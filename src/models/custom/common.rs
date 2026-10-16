//! Model-level glue shared by the modular causal and encoder-decoder graphs.
//!
//! Block computation lives in [`super::layer`], [`super::mixer`], and
//! [`super::feed_forward`]; per-pass tables live in [`super::pass`]. This
//! module validates whole stacks and public inputs, checks staged contexts
//! with one device read, assembles public outputs and router diagnostics, and
//! provides the readout helpers both models share.

use burn::{
    nn::Embedding,
    tensor::{Bool, DType, Int, Tensor, backend::Backend, bf16, f16},
};

use crate::{
    CausalLmInput, CausalLmOutput, Error, ForwardOptions, Result,
    architecture::{DecoderSpec, RouterDiagnostics, TokenMask},
    attention::AttentionConnectivity,
    experiment::{EncoderContext, ExecutionContext, HiddenState, read_int_scalars},
    intervention::Stack,
    moe::{SparseMoeOutput, load_balancing_loss},
};

pub(super) use crate::models::shared::select_hidden;
use crate::models::shared::{
    DeferredChecks, InputContract, InputNames, PaddedPositions, defer_input,
    validate_position_values,
};

use super::layer::{ENCODER_SLIDING_WINDOW, ensure_executable};

/// Complete hidden-state result retained before vocabulary projection.
pub(super) struct BackboneOutput<B: Backend> {
    /// Final normalized residual stream.
    pub(super) hidden: Tensor<B, 3>,
    /// Embedding, layer, and final states when requested.
    pub(super) hidden_states: Vec<Tensor<B, 3>>,
    /// Raw sparse outputs for each MoE layer.
    pub(super) routers: Vec<SparseMoeOutput<B>>,
}

/// Validates the executable capability boundary of one modular stack.
///
/// Beyond [`DecoderSpec::validate`], positions must fit Burn's portable
/// signed 32-bit index range, every block must have an executable kernel, and
/// an encoder stack (bidirectional by default) rejects causal recurrent
/// mixers and past-only sliding windows. Errors name fields below `field`.
pub(super) fn validate_stack_execution(
    spec: &DecoderSpec,
    stack: Stack,
    field: &str,
) -> Result<()> {
    spec.validate()?;
    if spec.max_position_embeddings > i32::MAX as usize {
        return Err(Error::configuration(
            format!("{field}.max_position_embeddings"),
            "must fit Burn's portable signed position-index range",
        ));
    }
    for (index, layer) in spec.layers.iter().enumerate() {
        let layer_field = format!("{field}.layers[{index}]");
        ensure_executable(layer, spec.hidden_size, &layer_field)?;
        if stack == Stack::Encoder {
            if layer.mixer.is_recurrent() {
                return Err(Error::BackendCapability(format!(
                    "{layer_field}.mixer is a causal recurrent scan and cannot represent a bidirectional encoder"
                )));
            }
            if matches!(
                layer.mixer.connectivity(),
                Some(AttentionConnectivity::SlidingWindow { .. })
            ) {
                return Err(Error::BackendCapability(format!(
                    "{layer_field}.mixer {ENCODER_SLIDING_WINDOW}"
                )));
            }
        }
    }
    Ok(())
}

/// Validates a public causal input and resolves its positions and rotary extent.
///
/// Omitted positions are valid-token ordinals, so left padding does not shift
/// them; explicit positions at padded columns are replaced by zero, so padding
/// sentinels are inert. Token, mask, and position values share one
/// device-to-host readback, which also yields the context `1 + max p`. Mask
/// and positions must share the token IDs' device. Errors name `{field}.*`.
pub(super) fn validate_and_resolve_input<B: Backend>(
    input: &CausalLmInput<B>,
    vocabulary_size: usize,
    max_positions: usize,
    field: &str,
) -> Result<(Tensor<B, 2, Int>, usize)> {
    validate_and_resolve_input_with(input, vocabulary_size, max_positions, field, |_| Ok(()))
}

/// [`validate_and_resolve_input`] plus caller checks that share its readback.
///
/// `extra` registers further deferred checks (for example a source mask's
/// row counts) after the input's own, so their errors follow the input's
/// value errors and every value still costs a single device-to-host read.
pub(super) fn validate_and_resolve_input_with<B: Backend>(
    input: &CausalLmInput<B>,
    vocabulary_size: usize,
    max_positions: usize,
    field: &str,
    extra: impl FnOnce(&mut DeferredChecks<'_, B>) -> Result<()>,
) -> Result<(Tensor<B, 2, Int>, usize)> {
    let token_ids = format!("{field}.token_ids");
    let token_mask = format!("{field}.token_mask");
    let position_ids = format!("{field}.position_ids");
    let names = InputNames {
        token_ids: &token_ids,
        token_mask: &token_mask,
        position_ids: &position_ids,
    };
    let mut checks = DeferredChecks::new();
    let pending = defer_input(
        input,
        &input.token_ids.device(),
        &InputContract::new(names, &token_ids, vocabulary_size)
            .with_position_limit(max_positions)
            .with_padded_positions(PaddedPositions::ZeroFill),
        &mut checks,
    )?;
    extra(&mut checks)?;
    let validated = pending.finish(&checks.run(&token_ids)?)?;
    Ok((
        validated.require_positions(&position_ids)?,
        validated.evaluated_context,
    ))
}

/// Builds the raw staged state of freshly embedded, already validated tokens.
///
/// `positions` and `evaluated_context` come from [`validate_and_resolve_input`],
/// so the context's position invariant already holds and only the sync-free
/// layout (including an attached `encoder`) is checked here. Encoder position
/// values are read once, with the target positions, when blocks run.
pub(super) fn embedded_state<B: Backend>(
    hidden: Tensor<B, 3>,
    positions: Tensor<B, 2, Int>,
    token_mask: Option<TokenMask<B>>,
    evaluated_context: usize,
    encoder: Option<EncoderContext<B>>,
) -> Result<HiddenState<B>> {
    let context = ExecutionContext {
        positions,
        token_mask,
        evaluated_context,
        attention: None,
        layer_attention: Vec::new(),
        cross_attention: None,
        layer_cross_attention: Vec::new(),
        encoder,
    };
    let [batch, tokens, _] = hidden.dims();
    context.validate_layout([batch, tokens], &hidden.device())?;
    Ok(HiddenState {
        hidden,
        context,
        next_layer: 0,
        normalized: false,
    })
}

/// Checks a staged context's position values before its blocks run.
///
/// Target positions must lie in `0..min(evaluated_context, max_positions)`:
/// the context invariant and the model's embedding range are checked with one
/// shared device read. An attached encoder's positions must be nonnegative,
/// which costs one more read only when a source is present. Layout is checked
/// separately and sync-free by [`ExecutionContext::validate_layout`].
pub(super) fn validate_staged_positions<B: Backend>(
    context: &ExecutionContext<B>,
    max_positions: usize,
    field: &str,
) -> Result<()> {
    validate_position_values(
        &context.positions,
        max_positions.min(context.evaluated_context),
        &format!("{field}.position_ids"),
    )?;
    if let Some(encoder) = &context.encoder {
        let minimum = read_int_scalars(vec![encoder.positions.clone().min()], "source positions")?;
        if minimum.first().is_some_and(|low| *low < 0) {
            return Err(Error::configuration(
                "execution.encoder.positions",
                "source positions must be nonnegative",
            ));
        }
    }
    Ok(())
}

/// Converts sparse outputs into opt-in public router diagnostics.
pub(super) fn router_diagnostics<B: Backend>(
    routers: &[SparseMoeOutput<B>],
    mask: Option<&TokenMask<B>>,
    options: &ForwardOptions,
) -> Result<Vec<RouterDiagnostics<B>>> {
    if !options.router_logits && !options.auxiliary_losses {
        return Ok(Vec::new());
    }
    routers
        .iter()
        .map(|router| {
            Ok(RouterDiagnostics {
                logits: options.router_logits.then(|| router.router_logits.clone()),
                load_balancing_loss: options
                    .auxiliary_losses
                    .then(|| router.load_balancing_loss(mask))
                    .transpose()?,
                z_loss: None,
            })
        })
        .collect()
}

/// Builds public output from an already-selected final hidden tensor.
pub(super) fn causal_output<B: Backend>(
    logits: Tensor<B, 3>,
    final_hidden: Tensor<B, 3>,
    hidden_states: Vec<Tensor<B, 3>>,
    routers: Vec<RouterDiagnostics<B>>,
    options: &ForwardOptions,
) -> CausalLmOutput<B> {
    CausalLmOutput {
        logits,
        final_hidden_state: options.hidden_states.then_some(final_hidden),
        hidden_states,
        attentions: Vec::new(),
        routers,
        load_balancing_loss: None,
    }
}

/// Rejects unsupported attention capture before any model computation begins.
pub(super) fn validate_forward_options(options: &ForwardOptions) -> Result<()> {
    if options.attentions {
        return Err(Error::BackendCapability(
            "modular attention-probability capture is not exposed by the shared Burn dispatch"
                .into(),
        ));
    }
    Ok(())
}

/// Returns whether a stack contains sparse experts, plain or clamped.
pub(super) fn has_moe(spec: &DecoderSpec) -> bool {
    spec.layers
        .iter()
        .any(|layer| layer.feed_forward.experts().is_some())
}

/// Looks up `[batch, token]` IDs in `table` and applies a stack's optional embedding multiplier.
///
/// Computes `h_0 = s · E[token]` in the table's dtype, with `s`
/// ([`DecoderSpec::embedding_scale`]) first rounded to that dtype by
/// [`round_to_dtype`]; `None` returns the plain lookup.
pub(super) fn embed_tokens<B: Backend>(
    table: &Embedding<B>,
    token_ids: Tensor<B, 2, Int>,
    scale: Option<f64>,
) -> Tensor<B, 3> {
    let embeddings = table.forward(token_ids);
    match scale {
        Some(scale) => {
            let scale = round_to_dtype(scale, embeddings.dtype());
            embeddings.mul_scalar(scale)
        }
        None => embeddings,
    }
}

/// Rounds a scalar multiplier to the precision of `dtype`.
///
/// F64 keeps the value, F16 and BF16 round through F32 (as Gemma's references
/// build their `sqrt(d)` normalizer: an F32 value converted to the activation
/// dtype), and every other floating dtype uses the F32 value. Multiplying a
/// tensor of `dtype` by the result therefore uses a scalar that dtype
/// represents exactly, independent of how a backend converts scalars.
pub(super) fn round_to_dtype(value: f64, dtype: DType) -> f64 {
    let single = value as f32;
    match dtype {
        DType::F64 => value,
        DType::F16 => f64::from(f16::from_f32(single).to_f32()),
        DType::BF16 => f64::from(bf16::from_f32(single).to_f32()),
        _ => f64::from(single),
    }
}

/// Applies a stack's optional final logit soft cap `c · tanh(z / c)` in the logits' dtype.
///
/// `cap` is [`DecoderSpec::final_logit_softcap`]; `None` returns the logits unchanged.
pub(super) fn soft_cap_logits<B: Backend>(logits: Tensor<B, 3>, cap: Option<f64>) -> Tensor<B, 3> {
    match cap {
        Some(cap) => logits.div_scalar(cap).tanh().mul_scalar(cap),
        None => logits,
    }
}

/// Applies an ordinary tied vocabulary projection with a `[vocabulary, hidden]` embedding table.
///
/// `logits[b, t, v] = Σ_i hidden[b, t, i] · E[v, i]`, computed by
/// [`crate::operations::matmul_transposed`] against the transposed view of `E`.
/// The table is never copied; Burn's `module::linear` reshapes the transposed
/// table and therefore copies it on Flex on every call.
///
/// # Errors
///
/// Propagates the shape, dtype, and device checks of `matmul_transposed`.
pub(super) fn tied_projection<B: Backend>(
    hidden: Tensor<B, 3>,
    embedding_weight: Tensor<B, 2>,
) -> Result<Tensor<B, 3>> {
    crate::operations::matmul_transposed(hidden, embedding_weight)
}

/// Recomputes auxiliary loss from gathered cached-step router tensors.
pub(super) fn gathered_router_diagnostics<B: Backend>(
    logits: Tensor<B, 3>,
    selected_experts: Tensor<B, 3, Int>,
    mask: Option<&TokenMask<B>>,
    options: &ForwardOptions,
) -> Result<RouterDiagnostics<B>> {
    Ok(RouterDiagnostics {
        logits: options.router_logits.then_some(logits.clone()),
        load_balancing_loss: options
            .auxiliary_losses
            .then(|| load_balancing_loss(logits, selected_experts, mask))
            .transpose()?,
        z_loss: None,
    })
}

/// Casts a boolean mask into a float multiplier matching a hidden tensor's dtype.
pub(super) fn mask_hidden<B: Backend>(
    hidden: Tensor<B, 3>,
    mask: Tensor<B, 2, Bool>,
) -> Tensor<B, 3> {
    let dtype = hidden.dtype();
    hidden * mask.float().cast(dtype).unsqueeze_dim(2)
}

/// Creates a token mask from a rectangular host validity vector.
pub(super) fn token_mask_from_values<B: Backend>(
    values: Vec<bool>,
    shape: [usize; 2],
    device: &B::Device,
) -> TokenMask<B> {
    TokenMask::new(crate::architecture::bool_tensor_from_values(
        values, shape, device,
    ))
}
