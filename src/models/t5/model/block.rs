//! The single T5 block and stack graph shared by every execution path.
//!
//! A T5 block is three pre-normalized residual sublayers:
//!
//! ```text
//! h1 = x  + SelfAttn(RMS_a(x))            softmax(Q K^T + bias + mask) V
//! h2 = h1 + CrossAttn(RMS_c(h1), E)       decoder only, over encoder states E
//! y  = h2 + FFN(RMS_f(h2))                dense ReLU/GELU or gated GELU/SiLU
//! ```
//!
//! Scores are unscaled (`Q K^T`, no `1 / sqrt(d_kv)`), the relative bias is
//! computed once per pass by [`super::bias`], and masks use Burn's
//! `true == excluded` convention. Residuals are `[batch, token, d_model]`;
//! attention tensors are `[batch, head, token, d_kv]`.
//!
//! Every path — plain and hooked stateless encoding and decoding, staged
//! encoder recipes, and cached decoding — runs [`T5Block::run`]. Paths differ
//! only in their inputs: the exclusion plane, the bias, causality, where
//! self-attention keys come from ([`SelfKeyValues`]: fresh projections or a
//! decode cache written ahead), and whether cross-attention projects encoder
//! states or reuses cached projections ([`CrossSource`]). Hooks are emitted in
//! the order [`T5Block::activation_points`] lists; an empty [`Hooks`]
//! collection makes every emission a no-op, so the plain paths share the
//! hooked graph without overhead.
//!
//! # Half-precision overflow guard
//!
//! T5 v1.1, FLAN, and UL2 activations can exceed the F16 range. Like the
//! reference `T5Block`, an F16 residual stream is clamped after each sublayer
//! to `±c`, where `c = 65504` (the F16 maximum) normally and `c = 64504`
//! (rounding to `64512` in F16) whenever any element is infinite. The choice
//! of `c` is made on the device, so the guard needs no host synchronization.
//! F32, BF16, and F64 streams are untouched (no extra operations). The
//! reference additionally keeps the FFN output projection `wo` in F32; this
//! crate keeps one parameter dtype, so the guard prevents `inf`/`NaN`
//! propagation but cannot recover the precision an F16 `wo` loses. BF16 is
//! the recommended reduced precision for T5 v1.1/FLAN/UL2.

use burn::tensor::{Bool, DType, Tensor, backend::Backend};

use crate::{
    Error, Result,
    architecture::TokenMask,
    attention::{AttentionOutputGate, AttentionProjection, GroupedQueryAttention},
    intervention::{Hooks, Layer, Stack},
    operations,
};

use super::{T5Block, T5LayerCrossAttention, T5LayerFeedForward, T5LayerSelfAttention, T5Stack};

/// Largest finite F16 value, `(2 - 2^-10) * 2^15`.
const F16_MAX: f64 = 65_504.0;

/// Reference clamp bound once any F16 activation has overflowed (`F16_MAX - 1000`).
const F16_INF_CLAMP: f64 = F16_MAX - 1000.0;

/// Supplies the keys and values a block's self-attention queries attend to.
///
/// The block projects keys and values for its own tokens; a stateless pass
/// attends to exactly those ([`FreshKeyValues`]), while cached decoding writes
/// them into its decode cache and attends over the whole retained history.
pub(super) trait SelfKeyValues<B: Backend> {
    /// Maps layer `layer`'s new `[batch, heads, tokens, d_kv]` keys and values
    /// to the keys and values its queries attend to.
    fn resolve(
        &mut self,
        layer: usize,
        keys: Tensor<B, 4>,
        values: Tensor<B, 4>,
    ) -> Result<(Tensor<B, 4>, Tensor<B, 4>)>;
}

/// Stateless self-attention: queries attend to this pass's own keys and values.
pub(super) struct FreshKeyValues;

impl<B: Backend> SelfKeyValues<B> for FreshKeyValues {
    /// Returns the freshly projected keys and values unchanged.
    fn resolve(
        &mut self,
        _layer: usize,
        keys: Tensor<B, 4>,
        values: Tensor<B, 4>,
    ) -> Result<(Tensor<B, 4>, Tensor<B, 4>)> {
        Ok((keys, values))
    }
}

/// Self-attention visibility and bias of one block evaluation.
#[derive(Clone, Debug)]
pub(super) struct SelfAttentionPlan<B: Backend> {
    /// Exclusions broadcastable to `[batch, heads, query, key]` (`true` = excluded).
    ///
    /// Usually a `[batch, 1, 1 or query, key]` plane; the block expands it to
    /// its own head count without copying.
    pub(super) excluded: Option<Tensor<B, 4, Bool>>,
    /// Additive bias `[1 or batch, heads, query, key]`.
    pub(super) bias: Option<Tensor<B, 4>>,
    /// Whether bottom-right causality additionally excludes future keys.
    pub(super) causal: bool,
}

impl<B: Backend> SelfAttentionPlan<B> {
    /// Excludes padded keys (`true == valid` token mask) and applies `bias`.
    pub(super) fn padded(
        key_mask: Option<&TokenMask<B>>,
        bias: Option<Tensor<B, 4>>,
        causal: bool,
    ) -> Self {
        Self {
            excluded: key_mask.map(key_exclusions),
            bias,
            causal,
        }
    }

    /// Causal self-attention over a padded target rectangle (`true == valid` mask).
    ///
    /// Valid queries exclude padded keys, exactly as [`Self::padded`] does.
    /// Padded queries exclude nothing beyond causality, so every score row
    /// keeps at least key zero. With left padding a padded query precedes
    /// every valid key of its row and would otherwise see no key at all;
    /// backends define such empty softmax rows differently (Flex returns
    /// zeros, others may produce `NaN`), and a `NaN` row poisons the backward
    /// pass even after its output is discarded. The caller zeroes padded
    /// outputs, and valid queries never read padded keys, so valid outputs are
    /// unchanged. Costs one `[batch, 1, token, token]` Boolean plane per pass,
    /// shared by every block and head.
    pub(super) fn padded_causal(mask: &TokenMask<B>, bias: Option<Tensor<B, 4>>) -> Self {
        let valid = mask.as_tensor().clone();
        let [batch, tokens] = valid.dims();
        let plane = [batch, 1, tokens, tokens];
        let invalid_keys = valid
            .clone()
            .bool_not()
            .reshape([batch, 1, 1, tokens])
            .expand(plane);
        let valid_queries = valid.reshape([batch, 1, tokens, 1]).expand(plane);
        Self {
            excluded: Some(invalid_keys.bool_and(valid_queries)),
            bias,
            causal: true,
        }
    }
}

/// Where a decoder block's cross-attention keys and values come from.
#[derive(Clone, Copy, Debug)]
pub(super) enum CrossSource<'a, B: Backend> {
    /// Project keys and values from encoder states `[batch, source, d_model]`.
    Encoder(
        /// Final encoder hidden states, `[batch, source, d_model]`.
        &'a Tensor<B, 3>,
    ),
    /// Reuse per-layer projections `[batch, heads, source, d_kv]` made once at prefill.
    Projected {
        /// Projected keys, one tensor per decoder layer.
        keys: &'a [Tensor<B, 4>],
        /// Projected values, one tensor per decoder layer.
        values: &'a [Tensor<B, 4>],
    },
}

/// Cross-attention inputs shared by every decoder block of one pass.
#[derive(Clone, Debug)]
pub(super) struct CrossAttentionPlan<'a, B: Backend> {
    /// Encoder states or their cached projections.
    pub(super) source: CrossSource<'a, B>,
    /// Padded-source exclusions broadcastable to `[batch, heads, query, source]`.
    pub(super) excluded: Option<Tensor<B, 4, Bool>>,
}

impl<'a, B: Backend> CrossAttentionPlan<'a, B> {
    /// Builds the plan for `source`, excluding padded encoder keys.
    pub(super) fn new(source: CrossSource<'a, B>, source_mask: Option<&TokenMask<B>>) -> Self {
        Self {
            source,
            excluded: source_mask.map(key_exclusions),
        }
    }
}

/// Inputs shared by every block of one stack pass.
#[derive(Clone, Debug)]
pub(super) struct StackPlan<'a, B: Backend> {
    /// Self-attention visibility and the stack's single relative bias.
    pub(super) self_attention: SelfAttentionPlan<B>,
    /// Cross-attention inputs; present exactly for decoder stacks.
    pub(super) cross_attention: Option<CrossAttentionPlan<'a, B>>,
}

/// Final hidden state plus optional per-layer snapshots of one stack pass.
#[derive(Debug)]
pub(super) struct T5StackOutput<B: Backend> {
    /// Final normalized stack output `[batch, token, d_model]`.
    pub(super) hidden: Tensor<B, 3>,
    /// Embedding, every block output, and the final normalization, when requested.
    pub(super) hidden_states: Vec<Tensor<B, 3>>,
}

/// Converts a `[batch, key]` validity mask to a `[batch, 1, 1, key]` exclusion plane.
///
/// Costs one `O(batch * key)` negation; heads and queries are broadcast views.
fn key_exclusions<B: Backend>(mask: &TokenMask<B>) -> Tensor<B, 4, Bool> {
    let [batch, keys] = mask.as_tensor().dims();
    mask.as_tensor()
        .clone()
        .bool_not()
        .reshape([batch, 1, 1, keys])
}

/// Expands an exclusion plane to the exact `[batch, heads, query, key]` score shape.
///
/// A plane that does not broadcast is reported with the attention's own
/// shape error rather than a backend panic.
fn expand_exclusions<B: Backend>(
    excluded: Option<Tensor<B, 4, Bool>>,
    shape: [usize; 4],
) -> Result<Option<Tensor<B, 4, Bool>>> {
    excluded
        .map(|plane| {
            let dims = plane.dims();
            if dims
                .iter()
                .zip(shape)
                .all(|(&actual, expected)| actual == expected || actual == 1)
            {
                Ok(if dims == shape {
                    plane
                } else {
                    plane.expand(shape)
                })
            } else {
                Err(Error::shape(
                    "t5.attention.mask",
                    format!("broadcastable to {shape:?}"),
                    format!("{dims:?}"),
                ))
            }
        })
        .transpose()
}

/// Applies the reference F16 overflow guard to a residual stream (see the module docs).
///
/// Non-F16 tensors are returned unchanged without any operation. For F16 the
/// bound `c` is `F16_MAX`, or `F16_INF_CLAMP` when any element is infinite,
/// selected on the device; values beyond `±c` (including infinities) are
/// replaced by `±c`. NaN is propagated, as `torch.clamp` does.
pub(super) fn clamp_f16_residual<B: Backend>(hidden: Tensor<B, 3>) -> Tensor<B, 3> {
    if hidden.dtype() != DType::F16 {
        return hidden;
    }
    let shape = hidden.dims();
    let overflowed = hidden.clone().is_inf().any().float().cast(DType::F16);
    // c = F16_MAX - 1000 * any(isinf(x)), rounded to F16 like the reference.
    let bound = overflowed
        .mul_scalar(F16_MAX - F16_INF_CLAMP)
        .neg()
        .add_scalar(F16_MAX)
        .reshape([1, 1, 1])
        .expand(shape);
    let lower = bound.clone().neg();
    let above = hidden.clone().greater(bound.clone());
    let hidden = hidden.mask_where(above, bound);
    let below = hidden.clone().lower(lower.clone());
    hidden.mask_where(below, lower)
}

impl<B: Backend> T5LayerSelfAttention<B> {
    /// Computes `x + SelfAttn(RMS(x))`, emitting the self-attention points of `layer`.
    ///
    /// `keys` maps this pass's freshly projected keys and values to the
    /// attended ones, which lets cached decoding share this exact graph.
    pub(super) fn forward_with(
        &self,
        hidden: Tensor<B, 3>,
        plan: SelfAttentionPlan<B>,
        keys: &mut dyn SelfKeyValues<B>,
        layer: Layer,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<Tensor<B, 3>> {
        let attention = &self.self_attention.attention;
        validate_residual(&hidden, attention.hidden_size, "t5.self_attention.input")?;
        let normalized = hooks.apply(
            layer.attention_input(),
            self.layer_norm.forward(hidden.clone()),
        )?;
        let mut projection = attention.project_self(normalized)?;
        (projection.key, projection.value) =
            keys.resolve(layer.index(), projection.key, projection.value)?;
        let [batch, _, query, _] = projection.query.dims();
        let key = projection.key.dims()[2];
        let excluded =
            expand_exclusions(plan.excluded, [batch, attention.query_heads, query, key])?;
        let attended = attention.attend_with_hooks(
            projection,
            excluded,
            plan.bias,
            plan.causal,
            layer.self_attention(),
            hooks,
        )?;
        Ok(hidden + hooks.apply(layer.attention_output(), attended)?)
    }
}

impl<B: Backend> T5LayerCrossAttention<B> {
    /// Computes `x + CrossAttn(RMS(x), source)`, emitting the cross-attention points of `layer`.
    ///
    /// With [`CrossSource::Projected`] only the queries are projected; the
    /// keys and values were projected from the bound encoder once at prefill.
    pub(super) fn forward_with(
        &self,
        hidden: Tensor<B, 3>,
        plan: &CrossAttentionPlan<'_, B>,
        layer: Layer,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<Tensor<B, 3>> {
        let attention = &self.enc_dec_attention.attention;
        validate_residual(&hidden, attention.hidden_size, "t5.cross_attention.input")?;
        let normalized = hooks.apply(
            layer.cross_attention_input(),
            self.layer_norm.forward(hidden.clone()),
        )?;
        let projection = match plan.source {
            CrossSource::Encoder(source) => {
                validate_residual(
                    source,
                    attention.hidden_size,
                    "t5.encoder_output.hidden_state",
                )?;
                if source.dims()[0] != normalized.dims()[0] {
                    return Err(Error::shape(
                        "t5.encoder_output.hidden_state",
                        format!(
                            "[{}, source>0, {}]",
                            normalized.dims()[0],
                            attention.hidden_size
                        ),
                        format!("{:?}", source.dims()),
                    ));
                }
                attention.project(normalized, source.clone())?
            }
            CrossSource::Projected { keys, values } => {
                let index = layer.index();
                let (Some(key), Some(value)) = (keys.get(index), values.get(index)) else {
                    return Err(Error::Cache(format!(
                        "T5 cross-attention layer {index} is outside {} cached layers",
                        keys.len().min(values.len())
                    )));
                };
                AttentionProjection {
                    query: project_queries(attention, normalized)?,
                    key: key.clone(),
                    value: value.clone(),
                    gate: None,
                }
            }
        };
        let [batch, _, query, _] = projection.query.dims();
        let source = projection.key.dims()[2];
        let excluded = expand_exclusions(
            plan.excluded.clone(),
            [batch, attention.query_heads, query, source],
        )?;
        let attended = attention.attend_with_hooks(
            projection,
            excluded,
            None,
            false,
            layer.cross_attention(),
            hooks,
        )?;
        Ok(hidden + hooks.apply(layer.cross_attention_output(), attended)?)
    }
}

impl<B: Backend> T5LayerFeedForward<B> {
    /// Computes `x + FFN(RMS(x))`, emitting the MLP points of `layer`.
    ///
    /// Dense and gated topologies use the shared hook-aware MLP forwards; with
    /// empty hooks they equal the plain forwards exactly.
    pub(super) fn forward_with(
        &self,
        hidden: Tensor<B, 3>,
        layer: Layer,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<Tensor<B, 3>> {
        let normalized = hooks.apply(layer.mlp_input(), self.layer_norm.forward(hidden.clone()))?;
        let projected = match (&self.dense, &self.gated, self.projection.is_gated()) {
            (Some(dense), None, false) => dense.forward_with_hooks(normalized, layer, hooks)?,
            (None, Some(gated), true) => gated.forward_with_hooks(normalized, layer, hooks)?,
            _ => {
                return Err(Error::configuration(
                    "t5.feed_forward",
                    "active dense/gated module disagrees with checkpoint projection",
                ));
            }
        };
        Ok(hidden + hooks.apply(layer.mlp_output(), projected)?)
    }
}

impl<B: Backend> T5Block<B> {
    /// Runs the one T5 block graph: self-attention, optional cross-attention, FFN.
    ///
    /// Encoder blocks (bidirectional, no cross-attention module) require
    /// `cross_attention == None`; decoder blocks (causal, with cross-attention)
    /// require it. The F16 overflow guard runs after every residual sublayer.
    pub(super) fn run(
        &self,
        hidden: Tensor<B, 3>,
        self_attention: SelfAttentionPlan<B>,
        keys: &mut dyn SelfKeyValues<B>,
        cross_attention: Option<&CrossAttentionPlan<'_, B>>,
        layer: Layer,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<Tensor<B, 3>> {
        match (
            cross_attention.is_some(),
            self.cross_attention.is_some(),
            self.self_attention.causal,
        ) {
            (false, false, false) | (true, true, true) => {}
            (true, false, _) => {
                return Err(Error::configuration(
                    "t5.decoder.cross_attention",
                    "decoder block has no encoder-decoder attention module",
                ));
            }
            (true, true, false) => {
                return Err(Error::configuration(
                    "t5.decoder.block",
                    "decoder block was initialized with bidirectional self-attention",
                ));
            }
            (false, _, _) => {
                return Err(Error::configuration(
                    "t5.encoder.block",
                    "encoder block was initialized with decoder sublayers",
                ));
            }
        }
        let hidden = hooks.apply(layer.residual_pre(), hidden)?;
        let mut hidden = clamp_f16_residual(self.self_attention.forward_with(
            hidden,
            self_attention,
            keys,
            layer,
            hooks,
        )?);
        if let (Some(plan), Some(cross)) = (cross_attention, &self.cross_attention) {
            hidden = clamp_f16_residual(cross.forward_with(hidden, plan, layer, hooks)?);
        }
        let hidden = clamp_f16_residual(self.feed_forward.forward_with(hidden, layer, hooks)?);
        hooks.apply(layer.residual_post(), hidden)
    }
}

impl<B: Backend> T5Stack<B> {
    /// Returns the address of block `index` in this stack.
    pub(super) fn layer(&self, index: usize) -> Layer {
        if self.is_decoder {
            Layer::decoder(index)
        } else {
            Layer::encoder(index)
        }
    }

    /// Returns this stack's hook namespace.
    pub(super) const fn stack(&self) -> Stack {
        if self.is_decoder {
            Stack::Decoder
        } else {
            Stack::Encoder
        }
    }

    /// Runs embedding-boundary hooks, every block, and the final normalization.
    ///
    /// `plan` is shared by every block: its exclusion plane and relative bias
    /// are built once per pass. Retained diagnostics are the embedding, every
    /// block output, and the final normalized state (`layers + 2` tensors).
    pub(super) fn run(
        &self,
        hidden: Tensor<B, 3>,
        plan: &StackPlan<'_, B>,
        keys: &mut dyn SelfKeyValues<B>,
        retain_hidden_states: bool,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<T5StackOutput<B>> {
        if self.is_decoder != plan.cross_attention.is_some() {
            return Err(Error::configuration(
                "t5.stack",
                "only decoder stacks accept and require encoder states",
            ));
        }
        let stack = self.stack();
        let mut hidden = hooks.apply(stack.embedding(), hidden)?;
        let mut hidden_states = diagnostic_buffer(self.block.len(), retain_hidden_states);
        if retain_hidden_states {
            hidden_states.push(hidden.clone());
        }
        for (index, block) in self.block.iter().enumerate() {
            hidden = block.run(
                hidden,
                plan.self_attention.clone(),
                keys,
                plan.cross_attention.as_ref(),
                self.layer(index),
                hooks,
            )?;
            if retain_hidden_states {
                hidden_states.push(hidden.clone());
            }
        }
        hidden = hooks.apply(stack.final_norm(), self.final_layer_norm.forward(hidden))?;
        if retain_hidden_states {
            hidden_states.push(hidden.clone());
        }
        Ok(T5StackOutput {
            hidden,
            hidden_states,
        })
    }
}

/// Allocates diagnostics only when requested, avoiding inference-path growth.
pub(super) fn diagnostic_buffer<B: Backend>(depth: usize, retain: bool) -> Vec<Tensor<B, 3>> {
    Vec::with_capacity(if retain { depth.saturating_add(2) } else { 0 })
}

/// Checks a `[batch>0, token>0, width]` residual before its first projection.
fn validate_residual<B: Backend>(hidden: &Tensor<B, 3>, width: usize, name: &str) -> Result<()> {
    let [batch, tokens, actual] = hidden.dims();
    if batch == 0 || tokens == 0 || actual != width {
        return Err(Error::shape(
            name,
            format!("[batch>0, token>0, {width}]"),
            format!("{:?}", hidden.dims()),
        ));
    }
    Ok(())
}

/// Projects only the queries `[batch, heads, token, d_kv]` for cached cross-attention.
///
/// T5 attention has no query normalization and no output gate; a module
/// edited to have either cannot reuse projections made by the full
/// projection and is rejected rather than silently mis-projected.
fn project_queries<B: Backend>(
    attention: &GroupedQueryAttention<B>,
    hidden: Tensor<B, 3>,
) -> Result<Tensor<B, 4>> {
    if attention.query_norm.is_some() || attention.output_gate != AttentionOutputGate::None {
        return Err(Error::configuration(
            "t5.cross_attention",
            "cached cross-attention requires ungated attention without query normalization",
        ));
    }
    let [batch, tokens, _] = hidden.dims();
    Ok(operations::linear(hidden, &attention.q_proj)?
        .reshape([batch, tokens, attention.query_heads, attention.head_dim])
        .swap_dims(1, 2))
}

#[cfg(test)]
#[path = "block_tests.rs"]
mod tests;
