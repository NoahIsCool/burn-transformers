//! One heterogeneous residual block and its single execution path.
//!
//! A block factors into a sequence mixer `M`, an optional cross-attention
//! branch `X`, and an optional feed-forward branch `F`, each wrapped by its own
//! normalization layout ([`NormLayout`]):
//!
//! ```text
//! h1 = R_m(x,  M(N_m(x)))          mixer branch
//! h2 = R_x(h1, X(N_x(h1), source)) cross-attention (seq2seq decoders only)
//! y  = R_f(h2, F(N_f(h2)))         feed-forward branch (absent for attention-only blocks)
//! ```
//!
//! where `R` is the residual combination of the layout (pre, post, sandwich,
//! or identity). [`ModularLayer::forward`] is the only implementation of this
//! computation: full-sequence model passes, staged range runners, and the
//! public single-block [`HiddenTransform`] entry point all call it with a
//! [`StackPass`] carrying positions, padding, overrides, and shared tables.
//! The wrappers differ only in how much validation they perform first.
//!
//! # Activation points
//!
//! A block emits, in order: `residual_pre`, `attention_input`, the mixer's
//! attention points, `attention_output`, the cross-attention input, points,
//! and output (decoders), `mlp_input`, the dense MLP points, `mlp_output`, and
//! `residual_post`. Attention-only blocks omit every MLP point.

use burn::{
    module::Module,
    tensor::{DType, Tensor, backend::Backend},
};

use crate::{
    Error, Result,
    architecture::{CrossAttentionSpec, DecoderLayerSpec, ResidualTopology},
    attention::{AttentionConnectivity, AttentionPrecisionMode},
    experiment::{ExecutionContext, HiddenTransform},
    intervention::{Hooks, Layer, PointInfo, Stack},
    moe::SparseMoeOutput,
};

use super::{
    feed_forward::ModularFeedForward,
    mixer::{ModularAttention, ModularMixer},
    norm::NormLayout,
    pass::{StackPass, explicit_cross_policy, explicit_self_policy},
};

/// Error text for a trailing window evaluated without a causal direction.
pub(super) const ENCODER_SLIDING_WINDOW: &str = "defines a trailing (past-only) window and cannot represent symmetric bidirectional encoder attention; use chunked-local connectivity or an explicit attention policy";

/// One heterogeneous residual block and optional seq2seq cross-attention branch.
#[derive(Module, Debug)]
pub struct ModularLayer<B: Backend> {
    /// Sequence mixer.
    mixer: ModularMixer<B>,
    /// Dense or sparse feed-forward operation; absent for attention-only blocks.
    feed_forward: Option<ModularFeedForward<B>>,
    /// Mixer-side normalization parameters.
    mixer_norm: NormLayout<B>,
    /// Feed-forward-side normalization parameters.
    feed_forward_norm: NormLayout<B>,
    /// Decoder-to-encoder attention, absent from causal-only and encoder layers.
    cross_attention: Option<ModularAttention<B>>,
    /// Independent cross-attention normalization using the mixer convention.
    cross_attention_norm: Option<NormLayout<B>>,
    /// Residual width consumed and returned by this block.
    #[module(skip)]
    hidden_size: usize,
}

impl<B: Backend> ModularLayer<B> {
    /// Validates one standalone block and allocates its parameters.
    ///
    /// `cross_attention` adds a decoder-to-encoder branch (seq2seq decoder
    /// blocks). Specification errors name fields below `layer` (and
    /// `cross_attention`); a well-formed block whose mechanism has no
    /// executable kernel (Mamba2, gated residuals) fails with a capability
    /// error before any parameter is allocated.
    pub fn init(
        spec: &DecoderLayerSpec,
        hidden_size: usize,
        cross_attention: Option<&CrossAttentionSpec>,
        device: &B::Device,
    ) -> Result<Self> {
        if hidden_size == 0 {
            return Err(Error::configuration(
                "layer.hidden_size",
                "must be non-zero",
            ));
        }
        spec.validate_at("layer", hidden_size)?;
        if let Some(cross) = cross_attention {
            cross.validate("cross_attention", hidden_size)?;
        }
        ensure_executable(spec, "layer")?;
        Self::build(spec, hidden_size, cross_attention, device)
    }

    /// Allocates a block whose specification the caller has already validated.
    pub(super) fn build(
        spec: &DecoderLayerSpec,
        hidden_size: usize,
        cross_attention: Option<&CrossAttentionSpec>,
        device: &B::Device,
    ) -> Result<Self> {
        let (cross_attention, cross_attention_norm) = match cross_attention {
            Some(cross) => (
                Some(ModularAttention::init(
                    &cross.attention(),
                    &cross.options,
                    hidden_size,
                    spec.mixer_norm,
                    device,
                )?),
                Some(NormLayout::init(
                    spec.mixer_norm,
                    spec.norm_placement,
                    hidden_size,
                    device,
                )?),
            ),
            None => (None, None),
        };
        let feed_forward = ModularFeedForward::init(&spec.feed_forward, hidden_size, device)?;
        let feed_forward_norm = if feed_forward.is_some() {
            NormLayout::init(
                spec.feed_forward_norm,
                spec.norm_placement,
                hidden_size,
                device,
            )?
        } else {
            NormLayout::identity()
        };
        Ok(Self {
            mixer: ModularMixer::init(&spec.mixer, hidden_size, spec.mixer_norm, device)?,
            feed_forward,
            mixer_norm: NormLayout::init(
                spec.mixer_norm,
                spec.norm_placement,
                hidden_size,
                device,
            )?,
            feed_forward_norm,
            cross_attention,
            cross_attention_norm,
            hidden_size,
        })
    }

    /// Returns this block's residual width, independent of other blocks in an experiment.
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Executes one independently constructed block using explicit positions and policy.
    pub fn run(
        &self,
        hidden: Tensor<B, 3>,
        context: &ExecutionContext<B>,
        layer: Layer,
    ) -> Result<Tensor<B, 3>> {
        self.run_with_hooks(hidden, context, layer, &mut Hooks::new())
    }

    /// Executes one block with a scope containing exactly its supported activation points.
    ///
    /// An explicit attention policy replaces the stack's causal/bidirectional
    /// default and the block's structural connectivity, keeps padding
    /// excluded, and adds its bias to any learned position bias. DeltaNet
    /// accepts only its ordinary causal policy, and a sliding window needs a
    /// causal direction. Cross-attention reads `context.encoder` with source
    /// padding. Validation runs inside the hook scope, so a rejected call
    /// clears earlier captures.
    pub fn run_with_hooks(
        &self,
        hidden: Tensor<B, 3>,
        context: &ExecutionContext<B>,
        layer: Layer,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<Tensor<B, 3>> {
        hooks.run(self.activation_points(layer), |hooks| {
            <Self as HiddenTransform<B>>::forward(self, hidden, context, layer, hooks)
        })
    }

    /// Returns only the intervention points implemented by this block's actual operators.
    pub fn activation_points(&self, layer: Layer) -> Vec<PointInfo> {
        let mut points = vec![layer.residual_pre().info(), layer.attention_input().info()];
        points.extend(self.mixer.activation_points(layer));
        points.push(layer.attention_output().info());
        if let Some(attention) = &self.cross_attention {
            points.push(layer.cross_attention_input().info());
            points.extend(attention.activation_points(layer.cross_attention()));
            points.push(layer.cross_attention_output().info());
        }
        if let Some(feed_forward) = &self.feed_forward {
            points.push(layer.mlp_input().info());
            points.extend(feed_forward.activation_points(layer));
            points.push(layer.mlp_output().info());
        }
        points.push(layer.residual_post().info());
        points
    }

    /// Evaluates every residual branch of this block over one stack pass.
    ///
    /// This is the single execution path of the block. Interventions apply
    /// before dependent computation; MoE layers also return their raw routing
    /// state for diagnostics. Callers validate inputs first.
    pub(super) fn forward(
        &self,
        hidden: Tensor<B, 3>,
        pass: &StackPass<'_, B>,
        layer: Layer,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<(Tensor<B, 3>, Option<SparseMoeOutput<B>>)> {
        let hidden = hooks.apply(layer.residual_pre(), hidden)?;
        let residual = hidden.clone();
        let prepared = hooks.apply(layer.attention_input(), self.mixer_norm.prepare(hidden)?)?;
        let mixed = self.mixer.forward(prepared, pass, layer, hooks)?;
        let mixed = hooks.apply(layer.attention_output(), mixed)?;
        let mut hidden = self.mixer_norm.finish(residual, mixed)?;

        if let (Some(attention), Some(norm)) = (&self.cross_attention, &self.cross_attention_norm) {
            let residual = hidden.clone();
            let prepared = hooks.apply(layer.cross_attention_input(), norm.prepare(hidden)?)?;
            let attended = attention.forward_cross(prepared, pass, layer, hooks)?;
            let attended = hooks.apply(layer.cross_attention_output(), attended)?;
            hidden = norm.finish(residual, attended)?;
        }

        let mut router = None;
        if let Some(feed_forward) = &self.feed_forward {
            let residual = hidden.clone();
            let prepared =
                hooks.apply(layer.mlp_input(), self.feed_forward_norm.prepare(hidden)?)?;
            let (projected, routing) = feed_forward.forward(prepared, layer, hooks)?;
            router = routing;
            let projected = hooks.apply(layer.mlp_output(), projected)?;
            hidden = self.feed_forward_norm.finish(residual, projected)?;
        }
        Ok((hooks.apply(layer.residual_post(), hidden)?, router))
    }

    /// Validates a block inside a range whose context values were checked once.
    ///
    /// Range runners validate the context's position values once per range,
    /// because the context is borrowed immutably and cannot change between
    /// blocks. Hooks may still replace hidden states, so this re-checks the
    /// width, dtype, device, layout, and policies without any device read.
    pub(super) fn validate_prevalidated(
        &self,
        hidden: &Tensor<B, 3>,
        context: &ExecutionContext<B>,
        layer: Layer,
    ) -> Result<()> {
        let [batch, tokens, _] = hidden.dims();
        self.validate_hidden(hidden)?;
        context.validate_layout([batch, tokens], &hidden.device())?;
        self.validate_policies(hidden, context, layer)
    }

    /// Checks the residual width and the activation dtype/device against this block.
    fn validate_hidden(&self, hidden: &Tensor<B, 3>) -> Result<()> {
        let width = hidden.dims()[2];
        if width != self.hidden_size {
            return Err(Error::shape(
                "custom.layer.hidden",
                format!("[batch, token, {}]", self.hidden_size),
                format!("{:?}", hidden.dims()),
            ));
        }
        let projection = match &self.mixer {
            ModularMixer::Attention(attention) => attention.attention.q_proj.weight.val(),
            ModularMixer::Mla(mla) => mla.attention.kv_a_proj_with_mqa.weight.val(),
            ModularMixer::DeltaNet(delta) => delta.in_proj_qkv.weight.val(),
        };
        let packed_weight = matches!(projection.dtype(), DType::QFloat(_));
        if projection.device() != hidden.device()
            || matches!(hidden.dtype(), DType::QFloat(_))
            || (!packed_weight && projection.dtype() != hidden.dtype())
        {
            return Err(Error::configuration(
                "custom.layer.hidden",
                "hidden states must share this block's device and floating projection dtype; packed weights accept floating activations",
            ));
        }
        Ok(())
    }

    /// Checks mixer, self-attention, and cross-attention policies against `hidden`.
    ///
    /// Without an explicit self-attention policy, an encoder-stack block uses
    /// bidirectional attention, which a recurrent mixer cannot run and a
    /// past-only sliding window cannot express; both are rejected.
    fn validate_policies(
        &self,
        hidden: &Tensor<B, 3>,
        context: &ExecutionContext<B>,
        layer: Layer,
    ) -> Result<()> {
        let [batch, tokens, _] = hidden.dims();
        let explicit = explicit_self_policy(context, layer);
        let encoder_default = explicit.is_none() && layer.stack() == Stack::Encoder;
        if matches!(&self.mixer, ModularMixer::DeltaNet(_)) {
            let unsupported = encoder_default
                || explicit.as_ref().is_some_and(|policy| {
                    !policy.causal || policy.allowed.is_some() || policy.bias.is_some()
                });
            if unsupported {
                return Err(Error::BackendCapability(
                    "DeltaNet requires ordinary causal execution; choose another mixer for bidirectional or custom connectivity".into(),
                ));
            }
        }
        if encoder_default
            && matches!(
                self.mixer.connectivity(),
                Some(AttentionConnectivity::SlidingWindow { .. })
            )
        {
            return Err(Error::BackendCapability(format!(
                "custom.layer.mixer {ENCODER_SLIDING_WINDOW}"
            )));
        }
        if let (Some(policy), Some(heads)) = (&explicit, self.mixer.attention_heads()) {
            policy.validate([batch, heads, tokens, tokens], &hidden.device())?;
            if policy
                .bias
                .as_ref()
                .is_some_and(|bias| bias.dtype() != hidden.dtype())
            {
                return Err(Error::configuration(
                    "custom.attention.bias",
                    "score bias must use the hidden-state dtype",
                ));
            }
        }
        if let Some(attention) = &self.cross_attention {
            let source = context.encoder.as_ref().ok_or_else(|| {
                Error::configuration(
                    "custom.cross_attention",
                    "this block requires context.encoder",
                )
            })?;
            if let Some(policy) = explicit_cross_policy(context, layer) {
                policy.validate(
                    [
                        batch,
                        attention.attention.query_heads,
                        tokens,
                        source.hidden.dims()[1],
                    ],
                    &hidden.device(),
                )?;
                if policy
                    .bias
                    .as_ref()
                    .is_some_and(|bias| bias.dtype() != hidden.dtype())
                {
                    return Err(Error::configuration(
                        "custom.cross_attention.bias",
                        "score bias must use the hidden-state dtype",
                    ));
                }
            }
            if source.hidden.dims()[2] != self.hidden_size
                || source.hidden.dtype() != hidden.dtype()
            {
                return Err(Error::configuration(
                    "custom.cross_attention",
                    "source width and dtype must match this block; insert an explicit source adapter otherwise",
                ));
            }
        }
        Ok(())
    }

    /// Selects numerical execution for attention branches; recurrent mixers are unchanged.
    pub(super) fn set_attention_precision(&mut self, mode: AttentionPrecisionMode) {
        if let ModularMixer::Attention(attention) = &mut self.mixer {
            attention.attention.precision_mode = mode;
        }
        if let Some(attention) = &mut self.cross_attention {
            attention.attention.precision_mode = mode;
        }
    }
}

impl<B: Backend> HiddenTransform<B> for ModularLayer<B> {
    /// Lists the actual mixer, optional cross-attention, and MLP points for this block.
    fn activation_points(&self, layer: Layer) -> Vec<PointInfo> {
        ModularLayer::activation_points(self, layer)
    }

    /// Validates the context (one device read of its positions), then applies this block.
    ///
    /// Encoder blocks default to bidirectional attention and decoder blocks to
    /// causal attention. Cross-attention consumes the supplied encoder context;
    /// the returned tensor preserves the residual axes and autodiff graph.
    fn forward(
        &self,
        hidden: Tensor<B, 3>,
        context: &ExecutionContext<B>,
        layer: Layer,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<Tensor<B, 3>> {
        let [batch, tokens, _] = hidden.dims();
        self.validate_hidden(&hidden)?;
        context.validate([batch, tokens], &hidden.device())?;
        self.validate_policies(&hidden, context, layer)?;
        let pass = StackPass::from_context(context, layer.stack(), hidden.dtype());
        Ok(ModularLayer::forward(self, hidden, &pass, layer, hooks)?.0)
    }
}

/// Rejects well-formed blocks whose mechanism has no executable kernel yet.
///
/// Errors name fields below `field`, such as `custom.causal.layers[2]`.
pub(super) fn ensure_executable(spec: &DecoderLayerSpec, field: &str) -> Result<()> {
    if spec.residual == ResidualTopology::Gated {
        return Err(Error::BackendCapability(format!(
            "{field}.residual requests a learned gate but the specification does not define its activation, shape, or initialization"
        )));
    }
    if matches!(spec.mixer, crate::architecture::MixerSpec::Mamba2(_)) {
        return Err(Error::BackendCapability(format!(
            "{field}.mixer requests Mamba2; the selective state-space kernel is not yet available as a shared module"
        )));
    }
    Ok(())
}

#[cfg(test)]
#[path = "layer_tests.rs"]
/// Standalone block construction, options, connectivity, and direction contracts.
mod tests;
