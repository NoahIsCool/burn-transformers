//! Explicit execution boundaries for composing transformer experiments.
//!
//! A [`HiddenState`] carries a tensor together with its positions and padding.
//! `embed`, `run_layers`, and `normalize` make the boundaries visible: running
//! blocks never silently reapplies embeddings or final normalization. Parameters
//! belong to ordinary Burn modules; these execution values are not checkpointed.
//!
//! Hidden tensors have axes `[batch, token, channel]`; positions and token
//! validity have `[batch, token]` axes. Attention policies operate on broadcastable
//! `[batch, head, query, key]` planes. User-supplied allowed links use `true`, while
//! kernel exclusions invert that convention and combine it with padding. Layer
//! policies may choose causal, bidirectional, or custom attention without changing
//! parameter ownership or silently converting tensor dtype/device.

use std::ops::Range;

use burn::{
    module::Module,
    tensor::{Bool, Int, Tensor, backend::Backend},
};

use crate::{
    EncoderOutput, Error, Result, TokenMask,
    intervention::{Hooks, Layer, PointInfo},
};

mod block;
pub(crate) mod bounds;
mod rotary;
pub use block::{ResidualBlock, WidthAdapter};
pub use rotary::RotaryAttention;

/// Attention connectivity and additive bias for a stateless operation.
///
/// An explicit `allowed` matrix uses **true = allowed**. It is intersected with
/// causality and valid-token padding, never substituted for those constraints.
#[derive(Clone, Debug)]
pub struct AttentionPolicy<B: Backend> {
    /// Whether keys later than the aligned query position are excluded.
    pub causal: bool,
    /// Optional `[batch, head, query, key]` connectivity, with singleton broadcast axes.
    pub allowed: Option<Tensor<B, 4, Bool>>,
    /// Optional additive score bias with singleton broadcast axes.
    pub bias: Option<Tensor<B, 4>>,
}

impl<B: Backend> AttentionPolicy<B> {
    /// Select autoregressive visibility.
    pub fn causal() -> Self {
        Self {
            causal: true,
            allowed: None,
            bias: None,
        }
    }

    /// Allow future and earlier tokens, subject to padding.
    pub fn bidirectional() -> Self {
        Self {
            causal: false,
            allowed: None,
            bias: None,
        }
    }

    /// Specify all allowed links, without an additional causal constraint.
    pub fn custom(allowed: Tensor<B, 4, Bool>) -> Self {
        Self {
            allowed: Some(allowed),
            ..Self::bidirectional()
        }
    }

    /// Add a bias without changing connectivity.
    pub fn with_bias(mut self, bias: Tensor<B, 4>) -> Self {
        self.bias = Some(bias);
        self
    }

    /// Restrict this policy to explicitly allowed links without changing causality.
    pub fn with_allowed(mut self, allowed: Tensor<B, 4, Bool>) -> Self {
        self.allowed = Some(allowed);
        self
    }

    /// Expand a validated bias to the primitive's exact attention-plane layout.
    pub fn expanded_bias(
        &self,
        shape: [usize; 4],
        device: &B::Device,
    ) -> Result<Option<Tensor<B, 4>>> {
        self.validate(shape, device)?;
        Ok(self.bias.as_ref().map(|bias| bias.clone().expand(shape)))
    }

    /// Validate broadcast extents and devices before attention executes.
    pub fn validate(&self, shape: [usize; 4], device: &B::Device) -> Result<()> {
        if shape.contains(&0) {
            return Err(Error::configuration(
                "attention.policy",
                "attention extents must be positive",
            ));
        }
        if let Some(allowed) = &self.allowed {
            validate_plane("attention.allowed", allowed.dims(), shape)?;
            if allowed.device() != *device {
                return Err(Error::configuration(
                    "attention.allowed",
                    "connectivity must be on the attention device",
                ));
            }
        }
        if let Some(bias) = &self.bias {
            validate_plane("attention.bias", bias.dims(), shape)?;
            if bias.device() != *device {
                return Err(Error::configuration(
                    "attention.bias",
                    "bias must be on the attention device",
                ));
            }
        }
        Ok(())
    }

    /// Combine explicit connectivity with an existing exclusion mask.
    ///
    /// The returned primitive mask uses **true = excluded**. Causality is kept
    /// in [`Self::causal`] for the attention operation to apply exactly once.
    pub fn exclusion_mask(
        &self,
        padding: Option<Tensor<B, 4, Bool>>,
        shape: [usize; 4],
        device: &B::Device,
    ) -> Result<Option<Tensor<B, 4, Bool>>> {
        self.validate(shape, device)?;
        if let Some(mask) = &padding {
            validate_plane("attention.padding", mask.dims(), shape)?;
            if mask.device() != *device {
                return Err(Error::configuration(
                    "attention.padding",
                    "padding must be on the attention device",
                ));
            }
        }
        let padding = padding.map(|mask| mask.int().bool());
        Ok(match (&self.allowed, padding) {
            (Some(allowed), Some(padding)) => {
                Some(allowed.clone().int().bool().bool_not().bool_or(padding))
            }
            (Some(allowed), None) => Some(allowed.clone().int().bool().bool_not()),
            (None, padding) => padding,
        }
        .map(|mask| mask.expand(shape)))
    }
}

/// Checks broadcasting into `[batch, head, query_token, key_token]` one axis at a time.
fn validate_plane(field: &str, actual: [usize; 4], expected: [usize; 4]) -> Result<()> {
    if !bounds::broadcasts_to(&actual, &expected) {
        return Err(Error::shape(
            field,
            format!("broadcastable to {expected:?}"),
            format!("{actual:?}"),
        ));
    }
    Ok(())
}

/// Source states supplied to cross-attention without a vocabulary head.
#[derive(Clone, Debug)]
pub struct EncoderContext<B: Backend> {
    /// Source representations, `[batch, source_token, channel]`.
    pub hidden: Tensor<B, 3>,
    /// Source positions, `[batch, source_token]`.
    pub positions: Tensor<B, 2, Int>,
    /// Source validity; true means a real token.
    pub token_mask: Option<TokenMask<B>>,
}

/// Positions, padding, attention choices, and optional source context for a pass.
///
/// # Validation cost
///
/// Position *values* live on the device. [`Self::new`] and [`Self::validate`]
/// read their range back to the host in one synchronization; blocks call the
/// sync-free [`Self::validate_layout`] instead, so a composed stack pays for
/// value checks once per pass rather than once per block. The fields are public
/// for experiments that rewrite a context directly: after assigning
/// `positions`, `evaluated_context`, or `encoder`, call [`Self::validate`] again.
#[derive(Clone, Debug)]
pub struct ExecutionContext<B: Backend> {
    /// Absolute or architecture-defined logical positions, `[batch, token]`.
    ///
    /// Invariant: every value lies in `0..evaluated_context`.
    pub positions: Tensor<B, 2, Int>,
    /// Token validity; absence means all positions are valid.
    pub token_mask: Option<TokenMask<B>>,
    /// Evaluated positional extent used by context-dependent position encodings.
    ///
    /// This is the largest logical position plus one; dynamic RoPE scaling reads it.
    pub evaluated_context: usize,
    /// Optional default override; absence retains the architecture's policy.
    pub attention: Option<AttentionPolicy<B>>,
    /// Per-layer overrides, taking precedence over [`Self::attention`].
    pub layer_attention: Vec<(Layer, AttentionPolicy<B>)>,
    /// Optional default decoder cross-attention override, separate from self-attention.
    pub cross_attention: Option<AttentionPolicy<B>>,
    /// Per-layer cross-attention overrides.
    pub layer_cross_attention: Vec<(Layer, AttentionPolicy<B>)>,
    /// Optional source for decoder cross-attention.
    pub encoder: Option<EncoderContext<B>>,
}

impl<B: Backend> ExecutionContext<B> {
    /// Construct a context from nonnegative positions and token validity.
    ///
    /// The evaluated context becomes `max(positions) + 1`. Reading the position
    /// range synchronizes the device once.
    pub fn new(positions: Tensor<B, 2, Int>, token_mask: Option<TokenMask<B>>) -> Result<Self> {
        let shape = positions.dims();
        if shape.contains(&0) {
            return Err(Error::configuration(
                "execution.positions",
                "positions must be nonempty",
            ));
        }
        let [low, high] = read_int_scalars(
            vec![positions.clone().min(), positions.clone().max()],
            "execution positions",
        )?
        .try_into()
        .map_err(|_| Error::BackendCapability("position range readback is incomplete".into()))?;
        if low < 0 {
            return Err(Error::configuration(
                "execution.positions",
                "positions must be nonnegative",
            ));
        }
        let evaluated_context = bounds::extent_after(high).ok_or_else(|| {
            Error::configuration("execution.positions", "positional extent overflows usize")
        })?;
        let device = positions.device();
        let context = Self::unchecked(positions, token_mask, evaluated_context);
        context.validate_layout(shape, &device)?;
        Ok(context)
    }

    /// Construct positions `0..tokens` for every row, with no padding.
    ///
    /// The range is known on the host, so no device synchronization occurs.
    /// Padded batches whose positions should skip padding need explicit
    /// positions and [`Self::new`].
    pub fn contiguous(batch: usize, tokens: usize, device: &B::Device) -> Result<Self> {
        if batch == 0 || tokens == 0 {
            return Err(Error::configuration(
                "execution.positions",
                format!("positions must be nonempty, got [{batch}, {tokens}]"),
            ));
        }
        let last = i64::try_from(tokens)
            .map_err(|_| Error::configuration("execution.positions", "token count exceeds i64"))?;
        let positions = Tensor::<B, 1, Int>::arange(0..last, device)
            .reshape([1, tokens])
            .expand([batch, tokens]);
        Ok(Self::unchecked(positions, None, tokens))
    }

    /// Assembles a context without validation; callers establish the position invariant.
    fn unchecked(
        positions: Tensor<B, 2, Int>,
        token_mask: Option<TokenMask<B>>,
        evaluated_context: usize,
    ) -> Self {
        Self {
            positions,
            token_mask,
            evaluated_context,
            attention: None,
            layer_attention: Vec::new(),
            cross_attention: None,
            layer_cross_attention: Vec::new(),
            encoder: None,
        }
    }

    /// Attach a decoder cross-attention source after checking its layout.
    ///
    /// Source positions are checked for nonnegativity with one device
    /// synchronization, matching [`Self::validate`].
    pub fn with_encoder(mut self, encoder: EncoderContext<B>) -> Result<Self> {
        self.encoder = Some(encoder);
        self.validate(self.positions.dims(), &self.positions.device())?;
        Ok(self)
    }

    /// Override attention for the whole pass.
    pub fn with_attention(mut self, policy: AttentionPolicy<B>) -> Self {
        self.attention = Some(policy);
        self
    }

    /// Override one layer; a repeated address replaces its earlier policy.
    pub fn with_layer_attention(mut self, layer: Layer, policy: AttentionPolicy<B>) -> Self {
        self.layer_attention
            .retain(|(existing, _)| *existing != layer);
        self.layer_attention.push((layer, policy));
        self
    }

    /// Resolve one layer's policy while preserving its default when no override exists.
    pub fn attention_for(&self, layer: Layer, default_causal: bool) -> AttentionPolicy<B> {
        self.layer_attention
            .iter()
            .rev()
            .find(|(candidate, _)| *candidate == layer)
            .map(|(_, policy)| policy.clone())
            .or_else(|| self.attention.clone())
            .unwrap_or_else(|| {
                if default_causal {
                    AttentionPolicy::causal()
                } else {
                    AttentionPolicy::bidirectional()
                }
            })
    }

    /// Override cross-attention for all decoder layers.
    pub fn with_cross_attention(mut self, policy: AttentionPolicy<B>) -> Self {
        self.cross_attention = Some(policy);
        self
    }

    /// Override cross-attention at one decoder layer.
    pub fn with_layer_cross_attention(mut self, layer: Layer, policy: AttentionPolicy<B>) -> Self {
        self.layer_cross_attention
            .retain(|(existing, _)| *existing != layer);
        self.layer_cross_attention.push((layer, policy));
        self
    }

    /// Resolve cross-attention connectivity; its default permits every valid source token.
    pub fn cross_attention_for(&self, layer: Layer) -> AttentionPolicy<B> {
        self.layer_cross_attention
            .iter()
            .rev()
            .find(|(candidate, _)| *candidate == layer)
            .map(|(_, policy)| policy.clone())
            .or_else(|| self.cross_attention.clone())
            .unwrap_or_else(AttentionPolicy::bidirectional)
    }

    /// Validate layouts and position values for a `[batch, token]` pass.
    ///
    /// Runs [`Self::validate_layout`], then reads the target position range (and
    /// the source minimum, when an encoder is attached) in one device
    /// synchronization. Call this once per pass, or after editing public fields.
    pub fn validate(&self, shape: [usize; 2], device: &B::Device) -> Result<()> {
        self.validate_layout(shape, device)?;
        self.validate_position_values()
    }

    /// Checks that target positions lie in `0..evaluated_context` and source positions are nonnegative.
    ///
    /// All extrema are concatenated and read back together: one synchronization
    /// regardless of whether an encoder is attached.
    fn validate_position_values(&self) -> Result<()> {
        let mut extrema = vec![self.positions.clone().min(), self.positions.clone().max()];
        if let Some(encoder) = &self.encoder {
            extrema.push(encoder.positions.clone().min());
        }
        let values = read_int_scalars(extrema, "execution positions")?;
        let (low, high) = (values[0], values[1]);
        if !bounds::range_within(low, high, self.evaluated_context) {
            return Err(Error::configuration(
                "execution.positions",
                format!(
                    "positions must lie within the nonnegative evaluated context 0..{}, found {low}..={high}",
                    self.evaluated_context
                ),
            ));
        }
        if values.get(2).is_some_and(|source_low| *source_low < 0) {
            return Err(Error::configuration(
                "execution.encoder.positions",
                "source positions must be nonnegative",
            ));
        }
        Ok(())
    }

    /// Validate shapes, devices, and the evaluated extent without reading device values.
    ///
    /// Blocks call this on every invocation; it inspects only tensor metadata,
    /// so it never synchronizes the device. Position values are covered by
    /// [`Self::new`] or [`Self::validate`].
    pub fn validate_layout(&self, shape: [usize; 2], device: &B::Device) -> Result<()> {
        if shape.contains(&0) || self.positions.dims() != shape {
            return Err(Error::shape(
                "execution.positions",
                format!("nonempty {shape:?}"),
                format!("{:?}", self.positions.dims()),
            ));
        }
        if self.positions.device() != *device {
            return Err(Error::configuration(
                "execution.positions",
                "positions and hidden states must share a device",
            ));
        }
        if self.evaluated_context == 0 {
            return Err(Error::configuration(
                "execution.evaluated_context",
                "must be positive",
            ));
        }
        if let Some(mask) = &self.token_mask {
            if mask.as_tensor().dims() != shape {
                return Err(Error::shape(
                    "execution.token_mask",
                    format!("{shape:?}"),
                    format!("{:?}", mask.as_tensor().dims()),
                ));
            }
            if mask.as_tensor().device() != *device {
                return Err(Error::configuration(
                    "execution.token_mask",
                    "padding and hidden states must share a device",
                ));
            }
        }
        if let Some(encoder) = &self.encoder {
            let [batch, tokens, width] = encoder.hidden.dims();
            if batch != shape[0]
                || tokens == 0
                || width == 0
                || encoder.positions.dims() != [batch, tokens]
            {
                return Err(Error::configuration(
                    "execution.encoder",
                    "source hidden states and positions must have matching nonempty batch/token axes",
                ));
            }
            if encoder.hidden.device() != *device || encoder.positions.device() != *device {
                return Err(Error::configuration(
                    "execution.encoder",
                    "source context must share the target device",
                ));
            }
            if let Some(mask) = &encoder.token_mask
                && (mask.as_tensor().dims() != [batch, tokens]
                    || mask.as_tensor().device() != *device)
            {
                return Err(Error::configuration(
                    "execution.encoder.token_mask",
                    "source padding must match source shape and device",
                ));
            }
        }
        Ok(())
    }
}

// Host readback of device-side integer extrema.

/// Reads one-element integer tensors from one device with a single synchronization.
///
/// Each input is a `[1]` tensor such as `positions.min()`. Concatenating them on
/// the device before the readback replaces one blocking transfer per scalar with
/// one in total. Integer storage may be I32 or I64; values widen to `i64`.
/// Readback failures become [`Error::BackendCapability`] rather than panics.
pub(crate) fn read_int_scalars<B: Backend>(
    scalars: Vec<Tensor<B, 1, Int>>,
    purpose: &str,
) -> Result<Vec<i64>> {
    let expected = scalars.len();
    if expected == 0 {
        return Ok(Vec::new());
    }
    let data = Tensor::cat(scalars, 0)
        .try_into_data()
        .map_err(|error| Error::BackendCapability(format!("cannot read {purpose}: {error}")))?;
    let values = data.iter::<i64>().collect::<Vec<_>>();
    if values.len() != expected {
        return Err(Error::BackendCapability(format!(
            "cannot read {purpose}: expected {expected} scalars, found {}",
            values.len()
        )));
    }
    Ok(values)
}

/// A representation at an explicit block and normalization boundary.
#[derive(Clone, Debug)]
pub struct HiddenState<B: Backend> {
    /// Current `[batch, token, channel]` representation; adapters may change channel width.
    pub hidden: Tensor<B, 3>,
    /// Positions, validity, and attention policy accompanying the representation.
    pub context: ExecutionContext<B>,
    /// Index of the next block in the current stack.
    pub next_layer: usize,
    /// Whether final stack normalization has already been applied.
    pub normalized: bool,
}

impl<B: Backend> HiddenState<B> {
    /// Start a raw stack representation before layer zero.
    pub fn new(hidden: Tensor<B, 3>, context: ExecutionContext<B>) -> Result<Self> {
        let state = Self {
            hidden,
            context,
            next_layer: 0,
            normalized: false,
        };
        state.validate(state.hidden.dims()[2])?;
        Ok(state)
    }

    /// Validate representation width, positions, padding, and devices before execution.
    pub fn validate(&self, width: usize) -> Result<()> {
        let [batch, tokens, actual_width] = self.hidden.dims();
        if width == 0 || actual_width != width || batch == 0 || tokens == 0 {
            return Err(Error::shape(
                "execution.hidden",
                format!("nonempty [batch, token, {width}]"),
                format!("{:?}", self.hidden.dims()),
            ));
        }
        self.context
            .validate([batch, tokens], &self.hidden.device())
    }

    /// Reject skipped/repeated blocks, out-of-range endpoints, and normalized input.
    pub fn validate_range(&self, range: Range<usize>, total: usize) -> Result<()> {
        if self.normalized {
            return Err(Error::configuration(
                "execution.range",
                "blocks require raw states before final normalization",
            ));
        }
        if !bounds::follows_cursor(self.next_layer, range.start, range.end, total) {
            return Err(Error::configuration(
                "execution.range",
                format!(
                    "expected range starting at {} with end at most {total}, got {range:?}",
                    self.next_layer
                ),
            ));
        }
        Ok(())
    }

    /// Export a normalized source representation for existing seq2seq consumers.
    pub fn into_encoder_output(self) -> Result<EncoderOutput<B>> {
        self.validate(self.hidden.dims()[2])?;
        if !self.normalized {
            return Err(Error::configuration(
                "execution.encoder",
                "normalize the source stack before exporting encoder output",
            ));
        }
        Ok(EncoderOutput {
            hidden_state: self.hidden,
            token_mask: self.context.token_mask,
            hidden_states: Vec::new(),
            attentions: Vec::new(),
        })
    }

    /// Export normalized source states with their positions for another decoder.
    pub fn into_encoder_context(self) -> Result<EncoderContext<B>> {
        self.validate(self.hidden.dims()[2])?;
        if !self.normalized {
            return Err(Error::configuration(
                "execution.encoder",
                "normalize the source stack before exporting encoder context",
            ));
        }
        Ok(EncoderContext {
            hidden: self.hidden,
            positions: self.context.positions,
            token_mask: self.context.token_mask,
        })
    }
}

/// An open extension point for a mixer, feed-forward operation, or custom processor.
///
/// Implement this trait on an ordinary registered Burn module. The caller owns
/// the outer hook scope; implementations emit their supported points with `apply`.
pub trait HiddenTransform<B: Backend>: Module<B> {
    /// List additional points emitted by this module at the supplied layer.
    fn activation_points(&self, _layer: Layer) -> Vec<PointInfo> {
        Vec::new()
    }

    /// Transform hidden states using explicit context and a semantic layer address.
    fn forward(
        &self,
        hidden: Tensor<B, 3>,
        context: &ExecutionContext<B>,
        layer: Layer,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<Tensor<B, 3>>;
}

#[cfg(all(test, feature = "flex"))]
#[path = "experiment_tests.rs"]
mod tests;
