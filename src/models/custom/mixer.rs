//! Sequence mixers of a modular block: grouped-query, latent, and differential attention,
//! Gated DeltaNet, and Mamba-2.
//!
//! Each mixer maps `[batch, token, hidden]` to the same shape. Attention mixers
//! combine three independently specified ingredients:
//!
//! * projections from the shared kernels ([`GroupedQueryAttention`] or
//!   [`MultiLatentAttention`]), configured through the spec's own translation
//!   ([`AttentionSpec::grouped_query_config`], [`MlaSpec::latent_config`]);
//! * at most one position encoding: RoPE, which rotates queries and keys by
//!   the pass's positions, or a learned T5 bias added to the scores;
//! * a structural [`AttentionConnectivity`] (global, sliding window, chunked
//!   local), applied unless an explicit policy replaces it.
//!
//! Position tables and exclusion planes come from the shared [`StackPass`],
//! so every block of a stack reuses them. Padded query rows are defined as
//! zero after the output projection; their softmax rows are not masked, which
//! keeps every row well defined without special-casing fully padded rows.
//!
//! # Recurrent mixers
//!
//! Gated DeltaNet and Mamba-2 are causal scans from a zero initial state and
//! have no attention connectivity. The token mask is semantic: a padded token
//! passes no recurrent time (Mamba-2 sets its step `dt = 0`, an exact
//! identity transition) and contributes no input, so left- and right-padded
//! rows equal their unpadded scans. Mamba-2 runs the shared chunked
//! state-space-duality scan ([`Mamba2Mode::ChunkedPrefill`]) of
//! [`crate::mamba2`], whose state arithmetic is F32.
//!
//! [`Mamba2Spec`] fixes only the tensor geometry, so the remaining Mamba-2
//! settings take the Hugging Face `Mamba2Config` defaults (transformers
//! `cfa309cf`): `chunk_size = 256`, `time_step_limit = (0, inf)` on the
//! chunked scan (a no-op bound, since `softplus >= 0`), no input/output
//! projection bias (`use_bias = False`), a convolution bias
//! (`use_conv_bias = True`), `n_groups = heads / group_size`, and the block's
//! mixer-normalization epsilon for the gated group RMS norm.

use burn::{
    module::Module,
    tensor::{Tensor, backend::Backend},
};

use crate::{
    Error, Result,
    architecture::{
        AttentionOptions, AttentionSpec, DifferentialAttentionSpec, Mamba2Spec, MixerSpec, MlaSpec,
        NormalizationSpec, PositionEncodingSpec, TokenMask,
    },
    attention::{
        AttentionConnectivity, AttentionPrecisionMode, DifferentialAttention, GroupedQueryAttention,
    },
    delta_net::GatedDeltaNet,
    intervention::{AttentionPoints, Hooks, Layer, PointInfo},
    mamba2::{Mamba2, Mamba2Config, Mamba2Mode, TimeStepClamp, TimeStepLimits},
    mla::MultiLatentAttention,
    position::{RopeConfig, T5RelativePositionBias, T5RelativePositionBiasConfig, apply_rotary},
};

/// Maximum SSD chunk width of a modular Mamba-2 mixer (Hugging Face `Mamba2Config` default).
const MAMBA2_CHUNK_SIZE: usize = 256;

/// Translates a validated [`Mamba2Spec`] into the shared mixer configuration.
///
/// `hidden_size` is the residual width and `norm_epsilon` the block's mixer
/// normalization epsilon, reused by the gated group RMS norm (as DeltaNet
/// reuses it). Settings the spec does not carry take the Hugging Face
/// `Mamba2Config` defaults listed in the module documentation. The result is
/// not validated here; [`Mamba2Config::validate`] rejects, for example, a
/// convolution kernel below two, which the spec alone accepts.
pub(super) fn mamba2_config(
    spec: &Mamba2Spec,
    hidden_size: usize,
    norm_epsilon: f64,
) -> Mamba2Config {
    Mamba2Config {
        hidden_size,
        num_heads: spec.heads,
        head_dim: spec.head_dim,
        state_size: spec.state_dim,
        // `group_size` heads share one B/C group; spec validation guarantees it divides `heads`.
        groups: spec.heads.checked_div(spec.group_size).unwrap_or(0),
        convolution_kernel: spec.convolution_kernel,
        chunk_size: MAMBA2_CHUNK_SIZE,
        time_step_clamp: TimeStepClamp::chunked_only(TimeStepLimits::at_least(0.0)),
        normalization_epsilon: norm_epsilon,
        projection_bias: false,
        convolution_bias: true,
    }
}

use super::pass::StackPass;

/// Grouped-query attention with one optional position encoding and a connectivity rule.
#[derive(Module, Debug)]
pub(super) struct ModularAttention<B: Backend> {
    /// Shared projected MHA/GQA/MQA implementation.
    pub(super) attention: GroupedQueryAttention<B>,
    /// Optional learned T5 displacement bias, `[buckets, heads]`.
    pub(super) relative_bias: Option<T5RelativePositionBias<B>>,
    /// Optional rotary frequency policy, which owns no trainable parameters.
    #[module(skip)]
    rope: Option<RopeConfig>,
    /// Structural key visibility used unless an explicit policy replaces it.
    #[module(skip)]
    connectivity: AttentionConnectivity,
}

impl<B: Backend> ModularAttention<B> {
    /// Builds projections, the position encoding, and connectivity from a validated spec.
    ///
    /// `norm` is the block's mixer normalization, whose epsilon and
    /// zero-centering the optional Q/K norm reuses. A learned T5 bias starts
    /// from the reference `N(0, 1/hidden_size)` initialization.
    pub(super) fn init(
        spec: &AttentionSpec,
        options: &AttentionOptions,
        hidden_size: usize,
        norm: NormalizationSpec,
        device: &B::Device,
    ) -> Result<Self> {
        let attention = spec
            .grouped_query_config(hidden_size, options, norm)
            .init(device)?;
        Ok(Self {
            attention,
            relative_bias: relative_bias(&spec.position, spec.query_heads, hidden_size, device)?,
            rope: spec.position.rope_config(spec.head_dim)?,
            connectivity: options.connectivity,
        })
    }

    /// Structural key visibility of this layer.
    pub(super) const fn connectivity(&self) -> AttentionConnectivity {
        self.connectivity
    }

    /// Lists projection points, plus score and pattern points under eager F32 softmax.
    pub(super) fn activation_points(&self, points: AttentionPoints) -> Vec<PointInfo> {
        if self.attention.precision_mode == AttentionPrecisionMode::EagerFp32Softmax {
            points.all()
        } else {
            vec![
                points.query().info(),
                points.key().info(),
                points.value().info(),
                points.head_context().info(),
            ]
        }
    }

    /// Evaluates position-aware self-attention over the pass's complete sequence.
    pub(super) fn forward_self(
        &self,
        hidden: Tensor<B, 3>,
        pass: &StackPass<'_, B>,
        layer: Layer,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<Tensor<B, 3>> {
        let mut projection = self.attention.project_self(hidden)?;
        if let Some(rope) = self.rope {
            let angles = pass.rotary_angles(rope)?;
            (projection.query, projection.key) =
                apply_rotary(projection.query, projection.key, &angles)?;
        }
        let [batch, heads, tokens, _] = projection.query.dims();
        let position_bias = self
            .relative_bias
            .as_ref()
            .map(|bias| pass.relative_bias(bias))
            .transpose()?;
        let inputs = pass
            .self_attention(layer, self.connectivity, heads)?
            .with_position_bias(position_bias, [batch, heads, tokens, tokens]);
        let output = self.attention.attend_with_hooks(
            projection,
            inputs.mask,
            inputs.bias,
            inputs.causal,
            layer.self_attention(),
            hooks,
        )?;
        Ok(zero_padded_queries(output, pass.token_mask()))
    }

    /// Evaluates decoder-to-encoder attention against the pass's source.
    pub(super) fn forward_cross(
        &self,
        query: Tensor<B, 3>,
        pass: &StackPass<'_, B>,
        layer: Layer,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<Tensor<B, 3>> {
        let source = pass.source().ok_or_else(|| {
            Error::configuration(
                "custom.cross_attention",
                "decoder layer requires encoder hidden states",
            )
        })?;
        let projection = self.attention.project(query, source.hidden.clone())?;
        let inputs = pass.cross_attention(layer, self.attention.query_heads)?;
        let output = self.attention.attend_with_hooks(
            projection,
            inputs.mask,
            inputs.bias,
            inputs.causal,
            layer.cross_attention(),
            hooks,
        )?;
        Ok(zero_padded_queries(output, pass.token_mask()))
    }
}

/// Multi-head latent attention with one optional position encoding and a connectivity rule.
#[derive(Module, Debug)]
pub(super) struct ModularMla<B: Backend> {
    /// Shared latent projections, compressed KV normalization, and attention kernel.
    pub(super) attention: MultiLatentAttention<B>,
    /// Optional learned T5 displacement bias added before the latent-attention softmax.
    pub(super) relative_bias: Option<T5RelativePositionBias<B>>,
    /// Optional rotary policy over the decoupled rotary sub-head.
    #[module(skip)]
    rope: Option<RopeConfig>,
    /// Structural key visibility used unless an explicit policy replaces it.
    #[module(skip)]
    connectivity: AttentionConnectivity,
}

impl<B: Backend> ModularMla<B> {
    /// Builds latent attention with its score scale, RoPE or T5 bias, and connectivity.
    ///
    /// Compressed KV width, query rank, and content/rotary/value head widths
    /// are independent of the residual width; `norm` supplies the latent RMS
    /// epsilon.
    pub(super) fn init(
        spec: &MlaSpec,
        hidden_size: usize,
        norm: NormalizationSpec,
        device: &B::Device,
    ) -> Result<Self> {
        Ok(Self {
            attention: spec
                .latent_config(hidden_size, norm.epsilon())
                .init(device)?,
            relative_bias: relative_bias(&spec.position, spec.query_heads, hidden_size, device)?,
            // MLA rotates only its decoupled rotary sub-head.
            rope: spec.position.rope_config(spec.rotary_head_dim)?,
            connectivity: spec.connectivity,
        })
    }

    /// Structural key visibility of this layer.
    pub(super) const fn connectivity(&self) -> AttentionConnectivity {
        self.connectivity
    }

    /// Computes latent self-attention on `[batch, token, hidden]` residuals.
    ///
    /// The query and the shared rotary key are rotated once with the pass's
    /// cached tables; connectivity, padding, and relative bias combine before
    /// the softmax. Padded query outputs are zeroed after the output projection.
    pub(super) fn forward(
        &self,
        hidden: Tensor<B, 3>,
        pass: &StackPass<'_, B>,
        layer: Layer,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<Tensor<B, 3>> {
        let [batch, tokens, _] = hidden.dims();
        let heads = self.attention.config.num_heads;
        let mut projection = self.attention.project_with_hooks(
            hidden.clone(),
            hidden,
            layer.self_attention(),
            hooks,
        )?;
        if let Some(rope) = self.rope {
            let angles = pass.rotary_angles(rope)?;
            (projection.query_rotary, projection.rotated_key) =
                apply_rotary(projection.query_rotary, projection.rotated_key, &angles)?;
        }
        let position_bias = self
            .relative_bias
            .as_ref()
            .map(|bias| pass.relative_bias(bias))
            .transpose()?;
        let inputs = pass
            .self_attention(layer, self.connectivity, heads)?
            .with_position_bias(position_bias, [batch, heads, tokens, tokens]);
        let output = self.attention.attend_with_hooks(
            projection,
            inputs.mask,
            inputs.bias,
            inputs.causal,
            layer.self_attention(),
            hooks,
        )?;
        Ok(zero_padded_queries(output, pass.token_mask()))
    }
}

/// Differential attention with optional RoPE and a connectivity rule.
#[derive(Module, Debug)]
pub(super) struct ModularDifferential<B: Backend> {
    /// Shared paired-head projections, lambda parameters, and kernel.
    pub(super) attention: DifferentialAttention<B>,
    /// Optional rotary policy over each full head.
    #[module(skip)]
    rope: Option<RopeConfig>,
    /// Structural key visibility used unless an explicit policy replaces it.
    #[module(skip)]
    connectivity: AttentionConnectivity,
}

impl<B: Backend> ModularDifferential<B> {
    /// Builds the shared layer of a validated spec; `norm` supplies the pair-norm epsilon.
    pub(super) fn init(
        spec: &DifferentialAttentionSpec,
        hidden_size: usize,
        norm: NormalizationSpec,
        device: &B::Device,
    ) -> Result<Self> {
        Ok(Self {
            attention: spec
                .differential_config(hidden_size, norm.epsilon())
                .init(device)?,
            rope: spec.position.rope_config(spec.head_dim)?,
            connectivity: spec.connectivity,
        })
    }

    /// Structural key visibility of this layer.
    pub(super) const fn connectivity(&self) -> AttentionConnectivity {
        self.connectivity
    }

    /// Computes differential self-attention on `[batch, token, hidden]` residuals.
    ///
    /// The kernel has no additive-bias input, so an explicit policy carrying a
    /// bias is rejected (validation reports it before any callback).
    pub(super) fn forward(
        &self,
        hidden: Tensor<B, 3>,
        pass: &StackPass<'_, B>,
        layer: Layer,
    ) -> Result<Tensor<B, 3>> {
        let mut projection = self.attention.project_self(hidden)?;
        if let Some(rope) = self.rope {
            let angles = pass.rotary_angles(rope)?;
            (projection.query, projection.key) =
                apply_rotary(projection.query, projection.key, &angles)?;
        }
        let inputs = pass.self_attention(layer, self.connectivity, self.attention.query_heads)?;
        if inputs.bias.is_some() {
            return Err(Error::BackendCapability(
                "differential attention has no additive score-bias input".into(),
            ));
        }
        let output = self
            .attention
            .attend(projection, inputs.mask, inputs.causal)?;
        Ok(zero_padded_queries(output, pass.token_mask()))
    }
}

/// One executable sequence-mixing implementation.
///
/// Burn records store an enum module by variant (by index in binary
/// recorders), so variants are append-only: the original `Attention`, `Mla`,
/// and `DeltaNet` keep their positions and later mixers follow them.
/// Every variant maps `[batch, token, hidden]` residual inputs to the same shape.
#[derive(Module, Debug)]
// Burn 0.21 does not implement `Module` for `Box<M>`. Keeping modules inline is required so
// generated records can visit every parameter; the per-layer host metadata cost is negligible
// relative to even tiny projection tensors.
#[allow(clippy::large_enum_variant)]
pub(super) enum ModularMixer<B: Backend> {
    /// Grouped-query attention, in any connectivity.
    Attention(
        /// Projected attention with its positional and connectivity policy.
        ModularAttention<B>,
    ),
    /// Multi-head latent attention, reusable in either stack.
    Mla(
        /// Compressed latent attention with independently selected position encoding.
        ModularMla<B>,
    ),
    /// Gated DeltaNet linear attention (causal recurrent scan).
    DeltaNet(
        /// Gated recurrent linear-attention module.
        GatedDeltaNet<B>,
    ),
    /// Differential attention over paired query heads.
    Differential(
        /// Paired-head projections, lambda parameters, and optional RoPE.
        ModularDifferential<B>,
    ),
    /// Mamba-2 selective state-space mixing (causal chunked scan).
    Mamba2(
        /// Shared Mamba-2 mixer built from the spec's geometry and documented defaults.
        Mamba2<B>,
    ),
}

impl<B: Backend> ModularMixer<B> {
    /// Builds the mixer of a validated spec from the shared kernels.
    pub(super) fn init(
        spec: &MixerSpec,
        hidden_size: usize,
        norm: NormalizationSpec,
        device: &B::Device,
    ) -> Result<Self> {
        if let Some((attention, options)) = spec.grouped_query_attention() {
            return ModularAttention::init(attention, &options, hidden_size, norm, device)
                .map(Self::Attention);
        }
        if let Some(config) = spec.delta_net_config(hidden_size, norm.epsilon()) {
            return config.init(device).map(Self::DeltaNet);
        }
        match spec {
            MixerSpec::Mla(mla) => ModularMla::init(mla, hidden_size, norm, device).map(Self::Mla),
            MixerSpec::DifferentialAttention(differential) => {
                ModularDifferential::init(differential, hidden_size, norm, device)
                    .map(Self::Differential)
            }
            MixerSpec::Mamba2(mamba) => mamba2_config(mamba, hidden_size, norm.epsilon())
                .init(device)
                .map(Self::Mamba2),
            // Translated by `grouped_query_attention` and `delta_net_config` above.
            MixerSpec::Attention(_)
            | MixerSpec::SlidingAttention(_)
            | MixerSpec::AttentionWithOptions { .. }
            | MixerSpec::DeltaNet { .. } => unreachable!("translated before this match"),
        }
    }

    /// Structural connectivity of an attention mixer; `None` for recurrent mixers.
    pub(super) fn connectivity(&self) -> Option<AttentionConnectivity> {
        match self {
            Self::Attention(attention) => Some(attention.connectivity()),
            Self::Mla(mla) => Some(mla.connectivity()),
            Self::Differential(differential) => Some(differential.connectivity()),
            Self::DeltaNet(_) | Self::Mamba2(_) => None,
        }
    }

    /// Number of query heads of an attention mixer; `None` for recurrent mixers.
    pub(super) fn attention_heads(&self) -> Option<usize> {
        match self {
            Self::Attention(attention) => Some(attention.attention.query_heads),
            Self::Mla(mla) => Some(mla.attention.config.num_heads),
            Self::Differential(differential) => Some(differential.attention.query_heads),
            Self::DeltaNet(_) | Self::Mamba2(_) => None,
        }
    }

    /// Lists the projection, latent, and attention points this mixer emits.
    ///
    /// Differential attention, DeltaNet, and Mamba-2 expose only the surrounding mixer points.
    pub(super) fn activation_points(&self, layer: Layer) -> Vec<PointInfo> {
        match self {
            Self::Attention(attention) => attention.activation_points(layer.self_attention()),
            Self::Mla(mla) => mla.attention.activation_points(layer.self_attention()),
            Self::Differential(_) | Self::DeltaNet(_) | Self::Mamba2(_) => Vec::new(),
        }
    }

    /// Display name of a causal recurrent mixer, or `None` for attention mixers.
    pub(super) const fn recurrent_name(&self) -> Option<&'static str> {
        match self {
            Self::DeltaNet(_) => Some("DeltaNet"),
            Self::Mamba2(_) => Some("Mamba2"),
            Self::Attention(_) | Self::Mla(_) | Self::Differential(_) => None,
        }
    }

    /// Applies the mixer over the pass's complete sequence.
    pub(super) fn forward(
        &self,
        hidden: Tensor<B, 3>,
        pass: &StackPass<'_, B>,
        layer: Layer,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<Tensor<B, 3>> {
        match self {
            Self::Attention(attention) => attention.forward_self(hidden, pass, layer, hooks),
            Self::Mla(mla) => mla.forward(hidden, pass, layer, hooks),
            Self::Differential(differential) => differential.forward(hidden, pass, layer),
            Self::DeltaNet(delta) => delta.forward(hidden, pass.token_mask()),
            Self::Mamba2(mamba) => {
                mamba.forward(hidden, pass.token_mask(), Mamba2Mode::ChunkedPrefill)
            }
        }
    }
}

/// Builds the learned T5 bias requested by `position`, or `None` for other encodings.
///
/// Random initialization follows the reference `N(0, 1/hidden_size)`
/// (`initializer_factor = 1`); checkpoint weights replace it when loaded.
fn relative_bias<B: Backend>(
    position: &PositionEncodingSpec,
    heads: usize,
    hidden_size: usize,
    device: &B::Device,
) -> Result<Option<T5RelativePositionBias<B>>> {
    position
        .t5_buckets()?
        .map(|buckets| {
            T5RelativePositionBiasConfig::new(
                heads,
                buckets.num_buckets(),
                buckets.max_distance(),
                buckets.is_bidirectional(),
            )
            .with_reference_initialization(hidden_size, 1.0)
            .init(device)
        })
        .transpose()
}

/// Defines padded query rows as zero without masking every score in their softmax row.
pub(super) fn zero_padded_queries<B: Backend>(
    output: Tensor<B, 3>,
    mask: Option<&TokenMask<B>>,
) -> Tensor<B, 3> {
    match mask {
        Some(mask) => {
            let dtype = output.dtype();
            output
                * mask
                    .as_tensor()
                    .clone()
                    .float()
                    .cast(dtype)
                    .unsqueeze_dim(2)
        }
        None => output,
    }
}

#[cfg(test)]
#[path = "mixer_tests.rs"]
/// Mamba-2 translation, shared-kernel parity, padding, caching, causality, hooks, and gradients.
mod tests;
