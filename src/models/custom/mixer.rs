//! Sequence mixers of a modular block: grouped-query attention, MLA, and Gated DeltaNet.
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

use burn::{
    module::Module,
    tensor::{Tensor, backend::Backend},
};

use crate::{
    Error, Result,
    architecture::{
        AttentionOptions, AttentionSpec, MixerSpec, MlaSpec, NormalizationSpec,
        PositionEncodingSpec, TokenMask,
    },
    attention::{AttentionConnectivity, AttentionPrecisionMode, GroupedQueryAttention},
    delta_net::GatedDeltaNet,
    intervention::{AttentionPoints, Hooks, Layer, PointInfo},
    mla::MultiLatentAttention,
    position::{RopeConfig, T5RelativePositionBias, T5RelativePositionBiasConfig, apply_rotary},
};

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

/// One executable sequence-mixing implementation.
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
}

impl<B: Backend> ModularMixer<B> {
    /// Builds the mixer of a validated spec; Mamba2 has no shared kernel yet.
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
            _ => Err(Error::BackendCapability(
                "the Mamba2 mixer is describable but has no shared executable kernel yet".into(),
            )),
        }
    }

    /// Structural connectivity of an attention mixer; `None` for recurrent mixers.
    pub(super) fn connectivity(&self) -> Option<AttentionConnectivity> {
        match self {
            Self::Attention(attention) => Some(attention.connectivity()),
            Self::Mla(mla) => Some(mla.connectivity()),
            Self::DeltaNet(_) => None,
        }
    }

    /// Number of query heads of an attention mixer; `None` for recurrent mixers.
    pub(super) fn attention_heads(&self) -> Option<usize> {
        match self {
            Self::Attention(attention) => Some(attention.attention.query_heads),
            Self::Mla(mla) => Some(mla.attention.config.num_heads),
            Self::DeltaNet(_) => None,
        }
    }

    /// Lists the projection, latent, and attention points this mixer emits.
    pub(super) fn activation_points(&self, layer: Layer) -> Vec<PointInfo> {
        match self {
            Self::Attention(attention) => attention.activation_points(layer.self_attention()),
            Self::Mla(mla) => mla.attention.activation_points(layer.self_attention()),
            Self::DeltaNet(_) => Vec::new(),
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
            Self::DeltaNet(delta) => delta.forward(hidden, pass.token_mask()),
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
