//! Parameterized building blocks shared by modular causal and encoder-decoder graphs.
//!
//! A block is decomposed into a sequence mixer, a feed-forward operator, and normalization
//! layouts around each residual branch. This mirrors the mathematical factorization
//! `x' = R_m(x, M(N_m(x)))`, `y = R_f(x', F(N_f(x')))`. Keeping the pieces independent makes
//! heterogeneous schedules possible without model-name checks in shared kernels.

use burn::{
    module::Module,
    nn::{LayerNorm, LayerNormConfig},
    tensor::{Bool, Int, Tensor, TensorData, backend::Backend, module::embedding},
};

use crate::{
    CausalLmInput, CausalLmOutput, Error, ForwardOptions, Result,
    architecture::{
        AttentionSpec, DecoderLayerSpec, DecoderSpec, FeedForwardSpec, MixerSpec, MlaSpec,
        NormPlacement, NormalizationSpec, PositionEncodingSpec, ResidualTopology,
        RopeScaling as SpecRopeScaling, RouterDiagnostics, TokenMask,
    },
    attention::{
        AttentionOutputGate, AttentionPrecisionMode, GroupedQueryAttention,
        GroupedQueryAttentionConfig, padding_attention_mask,
    },
    delta_net::{GatedDeltaNet, GatedDeltaNetConfig},
    experiment::{AttentionPolicy, ExecutionContext, HiddenTransform},
    feed_forward::{GatedFeedForward, GatedFeedForwardConfig, apply_activation},
    intervention::{AttentionPoints, Hooks, Layer, PointInfo, Stack},
    mla::{MultiLatentAttention, MultiLatentAttentionConfig},
    moe::{SparseMoe, SparseMoeConfig, SparseMoeOutput, load_balancing_loss},
    normalization::{RmsNorm, RmsNormConfig},
    position::{
        RopeConfig, RopeScaling, T5RelativePositionBias, T5RelativePositionBiasConfig, YarnConfig,
        apply_rotary, rotary_angles,
    },
};

use crate::models::shared::{
    DeferredChecks, InputContract, InputNames, PaddedPositions, defer_input,
};
pub(super) use crate::models::shared::{select_hidden, validate_position_values};

/// Whether a modular stack applies causal or bidirectional self-attention.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum AttentionDirection {
    /// Decoder attention may observe only its causal prefix.
    Causal,
    /// Encoder attention may observe all valid source positions.
    Bidirectional,
}

impl AttentionDirection {
    /// Returns the causal flag consumed by Burn's attention primitive.
    const fn is_causal(self) -> bool {
        matches!(self, Self::Causal)
    }
}

/// Runtime normalization supporting both scale-only RMS and mean-subtracting LayerNorm.
#[derive(Module, Debug)]
pub enum ModularNorm<B: Backend> {
    /// Crate-native RMS normalization, including zero-centered checkpoint weights.
    Rms(
        /// RMS normalization parameters and epsilon for the residual width.
        RmsNorm<B>,
    ),
    /// Burn LayerNorm with an optional learned additive bias.
    Layer(
        /// Mean-centering layer normalization parameters and epsilon.
        LayerNorm<B>,
    ),
}

impl<B: Backend> ModularNorm<B> {
    /// Initializes the exact normalization convention requested by a specification.
    pub fn init(spec: NormalizationSpec, hidden_size: usize, device: &B::Device) -> Result<Self> {
        spec.validate("custom.normalization")?;
        if hidden_size == 0 {
            return Err(Error::configuration(
                "custom.normalization.hidden_size",
                "must be non-zero",
            ));
        }
        match spec {
            NormalizationSpec::Rms {
                epsilon,
                zero_centered,
            } => RmsNormConfig::new(hidden_size)
                .with_epsilon(epsilon)
                .with_zero_centered(zero_centered)
                .init(device)
                .map(Self::Rms),
            NormalizationSpec::Layer { epsilon, bias } => Ok(Self::Layer(
                LayerNormConfig::new(hidden_size)
                    .with_epsilon(epsilon)
                    .with_bias(bias)
                    .init(device),
            )),
        }
    }

    /// Normalizes the final channel of a `[batch, sequence, hidden]` tensor.
    pub fn forward(&self, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
        match self {
            Self::Rms(norm) => norm.forward(hidden),
            Self::Layer(norm) => norm.forward(hidden),
        }
    }
}

/// Before/after normalization parameters for one residual sublayer.
#[derive(Module, Debug)]
pub struct NormLayout<B: Backend> {
    /// Independent pre-sublayer normalization when requested.
    before: Option<ModularNorm<B>>,
    /// Independent post-sublayer normalization when requested.
    after: Option<ModularNorm<B>>,
    /// Algebraic placement retained outside the parameter record.
    #[module(skip)]
    placement: NormPlacement,
    /// Explicitly bypasses normalization for an attention/MLP-only experiment.
    #[module(skip)]
    identity: bool,
}

impl<B: Backend> NormLayout<B> {
    /// Allocates exactly the normalization parameters implied by one placement.
    pub fn init(
        spec: NormalizationSpec,
        placement: NormPlacement,
        hidden_size: usize,
        device: &B::Device,
    ) -> Result<Self> {
        let before = matches!(placement, NormPlacement::Pre | NormPlacement::Sandwich)
            .then(|| ModularNorm::init(spec, hidden_size, device))
            .transpose()?;
        let after = matches!(placement, NormPlacement::Post | NormPlacement::Sandwich)
            .then(|| ModularNorm::init(spec, hidden_size, device))
            .transpose()?;
        Ok(Self {
            before,
            after,
            placement,
            identity: false,
        })
    }

    /// Creates an explicit unnormalized residual branch, `x + F(x)`.
    pub fn identity() -> Self {
        Self {
            before: None,
            after: None,
            placement: NormPlacement::Pre,
            identity: true,
        }
    }

    /// Applies the optional normalization immediately before a sublayer.
    pub fn prepare(&self, hidden: Tensor<B, 3>) -> Result<Tensor<B, 3>> {
        if self.identity {
            return Ok(hidden);
        }
        match (&self.before, self.placement) {
            (Some(norm), NormPlacement::Pre | NormPlacement::Sandwich) => Ok(norm.forward(hidden)),
            (None, NormPlacement::Post) => Ok(hidden),
            _ => Err(Error::configuration(
                "custom.norm_layout",
                "instantiated pre-normalization parameters disagree with their placement",
            )),
        }
    }

    /// Combines a sublayer value with its residual according to the requested norm topology.
    ///
    /// `Pre` computes `x + F(N(x))`; conventional `Post` computes `N(x + F(x))`;
    /// sandwich normalization computes `x + N_post(F(N_pre(x)))`, the form used by modern
    /// sandwich-normalized decoders such as Muse Glimmer.
    pub fn finish(&self, residual: Tensor<B, 3>, mixed: Tensor<B, 3>) -> Result<Tensor<B, 3>> {
        if self.identity {
            return Ok(residual + mixed);
        }
        match (&self.after, self.placement) {
            (None, NormPlacement::Pre) => Ok(residual + mixed),
            (Some(norm), NormPlacement::Post) => Ok(norm.forward(residual + mixed)),
            (Some(norm), NormPlacement::Sandwich) => Ok(residual + norm.forward(mixed)),
            _ => Err(Error::configuration(
                "custom.norm_layout",
                "instantiated post-normalization parameters disagree with their placement",
            )),
        }
    }
}

/// Position-aware projected attention used by an arbitrary modular layer.
#[derive(Module, Debug)]
struct ModularAttention<B: Backend> {
    /// Shared projected MHA/GQA/MQA implementation.
    attention: GroupedQueryAttention<B>,
    /// Optional learned displacement bias.
    relative_bias: Option<T5RelativePositionBias<B>>,
    /// Optional rotary frequency policy, which owns no trainable parameters.
    #[module(skip)]
    rope: Option<RopeConfig>,
    /// Optional number of visible trailing keys.
    #[module(skip)]
    sliding_window: Option<usize>,
}

impl<B: Backend> ModularAttention<B> {
    /// Initializes shared projections and exactly one requested positional mechanism.
    fn init(
        spec: &AttentionSpec,
        hidden_size: usize,
        norm: NormalizationSpec,
        sliding_window: Option<usize>,
        device: &B::Device,
    ) -> Result<Self> {
        let mut config = GroupedQueryAttentionConfig::new(
            hidden_size,
            spec.query_heads,
            spec.key_value_heads,
            spec.head_dim,
        );
        config.query_key_norm_epsilon = spec.query_key_norm.then_some(norm_epsilon(norm));
        config.query_key_norm_zero_centered = matches!(
            norm,
            NormalizationSpec::Rms {
                zero_centered: true,
                ..
            }
        );
        config.output_gate = if spec.output_gate {
            AttentionOutputGate::Sigmoid
        } else {
            AttentionOutputGate::None
        };
        // T5-family attention intentionally omits the usual 1/sqrt(d) score scaling.
        if matches!(&spec.position, PositionEncodingSpec::T5RelativeBias { .. }) {
            config.scale = Some(1.0);
        }
        let (rope, relative_bias) = match &spec.position {
            PositionEncodingSpec::None => (None, None),
            PositionEncodingSpec::Rope {
                theta,
                rotary_dim,
                scaling,
            } => (
                Some(build_rope(spec.head_dim, *rotary_dim, *theta, *scaling)?),
                None,
            ),
            PositionEncodingSpec::T5RelativeBias {
                buckets,
                max_distance,
                bidirectional,
            } => (
                None,
                Some(
                    T5RelativePositionBiasConfig::new(
                        spec.query_heads,
                        *buckets,
                        *max_distance,
                        *bidirectional,
                    )
                    .init(device)?,
                ),
            ),
        };
        Ok(Self {
            attention: config.init(device)?,
            relative_bias,
            rope,
            sliding_window,
        })
    }

    /// Advertises score planes only when the configured arithmetic can expose them.
    fn activation_points(&self, points: AttentionPoints) -> Vec<PointInfo> {
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

    /// Evaluates position-aware self-attention over a complete stateless sequence.
    #[allow(clippy::too_many_arguments)]
    fn forward_self(
        &self,
        hidden: Tensor<B, 3>,
        positions: Tensor<B, 2, Int>,
        token_mask: Option<&TokenMask<B>>,
        direction: AttentionDirection,
        evaluated_context: usize,
        layer: Layer,
        hooks: &mut Hooks<'_, B>,
        policy: Option<&AttentionPolicy<B>>,
    ) -> Result<Tensor<B, 3>> {
        let mut projection = self.attention.project_self(hidden)?;
        if let Some(rope) = self.rope {
            let angles = rotary_angles(rope, positions.clone(), evaluated_context)?;
            (projection.query, projection.key) =
                apply_rotary(projection.query, projection.key, &angles)?;
        }
        let [batch, _, sequence, _] = projection.query.dims();
        let padding = padding_attention_mask(
            None,
            token_mask,
            batch,
            self.attention.query_heads,
            sequence,
            sequence,
        )?;
        let mask = add_sliding_mask(
            padding,
            token_mask,
            batch,
            self.attention.query_heads,
            sequence,
            if policy.is_some() {
                None
            } else {
                self.sliding_window
            },
            &projection.query.device(),
        );
        let bias = self
            .relative_bias
            .as_ref()
            .map(|bias| relative_bias_from_positions(bias, positions))
            .transpose()?;
        let (mask, bias, causal) = resolve_attention_policy(
            policy,
            mask,
            bias,
            [batch, self.attention.query_heads, sequence, sequence],
            &projection.query.device(),
            direction.is_causal(),
        )?;
        let output = self.attention.attend_with_hooks(
            projection,
            mask,
            bias,
            causal,
            layer.self_attention(),
            hooks,
        )?;
        Ok(mask_padded_queries(output, token_mask))
    }

    /// Evaluates ordinary decoder-to-encoder cross attention.
    #[allow(clippy::too_many_arguments)]
    fn forward_cross(
        &self,
        query: Tensor<B, 3>,
        source: Tensor<B, 3>,
        query_mask: Option<&TokenMask<B>>,
        source_mask: Option<&TokenMask<B>>,
        layer: Layer,
        hooks: &mut Hooks<'_, B>,
        policy: Option<&AttentionPolicy<B>>,
    ) -> Result<Tensor<B, 3>> {
        let projection = self.attention.project(query, source)?;
        let [batch, _, query_sequence, _] = projection.query.dims();
        let key_sequence = projection.key.dims()[2];
        let mask = padding_attention_mask(
            None,
            source_mask,
            batch,
            self.attention.query_heads,
            query_sequence,
            key_sequence,
        )?;
        let (mask, bias, causal) = resolve_attention_policy(
            policy,
            mask,
            None,
            [
                batch,
                self.attention.query_heads,
                query_sequence,
                key_sequence,
            ],
            &projection.query.device(),
            false,
        )?;
        let output = self.attention.attend_with_hooks(
            projection,
            mask,
            bias,
            causal,
            layer.cross_attention(),
            hooks,
        )?;
        Ok(mask_padded_queries(output, query_mask))
    }
}

/// Shared latent attention with independently selected position encoding.
#[derive(Module, Debug)]
struct ModularMla<B: Backend> {
    /// Shared latent projections, compressed KV normalization, and attention kernel.
    attention: MultiLatentAttention<B>,
    /// Optional learned relative-position bias added before latent-attention softmax.
    relative_bias: Option<T5RelativePositionBias<B>>,
    /// Optional rotary encoding applied to the latent attention rotary channels.
    #[module(skip)]
    rope: Option<RopeConfig>,
}

impl<B: Backend> ModularMla<B> {
    /// Builds shared latent attention with optional RoPE or learned T5 relative bias.
    ///
    /// Compressed KV width, query rank, and content/rotary/value head widths
    /// remain independent of the residual width; normalization supplies its epsilon.
    fn init(
        spec: &MlaSpec,
        hidden_size: usize,
        norm: NormalizationSpec,
        device: &B::Device,
    ) -> Result<Self> {
        let mut config = MultiLatentAttentionConfig::new(
            hidden_size,
            spec.query_heads,
            spec.content_head_dim,
            spec.rotary_head_dim,
            spec.value_head_dim,
            spec.key_value_lora_rank,
            spec.query_lora_rank,
        );
        config.rms_norm_epsilon = norm_epsilon(norm);
        let (rope, relative_bias) = match &spec.position {
            PositionEncodingSpec::None => (None, None),
            PositionEncodingSpec::Rope {
                theta,
                rotary_dim,
                scaling,
            } => (
                Some(build_rope(
                    spec.rotary_head_dim,
                    *rotary_dim,
                    *theta,
                    *scaling,
                )?),
                None,
            ),
            PositionEncodingSpec::T5RelativeBias {
                buckets,
                max_distance,
                bidirectional,
            } => (
                None,
                Some(
                    T5RelativePositionBiasConfig::new(
                        spec.query_heads,
                        *buckets,
                        *max_distance,
                        *bidirectional,
                    )
                    .init(device)?,
                ),
            ),
        };
        Ok(Self {
            attention: config.init(device)?,
            relative_bias,
            rope,
        })
    }

    /// Computes latent self-attention on `[batch, token, hidden]` residuals.
    ///
    /// Projected content and rotary channels receive explicit positions; the
    /// selected connectivity, padding, and relative score bias combine before
    /// attention. Padded query outputs are zeroed after the output projection.
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        hidden: Tensor<B, 3>,
        positions: Tensor<B, 2, Int>,
        token_mask: Option<&TokenMask<B>>,
        direction: AttentionDirection,
        evaluated_context: usize,
        layer: Layer,
        hooks: &mut Hooks<'_, B>,
        policy: Option<&AttentionPolicy<B>>,
    ) -> Result<Tensor<B, 3>> {
        let [batch, sequence, _] = hidden.dims();
        let projection = self.attention.project_with_hooks(
            hidden.clone(),
            hidden,
            layer.self_attention(),
            hooks,
        )?;
        let projection = match self.rope {
            Some(rope) => projection.with_rotary(
                positions.clone(),
                positions.clone(),
                rope,
                evaluated_context,
            )?,
            None => projection,
        };
        let mask = padding_attention_mask(
            None,
            token_mask,
            batch,
            self.attention.config.num_heads,
            sequence,
            sequence,
        )?;
        let bias = self
            .relative_bias
            .as_ref()
            .map(|bias| relative_bias_from_positions(bias, positions))
            .transpose()?;
        let (mask, bias, causal) = resolve_attention_policy(
            policy,
            mask,
            bias,
            [batch, self.attention.config.num_heads, sequence, sequence],
            &projection.query_content.device(),
            direction.is_causal(),
        )?;
        let output = self.attention.attend_with_hooks(
            projection,
            mask,
            bias,
            causal,
            layer.self_attention(),
            hooks,
        )?;
        Ok(mask_padded_queries(output, token_mask))
    }
}

/// One supported sequence-mixing implementation.
#[derive(Module, Debug)]
// Burn 0.21 does not implement `Module` for `Box<M>`. Keeping modules inline is required so
// generated records can visit every parameter; the per-layer host metadata cost is negligible
// relative to even tiny projection tensors.
#[allow(clippy::large_enum_variant)]
enum ModularMixer<B: Backend> {
    /// Full or sliding projected attention.
    Attention(
        /// Projected attention module with its positional and window policy.
        ModularAttention<B>,
    ),
    /// Multi-head latent attention, reusable in either stack.
    Mla(
        /// Compressed latent-attention module with independently selected position encoding.
        ModularMla<B>,
    ),
    /// Gated DeltaNet linear attention.
    DeltaNet(
        /// Gated recurrent linear-attention module.
        GatedDeltaNet<B>,
    ),
}

impl<B: Backend> ModularMixer<B> {
    /// Initializes a mixer after capability validation has rejected unavailable kernels.
    fn init(
        spec: &MixerSpec,
        hidden_size: usize,
        norm: NormalizationSpec,
        device: &B::Device,
    ) -> Result<Self> {
        match spec {
            MixerSpec::Attention(attention) => {
                ModularAttention::init(attention, hidden_size, norm, None, device)
                    .map(Self::Attention)
            }
            MixerSpec::SlidingAttention(sliding) => ModularAttention::init(
                &sliding.attention,
                hidden_size,
                norm,
                Some(sliding.window),
                device,
            )
            .map(Self::Attention),
            MixerSpec::DeltaNet {
                key_heads,
                value_heads,
                key_head_dim,
                value_head_dim,
                convolution_kernel,
            } => {
                let mut config = GatedDeltaNetConfig::new(
                    hidden_size,
                    *key_heads,
                    *value_heads,
                    *key_head_dim,
                    *value_head_dim,
                );
                config.convolution_kernel = *convolution_kernel;
                config.normalization_epsilon = norm_epsilon(norm);
                config.init(device).map(Self::DeltaNet)
            }
            MixerSpec::Mla(mla) => ModularMla::init(mla, hidden_size, norm, device).map(Self::Mla),
            MixerSpec::Mamba2(_) => Err(Error::BackendCapability(
                "custom mixer passed validation without an executable shared kernel".into(),
            )),
        }
    }

    /// Applies a stateless mixer while preserving its exact causal direction.
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        hidden: Tensor<B, 3>,
        positions: Tensor<B, 2, Int>,
        token_mask: Option<&TokenMask<B>>,
        direction: AttentionDirection,
        evaluated_context: usize,
        layer: Layer,
        hooks: &mut Hooks<'_, B>,
        policy: Option<&AttentionPolicy<B>>,
    ) -> Result<Tensor<B, 3>> {
        match self {
            Self::Attention(attention) => attention.forward_self(
                hidden,
                positions,
                token_mask,
                direction,
                evaluated_context,
                layer,
                hooks,
                policy,
            ),
            Self::Mla(mla) => mla.forward(
                hidden,
                positions,
                token_mask,
                direction,
                evaluated_context,
                layer,
                hooks,
                policy,
            ),
            Self::DeltaNet(delta) => delta.forward(hidden, token_mask),
        }
    }
}

/// Dense gated MLP or sparse routed expert bank.
#[derive(Module, Debug)]
// See `ModularMixer`: indirection would make the Burn parameter record non-visitable.
#[allow(clippy::large_enum_variant)]
enum ModularFeedForward<B: Backend> {
    /// One always-active gated feed-forward network.
    Dense(
        /// Dense gate, content, and output projection parameters.
        GatedFeedForward<B>,
    ),
    /// Top-k routed and optional shared experts.
    MixtureOfExperts(
        /// Router and routed/shared expert parameters.
        SparseMoe<B>,
    ),
}

impl<B: Backend> ModularFeedForward<B> {
    /// Initializes the parameter topology declared by a feed-forward specification.
    fn init(spec: &FeedForwardSpec, hidden_size: usize, device: &B::Device) -> Result<Self> {
        match spec {
            FeedForwardSpec::Dense(dense) => {
                GatedFeedForwardConfig::new(hidden_size, dense.intermediate_size)
                    .with_activation(dense.activation)
                    .with_bias(dense.bias)
                    .init(device)
                    .map(Self::Dense)
            }
            FeedForwardSpec::MixtureOfExperts(moe) => {
                let mut config = SparseMoeConfig::new(
                    hidden_size,
                    moe.expert_intermediate_size,
                    moe.routed_experts,
                    moe.experts_per_token,
                );
                config.shared_experts = moe.shared_experts;
                config.activation = moe.activation;
                config.renormalize_selected = moe.renormalize_selected;
                config.routed_scale = moe.routed_scale;
                config.init(device).map(Self::MixtureOfExperts)
            }
        }
    }

    /// Evaluates the MLP and returns sparse-routing state only for an MoE layer.
    fn forward(
        &self,
        hidden: Tensor<B, 3>,
        layer: Layer,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<(Tensor<B, 3>, Option<SparseMoeOutput<B>>)> {
        match self {
            Self::Dense(dense) => {
                let gate = crate::operations::linear(hidden.clone(), &dense.gate_proj)?;
                let gate = hooks.apply(layer.mlp_pre_activation(), gate)?;
                let gate = apply_activation(gate, dense.activation);
                let gate = hooks.apply(layer.mlp_activation(), gate)?;
                let content = crate::operations::linear(hidden, &dense.up_proj)?;
                Ok((
                    crate::operations::linear(gate * content, &dense.down_proj)?,
                    None,
                ))
            }
            Self::MixtureOfExperts(moe) => {
                let output = moe.forward(hidden)?;
                Ok((output.hidden_state.clone(), Some(output)))
            }
        }
    }
}

/// One heterogeneous residual block and optional seq2seq cross-attention branch.
#[derive(Module, Debug)]
pub struct ModularLayer<B: Backend> {
    /// Sequence mixer.
    mixer: ModularMixer<B>,
    /// Dense or sparse feed-forward operation.
    feed_forward: ModularFeedForward<B>,
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
    /// Initializes one layer and an optional plain cross-attention branch.
    pub fn init(
        spec: &DecoderLayerSpec,
        hidden_size: usize,
        cross_attention: Option<(usize, usize)>,
        device: &B::Device,
    ) -> Result<Self> {
        spec.validate(0)?;
        if hidden_size == 0 || spec.residual == ResidualTopology::Gated {
            return Err(Error::configuration(
                "custom.layer",
                "width must be positive and residual gates require explicitly defined custom computation",
            ));
        }
        let (cross_attention, cross_attention_norm) = match cross_attention {
            Some((heads, head_dim)) => {
                let cross = AttentionSpec {
                    query_heads: heads,
                    key_value_heads: heads,
                    head_dim,
                    query_key_norm: false,
                    output_gate: false,
                    position: PositionEncodingSpec::None,
                };
                (
                    Some(ModularAttention::init(
                        &cross,
                        hidden_size,
                        spec.mixer_norm,
                        None,
                        device,
                    )?),
                    Some(NormLayout::init(
                        spec.mixer_norm,
                        spec.norm_placement,
                        hidden_size,
                        device,
                    )?),
                )
            }
            None => (None, None),
        };
        Ok(Self {
            mixer: ModularMixer::init(&spec.mixer, hidden_size, spec.mixer_norm, device)?,
            feed_forward: ModularFeedForward::init(&spec.feed_forward, hidden_size, device)?,
            mixer_norm: NormLayout::init(
                spec.mixer_norm,
                spec.norm_placement,
                hidden_size,
                device,
            )?,
            feed_forward_norm: NormLayout::init(
                spec.feed_forward_norm,
                spec.norm_placement,
                hidden_size,
                device,
            )?,
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
    /// An explicit attention policy replaces causal/trailing-window defaults, keeps padding
    /// excluded, and adds its bias to any learned position bias. DeltaNet accepts only its
    /// ordinary causal policy. Cross-attention reads `context.encoder` with source padding.
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

    /// Validates a staged call before any callback or parameter operation executes.
    ///
    /// This is the complete check used by the public [`HiddenTransform`] entry
    /// point: block geometry, the context's position values (a device-to-host
    /// read inside [`ExecutionContext::validate`]), and every attention policy.
    pub(super) fn validate_context(
        &self,
        hidden: &Tensor<B, 3>,
        context: &ExecutionContext<B>,
        layer: Layer,
    ) -> Result<()> {
        let [batch, tokens, _] = hidden.dims();
        self.validate_hidden(hidden)?;
        context.validate([batch, tokens], &hidden.device())?;
        self.validate_policies(hidden, context, layer)
    }

    /// Validates a block inside a range whose context values were checked once.
    ///
    /// Range runners call [`ExecutionContext::validate`] once for the whole
    /// range, because the context is borrowed immutably and its position values
    /// cannot change between blocks. Hooks may still replace the hidden states,
    /// so this re-checks everything that depends on them without reading
    /// tensor values back to the host.
    pub(super) fn validate_prevalidated(
        &self,
        hidden: &Tensor<B, 3>,
        context: &ExecutionContext<B>,
        layer: Layer,
    ) -> Result<()> {
        let [batch, tokens, _] = hidden.dims();
        self.validate_hidden(hidden)?;
        validate_context_layout(context, [batch, tokens], &hidden.device())?;
        self.validate_policies(hidden, context, layer)
    }

    /// Runs this block inside a range runner after [`ExecutionContext::validate`].
    ///
    /// Numerically identical to [`HiddenTransform::forward`]; only the
    /// per-block device-to-host reads of the shared context are skipped.
    pub(super) fn forward_prevalidated(
        &self,
        hidden: Tensor<B, 3>,
        context: &ExecutionContext<B>,
        layer: Layer,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<Tensor<B, 3>> {
        self.validate_prevalidated(&hidden, context, layer)?;
        self.forward_with_context(hidden, context, layer, hooks)
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
        let packed_weight = matches!(projection.dtype(), burn::tensor::DType::QFloat(_));
        if projection.device() != hidden.device()
            || matches!(hidden.dtype(), burn::tensor::DType::QFloat(_))
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
    fn validate_policies(
        &self,
        hidden: &Tensor<B, 3>,
        context: &ExecutionContext<B>,
        layer: Layer,
    ) -> Result<()> {
        let [batch, tokens, _] = hidden.dims();
        if matches!(&self.mixer, ModularMixer::DeltaNet(_)) {
            let policy = context.attention_for(layer, layer.stack() == Stack::Decoder);
            if !policy.causal || policy.allowed.is_some() || policy.bias.is_some() {
                return Err(Error::BackendCapability("DeltaNet requires ordinary causal execution; choose another mixer for bidirectional or custom connectivity".into()));
            }
        }
        if let Some(policy) = explicit_policy(context, layer) {
            match &self.mixer {
                ModularMixer::Attention(attention) => policy.validate(
                    [batch, attention.attention.query_heads, tokens, tokens],
                    &hidden.device(),
                )?,
                ModularMixer::Mla(mla) => policy.validate(
                    [batch, mla.attention.config.num_heads, tokens, tokens],
                    &hidden.device(),
                )?,
                ModularMixer::DeltaNet(_) => {}
            }
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
            let cross_policy = context.cross_attention_for(layer);
            cross_policy.validate(
                [
                    batch,
                    attention.attention.query_heads,
                    tokens,
                    source.hidden.dims()[1],
                ],
                &hidden.device(),
            )?;
            if cross_policy
                .bias
                .as_ref()
                .is_some_and(|bias| bias.dtype() != hidden.dtype())
            {
                return Err(Error::configuration(
                    "custom.cross_attention.bias",
                    "score bias must use the hidden-state dtype",
                ));
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

    /// Returns only the intervention points implemented by this block's actual operators.
    pub fn activation_points(&self, layer: Layer) -> Vec<PointInfo> {
        let mut points = vec![layer.residual_pre().info(), layer.attention_input().info()];
        match &self.mixer {
            ModularMixer::Attention(attention) => {
                points.extend(attention.activation_points(layer.self_attention()))
            }
            ModularMixer::Mla(mla) => {
                points.extend(mla.attention.activation_points(layer.self_attention()))
            }
            ModularMixer::DeltaNet(_) => {}
        }
        points.push(layer.attention_output().info());
        if let Some(attention) = &self.cross_attention {
            points.push(layer.cross_attention_input().info());
            points.extend(attention.activation_points(layer.cross_attention()));
            points.push(layer.cross_attention_output().info());
        }
        points.push(layer.mlp_input().info());
        if matches!(&self.feed_forward, ModularFeedForward::Dense(_)) {
            points.extend([
                layer.mlp_pre_activation().info(),
                layer.mlp_activation().info(),
            ]);
        }
        points.extend([layer.mlp_output().info(), layer.residual_post().info()]);
        points
    }

    /// Evaluates residual branches, applying interventions before dependent computation.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward(
        &self,
        hidden: Tensor<B, 3>,
        positions: Tensor<B, 2, Int>,
        token_mask: Option<&TokenMask<B>>,
        direction: AttentionDirection,
        evaluated_context: usize,
        encoder: Option<(&Tensor<B, 3>, Option<&TokenMask<B>>)>,
        layer: Layer,
        hooks: &mut Hooks<'_, B>,
        policy: Option<&AttentionPolicy<B>>,
        cross_policy: Option<&AttentionPolicy<B>>,
    ) -> Result<(Tensor<B, 3>, Option<SparseMoeOutput<B>>)> {
        let hidden = hooks.apply(layer.residual_pre(), hidden)?;
        let residual = hidden.clone();
        let prepared = self.mixer_norm.prepare(hidden)?;
        let prepared = hooks.apply(layer.attention_input(), prepared)?;
        let mixed = self.mixer.forward(
            prepared,
            positions,
            token_mask,
            direction,
            evaluated_context,
            layer,
            hooks,
            policy,
        )?;
        let mixed = hooks.apply(layer.attention_output(), mixed)?;
        let mut hidden = self.mixer_norm.finish(residual, mixed)?;

        match (&self.cross_attention, &self.cross_attention_norm, encoder) {
            (Some(attention), Some(norm), Some((source, source_mask))) => {
                let residual = hidden.clone();
                let prepared = hooks.apply(layer.cross_attention_input(), norm.prepare(hidden)?)?;
                let attended = attention.forward_cross(
                    prepared,
                    source.clone(),
                    token_mask,
                    source_mask,
                    layer,
                    hooks,
                    cross_policy,
                )?;
                let attended = hooks.apply(layer.cross_attention_output(), attended)?;
                hidden = norm.finish(residual, attended)?;
            }
            (None, None, None) => {}
            (Some(_), Some(_), None) => {
                return Err(Error::configuration(
                    "custom.cross_attention",
                    "decoder layer requires encoder hidden states",
                ));
            }
            _ => {
                return Err(Error::configuration(
                    "custom.cross_attention",
                    "instantiated cross-attention parameters disagree with the layer role",
                ));
            }
        }

        let residual = hidden.clone();
        let prepared = hooks.apply(layer.mlp_input(), self.feed_forward_norm.prepare(hidden)?)?;
        let (projected, router) = self.feed_forward.forward(prepared, layer, hooks)?;
        let projected = hooks.apply(layer.mlp_output(), projected)?;
        let hidden = self.feed_forward_norm.finish(residual, projected)?;
        Ok((hooks.apply(layer.residual_post(), hidden)?, router))
    }
}

impl<B: Backend> HiddenTransform<B> for ModularLayer<B> {
    /// Lists the actual mixer, optional cross-attention, and MLP points for this block.
    fn activation_points(&self, layer: Layer) -> Vec<PointInfo> {
        ModularLayer::activation_points(self, layer)
    }

    /// Applies this residual block using explicit positions, padding, and attention policies.
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
        self.validate_context(&hidden, context, layer)?;
        self.forward_with_context(hidden, context, layer, hooks)
    }
}

impl<B: Backend> ModularLayer<B> {
    /// Applies an already validated block using the context's positions and policies.
    ///
    /// Encoder blocks default to bidirectional attention and decoder blocks to
    /// causal attention; explicit policies replace those defaults.
    fn forward_with_context(
        &self,
        hidden: Tensor<B, 3>,
        context: &ExecutionContext<B>,
        layer: Layer,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<Tensor<B, 3>> {
        let direction = match layer.stack() {
            Stack::Encoder => AttentionDirection::Bidirectional,
            Stack::Decoder => AttentionDirection::Causal,
        };
        let encoder = if self.cross_attention.is_some() {
            context
                .encoder
                .as_ref()
                .map(|source| (&source.hidden, source.token_mask.as_ref()))
        } else {
            None
        };
        let policy = explicit_policy(context, layer);
        let cross_policy = context.cross_attention_for(layer);
        Ok(self
            .forward(
                hidden,
                context.positions.clone(),
                context.token_mask.as_ref(),
                direction,
                context.evaluated_context,
                encoder,
                layer,
                hooks,
                policy.as_ref(),
                Some(&cross_policy),
            )?
            .0)
    }
}

/// Sync-free mirror of the layout half of [`ExecutionContext::validate`].
///
/// Checks that positions and padding match the `[batch, token]` target grid and
/// device, and that any encoder source shares the batch and device. Position
/// values are not read; range runners validate them once per range.
fn validate_context_layout<B: Backend>(
    context: &ExecutionContext<B>,
    shape: [usize; 2],
    device: &B::Device,
) -> Result<()> {
    if shape.contains(&0) || context.positions.dims() != shape {
        return Err(Error::shape(
            "execution.positions",
            format!("nonempty {shape:?}"),
            format!("{:?}", context.positions.dims()),
        ));
    }
    if context.positions.device() != *device {
        return Err(Error::configuration(
            "execution.positions",
            "positions and hidden states must share a device",
        ));
    }
    if let Some(mask) = &context.token_mask {
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
    if let Some(encoder) = &context.encoder {
        if encoder.hidden.dims()[0] != shape[0] {
            return Err(Error::configuration(
                "execution.encoder",
                "source hidden states and positions must have matching nonempty batch/token axes",
            ));
        }
        if encoder.hidden.device() != *device {
            return Err(Error::configuration(
                "execution.encoder",
                "source context must share the target device",
            ));
        }
    }
    Ok(())
}

/// Resolves only intentional overrides, preserving structural sliding-window defaults otherwise.
fn explicit_policy<B: Backend>(
    context: &ExecutionContext<B>,
    layer: Layer,
) -> Option<AttentionPolicy<B>> {
    context
        .layer_attention
        .iter()
        .rev()
        .find(|(candidate, _)| *candidate == layer)
        .map(|(_, policy)| policy.clone())
        .or_else(|| context.attention.clone())
}

/// Adds explicit connectivity and bias to ordinary padding and learned position bias.
#[allow(clippy::type_complexity)]
fn resolve_attention_policy<B: Backend>(
    policy: Option<&AttentionPolicy<B>>,
    mask: Option<Tensor<B, 4, Bool>>,
    bias: Option<Tensor<B, 4>>,
    shape: [usize; 4],
    device: &B::Device,
    default_causal: bool,
) -> Result<(Option<Tensor<B, 4, Bool>>, Option<Tensor<B, 4>>, bool)> {
    let Some(policy) = policy else {
        return Ok((mask, bias, default_causal));
    };
    let mask = policy
        .exclusion_mask(mask, shape, device)?
        .map(|mask| mask.expand(shape));
    let bias = match (bias, policy.expanded_bias(shape, device)?) {
        (Some(existing), Some(extra)) => Some(existing + extra),
        (None, Some(extra)) => Some(extra),
        (existing, None) => existing,
    };
    Ok((mask, bias, policy.causal))
}

/// Complete hidden-state result retained before vocabulary projection.
pub(super) struct BackboneOutput<B: Backend> {
    /// Final normalized residual stream.
    pub(super) hidden: Tensor<B, 3>,
    /// Embedding, layer, and final states when requested.
    pub(super) hidden_states: Vec<Tensor<B, 3>>,
    /// Raw sparse outputs for each MoE layer.
    pub(super) routers: Vec<SparseMoeOutput<B>>,
}

/// Validates the executable capability boundary for one modular decoder stack.
pub(super) fn validate_decoder_execution(
    spec: &DecoderSpec,
    direction: AttentionDirection,
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
        if layer.residual == ResidualTopology::Gated {
            return Err(Error::BackendCapability(format!(
                "{layer_field}.residual requests a learned gate but the specification does not define its activation, shape, or initialization"
            )));
        }
        match (&layer.mixer, direction) {
            (MixerSpec::Mamba2(_), _) => {
                return Err(Error::BackendCapability(format!(
                    "{layer_field}.mixer requests Mamba2; the reusable selective state-space kernel is not executable yet"
                )));
            }
            (MixerSpec::DeltaNet { .. }, AttentionDirection::Bidirectional) => {
                return Err(Error::BackendCapability(format!(
                    "{layer_field}.mixer is causal recurrent DeltaNet and cannot represent a bidirectional encoder"
                )));
            }
            (MixerSpec::SlidingAttention(_), AttentionDirection::Bidirectional) => {
                return Err(Error::BackendCapability(format!(
                    "{layer_field}.mixer defines a trailing causal window and cannot represent symmetric bidirectional encoder attention"
                )));
            }
            _ => {}
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

/// Returns whether a stack contains sparse experts.
pub(super) fn has_moe(spec: &DecoderSpec) -> bool {
    spec.layers
        .iter()
        .any(|layer| matches!(&layer.feed_forward, FeedForwardSpec::MixtureOfExperts(_)))
}

/// Applies an ordinary tied vocabulary projection with embedding weights.
pub(super) fn tied_projection<B: Backend>(
    hidden: Tensor<B, 3>,
    embedding_weight: Tensor<B, 2>,
) -> Tensor<B, 3> {
    burn::tensor::module::linear(hidden, embedding_weight.transpose(), None)
}

/// Adds a trailing window measured in valid-token ordinals, preserving key padding.
///
/// Let `r_i = sum_{j <= i} valid_j`. A valid causal key `k` is too old for query `q`
/// exactly when `r_q - r_k >= window`. Padding columns therefore consume no history, matching
/// the decode state's compact token storage. Caller-supplied position IDs affect positional
/// encodings only: gaps or offsets in those IDs do not discard otherwise recent keys.
fn add_sliding_mask<B: Backend>(
    padding: Option<Tensor<B, 4, Bool>>,
    token_mask: Option<&TokenMask<B>>,
    batch: usize,
    heads: usize,
    sequence: usize,
    window: Option<usize>,
    device: &B::Device,
) -> Option<Tensor<B, 4, Bool>> {
    let Some(window) = window.filter(|&window| window < sequence) else {
        return padding;
    };
    let ordinals = match token_mask {
        Some(mask) => mask.as_tensor().clone().int().cumsum(1),
        None => Tensor::<B, 1, Int>::arange(0..sequence as i64, device)
            .reshape([1, sequence])
            .expand([batch, sequence]),
    };
    let query = ordinals
        .clone()
        .reshape([batch, 1, sequence, 1])
        .expand([batch, heads, sequence, sequence]);
    let key = ordinals
        .reshape([batch, 1, 1, sequence])
        .expand([batch, heads, sequence, sequence]);
    let too_old = (query - key).greater_equal_elem(window as i64);
    Some(match padding {
        Some(padding) => padding.bool_or(too_old),
        None => too_old,
    })
}

/// Defines padded query rows as zero without masking every score in their softmax row.
fn mask_padded_queries<B: Backend>(
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

/// Extracts the normalization epsilon reused by optional per-head Q/K normalization.
fn norm_epsilon(spec: NormalizationSpec) -> f64 {
    match spec {
        NormalizationSpec::Rms { epsilon, .. } | NormalizationSpec::Layer { epsilon, .. } => {
            epsilon
        }
    }
}

/// Converts a serializable RoPE policy into the validated shared runtime representation.
fn build_rope(
    head_dim: usize,
    rotary_dim: usize,
    theta: f64,
    scaling: SpecRopeScaling,
) -> Result<RopeConfig> {
    let scaling = match scaling {
        SpecRopeScaling::None => RopeScaling::None,
        SpecRopeScaling::Yarn {
            factor,
            original_max_positions,
            attention_factor,
            beta_slow,
            beta_fast,
        } => {
            let mut yarn = YarnConfig::new(factor, original_max_positions)
                .map_err(|error| Error::configuration("custom.rope.yarn", error.to_string()))?
                .with_betas(beta_fast, beta_slow)
                .map_err(|error| Error::configuration("custom.rope.yarn", error.to_string()))?;
            if let Some(attention_factor) = attention_factor {
                yarn = yarn
                    .with_attention_factor(attention_factor)
                    .map_err(|error| Error::configuration("custom.rope.yarn", error.to_string()))?;
            }
            RopeScaling::Yarn(yarn)
        }
    };
    RopeConfig::new(head_dim, rotary_dim, theta, scaling)
        .map_err(|error| Error::configuration("custom.rope", error.to_string()))
}

/// Expands T5 bucket bias from the caller's exact absolute positions.
///
/// The shared family fast path accepts contiguous ranges through a query offset. A custom graph
/// may deliberately supply gaps or per-row offsets, so it constructs `bucket(p_key - p_query)`
/// for every pair instead. Only integer bucket selection is materialized on the host; embedding
/// lookup and the resulting attention computation remain differentiable with respect to the
/// learned bias table.
pub(super) fn relative_bias_from_positions<B: Backend>(
    bias: &T5RelativePositionBias<B>,
    positions: Tensor<B, 2, Int>,
) -> Result<Tensor<B, 4>> {
    let [batch, sequence] = positions.dims();
    let device = positions.device();
    let positions = positions
        .try_into_data()
        .map_err(|error| {
            Error::BackendCapability(format!(
                "failed to inspect custom T5 relative positions on the backend: {error}"
            ))
        })?
        .convert::<i64>()
        .into_vec::<i64>()
        .map_err(|error| {
            Error::BackendCapability(format!(
                "failed to inspect custom T5 relative positions: {error}"
            ))
        })?;
    let element_count = batch
        .checked_mul(sequence)
        .and_then(|count| count.checked_mul(sequence))
        .ok_or_else(|| {
            Error::configuration(
                "custom.t5_relative_bias",
                "batch times sequence squared overflows usize",
            )
        })?;
    let mut indices = Vec::new();
    indices.try_reserve_exact(element_count).map_err(|_| {
        Error::BackendCapability(format!(
            "could not reserve {element_count} custom T5 relative-bias indices"
        ))
    })?;
    let buckets = bias.buckets();
    for row in positions.chunks_exact(sequence) {
        for &query in row {
            let query = usize::try_from(query)
                .map_err(|_| Error::configuration("custom.position_ids", "must be non-negative"))?;
            for &key in row {
                let key = usize::try_from(key).map_err(|_| {
                    Error::configuration("custom.position_ids", "must be non-negative")
                })?;
                indices.push(
                    i64::try_from(buckets.bucket_for_positions(query, key)).map_err(|_| {
                        Error::configuration(
                            "custom.t5_relative_bias",
                            "bucket index does not fit Burn's signed integer representation",
                        )
                    })?,
                );
            }
        }
    }
    let indices = Tensor::<B, 2, Int>::from_data(
        TensorData::new(indices, [batch * sequence, sequence]),
        &device,
    );
    let weight = bias.weight.val();
    let heads = weight.dims()[1];
    Ok(embedding(weight, indices)
        .reshape([batch, sequence, sequence, heads])
        .permute([0, 3, 1, 2]))
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
