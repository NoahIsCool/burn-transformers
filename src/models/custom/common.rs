//! Parameterized building blocks shared by modular causal and encoder-decoder graphs.
//!
//! A block is decomposed into a sequence mixer, a feed-forward operator, and normalization
//! layouts around each residual branch. This mirrors the mathematical factorization
//! `x' = R_m(x, M(N_m(x)))`, `y = R_f(x', F(N_f(x')))`. Keeping the pieces independent makes
//! heterogeneous schedules possible without model-name checks in shared kernels.

use burn::{
    module::Module,
    nn::{LayerNorm, LayerNormConfig},
    tensor::{
        Bool, ElementConversion, Int, Tensor, TensorData, backend::Backend, module::embedding,
    },
};

use crate::{
    CausalLmInput, CausalLmOutput, Error, ForwardOptions, LogitsSelection, Result,
    architecture::{
        AttentionSpec, DecoderLayerSpec, DecoderSpec, FeedForwardSpec, MixerSpec, NormPlacement,
        NormalizationSpec, PositionEncodingSpec, ResidualTopology, RopeScaling as SpecRopeScaling,
        RouterDiagnostics, TokenMask,
    },
    attention::{
        AttentionOutputGate, GroupedQueryAttention, GroupedQueryAttentionConfig,
        padding_attention_mask,
    },
    delta_net::{GatedDeltaNet, GatedDeltaNetConfig},
    feed_forward::{GatedFeedForward, GatedFeedForwardConfig},
    moe::{SparseMoe, SparseMoeConfig, SparseMoeOutput, load_balancing_loss},
    normalization::{RmsNorm, RmsNormConfig},
    position::{
        RopeConfig, RopeScaling, T5RelativePositionBias, T5RelativePositionBiasConfig, YarnConfig,
        apply_rotary, rotary_angles,
    },
};

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
pub(super) enum ModularNorm<B: Backend> {
    /// Crate-native RMS normalization, including zero-centered checkpoint weights.
    Rms(RmsNorm<B>),
    /// Burn LayerNorm with an optional learned additive bias.
    Layer(LayerNorm<B>),
}

impl<B: Backend> ModularNorm<B> {
    /// Initializes the exact normalization convention requested by a specification.
    pub(super) fn init(
        spec: NormalizationSpec,
        hidden_size: usize,
        device: &B::Device,
    ) -> Result<Self> {
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
    pub(super) fn forward(&self, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
        match self {
            Self::Rms(norm) => norm.forward(hidden),
            Self::Layer(norm) => norm.forward(hidden),
        }
    }
}

/// Before/after normalization parameters for one residual sublayer.
#[derive(Module, Debug)]
struct NormLayout<B: Backend> {
    /// Independent pre-sublayer normalization when requested.
    before: Option<ModularNorm<B>>,
    /// Independent post-sublayer normalization when requested.
    after: Option<ModularNorm<B>>,
    /// Algebraic placement retained outside the parameter record.
    #[module(skip)]
    placement: NormPlacement,
}

impl<B: Backend> NormLayout<B> {
    /// Allocates exactly the normalization parameters implied by one placement.
    fn init(
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
        })
    }

    /// Applies the optional normalization immediately before a sublayer.
    fn prepare(&self, hidden: Tensor<B, 3>) -> Result<Tensor<B, 3>> {
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
    fn finish(&self, residual: Tensor<B, 3>, mixed: Tensor<B, 3>) -> Result<Tensor<B, 3>> {
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

    /// Evaluates position-aware self-attention over a complete stateless sequence.
    fn forward_self(
        &self,
        hidden: Tensor<B, 3>,
        positions: Tensor<B, 2, Int>,
        token_mask: Option<&TokenMask<B>>,
        direction: AttentionDirection,
        evaluated_context: usize,
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
            batch,
            self.attention.query_heads,
            sequence,
            self.sliding_window,
            &projection.query.device(),
        );
        let bias = self
            .relative_bias
            .as_ref()
            .map(|bias| relative_bias_from_positions(bias, positions))
            .transpose()?;
        let output = self
            .attention
            .attend(projection, mask, bias, direction.is_causal())?;
        Ok(mask_padded_queries(output, token_mask))
    }

    /// Evaluates ordinary decoder-to-encoder cross attention.
    fn forward_cross(
        &self,
        query: Tensor<B, 3>,
        source: Tensor<B, 3>,
        query_mask: Option<&TokenMask<B>>,
        source_mask: Option<&TokenMask<B>>,
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
        let output = self.attention.attend(projection, mask, None, false)?;
        Ok(mask_padded_queries(output, query_mask))
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
    Attention(ModularAttention<B>),
    /// Gated DeltaNet linear attention.
    DeltaNet(GatedDeltaNet<B>),
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
            MixerSpec::Mla(_) | MixerSpec::Mamba2(_) => Err(Error::BackendCapability(
                "custom mixer passed validation without an executable shared kernel".into(),
            )),
        }
    }

    /// Applies a stateless mixer while preserving its exact causal direction.
    fn forward(
        &self,
        hidden: Tensor<B, 3>,
        positions: Tensor<B, 2, Int>,
        token_mask: Option<&TokenMask<B>>,
        direction: AttentionDirection,
        evaluated_context: usize,
    ) -> Result<Tensor<B, 3>> {
        match self {
            Self::Attention(attention) => {
                attention.forward_self(hidden, positions, token_mask, direction, evaluated_context)
            }
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
    Dense(GatedFeedForward<B>),
    /// Top-k routed and optional shared experts.
    MixtureOfExperts(SparseMoe<B>),
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
    fn forward(&self, hidden: Tensor<B, 3>) -> Result<(Tensor<B, 3>, Option<SparseMoeOutput<B>>)> {
        match self {
            Self::Dense(dense) => Ok((dense.forward(hidden), None)),
            Self::MixtureOfExperts(moe) => {
                let output = moe.forward(hidden)?;
                Ok((output.hidden_state.clone(), Some(output)))
            }
        }
    }
}

/// One heterogeneous residual block and optional seq2seq cross-attention branch.
#[derive(Module, Debug)]
pub(super) struct ModularLayer<B: Backend> {
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
}

impl<B: Backend> ModularLayer<B> {
    /// Initializes one layer and an optional plain cross-attention branch.
    pub(super) fn init(
        spec: &DecoderLayerSpec,
        hidden_size: usize,
        cross_attention: Option<(usize, usize)>,
        device: &B::Device,
    ) -> Result<Self> {
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
        })
    }

    /// Evaluates self mixing, optional cross attention, and feed-forward residual branches.
    pub(super) fn forward(
        &self,
        hidden: Tensor<B, 3>,
        positions: Tensor<B, 2, Int>,
        token_mask: Option<&TokenMask<B>>,
        direction: AttentionDirection,
        evaluated_context: usize,
        encoder: Option<(&Tensor<B, 3>, Option<&TokenMask<B>>)>,
    ) -> Result<(Tensor<B, 3>, Option<SparseMoeOutput<B>>)> {
        let residual = hidden.clone();
        let mixed = self.mixer.forward(
            self.mixer_norm.prepare(hidden)?,
            positions,
            token_mask,
            direction,
            evaluated_context,
        )?;
        let mut hidden = self.mixer_norm.finish(residual, mixed)?;

        match (&self.cross_attention, &self.cross_attention_norm, encoder) {
            (Some(attention), Some(norm), Some((source, source_mask))) => {
                let residual = hidden.clone();
                let attended = attention.forward_cross(
                    norm.prepare(hidden)?,
                    source.clone(),
                    token_mask,
                    source_mask,
                )?;
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
        let (projected, router) = self
            .feed_forward
            .forward(self.feed_forward_norm.prepare(hidden)?)?;
        Ok((self.feed_forward_norm.finish(residual, projected)?, router))
    }
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
            (MixerSpec::Mla(_), _) => {
                return Err(Error::BackendCapability(format!(
                    "{layer_field}.mixer requests MLA; the reusable absorbed/unabsorbed MLA kernel is not executable yet"
                )));
            }
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

/// Creates a position tensor and validates all public causal input invariants.
pub(super) fn validate_and_resolve_input<B: Backend>(
    input: &CausalLmInput<B>,
    vocabulary_size: usize,
    max_positions: usize,
    field: &str,
) -> Result<Tensor<B, 2, Int>> {
    let [batch, sequence] = input.token_ids.dims();
    if batch == 0 || sequence == 0 {
        return Err(Error::shape(
            format!("{field}.token_ids"),
            "non-empty [batch, sequence]",
            format!("[{batch}, {sequence}]"),
        ));
    }
    validate_token_ids(input.token_ids.clone(), vocabulary_size, field)?;
    let device = input.token_ids.device();
    validate_mask(input.token_mask.as_ref(), batch, sequence, &device, field)?;
    let positions = match &input.position_ids {
        Some(positions) => {
            if positions.dims() != [batch, sequence] {
                return Err(Error::shape(
                    format!("{field}.position_ids"),
                    format!("[{batch}, {sequence}]"),
                    format!("{:?}", positions.dims()),
                ));
            }
            if positions.device() != device {
                return Err(Error::BackendCapability(format!(
                    "{field} token and position tensors must reside on the same device"
                )));
            }
            match &input.token_mask {
                Some(mask) => positions
                    .clone()
                    .mask_fill(mask.as_tensor().clone().bool_not(), 0),
                None => positions.clone(),
            }
        }
        None => match &input.token_mask {
            Some(mask) => {
                let valid = mask.as_tensor().clone();
                (valid.clone().int().cumsum(1) - 1).mask_fill(valid.bool_not(), 0)
            }
            None => Tensor::<B, 1, Int>::arange(0..sequence as i64, &input.token_ids.device())
                .reshape([1, sequence])
                .expand([batch, sequence]),
        },
    };
    validate_positions(positions.clone(), max_positions, field)?;
    Ok(positions)
}

/// Returns one plus the largest valid logical position.
pub(super) fn evaluated_context<B: Backend>(positions: Tensor<B, 2, Int>) -> Result<usize> {
    let maximum = positions
        .max()
        .try_into_scalar()
        .map_err(|error| {
            Error::BackendCapability(format!("failed to inspect custom position IDs: {error}"))
        })?
        .elem::<i64>();
    usize::try_from(maximum)
        .map_err(|_| Error::configuration("custom.position_ids", "must be non-negative"))?
        .checked_add(1)
        .ok_or_else(|| Error::configuration("custom.position_ids", "context overflows usize"))
}

/// Selects hidden positions before the expensive vocabulary projection.
pub(super) fn select_hidden<B: Backend>(
    hidden: Tensor<B, 3>,
    mask: Option<&TokenMask<B>>,
    selection: &LogitsSelection,
) -> Result<Tensor<B, 3>> {
    let [batch, sequence, width] = hidden.dims();
    validate_logits_selection(selection, sequence)?;
    match selection {
        LogitsSelection::All => Ok(hidden),
        LogitsSelection::Last => match mask {
            None => Ok(hidden.narrow(1, sequence - 1, 1)),
            Some(mask) => {
                let positions = Tensor::<B, 1, Int>::arange(0..sequence as i64, &hidden.device())
                    .reshape([1, sequence])
                    .expand([batch, sequence]);
                let last = positions
                    .mask_fill(mask.as_tensor().clone().bool_not(), -1)
                    .argmax(1)
                    .unsqueeze_dim::<3>(2)
                    .expand([batch, 1, width]);
                Ok(hidden.gather(1, last))
            }
        },
        LogitsSelection::Positions(positions) => {
            let count = positions.len();
            let indices = positions
                .iter()
                .map(|&value| value as i64)
                .collect::<Vec<_>>();
            let device = hidden.device();
            Ok(hidden.select(
                1,
                Tensor::<B, 1, Int>::from_data(TensorData::new(indices, [count]), &device),
            ))
        }
    }
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

/// Adds the exclusion mask for a causal trailing attention window.
fn add_sliding_mask<B: Backend>(
    padding: Option<Tensor<B, 4, Bool>>,
    batch: usize,
    heads: usize,
    sequence: usize,
    window: Option<usize>,
    device: &B::Device,
) -> Option<Tensor<B, 4, Bool>> {
    let window = window?;
    let query = Tensor::<B, 1, Int>::arange(0..sequence as i64, device)
        .reshape([1, 1, sequence, 1])
        .expand([batch, heads, sequence, sequence]);
    let key = Tensor::<B, 1, Int>::arange(0..sequence as i64, device)
        .reshape([1, 1, 1, sequence])
        .expand([batch, heads, sequence, sequence]);
    let too_old = (key + window as i64).lower_equal(query);
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
        .into_data()
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

/// Checks token values before the backend embedding kernel receives indices.
fn validate_token_ids<B: Backend>(
    token_ids: Tensor<B, 2, Int>,
    vocabulary: usize,
    field: &str,
) -> Result<()> {
    let minimum = token_ids
        .clone()
        .min()
        .try_into_scalar()
        .map_err(|error| {
            Error::BackendCapability(format!("failed to validate custom token IDs: {error}"))
        })?
        .elem::<i64>();
    let maximum = token_ids
        .max()
        .try_into_scalar()
        .map_err(|error| {
            Error::BackendCapability(format!("failed to validate custom token IDs: {error}"))
        })?
        .elem::<i64>();
    if minimum < 0 || maximum as usize >= vocabulary {
        return Err(Error::shape(
            format!("{field}.token_ids"),
            format!("values in 0..{vocabulary}"),
            format!("observed range {minimum}..={maximum}"),
        ));
    }
    Ok(())
}

/// Checks a token-valid mask and rejects rows whose attention softmax would have no valid key.
fn validate_mask<B: Backend>(
    mask: Option<&TokenMask<B>>,
    batch: usize,
    sequence: usize,
    token_device: &B::Device,
    field: &str,
) -> Result<()> {
    let Some(mask) = mask else {
        return Ok(());
    };
    if mask.as_tensor().dims() != [batch, sequence] {
        return Err(Error::shape(
            format!("{field}.token_mask"),
            format!("[{batch}, {sequence}]"),
            format!("{:?}", mask.as_tensor().dims()),
        ));
    }
    if mask.as_tensor().device() != *token_device {
        return Err(Error::BackendCapability(format!(
            "{field} token and mask tensors must reside on the same device"
        )));
    }
    let minimum = mask
        .as_tensor()
        .clone()
        .int()
        .sum_dim(1)
        .min()
        .try_into_scalar()
        .map_err(|error| {
            Error::BackendCapability(format!("failed to validate custom token mask: {error}"))
        })?
        .elem::<i64>();
    if minimum == 0 {
        return Err(Error::shape(
            format!("{field}.token_mask"),
            "at least one valid token in every row",
            "a fully masked row",
        ));
    }
    Ok(())
}

/// Checks that absolute positions fit the model's declared context.
fn validate_positions<B: Backend>(
    positions: Tensor<B, 2, Int>,
    capacity: usize,
    field: &str,
) -> Result<()> {
    let minimum = positions
        .clone()
        .min()
        .try_into_scalar()
        .map_err(|error| {
            Error::BackendCapability(format!("failed to validate custom positions: {error}"))
        })?
        .elem::<i64>();
    let maximum = positions
        .max()
        .try_into_scalar()
        .map_err(|error| {
            Error::BackendCapability(format!("failed to validate custom positions: {error}"))
        })?
        .elem::<i64>();
    if minimum < 0 || maximum as usize >= capacity {
        return Err(Error::shape(
            format!("{field}.position_ids"),
            format!("values in 0..{capacity}"),
            format!("observed range {minimum}..={maximum}"),
        ));
    }
    Ok(())
}

/// Validates explicit selective-logit indices.
fn validate_logits_selection(selection: &LogitsSelection, sequence: usize) -> Result<()> {
    if let LogitsSelection::Positions(positions) = selection
        && (positions.is_empty() || positions.iter().any(|&position| position >= sequence))
    {
        return Err(Error::shape(
            "forward_options.logits.positions",
            format!("a non-empty list of indices below {sequence}"),
            format!("{positions:?}"),
        ));
    }
    Ok(())
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
    TokenMask::new(Tensor::<B, 2, Bool>::from_data(
        TensorData::new(values, shape),
        device,
    ))
}
