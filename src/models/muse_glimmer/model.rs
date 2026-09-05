//! Executable Muse-Glimmer dense hybrid decoder.
//!
//! A decoder layer applies a four-norm "sandwich" residual layout:
//! `x' = x + PostAttnNorm(Attention(InputNorm(x)))` and
//! `y = x' + PostFfnNorm(SwiGLU(PreFfnNorm(x')))`. All four layer norms use
//! zero-centered checkpoint scales, `scale = 1 + weight`; the two post norms
//! intentionally use the smaller `post_norm_eps` stabilizer. Token embeddings
//! are RMS-normalized without a learned scale, while the final norm stores an
//! ordinary (non-centered) learned scale.
//!
//! Attention is grouped-query attention with a distinct gate projection. After
//! scale-free per-head Q/K normalization, queries are multiplied by
//! `qk_scale_factor`; dot products still receive the standard `1/sqrt(d)`
//! factor. Muse's eager reference evaluates the attention softmax in FP32 before
//! restoring the query dtype. Burn's optimized Flex and CubeCL attention paths
//! preserve that accumulator precision for half inputs. Passing the conventional
//! scale as `None` is semantically exact and is also required for Burn 0.21 to
//! retain fused CubeCL dispatch eligibility. The attention result is multiplied
//! by `sigmoid(gate_proj(h))` before the output projection. Three local RoPE
//! layers normally precede each global NoPE layer. Finally, logits follow the
//! exact bounded map `T * tanh(output_multiplier * raw_logits / T)`.

use burn::{
    module::Module,
    nn::{Embedding, EmbeddingConfig, Linear, LinearConfig, LinearLayout},
    tensor::{
        Bool, DType, ElementConversion, Int, Tensor, TensorData,
        activation::sigmoid,
        backend::Backend,
        module::{attention, linear as tensor_linear},
        ops::AttentionModuleOptions,
    },
};

use crate::{
    ArchitectureConfig, CausalLmInput, CausalLmOutput, Error, ForwardOptions, LogitsSelection,
    Result,
    architecture::TokenMask,
    attention::{padding_attention_mask, repeat_key_value},
    feed_forward::{GatedFeedForward, GatedFeedForwardConfig, apply_activation},
    normalization::{RmsNorm, RmsNormConfig},
    operations::linear,
    position::{
        DynamicNtkScaling, LinearRopeScaling, RopeConfig, RopeScaling, YarnConfig, apply_rotary,
        rotary_angles,
    },
};

use super::{MuseGlimmerLayerType, MuseGlimmerTextConfig};
use crate::models::config::RopeType;

mod decode;

pub use decode::{MuseGlimmerDecodeState, MuseGlimmerLayerCache};

/// Projected Q/K/V and the independent Muse output gate.
#[derive(Clone, Debug)]
struct MuseAttentionProjection<B: Backend> {
    /// Queries shaped `[batch, query_heads, query_sequence, head_dim]`.
    query: Tensor<B, 4>,
    /// Native keys shaped `[batch, key_value_heads, key_sequence, head_dim]`.
    key: Tensor<B, 4>,
    /// Native values shaped `[batch, key_value_heads, key_sequence, head_dim]`.
    value: Tensor<B, 4>,
    /// Unactivated gate shaped `[batch, query_sequence, query_heads * head_dim]`.
    gate: Tensor<B, 3>,
}

/// Muse-Glimmer grouped-query attention with scale-free Q/K norm and a separate gate.
#[derive(Module, Debug)]
pub struct MuseGlimmerAttention<B: Backend> {
    /// Query projection without a fused gate payload.
    pub q_proj: Linear<B>,
    /// Native key-head projection.
    pub k_proj: Linear<B>,
    /// Native value-head projection.
    pub v_proj: Linear<B>,
    /// Independent attention-output gate projection.
    pub gate_proj: Linear<B>,
    /// Projection from concatenated query heads to the residual width.
    pub o_proj: Linear<B>,
    /// Residual-stream width accepted by every input projection.
    pub hidden_size: usize,
    /// Number of independently projected query heads.
    pub query_heads: usize,
    /// Number of native key/value heads retained in the cache.
    pub key_value_heads: usize,
    /// Scalar width of each attention head.
    pub head_dim: usize,
    /// Stabilizer for weightless per-head RMS normalization.
    pub qk_norm_epsilon: f64,
    /// Multiplier applied to normalized queries before dot products.
    pub qk_scale_factor: f64,
}

impl<B: Backend> MuseGlimmerAttention<B> {
    /// Initializes exact checkpoint-shaped attention projections.
    fn init(config: &MuseGlimmerTextConfig, device: &B::Device) -> Result<Self> {
        let query_width = checked_product(
            "muse_glimmer.attention.query_width",
            config.num_attention_heads,
            config.head_dim,
        )?;
        let key_value_width = checked_product(
            "muse_glimmer.attention.key_value_width",
            config.num_key_value_heads,
            config.head_dim,
        )?;
        let projected = |input, output, bias| {
            LinearConfig::new(input, output)
                .with_bias(bias)
                .with_layout(LinearLayout::Col)
                .init(device)
        };
        Ok(Self {
            q_proj: projected(config.hidden_size, query_width, config.attention_bias),
            k_proj: projected(config.hidden_size, key_value_width, config.attention_bias),
            v_proj: projected(config.hidden_size, key_value_width, config.attention_bias),
            gate_proj: projected(config.hidden_size, query_width, false),
            o_proj: projected(query_width, config.hidden_size, config.attention_bias),
            hidden_size: config.hidden_size,
            query_heads: config.num_attention_heads,
            key_value_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            qk_norm_epsilon: config.rms_norm_eps,
            qk_scale_factor: config.qk_scale_factor,
        })
    }

    /// Projects one hidden sequence and applies Muse's scale-free Q/K normalization.
    fn project_self(&self, hidden: Tensor<B, 3>) -> Result<MuseAttentionProjection<B>> {
        let [batch, sequence, width] = hidden.dims();
        if batch == 0 || sequence == 0 || width != self.hidden_size {
            return Err(Error::shape(
                "muse_glimmer.attention_input",
                format!("[batch>0, sequence>0, {}]", self.hidden_size),
                format!("[{batch}, {sequence}, {width}]"),
            ));
        }
        let query = linear(hidden.clone(), &self.q_proj)?
            .reshape([batch, sequence, self.query_heads, self.head_dim])
            .swap_dims(1, 2);
        let key = linear(hidden.clone(), &self.k_proj)?
            .reshape([batch, sequence, self.key_value_heads, self.head_dim])
            .swap_dims(1, 2);
        let value = linear(hidden.clone(), &self.v_proj)?
            .reshape([batch, sequence, self.key_value_heads, self.head_dim])
            .swap_dims(1, 2);
        let gate = linear(hidden, &self.gate_proj)?;
        Ok(MuseAttentionProjection {
            query: scale_free_rms_norm(query, self.qk_norm_epsilon) * self.qk_scale_factor,
            key: scale_free_rms_norm(key, self.qk_norm_epsilon),
            value,
            gate,
        })
    }

    /// Evaluates already positioned attention and gates its result before `o_proj`.
    ///
    /// `mask` follows Burn's kernel convention: `true` excludes a query/key pair.
    fn attend(
        &self,
        projection: MuseAttentionProjection<B>,
        mask: Option<Tensor<B, 4, Bool>>,
        causal: bool,
    ) -> Result<Tensor<B, 3>> {
        let [batch, query_heads, query_sequence, query_dim] = projection.query.dims();
        let [key_batch, key_value_heads, key_sequence, key_dim] = projection.key.dims();
        let [value_batch, value_heads, value_sequence, value_dim] = projection.value.dims();
        let query_width = self.query_heads * self.head_dim;
        if query_heads != self.query_heads
            || query_dim != self.head_dim
            || key_batch != batch
            || value_batch != batch
            || key_value_heads != self.key_value_heads
            || value_heads != self.key_value_heads
            || key_sequence == 0
            || value_sequence != key_sequence
            || key_dim != self.head_dim
            || value_dim != self.head_dim
            || projection.gate.dims() != [batch, query_sequence, query_width]
        {
            return Err(Error::shape(
                "muse_glimmer.attention_projection",
                "compatible Q/K/V heads and a [batch, query, query_width] gate",
                format!(
                    "q={:?}, k={:?}, v={:?}, gate={:?}",
                    projection.query.dims(),
                    projection.key.dims(),
                    projection.value.dims(),
                    projection.gate.dims()
                ),
            ));
        }
        if let Some(mask) = &mask
            && mask.dims() != [batch, self.query_heads, query_sequence, key_sequence]
        {
            return Err(Error::shape(
                "muse_glimmer.attention_mask",
                format!(
                    "[{batch}, {}, {query_sequence}, {key_sequence}]",
                    self.query_heads
                ),
                format!("{:?}", mask.dims()),
            ));
        }
        let key = repeat_key_value(projection.key, self.query_heads)?;
        let value = repeat_key_value(projection.value, self.query_heads)?;
        let context = attention(
            projection.query,
            key,
            value,
            mask,
            None,
            muse_attention_options(causal),
        )
        .swap_dims(1, 2)
        .reshape([batch, query_sequence, query_width]);
        linear(context * sigmoid(projection.gate), &self.o_proj)
    }
}

/// Builds Muse's backend attention options without disguising the default scale.
///
/// Burn defines `scale: None` as `1 / sqrt(head_dim)`, which is precisely Muse's
/// checkpoint equation. On Burn 0.21, setting the same value explicitly is not a
/// harmless representation detail: CubeCL classifies any custom scale as a
/// fallback-only feature. Keeping the scale implicit therefore preserves both
/// the reference mathematics and fused-attention eligibility.
fn muse_attention_options(causal: bool) -> AttentionModuleOptions {
    AttentionModuleOptions {
        scale: None,
        softcap: None,
        is_causal: causal,
    }
}

/// One dense Muse decoder block with sandwich normalization.
#[derive(Module, Debug)]
pub struct MuseGlimmerDecoderLayer<B: Backend> {
    /// Gated grouped-query attention.
    pub self_attn: MuseGlimmerAttention<B>,
    /// Dense SwiGLU feed-forward network.
    pub mlp: GatedFeedForward<B>,
    /// Zero-centered pre-attention norm using `rms_norm_eps`.
    pub input_layernorm: RmsNorm<B>,
    /// Zero-centered post-attention norm using `post_norm_eps`.
    pub post_attention_layernorm: RmsNorm<B>,
    /// Zero-centered pre-feed-forward norm using `rms_norm_eps`.
    pub pre_feedforward_layernorm: RmsNorm<B>,
    /// Zero-centered post-feed-forward norm using `post_norm_eps`.
    pub post_feedforward_layernorm: RmsNorm<B>,
    /// Whether the attention kernel uses a local or global causal mask.
    #[module(skip)]
    pub layer_type: MuseGlimmerLayerType,
    /// Layer-specific rotary configuration; `None` means exact NoPE.
    #[module(skip)]
    rope: Option<RopeConfig>,
}

impl<B: Backend> MuseGlimmerDecoderLayer<B> {
    /// Initializes one layer from its explicit schedule and per-layer theta.
    fn init(
        config: &MuseGlimmerTextConfig,
        layer_type: MuseGlimmerLayerType,
        theta: f64,
        device: &B::Device,
    ) -> Result<Self> {
        let pre_norm = || {
            RmsNormConfig::new(config.hidden_size)
                .with_epsilon(config.rms_norm_eps)
                .with_zero_centered(true)
                .init(device)
        };
        let post_norm = || {
            RmsNormConfig::new(config.hidden_size)
                .with_epsilon(config.post_norm_eps)
                .with_zero_centered(true)
                .init(device)
        };
        Ok(Self {
            self_attn: MuseGlimmerAttention::init(config, device)?,
            mlp: GatedFeedForwardConfig::new(config.hidden_size, config.intermediate_size)
                .init(device)?,
            input_layernorm: pre_norm()?,
            post_attention_layernorm: post_norm()?,
            pre_feedforward_layernorm: pre_norm()?,
            post_feedforward_layernorm: post_norm()?,
            layer_type,
            rope: (theta > 0.0)
                .then(|| muse_rope_config(config, theta))
                .transpose()?,
        })
    }

    /// Evaluates one stateless sandwich-normalized residual layer.
    fn forward(
        &self,
        hidden: Tensor<B, 3>,
        positions: Tensor<B, 2, Int>,
        token_mask: Option<&TokenMask<B>>,
        evaluated_context: usize,
        sliding_window: Option<usize>,
    ) -> Result<Tensor<B, 3>> {
        let residual = hidden.clone();
        let normalized = self.input_layernorm.forward(hidden);
        let mut projection = self.self_attn.project_self(normalized)?;
        if let Some(rope) = self.rope {
            let angles = rotary_angles(rope, positions, evaluated_context)?;
            (projection.query, projection.key) =
                apply_rotary(projection.query, projection.key, &angles)?;
        }
        let [batch, _, sequence, _] = projection.query.dims();
        let local_window = match self.layer_type {
            MuseGlimmerLayerType::SlidingAttention => Some(sliding_window.ok_or_else(|| {
                Error::configuration(
                    "muse_glimmer.sliding_window",
                    "is required by a sliding-attention layer",
                )
            })?),
            MuseGlimmerLayerType::FullAttention => None,
        };
        let mask = stateless_attention_mask(
            token_mask,
            batch,
            self.self_attn.query_heads,
            sequence,
            local_window,
            &projection.query.device(),
        )?;
        let attended = self.self_attn.attend(projection, mask, true)?;
        let hidden = residual + self.post_attention_layernorm.forward(attended);
        let residual = hidden.clone();
        let fed =
            muse_gated_feed_forward(&self.mlp, self.pre_feedforward_layernorm.forward(hidden))?;
        Ok(residual + self.post_feedforward_layernorm.forward(fed))
    }

    /// Returns whether this layer applies rotary positions.
    #[must_use]
    pub const fn uses_rope(&self) -> bool {
        self.rope.is_some()
    }
}

/// Muse token embedding, dense decoder stack, and final learned normalization.
#[derive(Module, Debug)]
pub struct MuseGlimmerModel<B: Backend> {
    /// Raw token embedding table; a scale-free RMS norm is applied after lookup.
    pub embed_tokens: Embedding<B>,
    /// Ordered hybrid decoder layers.
    pub layers: Vec<MuseGlimmerDecoderLayer<B>>,
    /// Conventional learned RMSNorm after the final decoder layer.
    pub norm: RmsNorm<B>,
    /// Validated family configuration excluded from parameter records.
    #[module(skip)]
    pub config: MuseGlimmerTextConfig,
}

impl<B: Backend> MuseGlimmerModel<B> {
    /// Initializes the exact dense text topology without allocating cache state.
    pub fn init(config: MuseGlimmerTextConfig, device: &B::Device) -> Result<Self> {
        ArchitectureConfig::validate(&config)?;
        if config.attention_dropout != 0.0 {
            return Err(Error::BackendCapability(
                "Muse-Glimmer inference supports the released zero attention dropout only".into(),
            ));
        }
        let schedule = config.resolved_layer_types();
        let theta = config.resolved_layer_rope_theta();
        let layers = schedule
            .into_iter()
            .zip(theta)
            .map(|(layer_type, theta)| {
                MuseGlimmerDecoderLayer::init(&config, layer_type, theta, device)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            embed_tokens: EmbeddingConfig::new(config.vocab_size, config.hidden_size).init(device),
            layers,
            norm: RmsNormConfig::new(config.hidden_size)
                .with_epsilon(config.rms_norm_eps)
                .init(device)?,
            config,
        })
    }

    /// Evaluates the stateless backbone and retains only requested hidden states.
    fn forward(
        &self,
        input: &CausalLmInput<B>,
        retain_hidden_states: bool,
    ) -> Result<(Tensor<B, 3>, Vec<Tensor<B, 3>>)> {
        let [batch, sequence] = input.token_ids.dims();
        validate_nonempty_input(batch, sequence)?;
        validate_input_devices(input, &self.embed_tokens.weight.lazy_device())?;
        validate_token_ids(input.token_ids.clone(), self.config.vocab_size)?;
        validate_mask(input.token_mask.as_ref(), batch, sequence, false)?;
        let positions = resolve_positions(input, batch, sequence)?;
        validate_position_range(positions.clone(), self.config.max_position_embeddings)?;
        let evaluated_context = evaluated_context(positions.clone())?;
        let mut hidden = scale_free_rms_norm(
            self.embed_tokens.forward(input.token_ids.clone()),
            self.config.rms_norm_eps,
        );
        let mut hidden_states = Vec::with_capacity(if retain_hidden_states {
            self.layers.len() + 2
        } else {
            0
        });
        if retain_hidden_states {
            hidden_states.push(hidden.clone());
        }
        let window = self.config.resolved_sliding_window();
        for layer in &self.layers {
            hidden = layer.forward(
                hidden,
                positions.clone(),
                input.token_mask.as_ref(),
                evaluated_context,
                window,
            )?;
            if retain_hidden_states {
                hidden_states.push(hidden.clone());
            }
        }
        hidden = self.norm.forward(hidden);
        if retain_hidden_states {
            hidden_states.push(hidden.clone());
        }
        Ok((hidden, hidden_states))
    }
}

/// Decoder-only Muse-Glimmer language model.
#[derive(Module, Debug)]
pub struct MuseGlimmerForCausalLm<B: Backend> {
    /// Dense hybrid text backbone.
    pub model: MuseGlimmerModel<B>,
    /// Independent vocabulary projection, absent only for experimental tied configs.
    pub lm_head: Option<Linear<B>>,
}

impl<B: Backend> MuseGlimmerForCausalLm<B> {
    /// Initializes a differentiable full-precision graph.
    pub fn init(config: MuseGlimmerTextConfig, device: &B::Device) -> Result<Self> {
        ArchitectureConfig::validate(&config)?;
        let hidden = config.hidden_size;
        let vocabulary = config.vocab_size;
        let tied = config.tie_word_embeddings;
        Ok(Self {
            model: MuseGlimmerModel::init(config, device)?,
            lm_head: (!tied).then(|| {
                LinearConfig::new(hidden, vocabulary)
                    .with_bias(false)
                    .with_layout(LinearLayout::Col)
                    .init(device)
            }),
        })
    }

    /// Evaluates stateless causal inference and applies Muse's final logit softcap.
    pub fn forward(
        &self,
        input: CausalLmInput<B>,
        options: &ForwardOptions,
    ) -> Result<CausalLmOutput<B>> {
        if options.attentions {
            return Err(Error::BackendCapability(
                "Muse-Glimmer attention-probability capture is not exposed by Burn's fused path"
                    .into(),
            ));
        }
        let (hidden, hidden_states) = self.model.forward(&input, options.hidden_states)?;
        let selected = select_hidden(hidden.clone(), input.token_mask.as_ref(), &options.logits)?;
        let logits = self.project_logits(selected)?;
        Ok(CausalLmOutput {
            logits,
            final_hidden_state: options.hidden_states.then_some(hidden),
            hidden_states,
            attentions: Vec::new(),
            routers: Vec::new(),
        })
    }

    /// Returns the validated text configuration.
    #[must_use]
    pub const fn config(&self) -> &MuseGlimmerTextConfig {
        &self.model.config
    }

    /// Projects selected hidden states and applies `T * tanh(m * logits / T)`.
    fn project_logits(&self, hidden: Tensor<B, 3>) -> Result<Tensor<B, 3>> {
        let hidden_device = hidden.device();
        let raw = match &self.lm_head {
            Some(head) => {
                ensure_model_device(
                    "muse_glimmer.lm_head",
                    &head.weight.lazy_device(),
                    &hidden_device,
                )?;
                linear(hidden, head)?
            }
            None => {
                ensure_model_device(
                    "muse_glimmer.model.embed_tokens",
                    &self.model.embed_tokens.weight.lazy_device(),
                    &hidden_device,
                )?;
                tied_projection(hidden, self.model.embed_tokens.weight.val())?
            }
        };
        let cap = self.config().final_logit_softcapping;
        Ok(((raw * self.config().output_multiplier) / cap).tanh() * cap)
    }
}

/// Applies a learned-scale-free RMS norm in FP32 and restores the input dtype.
fn scale_free_rms_norm<B: Backend, const D: usize>(
    input: Tensor<B, D>,
    epsilon: f64,
) -> Tensor<B, D> {
    let dtype = input.dtype();
    let input = input.cast(DType::F32);
    let rms = (input.clone().square().mean_dim(D - 1) + epsilon).sqrt();
    (input / rms).cast(dtype)
}

/// Applies a tied embedding matrix as a vocabulary projection.
fn tied_projection<B: Backend>(
    hidden: Tensor<B, 3>,
    embedding: Tensor<B, 2>,
) -> Result<Tensor<B, 3>> {
    let [batch, sequence, hidden_width] = hidden.dims();
    let [vocabulary, embedding_width] = embedding.dims();
    if hidden_width != embedding_width || vocabulary == 0 {
        return Err(Error::shape(
            "muse_glimmer.tied_lm_head",
            format!("embedding [vocab>0, {hidden_width}]"),
            format!("[{vocabulary}, {embedding_width}]"),
        ));
    }
    Ok(tensor_linear(hidden, embedding.transpose(), None).reshape([batch, sequence, vocabulary]))
}

/// Evaluates Muse's SwiGLU with packed-weight-preserving linear dispatch.
///
/// The equation is `down(SiLU(gate(x)) * up(x))`. Keeping all three maps at
/// the shared [`linear`] boundary lets CubeCL consume Q4/Q8 weights directly
/// instead of materializing a complete dequantized projection matrix.
fn muse_gated_feed_forward<B: Backend, const D: usize>(
    mlp: &GatedFeedForward<B>,
    input: Tensor<B, D>,
) -> Result<Tensor<B, D>> {
    let gate = apply_activation(linear(input.clone(), &mlp.gate_proj)?, mlp.activation);
    let content = linear(input, &mlp.up_proj)?;
    linear(gate * content, &mlp.down_proj)
}

/// Combines padding exclusions with an optional left-bounded causal window.
fn stateless_attention_mask<B: Backend>(
    token_mask: Option<&TokenMask<B>>,
    batch: usize,
    heads: usize,
    sequence: usize,
    window: Option<usize>,
    device: &B::Device,
) -> Result<Option<Tensor<B, 4, Bool>>> {
    let padding = padding_attention_mask(token_mask, token_mask, batch, heads, sequence, sequence)?;
    let Some(window) = window else {
        return Ok(padding);
    };
    let query = Tensor::<B, 1, Int>::arange(0..sequence as i64, device)
        .reshape([1, 1, sequence, 1])
        .expand([batch, heads, sequence, sequence]);
    let key = Tensor::<B, 1, Int>::arange(0..sequence as i64, device)
        .reshape([1, 1, 1, sequence])
        .expand([batch, heads, sequence, sequence]);
    let too_old = (key + window as i64).lower_equal(query);
    Ok(Some(match padding {
        Some(padding) => padding.bool_or(too_old),
        None => too_old,
    }))
}

/// Converts per-layer serialized RoPE metadata to a shared validated config.
fn muse_rope_config(config: &MuseGlimmerTextConfig, theta: f64) -> Result<RopeConfig> {
    let parameters = config.rope_parameters.clone().unwrap_or_default();
    let scaling = match parameters.rope_type {
        RopeType::Default => RopeScaling::None,
        RopeType::Linear => LinearRopeScaling::new(parameters.factor.unwrap_or(1.0))
            .map(RopeScaling::from)
            .map_err(|error| Error::configuration("muse_glimmer.rope", error.to_string()))?,
        RopeType::Dynamic => DynamicNtkScaling::new(
            parameters.factor.unwrap_or(1.0),
            parameters
                .original_max_position_embeddings
                .unwrap_or(config.max_position_embeddings),
        )
        .map(RopeScaling::from)
        .map_err(|error| Error::configuration("muse_glimmer.rope", error.to_string()))?,
        RopeType::Yarn => {
            let mut yarn = YarnConfig::new(
                parameters.factor.unwrap_or(1.0),
                parameters
                    .original_max_position_embeddings
                    .unwrap_or(config.max_position_embeddings),
            )
            .map_err(|error| Error::configuration("muse_glimmer.rope", error.to_string()))?;
            if let Some(factor) = parameters.attention_factor {
                yarn = yarn.with_attention_factor(factor).map_err(|error| {
                    Error::configuration("muse_glimmer.rope", error.to_string())
                })?;
            }
            yarn = yarn
                .with_beta_overrides(parameters.beta_fast, parameters.beta_slow)
                .map_err(|error| Error::configuration("muse_glimmer.rope", error.to_string()))?;
            RopeScaling::Yarn(yarn)
        }
        unsupported => {
            return Err(Error::configuration(
                "muse_glimmer.rope_type",
                format!("{unsupported:?} is not implemented for Muse-Glimmer"),
            ));
        }
    };
    RopeConfig::new(config.head_dim, config.head_dim, theta, scaling)
        .map_err(|error| Error::configuration("muse_glimmer.rope", error.to_string()))
}

/// Rejects empty tensor extents before embedding or reduction kernels execute.
fn validate_nonempty_input(batch: usize, sequence: usize) -> Result<()> {
    if batch == 0 || sequence == 0 {
        return Err(Error::shape(
            "muse_glimmer.input_ids",
            "non-empty [batch, sequence]",
            format!("[{batch}, {sequence}]"),
        ));
    }
    Ok(())
}

/// Rejects stateless input tensors that do not share the model parameter device.
///
/// Device identity is checked before any synchronizing reduction, positional
/// arithmetic, or embedding lookup so callers receive a contextual error
/// rather than a backend-specific mixed-device failure.
fn validate_input_devices<B: Backend>(
    input: &CausalLmInput<B>,
    model_device: &B::Device,
) -> Result<()> {
    ensure_model_device(
        "muse_glimmer.input_ids",
        &input.token_ids.device(),
        model_device,
    )?;
    if let Some(mask) = &input.token_mask {
        ensure_model_device(
            "muse_glimmer.token_mask",
            &mask.as_tensor().device(),
            model_device,
        )?;
    }
    if let Some(positions) = &input.position_ids {
        ensure_model_device(
            "muse_glimmer.position_ids",
            &positions.device(),
            model_device,
        )?;
    }
    Ok(())
}

/// Reports one stateless tensor/parameter device mismatch contextually.
fn ensure_model_device<D: PartialEq>(tensor: &str, actual: &D, expected: &D) -> Result<()> {
    if actual != expected {
        return Err(Error::BackendCapability(format!(
            "{tensor} and Muse-Glimmer model parameters are on different devices"
        )));
    }
    Ok(())
}

/// Validates token values before the embedding backend receives indices.
fn validate_token_ids<B: Backend>(token_ids: Tensor<B, 2, Int>, vocabulary: usize) -> Result<()> {
    let minimum = token_ids
        .clone()
        .min()
        .try_into_scalar()
        .map_err(|error| Error::BackendCapability(format!("failed to inspect token IDs: {error}")))?
        .elem::<i64>();
    let maximum = token_ids
        .max()
        .try_into_scalar()
        .map_err(|error| Error::BackendCapability(format!("failed to inspect token IDs: {error}")))?
        .elem::<i64>();
    if minimum < 0 || maximum as usize >= vocabulary {
        return Err(Error::shape(
            "muse_glimmer.input_ids",
            format!("values in 0..{vocabulary}"),
            format!("observed range {minimum}..={maximum}"),
        ));
    }
    Ok(())
}

/// Validates a semantic token mask and its nonempty-row policy.
fn validate_mask<B: Backend>(
    mask: Option<&TokenMask<B>>,
    batch: usize,
    sequence: usize,
    permit_empty_rows: bool,
) -> Result<()> {
    let Some(mask) = mask else {
        return Ok(());
    };
    if mask.as_tensor().dims() != [batch, sequence] {
        return Err(Error::shape(
            "muse_glimmer.token_mask",
            format!("[{batch}, {sequence}]"),
            format!("{:?}", mask.as_tensor().dims()),
        ));
    }
    if !permit_empty_rows {
        let minimum = mask
            .as_tensor()
            .clone()
            .int()
            .sum_dim(1)
            .min()
            .try_into_scalar()
            .map_err(|error| Error::BackendCapability(format!("failed to inspect mask: {error}")))?
            .elem::<i64>();
        if minimum == 0 {
            return Err(Error::shape(
                "muse_glimmer.token_mask",
                "at least one valid token in every prompt row",
                "a fully masked row",
            ));
        }
    }
    Ok(())
}

/// Resolves explicit positions or padding-independent cumulative token ordinals.
fn resolve_positions<B: Backend>(
    input: &CausalLmInput<B>,
    batch: usize,
    sequence: usize,
) -> Result<Tensor<B, 2, Int>> {
    if let Some(positions) = &input.position_ids {
        if positions.dims() != [batch, sequence] {
            return Err(Error::shape(
                "muse_glimmer.position_ids",
                format!("[{batch}, {sequence}]"),
                format!("{:?}", positions.dims()),
            ));
        }
        return Ok(positions.clone());
    }
    let device = input.token_ids.device();
    match &input.token_mask {
        Some(mask) => {
            let valid = mask.as_tensor().clone();
            Ok((valid.clone().int().cumsum(1) - 1).mask_fill(valid.bool_not(), 0))
        }
        None => Ok(Tensor::<B, 1, Int>::arange(0..sequence as i64, &device)
            .reshape([1, sequence])
            .expand([batch, sequence])),
    }
}

/// Validates absolute position IDs against the configured context bound.
fn validate_position_range<B: Backend>(
    positions: Tensor<B, 2, Int>,
    capacity: usize,
) -> Result<()> {
    let minimum = positions
        .clone()
        .min()
        .try_into_scalar()
        .map_err(|error| Error::BackendCapability(format!("failed to inspect positions: {error}")))?
        .elem::<i64>();
    let maximum = positions
        .max()
        .try_into_scalar()
        .map_err(|error| Error::BackendCapability(format!("failed to inspect positions: {error}")))?
        .elem::<i64>();
    if minimum < 0 || maximum as usize >= capacity {
        return Err(Error::shape(
            "muse_glimmer.position_ids",
            format!("values in 0..{capacity}"),
            format!("observed range {minimum}..={maximum}"),
        ));
    }
    Ok(())
}

/// Returns one plus the largest position in a nonempty tensor.
fn evaluated_context<B: Backend>(positions: Tensor<B, 2, Int>) -> Result<usize> {
    let maximum = positions
        .max()
        .try_into_scalar()
        .map_err(|error| Error::BackendCapability(format!("failed to inspect positions: {error}")))?
        .elem::<i64>();
    usize::try_from(maximum)
        .map_err(|_| Error::configuration("muse_glimmer.position_ids", "must be non-negative"))?
        .checked_add(1)
        .ok_or_else(|| Error::configuration("muse_glimmer.position_ids", "context overflows usize"))
}

/// Selects hidden positions before the expensive vocabulary projection.
fn select_hidden<B: Backend>(
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
                let device = hidden.device();
                let positions = Tensor::<B, 1, Int>::arange(0..sequence as i64, &device)
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
            let indices = positions
                .iter()
                .map(|&value| value as i64)
                .collect::<Vec<_>>();
            let count = indices.len();
            let device = hidden.device();
            Ok(hidden.select(
                1,
                Tensor::<B, 1, Int>::from_data(TensorData::new(indices, [count]), &device),
            ))
        }
    }
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

/// Multiplies two shape terms without integer wraparound.
fn checked_product(field: &'static str, left: usize, right: usize) -> Result<usize> {
    left.checked_mul(right)
        .ok_or_else(|| Error::configuration(field, "dimension product overflows usize"))
}

#[cfg(test)]
mod tests {
    use burn::{
        backend::Flex,
        module::Param,
        tensor::{Int, Tensor, TensorData, Tolerance, backend::BackendTypes, f16},
    };

    use super::*;
    use crate::{
        CausalLmInput, ForwardOptions, checkpoint::TensorSink,
        models::muse_glimmer::MuseGlimmerCheckpointOptions,
    };

    type TestBackend = Flex<f32, i32>;

    /// Builds a four-layer miniature with the released local/local/local/global schedule.
    fn tiny_config() -> MuseGlimmerTextConfig {
        MuseGlimmerTextConfig {
            vocab_size: 32,
            hidden_size: 8,
            intermediate_size: 12,
            num_hidden_layers: 4,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 4,
            max_position_embeddings: 32,
            sliding_window: Some(3),
            bos_token_id: Some(crate::models::config::TokenIds::One(1)),
            eos_token_id: Some(crate::models::config::TokenIds::One(2)),
            ..Default::default()
        }
    }

    /// Installs deterministic small parameters without relying on process-global RNG state.
    fn initialized_model<B: Backend>(
        config: MuseGlimmerTextConfig,
        device: &B::Device,
    ) -> MuseGlimmerForCausalLm<B> {
        let contract = config
            .checkpoint_contract(MuseGlimmerCheckpointOptions::default())
            .expect("tiny contract");
        let mut model = MuseGlimmerForCausalLm::init(config, device).expect("tiny model");
        for (tensor_index, expected) in contract.expected_tensors().iter().enumerate() {
            let shape = expected
                .shape
                .iter()
                .map(|&dimension| dimension as usize)
                .collect::<Vec<_>>();
            let elements = shape.iter().product();
            let values = (0..elements)
                .map(|element| {
                    let code = (element + tensor_index * 5) % 29;
                    (code as f32 - 14.0) * 0.003
                })
                .collect::<Vec<_>>();
            TensorSink::write_tensor(&mut model, &expected.name, TensorData::new(values, shape))
                .unwrap_or_else(|error| panic!("{}: {error}", expected.name));
        }
        model
    }

    /// Returns an initialized miniature and its device.
    fn tiny_model() -> (
        MuseGlimmerForCausalLm<TestBackend>,
        <TestBackend as BackendTypes>::Device,
    ) {
        let device = Default::default();
        (initialized_model(tiny_config(), &device), device)
    }

    /// Builds an isolated attention fixture whose values expose its softmax weights.
    ///
    /// The value rows are basis vectors and the output projection is the identity,
    /// so the first three result channels equal the three attention probabilities.
    /// A saturated gate retains them unchanged. This makes the fixture sensitive
    /// to a one-ULP half-softmax regression without involving checkpoint loading.
    fn softmax_oracle_attention(
        dtype: DType,
    ) -> (
        MuseGlimmerAttention<TestBackend>,
        MuseAttentionProjection<TestBackend>,
    ) {
        let device = Default::default();
        let config = MuseGlimmerTextConfig {
            hidden_size: 4,
            intermediate_size: 8,
            num_hidden_layers: 1,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            head_dim: 4,
            max_position_embeddings: 8,
            sliding_window: Some(2),
            layer_types: vec![MuseGlimmerLayerType::FullAttention],
            layer_rope_theta: vec![0.0],
            vocab_size: 8,
            bos_token_id: None,
            eos_token_id: None,
            ..Default::default()
        };
        let mut attention = MuseGlimmerAttention::<TestBackend>::init(&config, &device).unwrap();
        attention.o_proj.weight = Param::from_tensor(
            Tensor::from_floats(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                &device,
            )
            .cast(dtype),
        );
        let projection = MuseAttentionProjection {
            // With head_dim=4 the implicit scale is 1/2, so these dot
            // products produce exact scores [-0.125, 5.875, 0.4375].
            query: Tensor::from_floats([[[[2.0, 0.0, 0.0, 0.0]]]], &device).cast(dtype),
            key: Tensor::<TestBackend, 3>::from_floats(
                [[
                    [-0.125, 0.0, 0.0, 0.0],
                    [5.875, 0.0, 0.0, 0.0],
                    [0.4375, 0.0, 0.0, 0.0],
                ]],
                &device,
            )
            .unsqueeze_dim::<4>(0)
            .cast(dtype),
            value: Tensor::<TestBackend, 3>::from_floats(
                [[
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                ]],
                &device,
            )
            .unsqueeze_dim::<4>(0)
            .cast(dtype),
            gate: Tensor::<TestBackend, 3>::full([1, 1, 4], 80.0, &device).cast(dtype),
        };
        (attention, projection)
    }

    /// F32 attention matches the literal upstream eager-softmax oracle.
    ///
    /// The expected probabilities were generated from
    /// `eager_attention_forward` in Hugging Face Transformers revision
    /// `cfa309cfa34b5497a32fd50af49aa97aac94c054`. Keeping literals here makes
    /// the test independent of Burn's own softmax implementation.
    #[test]
    fn attention_matches_pinned_fp32_softmax_oracle() {
        let (attention, projection) = softmax_oracle_attention(DType::F32);
        let actual = attention
            .attend(projection, None, false)
            .expect("valid isolated attention");
        let expected = TensorData::new(
            vec![0.002_461_939_6_f32, 0.993_217_3, 0.004_320_838_5, 0.0],
            [1, 1, 4],
        );
        actual
            .into_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::absolute(1e-7));
    }

    /// F16 attention rounds the pinned FP32-softmax result, not a half-softmax.
    ///
    /// The third expected half has bits `0x1c6d`; evaluating exponentiation and
    /// normalization in F16 instead produces adjacent value `0x1c6c`. This
    /// one-ULP discriminator directly covers Muse's explicit upstream
    /// `softmax(..., dtype=torch.float32).to(query.dtype)` requirement at
    /// Transformers revision `cfa309cfa34b5497a32fd50af49aa97aac94c054`.
    #[test]
    fn f16_attention_uses_fp32_softmax_before_rounding() {
        let (attention, projection) = softmax_oracle_attention(DType::F16);
        let actual = attention
            .attend(projection, None, false)
            .expect("valid isolated attention");
        assert_eq!(actual.dtype(), DType::F16);
        let bits = actual
            .into_data()
            .to_vec::<f16>()
            .expect("F16 result")
            .into_iter()
            .map(f16::to_bits)
            .collect::<Vec<_>>();
        assert_eq!(bits, [0x190b, 0x3bf2, 0x1c6d, 0x0000]);
        assert_ne!(bits[2], 0x1c6c, "half-precision softmax sentinel");
    }

    /// Default Muse options retain Burn's optimized-attention dispatch contract.
    ///
    /// Burn 0.21 CubeCL falls back whenever `scale.is_some()`, even when that
    /// value redundantly equals `1/sqrt(head_dim)`. This structural assertion
    /// prevents the family wiring from silently disabling fused kernels again.
    #[test]
    fn attention_options_preserve_fused_dispatch_eligibility() {
        assert_eq!(
            muse_attention_options(true),
            AttentionModuleOptions {
                scale: None,
                softcap: None,
                is_causal: true,
            }
        );
        assert_eq!(muse_attention_options(false).scale, None);
    }

    /// Stateless inference emits exact requested shapes and dense diagnostics.
    #[test]
    fn stateless_forward_shapes_and_options_are_exact() {
        let (model, device) = tiny_model();
        let output = model
            .forward(
                CausalLmInput {
                    token_ids: Tensor::<TestBackend, 2, Int>::from_ints(
                        [[1, 2, 3], [4, 5, 6]],
                        &device,
                    ),
                    token_mask: None,
                    position_ids: None,
                },
                &ForwardOptions {
                    logits: LogitsSelection::All,
                    hidden_states: true,
                    ..Default::default()
                },
            )
            .expect("tiny forward");
        assert_eq!(output.logits.dims(), [2, 3, 32]);
        assert_eq!(output.hidden_states.len(), 6);
        assert_eq!(output.final_hidden_state.unwrap().dims(), [2, 3, 8]);
        assert!(output.attentions.is_empty());
        assert!(output.routers.is_empty());
    }

    /// NoPE makes a global layer invariant to otherwise different position IDs.
    #[test]
    fn nope_layer_does_not_consume_positions() {
        let (model, device) = tiny_model();
        let layer = &model.model.layers[3];
        assert!(!layer.uses_rope());
        let hidden = Tensor::<TestBackend, 3>::from_floats(
            [[[0.1, -0.2, 0.3, 0.4, -0.5, 0.6, 0.7, -0.8]]],
            &device,
        );
        let first = layer
            .forward(
                hidden.clone(),
                Tensor::from_ints([[0]], &device),
                None,
                1,
                None,
            )
            .unwrap();
        let later = layer
            .forward(hidden, Tensor::from_ints([[17]], &device), None, 18, None)
            .unwrap();
        first
            .into_data()
            .assert_approx_eq::<f32>(&later.into_data(), Tolerance::absolute(1e-6));
    }

    /// The separate gate is sigmoid-activated before the output projection.
    #[test]
    fn attention_gate_precedes_output_projection() {
        let device = Default::default();
        let config = MuseGlimmerTextConfig {
            hidden_size: 2,
            intermediate_size: 4,
            num_hidden_layers: 1,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            head_dim: 2,
            max_position_embeddings: 8,
            sliding_window: Some(2),
            layer_types: vec![MuseGlimmerLayerType::FullAttention],
            layer_rope_theta: vec![0.0],
            vocab_size: 8,
            bos_token_id: None,
            eos_token_id: None,
            ..Default::default()
        };
        let mut attention = MuseGlimmerAttention::<TestBackend>::init(&config, &device).unwrap();
        let identity = Tensor::from_floats([[1.0, 0.0], [0.0, 1.0]], &device);
        attention.q_proj.weight = Param::from_tensor(identity.clone());
        attention.k_proj.weight = Param::from_tensor(identity.clone());
        attention.v_proj.weight = Param::from_tensor(identity.clone());
        attention.o_proj.weight = Param::from_tensor(identity);
        attention.gate_proj.weight = Param::from_tensor(Tensor::zeros([2, 2], &device));
        let hidden = Tensor::from_floats([[[3.0, 4.0]]], &device);
        let projection = attention.project_self(hidden).unwrap();
        let value = projection.value.clone().swap_dims(1, 2).reshape([1, 1, 2]);
        let actual = attention.attend(projection, None, true).unwrap();
        (actual * 2.0)
            .into_data()
            .assert_approx_eq::<f32>(&value.into_data(), Tolerance::absolute(1e-6));
    }

    /// Query heads are scale-free normalized and then multiplied by the configured factor.
    #[test]
    fn query_normalization_applies_extra_scale() {
        let (model, device) = tiny_model();
        let attention = &model.model.layers[0].self_attn;
        let hidden = Tensor::<TestBackend, 3>::ones([1, 1, 8], &device);
        let projection = attention.project_self(hidden).unwrap();
        let query_rms = projection
            .query
            .square()
            .mean_dim(3)
            .sqrt()
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        for value in query_rms {
            assert!((value - attention.qk_scale_factor as f32).abs() < 2e-3);
        }
    }

    /// The two residual branches follow the documented four-norm sandwich order.
    #[test]
    fn decoder_layer_matches_explicit_sandwich_equation() {
        let (model, device) = tiny_model();
        let layer = &model.model.layers[3];
        let hidden = Tensor::<TestBackend, 3>::from_floats(
            [[[0.2, -0.1, 0.3, -0.4, 0.5, -0.6, 0.7, 0.8]]],
            &device,
        );
        let position = Tensor::from_ints([[0]], &device);
        let normalized = layer.input_layernorm.forward(hidden.clone());
        let projection = layer.self_attn.project_self(normalized).unwrap();
        let attended = layer.self_attn.attend(projection, None, true).unwrap();
        let after_attention = hidden.clone() + layer.post_attention_layernorm.forward(attended);
        let fed = layer
            .mlp
            .forward(
                layer
                    .pre_feedforward_layernorm
                    .forward(after_attention.clone()),
            )
            .expect("validated feed-forward projection");
        let expected = after_attention + layer.post_feedforward_layernorm.forward(fed);
        let actual = layer.forward(hidden, position, None, 1, Some(3)).unwrap();
        actual
            .into_data()
            .assert_approx_eq::<f32>(&expected.into_data(), Tolerance::absolute(1e-6));
    }

    /// Final logits match the exact output multiplier and tanh softcap equation.
    #[test]
    fn final_logit_softcap_matches_closed_form_and_bounds() {
        let (model, device) = tiny_model();
        let hidden = Tensor::<TestBackend, 3>::from_floats(
            [[[0.5, -1.0, 1.5, 0.25, -0.75, 2.0, 0.0, -0.5]]],
            &device,
        );
        let raw = linear(hidden.clone(), model.lm_head.as_ref().unwrap()).unwrap();
        let expected = ((raw * model.config().output_multiplier)
            / model.config().final_logit_softcapping)
            .tanh()
            * model.config().final_logit_softcapping;
        let actual = model.project_logits(hidden).unwrap();
        actual
            .clone()
            .into_data()
            .assert_approx_eq::<f32>(&expected.into_data(), Tolerance::absolute(1e-6));
        assert!(
            actual.abs().max().into_scalar().elem::<f32>()
                <= model.config().final_logit_softcapping as f32
        );
    }

    /// Invalid input values and unsupported diagnostics return contextual errors.
    #[test]
    fn public_forward_rejects_invalid_inputs_without_panicking() {
        let (model, device) = tiny_model();
        let invalid_token = model.forward(
            CausalLmInput {
                token_ids: Tensor::from_ints([[32]], &device),
                token_mask: None,
                position_ids: None,
            },
            &ForwardOptions::default(),
        );
        assert!(invalid_token.is_err());
        let attention_request = model.forward(
            CausalLmInput {
                token_ids: Tensor::from_ints([[1]], &device),
                token_mask: None,
                position_ids: None,
            },
            &ForwardOptions {
                attentions: true,
                ..Default::default()
            },
        );
        assert!(matches!(
            attention_request,
            Err(Error::BackendCapability(_))
        ));
    }

    /// Stateless device preflight identifies every independently supplied tensor.
    #[test]
    fn stateless_device_affinity_errors_are_contextual() {
        assert!(ensure_model_device("muse_glimmer.input_ids", &0_u8, &0_u8).is_ok());
        for tensor in [
            "muse_glimmer.input_ids",
            "muse_glimmer.token_mask",
            "muse_glimmer.position_ids",
            "muse_glimmer.lm_head",
        ] {
            let error = ensure_model_device(tensor, &1_u8, &0_u8)
                .expect_err("different device identities must fail preflight");
            assert!(error.to_string().contains(tensor));
            assert!(error.to_string().contains("model parameters"));
        }
    }

    /// Stateless graph construction remains differentiable through embeddings and logits.
    #[cfg(feature = "autodiff")]
    #[test]
    fn autodiff_smoke_reaches_embedding_weights() {
        use burn::backend::Autodiff;

        type AdBackend = Autodiff<TestBackend>;
        let device = Default::default();
        let model = MuseGlimmerForCausalLm::<AdBackend>::init(tiny_config(), &device).unwrap();
        let loss = model
            .forward(
                CausalLmInput {
                    token_ids: Tensor::<AdBackend, 2, Int>::from_ints([[1, 2]], &device),
                    token_mask: None,
                    position_ids: None,
                },
                &ForwardOptions::default(),
            )
            .unwrap()
            .logits
            .sum();
        let gradients = loss.backward();
        assert!(model.model.embed_tokens.weight.grad(&gradients).is_some());
    }
}
