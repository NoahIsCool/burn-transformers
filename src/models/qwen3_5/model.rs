//! Burn implementation of the Qwen3.5/Qwen3.8 text decoder.
//!
//! The architecture is a pre-normalized residual decoder. Three Gated DeltaNet
//! layers normally alternate with one gated full-attention layer, while every
//! layer shares the same dense SwiGLU feed-forward topology. Qwen3.8 stores
//! zero-centered RMSNorm parameters (`scale = 1 + weight`) throughout the text
//! decoder and rotates only a prefix of each full-attention head.
//!
//! This module owns exact family wiring. Projection, rotary, attention,
//! DeltaNet, feed-forward, and normalization mathematics remain in shared
//! modules so custom experiments can recombine them without model-name checks.

use burn::{
    module::Module,
    nn::{Embedding, EmbeddingConfig, Linear, LinearConfig, LinearLayout},
    tensor::{ElementConversion, Int, Tensor, TensorData, backend::Backend},
};

use crate::{
    ArchitectureConfig, CausalLmInput, CausalLmOutput, Error, ForwardOptions, LogitsSelection,
    Result,
    architecture::TokenMask,
    attention::{
        AttentionOutputGate, GroupedQueryAttention, GroupedQueryAttentionConfig,
        padding_attention_mask,
    },
    delta_net::{GatedDeltaNet, GatedDeltaNetConfig},
    feed_forward::{GatedFeedForward, GatedFeedForwardConfig},
    normalization::{RmsNorm, RmsNormConfig},
    operations,
    position::{
        DynamicNtkScaling, LinearRopeScaling, RopeConfig, RopeScaling, YarnConfig, apply_rotary,
        rotary_angles,
    },
};

use super::{Qwen3_5LayerType, Qwen3_5TextConfig};
use crate::models::config::RopeType;

mod decode;
mod mtp;
mod speculative;

pub use decode::{Qwen3_5DecodeState, Qwen3_5LayerCache};
pub use mtp::{Qwen3_5Mtp, Qwen3_5MtpOutput, Qwen3_5MtpPrediction};
pub use speculative::{
    Qwen3_5SpeculativeConfig, Qwen3_5SpeculativeDecoder, Qwen3_5SpeculativeOutput,
};

/// One Qwen decoder block with exactly one active sequence mixer.
#[derive(Module, Debug)]
pub struct Qwen3_5DecoderLayer<B: Backend> {
    /// Gated DeltaNet module for a linear-attention layer.
    pub linear_attn: Option<GatedDeltaNet<B>>,
    /// Gated GQA module for a full-attention layer.
    pub self_attn: Option<GroupedQueryAttention<B>>,
    /// Dense SwiGLU feed-forward network.
    pub mlp: GatedFeedForward<B>,
    /// Zero-centered RMSNorm before the sequence mixer.
    pub input_layernorm: RmsNorm<B>,
    /// Zero-centered RMSNorm before the feed-forward network.
    pub post_attention_layernorm: RmsNorm<B>,
    /// Serialized mixer schedule entry; it owns no parameters.
    #[module(skip)]
    pub layer_type: Qwen3_5LayerType,
}

impl<B: Backend> Qwen3_5DecoderLayer<B> {
    /// Initializes the block selected by one schedule entry.
    fn init(
        config: &Qwen3_5TextConfig,
        layer_type: Qwen3_5LayerType,
        device: &B::Device,
    ) -> Result<Self> {
        let norm = || {
            RmsNormConfig::new(config.hidden_size)
                .with_epsilon(config.rms_norm_eps)
                .with_zero_centered(true)
                .init(device)
        };
        let (linear_attn, self_attn) = match layer_type {
            Qwen3_5LayerType::LinearAttention => {
                let mut delta = GatedDeltaNetConfig::new(
                    config.hidden_size,
                    config.linear_num_key_heads,
                    config.linear_num_value_heads,
                    config.linear_key_head_dim,
                    config.linear_value_head_dim,
                );
                delta.convolution_kernel = config.linear_conv_kernel_dim;
                delta.normalization_epsilon = config.rms_norm_eps;
                (Some(delta.init(device)?), None)
            }
            Qwen3_5LayerType::FullAttention => {
                let mut attention = GroupedQueryAttentionConfig::new(
                    config.hidden_size,
                    config.num_attention_heads,
                    config.num_key_value_heads,
                    config.head_dim,
                );
                attention.bias = config.attention_bias;
                attention.query_key_norm_epsilon = Some(config.rms_norm_eps);
                attention.query_key_norm_zero_centered = true;
                attention.output_gate = if config.attn_output_gate {
                    AttentionOutputGate::Sigmoid
                } else {
                    AttentionOutputGate::None
                };
                (None, Some(attention.init(device)?))
            }
        };
        Ok(Self {
            linear_attn,
            self_attn,
            mlp: GatedFeedForwardConfig::new(config.hidden_size, config.intermediate_size)
                .init(device)?,
            input_layernorm: norm()?,
            post_attention_layernorm: norm()?,
            layer_type,
        })
    }

    /// Evaluates one stateless residual block.
    fn forward(
        &self,
        hidden: Tensor<B, 3>,
        positions: Tensor<B, 2, Int>,
        token_mask: Option<&TokenMask<B>>,
        rope: RopeConfig,
        evaluated_context: usize,
    ) -> Result<Tensor<B, 3>> {
        let residual = hidden.clone();
        let normalized = self.input_layernorm.forward(hidden);
        let mixed = match self.layer_type {
            Qwen3_5LayerType::LinearAttention => self
                .linear_attn
                .as_ref()
                .ok_or_else(|| {
                    Error::Checkpoint("Qwen linear layer has no DeltaNet module".into())
                })?
                .forward(normalized, token_mask)?,
            Qwen3_5LayerType::FullAttention => {
                let attention = self.self_attn.as_ref().ok_or_else(|| {
                    Error::Checkpoint("Qwen full-attention layer has no attention module".into())
                })?;
                let mut projection = attention.project_self(normalized)?;
                let angles = rotary_angles(rope, positions, evaluated_context)?;
                (projection.query, projection.key) =
                    apply_rotary(projection.query, projection.key, &angles)?;
                let [batch, _, query_sequence, _] = projection.query.dims();
                let key_sequence = projection.key.dims()[2];
                let mask = padding_attention_mask(
                    token_mask,
                    token_mask,
                    batch,
                    attention.query_heads,
                    query_sequence,
                    key_sequence,
                )?;
                attention.attend(projection, mask, None, true)?
            }
        };
        let hidden = residual + mixed;
        let residual = hidden.clone();
        Ok(residual
            + self
                .mlp
                .forward(self.post_attention_layernorm.forward(hidden))?)
    }
}

/// Parameterized Qwen text backbone without the vocabulary projection.
#[derive(Module, Debug)]
pub struct Qwen3_5TextModel<B: Backend> {
    /// Token embedding table.
    pub embed_tokens: Embedding<B>,
    /// Ordered hybrid decoder blocks.
    pub layers: Vec<Qwen3_5DecoderLayer<B>>,
    /// Final zero-centered RMSNorm.
    pub norm: RmsNorm<B>,
    /// Validated family configuration excluded from parameter records.
    #[module(skip)]
    pub config: Qwen3_5TextConfig,
    /// Validated partial rotary configuration.
    #[module(skip)]
    pub rope: RopeConfig,
}

impl<B: Backend> Qwen3_5TextModel<B> {
    /// Initializes lazy parameters for the exact configured hybrid schedule.
    pub fn init(config: Qwen3_5TextConfig, device: &B::Device) -> Result<Self> {
        ArchitectureConfig::validate(&config)?;
        let rope = qwen_rope_config(&config)?;
        let layers = config
            .resolved_layer_types()
            .into_iter()
            .map(|layer_type| Qwen3_5DecoderLayer::init(&config, layer_type, device))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            embed_tokens: EmbeddingConfig::new(config.vocab_size, config.hidden_size).init(device),
            layers,
            norm: RmsNormConfig::new(config.hidden_size)
                .with_epsilon(config.rms_norm_eps)
                .with_zero_centered(true)
                .init(device)?,
            config,
            rope,
        })
    }

    /// Evaluates the text backbone and optionally retains per-layer hidden states.
    fn forward(
        &self,
        input: &CausalLmInput<B>,
        retain_hidden_states: bool,
    ) -> Result<(Tensor<B, 3>, Vec<Tensor<B, 3>>)> {
        let [batch, sequence] = input.token_ids.dims();
        if batch == 0 || sequence == 0 {
            return Err(Error::shape(
                "qwen.input_ids",
                "non-empty [batch, sequence]",
                format!("[{batch}, {sequence}]"),
            ));
        }
        validate_input_devices(input, &self.embed_tokens.weight.lazy_device())?;
        validate_token_ids(input.token_ids.clone(), self.config.vocab_size)?;
        validate_mask(input.token_mask.as_ref(), batch, sequence)?;
        let positions = resolve_positions(input, batch, sequence)?;
        decode::validate_position_range(positions.clone(), self.config.max_position_embeddings)?;
        let evaluated_context = decode::evaluated_context(positions.clone())?;
        let mut hidden = self.embed_tokens.forward(input.token_ids.clone());
        let mut hidden_states = Vec::with_capacity(if retain_hidden_states {
            self.layers.len() + 2
        } else {
            0
        });
        if retain_hidden_states {
            hidden_states.push(hidden.clone());
        }
        for layer in &self.layers {
            hidden = layer.forward(
                hidden,
                positions.clone(),
                input.token_mask.as_ref(),
                self.rope,
                evaluated_context,
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

/// Decoder-only Qwen language model with a tied or independent output head.
#[derive(Module, Debug)]
pub struct Qwen3_5ForCausalLm<B: Backend> {
    /// Hybrid text backbone.
    pub model: Qwen3_5TextModel<B>,
    /// Independent vocabulary projection, absent when the embedding is tied.
    ///
    /// Tied checkpoints reuse `model.embed_tokens.weight` directly during the
    /// forward pass. Keeping that case parameter-free guarantees one physical
    /// allocation instead of relying on two differently mapped Burn parameters.
    pub lm_head: Option<Linear<B>>,
    /// Optional official multi-token-prediction auxiliary graph.
    pub mtp: Option<Qwen3_5Mtp<B>>,
}

impl<B: Backend> Qwen3_5ForCausalLm<B> {
    /// Initializes a validated inference/training graph with lazy parameters.
    pub fn init(config: Qwen3_5TextConfig, device: &B::Device) -> Result<Self> {
        ArchitectureConfig::validate(&config)?;
        let hidden_size = config.hidden_size;
        let vocabulary_size = config.vocab_size;
        let tie_word_embeddings = config.tie_word_embeddings;
        Ok(Self {
            mtp: (config.mtp_num_hidden_layers > 0)
                .then(|| Qwen3_5Mtp::init(&config, device))
                .transpose()?,
            model: Qwen3_5TextModel::init(config, device)?,
            lm_head: (!tie_word_embeddings).then(|| {
                operations::init_linear(
                    &LinearConfig::new(hidden_size, vocabulary_size)
                        .with_bias(false)
                        .with_layout(LinearLayout::Col),
                    device,
                )
            }),
        })
    }

    /// Evaluates stateless Qwen inference with selective vocabulary projection.
    pub fn forward(
        &self,
        input: CausalLmInput<B>,
        options: &ForwardOptions,
    ) -> Result<CausalLmOutput<B>> {
        if options.attentions {
            return Err(Error::BackendCapability(
                "Qwen attention probability capture requires the reference attention path".into(),
            ));
        }
        let (hidden, hidden_states) = self.model.forward(&input, options.hidden_states)?;
        let selected = select_hidden(hidden.clone(), input.token_mask.as_ref(), &options.logits)?;
        let logits = self.project_hidden(selected)?;
        Ok(CausalLmOutput {
            logits,
            final_hidden_state: options.hidden_states.then_some(hidden),
            hidden_states,
            attentions: Vec::new(),
            routers: Vec::new(),
            load_balancing_loss: None,
        })
    }

    /// Returns the exact validated checkpoint configuration.
    pub fn config(&self) -> &Qwen3_5TextConfig {
        &self.model.config
    }

    /// Runs the shifted MTP graph and the base model's shared LM head.
    ///
    /// `next_token_ids[b,t]` is paired with the target final hidden state at the
    /// preceding autoregressive position: `x_(t+1)` combines with `h_t` to
    /// predict `x_(t+2)`. Incremental callers must preserve this one-token shift.
    pub fn forward_mtp(
        &self,
        next_token_ids: Tensor<B, 2, Int>,
        target_hidden_state: Tensor<B, 3>,
        position_ids: Tensor<B, 2, Int>,
        step: usize,
    ) -> Result<Qwen3_5MtpPrediction<B>> {
        let mtp = self.mtp.as_ref().ok_or_else(|| {
            Error::configuration(
                "qwen3_5.mtp_num_hidden_layers",
                "model was initialized without the MTP graph",
            )
        })?;
        let [batch, sequence] = next_token_ids.dims();
        if batch == 0 || sequence == 0 {
            return Err(Error::shape(
                "qwen.mtp.next_token_ids",
                "non-empty [batch, sequence]",
                format!("[{batch}, {sequence}]"),
            ));
        }
        let model_device = self.model.embed_tokens.weight.lazy_device();
        ensure_model_device(
            "qwen.mtp.next_token_ids",
            &next_token_ids.device(),
            &model_device,
        )?;
        ensure_model_device(
            "qwen.mtp.target_hidden_state",
            &target_hidden_state.device(),
            &model_device,
        )?;
        ensure_model_device(
            "qwen.mtp.position_ids",
            &position_ids.device(),
            &model_device,
        )?;
        validate_token_ids(next_token_ids.clone(), self.config().vocab_size)?;
        decode::validate_position_range(
            position_ids.clone(),
            self.config().max_position_embeddings,
        )?;
        let context = decode::evaluated_context(position_ids.clone())?;
        let embedding = self.model.embed_tokens.forward(next_token_ids);
        let output = mtp.forward(
            embedding,
            target_hidden_state,
            position_ids,
            step,
            self.model.rope,
            context,
        )?;
        let logits = self.project_hidden(output.hidden_state.clone())?;
        Ok(Qwen3_5MtpPrediction {
            logits,
            hidden_state: output.hidden_state,
        })
    }

    /// Projects normalized hidden states with the shared base-model LM head.
    pub(crate) fn project_hidden(&self, hidden: Tensor<B, 3>) -> Result<Tensor<B, 3>> {
        let hidden_device = hidden.device();
        match &self.lm_head {
            Some(head) => {
                ensure_model_device("qwen.lm_head", &head.weight.lazy_device(), &hidden_device)?;
                operations::linear(hidden, head)
            }
            None => {
                ensure_model_device(
                    "qwen.model.embed_tokens",
                    &self.model.embed_tokens.weight.lazy_device(),
                    &hidden_device,
                )?;
                Ok(burn::tensor::module::linear(
                    hidden,
                    self.model.embed_tokens.weight.val().transpose(),
                    None,
                ))
            }
        }
    }
}

/// Translates Hugging Face RoPE metadata to the shared validated implementation.
fn qwen_rope_config(config: &Qwen3_5TextConfig) -> Result<RopeConfig> {
    let parameters = config.rope_parameters.clone().unwrap_or_default();
    let theta = parameters
        .rope_theta
        .or(config.rope_theta)
        .unwrap_or(10_000.0);
    let partial = parameters
        .partial_rotary_factor
        .or(config.partial_rotary_factor)
        .unwrap_or(1.0);
    let scaling = match parameters.rope_type {
        RopeType::Default => RopeScaling::None,
        RopeType::Linear => LinearRopeScaling::new(parameters.factor.unwrap_or(1.0))
            .map(RopeScaling::from)
            .map_err(|error| Error::configuration("qwen3_5.rope", error.to_string()))?,
        RopeType::Dynamic => DynamicNtkScaling::new(
            parameters.factor.unwrap_or(1.0),
            parameters
                .original_max_position_embeddings
                .unwrap_or(config.max_position_embeddings),
        )
        .map(RopeScaling::from)
        .map_err(|error| Error::configuration("qwen3_5.rope", error.to_string()))?,
        RopeType::Yarn => {
            let mut yarn = YarnConfig::new(
                parameters.factor.unwrap_or(1.0),
                parameters
                    .original_max_position_embeddings
                    .unwrap_or(config.max_position_embeddings),
            )
            .map_err(|error| Error::configuration("qwen3_5.rope", error.to_string()))?;
            if let Some(factor) = parameters.attention_factor {
                yarn = yarn
                    .with_attention_factor(factor)
                    .map_err(|error| Error::configuration("qwen3_5.rope", error.to_string()))?;
            }
            yarn = yarn
                .with_beta_overrides(parameters.beta_fast, parameters.beta_slow)
                .map_err(|error| Error::configuration("qwen3_5.rope", error.to_string()))?;
            RopeScaling::Yarn(yarn)
        }
        unsupported => {
            return Err(Error::configuration(
                "qwen3_5.rope_type",
                format!("{unsupported:?} is not implemented for the Qwen text runtime"),
            ));
        }
    };
    RopeConfig::from_partial_factor(config.head_dim, partial, theta, scaling)
        .map_err(|error| Error::configuration("qwen3_5.rope", error.to_string()))
}

/// Rejects stateless input tensors that do not share the model parameter device.
///
/// This preflight precedes reductions, positional arithmetic, and embedding
/// lookup so mixed-device user input remains a recoverable library error.
fn validate_input_devices<B: Backend>(
    input: &CausalLmInput<B>,
    model_device: &B::Device,
) -> Result<()> {
    ensure_model_device("qwen.input_ids", &input.token_ids.device(), model_device)?;
    if let Some(mask) = &input.token_mask {
        ensure_model_device("qwen.token_mask", &mask.as_tensor().device(), model_device)?;
    }
    if let Some(positions) = &input.position_ids {
        ensure_model_device("qwen.position_ids", &positions.device(), model_device)?;
    }
    Ok(())
}

/// Reports a stateless tensor/parameter device mismatch through the error taxonomy.
fn ensure_model_device<D: PartialEq>(tensor: &str, actual: &D, expected: &D) -> Result<()> {
    if actual != expected {
        return Err(Error::BackendCapability(format!(
            "{tensor} and Qwen3.5 model parameters are on different devices"
        )));
    }
    Ok(())
}

/// Validates token values before the embedding backend sees an index.
fn validate_token_ids<B: Backend>(token_ids: Tensor<B, 2, Int>, vocabulary: usize) -> Result<()> {
    let minimum = token_ids
        .clone()
        .min()
        .try_into_scalar()
        .map_err(|error| {
            Error::BackendCapability(format!("failed to validate token IDs: {error}"))
        })?
        .elem::<i64>();
    let maximum = token_ids
        .max()
        .try_into_scalar()
        .map_err(|error| {
            Error::BackendCapability(format!("failed to validate token IDs: {error}"))
        })?
        .elem::<i64>();
    if minimum < 0 || maximum as usize >= vocabulary {
        return Err(Error::shape(
            "qwen.input_ids",
            format!("token values in 0..{vocabulary}"),
            format!("observed range {minimum}..={maximum}"),
        ));
    }
    Ok(())
}

/// Validates a semantic mask and rejects rows with no usable token.
fn validate_mask<B: Backend>(
    mask: Option<&TokenMask<B>>,
    batch: usize,
    sequence: usize,
) -> Result<()> {
    let Some(mask) = mask else {
        return Ok(());
    };
    if mask.as_tensor().dims() != [batch, sequence] {
        return Err(Error::shape(
            "qwen.token_mask",
            format!("[{batch}, {sequence}]"),
            format!("{:?}", mask.as_tensor().dims()),
        ));
    }
    let minimum_valid = mask
        .as_tensor()
        .clone()
        .int()
        .sum_dim(1)
        .min()
        .try_into_scalar()
        .map_err(|error| Error::BackendCapability(format!("failed to validate mask: {error}")))?
        .elem::<i64>();
    if minimum_valid == 0 {
        return Err(Error::shape(
            "qwen.token_mask",
            "at least one valid token in every batch row",
            "a fully masked row",
        ));
    }
    Ok(())
}

/// Resolves explicit positions or derives padding-independent text positions.
fn resolve_positions<B: Backend>(
    input: &CausalLmInput<B>,
    batch: usize,
    sequence: usize,
) -> Result<Tensor<B, 2, Int>> {
    if let Some(positions) = &input.position_ids {
        if positions.dims() != [batch, sequence] {
            return Err(Error::shape(
                "qwen.position_ids",
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

/// Selects requested sequence positions before the expensive vocabulary map.
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
            let device = hidden.device();
            let indices = positions
                .iter()
                .map(|position| *position as i64)
                .collect::<Vec<_>>();
            let count = indices.len();
            Ok(hidden.select(
                1,
                Tensor::<B, 1, Int>::from_data(TensorData::new(indices, [count]), &device),
            ))
        }
    }
}

/// Validates selective-logit indices before potentially stateful model work.
fn validate_logits_selection(selection: &LogitsSelection, sequence: usize) -> Result<()> {
    if let LogitsSelection::Positions(positions) = selection
        && (positions.is_empty() || positions.iter().any(|position| *position >= sequence))
    {
        return Err(Error::shape(
            "forward_options.logits.positions",
            format!("a non-empty list of indices below {sequence}"),
            format!("{positions:?}"),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use burn::{
        backend::Flex,
        tensor::{Bool, Int, Tensor, backend::Backend},
    };

    use super::*;
    use crate::models::config::ModelConfig;

    type TestBackend = Flex<f32, i32>;

    /// Builds a tiny hybrid model that exercises both mixer families.
    fn tiny_config() -> Qwen3_5TextConfig {
        Qwen3_5TextConfig {
            vocab_size: 32,
            hidden_size: 8,
            intermediate_size: 16,
            num_hidden_layers: 4,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 4,
            max_position_embeddings: 64,
            layer_types: vec![
                Qwen3_5LayerType::LinearAttention,
                Qwen3_5LayerType::LinearAttention,
                Qwen3_5LayerType::LinearAttention,
                Qwen3_5LayerType::FullAttention,
            ],
            linear_key_head_dim: 4,
            linear_value_head_dim: 4,
            linear_num_key_heads: 1,
            linear_num_value_heads: 2,
            linear_conv_kernel_dim: 3,
            partial_rotary_factor: Some(1.0),
            rope_theta: Some(10_000.0),
            mtp_num_hidden_layers: 0,
            ..Default::default()
        }
    }

    /// Tiny stateless inference returns only requested logits and diagnostics.
    #[test]
    fn tiny_hybrid_forward_respects_output_selection() {
        let device = Default::default();
        TestBackend::seed(&device, 11);
        let model = Qwen3_5ForCausalLm::<TestBackend>::init(tiny_config(), &device)
            .expect("tiny model must initialize");
        let input = CausalLmInput {
            token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[1, 2, 3], [4, 5, 6]], &device),
            token_mask: None,
            position_ids: None,
        };
        let output = model
            .forward(
                input,
                &ForwardOptions {
                    logits: LogitsSelection::Positions(vec![0, 2]),
                    hidden_states: true,
                    ..Default::default()
                },
            )
            .expect("forward must succeed");
        assert_eq!(output.logits.dims(), [2, 2, 32]);
        assert_eq!(output.hidden_states.len(), 6);
        assert_eq!(
            output.final_hidden_state.expect("requested").dims(),
            [2, 3, 8]
        );
    }

    /// Tied checkpoints project with the embedding table and allocate no second matrix.
    #[test]
    fn tied_embedding_head_runs_without_duplicate_parameter() {
        let device = Default::default();
        TestBackend::seed(&device, 17);
        let mut config = tiny_config();
        config.tie_word_embeddings = true;
        let model = Qwen3_5ForCausalLm::<TestBackend>::init(config, &device)
            .expect("tied tiny model must initialize");
        assert!(model.lm_head.is_none());

        let output = model
            .forward(
                CausalLmInput {
                    token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[1, 2]], &device),
                    token_mask: None,
                    position_ids: None,
                },
                &ForwardOptions::default(),
            )
            .expect("embedding-backed vocabulary projection must run");
        assert_eq!(output.logits.dims(), [1, 1, 32]);
    }

    /// A gradient must reach its leaf parameter and contain a finite learning signal.
    #[cfg(feature = "autodiff")]
    fn assert_finite_nonzero_gradient<const D: usize>(
        gradient: Option<Tensor<TestBackend, D>>,
        parameter: &str,
    ) {
        let values = gradient
            .unwrap_or_else(|| panic!("{parameter} is disconnected from the stateless loss"))
            .into_data()
            .to_vec::<f32>()
            .expect("Flex gradients are F32");
        assert!(
            values.iter().all(|value| value.is_finite()),
            "{parameter} has a non-finite gradient"
        );
        assert!(
            values.iter().any(|value| *value != 0.0),
            "{parameter} has an entirely zero gradient"
        );
    }

    /// The complete 3:1 hybrid graph propagates gradients through both mixer families.
    ///
    /// A tied vocabulary head matches the 0.8B checkpoint; the untied case also
    /// proves that embedding gradients pass through the backbone rather than
    /// only its shared output head. Partial RoPE exercises both rotated and
    /// unrotated attention channels. Four causal
    /// positions make the DeltaNet decay parameters affect retained history;
    /// a single-token example could not prove that recurrent learning signal.
    /// Squared logit loss is used only to probe connectivity and finite values,
    /// not to prescribe an eventual training objective.
    #[cfg(feature = "autodiff")]
    #[test]
    fn stateless_hybrid_graph_has_finite_nonzero_parameter_gradients() {
        use burn::backend::Autodiff;

        type AdBackend = Autodiff<TestBackend>;
        let device = Default::default();
        for tied in [false, true] {
            AdBackend::seed(&device, 29);
            let mut config = tiny_config();
            config.tie_word_embeddings = tied;
            config.partial_rotary_factor = Some(0.5);
            let model = Qwen3_5ForCausalLm::<AdBackend>::init(config, &device)
                .expect("tiny differentiable hybrid model initializes");
            assert_eq!(model.lm_head.is_none(), tied);
            let output = model
                .forward(
                    CausalLmInput {
                        token_ids: Tensor::<AdBackend, 2, Int>::from_ints([[1, 2, 3, 4]], &device),
                        token_mask: None,
                        position_ids: None,
                    },
                    &ForwardOptions {
                        logits: LogitsSelection::All,
                        ..ForwardOptions::default()
                    },
                )
                .expect("stateless hybrid autodiff forward succeeds");
            let gradients = output.logits.square().mean().backward();
            assert_finite_nonzero_gradient(
                model.model.embed_tokens.weight.grad(&gradients),
                "embedding and tied vocabulary head",
            );
            assert_finite_nonzero_gradient(model.model.norm.weight.grad(&gradients), "final norm");
            if let Some(head) = &model.lm_head {
                assert_finite_nonzero_gradient(
                    head.weight.grad(&gradients),
                    "untied vocabulary head",
                );
            }

            for (index, layer) in model.model.layers.iter().enumerate() {
                for (name, projection) in [
                    ("ffn.gate", &layer.mlp.gate_proj),
                    ("ffn.up", &layer.mlp.up_proj),
                    ("ffn.down", &layer.mlp.down_proj),
                ] {
                    assert_finite_nonzero_gradient(
                        projection.weight.grad(&gradients),
                        &format!("layer {index} {name}"),
                    );
                }
                for (name, norm) in [
                    ("input norm", &layer.input_layernorm),
                    ("post-attention norm", &layer.post_attention_layernorm),
                ] {
                    assert_finite_nonzero_gradient(
                        norm.weight.grad(&gradients),
                        &format!("layer {index} {name}"),
                    );
                }
                if let Some(delta) = &layer.linear_attn {
                    for (name, projection) in [
                        ("delta.qkv", &delta.in_proj_qkv),
                        ("delta.output_gate", &delta.in_proj_z),
                        ("delta.beta", &delta.in_proj_b),
                        ("delta.decay", &delta.in_proj_a),
                        ("delta.output", &delta.out_proj),
                    ] {
                        assert_finite_nonzero_gradient(
                            projection.weight.grad(&gradients),
                            &format!("layer {index} {name}"),
                        );
                    }
                    assert_finite_nonzero_gradient(
                        delta.conv1d.weight.grad(&gradients),
                        &format!("layer {index} delta.convolution"),
                    );
                    for (name, parameter) in [
                        ("delta.dt_bias", &delta.dt_bias),
                        ("delta.a_log", &delta.a_log),
                        ("delta.norm", &delta.norm.weight),
                    ] {
                        assert_finite_nonzero_gradient(
                            parameter.grad(&gradients),
                            &format!("layer {index} {name}"),
                        );
                    }
                }
                if let Some(attention) = &layer.self_attn {
                    for (name, projection) in [
                        ("attention.query_and_gate", &attention.q_proj),
                        ("attention.key", &attention.k_proj),
                        ("attention.value", &attention.v_proj),
                        ("attention.output", &attention.o_proj),
                    ] {
                        assert_finite_nonzero_gradient(
                            projection.weight.grad(&gradients),
                            &format!("layer {index} {name}"),
                        );
                    }
                    for (name, norm) in [
                        ("attention.query_norm", attention.query_norm.as_ref()),
                        ("attention.key_norm", attention.key_norm.as_ref()),
                    ] {
                        assert_finite_nonzero_gradient(
                            norm.expect("Qwen enables Q/K normalization")
                                .weight
                                .grad(&gradients),
                            &format!("layer {index} {name}"),
                        );
                    }
                }
            }
        }
    }

    /// Last-logit selection follows each row's last valid token under padding.
    #[test]
    fn last_selection_uses_semantic_mask() {
        let device = Default::default();
        let hidden = Tensor::from_floats(
            [
                [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
                [[4.0, 40.0], [5.0, 50.0], [6.0, 60.0]],
            ],
            &device,
        );
        let mask = TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
            [[true, true, false], [true, false, false]].into(),
            &device,
        ));
        let selected = select_hidden(hidden, Some(&mask), &LogitsSelection::Last)
            .expect("valid selection")
            .into_data()
            .to_vec::<f32>()
            .expect("f32 values");
        assert_eq!(selected, vec![2.0, 20.0, 4.0, 40.0]);
    }

    /// Invalid IDs, masks, positions, and explicit selections return errors.
    #[test]
    fn input_validation_precedes_backend_indexing() {
        let device = Default::default();
        assert!(
            validate_token_ids(
                Tensor::<TestBackend, 2, Int>::from_ints([[32]], &device),
                32
            )
            .is_err()
        );
        let empty = TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
            [[false, false]].into(),
            &device,
        ));
        assert!(validate_mask(Some(&empty), 1, 2).is_err());
        assert!(
            select_hidden(
                Tensor::<TestBackend, 3>::zeros([1, 2, 3], &device),
                None,
                &LogitsSelection::Positions(vec![2]),
            )
            .is_err()
        );
    }

    /// Stateless execution enforces the same declared context as cached decoding.
    #[test]
    fn stateless_positions_fail_before_embedding_or_attention_outside_context() {
        let device = Default::default();
        let model = Qwen3_5ForCausalLm::<TestBackend>::init(tiny_config(), &device)
            .expect("tiny model must initialize");
        let token_ids = Tensor::<TestBackend, 2, Int>::from_ints([[1, 2]], &device);

        for positions in [[0, -1], [0, 64]] {
            let error = model
                .forward(
                    CausalLmInput {
                        token_ids: token_ids.clone(),
                        token_mask: None,
                        position_ids: Some(Tensor::from_ints([positions], &device)),
                    },
                    &ForwardOptions::default(),
                )
                .expect_err("invalid stateless positions must be recoverable");
            assert!(matches!(error, Error::Shape { .. }));
            assert!(error.to_string().contains("qwen.position_ids"));
        }
    }

    /// Stateless device preflight identifies every independently supplied tensor.
    #[test]
    fn stateless_device_affinity_errors_are_contextual() {
        assert!(ensure_model_device("qwen.input_ids", &0_u8, &0_u8).is_ok());
        for tensor in [
            "qwen.input_ids",
            "qwen.token_mask",
            "qwen.position_ids",
            "qwen.mtp.target_hidden_state",
            "qwen.lm_head",
        ] {
            let error = ensure_model_device(tensor, &1_u8, &0_u8)
                .expect_err("different device identities must fail preflight");
            assert!(error.to_string().contains(tensor));
            assert!(error.to_string().contains("model parameters"));
        }
    }

    /// The official reduced Qwen3.8 fixture constructs all lazy parameter shapes.
    #[test]
    fn official_qwen38_topology_initializes_lazily() {
        let device = Default::default();
        let config = Qwen3_5TextConfig::from_json_str(include_str!(
            "../../../tests/fixtures/qwen3_8_27b.json"
        ))
        .expect("pinned fixture must parse");
        let model = Qwen3_5ForCausalLm::<TestBackend>::init(config, &device)
            .expect("official topology must initialize lazily");
        assert_eq!(model.model.layers.len(), 64);
        assert_eq!(
            model.mtp.as_ref().expect("official MTP graph").layers.len(),
            1
        );
        assert_eq!(model.config().vocab_size, 248_320);
        assert_eq!(model.model.rope.rotary_dimension(), 64);
    }
}
