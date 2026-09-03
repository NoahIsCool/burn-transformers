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
    position::{
        DynamicNtkScaling, LinearRopeScaling, RopeConfig, RopeScaling, YarnConfig, apply_rotary,
        rotary_angles,
    },
};

use super::{Qwen3_5LayerType, Qwen3_5TextConfig};
use crate::models::config::RopeType;

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
                let mut projection = attention.project_self(normalized);
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
                .forward(self.post_attention_layernorm.forward(hidden)))
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
        validate_token_ids(input.token_ids.clone(), self.config.vocab_size)?;
        validate_mask(input.token_mask.as_ref(), batch, sequence)?;
        let positions = resolve_positions(input, batch, sequence)?;
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
                sequence,
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

/// Decoder-only Qwen language model with an untied output head.
#[derive(Module, Debug)]
pub struct Qwen3_5ForCausalLm<B: Backend> {
    /// Hybrid text backbone.
    pub model: Qwen3_5TextModel<B>,
    /// Vocabulary projection. Released Qwen3.8 checkpoints do not tie it.
    pub lm_head: Linear<B>,
}

impl<B: Backend> Qwen3_5ForCausalLm<B> {
    /// Initializes a validated inference/training graph with lazy parameters.
    pub fn init(config: Qwen3_5TextConfig, device: &B::Device) -> Result<Self> {
        if config.tie_word_embeddings {
            return Err(Error::configuration(
                "qwen3_5.tie_word_embeddings",
                "tied Qwen heads are not represented by this checkpoint family",
            ));
        }
        let hidden_size = config.hidden_size;
        let vocabulary_size = config.vocab_size;
        Ok(Self {
            model: Qwen3_5TextModel::init(config, device)?,
            lm_head: LinearConfig::new(hidden_size, vocabulary_size)
                .with_bias(false)
                .with_layout(LinearLayout::Col)
                .init(device),
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
        let logits = self.lm_head.forward(selected);
        Ok(CausalLmOutput {
            logits,
            final_hidden_state: options.hidden_states.then_some(hidden),
            hidden_states,
            attentions: Vec::new(),
            routers: Vec::new(),
        })
    }

    /// Returns the exact validated checkpoint configuration.
    pub fn config(&self) -> &Qwen3_5TextConfig {
        &self.model.config
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
            if let (Some(fast), Some(slow)) = (parameters.beta_fast, parameters.beta_slow) {
                yarn = yarn
                    .with_betas(fast, slow)
                    .map_err(|error| Error::configuration("qwen3_5.rope", error.to_string()))?;
            }
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
            if positions.is_empty() || positions.iter().any(|position| *position >= sequence) {
                return Err(Error::shape(
                    "forward_options.logits.positions",
                    format!("a non-empty list of indices below {sequence}"),
                    format!("{positions:?}"),
                ));
            }
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
        assert_eq!(model.config().vocab_size, 248_320);
        assert_eq!(model.model.rope.rotary_dimension(), 64);
    }
}
