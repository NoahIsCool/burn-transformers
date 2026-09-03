//! Stateless Burn implementation of the T5/FLAN/UL2 transformer graph.
//!
//! T5 is a pre-normalized encoder-decoder. Encoder self-attention is
//! bidirectional, decoder self-attention is causal, and every decoder block has
//! an additional cross-attention sublayer over the encoder's final states. The
//! first self-attention block in each stack owns a learned relative-position
//! table; its computed bias is reused by every later block in that stack.
//!
//! Unlike most modern decoder-only transformers, T5 does **not** divide dot
//! products by `sqrt(d_kv)`: attention uses `softmax(QK^T + bias)`. The shared
//! attention primitive is therefore configured with an explicit scale of one.
//! This module intentionally omits stochastic dropout during inference. All
//! parameterized operations remain ordinary Burn tensors and are suitable for
//! a future training-mode wrapper.

use burn::{
    module::{Initializer, Module},
    nn::{Embedding, EmbeddingConfig, Linear, LinearConfig, LinearLayout},
    tensor::{ElementConversion, Int, Tensor, TensorData, backend::Backend, module::linear},
};

use crate::{
    CausalLmInput, CausalLmOutput, DecodeInput, Error, ForwardOptions, LogitsSelection, Result,
    Seq2SeqLanguageModel, cache::CacheConfig, cache::TensorDecodeState,
};
use crate::{
    architecture::{EncoderOutput, Seq2SeqInput, Seq2SeqOutput, TokenMask},
    attention::{GroupedQueryAttention, GroupedQueryAttentionConfig, padding_attention_mask},
    feed_forward::{
        DenseFeedForward, DenseFeedForwardConfig, GatedFeedForward, GatedFeedForwardConfig,
    },
    normalization::{RmsNorm, RmsNormConfig},
    position::{T5RelativePositionBias, T5RelativePositionBiasConfig},
};

use super::{T5Config, T5FeedForwardProjection};

/// T5 multi-head attention projections plus an optional learned bias table.
#[derive(Module, Debug)]
pub struct T5Attention<B: Backend> {
    /// Bias-free Q/K/V/O projections implemented by the shared MHA primitive.
    pub attention: GroupedQueryAttention<B>,
    /// Learned table present only on the first self-attention block of a stack.
    pub relative_attention_bias: Option<T5RelativePositionBias<B>>,
    /// Residual-stream width checked before lower-level linear operations.
    #[module(skip)]
    pub hidden_size: usize,
}

impl<B: Backend> T5Attention<B> {
    /// Initializes T5's unscaled, bias-free full multi-head attention.
    fn init(
        config: &T5Config,
        relative_bias_is_bidirectional: Option<bool>,
        device: &B::Device,
    ) -> Result<Self> {
        let mut attention = GroupedQueryAttentionConfig::new(
            config.d_model,
            config.num_heads,
            config.num_heads,
            config.d_kv,
        );
        // T5 intentionally uses QK^T rather than QK^T / sqrt(d_kv).
        attention.scale = Some(1.0);
        let relative_attention_bias = relative_bias_is_bidirectional
            .map(|bidirectional| {
                T5RelativePositionBiasConfig::new(
                    config.num_heads,
                    config.relative_attention_num_buckets,
                    config.relative_attention_max_distance,
                    bidirectional,
                )
                .with_initializer_std(config.initializer_factor / (config.d_model as f64).sqrt())
                .init(device)
            })
            .transpose()?;
        let mut attention = attention.init(device)?;
        let factor = config.initializer_factor;
        let inner_width = config.attention_inner_dim();
        // Mesh TensorFlow chooses these variances specifically because T5 does
        // not scale QK scores at runtime. Preserve them for newly trained
        // experiments as well as checkpoint-loaded inference.
        attention.q_proj = t5_linear(
            config.d_model,
            inner_width,
            factor / ((config.d_model * config.d_kv) as f64).sqrt(),
            device,
        );
        attention.k_proj = t5_linear(
            config.d_model,
            inner_width,
            factor / (config.d_model as f64).sqrt(),
            device,
        );
        attention.v_proj = t5_linear(
            config.d_model,
            inner_width,
            factor / (config.d_model as f64).sqrt(),
            device,
        );
        attention.o_proj = t5_linear(
            inner_width,
            config.d_model,
            factor / (inner_width as f64).sqrt(),
            device,
        );
        Ok(Self {
            attention,
            relative_attention_bias,
            hidden_size: config.d_model,
        })
    }

    /// Computes this block's learned relative bias for one attention rectangle.
    fn position_bias(
        &self,
        query_length: usize,
        key_length: usize,
        query_offset: usize,
        device: &B::Device,
    ) -> Result<Tensor<B, 4>> {
        self.relative_attention_bias
            .as_ref()
            .ok_or_else(|| {
                Error::configuration(
                    "t5.relative_attention_bias",
                    "only the first block in a stack owns the shared bias table",
                )
            })?
            .forward(query_length, key_length, query_offset, device)
    }

    /// Evaluates self- or cross-attention with validated shapes and token masking.
    ///
    /// `key_mask` uses the crate-wide `true == valid` convention. Query padding is
    /// deliberately not masked: the official T5 graph masks keys and still
    /// computes padded query rows, which are ignored by the loss or logit selector.
    fn forward(
        &self,
        query_hidden: Tensor<B, 3>,
        key_value_hidden: Tensor<B, 3>,
        key_mask: Option<&TokenMask<B>>,
        position_bias: Option<Tensor<B, 4>>,
        causal: bool,
    ) -> Result<Tensor<B, 3>> {
        let [batch, query_sequence, query_width] = query_hidden.dims();
        let [key_batch, key_sequence, key_width] = key_value_hidden.dims();
        if batch == 0 || query_sequence == 0 || key_sequence == 0 {
            return Err(Error::shape(
                "t5.attention",
                "non-empty query and key/value tensors",
                format!(
                    "query={:?}, key_value={:?}",
                    query_hidden.dims(),
                    key_value_hidden.dims()
                ),
            ));
        }
        if batch != key_batch || query_width != self.hidden_size || key_width != self.hidden_size {
            return Err(Error::shape(
                "t5.attention",
                format!(
                    "query [batch, query, {}] and key/value [same_batch, key, {}]",
                    self.hidden_size, self.hidden_size
                ),
                format!(
                    "query={:?}, key_value={:?}",
                    query_hidden.dims(),
                    key_value_hidden.dims()
                ),
            ));
        }
        validate_mask_shape(key_mask, batch, key_sequence, "t5.attention.key_mask")?;
        let projection = self.attention.project(query_hidden, key_value_hidden)?;
        let mask = padding_attention_mask(
            None,
            key_mask,
            batch,
            self.attention.query_heads,
            query_sequence,
            key_sequence,
        )?;
        self.attention
            .attend(projection, mask, position_bias, causal)
    }
}

/// Pre-normalized T5 self-attention residual sublayer.
#[derive(Module, Debug)]
pub struct T5LayerSelfAttention<B: Backend> {
    /// Unscaled self-attention projections and optional relative-bias table.
    pub self_attention: T5Attention<B>,
    /// T5 scale-only RMS normalization applied before projection.
    pub layer_norm: RmsNorm<B>,
    /// Whether this sublayer belongs to the causal decoder stack.
    #[module(skip)]
    pub causal: bool,
}

impl<B: Backend> T5LayerSelfAttention<B> {
    /// Initializes one encoder or decoder self-attention residual branch.
    fn init(
        config: &T5Config,
        causal: bool,
        owns_relative_bias: bool,
        device: &B::Device,
    ) -> Result<Self> {
        Ok(Self {
            self_attention: T5Attention::init(
                config,
                owns_relative_bias.then_some(!causal),
                device,
            )?,
            layer_norm: t5_norm(config, device)?,
            causal,
        })
    }

    /// Applies pre-normalization, self-attention, and the residual addition.
    fn forward(
        &self,
        hidden: Tensor<B, 3>,
        token_mask: Option<&TokenMask<B>>,
        position_bias: Tensor<B, 4>,
    ) -> Result<Tensor<B, 3>> {
        let residual = hidden.clone();
        let normalized = self.layer_norm.forward(hidden);
        let attended = self.self_attention.forward(
            normalized.clone(),
            normalized,
            token_mask,
            Some(position_bias),
            self.causal,
        )?;
        Ok(residual + attended)
    }
}

/// Pre-normalized decoder-to-encoder cross-attention residual sublayer.
#[derive(Module, Debug)]
pub struct T5LayerCrossAttention<B: Backend> {
    /// Query-from-decoder and key/value-from-encoder projections.
    pub enc_dec_attention: T5Attention<B>,
    /// T5 scale-only RMS normalization applied to decoder queries.
    pub layer_norm: RmsNorm<B>,
}

impl<B: Backend> T5LayerCrossAttention<B> {
    /// Initializes a cross-attention branch without relative-position parameters.
    fn init(config: &T5Config, device: &B::Device) -> Result<Self> {
        Ok(Self {
            enc_dec_attention: T5Attention::init(config, None, device)?,
            layer_norm: t5_norm(config, device)?,
        })
    }

    /// Applies decoder query normalization, cross-attention, and the residual.
    fn forward(
        &self,
        hidden: Tensor<B, 3>,
        encoder_hidden: Tensor<B, 3>,
        encoder_mask: Option<&TokenMask<B>>,
    ) -> Result<Tensor<B, 3>> {
        let residual = hidden.clone();
        let query = self.layer_norm.forward(hidden);
        Ok(residual
            + self
                .enc_dec_attention
                .forward(query, encoder_hidden, encoder_mask, None, false)?)
    }
}

/// Pre-normalized dense or gated T5 feed-forward residual sublayer.
#[derive(Module, Debug)]
pub struct T5LayerFeedForward<B: Backend> {
    /// Single-branch ReLU/GELU implementation when selected by the checkpoint.
    pub dense: Option<DenseFeedForward<B>>,
    /// Two-branch gated GELU/SiLU implementation when selected by the checkpoint.
    pub gated: Option<GatedFeedForward<B>>,
    /// T5 scale-only RMS normalization applied before the MLP.
    pub layer_norm: RmsNorm<B>,
    /// Serialized projection convention used to validate the active module.
    #[module(skip)]
    pub projection: T5FeedForwardProjection,
}

impl<B: Backend> T5LayerFeedForward<B> {
    /// Initializes exactly one feed-forward topology from checkpoint metadata.
    fn init(config: &T5Config, device: &B::Device) -> Result<Self> {
        use crate::architecture::Activation;

        let (mut dense, mut gated) = match config.feed_forward_proj {
            T5FeedForwardProjection::Relu => (
                Some(
                    DenseFeedForwardConfig::new(config.d_model, config.d_ff)
                        .with_activation(Activation::Relu)
                        .init(device)?,
                ),
                None,
            ),
            T5FeedForwardProjection::Gelu => (
                Some(
                    DenseFeedForwardConfig::new(config.d_model, config.d_ff)
                        .with_activation(Activation::Gelu)
                        .init(device)?,
                ),
                None,
            ),
            T5FeedForwardProjection::GatedGelu => (
                None,
                Some(
                    GatedFeedForwardConfig::new(config.d_model, config.d_ff)
                        .with_activation(Activation::GeluTanh)
                        .init(device)?,
                ),
            ),
            T5FeedForwardProjection::GatedGeluNew => (
                None,
                Some(
                    GatedFeedForwardConfig::new(config.d_model, config.d_ff)
                        .with_activation(Activation::GeluTanh)
                        .init(device)?,
                ),
            ),
            T5FeedForwardProjection::GatedSilu => (
                None,
                Some(
                    GatedFeedForwardConfig::new(config.d_model, config.d_ff)
                        .with_activation(Activation::Silu)
                        .init(device)?,
                ),
            ),
        };
        let input_std = config.initializer_factor / (config.d_model as f64).sqrt();
        let output_std = config.initializer_factor / (config.d_ff as f64).sqrt();
        if let Some(dense) = &mut dense {
            dense.up_proj = t5_linear(config.d_model, config.d_ff, input_std, device);
            dense.down_proj = t5_linear(config.d_ff, config.d_model, output_std, device);
        }
        if let Some(gated) = &mut gated {
            gated.gate_proj = t5_linear(config.d_model, config.d_ff, input_std, device);
            gated.up_proj = t5_linear(config.d_model, config.d_ff, input_std, device);
            gated.down_proj = t5_linear(config.d_ff, config.d_model, output_std, device);
        }
        Ok(Self {
            dense,
            gated,
            layer_norm: t5_norm(config, device)?,
            projection: config.feed_forward_proj,
        })
    }

    /// Applies the selected MLP to normalized states and adds the residual.
    fn forward(&self, hidden: Tensor<B, 3>) -> Result<Tensor<B, 3>> {
        let residual = hidden.clone();
        let normalized = self.layer_norm.forward(hidden);
        let projected = match (&self.dense, &self.gated, self.projection.is_gated()) {
            (Some(dense), None, false) => dense.forward(normalized),
            (None, Some(gated), true) => gated.forward(normalized),
            _ => {
                return Err(Error::configuration(
                    "t5.feed_forward",
                    "active dense/gated module disagrees with checkpoint projection",
                ));
            }
        };
        Ok(residual + projected)
    }
}

/// One exact T5 encoder or decoder block.
#[derive(Module, Debug)]
pub struct T5Block<B: Backend> {
    /// Encoder-bidirectional or decoder-causal self-attention branch.
    pub self_attention: T5LayerSelfAttention<B>,
    /// Decoder cross-attention branch; absent from encoder blocks.
    pub cross_attention: Option<T5LayerCrossAttention<B>>,
    /// Dense or gated feed-forward branch.
    pub feed_forward: T5LayerFeedForward<B>,
}

impl<B: Backend> T5Block<B> {
    /// Initializes one block, placing the learned position table only when requested.
    fn init(
        config: &T5Config,
        decoder: bool,
        owns_relative_bias: bool,
        device: &B::Device,
    ) -> Result<Self> {
        Ok(Self {
            self_attention: T5LayerSelfAttention::init(
                config,
                decoder,
                owns_relative_bias,
                device,
            )?,
            cross_attention: decoder
                .then(|| T5LayerCrossAttention::init(config, device))
                .transpose()?,
            feed_forward: T5LayerFeedForward::init(config, device)?,
        })
    }

    /// Evaluates one encoder block using bidirectional self-attention.
    fn forward_encoder(
        &self,
        hidden: Tensor<B, 3>,
        token_mask: Option<&TokenMask<B>>,
        position_bias: Tensor<B, 4>,
    ) -> Result<Tensor<B, 3>> {
        if self.cross_attention.is_some() || self.self_attention.causal {
            return Err(Error::configuration(
                "t5.encoder.block",
                "encoder block was initialized with decoder sublayers",
            ));
        }
        self.feed_forward.forward(
            self.self_attention
                .forward(hidden, token_mask, position_bias)?,
        )
    }

    /// Evaluates causal self-attention, cross-attention, then the decoder FFN.
    fn forward_decoder(
        &self,
        hidden: Tensor<B, 3>,
        decoder_mask: Option<&TokenMask<B>>,
        position_bias: Tensor<B, 4>,
        encoder_hidden: Tensor<B, 3>,
        encoder_mask: Option<&TokenMask<B>>,
    ) -> Result<Tensor<B, 3>> {
        if !self.self_attention.causal {
            return Err(Error::configuration(
                "t5.decoder.block",
                "decoder block was initialized with bidirectional self-attention",
            ));
        }
        let hidden = self
            .self_attention
            .forward(hidden, decoder_mask, position_bias)?;
        let cross_attention = self.cross_attention.as_ref().ok_or_else(|| {
            Error::configuration(
                "t5.decoder.cross_attention",
                "decoder block has no encoder-decoder attention module",
            )
        })?;
        let hidden = cross_attention.forward(hidden, encoder_hidden, encoder_mask)?;
        self.feed_forward.forward(hidden)
    }
}

/// Parameterized T5 encoder or decoder stack without token embeddings.
#[derive(Module, Debug)]
pub struct T5Stack<B: Backend> {
    /// Ordered blocks; only block zero owns relative-position parameters.
    pub block: Vec<T5Block<B>>,
    /// Scale-only RMS normalization after the final residual block.
    pub final_layer_norm: RmsNorm<B>,
    /// Whether this is the causal decoder stack.
    #[module(skip)]
    pub is_decoder: bool,
}

/// Hidden state plus optional per-layer diagnostic snapshots.
struct T5StackOutput<B: Backend> {
    /// Final normalized stack output.
    hidden: Tensor<B, 3>,
    /// Embedding, residual-block, and final-normalization states when requested.
    hidden_states: Vec<Tensor<B, 3>>,
}

impl<B: Backend> T5Stack<B> {
    /// Initializes every block and a single stack-level shared bias owner.
    fn init(config: &T5Config, is_decoder: bool, device: &B::Device) -> Result<Self> {
        let depth = if is_decoder {
            config.decoder_layers()
        } else {
            config.num_layers
        };
        let block = (0..depth)
            .map(|index| T5Block::init(config, is_decoder, index == 0, device))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            block,
            final_layer_norm: t5_norm(config, device)?,
            is_decoder,
        })
    }

    /// Retrieves the first block's position table and computes one shared bias.
    fn position_bias(
        &self,
        query_length: usize,
        key_length: usize,
        device: &B::Device,
    ) -> Result<Tensor<B, 4>> {
        self.block
            .first()
            .ok_or_else(|| Error::configuration("t5.stack.block", "stack must be non-empty"))?
            .self_attention
            .self_attention
            .position_bias(query_length, key_length, 0, device)
    }

    /// Evaluates the bidirectional encoder stack.
    fn forward_encoder(
        &self,
        mut hidden: Tensor<B, 3>,
        token_mask: Option<&TokenMask<B>>,
        retain_hidden_states: bool,
    ) -> Result<T5StackOutput<B>> {
        if self.is_decoder {
            return Err(Error::configuration(
                "t5.encoder",
                "cannot run an encoder pass through a decoder stack",
            ));
        }
        let [_, sequence, _] = hidden.dims();
        let position_bias = self.position_bias(sequence, sequence, &hidden.device())?;
        let mut hidden_states = diagnostic_buffer(self.block.len(), retain_hidden_states);
        if retain_hidden_states {
            hidden_states.push(hidden.clone());
        }
        for block in &self.block {
            hidden = block.forward_encoder(hidden, token_mask, position_bias.clone())?;
            if retain_hidden_states {
                hidden_states.push(hidden.clone());
            }
        }
        hidden = self.final_layer_norm.forward(hidden);
        if retain_hidden_states {
            hidden_states.push(hidden.clone());
        }
        Ok(T5StackOutput {
            hidden,
            hidden_states,
        })
    }

    /// Evaluates the causal decoder stack with fresh cross-attention projections.
    fn forward_decoder(
        &self,
        mut hidden: Tensor<B, 3>,
        decoder_mask: Option<&TokenMask<B>>,
        encoder_hidden: Tensor<B, 3>,
        encoder_mask: Option<&TokenMask<B>>,
        retain_hidden_states: bool,
    ) -> Result<T5StackOutput<B>> {
        if !self.is_decoder {
            return Err(Error::configuration(
                "t5.decoder",
                "cannot run a decoder pass through an encoder stack",
            ));
        }
        let [_, sequence, _] = hidden.dims();
        let position_bias = self.position_bias(sequence, sequence, &hidden.device())?;
        let mut hidden_states = diagnostic_buffer(self.block.len(), retain_hidden_states);
        if retain_hidden_states {
            hidden_states.push(hidden.clone());
        }
        for block in &self.block {
            hidden = block.forward_decoder(
                hidden,
                decoder_mask,
                position_bias.clone(),
                encoder_hidden.clone(),
                encoder_mask,
            )?;
            if retain_hidden_states {
                hidden_states.push(hidden.clone());
            }
        }
        hidden = self.final_layer_norm.forward(hidden);
        if retain_hidden_states {
            hidden_states.push(hidden.clone());
        }
        Ok(T5StackOutput {
            hidden,
            hidden_states,
        })
    }
}

/// Complete T5-family encoder-decoder model with conditional-generation head.
#[derive(Module, Debug)]
pub struct T5ForConditionalGeneration<B: Backend> {
    /// Single token embedding table used by both stacks and tied vocabulary heads.
    pub shared: Embedding<B>,
    /// Bidirectional source encoder.
    pub encoder: T5Stack<B>,
    /// Causal target decoder with cross-attention in every block.
    pub decoder: T5Stack<B>,
    /// Separate vocabulary projection used by T5 v1.1/FLAN/UL2 checkpoints.
    ///
    /// Original T5 ties this projection to [`Self::shared`] and therefore does
    /// not allocate a second matrix. Checkpoints with
    /// `tie_word_embeddings = false` contain this parameter explicitly.
    pub lm_head: Option<Linear<B>>,
    /// Validated checkpoint topology, excluded from parameter records.
    #[module(skip)]
    pub config: T5Config,
}

impl<B: Backend> T5ForConditionalGeneration<B> {
    /// Initializes a validated stateless T5/FLAN/UL2 graph.
    pub fn init(config: T5Config, device: &B::Device) -> Result<Self> {
        crate::ArchitectureConfig::validate(&config)?;
        let shared = EmbeddingConfig::new(config.vocab_size, config.d_model)
            .with_initializer(Initializer::Normal {
                mean: 0.0,
                std: config.initializer_factor,
            })
            .init(device);
        Ok(Self {
            encoder: T5Stack::init(&config, false, device)?,
            decoder: T5Stack::init(&config, true, device)?,
            lm_head: (!config.tie_word_embeddings).then(|| {
                t5_linear(
                    config.d_model,
                    config.vocab_size,
                    config.initializer_factor,
                    device,
                )
            }),
            shared,
            config,
        })
    }

    /// Returns the exact validated checkpoint configuration.
    #[must_use]
    pub const fn config(&self) -> &T5Config {
        &self.config
    }

    /// Encodes source tokens with bidirectional attention.
    pub fn encode(
        &self,
        source: CausalLmInput<B>,
        options: &ForwardOptions,
    ) -> Result<EncoderOutput<B>> {
        validate_options(options)?;
        let [batch, sequence] = validate_token_input(&source, self.config.vocab_size, "encoder")?;
        let hidden = self.shared.forward(source.token_ids.clone());
        debug_assert_eq!(hidden.dims(), [batch, sequence, self.config.d_model]);
        let output = self.encoder.forward_encoder(
            hidden,
            source.token_mask.as_ref(),
            options.hidden_states,
        )?;
        Ok(EncoderOutput {
            hidden_state: output.hidden,
            token_mask: source.token_mask,
            hidden_states: output.hidden_states,
            attentions: Vec::new(),
        })
    }

    /// Evaluates the complete stateless encoder-decoder graph.
    pub fn forward(
        &self,
        input: Seq2SeqInput<B>,
        options: &ForwardOptions,
    ) -> Result<Seq2SeqOutput<B>> {
        let encoder = self.encode(input.encoder, options)?;
        let decoder = self.decode_stateless(&encoder, input.decoder, options)?;
        Ok(Seq2SeqOutput {
            decoder,
            encoder,
            cross_attentions: Vec::new(),
        })
    }

    /// Runs the decoder without persistent KV or cross-attention state.
    fn decode_stateless(
        &self,
        encoder: &EncoderOutput<B>,
        decoder: CausalLmInput<B>,
        options: &ForwardOptions,
    ) -> Result<CausalLmOutput<B>> {
        validate_options(options)?;
        let [batch, sequence] = validate_token_input(&decoder, self.config.vocab_size, "decoder")?;
        validate_encoder_output(encoder, batch, self.config.d_model)?;
        let hidden = self.shared.forward(decoder.token_ids.clone());
        debug_assert_eq!(hidden.dims(), [batch, sequence, self.config.d_model]);
        let output = self.decoder.forward_decoder(
            hidden,
            decoder.token_mask.as_ref(),
            encoder.hidden_state.clone(),
            encoder.token_mask.as_ref(),
            options.hidden_states,
        )?;
        let selected = select_hidden(
            output.hidden.clone(),
            decoder.token_mask.as_ref(),
            &options.logits,
        )?;
        let logits = self.project_logits(selected)?;
        Ok(CausalLmOutput {
            logits,
            final_hidden_state: options.hidden_states.then_some(output.hidden),
            hidden_states: output.hidden_states,
            attentions: Vec::new(),
            routers: Vec::new(),
        })
    }

    /// Applies the tied/scaled or independent/unscaled vocabulary projection.
    fn project_logits(&self, hidden: Tensor<B, 3>) -> Result<Tensor<B, 3>> {
        if self.config.tie_word_embeddings {
            if self.lm_head.is_some() {
                return Err(Error::configuration(
                    "t5.lm_head",
                    "tied checkpoints must not own a separate vocabulary projection",
                ));
            }
            let hidden = hidden.mul_scalar(1.0 / (self.config.d_model as f64).sqrt());
            return Ok(linear(hidden, self.shared.weight.val().transpose(), None));
        }
        self.lm_head
            .as_ref()
            .ok_or_else(|| {
                Error::configuration(
                    "t5.lm_head",
                    "untied checkpoints require a separate vocabulary projection",
                )
            })
            .map(|head| head.forward(hidden))
    }
}

impl<B: Backend> Seq2SeqLanguageModel<B> for T5ForConditionalGeneration<B> {
    type Config = T5Config;
    type State = TensorDecodeState<B>;

    /// Returns the validated T5-family configuration used to build this graph.
    fn config(&self) -> &Self::Config {
        T5ForConditionalGeneration::config(self)
    }

    /// Runs the stateless bidirectional encoder implementation.
    fn encode(
        &self,
        source: CausalLmInput<B>,
        options: &ForwardOptions,
    ) -> Result<EncoderOutput<B>> {
        T5ForConditionalGeneration::encode(self, source, options)
    }

    /// Runs stateless encoder, causal decoder, cross-attention, and LM projection.
    fn forward(
        &self,
        input: Seq2SeqInput<B>,
        options: &ForwardOptions,
    ) -> Result<Seq2SeqOutput<B>> {
        T5ForConditionalGeneration::forward(self, input, options)
    }

    /// Reports the deliberately isolated cached-decoder integration boundary.
    fn new_decode_state(
        &self,
        _config: CacheConfig,
        _batch_size: usize,
        _device: &B::Device,
    ) -> Result<Self::State> {
        Err(cached_decoder_unavailable())
    }

    /// Reports that stateful decoder prefill is deferred from this vertical slice.
    fn prefill_decoder(
        &self,
        _encoder: &EncoderOutput<B>,
        _decoder: CausalLmInput<B>,
        _state: &mut Self::State,
        _options: &ForwardOptions,
    ) -> Result<Seq2SeqOutput<B>> {
        Err(cached_decoder_unavailable())
    }

    /// Reports that recurrent decoder steps await the cross-attention cache API.
    fn decode(
        &self,
        _encoder: &EncoderOutput<B>,
        _input: DecodeInput<B>,
        _state: &mut Self::State,
        _options: &ForwardOptions,
    ) -> Result<Seq2SeqOutput<B>> {
        Err(cached_decoder_unavailable())
    }
}

/// Initializes one conventional T5 scale-only RMS normalization.
fn t5_norm<B: Backend>(config: &T5Config, device: &B::Device) -> Result<RmsNorm<B>> {
    let mut norm = RmsNormConfig::new(config.d_model)
        .with_epsilon(config.layer_norm_epsilon)
        .init(device)?;
    norm.weight = Initializer::Constant {
        value: config.initializer_factor,
    }
    .init([config.d_model], device);
    Ok(norm)
}

/// Initializes a bias-free checkpoint-layout linear map with an exact T5 variance.
fn t5_linear<B: Backend>(input: usize, output: usize, std: f64, device: &B::Device) -> Linear<B> {
    LinearConfig::new(input, output)
        .with_bias(false)
        .with_layout(LinearLayout::Col)
        .with_initializer(Initializer::Normal { mean: 0.0, std })
        .init(device)
}

/// Allocates diagnostics only when requested, avoiding inference-path growth.
fn diagnostic_buffer<B: Backend>(depth: usize, retain: bool) -> Vec<Tensor<B, 3>> {
    Vec::with_capacity(if retain { depth.saturating_add(2) } else { 0 })
}

/// Rejects diagnostics that the fused Burn attention boundary cannot expose.
fn validate_options(options: &ForwardOptions) -> Result<()> {
    if options.attentions {
        return Err(Error::BackendCapability(
            "T5 attention probability capture requires a reference attention path".into(),
        ));
    }
    Ok(())
}

/// Validates token IDs, mask semantics, and unsupported absolute positions.
fn validate_token_input<B: Backend>(
    input: &CausalLmInput<B>,
    vocabulary: usize,
    stack: &'static str,
) -> Result<[usize; 2]> {
    let [batch, sequence] = input.token_ids.dims();
    if batch == 0 || sequence == 0 {
        return Err(Error::shape(
            format!("t5.{stack}.input_ids"),
            "non-empty [batch, sequence]",
            format!("[{batch}, {sequence}]"),
        ));
    }
    if let Some(positions) = &input.position_ids {
        if positions.dims() != [batch, sequence] {
            return Err(Error::shape(
                format!("t5.{stack}.position_ids"),
                format!("[{batch}, {sequence}]"),
                format!("{:?}", positions.dims()),
            ));
        }
        return Err(Error::BackendCapability(format!(
            "T5 {stack} uses relative positions and does not accept explicit position IDs"
        )));
    }
    validate_token_ids(input.token_ids.clone(), vocabulary, stack)?;
    validate_nonempty_mask(input.token_mask.as_ref(), batch, sequence, stack)?;
    Ok([batch, sequence])
}

/// Synchronizes the integer range before an embedding backend can index it.
fn validate_token_ids<B: Backend>(
    token_ids: Tensor<B, 2, Int>,
    vocabulary: usize,
    stack: &'static str,
) -> Result<()> {
    let minimum = token_ids
        .clone()
        .min()
        .try_into_scalar()
        .map_err(|error| {
            Error::BackendCapability(format!("failed to validate T5 token IDs: {error}"))
        })?
        .elem::<i64>();
    let maximum = token_ids
        .max()
        .try_into_scalar()
        .map_err(|error| {
            Error::BackendCapability(format!("failed to validate T5 token IDs: {error}"))
        })?
        .elem::<i64>();
    if minimum < 0 || maximum as usize >= vocabulary {
        return Err(Error::shape(
            format!("t5.{stack}.input_ids"),
            format!("token values in 0..{vocabulary}"),
            format!("observed range {minimum}..={maximum}"),
        ));
    }
    Ok(())
}

/// Validates a semantic mask and rejects rows without any usable key.
fn validate_nonempty_mask<B: Backend>(
    mask: Option<&TokenMask<B>>,
    batch: usize,
    sequence: usize,
    stack: &'static str,
) -> Result<()> {
    validate_mask_shape(mask, batch, sequence, &format!("t5.{stack}.token_mask"))?;
    let Some(mask) = mask else {
        return Ok(());
    };
    let minimum_valid = mask
        .as_tensor()
        .clone()
        .int()
        .sum_dim(1)
        .min()
        .try_into_scalar()
        .map_err(|error| Error::BackendCapability(format!("failed to validate T5 mask: {error}")))?
        .elem::<i64>();
    if minimum_valid == 0 {
        return Err(Error::shape(
            format!("t5.{stack}.token_mask"),
            "at least one valid token in every batch row",
            "a fully masked row",
        ));
    }
    Ok(())
}

/// Checks a semantic mask extent without synchronizing its values.
fn validate_mask_shape<B: Backend>(
    mask: Option<&TokenMask<B>>,
    batch: usize,
    sequence: usize,
    name: &str,
) -> Result<()> {
    if let Some(mask) = mask
        && mask.as_tensor().dims() != [batch, sequence]
    {
        return Err(Error::shape(
            name,
            format!("[{batch}, {sequence}]"),
            format!("{:?}", mask.as_tensor().dims()),
        ));
    }
    Ok(())
}

/// Validates externally supplied encoder states before cross-attention projection.
fn validate_encoder_output<B: Backend>(
    encoder: &EncoderOutput<B>,
    decoder_batch: usize,
    hidden_size: usize,
) -> Result<()> {
    let [batch, sequence, width] = encoder.hidden_state.dims();
    if batch != decoder_batch || sequence == 0 || width != hidden_size {
        return Err(Error::shape(
            "t5.encoder_output.hidden_state",
            format!("[{decoder_batch}, source>0, {hidden_size}]"),
            format!("[{batch}, {sequence}, {width}]"),
        ));
    }
    validate_nonempty_mask(
        encoder.token_mask.as_ref(),
        batch,
        sequence,
        "encoder_output",
    )
}

/// Selects decoder positions before the vocabulary projection.
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
                let sequence_end = i64::try_from(sequence).map_err(|_| {
                    Error::configuration(
                        "t5.decoder.sequence",
                        "sequence length does not fit in i64",
                    )
                })?;
                let device = hidden.device();
                let positions = Tensor::<B, 1, Int>::arange(0..sequence_end, &device)
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
            let indices = positions
                .iter()
                .map(|position| {
                    i64::try_from(*position).map_err(|_| {
                        Error::configuration(
                            "forward_options.logits.positions",
                            "position does not fit in i64",
                        )
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            let count = indices.len();
            let device = hidden.device();
            Ok(hidden.select(
                1,
                Tensor::<B, 1, Int>::from_data(TensorData::new(indices, [count]), &device),
            ))
        }
    }
}

/// Returns the explicit boundary while decoder cache ownership is being stabilized.
fn cached_decoder_unavailable() -> Error {
    Error::BackendCapability(
        "T5 cached decoding is not part of the stateless vertical slice; use forward until cross-attention cache integration lands"
            .into(),
    )
}

#[cfg(test)]
mod tests {
    use burn::{
        backend::Flex,
        module::Param,
        tensor::{Bool, Int, Tensor, TensorData, Tolerance},
    };

    use super::*;
    use crate::models::config::ModelConfig;

    type TestBackend = Flex<f32, i32>;

    /// Returns a tiny topology that still exercises every T5 attention path.
    fn tiny_config(projection: T5FeedForwardProjection) -> T5Config {
        T5Config {
            vocab_size: 16,
            d_model: 4,
            d_kv: 2,
            d_ff: 8,
            num_layers: 1,
            num_decoder_layers: Some(1),
            num_heads: 2,
            relative_attention_num_buckets: 8,
            relative_attention_max_distance: 32,
            dropout_rate: 0.0,
            feed_forward_proj: projection,
            dense_act_fn: None,
            is_gated_act: None,
            tie_word_embeddings: true,
            ..T5Config::default()
        }
    }

    /// Replaces all four runtime projection matrices with two-dimensional identity.
    fn make_attention_identity(attention: &mut T5Attention<TestBackend>) {
        let device = attention.attention.q_proj.weight.device();
        let identity = Tensor::eye(2, &device);
        attention.attention.q_proj.weight = Param::from_tensor(identity.clone());
        attention.attention.k_proj.weight = Param::from_tensor(identity.clone());
        attention.attention.v_proj.weight = Param::from_tensor(identity.clone());
        attention.attention.o_proj.weight = Param::from_tensor(identity);
    }

    /// Reads the first position from a `[1, sequence, width]` tensor.
    fn first_position(tensor: Tensor<TestBackend, 3>) -> Vec<f32> {
        let width = tensor.dims()[2];
        tensor
            .narrow(1, 0, 1)
            .reshape([width])
            .into_data()
            .to_vec::<f32>()
            .expect("Flex exposes f32 values")
    }

    /// Decoder causality hides future values while encoder attention observes them.
    #[test]
    fn encoder_is_bidirectional_and_decoder_self_attention_is_causal() {
        let device = Default::default();
        let config = T5Config {
            d_model: 2,
            d_kv: 2,
            num_heads: 1,
            d_ff: 4,
            vocab_size: 8,
            num_layers: 1,
            num_decoder_layers: Some(1),
            relative_attention_num_buckets: 8,
            relative_attention_max_distance: 32,
            dropout_rate: 0.0,
            ..T5Config::default()
        };
        let mut encoder = T5LayerSelfAttention::init(&config, false, false, &device)
            .expect("valid encoder attention");
        let mut decoder = T5LayerSelfAttention::init(&config, true, false, &device)
            .expect("valid decoder attention");
        make_attention_identity(&mut encoder.self_attention);
        make_attention_identity(&mut decoder.self_attention);
        let first = Tensor::from_floats([[[1.0, 0.0], [0.0, 1.0]]], &device);
        let changed = Tensor::from_floats([[[1.0, 0.0], [1.0, 1.0]]], &device);
        let zero_bias = Tensor::zeros([1, 1, 2, 2], &device);

        let encoder_first = first_position(
            encoder
                .forward(first.clone(), None, zero_bias.clone())
                .expect("encoder forward"),
        );
        let encoder_changed = first_position(
            encoder
                .forward(changed.clone(), None, zero_bias.clone())
                .expect("encoder changed forward"),
        );
        assert!(
            encoder_first
                .iter()
                .zip(&encoder_changed)
                .any(|(left, right)| (left - right).abs() > 1e-4)
        );

        let decoder_first = first_position(
            decoder
                .forward(first, None, zero_bias.clone())
                .expect("decoder forward"),
        );
        let decoder_changed = first_position(
            decoder
                .forward(changed, None, zero_bias)
                .expect("decoder changed forward"),
        );
        assert_eq!(decoder_first, decoder_changed);
    }

    /// A false key mask prevents padded source content from affecting real queries.
    #[test]
    fn semantic_padding_mask_excludes_encoder_keys() {
        let device = Default::default();
        let config = T5Config {
            d_model: 2,
            d_kv: 2,
            num_heads: 1,
            d_ff: 4,
            vocab_size: 8,
            num_layers: 1,
            num_decoder_layers: Some(1),
            relative_attention_num_buckets: 8,
            relative_attention_max_distance: 32,
            dropout_rate: 0.0,
            ..T5Config::default()
        };
        let mut layer =
            T5LayerSelfAttention::init(&config, false, false, &device).expect("valid attention");
        make_attention_identity(&mut layer.self_attention);
        let mask = TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
            [[true, false]].into(),
            &device,
        ));
        let first = Tensor::from_floats([[[1.0, 0.0], [0.0, 1.0]]], &device);
        let changed = Tensor::from_floats([[[1.0, 0.0], [1.0, 1.0]]], &device);
        let bias = Tensor::zeros([1, 1, 2, 2], &device);
        let first = first_position(
            layer
                .forward(first, Some(&mask), bias.clone())
                .expect("masked forward"),
        );
        let changed = first_position(
            layer
                .forward(changed, Some(&mask), bias)
                .expect("masked changed forward"),
        );
        assert_eq!(first, changed);
    }

    /// Decoder cross-attention consumes encoder key/value states.
    #[test]
    fn cross_attention_changes_with_encoder_content() {
        let device = Default::default();
        let config = T5Config {
            d_model: 2,
            d_kv: 2,
            num_heads: 1,
            d_ff: 4,
            vocab_size: 8,
            num_layers: 1,
            num_decoder_layers: Some(1),
            relative_attention_num_buckets: 8,
            relative_attention_max_distance: 32,
            dropout_rate: 0.0,
            ..T5Config::default()
        };
        let mut layer =
            T5LayerCrossAttention::init(&config, &device).expect("valid cross attention");
        make_attention_identity(&mut layer.enc_dec_attention);
        let query = Tensor::from_floats([[[1.0, 0.0]]], &device);
        let first_encoder = Tensor::from_floats([[[1.0, 0.0], [0.0, 1.0]]], &device);
        let changed_encoder = Tensor::from_floats([[[0.0, 1.0], [0.0, 1.0]]], &device);
        let first = layer
            .forward(query.clone(), first_encoder, None)
            .expect("first cross attention")
            .into_data();
        let changed = layer
            .forward(query, changed_encoder, None)
            .expect("changed cross attention")
            .into_data();
        assert_ne!(first, changed);
    }

    /// Every serialized FFN spelling selects exactly one matching shared module.
    #[test]
    fn feed_forward_variants_select_exact_topology_and_activation() {
        use crate::architecture::Activation;

        let device = Default::default();
        for (projection, activation, gated) in [
            (T5FeedForwardProjection::Relu, Activation::Relu, false),
            (T5FeedForwardProjection::Gelu, Activation::Gelu, false),
            (
                T5FeedForwardProjection::GatedGelu,
                Activation::GeluTanh,
                true,
            ),
            (
                T5FeedForwardProjection::GatedGeluNew,
                Activation::GeluTanh,
                true,
            ),
            (T5FeedForwardProjection::GatedSilu, Activation::Silu, true),
        ] {
            let layer = T5LayerFeedForward::<TestBackend>::init(&tiny_config(projection), &device)
                .expect("valid feed-forward variant");
            if gated {
                assert!(layer.dense.is_none());
                assert_eq!(
                    layer.gated.expect("gated projection").activation,
                    activation
                );
            } else {
                assert!(layer.gated.is_none());
                assert_eq!(
                    layer.dense.expect("dense projection").activation,
                    activation
                );
            }
        }
    }

    /// T5's global initializer factor also scales every RMSNorm weight.
    #[test]
    fn t5_norm_initializer_honors_global_factor() {
        let device = Default::default();
        let config = T5Config {
            d_model: 3,
            initializer_factor: 2.5,
            ..tiny_config(T5FeedForwardProjection::Relu)
        };
        let values = t5_norm::<TestBackend>(&config, &device)
            .expect("valid norm")
            .weight
            .val()
            .into_data();
        assert_eq!(values, TensorData::from([2.5_f32, 2.5, 2.5]));
    }

    /// Tied logits reuse the sole embedding table and apply `d_model^-1/2`.
    #[test]
    fn tied_lm_head_matches_shared_weight_scaling_equation() {
        let device = Default::default();
        let mut config = tiny_config(T5FeedForwardProjection::Relu);
        config.vocab_size = 3;
        config.d_model = 2;
        config.d_kv = 2;
        config.num_heads = 1;
        config.d_ff = 4;
        let mut model =
            T5ForConditionalGeneration::<TestBackend>::init(config, &device).expect("valid model");
        model.shared.weight = Param::from_tensor(Tensor::from_floats(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            &device,
        ));
        let hidden = Tensor::from_floats([[[2.0_f32.sqrt(), 0.0]]], &device);
        model
            .project_logits(hidden)
            .expect("tied projection")
            .into_data()
            .assert_approx_eq::<f32>(
                &TensorData::from([[[1.0, 0.0, 1.0]]]),
                Tolerance::absolute(1e-5),
            );
    }

    /// T5 v1.1/FLAN disable scaling and use their checkpoint's separate head.
    #[test]
    fn untied_flag_uses_separate_head_without_decoder_scaling() {
        let device = Default::default();
        let mut config = tiny_config(T5FeedForwardProjection::GatedGelu);
        config.vocab_size = 2;
        config.d_model = 2;
        config.d_kv = 2;
        config.num_heads = 1;
        config.d_ff = 4;
        config.tie_word_embeddings = false;
        let mut model = T5ForConditionalGeneration::<TestBackend>::init(config, &device)
            .expect("valid T5 v1.1 model");
        model.shared.weight =
            Param::from_tensor(Tensor::from_floats([[100.0, 0.0], [0.0, 100.0]], &device));
        model
            .lm_head
            .as_mut()
            .expect("untied checkpoint head")
            .weight = Param::from_tensor(Tensor::eye(2, &device));

        model
            .project_logits(Tensor::from_floats([[[2.0, 4.0]]], &device))
            .expect("untied unscaled projection")
            .into_data()
            .assert_approx_eq::<f32>(&TensorData::from([[[2.0, 4.0]]]), Tolerance::absolute(1e-6));
    }

    /// A tiny complete graph returns requested encoder and decoder diagnostics.
    #[test]
    fn stateless_forward_covers_encoder_decoder_and_cross_attention() {
        let device = Default::default();
        TestBackend::seed(&device, 19);
        let model = T5ForConditionalGeneration::<TestBackend>::init(
            tiny_config(T5FeedForwardProjection::GatedGelu),
            &device,
        )
        .expect("valid tiny model");
        let encoder_mask = TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
            [[true, true, false], [true, true, true]].into(),
            &device,
        ));
        let decoder_mask = TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
            [[true, true], [true, false]].into(),
            &device,
        ));
        let output = model
            .forward(
                Seq2SeqInput {
                    encoder: CausalLmInput {
                        token_ids: Tensor::<TestBackend, 2, Int>::from_ints(
                            [[1, 2, 0], [3, 4, 5]],
                            &device,
                        ),
                        token_mask: Some(encoder_mask),
                        position_ids: None,
                    },
                    decoder: CausalLmInput {
                        token_ids: Tensor::<TestBackend, 2, Int>::from_ints(
                            [[0, 6], [0, 7]],
                            &device,
                        ),
                        token_mask: Some(decoder_mask),
                        position_ids: None,
                    },
                },
                &ForwardOptions {
                    logits: LogitsSelection::All,
                    hidden_states: true,
                    ..ForwardOptions::default()
                },
            )
            .expect("stateless T5 forward");
        assert_eq!(output.encoder.hidden_state.dims(), [2, 3, 4]);
        assert_eq!(output.encoder.hidden_states.len(), 3);
        assert_eq!(output.decoder.logits.dims(), [2, 2, 16]);
        assert_eq!(output.decoder.hidden_states.len(), 3);
        assert_eq!(
            output
                .decoder
                .final_hidden_state
                .expect("requested final state")
                .dims(),
            [2, 2, 4]
        );
        assert!(output.cross_attentions.is_empty());
    }

    /// Every public malformed shape is rejected before indexing or reshape panics.
    #[test]
    fn stateless_input_validation_is_contextual() {
        let device = Default::default();
        let model = T5ForConditionalGeneration::<TestBackend>::init(
            tiny_config(T5FeedForwardProjection::Relu),
            &device,
        )
        .expect("valid model");
        let empty_mask = TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
            [[false, false]].into(),
            &device,
        ));
        let error = model
            .encode(
                CausalLmInput {
                    token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[1, 2]], &device),
                    token_mask: Some(empty_mask),
                    position_ids: None,
                },
                &ForwardOptions::default(),
            )
            .expect_err("fully masked rows are invalid");
        assert!(error.to_string().contains("fully masked row"));

        let error = model
            .encode(
                CausalLmInput {
                    token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[16]], &device),
                    token_mask: None,
                    position_ids: None,
                },
                &ForwardOptions::default(),
            )
            .expect_err("out-of-range IDs are invalid");
        assert!(error.to_string().contains("0..16"));

        let error = model
            .encode(
                CausalLmInput {
                    token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[1]], &device),
                    token_mask: None,
                    position_ids: Some(Tensor::zeros([1, 1], &device)),
                },
                &ForwardOptions::default(),
            )
            .expect_err("absolute positions are unsupported");
        assert!(matches!(error, Error::BackendCapability(_)));
    }

    /// Last-valid and explicit-position selection retain order for every batch row.
    #[test]
    fn logit_selection_handles_gapped_masks_and_ordered_positions() {
        let device = Default::default();
        let hidden = Tensor::<TestBackend, 3>::from_floats(
            [[[0.0], [1.0], [2.0]], [[3.0], [4.0], [5.0]]],
            &device,
        );
        let mask = TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
            [[true, false, true], [false, true, false]].into(),
            &device,
        ));
        select_hidden(hidden.clone(), Some(&mask), &LogitsSelection::Last)
            .expect("semantic last position")
            .into_data()
            .assert_approx_eq::<f32>(
                &TensorData::from([[[2.0]], [[4.0]]]),
                Tolerance::absolute(1e-6),
            );
        select_hidden(
            hidden.clone(),
            None,
            &LogitsSelection::Positions(vec![2, 0]),
        )
        .expect("ordered explicit positions")
        .into_data()
        .assert_approx_eq::<f32>(
            &TensorData::from([[[2.0], [0.0]], [[5.0], [3.0]]]),
            Tolerance::absolute(1e-6),
        );
        assert!(
            select_hidden(
                hidden.clone(),
                None,
                &LogitsSelection::Positions(Vec::new())
            )
            .is_err()
        );
        assert!(select_hidden(hidden, None, &LogitsSelection::Positions(vec![3])).is_err());
    }

    /// Official FLAN and UL2 schemas preserve their distinct gated activations.
    #[test]
    fn official_flan_and_ul2_fixtures_initialize_when_reduced() {
        let device = Default::default();
        let mut flan =
            T5Config::from_json_str(include_str!("../../../tests/fixtures/flan_t5_xl.json"))
                .expect("official FLAN fixture");
        let mut ul2 = T5Config::from_json_str(include_str!("../../../tests/fixtures/ul2.json"))
            .expect("official UL2 fixture");
        assert_eq!(flan.feed_forward_proj, T5FeedForwardProjection::GatedGelu);
        assert_eq!(ul2.feed_forward_proj, T5FeedForwardProjection::GatedSilu);
        reduce_official_config(&mut flan);
        reduce_official_config(&mut ul2);

        for config in [flan, ul2] {
            let expects_separate_head = !config.tie_word_embeddings;
            let model = T5ForConditionalGeneration::<TestBackend>::init(config, &device)
                .expect("reduced official topology should initialize");
            assert_eq!(model.encoder.block.len(), 1);
            assert_eq!(model.decoder.block.len(), 1);
            assert!(model.encoder.block[0].feed_forward.gated.is_some());
            assert_eq!(model.lm_head.is_some(), expects_separate_head);
            assert!(
                model.encoder.block[0]
                    .self_attention
                    .self_attention
                    .relative_attention_bias
                    .is_some()
            );
            assert!(
                model.decoder.block[0]
                    .self_attention
                    .self_attention
                    .relative_attention_bias
                    .is_some()
            );
        }
    }

    /// Shrinks only allocation dimensions while retaining a fixture's family choices.
    fn reduce_official_config(config: &mut T5Config) {
        config.vocab_size = 32;
        config.d_model = 8;
        config.d_kv = 4;
        config.d_ff = 16;
        config.num_heads = 2;
        config.num_layers = 1;
        config.num_decoder_layers = Some(1);
        config.dropout_rate = 0.0;
    }

    /// The trait advertises stateless methods but fails cached calls explicitly.
    #[test]
    fn cached_trait_boundary_is_explicit() {
        let device = Default::default();
        let model = T5ForConditionalGeneration::<TestBackend>::init(
            tiny_config(T5FeedForwardProjection::Relu),
            &device,
        )
        .expect("valid model");
        let cache = CacheConfig::new(1, 8).expect("valid cache config");
        let error = <T5ForConditionalGeneration<TestBackend> as Seq2SeqLanguageModel<
            TestBackend,
        >>::new_decode_state(&model, cache, 1, &device)
        .expect_err("cached slice is intentionally isolated");
        assert!(error.to_string().contains("stateless vertical slice"));
    }

    /// T5 attention scale remains one even when `d_kv` is not one.
    #[test]
    fn attention_uses_unscaled_dot_products() {
        let device = Default::default();
        let config = tiny_config(T5FeedForwardProjection::Relu);
        let attention =
            T5Attention::<TestBackend>::init(&config, None, &device).expect("valid attention");
        assert_eq!(attention.attention.scale, Some(1.0));
        assert_eq!(config.attention_inner_dim(), 4);
    }
}
