//! Burn implementation of the T5/FLAN/UL2 transformer graph.
//!
//! T5 is a pre-normalized encoder-decoder. Encoder self-attention is
//! bidirectional, decoder self-attention is causal, and every decoder block has
//! an additional cross-attention sublayer over the encoder's final states. The
//! first self-attention block in each stack owns a learned relative-position
//! table; its computed bias is built once per pass and reused by every later
//! block in that stack (see [`bias`]).
//!
//! Unlike most modern decoder-only transformers, T5 does **not** divide dot
//! products by `sqrt(d_kv)`: attention uses `softmax(QK^T + bias)`. The shared
//! attention primitive is therefore configured with an explicit scale of one.
//! This graph omits stochastic dropout during both inference and autodiff
//! execution, regardless of configured dropout metadata. Its floating parameters
//! can be optimized; training-time regularization must be composed explicitly.
//!
//! # One graph for every path
//!
//! The block equations are written once, in [`block`]: plain and hooked
//! stateless passes, the staged encoder recipe, and cached decoding all run
//! [`T5Block::run`] and differ only in their masks, bias, and key/value
//! sources. Stateful inference keeps decoder self-KV and immutable encoder
//! cross-KV outside this parameter graph in [`T5DecodeState`] and evaluates
//! every batch row in one pass (see [`decode`]).
//!
//! # Experiment interfaces
//!
//! [`T5ForConditionalGeneration`] and [`T5Encoder`] implement
//! [`crate::architecture::HookedModel`], and [`T5Encoder`] implements
//! [`crate::architecture::StagedModel`] over the encoder stack, so
//! family-generic caches and patching sweeps run on T5 unchanged (see
//! `experiment.rs`).
//!
//! # Padded targets
//!
//! A target mask (`true == valid`) never lets padding enter a valid token's
//! attention: padded keys are excluded and valid tokens use *logical*
//! positions that count valid tokens only, so left, right, and interior
//! padding produce the same valid outputs as the unpadded rows. The whole
//! batch runs as one rectangle. Padded queries still attend to their causal
//! keys, so no softmax row is ever empty (which some backends would turn
//! into `NaN` in the forward and backward passes); padded positions of the
//! returned hidden states, diagnostics, and logits are then set to exact
//! zeros.

use burn::{
    module::{Initializer, Module},
    nn::{Embedding, EmbeddingConfig, Linear, LinearConfig, LinearLayout},
    tensor::{Bool, DType, Tensor, backend::Backend},
};

use crate::{CausalLmInput, CausalLmOutput, Error, ForwardOptions, Result};
use crate::{
    architecture::{EncoderOutput, Seq2SeqInput, Seq2SeqOutput, TokenMask},
    attention::{AttentionPrecisionMode, GroupedQueryAttention, GroupedQueryAttentionConfig},
    feed_forward::{
        DenseFeedForward, DenseFeedForwardConfig, GatedFeedForward, GatedFeedForwardConfig,
    },
    intervention::{AttentionPoints, Hooks, Layer, PointInfo},
    normalization::{RmsNorm, RmsNormConfig},
    operations,
    position::{T5RelativePositionBias, T5RelativePositionBiasConfig},
};

use super::{T5Config, T5FeedForwardProjection, T5Stacks};
use crate::models::shared::{
    DeferredChecks, DeferredInput, ImplicitPositions, InputContract, InputNames, ValidatedInput,
    check_nonempty_mask, defer_input, ensure_grid, ensure_nonempty, ensure_same_device,
    plan::{ScatterLayout, plan_scatter},
    select_hidden, validate_input,
};

mod bias;
mod block;
mod decode;
mod experiment;
mod packing;
mod padded_hooks;
mod recipe;

use block::{CrossAttentionPlan, CrossSource, FreshKeyValues, SelfAttentionPlan, StackPlan};

pub use decode::T5DecodeState;
pub use recipe::{T5Encoder, T5EncoderLayerSpec, T5EncoderRecipe};

/// Tensor names reported by T5 encoder input errors.
const ENCODER_INPUT: InputNames<'static> = InputNames {
    token_ids: "t5.encoder.input_ids",
    token_mask: "t5.encoder.token_mask",
    position_ids: "t5.encoder.position_ids",
};

/// Tensor names reported by T5 decoder input errors.
const DECODER_INPUT: InputNames<'static> = InputNames {
    token_ids: "t5.decoder.input_ids",
    token_mask: "t5.decoder.token_mask",
    position_ids: "t5.decoder.position_ids",
};

/// Name of a supplied encoder result's source mask in errors.
const ENCODER_OUTPUT_MASK: &str = "t5.encoder_output.token_mask";

/// Owner named when a tensor is on another device than the weights.
const PARAMETERS: &str = "T5 model parameters";

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

    /// Lists only attention tensors supported by the configured numerical path.
    fn activation_points(&self, points: AttentionPoints) -> Vec<PointInfo> {
        let mut available = vec![
            points.query().info(),
            points.key().info(),
            points.value().info(),
        ];
        if self.attention.precision_mode == AttentionPrecisionMode::EagerFp32Softmax {
            available.extend([points.scores().info(), points.pattern().info()]);
        }
        available.push(points.head_context().info());
        available
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

    /// Lists the tensors visited by this block in forward execution order.
    fn activation_points(&self, layer: Layer) -> Vec<PointInfo> {
        let mut points = vec![layer.residual_pre().info(), layer.attention_input().info()];
        points.extend(
            self.self_attention
                .self_attention
                .activation_points(layer.self_attention()),
        );
        points.push(layer.attention_output().info());
        if let Some(cross_attention) = &self.cross_attention {
            points.push(layer.cross_attention_input().info());
            points.extend(
                cross_attention
                    .enc_dec_attention
                    .activation_points(layer.cross_attention()),
            );
            points.push(layer.cross_attention_output().info());
        }
        points.extend([
            layer.mlp_input().info(),
            layer.mlp_pre_activation().info(),
            layer.mlp_activation().info(),
            layer.mlp_output().info(),
            layer.residual_post().info(),
        ]);
        points
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

    /// Returns the learned relative-position table owned by block zero.
    ///
    /// Every block of the stack reuses the bias this table produces.
    fn relative_table(&self) -> Result<&T5RelativePositionBias<B>> {
        self.block
            .first()
            .and_then(|block| {
                block
                    .self_attention
                    .self_attention
                    .relative_attention_bias
                    .as_ref()
            })
            .ok_or_else(|| {
                Error::configuration(
                    "t5.relative_attention_bias",
                    "block zero of every stack must own the shared bias table",
                )
            })
    }

    /// Lists every tensor address for this stack, including embedding boundaries.
    fn activation_points(&self) -> Vec<PointInfo> {
        let stack = self.stack();
        let mut points = vec![stack.embedding().info()];
        for (index, block) in self.block.iter().enumerate() {
            points.extend(block.activation_points(self.layer(index)));
        }
        points.push(stack.final_norm().info());
        points
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
    /// Initializes a validated T5/FLAN/UL2 parameter graph.
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
        let model_device = self.shared.weight.lazy_device();
        validate_token_input(
            &source,
            self.config.vocab_size,
            &model_device,
            ENCODER_INPUT,
            "encoder",
        )?;
        self.encode_validated(source, options)
    }

    /// Runs the encoder stack on a source that already passed [`validate_token_input`].
    fn encode_validated(
        &self,
        source: CausalLmInput<B>,
        options: &ForwardOptions,
    ) -> Result<EncoderOutput<B>> {
        self.encode_with(source, options, &mut Hooks::new())
    }

    /// Runs the bidirectional encoder on a validated source, emitting encoder points.
    ///
    /// Padded source keys are excluded; the relative bias is the stack's
    /// single `[1, heads, source, source]` plane, built once for every block.
    fn encode_with(
        &self,
        source: CausalLmInput<B>,
        options: &ForwardOptions,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<EncoderOutput<B>> {
        let [_, sequence] = source.token_ids.dims();
        let hidden = self.shared.forward(source.token_ids.clone());
        let bias = bias::contiguous_bias(
            self.encoder.relative_table()?,
            sequence,
            sequence,
            0,
            &hidden.device(),
        )?;
        let plan = StackPlan {
            self_attention: SelfAttentionPlan::padded(
                source.token_mask.as_ref(),
                Some(bias),
                false,
            ),
            cross_attention: None,
        };
        let output = self.encoder.run(
            hidden,
            &plan,
            &mut FreshKeyValues,
            options.hidden_states,
            hooks,
        )?;
        Ok(EncoderOutput {
            hidden_state: output.hidden,
            token_mask: source.token_mask,
            hidden_states: output.hidden_states,
            attentions: Vec::new(),
        })
    }

    /// Evaluates the complete stateless encoder-decoder graph.
    ///
    /// Source and target are validated together with one device-to-host
    /// readback before the encoder runs. The resulting encoder mask is the
    /// validated source mask, so the decoder re-checks only the sync-free
    /// encoder layout (for example a source/target batch mismatch).
    pub fn forward(
        &self,
        input: Seq2SeqInput<B>,
        options: &ForwardOptions,
    ) -> Result<Seq2SeqOutput<B>> {
        validate_options(options)?;
        let target = self.validate_seq2seq_input(&input, TargetColumns::Read)?;
        let encoder = self.encode_validated(input.encoder, options)?;
        validate_encoder_layout(
            &encoder,
            input.decoder.token_ids.dims()[0],
            self.config.d_model,
            self.shared.weight.val().dtype(),
        )?;
        let decoder = self.decode_validated(&encoder, input.decoder, options, target.columns)?;
        Ok(Seq2SeqOutput {
            decoder,
            encoder,
            cross_attentions: Vec::new(),
        })
    }

    /// Returns the stateless encoder and decoder activation addresses in execution order.
    ///
    /// Attention and MLP inputs are normalized; their outputs are residual branches
    /// before addition. For gated MLPs, `mlp_pre_activation` and `mlp_activation`
    /// refer to the gate branch before and after its nonlinearity, respectively.
    /// Q/K/V and head-context tensors are available in both stacks. Attention
    /// scores and patterns appear only when the corresponding attention module
    /// uses [`AttentionPrecisionMode::EagerFp32Softmax`].
    #[must_use]
    pub fn activation_points(&self) -> Vec<PointInfo> {
        let mut points = self.encoder.activation_points();
        points.extend(self.decoder.activation_points());
        points
    }

    /// Encodes source tokens with selective activation capture or intervention.
    ///
    /// This is the natural entry point for encoder classifiers and representation
    /// experiments. Hooks can call any ordinary Burn module and retain gradients.
    /// Decoder hook addresses are rejected before execution. Padding follows the
    /// same `true == valid` convention as [`Self::encode`].
    pub fn encode_with_hooks(
        &self,
        source: CausalLmInput<B>,
        options: &ForwardOptions,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<EncoderOutput<B>> {
        if !hooks.has_work() {
            return self.encode(source, options);
        }
        hooks.run(self.encoder.activation_points(), |hooks| {
            self.encode_hooked(source, options, hooks)
        })
    }

    /// Executes the complete stateless graph with selective tensor interventions.
    ///
    /// Register callbacks using addresses such as `Layer::encoder(0).mlp_activation()`
    /// or `Layer::decoder(0).cross_attention_output()`. Replacements must preserve
    /// shape, dtype, and device; subsequent operations consume the returned tensor.
    /// Edits preserve the autodiff graph. Captures and observation callbacks
    /// detach by default; use [`Hooks::capture_with_grad`] to retain connectivity.
    ///
    /// Decoder hooks retain the original batch and token coordinates, including
    /// left, right, and interior padding. Padded decoder activations are zero
    /// before observation and after edits; padding cannot be revived by a callback.
    /// Valid tokens use logical relative positions that exclude padding. Source
    /// padding retains ordinary encoder semantics and is excluded in cross-attention.
    /// Stateful decode caches do not participate in stateless interventions.
    /// Empty hook collections delegate to [`Self::forward`].
    pub fn forward_with_hooks(
        &self,
        input: Seq2SeqInput<B>,
        options: &ForwardOptions,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<Seq2SeqOutput<B>> {
        if !hooks.has_work() {
            return self.forward(input, options);
        }
        let validity = match self.decoder_hook_validity(&input) {
            Ok(validity) => validity,
            // Even validity-plan failures belong to a run and invalidate old
            // captures. This error-only scope executes no model or user callback.
            Err(error) => return hooks.run(self.activation_points(), |_| Err(error)),
        };
        hooks.run_with_validity(self.activation_points(), validity, |hooks| {
            validate_options(options)?;
            let target = self.validate_seq2seq_input(&input, TargetColumns::Skip)?;
            let mut decoder_input = input.decoder;
            // Tokenizers commonly emit an all-valid mask for an unpadded batch. The
            // validation readback already established whether any padding exists,
            // so normalizing it away needs no further synchronization.
            if target.fully_valid {
                decoder_input.token_mask = None;
            }
            let encoder = self.encode_hooked_validated(input.encoder, options, hooks)?;
            let decoder = self.decode_stateless_hooked(&encoder, decoder_input, options, hooks)?;
            Ok(Seq2SeqOutput {
                decoder,
                encoder,
                cross_attentions: Vec::new(),
            })
        })
    }

    /// Validates the source, then runs the encoder inside an encoder-only hook session.
    fn encode_hooked(
        &self,
        source: CausalLmInput<B>,
        options: &ForwardOptions,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<EncoderOutput<B>> {
        validate_options(options)?;
        let model_device = self.shared.weight.lazy_device();
        validate_token_input(
            &source,
            self.config.vocab_size,
            &model_device,
            ENCODER_INPUT,
            "encoder",
        )?;
        self.encode_hooked_validated(source, options, hooks)
    }

    /// Runs the encoder with hooks on a source that already passed validation.
    fn encode_hooked_validated(
        &self,
        source: CausalLmInput<B>,
        options: &ForwardOptions,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<EncoderOutput<B>> {
        self.encode_with(source, options, hooks)
    }

    /// Validates source and target of a complete pass with one readback.
    ///
    /// Runs before the encoder and before any user callback. Layout and device
    /// errors of both inputs are reported first; value errors follow in the
    /// order source tokens, source mask, target tokens, target mask. With
    /// [`TargetColumns::Read`] the target mask rides along in the same
    /// transfer, so masked decoding needs no further synchronization.
    fn validate_seq2seq_input(
        &self,
        input: &Seq2SeqInput<B>,
        columns: TargetColumns,
    ) -> Result<ValidatedTarget> {
        let model_device = self.shared.weight.lazy_device();
        let vocabulary = self.config.vocab_size;
        let mut checks = DeferredChecks::new();
        let source = defer_token_input(
            &input.encoder,
            vocabulary,
            &model_device,
            ENCODER_INPUT,
            "encoder",
            &mut checks,
        )?;
        let target = defer_token_input(
            &input.decoder,
            vocabulary,
            &model_device,
            DECODER_INPUT,
            "decoder",
            &mut checks,
        )?;
        ValidatedTarget::read(checks, &input.decoder, target, Some(source), columns)
    }

    /// Implements rectangular decoder execution inside the same hook session.
    ///
    /// The caller validated both inputs with [`Self::validate_seq2seq_input`]
    /// and produced `encoder` from the validated source, so only the sync-free
    /// encoder layout is re-checked here. A target mask selects logical
    /// positions, because hooks skip the host column readback that would
    /// identify one-run (left or right) padding.
    fn decode_stateless_hooked(
        &self,
        encoder: &EncoderOutput<B>,
        decoder: CausalLmInput<B>,
        options: &ForwardOptions,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<CausalLmOutput<B>> {
        let padding = match decoder.token_mask.clone() {
            None => TargetPadding::None,
            Some(mask) => TargetPadding::Logical(mask),
        };
        self.decode_with(encoder, decoder, padding, options, hooks)
    }

    /// Runs a validated target as one batched rectangle.
    ///
    /// `columns` holds each row's valid columns exactly when `decoder` has a
    /// mask; every row has at least one valid column. The columns decide the
    /// bias geometry: one-run padding keeps the contiguous bias, interior
    /// padding needs logical positions (see [`TargetPadding`]).
    fn decode_validated(
        &self,
        encoder: &EncoderOutput<B>,
        mut decoder: CausalLmInput<B>,
        options: &ForwardOptions,
        columns: Option<Vec<Vec<usize>>>,
    ) -> Result<CausalLmOutput<B>> {
        let sequence = decoder.token_ids.dims()[1];
        let padding = match (decoder.token_mask.take(), columns) {
            (Some(mask), Some(columns)) => TargetPadding::classify(mask, &columns, sequence),
            // Without host columns the layout is unknown; logical positions are exact.
            (Some(mask), None) => TargetPadding::Logical(mask),
            (None, _) => TargetPadding::None,
        };
        decoder.token_mask = padding.mask().cloned();
        self.decode_with(encoder, decoder, padding, options, &mut Hooks::new())
    }

    /// Runs the causal decoder over a whole target rectangle, emitting decoder points.
    ///
    /// Valid target queries exclude padded keys; padded queries keep their
    /// causal keys so no score row is empty (see
    /// [`SelfAttentionPlan::padded_causal`]). Padded positions of the final
    /// hidden state and every diagnostic are set to exact zeros, so their
    /// logits are zero; valid positions equal an unpadded evaluation of each
    /// row (see [`TargetPadding`] for the bias). Cross-attention projects the
    /// encoder states in every block and excludes padded source keys.
    fn decode_with(
        &self,
        encoder: &EncoderOutput<B>,
        decoder: CausalLmInput<B>,
        padding: TargetPadding<B>,
        options: &ForwardOptions,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<CausalLmOutput<B>> {
        let [batch, sequence] = decoder.token_ids.dims();
        validate_encoder_layout(
            encoder,
            batch,
            self.config.d_model,
            self.shared.weight.val().dtype(),
        )?;
        let hidden = self.shared.forward(decoder.token_ids.clone());
        let device = hidden.device();
        let table = self.decoder.relative_table()?;
        let bias = match &padding {
            TargetPadding::None | TargetPadding::Contiguous(_) => {
                bias::contiguous_bias(table, sequence, sequence, 0, &device)?
            }
            TargetPadding::Logical(mask) => {
                let positions = padded_hooks::logical_decoder_positions(mask)?;
                // Logical positions lie in 0..sequence.
                bias::positional_bias(table, positions.clone(), positions, sequence - 1)?
            }
        };
        let self_attention = match padding.mask() {
            Some(mask) => SelfAttentionPlan::padded_causal(mask, Some(bias)),
            None => SelfAttentionPlan::padded(None, Some(bias), true),
        };
        let plan = StackPlan {
            self_attention,
            cross_attention: Some(CrossAttentionPlan::new(
                CrossSource::Encoder(&encoder.hidden_state),
                encoder.token_mask.as_ref(),
            )),
        };
        let output = self.decoder.run(
            hidden,
            &plan,
            &mut FreshKeyValues,
            options.hidden_states,
            hooks,
        )?;
        let zero_padding = |hidden: Tensor<B, 3>| match padding.mask() {
            Some(mask) => zero_invalid_positions(hidden, mask),
            None => hidden,
        };
        let hidden = zero_padding(output.hidden);
        let hidden_states = output.hidden_states.into_iter().map(zero_padding).collect();
        let selected = select_hidden(hidden.clone(), padding.mask(), &options.logits)?;
        Ok(CausalLmOutput {
            logits: self.project_logits(selected)?,
            final_hidden_state: options.hidden_states.then_some(hidden),
            hidden_states,
            attentions: Vec::new(),
            routers: Vec::new(),
            load_balancing_loss: None,
        })
    }

    /// Applies the tied/scaled or independent/unscaled vocabulary projection.
    ///
    /// A tied head computes `(h / sqrt(d_model)) · Eᵀ` against the transposed
    /// view of the shared `[vocabulary, d_model]` table without copying it
    /// ([`operations::matmul_transposed`]).
    fn project_logits(&self, hidden: Tensor<B, 3>) -> Result<Tensor<B, 3>> {
        let hidden_device = hidden.device();
        if self.config.tie_word_embeddings {
            if self.lm_head.is_some() {
                return Err(Error::configuration(
                    "t5.lm_head",
                    "tied checkpoints must not own a separate vocabulary projection",
                ));
            }
            ensure_same_device(
                "t5.shared",
                &self.shared.weight.lazy_device(),
                &hidden_device,
                PARAMETERS,
            )?;
            let hidden = hidden.mul_scalar(1.0 / (self.config.d_model as f64).sqrt());
            return operations::matmul_transposed(hidden, self.shared.weight.val());
        }
        self.lm_head
            .as_ref()
            .ok_or_else(|| {
                Error::configuration(
                    "t5.lm_head",
                    "untied checkpoints require a separate vocabulary projection",
                )
            })
            .and_then(|head| {
                ensure_same_device(
                    "t5.lm_head",
                    &head.weight.lazy_device(),
                    &hidden_device,
                    PARAMETERS,
                )?;
                operations::linear(hidden, head)
            })
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

/// Initializes a bias-free, gradient-safe linear map with an exact T5 variance.
fn t5_linear<B: Backend>(input: usize, output: usize, std: f64, device: &B::Device) -> Linear<B> {
    LinearConfig::new(input, output)
        .with_bias(false)
        // Burn 0.21's Col-layout initialization mapper transposes through an
        // untracked tensor and therefore disconnects weight gradients. Keep
        // the differentiable runtime parameter in [input, output] order; the
        // generic checkpoint plan explicitly transposes official [output, input]
        // matrices once before this architecture's sink installs them.
        .with_layout(LinearLayout::Row)
        .with_initializer(Initializer::Normal { mean: 0.0, std })
        .init(device)
}

/// How a stateless target's padding shapes the batched decoder pass.
///
/// Valid tokens of a padded row must see exactly the relative bias an
/// unpadded row of their valid tokens would see: `b(p_k - p_q)` with logical
/// positions `p` that count valid tokens only. When a row's valid tokens form
/// one run (left or right padding), physical displacements between valid
/// tokens already equal logical ones, so the ordinary `[1, heads, S, S]` bias
/// is exact for every valid pair; only interior padding needs the per-row
/// logical bias `[batch, heads, S, S]`.
#[derive(Debug, Clone)]
enum TargetPadding<B: Backend> {
    /// Every token is valid.
    None,
    /// Each row's valid tokens form one run; the contiguous bias is exact.
    Contiguous(
        /// Target validity mask `[batch, sequence]`.
        TokenMask<B>,
    ),
    /// Interior padding, or an unknown layout: logical-position bias.
    Logical(
        /// Target validity mask `[batch, sequence]`, used to count logical positions.
        TokenMask<B>,
    ),
}

impl<B: Backend> TargetPadding<B> {
    /// Classifies a mask from its host valid columns (one list per row).
    fn classify(mask: TokenMask<B>, columns: &[Vec<usize>], sequence: usize) -> Self {
        let mut all_valid = true;
        for row in columns {
            match plan_scatter(row, sequence) {
                ScatterLayout::Identity => {}
                ScatterLayout::Contiguous { .. } => all_valid = false,
                ScatterLayout::Gather { .. } => return Self::Logical(mask),
            }
        }
        if all_valid {
            Self::None
        } else {
            Self::Contiguous(mask)
        }
    }

    /// Returns the target mask, or `None` for an unpadded target.
    const fn mask(&self) -> Option<&TokenMask<B>> {
        match self {
            Self::None => None,
            Self::Contiguous(mask) | Self::Logical(mask) => Some(mask),
        }
    }
}

/// Replaces the states of padded `[batch, token]` positions with exact zeros.
///
/// Uses a select (`mask_fill`), not a product, so non-finite values computed
/// at padded positions cannot leak; gradients of padded entries are zero.
fn zero_invalid_positions<B: Backend>(hidden: Tensor<B, 3>, mask: &TokenMask<B>) -> Tensor<B, 3> {
    let shape = hidden.dims();
    let invalid: Tensor<B, 3, Bool> = mask
        .as_tensor()
        .clone()
        .bool_not()
        .unsqueeze_dim::<3>(2)
        .expand(shape);
    hidden.mask_fill(invalid, 0.0)
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

/// Validates T5 token input: grid, devices, vocabulary range, and mask rows.
///
/// Token and mask values share one device-to-host readback; the result also
/// reports whether the mask contains any padding.
fn validate_token_input<B: Backend>(
    input: &CausalLmInput<B>,
    vocabulary: usize,
    device: &B::Device,
    names: InputNames<'static>,
    stack: &str,
) -> Result<ValidatedInput<B>> {
    let mut checks = DeferredChecks::new();
    let pending = defer_token_input(input, vocabulary, device, names, stack, &mut checks)?;
    pending.finish(&checks.run(names.token_ids)?)
}

/// Checks a T5 input's layout and registers its token and mask value checks.
///
/// T5 encodes positions with relative buckets, so explicit `position_ids` are
/// rejected once their extent and device are known. The token range and the
/// non-empty-row check are deferred to `checks`, so several inputs can share
/// one readback.
fn defer_token_input<'a, B: Backend>(
    input: &CausalLmInput<B>,
    vocabulary: usize,
    device: &B::Device,
    names: InputNames<'static>,
    stack: &str,
    checks: &mut DeferredChecks<'a, B>,
) -> Result<DeferredInput<'a, B>> {
    let [batch, sequence] = input.token_ids.dims();
    ensure_nonempty(batch, sequence, names.token_ids)?;
    if let Some(positions) = &input.position_ids {
        ensure_same_device(names.position_ids, &positions.device(), device, PARAMETERS)?;
        ensure_grid(positions.dims(), batch, sequence, names.position_ids)?;
        return Err(Error::BackendCapability(format!(
            "T5 {stack} uses relative positions and does not accept explicit position IDs"
        )));
    }
    defer_input(
        input,
        device,
        &InputContract::new(names, PARAMETERS, vocabulary)
            .with_implicit_positions(ImplicitPositions::Unused),
        checks,
    )
}

/// Whether a fused T5 validation also reads the target mask for row compaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TargetColumns {
    /// Read the target mask with the checks (masked stateless decoding).
    Read,
    /// Skip it (hooked decoding keeps the rectangular grid).
    Skip,
}

/// Target summary produced by one fused T5 validation readback.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ValidatedTarget {
    /// Whether the target has no padding (no mask, or an all-true mask).
    fully_valid: bool,
    /// Each target row's valid columns, when requested and a mask is present.
    columns: Option<Vec<Vec<usize>>>,
}

impl ValidatedTarget {
    /// Runs the registered checks once and summarizes the target.
    ///
    /// `source`, when given, was registered on the same `checks` and is
    /// finished from the same readout.
    fn read<B: Backend>(
        mut checks: DeferredChecks<'_, B>,
        decoder: &CausalLmInput<B>,
        target: DeferredInput<'_, B>,
        source: Option<DeferredInput<'_, B>>,
        columns: TargetColumns,
    ) -> Result<Self> {
        let mask = decoder
            .token_mask
            .as_ref()
            .filter(|_| columns == TargetColumns::Read)
            .map(|mask| checks.mask_columns(mask));
        let readout = checks.run("T5 input statistics")?;
        if let Some(source) = source {
            source.finish(&readout)?;
        }
        Ok(Self {
            fully_valid: target.finish(&readout)?.fully_valid,
            columns: mask.map(|mask| readout.valid_columns(mask)),
        })
    }
}

/// Validates externally supplied encoder states before cross-attention projection.
///
/// Checks the layout and dtype, then reads the source mask's row counts.
fn validate_encoder_output<B: Backend>(
    encoder: &EncoderOutput<B>,
    decoder_batch: usize,
    hidden_size: usize,
    decoder_dtype: DType,
) -> Result<()> {
    let mut checks = DeferredChecks::new();
    defer_encoder_output(
        &mut checks,
        encoder,
        decoder_batch,
        hidden_size,
        decoder_dtype,
    )?;
    checks.run(ENCODER_OUTPUT_MASK).map(|_| ())
}

/// Checks the encoder layout now and defers its non-empty source-mask check.
fn defer_encoder_output<B: Backend>(
    checks: &mut DeferredChecks<'_, B>,
    encoder: &EncoderOutput<B>,
    decoder_batch: usize,
    hidden_size: usize,
    decoder_dtype: DType,
) -> Result<()> {
    validate_encoder_layout(encoder, decoder_batch, hidden_size, decoder_dtype)?;
    let [batch, sequence, _] = encoder.hidden_state.dims();
    check_nonempty_mask(
        checks,
        encoder.token_mask.as_ref(),
        batch,
        sequence,
        ENCODER_OUTPUT_MASK,
    )
}

/// Checks encoder extents, dtype, and source-mask grid without reading values.
fn validate_encoder_layout<B: Backend>(
    encoder: &EncoderOutput<B>,
    decoder_batch: usize,
    hidden_size: usize,
    decoder_dtype: DType,
) -> Result<()> {
    let [batch, sequence, width] = encoder.hidden_state.dims();
    if batch != decoder_batch || sequence == 0 || width != hidden_size {
        return Err(Error::shape(
            "t5.encoder_output.hidden_state",
            format!("[{decoder_batch}, source>0, {hidden_size}]"),
            format!("[{batch}, {sequence}, {width}]"),
        ));
    }
    // Cross projections multiply the supplied source by decoder parameters.
    // Read the actual parameter dtype: a checkpoint may select F16 even when
    // the backend's default floating element type is F32. Do not silently cast
    // external encoder states or constrain higher-precision normalization weights.
    if encoder.hidden_state.dtype() != decoder_dtype {
        return Err(Error::BackendCapability(format!(
            "t5.encoder_output.hidden_state dtype must match decoder parameters {decoder_dtype:?}, found {:?}",
            encoder.hidden_state.dtype(),
        )));
    }
    if let Some(mask) = &encoder.token_mask {
        ensure_grid(
            mask.as_tensor().dims(),
            batch,
            sequence,
            ENCODER_OUTPUT_MASK,
        )?;
    }
    Ok(())
}

#[cfg(test)]
#[path = "model_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "model/intervention_tests.rs"]
mod intervention_tests;
