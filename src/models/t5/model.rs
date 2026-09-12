//! Burn implementation of the T5/FLAN/UL2 transformer graph.
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
//! This graph omits stochastic dropout during both inference and autodiff
//! execution, regardless of configured dropout metadata. Its floating parameters
//! can be optimized; training-time regularization must be composed explicitly.
//! Stateful inference keeps decoder self-KV and
//! immutable encoder cross-KV outside this parameter graph in [`T5DecodeState`].

use burn::{
    module::{Initializer, Module},
    nn::{Embedding, EmbeddingConfig, Linear, LinearConfig, LinearLayout},
    tensor::{DType, ElementConversion, Int, Tensor, TensorData, backend::Backend, module::linear},
};

use crate::{CausalLmInput, CausalLmOutput, Error, ForwardOptions, LogitsSelection, Result};
use crate::{
    architecture::{EncoderOutput, Seq2SeqInput, Seq2SeqOutput, TokenMask},
    attention::{
        AttentionPrecisionMode, GroupedQueryAttention, GroupedQueryAttentionConfig,
        padding_attention_mask,
    },
    experiment::AttentionPolicy,
    feed_forward::{
        DenseFeedForward, DenseFeedForwardConfig, GatedFeedForward, GatedFeedForwardConfig,
        apply_activation,
    },
    intervention::{AttentionPoints, Hooks, Layer, PointInfo, Stack},
    normalization::{RmsNorm, RmsNormConfig},
    operations,
    position::{T5RelativePositionBias, T5RelativePositionBiasConfig},
};

use super::{T5Config, T5FeedForwardProjection};

mod decode;
mod padded_hooks;
mod recipe;

pub use decode::T5DecodeState;
pub use recipe::{T5Encoder, T5EncoderLayerSpec, T5EncoderRecipe};

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
        self.forward_impl(
            query_hidden,
            key_value_hidden,
            key_mask,
            position_bias,
            causal,
            None,
        )
    }

    /// Runs validated attention with optional projection and head-level hooks.
    fn forward_impl(
        &self,
        query_hidden: Tensor<B, 3>,
        key_value_hidden: Tensor<B, 3>,
        key_mask: Option<&TokenMask<B>>,
        position_bias: Option<Tensor<B, 4>>,
        causal: bool,
        intervention: Option<(AttentionPoints, &mut Hooks<'_, B>)>,
    ) -> Result<Tensor<B, 3>> {
        let policy = if causal {
            AttentionPolicy::causal()
        } else {
            AttentionPolicy::bidirectional()
        };
        self.forward_policy(
            query_hidden,
            key_value_hidden,
            key_mask,
            position_bias,
            policy,
            intervention,
        )
    }

    /// Applies an explicit per-pass visibility policy without changing parameters.
    fn forward_policy(
        &self,
        query_hidden: Tensor<B, 3>,
        key_value_hidden: Tensor<B, 3>,
        key_mask: Option<&TokenMask<B>>,
        position_bias: Option<Tensor<B, 4>>,
        policy: AttentionPolicy<B>,
        intervention: Option<(AttentionPoints, &mut Hooks<'_, B>)>,
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
        let mask = policy.exclusion_mask(
            mask,
            [
                batch,
                self.attention.query_heads,
                query_sequence,
                key_sequence,
            ],
            &projection.query.device(),
        )?;
        if policy
            .bias
            .as_ref()
            .is_some_and(|bias| bias.dtype() != projection.query.dtype())
        {
            return Err(Error::configuration(
                "t5.attention.bias",
                "bias dtype must match attention projections",
            ));
        }
        let explicit_bias = policy.expanded_bias(
            [
                batch,
                self.attention.query_heads,
                query_sequence,
                key_sequence,
            ],
            &projection.query.device(),
        )?;
        let position_bias = match (position_bias, explicit_bias) {
            (Some(relative), Some(extra)) => Some(relative + extra),
            (Some(relative), None) => Some(relative),
            (None, extra) => extra,
        };
        match intervention {
            Some((points, hooks)) => self.attention.attend_with_hooks(
                projection,
                mask,
                position_bias,
                policy.causal,
                points,
                hooks,
            ),
            None => self
                .attention
                .attend(projection, mask, position_bias, policy.causal),
        }
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
            (Some(dense), None, false) => dense.forward(normalized)?,
            (None, Some(gated), true) => gated.forward(normalized)?,
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

/// Per-pass block inputs kept outside the parameter graph.
struct T5BlockContext<'a, B: Backend> {
    /// Valid self-attention token positions in the current stack.
    token_mask: Option<&'a TokenMask<B>>,
    /// Shared relative-position bias broadcast over attention scores.
    position_bias: Tensor<B, 4>,
    /// Encoder hidden states and source-key validity for decoder cross-attention.
    encoder: Option<(&'a Tensor<B, 3>, Option<&'a TokenMask<B>>)>,
    /// Stack and block identity used to address activation hooks.
    layer: Layer,
    /// Optional self-attention policy overriding the block's causal convention.
    attention_policy: Option<AttentionPolicy<B>>,
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

    /// Runs the ordinary block graph with shape-preserving tensor interventions.
    fn forward_with_hooks(
        &self,
        hidden: Tensor<B, 3>,
        token_mask: Option<&TokenMask<B>>,
        position_bias: Tensor<B, 4>,
        encoder: Option<(&Tensor<B, 3>, Option<&TokenMask<B>>)>,
        layer: Layer,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<Tensor<B, 3>> {
        self.forward_context_with_hooks(
            hidden,
            T5BlockContext {
                token_mask,
                position_bias,
                encoder,
                layer,
                attention_policy: None,
            },
            hooks,
        )
    }

    /// Runs a block with context-owned attention behavior and activation hooks.
    fn forward_context_with_hooks(
        &self,
        hidden: Tensor<B, 3>,
        context: T5BlockContext<'_, B>,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<Tensor<B, 3>> {
        let T5BlockContext {
            token_mask,
            position_bias,
            encoder,
            layer,
            attention_policy,
        } = context;
        match (
            encoder.is_some(),
            self.cross_attention.is_some(),
            self.self_attention.causal,
        ) {
            (false, false, false) | (true, true, true) => {}
            _ => {
                return Err(Error::configuration(
                    "t5.block",
                    "encoder/decoder context disagrees with block sublayers",
                ));
            }
        }
        let hidden = hooks.apply(layer.residual_pre(), hidden)?;
        let residual = hidden.clone();
        let normalized = hooks.apply(
            layer.attention_input(),
            self.self_attention.layer_norm.forward(hidden),
        )?;
        let policy = attention_policy.unwrap_or_else(|| {
            if self.self_attention.causal {
                AttentionPolicy::causal()
            } else {
                AttentionPolicy::bidirectional()
            }
        });
        let attended = self.self_attention.self_attention.forward_policy(
            normalized.clone(),
            normalized,
            token_mask,
            Some(position_bias),
            policy,
            Some((layer.self_attention(), hooks)),
        )?;
        let mut hidden = residual + hooks.apply(layer.attention_output(), attended)?;
        if let Some((source, source_mask)) = encoder {
            let cross_attention = self.cross_attention.as_ref().ok_or_else(|| {
                Error::configuration(
                    "t5.decoder.cross_attention",
                    "missing cross-attention module",
                )
            })?;
            let residual = hidden.clone();
            let query = hooks.apply(
                layer.cross_attention_input(),
                cross_attention.layer_norm.forward(hidden),
            )?;
            let attended = cross_attention.enc_dec_attention.forward_impl(
                query,
                source.clone(),
                source_mask,
                None,
                false,
                Some((layer.cross_attention(), hooks)),
            )?;
            hidden = residual + hooks.apply(layer.cross_attention_output(), attended)?;
        }
        let residual = hidden.clone();
        let normalized = hooks.apply(
            layer.mlp_input(),
            self.feed_forward.layer_norm.forward(hidden),
        )?;
        let projected = match (
            &self.feed_forward.dense,
            &self.feed_forward.gated,
            self.feed_forward.projection.is_gated(),
        ) {
            (Some(dense), None, false) => {
                let pre_activation = hooks.apply(
                    layer.mlp_pre_activation(),
                    operations::linear(normalized, &dense.up_proj)?,
                )?;
                let activation = hooks.apply(
                    layer.mlp_activation(),
                    apply_activation(pre_activation, dense.activation),
                )?;
                operations::linear(activation, &dense.down_proj)?
            }
            (None, Some(gated), true) => {
                let pre_activation = hooks.apply(
                    layer.mlp_pre_activation(),
                    operations::linear(normalized.clone(), &gated.gate_proj)?,
                )?;
                let gate = hooks.apply(
                    layer.mlp_activation(),
                    apply_activation(pre_activation, gated.activation),
                )?;
                let content = operations::linear(normalized, &gated.up_proj)?;
                operations::linear(gate * content, &gated.down_proj)?
            }
            _ => {
                return Err(Error::configuration(
                    "t5.feed_forward",
                    "active dense/gated module disagrees with checkpoint projection",
                ));
            }
        };
        let hidden = residual + hooks.apply(layer.mlp_output(), projected)?;
        hooks.apply(layer.residual_post(), hidden)
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

    /// Lists every tensor address for this stack, including embedding boundaries.
    fn activation_points(&self) -> Vec<PointInfo> {
        let stack = if self.is_decoder {
            Stack::Decoder
        } else {
            Stack::Encoder
        };
        let mut points = vec![stack.embedding().info()];
        for (index, block) in self.block.iter().enumerate() {
            let layer = if self.is_decoder {
                Layer::decoder(index)
            } else {
                Layer::encoder(index)
            };
            points.extend(block.activation_points(layer));
        }
        points.push(stack.final_norm().info());
        points
    }

    /// Executes an embedded stack while preserving T5's shared relative bias.
    fn forward_with_hooks(
        &self,
        hidden: Tensor<B, 3>,
        token_mask: Option<&TokenMask<B>>,
        encoder: Option<(&Tensor<B, 3>, Option<&TokenMask<B>>)>,
        retain_hidden_states: bool,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<T5StackOutput<B>> {
        if self.is_decoder != encoder.is_some() {
            return Err(Error::configuration(
                "t5.stack",
                "only decoder stacks accept and require encoder states",
            ));
        }
        let stack = if self.is_decoder {
            Stack::Decoder
        } else {
            Stack::Encoder
        };
        let mut hidden = hooks.apply(stack.embedding(), hidden)?;
        let [_, sequence, _] = hidden.dims();
        let position_bias = if self.is_decoder {
            if let Some(mask) = token_mask {
                let positions = padded_hooks::logical_decoder_positions(mask)?;
                let table = self
                    .block
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
                            "t5.decoder.relative_bias",
                            "missing shared relative-position table",
                        )
                    })?;
                recipe::relative_bias_for_positions(table, &positions)?
            } else {
                self.position_bias(sequence, sequence, &hidden.device())?
            }
        } else {
            self.position_bias(sequence, sequence, &hidden.device())?
        };
        let mut hidden_states = diagnostic_buffer(self.block.len(), retain_hidden_states);
        if retain_hidden_states {
            hidden_states.push(hidden.clone());
        }
        for (index, block) in self.block.iter().enumerate() {
            let layer = if self.is_decoder {
                Layer::decoder(index)
            } else {
                Layer::encoder(index)
            };
            hidden = block.forward_with_hooks(
                hidden,
                token_mask,
                position_bias.clone(),
                encoder,
                layer,
                hooks,
            )?;
            if retain_hidden_states {
                hidden_states.push(hidden.clone());
            }
        }
        hidden = hooks.apply(stack.final_norm(), self.final_layer_norm.forward(hidden))?;
        if retain_hidden_states {
            hidden_states.push(hidden.clone());
        }
        Ok(T5StackOutput {
            hidden,
            hidden_states,
        })
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
        validate_input_devices(&source, "encoder", &model_device)?;
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
            let decoder_input = self.prepare_hooked_decoder_input(input.decoder)?;
            let encoder = self.encode_hooked(input.encoder, options, hooks)?;
            let decoder = self.decode_stateless_hooked(&encoder, decoder_input, options, hooks)?;
            Ok(Seq2SeqOutput {
                decoder,
                encoder,
                cross_attentions: Vec::new(),
            })
        })
    }

    /// Implements encoder execution inside an already validated hook session.
    fn encode_hooked(
        &self,
        source: CausalLmInput<B>,
        options: &ForwardOptions,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<EncoderOutput<B>> {
        validate_options(options)?;
        let model_device = self.shared.weight.lazy_device();
        validate_input_devices(&source, "encoder", &model_device)?;
        validate_token_input(&source, self.config.vocab_size, "encoder")?;
        let hidden = self.shared.forward(source.token_ids.clone());
        let output = self.encoder.forward_with_hooks(
            hidden,
            source.token_mask.as_ref(),
            None,
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

    /// Validates target inputs before any user callback can execute.
    fn prepare_hooked_decoder_input(
        &self,
        mut decoder: CausalLmInput<B>,
    ) -> Result<CausalLmInput<B>> {
        let model_device = self.shared.weight.lazy_device();
        validate_input_devices(&decoder, "decoder", &model_device)?;
        validate_token_input(&decoder, self.config.vocab_size, "decoder")?;
        if let Some(mask) = &decoder.token_mask {
            // Tokenizers commonly emit an all-valid mask for an unpadded batch.
            // Validate it above before normalization so malformed shapes/devices
            // retain the same contextual errors as ordinary decoder execution.
            let all_valid = mask
                .as_tensor()
                .clone()
                .all()
                .try_into_scalar()
                .map_err(|error| {
                    Error::BackendCapability(format!(
                        "failed to validate T5 decoder token mask for hooks: {error}"
                    ))
                })?
                .elem::<bool>();
            if all_valid {
                decoder.token_mask = None;
            }
        }
        Ok(decoder)
    }

    /// Implements rectangular decoder execution inside the same hook session.
    fn decode_stateless_hooked(
        &self,
        encoder: &EncoderOutput<B>,
        decoder: CausalLmInput<B>,
        options: &ForwardOptions,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<CausalLmOutput<B>> {
        validate_options(options)?;
        let model_device = self.shared.weight.lazy_device();
        validate_encoder_devices(encoder, &model_device)?;
        let batch = decoder.token_ids.dims()[0];
        validate_encoder_output(
            encoder,
            batch,
            self.config.d_model,
            self.shared.weight.val().dtype(),
        )?;
        let hidden = self.shared.forward(decoder.token_ids.clone());
        let output = self.decoder.forward_with_hooks(
            hidden,
            decoder.token_mask.as_ref(),
            Some((&encoder.hidden_state, encoder.token_mask.as_ref())),
            options.hidden_states,
            hooks,
        )?;
        let selected = select_hidden(
            output.hidden.clone(),
            decoder.token_mask.as_ref(),
            &options.logits,
        )?;
        Ok(CausalLmOutput {
            logits: self.project_logits(selected)?,
            final_hidden_state: options.hidden_states.then_some(output.hidden),
            hidden_states: output.hidden_states,
            attentions: Vec::new(),
            routers: Vec::new(),
            load_balancing_loss: None,
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
        let model_device = self.shared.weight.lazy_device();
        validate_input_devices(&decoder, "decoder", &model_device)?;
        validate_encoder_devices(encoder, &model_device)?;
        let [batch, sequence] = validate_token_input(&decoder, self.config.vocab_size, "decoder")?;
        validate_encoder_output(
            encoder,
            batch,
            self.config.d_model,
            self.shared.weight.val().dtype(),
        )?;
        if decoder.token_mask.is_some() {
            return self.decode_stateless_masked(encoder, decoder, options);
        }
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
            load_balancing_loss: None,
        })
    }

    /// Applies the tied/scaled or independent/unscaled vocabulary projection.
    fn project_logits(&self, hidden: Tensor<B, 3>) -> Result<Tensor<B, 3>> {
        let hidden_device = hidden.device();
        if self.config.tie_word_embeddings {
            if self.lm_head.is_some() {
                return Err(Error::configuration(
                    "t5.lm_head",
                    "tied checkpoints must not own a separate vocabulary projection",
                ));
            }
            ensure_model_device(
                "t5.shared",
                &self.shared.weight.lazy_device(),
                &hidden_device,
            )?;
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
            .and_then(|head| {
                ensure_model_device("t5.lm_head", &head.weight.lazy_device(), &hidden_device)?;
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

/// Rejects stateless token tensors that do not share the model parameter device.
///
/// T5 rejects explicit absolute positions semantically, but their device is
/// still checked first so every independently supplied tensor obeys one stable
/// affinity contract.
fn validate_input_devices<B: Backend>(
    input: &CausalLmInput<B>,
    stack: &'static str,
    model_device: &B::Device,
) -> Result<()> {
    ensure_model_device(
        &format!("t5.{stack}.input_ids"),
        &input.token_ids.device(),
        model_device,
    )?;
    if let Some(mask) = &input.token_mask {
        ensure_model_device(
            &format!("t5.{stack}.token_mask"),
            &mask.as_tensor().device(),
            model_device,
        )?;
    }
    if let Some(positions) = &input.position_ids {
        ensure_model_device(
            &format!("t5.{stack}.position_ids"),
            &positions.device(),
            model_device,
        )?;
    }
    Ok(())
}

/// Validates the cross-attention tensors of an externally supplied encoder result.
fn validate_encoder_devices<B: Backend>(
    encoder: &EncoderOutput<B>,
    model_device: &B::Device,
) -> Result<()> {
    ensure_model_device(
        "t5.encoder_output.hidden_state",
        &encoder.hidden_state.device(),
        model_device,
    )?;
    if let Some(mask) = &encoder.token_mask {
        ensure_model_device(
            "t5.encoder_output.token_mask",
            &mask.as_tensor().device(),
            model_device,
        )?;
    }
    Ok(())
}

/// Reports one stateless tensor/parameter device mismatch contextually.
fn ensure_model_device<D: PartialEq>(tensor: &str, actual: &D, expected: &D) -> Result<()> {
    if actual != expected {
        return Err(Error::BackendCapability(format!(
            "{tensor} and T5 model parameters are on different devices"
        )));
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

#[cfg(test)]
#[path = "model_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "model/intervention_tests.rs"]
mod intervention_tests;
