//! The executable Gemma 2 / Gemma 3 text decoder graph.
//!
//! # The block
//!
//! A Gemma block normalizes both the input and the output of each residual
//! sublayer ("sandwich" norms), with zero-centered RMSNorm
//! `N(x) = (1 + w) ⊙ x · rsqrt(mean(x²) + ε)` computed in F32:
//!
//! ```text
//! a  = Attn(N_input(x))                 Q/K RMSNorm (Gemma 3), rotation, attention
//! x' = x + N_post_attention(a)
//! m  = GeGLU(N_pre_feedforward(x'))     GeGLU(u) = down(gelu_tanh(gate u) ⊙ up u)
//! y  = x' + N_post_feedforward(m)
//! ```
//!
//! Attention scores are `softcap(q·kᵀ · query_pre_attn_scalar^(-1/2))` with
//! `softcap(s) = c · tanh(s / c)` on Gemma 2 and the identity on Gemma 3;
//! sliding layers see the `W` newest positions (`p - W < j <= p`), global
//! layers every earlier one. Every layer rotates queries and keys; Gemma 3's
//! sliding and global layers use different RoPE sets, which the graph's
//! rotary schedule assigns per layer.
//!
//! # The decoder
//!
//! ```text
//! h_0     = Embed(tokens) · round_dtype(sqrt(d))
//! h_{l+1} = Block_l(h_l)
//! logits  = softcap_final(N_final(h_L) · Wᵀ)       W = lm_head, or Embed when tied
//! ```
//!
//! [`GemmaForCausalLm`] implements the crate-private dense-decoder interface
//! of the Llama family's execution engine (`DecoderGraph`), so the stateless
//! pass, activation hooks, staged execution, and the genuinely batched
//! cached prefill/decode are the very code paths the Llama family runs,
//! including ring caches for sliding layers below the cache capacity; only
//! the embedding scale, the block above, and the soft-capped head are
//! Gemma's own.
//!
//! # Tensor axes and precision
//!
//! Tokens and positions are `[batch, token]`, residuals `[batch, token, d]`,
//! queries `[batch, H, token, h]`, keys and values keep their native
//! `[batch, K, token, h]` heads, and logits are `[batch, selected, V]`. The
//! embedding multiplier is rounded to the activation dtype exactly as the
//! references round it; norms reduce and scale in F32 and round once;
//! rotary tables are built once per call per RoPE set in F32 and rounded to
//! the activation dtype; soft-caps run in the activation dtype. Attention
//! uses the backend's fused kernels (with a native soft-cap) unless
//! [`GemmaForCausalLm::with_attention_precision`] selects the eager
//! F32-softmax graph, which also exposes score and pattern hooks.
//!
//! # Padding
//!
//! As in the Llama family: padding never consumes a position, padded keys
//! are excluded from every query, padded queries still attend (so no score
//! row is empty), hooked passes zero padded entries at every point, and
//! cached passes return exact zeros at padded columns.

use burn::{
    module::Module,
    nn::{Embedding, EmbeddingConfig, Linear, LinearConfig, LinearLayout},
    tensor::{Int, Tensor, backend::Backend},
};

use crate::{
    CausalLmInput, CausalLmOutput, ForwardOptions, Result,
    attention::{AttentionConnectivity, AttentionPrecisionMode, GroupedQueryAttention},
    feed_forward::GatedFeedForward,
    intervention::{Hooks, Layer, PointInfo},
    models::{
        llama::model::{
            engine,
            graph::{
                DecoderBlock, DecoderGeometry, DecoderGraph, GraphLabels, LayerKeys,
                RotarySchedule, missing_rotary_tables, rotate, vocabulary_projection,
            },
        },
        shared::InputNames,
    },
    normalization::RmsNorm,
    operations,
};

use super::architecture::{GemmaArchitecture, GemmaLayerSchedule};
use super::config::GemmaConfig;

mod decode;
mod experiment;
#[cfg(test)]
pub(super) mod test_support;

pub use decode::GemmaDecodeState;

/// Names the Gemma family uses in errors raised by the shared execution paths.
pub(super) const LABELS: GraphLabels = GraphLabels {
    family: "Gemma",
    prefix: "gemma",
    inputs: InputNames {
        token_ids: "gemma.input_ids",
        token_mask: "gemma.token_mask",
        position_ids: "gemma.position_ids",
    },
    parameters: "Gemma model parameters",
    state: "Gemma decode state",
};

// The block.

/// One sandwich-normalized Gemma decoder block.
///
/// Field names are the reference's, so the checkpoint contract maps every
/// norm by identity.
#[derive(Module, Debug)]
pub struct GemmaDecoderLayer<B: Backend> {
    /// Grouped-query self-attention with explicit scale, optional soft-cap,
    /// and (Gemma 3) zero-centered per-head Q/K norms.
    pub self_attn: GroupedQueryAttention<B>,
    /// GeGLU feed-forward network.
    pub mlp: GatedFeedForward<B>,
    /// Norm of the attention input.
    pub input_layernorm: RmsNorm<B>,
    /// Norm of the attention output, before the residual addition.
    pub post_attention_layernorm: RmsNorm<B>,
    /// Norm of the MLP input.
    pub pre_feedforward_layernorm: RmsNorm<B>,
    /// Norm of the MLP output, before the residual addition.
    pub post_feedforward_layernorm: RmsNorm<B>,
    /// Attention kind and key visibility of this layer.
    #[module(skip)]
    pub schedule: GemmaLayerSchedule,
}

impl<B: Backend> GemmaDecoderLayer<B> {
    /// Allocates block `index` of a validated architecture.
    fn init(architecture: &GemmaArchitecture, index: usize, device: &B::Device) -> Result<Self> {
        let mut self_attn = architecture.attention_config().init(device)?;
        // The shared attention builder uses default RMSNorm conventions;
        // Gemma 3's Q/K norms follow the family's F32 scale-then-cast rule.
        let family_norm = architecture.rms_norm(1);
        for norm in [&mut self_attn.query_norm, &mut self_attn.key_norm]
            .into_iter()
            .flatten()
        {
            norm.round_before_scale = family_norm.round_before_scale;
            norm.root_mode = family_norm.root_mode;
        }
        let norm = || architecture.rms_norm(architecture.hidden_size).init(device);
        Ok(Self {
            self_attn,
            mlp: architecture.mlp_config().init(device)?,
            input_layernorm: norm()?,
            post_attention_layernorm: norm()?,
            pre_feedforward_layernorm: norm()?,
            post_feedforward_layernorm: norm()?,
            schedule: architecture.layers[index],
        })
    }

    /// Computes the attention sublayer's contribution to the residual stream.
    fn attention_branch(
        &self,
        residual: Tensor<B, 3>,
        index: usize,
        keys: &mut impl LayerKeys<B>,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<Tensor<B, 3>> {
        let layer = Layer::decoder(index);
        let input = hooks.apply(
            layer.attention_input(),
            self.input_layernorm.forward(residual),
        )?;
        let mut projection = self.self_attn.project_self(input)?;
        let angles = keys
            .angles(index)
            .ok_or_else(|| missing_rotary_tables(&LABELS))?;
        rotate(&mut projection, angles)?;
        let output = keys.attend(
            index,
            &self.self_attn,
            projection,
            layer.self_attention(),
            hooks,
        )?;
        hooks.apply(
            layer.attention_output(),
            self.post_attention_layernorm.forward(output),
        )
    }

    /// Computes the MLP sublayer's contribution to the residual stream.
    fn mlp_branch(
        &self,
        hidden: Tensor<B, 3>,
        layer: Layer,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<Tensor<B, 3>> {
        let input = hooks.apply(
            layer.mlp_input(),
            self.pre_feedforward_layernorm.forward(hidden),
        )?;
        let output = self.mlp.forward_with_hooks(input, layer, hooks)?;
        hooks.apply(
            layer.mlp_output(),
            self.post_feedforward_layernorm.forward(output),
        )
    }

    /// Lists the points this block emits as block `index`, in emission order.
    ///
    /// `attention.input` and `mlp.input` follow the pre-norms;
    /// `attention.output` and `mlp.output` follow the post-norms, i.e. they
    /// are exactly the branches added to the residual. Score and pattern
    /// points exist only under [`AttentionPrecisionMode::EagerFp32Softmax`].
    #[must_use]
    pub fn activation_points(&self, index: usize) -> Vec<PointInfo> {
        self.block_points(index)
    }
}

impl<B: Backend> DecoderBlock<B> for GemmaDecoderLayer<B> {
    /// Runs the sandwich-normalized block equation of the module documentation.
    fn forward<K: LayerKeys<B>>(
        &self,
        hidden: Tensor<B, 3>,
        index: usize,
        keys: &mut K,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<Tensor<B, 3>> {
        let layer = Layer::decoder(index);
        let residual = hooks.apply(layer.residual_pre(), hidden)?;
        let attention = self.attention_branch(residual.clone(), index, keys, hooks)?;
        let hidden = residual + attention;
        let mlp = self.mlp_branch(hidden.clone(), layer, hooks)?;
        hooks.apply(layer.residual_post(), hidden + mlp)
    }

    /// Returns the layer's global or sliding-window rule.
    fn connectivity(&self) -> AttentionConnectivity {
        self.schedule.connectivity
    }

    /// Returns the attention operation graph selected for this layer.
    fn attention_precision(&self) -> AttentionPrecisionMode {
        self.self_attn.precision_mode
    }
}

// The backbone.

/// Token embedding, decoder blocks, and final RMSNorm.
#[derive(Module, Debug)]
pub struct GemmaModel<B: Backend> {
    /// Token embedding table `[V, d]`, scaled by `sqrt(d)` on lookup.
    pub embed_tokens: Embedding<B>,
    /// Decoder blocks in order.
    pub layers: Vec<GemmaDecoderLayer<B>>,
    /// Final zero-centered RMSNorm.
    pub norm: RmsNorm<B>,
    /// Checkpoint configuration, excluded from parameter records.
    #[module(skip)]
    pub config: GemmaConfig,
    /// Validated executable topology derived from `config`.
    #[module(skip)]
    pub architecture: GemmaArchitecture,
}

impl<B: Backend> GemmaModel<B> {
    /// Allocates a randomly initialized backbone for a validated configuration.
    fn init(config: GemmaConfig, device: &B::Device) -> Result<Self> {
        let architecture = config.architecture()?;
        let layers = (0..architecture.num_layers())
            .map(|index| GemmaDecoderLayer::init(&architecture, index, device))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            embed_tokens: EmbeddingConfig::new(architecture.vocab_size, architecture.hidden_size)
                .init(device),
            layers,
            norm: architecture
                .rms_norm(architecture.hidden_size)
                .init(device)?,
            config,
            architecture,
        })
    }
}

// The causal language model.

/// A Gemma 2 or Gemma 3 text decoder-only language model.
#[derive(Module, Debug)]
pub struct GemmaForCausalLm<B: Backend> {
    /// Embedding, blocks, and final norm.
    pub model: GemmaModel<B>,
    /// Independent vocabulary projection `[V, d]`; `None` when tied to the embedding.
    pub lm_head: Option<Linear<B>>,
}

impl<B: Backend> GemmaForCausalLm<B> {
    /// Validates `config` and allocates a randomly initialized model.
    ///
    /// Norm weights start at zero (scale one). Load weights with
    /// [`Self::load_checkpoint`] or the automatic registry.
    ///
    /// # Errors
    ///
    /// Returns the configuration's validation error or a parameter
    /// allocation error.
    pub fn init(config: GemmaConfig, device: &B::Device) -> Result<Self> {
        let model = GemmaModel::init(config, device)?;
        let architecture = &model.architecture;
        let lm_head = (!architecture.tie_word_embeddings).then(|| {
            operations::init_linear(
                &LinearConfig::new(architecture.hidden_size, architecture.vocab_size)
                    .with_bias(false)
                    .with_layout(LinearLayout::Col),
                device,
            )
        });
        Ok(Self { model, lm_head })
    }

    /// Returns the checkpoint configuration.
    #[must_use]
    pub const fn config(&self) -> &GemmaConfig {
        &self.model.config
    }

    /// Returns the validated executable topology.
    #[must_use]
    pub const fn architecture(&self) -> &GemmaArchitecture {
        &self.model.architecture
    }

    /// Selects the attention operation graph of every layer.
    ///
    /// [`AttentionPrecisionMode::EagerFp32Softmax`] materializes scores
    /// (soft-capped before exclusion), runs softmax in F32 as the eager
    /// reference does, and exposes the score and pattern hook points and
    /// [`ForwardOptions::attentions`]; the default backend graph uses fused
    /// kernels.
    #[must_use]
    pub fn with_attention_precision(mut self, mode: AttentionPrecisionMode) -> Self {
        for layer in &mut self.model.layers {
            layer.self_attn.precision_mode = mode;
        }
        self
    }

    /// Evaluates the stateless, differentiable forward pass.
    ///
    /// Implicit positions are valid-token ordinals, so left padding does not
    /// shift rotary phases. `options.attentions` requires
    /// [`AttentionPrecisionMode::EagerFp32Softmax`]; router outputs are
    /// empty because every block is dense. `hidden_states` holds the scaled
    /// embedding (the input of layer 0), every block output, and the
    /// final-normalized states.
    ///
    /// # Errors
    ///
    /// Returns input validation errors (see `models::shared`), or a
    /// capability error for attention capture under fused kernels.
    pub fn forward(
        &self,
        input: CausalLmInput<B>,
        options: &ForwardOptions,
    ) -> Result<CausalLmOutput<B>> {
        engine::forward(self, input, options)
    }
}

impl<B: Backend> DecoderGraph<B> for GemmaForCausalLm<B> {
    /// Sandwich-normalized Gemma blocks.
    type Block = GemmaDecoderLayer<B>;

    /// `gemma.*` error fields and `Gemma` messages.
    const LABELS: GraphLabels = LABELS;

    /// Reads the geometry of the validated architecture.
    fn geometry(&self) -> DecoderGeometry {
        let architecture = self.architecture();
        DecoderGeometry {
            vocab_size: architecture.vocab_size,
            hidden_size: architecture.hidden_size,
            query_heads: architecture.query_heads,
            key_value_heads: architecture.key_value_heads,
            head_dim: architecture.head_dim,
            max_position_embeddings: architecture.max_position_embeddings,
        }
    }

    /// Sliding layers rotate with the local set, global layers with the global set.
    fn rotary_schedule(&self) -> RotarySchedule {
        self.architecture().rotary_schedule()
    }

    /// Returns the decoder blocks.
    fn blocks(&self) -> &[GemmaDecoderLayer<B>] {
        &self.model.layers
    }

    /// Returns the token embedding table.
    fn embedding(&self) -> &Embedding<B> {
        &self.model.embed_tokens
    }

    /// Looks up embeddings and multiplies them by `sqrt(d)` rounded to their dtype.
    fn embed(&self, token_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let embeddings = self.model.embed_tokens.forward(token_ids);
        let scale = self.architecture().embedding_scale(embeddings.dtype());
        embeddings.mul_scalar(scale)
    }

    /// Applies the final zero-centered RMSNorm.
    fn final_norm(&self, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
        self.model.norm.forward(hidden)
    }

    /// Applies the tied or independent head, then the optional final soft-cap.
    fn project_logits(&self, hidden: Tensor<B, 3>) -> Result<Tensor<B, 3>> {
        let logits = vocabulary_projection(
            &LABELS,
            hidden,
            self.lm_head.as_ref(),
            &self.model.embed_tokens,
        )?;
        Ok(match self.architecture().final_logit_softcap {
            Some(cap) => logits.div_scalar(cap).tanh().mul_scalar(cap),
            None => logits,
        })
    }
}

#[cfg(test)]
#[path = "model_tests.rs"]
mod tests;
