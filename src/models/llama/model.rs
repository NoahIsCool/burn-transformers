//! The executable Llama-family decoder graph.
//!
//! # One block for every path
//!
//! A block (see the [`architecture`](super::LlamaArchitecture) translation
//! table) is two residual sublayers whose normalization is either applied to
//! the sublayer *input* (`x + F(N(x))`, every family but OLMo 2) or to its
//! *output* (`x + N(F(x))`, OLMo 2):
//!
//! ```text
//! a = Attn(x)            queries/keys rotated unless the layer is NoPE
//! x' = x + branch(a)
//! y  = x' + branch(SwiGLU(x'))      SwiGLU(u) = down(silu(gate u) ⊙ up u)
//! ```
//!
//! The block's `DecoderBlock::forward` is the only implementation of that
//! equation. The stateless pass, the hooked pass, staged execution, and the
//! batched cached step differ only in where attention finds its keys, which
//! the crate-private `LayerKeys` trait abstracts: the `sequence` submodule
//! attends over the input's own tokens with causal/sliding/padding masks, and
//! the `decode` submodule attends over a decode cache written ahead or read
//! through per-step plans.
//!
//! # A shared execution engine
//!
//! Those paths are written once against the crate-private `DecoderGraph`
//! interface of the `graph` submodule, never against Llama types, so other
//! dense decoders reuse them: [`LlamaForCausalLm`] implements the interface
//! with the blocks above, and the Gemma family
//! ([`crate::models::gemma`]) implements it with sandwich-normalized,
//! soft-capped blocks and per-layer-type rotary tables, inheriting the
//! stateless, hooked, staged, and genuinely batched cached paths unchanged.
//!
//! # Tensor axes and precision
//!
//! Tokens and positions are `[batch, token]`, residuals `[batch, token, d]`,
//! queries `[batch, H, token, h]`, keys and values keep their native
//! `[batch, K, token, h]` heads, and logits are `[batch, selected, V]`.
//! RMSNorm reduces in F32 and rounds as the reference does
//! ([`NormRounding`](super::NormRounding)); rotary tables are built once per
//! call in F32 and converted once to the rotation dtype
//! ([`RotaryPrecision`](super::RotaryPrecision)). Attention uses the backend's
//! fused kernels unless [`LlamaForCausalLm::with_attention_precision`] selects
//! the eager F32-softmax graph, which also exposes score and pattern hooks.
//!
//! # Padding
//!
//! Padding never consumes a position: implicit positions are valid-token
//! ordinals, and a padded key is excluded from every query. A padded *query*
//! still attends (to the valid keys up to the first ordinal), so no score row
//! is ever fully excluded and no backend's convention for empty softmax rows
//! matters. Values at padded positions are therefore finite but unspecified;
//! hooked passes zero them at every activation point (see
//! [`LlamaForCausalLm::forward_with_hooks`]), and cached passes return exact
//! zeros there.

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
    models::shared::InputNames,
    normalization::RmsNorm,
    operations,
};

use super::architecture::{LlamaArchitecture, LlamaLayerSchedule, SublayerNorm};
use super::config::LlamaFamilyConfig;

mod decode;
mod experiment;
pub(crate) mod graph;
mod sequence;
mod step;
#[cfg(test)]
pub(super) mod test_support;

pub use decode::LlamaDecodeState;

pub(crate) mod engine {
    //! The execution entry points shared by every `DecoderGraph` (see [`super::graph`]).
    //!
    //! A family whose model implements the graph interface gets its
    //! stateless, hooked, staged, and batched cached passes by calling these
    //! functions with itself; the Llama family's own inherent methods do
    //! exactly that.

    pub(crate) use super::decode::{Phase, forward_cached};
    pub(crate) use super::experiment::{
        activation_points, embed_with_hooks, forward, forward_with_hooks, normalize_with_hooks,
        run_layers_with_hooks,
    };
}
use graph::{
    DecoderBlock, DecoderGeometry, DecoderGraph, GraphLabels, LayerKeys, RotarySchedule,
    missing_rotary_tables, rotate, vocabulary_projection,
};

/// Names the Llama family uses in errors raised by the shared execution paths.
const LABELS: GraphLabels = GraphLabels {
    family: "Llama-family",
    prefix: "llama",
    inputs: InputNames {
        token_ids: "llama.input_ids",
        token_mask: "llama.token_mask",
        position_ids: "llama.position_ids",
    },
    parameters: "Llama-family model parameters",
    state: "Llama-family decode state",
};

// The block.

/// One Llama-family decoder block.
///
/// Norms are named by role, not by checkpoint position: `attention_norm`
/// wraps the attention sublayer and `mlp_norm` the MLP sublayer, as an input
/// norm or an output norm according to `sublayer_norm`.
#[derive(Module, Debug)]
pub struct LlamaDecoderLayer<B: Backend> {
    /// Grouped-query self-attention, with optional biases and Q/K norms.
    pub self_attn: GroupedQueryAttention<B>,
    /// SwiGLU feed-forward network.
    pub mlp: GatedFeedForward<B>,
    /// RMSNorm of the attention sublayer.
    pub attention_norm: RmsNorm<B>,
    /// RMSNorm of the MLP sublayer.
    pub mlp_norm: RmsNorm<B>,
    /// Rotary and key-visibility schedule of this layer.
    #[module(skip)]
    pub schedule: LlamaLayerSchedule,
    /// Whether the norms wrap sublayer inputs or outputs.
    #[module(skip)]
    pub sublayer_norm: SublayerNorm,
}

impl<B: Backend> LlamaDecoderLayer<B> {
    /// Allocates block `index` of a validated architecture.
    fn init(architecture: &LlamaArchitecture, index: usize, device: &B::Device) -> Result<Self> {
        let mut self_attn = architecture.attention_config().init(device)?;
        // The shared attention builder uses default RMSNorm conventions;
        // Q/K norms follow the family's rounding and root operation instead.
        let family_norm = architecture.rms_norm(1);
        for norm in [&mut self_attn.query_norm, &mut self_attn.key_norm]
            .into_iter()
            .flatten()
        {
            norm.round_before_scale = family_norm.round_before_scale;
            norm.root_mode = family_norm.root_mode;
        }
        Ok(Self {
            self_attn,
            mlp: architecture.mlp_config().init(device)?,
            attention_norm: architecture
                .rms_norm(architecture.hidden_size)
                .init(device)?,
            mlp_norm: architecture
                .rms_norm(architecture.hidden_size)
                .init(device)?,
            schedule: architecture.layers[index],
            sublayer_norm: architecture.sublayer_norm,
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
        let input = self.input_norm(&self.attention_norm, residual);
        let input = hooks.apply(layer.attention_input(), input)?;
        let mut projection = self.self_attn.project_self(input)?;
        if self.schedule.rotary {
            let angles = keys
                .angles(index)
                .ok_or_else(|| missing_rotary_tables(&LABELS))?;
            rotate(&mut projection, angles)?;
        }
        let output = keys.attend(
            index,
            &self.self_attn,
            projection,
            layer.self_attention(),
            hooks,
        )?;
        let output = self.output_norm(&self.attention_norm, output);
        hooks.apply(layer.attention_output(), output)
    }

    /// Computes the MLP sublayer's contribution to the residual stream.
    fn mlp_branch(
        &self,
        hidden: Tensor<B, 3>,
        layer: Layer,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<Tensor<B, 3>> {
        let input = self.input_norm(&self.mlp_norm, hidden);
        let input = hooks.apply(layer.mlp_input(), input)?;
        let output = self.mlp.forward_with_hooks(input, layer, hooks)?;
        let output = self.output_norm(&self.mlp_norm, output);
        hooks.apply(layer.mlp_output(), output)
    }

    /// Applies `norm` to a sublayer input in input-normalized blocks.
    fn input_norm(&self, norm: &RmsNorm<B>, input: Tensor<B, 3>) -> Tensor<B, 3> {
        match self.sublayer_norm {
            SublayerNorm::Input => norm.forward(input),
            SublayerNorm::Output => input,
        }
    }

    /// Applies `norm` to a sublayer output in output-normalized blocks.
    fn output_norm(&self, norm: &RmsNorm<B>, output: Tensor<B, 3>) -> Tensor<B, 3> {
        match self.sublayer_norm {
            SublayerNorm::Input => output,
            SublayerNorm::Output => norm.forward(output),
        }
    }

    /// Lists the points this block emits as block `index`, in emission order.
    ///
    /// `attention.output` and `mlp.output` are the branches added to the
    /// residual, i.e. after OLMo 2's output norms. Score and pattern points
    /// exist only under [`AttentionPrecisionMode::EagerFp32Softmax`].
    #[must_use]
    pub fn activation_points(&self, index: usize) -> Vec<PointInfo> {
        self.block_points(index)
    }
}

impl<B: Backend> DecoderBlock<B> for LlamaDecoderLayer<B> {
    /// Runs the input- or output-normalized block equation of the module documentation.
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
pub struct LlamaModel<B: Backend> {
    /// Token embedding table `[V, d]`.
    pub embed_tokens: Embedding<B>,
    /// Decoder blocks in order.
    pub layers: Vec<LlamaDecoderLayer<B>>,
    /// Final RMSNorm.
    pub norm: RmsNorm<B>,
    /// Checkpoint configuration, excluded from parameter records.
    #[module(skip)]
    pub config: LlamaFamilyConfig,
    /// Validated executable topology derived from `config`.
    #[module(skip)]
    pub architecture: LlamaArchitecture,
}

impl<B: Backend> LlamaModel<B> {
    /// Allocates a randomly initialized backbone for a validated configuration.
    fn init(config: LlamaFamilyConfig, device: &B::Device) -> Result<Self> {
        let architecture = config.architecture()?;
        let layers = (0..architecture.num_layers())
            .map(|index| LlamaDecoderLayer::init(&architecture, index, device))
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

/// A Llama-family decoder-only language model.
#[derive(Module, Debug)]
pub struct LlamaForCausalLm<B: Backend> {
    /// Embedding, blocks, and final norm.
    pub model: LlamaModel<B>,
    /// Independent vocabulary projection `[V, d]`; `None` when tied to the embedding.
    pub lm_head: Option<Linear<B>>,
}

impl<B: Backend> LlamaForCausalLm<B> {
    /// Validates `config` and allocates a randomly initialized model.
    ///
    /// Load weights with [`Self::load_checkpoint`] or the automatic registry.
    ///
    /// # Errors
    ///
    /// Returns the configuration's validation error or a parameter
    /// allocation error.
    pub fn init(config: LlamaFamilyConfig, device: &B::Device) -> Result<Self> {
        let model = LlamaModel::init(config, device)?;
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
    pub const fn config(&self) -> &LlamaFamilyConfig {
        &self.model.config
    }

    /// Returns the validated executable topology.
    #[must_use]
    pub const fn architecture(&self) -> &LlamaArchitecture {
        &self.model.architecture
    }

    /// Selects the attention operation graph of every layer.
    ///
    /// [`AttentionPrecisionMode::EagerFp32Softmax`] materializes scores, runs
    /// softmax in F32 as the eager reference does, and exposes the score and
    /// pattern hook points and [`ForwardOptions::attentions`]; the default
    /// backend graph uses fused kernels.
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
    /// [`AttentionPrecisionMode::EagerFp32Softmax`]; router outputs are empty
    /// because every block is dense.
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
        experiment::forward(self, input, options)
    }
}

impl<B: Backend> DecoderGraph<B> for LlamaForCausalLm<B> {
    /// Llama blocks with input or output sublayer norms.
    type Block = LlamaDecoderLayer<B>;

    /// `llama.*` error fields and `Llama-family` messages.
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

    /// One parameter set shared by every rotary layer; NoPE layers use none.
    fn rotary_schedule(&self) -> RotarySchedule {
        let architecture = self.architecture();
        RotarySchedule {
            sets: vec![architecture.rope],
            layers: architecture
                .layers
                .iter()
                .map(|layer| layer.rotary.then_some(0))
                .collect(),
            precision: architecture.rotary_precision,
        }
    }

    /// Returns the decoder blocks.
    fn blocks(&self) -> &[LlamaDecoderLayer<B>] {
        &self.model.layers
    }

    /// Returns the token embedding table.
    fn embedding(&self) -> &Embedding<B> {
        &self.model.embed_tokens
    }

    /// Looks up token embeddings without scaling.
    fn embed(&self, token_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        self.model.embed_tokens.forward(token_ids)
    }

    /// Applies the final RMSNorm.
    fn final_norm(&self, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
        self.model.norm.forward(hidden)
    }

    /// Applies the independent or tied vocabulary projection.
    fn project_logits(&self, hidden: Tensor<B, 3>) -> Result<Tensor<B, 3>> {
        vocabulary_projection(
            &LABELS,
            hidden,
            self.lm_head.as_ref(),
            &self.model.embed_tokens,
        )
    }
}

#[cfg(test)]
#[path = "model_tests.rs"]
mod tests;
