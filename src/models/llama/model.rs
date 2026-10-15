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
//! `LlamaDecoderLayer::forward` is the only implementation of that
//! equation. The stateless pass, the hooked pass, staged execution, and the
//! batched cached step differ only in where attention finds its keys, which
//! the private `LayerKeys` trait abstracts: the `sequence` submodule attends
//! over the input's own tokens with causal/sliding/padding masks, and the
//! `decode` submodule attends over a decode cache written ahead or read
//! through per-step plans.
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
    tensor::{DType, Int, Tensor, backend::Backend},
};

use crate::{
    CausalLmInput, CausalLmOutput, Error, ForwardOptions, Result,
    attention::{AttentionPrecisionMode, AttentionProjection, GroupedQueryAttention},
    feed_forward::{GatedFeedForward, mlp_hook_points},
    intervention::{AttentionPoints, Hooks, Layer, PointInfo, Stack},
    normalization::RmsNorm,
    operations,
    position::{RotaryAngles, RotaryFrequencies, apply_rotary, rotary_angles_from_frequencies},
};

use super::architecture::{LlamaArchitecture, LlamaLayerSchedule, SublayerNorm};
use super::config::LlamaFamilyConfig;
use crate::models::shared::{
    InputContract, InputNames, ensure_same_device, select_hidden, validate_input,
};

mod decode;
mod experiment;
mod sequence;
mod step;
#[cfg(test)]
pub(super) mod test_support;

pub use decode::LlamaDecodeState;
use sequence::SequenceKeys;

/// Tensor names reported by input errors.
pub(super) const INPUT_NAMES: InputNames<'static> = InputNames {
    token_ids: "llama.input_ids",
    token_mask: "llama.token_mask",
    position_ids: "llama.position_ids",
};

/// Owner named when an input lives on another device than the parameters.
const PARAMETERS: &str = "Llama-family model parameters";

// The shared block.

/// Where one attention call finds its keys and values.
///
/// Implementations own the masks and rotary tables of one call; the block
/// projects, normalizes, and rotates, then hands the projection over.
pub(super) trait LayerKeys<B: Backend> {
    /// Rotary tables of this call, already in the rotation dtype.
    fn angles(&self) -> Option<&RotaryAngles<B>>;

    /// Attends `projection` (rotated where the layer rotates) for block `layer`.
    ///
    /// Returns the output-projected `[batch, token, d]` attention output.
    fn attend(
        &mut self,
        layer: usize,
        attention: &GroupedQueryAttention<B>,
        projection: AttentionProjection<B>,
        points: AttentionPoints,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<Tensor<B, 3>>;
}

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

    /// Runs block `index` on `[batch, token, d]` states, emitting its hook points.
    ///
    /// Points are emitted in [`Self::activation_points`] order; with empty
    /// hooks every emission is a no-op.
    pub(super) fn forward(
        &self,
        hidden: Tensor<B, 3>,
        index: usize,
        keys: &mut impl LayerKeys<B>,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<Tensor<B, 3>> {
        let layer = Layer::decoder(index);
        let residual = hooks.apply(layer.residual_pre(), hidden)?;
        let attention = self.attention_branch(residual.clone(), index, keys, hooks)?;
        let hidden = residual + attention;
        let mlp = self.mlp_branch(hidden.clone(), layer, hooks)?;
        hooks.apply(layer.residual_post(), hidden + mlp)
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
            let angles = keys.angles().ok_or_else(|| {
                Error::configuration("llama.rotary", "a rotary layer ran without rotary tables")
            })?;
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
        let layer = Layer::decoder(index);
        let attention = layer.self_attention();
        let mut points = vec![
            layer.residual_pre().info(),
            layer.attention_input().info(),
            attention.query().info(),
            attention.key().info(),
            attention.value().info(),
        ];
        if self.self_attn.precision_mode == AttentionPrecisionMode::EagerFp32Softmax {
            points.extend([attention.scores().info(), attention.pattern().info()]);
        }
        points.extend([
            attention.head_context().info(),
            layer.attention_output().info(),
            layer.mlp_input().info(),
        ]);
        points.extend(mlp_hook_points(layer));
        points.extend([layer.mlp_output().info(), layer.residual_post().info()]);
        points
    }
}

/// Rotates queries and keys in the dtype of `angles`.
///
/// Tables in the activation dtype rotate directly (Llama-style references
/// round the tables first). F32 tables for reduced-precision activations
/// rotate in F32 and round the results once (OLMo 2); in F32 both coincide.
fn rotate<B: Backend>(
    projection: &mut AttentionProjection<B>,
    angles: &RotaryAngles<B>,
) -> Result<()> {
    let activation = projection.query.dtype();
    let rotation = angles.cosine.dtype();
    let query = projection.query.clone().cast(rotation);
    let key = projection.key.clone().cast(rotation);
    let (query, key) = apply_rotary(query, key, angles)?;
    projection.query = query.cast(activation);
    projection.key = key.cast(activation);
    Ok(())
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

/// Final states and requested diagnostics of one backbone pass.
pub(super) struct Backbone<B: Backend> {
    /// Final-normalized states `[batch, token, d]`.
    pub(super) hidden: Tensor<B, 3>,
    /// Embedding, every block output, and the final states, when requested.
    pub(super) hidden_states: Vec<Tensor<B, 3>>,
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

    /// Returns the stateless input contract: valid-token ordinal positions below the trained context.
    fn input_contract(&self) -> InputContract<'static> {
        InputContract::new(INPUT_NAMES, PARAMETERS, self.architecture.vocab_size)
            .with_position_limit(self.architecture.max_position_embeddings)
    }

    /// Returns the activation dtype, which is the embedding table's dtype.
    fn activation_dtype(&self) -> DType {
        self.embed_tokens.weight.val().dtype()
    }

    /// Builds the rotary tables of one call, or `None` when no layer rotates.
    ///
    /// Frequencies are evaluated once on the host for `evaluated_context`,
    /// angles once on the device in F32, and the tables converted once to the
    /// rotation dtype; every rotary layer then shares them.
    pub(super) fn rotary_tables(
        &self,
        positions: Tensor<B, 2, Int>,
        evaluated_context: usize,
    ) -> Result<Option<RotaryAngles<B>>> {
        let architecture = &self.architecture;
        if !architecture.layers.iter().any(|layer| layer.rotary) {
            return Ok(None);
        }
        let frequencies = RotaryFrequencies::new(architecture.rope, evaluated_context)?;
        let angles = rotary_angles_from_frequencies(&frequencies, positions)?;
        let dtype = architecture
            .rotary_precision
            .rotation_dtype(self.activation_dtype());
        Ok(Some(angles.to_dtype(dtype)))
    }

    /// Runs embedding, every block, and the final norm inside the caller's hook scope.
    fn forward(
        &self,
        input: &CausalLmInput<B>,
        retain_hidden_states: bool,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<Backbone<B>> {
        let device = self.embed_tokens.weight.lazy_device();
        let validated = validate_input(input, &device, &self.input_contract())?;
        let positions = validated.require_positions(INPUT_NAMES.position_ids)?;
        let angles = self.rotary_tables(positions, validated.evaluated_context)?;
        // A mask proven all-valid is dropped: ordinals then equal columns, so
        // global layers keep the fused causal path and build no plane.
        let padding = input.token_mask.as_ref().filter(|_| !validated.fully_valid);
        let mut keys = SequenceKeys::new(self, angles, padding, input.token_ids.dims())?;
        let mut hidden = hooks.apply(
            Stack::Decoder.embedding(),
            self.embed_tokens.forward(input.token_ids.clone()),
        )?;
        let mut hidden_states = Vec::new();
        for (index, layer) in self.layers.iter().enumerate() {
            if retain_hidden_states {
                hidden_states.push(hidden.clone());
            }
            hidden = layer.forward(hidden, index, &mut keys, hooks)?;
        }
        if retain_hidden_states {
            hidden_states.push(hidden.clone());
        }
        let hidden = hooks.apply(Stack::Decoder.final_norm(), self.norm.forward(hidden))?;
        if retain_hidden_states {
            hidden_states.push(hidden.clone());
        }
        Ok(Backbone {
            hidden,
            hidden_states,
        })
    }

    /// Lists every point of a complete pass, in emission order.
    fn activation_points(&self) -> Vec<PointInfo> {
        let mut points = vec![Stack::Decoder.embedding().info()];
        for (index, layer) in self.layers.iter().enumerate() {
            points.extend(layer.activation_points(index));
        }
        points.push(Stack::Decoder.final_norm().info());
        points
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
        if options.attentions {
            return self.forward_capturing_attentions(input, options);
        }
        let backbone = self
            .model
            .forward(&input, options.hidden_states, &mut Hooks::new())?;
        self.finish(input, options, backbone)
    }

    /// Projects the requested positions and assembles the output.
    fn finish(
        &self,
        input: CausalLmInput<B>,
        options: &ForwardOptions,
        backbone: Backbone<B>,
    ) -> Result<CausalLmOutput<B>> {
        let selected = select_hidden(
            backbone.hidden.clone(),
            input.token_mask.as_ref(),
            &options.logits,
        )?;
        let mut output = CausalLmOutput::from_logits(self.project_logits(selected)?);
        output.final_hidden_state = options.hidden_states.then_some(backbone.hidden);
        output.hidden_states = backbone.hidden_states;
        Ok(output)
    }

    /// Applies the independent or tied vocabulary projection to `[batch, token, d]`.
    pub(super) fn project_logits(&self, hidden: Tensor<B, 3>) -> Result<Tensor<B, 3>> {
        let device = hidden.device();
        match &self.lm_head {
            Some(head) => {
                ensure_same_device(
                    "llama.lm_head",
                    &head.weight.lazy_device(),
                    &device,
                    PARAMETERS,
                )?;
                operations::linear(hidden, head)
            }
            None => {
                let table = &self.model.embed_tokens.weight;
                ensure_same_device(
                    "llama.model.embed_tokens",
                    &table.lazy_device(),
                    &device,
                    PARAMETERS,
                )?;
                Ok(burn::tensor::module::linear(
                    hidden,
                    table.val().transpose(),
                    None,
                ))
            }
        }
    }
}

#[cfg(test)]
#[path = "model_tests.rs"]
mod tests;
