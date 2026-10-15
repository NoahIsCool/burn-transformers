//! The dense-decoder interface shared by the Llama and Gemma families.
//!
//! The stateless, hooked, staged, and batched cached execution paths in this
//! directory never name a family. They run against a [`DecoderGraph`]: an
//! embedding, a stack of [`DecoderBlock`]s that attend through
//! [`LayerKeys`], a final normalization, and a vocabulary projection, plus the
//! integer geometry and rotary schedule those paths plan with:
//!
//! ```text
//! h_0     = Embed(tokens)              family-owned (Gemma scales by sqrt(d))
//! h_{l+1} = Block_l(h_l; keys_l)       family-owned norm placement, attention, MLP
//! logits  = Project(FinalNorm(h_L))    family-owned head (tied or not, soft-capped)
//! ```
//!
//! A family implements the two traits for its own block and model types and
//! inherits every execution path, including the genuinely batched cache step
//! of the `decode` submodule. Only the *keys* differ between paths: a block
//! projects, normalizes, and rotates its queries and keys, then hands them
//! to [`LayerKeys::attend`], which attends over the input's own tokens
//! (stateless) or over a decode cache (cached).
//!
//! # Rotary schedule
//!
//! A graph declares a few RoPE parameter sets and which set, if any, each
//! layer rotates with ([`RotarySchedule`]): Llama-family models rotate every
//! rotary layer with one set (SmolLM3's NoPE layers with none), Gemma 3
//! rotates sliding and global layers with different bases and scalings.
//! Every call evaluates each *used* set's frequencies once on the host,
//! its angles once on the device in F32, and converts the tables once to the
//! rotation dtype ([`RotaryPrecision`]); the layers of a set share them
//! ([`RotaryTables`]).
//!
//! # Labels
//!
//! Errors name the family through its [`GraphLabels`]: configuration fields
//! carry the family prefix (`llama.position_ids`, `gemma.execution.hidden`)
//! and messages its display name, so a shared path reports errors exactly as
//! a family-private one would.

use burn::{
    nn::{Embedding, Linear},
    tensor::{DType, Int, Tensor, backend::Backend},
};

use crate::{
    Error, Result,
    attention::{
        AttentionConnectivity, AttentionPrecisionMode, AttentionProjection, GroupedQueryAttention,
    },
    feed_forward::mlp_hook_points,
    intervention::{AttentionPoints, Hooks, Layer, Point, PointInfo},
    models::shared::{InputContract, InputNames, ensure_same_device},
    operations,
    position::{
        RopeConfig, RopeScaling, RotaryAngles, RotaryFrequencies, apply_rotary,
        rotary_angles_from_frequencies,
    },
};

use super::super::RotaryPrecision;

// Family identity.

/// Names a family uses in the errors and diagnostics of shared execution paths.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GraphLabels {
    /// Display name used in messages, e.g. `Llama-family` or `Gemma`.
    pub(crate) family: &'static str,
    /// First segment of configuration error fields, e.g. `llama`.
    pub(crate) prefix: &'static str,
    /// Tensor names reported by input errors.
    pub(crate) inputs: InputNames<'static>,
    /// Owner named when an input lives on another device than the parameters.
    pub(crate) parameters: &'static str,
    /// Owner named when an input lives on another device than the decode state.
    pub(crate) state: &'static str,
}

impl GraphLabels {
    /// Returns the configuration error field `<prefix>.<suffix>`.
    pub(crate) fn field(&self, suffix: &str) -> String {
        format!("{}.{suffix}", self.prefix)
    }
}

/// Integer geometry shared by every block of a dense decoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DecoderGeometry {
    /// Vocabulary size `V`: embedding rows and logits per token.
    pub(crate) vocab_size: usize,
    /// Residual width `d`.
    pub(crate) hidden_size: usize,
    /// Query heads `H`, the head extent of exclusion planes.
    pub(crate) query_heads: usize,
    /// Native key/value heads `K` stored by the decode cache.
    pub(crate) key_value_heads: usize,
    /// Per-head width `h` of queries, keys, and values.
    pub(crate) head_dim: usize,
    /// Positions (and cache capacities) must lie below this trained context.
    pub(crate) max_position_embeddings: usize,
}

// Rotary schedule and per-call tables.

/// The RoPE parameter sets of a decoder and the set each layer rotates with.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RotarySchedule {
    /// Distinct parameter sets.
    pub(crate) sets: Vec<RopeConfig>,
    /// Per layer: the index into `sets` it rotates with, or `None` (NoPE).
    pub(crate) layers: Vec<Option<usize>>,
    /// The dtype in which queries and keys are rotated.
    pub(crate) precision: RotaryPrecision,
}

impl RotarySchedule {
    /// Returns, per set, whether at least one layer rotates with it.
    fn used_sets(&self) -> Vec<bool> {
        let mut used = vec![false; self.sets.len()];
        for set in self.layers.iter().flatten() {
            if let Some(flag) = used.get_mut(*set) {
                *flag = true;
            }
        }
        used
    }

    /// Returns the tightest original context of a used context-dependent set.
    ///
    /// Cached keys are rotated once, with the frequencies of the context at
    /// the time they were written; a policy whose frequencies change past its
    /// original context (dynamic NTK, LongRoPE) therefore cannot extend a
    /// cache beyond it. Returns that limit and the scaling that imposes it.
    pub(crate) fn cached_context_limit(&self) -> Option<(usize, RopeScaling)> {
        self.sets
            .iter()
            .zip(self.used_sets())
            .filter(|(_, used)| *used)
            .filter_map(|(set, _)| {
                let scaling = set.scaling();
                scaling.cached_context_limit().map(|limit| (limit, scaling))
            })
            .min_by_key(|(limit, _)| *limit)
    }
}

/// The rotary tables of one call, already in the rotation dtype.
#[derive(Debug, Clone)]
pub(crate) struct RotaryTables<B: Backend> {
    /// Per set: its tables when some layer uses it.
    tables: Vec<Option<RotaryAngles<B>>>,
    /// Per layer: the set it rotates with, or `None`.
    layers: Vec<Option<usize>>,
}

impl<B: Backend> RotaryTables<B> {
    /// Returns the tables layer `layer` rotates with, or `None` for a NoPE layer.
    pub(crate) fn for_layer(&self, layer: usize) -> Option<&RotaryAngles<B>> {
        let set = (*self.layers.get(layer)?)?;
        self.tables.get(set)?.as_ref()
    }
}

/// Builds every used set's tables for `[batch, token]` `positions`.
///
/// Frequencies are evaluated once per set on the host for
/// `evaluated_context`, angles once on the device in F32, and the tables
/// converted once to the schedule's rotation dtype for activations of
/// `activation` dtype.
///
/// # Errors
///
/// Returns frequency-evaluation errors (for example a context beyond a
/// policy's support) and position-tensor errors.
pub(crate) fn rotary_tables<B: Backend>(
    schedule: RotarySchedule,
    activation: DType,
    positions: Tensor<B, 2, Int>,
    evaluated_context: usize,
) -> Result<RotaryTables<B>> {
    let dtype = schedule.precision.rotation_dtype(activation);
    let used = schedule.used_sets();
    let tables = schedule
        .sets
        .iter()
        .zip(used)
        .map(|(set, used)| {
            if !used {
                return Ok(None);
            }
            let frequencies = RotaryFrequencies::new(*set, evaluated_context)?;
            let angles = rotary_angles_from_frequencies(&frequencies, positions.clone())?;
            Ok(Some(angles.to_dtype(dtype)))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(RotaryTables {
        tables,
        layers: schedule.layers,
    })
}

/// Rotates queries and keys in the dtype of `angles`.
///
/// Tables in the activation dtype rotate directly (most references round the
/// tables first). F32 tables for reduced-precision activations rotate in F32
/// and round the results once (OLMo 2); in F32 both coincide.
///
/// # Errors
///
/// Returns rotary shape errors.
pub(crate) fn rotate<B: Backend>(
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

// Blocks and keys.

/// Where one attention call finds its keys and values.
///
/// Implementations own the masks and rotary tables of one call; the block
/// projects, normalizes, and rotates, then hands the projection over.
pub(crate) trait LayerKeys<B: Backend> {
    /// Rotary tables of block `layer` in this call, already in the rotation dtype.
    fn angles(&self, layer: usize) -> Option<&RotaryAngles<B>>;

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

/// One residual block of a dense decoder.
pub(crate) trait DecoderBlock<B: Backend> {
    /// Runs block `index` on `[batch, token, d]` states, emitting its hook points.
    ///
    /// Points are emitted in [`Self::block_points`] order; with empty
    /// hooks every emission is a no-op.
    fn forward<K: LayerKeys<B>>(
        &self,
        hidden: Tensor<B, 3>,
        index: usize,
        keys: &mut K,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<Tensor<B, 3>>;

    /// Returns which earlier keys a query may see, on top of causality and padding.
    fn connectivity(&self) -> AttentionConnectivity;

    /// Returns the attention operation graph of this block.
    fn attention_precision(&self) -> AttentionPrecisionMode;

    /// Lists the points this block emits as block `index`, in emission order.
    fn block_points(&self, index: usize) -> Vec<PointInfo> {
        decoder_block_points(
            index,
            self.attention_precision() == AttentionPrecisionMode::EagerFp32Softmax,
        )
    }
}

/// Lists the points of one attention-plus-MLP block, in emission order.
///
/// ```text
/// residual.pre, attention.input, attention.{query, key, value},
/// [attention.{scores, pattern} when eager], attention.head_context,
/// attention.output, mlp.input, mlp.pre_activation, mlp.activation,
/// mlp.output, residual.post
/// ```
///
/// `attention.output` and `mlp.output` are the branches added to the
/// residual stream, after any output norm. Every hidden-width point among
/// them is listed in [`HIDDEN_POINTS`].
pub(crate) fn decoder_block_points(index: usize, eager: bool) -> Vec<PointInfo> {
    let layer = Layer::decoder(index);
    let attention = layer.self_attention();
    let mut points = vec![
        layer.residual_pre().info(),
        layer.attention_input().info(),
        attention.query().info(),
        attention.key().info(),
        attention.value().info(),
    ];
    if eager {
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

/// Every `[batch, token, width]` point of a block listed by [`decoder_block_points`].
pub(crate) const HIDDEN_POINTS: [fn(Layer) -> Point<3>; 8] = [
    Layer::residual_pre,
    Layer::attention_input,
    Layer::attention_output,
    Layer::mlp_input,
    Layer::mlp_pre_activation,
    Layer::mlp_activation,
    Layer::mlp_output,
    Layer::residual_post,
];

// The decoder.

/// A dense decoder the shared execution paths can run.
pub(crate) trait DecoderGraph<B: Backend> {
    /// The family's residual block.
    type Block: DecoderBlock<B>;

    /// Names used in errors and diagnostics.
    const LABELS: GraphLabels;

    /// Returns the integer geometry shared by every block.
    fn geometry(&self) -> DecoderGeometry;

    /// Returns the RoPE parameter sets and the set each layer rotates with.
    fn rotary_schedule(&self) -> RotarySchedule;

    /// Returns the blocks in order.
    fn blocks(&self) -> &[Self::Block];

    /// Returns the token embedding table, whose dtype is the activation dtype.
    fn embedding(&self) -> &Embedding<B>;

    /// Embeds `[batch, token]` IDs into the `[batch, token, d]` residual stream.
    fn embed(&self, token_ids: Tensor<B, 2, Int>) -> Tensor<B, 3>;

    /// Applies the final normalization to `[batch, token, d]` states.
    fn final_norm(&self, hidden: Tensor<B, 3>) -> Tensor<B, 3>;

    /// Projects `[batch, token, d]` final states to `[batch, token, V]` logits.
    ///
    /// # Errors
    ///
    /// Returns a device error when `hidden` lives elsewhere than the head.
    fn project_logits(&self, hidden: Tensor<B, 3>) -> Result<Tensor<B, 3>>;

    /// Returns the activation dtype, which is the embedding table's dtype.
    fn activation_dtype(&self) -> DType {
        self.embedding().weight.val().dtype()
    }

    /// Returns the device holding the parameters.
    fn parameter_device(&self) -> B::Device {
        self.embedding().weight.lazy_device()
    }

    /// Returns the stateless input contract: valid-token ordinal positions below the trained context.
    fn input_contract(&self) -> InputContract<'static> {
        let geometry = self.geometry();
        InputContract::new(
            Self::LABELS.inputs,
            Self::LABELS.parameters,
            geometry.vocab_size,
        )
        .with_position_limit(geometry.max_position_embeddings)
    }

    /// Builds the rotary tables of one call for `[batch, token]` positions.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`rotary_tables`].
    fn rotary_tables(
        &self,
        positions: Tensor<B, 2, Int>,
        evaluated_context: usize,
    ) -> Result<RotaryTables<B>> {
        rotary_tables(
            self.rotary_schedule(),
            self.activation_dtype(),
            positions,
            evaluated_context,
        )
    }
}

/// Applies an independent head, or the transposed embedding when tied.
///
/// `hidden` is `[batch, token, d]`; the result is `[batch, token, V]`. A tied
/// head multiplies by the transposed view of the `[V, d]` table through
/// [`operations::matmul_transposed`], which never copies the table.
///
/// # Errors
///
/// Returns a device error naming the head (or the tied embedding) when
/// `hidden` lives on another device.
pub(crate) fn vocabulary_projection<B: Backend>(
    labels: &GraphLabels,
    hidden: Tensor<B, 3>,
    lm_head: Option<&Linear<B>>,
    embedding: &Embedding<B>,
) -> Result<Tensor<B, 3>> {
    let device = hidden.device();
    match lm_head {
        Some(head) => {
            ensure_same_device(
                &labels.field("lm_head"),
                &head.weight.lazy_device(),
                &device,
                labels.parameters,
            )?;
            operations::linear(hidden, head)
        }
        None => {
            let table = &embedding.weight;
            ensure_same_device(
                &labels.field("model.embed_tokens"),
                &table.lazy_device(),
                &device,
                labels.parameters,
            )?;
            operations::matmul_transposed(hidden, table.val())
        }
    }
}

/// Reports a rotary layer that ran without rotary tables.
pub(crate) fn missing_rotary_tables(labels: &GraphLabels) -> Error {
    Error::configuration(
        labels.field("rotary"),
        "a rotary layer ran without rotary tables",
    )
}

#[cfg(test)]
#[path = "graph_tests.rs"]
mod tests;
