//! Shared inputs and memoized position/mask tables of one stack execution.
//!
//! Every block of a stack pass sees the same `[batch, token]` positions,
//! token validity, attention overrides, and (for decoders) encoder source.
//! Several tensors depend only on those inputs and on a block's static
//! configuration, never on its parameters or hidden states:
//!
//! | table | key | shape |
//! | --- | --- | --- |
//! | rotary cosine/sine tables | [`RopeConfig`] | `[batch, 1, token, rotary]` |
//! | T5 relative bucket IDs | [`T5RelativePositionBuckets`] | `[batch or 1, query, key]` |
//! | connectivity + key-padding exclusions | [`AttentionConnectivity`] | `[batch or 1, 1, query, key]` |
//! | source key-padding exclusions | (one per pass) | `[batch, 1, 1, source]` |
//!
//! [`StackPass`] builds each table the first time a block asks for it and
//! returns cheap handle clones afterwards, so an `L`-layer stack computes one
//! table per *distinct* configuration instead of `L`. Exclusion planes are
//! stored with a singleton head axis and expanded (as views, without copies)
//! to each layer's head count.
//!
//! # Relative-position buckets for arbitrary positions
//!
//! The custom graphs accept gapped, offset, or per-row positions, so the
//! contiguous-range [`crate::position::T5RelativeBucketTable`] does not apply.
//! Instead the host evaluates the reference bucket function once for each
//! displacement in `[-R, R]`, `R = min(max_distance, C - 1)` (see
//! [`super::plan`] for why this clamp changes no bucket), and the device
//! gathers that `2R + 1` lookup table at `clamp(p_k - p_q, -R, R) + R`. This
//! replaces the former per-layer `O(batch * token^2)` host loop, device
//! readback, and index upload with `O(R)` host work and one small upload per
//! pass. When every row uses the shared positions `0..token`, the IDs are
//! built once with a batch axis of one and broadcast by the attention kernel.
//!
//! # Attention overrides
//!
//! Without an override, a block uses its structural defaults: causal
//! attention in decoder stacks, bidirectional attention in encoder stacks, and
//! its configured [`AttentionConnectivity`]. An explicit [`AttentionPolicy`]
//! (from [`ExecutionContext`] overrides, or the bidirectional policy of a
//! full-sequence bidirectional forward) replaces causality *and* structural
//! connectivity, intersects its `allowed` links with key padding, and adds its
//! bias to any learned position bias. Cross-attention defaults to every valid
//! source token.
//!
//! # Invariants
//!
//! Callers validate before constructing a pass: positions lie in
//! `0..evaluated_context`, masks and positions share the `[batch, token]` grid
//! and device, and overrides broadcast to their attention planes. A pass is
//! single-threaded; its memo uses interior mutability so blocks can share it
//! by reference.

use core::cell::RefCell;

use burn::tensor::{Bool, DType, Int, Tensor, TensorData, backend::Backend, module::embedding};

use crate::{
    Error, Result,
    architecture::TokenMask,
    attention::{AttentionConnectivity, MaskGeometry},
    experiment::{AttentionPolicy, ExecutionContext},
    intervention::{Layer, Stack},
    position::{
        RopeConfig, RotaryAngles, RotaryFrequencies, T5RelativePositionBias,
        T5RelativePositionBuckets, rotary_angles_from_frequencies,
    },
};

use super::plan::{displacement_table, slot_displacement};

/// Encoder states read by every decoder cross-attention branch of a pass.
#[derive(Clone, Debug)]
pub(super) struct SourceInputs<'a, B: Backend> {
    /// Final encoder states, `[batch, source_token, hidden]`.
    pub(super) hidden: &'a Tensor<B, 3>,
    /// Source validity (`true == real token`); `None` means every token is valid.
    pub(super) token_mask: Option<&'a TokenMask<B>>,
}

/// Where explicit attention overrides come from.
#[derive(Clone, Debug)]
enum Overrides<'a, B: Backend> {
    /// No overrides: every block uses its structural defaults.
    Structural,
    /// One self-attention policy for every block (for example bidirectional execution).
    SelfAttention(AttentionPolicy<B>),
    /// Per-layer and default overrides of a staged execution context.
    Context(&'a ExecutionContext<B>),
}

/// Exclusions, additive bias, and causality resolved for one attention call.
#[derive(Clone, Debug)]
pub(super) struct AttentionInputs<B: Backend> {
    /// `[batch, heads, query, key]` exclusions (`true == excluded`), often an expanded view.
    pub(super) mask: Option<Tensor<B, 4, Bool>>,
    /// Additive score bias broadcastable to `[batch, heads, query, key]`.
    pub(super) bias: Option<Tensor<B, 4>>,
    /// Whether the kernel excludes keys after the bottom-right-aligned query.
    pub(super) causal: bool,
}

impl<B: Backend> AttentionInputs<B> {
    /// Adds a learned position bias `[1 or batch, heads, query, key]` to any override bias.
    ///
    /// A lone position bias keeps its batch-one layout so the kernel can
    /// broadcast it; a sum is formed at the full `shape`.
    pub(super) fn with_position_bias(
        mut self,
        position_bias: Option<Tensor<B, 4>>,
        shape: [usize; 4],
    ) -> Self {
        self.bias = match (self.bias, position_bias) {
            (Some(existing), Some(position)) => Some(existing + position.expand(shape)),
            (None, position) => position,
            (existing, None) => existing,
        };
        self
    }
}

/// Lazily built tables shared by every block of a pass.
#[derive(Debug)]
struct PassTables<B: Backend> {
    /// Rotary tables in the activation dtype, one per distinct RoPE configuration.
    rotary: Vec<(RopeConfig, RotaryAngles<B>)>,
    /// `[batch or 1, query, key]` T5 bucket IDs, one per distinct bucket mapping.
    relative_buckets: Vec<(T5RelativePositionBuckets, Tensor<B, 3, Int>)>,
    /// `[batch or 1, 1, query, key]` exclusions, one per distinct connectivity.
    exclusions: Vec<(AttentionConnectivity, Option<Tensor<B, 4, Bool>>)>,
    /// `[batch, 1, 1, source]` source key padding, built on first use.
    source_padding: Option<Option<Tensor<B, 4, Bool>>>,
}

impl<B: Backend> Default for PassTables<B> {
    /// An empty memo; every table is built on first use.
    fn default() -> Self {
        Self {
            rotary: Vec::new(),
            relative_buckets: Vec::new(),
            exclusions: Vec::new(),
            source_padding: None,
        }
    }
}

/// Positions, padding, overrides, source, and memoized tables of one stack pass.
#[derive(Debug)]
pub(super) struct StackPass<'a, B: Backend> {
    /// Target positions `[batch, token]`, all in `0..evaluated_context`.
    positions: Tensor<B, 2, Int>,
    /// Target validity; `None` means every token is valid.
    token_mask: Option<&'a TokenMask<B>>,
    /// One plus the largest target position.
    evaluated_context: usize,
    /// Stack whose structural attention direction applies.
    stack: Stack,
    /// Encoder states for cross-attention, absent outside seq2seq decoders.
    source: Option<SourceInputs<'a, B>>,
    /// Explicit attention overrides.
    overrides: Overrides<'a, B>,
    /// Whether every row uses positions `0..token`, so tables may drop the batch axis.
    row_invariant_positions: bool,
    /// Activation dtype that rotary tables are converted to once.
    dtype: DType,
    /// Tables built on first use.
    tables: RefCell<PassTables<B>>,
}

impl<'a, B: Backend> StackPass<'a, B> {
    /// Starts a pass with structural attention over validated positions.
    ///
    /// `positions` is `[batch, token]` with values in `0..evaluated_context`;
    /// `dtype` is the activation dtype of the blocks.
    pub(super) fn new(
        positions: Tensor<B, 2, Int>,
        token_mask: Option<&'a TokenMask<B>>,
        evaluated_context: usize,
        stack: Stack,
        dtype: DType,
    ) -> Self {
        Self {
            positions,
            token_mask,
            evaluated_context,
            stack,
            source: None,
            overrides: Overrides::Structural,
            row_invariant_positions: false,
            dtype,
            tables: RefCell::new(PassTables::default()),
        }
    }

    /// Reads positions, padding, overrides, and any encoder source from a staged context.
    ///
    /// The context must already be validated for the hidden states it accompanies.
    pub(super) fn from_context(
        context: &'a ExecutionContext<B>,
        stack: Stack,
        dtype: DType,
    ) -> Self {
        let mut pass = Self::new(
            context.positions.clone(),
            context.token_mask.as_ref(),
            context.evaluated_context,
            stack,
            dtype,
        );
        pass.source = context.encoder.as_ref().map(|source| SourceInputs {
            hidden: &source.hidden,
            token_mask: source.token_mask.as_ref(),
        });
        pass.overrides = Overrides::Context(context);
        pass
    }

    /// Declares that every row uses positions `0..token` (no explicit IDs and no padding).
    ///
    /// Relative-position tables are then built with a batch axis of one.
    #[must_use]
    pub(super) fn with_row_invariant_positions(mut self, row_invariant: bool) -> Self {
        self.row_invariant_positions = row_invariant;
        self
    }

    /// Attaches encoder states for decoder cross-attention.
    #[must_use]
    pub(super) fn with_source(
        mut self,
        hidden: &'a Tensor<B, 3>,
        token_mask: Option<&'a TokenMask<B>>,
    ) -> Self {
        self.source = Some(SourceInputs { hidden, token_mask });
        self
    }

    /// Applies one self-attention policy to every block, replacing structural defaults.
    #[must_use]
    pub(super) fn with_self_attention_policy(mut self, policy: AttentionPolicy<B>) -> Self {
        self.overrides = Overrides::SelfAttention(policy);
        self
    }

    /// Target validity of this pass.
    pub(super) fn token_mask(&self) -> Option<&'a TokenMask<B>> {
        self.token_mask
    }

    /// Encoder source of this pass, when one is attached.
    pub(super) fn source(&self) -> Option<&SourceInputs<'a, B>> {
        self.source.as_ref()
    }

    /// Whether blocks default to causal self-attention (decoder stacks).
    fn default_causal(&self) -> bool {
        self.stack == Stack::Decoder
    }

    /// Returns the explicit self-attention policy of `layer`, if any.
    pub(super) fn self_policy(&self, layer: Layer) -> Option<AttentionPolicy<B>> {
        match &self.overrides {
            Overrides::Structural => None,
            Overrides::SelfAttention(policy) => Some(policy.clone()),
            Overrides::Context(context) => explicit_self_policy(context, layer),
        }
    }

    /// Returns the explicit cross-attention policy of `layer`, if any.
    fn cross_policy(&self, layer: Layer) -> Option<AttentionPolicy<B>> {
        match &self.overrides {
            Overrides::Structural | Overrides::SelfAttention(_) => None,
            Overrides::Context(context) => explicit_cross_policy(context, layer),
        }
    }

    /// Rotary tables for `rope` over this pass's positions, in the activation dtype.
    pub(super) fn rotary_angles(&self, rope: RopeConfig) -> Result<RotaryAngles<B>> {
        if let Some((_, angles)) = self
            .tables
            .borrow()
            .rotary
            .iter()
            .find(|(config, _)| *config == rope)
        {
            return Ok(angles.clone());
        }
        let frequencies = RotaryFrequencies::new(rope, self.evaluated_context)?;
        let angles = rotary_angles_from_frequencies(&frequencies, self.positions.clone())?
            .to_dtype(self.dtype);
        self.tables.borrow_mut().rotary.push((rope, angles.clone()));
        Ok(angles)
    }

    /// Learned T5 bias `[batch or 1, heads, query, key]` for this pass's positions.
    ///
    /// Entry `(b, h, i, j)` is `bias.weight[bucket(p[b, j] - p[b, i]), h]`; the
    /// embedding lookup stays differentiable with respect to the table.
    pub(super) fn relative_bias(&self, bias: &T5RelativePositionBias<B>) -> Result<Tensor<B, 4>> {
        let ids = self.relative_bucket_ids(bias.buckets())?;
        let [rows, queries, keys] = ids.dims();
        let weight = bias.weight.val();
        let heads = weight.dims()[1];
        Ok(embedding(weight, ids.reshape([rows * queries, keys]))
            .reshape([rows, queries, keys, heads])
            .permute([0, 3, 1, 2]))
    }

    /// Memoized `[batch or 1, query, key]` bucket IDs of one bucket mapping.
    fn relative_bucket_ids(&self, buckets: T5RelativePositionBuckets) -> Result<Tensor<B, 3, Int>> {
        if let Some((_, ids)) = self
            .tables
            .borrow()
            .relative_buckets
            .iter()
            .find(|(mapping, _)| *mapping == buckets)
        {
            return Ok(ids.clone());
        }
        let ids = self.build_relative_bucket_ids(buckets)?;
        self.tables
            .borrow_mut()
            .relative_buckets
            .push((buckets, ids.clone()));
        Ok(ids)
    }

    /// Gathers host-evaluated buckets at the clamped device displacement `p_k - p_q`.
    fn build_relative_bucket_ids(
        &self,
        buckets: T5RelativePositionBuckets,
    ) -> Result<Tensor<B, 3, Int>> {
        let table = displacement_table(buckets.max_distance(), self.evaluated_context)
            .filter(|table| table.length <= PORTABLE_INDEX_LIMIT)
            .ok_or_else(|| {
                Error::configuration(
                    "custom.t5_relative_bias",
                    format!(
                        "a displacement table for max_distance {} over context {} exceeds the portable 32-bit index range",
                        buckets.max_distance(),
                        self.evaluated_context
                    ),
                )
            })?;
        let mut lookup = Vec::new();
        lookup.try_reserve_exact(table.length).map_err(|_| {
            Error::BackendCapability(format!(
                "could not reserve {} host T5 relative buckets",
                table.length
            ))
        })?;
        for slot in 0..table.length {
            let bucket = buckets.bucket(slot_displacement(slot, table.radius));
            // Bucket counts are bounded by the configuration's i64 check.
            lookup.push(i64::try_from(bucket).map_err(|_| {
                Error::configuration(
                    "custom.t5_relative_bias",
                    "bucket index does not fit in i64",
                )
            })?);
        }
        let device = self.positions.device();
        let lookup =
            Tensor::<B, 1, Int>::from_data(TensorData::new(lookup, [table.length]), &device);
        let positions = if self.row_invariant_positions {
            let tokens = self.positions.dims()[1];
            self.positions.clone().slice([0..1, 0..tokens])
        } else {
            self.positions.clone()
        };
        let [rows, tokens] = positions.dims();
        let plane = [rows, tokens, tokens];
        // `radius < evaluated_context`, which validated positions bound, so it
        // fits the backend integer dtype.
        let radius = table.radius as i64;
        let keys = positions.clone().reshape([rows, 1, tokens]).expand(plane);
        let queries = positions.reshape([rows, tokens, 1]).expand(plane);
        let slots = (keys - queries).clamp(-radius, radius) + radius;
        Ok(lookup
            .select(0, slots.reshape([rows * tokens * tokens]))
            .reshape(plane))
    }

    /// Resolves exclusions, bias, and causality for one self-attention call.
    ///
    /// Structural defaults use the stack direction and `connectivity`; an
    /// explicit policy replaces both and keeps only key padding from the pass.
    pub(super) fn self_attention(
        &self,
        layer: Layer,
        connectivity: AttentionConnectivity,
        heads: usize,
    ) -> Result<AttentionInputs<B>> {
        let [batch, tokens] = self.positions.dims();
        let shape = [batch, heads, tokens, tokens];
        match self.self_policy(layer) {
            None => Ok(AttentionInputs {
                mask: self
                    .exclusions(connectivity)?
                    .map(|plane| plane.expand(shape)),
                bias: None,
                causal: self.default_causal(),
            }),
            Some(policy) => self.apply_policy(
                policy,
                self.exclusions(AttentionConnectivity::Global)?,
                shape,
            ),
        }
    }

    /// Resolves exclusions, bias, and causality for one cross-attention call.
    ///
    /// The default sees every valid source token; an explicit policy restricts
    /// links and may add bias or causality.
    pub(super) fn cross_attention(&self, layer: Layer, heads: usize) -> Result<AttentionInputs<B>> {
        let source = self.source.as_ref().ok_or_else(|| {
            Error::configuration(
                "custom.cross_attention",
                "decoder layer requires encoder hidden states",
            )
        })?;
        let [batch, queries] = self.positions.dims();
        let keys = source.hidden.dims()[1];
        let shape = [batch, heads, queries, keys];
        let padding = self.source_padding(source, batch, keys);
        match self.cross_policy(layer) {
            None => Ok(AttentionInputs {
                mask: padding.map(|plane| plane.expand(shape)),
                bias: None,
                causal: false,
            }),
            Some(policy) => self.apply_policy(policy, padding, shape),
        }
    }

    /// Intersects a policy's allowed links with key padding and expands its bias.
    fn apply_policy(
        &self,
        policy: AttentionPolicy<B>,
        padding: Option<Tensor<B, 4, Bool>>,
        shape: [usize; 4],
    ) -> Result<AttentionInputs<B>> {
        policy.validate(shape, &self.positions.device())?;
        let excluded = policy
            .allowed
            .map(|allowed| allowed.bool_not().expand(shape));
        let padding = padding.map(|plane| plane.expand(shape));
        let mask = match (excluded, padding) {
            (Some(excluded), Some(padding)) => Some(excluded.bool_or(padding)),
            (excluded, None) => excluded,
            (None, padding) => padding,
        };
        Ok(AttentionInputs {
            mask,
            bias: policy.bias.map(|bias| bias.expand(shape)),
            causal: policy.causal,
        })
    }

    /// Memoized `[batch or 1, 1, query, key]` key-padding and connectivity exclusions.
    ///
    /// Causality is left to the attention kernel. Windows and chunks count
    /// valid tokens (see [`AttentionConnectivity`]); padded query rows are not
    /// excluded here because their outputs are zeroed after attention.
    fn exclusions(
        &self,
        connectivity: AttentionConnectivity,
    ) -> Result<Option<Tensor<B, 4, Bool>>> {
        if let Some((_, plane)) = self
            .tables
            .borrow()
            .exclusions
            .iter()
            .find(|(rule, _)| *rule == connectivity)
        {
            return Ok(plane.clone());
        }
        let [batch, tokens] = self.positions.dims();
        let plane = connectivity.exclusion_mask::<B>(
            false,
            MaskGeometry::new(batch, 1, tokens, tokens),
            None,
            self.token_mask,
            &self.positions.device(),
        )?;
        self.tables
            .borrow_mut()
            .exclusions
            .push((connectivity, plane.clone()));
        Ok(plane)
    }

    /// Memoized `[batch, 1, 1, source]` source key padding.
    fn source_padding(
        &self,
        source: &SourceInputs<'a, B>,
        batch: usize,
        keys: usize,
    ) -> Option<Tensor<B, 4, Bool>> {
        if let Some(plane) = &self.tables.borrow().source_padding {
            return plane.clone();
        }
        let plane = source.token_mask.map(|mask| {
            mask.as_tensor()
                .clone()
                .bool_not()
                .reshape([batch, 1, 1, keys])
        });
        self.tables.borrow_mut().source_padding = Some(plane.clone());
        plane
    }
}

/// Largest displacement table, so every slot fits a signed 32-bit integer dtype.
const PORTABLE_INDEX_LIMIT: usize = i32::MAX as usize;

/// The explicit self-attention override for `layer`: its own policy, else the context default.
pub(super) fn explicit_self_policy<B: Backend>(
    context: &ExecutionContext<B>,
    layer: Layer,
) -> Option<AttentionPolicy<B>> {
    context
        .layer_attention
        .iter()
        .rev()
        .find(|(candidate, _)| *candidate == layer)
        .map(|(_, policy)| policy.clone())
        .or_else(|| context.attention.clone())
}

/// The explicit cross-attention override for `layer`: its own policy, else the context default.
pub(super) fn explicit_cross_policy<B: Backend>(
    context: &ExecutionContext<B>,
    layer: Layer,
) -> Option<AttentionPolicy<B>> {
    context
        .layer_cross_attention
        .iter()
        .rev()
        .find(|(candidate, _)| *candidate == layer)
        .map(|(_, policy)| policy.clone())
        .or_else(|| context.cross_attention.clone())
}

#[cfg(test)]
#[path = "pass_tests.rs"]
/// Pass tables against the per-layer reference algorithms they replace.
mod tests;
