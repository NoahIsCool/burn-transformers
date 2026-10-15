//! Compact, right-padded layout of one batched stateful (prefill or decode) call.
//!
//! A cached call receives a rectangular `[B, S]` input whose mask may pad any
//! columns. Padding must consume neither cache positions, rotary positions,
//! nor causal-convolution history, so the whole batch runs in one *compact*
//! layout: row `b`'s `n_b` valid tokens occupy slots `0..n_b` of a `[B, N]`
//! block, `N = max_b n_b`, in their original order, and slots `n_b..N` are
//! padding. Slot `i` of row `b` is cache position `W_b + i`, where `W_b` is
//! the row's working length. Consequently
//!
//! * every cache group appends with the right-padded alignment (row `b`
//!   writes slots `0..n_b` at positions `W_b..W_b + n_b`);
//! * a single `[B, N]` validity mask (`i < n_b`) serves DeltaNet, attention
//!   key validity, and output zeroing;
//! * a query in slot `i` may attend to cache positions `<= W_b + i`, which is
//!   the bottom-right causal rule whenever every row shares one `W` and `n`.
//!
//! The common layouts cost nothing extra: without a mask the compact block is
//! the input itself, and when every row's valid columns are a prefix `0..n_b`
//! compaction is a narrow. Left or interior padding costs one gather for the
//! token IDs (and explicit positions) and one per restored output. A row with
//! `n_b = 0` (a finished sequence during decoding) is pure padding; it writes
//! no cache entry and its outputs are exact zeros.
//!
//! [`CachedAttention`] runs one full-attention layer of such a batch against
//! a native-KV cache group: it writes the step's rotated keys and values into
//! their slots ahead of attention and attends over one zero-copy view of each
//! row's `[0, W_b + n_b)` keys.
//!
//! Both pieces are family-independent; the Qwen3.5 and Qwen3-MoE decoders
//! share them.

use core::num::NonZeroUsize;

use burn::tensor::{Bool, DType, Int, Tensor, TensorData, backend::Backend};

use crate::{
    Error, LogitsSelection, Result,
    architecture::TokenMask,
    attention::GroupedQueryAttention,
    cache::{KvBatchReadPlan, TensorDecodeState, TensorStepWrites},
    position::{
        RopeConfig, RotaryAngles, RotaryFrequencies, apply_rotary, rotary_angles_from_frequencies,
    },
};

use crate::models::{
    qwen3_5::plan::{
        all_equal, columns_below, compact_row_plan, prefix_rows, row_counts, signed_index,
        slot_positions, slot_validity, strictly_increasing, widest,
    },
    shared::select_hidden_with_last_columns,
};

/// Compact right-padded layout of one stateful call, built once per call.
#[derive(Debug, Clone)]
pub(crate) struct CompactBatch<B: Backend> {
    /// Caller columns `S`, including padding.
    sequence: usize,
    /// Compact slots `N = max_b n_b`.
    width: usize,
    /// Valid tokens `n_b` appended by every row.
    counts: Vec<usize>,
    /// Working cache length `W_b` of every row before the call.
    starts: Vec<usize>,
    /// `[B, N]` caller column feeding every slot; `None` for prefix layouts.
    gather: Option<Tensor<B, 2, Int>>,
    /// `[B, S]` slot restored to every caller column (`N` addresses a zero
    /// slot); `None` for prefix layouts.
    restore: Option<Tensor<B, 2, Int>>,
    /// `[B, N]` validity of every slot (`true` = token); `None` when every
    /// row fills all `N` slots.
    valid: Option<Tensor<B, 2, Bool>>,
    /// `[B, N]` cache position `W_b + i` of every slot.
    cache_positions: Tensor<B, 2, Int>,
}

impl<B: Backend> CompactBatch<B> {
    /// Plans the compact layout from each row's valid caller columns and working length.
    ///
    /// `valid_columns[b]` must be strictly increasing and below `sequence`
    /// (the shared stateful preparation guarantees it) and at least one row
    /// must append a token.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Shape`] for mismatched row counts, columns outside the
    /// rectangle or not strictly increasing, or a call in which no row
    /// appends a token, and [`Error::Configuration`] for a cache position
    /// beyond `i64`.
    pub(crate) fn new(
        valid_columns: &[Vec<usize>],
        sequence: usize,
        starts: &[usize],
        device: &B::Device,
    ) -> Result<Self> {
        let batch = valid_columns.len();
        if batch == 0 || starts.len() != batch {
            return Err(Error::shape(
                "compact_batch.rows",
                format!("{batch} working lengths for a non-empty batch"),
                format!("{} working lengths", starts.len()),
            ));
        }
        // The host plan below is verified in `verus/qwen.rs`: each helper
        // returns exactly the predicate, count, width, flag, or position its
        // documentation states.
        if !columns_below(valid_columns, sequence) {
            return Err(Error::shape(
                "compact_batch.valid_columns",
                format!("columns below {sequence}"),
                format!("{valid_columns:?}"),
            ));
        }
        let counts = row_counts(valid_columns);
        let width = widest(&counts);
        if width == 0 {
            return Err(Error::shape(
                "compact_batch.token_counts",
                "at least one row with a valid token",
                "only fully masked rows",
            ));
        }
        let prefix = prefix_rows(valid_columns);
        let index = |values: Vec<i64>, columns: usize| {
            Tensor::<B, 2, Int>::from_data(TensorData::new(values, [batch, columns]), device)
        };
        let (gather, restore) = if prefix {
            (None, None)
        } else {
            let mut sources = Vec::with_capacity(batch * width);
            let mut slots = Vec::with_capacity(batch * sequence);
            for columns in valid_columns {
                if !strictly_increasing(columns) {
                    return Err(Error::shape(
                        "compact_batch.valid_columns",
                        "strictly increasing columns",
                        format!("{columns:?}"),
                    ));
                }
                // The checks above establish the plan's preconditions: slot k
                // reads column c_k, padding slots read a valid column of
                // their row (column 0 for an inactive row), and restoring
                // maps c_k back to k and every other column to the zero slot
                // `width`.
                let plan = compact_row_plan(columns, sequence, width);
                sources.extend(plan.sources.into_iter().map(signed_index));
                slots.extend(plan.slots.into_iter().map(signed_index));
            }
            (Some(index(sources, width)), Some(index(slots, sequence)))
        };
        let valid = slot_validity(&counts, width).map(|rows| {
            Tensor::<B, 2, Bool>::from_data(TensorData::new(rows.concat(), [batch, width]), device)
        });
        let positions = slot_positions(starts, width)
            .ok_or_else(|| {
                Error::configuration(
                    "compact_batch.cache_positions",
                    "a cache position exceeds the signed index range",
                )
            })?
            .concat();
        Ok(Self {
            sequence,
            width,
            counts,
            starts: starts.to_vec(),
            gather,
            restore,
            valid,
            cache_positions: index(positions, width),
        })
    }

    /// Returns the tokens `n_b` every row appends.
    pub(crate) fn counts(&self) -> &[usize] {
        &self.counts
    }

    /// Returns whether every row appends `N` tokens after one shared working length.
    ///
    /// Then every cached key is valid for every row and causality is exactly
    /// the bottom-right rule, so attention needs no explicit mask.
    pub(crate) fn is_uniform(&self) -> bool {
        self.valid.is_none() && all_equal(&self.starts)
    }

    /// Returns the `[B, N]` cache position `W_b + i` of every slot.
    pub(crate) const fn cache_positions(&self) -> &Tensor<B, 2, Int> {
        &self.cache_positions
    }

    /// Returns the `[B, N]` slot validity as a token mask, or `None` without padding.
    pub(crate) fn token_mask(&self) -> Option<TokenMask<B>> {
        self.valid.clone().map(TokenMask::new)
    }

    /// Moves rectangular `[B, S]` IDs (tokens or positions) into compact `[B, N]` slots.
    ///
    /// Padding slots receive `padding_value`. Callers may put arbitrary
    /// sentinels in padded columns (validation only reads valid columns), but
    /// a padding slot is still evaluated, so it must hold an ID the model can
    /// embed, such as token 0.
    pub(crate) fn compact(
        &self,
        rectangular: &Tensor<B, 2, Int>,
        padding_value: i64,
    ) -> Tensor<B, 2, Int> {
        let compact = match &self.gather {
            Some(sources) => rectangular.clone().gather(1, sources.clone()),
            None if self.width == self.sequence => rectangular.clone(),
            None => rectangular.clone().narrow(1, 0, self.width),
        };
        match &self.valid {
            Some(valid) => compact.mask_fill(valid.clone().bool_not(), padding_value),
            None => compact,
        }
    }

    /// Sets every padding slot of compact `[B, N, d]` values to exact zero.
    ///
    /// Padding slots hold finite but meaningless values (or, on backends that
    /// return NaN for a fully excluded attention row, non-finite ones).
    /// Replacing them, rather than multiplying by a mask, removes either.
    pub(crate) fn zero_padding(&self, compact: Tensor<B, 3>) -> Tensor<B, 3> {
        match &self.valid {
            None => compact,
            Some(valid) => {
                let [batch, width, depth] = compact.dims();
                let padding = valid
                    .clone()
                    .bool_not()
                    .reshape([batch, width, 1])
                    .expand([batch, width, depth]);
                compact.mask_fill(padding, 0.0)
            }
        }
    }

    /// Restores compact `[B, N, d]` values to the caller's `[B, S, d]` columns.
    ///
    /// Padding columns, and every column of an inactive row, are exact zeros.
    pub(crate) fn restore(&self, compact: Tensor<B, 3>) -> Tensor<B, 3> {
        let compact = self.zero_padding(compact);
        let [batch, width, depth] = compact.dims();
        let zeros = |columns: usize| {
            Tensor::<B, 3>::zeros(
                [batch, columns, depth],
                (&compact.device(), compact.dtype()),
            )
        };
        match &self.restore {
            Some(slots) => {
                // Slot `N` of `[compact | 0]` is the zero row addressed by padding.
                let padded = Tensor::cat(vec![compact.clone(), zeros(1)], 1);
                padded.gather(
                    1,
                    slots.clone().reshape([batch, self.sequence, 1]).expand([
                        batch,
                        self.sequence,
                        depth,
                    ]),
                )
            }
            None if width == self.sequence => compact,
            None => Tensor::cat(vec![compact.clone(), zeros(self.sequence - width)], 1),
        }
    }

    /// Selects every row's last valid slot `[B, 1, d]`; inactive rows select zeros.
    pub(crate) fn last(&self, compact: Tensor<B, 3>) -> Tensor<B, 3> {
        if self.valid.is_none() {
            return compact.narrow(1, self.width - 1, 1);
        }
        let [batch, _, depth] = compact.dims();
        let device = compact.device();
        // An inactive row reads slot 0, which `zero_padding` cleared.
        let slots = self
            .counts
            .iter()
            .map(|&count| signed_index(count.saturating_sub(1)))
            .collect::<Vec<_>>();
        let slots = Tensor::<B, 2, Int>::from_data(TensorData::new(slots, [batch, 1]), &device)
            .reshape([batch, 1, 1])
            .expand([batch, 1, depth]);
        self.zero_padding(compact).gather(1, slots)
    }

    /// Selects the hidden states whose logits a call requested.
    ///
    /// [`LogitsSelection::Last`] reads the compact block directly; `All` and
    /// explicit positions address caller columns and therefore restore first.
    /// `last_columns` are the caller's last valid columns.
    ///
    /// # Errors
    ///
    /// Returns the errors of the shared selection for an invalid selection.
    pub(crate) fn select(
        &self,
        compact: Tensor<B, 3>,
        selection: &LogitsSelection,
        last_columns: &[usize],
    ) -> Result<Tensor<B, 3>> {
        match selection {
            LogitsSelection::Last => Ok(self.last(compact)),
            other => select_hidden_with_last_columns(self.restore(compact), last_columns, other),
        }
    }
}

/// Per-step state shared by every full-attention layer of one native-KV group.
///
/// Rotary tables, the read plan through the step's own tokens, and the
/// exclusion masks depend only on the step's positions and lengths, not on
/// the layer, so they are built once and reused: the tables when the step
/// starts, each mask (one per distinct sliding window) by the first layer
/// that needs it.
pub(crate) struct CachedAttention<B: Backend> {
    /// Cosine/sine tables `[B, 1, N, rotary]` in the activation dtype.
    rotary: RotaryAngles<B>,
    /// Batched read of `[0, W_b + n_b)` after each layer's write-ahead.
    plan: KvBatchReadPlan<B>,
    /// Native-KV group the layers read and write.
    group: usize,
    /// Exclusion planes `[B, 1, N, K]` (`true` = excluded, broadcast over heads
    /// by attention) per sliding window (`None` = global).
    masks: Vec<(Option<NonZeroUsize>, Tensor<B, 4, Bool>)>,
}

impl<B: Backend> CachedAttention<B> {
    /// Builds the step's rotary tables (frequencies computed once) and read plan.
    ///
    /// `context` must be the call's largest evaluated rotary context; callers
    /// guarantee it lies where the RoPE policy's frequencies are
    /// context-independent, so one table serves every row.
    ///
    /// # Errors
    ///
    /// Returns rotary or cache planning errors.
    pub(crate) fn new(
        rope: RopeConfig,
        context: usize,
        positions: Tensor<B, 2, Int>,
        dtype: DType,
        tensors: &TensorDecodeState<B>,
        writes: &TensorStepWrites<B>,
        group: usize,
    ) -> Result<Self> {
        let frequencies = RotaryFrequencies::new(rope, context)?;
        Ok(Self {
            rotary: rotary_angles_from_frequencies(&frequencies, positions)?.to_dtype(dtype),
            plan: tensors.kv_read_plan_through(writes, group)?,
            group,
            masks: Vec::new(),
        })
    }

    /// Projects, rotates, writes ahead, and attends one full-attention layer.
    ///
    /// The new keys and values land in their cache slots before attention,
    /// which then reads keys and values `[B, kv_heads, K, d]` through one view.
    /// A query at cache position `q` sees keys `k <= q` and, with a sliding
    /// `window`, only `k > q - window`. A uniform batch without a window uses
    /// the backend's bottom-right causal path and no mask. Invalid slots
    /// (beyond a row's `W_b + n_b`) have their values zeroed so stale
    /// non-finite bytes cannot poison a row; padding outputs are zeroed for
    /// the same reason.
    ///
    /// # Errors
    ///
    /// Returns projection, rotary, cache, or attention errors.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn attend(
        &mut self,
        module: &GroupedQueryAttention<B>,
        normalized: Tensor<B, 3>,
        cache_layer: usize,
        window: Option<NonZeroUsize>,
        layout: &CompactBatch<B>,
        tensors: &mut TensorDecodeState<B>,
        writes: &mut TensorStepWrites<B>,
    ) -> Result<Tensor<B, 3>> {
        let mut projection = module.project_self(normalized)?;
        (projection.query, projection.key) =
            apply_rotary(projection.query, projection.key, &self.rotary)?;
        tensors.write_kv_ahead(
            writes,
            self.group,
            cache_layer,
            projection.key.clone(),
            projection.value.clone(),
        )?;
        let segment = tensors
            .read_kv_through(writes, self.group, cache_layer, &self.plan)?
            .materialize()
            .ok_or_else(|| Error::Cache("cached attention step read no key".into()))?;
        // With every row sharing one cursor and width and no window,
        // causality is the bottom-right rule over `[0, W + N)`.
        let causal = window.is_none() && layout.is_uniform();
        let mask = if causal {
            None
        } else if let Some((_, mask)) = self.masks.iter().find(|(known, _)| *known == window) {
            Some(mask.clone())
        } else {
            let mask =
                segment.exclusion_mask(layout.cache_positions(), module.query_heads, window)?;
            self.masks.push((window, mask.clone()));
            Some(mask)
        };
        projection.key = segment.keys().clone();
        projection.value = segment.masked_values();
        let mixed = module.attend(projection, mask, None, causal)?;
        Ok(layout.zero_padding(mixed))
    }
}

/// Rejects an evaluated rotary context that keys already in the cache cannot follow.
///
/// Context-dependent policies (dynamic NTK and LongRoPE) change their
/// frequencies once the evaluated context passes
/// [`RopeScaling::cached_context_limit`](crate::position::RopeScaling::cached_context_limit).
/// Cached keys were rotated with the earlier frequencies, so extending the
/// cache past that limit would silently mix two rotary bases; re-rotating the
/// history is not implemented. Static policies have no limit. Below the
/// limit the frequencies do not depend on the context, which lets one call
/// rotate every row with the tables of its largest context. `family` names
/// the model in the error.
///
/// # Errors
///
/// Returns [`Error::BackendCapability`] for a context beyond the limit.
pub(crate) fn ensure_cacheable_rotary_context(
    rope: RopeConfig,
    context: usize,
    family: &str,
) -> Result<()> {
    match rope.scaling().cached_context_limit() {
        Some(limit) if context > limit => Err(Error::BackendCapability(format!(
            "{family} cached decoding with a context-dependent RoPE policy cannot extend the evaluated context to {context} beyond {limit} without re-rotating cached keys"
        ))),
        _ => Ok(()),
    }
}

#[cfg(test)]
#[path = "batch_tests.rs"]
mod tests;
