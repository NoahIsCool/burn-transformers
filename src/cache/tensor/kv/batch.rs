//! Batched native-KV writes and reads that cover every row of a layer.
//!
//! # Writes
//!
//! A model step appends `n_b` tokens to row `b` at absolute positions
//! `[s_b, s_b + n_b)`. Its new keys and values arrive as one
//! `[batch, kv_heads, block, head_dim]` tensor in which row `b`'s tokens occupy
//! columns `[c_b, c_b + n_b)`; [`KvBatchAppend`] records `(s_b, c_b, n_b)`, and
//! the helpers [`KvBatchAppend::right_padded`] and [`KvBatchAppend::left_padded`]
//! cover the two usual padding conventions. Rows with `n_b = 0` are untouched.
//!
//! Adjacent rows with identical `(s_b, c_b, n_b)` form a *run* (the verified
//! `row_runs`); a run is written with one `slice_assign` per physical segment
//! over its whole row range. One-token decode at equal lengths and unpadded
//! prefill are a single run, so a layer costs one assignment (two for a
//! wrapping ring, four for a wrapping mirrored ring) instead of one per row;
//! fully ragged rows fall back to exactly the per-row cost. A scatter was
//! rejected as the general path: Burn's Flex `scatter_nd` copies the whole
//! destination tensor, which is `O(capacity)` per call, whereas
//! `slice_assign` on a uniquely owned cache tensor is in place.
//!
//! # Reads
//!
//! A batched read returns, per storage segment, keys and values
//! `[batch, kv_heads, keys, head_dim]` together with each key's absolute
//! position `[batch, keys]` (Int) and validity `[batch, keys]` (Bool, `true`
//! = valid). Keys are in **slot order**: chronological for full storage and
//! physical ring order for ring storage. That order costs no copy — full
//! storage is the prefix view `[0, max_b L_b)` and a ring is its own slot
//! range — and attention is invariant to permuting keys as long as masks and
//! position-dependent biases are derived from the returned positions (keys are
//! cached after rotary embedding). [`KvBatchSegment::exclusion_mask`] builds
//! the causal, sliding-window, and padding mask from them.
//!
//! During a transaction with separate staging, the proposal tokens form a
//! second segment. Concatenating segments ([`KvBatchRead::materialize`])
//! copies; groups using in-place staging, or reads outside a transaction,
//! need no copy at all.
//!
//! A [`KvBatchReadPlan`] holds the per-step positions and masks. Every layer
//! of a group shares the same lengths within a step, so a model plans once per
//! step and reads each layer through the plan, uploading the small position
//! and mask tensors once instead of once per layer. A plan reads only the
//! group that built it, and a working plan only within the transaction it was
//! built in.
//!
//! # Invalid slots
//!
//! Because every row is read over the same slot range, a row shorter than the
//! longest one, or a ring slot outside a row's window, yields *invalid* keys.
//! Their keys and values are unspecified bytes: zeros, older tokens, rejected
//! speculative proposals, abandoned write-ahead tokens, or rows left behind by
//! a truncate or reorder. Such bytes can be non-finite (an F16 draft that
//! overflowed, say). Excluding a key by *replacing* its score with `-inf`
//! (`mask_fill`) neutralizes a non-finite key, but its value still enters
//! `softmax(scores) · V` with weight `0`, and `0 · NaN = NaN`. Consumers must
//! therefore use [`KvBatchSegment::masked_values`] (or zero invalid values
//! themselves) unless they can rule out non-finite cache contents.

use core::num::NonZeroUsize;
use core::ops::Range;

use burn::tensor::{Bool, Int, Tensor, TensorData, backend::Backend};

use crate::attention::position_limit;
use crate::cache::DecodeStateError;
use crate::cache::identity::IdentityToken;
use crate::cache::retention::WorkingWindow;
use crate::cache::rows::{RowRun, row_runs};
use crate::cache::tensor::plan::{first_index_above, portable_indices};

use super::plan::{committed_read_rows, plan_direct_run, working_read_rows};
use super::{
    NativeKvCache, NativeKvCacheSpec, PreparedDirectSegment, TensorCacheError, TensorCacheResult,
    direct_write_error, validate_plane_count, validate_sequence, validate_tensor,
    validate_write_end,
};

mod layout;
mod slots;

use layout::{
    AppendViolation, StageViolation, append_ends, linear_stage_runs, one_hot_counts,
    plan_stage_updates, proposal_cursor_of, proposal_starts, validate_append_layout,
};
pub(crate) use layout::{left_padded_columns, token_segment_table};
use slots::{
    first_unequal_vector, full_slot_mask, full_slots_all_valid, max_value, ring_slot_table,
    staged_slot_mask, staged_slot_positions, staged_slots_all_valid,
};

// Write layouts.

/// Per-row placement of one step's new tokens in a batched input block.
///
/// Row `b` appends `counts[b]` tokens at absolute cache positions
/// `[starts[b], starts[b] + counts[b])`, read from columns
/// `[columns[b], columns[b] + counts[b])` of a `[batch, heads, block, dim]`
/// input. Interior padding is not representable: callers compact each row's
/// tokens into one contiguous column range first.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvBatchAppend {
    /// Absolute position of each row's first new token.
    starts: Vec<usize>,
    /// Input column of each row's first new token.
    columns: Vec<usize>,
    /// New tokens per row; zero leaves the row untouched.
    counts: Vec<usize>,
    /// Token-axis extent of the input block.
    block: usize,
}

impl KvBatchAppend {
    /// Places each row's tokens in the first `counts[b]` columns (right padding).
    ///
    /// # Errors
    ///
    /// Returns an error for mismatched vector lengths, a row longer than the
    /// block, or a position range that overflows `usize`.
    pub fn right_padded(
        starts: &[usize],
        counts: &[usize],
        block: usize,
    ) -> TensorCacheResult<Self> {
        Self::with_columns(starts, &vec![0; counts.len()], counts, block)
    }

    /// Places each row's tokens in the last `counts[b]` columns (left padding).
    ///
    /// Left padding is the usual layout for batched generation because every
    /// row's newest token then sits in the last column.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`right_padded`](Self::right_padded).
    pub fn left_padded(
        starts: &[usize],
        counts: &[usize],
        block: usize,
    ) -> TensorCacheResult<Self> {
        let columns = left_padded_columns(counts, block).map_err(|sequence| {
            TensorCacheError::AppendOutsideBlock {
                sequence,
                column: 0,
                count: counts[sequence],
                block,
            }
        })?;
        Self::with_columns(starts, &columns, counts, block)
    }

    /// Places each row's tokens at explicit columns.
    ///
    /// # Errors
    ///
    /// Returns an error for mismatched vector lengths, a column range outside
    /// the block, or a position range that overflows `usize`.
    pub fn with_columns(
        starts: &[usize],
        columns: &[usize],
        counts: &[usize],
        block: usize,
    ) -> TensorCacheResult<Self> {
        // The verified validator accepts exactly the layouts whose rows fit
        // the block and the position space; every other method relies on it.
        validate_append_layout(starts, columns, counts, block).map_err(
            |violation| match violation {
                AppendViolation::CountMismatch { expected, actual } => {
                    DecodeStateError::SequenceCountMismatch { expected, actual }.into()
                }
                AppendViolation::OutsideBlock {
                    sequence,
                    column,
                    count,
                } => TensorCacheError::AppendOutsideBlock {
                    sequence,
                    column,
                    count,
                    block,
                },
                AppendViolation::PositionOverflow { start, tokens } => {
                    TensorCacheError::WritePositionOverflow { start, tokens }
                }
            },
        )?;
        Ok(Self {
            starts: starts.to_vec(),
            columns: columns.to_vec(),
            counts: counts.to_vec(),
            block,
        })
    }

    /// Returns the number of rows described.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.counts.len()
    }

    /// Returns each row's first absolute position.
    #[must_use]
    pub fn starts(&self) -> &[usize] {
        &self.starts
    }

    /// Returns each row's first input column.
    #[must_use]
    pub fn columns(&self) -> &[usize] {
        &self.columns
    }

    /// Returns each row's new-token count.
    #[must_use]
    pub fn counts(&self) -> &[usize] {
        &self.counts
    }

    /// Returns the input block's token-axis extent.
    #[must_use]
    pub const fn block(&self) -> usize {
        self.block
    }

    /// Returns each row's exclusive absolute end `start + count`.
    #[must_use]
    pub fn ends(&self) -> Vec<usize> {
        // Construction validated that every sum fits in `usize`.
        append_ends(&self.starts, &self.counts)
    }
}

/// Which allocation a prepared batched write targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BatchTarget {
    /// Committed keys and values (direct writes and in-place staging).
    Committed,
    /// Separate linear proposal buffers.
    Staging,
}

/// One run of rows and its physical destinations.
#[derive(Debug)]
struct PreparedRun {
    /// Destination rows written together.
    rows: Range<usize>,
    /// Matching rows of the input tensor (equal to `rows` for a batched input,
    /// `0..1` for a single-row input).
    source_rows: Range<usize>,
    /// Input column ranges and their destination slot ranges.
    segments: Vec<PreparedDirectSegment>,
}

/// Validated destinations and staged-length updates of a staging write.
#[derive(Debug)]
struct StagePlan {
    /// Allocation receiving the proposals.
    target: BatchTarget,
    /// Row runs with their physical segments.
    runs: Vec<PreparedRun>,
    /// `(row, staged length)` pairs published with the write.
    staged_updates: Vec<(usize, usize)>,
}

impl StagePlan {
    /// Attaches the input tensors, producing the infallible write capability.
    fn into_write<B: Backend>(
        self,
        layer: usize,
        planes: Vec<Tensor<B, 4>>,
    ) -> PreparedNativeKvBatchWrite<B> {
        PreparedNativeKvBatchWrite {
            layer,
            planes,
            target: self.target,
            runs: self.runs,
            staged_updates: self.staged_updates,
        }
    }
}

/// A fully validated batched write; applying it cannot fail.
///
/// The private fields mean only [`NativeKvCache`] can create one, after every
/// shape, device, capacity, and transaction check has passed.
#[derive(Debug)]
pub(crate) struct PreparedNativeKvBatchWrite<B: Backend> {
    /// Destination layer.
    layer: usize,
    /// Input planes `[rows, heads, block, dim]`, one per stored plane.
    planes: Vec<Tensor<B, 4>>,
    /// Allocation receiving the tokens.
    target: BatchTarget,
    /// Row runs with their physical segments.
    runs: Vec<PreparedRun>,
    /// `(row, staged length)` pairs published with a staging write.
    staged_updates: Vec<(usize, usize)>,
}

impl<B: Backend> NativeKvCache<B> {
    /// Writes one layer's committed tokens for every row in one operation.
    ///
    /// See [`KvBatchAppend`] for the input layout. Ring storage keeps only the
    /// newest window of an oversized row, exactly like
    /// [`write_sequence`](Self::write_sequence). Direct writes are forbidden
    /// while speculative staging is active.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid layer, a row-count or tensor
    /// shape/dtype/device mismatch, a write past logical capacity, or an
    /// active transaction. Nothing is written on error.
    pub fn write_batch(
        &mut self,
        layer: usize,
        append: &KvBatchAppend,
        keys: Tensor<B, 4>,
        values: Tensor<B, 4>,
    ) -> TensorCacheResult<()> {
        let planes = self.pair_planes(keys, values)?;
        let write = self.prepare_batch_write(layer, append, planes)?;
        self.apply_batch_writes(vec![write]);
        Ok(())
    }

    /// Writes one layer's committed latents `[batch, block, width]` for every row.
    ///
    /// The latent counterpart of [`write_batch`](Self::write_batch).
    ///
    /// # Errors
    ///
    /// Returns an error for a key/value group or the errors of
    /// [`write_batch`](Self::write_batch).
    pub fn write_latent_batch(
        &mut self,
        layer: usize,
        append: &KvBatchAppend,
        latent: Tensor<B, 3>,
    ) -> TensorCacheResult<()> {
        let planes = self.latent_planes(latent)?;
        let write = self.prepare_batch_write(layer, append, planes)?;
        self.apply_batch_writes(vec![write]);
        Ok(())
    }

    /// Stages one layer's proposal tokens for every row in one operation.
    ///
    /// `append.starts()[b]` must equal the row's working length, i.e. its
    /// committed baseline plus the tokens already staged for this layer; this
    /// cross-check catches a model that skipped or repeated a staging step.
    ///
    /// # Errors
    ///
    /// Returns an error outside a transaction, for a start that disagrees
    /// with the working cursor, for staging- or logical-capacity overflow, or
    /// for invalid tensor metadata. Nothing is staged on error.
    pub fn stage_batch(
        &mut self,
        layer: usize,
        append: &KvBatchAppend,
        keys: Tensor<B, 4>,
        values: Tensor<B, 4>,
    ) -> TensorCacheResult<()> {
        let planes = self.pair_planes(keys, values)?;
        let write = self.prepare_batch_stage(layer, append, planes)?;
        self.apply_batch_writes(vec![write]);
        Ok(())
    }

    /// Stages one layer's proposal latents `[batch, block, width]` for every row.
    ///
    /// The latent counterpart of [`stage_batch`](Self::stage_batch).
    ///
    /// # Errors
    ///
    /// Returns an error for a key/value group or the errors of
    /// [`stage_batch`](Self::stage_batch).
    pub fn stage_latent_batch(
        &mut self,
        layer: usize,
        append: &KvBatchAppend,
        latent: Tensor<B, 3>,
    ) -> TensorCacheResult<()> {
        let planes = self.latent_planes(latent)?;
        let write = self.prepare_batch_stage(layer, append, planes)?;
        self.apply_batch_writes(vec![write]);
        Ok(())
    }

    /// Validates and plans a batched committed write without mutating storage.
    pub(crate) fn prepare_batch_write(
        &self,
        layer: usize,
        append: &KvBatchAppend,
        planes: Vec<Tensor<B, 4>>,
    ) -> TensorCacheResult<PreparedNativeKvBatchWrite<B>> {
        if self.staging_active {
            return Err(TensorCacheError::TransactionActive {
                operation: "write committed KV",
            });
        }
        self.validate_batch_planes(layer, append, &planes)?;
        for (&start, &count) in append.starts.iter().zip(&append.counts) {
            validate_write_end(start, count, self.config.capacity())?;
        }
        let runs = row_runs(&append.starts, &append.columns, &append.counts)
            .into_iter()
            .map(|run| self.prepare_run(run, 0))
            .collect::<TensorCacheResult<_>>()?;
        Ok(PreparedNativeKvBatchWrite {
            layer,
            planes,
            target: BatchTarget::Committed,
            runs,
            staged_updates: Vec::new(),
        })
    }

    /// Validates and plans a batched staging write without mutating storage.
    pub(crate) fn prepare_batch_stage(
        &self,
        layer: usize,
        append: &KvBatchAppend,
        planes: Vec<Tensor<B, 4>>,
    ) -> TensorCacheResult<PreparedNativeKvBatchWrite<B>> {
        if !self.staging_active {
            return Err(TensorCacheError::TransactionNotActive);
        }
        self.validate_batch_planes(layer, append, &planes)?;
        let plan = self.plan_stage(layer, &append.starts, &append.columns, &append.counts, 0)?;
        Ok(plan.into_write(layer, planes))
    }

    /// Validates and plans staging one row's `[1, kv_heads, tokens, head_dim]` proposal.
    ///
    /// This is [`stage_sequence`](Self::stage_sequence) split into a fallible
    /// prepare phase and an infallible apply phase, so a model step can
    /// validate every staging write before publishing any.
    pub(crate) fn prepare_row_stage(
        &self,
        layer: usize,
        sequence: usize,
        planes: Vec<Tensor<B, 4>>,
    ) -> TensorCacheResult<PreparedNativeKvBatchWrite<B>> {
        if !self.staging_active {
            return Err(TensorCacheError::TransactionNotActive);
        }
        let count = self.validate_planes(layer, sequence, &planes)?;
        let rows = self.config.sequence_count();
        // Only row `sequence` is active, so every staging run starts at the
        // source offset `sequence` (verified one-hot counts).
        let counts = one_hot_counts(rows, sequence, count);
        let starts = proposal_starts(
            &self.staged_lengths[layer],
            self.staging_baseline.as_deref(),
            rows,
        );
        let plan = self.plan_stage(layer, &starts, &vec![0; rows], &counts, sequence)?;
        Ok(plan.into_write(layer, planes))
    }

    /// Checks cursors and capacities, then plans the runs of a staging write.
    ///
    /// Input row `b - source_offset` feeds destination row `b`; a batched
    /// input uses offset zero and a single-row input uses its row index.
    fn plan_stage(
        &self,
        layer: usize,
        starts: &[usize],
        columns: &[usize],
        counts: &[usize],
        source_offset: usize,
    ) -> TensorCacheResult<StagePlan> {
        let staged = &self.staged_lengths[layer];
        let staging_capacity = self.spec.staging_capacity();
        let capacity = self.config.capacity();
        // The verified planner checks each active row's cursor, staging
        // capacity, and logical capacity and lists exactly the active rows.
        let staged_updates = plan_stage_updates(
            starts,
            counts,
            staged,
            self.staging_baseline.as_deref(),
            staging_capacity,
            capacity,
        )
        .map_err(|violation| match violation {
            StageViolation::CursorMismatch {
                sequence,
                expected,
                actual,
            } => TensorCacheError::TransactionCursorMismatch {
                sequence,
                cursor: "proposal start",
                expected,
                actual,
            },
            StageViolation::StagingCapacityExceeded { requested } => {
                TensorCacheError::StagingCapacityExceeded {
                    requested,
                    capacity: staging_capacity,
                }
            }
            StageViolation::WriteEnd {
                start,
                tokens,
                violation,
            } => TensorCacheError::write_end(start, tokens, capacity, violation),
        })?;
        if self.spec.in_place_staging() {
            // In-place proposals land at their absolute positions, which the
            // cursor check above tied to `baseline + staged`.
            if self.staging_baseline.is_none() {
                return Err(TensorCacheError::TransactionNotActive);
            }
            let runs = row_runs(starts, columns, counts)
                .into_iter()
                .map(|run| self.prepare_run(run, source_offset))
                .collect::<TensorCacheResult<_>>()?;
            return Ok(StagePlan {
                target: BatchTarget::Committed,
                runs,
                staged_updates,
            });
        }
        self.staged_planes(self.layer(layer)?)?;
        // Separate staging is linear: proposal offset `staged_b + j`. The
        // checks above bound every active row's offsets by the staging
        // capacity, so the verified run planner's preconditions hold.
        let runs = linear_stage_runs(staged, columns, counts, source_offset)
            .into_iter()
            .map(|run| PreparedRun {
                rows: run.rows,
                source_rows: run.source_rows,
                segments: vec![PreparedDirectSegment {
                    input: run.input,
                    destinations: core::iter::once(run.destination).collect(),
                }],
            })
            .collect();
        Ok(StagePlan {
            target: BatchTarget::Staging,
            runs,
            staged_updates,
        })
    }

    /// Applies prepared batched writes; every fallible check already ran.
    pub(crate) fn apply_batch_writes(&mut self, writes: Vec<PreparedNativeKvBatchWrite<B>>) {
        for write in writes {
            let layer = write.layer;
            let updates = self.apply_batch_tensors(write);
            self.record_staged_lengths(layer, &updates);
        }
    }

    /// Performs a prepared write's tensor assignments and returns its staged-length updates.
    ///
    /// Splitting the two lets a write-ahead place proposal tokens before
    /// attention while the staged lengths, which make them part of the
    /// transaction, are only published with the whole model step.
    pub(crate) fn apply_batch_tensors(
        &mut self,
        write: PreparedNativeKvBatchWrite<B>,
    ) -> Vec<(usize, usize)> {
        let spec = self.spec;
        let layer = &mut self.layers[write.layer];
        let targets = match write.target {
            BatchTarget::Committed => &mut layer.planes,
            BatchTarget::Staging => match &mut layer.staged {
                Some(planes) => planes,
                // Preparation proved the staging planes exist.
                None => return write.staged_updates,
            },
        };
        for run in &write.runs {
            for segment in &run.segments {
                for destination in &segment.destinations {
                    for (target, source) in targets.iter_mut().zip(&write.planes) {
                        assign_rows(
                            target,
                            source,
                            run,
                            segment.input.clone(),
                            destination.clone(),
                            spec,
                        );
                    }
                }
            }
        }
        write.staged_updates
    }

    /// Requires row `sequence`'s next proposal to begin at absolute position `start`.
    ///
    /// The working cursor of a row is its committed baseline plus the tokens
    /// staged so far; a mismatch means some layer was staged outside the step
    /// that is being published.
    pub(crate) fn check_proposal_start(
        &self,
        layer: usize,
        sequence: usize,
        start: usize,
    ) -> TensorCacheResult<()> {
        self.layer(layer)?;
        validate_sequence(sequence, self.config.sequence_count())?;
        let Some(baseline) = &self.staging_baseline else {
            return Ok(());
        };
        let expected = proposal_cursor_of(baseline[sequence], self.staged_lengths[layer][sequence]);
        if expected == start {
            Ok(())
        } else {
            Err(TensorCacheError::TransactionCursorMismatch {
                sequence,
                cursor: "proposal start",
                expected,
                actual: start,
            })
        }
    }

    /// Rechecks a write-ahead at publication and returns its staged-length updates.
    ///
    /// The cursor, staging capacity, and logical capacity are validated
    /// exactly as for [`stage_batch`](Self::stage_batch).
    pub(crate) fn staged_updates_for(
        &self,
        layer: usize,
        starts: &[usize],
        counts: &[usize],
    ) -> TensorCacheResult<Vec<(usize, usize)>> {
        self.layer(layer)?;
        let plan = self.plan_stage(layer, starts, &vec![0; counts.len()], counts, 0)?;
        Ok(plan.staged_updates)
    }

    /// Publishes `(row, staged length)` updates for one layer.
    pub(crate) fn record_staged_lengths(&mut self, layer: usize, updates: &[(usize, usize)]) {
        for &(sequence, staged) in updates {
            self.staged_lengths[layer][sequence] = staged;
        }
    }

    /// Plans the physical segments of one run, shifting inputs to the run's column.
    ///
    /// Input row `b - source_offset` feeds destination row `b`. Every run
    /// comes from `row_runs`, so it holds only active rows; a batched input
    /// uses offset zero and a single-row input's one-hot counts make its row
    /// the only active one, so every run starts at or after `source_offset`.
    /// Validated layouts keep each run's columns inside the input block.
    fn prepare_run(&self, run: RowRun, source_offset: usize) -> TensorCacheResult<PreparedRun> {
        let storage = self.spec.storage();
        // The Verus-verified planner shifts the inputs and rows and writes
        // each token to its storage slot.
        let plan = plan_direct_run(
            run,
            source_offset,
            storage,
            self.window.map(NonZeroUsize::get),
        )
        .map_err(|violation| {
            direct_write_error(
                storage,
                run.start,
                run.count,
                self.config.capacity(),
                violation,
            )
        })?;
        Ok(PreparedRun {
            rows: plan.rows,
            source_rows: plan.source_rows,
            segments: plan
                .segments
                .into_iter()
                .map(PreparedDirectSegment::from)
                .collect(),
        })
    }

    /// Validates batched input planes `[rows, heads, block, dim]` against the layer and layout.
    fn validate_batch_planes(
        &self,
        layer: usize,
        append: &KvBatchAppend,
        planes: &[Tensor<B, 4>],
    ) -> TensorCacheResult<()> {
        self.layer(layer)?;
        let rows = self.config.sequence_count();
        if append.rows() != rows {
            return Err(TensorCacheError::BatchSizeMismatch {
                configured: rows,
                requested: append.rows(),
            });
        }
        validate_plane_count(self.spec, planes.len())?;
        let expected = [
            rows,
            self.spec.key_value_heads(),
            append.block(),
            self.spec.head_dimension(),
        ];
        for (plane, name) in planes.iter().zip(["batched keys", "batched values"]) {
            validate_tensor(plane, name, expected, self.spec.dtype(), &self.device)?;
        }
        Ok(())
    }
}

/// Assigns one input column range of a run's rows to one destination slot range.
fn assign_rows<B: Backend>(
    destination: &mut Tensor<B, 4>,
    source: &Tensor<B, 4>,
    run: &PreparedRun,
    input: Range<usize>,
    physical: Range<usize>,
    spec: NativeKvCacheSpec,
) {
    let heads = 0..spec.key_value_heads();
    let channels = 0..spec.head_dimension();
    let values = source.clone().slice([
        run.source_rows.clone(),
        heads.clone(),
        input,
        channels.clone(),
    ]);
    destination.inplace(|tensor| {
        tensor.slice_assign([run.rows.clone(), heads, physical, channels], values)
    });
}

// Read plans.

/// Which allocation a read segment comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SegmentSource {
    /// Committed keys and values.
    Committed,
    /// Separate linear proposal buffers.
    Staged,
}

/// One planned segment: a slot range plus per-key positions and validity.
#[derive(Debug, Clone)]
struct PlannedSegment<B: Backend> {
    /// Allocation to slice.
    source: SegmentSource,
    /// Slot range sliced from every row.
    slots: Range<usize>,
    /// Absolute key positions `[batch, slots]`; arbitrary where invalid.
    positions: Tensor<B, 2, Int>,
    /// Key validity `[batch, slots]`, or `None` when every key is valid.
    valid: Option<Tensor<B, 2, Bool>>,
}

/// The transaction state a working read plan was built against.
#[derive(Debug, Clone, PartialEq, Eq)]
struct PlannedStaging {
    /// The group's transaction counter when the plan was built.
    epoch: u64,
    /// Proposal lengths per row that every layer read must still match.
    lengths: Vec<usize>,
}

/// Per-step plan for reading every layer of one KV group in batch.
///
/// Build it once per step with [`NativeKvCache::plan_batch_read`] or
/// [`NativeKvCache::plan_working_batch_read`], then call
/// [`NativeKvCache::read_batch`] for each layer. A plan is stamped with the
/// identity of the group that built it and is rejected by every other group,
/// even one with identical geometry, because staging mode and contents
/// differ between groups. A working plan also records its transaction and
/// staged lengths, and every read checks them, so it cannot read a layer
/// whose proposals have since changed or a later transaction's proposals.
///
/// Committed positions are a function of the lengths the plan was built for.
/// Full storage keeps every position at its own slot, so such a plan stays
/// correct while those prefixes are unchanged; a ring's slots are reused, so
/// a ring plan is only valid until the group's next committed write. Plan
/// after the lengths change (in practice, once per step).
#[derive(Debug, Clone)]
pub struct KvBatchReadPlan<B: Backend> {
    /// Visible absolute interval of each row.
    retained: Vec<Range<usize>>,
    /// Planned segments in storage order.
    segments: Vec<PlannedSegment<B>>,
    /// Identity of the group that built the plan.
    source: IdentityToken,
    /// Transaction and staged lengths assumed by a working plan; `None` for committed plans.
    staging: Option<PlannedStaging>,
}

impl<B: Backend> KvBatchReadPlan<B> {
    /// Returns each row's visible absolute key interval.
    #[must_use]
    pub fn retained_positions(&self) -> &[Range<usize>] {
        &self.retained
    }

    /// Returns the number of key slots each layer read will return.
    #[must_use]
    pub fn key_slots(&self) -> usize {
        self.segments
            .iter()
            .map(|segment| segment.slots.len())
            .sum()
    }
}

/// One batched layer read.
#[derive(Debug, Clone)]
pub struct KvBatchRead<B: Backend> {
    /// Visible absolute interval of each row.
    retained: Vec<Range<usize>>,
    /// Segments in storage order.
    segments: Vec<KvBatchSegment<B>>,
}

/// Keys, values, positions, and validity for one storage segment.
///
/// Keys and values of invalid entries are unspecified and may be non-finite;
/// see the module documentation on invalid slots and
/// [`masked_values`](Self::masked_values).
#[derive(Debug, Clone)]
pub struct KvBatchSegment<B: Backend> {
    /// Keys `[batch, kv_heads, keys, head_dim]`.
    keys: Tensor<B, 4>,
    /// Values with the same geometry.
    values: Tensor<B, 4>,
    /// Absolute key positions `[batch, keys]`.
    positions: Tensor<B, 2, Int>,
    /// Key validity `[batch, keys]` (`true` = valid), or `None` if all are valid.
    valid: Option<Tensor<B, 2, Bool>>,
}

impl<B: Backend> KvBatchRead<B> {
    /// Appends another segment, such as a step's new tokens, after the cached ones.
    ///
    /// # Errors
    ///
    /// Returns an error when the segment's batch size differs from the read's.
    pub fn with_segment(mut self, segment: KvBatchSegment<B>) -> TensorCacheResult<Self> {
        let rows = segment.positions.dims()[0];
        if rows != self.retained.len() {
            return Err(TensorCacheError::BatchSizeMismatch {
                configured: self.retained.len(),
                requested: rows,
            });
        }
        self.segments.push(segment);
        Ok(self)
    }

    /// Returns each row's visible absolute key interval.
    #[must_use]
    pub fn retained_positions(&self) -> &[Range<usize>] {
        &self.retained
    }

    /// Returns the segments in storage order.
    #[must_use]
    pub fn segments(&self) -> &[KvBatchSegment<B>] {
        &self.segments
    }

    /// Joins the segments along the key axis.
    ///
    /// One segment is returned as is (no copy). Two segments, which only
    /// separate staging during a transaction produces, are concatenated.
    /// Returns `None` when no row has a visible key.
    #[must_use]
    pub fn materialize(&self) -> Option<KvBatchSegment<B>> {
        match self.segments.as_slice() {
            [] => None,
            [segment] => Some(segment.clone()),
            segments => Some(concatenate(segments)),
        }
    }
}

impl<B: Backend> KvBatchSegment<B> {
    /// Assembles a segment from validated parts.
    pub(crate) const fn from_parts(
        keys: Tensor<B, 4>,
        values: Tensor<B, 4>,
        positions: Tensor<B, 2, Int>,
        valid: Option<Tensor<B, 2, Bool>>,
    ) -> Self {
        Self {
            keys,
            values,
            positions,
            valid,
        }
    }

    /// Borrows keys `[batch, kv_heads, keys, head_dim]`.
    #[must_use]
    pub const fn keys(&self) -> &Tensor<B, 4> {
        &self.keys
    }

    /// Borrows values `[batch, kv_heads, keys, head_dim]`.
    #[must_use]
    pub const fn values(&self) -> &Tensor<B, 4> {
        &self.values
    }

    /// Returns a latent group's keys as `[batch, keys, width]` (a free reshape).
    #[must_use]
    pub fn latent(&self) -> Tensor<B, 3> {
        let [batch, _, keys, width] = self.keys.dims();
        self.keys.clone().reshape([batch, keys, width])
    }

    /// Borrows absolute key positions `[batch, keys]`; entries of invalid keys are arbitrary.
    #[must_use]
    pub const fn key_positions(&self) -> &Tensor<B, 2, Int> {
        &self.positions
    }

    /// Returns values with every invalid key's entries set to zero.
    ///
    /// Attention computes `softmax(scores) · V`; an excluded key has weight
    /// `0`, but `0 · NaN` and `0 · ∞` are `NaN`, so a non-finite stale value in
    /// an invalid slot would poison its row. Zeroing makes the product
    /// exactly `0`. When every key is valid the values are returned as is (a
    /// shared handle, no copy); otherwise this writes one masked copy of the
    /// values `[batch, kv_heads, keys, head_dim]`.
    #[must_use]
    pub fn masked_values(&self) -> Tensor<B, 4> {
        let Some(valid) = &self.valid else {
            return self.values.clone();
        };
        let [batch, heads, keys, channels] = self.values.dims();
        let invalid = valid
            .clone()
            .bool_not()
            .reshape([batch, 1, keys, 1])
            .expand([batch, heads, keys, channels]);
        self.values.clone().mask_fill(invalid, 0.0)
    }

    /// Borrows key validity `[batch, keys]` (`true` = valid), or `None` if all are valid.
    #[must_use]
    pub const fn key_valid(&self) -> Option<&Tensor<B, 2, Bool>> {
        self.valid.as_ref()
    }

    /// Builds the attention exclusion mask `[batch, heads, queries, keys]` (`true` = excluded).
    ///
    /// With query positions `q` and key positions `k`, a key is excluded when
    /// it is invalid, lies in the future (`k > q`), or, given a window `W`,
    /// has left the window (`k + W <= q`, i.e. only `q - W < k <= q` remain).
    /// A query with no remaining key yields a fully excluded row; callers must
    /// handle such rows (for example inactive batch rows) explicitly.
    ///
    /// Apply the mask by replacing scores (`mask_fill` with `-inf`), never by
    /// adding a bias: invalid keys may be non-finite, and `NaN + (-inf)` stays
    /// `NaN`. Values need [`masked_values`](Self::masked_values) as well.
    ///
    /// # Errors
    ///
    /// Returns an error when `query_positions` is not `[batch, queries]` with
    /// this segment's batch size, is on another device, or `heads` is zero.
    pub fn exclusion_mask(
        &self,
        query_positions: &Tensor<B, 2, Int>,
        heads: usize,
        window: Option<NonZeroUsize>,
    ) -> TensorCacheResult<Tensor<B, 4, Bool>> {
        let [batch, keys] = self.positions.dims();
        let [query_batch, queries] = query_positions.dims();
        if query_batch != batch || queries == 0 {
            return Err(TensorCacheError::ShapeMismatch {
                tensor: "query positions",
                expected: vec![batch, queries.max(1)],
                actual: vec![query_batch, queries],
            });
        }
        if heads == 0 {
            return Err(TensorCacheError::ZeroDimension { field: "heads" });
        }
        if query_positions.device() != self.positions.device() {
            return Err(TensorCacheError::DeviceMismatch {
                tensor: "query positions",
            });
        }
        let shape = [batch, 1, queries, keys];
        let key = self
            .positions
            .clone()
            .reshape([batch, 1, 1, keys])
            .expand(shape);
        let query = query_positions
            .clone()
            .reshape([batch, 1, queries, 1])
            .expand(shape);
        let mut excluded = key.clone().greater(query.clone());
        if let Some(window) = window {
            // Compare the distance `query - key` with the window rather than
            // forming `key + window`: backends convert scalars to their integer
            // element type without range checks, so a wide window wrapped
            // negative and excluded every key. Both positions are
            // non-negative, so the difference cannot overflow, and a window
            // wider than any representable distance excludes nothing.
            let width = i64::try_from(window.get()).unwrap_or(i64::MAX);
            if width <= position_limit(self.positions.dtype()) {
                excluded = excluded.bool_or(query.sub(key).greater_equal_elem(width));
            }
        }
        if let Some(valid) = &self.valid {
            excluded = excluded.bool_or(
                valid
                    .clone()
                    .bool_not()
                    .reshape([batch, 1, 1, keys])
                    .expand(shape),
            );
        }
        Ok(excluded.expand([batch, heads, queries, keys]))
    }
}

/// Concatenates segments along the key axis, filling validity where it was implicit.
fn concatenate<B: Backend>(segments: &[KvBatchSegment<B>]) -> KvBatchSegment<B> {
    let keys = Tensor::cat(
        segments
            .iter()
            .map(|segment| segment.keys.clone())
            .collect(),
        2,
    );
    let values = Tensor::cat(
        segments
            .iter()
            .map(|segment| segment.values.clone())
            .collect(),
        2,
    );
    let positions = Tensor::cat(
        segments
            .iter()
            .map(|segment| segment.positions.clone())
            .collect(),
        1,
    );
    let valid = segments
        .iter()
        .any(|segment| segment.valid.is_some())
        .then(|| {
            Tensor::cat(
                segments
                    .iter()
                    .map(|segment| {
                        segment.valid.clone().unwrap_or_else(|| {
                            // An implicit all-valid mask becomes explicit:
                            // `0 == 0` builds an all-true Bool tensor on the
                            // device without uploading host data.
                            let [batch, keys] = segment.positions.dims();
                            Tensor::<B, 2, Int>::zeros([batch, keys], &segment.positions.device())
                                .equal_elem(0)
                        })
                    })
                    .collect(),
                1,
            )
        });
    KvBatchSegment {
        keys,
        values,
        positions,
        valid,
    }
}

impl<B: Backend> NativeKvCache<B> {
    /// Plans a batched read of each row's committed window ending at `lengths[b]`.
    ///
    /// Full storage reads the prefix view `[0, max_b L_b)`; ring storage reads
    /// its first `min(W, max_b L_b)` physical slots. Keys beyond a row's
    /// visible window are marked invalid. The plan is valid for every layer
    /// of this group while the committed contents of `[0, L_b)` are unchanged
    /// (for a ring: until its next committed write); see [`KvBatchReadPlan`].
    ///
    /// # Errors
    ///
    /// Returns an error for a row-count mismatch, a length beyond logical
    /// capacity, or a position outside Burn's portable `i32` index range.
    pub fn plan_batch_read(&self, lengths: &[usize]) -> TensorCacheResult<KvBatchReadPlan<B>> {
        self.validate_read_lengths(lengths)?;
        // The verified planner keeps `min(L_b, W)` (or `L_b`) positions per row.
        let rows = committed_read_rows(lengths, self.window.map(NonZeroUsize::get));
        let segments = self
            .committed_plan_segment(lengths, &rows.keep)?
            .into_iter()
            .collect();
        Ok(self.plan(rows.retained, segments, None))
    }

    /// Plans a batched read of each row's working window during speculation.
    ///
    /// `committed` must equal the transaction baseline. The visible interval
    /// of row `b` is the newest `min(C_b + P_b, W)` positions of its committed
    /// tokens followed by its `P_b` staged proposals (the verified working
    /// window). In-place groups return one committed segment; separate staging
    /// adds a proposal segment. Outside a transaction this equals
    /// [`plan_batch_read`](Self::plan_batch_read).
    ///
    /// # Errors
    ///
    /// Returns an error for a baseline mismatch, staged lengths that differ
    /// between layers, or the errors of [`plan_batch_read`](Self::plan_batch_read).
    pub fn plan_working_batch_read(
        &self,
        committed: &[usize],
    ) -> TensorCacheResult<KvBatchReadPlan<B>> {
        if !self.staging_active {
            return self.plan_batch_read(committed);
        }
        self.validate_read_lengths(committed)?;
        for (sequence, &length) in committed.iter().enumerate() {
            self.validate_baseline(sequence, length)?;
        }
        let staged = self.staged_lengths[0].clone();
        if let Some(layer) = first_unequal_vector(&self.staged_lengths, &staged) {
            return Err(TensorCacheError::BatchPlanMismatch {
                layer,
                reason: "staged lengths differ between layers of the group",
            });
        }
        let capacity = self.config.capacity();
        // The verified planner checks every working end and splits each
        // row's window into committed and staged parts.
        let rows = working_read_rows(
            committed,
            &staged,
            self.window.map(NonZeroUsize::get),
            capacity,
        )
        .map_err(|(row, violation)| {
            TensorCacheError::write_end(committed[row], staged[row], capacity, violation)
        })?;
        let staging = PlannedStaging {
            epoch: self.staging_epoch,
            lengths: staged.clone(),
        };
        if self.spec.in_place_staging() {
            let mut plan = self.plan_batch_read(&rows.ends)?;
            plan.staging = Some(staging);
            return Ok(plan);
        }
        let mut segments: Vec<PlannedSegment<B>> = self
            .committed_plan_segment(committed, &rows.keep)?
            .into_iter()
            .collect();
        segments.extend(self.staged_plan_segment(committed, &staged, &rows.windows)?);
        Ok(self.plan(rows.retained, segments, Some(staging)))
    }

    /// Reads one layer through a batched read plan.
    ///
    /// Slicing is the only tensor work: a single-segment plan returns views of
    /// the cache allocation, and positions and masks are shared handles.
    /// Keys and values of invalid entries are unspecified (see the module
    /// documentation on invalid slots).
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid layer, a plan built by another group,
    /// or a working plan from another transaction or whose staged lengths no
    /// longer match.
    pub fn read_batch(
        &self,
        layer: usize,
        plan: &KvBatchReadPlan<B>,
    ) -> TensorCacheResult<KvBatchRead<B>> {
        let cache_layer = self.layer(layer)?;
        if plan.source != self.identity {
            return Err(TensorCacheError::BatchPlanMismatch {
                layer,
                reason: "the plan was built by another cache group",
            });
        }
        if let Some(staging) = &plan.staging {
            if !self.staging_active || staging.epoch != self.staging_epoch {
                return Err(TensorCacheError::BatchPlanMismatch {
                    layer,
                    reason: "the plan was built in another transaction",
                });
            }
            if self.staged_lengths[layer] != staging.lengths {
                return Err(TensorCacheError::BatchPlanMismatch {
                    layer,
                    reason: "staged proposals changed since the plan was built",
                });
            }
        }
        let rows = 0..self.config.sequence_count();
        let heads = 0..self.spec.key_value_heads();
        let channels = 0..self.spec.head_dimension();
        let segments = plan
            .segments
            .iter()
            .map(|segment| {
                let planes = match segment.source {
                    SegmentSource::Committed => cache_layer.planes.as_slice(),
                    SegmentSource::Staged => self.staged_planes(cache_layer)?,
                };
                let slices = [
                    rows.clone(),
                    heads.clone(),
                    segment.slots.clone(),
                    channels.clone(),
                ];
                let keys = planes[0].clone().slice(slices.clone());
                // A latent group's single plane serves as both keys and values.
                let values = planes
                    .get(1)
                    .map_or_else(|| keys.clone(), |plane| plane.clone().slice(slices));
                Ok(KvBatchSegment {
                    keys,
                    values,
                    positions: segment.positions.clone(),
                    valid: segment.valid.clone(),
                })
            })
            .collect::<TensorCacheResult<_>>()?;
        Ok(KvBatchRead {
            retained: plan.retained.clone(),
            segments,
        })
    }

    /// Assembles a plan stamped with this group's identity.
    fn plan(
        &self,
        retained: Vec<Range<usize>>,
        segments: Vec<PlannedSegment<B>>,
        staging: Option<PlannedStaging>,
    ) -> KvBatchReadPlan<B> {
        KvBatchReadPlan {
            retained,
            segments,
            source: self.identity.clone(),
            staging,
        }
    }

    /// Checks one length per row, each within logical capacity.
    fn validate_read_lengths(&self, lengths: &[usize]) -> TensorCacheResult<()> {
        if lengths.len() != self.config.sequence_count() {
            return Err(DecodeStateError::SequenceCountMismatch {
                expected: self.config.sequence_count(),
                actual: lengths.len(),
            }
            .into());
        }
        match first_index_above(lengths, self.config.capacity()) {
            Some(row) => Err(TensorCacheError::ReadExceedsCapacity {
                length: lengths[row],
                capacity: self.config.capacity(),
            }),
            None => Ok(()),
        }
    }

    /// Plans the committed segment that keeps the newest `keep[b]` positions before `lengths[b]`.
    fn committed_plan_segment(
        &self,
        lengths: &[usize],
        keep: &[usize],
    ) -> TensorCacheResult<Option<PlannedSegment<B>>> {
        let rows = lengths.len();
        match self.window {
            None => {
                // Slot j holds position j; row b sees positions [L_b - keep_b, L_b)
                // (the verified full-storage validity table).
                let slots = max_value(lengths);
                if slots == 0 {
                    return Ok(None);
                }
                let positions = Tensor::<B, 1, Int>::arange(0..to_i64(slots)?, &self.device)
                    .reshape([1, slots])
                    .expand([rows, slots]);
                let valid = (!full_slots_all_valid(lengths, keep, slots)).then(|| {
                    Tensor::from_bool(
                        TensorData::new(full_slot_mask(lengths, keep, slots), [rows, slots]),
                        &self.device,
                    )
                });
                Ok(Some(PlannedSegment {
                    source: SegmentSource::Committed,
                    slots: 0..slots,
                    positions,
                    valid,
                }))
            }
            Some(window) => {
                let width = window.get();
                let slots = max_value(lengths).min(width);
                if slots == 0 {
                    return Ok(None);
                }
                // The verified table gives every slot's newest position and
                // whether it lies in the row's visible window.
                let (positions, valid) = ring_slot_table(lengths, keep, width, slots);
                let positions = key_positions(&positions)?;
                let all_valid = valid.iter().all(|&visible| visible);
                Ok(Some(PlannedSegment {
                    source: SegmentSource::Committed,
                    slots: 0..slots,
                    positions: Tensor::from_ints(
                        TensorData::new(positions, [rows, slots]),
                        &self.device,
                    ),
                    valid: (!all_valid).then(|| {
                        Tensor::from_bool(TensorData::new(valid, [rows, slots]), &self.device)
                    }),
                }))
            }
        }
    }

    /// Plans the separate-staging segment: proposal `j` of row `b` is position `C_b + j`.
    fn staged_plan_segment(
        &self,
        committed: &[usize],
        staged: &[usize],
        windows: &[WorkingWindow],
    ) -> TensorCacheResult<Option<PlannedSegment<B>>> {
        let rows = committed.len();
        let slots = max_value(staged);
        if slots == 0 {
            return Ok(None);
        }
        // Proposal `j` of row `b` is position `C_b + j`, valid from the row's
        // skipped proposals up to its staged length (verified tables).
        let positions = key_positions(&staged_slot_positions(committed, slots))?;
        let valid = (!staged_slots_all_valid(staged, windows, slots)).then(|| {
            Tensor::from_bool(
                TensorData::new(staged_slot_mask(staged, windows, slots), [rows, slots]),
                &self.device,
            )
        });
        Ok(Some(PlannedSegment {
            source: SegmentSource::Staged,
            slots: 0..slots,
            positions: Tensor::from_ints(TensorData::new(positions, [rows, slots]), &self.device),
            valid,
        }))
    }
}

/// Converts key positions to Burn's portable `i32` index type (the verified `portable_indices`).
fn key_positions(positions: &[usize]) -> TensorCacheResult<Vec<i32>> {
    portable_indices(positions).map_err(|index| TensorCacheError::TensorIndexOverflow {
        axis: "key position",
        index: positions[index],
    })
}

/// Converts a slot count to the `i64` range accepted by `arange`.
fn to_i64(value: usize) -> TensorCacheResult<i64> {
    i64::try_from(value).map_err(|_| TensorCacheError::TensorIndexOverflow {
        axis: "key slot",
        index: value,
    })
}

#[cfg(test)]
#[path = "batch_tests.rs"]
mod tests;
