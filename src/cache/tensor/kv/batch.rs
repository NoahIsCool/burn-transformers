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
//! and mask tensors once instead of once per layer.

use core::num::NonZeroUsize;
use core::ops::Range;

use burn::tensor::{Bool, Int, Tensor, TensorData, backend::Backend};

use crate::cache::DecodeStateError;
use crate::cache::retention::{WorkingWindow, ring_slot_position, ring_working_window};
use crate::cache::rows::{RowRun, row_runs};

use super::{
    KvCacheStorage, NativeKvCache, NativeKvCacheSpec, PreparedDirectSegment, TensorCacheError,
    TensorCacheResult, validate_plane_count, validate_sequence, validate_tensor,
    validate_write_end,
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
        let columns = counts
            .iter()
            .enumerate()
            .map(|(sequence, &count)| {
                block
                    .checked_sub(count)
                    .ok_or(TensorCacheError::AppendOutsideBlock {
                        sequence,
                        column: 0,
                        count,
                        block,
                    })
            })
            .collect::<TensorCacheResult<Vec<_>>>()?;
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
        for actual in [starts.len(), columns.len()] {
            if actual != counts.len() {
                return Err(DecodeStateError::SequenceCountMismatch {
                    expected: counts.len(),
                    actual,
                }
                .into());
            }
        }
        for (sequence, ((&start, &column), &count)) in
            starts.iter().zip(columns).zip(counts).enumerate()
        {
            if column.checked_add(count).is_none_or(|end| end > block) {
                return Err(TensorCacheError::AppendOutsideBlock {
                    sequence,
                    column,
                    count,
                    block,
                });
            }
            start
                .checked_add(count)
                .ok_or(TensorCacheError::WritePositionOverflow {
                    start,
                    tokens: count,
                })?;
        }
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
        // Construction proved that every sum fits in `usize`.
        self.starts
            .iter()
            .zip(&self.counts)
            .map(|(&start, &count)| start + count)
            .collect()
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
        let mut counts = vec![0; rows];
        counts[sequence] = count;
        let staged = &self.staged_lengths[layer];
        let starts: Vec<usize> = (0..rows)
            .map(|row| {
                let baseline = self.staging_baseline.as_ref().map_or(0, |base| base[row]);
                baseline.saturating_add(staged[row])
            })
            .collect();
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
        let mut staged_updates = Vec::new();
        for (sequence, (&start, &count)) in starts.iter().zip(counts).enumerate() {
            if count == 0 {
                continue;
            }
            if let Some(baseline) = &self.staging_baseline {
                let expected = baseline[sequence] + staged[sequence];
                if start != expected {
                    return Err(TensorCacheError::TransactionCursorMismatch {
                        sequence,
                        cursor: "proposal start",
                        expected,
                        actual: start,
                    });
                }
            }
            let end = staged[sequence]
                .checked_add(count)
                .filter(|&end| end <= self.spec.staging_capacity())
                .ok_or(TensorCacheError::StagingCapacityExceeded {
                    requested: staged[sequence].saturating_add(count),
                    capacity: self.spec.staging_capacity(),
                })?;
            validate_write_end(start, count, self.config.capacity())?;
            staged_updates.push((sequence, end));
        }
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
        // Separate staging is linear: proposal offset `staged_b + j`.
        let runs = row_runs(staged, columns, counts)
            .into_iter()
            .map(|run| PreparedRun {
                rows: run.first_row..run.end_row,
                source_rows: run.first_row - source_offset..run.end_row - source_offset,
                segments: vec![PreparedDirectSegment {
                    input: run.column..run.column + run.count,
                    destinations: core::iter::once(run.start..run.start + run.count).collect(),
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
        match &self.staging_baseline {
            Some(baseline)
                if baseline[sequence] + self.staged_lengths[layer][sequence] != start =>
            {
                Err(TensorCacheError::TransactionCursorMismatch {
                    sequence,
                    cursor: "proposal start",
                    expected: baseline[sequence] + self.staged_lengths[layer][sequence],
                    actual: start,
                })
            }
            _ => Ok(()),
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
    /// Input row `b - source_offset` feeds destination row `b`.
    fn prepare_run(&self, run: RowRun, source_offset: usize) -> TensorCacheResult<PreparedRun> {
        let segments = self
            .prepare_direct_segments(run.start, run.count)?
            .into_iter()
            .map(|segment| PreparedDirectSegment {
                input: segment.input.start + run.column..segment.input.end + run.column,
                destinations: segment.destinations,
            })
            .collect();
        Ok(PreparedRun {
            rows: run.first_row..run.end_row,
            source_rows: run.first_row - source_offset..run.end_row - source_offset,
            segments,
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

/// Per-step plan for reading every layer of one KV group in batch.
///
/// Build it once per step with [`NativeKvCache::plan_batch_read`] or
/// [`NativeKvCache::plan_working_batch_read`], then call
/// [`NativeKvCache::read_batch`] for each layer. The plan records the staged
/// lengths it assumed and every read checks them, so a plan cannot silently
/// read a layer whose proposals have since changed.
#[derive(Debug, Clone)]
pub struct KvBatchReadPlan<B: Backend> {
    /// Visible absolute interval of each row.
    retained: Vec<Range<usize>>,
    /// Planned segments in storage order.
    segments: Vec<PlannedSegment<B>>,
    /// Separate-staging lengths assumed by a working plan.
    staged: Option<Vec<usize>>,
    /// Storage mode of the planned group.
    storage: KvCacheStorage,
    /// Ring width of the planned group.
    window: Option<NonZeroUsize>,
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
            let width = i64::try_from(window.get()).unwrap_or(i64::MAX);
            excluded = excluded.bool_or(key.add_scalar(width).lower_equal(query));
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
    /// and at any time, since committed tensors only change through committed
    /// writes.
    ///
    /// # Errors
    ///
    /// Returns an error for a row-count mismatch, a length beyond logical
    /// capacity, or a position outside Burn's portable `i32` index range.
    pub fn plan_batch_read(&self, lengths: &[usize]) -> TensorCacheResult<KvBatchReadPlan<B>> {
        self.validate_read_lengths(lengths)?;
        let keep: Vec<usize> = match self.window {
            Some(window) => lengths
                .iter()
                .map(|&length| length.min(window.get()))
                .collect(),
            None => lengths.to_vec(),
        };
        let retained = lengths
            .iter()
            .zip(&keep)
            .map(|(&length, &keep)| length - keep..length)
            .collect();
        let segments = self
            .committed_plan_segment(lengths, &keep)?
            .into_iter()
            .collect();
        Ok(self.plan(retained, segments, None))
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
        if let Some(layer) = self
            .staged_lengths
            .iter()
            .position(|lengths| *lengths != staged)
        {
            return Err(TensorCacheError::BatchPlanMismatch {
                layer,
                reason: "staged lengths differ between layers of the group",
            });
        }
        let working = committed
            .iter()
            .zip(&staged)
            .map(|(&length, &proposed)| {
                validate_write_end(length, proposed, self.config.capacity())
            })
            .collect::<TensorCacheResult<Vec<_>>>()?;
        if self.spec.in_place_staging() {
            let mut plan = self.plan_batch_read(&working)?;
            plan.staged = Some(staged);
            return Ok(plan);
        }
        let windows: Vec<WorkingWindow> = committed
            .iter()
            .zip(&staged)
            .map(|(&length, &proposed)| match self.window {
                Some(window) => ring_working_window(length, proposed, window.get()),
                None => WorkingWindow {
                    retained_start: 0,
                    committed_keep: length,
                    staged_skip: 0,
                },
            })
            .collect();
        let retained = windows
            .iter()
            .zip(&working)
            .map(|(window, &end)| window.retained_start..end)
            .collect();
        let keep: Vec<usize> = windows.iter().map(|window| window.committed_keep).collect();
        let mut segments: Vec<PlannedSegment<B>> = self
            .committed_plan_segment(committed, &keep)?
            .into_iter()
            .collect();
        segments.extend(self.staged_plan_segment(committed, &staged, &windows)?);
        Ok(self.plan(retained, segments, Some(staged)))
    }

    /// Reads one layer through a batched read plan.
    ///
    /// Slicing is the only tensor work: a single-segment plan returns views of
    /// the cache allocation, and positions and masks are shared handles.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid layer, a plan built for another
    /// geometry, or a working plan whose staged lengths no longer match.
    pub fn read_batch(
        &self,
        layer: usize,
        plan: &KvBatchReadPlan<B>,
    ) -> TensorCacheResult<KvBatchRead<B>> {
        let cache_layer = self.layer(layer)?;
        if plan.storage != self.spec.storage()
            || plan.window != self.window
            || plan.retained.len() != self.config.sequence_count()
        {
            return Err(TensorCacheError::BatchPlanMismatch {
                layer,
                reason: "the plan was built for a different cache geometry",
            });
        }
        if let Some(staged) = &plan.staged
            && (!self.staging_active || self.staged_lengths[layer] != *staged)
        {
            return Err(TensorCacheError::BatchPlanMismatch {
                layer,
                reason: "staged proposals changed since the plan was built",
            });
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

    /// Assembles a plan stamped with this cache's geometry.
    fn plan(
        &self,
        retained: Vec<Range<usize>>,
        segments: Vec<PlannedSegment<B>>,
        staged: Option<Vec<usize>>,
    ) -> KvBatchReadPlan<B> {
        KvBatchReadPlan {
            retained,
            segments,
            staged,
            storage: self.spec.storage(),
            window: self.window,
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
        match lengths
            .iter()
            .find(|&&length| length > self.config.capacity())
        {
            Some(&length) => Err(TensorCacheError::ReadExceedsCapacity {
                length,
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
                // Slot j holds position j; row b sees positions [L_b - keep_b, L_b).
                let slots = lengths.iter().copied().max().unwrap_or(0);
                if slots == 0 {
                    return Ok(None);
                }
                let positions = Tensor::<B, 1, Int>::arange(0..to_i64(slots)?, &self.device)
                    .reshape([1, slots])
                    .expand([rows, slots]);
                let all_valid = lengths
                    .iter()
                    .zip(keep)
                    .all(|(&length, &keep)| length == slots && keep == length);
                let valid = (!all_valid).then(|| {
                    host_mask(rows, slots, &self.device, |row, slot| {
                        slot < lengths[row] && slot >= lengths[row] - keep[row]
                    })
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
                let slots = lengths.iter().copied().max().unwrap_or(0).min(width);
                if slots == 0 {
                    return Ok(None);
                }
                let mut positions = Vec::with_capacity(rows * slots);
                let mut valid = Vec::with_capacity(rows * slots);
                for (&length, &keep) in lengths.iter().zip(keep) {
                    for slot in 0..slots {
                        let position = ring_slot_position(length, slot, width);
                        let visible = position.is_some_and(|position| position + keep >= length);
                        positions.push(to_i32(position.unwrap_or(0))?);
                        valid.push(visible);
                    }
                }
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
        let slots = staged.iter().copied().max().unwrap_or(0);
        if slots == 0 {
            return Ok(None);
        }
        let mut positions = Vec::with_capacity(rows * slots);
        for &length in committed {
            for slot in 0..slots {
                positions.push(to_i32(length + slot)?);
            }
        }
        let all_valid = staged
            .iter()
            .zip(windows)
            .all(|(&proposed, window)| proposed == slots && window.staged_skip == 0);
        let valid = (!all_valid).then(|| {
            host_mask(rows, slots, &self.device, |row, slot| {
                slot >= windows[row].staged_skip && slot < staged[row]
            })
        });
        Ok(Some(PlannedSegment {
            source: SegmentSource::Staged,
            slots: 0..slots,
            positions: Tensor::from_ints(TensorData::new(positions, [rows, slots]), &self.device),
            valid,
        }))
    }
}

/// Uploads a `[rows, slots]` Bool mask computed on the host.
fn host_mask<B: Backend>(
    rows: usize,
    slots: usize,
    device: &B::Device,
    valid: impl Fn(usize, usize) -> bool,
) -> Tensor<B, 2, Bool> {
    let mask = (0..rows)
        .flat_map(|row| (0..slots).map(move |slot| (row, slot)))
        .map(|(row, slot)| valid(row, slot))
        .collect::<Vec<_>>();
    Tensor::from_bool(TensorData::new(mask, [rows, slots]), device)
}

/// Converts a position to Burn's portable `i32` index type.
fn to_i32(value: usize) -> TensorCacheResult<i32> {
    i32::try_from(value).map_err(|_| TensorCacheError::TensorIndexOverflow {
        axis: "key position",
        index: value,
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
