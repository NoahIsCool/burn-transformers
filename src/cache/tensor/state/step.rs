//! One model step's cache updates, validated and published as a unit.
//!
//! A cached forward pass appends `n_b` tokens to every row `b` of one bank
//! (the main decoder or the MTP stack). Every layer of every cache group must
//! record exactly the tokens of that step, and the logical lengths may only
//! advance once all of them are known to be valid. [`TensorStepWrites`]
//! collects the step's per-row or batched KV tensors and recurrent states;
//! [`TensorDecodeState::publish_step`] then
//!
//! 1. checks *coverage*: each `(group, layer, active row)` is written exactly
//!    once, inactive rows (`n_b = 0`) are never written, and each tensor's
//!    token extent equals `n_b`;
//! 2. plans the length advance with the verified ledger (capacity, overflow);
//! 3. prepares every tensor write — direct writes outside a transaction, or
//!    proposal staging inside one — so every shape, device, cursor, staging
//!    capacity, and ring-planning error surfaces here;
//! 4. publishes the lengths and applies the prepared writes, which cannot
//!    fail.
//!
//! A rejected step therefore changes nothing (with one documented exception,
//! write-ahead, below). This replaces the per-family `PendingKvWrite` structs
//! and coverage checks, works for any number of KV and recurrent groups, and
//! makes speculative staging all-or-nothing per step.
//!
//! # Write-ahead
//!
//! For full-storage groups, [`TensorDecodeState::write_kv_ahead`] writes a
//! layer's new tokens into slots `[L_b, L_b + n_b)` *before* attention, so
//! the layer can attend over one zero-copy view `[0, L_b + n_b)` instead of
//! concatenating its whole history with the new tokens. Those slots are
//! unreachable until the step is published, so a failed step keeps *logical*
//! atomicity: lengths and every readable position are unchanged, while slots
//! beyond the lengths may hold the abandoned tokens (later writes overwrite
//! them). Rings cannot write ahead, because a wrapped write would overwrite a
//! still-visible slot. Inside a transaction only in-place groups can.

use core::ops::Range;

use burn::tensor::{Bool, Int, Tensor, TensorData, backend::Backend};

use crate::cache::SequenceLengths;
use crate::cache::tensor::kv::{
    KvBatchAppend, KvBatchRead, KvBatchReadPlan, KvBatchSegment, KvCacheRead,
};
use crate::cache::tensor::kv::{PreparedNativeKvBatchWrite, PreparedNativeKvWrite};
use crate::cache::tensor::recurrent::{
    PreparedRecurrentBatchWrite, PreparedRecurrentStage, PreparedRecurrentStateWrite,
    RecurrentStateRead,
};

use super::{
    KvCacheStorage, TensorCacheBank, TensorCacheError, TensorCacheResult, TensorDecodeState,
    validate_count,
};

/// Bank whose groups a step writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StepBank {
    /// The decoder's ordinary cache groups and committed cursor.
    Main,
    /// The auxiliary multi-token-prediction groups and their own cursor.
    Mtp,
}

/// Where each row's new tokens sit on a batched input's token axis.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenAlignment {
    /// Row `b` occupies the first `n_b` columns (padding on the right).
    RightPadded,
    /// Row `b` occupies the last `n_b` columns (padding on the left).
    LeftPadded,
    /// Row `b` starts at the given column.
    Columns(Vec<usize>),
}

/// One recorded KV update.
#[derive(Debug)]
enum KvEntry<B: Backend> {
    /// One row's planes `[1, heads, n_b, dim]` (keys and values, or one latent).
    Row {
        /// Cache group.
        group: usize,
        /// Layer within the group.
        layer: usize,
        /// Batch row.
        row: usize,
        /// New planes in storage order.
        planes: Vec<Tensor<B, 4>>,
    },
    /// Every row's planes `[batch, heads, block, dim]`.
    Batch {
        /// Cache group.
        group: usize,
        /// Layer within the group.
        layer: usize,
        /// New planes in storage order.
        planes: Vec<Tensor<B, 4>>,
    },
    /// A layer already written ahead at the recorded starts.
    WrittenAhead {
        /// Cache group.
        group: usize,
        /// Layer within the group.
        layer: usize,
        /// Working lengths when the tokens were written.
        starts: Vec<usize>,
    },
}

/// One recorded recurrent update.
#[derive(Debug)]
enum RecurrentEntry<B: Backend> {
    /// One row's terminal state after all of its new tokens.
    RowTerminal {
        /// Cache group.
        group: usize,
        /// Layer within the group.
        layer: usize,
        /// Batch row.
        row: usize,
        /// Matrix `[1, heads, rows, columns]`.
        matrix: Tensor<B, 4>,
        /// Convolution history `[1, channels, width]`.
        convolution: Tensor<B, 3>,
    },
    /// One row's state after each new token.
    RowSnapshots {
        /// Cache group.
        group: usize,
        /// Layer within the group.
        layer: usize,
        /// Batch row.
        row: usize,
        /// Matrices `[1, n_b, heads, rows, columns]`.
        matrices: Tensor<B, 5>,
        /// Histories `[1, n_b, channels, width]`.
        convolutions: Tensor<B, 4>,
    },
    /// Every row's terminal state; inactive rows are ignored.
    BatchTerminal {
        /// Cache group.
        group: usize,
        /// Layer within the group.
        layer: usize,
        /// Matrices `[batch, heads, rows, columns]`.
        matrix: Tensor<B, 4>,
        /// Histories `[batch, channels, width]`.
        convolution: Tensor<B, 3>,
    },
}

/// The pending cache updates of one model step for one bank.
///
/// Create it with [`TensorDecodeState::step`] (or
/// [`mtp_step`](TensorDecodeState::mtp_step)), record each layer's new keys,
/// values, and recurrent states while the forward pass runs, then hand it to
/// [`TensorDecodeState::publish_step`]. Recording never fails; every check
/// happens at publication, before anything changes.
#[derive(Debug)]
pub struct TensorStepWrites<B: Backend> {
    /// Bank receiving the step.
    bank: StepBank,
    /// New tokens per row; zero marks an inactive row.
    counts: Vec<usize>,
    /// Token placement of batched inputs.
    alignment: TokenAlignment,
    /// Recorded KV updates.
    kv: Vec<KvEntry<B>>,
    /// Recorded recurrent updates.
    recurrent: Vec<RecurrentEntry<B>>,
}

impl<B: Backend> TensorStepWrites<B> {
    /// Returns the bank this step writes.
    #[must_use]
    pub const fn bank(&self) -> StepBank {
        self.bank
    }

    /// Returns the new-token count of every row.
    #[must_use]
    pub fn token_counts(&self) -> &[usize] {
        &self.counts
    }

    /// Sets where batched inputs place each row's tokens (right padding by default).
    #[must_use]
    pub fn with_alignment(mut self, alignment: TokenAlignment) -> Self {
        self.alignment = alignment;
        self
    }

    /// Records one row's new keys and values `[1, kv_heads, n_b, head_dim]`.
    pub fn push_kv_row(
        &mut self,
        group: usize,
        layer: usize,
        row: usize,
        keys: Tensor<B, 4>,
        values: Tensor<B, 4>,
    ) {
        self.kv.push(KvEntry::Row {
            group,
            layer,
            row,
            planes: vec![keys, values],
        });
    }

    /// Records one row's new latents `[1, n_b, width]` for a latent group.
    pub fn push_latent_row(
        &mut self,
        group: usize,
        layer: usize,
        row: usize,
        latent: Tensor<B, 3>,
    ) {
        self.kv.push(KvEntry::Row {
            group,
            layer,
            row,
            planes: vec![latent.unsqueeze_dim(1)],
        });
    }

    /// Records every row's new keys and values `[batch, kv_heads, block, head_dim]`.
    ///
    /// Row `b`'s tokens are located by the step's [`TokenAlignment`].
    pub fn push_kv_batch(
        &mut self,
        group: usize,
        layer: usize,
        keys: Tensor<B, 4>,
        values: Tensor<B, 4>,
    ) {
        self.kv.push(KvEntry::Batch {
            group,
            layer,
            planes: vec![keys, values],
        });
    }

    /// Records every row's new latents `[batch, block, width]` for a latent group.
    ///
    /// Row `b`'s tokens are located by the step's [`TokenAlignment`].
    pub fn push_latent_batch(&mut self, group: usize, layer: usize, latent: Tensor<B, 3>) {
        self.kv.push(KvEntry::Batch {
            group,
            layer,
            planes: vec![latent.unsqueeze_dim(1)],
        });
    }

    /// Records one row's recurrent state after all of its new tokens.
    ///
    /// Inside a transaction this is accepted only for single-token steps,
    /// where the terminal state *is* the post-token snapshot; multi-token
    /// speculative steps need [`push_recurrent_row_snapshots`](Self::push_recurrent_row_snapshots)
    /// because a recurrence cannot be rewound to an accepted prefix.
    pub fn push_recurrent_row(
        &mut self,
        group: usize,
        layer: usize,
        row: usize,
        matrix: Tensor<B, 4>,
        convolution: Tensor<B, 3>,
    ) {
        self.recurrent.push(RecurrentEntry::RowTerminal {
            group,
            layer,
            row,
            matrix,
            convolution,
        });
    }

    /// Records one row's recurrent state after each of its `n_b` new tokens.
    ///
    /// Inputs are `[1, n_b, heads, rows, columns]` and `[1, n_b, channels,
    /// width]`. Outside a transaction only the last snapshot is committed.
    pub fn push_recurrent_row_snapshots(
        &mut self,
        group: usize,
        layer: usize,
        row: usize,
        matrices: Tensor<B, 5>,
        convolutions: Tensor<B, 4>,
    ) {
        self.recurrent.push(RecurrentEntry::RowSnapshots {
            group,
            layer,
            row,
            matrices,
            convolutions,
        });
    }

    /// Records every row's terminal recurrent state `[batch, ...]`.
    ///
    /// Inactive rows keep their committed state whatever the input holds.
    /// Inside a transaction every active row must have exactly one new token.
    pub fn push_recurrent_batch(
        &mut self,
        group: usize,
        layer: usize,
        matrix: Tensor<B, 4>,
        convolution: Tensor<B, 3>,
    ) {
        self.recurrent.push(RecurrentEntry::BatchTerminal {
            group,
            layer,
            matrix,
            convolution,
        });
    }

    /// Returns whether the step recorded a write-ahead for `(group, layer)`.
    fn wrote_ahead(&self, group: usize, layer: usize) -> bool {
        self.kv.iter().any(|entry| {
            matches!(entry, KvEntry::WrittenAhead { group: g, layer: l, .. } if *g == group && *l == layer)
        })
    }

    /// Resolves each row's first input column for a batched input of width `block`.
    fn columns(&self, block: usize) -> TensorCacheResult<Vec<usize>> {
        match &self.alignment {
            TokenAlignment::RightPadded => Ok(vec![0; self.counts.len()]),
            TokenAlignment::LeftPadded => self
                .counts
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
                .collect(),
            TokenAlignment::Columns(columns) => Ok(columns.clone()),
        }
    }

    /// Builds the batched append layout of this step for a block of width `block`.
    fn append(&self, starts: &[usize], block: usize) -> TensorCacheResult<KvBatchAppend> {
        KvBatchAppend::with_columns(starts, &self.columns(block)?, &self.counts, block)
    }
}

/// Tracks which rows of one `(group, layer)` a step covers.
struct Coverage {
    /// `[group][layer][row]` coverage flags.
    covered: Vec<Vec<Vec<bool>>>,
    /// Cache family named in diagnostics.
    kind: &'static str,
}

impl Coverage {
    /// Creates empty coverage for groups with the given layer counts.
    fn new(kind: &'static str, layers: impl Iterator<Item = usize>, rows: usize) -> Self {
        Self {
            covered: layers.map(|count| vec![vec![false; rows]; count]).collect(),
            kind,
        }
    }

    /// Marks one row, or every active row when `row` is `None`.
    fn mark(
        &mut self,
        group: usize,
        layer: usize,
        row: Option<usize>,
        counts: &[usize],
    ) -> TensorCacheResult<()> {
        let groups = self.covered.len();
        let layers = self
            .covered
            .get_mut(group)
            .ok_or(TensorCacheError::CacheOutOfBounds {
                kind: self.kind,
                index: group,
                count: groups,
            })?;
        let layer_count = layers.len();
        let rows = layers
            .get_mut(layer)
            .ok_or(TensorCacheError::LayerOutOfBounds {
                layer,
                layers: layer_count,
            })?;
        let marked: Vec<usize> = match row {
            Some(row) => {
                if row >= counts.len() {
                    return Err(TensorCacheError::SequenceOutOfBounds {
                        sequence: row,
                        batch_size: counts.len(),
                    });
                }
                if counts[row] == 0 {
                    return Err(coverage_error(
                        self.kind,
                        group,
                        layer,
                        row,
                        "inactive row written",
                    ));
                }
                vec![row]
            }
            None => (0..counts.len()).filter(|&row| counts[row] > 0).collect(),
        };
        for row in marked {
            if core::mem::replace(&mut rows[row], true) {
                return Err(coverage_error(
                    self.kind,
                    group,
                    layer,
                    row,
                    "duplicate write",
                ));
            }
        }
        Ok(())
    }

    /// Requires every active row of every layer of every group to be covered.
    fn finish(&self, counts: &[usize]) -> TensorCacheResult<()> {
        for (group, layers) in self.covered.iter().enumerate() {
            for (layer, rows) in layers.iter().enumerate() {
                if let Some(row) = (0..counts.len()).find(|&row| counts[row] > 0 && !rows[row]) {
                    return Err(coverage_error(
                        self.kind,
                        group,
                        layer,
                        row,
                        "missing write",
                    ));
                }
            }
        }
        Ok(())
    }
}

/// Builds a step-coverage error.
const fn coverage_error(
    kind: &'static str,
    group: usize,
    layer: usize,
    sequence: usize,
    problem: &'static str,
) -> TensorCacheError {
    TensorCacheError::StepWriteCoverage {
        kind,
        group,
        layer,
        sequence,
        problem,
    }
}

/// A prepared KV publication of one group.
enum PreparedKv<B: Backend> {
    /// Per-row direct write.
    Row(usize, PreparedNativeKvWrite<B>),
    /// Batched write or staging.
    Batch(usize, PreparedNativeKvBatchWrite<B>),
    /// Staged-length updates for a layer written ahead inside a transaction.
    StagedLengths(usize, usize, Vec<(usize, usize)>),
}

/// A prepared recurrent publication of one group.
enum PreparedRecurrent<B: Backend> {
    /// Per-row direct terminal state.
    Row(usize, PreparedRecurrentStateWrite<B>),
    /// Whole-batch direct terminal state.
    Batch(usize, PreparedRecurrentBatchWrite<B>),
    /// Snapshot staging.
    Stage(usize, PreparedRecurrentStage<B>),
}

impl<B: Backend> TensorDecodeState<B> {
    /// Starts recording one main-bank step that appends `token_counts[b]` tokens to row `b`.
    ///
    /// # Errors
    ///
    /// Returns an error when the count vector does not have one entry per row.
    pub fn step(&self, token_counts: &[usize]) -> TensorCacheResult<TensorStepWrites<B>> {
        self.step_for(StepBank::Main, token_counts)
    }

    /// Starts recording one MTP-bank step with the MTP stack's own token counts.
    ///
    /// # Errors
    ///
    /// Returns an error when MTP is disabled or the count vector has the
    /// wrong length.
    pub fn mtp_step(&self, token_counts: &[usize]) -> TensorCacheResult<TensorStepWrites<B>> {
        self.step_for(StepBank::Mtp, token_counts)
    }

    /// Shared step constructor.
    fn step_for(
        &self,
        bank: StepBank,
        token_counts: &[usize],
    ) -> TensorCacheResult<TensorStepWrites<B>> {
        if bank == StepBank::Mtp && self.mtp.is_none() {
            return Err(TensorCacheError::MtpNotEnabled);
        }
        validate_count(self.batch_size(), token_counts.len())?;
        Ok(TensorStepWrites {
            bank,
            counts: token_counts.to_vec(),
            alignment: TokenAlignment::RightPadded,
            kv: Vec::new(),
            recurrent: Vec::new(),
        })
    }

    /// Returns the bank and logical lengths a step targets.
    fn step_target(
        &self,
        bank: StepBank,
    ) -> TensorCacheResult<(&TensorCacheBank<B>, &SequenceLengths)> {
        match bank {
            StepBank::Main => Ok((&self.main, &self.metadata)),
            StepBank::Mtp => match (&self.mtp, &self.mtp_metadata) {
                (Some(bank), Some(lengths)) => Ok((bank, lengths)),
                _ => Err(TensorCacheError::MtpNotEnabled),
            },
        }
    }

    /// Reads one row's working KV history using this state's own cursors.
    ///
    /// Unlike [`NativeKvCache::read_working_sequence`](super::NativeKvCache::read_working_sequence),
    /// the caller cannot pass a stale committed length, and inside a
    /// transaction the layer's staged proposal must equal the row's logical
    /// proposal length, so a layer left behind by a failed or skipped staging
    /// step is reported instead of silently attended over.
    ///
    /// # Errors
    ///
    /// Returns an error for a disabled bank, invalid indices, or a staged
    /// length that disagrees with the working cursor.
    pub fn read_kv(
        &self,
        bank: StepBank,
        group: usize,
        layer: usize,
        row: usize,
    ) -> TensorCacheResult<KvCacheRead<B>> {
        let (cache_bank, lengths) = self.step_target(bank)?;
        let cache = cache_bank.native_kv(group)?;
        let (committed, proposed) = row_cursors(lengths, row)?;
        if cache.has_active_staging() {
            check_staged(row, proposed, cache.staged_length(layer, row)?)?;
        }
        cache.read_working_sequence(layer, row, committed)
    }

    /// Reads one row's working recurrent state, checking its snapshot count.
    ///
    /// Inside a transaction the layer must hold exactly one snapshot per
    /// proposal token; the read then returns the snapshot after the last one.
    ///
    /// # Errors
    ///
    /// Returns an error for a disabled bank, invalid indices, or a snapshot
    /// count that disagrees with the working cursor.
    pub fn read_recurrent(
        &self,
        bank: StepBank,
        group: usize,
        layer: usize,
        row: usize,
    ) -> TensorCacheResult<RecurrentStateRead<B>> {
        let (cache_bank, lengths) = self.step_target(bank)?;
        let cache = cache_bank.recurrent(group)?;
        let (_, proposed) = row_cursors(lengths, row)?;
        if cache.has_active_staging() {
            check_staged(row, proposed, cache.staged_length(layer, row)?)?;
        }
        cache.read_working_sequence(layer, row)
    }

    /// Plans a batched read of one group's working history (committed plus staged).
    ///
    /// Use it to attend over the history *before* the step's own tokens; the
    /// step's tokens then join through [`step_token_segment`](Self::step_token_segment).
    ///
    /// # Errors
    ///
    /// Returns an error for a disabled bank, an invalid group, or the errors of
    /// [`NativeKvCache::plan_working_batch_read`](super::NativeKvCache::plan_working_batch_read).
    pub fn kv_read_plan(
        &self,
        bank: StepBank,
        group: usize,
    ) -> TensorCacheResult<KvBatchReadPlan<B>> {
        let (cache_bank, lengths) = self.step_target(bank)?;
        cache_bank
            .native_kv(group)?
            .plan_working_batch_read(lengths.lengths())
    }

    /// Plans a batched read of one group through a step written ahead.
    ///
    /// The plan covers `[0, W_b + n_b)` for working length `W_b`, i.e. the
    /// history plus this step's tokens, as one segment of full storage. Read
    /// each layer with [`read_kv_through`](Self::read_kv_through) after
    /// [`write_kv_ahead`](Self::write_kv_ahead).
    ///
    /// # Errors
    ///
    /// Returns an error when the group cannot write ahead (see
    /// [`write_kv_ahead`](Self::write_kv_ahead)) or the step overflows capacity.
    pub fn kv_read_plan_through(
        &self,
        step: &TensorStepWrites<B>,
        group: usize,
    ) -> TensorCacheResult<KvBatchReadPlan<B>> {
        let (bank, lengths) = self.step_target(step.bank)?;
        let cache = bank.native_kv(group)?;
        validate_write_ahead(
            cache.spec().storage(),
            cache.spec().in_place_staging(),
            lengths.has_active_transaction(),
        )?;
        let ends = lengths
            .working_lengths()
            .iter()
            .zip(&step.counts)
            .map(|(&length, &count)| {
                length
                    .checked_add(count)
                    .ok_or(TensorCacheError::WritePositionOverflow {
                        start: length,
                        tokens: count,
                    })
            })
            .collect::<TensorCacheResult<Vec<_>>>()?;
        cache.plan_batch_read(&ends)
    }

    /// Writes one layer's new tokens into full storage ahead of attention.
    ///
    /// Inputs are `[batch, kv_heads, block, head_dim]`, placed by the step's
    /// [`TokenAlignment`]. The tokens land in slots `[W_b, W_b + n_b)`, which
    /// stay unreachable until [`publish_step`](Self::publish_step) succeeds;
    /// the step records the layer as covered. See the module documentation
    /// for the logical-atomicity contract.
    ///
    /// # Errors
    ///
    /// Returns an error for ring storage, for a transaction on a group without
    /// in-place staging, or for any write validation error. Nothing is
    /// written on error.
    pub fn write_kv_ahead(
        &mut self,
        step: &mut TensorStepWrites<B>,
        group: usize,
        layer: usize,
        keys: Tensor<B, 4>,
        values: Tensor<B, 4>,
    ) -> TensorCacheResult<()> {
        self.write_planes_ahead(step, group, layer, vec![keys, values])
    }

    /// Writes one layer's new latents `[batch, block, width]` ahead of attention.
    ///
    /// The latent counterpart of [`write_kv_ahead`](Self::write_kv_ahead).
    ///
    /// # Errors
    ///
    /// Returns the errors of [`write_kv_ahead`](Self::write_kv_ahead), or an
    /// error for a key/value group.
    pub fn write_latent_ahead(
        &mut self,
        step: &mut TensorStepWrites<B>,
        group: usize,
        layer: usize,
        latent: Tensor<B, 3>,
    ) -> TensorCacheResult<()> {
        self.write_planes_ahead(step, group, layer, vec![latent.unsqueeze_dim(1)])
    }

    /// Plane-generic core of the write-ahead methods.
    fn write_planes_ahead(
        &mut self,
        step: &mut TensorStepWrites<B>,
        group: usize,
        layer: usize,
        planes: Vec<Tensor<B, 4>>,
    ) -> TensorCacheResult<()> {
        let (bank, lengths) = self.step_target(step.bank)?;
        let starts = lengths.working_lengths().to_vec();
        let staging = lengths.has_active_transaction();
        let cache = bank.native_kv(group)?;
        validate_write_ahead(
            cache.spec().storage(),
            cache.spec().in_place_staging(),
            staging,
        )?;
        let append = step.append(&starts, planes[0].dims()[2])?;
        let write = if staging {
            cache.prepare_batch_stage(layer, &append, planes)?
        } else {
            cache.prepare_batch_write(layer, &append, planes)?
        };
        let bank = match step.bank {
            StepBank::Main => &mut self.main,
            StepBank::Mtp => self.mtp.as_mut().ok_or(TensorCacheError::MtpNotEnabled)?,
        };
        // Staged lengths are published with the step, not here.
        let _deferred = bank.native_kv[group].apply_batch_tensors(write);
        step.kv.push(KvEntry::WrittenAhead {
            group,
            layer,
            starts,
        });
        Ok(())
    }

    /// Reads one layer written ahead through a [`kv_read_plan_through`](Self::kv_read_plan_through) plan.
    ///
    /// # Errors
    ///
    /// Returns an error when the step has not written this layer ahead, or
    /// for the errors of [`NativeKvCache::read_batch`](super::NativeKvCache::read_batch).
    pub fn read_kv_through(
        &self,
        step: &TensorStepWrites<B>,
        group: usize,
        layer: usize,
        plan: &KvBatchReadPlan<B>,
    ) -> TensorCacheResult<KvBatchRead<B>> {
        if !step.wrote_ahead(group, layer) {
            return Err(TensorCacheError::BatchPlanMismatch {
                layer,
                reason: "the step has not written this layer ahead",
            });
        }
        let (bank, _) = self.step_target(step.bank)?;
        bank.native_kv(group)?.read_batch(layer, plan)
    }

    /// Describes a step's batched new tokens as a read segment.
    ///
    /// Column `c` of row `b` holds absolute position `W_b + (c - c_b)` when it
    /// lies inside the row's token range and is invalid otherwise, so
    /// appending this segment to a history read yields every key a query of
    /// this step may attend to.
    ///
    /// # Errors
    ///
    /// Returns an error for a disabled bank, a misplaced token range, or a
    /// position beyond Burn's portable `i32` range.
    pub fn step_token_segment(
        &self,
        step: &TensorStepWrites<B>,
        keys: Tensor<B, 4>,
        values: Tensor<B, 4>,
    ) -> TensorCacheResult<KvBatchSegment<B>> {
        let (_, lengths) = self.step_target(step.bank)?;
        let [rows, _, block, _] = keys.dims();
        validate_count(step.counts.len(), rows)?;
        let append = step.append(lengths.working_lengths(), block)?;
        let mut positions = Vec::with_capacity(rows * block);
        let mut valid = Vec::with_capacity(rows * block);
        for row in 0..rows {
            let column = append.columns()[row];
            let tokens = column..column + append.counts()[row];
            for slot in 0..block {
                let inside = tokens.contains(&slot);
                let position = if inside {
                    append.starts()[row] + (slot - column)
                } else {
                    0
                };
                positions.push(i32::try_from(position).map_err(|_| {
                    TensorCacheError::TensorIndexOverflow {
                        axis: "key position",
                        index: position,
                    }
                })?);
                valid.push(inside);
            }
        }
        let device = keys.device();
        let all_valid = valid.iter().all(|&inside| inside);
        Ok(KvBatchSegment::from_parts(
            keys,
            values,
            Tensor::<B, 2, Int>::from_ints(TensorData::new(positions, [rows, block]), &device),
            (!all_valid).then(|| {
                Tensor::<B, 2, Bool>::from_data(TensorData::new(valid, [rows, block]), &device)
            }),
        ))
    }

    /// Describes a step's batched new latents `[batch, block, width]` as a read segment.
    ///
    /// The latent counterpart of [`step_token_segment`](Self::step_token_segment);
    /// the segment's keys and values are both the latent.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`step_token_segment`](Self::step_token_segment).
    pub fn step_latent_segment(
        &self,
        step: &TensorStepWrites<B>,
        latent: Tensor<B, 3>,
    ) -> TensorCacheResult<KvBatchSegment<B>> {
        let latent = latent.unsqueeze_dim::<4>(1);
        self.step_token_segment(step, latent.clone(), latent)
    }

    /// Validates a complete step and publishes it atomically.
    ///
    /// Outside a transaction the tokens are committed; inside one they are
    /// staged as proposals and only the working cursor advances. See the
    /// module documentation for the exact checks.
    ///
    /// # Errors
    ///
    /// Returns an error for incomplete, duplicate, or inconsistent coverage,
    /// capacity overflow, or any tensor validation failure. On error nothing
    /// changes, except slots beyond the lengths that a write-ahead filled.
    pub fn publish_step(&mut self, step: TensorStepWrites<B>) -> TensorCacheResult<()> {
        let bank_kind = step.bank;
        let (bank, lengths) = self.step_target(bank_kind)?;
        validate_count(self.batch_size(), step.counts.len())?;
        validate_coverage(bank, &step)?;
        let next = lengths.plan_advance(&step.counts)?;
        let cursors = StepCursors {
            starts: lengths.working_lengths().to_vec(),
            proposals: lengths
                .has_active_transaction()
                .then(|| lengths.proposal_lengths()),
        };
        let prepared = prepare_step(bank, step, &cursors)?;

        // Every recoverable check has passed; publication is infallible.
        let (bank, lengths) = match bank_kind {
            StepBank::Main => (&mut self.main, &mut self.metadata),
            StepBank::Mtp => match (&mut self.mtp, &mut self.mtp_metadata) {
                (Some(bank), Some(lengths)) => (bank, lengths),
                _ => return Err(TensorCacheError::MtpNotEnabled),
            },
        };
        lengths.publish_advance(next);
        apply_step(bank, prepared);
        self.sync_history_locks();
        Ok(())
    }
}

/// Returns a row's committed length and proposal length.
fn row_cursors(lengths: &SequenceLengths, row: usize) -> TensorCacheResult<(usize, usize)> {
    let committed = lengths.length(row)?;
    Ok((committed, lengths.working_lengths()[row] - committed))
}

/// Requires a layer's staged count to equal the row's logical proposal length.
fn check_staged(row: usize, proposed: usize, staged: usize) -> TensorCacheResult<()> {
    if staged == proposed {
        Ok(())
    } else {
        Err(TensorCacheError::TransactionCursorMismatch {
            sequence: row,
            cursor: "staged proposal length",
            expected: proposed,
            actual: staged,
        })
    }
}

/// Rejects write-ahead for storage or transactions where it could expose unpublished tokens.
fn validate_write_ahead(
    storage: KvCacheStorage,
    in_place: bool,
    staging: bool,
) -> TensorCacheResult<()> {
    if storage != KvCacheStorage::Full {
        return Err(TensorCacheError::WriteAheadUnsupported {
            storage: storage.as_str(),
            reason: "a ring write would overwrite a slot that is still visible",
        });
    }
    if staging && !in_place {
        return Err(TensorCacheError::WriteAheadUnsupported {
            storage: storage.as_str(),
            reason: "inside a transaction only groups with in-place staging can write ahead",
        });
    }
    Ok(())
}

/// Checks that the step writes every active row of every layer of the bank exactly once.
fn validate_coverage<B: Backend>(
    bank: &TensorCacheBank<B>,
    step: &TensorStepWrites<B>,
) -> TensorCacheResult<()> {
    let rows = step.counts.len();
    let mut kv = Coverage::new(
        "native KV",
        bank.native_kv.iter().map(|cache| cache.spec().layers()),
        rows,
    );
    for entry in &step.kv {
        let (group, layer, row) = match entry {
            KvEntry::Row {
                group, layer, row, ..
            } => (*group, *layer, Some(*row)),
            KvEntry::Batch { group, layer, .. } | KvEntry::WrittenAhead { group, layer, .. } => {
                (*group, *layer, None)
            }
        };
        kv.mark(group, layer, row, &step.counts)?;
        if let KvEntry::Row { planes, .. } = entry
            && let Some(row) = row
            && planes
                .first()
                .is_some_and(|plane| plane.dims()[2] != step.counts[row])
        {
            return Err(coverage_error(
                "native KV",
                group,
                layer,
                row,
                "token count differs from the step",
            ));
        }
    }
    kv.finish(&step.counts)?;

    let mut recurrent = Coverage::new(
        "recurrent",
        bank.recurrent.iter().map(|cache| cache.spec().layers()),
        rows,
    );
    for entry in &step.recurrent {
        let (group, layer, row, tokens) = match entry {
            RecurrentEntry::RowTerminal {
                group, layer, row, ..
            } => (*group, *layer, Some(*row), None),
            RecurrentEntry::RowSnapshots {
                group,
                layer,
                row,
                matrices,
                ..
            } => (*group, *layer, Some(*row), Some(matrices.dims()[1])),
            RecurrentEntry::BatchTerminal { group, layer, .. } => (*group, *layer, None, None),
        };
        recurrent.mark(group, layer, row, &step.counts)?;
        if let (Some(row), Some(tokens)) = (row, tokens)
            && tokens != step.counts[row]
        {
            return Err(coverage_error(
                "recurrent",
                group,
                layer,
                row,
                "token count differs from the step",
            ));
        }
    }
    recurrent.finish(&step.counts)
}

/// Every prepared write of one step, ready for infallible publication.
struct PreparedStep<B: Backend> {
    /// KV publications in recording order.
    kv: Vec<PreparedKv<B>>,
    /// Recurrent publications in recording order.
    recurrent: Vec<PreparedRecurrent<B>>,
}

/// Logical cursors a step is validated against.
struct StepCursors {
    /// Working length of each row: where the step's tokens begin.
    starts: Vec<usize>,
    /// Proposal length of each row inside a transaction, `None` outside one.
    proposals: Option<Vec<usize>>,
}

/// Prepares every write of a covered step without mutating anything.
fn prepare_step<B: Backend>(
    bank: &TensorCacheBank<B>,
    step: TensorStepWrites<B>,
    cursors: &StepCursors,
) -> TensorCacheResult<PreparedStep<B>> {
    let staging = cursors.proposals.is_some();
    let mut kv = Vec::with_capacity(step.kv.len());
    for entry in &step.kv {
        kv.push(prepare_kv(bank, &step, entry, cursors)?);
    }
    let active: Vec<bool> = step.counts.iter().map(|&count| count > 0).collect();
    let mut recurrent = Vec::with_capacity(step.recurrent.len());
    for entry in step.recurrent {
        if let (Some(proposals), Some((group, layer, rows))) =
            (&cursors.proposals, recurrent_rows(&entry, &active))
        {
            check_recurrent_proposals(bank, group, layer, &rows, proposals)?;
        }
        prepare_recurrent(bank, entry, &step.counts, &active, staging, &mut recurrent)?;
    }
    Ok(PreparedStep { kv, recurrent })
}

/// Returns the group, layer, and rows a recurrent entry stages.
fn recurrent_rows<B: Backend>(
    entry: &RecurrentEntry<B>,
    active: &[bool],
) -> Option<(usize, usize, Vec<usize>)> {
    match entry {
        RecurrentEntry::RowTerminal {
            group, layer, row, ..
        }
        | RecurrentEntry::RowSnapshots {
            group, layer, row, ..
        } => Some((*group, *layer, vec![*row])),
        RecurrentEntry::BatchTerminal { group, layer, .. } => Some((
            *group,
            *layer,
            (0..active.len()).filter(|&row| active[row]).collect(),
        )),
    }
}

/// Requires a recurrent layer's staged snapshots to equal the logical proposal length.
///
/// Snapshot `i` is the state after proposal token `i`, so the next snapshot
/// must land exactly at the row's proposal length; a mismatch means an
/// earlier step staged this layer outside the step API.
fn check_recurrent_proposals<B: Backend>(
    bank: &TensorCacheBank<B>,
    group: usize,
    layer: usize,
    rows: &[usize],
    proposals: &[usize],
) -> TensorCacheResult<()> {
    let staged = bank.recurrent(group)?.staged_lengths_of(layer)?;
    match rows.iter().find(|&&row| staged[row] != proposals[row]) {
        Some(&sequence) => Err(TensorCacheError::TransactionCursorMismatch {
            sequence,
            cursor: "recurrent proposal start",
            expected: proposals[sequence],
            actual: staged[sequence],
        }),
        None => Ok(()),
    }
}

/// Prepares one KV entry as a direct write, a staging write, or deferred staged lengths.
fn prepare_kv<B: Backend>(
    bank: &TensorCacheBank<B>,
    step: &TensorStepWrites<B>,
    entry: &KvEntry<B>,
    cursors: &StepCursors,
) -> TensorCacheResult<PreparedKv<B>> {
    let starts = cursors.starts.as_slice();
    let staging = cursors.proposals.is_some();
    match entry {
        KvEntry::Row {
            group,
            layer,
            row,
            planes,
        } => {
            let cache = bank.native_kv(*group)?;
            Ok(if staging {
                cache.check_proposal_start(*layer, *row, starts[*row])?;
                PreparedKv::Batch(
                    *group,
                    cache.prepare_row_stage(*layer, *row, planes.clone())?,
                )
            } else {
                PreparedKv::Row(
                    *group,
                    cache.prepare_planes_write(*layer, *row, starts[*row], planes.clone())?,
                )
            })
        }
        KvEntry::Batch {
            group,
            layer,
            planes,
        } => {
            let cache = bank.native_kv(*group)?;
            let block = planes.first().map_or(0, |plane| plane.dims()[2]);
            let append = step.append(starts, block)?;
            Ok(PreparedKv::Batch(
                *group,
                if staging {
                    cache.prepare_batch_stage(*layer, &append, planes.clone())?
                } else {
                    cache.prepare_batch_write(*layer, &append, planes.clone())?
                },
            ))
        }
        KvEntry::WrittenAhead {
            group,
            layer,
            starts: written,
        } => {
            if let Some(sequence) = (0..starts.len()).find(|&row| written[row] != starts[row]) {
                return Err(TensorCacheError::TransactionCursorMismatch {
                    sequence,
                    cursor: "write-ahead start",
                    expected: starts[sequence],
                    actual: written[sequence],
                });
            }
            let updates = if staging {
                bank.native_kv(*group)?
                    .staged_updates_for(*layer, starts, &step.counts)?
            } else {
                Vec::new()
            };
            Ok(PreparedKv::StagedLengths(*group, *layer, updates))
        }
    }
}

/// Prepares one recurrent entry, appending one or more prepared writes.
fn prepare_recurrent<B: Backend>(
    bank: &TensorCacheBank<B>,
    entry: RecurrentEntry<B>,
    counts: &[usize],
    active: &[bool],
    staging: bool,
    prepared: &mut Vec<PreparedRecurrent<B>>,
) -> TensorCacheResult<()> {
    match entry {
        RecurrentEntry::RowTerminal {
            group,
            layer,
            row,
            matrix,
            convolution,
        } => {
            let cache = bank.recurrent(group)?;
            if staging {
                if counts[row] != 1 {
                    return Err(coverage_error(
                        "recurrent",
                        group,
                        layer,
                        row,
                        "a multi-token transactional step needs per-token snapshots",
                    ));
                }
                let stage = cache.prepare_row_snapshots(
                    layer,
                    row,
                    matrix.unsqueeze_dim(1),
                    convolution.unsqueeze_dim(1),
                )?;
                prepared.push(PreparedRecurrent::Stage(group, stage));
            } else {
                let write = cache.prepare_sequence_state(layer, row, matrix, convolution)?;
                prepared.push(PreparedRecurrent::Row(group, write));
            }
        }
        RecurrentEntry::RowSnapshots {
            group,
            layer,
            row,
            matrices,
            convolutions,
        } => {
            let cache = bank.recurrent(group)?;
            if staging {
                let stage = cache.prepare_row_snapshots(layer, row, matrices, convolutions)?;
                prepared.push(PreparedRecurrent::Stage(group, stage));
            } else {
                // Only the state after the row's last token is committed.
                let [_, tokens, heads, rows, columns] = matrices.dims();
                let [_, _, channels, width] = convolutions.dims();
                let last = tokens - 1;
                let write = cache.prepare_sequence_state(
                    layer,
                    row,
                    matrices
                        .slice([0..1, last..tokens, 0..heads, 0..rows, 0..columns])
                        .reshape([1, heads, rows, columns]),
                    convolutions
                        .slice([0..1, last..tokens, 0..channels, 0..width])
                        .reshape([1, channels, width]),
                )?;
                prepared.push(PreparedRecurrent::Row(group, write));
            }
        }
        RecurrentEntry::BatchTerminal {
            group,
            layer,
            matrix,
            convolution,
        } => {
            let cache = bank.recurrent(group)?;
            if !active.contains(&true) {
                return Ok(());
            }
            if !staging {
                let write = cache.prepare_batch_state(layer, matrix, convolution, active)?;
                prepared.push(PreparedRecurrent::Batch(group, write));
                return Ok(());
            }
            if let Some(row) = (0..counts.len()).find(|&row| counts[row] > 1) {
                return Err(coverage_error(
                    "recurrent",
                    group,
                    layer,
                    row,
                    "a multi-token transactional step needs per-token snapshots",
                ));
            }
            prepare_batch_snapshot_stages(
                cache,
                group,
                layer,
                matrix,
                convolution,
                active,
                prepared,
            )?;
        }
    }
    Ok(())
}

/// Stages single-token batch states, as one block when every row is active and aligned.
fn prepare_batch_snapshot_stages<B: Backend>(
    cache: &crate::cache::RecurrentStateCache<B>,
    group: usize,
    layer: usize,
    matrix: Tensor<B, 4>,
    convolution: Tensor<B, 3>,
    active: &[bool],
    prepared: &mut Vec<PreparedRecurrent<B>>,
) -> TensorCacheResult<()> {
    let uniform = active.iter().all(|&row| row)
        && cache
            .staged_lengths_of(layer)?
            .windows(2)
            .all(|pair| pair[0] == pair[1]);
    if uniform {
        let stage = cache.prepare_batch_snapshots(
            layer,
            matrix.unsqueeze_dim(1),
            convolution.unsqueeze_dim(1),
        )?;
        prepared.push(PreparedRecurrent::Stage(group, stage));
        return Ok(());
    }
    let [_, heads, rows, columns] = matrix.dims();
    let [_, channels, width] = convolution.dims();
    for row in (0..active.len()).filter(|&row| active[row]) {
        let row_range: Range<usize> = row..row + 1;
        let stage = cache.prepare_row_snapshots(
            layer,
            row,
            matrix
                .clone()
                .slice([row_range.clone(), 0..heads, 0..rows, 0..columns])
                .unsqueeze_dim(1),
            convolution
                .clone()
                .slice([row_range, 0..channels, 0..width])
                .unsqueeze_dim(1),
        )?;
        prepared.push(PreparedRecurrent::Stage(group, stage));
    }
    Ok(())
}

/// Applies prepared step writes to a bank; nothing here can fail.
fn apply_step<B: Backend>(bank: &mut TensorCacheBank<B>, prepared: PreparedStep<B>) {
    for write in prepared.kv {
        match write {
            PreparedKv::Row(group, write) => {
                bank.native_kv[group].apply_prepared_writes(vec![write])
            }
            PreparedKv::Batch(group, write) => {
                bank.native_kv[group].apply_batch_writes(vec![write])
            }
            PreparedKv::StagedLengths(group, layer, updates) => {
                bank.native_kv[group].record_staged_lengths(layer, &updates);
            }
        }
    }
    for write in prepared.recurrent {
        match write {
            PreparedRecurrent::Row(group, write) => {
                bank.recurrent[group].apply_prepared_state_writes(vec![write]);
            }
            PreparedRecurrent::Batch(group, write) => {
                bank.recurrent[group].apply_batch_state_writes(vec![write]);
            }
            PreparedRecurrent::Stage(group, stage) => {
                bank.recurrent[group].apply_snapshot_stages(vec![stage]);
            }
        }
    }
}

#[cfg(test)]
#[path = "step_tests.rs"]
mod tests;
