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
//! A rejected step changes no length and no readable cache position; only
//! write-ahead (below) may leave bytes in slots beyond the lengths. The step
//! API is intended to replace the per-family `PendingKvWrite` structs and
//! coverage checks (families migrate to it next); it works for any number of
//! KV, latent, and recurrent groups and makes speculative staging
//! all-or-nothing per step.
//!
//! # Step validity
//!
//! A step is derived from the cursors of one bank of one decode state at one
//! moment: its tokens' positions start at the working lengths, and its
//! write-ahead tokens sit in physical slots computed from them. Every step
//! therefore carries a [`StepOrigin`] — the state's identity and the bank's
//! lifecycle version — and every step operation rejects a step whose origin
//! is no longer current with [`TensorCacheError::StaleStep`]. The version
//! advances whenever anything other than the step itself could have moved
//! those cursors or slots: advance, begin/commit/rollback, reorder, truncate,
//! reset, mutable bank access, MTP attachment, and the publication or
//! write-ahead of *another* step. A step's own write-ahead re-stamps it, so
//! of two steps on one bank only the one that wrote ahead last stays valid.
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
//! them, and batched reads mark them invalid). Rings cannot write ahead,
//! because a wrapped write would overwrite a still-visible slot. Inside a
//! transaction only in-place groups can. Each layer can be written ahead at
//! most once per step and never combined with a pushed update of that layer:
//! a write-ahead of an already recorded layer is refused before it touches
//! any tensor, and a push recorded after a write-ahead fails publication's
//! coverage check.

use core::ops::Range;

use burn::tensor::{Bool, Int, Tensor, TensorData, backend::Backend};

use crate::cache::SequenceLengths;
use crate::cache::identity::IdentityToken;
use crate::cache::tensor::kv::{
    KvBatchAppend, KvBatchRead, KvBatchReadPlan, KvBatchSegment, KvCacheRead, WriteAheadViolation,
    left_padded_columns, token_segment_table, write_ahead_allowed,
};
use crate::cache::tensor::kv::{PreparedNativeKvBatchWrite, PreparedNativeKvWrite};
use crate::cache::tensor::plan::{checked_ends, first_index_above, portable_indices};
use crate::cache::tensor::recurrent::{
    PreparedRecurrentBatchWrite, PreparedRecurrentStage, PreparedRecurrentStateWrite,
    RecurrentStateRead,
};
use crate::cache::tensor::recurrent::{first_nonuniform_offset, last_snapshot};

use super::{
    KvCacheStorage, TensorCacheBank, TensorCacheError, TensorCacheResult, TensorDecodeState,
    validate_count,
};

mod coverage;

pub use coverage::StepBank;
use coverage::{
    BankVersions, CoverageViolation, all_set, first_active_row, first_uncovered_row,
    first_unstaged_proposal, initial_versions, mark_covered, set_positions,
};

impl StepBank {
    /// Returns the bank's name for diagnostics.
    const fn as_str(self) -> &'static str {
        match self {
            Self::Main => "main",
            Self::Mtp => "MTP",
        }
    }
}

/// The decode state and bank version a step was derived from.
#[derive(Debug, Clone)]
struct StepOrigin {
    /// Identity of the decode state that created the step.
    state: IdentityToken,
    /// Lifecycle version of the step's bank when the step was created or last wrote ahead.
    version: u64,
}

/// Per-bank lifecycle versions that bind step write sets to one decode state.
///
/// A version is a counter that the owning [`TensorDecodeState`] advances on
/// every change to a bank's cursors, transaction, or tensors made outside
/// the step being validated (see the module documentation). The counters
/// are the Verus-verified `BankVersions`: a bump changes exactly its bank's
/// version, to a value different from the old one. Counters wrap; reaching
/// a collision would take 2^64 lifecycle operations while one stale step is
/// kept alive, which is not a practical concern.
#[derive(Debug)]
pub(super) struct StepVersions {
    /// Identity of the owning decode state, shared by every step it creates.
    state: IdentityToken,
    /// Lifecycle versions of the main and MTP banks.
    banks: BankVersions,
}

impl StepVersions {
    /// Starts both banks at version zero under a fresh state identity.
    pub(super) fn new() -> Self {
        Self {
            state: IdentityToken::new(),
            banks: initial_versions(),
        }
    }

    /// Invalidates every outstanding step of one bank.
    pub(super) const fn bump(&mut self, bank: StepBank) {
        self.banks.bump(bank);
    }

    /// Invalidates every outstanding step of both banks.
    pub(super) const fn bump_all(&mut self) {
        self.bump(StepBank::Main);
        self.bump(StepBank::Mtp);
    }

    /// Returns a bank's current version.
    const fn current(&self, bank: StepBank) -> u64 {
        self.banks.current(bank)
    }

    /// Stamps a new step for `bank` with the current state identity and version.
    fn stamp(&self, bank: StepBank) -> StepOrigin {
        StepOrigin {
            state: self.state.clone(),
            version: self.current(bank),
        }
    }

    /// Accepts a step only if this state created it and its bank is unchanged since.
    fn check<B: Backend>(&self, step: &TensorStepWrites<B>) -> TensorCacheResult<()> {
        let reason = if step.origin.state != self.state {
            "it was created by another decode state"
        } else if step.origin.version != self.current(step.bank) {
            "its bank changed after the step was created or last wrote ahead"
        } else {
            return Ok(());
        };
        Err(TensorCacheError::StaleStep {
            bank: step.bank.as_str(),
            reason,
        })
    }
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

impl<B: Backend> KvEntry<B> {
    /// Returns the `(group, layer)` this entry updates.
    const fn target(&self) -> (usize, usize) {
        match self {
            Self::Row { group, layer, .. }
            | Self::Batch { group, layer, .. }
            | Self::WrittenAhead { group, layer, .. } => (*group, *layer),
        }
    }
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
///
/// A step belongs to the state and bank version it was created at; see the
/// module documentation on step validity.
#[derive(Debug)]
pub struct TensorStepWrites<B: Backend> {
    /// Bank receiving the step.
    bank: StepBank,
    /// State identity and bank version the step's cursors were derived from.
    origin: StepOrigin,
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
            matches!(entry, KvEntry::WrittenAhead { .. }) && entry.target() == (group, layer)
        })
    }

    /// Returns whether the step already records any KV update for `(group, layer)`.
    fn records_kv(&self, group: usize, layer: usize) -> bool {
        self.kv.iter().any(|entry| entry.target() == (group, layer))
    }

    /// Returns the first row with new tokens (row 0 when every row is inactive).
    fn first_active_row(&self) -> usize {
        first_active_row(&self.counts)
    }

    /// Resolves each row's first input column for a batched input of width `block`.
    fn columns(&self, block: usize) -> TensorCacheResult<Vec<usize>> {
        match &self.alignment {
            TokenAlignment::RightPadded => Ok(vec![0; self.counts.len()]),
            TokenAlignment::LeftPadded => {
                left_padded_columns(&self.counts, block).map_err(|sequence| {
                    TensorCacheError::AppendOutsideBlock {
                        sequence,
                        column: 0,
                        count: self.counts[sequence],
                        block,
                    }
                })
            }
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
        // The verified marker accepts exactly unmarked active rows.
        mark_covered(rows, row, counts).map_err(|violation| match violation {
            CoverageViolation::RowOutOfBounds { row } => TensorCacheError::SequenceOutOfBounds {
                sequence: row,
                batch_size: counts.len(),
            },
            CoverageViolation::InactiveRow { row } => {
                coverage_error(self.kind, group, layer, row, "inactive row written")
            }
            CoverageViolation::Duplicate { row } => {
                coverage_error(self.kind, group, layer, row, "duplicate write")
            }
        })
    }

    /// Requires every active row of every layer of every group to be covered.
    fn finish(&self, counts: &[usize]) -> TensorCacheResult<()> {
        for (group, layers) in self.covered.iter().enumerate() {
            for (layer, rows) in layers.iter().enumerate() {
                if let Some(row) = first_uncovered_row(rows, counts) {
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
            origin: self.versions.stamp(bank),
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
    /// The plan reads only this group; build it after the last lifecycle
    /// operation and use it until the step is published.
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
    /// Returns an error for a stale step, a group that cannot write ahead (see
    /// [`write_kv_ahead`](Self::write_kv_ahead)), or a step that overflows
    /// capacity.
    pub fn kv_read_plan_through(
        &self,
        step: &TensorStepWrites<B>,
        group: usize,
    ) -> TensorCacheResult<KvBatchReadPlan<B>> {
        self.versions.check(step)?;
        let (bank, lengths) = self.step_target(step.bank)?;
        let cache = bank.native_kv(group)?;
        validate_write_ahead(
            cache.spec().storage(),
            cache.spec().in_place_staging(),
            lengths.has_active_transaction(),
        )?;
        let working = lengths.working_lengths();
        let ends = checked_ends(working, &step.counts).map_err(|row| {
            TensorCacheError::WritePositionOverflow {
                start: working[row],
                tokens: step.counts[row],
            }
        })?;
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
    /// The write re-stamps `step` with the bank's new version, so any other
    /// outstanding step of this bank becomes stale.
    ///
    /// # Errors
    ///
    /// Returns an error for a stale step, a layer the step already covers,
    /// ring storage, a transaction on a group without in-place staging, or
    /// any write validation error. Nothing is written on error.
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
    ///
    /// Every check runs before the single tensor assignment; afterwards the
    /// bank's version advances and only `step` is re-stamped with it.
    fn write_planes_ahead(
        &mut self,
        step: &mut TensorStepWrites<B>,
        group: usize,
        layer: usize,
        planes: Vec<Tensor<B, 4>>,
    ) -> TensorCacheResult<()> {
        self.versions.check(step)?;
        if step.records_kv(group, layer) {
            return Err(coverage_error(
                "native KV",
                group,
                layer,
                step.first_active_row(),
                "duplicate write",
            ));
        }
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
        // The written slots now belong to this step alone: any other step of
        // this bank, which may hold write-aheads at the same slots, is stale.
        self.versions.bump(step.bank);
        step.origin = self.versions.stamp(step.bank);
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
    /// Returns an error for a stale step, a layer the step has not written
    /// ahead, or the errors of
    /// [`NativeKvCache::read_batch`](super::NativeKvCache::read_batch).
    pub fn read_kv_through(
        &self,
        step: &TensorStepWrites<B>,
        group: usize,
        layer: usize,
        plan: &KvBatchReadPlan<B>,
    ) -> TensorCacheResult<KvBatchRead<B>> {
        self.versions.check(step)?;
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
    /// Returns an error for a stale step, a disabled bank, a misplaced token
    /// range, or a position beyond Burn's portable `i32` range.
    pub fn step_token_segment(
        &self,
        step: &TensorStepWrites<B>,
        keys: Tensor<B, 4>,
        values: Tensor<B, 4>,
    ) -> TensorCacheResult<KvBatchSegment<B>> {
        self.versions.check(step)?;
        let (_, lengths) = self.step_target(step.bank)?;
        let [rows, _, block, _] = keys.dims();
        validate_count(step.counts.len(), rows)?;
        let append = step.append(lengths.working_lengths(), block)?;
        // `append` passed the verified layout validation, so the verified
        // table builder's precondition holds; `rows == append.rows()` above.
        let (positions, valid) =
            token_segment_table(append.starts(), append.columns(), append.counts(), block);
        let positions = portable_indices(&positions).map_err(|index| {
            TensorCacheError::TensorIndexOverflow {
                axis: "key position",
                index: positions[index],
            }
        })?;
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
    /// Returns an error for a stale step, incomplete, duplicate, or
    /// inconsistent coverage, capacity overflow, or any tensor validation
    /// failure. On error nothing changes, except slots beyond the lengths that
    /// a write-ahead filled.
    pub fn publish_step(&mut self, step: TensorStepWrites<B>) -> TensorCacheResult<()> {
        self.versions.check(&step)?;
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
        self.versions.bump(bank_kind);
        self.sync_history_locks();
        Ok(())
    }
}

/// Returns a row's committed length and proposal length.
fn row_cursors(lengths: &SequenceLengths, row: usize) -> TensorCacheResult<(usize, usize)> {
    Ok((lengths.length(row)?, lengths.proposal_length(row)?))
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
    // The verified rule admits exactly full storage, in place inside a transaction.
    write_ahead_allowed(storage, in_place, staging).map_err(|violation| {
        TensorCacheError::WriteAheadUnsupported {
            storage: storage.as_str(),
            reason: match violation {
                WriteAheadViolation::RingStorage => {
                    "a ring write would overwrite a slot that is still visible"
                }
                WriteAheadViolation::SeparateStaging => {
                    "inside a transaction only groups with in-place staging can write ahead"
                }
            },
        }
    })
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
        RecurrentEntry::BatchTerminal { group, layer, .. } => {
            Some((*group, *layer, set_positions(active)))
        }
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
    // Coverage validated every row, so each indexes both vectors.
    match first_unstaged_proposal(rows, staged, proposals) {
        Some(position) => {
            let sequence = rows[position];
            Err(TensorCacheError::TransactionCursorMismatch {
                sequence,
                cursor: "recurrent proposal start",
                expected: proposals[sequence],
                actual: staged[sequence],
            })
        }
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
            // The version check in `publish_step` proves no lifecycle
            // operation ran since the write, so the cursors cannot have moved.
            debug_assert_eq!(written.as_slice(), starts, "write-ahead cursors moved");
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
                // Coverage proved the row active with exactly `tokens` new
                // tokens; the verified selector still rejects an empty row.
                let Some(last) = last_snapshot(tokens) else {
                    return Err(coverage_error(
                        "recurrent",
                        group,
                        layer,
                        row,
                        "row has no tokens",
                    ));
                };
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
            if let Some(row) = first_index_above(counts, 1) {
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
    let uniform =
        all_set(active) && first_nonuniform_offset(cache.staged_lengths_of(layer)?).is_none();
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
    for row in set_positions(active) {
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
