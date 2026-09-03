//! Composable model-level decode state and lifecycle coordination.

use std::sync::Arc;

use burn::tensor::backend::Backend;

use crate::architecture::DecodeState as ArchitectureDecodeState;
use crate::cache::{CacheConfig, DecodeState as DecodeMetadata, DecodeStateError};

use super::kv::{validate_batch, validate_reorder_indices};
use super::{
    NativeKvCache, NativeKvCacheSpec, RecurrentStateCache, RecurrentStateSpec, TensorCacheError,
    TensorCacheResult,
};

/// A composable collection of attention and recurrent cache allocations.
///
/// A bank can represent the main decoder or an auxiliary multi-token-prediction
/// (MTP) stack. Every cache in the bank shares batch size, logical capacity, and
/// device, while individual groups may have different head/state geometry.
#[derive(Debug)]
pub struct TensorCacheBank<B: Backend> {
    config: CacheConfig,
    batch_size: usize,
    device: B::Device,
    native_kv: Vec<NativeKvCache<B>>,
    recurrent: Vec<RecurrentStateCache<B>>,
}

impl<B: Backend> TensorCacheBank<B> {
    /// Creates an empty cache bank without allocating layer tensors.
    ///
    /// # Errors
    ///
    /// Returns an error when `batch_size` differs from the sequence count in
    /// `config`.
    pub fn new(
        config: CacheConfig,
        batch_size: usize,
        device: &B::Device,
    ) -> TensorCacheResult<Self> {
        validate_batch(config, batch_size)?;
        Ok(Self {
            config,
            batch_size,
            device: device.clone(),
            native_kv: Vec::new(),
            recurrent: Vec::new(),
        })
    }

    /// Returns the shared logical cache configuration.
    #[must_use]
    pub const fn config(&self) -> CacheConfig {
        self.config
    }

    /// Returns the fixed batch-row count.
    #[must_use]
    pub const fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Returns the device shared by every tensor in this bank.
    #[must_use]
    pub const fn device(&self) -> &B::Device {
        &self.device
    }

    /// Allocates and appends one homogeneous native-KV cache group.
    ///
    /// # Errors
    ///
    /// Returns an error if a transaction is active or allocation geometry is
    /// incompatible with this bank's config, backend, or device.
    pub fn add_native_kv(&mut self, spec: NativeKvCacheSpec) -> TensorCacheResult<usize> {
        self.ensure_topology_mutation_allowed("add native KV cache")?;
        let cache = NativeKvCache::new(self.config, self.batch_size, spec, &self.device)?;
        let index = self.native_kv.len();
        self.native_kv.push(cache);
        Ok(index)
    }

    /// Allocates and appends one homogeneous recurrent-state cache group.
    ///
    /// # Errors
    ///
    /// Returns an error if a transaction is active or allocation geometry is
    /// incompatible with this bank's backend or device.
    pub fn add_recurrent(&mut self, spec: RecurrentStateSpec) -> TensorCacheResult<usize> {
        self.ensure_topology_mutation_allowed("add recurrent cache")?;
        let cache = RecurrentStateCache::new(self.config, self.batch_size, spec, &self.device)?;
        let index = self.recurrent.len();
        self.recurrent.push(cache);
        Ok(index)
    }

    /// Returns the number of native-KV cache groups.
    #[must_use]
    pub fn native_kv_count(&self) -> usize {
        self.native_kv.len()
    }

    /// Returns the number of recurrent-state cache groups.
    #[must_use]
    pub fn recurrent_count(&self) -> usize {
        self.recurrent.len()
    }

    /// Borrows one native-KV cache group.
    ///
    /// # Errors
    ///
    /// Returns an error when `index` is out of bounds.
    pub fn native_kv(&self, index: usize) -> TensorCacheResult<&NativeKvCache<B>> {
        self.native_kv
            .get(index)
            .ok_or(TensorCacheError::CacheOutOfBounds {
                kind: "native KV",
                index,
                count: self.native_kv.len(),
            })
    }

    /// Mutably borrows one native-KV cache group.
    ///
    /// # Errors
    ///
    /// Returns an error when `index` is out of bounds.
    pub fn native_kv_mut(&mut self, index: usize) -> TensorCacheResult<&mut NativeKvCache<B>> {
        let count = self.native_kv.len();
        self.native_kv
            .get_mut(index)
            .ok_or(TensorCacheError::CacheOutOfBounds {
                kind: "native KV",
                index,
                count,
            })
    }

    /// Borrows one recurrent-state cache group.
    ///
    /// # Errors
    ///
    /// Returns an error when `index` is out of bounds.
    pub fn recurrent(&self, index: usize) -> TensorCacheResult<&RecurrentStateCache<B>> {
        self.recurrent
            .get(index)
            .ok_or(TensorCacheError::CacheOutOfBounds {
                kind: "recurrent",
                index,
                count: self.recurrent.len(),
            })
    }

    /// Mutably borrows one recurrent-state cache group.
    ///
    /// # Errors
    ///
    /// Returns an error when `index` is out of bounds.
    pub fn recurrent_mut(
        &mut self,
        index: usize,
    ) -> TensorCacheResult<&mut RecurrentStateCache<B>> {
        let count = self.recurrent.len();
        self.recurrent
            .get_mut(index)
            .ok_or(TensorCacheError::CacheOutOfBounds {
                kind: "recurrent",
                index,
                count,
            })
    }

    /// Returns whether any cache group currently accepts staged values.
    #[must_use]
    pub fn has_active_staging(&self) -> bool {
        self.native_kv.iter().any(NativeKvCache::has_active_staging)
            || self
                .recurrent
                .iter()
                .any(RecurrentStateCache::has_active_staging)
    }

    /// Verifies every populated group can begin a speculative transaction.
    fn validate_can_begin_staging(&self) -> TensorCacheResult<()> {
        if self.has_active_staging() {
            return Err(TensorCacheError::TransactionAlreadyActive);
        }
        if self
            .native_kv
            .iter()
            .any(|cache| cache.spec().staging_capacity() == 0)
            || self
                .recurrent
                .iter()
                .any(|cache| cache.spec().staging_capacity() == 0)
        {
            return Err(TensorCacheError::StagingDisabled);
        }
        Ok(())
    }

    /// Begins staging after a model-level all-bank preflight succeeds.
    fn begin_staging(&mut self) -> TensorCacheResult<()> {
        for cache in &mut self.native_kv {
            cache.begin_staging()?;
        }
        for cache in &mut self.recurrent {
            cache.begin_staging()?;
        }
        Ok(())
    }

    /// Checks complete physical staging before any bank mutates committed data.
    fn validate_staging_commit(
        &self,
        baseline: &[usize],
        proposed: &[usize],
        accepted: &[usize],
    ) -> TensorCacheResult<()> {
        for cache in &self.native_kv {
            cache.validate_staging_commit(baseline, proposed, accepted)?;
        }
        for cache in &self.recurrent {
            cache.validate_staging_commit(proposed, accepted)?;
        }
        Ok(())
    }

    /// Commits previously validated accepted prefixes in every cache group.
    fn commit_staging(&mut self, baseline: &[usize], accepted: &[usize]) -> TensorCacheResult<()> {
        for cache in &mut self.native_kv {
            cache.commit_staging(baseline, accepted)?;
        }
        for cache in &mut self.recurrent {
            cache.commit_staging(accepted)?;
        }
        Ok(())
    }

    /// Discards all proposal storage without changing committed tensors.
    fn rollback_staging(&mut self) {
        for cache in &mut self.native_kv {
            cache.rollback_staging();
        }
        for cache in &mut self.recurrent {
            cache.rollback_staging();
        }
    }

    /// Clears every state tensor and transaction marker.
    fn reset(&mut self) {
        for cache in &mut self.native_kv {
            cache.reset();
        }
        for cache in &mut self.recurrent {
            cache.reset();
        }
    }

    /// Reorders or duplicates every tensor batch dimension.
    fn reorder(&mut self, indices: &[usize]) -> TensorCacheResult<()> {
        for cache in &mut self.native_kv {
            cache.reorder(indices)?;
        }
        for cache in &mut self.recurrent {
            cache.reorder(indices)?;
        }
        Ok(())
    }

    /// Validates that all physical representations support a rewind.
    fn validate_truncate(&self, current: &[usize], requested: &[usize]) -> TensorCacheResult<()> {
        for cache in &self.native_kv {
            cache.validate_truncate(current, requested)?;
        }
        for cache in &self.recurrent {
            cache.validate_truncate(current, requested)?;
        }
        Ok(())
    }

    /// Applies row clearing required by a previously validated rewind.
    fn apply_truncate(&mut self, current: &[usize], requested: &[usize]) {
        for cache in &mut self.native_kv {
            cache.apply_truncate(current, requested);
        }
        for cache in &mut self.recurrent {
            cache.apply_truncate(current, requested);
        }
    }

    /// Prevents bank topology changes while any group is staging.
    fn ensure_topology_mutation_allowed(&self, operation: &'static str) -> TensorCacheResult<()> {
        if self.has_active_staging() {
            Err(TensorCacheError::TransactionActive { operation })
        } else {
            Ok(())
        }
    }
}

#[derive(Clone)]
struct TensorTransactionToken(Arc<()>);

impl TensorTransactionToken {
    /// Creates a transaction identity that cannot collide with another state.
    fn new() -> Self {
        Self(Arc::new(()))
    }
}

impl core::fmt::Debug for TensorTransactionToken {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_str("TensorTransactionToken(..)")
    }
}

impl PartialEq for TensorTransactionToken {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for TensorTransactionToken {}

/// Single-use identity returned for a tensor-cache speculative transaction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorCacheMark {
    token: TensorTransactionToken,
    baseline_lengths: Vec<usize>,
}

impl TensorCacheMark {
    /// Returns the committed logical lengths captured before the proposal.
    #[must_use]
    pub fn baseline_lengths(&self) -> &[usize] {
        &self.baseline_lengths
    }
}

#[derive(Debug)]
struct ActiveTensorTransaction {
    token: TensorTransactionToken,
    baseline_lengths: Vec<usize>,
}

/// External, composable model state implementing the architecture lifecycle.
///
/// `main` contains the decoder's ordinary layer caches. `mtp`, when enabled,
/// is a distinct bank for multi-token prediction layers: it shares lifecycle
/// metadata and beam ordering but never aliases main-model tensors. Neither
/// bank is a Burn module or checkpoint parameter.
#[derive(Debug)]
pub struct TensorDecodeState<B: Backend> {
    metadata: DecodeMetadata,
    main: TensorCacheBank<B>,
    mtp: Option<TensorCacheBank<B>>,
    device: B::Device,
    active_transaction: Option<ActiveTensorTransaction>,
}

impl<B: Backend> TensorDecodeState<B> {
    /// Creates an empty model state with no layer cache groups yet.
    ///
    /// # Errors
    ///
    /// Returns [`TensorCacheError::BatchSizeMismatch`] when `config` does not
    /// describe exactly `batch_size` sequences.
    pub fn new(
        config: CacheConfig,
        batch_size: usize,
        device: &B::Device,
    ) -> TensorCacheResult<Self> {
        validate_batch(config, batch_size)?;
        Ok(Self {
            metadata: DecodeMetadata::new(config),
            main: TensorCacheBank::new(config, batch_size, device)?,
            mtp: None,
            device: device.clone(),
            active_transaction: None,
        })
    }

    /// Returns backend-independent logical length metadata.
    #[must_use]
    pub const fn metadata(&self) -> &DecodeMetadata {
        &self.metadata
    }

    /// Borrows the main decoder cache bank.
    #[must_use]
    pub const fn main(&self) -> &TensorCacheBank<B> {
        &self.main
    }

    /// Mutably borrows the main decoder cache bank.
    #[must_use]
    pub const fn main_mut(&mut self) -> &mut TensorCacheBank<B> {
        &mut self.main
    }

    /// Borrows the independent MTP cache bank when enabled.
    #[must_use]
    pub const fn mtp(&self) -> Option<&TensorCacheBank<B>> {
        self.mtp.as_ref()
    }

    /// Mutably borrows the independent MTP cache bank when enabled.
    #[must_use]
    pub const fn mtp_mut(&mut self) -> Option<&mut TensorCacheBank<B>> {
        self.mtp.as_mut()
    }

    /// Enables an empty MTP bank on the same config and device.
    ///
    /// # Errors
    ///
    /// Returns an error if MTP is already enabled or a transaction is active.
    pub fn enable_mtp(&mut self) -> TensorCacheResult<&mut TensorCacheBank<B>> {
        self.ensure_no_transaction("enable MTP")?;
        if self.mtp.is_some() {
            return Err(TensorCacheError::MtpAlreadyEnabled);
        }
        let bank =
            TensorCacheBank::new(self.metadata.config(), self.main.batch_size(), &self.device)?;
        Ok(self.mtp.insert(bank))
    }

    /// Attaches an independently assembled MTP bank after compatibility checks.
    ///
    /// # Errors
    ///
    /// Returns an error for an existing MTP bank, active transaction, different
    /// cache config, batch size, or device.
    pub fn attach_mtp(&mut self, bank: TensorCacheBank<B>) -> TensorCacheResult<()> {
        self.ensure_no_transaction("attach MTP")?;
        if self.mtp.is_some() {
            return Err(TensorCacheError::MtpAlreadyEnabled);
        }
        if bank.config() != self.main.config() {
            return Err(TensorCacheError::IncompatibleBank {
                bank: "MTP",
                reason: "cache configuration differs from main state",
            });
        }
        if bank.batch_size() != self.main.batch_size() {
            return Err(TensorCacheError::IncompatibleBank {
                bank: "MTP",
                reason: "batch size differs from main state",
            });
        }
        if bank.device() != &self.device {
            return Err(TensorCacheError::IncompatibleBank {
                bank: "MTP",
                reason: "device differs from main state",
            });
        }
        self.mtp = Some(bank);
        Ok(())
    }

    /// Atomically advances all logical sequence lengths.
    ///
    /// Models should populate every layer cache for these tokens before calling
    /// this method. During speculation the staged length for each cache group
    /// is checked against this logical suffix at commit time.
    ///
    /// # Errors
    ///
    /// Returns metadata errors for count, integer, or capacity overflow.
    pub fn advance(&mut self, token_counts: &[usize]) -> TensorCacheResult<()> {
        self.metadata.advance_all(token_counts)?;
        Ok(())
    }

    /// Begins one all-bank speculative transaction.
    ///
    /// Every populated cache group must have non-zero staging capacity. The
    /// transaction records logical baselines but writes no committed tensors.
    ///
    /// # Errors
    ///
    /// Returns an error for nested transactions or a cache without staging.
    pub fn begin_transaction(&mut self) -> TensorCacheResult<TensorCacheMark> {
        if self.active_transaction.is_some() {
            return Err(TensorCacheError::TransactionAlreadyActive);
        }
        self.main.validate_can_begin_staging()?;
        if let Some(mtp) = &self.mtp {
            mtp.validate_can_begin_staging()?;
        }
        self.main.begin_staging()?;
        if let Some(mtp) = &mut self.mtp {
            mtp.begin_staging()?;
        }

        let token = TensorTransactionToken::new();
        let baseline_lengths = self.metadata.lengths().to_vec();
        self.active_transaction = Some(ActiveTensorTransaction {
            token: token.clone(),
            baseline_lengths: baseline_lengths.clone(),
        });
        Ok(TensorCacheMark {
            token,
            baseline_lengths,
        })
    }

    /// Commits a possibly ragged accepted prefix from staged proposal tensors.
    ///
    /// Validation covers every cache group before any committed tensor changes.
    /// KV prefixes are copied from linear staging into full/ring storage;
    /// recurrent caches select snapshot `accepted - 1`. Rejected values never
    /// touch committed wrapped-ring slots.
    ///
    /// # Errors
    ///
    /// Returns an error for a stale mark, count mismatch, over-acceptance, or
    /// physically incomplete staging. Validation failure leaves the transaction
    /// active and recoverable by [`rollback`](Self::rollback).
    pub fn commit_prefix(
        &mut self,
        mark: TensorCacheMark,
        accepted: &[usize],
    ) -> TensorCacheResult<()> {
        let active = self.validate_mark(&mark)?;
        validate_count(self.main.batch_size(), accepted.len())?;
        let baseline = active.baseline_lengths.clone();
        let proposed = proposal_lengths(&baseline, self.metadata.lengths());
        validate_accepted(&proposed, accepted)?;
        self.main
            .validate_staging_commit(&baseline, &proposed, accepted)?;
        if let Some(mtp) = &self.mtp {
            mtp.validate_staging_commit(&baseline, &proposed, accepted)?;
        }

        self.main.commit_staging(&baseline, accepted)?;
        if let Some(mtp) = &mut self.mtp {
            mtp.commit_staging(&baseline, accepted)?;
        }
        let committed: Vec<usize> = baseline
            .iter()
            .zip(accepted)
            .map(|(&start, &count)| start + count)
            .collect();
        self.metadata.truncate_all(&committed)?;
        self.active_transaction = None;
        Ok(())
    }

    /// Discards staged tensors and restores the transaction's logical baseline.
    ///
    /// Because proposals were never written into committed rings, rollback is
    /// exact even after a full sliding window has wrapped.
    ///
    /// # Errors
    ///
    /// Returns an error when the mark is stale or belongs to another state.
    pub fn rollback(&mut self, mark: TensorCacheMark) -> TensorCacheResult<()> {
        let active = self.validate_mark(&mark)?;
        let baseline = active.baseline_lengths.clone();
        self.main.rollback_staging();
        if let Some(mtp) = &mut self.mtp {
            mtp.rollback_staging();
        }
        self.metadata.truncate_all(&baseline)?;
        self.active_transaction = None;
        Ok(())
    }

    /// Clears every tensor and logical length for session reuse.
    ///
    /// # Errors
    ///
    /// Returns an error while a speculative transaction is active.
    pub fn reset(&mut self) -> TensorCacheResult<()> {
        self.ensure_no_transaction("reset")?;
        self.main.reset();
        if let Some(mtp) = &mut self.mtp {
            mtp.reset();
        }
        self.metadata.reset()?;
        Ok(())
    }

    /// Reorders or duplicates all main, MTP, and metadata batch rows.
    ///
    /// Duplicate source indices implement beam expansion without reallocating a
    /// larger batch; output row count must remain unchanged.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid indices or an active transaction.
    pub fn reorder(&mut self, indices: &[usize]) -> TensorCacheResult<()> {
        self.ensure_no_transaction("reorder")?;
        validate_reorder_indices(indices, self.main.batch_size())?;
        self.main.reorder(indices)?;
        if let Some(mtp) = &mut self.mtp {
            mtp.reorder(indices)?;
        }
        self.metadata.reorder(indices)?;
        Ok(())
    }

    /// Rewinds logical lengths when every physical representation can do so.
    ///
    /// Full KV caches can always rewind because their prefix remains stored.
    /// Wrapped rings reject non-zero rewinds after eviction, and recurrent
    /// terminal states reject non-zero intermediate rewinds because recurrence
    /// is not invertible. Any representation may truncate to zero; affected
    /// tensor rows are then cleared exactly.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid lengths, active speculation, or an unsafe
    /// wrapped/recurrent rewind. Validation is atomic.
    pub fn truncate(&mut self, lengths: &[usize]) -> TensorCacheResult<()> {
        self.ensure_no_transaction("truncate")?;
        validate_count(self.main.batch_size(), lengths.len())?;
        let current = self.metadata.lengths().to_vec();
        for (sequence, (&from, &to)) in current.iter().zip(lengths).enumerate() {
            if to > from {
                return Err(DecodeStateError::CannotExtendByTruncation {
                    sequence,
                    current: from,
                    requested: to,
                }
                .into());
            }
        }
        self.main.validate_truncate(&current, lengths)?;
        if let Some(mtp) = &self.mtp {
            mtp.validate_truncate(&current, lengths)?;
        }

        self.main.apply_truncate(&current, lengths);
        if let Some(mtp) = &mut self.mtp {
            mtp.apply_truncate(&current, lengths);
        }
        self.metadata.truncate_all(lengths)?;
        Ok(())
    }

    /// Returns whether an all-bank speculative transaction is active.
    #[must_use]
    pub const fn has_active_transaction(&self) -> bool {
        self.active_transaction.is_some()
    }

    /// Returns the state tensor device.
    #[must_use]
    pub const fn device(&self) -> &B::Device {
        &self.device
    }

    /// Validates the identity and baseline encoded by a transaction mark.
    fn validate_mark(&self, mark: &TensorCacheMark) -> TensorCacheResult<&ActiveTensorTransaction> {
        let active = self
            .active_transaction
            .as_ref()
            .ok_or(TensorCacheError::TransactionNotActive)?;
        if active.token != mark.token || active.baseline_lengths != mark.baseline_lengths {
            return Err(TensorCacheError::StaleTransactionMark);
        }
        Ok(active)
    }

    /// Rejects lifecycle operations while staged values are reachable.
    fn ensure_no_transaction(&self, operation: &'static str) -> TensorCacheResult<()> {
        if self.has_active_transaction() {
            Err(TensorCacheError::TransactionActive { operation })
        } else {
            Ok(())
        }
    }
}

impl<B: Backend> ArchitectureDecodeState<B> for TensorDecodeState<B> {
    fn batch_size(&self) -> usize {
        self.main.batch_size()
    }

    fn capacity(&self) -> usize {
        self.metadata.config().capacity()
    }

    fn lengths(&self) -> &[usize] {
        self.metadata.lengths()
    }

    fn reset(&mut self) -> crate::Result<()> {
        TensorDecodeState::reset(self).map_err(Into::into)
    }

    fn reorder(&mut self, indices: &[usize]) -> crate::Result<()> {
        TensorDecodeState::reorder(self, indices).map_err(Into::into)
    }

    fn truncate(&mut self, lengths: &[usize]) -> crate::Result<()> {
        TensorDecodeState::truncate(self, lengths).map_err(Into::into)
    }

    fn device(&self) -> &B::Device {
        TensorDecodeState::device(self)
    }
}

/// Validates a per-sequence vector length.
fn validate_count(expected: usize, actual: usize) -> TensorCacheResult<()> {
    if expected == actual {
        Ok(())
    } else {
        Err(DecodeStateError::SequenceCountMismatch { expected, actual }.into())
    }
}

/// Computes each proposal suffix from current and baseline logical lengths.
fn proposal_lengths(baseline: &[usize], current: &[usize]) -> Vec<usize> {
    baseline
        .iter()
        .zip(current)
        .map(|(&start, &end)| end - start)
        .collect()
}

/// Ensures no accepted count exceeds its logical proposal length.
fn validate_accepted(proposed: &[usize], accepted: &[usize]) -> TensorCacheResult<()> {
    for (sequence, (&proposed, &accepted)) in proposed.iter().zip(accepted).enumerate() {
        if accepted > proposed {
            return Err(TensorCacheError::AcceptedPrefixExceedsProposal {
                sequence,
                accepted,
                proposed,
            });
        }
    }
    Ok(())
}
