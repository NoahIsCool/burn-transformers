//! Composable model-level decode state and lifecycle coordination.
//!
//! The coordinator keeps absolute lengths alongside independent native-KV and
//! recurrent banks. KV tensors use `[batch, kv_head, time, channel]`; recurrent
//! geometry is declared per group. Main and optional auxiliary banks share a
//! device and fixed batch/capacity while retaining their own accepted progress.
//!
//! Ordinary writes are prepared before publication. Speculation stages KV values
//! and recurrent terminal snapshots, validates every accepted prefix, then commits
//! tensor storage and metadata together. Reset/reorder/truncate cannot cross an
//! active transaction. These mutable execution tensors remain outside trainable
//! model parameters, and storage dtype is explicit in each group's specification.

use std::sync::Arc;

use burn::tensor::backend::Backend;

use crate::architecture::DecodeState as ArchitectureDecodeState;
use crate::cache::{
    CacheConfig, DecodeState as DecodeMetadata, DecodeStateError, TransactionalCache,
};

use super::kv::{KvCommitPlan, PreparedNativeKvWrite, validate_batch, validate_reorder_indices};
use super::recurrent::{PreparedRecurrentStateWrite, RecurrentCommitPlan};
use super::{
    NativeKvCache, NativeKvCacheSpec, RecurrentStateCache, RecurrentStateSpec, TensorCacheError,
    TensorCacheResult,
};

// Bank allocation and coordinated tensor writes.

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
    staging_active: bool,
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
            staging_active: false,
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

    /// Seals already validated ordinary writes into one bank-level capability.
    ///
    /// Both group indices and the bank transaction state are checked before a
    /// model advances logical metadata. The returned value can subsequently be
    /// applied without another lookup or recoverable branch.
    ///
    /// # Errors
    ///
    /// Returns an error while speculative staging is active or when either
    /// requested group index is absent.
    pub(crate) fn prepare_direct_writes(
        &self,
        native_kv: Option<(usize, Vec<PreparedNativeKvWrite<B>>)>,
        recurrent: Option<(usize, Vec<PreparedRecurrentStateWrite<B>>)>,
    ) -> TensorCacheResult<PreparedTensorBankWrites<B>> {
        if self.staging_active {
            return Err(TensorCacheError::TransactionActive {
                operation: "prepare ordinary cache writes",
            });
        }
        if let Some((index, _)) = &native_kv {
            self.native_kv(*index)?;
        }
        if let Some((index, _)) = &recurrent {
            self.recurrent(*index)?;
        }
        Ok(PreparedTensorBankWrites {
            native_kv,
            recurrent,
        })
    }

    /// Publishes a bank-level capability after all recoverable work succeeds.
    ///
    /// The prepare phase proved every index and individual tensor write, so the
    /// application phase consists solely of Burn assignments and cannot return
    /// an error between cache families.
    pub(crate) fn apply_prepared_direct_writes(&mut self, writes: PreparedTensorBankWrites<B>) {
        if let Some((index, writes)) = writes.native_kv {
            self.native_kv[index].apply_prepared_writes(writes);
        }
        if let Some((index, writes)) = writes.recurrent {
            self.recurrent[index].apply_prepared_state_writes(writes);
        }
    }

    /// Returns whether this bank is locked for a model-level transaction.
    ///
    /// The marker is bank-owned rather than inferred from child caches, so an
    /// empty bank is still topology-locked after a transaction begins.
    #[must_use]
    pub const fn has_active_staging(&self) -> bool {
        self.staging_active
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
        self.staging_active = true;
        Ok(())
    }

    /// Creates an all-cache commit plan before any committed tensor mutates.
    fn prepare_staging_commit(
        &self,
        baseline: &[usize],
        proposed: &[usize],
        accepted: &[usize],
    ) -> TensorCacheResult<TensorBankCommitPlan> {
        if !self.staging_active {
            return Err(TensorCacheError::TransactionNotActive);
        }
        let native_kv = self
            .native_kv
            .iter()
            .map(|cache| cache.prepare_staging_commit(baseline, proposed, accepted))
            .collect::<TensorCacheResult<_>>()?;
        let recurrent = self
            .recurrent
            .iter()
            .map(|cache| cache.prepare_staging_commit(proposed, accepted))
            .collect::<TensorCacheResult<_>>()?;
        Ok(TensorBankCommitPlan {
            native_kv,
            recurrent,
        })
    }

    /// Consumes a fully validated plan using only infallible cache operations.
    fn apply_staging_commit(&mut self, plan: TensorBankCommitPlan) {
        for (cache, cache_plan) in self.native_kv.iter_mut().zip(plan.native_kv) {
            cache.apply_staging_commit(cache_plan);
        }
        for (cache, cache_plan) in self.recurrent.iter_mut().zip(plan.recurrent) {
            cache.apply_staging_commit(cache_plan);
        }
        self.staging_active = false;
    }

    /// Discards all proposal storage without changing committed tensors.
    fn rollback_staging(&mut self) {
        for cache in &mut self.native_kv {
            cache.rollback_staging();
        }
        for cache in &mut self.recurrent {
            cache.rollback_staging();
        }
        self.staging_active = false;
    }

    /// Clears every state tensor and transaction marker.
    fn reset(&mut self) {
        for cache in &mut self.native_kv {
            cache.reset();
        }
        for cache in &mut self.recurrent {
            cache.reset();
        }
        self.staging_active = false;
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

/// Commit capabilities prepared for every tensor cache in one bank.
#[derive(Debug)]
struct TensorBankCommitPlan {
    native_kv: Vec<KvCommitPlan>,
    recurrent: Vec<RecurrentCommitPlan>,
}

/// Fully prepared ordinary writes spanning both cache families in one bank.
///
/// This capability is distinct from speculative staging: it lets a normal
/// prefill or decode call defer all physical cache publication until its model
/// computation and every recoverable write validation have succeeded.
#[derive(Debug)]
pub(crate) struct PreparedTensorBankWrites<B: Backend> {
    native_kv: Option<(usize, Vec<PreparedNativeKvWrite<B>>)>,
    recurrent: Option<(usize, Vec<PreparedRecurrentStateWrite<B>>)>,
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
    /// Formats the tensor transaction token without exposing its allocation address.
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_str("TensorTransactionToken(..)")
    }
}

impl PartialEq for TensorTransactionToken {
    /// Compares allocation identity so marks cannot authorize a different tensor-cache transaction.
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for TensorTransactionToken {}

// Session transactions and lifecycle operations.

/// Single-use identity returned for a tensor-cache speculative transaction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorCacheMark {
    token: TensorTransactionToken,
    baseline_lengths: Vec<usize>,
    mtp_baseline_lengths: Option<Vec<usize>>,
}

impl TensorCacheMark {
    /// Returns the committed logical lengths captured before the proposal.
    #[must_use]
    pub fn baseline_lengths(&self) -> &[usize] {
        &self.baseline_lengths
    }

    /// Returns independent MTP baselines when an auxiliary bank was enabled.
    #[must_use]
    pub fn mtp_baseline_lengths(&self) -> Option<&[usize]> {
        self.mtp_baseline_lengths.as_deref()
    }
}

#[derive(Debug)]
struct ActiveTensorTransaction {
    token: TensorTransactionToken,
    baseline_lengths: Vec<usize>,
    working_lengths: Vec<usize>,
    mtp_baseline_lengths: Option<Vec<usize>>,
    mtp_working_lengths: Option<Vec<usize>>,
}

/// External, composable model state implementing the architecture lifecycle.
///
/// `main` contains the decoder's ordinary layer caches. `mtp`, when enabled,
/// is a distinct bank for multi-token prediction layers: it shares batch-row
/// ordering but owns an independent logical cursor and accepted-prefix policy.
/// Neither bank is a Burn module or checkpoint parameter.
///
/// The crate-wide [`TransactionalCache`] implementation controls a uniform main
/// prefix and is sufficient for generic speculative generation. A non-empty
/// MTP proposal intentionally returns
/// [`TensorCacheError::MtpAcceptedPrefixRequired`]; architecture-aware Qwen MTP
/// code uses [`commit_prefixes`](Self::commit_prefixes) to provide both ragged
/// prefix vectors without downcasting cache internals.
#[derive(Debug)]
pub struct TensorDecodeState<B: Backend> {
    metadata: DecodeMetadata,
    main: TensorCacheBank<B>,
    mtp: Option<TensorCacheBank<B>>,
    mtp_metadata: Option<DecodeMetadata>,
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
            mtp_metadata: None,
            device: device.clone(),
            active_transaction: None,
        })
    }

    /// Returns backend-independent logical length metadata.
    #[must_use]
    pub const fn metadata(&self) -> &DecodeMetadata {
        &self.metadata
    }

    /// Returns the validated logical cache configuration.
    #[must_use]
    pub const fn config(&self) -> CacheConfig {
        self.metadata.config()
    }

    /// Returns the fixed number of independent sequence rows.
    #[must_use]
    pub const fn batch_size(&self) -> usize {
        self.main.batch_size()
    }

    /// Returns the maximum logical token count for every sequence.
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.metadata.config().capacity()
    }

    /// Returns each sequence's number of committed logical tokens.
    #[must_use]
    pub fn lengths(&self) -> &[usize] {
        self.metadata.lengths()
    }

    /// Returns lengths visible to the next main-model speculative step.
    ///
    /// Outside a transaction these equal [`lengths`](Self::lengths). During a
    /// transaction they include tentative proposal tokens while `lengths()`
    /// remains the architecture contract's committed cursor.
    #[must_use]
    pub fn working_lengths(&self) -> &[usize] {
        self.active_transaction
            .as_ref()
            .map_or_else(|| self.metadata.lengths(), |active| &active.working_lengths)
    }

    /// Returns committed MTP lengths when the auxiliary bank is enabled.
    #[must_use]
    pub fn mtp_lengths(&self) -> Option<&[usize]> {
        self.mtp_metadata.as_ref().map(DecodeMetadata::lengths)
    }

    /// Returns MTP lengths visible to the next speculative MTP step.
    #[must_use]
    pub fn mtp_working_lengths(&self) -> Option<&[usize]> {
        match &self.active_transaction {
            Some(active) => active.mtp_working_lengths.as_deref(),
            None => self.mtp_lengths(),
        }
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
        self.mtp_metadata = Some(DecodeMetadata::new(self.metadata.config()));
        Ok(self.mtp.insert(bank))
    }

    /// Attaches an independently assembled MTP bank after compatibility checks.
    ///
    /// # Errors
    ///
    /// Returns an error for an existing MTP bank, active transaction, different
    /// cache config or device.
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
        if bank.device() != &self.device {
            return Err(TensorCacheError::IncompatibleBank {
                bank: "MTP",
                reason: "device differs from main state",
            });
        }
        self.mtp = Some(bank);
        self.mtp_metadata = Some(DecodeMetadata::new(self.metadata.config()));
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
        if let Some(active) = &mut self.active_transaction {
            advance_working_lengths(
                &mut active.working_lengths,
                token_counts,
                self.metadata.config().capacity(),
            )?;
        } else {
            self.metadata.advance_all(token_counts)?;
        }
        Ok(())
    }

    /// Atomically advances the independent MTP logical cursor.
    ///
    /// MTP may consume a shifted stream and therefore may lag or advance by a
    /// different proposal count than the main decoder. During a transaction
    /// only the MTP working cursor changes.
    ///
    /// # Errors
    ///
    /// Returns an error when MTP is disabled or for count/capacity overflow.
    pub fn advance_mtp(&mut self, token_counts: &[usize]) -> TensorCacheResult<()> {
        let metadata = self
            .mtp_metadata
            .as_mut()
            .ok_or(TensorCacheError::MtpNotEnabled)?;
        if let Some(active) = &mut self.active_transaction {
            let working = active
                .mtp_working_lengths
                .as_mut()
                .ok_or(TensorCacheError::MtpNotEnabled)?;
            advance_working_lengths(working, token_counts, metadata.config().capacity())?;
        } else {
            metadata.advance_all(token_counts)?;
        }
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
        let mtp_baseline_lengths = self
            .mtp_metadata
            .as_ref()
            .map(|metadata| metadata.lengths().to_vec());
        self.active_transaction = Some(ActiveTensorTransaction {
            token: token.clone(),
            baseline_lengths: baseline_lengths.clone(),
            working_lengths: baseline_lengths.clone(),
            mtp_baseline_lengths: mtp_baseline_lengths.clone(),
            mtp_working_lengths: mtp_baseline_lengths.clone(),
        });
        Ok(TensorCacheMark {
            token,
            baseline_lengths,
            mtp_baseline_lengths,
        })
    }

    /// Commits a ragged main-model prefix when MTP staged no tokens.
    ///
    /// Use [`commit_prefixes`](Self::commit_prefixes) when MTP has an
    /// independent proposal. Validation covers every cache group before any
    /// committed tensor changes.
    ///
    /// # Errors
    ///
    /// Returns an error for a stale mark, invalid prefix, incomplete physical
    /// staging, or an MTP proposal that needs its own accepted-prefix vector.
    pub fn commit_prefix(
        &mut self,
        mark: TensorCacheMark,
        accepted: &[usize],
    ) -> TensorCacheResult<()> {
        self.commit_prefixes(mark, accepted, None)
    }

    /// Commits independent main and MTP accepted prefixes atomically.
    ///
    /// KV prefixes are copied from linear staging into full/ring storage;
    /// recurrent caches select snapshot `accepted - 1`. Rejected values never
    /// touch committed wrapped-ring slots. The main and MTP proposal lengths
    /// may differ for every sequence because shifted MTP decoding has its own
    /// cursor.
    ///
    /// # Errors
    ///
    /// Returns an error for a stale mark, count mismatch, over-acceptance, or
    /// physically incomplete staging. Validation failure leaves the transaction
    /// active and recoverable by [`rollback`](Self::rollback).
    pub fn commit_prefixes(
        &mut self,
        mark: TensorCacheMark,
        main_accepted: &[usize],
        mtp_accepted: Option<&[usize]>,
    ) -> TensorCacheResult<()> {
        let active = self.validate_mark(&mark)?;
        let main_baseline = active.baseline_lengths.clone();
        let main_working = active.working_lengths.clone();
        let mtp_baseline = active.mtp_baseline_lengths.clone();
        let mtp_working = active.mtp_working_lengths.clone();
        validate_count(self.main.batch_size(), main_accepted.len())?;
        let main_proposed = proposal_lengths(&main_baseline, &main_working);
        validate_accepted(&main_proposed, main_accepted)?;

        let main_committed = committed_lengths(&main_baseline, main_accepted)?;
        let next_main_metadata =
            DecodeMetadata::from_lengths(self.metadata.config(), main_committed)?;
        let main_plan =
            self.main
                .prepare_staging_commit(&main_baseline, &main_proposed, main_accepted)?;

        let (mtp_plan, next_mtp_metadata) = match (
            self.mtp.as_ref(),
            self.mtp_metadata.as_ref(),
            mtp_baseline.as_deref(),
            mtp_working.as_deref(),
        ) {
            (Some(mtp), Some(metadata), Some(baseline), Some(working)) => {
                let proposed = proposal_lengths(baseline, working);
                let accepted = match mtp_accepted {
                    Some(accepted) => accepted.to_vec(),
                    None if proposed.iter().all(|&count| count == 0) => {
                        // Empty MTP proposals have the unique valid zero prefix.
                        vec![0; self.batch_size()]
                    }
                    None => return Err(TensorCacheError::MtpAcceptedPrefixRequired),
                };
                validate_count(self.batch_size(), accepted.len())?;
                validate_accepted(&proposed, &accepted)?;
                let committed = committed_lengths(baseline, &accepted)?;
                let next = DecodeMetadata::from_lengths(metadata.config(), committed)?;
                let plan = mtp.prepare_staging_commit(baseline, &proposed, &accepted)?;
                (Some(plan), Some(next))
            }
            (None, None, None, None) => {
                if mtp_accepted.is_some() {
                    return Err(TensorCacheError::MtpNotEnabled);
                }
                (None, None)
            }
            _ => return Err(TensorCacheError::MtpNotEnabled),
        };

        self.main.apply_staging_commit(main_plan);
        if let (Some(mtp), Some(plan)) = (&mut self.mtp, mtp_plan) {
            mtp.apply_staging_commit(plan);
        }
        self.metadata = next_main_metadata;
        self.mtp_metadata = next_mtp_metadata;
        self.active_transaction = None;
        Ok(())
    }

    /// Commits the same accepted proposal length for every sequence row.
    ///
    /// # Errors
    ///
    /// Returns the same validation errors as [`commit_prefix`](Self::commit_prefix).
    pub fn commit_uniform_prefix(
        &mut self,
        mark: TensorCacheMark,
        accepted: usize,
    ) -> TensorCacheResult<()> {
        let counts = vec![accepted; self.batch_size()];
        self.commit_prefix(mark, &counts)
    }

    /// Commits uniform but independent main and MTP proposal prefixes.
    ///
    /// # Errors
    ///
    /// Returns the same validation errors as
    /// [`commit_prefixes`](Self::commit_prefixes).
    pub fn commit_uniform_prefixes(
        &mut self,
        mark: TensorCacheMark,
        main_accepted: usize,
        mtp_accepted: Option<usize>,
    ) -> TensorCacheResult<()> {
        let main = vec![main_accepted; self.batch_size()];
        let mtp = mtp_accepted.map(|count| vec![count; self.batch_size()]);
        self.commit_prefixes(mark, &main, mtp.as_deref())
    }

    /// Discards staged tensors without changing committed logical metadata.
    ///
    /// Because proposals were never written into committed rings, rollback is
    /// exact even after a full sliding window has wrapped.
    ///
    /// # Errors
    ///
    /// Returns an error when the mark is stale or belongs to another state.
    pub fn rollback(&mut self, mark: TensorCacheMark) -> TensorCacheResult<()> {
        self.validate_mark(&mark)?;
        self.main.rollback_staging();
        if let Some(mtp) = &mut self.mtp {
            mtp.rollback_staging();
        }
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
        self.metadata = DecodeMetadata::new(self.metadata.config());
        if let Some(metadata) = &mut self.mtp_metadata {
            *metadata = DecodeMetadata::new(metadata.config());
        }
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
        let reordered_main: Vec<usize> = indices
            .iter()
            .map(|&index| self.metadata.lengths()[index])
            .collect();
        let next_main = DecodeMetadata::from_lengths(self.metadata.config(), reordered_main)?;
        let next_mtp = self
            .mtp_metadata
            .as_ref()
            .map(|metadata| {
                let lengths = indices
                    .iter()
                    .map(|&index| metadata.lengths()[index])
                    .collect::<Vec<_>>();
                DecodeMetadata::from_lengths(metadata.config(), lengths)
            })
            .transpose()?;
        self.main.reorder(indices)?;
        if let Some(mtp) = &mut self.mtp {
            mtp.reorder(indices)?;
        }
        self.metadata = next_main;
        self.mtp_metadata = next_mtp;
        Ok(())
    }

    /// Rewinds main-model lengths when every physical representation can do so.
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
        validate_rewind(&current, lengths)?;
        self.main.validate_truncate(&current, lengths)?;
        let next = DecodeMetadata::from_lengths(self.metadata.config(), lengths.to_vec())?;
        self.main.apply_truncate(&current, lengths);
        self.metadata = next;
        Ok(())
    }

    /// Rewinds independent MTP lengths when its representations support it.
    ///
    /// Keeping this operation separate prevents a target-model rewind from
    /// silently moving a shifted MTP cursor to the same absolute position.
    ///
    /// # Errors
    ///
    /// Returns an error when MTP is disabled, a transaction is active, a length
    /// extends the cursor, or a wrapped/recurrent representation cannot rewind.
    pub fn truncate_mtp(&mut self, lengths: &[usize]) -> TensorCacheResult<()> {
        self.ensure_no_transaction("truncate MTP")?;
        validate_count(self.batch_size(), lengths.len())?;
        let metadata = self
            .mtp_metadata
            .as_ref()
            .ok_or(TensorCacheError::MtpNotEnabled)?;
        let bank = self.mtp.as_ref().ok_or(TensorCacheError::MtpNotEnabled)?;
        let current = metadata.lengths().to_vec();
        validate_rewind(&current, lengths)?;
        bank.validate_truncate(&current, lengths)?;
        let next = DecodeMetadata::from_lengths(metadata.config(), lengths.to_vec())?;
        self.mtp
            .as_mut()
            .expect("MTP bank was validated as present")
            .apply_truncate(&current, lengths);
        self.mtp_metadata = Some(next);
        Ok(())
    }

    /// Atomically rewinds the main and MTP banks to independent prefix lengths.
    ///
    /// Shifted auxiliary streams generally do not share the target model's
    /// absolute cursor. Architecture wrappers therefore derive the surviving
    /// auxiliary prefix and pass both vectors here. Every logical and physical
    /// rewind is validated before either bank or either metadata record is
    /// changed, so an unsafe ring/recurrent rewind leaves the complete state
    /// untouched.
    ///
    /// # Errors
    ///
    /// Returns an error when MTP is disabled, a transaction is active, either
    /// vector has the wrong batch size, a requested length extends its cursor,
    /// or any physical representation cannot reconstruct the requested prefix.
    pub fn truncate_prefixes(
        &mut self,
        main_lengths: &[usize],
        mtp_lengths: &[usize],
    ) -> TensorCacheResult<()> {
        self.ensure_no_transaction("truncate main and MTP")?;
        validate_count(self.batch_size(), main_lengths.len())?;
        validate_count(self.batch_size(), mtp_lengths.len())?;

        let main_current = self.metadata.lengths().to_vec();
        validate_rewind(&main_current, main_lengths)?;
        self.main.validate_truncate(&main_current, main_lengths)?;
        let next_main =
            DecodeMetadata::from_lengths(self.metadata.config(), main_lengths.to_vec())?;

        let mtp_metadata = self
            .mtp_metadata
            .as_ref()
            .ok_or(TensorCacheError::MtpNotEnabled)?;
        let mtp_bank = self.mtp.as_ref().ok_or(TensorCacheError::MtpNotEnabled)?;
        let mtp_current = mtp_metadata.lengths().to_vec();
        validate_rewind(&mtp_current, mtp_lengths)?;
        mtp_bank.validate_truncate(&mtp_current, mtp_lengths)?;
        let next_mtp = DecodeMetadata::from_lengths(mtp_metadata.config(), mtp_lengths.to_vec())?;

        self.main.apply_truncate(&main_current, main_lengths);
        self.mtp
            .as_mut()
            .expect("MTP bank was validated as present")
            .apply_truncate(&mtp_current, mtp_lengths);
        self.metadata = next_main;
        self.mtp_metadata = Some(next_mtp);
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
        if active.token != mark.token
            || active.baseline_lengths != mark.baseline_lengths
            || active.mtp_baseline_lengths != mark.mtp_baseline_lengths
        {
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

impl<B: Backend> TransactionalCache for TensorDecodeState<B> {
    type Mark = TensorCacheMark;
    type Error = TensorCacheError;

    /// Begins a coordinated tensor transaction with external staging for speculative writes.
    fn mark(&mut self) -> Result<Self::Mark, Self::Error> {
        self.begin_transaction()
    }

    /// Publishes a uniform accepted prefix across cache banks after all commit checks succeed.
    fn commit_prefix(
        &mut self,
        mark: Self::Mark,
        accepted_tokens: usize,
    ) -> Result<(), Self::Error> {
        self.commit_uniform_prefix(mark, accepted_tokens)
    }

    /// Discards staged proposals and restores the tensor-state transaction baseline.
    fn rollback(&mut self, mark: Self::Mark) -> Result<(), Self::Error> {
        TensorDecodeState::rollback(self, mark)
    }
}

impl<B: Backend> ArchitectureDecodeState<B> for TensorDecodeState<B> {
    /// Returns the fixed number of independently tracked cache rows.
    fn batch_size(&self) -> usize {
        TensorDecodeState::batch_size(self)
    }

    /// Returns logical context capacity, independent of a layer's physical retention window.
    fn capacity(&self) -> usize {
        TensorDecodeState::capacity(self)
    }

    /// Borrows absolute processed-token counts used by positions and causal offsets.
    fn lengths(&self) -> &[usize] {
        TensorDecodeState::lengths(self)
    }

    /// Resets logical progress and recurrent state through the checked tensor lifecycle operation.
    fn reset(&mut self) -> crate::Result<()> {
        TensorDecodeState::reset(self).map_err(Into::into)
    }

    /// Gathers or duplicates cache rows according to validated beam-parent indices.
    fn reorder(&mut self, indices: &[usize]) -> crate::Result<()> {
        TensorDecodeState::reorder(self, indices).map_err(Into::into)
    }

    /// Truncates every row to a supported earlier state, propagating typed lifecycle errors.
    fn truncate(&mut self, lengths: &[usize]) -> crate::Result<()> {
        TensorDecodeState::truncate(self, lengths).map_err(Into::into)
    }

    /// Returns the backend device shared by the session's cache allocations.
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

/// Rejects a logical rewind that would extend any sequence cursor.
fn validate_rewind(current: &[usize], requested: &[usize]) -> TensorCacheResult<()> {
    for (sequence, (&from, &to)) in current.iter().zip(requested).enumerate() {
        if to > from {
            return Err(DecodeStateError::CannotExtendByTruncation {
                sequence,
                current: from,
                requested: to,
            }
            .into());
        }
    }
    Ok(())
}

/// Advances a tentative cursor atomically without publishing it as committed.
fn advance_working_lengths(
    working: &mut Vec<usize>,
    token_counts: &[usize],
    capacity: usize,
) -> TensorCacheResult<()> {
    validate_count(working.len(), token_counts.len())?;
    let mut next = Vec::with_capacity(working.len());
    for (sequence, (&length, &additional)) in working.iter().zip(token_counts).enumerate() {
        let requested = length
            .checked_add(additional)
            .ok_or(DecodeStateError::LengthOverflow {
                sequence,
                length,
                additional,
            })?;
        if requested > capacity {
            return Err(DecodeStateError::CapacityExceeded {
                sequence,
                requested,
                capacity,
            }
            .into());
        }
        next.push(requested);
    }
    *working = next;
    Ok(())
}

/// Computes each proposal suffix from current and baseline logical lengths.
fn proposal_lengths(baseline: &[usize], current: &[usize]) -> Vec<usize> {
    baseline
        .iter()
        .zip(current)
        .map(|(&start, &end)| end - start)
        .collect()
}

/// Computes committed endpoints with checked arithmetic for public inputs.
fn committed_lengths(baseline: &[usize], accepted: &[usize]) -> TensorCacheResult<Vec<usize>> {
    baseline
        .iter()
        .zip(accepted)
        .enumerate()
        .map(|(sequence, (&length, &additional))| {
            length.checked_add(additional).ok_or_else(|| {
                DecodeStateError::LengthOverflow {
                    sequence,
                    length,
                    additional,
                }
                .into()
            })
        })
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
