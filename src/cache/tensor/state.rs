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

use burn::tensor::{Int, Tensor, backend::Backend};

use crate::architecture::DecodeState as ArchitectureDecodeState;
use crate::cache::lengths::LengthViolation;
use crate::cache::state::length_error;
use crate::cache::{CacheConfig, CacheMark, DecodeStateError, SequenceLengths, TransactionalCache};

use super::kv::{KvCommitPlan, selection_tensor, validate_batch, validate_reorder_indices};
use super::recurrent::RecurrentCommitPlan;
use super::{
    KvCacheStorage, NativeKvCache, NativeKvCacheSpec, RecurrentStateCache, RecurrentStateSpec,
    TensorCacheError, TensorCacheResult,
};

mod step;

use step::StepVersions;
pub use step::{StepBank, TensorStepWrites, TokenAlignment};

// Bank allocation and coordinated tensor writes.

/// A composable collection of attention and recurrent cache allocations.
///
/// A bank can represent the main decoder or an auxiliary multi-token-prediction
/// (MTP) stack. Every cache in the bank shares batch size, logical capacity, and
/// device, while individual groups may have different head/state geometry.
#[derive(Debug)]
pub struct TensorCacheBank<B: Backend> {
    /// Logical geometry shared by all cache groups.
    config: CacheConfig,
    /// Fixed number of independently tracked tensor rows.
    batch_size: usize,
    /// Device shared by every tensor allocation in this bank.
    device: B::Device,
    /// Homogeneous attention-cache groups indexed by model wiring.
    native_kv: Vec<NativeKvCache<B>>,
    /// Homogeneous recurrent-cache groups indexed by model wiring.
    recurrent: Vec<RecurrentStateCache<B>>,
    /// Topology lock while the bank participates in a transaction.
    staging_active: bool,
    /// Whether the owning decode state has committed tokens for this bank.
    ///
    /// A group added afterwards would start from zero tensors while the
    /// logical lengths claim real history, so topology changes are refused
    /// until the state is reset.
    holds_history: bool,
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
            holds_history: false,
        })
    }

    /// Creates an empty bank whose batch size is `config.sequence_count()`.
    #[must_use]
    pub fn with_config(config: CacheConfig, device: &B::Device) -> Self {
        Self {
            config,
            batch_size: config.sequence_count(),
            device: device.clone(),
            native_kv: Vec::new(),
            recurrent: Vec::new(),
            staging_active: false,
            holds_history: false,
        }
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
    /// Groups can only be added while the owning decode state holds no
    /// tokens: a new group starts from zero tensors and would otherwise
    /// silently expose zero-filled "history" to attention.
    ///
    /// # Errors
    ///
    /// Returns an error if a transaction is active, the owning state already
    /// holds tokens, or allocation geometry is incompatible with this bank's
    /// config, backend, or device.
    pub fn add_native_kv(&mut self, spec: NativeKvCacheSpec) -> TensorCacheResult<usize> {
        self.ensure_topology_mutation_allowed("add native KV cache")?;
        let cache = NativeKvCache::new(self.config, self.batch_size, spec, &self.device)?;
        let index = self.native_kv.len();
        self.native_kv.push(cache);
        Ok(index)
    }

    /// Allocates and appends one homogeneous recurrent-state cache group.
    ///
    /// Like [`add_native_kv`](Self::add_native_kv), this is refused once the
    /// owning state holds tokens, because a new recurrent group would silently
    /// restart from the zero state.
    ///
    /// # Errors
    ///
    /// Returns an error if a transaction is active, the owning state already
    /// holds tokens, or allocation geometry is incompatible with this bank's
    /// backend or device.
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
    ///
    /// `baseline` is this bank's committed lengths, which in-place KV groups
    /// use as their first proposal slot.
    fn begin_staging(&mut self, baseline: &[usize]) -> TensorCacheResult<()> {
        for cache in &mut self.native_kv {
            cache.begin_staging_from(baseline)?;
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
    ) -> TensorCacheResult<TensorBankCommitPlan<B>> {
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
    fn apply_staging_commit(&mut self, plan: TensorBankCommitPlan<B>) {
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

    /// Reorders or duplicates every committed tensor's batch rows.
    ///
    /// `selection` holds validated source rows, built once by the decode state
    /// and shared by every group of both banks.
    fn reorder_with(&mut self, selection: &Tensor<B, 1, Int>) {
        for cache in &mut self.native_kv {
            cache.reorder_rows(selection);
        }
        for cache in &mut self.recurrent {
            cache.reorder_rows(selection);
        }
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

    /// Prevents bank topology changes while staging or after tokens were committed.
    fn ensure_topology_mutation_allowed(&self, operation: &'static str) -> TensorCacheResult<()> {
        if self.has_active_staging() {
            Err(TensorCacheError::TransactionActive { operation })
        } else if self.holds_history {
            Err(TensorCacheError::TopologyAfterHistory { operation })
        } else {
            Ok(())
        }
    }
}

/// Commit capabilities prepared for every tensor cache in one bank.
#[derive(Debug)]
struct TensorBankCommitPlan<B: Backend> {
    /// Per-group KV plans retaining validated staging allocations.
    native_kv: Vec<KvCommitPlan<B>>,
    /// Per-group recurrent plans retaining validated snapshot allocations.
    recurrent: Vec<RecurrentCommitPlan<B>>,
}

// Session transactions and lifecycle operations.

/// Single-use identity returned for a tensor-cache speculative transaction.
///
/// It pairs the main-bank [`CacheMark`] with an independent MTP mark when an
/// auxiliary bank is enabled. Both marks come from the same verified ledger
/// implementation as [`SequenceLengths`], so the tensor and metadata
/// transactions share one state machine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorCacheMark {
    /// Main-bank transaction identity and committed baseline.
    main: CacheMark,
    /// Independent auxiliary identity and baseline, when an MTP bank is enabled.
    mtp: Option<CacheMark>,
}

impl TensorCacheMark {
    /// Returns the committed logical lengths captured before the proposal.
    #[must_use]
    pub fn baseline_lengths(&self) -> &[usize] {
        self.main.baseline_lengths()
    }

    /// Returns independent MTP baselines when an auxiliary bank was enabled.
    #[must_use]
    pub fn mtp_baseline_lengths(&self) -> Option<&[usize]> {
        self.mtp.as_ref().map(CacheMark::baseline_lengths)
    }
}

/// External, composable model state implementing the architecture lifecycle.
///
/// `main` contains the decoder's ordinary layer caches. `mtp`, when enabled,
/// is a distinct bank for multi-token prediction layers: it shares batch-row
/// ordering but owns an independent logical cursor and accepted-prefix policy.
/// Neither bank is a Burn module or checkpoint parameter.
///
/// Logical lengths of each bank live in a [`SequenceLengths`] whose verified
/// ledger also runs the speculative transaction, so [`lengths`](Self::lengths)
/// is always the committed cursor and [`working_lengths`](Self::working_lengths)
/// the tentative one, exactly as for the metadata-only type.
///
/// The crate-wide [`TransactionalCache`] implementation controls a uniform main
/// prefix and is sufficient for generic speculative generation. A non-empty
/// MTP proposal intentionally returns
/// [`TensorCacheError::MtpAcceptedPrefixRequired`]; architecture-aware Qwen MTP
/// code uses [`commit_prefixes`](Self::commit_prefixes) to provide both ragged
/// prefix vectors without downcasting cache internals.
#[derive(Debug)]
pub struct TensorDecodeState<B: Backend> {
    /// Committed and working main-bank lengths plus the transaction identity.
    metadata: SequenceLengths,
    /// Ordinary decoder attention and recurrent allocations.
    main: TensorCacheBank<B>,
    /// Optional auxiliary multi-token-prediction allocations.
    mtp: Option<TensorCacheBank<B>>,
    /// Independent auxiliary lengths and transaction; present exactly when `mtp` is.
    mtp_metadata: Option<SequenceLengths>,
    /// Device required by all attached tensor banks.
    device: B::Device,
    /// State identity and per-bank lifecycle versions that step write sets must match.
    versions: StepVersions,
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
            metadata: SequenceLengths::new(config),
            main: TensorCacheBank::new(config, batch_size, device)?,
            mtp: None,
            mtp_metadata: None,
            device: device.clone(),
            versions: StepVersions::new(),
        })
    }

    /// Creates an empty model state whose batch size is `config.sequence_count()`.
    ///
    /// Equivalent to [`new`](Self::new) without repeating the batch size.
    #[must_use]
    pub fn with_config(config: CacheConfig, device: &B::Device) -> Self {
        Self {
            metadata: SequenceLengths::new(config),
            main: TensorCacheBank::with_config(config, device),
            mtp: None,
            mtp_metadata: None,
            device: device.clone(),
            versions: StepVersions::new(),
        }
    }

    /// Returns the main bank's logical lengths, including any active transaction.
    #[must_use]
    pub const fn metadata(&self) -> &SequenceLengths {
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
        self.metadata.working_lengths()
    }

    /// Returns committed MTP lengths when the auxiliary bank is enabled.
    #[must_use]
    pub fn mtp_lengths(&self) -> Option<&[usize]> {
        self.mtp_metadata.as_ref().map(SequenceLengths::lengths)
    }

    /// Returns MTP lengths visible to the next speculative MTP step.
    #[must_use]
    pub fn mtp_working_lengths(&self) -> Option<&[usize]> {
        self.mtp_metadata
            .as_ref()
            .map(SequenceLengths::working_lengths)
    }

    /// Borrows the main decoder cache bank.
    #[must_use]
    pub const fn main(&self) -> &TensorCacheBank<B> {
        &self.main
    }

    /// Mutably borrows the main decoder cache bank.
    ///
    /// Mutable access can change tensors behind the state's cursors, so it
    /// invalidates every outstanding main-bank [`TensorStepWrites`].
    #[must_use]
    pub const fn main_mut(&mut self) -> &mut TensorCacheBank<B> {
        self.versions.bump(StepBank::Main);
        &mut self.main
    }

    /// Borrows the independent MTP cache bank when enabled.
    #[must_use]
    pub const fn mtp(&self) -> Option<&TensorCacheBank<B>> {
        self.mtp.as_ref()
    }

    /// Mutably borrows the independent MTP cache bank when enabled.
    ///
    /// Like [`main_mut`](Self::main_mut), this invalidates every outstanding
    /// MTP-bank [`TensorStepWrites`].
    #[must_use]
    pub const fn mtp_mut(&mut self) -> Option<&mut TensorCacheBank<B>> {
        self.versions.bump(StepBank::Mtp);
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
        self.mtp_metadata = Some(SequenceLengths::new(self.metadata.config()));
        self.versions.bump(StepBank::Mtp);
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
        self.mtp_metadata = Some(SequenceLengths::new(self.metadata.config()));
        self.versions.bump(StepBank::Mtp);
        Ok(())
    }

    /// Atomically advances all logical sequence lengths.
    ///
    /// Models should populate every layer cache for these tokens before calling
    /// this method. During speculation only the working cursor moves, and the
    /// staged length for each cache group is checked against this logical
    /// suffix at commit time.
    ///
    /// # Errors
    ///
    /// Returns metadata errors for count, integer, or capacity overflow.
    pub fn advance(&mut self, token_counts: &[usize]) -> TensorCacheResult<()> {
        self.metadata.advance_all(token_counts)?;
        self.versions.bump(StepBank::Main);
        self.sync_history_locks();
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
        self.mtp_metadata
            .as_mut()
            .ok_or(TensorCacheError::MtpNotEnabled)?
            .advance_all(token_counts)?;
        self.versions.bump(StepBank::Mtp);
        self.sync_history_locks();
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
        if self.has_active_transaction() {
            return Err(TensorCacheError::TransactionAlreadyActive);
        }
        self.main.validate_can_begin_staging()?;
        if let Some(mtp) = &self.mtp {
            mtp.validate_can_begin_staging()?;
        }
        self.main.begin_staging(self.metadata.lengths())?;
        if let (Some(mtp), Some(metadata)) = (&mut self.mtp, &self.mtp_metadata) {
            mtp.begin_staging(metadata.lengths())?;
        }
        // Steps derived outside the transaction would write committed slots.
        self.versions.bump_all();
        // Neither ledger is active (checked above), so both marks succeed.
        let main = self.metadata.mark()?;
        let mtp = self
            .mtp_metadata
            .as_mut()
            .map(TransactionalCache::mark)
            .transpose()?;
        Ok(TensorCacheMark { main, mtp })
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
        self.validate_mark(&mark)?;
        let batch_size = self.batch_size();
        validate_count(batch_size, main_accepted.len())?;
        let main_next = self
            .metadata
            .plan_commit(main_accepted)
            .map_err(commit_error)?;
        let main_proposed = self.metadata.proposal_lengths();
        let main_plan = self.main.prepare_staging_commit(
            self.metadata.lengths(),
            &main_proposed,
            main_accepted,
        )?;

        let (mtp_plan, mtp_next) = match (self.mtp.as_ref(), self.mtp_metadata.as_ref()) {
            (Some(bank), Some(metadata)) => {
                let proposed = metadata.proposal_lengths();
                let accepted = match mtp_accepted {
                    Some(accepted) => accepted.to_vec(),
                    None if proposed.iter().all(|&count| count == 0) => {
                        // Empty MTP proposals have the unique valid zero prefix.
                        vec![0; batch_size]
                    }
                    None => return Err(TensorCacheError::MtpAcceptedPrefixRequired),
                };
                validate_count(batch_size, accepted.len())?;
                let next = metadata.plan_commit(&accepted).map_err(commit_error)?;
                let plan = bank.prepare_staging_commit(metadata.lengths(), &proposed, &accepted)?;
                (Some(plan), Some(next))
            }
            (None, None) => {
                if mtp_accepted.is_some() {
                    return Err(TensorCacheError::MtpNotEnabled);
                }
                (None, None)
            }
            _ => return Err(TensorCacheError::MtpNotEnabled),
        };

        // Every recoverable check has passed; publication is infallible.
        self.main.apply_staging_commit(main_plan);
        if let (Some(bank), Some(plan)) = (&mut self.mtp, mtp_plan) {
            bank.apply_staging_commit(plan);
        }
        self.metadata.finish_transaction(main_next);
        if let (Some(metadata), Some(next)) = (&mut self.mtp_metadata, mtp_next) {
            metadata.finish_transaction(next);
        }
        self.versions.bump_all();
        self.sync_history_locks();
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
        let main_next = self
            .metadata
            .plan_rollback()
            .map_err(|_| TensorCacheError::TransactionNotActive)?;
        let mtp_next = self
            .mtp_metadata
            .as_ref()
            .map(SequenceLengths::plan_rollback)
            .transpose()
            .map_err(|_| TensorCacheError::TransactionNotActive)?;
        self.main.rollback_staging();
        if let Some(mtp) = &mut self.mtp {
            mtp.rollback_staging();
        }
        self.metadata.finish_transaction(main_next);
        if let (Some(metadata), Some(next)) = (&mut self.mtp_metadata, mtp_next) {
            metadata.finish_transaction(next);
        }
        self.versions.bump_all();
        Ok(())
    }

    /// Clears every tensor and logical length for session reuse.
    ///
    /// # Errors
    ///
    /// Returns an error while a speculative transaction is active.
    pub fn reset(&mut self) -> TensorCacheResult<()> {
        self.ensure_no_transaction("reset")?;
        let main_next = self.metadata.plan_reset()?;
        let mtp_next = self
            .mtp_metadata
            .as_ref()
            .map(SequenceLengths::plan_reset)
            .transpose()?;
        self.main.reset();
        if let Some(mtp) = &mut self.mtp {
            mtp.reset();
        }
        self.metadata.publish(main_next);
        if let (Some(metadata), Some(next)) = (&mut self.mtp_metadata, mtp_next) {
            metadata.publish(next);
        }
        self.versions.bump_all();
        self.sync_history_locks();
        Ok(())
    }

    /// Reorders or duplicates all main, MTP, and metadata batch rows.
    ///
    /// Duplicate source indices implement beam expansion without reallocating a
    /// larger batch; output row count must remain unchanged. One device
    /// selector is built and shared by every cache group in both banks.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid indices or an active transaction.
    pub fn reorder(&mut self, indices: &[usize]) -> TensorCacheResult<()> {
        self.ensure_no_transaction("reorder")?;
        validate_reorder_indices(indices, self.main.batch_size())?;
        let main_next = self.metadata.plan_reorder(indices)?;
        let mtp_next = self
            .mtp_metadata
            .as_ref()
            .map(|metadata| metadata.plan_reorder(indices))
            .transpose()?;
        let selection = selection_tensor::<B>(indices, &self.device)?;
        self.main.reorder_with(&selection);
        if let Some(mtp) = &mut self.mtp {
            mtp.reorder_with(&selection);
        }
        self.metadata.publish(main_next);
        if let (Some(metadata), Some(next)) = (&mut self.mtp_metadata, mtp_next) {
            metadata.publish(next);
        }
        self.versions.bump_all();
        self.sync_history_locks();
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
        let next = self.metadata.plan_truncate(lengths)?;
        self.main.validate_truncate(&current, lengths)?;
        self.main.apply_truncate(&current, lengths);
        self.metadata.publish(next);
        self.versions.bump(StepBank::Main);
        self.sync_history_locks();
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
        let (Some(metadata), Some(bank)) = (self.mtp_metadata.as_mut(), self.mtp.as_mut()) else {
            return Err(TensorCacheError::MtpNotEnabled);
        };
        let current = metadata.lengths().to_vec();
        let next = metadata.plan_truncate(lengths)?;
        bank.validate_truncate(&current, lengths)?;
        bank.apply_truncate(&current, lengths);
        metadata.publish(next);
        self.versions.bump(StepBank::Mtp);
        self.sync_history_locks();
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
        let main_next = self.metadata.plan_truncate(main_lengths)?;
        self.main.validate_truncate(&main_current, main_lengths)?;

        let (Some(mtp_metadata), Some(mtp_bank)) = (self.mtp_metadata.as_mut(), self.mtp.as_mut())
        else {
            return Err(TensorCacheError::MtpNotEnabled);
        };
        let mtp_current = mtp_metadata.lengths().to_vec();
        let mtp_next = mtp_metadata.plan_truncate(mtp_lengths)?;
        mtp_bank.validate_truncate(&mtp_current, mtp_lengths)?;

        self.main.apply_truncate(&main_current, main_lengths);
        mtp_bank.apply_truncate(&mtp_current, mtp_lengths);
        self.metadata.publish(main_next);
        mtp_metadata.publish(mtp_next);
        self.versions.bump_all();
        self.sync_history_locks();
        Ok(())
    }

    /// Returns whether an all-bank speculative transaction is active.
    #[must_use]
    pub const fn has_active_transaction(&self) -> bool {
        self.metadata.has_active_transaction()
    }

    /// Returns the state tensor device.
    #[must_use]
    pub const fn device(&self) -> &B::Device {
        &self.device
    }

    /// Records in each bank whether its committed lengths hold any token.
    ///
    /// Called after every successful length change so a bank can refuse new
    /// groups exactly while their zero tensors would contradict the lengths.
    fn sync_history_locks(&mut self) {
        self.main.holds_history = self.metadata.lengths().iter().any(|&length| length > 0);
        if let (Some(bank), Some(metadata)) = (&mut self.mtp, &self.mtp_metadata) {
            bank.holds_history = metadata.lengths().iter().any(|&length| length > 0);
        }
    }

    /// Validates that a mark names the active transaction of both banks.
    fn validate_mark(&self, mark: &TensorCacheMark) -> TensorCacheResult<()> {
        if !self.has_active_transaction() {
            return Err(TensorCacheError::TransactionNotActive);
        }
        let main_matches = self.metadata.check_mark(&mark.main).is_ok();
        let mtp_matches = match (&self.mtp_metadata, &mark.mtp) {
            (Some(metadata), Some(mark)) => metadata.check_mark(mark).is_ok(),
            (None, None) => true,
            _ => false,
        };
        if main_matches && mtp_matches {
            Ok(())
        } else {
            Err(TensorCacheError::StaleTransactionMark)
        }
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
    /// Single-use checkpoint of both banks' independent committed lengths.
    type Mark = TensorCacheMark;
    /// Recoverable tensor, metadata, and transaction validation errors.
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

/// Maps a rejected verified commit plan to the tensor-state error vocabulary.
fn commit_error(violation: LengthViolation) -> TensorCacheError {
    match violation {
        LengthViolation::PrefixExceedsProposal {
            sequence,
            accepted,
            proposed,
        } => TensorCacheError::AcceptedPrefixExceedsProposal {
            sequence,
            accepted,
            proposed,
        },
        LengthViolation::TransactionInactive => TensorCacheError::TransactionNotActive,
        violation => length_error(violation, "commit").into(),
    }
}

#[cfg(test)]
#[path = "state_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "state_model_tests.rs"]
mod model_tests;
