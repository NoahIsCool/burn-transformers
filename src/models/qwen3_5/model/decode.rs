//! Cache-aware Qwen3.5/Qwen3.8 causal inference, one batched pass per call.
//!
//! Qwen's hybrid schedule has two physically different histories. A full-attention
//! layer retains every native key/value head, whereas a DeltaNet layer retains only
//! its terminal association matrix and the last `kernel - 1` convolution inputs.
//! [`Qwen3_5DecodeState`] maps model-layer indices monotonically into one cache group
//! of each kind, keeping runtime state outside Burn's parameter [`burn::module::Module`] graph.
//!
//! # Batched execution
//!
//! Every prefill or decode call runs the whole batch through each layer once.
//! The rectangular input is first arranged in the compact right-padded layout
//! of [`CompactBatch`]: row `b`'s `n_b` valid tokens occupy slots `0..n_b`,
//! slot `i` is cache position `W_b + i`, and padding consumes neither a cache
//! position, a rotary position, nor convolution history. Then, per layer:
//!
//! * full attention writes the new rotated keys and values straight into
//!   their cache slots ([`TensorDecodeState::write_kv_ahead`]) and attends over
//!   one zero-copy view of `[0, W_b + n_b)`. A batch whose rows share one
//!   cursor and token count uses the backend's bottom-right causal path with no
//!   mask; any other batch uses one `[B, 1, N, K]` exclusion plane built from
//!   cache positions and shared by every layer;
//! * DeltaNet reads every row's state at once, runs the masked chunked
//!   prefill or the recurrence over the batch, and records the new batch state.
//!
//! All writes belong to one [`TensorStepWrites`], published only after the
//! vocabulary projection succeeds; a failed call leaves every length and every
//! readable cache position unchanged (slots beyond the lengths may hold the
//! abandoned write-ahead tokens, which nothing can read).
//!
//! # Speculative transactions
//!
//! Inside a transaction the step stages proposals instead of committing them.
//! Full-attention groups stage in place, so keys still attend through one
//! view. A DeltaNet recurrence cannot be rewound, so a multi-token step keeps
//! the state after every token ([`GatedDeltaNet::forward_with_snapshots`]) and
//! a later prefix commit selects the snapshot at the accepted boundary.
//!
//! [`GatedDeltaNet::forward_with_snapshots`]: crate::delta_net::GatedDeltaNet::forward_with_snapshots

use burn::tensor::{DType, Int, Tensor, backend::Backend};

use crate::{
    CausalLanguageModel, CausalLmInput, CausalLmOutput, DecodeInput, Error, ForwardOptions, Result,
    architecture::DecodeState as ArchitectureDecodeState,
    cache::{
        CacheConfig, DecodeStateError, KvCacheStorage, NativeKvCacheSpec, RecurrentStateSpec,
        TensorCacheError, TensorCacheMark, TensorDecodeState, TensorStepWrites,
    },
    delta_net::{DeltaNetMode, GatedDeltaNetState},
    models::shared::{
        DeferredChecks, EmptyRows, StatefulContract, StatefulRows, ensure_same_device,
        implicit_position_ids, prepare_stateful_rows, select_hidden_with_last_columns,
        validate_logits_selection,
    },
};

use super::{
    INPUT_NAMES, Qwen3_5DecoderLayer, Qwen3_5ForCausalLm, Qwen3_5MtpPrediction, Qwen3_5TextModel,
    batch::{CachedAttention, CompactBatch, ensure_cacheable_rotary_context},
};
use crate::models::qwen3_5::{
    Qwen3_5LayerType, Qwen3_5TextConfig,
    plan::{self, shifted_mtp_length},
};

/// Single native-KV group within either the target or MTP cache bank.
const NATIVE_KV_GROUP: usize = 0;
/// Single recurrent group containing all target DeltaNet layer states.
const RECURRENT_GROUP: usize = 0;
/// Owner named when an input tensor is on another device than the cache.
const DECODE_STATE: &str = "Qwen decode state";

/// Public stateful operation whose empty-row semantics differ intentionally.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum StatefulPhase {
    /// Every prompt row must contain at least one token.
    Prefill,
    /// Finished rows may contribute zero new tokens while peers continue.
    Decode,
}

impl StatefulPhase {
    /// Returns the DeltaNet kernel appropriate for this operation.
    const fn delta_mode(self) -> DeltaNetMode {
        match self {
            Self::Prefill => DeltaNetMode::ChunkedPrefill,
            Self::Decode => DeltaNetMode::Recurrent,
        }
    }

    /// Returns the phase's policy for rows without a valid token.
    const fn empty_rows(self) -> EmptyRows {
        match self {
            Self::Prefill => EmptyRows::Reject,
            Self::Decode => EmptyRows::Permit,
        }
    }
}

/// Physical cache slot used by one layer in the Qwen hybrid schedule.
///
/// Slot numbers are dense within each cache family. For example, the schedule
/// `[DeltaNet, DeltaNet, Full, DeltaNet, Full]` maps to
/// `[DeltaNet(0), DeltaNet(1), Full(0), DeltaNet(2), Full(1)]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Qwen3_5LayerCache {
    /// Layer slot in the single recurrent DeltaNet cache group.
    DeltaNet(
        /// Dense layer slot in the recurrent group.
        usize,
    ),
    /// Layer slot in the single native-KV full-attention cache group.
    FullAttention(
        /// Dense layer slot in the native-KV group.
        usize,
    ),
}

impl Qwen3_5LayerCache {
    /// Returns the dense layer index inside the selected cache group.
    #[must_use]
    pub const fn group_layer(self) -> usize {
        match self {
            Self::DeltaNet(layer) | Self::FullAttention(layer) => layer,
        }
    }

    /// Returns the Qwen mixer family represented by this slot.
    #[must_use]
    pub const fn layer_type(self) -> Qwen3_5LayerType {
        match self {
            Self::DeltaNet(_) => Qwen3_5LayerType::LinearAttention,
            Self::FullAttention(_) => Qwen3_5LayerType::FullAttention,
        }
    }
}

/// Immutable cache geometry associated with one Qwen model instance.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Qwen3_5CacheLayout {
    /// Model-layer order mapped to dense slots within each physical cache family.
    schedule: Vec<Qwen3_5LayerCache>,
    /// Native-head geometry, present when a target layer uses full attention.
    native_kv: Option<NativeKvCacheSpec>,
    /// DeltaNet matrix and convolution geometry, present for recurrent layers.
    recurrent: Option<RecurrentStateSpec>,
    /// Separate native-KV geometry for the optional MTP prediction layers.
    mtp_native_kv: Option<NativeKvCacheSpec>,
    /// Maximum speculative suffix length retained before commit or rollback.
    staging_capacity: usize,
}

/// Qwen-specific external inference state backed by shared Burn tensor caches.
///
/// Exactly one native-KV group is allocated when the schedule contains full
/// attention, and exactly one recurrent group is allocated when it contains
/// DeltaNet. The wrapper owns only immutable layer mapping plus the generic
/// [`TensorDecodeState`]; model weights and session history never share a record.
#[derive(Debug)]
pub struct Qwen3_5DecodeState<B: Backend> {
    /// Shared physical tensors, logical cursors, and optional speculative transaction.
    tensors: TensorDecodeState<B>,
    /// Immutable family-to-cache mapping used to reject incompatible models.
    layout: Qwen3_5CacheLayout,
}

impl<B: Backend> Qwen3_5DecodeState<B> {
    /// Allocates a state whose topology exactly matches `model`, without speculative staging.
    ///
    /// Full-attention storage deliberately remains full even when `config`
    /// contains a sliding-window hint: Qwen labels these layers global and
    /// evicting their old keys would change the architecture's function.
    fn new(
        model: &Qwen3_5ForCausalLm<B>,
        config: CacheConfig,
        batch_size: usize,
        device: &B::Device,
    ) -> Result<Self> {
        Self::new_with_staging(model, config, batch_size, 0, device)
    }

    /// Allocates target and MTP banks with an equal bounded proposal suffix.
    fn new_with_staging(
        model: &Qwen3_5ForCausalLm<B>,
        config: CacheConfig,
        batch_size: usize,
        staging_capacity: usize,
        device: &B::Device,
    ) -> Result<Self> {
        if config.capacity() > model.config().max_position_embeddings {
            return Err(Error::configuration(
                "qwen3_5.decode_state.capacity",
                format!(
                    "{} exceeds max_position_embeddings {}",
                    config.capacity(),
                    model.config().max_position_embeddings
                ),
            ));
        }
        if config.capacity() > i32::MAX as usize {
            return Err(Error::configuration(
                "qwen3_5.decode_state.capacity",
                "must fit Burn's portable signed position-index range",
            ));
        }
        let layout = cache_layout(model, staging_capacity)?;
        let mut tensors = TensorDecodeState::new(config, batch_size, device)?;
        if let Some(spec) = layout.native_kv {
            let group = tensors.main_mut().add_native_kv(spec)?;
            if group != NATIVE_KV_GROUP {
                return Err(Error::Cache(format!(
                    "Qwen native-KV group index must be {NATIVE_KV_GROUP}, found {group}"
                )));
            }
        }
        if let Some(spec) = layout.recurrent {
            let group = tensors.main_mut().add_recurrent(spec)?;
            if group != RECURRENT_GROUP {
                return Err(Error::Cache(format!(
                    "Qwen recurrent group index must be {RECURRENT_GROUP}, found {group}"
                )));
            }
        }
        if let Some(spec) = layout.mtp_native_kv {
            let group = tensors.enable_mtp()?.add_native_kv(spec)?;
            if group != NATIVE_KV_GROUP {
                return Err(Error::Cache(format!(
                    "Qwen MTP native-KV group index must be {NATIVE_KV_GROUP}, found {group}"
                )));
            }
        }
        Ok(Self { tensors, layout })
    }

    /// Borrows the generic tensor state for cache inspection.
    #[must_use]
    pub const fn tensor_state(&self) -> &TensorDecodeState<B> {
        &self.tensors
    }

    /// Returns one model layer's deterministic physical-cache mapping.
    #[must_use]
    pub fn layer_cache(&self, model_layer: usize) -> Option<Qwen3_5LayerCache> {
        self.layout.schedule.get(model_layer).copied()
    }

    /// Returns the number of full-attention layers sharing the native-KV group.
    #[must_use]
    pub fn full_attention_layer_count(&self) -> usize {
        self.layout.native_kv.map_or(0, NativeKvCacheSpec::layers)
    }

    /// Returns the number of DeltaNet layers sharing the recurrent group.
    #[must_use]
    pub fn delta_net_layer_count(&self) -> usize {
        self.layout.recurrent.map_or(0, RecurrentStateSpec::layers)
    }

    /// Returns the number of independently cached full-attention MTP layers.
    #[must_use]
    pub fn mtp_layer_count(&self) -> usize {
        self.layout
            .mtp_native_kv
            .map_or(0, NativeKvCacheSpec::layers)
    }

    /// Returns the configured per-row speculative suffix capacity.
    #[must_use]
    pub const fn staging_capacity(&self) -> usize {
        self.layout.staging_capacity
    }

    /// Begins an atomic target-plus-MTP proposal suffix.
    pub fn mark(&mut self) -> Result<TensorCacheMark> {
        self.tensors.begin_transaction().map_err(Into::into)
    }

    /// Atomically retains independent prefixes from the target and shifted MTP suffixes.
    pub fn commit_prefixes(
        &mut self,
        mark: TensorCacheMark,
        target_tokens: usize,
        mtp_tokens: usize,
    ) -> Result<()> {
        self.tensors
            .commit_uniform_prefixes(mark, target_tokens, Some(mtp_tokens))
            .map_err(Into::into)
    }

    /// Discards all target and MTP tensors staged since `mark`.
    pub fn rollback(&mut self, mark: TensorCacheMark) -> Result<()> {
        self.tensors.rollback(mark).map_err(Into::into)
    }

    /// Verifies that cached geometry and layer ordering belong to `model`.
    pub(super) fn validate_for_model(&self, model: &Qwen3_5ForCausalLm<B>) -> Result<()> {
        let expected = cache_layout(model, self.layout.staging_capacity)?;
        if self.layout != expected {
            return Err(Error::Cache(
                "Qwen decode state was created for an incompatible model layout".into(),
            ));
        }
        let bank = self.tensors.main();
        let expected_native_groups = usize::from(expected.native_kv.is_some());
        let expected_recurrent_groups = usize::from(expected.recurrent.is_some());
        if bank.native_kv_count() != expected_native_groups
            || bank.recurrent_count() != expected_recurrent_groups
        {
            return Err(Error::Cache(
                "Qwen decode state cache-group count no longer matches its layout".into(),
            ));
        }
        if let Some(spec) = expected.native_kv
            && bank.native_kv(NATIVE_KV_GROUP)?.spec() != spec
        {
            return Err(Error::Cache(
                "Qwen native-KV cache geometry differs from the model".into(),
            ));
        }
        if let Some(spec) = expected.recurrent
            && bank.recurrent(RECURRENT_GROUP)?.spec() != spec
        {
            return Err(Error::Cache(
                "Qwen recurrent cache geometry differs from the model".into(),
            ));
        }
        match (expected.mtp_native_kv, self.tensors.mtp()) {
            (None, None) => {}
            (Some(spec), Some(bank))
                if bank.native_kv_count() == 1
                    && bank.recurrent_count() == 0
                    && bank.native_kv(NATIVE_KV_GROUP)?.spec() == spec => {}
            _ => {
                return Err(Error::Cache(
                    "Qwen MTP cache geometry differs from the model".into(),
                ));
            }
        }
        Ok(())
    }
}

impl<B: Backend> ArchitectureDecodeState<B> for Qwen3_5DecodeState<B> {
    /// Returns the number of logical sequence rows represented by this state.
    fn batch_size(&self) -> usize {
        self.tensors.batch_size()
    }

    /// Returns the maximum logical token count per row, excluding padding.
    fn capacity(&self) -> usize {
        self.tensors.capacity()
    }

    /// Borrows committed semantic token counts in the current batch order.
    fn lengths(&self) -> &[usize] {
        self.tensors.lengths()
    }

    /// Resets logical lengths and all cache groups through the shared state contract.
    fn reset(&mut self) -> Result<()> {
        self.tensors.reset().map_err(Into::into)
    }

    /// Reorders or duplicates sequence rows across metadata and every cache group atomically.
    fn reorder(&mut self, indices: &[usize]) -> Result<()> {
        self.tensors.reorder(indices).map_err(Into::into)
    }

    /// Truncates target history and bounds shifted MTP history by `target_length - 1`.
    ///
    /// Shared cache validation rejects unavailable recurrent snapshots or an
    /// invalid row count before any prefix is committed.
    fn truncate(&mut self, lengths: &[usize]) -> Result<()> {
        let Some(current_mtp) = self.tensors.mtp_lengths() else {
            return self.tensors.truncate(lengths).map_err(Into::into);
        };
        if lengths.len() != current_mtp.len() {
            return Err(
                TensorCacheError::Metadata(DecodeStateError::SequenceCountMismatch {
                    expected: current_mtp.len(),
                    actual: lengths.len(),
                })
                .into(),
            );
        }
        let mtp_lengths = lengths
            .iter()
            .zip(current_mtp)
            .map(|(&target, &mtp)| shifted_mtp_length(target, mtp))
            .collect::<Vec<_>>();
        self.tensors
            .truncate_prefixes(lengths, &mtp_lengths)
            .map_err(Into::into)
    }

    /// Returns the device holding this state, independent of model parameter ownership.
    fn device(&self) -> &B::Device {
        self.tensors.device()
    }
}

impl<B: Backend> CausalLanguageModel<B> for Qwen3_5ForCausalLm<B> {
    /// Hybrid topology used to allocate and validate inference state.
    type Config = Qwen3_5TextConfig;
    /// External target and optional MTP cache state for one inference session.
    type State = Qwen3_5DecodeState<B>;

    /// Delegates to the stateless family graph without allocating or mutating decode history.
    fn forward(
        &self,
        input: CausalLmInput<B>,
        options: &ForwardOptions,
    ) -> Result<CausalLmOutput<B>> {
        Qwen3_5ForCausalLm::forward(self, input, options)
    }

    /// Allocates external Qwen hybrid cache groups after validating model/device geometry.
    fn new_decode_state(
        &self,
        config: CacheConfig,
        batch_size: usize,
        device: &B::Device,
    ) -> Result<Self::State> {
        Qwen3_5DecodeState::new(self, config, batch_size, device)
    }

    /// Evaluates prompt rows with at least one valid token and commits all cache writes together.
    fn prefill(
        &self,
        input: CausalLmInput<B>,
        state: &mut Self::State,
        options: &ForwardOptions,
    ) -> Result<CausalLmOutput<B>> {
        self.forward_with_decode_state(input, state, options, StatefulPhase::Prefill)
    }

    /// Appends valid tokens while fully masked rows retain their state and return zero outputs.
    fn decode(
        &self,
        input: DecodeInput<B>,
        state: &mut Self::State,
        options: &ForwardOptions,
    ) -> Result<CausalLmOutput<B>> {
        self.forward_with_decode_state(
            CausalLmInput {
                token_ids: input.token_ids,
                token_mask: input.token_mask,
                position_ids: input.position_ids,
            },
            state,
            options,
            StatefulPhase::Decode,
        )
    }
}

impl<B: Backend> Qwen3_5ForCausalLm<B> {
    /// Allocates the batch-one transactional state required by MTP speculation.
    ///
    /// A cycle verifies the target lead token plus up to `K` drafts in one
    /// pass, staging `K + 1` target tokens, and re-runs the MTP layer over the
    /// accepted pairs, staging at most `K + 1` MTP tokens. Both banks use
    /// independent logical cursors and an atomic prefix commit.
    pub fn new_speculative_decode_state(
        &self,
        config: CacheConfig,
        batch_size: usize,
        max_draft_tokens: usize,
        device: &B::Device,
    ) -> Result<Qwen3_5DecodeState<B>> {
        if batch_size != 1 {
            return Err(Error::Generation(format!(
                "Qwen MTP speculative decoding supports batch size one, found {batch_size}"
            )));
        }
        if max_draft_tokens == 0 {
            return Err(Error::Generation(
                "Qwen MTP max_draft_tokens must be greater than zero".into(),
            ));
        }
        if self.mtp.is_none() {
            return Err(Error::Generation(
                "Qwen model has no configured MTP prediction layer".into(),
            ));
        }
        if self.mtp.as_ref().is_some_and(|mtp| mtp.layers.len() != 1) {
            return Err(Error::BackendCapability(
                "cached Qwen MTP currently supports the released single-layer auxiliary graph"
                    .into(),
            ));
        }
        let staging = max_draft_tokens
            .checked_add(1)
            .ok_or_else(|| Error::Generation("Qwen MTP staging capacity overflows usize".into()))?;
        if staging > config.capacity() {
            return Err(Error::Generation(format!(
                "Qwen MTP needs staging capacity {staging}, exceeding cache capacity {}",
                config.capacity()
            )));
        }
        Qwen3_5DecodeState::new_with_staging(self, config, batch_size, staging, device)
    }

    /// Evaluates shifted MTP inputs against its independent incremental KV cache.
    ///
    /// Pair `j` combines `next_token_ids[0, j] = x_(t+1)` with the hidden state
    /// `h_t` that preceded it (see [`Self::forward_mtp`]); all pairs of one call
    /// append consecutively to the MTP cache, inside or outside a transaction.
    /// Prefill passes the shifted prompt pairs; speculative verification passes
    /// the accepted target pairs; drafting passes one pair per step whose
    /// hidden state is the previous MTP output, as the released runtime does.
    /// `step` selects the MTP layer `step mod mtp_num_hidden_layers`.
    pub fn forward_mtp_cached(
        &self,
        next_token_ids: Tensor<B, 2, Int>,
        target_or_mtp_hidden: Tensor<B, 3>,
        position_ids: Option<Tensor<B, 2, Int>>,
        step: usize,
        state: &mut Qwen3_5DecodeState<B>,
    ) -> Result<Qwen3_5MtpPrediction<B>> {
        self.forward_mtp_cached_inner(
            next_token_ids,
            target_or_mtp_hidden,
            position_ids,
            step,
            state,
            false,
        )
    }

    /// Runs [`Self::forward_mtp_cached`] on host token IDs with implicit positions.
    ///
    /// The speculative decoder already holds its tokens on the host, so they
    /// are range-checked there and the call needs no device-to-host readback
    /// at all (the public entry point must read device token IDs back once).
    pub(super) fn forward_mtp_host_tokens(
        &self,
        tokens: &[u32],
        target_or_mtp_hidden: Tensor<B, 3>,
        step: usize,
        state: &mut Qwen3_5DecodeState<B>,
    ) -> Result<Qwen3_5MtpPrediction<B>> {
        let vocabulary = self.config().vocab_size;
        if let Some(&token) = tokens.iter().find(|&&token| token as usize >= vocabulary) {
            return Err(Error::shape(
                INPUT_NAMES.token_ids,
                format!("token values in 0..{vocabulary}"),
                token.to_string(),
            ));
        }
        let values = tokens.iter().copied().map(i64::from).collect::<Vec<_>>();
        let ids = Tensor::from_data(
            burn::tensor::TensorData::new(values, [1, tokens.len()]),
            state.tensors.device(),
        );
        self.forward_mtp_cached_inner(ids, target_or_mtp_hidden, None, step, state, true)
    }

    /// Shared body of the cached MTP entry points.
    ///
    /// `host_validated` skips the device range check of token IDs that the
    /// caller already checked on the host.
    fn forward_mtp_cached_inner(
        &self,
        next_token_ids: Tensor<B, 2, Int>,
        target_or_mtp_hidden: Tensor<B, 3>,
        position_ids: Option<Tensor<B, 2, Int>>,
        step: usize,
        state: &mut Qwen3_5DecodeState<B>,
        host_validated: bool,
    ) -> Result<Qwen3_5MtpPrediction<B>> {
        state.validate_for_model(self)?;
        let mtp = self.mtp.as_ref().ok_or_else(|| {
            Error::Generation("Qwen model has no configured MTP prediction layer".into())
        })?;
        let [batch, sequence] = next_token_ids.dims();
        if batch != 1 || batch != state.tensors.batch_size() || sequence == 0 {
            return Err(Error::shape(
                "qwen.mtp.next_token_ids",
                "[1, nonzero_sequence] matching a batch-one decode state",
                format!("[{batch}, {sequence}]"),
            ));
        }
        let device = state.tensors.device().clone();
        ensure_same_device(
            "qwen.mtp.next_token_ids",
            &next_token_ids.device(),
            &device,
            DECODE_STATE,
        )?;
        ensure_same_device(
            "qwen.mtp.hidden_state",
            &target_or_mtp_hidden.device(),
            &device,
            DECODE_STATE,
        )?;
        let baseline = state
            .tensors
            .mtp_working_lengths()
            .and_then(|lengths| lengths.first().copied())
            .ok_or_else(|| Error::Cache("Qwen MTP working metadata is not enabled".into()))?;
        let end = baseline
            .checked_add(sequence)
            .ok_or_else(|| Error::Generation("Qwen MTP logical position overflows usize".into()))?;
        if end > state.tensors.capacity() {
            return Err(
                TensorCacheError::Metadata(DecodeStateError::CapacityExceeded {
                    sequence: 0,
                    requested: end,
                    capacity: state.tensors.capacity(),
                })
                .into(),
            );
        }
        // Tokens and explicit positions share one readback; implicit positions
        // `baseline..end` need none because their extent is known on the host,
        // and host-validated tokens none at all.
        let mut checks = DeferredChecks::new();
        if !host_validated {
            checks.values_below(
                &next_token_ids,
                self.config().vocab_size,
                INPUT_NAMES.token_ids,
                "token",
            );
        }
        let (positions, explicit_range) = match position_ids {
            Some(positions) => {
                if positions.dims() != [1, sequence] {
                    return Err(Error::shape(
                        "qwen.mtp.position_ids",
                        format!("[1, {sequence}]"),
                        format!("{:?}", positions.dims()),
                    ));
                }
                ensure_same_device(
                    "qwen.mtp.position_ids",
                    &positions.device(),
                    &device,
                    DECODE_STATE,
                )?;
                let range = checks.values_below(
                    &positions,
                    self.config().max_position_embeddings,
                    INPUT_NAMES.position_ids,
                    "position",
                );
                (positions, Some(range))
            }
            None => (
                implicit_position_ids(baseline, sequence, &device, INPUT_NAMES.position_ids)?,
                None,
            ),
        };
        let readout = checks.run("qwen.mtp")?;
        let evaluated_context = match explicit_range {
            Some(range) => readout.context(range, INPUT_NAMES.position_ids)?,
            None => end,
        };
        ensure_cacheable_rotary_context(self.model.rope, evaluated_context, "Qwen")?;
        // One call writes one MTP layer, so a step covers the whole MTP group
        // only for the released single-layer graph.
        if mtp.layers.len() != 1 {
            return Err(Error::BackendCapability(
                "cached Qwen MTP currently supports the released single-layer auxiliary graph"
                    .into(),
            ));
        }
        let layer_index = step % mtp.layers.len();
        let layout = CompactBatch::new(&[(0..sequence).collect()], sequence, &[baseline], &device)?;
        let embedding = self.model.embed_tokens.forward(next_token_ids);
        let hidden = mtp.fuse(embedding, target_or_mtp_hidden)?;
        let mut writes = state.tensors.mtp_step(&[sequence])?;
        let mut attention = CachedAttention::new(
            self.model.rope,
            evaluated_context,
            positions,
            hidden.dtype(),
            &state.tensors,
            &writes,
            NATIVE_KV_GROUP,
        )?;
        let hidden = mtp.layers[layer_index].cached_step(
            hidden,
            Qwen3_5LayerCache::FullAttention(layer_index),
            &layout,
            &mut state.tensors,
            &mut writes,
            Some(&mut attention),
            DeltaStep::Terminal(DeltaNetMode::Recurrent),
        )?;
        let hidden = mtp.norm.forward(hidden);
        let logits = self.project_hidden(hidden.clone())?;
        state.tensors.publish_step(writes)?;
        Ok(Qwen3_5MtpPrediction {
            logits,
            hidden_state: hidden,
        })
    }

    /// Aligns a fresh MTP cache with an already-prefilled target prompt.
    ///
    /// For prompt tokens `x_0..x_(L-1)` and target states `h_0..h_(L-1)`, the
    /// auxiliary cache receives exactly the shifted `L-1` pairs
    /// `(x_(i+1), h_i)`. Therefore its committed cursor is one behind the target
    /// cursor at every speculative-cycle boundary.
    pub fn prefill_mtp_cache(
        &self,
        prompt_token_ids: Tensor<B, 2, Int>,
        target_final_hidden_states: Tensor<B, 3>,
        state: &mut Qwen3_5DecodeState<B>,
    ) -> Result<()> {
        state.validate_for_model(self)?;
        if state.tensors.has_active_transaction() {
            return Err(TensorCacheError::TransactionActive {
                operation: "prefill Qwen MTP cache",
            }
            .into());
        }
        let [batch, sequence] = prompt_token_ids.dims();
        if batch != 1
            || sequence == 0
            || target_final_hidden_states.dims() != [1, sequence, self.config().hidden_size]
        {
            return Err(Error::shape(
                "qwen.mtp.prefill",
                format!(
                    "token IDs [1, L] and target hidden states [1, L, {}] for L > 0",
                    self.config().hidden_size
                ),
                format!(
                    "tokens={:?}, hidden={:?}",
                    prompt_token_ids.dims(),
                    target_final_hidden_states.dims()
                ),
            ));
        }
        if state.tensors.lengths() != [sequence] {
            return Err(Error::Cache(format!(
                "Qwen target cache length must equal MTP prefill prompt length {sequence}, found {:?}",
                state.tensors.lengths()
            )));
        }
        if state.tensors.mtp_lengths() != Some([0].as_slice()) {
            return Err(Error::Cache(
                "Qwen MTP prefill requires a fresh auxiliary cache".into(),
            ));
        }
        if sequence == 1 {
            return Ok(());
        }
        let shifted_tokens = prompt_token_ids.narrow(1, 1, sequence - 1);
        let shifted_hidden = target_final_hidden_states.narrow(1, 0, sequence - 1);
        self.forward_mtp_cached(shifted_tokens, shifted_hidden, None, 0, state)?;
        Ok(())
    }

    /// Evaluates a rectangular prompt/decode segment and appends only valid tokens.
    ///
    /// If row `b` has working length `W_b`, its `i`th valid implicit position
    /// is `W_b + i` and its full-attention query can see keys `j <= W_b + i`.
    /// Padded positions and every position of a row without valid tokens
    /// return zeros and leave that row's cache untouched.
    fn forward_with_decode_state(
        &self,
        input: CausalLmInput<B>,
        state: &mut Qwen3_5DecodeState<B>,
        options: &ForwardOptions,
        phase: StatefulPhase,
    ) -> Result<CausalLmOutput<B>> {
        self.cached_forward(input, state, options, phase, false)
    }

    /// Runs one batched cached call; `keep_final_hidden` also returns the final states.
    ///
    /// `final_hidden_state` is filled when either `options.hidden_states` or
    /// `keep_final_hidden` is set; speculative verification uses the latter to
    /// read every verified position's state without the per-layer trace.
    pub(super) fn cached_forward(
        &self,
        input: CausalLmInput<B>,
        state: &mut Qwen3_5DecodeState<B>,
        options: &ForwardOptions,
        phase: StatefulPhase,
        keep_final_hidden: bool,
    ) -> Result<CausalLmOutput<B>> {
        if options.attentions {
            return Err(Error::BackendCapability(
                "Qwen attention probability capture requires the reference attention path".into(),
            ));
        }
        state.validate_for_model(self)?;
        let rows = prepare_stateful_input(self, &input, state, phase)?;
        validate_logits_selection(&options.logits, rows.sequence)?;
        let keep_final = options.hidden_states || keep_final_hidden;
        if rows.token_counts.iter().all(|&count| count == 0) {
            // Every row is finished: an exact no-op with zero outputs.
            return self.inactive_output(&rows, options, keep_final, state.tensors.device());
        }
        let device = state.tensors.device().clone();
        let layout = CompactBatch::new(
            &rows.valid_indices,
            rows.sequence,
            state.tensors.working_lengths(),
            &device,
        )?;
        // Padding slots are evaluated (and discarded), so they carry token 0
        // and position 0 instead of the caller's arbitrary sentinels.
        let tokens = layout.compact(&input.token_ids, 0);
        let positions = match &input.position_ids {
            Some(positions) => layout.compact(positions, 0),
            None => layout.cache_positions().clone(),
        };
        // Every active context is within the cacheable range, where rotary
        // frequencies do not depend on the context, so the largest serves all rows.
        let context = rows
            .active_contexts()
            .map(|(_, context)| context)
            .max()
            .unwrap_or(1);
        let pass = self.model.run_cached(
            tokens,
            positions,
            context,
            &layout,
            &mut state.tensors,
            &state.layout.schedule,
            phase,
            options.hidden_states,
        )?;
        let selected = layout.select(pass.hidden.clone(), &options.logits, rows.last_columns())?;
        let logits = self.project_hidden(selected)?;
        let final_hidden_state = keep_final.then(|| layout.restore(pass.hidden));
        let hidden_states = pass
            .hidden_states
            .into_iter()
            .map(|hidden| layout.restore(hidden))
            .collect();
        state.tensors.publish_step(pass.writes)?;
        Ok(CausalLmOutput {
            logits,
            final_hidden_state,
            hidden_states,
            attentions: Vec::new(),
            routers: Vec::new(),
            load_balancing_loss: None,
        })
    }

    /// Returns the zero outputs of a call in which no row appends a token.
    ///
    /// No layer is evaluated and no cache changes. Qwen's vocabulary
    /// projections have no bias, so zero hidden states give exact zero logits.
    fn inactive_output(
        &self,
        rows: &StatefulRows,
        options: &ForwardOptions,
        keep_final: bool,
        device: &B::Device,
    ) -> Result<CausalLmOutput<B>> {
        let zeros = Tensor::<B, 3>::zeros(
            [rows.batch, rows.sequence, self.config().hidden_size],
            (device, self.model.embed_tokens.weight.val().dtype()),
        );
        let selected =
            select_hidden_with_last_columns(zeros.clone(), rows.last_columns(), &options.logits)?;
        let hidden_states = if options.hidden_states {
            vec![zeros.clone(); self.model.layers.len() + 2]
        } else {
            Vec::new()
        };
        Ok(CausalLmOutput {
            logits: self.project_hidden(selected)?,
            final_hidden_state: keep_final.then_some(zeros),
            hidden_states,
            attentions: Vec::new(),
            routers: Vec::new(),
            load_balancing_loss: None,
        })
    }
}

/// Compact outputs and unpublished cache writes of one batched pass.
struct CachedPass<B: Backend> {
    /// Final-normalized compact hidden states `[B, N, hidden]`.
    hidden: Tensor<B, 3>,
    /// Embedding, every block output, and the normalized output, when requested.
    hidden_states: Vec<Tensor<B, 3>>,
    /// Every layer's cache update, published by the caller after projection.
    writes: TensorStepWrites<B>,
}

impl<B: Backend> Qwen3_5TextModel<B> {
    /// Runs every layer once over the compact batch and records its cache writes.
    ///
    /// `tokens` and `positions` are compact `[B, N]` (see [`CompactBatch`]);
    /// `context` is the largest evaluated rotary context of the call.
    #[allow(clippy::too_many_arguments)]
    fn run_cached(
        &self,
        tokens: Tensor<B, 2, Int>,
        positions: Tensor<B, 2, Int>,
        context: usize,
        layout: &CompactBatch<B>,
        tensors: &mut TensorDecodeState<B>,
        schedule: &[Qwen3_5LayerCache],
        phase: StatefulPhase,
        retain_hidden_states: bool,
    ) -> Result<CachedPass<B>> {
        if schedule.len() != self.layers.len() {
            return Err(Error::Cache(format!(
                "Qwen cache schedule has {} entries for {} model layers",
                schedule.len(),
                self.layers.len()
            )));
        }
        let mut writes = tensors.step(layout.counts())?;
        // A recurrence cannot be rewound, so a staged multi-token step keeps
        // the state after every token for a later prefix commit.
        let delta =
            if tensors.has_active_transaction() && layout.counts().iter().any(|&count| count > 1) {
                DeltaStep::Snapshots
            } else {
                DeltaStep::Terminal(phase.delta_mode())
            };
        let mut hidden = self.embed_tokens.forward(tokens);
        let mut attention = schedule
            .iter()
            .any(|slot| matches!(slot, Qwen3_5LayerCache::FullAttention(_)))
            .then(|| {
                CachedAttention::new(
                    self.rope,
                    context,
                    positions,
                    hidden.dtype(),
                    tensors,
                    &writes,
                    NATIVE_KV_GROUP,
                )
            })
            .transpose()?;
        let mut hidden_states = Vec::with_capacity(if retain_hidden_states {
            self.layers.len() + 2
        } else {
            0
        });
        if retain_hidden_states {
            hidden_states.push(hidden.clone());
        }
        for (layer, &slot) in self.layers.iter().zip(schedule) {
            hidden = layer.cached_step(
                hidden,
                slot,
                layout,
                tensors,
                &mut writes,
                attention.as_mut(),
                delta,
            )?;
            if retain_hidden_states {
                hidden_states.push(hidden.clone());
            }
        }
        hidden = self.norm.forward(hidden);
        if retain_hidden_states {
            hidden_states.push(hidden.clone());
        }
        Ok(CachedPass {
            hidden,
            hidden_states,
            writes,
        })
    }
}

/// How a DeltaNet layer evolves and records its recurrent state in one step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DeltaStep {
    /// Run the given kernel and record every row's terminal state.
    Terminal(DeltaNetMode),
    /// Run the recurrence and record every row's state after each token.
    Snapshots,
}

impl<B: Backend> Qwen3_5DecoderLayer<B> {
    /// Evaluates one residual block over the compact batch against its cache slot.
    ///
    /// Full attention needs the step's [`CachedAttention`]; DeltaNet reads every
    /// row's working state and records the update as `delta` prescribes.
    #[allow(clippy::too_many_arguments)]
    fn cached_step(
        &self,
        hidden: Tensor<B, 3>,
        slot: Qwen3_5LayerCache,
        layout: &CompactBatch<B>,
        tensors: &mut TensorDecodeState<B>,
        writes: &mut TensorStepWrites<B>,
        attention: Option<&mut CachedAttention<B>>,
        delta: DeltaStep,
    ) -> Result<Tensor<B, 3>> {
        if slot.layer_type() != self.layer_type {
            return Err(Error::Cache(format!(
                "Qwen layer type {:?} was mapped to incompatible cache slot {slot:?}",
                self.layer_type
            )));
        }
        let residual = hidden.clone();
        let normalized = self.input_layernorm.forward(hidden);
        let mixed = match slot {
            Qwen3_5LayerCache::FullAttention(cache_layer) => {
                let module = self.self_attn.as_ref().ok_or_else(|| {
                    Error::Checkpoint("Qwen full-attention layer has no attention module".into())
                })?;
                let attention = attention.ok_or_else(|| {
                    Error::Cache("Qwen full-attention layer has no cached attention step".into())
                })?;
                attention.attend(
                    module,
                    normalized,
                    cache_layer,
                    None,
                    layout,
                    tensors,
                    writes,
                )?
            }
            Qwen3_5LayerCache::DeltaNet(cache_layer) => {
                self.delta_step(normalized, cache_layer, layout, tensors, writes, delta)?
            }
        };
        finish_residual(self, residual, mixed)
    }

    /// Runs the DeltaNet mixer over the batch and records its new states.
    ///
    /// Padding slots are masked, so they neither update the matrix nor enter
    /// the convolution history, and inactive rows keep their state.
    fn delta_step(
        &self,
        normalized: Tensor<B, 3>,
        cache_layer: usize,
        layout: &CompactBatch<B>,
        tensors: &TensorDecodeState<B>,
        writes: &mut TensorStepWrites<B>,
        delta: DeltaStep,
    ) -> Result<Tensor<B, 3>> {
        let module = self
            .linear_attn
            .as_ref()
            .ok_or_else(|| Error::Checkpoint("Qwen linear layer has no DeltaNet module".into()))?;
        let read = tensors
            .main()
            .recurrent(RECURRENT_GROUP)?
            .read_working_batch(cache_layer)?;
        let mut state = GatedDeltaNetState {
            convolution: read.convolution().clone(),
            recurrent: read.matrix().clone(),
        };
        let mask = layout.token_mask();
        match delta {
            DeltaStep::Terminal(mode) => {
                let mixed =
                    module.forward_with_state(normalized, mask.as_ref(), &mut state, mode)?;
                writes.push_recurrent_batch(
                    RECURRENT_GROUP,
                    cache_layer,
                    state.recurrent,
                    state.convolution,
                );
                Ok(mixed)
            }
            DeltaStep::Snapshots => {
                let (mixed, snapshots) =
                    module.forward_with_snapshots(normalized, mask.as_ref(), &mut state)?;
                // Row b's tokens are slots 0..n_b, so its snapshots are too.
                for (row, &count) in layout.counts().iter().enumerate() {
                    if count == 0 {
                        continue;
                    }
                    writes.push_recurrent_row_snapshots(
                        RECURRENT_GROUP,
                        cache_layer,
                        row,
                        snapshots
                            .recurrent
                            .clone()
                            .narrow(0, row, 1)
                            .narrow(1, 0, count),
                        snapshots
                            .convolution
                            .clone()
                            .narrow(0, row, 1)
                            .narrow(1, 0, count),
                    );
                }
                Ok(mixed)
            }
        }
    }
}

/// Applies the post-mixer residual and pre-normalized SwiGLU sublayer.
fn finish_residual<B: Backend>(
    layer: &Qwen3_5DecoderLayer<B>,
    residual: Tensor<B, 3>,
    mixed: Tensor<B, 3>,
) -> Result<Tensor<B, 3>> {
    let hidden = residual + mixed;
    let residual = hidden.clone();
    Ok(residual
        + layer
            .mlp
            .forward(layer.post_attention_layernorm.forward(hidden))?)
}

/// Builds the model-layer mapping and homogeneous cache specifications.
///
/// Full-attention groups stage speculative proposals in place: full storage
/// never evicts, so a proposal can sit in its final slot and every layer reads
/// history plus proposals through one view. Only the DeltaNet group allocates
/// separate per-token snapshot buffers.
fn cache_layout<B: Backend>(
    model: &Qwen3_5ForCausalLm<B>,
    staging_capacity: usize,
) -> Result<Qwen3_5CacheLayout> {
    if !model.model.supports_family_cache {
        return Err(Error::BackendCapability(
            "editable Qwen recipe backbones use stateless execution; cached family decoding requires Qwen3_5ForCausalLm::init with an unchanged family configuration".into(),
        ));
    }
    let layer_types: Vec<Qwen3_5LayerType> = model
        .model
        .layers
        .iter()
        .map(|layer| layer.layer_type)
        .collect();
    let schedule = cache_schedule(&layer_types);
    let full_layers = schedule
        .iter()
        .filter(|slot| matches!(slot, Qwen3_5LayerCache::FullAttention(_)))
        .count();
    let delta_layers = schedule.len() - full_layers;
    let activation_dtype = model.model.embed_tokens.weight.val().dtype();
    let attention_spec = |layers: usize| {
        NativeKvCacheSpec::new(
            layers,
            model.config().num_key_value_heads,
            model.config().head_dim,
            KvCacheStorage::Full,
            activation_dtype,
        )
        .map(|spec| {
            spec.with_staging_capacity(staging_capacity)
                .with_in_place_staging()
        })
    };
    let native_kv = (full_layers > 0)
        .then(|| attention_spec(full_layers))
        .transpose()?;
    let recurrent = if delta_layers == 0 {
        None
    } else {
        let delta = model
            .model
            .layers
            .iter()
            .find_map(|layer| layer.linear_attn.as_ref())
            .ok_or_else(|| {
                Error::Checkpoint("Qwen schedule contains DeltaNet without a module".into())
            })?;
        let config = delta.config();
        let convolution_width = config.convolution_kernel.checked_sub(1).ok_or_else(|| {
            Error::configuration(
                "qwen3_5.linear_conv_kernel_dim",
                "cannot derive causal-convolution history width",
            )
        })?;
        if convolution_width == 0 {
            return Err(Error::configuration(
                "qwen3_5.linear_conv_kernel_dim",
                "stateful decoding currently requires a kernel of at least two",
            ));
        }
        Some(
            RecurrentStateSpec::new_with_dtypes(
                delta_layers,
                config.value_heads,
                config.key_head_dim,
                config.value_head_dim,
                config.convolution_channels()?,
                convolution_width,
                DType::F32,
                activation_dtype,
            )?
            .with_staging_capacity(staging_capacity),
        )
    };
    let mtp_native_kv = model
        .mtp
        .as_ref()
        .map(|mtp| attention_spec(mtp.layers.len()))
        .transpose()?;
    Ok(Qwen3_5CacheLayout {
        schedule,
        native_kv,
        recurrent,
        mtp_native_kv,
        staging_capacity,
    })
}

/// Maps each layer to a dense slot of its cache family, in schedule order.
///
/// The `k`th DeltaNet layer gets recurrent slot `k` and the `k`th
/// full-attention layer native-KV slot `k`, so each family's slots are exactly
/// `0..count` (the integer core, [`plan::cache_slots`], is verified in
/// `verus/qwen.rs`).
fn cache_schedule(layer_types: &[Qwen3_5LayerType]) -> Vec<Qwen3_5LayerCache> {
    let full_attention = layer_types
        .iter()
        .map(|layer_type| *layer_type == Qwen3_5LayerType::FullAttention)
        .collect::<Vec<_>>();
    let slots = plan::cache_slots(&full_attention);
    full_attention
        .iter()
        .zip(slots.slots)
        .map(|(&full, slot)| {
            if full {
                Qwen3_5LayerCache::FullAttention(slot)
            } else {
                Qwen3_5LayerCache::DeltaNet(slot)
            }
        })
        .collect()
}

/// Validates devices, shapes, positions, masks, and all capacity transitions.
///
/// The shared preparation reads the mask, token IDs, and explicit positions in
/// one transfer against the working cursors (speculative transactions append
/// beyond committed lengths); this wrapper adds the model-device and
/// context-dependent RoPE guards.
fn prepare_stateful_input<B: Backend>(
    model: &Qwen3_5ForCausalLm<B>,
    input: &CausalLmInput<B>,
    state: &Qwen3_5DecodeState<B>,
    phase: StatefulPhase,
) -> Result<StatefulRows> {
    ensure_same_device(
        "qwen.model",
        &model.model.embed_tokens.weight.lazy_device(),
        state.tensors.device(),
        DECODE_STATE,
    )?;
    let rows = prepare_stateful_rows(
        input,
        state.tensors.device(),
        state.tensors.working_lengths(),
        state.tensors.capacity(),
        &StatefulContract {
            names: INPUT_NAMES,
            owner: DECODE_STATE,
            vocabulary: model.config().vocab_size,
            position_limit: Some(model.config().max_position_embeddings),
            empty_rows: phase.empty_rows(),
        },
    )?;
    for (_, context) in rows.active_contexts() {
        ensure_cacheable_rotary_context(model.model.rope, context, "Qwen")?;
    }
    Ok(rows)
}

#[cfg(test)]
#[path = "decode_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "per_row_tests.rs"]
mod per_row_tests;
