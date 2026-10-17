//! Cache-aware Qwen3.5/Qwen3.8 causal inference.
//!
//! Qwen's hybrid schedule has two physically different histories. A full-attention
//! layer retains every native key/value head, whereas a DeltaNet layer retains only
//! its terminal association matrix and the last `kernel - 1` convolution inputs.
//! [`Qwen3_5DecodeState`] maps model-layer indices monotonically into one cache group
//! of each kind, keeping runtime state outside Burn's parameter [`burn::module::Module`] graph.
//!
//! Masked rectangular batches are compacted one row at a time before model math.
//! This is more than an optimization detail: a padding token must not shift the
//! causal-convolution history or consume a logical RoPE position. Results at masked
//! output positions are defined as zero; valid positions are scattered back to the
//! caller's rectangular shape.

use std::collections::BTreeSet;

use burn::tensor::{Bool, DType, ElementConversion, Int, Tensor, TensorData, backend::Backend};

use crate::{
    CausalLanguageModel, CausalLmInput, CausalLmOutput, DecodeInput, Error, ForwardOptions, Result,
    architecture::DecodeState as ArchitectureDecodeState,
    cache::{
        CacheConfig, DecodeStateError, KvCacheStorage, NativeKvCacheSpec, RecurrentStateSpec,
        TensorCacheBank, TensorCacheError, TensorCacheMark, TensorDecodeState,
    },
    delta_net::{DeltaNetMode, GatedDeltaNetState},
    position::{
        RopeScaling, RotaryAngles, apply_rotary, derive_position_ids, rotary_angles_for_dtype,
    },
};

use super::{
    Qwen3_5DecoderLayer, Qwen3_5ForCausalLm, Qwen3_5MtpPrediction, Qwen3_5TextModel, select_hidden,
    validate_logits_selection, validate_token_ids,
};
use crate::models::qwen3_5::{Qwen3_5LayerType, Qwen3_5TextConfig};

/// Single native-KV group within either the target or MTP cache bank.
const NATIVE_KV_GROUP: usize = 0;
/// Single recurrent group containing all target DeltaNet layer states.
const RECURRENT_GROUP: usize = 0;

/// Public stateful operation whose empty-row semantics differ intentionally.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StatefulPhase {
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

    /// Returns whether a semantic-mask row may contain no valid token.
    const fn permits_inactive_rows(self) -> bool {
        matches!(self, Self::Decode)
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
    /// Allocates a state whose topology exactly matches `model`.
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
        if model.mtp.as_ref().is_some_and(|mtp| mtp.layers.len() != 1) {
            return Err(Error::BackendCapability(
                "cached Qwen MTP currently supports the released single-layer auxiliary graph"
                    .into(),
            ));
        }
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
            .map(|(&target, &mtp)| mtp.min(target.saturating_sub(1)))
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
    /// The target stages `K + 1` tokens (the guaranteed target lead plus as many
    /// accepted drafts), while the shifted MTP bank stages at most `K + 1`
    /// inputs. Both use independent logical cursors and an atomic prefix commit.
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
    /// Before the first call, callers prefill pairs `(x_(i+1), h_i)` for the
    /// prompt. During speculation each call contains exactly one pair and is
    /// staged transactionally. The returned hidden state is recurrently fed to
    /// the next draft step, as prescribed by the released Qwen runtime.
    pub fn forward_mtp_cached(
        &self,
        next_token_ids: Tensor<B, 2, Int>,
        target_or_mtp_hidden: Tensor<B, 3>,
        position_ids: Option<Tensor<B, 2, Int>>,
        step: usize,
        state: &mut Qwen3_5DecodeState<B>,
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
        if state.tensors.has_active_transaction() && sequence != 1 {
            return Err(Error::Generation(
                "transactional Qwen MTP decoding accepts exactly one token per call".into(),
            ));
        }
        validate_token_ids(next_token_ids.clone(), self.config().vocab_size)?;
        ensure_device(
            "qwen.mtp.next_token_ids",
            &next_token_ids.device(),
            state.tensors.device(),
        )?;
        ensure_device(
            "qwen.mtp.hidden_state",
            &target_or_mtp_hidden.device(),
            state.tensors.device(),
        )?;
        let committed = state
            .tensors
            .mtp_lengths()
            .and_then(|lengths| lengths.first().copied())
            .ok_or_else(|| Error::Cache("Qwen MTP metadata is not enabled".into()))?;
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
        let positions = match position_ids {
            Some(positions) => {
                if positions.dims() != [1, sequence] {
                    return Err(Error::shape(
                        "qwen.mtp.position_ids",
                        format!("[1, {sequence}]"),
                        format!("{:?}", positions.dims()),
                    ));
                }
                ensure_device(
                    "qwen.mtp.position_ids",
                    &positions.device(),
                    state.tensors.device(),
                )?;
                validate_position_range(positions.clone(), self.config().max_position_embeddings)?;
                positions
            }
            None => implicit_positions::<B>(baseline, sequence, state.tensors.device())?,
        };
        let evaluated_context = evaluated_context(positions.clone())?;
        validate_cached_rope_context(self.model.rope.scaling(), evaluated_context)?;
        let embedding = self.model.embed_tokens.forward(next_token_ids);
        let mut hidden = mtp.fuse(embedding, target_or_mtp_hidden)?;
        let layer_count = mtp.layers.len();
        if layer_count == 0 {
            return Err(Error::Cache("Qwen MTP graph contains no layer".into()));
        }
        let layer_index = step % layer_count;
        let rotary = rotary_angles_for_dtype(
            self.model.rope,
            positions,
            evaluated_context,
            hidden.dtype(),
        )?;
        let mut pending = PendingQwenCacheWrites::new();
        let bank = state
            .tensors
            .mtp()
            .ok_or_else(|| Error::Cache("Qwen MTP cache bank is not enabled".into()))?;
        hidden = mtp.layers[layer_index].forward_cached(
            hidden,
            Some(&rotary),
            0,
            committed,
            baseline,
            bank,
            Qwen3_5LayerCache::FullAttention(layer_index),
            &mut pending,
            DeltaNetMode::Recurrent,
        )?;
        hidden = mtp.norm.forward(hidden);
        let write = match (
            pending.native_kv.pop(),
            pending.native_kv.is_empty(),
            pending.recurrent.is_empty(),
        ) {
            (Some(write), true, true) => write,
            _ => {
                return Err(Error::Cache(
                    "Qwen MTP layer produced an invalid cache write set".into(),
                ));
            }
        };
        let logits = self.project_hidden(hidden.clone())?;
        commit_mtp_write(&mut state.tensors, write, sequence)?;
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
    /// Each row is compacted according to `token_mask`, evaluated against its own
    /// committed length, and scattered back into the rectangular output. If row
    /// `b` has baseline `L_b`, its `i`th valid implicit position is `L_b + i` and
    /// its full-attention query can see keys `j <= L_b + i`.
    fn forward_with_decode_state(
        &self,
        input: CausalLmInput<B>,
        state: &mut Qwen3_5DecodeState<B>,
        options: &ForwardOptions,
        phase: StatefulPhase,
    ) -> Result<CausalLmOutput<B>> {
        if options.attentions {
            return Err(Error::BackendCapability(
                "Qwen attention probability capture requires the reference attention path".into(),
            ));
        }
        state.validate_for_model(self)?;
        let prepared = prepare_stateful_input(self, &input, state, phase)?;
        validate_logits_selection(&options.logits, prepared.sequence)?;
        let device = input.token_ids.device();
        let diagnostic_count = self.model.layers.len() + 2;
        let mut pending = PendingQwenCacheWrites::new();
        let mut final_rows = Vec::with_capacity(prepared.batch);
        let mut diagnostic_rows: Vec<Vec<Tensor<B, 3>>> = if options.hidden_states {
            (0..diagnostic_count)
                .map(|_| Vec::with_capacity(prepared.batch))
                .collect()
        } else {
            Vec::new()
        };

        for sequence_index in 0..prepared.batch {
            let valid = &prepared.valid_indices[sequence_index];
            if valid.is_empty() {
                let empty = empty_hidden_row(self, prepared.sequence, &device);
                final_rows.push(empty.clone());
                if options.hidden_states {
                    for destination in &mut diagnostic_rows {
                        destination.push(empty.clone());
                    }
                }
                continue;
            }
            let selector = selection_tensor::<B>(valid, &device)?;
            let tokens = input
                .token_ids
                .clone()
                .slice([sequence_index..sequence_index + 1, 0..prepared.sequence])
                .select(1, selector.clone());
            let committed_baseline = state.tensors.lengths()[sequence_index];
            let baseline = state.tensors.working_lengths()[sequence_index];
            let positions = match &input.position_ids {
                Some(position_ids) => position_ids
                    .clone()
                    .slice([sequence_index..sequence_index + 1, 0..prepared.sequence])
                    .select(1, selector.clone()),
                None => implicit_positions::<B>(baseline, valid.len(), &device)?,
            };
            let evaluated_context = prepared.evaluated_contexts[sequence_index];
            let (compact, compact_diagnostics) = self.model.forward_compact_row(
                tokens,
                positions,
                sequence_index,
                committed_baseline,
                baseline,
                evaluated_context,
                &state.tensors,
                &state.layout.schedule,
                &mut pending,
                phase.delta_mode(),
                options.hidden_states,
            )?;
            final_rows.push(scatter_compact(compact, prepared.sequence, valid)?);
            if options.hidden_states {
                if compact_diagnostics.len() != diagnostic_count {
                    return Err(Error::Cache(format!(
                        "Qwen hidden-state count must be {diagnostic_count}, found {}",
                        compact_diagnostics.len()
                    )));
                }
                for (destination, compact_state) in
                    diagnostic_rows.iter_mut().zip(compact_diagnostics)
                {
                    destination.push(scatter_compact(compact_state, prepared.sequence, valid)?);
                }
            }
        }

        let hidden = Tensor::cat(final_rows, 0);
        let hidden_states = diagnostic_rows
            .into_iter()
            .map(|rows| Tensor::cat(rows, 0))
            .collect();
        let selected = select_hidden(hidden.clone(), input.token_mask.as_ref(), &options.logits)?;
        let logits = self.project_hidden(selected)?;
        pending.commit(&mut state.tensors, &state.layout, &prepared.token_counts)?;
        Ok(CausalLmOutput {
            logits,
            final_hidden_state: options.hidden_states.then_some(hidden),
            hidden_states,
            attentions: Vec::new(),
            routers: Vec::new(),
            load_balancing_loss: None,
        })
    }
}

impl<B: Backend> Qwen3_5TextModel<B> {
    /// Evaluates one compacted row and records, but does not publish, its state.
    #[allow(clippy::too_many_arguments)]
    fn forward_compact_row(
        &self,
        token_ids: Tensor<B, 2, Int>,
        positions: Tensor<B, 2, Int>,
        sequence_index: usize,
        committed_baseline: usize,
        baseline: usize,
        evaluated_context: usize,
        state: &TensorDecodeState<B>,
        schedule: &[Qwen3_5LayerCache],
        pending: &mut PendingQwenCacheWrites<B>,
        delta_mode: DeltaNetMode,
        retain_hidden_states: bool,
    ) -> Result<(Tensor<B, 3>, Vec<Tensor<B, 3>>)> {
        let mut hidden = self.embed_tokens.forward(token_ids);
        let mut hidden_states = Vec::with_capacity(if retain_hidden_states {
            self.layers.len() + 2
        } else {
            0
        });
        if retain_hidden_states {
            hidden_states.push(hidden.clone());
        }
        let rotary = if schedule
            .iter()
            .any(|slot| matches!(slot, Qwen3_5LayerCache::FullAttention(_)))
        {
            Some(rotary_angles_for_dtype(
                self.rope,
                positions,
                evaluated_context,
                hidden.dtype(),
            )?)
        } else {
            None
        };
        if schedule.len() != self.layers.len() {
            return Err(Error::Cache(format!(
                "Qwen cache schedule has {} entries for {} model layers",
                schedule.len(),
                self.layers.len()
            )));
        }
        for (layer, slot) in self.layers.iter().zip(schedule.iter().copied()) {
            hidden = layer.forward_cached(
                hidden,
                rotary.as_ref(),
                sequence_index,
                committed_baseline,
                baseline,
                state.main(),
                slot,
                pending,
                delta_mode,
            )?;
            if retain_hidden_states {
                hidden_states.push(hidden.clone());
            }
        }
        hidden = self.norm.forward(hidden);
        if retain_hidden_states {
            hidden_states.push(hidden.clone());
        }
        Ok((hidden, hidden_states))
    }
}

impl<B: Backend> Qwen3_5DecoderLayer<B> {
    /// Evaluates one residual block against its mapped external cache slot.
    ///
    /// Native keys are rotated before being deferred, so later decode calls never
    /// need the old hidden states or old position IDs. DeltaNet defers its FP32
    /// terminal matrix and activation-dtype convolution tail after the complete
    /// compact segment. The caller publishes both only after the full operation.
    #[allow(clippy::too_many_arguments)]
    fn forward_cached(
        &self,
        hidden: Tensor<B, 3>,
        rotary: Option<&RotaryAngles<B>>,
        sequence_index: usize,
        committed_baseline: usize,
        baseline: usize,
        bank: &TensorCacheBank<B>,
        slot: Qwen3_5LayerCache,
        pending: &mut PendingQwenCacheWrites<B>,
        delta_mode: DeltaNetMode,
    ) -> Result<Tensor<B, 3>> {
        if slot.layer_type() != self.layer_type {
            return Err(Error::Cache(format!(
                "Qwen layer type {:?} was mapped to incompatible cache slot {slot:?}",
                self.layer_type
            )));
        }
        let residual = hidden.clone();
        let normalized = self.input_layernorm.forward(hidden);
        match slot {
            Qwen3_5LayerCache::FullAttention(cache_layer) => {
                let attention = self.self_attn.as_ref().ok_or_else(|| {
                    Error::Checkpoint("Qwen full-attention layer has no attention module".into())
                })?;
                let rotary = rotary.ok_or_else(|| {
                    Error::Cache("Qwen full-attention layer has no rotary angles".into())
                })?;
                let mut projection = attention.project_self(normalized)?;
                (projection.query, projection.key) =
                    apply_rotary(projection.query, projection.key, rotary)?;
                let new_keys = projection.key.clone();
                let new_values = projection.value.clone();
                let query_sequence = projection.query.dims()[2];
                let cache = bank.native_kv(NATIVE_KV_GROUP)?;
                let history = if cache.has_active_staging() {
                    cache.read_working_sequence(cache_layer, sequence_index, committed_baseline)?
                } else {
                    cache.read_sequence(cache_layer, sequence_index, baseline)?
                };
                if history.retained_positions() != (0..baseline) {
                    return Err(Error::Cache(
                        "Qwen full-attention cache evicted required global history".into(),
                    ));
                }
                if let Some((past_keys, past_values)) = history.materialize() {
                    projection.key = Tensor::cat(vec![past_keys, projection.key], 2);
                    projection.value = Tensor::cat(vec![past_values, projection.value], 2);
                }
                let mask = cached_causal_mask::<B>(
                    baseline,
                    query_sequence,
                    attention.query_heads,
                    &projection.query.device(),
                )?;
                let mixed = attention.attend(projection, Some(mask), None, false)?;
                let hidden = finish_residual(self, residual, mixed)?;
                pending.native_kv.push(PendingNativeKvWrite {
                    cache_layer,
                    sequence: sequence_index,
                    start: baseline,
                    keys: new_keys,
                    values: new_values,
                });
                Ok(hidden)
            }
            Qwen3_5LayerCache::DeltaNet(cache_layer) => {
                let delta = self.linear_attn.as_ref().ok_or_else(|| {
                    Error::Checkpoint("Qwen linear layer has no DeltaNet module".into())
                })?;
                let cache = bank.recurrent(RECURRENT_GROUP)?;
                let cached = if cache.has_active_staging() {
                    cache.read_working_sequence(cache_layer, sequence_index)?
                } else {
                    cache.read_sequence(cache_layer, sequence_index)?
                };
                let mut recurrent = GatedDeltaNetState {
                    convolution: cached.convolution().clone(),
                    recurrent: cached.matrix().clone(),
                };
                let token_count = normalized.dims()[1];
                let mixed =
                    delta.forward_with_state(normalized, None, &mut recurrent, delta_mode)?;
                let hidden = finish_residual(self, residual, mixed)?;
                pending.recurrent.push(PendingRecurrentWrite {
                    cache_layer,
                    sequence: sequence_index,
                    token_count,
                    matrix: recurrent.recurrent,
                    convolution: recurrent.convolution,
                });
                Ok(hidden)
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
fn cache_layout<B: Backend>(
    model: &Qwen3_5ForCausalLm<B>,
    staging_capacity: usize,
) -> Result<Qwen3_5CacheLayout> {
    if !model.model.supports_family_cache {
        return Err(Error::BackendCapability(
            "editable Qwen recipe backbones use stateless execution; cached family decoding requires Qwen3_5ForCausalLm::init with an unchanged family configuration".into(),
        ));
    }
    let mut full_layers = 0usize;
    let mut delta_layers = 0usize;
    let schedule = model
        .model
        .layers
        .iter()
        .map(|layer| match layer.layer_type {
            Qwen3_5LayerType::LinearAttention => {
                let slot = Qwen3_5LayerCache::DeltaNet(delta_layers);
                delta_layers += 1;
                slot
            }
            Qwen3_5LayerType::FullAttention => {
                let slot = Qwen3_5LayerCache::FullAttention(full_layers);
                full_layers += 1;
                slot
            }
        })
        .collect();
    let activation_dtype = model.model.embed_tokens.weight.val().dtype();
    let native_kv = (full_layers > 0)
        .then(|| {
            NativeKvCacheSpec::new(
                full_layers,
                model.config().num_key_value_heads,
                model.config().head_dim,
                KvCacheStorage::Full,
                activation_dtype,
            )
            .map(|spec| spec.with_staging_capacity(staging_capacity))
        })
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
        .map(|mtp| {
            NativeKvCacheSpec::new(
                mtp.layers.len(),
                model.config().num_key_value_heads,
                model.config().head_dim,
                KvCacheStorage::Full,
                activation_dtype,
            )
            .map(|spec| spec.with_staging_capacity(staging_capacity))
        })
        .transpose()?;
    Ok(Qwen3_5CacheLayout {
        schedule,
        native_kv,
        recurrent,
        mtp_native_kv,
        staging_capacity,
    })
}

/// Deferred native-KV append produced by one successful attention layer.
struct PendingNativeKvWrite<B: Backend> {
    /// Dense layer slot within the native-KV group.
    cache_layer: usize,
    /// Independent batch row receiving the suffix.
    sequence: usize,
    /// Absolute logical cursor where the new suffix begins.
    start: usize,
    /// New rotated keys shaped `[1, kv_heads, suffix, head_dim]`.
    keys: Tensor<B, 4>,
    /// New values with the same native-head axes as the keys.
    values: Tensor<B, 4>,
}

/// Publishes one complete MTP-layer KV suffix and advances only its cursor.
fn commit_mtp_write<B: Backend>(
    state: &mut TensorDecodeState<B>,
    write: PendingNativeKvWrite<B>,
    token_count: usize,
) -> Result<()> {
    let PendingNativeKvWrite {
        cache_layer,
        sequence,
        start,
        keys,
        values,
    } = write;
    let working = state
        .mtp_working_lengths()
        .and_then(|lengths| lengths.get(sequence).copied())
        .ok_or_else(|| Error::Cache("Qwen MTP metadata is not enabled".into()))?;
    if start != working || keys.dims()[2] != token_count || values.dims()[2] != token_count {
        return Err(Error::Cache(
            "Qwen MTP cache write does not cover its complete logical suffix".into(),
        ));
    }
    if state.has_active_transaction() {
        state
            .mtp_mut()
            .ok_or_else(|| Error::Cache("Qwen MTP cache bank is not enabled".into()))?
            .native_kv_mut(NATIVE_KV_GROUP)?
            .stage_sequence(cache_layer, sequence, keys, values)?;
        state.advance_mtp(&[token_count])?;
        return Ok(());
    }

    let bank = state
        .mtp()
        .ok_or_else(|| Error::Cache("Qwen MTP cache bank is not enabled".into()))?;
    let cache = bank.native_kv(NATIVE_KV_GROUP)?;
    let write = cache.prepare_sequence_write(cache_layer, sequence, start, keys, values)?;
    let writes = bank.prepare_direct_writes(Some((NATIVE_KV_GROUP, vec![write])), None)?;
    state.advance_mtp(&[token_count])?;
    state
        .mtp_mut()
        .ok_or_else(|| Error::Cache("Qwen MTP cache bank is not enabled".into()))?
        .apply_prepared_direct_writes(writes);
    Ok(())
}

/// Deferred terminal-state replacement produced by one DeltaNet layer.
struct PendingRecurrentWrite<B: Backend> {
    /// Dense DeltaNet layer slot within the recurrent group.
    cache_layer: usize,
    /// Independent batch row whose terminal state will be replaced.
    sequence: usize,
    /// Number of semantic tokens consumed to produce this terminal state.
    token_count: usize,
    /// Terminal association matrix with batch, value-head, key, and value axes.
    matrix: Tensor<B, 4>,
    /// Terminal causal-convolution history with batch, channel, and time axes.
    convolution: Tensor<B, 3>,
}

/// Scratch cache updates for one complete public prefill or decode operation.
///
/// Model computation may return recoverable errors after some earlier layers
/// have succeeded. Holding their outputs here keeps the committed cache exactly
/// unchanged until every layer and batch row has completed. The final commit is
/// a two-phase operation: every tensor write is prepared first, then logical
/// lengths advance and opaque write capabilities are applied without a further
/// fallible branch.
struct PendingQwenCacheWrites<B: Backend> {
    /// Unpublished per-row full-attention suffixes for this operation.
    native_kv: Vec<PendingNativeKvWrite<B>>,
    /// Unpublished per-row DeltaNet terminal states for this operation.
    recurrent: Vec<PendingRecurrentWrite<B>>,
}

impl<B: Backend> PendingQwenCacheWrites<B> {
    /// Creates an empty operation-local write set.
    fn new() -> Self {
        Self {
            native_kv: Vec::new(),
            recurrent: Vec::new(),
        }
    }

    /// Validates, atomically publishes metadata, and applies every cache write.
    ///
    /// For each active row `b` and physical layer slot `l`, exactly one update
    /// must exist at the old cursor `L_b` and contain `N_b` new tokens. This
    /// coverage invariant prevents an apparently successful call from advancing
    /// metadata when one layer's physical state was omitted or duplicated.
    fn commit(
        self,
        state: &mut TensorDecodeState<B>,
        layout: &Qwen3_5CacheLayout,
        token_counts: &[usize],
    ) -> Result<()> {
        self.validate_coverage(state, layout, token_counts)?;
        if state.has_active_transaction() {
            // Speculative verification advances one target token at a time.
            // That restriction is mathematically necessary for DeltaNet: its
            // recurrence is not invertible, so every accepted prefix requires
            // the post-token state snapshot at that exact boundary.
            if layout.recurrent.is_some() && token_counts.iter().any(|&count| count > 1) {
                return Err(Error::Cache(
                    "transactional Qwen DeltaNet decode accepts at most one token per row and call"
                        .into(),
                ));
            }
            let PendingQwenCacheWrites {
                native_kv,
                recurrent,
            } = self;
            if !native_kv.is_empty() {
                let cache = state.main_mut().native_kv_mut(NATIVE_KV_GROUP)?;
                for write in native_kv {
                    cache.stage_sequence(
                        write.cache_layer,
                        write.sequence,
                        write.keys,
                        write.values,
                    )?;
                }
            }
            if !recurrent.is_empty() {
                let cache = state.main_mut().recurrent_mut(RECURRENT_GROUP)?;
                for write in recurrent {
                    cache.stage_sequence_snapshots(
                        write.cache_layer,
                        write.sequence,
                        write.matrix.unsqueeze_dim::<5>(1),
                        write.convolution.unsqueeze_dim::<4>(1),
                    )?;
                }
            }
            state.advance(token_counts)?;
            return Ok(());
        }
        let bank = state.main();
        let native_kv = if self.native_kv.is_empty() {
            None
        } else {
            let cache = bank.native_kv(NATIVE_KV_GROUP)?;
            let writes = self
                .native_kv
                .into_iter()
                .map(|write| {
                    cache.prepare_sequence_write(
                        write.cache_layer,
                        write.sequence,
                        write.start,
                        write.keys,
                        write.values,
                    )
                })
                .collect::<core::result::Result<Vec<_>, _>>()?;
            Some((NATIVE_KV_GROUP, writes))
        };
        let recurrent = if self.recurrent.is_empty() {
            None
        } else {
            let cache = bank.recurrent(RECURRENT_GROUP)?;
            let writes = self
                .recurrent
                .into_iter()
                .map(|write| {
                    cache.prepare_sequence_state(
                        write.cache_layer,
                        write.sequence,
                        write.matrix,
                        write.convolution,
                    )
                })
                .collect::<core::result::Result<Vec<_>, _>>()?;
            Some((RECURRENT_GROUP, writes))
        };
        let writes = bank.prepare_direct_writes(native_kv, recurrent)?;

        // `advance` is the final recoverable step and is itself atomic across
        // rows. Once it succeeds, applying the opaque bank capability contains
        // only tensor assignments, so callers can never observe half-published
        // Result-level state.
        state.advance(token_counts)?;
        state.main_mut().apply_prepared_direct_writes(writes);
        Ok(())
    }

    /// Checks exact active-row/layer coverage before any tensor is consumed.
    fn validate_coverage(
        &self,
        state: &TensorDecodeState<B>,
        layout: &Qwen3_5CacheLayout,
        token_counts: &[usize],
    ) -> Result<()> {
        let batch = state.batch_size();
        if token_counts.len() != batch {
            return Err(Error::Cache(format!(
                "Qwen pending writes describe {} rows for batch {batch}",
                token_counts.len()
            )));
        }
        let active_rows = token_counts.iter().filter(|&&count| count > 0).count();
        let full_layers = layout.native_kv.map_or(0, NativeKvCacheSpec::layers);
        let recurrent_layers = layout.recurrent.map_or(0, RecurrentStateSpec::layers);
        let expected_native = full_layers.checked_mul(active_rows).ok_or_else(|| {
            Error::configuration("qwen.cache_commit", "native write count overflows usize")
        })?;
        let expected_recurrent = recurrent_layers.checked_mul(active_rows).ok_or_else(|| {
            Error::configuration("qwen.cache_commit", "recurrent write count overflows usize")
        })?;
        if self.native_kv.len() != expected_native || self.recurrent.len() != expected_recurrent {
            return Err(Error::Cache(format!(
                "Qwen pending cache coverage mismatch: native {}/{expected_native}, recurrent {}/{expected_recurrent}",
                self.native_kv.len(),
                self.recurrent.len()
            )));
        }

        let mut native_seen = BTreeSet::new();
        for write in &self.native_kv {
            let count = token_counts.get(write.sequence).copied().ok_or_else(|| {
                Error::Cache(format!(
                    "Qwen native write references batch row {} outside {batch}",
                    write.sequence
                ))
            })?;
            if count == 0
                || write.cache_layer >= full_layers
                || write.start != state.working_lengths()[write.sequence]
                || write.keys.dims()[2] != count
                || write.values.dims()[2] != count
                || !native_seen.insert((write.cache_layer, write.sequence))
            {
                return Err(Error::Cache(format!(
                    "invalid or duplicate Qwen native write for layer {}, row {}",
                    write.cache_layer, write.sequence
                )));
            }
        }
        let mut recurrent_seen = BTreeSet::new();
        for write in &self.recurrent {
            let count = token_counts.get(write.sequence).copied().ok_or_else(|| {
                Error::Cache(format!(
                    "Qwen recurrent write references batch row {} outside {batch}",
                    write.sequence
                ))
            })?;
            if count == 0
                || write.cache_layer >= recurrent_layers
                || write.token_count != count
                || !recurrent_seen.insert((write.cache_layer, write.sequence))
            {
                return Err(Error::Cache(format!(
                    "invalid or duplicate Qwen recurrent write for layer {}, row {}",
                    write.cache_layer, write.sequence
                )));
            }
        }
        for (sequence, &token_count) in token_counts.iter().enumerate() {
            if token_count == 0 {
                continue;
            }
            for layer in 0..full_layers {
                if !native_seen.contains(&(layer, sequence)) {
                    return Err(Error::Cache(format!(
                        "missing Qwen native write for layer {layer}, row {sequence}"
                    )));
                }
            }
            for layer in 0..recurrent_layers {
                if !recurrent_seen.contains(&(layer, sequence)) {
                    return Err(Error::Cache(format!(
                        "missing Qwen recurrent write for layer {layer}, row {sequence}"
                    )));
                }
            }
        }
        Ok(())
    }
}

/// Validated rectangular input metadata needed before any cache mutation.
struct PreparedStatefulInput {
    /// Number of independent rows in the rectangular input.
    batch: usize,
    /// Input column count, including padding that will not enter the cache.
    sequence: usize,
    /// Original column indices of semantic tokens in each row.
    valid_indices: Vec<Vec<usize>>,
    /// Per-row append lengths; zero identifies an inactive decode row.
    token_counts: Vec<usize>,
    /// Per-row RoPE context extent including explicit absolute positions.
    evaluated_contexts: Vec<usize>,
}

/// Validates devices, shapes, positions, masks, and all capacity transitions.
fn prepare_stateful_input<B: Backend>(
    model: &Qwen3_5ForCausalLm<B>,
    input: &CausalLmInput<B>,
    state: &Qwen3_5DecodeState<B>,
    phase: StatefulPhase,
) -> Result<PreparedStatefulInput> {
    let [batch, sequence] = input.token_ids.dims();
    if batch == 0 || sequence == 0 {
        return Err(Error::shape(
            "qwen.input_ids",
            "non-empty [batch, sequence]",
            format!("[{batch}, {sequence}]"),
        ));
    }
    if batch != state.tensors.batch_size() {
        return Err(TensorCacheError::BatchSizeMismatch {
            configured: state.tensors.batch_size(),
            requested: batch,
        }
        .into());
    }
    ensure_device(
        "qwen.input_ids",
        &input.token_ids.device(),
        state.tensors.device(),
    )?;
    ensure_device(
        "qwen.model",
        &model.model.embed_tokens.weight.val().device(),
        state.tensors.device(),
    )?;
    if let Some(mask) = &input.token_mask {
        ensure_device(
            "qwen.token_mask",
            &mask.as_tensor().device(),
            state.tensors.device(),
        )?;
        validate_stateful_mask_shape(mask, batch, sequence)?;
    }
    if let Some(positions) = &input.position_ids {
        ensure_device(
            "qwen.position_ids",
            &positions.device(),
            state.tensors.device(),
        )?;
        if positions.dims() != [batch, sequence] {
            return Err(Error::shape(
                "qwen.position_ids",
                format!("[{batch}, {sequence}]"),
                format!("{:?}", positions.dims()),
            ));
        }
    }
    let valid_indices = semantic_indices(input.token_mask.as_ref(), batch, sequence)?;
    if !phase.permits_inactive_rows() && valid_indices.iter().any(Vec::is_empty) {
        return Err(Error::shape(
            "qwen.token_mask",
            "at least one valid token in every prefill row",
            "a fully masked prompt row",
        ));
    }
    let token_counts = valid_indices.iter().map(Vec::len).collect::<Vec<_>>();
    let mut implicit_contexts = Vec::with_capacity(batch);
    for (sequence_index, (&baseline, &additional)) in state
        .tensors
        .working_lengths()
        .iter()
        .zip(&token_counts)
        .enumerate()
    {
        let requested = baseline.checked_add(additional).ok_or({
            TensorCacheError::Metadata(DecodeStateError::LengthOverflow {
                sequence: sequence_index,
                length: baseline,
                additional,
            })
        })?;
        if requested > state.tensors.capacity() {
            return Err(
                TensorCacheError::Metadata(DecodeStateError::CapacityExceeded {
                    sequence: sequence_index,
                    requested,
                    capacity: state.tensors.capacity(),
                })
                .into(),
            );
        }
        implicit_contexts.push(requested);
    }
    let mut evaluated_contexts = Vec::with_capacity(batch);
    for (sequence_index, valid) in valid_indices.iter().enumerate() {
        if valid.is_empty() {
            evaluated_contexts.push(implicit_contexts[sequence_index]);
            continue;
        }
        let selector = selection_tensor::<B>(valid, state.tensors.device())?;
        let compact_tokens = input
            .token_ids
            .clone()
            .slice([sequence_index..sequence_index + 1, 0..sequence])
            .select(1, selector.clone());
        validate_token_ids(compact_tokens, model.config().vocab_size)?;
        let context = match &input.position_ids {
            Some(positions) => {
                let compact = positions
                    .clone()
                    .slice([sequence_index..sequence_index + 1, 0..sequence])
                    .select(1, selector);
                validate_position_range(compact.clone(), model.config().max_position_embeddings)?;
                evaluated_context(compact)?
            }
            None => implicit_contexts[sequence_index],
        };
        validate_cached_rope_context(model.model.rope.scaling(), context)?;
        evaluated_contexts.push(context);
    }
    Ok(PreparedStatefulInput {
        batch,
        sequence,
        valid_indices,
        token_counts,
        evaluated_contexts,
    })
}

/// Validates only rectangular mask geometry, leaving empty-row policy to phase.
fn validate_stateful_mask_shape<B: Backend>(
    mask: &crate::architecture::TokenMask<B>,
    batch: usize,
    sequence: usize,
) -> Result<()> {
    if mask.as_tensor().dims() != [batch, sequence] {
        return Err(Error::shape(
            "qwen.token_mask",
            format!("[{batch}, {sequence}]"),
            format!("{:?}", mask.as_tensor().dims()),
        ));
    }
    Ok(())
}

/// Converts a semantic mask into stable per-row source-column indices.
fn semantic_indices<B: Backend>(
    mask: Option<&crate::architecture::TokenMask<B>>,
    batch: usize,
    sequence: usize,
) -> Result<Vec<Vec<usize>>> {
    let Some(mask) = mask else {
        let dense = (0..sequence).collect::<Vec<_>>();
        return Ok((0..batch).map(|_| dense.clone()).collect());
    };
    let values =
        crate::architecture::tensor_bool_values(mask.as_tensor().clone(), "Qwen token mask")?;
    Ok(values
        .chunks_exact(sequence)
        .map(|row| {
            row.iter()
                .enumerate()
                .filter_map(|(index, &valid)| valid.then_some(index))
                .collect()
        })
        .collect())
}

/// Creates a checked Burn selector for valid columns in one row.
fn selection_tensor<B: Backend>(
    indices: &[usize],
    device: &B::Device,
) -> Result<Tensor<B, 1, Int>> {
    let values = indices
        .iter()
        .map(|&index| {
            i64::try_from(index).map_err(|_| {
                Error::configuration(
                    "qwen.token_mask",
                    format!("column index {index} exceeds signed tensor-index range"),
                )
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Tensor::from_data(
        TensorData::new(values, [indices.len()]),
        device,
    ))
}

/// Derives compact implicit positions `baseline .. baseline + tokens`.
fn implicit_positions<B: Backend>(
    baseline: usize,
    tokens: usize,
    device: &B::Device,
) -> Result<Tensor<B, 2, Int>> {
    let positions = derive_position_ids(&vec![true; tokens], baseline, 0)
        .map_err(|error| Error::configuration("qwen.position_ids", error.to_string()))?
        .into_iter()
        .map(|position| {
            i64::try_from(position).map_err(|_| {
                Error::configuration("qwen.position_ids", "implicit position exceeds i64")
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Tensor::from_data(
        TensorData::new(positions, [1, tokens]),
        device,
    ))
}

/// Returns one plus the largest explicit or derived position in a compact row.
pub(super) fn evaluated_context<B: Backend>(positions: Tensor<B, 2, Int>) -> Result<usize> {
    let maximum = positions
        .max()
        .try_into_scalar()
        .map_err(|error| {
            Error::BackendCapability(format!("failed to inspect Qwen position IDs: {error}"))
        })?
        .elem::<i64>();
    let maximum = usize::try_from(maximum)
        .map_err(|_| Error::configuration("qwen.position_ids", "positions must be non-negative"))?;
    maximum
        .checked_add(1)
        .ok_or_else(|| Error::configuration("qwen.position_ids", "evaluated context overflows"))
}

/// Validates every explicit position before a cache tensor is mutated.
pub(super) fn validate_position_range<B: Backend>(
    positions: Tensor<B, 2, Int>,
    maximum_context: usize,
) -> Result<()> {
    let minimum = positions
        .clone()
        .min()
        .try_into_scalar()
        .map_err(|error| {
            Error::BackendCapability(format!("failed to validate Qwen positions: {error}"))
        })?
        .elem::<i64>();
    let maximum = positions
        .max()
        .try_into_scalar()
        .map_err(|error| {
            Error::BackendCapability(format!("failed to validate Qwen positions: {error}"))
        })?
        .elem::<i64>();
    if minimum < 0 || usize::try_from(maximum).map_or(true, |value| value >= maximum_context) {
        return Err(Error::shape(
            "qwen.position_ids",
            format!("position values in 0..{maximum_context}"),
            format!("observed range {minimum}..={maximum}"),
        ));
    }
    Ok(())
}

/// Rejects dynamic-NTK transitions that would invalidate already cached keys.
fn validate_cached_rope_context(scaling: RopeScaling, evaluated_context: usize) -> Result<()> {
    if let RopeScaling::DynamicNtk(dynamic) = scaling
        && evaluated_context > dynamic.original_max_position_embeddings()
    {
        return Err(Error::BackendCapability(
            "dynamic-NTK decoding beyond its original context requires re-rotating cached keys"
                .into(),
        ));
    }
    Ok(())
}

/// Builds the explicit cached causal mask `excluded(q, k) = k > baseline + q`.
fn cached_causal_mask<B: Backend>(
    baseline: usize,
    query_sequence: usize,
    heads: usize,
    device: &B::Device,
) -> Result<Tensor<B, 4, Bool>> {
    let key_sequence = baseline.checked_add(query_sequence).ok_or_else(|| {
        Error::configuration("qwen.attention_mask", "key sequence length overflows usize")
    })?;
    let query_start = i64::try_from(baseline)
        .map_err(|_| Error::configuration("qwen.attention_mask", "query offset exceeds i64"))?;
    let query_end = i64::try_from(key_sequence)
        .map_err(|_| Error::configuration("qwen.attention_mask", "key length exceeds i64"))?;
    let key_end = query_end;
    let query = Tensor::<B, 1, Int>::arange(query_start..query_end, device)
        .reshape([1, 1, query_sequence, 1])
        .expand([1, heads, query_sequence, key_sequence]);
    let key = Tensor::<B, 1, Int>::arange(0..key_end, device)
        .reshape([1, 1, 1, key_sequence])
        .expand([1, heads, query_sequence, key_sequence]);
    Ok(key.greater(query))
}

/// Creates the defined output for a decode row containing zero valid tokens.
///
/// No layer is evaluated and no cache update is recorded. Returning zero hidden
/// states preserves the rectangular batch shape; Qwen's bias-free vocabulary
/// projection consequently also produces exact zero logits for this inactive
/// row.
fn empty_hidden_row<B: Backend>(
    model: &Qwen3_5ForCausalLm<B>,
    sequence: usize,
    device: &B::Device,
) -> Tensor<B, 3> {
    Tensor::zeros(
        [1, sequence, model.config().hidden_size],
        (device, model.model.embed_tokens.weight.val().dtype()),
    )
}

/// Scatters compact valid-token states back into a zeroed rectangular row.
fn scatter_compact<B: Backend>(
    compact: Tensor<B, 3>,
    sequence: usize,
    indices: &[usize],
) -> Result<Tensor<B, 3>> {
    if compact.dims()[1] != indices.len()
        || indices.windows(2).any(|pair| pair[0] >= pair[1])
        || indices.last().is_some_and(|&index| index >= sequence)
    {
        return Err(Error::shape(
            "qwen.compact_hidden",
            "compact width matching sorted, unique in-range output indices",
            format!("compact={:?}, indices={indices:?}", compact.dims()),
        ));
    }
    let width = compact.dims()[2];
    let device = compact.device();
    let dtype = compact.dtype();
    let mut segments = Vec::with_capacity(indices.len().saturating_mul(2).saturating_add(1));
    let mut output_cursor = 0usize;
    for (compact_index, &output_index) in indices.iter().enumerate() {
        if output_index > output_cursor {
            segments.push(Tensor::zeros(
                [1, output_index - output_cursor, width],
                (&device, dtype),
            ));
        }
        segments.push(compact.clone().narrow(1, compact_index, 1));
        output_cursor = output_index + 1;
    }
    if output_cursor < sequence {
        segments.push(Tensor::zeros(
            [1, sequence - output_cursor, width],
            (&device, dtype),
        ));
    }
    Ok(Tensor::cat(segments, 1))
}

/// Ensures model, input, and cache tensors share one backend device.
fn ensure_device<D: PartialEq>(tensor: &'static str, actual: &D, expected: &D) -> Result<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(Error::BackendCapability(format!(
            "{tensor} must reside on the Qwen decode-state device"
        )))
    }
}

#[cfg(test)]
#[path = "decode_tests.rs"]
mod tests;
