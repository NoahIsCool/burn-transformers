//! Preallocated cache-aware Muse-Glimmer inference.
//!
//! Local and global attention layers have different physical retention needs.
//! Sliding layers share one mirrored-ring group whose tensors are shaped
//! `[batch, kv_heads, 2 * window, head_dim]`; mirroring makes the newest logical
//! window one contiguous view after wraparound. NoPE global layers share a full
//! group shaped `[batch, kv_heads, capacity, head_dim]`. A deterministic schedule
//! maps model-layer indices to dense slots inside those groups.
//!
//! Public operations are failure-atomic. Every row is compacted according to the
//! semantic `true == valid` mask, all layer results and KV appends are computed in
//! operation-local handles, and every physical write is prevalidated. Logical
//! lengths advance only after the complete write set is known to be valid, after
//! which prepared assignments are infallible. A returned error therefore leaves
//! both lengths and cache bytes at the retry baseline.

use std::collections::{BTreeMap, BTreeSet};

use burn::tensor::{Int, Tensor, backend::Backend};

use crate::{
    CausalLanguageModel, CausalLmInput, CausalLmOutput, DecodeInput, Error, ForwardOptions, Result,
    architecture::DecodeState as ArchitectureDecodeState,
    cache::{CacheConfig, KvCacheStorage, NativeKvCacheSpec, TensorCacheError, TensorDecodeState},
    models::shared::{
        EmptyRows, StatefulContract, StatefulRows, concat_rows, ensure_same_device,
        ensure_static_rotary_context, implicit_position_ids, prepare_stateful_rows,
        select_hidden_with_last_columns, validate_logits_selection,
    },
    position::{apply_rotary, rotary_angles},
};

use super::{
    INPUT_NAMES, MuseGlimmerDecoderLayer, MuseGlimmerForCausalLm, MuseGlimmerModel,
    scale_free_rms_norm,
};
use crate::models::muse_glimmer::{MuseGlimmerLayerType, MuseGlimmerTextConfig};

/// Owner named when an input tensor is on another device than the cache.
const DECODE_STATE: &str = "Muse decode state";

/// Stateful operation whose empty-row semantics differ intentionally.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StatefulPhase {
    /// Every initial prompt row must contain at least one semantic token.
    Prefill,
    /// Finished rows may remain fully masked while active peers continue.
    Decode,
}

impl StatefulPhase {
    /// Returns the phase's policy for rows without a valid token.
    const fn empty_rows(self) -> EmptyRows {
        match self {
            Self::Prefill => EmptyRows::Reject,
            Self::Decode => EmptyRows::Permit,
        }
    }
}

/// Physical cache slot used by one Muse decoder layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MuseGlimmerLayerCache {
    /// Dense slot in the mirrored sliding-window cache group.
    Sliding {
        /// Native-KV group index in the shared tensor bank.
        group: usize,
        /// Layer index inside that homogeneous group.
        layer: usize,
    },
    /// Dense slot in the full-history cache group.
    Full {
        /// Native-KV group index in the shared tensor bank.
        group: usize,
        /// Layer index inside that homogeneous group.
        layer: usize,
    },
}

impl MuseGlimmerLayerCache {
    /// Returns the native-KV group and layer indices.
    #[must_use]
    pub const fn coordinates(self) -> (usize, usize) {
        match self {
            Self::Sliding { group, layer } | Self::Full { group, layer } => (group, layer),
        }
    }

    /// Returns the physical retention strategy represented by this slot.
    #[must_use]
    pub const fn storage(self) -> KvCacheStorage {
        match self {
            Self::Sliding { .. } => KvCacheStorage::MirroredSlidingRing,
            Self::Full { .. } => KvCacheStorage::Full,
        }
    }

    /// Returns the local window for sliding slots and no bound for full slots.
    fn window(self, window: Option<usize>) -> Result<Option<usize>> {
        match self {
            Self::Sliding { .. } => window.map(Some).ok_or_else(|| {
                Error::configuration(
                    "muse_glimmer.decode_state.window",
                    "a sliding cache slot requires an effective window",
                )
            }),
            Self::Full { .. } => Ok(None),
        }
    }
}

/// Immutable physical layout associated with one model and logical capacity.
#[derive(Debug, Clone, PartialEq, Eq)]
struct MuseGlimmerCacheLayout {
    /// Maps each model layer to its physical cache group and local layer slot.
    schedule: Vec<MuseGlimmerLayerCache>,
    /// Homogeneous KV allocation specifications in tensor-bank order.
    groups: Vec<NativeKvCacheSpec>,
    /// Effective retained token window for sliding groups; absent for full-only models.
    window: Option<usize>,
}

/// External Muse decode state backed by shared preallocated tensor caches.
#[derive(Debug)]
pub struct MuseGlimmerDecodeState<B: Backend> {
    /// Shared tensor bank and committed semantic token counts.
    tensors: TensorDecodeState<B>,
    /// Model geometry used to validate reuse of this state.
    layout: MuseGlimmerCacheLayout,
}

impl<B: Backend> MuseGlimmerDecodeState<B> {
    /// Allocates local mirrored rings and global full-history tensors.
    fn new(
        model: &MuseGlimmerForCausalLm<B>,
        requested: CacheConfig,
        batch_size: usize,
        device: &B::Device,
    ) -> Result<Self> {
        if requested.capacity() > model.config().max_position_embeddings {
            return Err(Error::configuration(
                "muse_glimmer.decode_state.capacity",
                format!(
                    "{} exceeds max_position_embeddings {}",
                    requested.capacity(),
                    model.config().max_position_embeddings
                ),
            ));
        }
        if requested.capacity() > i32::MAX as usize {
            return Err(Error::configuration(
                "muse_glimmer.decode_state.capacity",
                "must fit Burn's portable signed position-index range",
            ));
        }
        let physical_window = model
            .config()
            .resolved_sliding_window()
            .map(|window| window.min(requested.capacity()));
        let internal_config = CacheConfig::new(requested.sequence_count(), requested.capacity())
            .map_err(|error| {
                Error::configuration("muse_glimmer.decode_state", error.to_string())
            })?;
        let internal_config = match physical_window {
            Some(window) => internal_config
                .with_sliding_window(window)
                .map_err(|error| {
                    Error::configuration("muse_glimmer.decode_state", error.to_string())
                })?,
            None => internal_config,
        };
        let layout = cache_layout(model, physical_window)?;
        let mut tensors = TensorDecodeState::new(internal_config, batch_size, device)?;
        for (expected_group, spec) in layout.groups.iter().copied().enumerate() {
            let group = tensors.main_mut().add_native_kv(spec)?;
            if group != expected_group {
                return Err(Error::Cache(format!(
                    "Muse cache group index must be {expected_group}, found {group}"
                )));
            }
        }
        Ok(Self { tensors, layout })
    }

    /// Borrows the shared tensor state for memory and cache diagnostics.
    #[must_use]
    pub const fn tensor_state(&self) -> &TensorDecodeState<B> {
        &self.tensors
    }

    /// Returns one model layer's deterministic physical slot.
    #[must_use]
    pub fn layer_cache(&self, model_layer: usize) -> Option<MuseGlimmerLayerCache> {
        self.layout.schedule.get(model_layer).copied()
    }

    /// Returns the effective ring width, or `None` for a full-attention-only model.
    #[must_use]
    pub const fn sliding_window(&self) -> Option<usize> {
        self.layout.window
    }

    /// Verifies that state geometry belongs to the supplied model.
    fn validate_for_model(&self, model: &MuseGlimmerForCausalLm<B>) -> Result<()> {
        let expected = cache_layout(model, self.layout.window)?;
        if expected != self.layout || self.tensors.main().native_kv_count() != expected.groups.len()
        {
            return Err(Error::Cache(
                "Muse-Glimmer decode state is incompatible with the model".into(),
            ));
        }
        for (group, spec) in expected.groups.iter().copied().enumerate() {
            if self.tensors.main().native_kv(group)?.spec() != spec {
                return Err(Error::Cache(format!(
                    "Muse-Glimmer cache group {group} geometry differs from the model"
                )));
            }
        }
        Ok(())
    }
}

impl<B: Backend> ArchitectureDecodeState<B> for MuseGlimmerDecodeState<B> {
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

    /// Truncates full and sliding histories together when all requested prefixes remain retained.
    ///
    /// Ring-buffer eviction can make an older prefix unavailable; the shared
    /// cache rejects such rollback instead of reconstructing missing KV values.
    fn truncate(&mut self, lengths: &[usize]) -> Result<()> {
        self.tensors.truncate(lengths).map_err(Into::into)
    }

    /// Returns the device holding this state, independent of model parameter ownership.
    fn device(&self) -> &B::Device {
        self.tensors.device()
    }
}

impl<B: Backend> CausalLanguageModel<B> for MuseGlimmerForCausalLm<B> {
    /// Validated Muse text architecture associated with the model.
    type Config = MuseGlimmerTextConfig;
    /// External hybrid KV history used by prefill and decode.
    type State = MuseGlimmerDecodeState<B>;

    /// Delegates to the stateless family graph without allocating or mutating decode history.
    fn forward(
        &self,
        input: CausalLmInput<B>,
        options: &ForwardOptions,
    ) -> Result<CausalLmOutput<B>> {
        MuseGlimmerForCausalLm::forward(self, input, options)
    }

    /// Allocates external Muse-Glimmer cache groups after validating model/device geometry.
    fn new_decode_state(
        &self,
        config: CacheConfig,
        batch_size: usize,
        device: &B::Device,
    ) -> Result<Self::State> {
        MuseGlimmerDecodeState::new(self, config, batch_size, device)
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

impl<B: Backend> MuseGlimmerForCausalLm<B> {
    /// Evaluates one masked segment against external committed KV state.
    fn forward_with_decode_state(
        &self,
        input: CausalLmInput<B>,
        state: &mut MuseGlimmerDecodeState<B>,
        options: &ForwardOptions,
        phase: StatefulPhase,
    ) -> Result<CausalLmOutput<B>> {
        if options.attentions {
            return Err(Error::BackendCapability(
                "Muse-Glimmer cached attention-probability capture is unavailable".into(),
            ));
        }
        state.validate_for_model(self)?;
        if state.tensors.has_active_transaction() {
            return Err(TensorCacheError::TransactionActive {
                operation: "run ordinary Muse-Glimmer prefill/decode",
            }
            .into());
        }
        let prepared = prepare_stateful_input(self, &input, state, phase)?;
        validate_logits_selection(&options.logits, prepared.sequence)?;
        let device = input.token_ids.device();
        let diagnostic_count = self.model.layers.len() + 2;
        let mut pending = PendingKvWrites::new();
        let mut final_rows = Vec::with_capacity(prepared.batch);
        let mut hidden_rows: Vec<Vec<Tensor<B, 3>>> = if options.hidden_states {
            (0..diagnostic_count)
                .map(|_| Vec::with_capacity(prepared.batch))
                .collect()
        } else {
            Vec::new()
        };

        for row in 0..prepared.batch {
            let Some(plan) = prepared.compact_row::<B>(row, &device)? else {
                let empty = empty_hidden_row(self, prepared.sequence, &device);
                final_rows.push(empty.clone());
                if options.hidden_states {
                    for destination in &mut hidden_rows {
                        destination.push(empty.clone());
                    }
                }
                continue;
            };
            let tokens = plan.compact(&input.token_ids);
            let baseline = state.tensors.lengths()[row];
            let positions = match &input.position_ids {
                Some(position_ids) => plan.compact(position_ids),
                None => implicit_position_ids(
                    baseline,
                    plan.count(),
                    &device,
                    INPUT_NAMES.position_ids,
                )?,
            };
            let compact = self.model.forward_compact_row(
                tokens,
                positions,
                row,
                baseline,
                prepared.evaluated_contexts[row],
                state,
                &mut pending,
                options.hidden_states,
            )?;
            final_rows.push(plan.scatter(compact.hidden)?);
            if options.hidden_states {
                if compact.hidden_states.len() != diagnostic_count {
                    return Err(Error::Cache(format!(
                        "Muse hidden-state count must be {diagnostic_count}, found {}",
                        compact.hidden_states.len()
                    )));
                }
                for (destination, hidden) in hidden_rows.iter_mut().zip(compact.hidden_states) {
                    destination.push(plan.scatter(hidden)?);
                }
            }
        }

        let hidden = concat_rows(final_rows, "muse_glimmer.hidden")?;
        let hidden_states = hidden_rows
            .into_iter()
            .map(|rows| concat_rows(rows, "muse_glimmer.hidden_states"))
            .collect::<Result<Vec<_>>>()?;
        let selected = select_hidden_with_last_columns(
            hidden.clone(),
            prepared.last_columns(),
            &options.logits,
        )?;
        let logits = self.project_logits(selected)?;
        pending.commit(state, &prepared.token_counts)?;
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

impl<B: Backend> MuseGlimmerModel<B> {
    /// Runs one compact semantic row while deferring every physical cache write.
    #[allow(clippy::too_many_arguments)]
    fn forward_compact_row(
        &self,
        token_ids: Tensor<B, 2, Int>,
        positions: Tensor<B, 2, Int>,
        row: usize,
        baseline: usize,
        evaluated_context: usize,
        state: &MuseGlimmerDecodeState<B>,
        pending: &mut PendingKvWrites<B>,
        retain_hidden_states: bool,
    ) -> Result<CompactRowOutput<B>> {
        let mut hidden = scale_free_rms_norm(
            self.embed_tokens.forward(token_ids),
            self.config.rms_norm_eps,
        );
        let mut hidden_states = Vec::with_capacity(if retain_hidden_states {
            self.layers.len() + 2
        } else {
            0
        });
        if retain_hidden_states {
            hidden_states.push(hidden.clone());
        }
        for (layer_index, layer) in self.layers.iter().enumerate() {
            let cache_slot = state.layout.schedule[layer_index];
            hidden = layer.forward_cached(
                hidden,
                positions.clone(),
                evaluated_context,
                row,
                baseline,
                cache_slot,
                state,
                layer_index,
                pending,
            )?;
            if retain_hidden_states {
                hidden_states.push(hidden.clone());
            }
        }
        hidden = self.norm.forward(hidden);
        if retain_hidden_states {
            hidden_states.push(hidden.clone());
        }
        Ok(CompactRowOutput {
            hidden,
            hidden_states,
        })
    }
}

/// Result of one compact row before rectangular scatter.
struct CompactRowOutput<B: Backend> {
    /// Final normalized compact row, shaped `[1, valid_tokens, hidden_size]`.
    hidden: Tensor<B, 3>,
    /// Requested embedding, block-residual, and final-normalized row snapshots.
    hidden_states: Vec<Tensor<B, 3>>,
}

impl<B: Backend> MuseGlimmerDecoderLayer<B> {
    /// Evaluates one cached sandwich layer without publishing its new keys.
    #[allow(clippy::too_many_arguments)]
    fn forward_cached(
        &self,
        hidden: Tensor<B, 3>,
        positions: Tensor<B, 2, Int>,
        evaluated_context: usize,
        row: usize,
        baseline: usize,
        cache_slot: MuseGlimmerLayerCache,
        state: &MuseGlimmerDecodeState<B>,
        model_layer: usize,
        pending: &mut PendingKvWrites<B>,
    ) -> Result<Tensor<B, 3>> {
        let residual = hidden.clone();
        let normalized = self.input_layernorm.forward(hidden);
        let mut projection = self.self_attn.project_self(normalized)?;
        if let Some(rope) = self.rope {
            let angles = rotary_angles(rope, positions, evaluated_context)?;
            (projection.query, projection.key) =
                apply_rotary(projection.query, projection.key, &angles)?;
        }
        let new_keys = projection.key.clone();
        let new_values = projection.value.clone();
        let (group, group_layer) = cache_slot.coordinates();
        let cache = state.tensors.main().native_kv(group)?;
        let history = cache.read_sequence(group_layer, row, baseline)?;
        let retained = history.retained_positions();
        let local_window = cache_slot.window(state.layout.window)?;
        let expected_start = local_window.map_or(0, |window| baseline.saturating_sub(window));
        if retained != (expected_start..baseline) {
            return Err(Error::Cache(format!(
                "Muse layer {model_layer} retained {retained:?}, expected {expected_start}..{baseline}"
            )));
        }
        if let Some((past_keys, past_values)) = history.materialize() {
            projection.key = Tensor::cat(vec![past_keys, projection.key], 2);
            projection.value = Tensor::cat(vec![past_values, projection.value], 2);
        }
        let attended = self.self_attn.attend_cached(projection, local_window)?;
        let hidden = residual + self.post_attention_layernorm.forward(attended);
        let residual = hidden.clone();
        let fed = super::muse_gated_feed_forward(
            &self.mlp,
            self.pre_feedforward_layernorm.forward(hidden),
        )?;
        let hidden = residual + self.post_feedforward_layernorm.forward(fed);
        pending.writes.push(PendingKvWrite {
            model_layer,
            group,
            group_layer,
            row,
            start: baseline,
            keys: new_keys,
            values: new_values,
        });
        Ok(hidden)
    }
}

/// One deferred native-KV append tagged with its model and group coordinates.
struct PendingKvWrite<B: Backend> {
    /// Model layer whose append must appear exactly once for this active row.
    model_layer: usize,
    /// Native-KV group receiving the append.
    group: usize,
    /// Layer slot within the destination KV group.
    group_layer: usize,
    /// Batch row receiving the append.
    row: usize,
    /// Committed semantic token count before this append.
    start: usize,
    /// New keys shaped `[1, kv_heads, valid_tokens, head_dim]`.
    keys: Tensor<B, 4>,
    /// New values with the same geometry as the staged keys.
    values: Tensor<B, 4>,
}

/// Complete operation-local write set used for atomic publication.
struct PendingKvWrites<B: Backend> {
    /// Deferred appends checked for complete layer/row coverage before publication.
    writes: Vec<PendingKvWrite<B>>,
}

impl<B: Backend> PendingKvWrites<B> {
    /// Creates an empty scratch set.
    fn new() -> Self {
        Self { writes: Vec::new() }
    }

    /// Prepares every group, advances metadata, then publishes infallible writes.
    fn commit(self, state: &mut MuseGlimmerDecodeState<B>, token_counts: &[usize]) -> Result<()> {
        self.validate_coverage(state, token_counts)?;
        let mut grouped = BTreeMap::<usize, Vec<_>>::new();
        for write in self.writes {
            let cache = state.tensors.main().native_kv(write.group)?;
            let prepared = cache.prepare_sequence_write(
                write.group_layer,
                write.row,
                write.start,
                write.keys,
                write.values,
            )?;
            grouped.entry(write.group).or_default().push(prepared);
        }
        let mut prepared_groups = Vec::with_capacity(grouped.len());
        for (group, writes) in grouped {
            prepared_groups.push(
                state
                    .tensors
                    .main()
                    .prepare_direct_writes(Some((group, writes)), None)?,
            );
        }
        state.tensors.advance(token_counts)?;
        for prepared in prepared_groups {
            state
                .tensors
                .main_mut()
                .apply_prepared_direct_writes(prepared);
        }
        Ok(())
    }

    /// Proves one append exists for every active row and model layer.
    fn validate_coverage(
        &self,
        state: &MuseGlimmerDecodeState<B>,
        token_counts: &[usize],
    ) -> Result<()> {
        if token_counts.len() != state.tensors.batch_size() {
            return Err(Error::Cache(format!(
                "Muse pending counts describe {} rows for batch {}",
                token_counts.len(),
                state.tensors.batch_size()
            )));
        }
        let active = token_counts.iter().filter(|&&count| count > 0).count();
        let expected = state
            .layout
            .schedule
            .len()
            .checked_mul(active)
            .ok_or_else(|| {
                Error::configuration("muse_glimmer.cache_commit", "write count overflows usize")
            })?;
        if self.writes.len() != expected {
            return Err(Error::Cache(format!(
                "Muse pending cache coverage is {}/{expected}",
                self.writes.len()
            )));
        }
        let mut seen = BTreeSet::new();
        for write in &self.writes {
            let count = token_counts.get(write.row).copied().ok_or_else(|| {
                Error::Cache(format!("Muse cache write row {} is invalid", write.row))
            })?;
            let expected_slot = state.layout.schedule.get(write.model_layer).copied();
            if count == 0
                || expected_slot.map(MuseGlimmerLayerCache::coordinates)
                    != Some((write.group, write.group_layer))
                || write.start != state.tensors.lengths()[write.row]
                || write.keys.dims()[2] != count
                || write.values.dims()[2] != count
                || !seen.insert((write.model_layer, write.row))
            {
                return Err(Error::Cache(format!(
                    "invalid or duplicate Muse write for layer {}, row {}",
                    write.model_layer, write.row
                )));
            }
        }
        Ok(())
    }
}

/// Validates shapes, devices, masks, positions, and capacity transitions.
///
/// The shared preparation reads the mask, token IDs, and explicit positions in
/// one transfer; this wrapper adds the model-device and dynamic-NTK guards.
fn prepare_stateful_input<B: Backend>(
    model: &MuseGlimmerForCausalLm<B>,
    input: &CausalLmInput<B>,
    state: &MuseGlimmerDecodeState<B>,
    phase: StatefulPhase,
) -> Result<StatefulRows> {
    ensure_same_device(
        "muse_glimmer.model",
        &model.model.embed_tokens.weight.lazy_device(),
        state.tensors.device(),
        DECODE_STATE,
    )?;
    let rows = prepare_stateful_rows(
        input,
        state.tensors.device(),
        state.tensors.lengths(),
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
        for rope in model.model.layers.iter().filter_map(|layer| layer.rope) {
            ensure_static_rotary_context(rope.scaling(), context, "Muse")?;
        }
    }
    Ok(rows)
}

/// Resolves the two homogeneous groups and dense per-layer schedule.
fn cache_layout<B: Backend>(
    model: &MuseGlimmerForCausalLm<B>,
    window: Option<usize>,
) -> Result<MuseGlimmerCacheLayout> {
    let schedule_types = model.config().resolved_layer_types();
    let sliding_count = schedule_types
        .iter()
        .filter(|&&kind| kind == MuseGlimmerLayerType::SlidingAttention)
        .count();
    let full_count = schedule_types.len() - sliding_count;
    let window = match (sliding_count, window) {
        (0, _) => None,
        (_, Some(0)) => {
            return Err(Error::configuration(
                "muse_glimmer.decode_state.window",
                "must be non-zero",
            ));
        }
        (_, Some(window)) => Some(window),
        (_, None) => {
            return Err(Error::configuration(
                "muse_glimmer.decode_state.window",
                "is required by the sliding cache group",
            ));
        }
    };
    let dtype = model.model.embed_tokens.weight.val().dtype();
    let mut groups = Vec::with_capacity(2);
    let sliding_group = if sliding_count > 0 {
        let group = groups.len();
        groups.push(NativeKvCacheSpec::new(
            sliding_count,
            model.config().num_key_value_heads,
            model.config().head_dim,
            KvCacheStorage::MirroredSlidingRing,
            dtype,
        )?);
        Some(group)
    } else {
        None
    };
    let full_group = if full_count > 0 {
        let group = groups.len();
        groups.push(NativeKvCacheSpec::new(
            full_count,
            model.config().num_key_value_heads,
            model.config().head_dim,
            KvCacheStorage::Full,
            dtype,
        )?);
        Some(group)
    } else {
        None
    };
    let mut next_sliding = 0;
    let mut next_full = 0;
    let schedule = schedule_types
        .into_iter()
        .map(|kind| -> Result<MuseGlimmerLayerCache> {
            match kind {
                MuseGlimmerLayerType::SlidingAttention => {
                    let group = sliding_group.ok_or_else(|| {
                        Error::Cache(
                            "Muse schedule contains a sliding layer without a sliding cache group"
                                .into(),
                        )
                    })?;
                    let slot = MuseGlimmerLayerCache::Sliding {
                        group,
                        layer: next_sliding,
                    };
                    next_sliding += 1;
                    Ok(slot)
                }
                MuseGlimmerLayerType::FullAttention => {
                    let group = full_group.ok_or_else(|| {
                        Error::Cache(
                            "Muse schedule contains a full layer without a full cache group".into(),
                        )
                    })?;
                    let slot = MuseGlimmerLayerCache::Full {
                        group,
                        layer: next_full,
                    };
                    next_full += 1;
                    Ok(slot)
                }
            }
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(MuseGlimmerCacheLayout {
        schedule,
        groups,
        window,
    })
}

/// Produces a zero-filled inactive rectangular row in the model activation dtype.
fn empty_hidden_row<B: Backend>(
    model: &MuseGlimmerForCausalLm<B>,
    sequence: usize,
    device: &B::Device,
) -> Tensor<B, 3> {
    Tensor::zeros(
        [1, sequence, model.config().hidden_size],
        (device, model.model.embed_tokens.weight.val().dtype()),
    )
}

#[cfg(test)]
#[path = "decode_tests.rs"]
mod tests;
