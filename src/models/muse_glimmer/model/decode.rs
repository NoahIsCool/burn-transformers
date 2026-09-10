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

use burn::tensor::{Int, Tensor, TensorData, backend::Backend};

use crate::{
    CausalLanguageModel, CausalLmInput, CausalLmOutput, DecodeInput, Error, ForwardOptions, Result,
    architecture::{DecodeState as ArchitectureDecodeState, TokenMask},
    cache::{
        CacheConfig, DecodeStateError, KvCacheStorage, NativeKvCacheSpec, TensorCacheError,
        TensorDecodeState,
    },
    position::{RopeScaling, apply_rotary, derive_position_ids, rotary_angles},
};

use super::{
    MuseGlimmerDecoderLayer, MuseGlimmerForCausalLm, MuseGlimmerModel, evaluated_context,
    scale_free_rms_norm, select_hidden, validate_logits_selection, validate_position_range,
    validate_token_ids,
};
use crate::models::muse_glimmer::{MuseGlimmerLayerType, MuseGlimmerTextConfig};

/// Stateful operation whose empty-row semantics differ intentionally.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StatefulPhase {
    /// Every initial prompt row must contain at least one semantic token.
    Prefill,
    /// Finished rows may remain fully masked while active peers continue.
    Decode,
}

impl StatefulPhase {
    /// Returns whether a semantic row may contain no valid token.
    const fn permits_inactive_rows(self) -> bool {
        matches!(self, Self::Decode)
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
    schedule: Vec<MuseGlimmerLayerCache>,
    groups: Vec<NativeKvCacheSpec>,
    window: Option<usize>,
}

/// External Muse decode state backed by shared preallocated tensor caches.
#[derive(Debug)]
pub struct MuseGlimmerDecodeState<B: Backend> {
    tensors: TensorDecodeState<B>,
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
    fn batch_size(&self) -> usize {
        self.tensors.batch_size()
    }

    fn capacity(&self) -> usize {
        self.tensors.capacity()
    }

    fn lengths(&self) -> &[usize] {
        self.tensors.lengths()
    }

    fn reset(&mut self) -> Result<()> {
        self.tensors.reset().map_err(Into::into)
    }

    fn reorder(&mut self, indices: &[usize]) -> Result<()> {
        self.tensors.reorder(indices).map_err(Into::into)
    }

    fn truncate(&mut self, lengths: &[usize]) -> Result<()> {
        self.tensors.truncate(lengths).map_err(Into::into)
    }

    fn device(&self) -> &B::Device {
        self.tensors.device()
    }
}

impl<B: Backend> CausalLanguageModel<B> for MuseGlimmerForCausalLm<B> {
    type Config = MuseGlimmerTextConfig;
    type State = MuseGlimmerDecodeState<B>;

    fn forward(
        &self,
        input: CausalLmInput<B>,
        options: &ForwardOptions,
    ) -> Result<CausalLmOutput<B>> {
        MuseGlimmerForCausalLm::forward(self, input, options)
    }

    fn new_decode_state(
        &self,
        config: CacheConfig,
        batch_size: usize,
        device: &B::Device,
    ) -> Result<Self::State> {
        MuseGlimmerDecodeState::new(self, config, batch_size, device)
    }

    fn prefill(
        &self,
        input: CausalLmInput<B>,
        state: &mut Self::State,
        options: &ForwardOptions,
    ) -> Result<CausalLmOutput<B>> {
        self.forward_with_decode_state(input, state, options, StatefulPhase::Prefill)
    }

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
            let valid = &prepared.valid_indices[row];
            if valid.is_empty() {
                let empty = empty_hidden_row(self, prepared.sequence, &device);
                final_rows.push(empty.clone());
                if options.hidden_states {
                    for destination in &mut hidden_rows {
                        destination.push(empty.clone());
                    }
                }
                continue;
            }
            let selector = selection_tensor(valid, &device)?;
            let tokens = input
                .token_ids
                .clone()
                .slice([row..row + 1, 0..prepared.sequence])
                .select(1, selector.clone());
            let baseline = state.tensors.lengths()[row];
            let positions = match &input.position_ids {
                Some(position_ids) => position_ids
                    .clone()
                    .slice([row..row + 1, 0..prepared.sequence])
                    .select(1, selector),
                None => implicit_positions(baseline, valid.len(), &device)?,
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
            final_rows.push(scatter_compact(compact.hidden, prepared.sequence, valid)?);
            if options.hidden_states {
                if compact.hidden_states.len() != diagnostic_count {
                    return Err(Error::Cache(format!(
                        "Muse hidden-state count must be {diagnostic_count}, found {}",
                        compact.hidden_states.len()
                    )));
                }
                for (destination, hidden) in hidden_rows.iter_mut().zip(compact.hidden_states) {
                    destination.push(scatter_compact(hidden, prepared.sequence, valid)?);
                }
            }
        }

        let hidden = Tensor::cat(final_rows, 0);
        let hidden_states = hidden_rows
            .into_iter()
            .map(|rows| Tensor::cat(rows, 0))
            .collect();
        let selected = select_hidden(hidden.clone(), input.token_mask.as_ref(), &options.logits)?;
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
    hidden: Tensor<B, 3>,
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
    model_layer: usize,
    group: usize,
    group_layer: usize,
    row: usize,
    start: usize,
    keys: Tensor<B, 4>,
    values: Tensor<B, 4>,
}

/// Complete operation-local write set used for atomic publication.
struct PendingKvWrites<B: Backend> {
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

/// Host-side metadata prepared before any model layer executes.
struct PreparedStatefulInput {
    batch: usize,
    sequence: usize,
    valid_indices: Vec<Vec<usize>>,
    token_counts: Vec<usize>,
    evaluated_contexts: Vec<usize>,
}

/// Validates shapes, devices, masks, positions, and capacity transitions.
fn prepare_stateful_input<B: Backend>(
    model: &MuseGlimmerForCausalLm<B>,
    input: &CausalLmInput<B>,
    state: &MuseGlimmerDecodeState<B>,
    phase: StatefulPhase,
) -> Result<PreparedStatefulInput> {
    let [batch, sequence] = input.token_ids.dims();
    if batch == 0 || sequence == 0 {
        return Err(Error::shape(
            "muse_glimmer.input_ids",
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
        "muse_glimmer.input_ids",
        &input.token_ids.device(),
        state.tensors.device(),
    )?;
    ensure_device(
        "muse_glimmer.model",
        &model.model.embed_tokens.weight.val().device(),
        state.tensors.device(),
    )?;
    if let Some(mask) = &input.token_mask {
        ensure_device(
            "muse_glimmer.token_mask",
            &mask.as_tensor().device(),
            state.tensors.device(),
        )?;
        if mask.as_tensor().dims() != [batch, sequence] {
            return Err(Error::shape(
                "muse_glimmer.token_mask",
                format!("[{batch}, {sequence}]"),
                format!("{:?}", mask.as_tensor().dims()),
            ));
        }
    }
    if let Some(positions) = &input.position_ids {
        ensure_device(
            "muse_glimmer.position_ids",
            &positions.device(),
            state.tensors.device(),
        )?;
        if positions.dims() != [batch, sequence] {
            return Err(Error::shape(
                "muse_glimmer.position_ids",
                format!("[{batch}, {sequence}]"),
                format!("{:?}", positions.dims()),
            ));
        }
    }
    let valid_indices = semantic_indices(input.token_mask.as_ref(), batch, sequence)?;
    if !phase.permits_inactive_rows() && valid_indices.iter().any(Vec::is_empty) {
        return Err(Error::shape(
            "muse_glimmer.token_mask",
            "at least one valid token in every prefill row",
            "a fully masked prompt row",
        ));
    }
    let token_counts = valid_indices.iter().map(Vec::len).collect::<Vec<_>>();
    let mut implicit_contexts = Vec::with_capacity(batch);
    for (row, (&baseline, &additional)) in state
        .tensors
        .lengths()
        .iter()
        .zip(&token_counts)
        .enumerate()
    {
        let requested = baseline.checked_add(additional).ok_or({
            TensorCacheError::Metadata(DecodeStateError::LengthOverflow {
                sequence: row,
                length: baseline,
                additional,
            })
        })?;
        if requested > state.tensors.capacity() {
            return Err(
                TensorCacheError::Metadata(DecodeStateError::CapacityExceeded {
                    sequence: row,
                    requested,
                    capacity: state.tensors.capacity(),
                })
                .into(),
            );
        }
        implicit_contexts.push(requested);
    }
    let mut evaluated_contexts = Vec::with_capacity(batch);
    for (row, valid) in valid_indices.iter().enumerate() {
        if valid.is_empty() {
            evaluated_contexts.push(implicit_contexts[row]);
            continue;
        }
        let selector = selection_tensor(valid, state.tensors.device())?;
        let tokens = input
            .token_ids
            .clone()
            .slice([row..row + 1, 0..sequence])
            .select(1, selector.clone());
        validate_token_ids(tokens, model.config().vocab_size)?;
        let context = match &input.position_ids {
            Some(positions) => {
                let compact = positions
                    .clone()
                    .slice([row..row + 1, 0..sequence])
                    .select(1, selector);
                validate_position_range(compact.clone(), model.config().max_position_embeddings)?;
                evaluated_context(compact)?
            }
            None => implicit_contexts[row],
        };
        validate_cached_rope_context(model, context)?;
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

/// Reads a semantic mask into stable per-row source-column indices.
fn semantic_indices<B: Backend>(
    mask: Option<&TokenMask<B>>,
    batch: usize,
    sequence: usize,
) -> Result<Vec<Vec<usize>>> {
    let Some(mask) = mask else {
        let dense = (0..sequence).collect::<Vec<_>>();
        return Ok((0..batch).map(|_| dense.clone()).collect());
    };
    let values =
        crate::architecture::tensor_bool_values(mask.as_tensor().clone(), "Muse token mask")?;
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

/// Builds a checked backend selector from host semantic indices.
fn selection_tensor<B: Backend>(
    indices: &[usize],
    device: &B::Device,
) -> Result<Tensor<B, 1, Int>> {
    let values = indices
        .iter()
        .map(|&index| {
            i64::try_from(index).map_err(|_| {
                Error::configuration(
                    "muse_glimmer.token_mask",
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

/// Creates compact implicit positions `baseline .. baseline + token_count`.
fn implicit_positions<B: Backend>(
    baseline: usize,
    tokens: usize,
    device: &B::Device,
) -> Result<Tensor<B, 2, Int>> {
    let positions = derive_position_ids(&vec![true; tokens], baseline, 0)
        .map_err(|error| Error::configuration("muse_glimmer.position_ids", error.to_string()))?
        .into_iter()
        .map(|position| {
            i64::try_from(position).map_err(|_| {
                Error::configuration("muse_glimmer.position_ids", "position exceeds i64")
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Tensor::from_data(
        TensorData::new(positions, [1, tokens]),
        device,
    ))
}

/// Rejects dynamic-NTK changes that would invalidate already rotated local keys.
fn validate_cached_rope_context<B: Backend>(
    model: &MuseGlimmerForCausalLm<B>,
    context: usize,
) -> Result<()> {
    for layer in &model.model.layers {
        if let Some(rope) = layer.rope
            && let RopeScaling::DynamicNtk(dynamic) = rope.scaling()
            && context > dynamic.original_max_position_embeddings()
        {
            return Err(Error::BackendCapability(
                "dynamic-NTK Muse decode beyond the original context requires re-rotating cached keys"
                    .into(),
            ));
        }
    }
    Ok(())
}

/// Scatters compact floating states back into rectangular source columns.
fn scatter_compact<B: Backend>(
    compact: Tensor<B, 3>,
    sequence: usize,
    indices: &[usize],
) -> Result<Tensor<B, 3>> {
    let [batch, compact_sequence, width] = compact.dims();
    if batch != 1
        || compact_sequence != indices.len()
        || indices.windows(2).any(|pair| pair[0] >= pair[1])
        || indices.last().is_some_and(|&index| index >= sequence)
    {
        return Err(Error::shape(
            "muse_glimmer.compact_hidden",
            "[1, valid_columns, width] with sorted unique indices",
            format!("shape={:?}, indices={indices:?}", compact.dims()),
        ));
    }
    let device = compact.device();
    let dtype = compact.dtype();
    let mut segments = Vec::with_capacity(indices.len().saturating_mul(2).saturating_add(1));
    let mut cursor = 0;
    for (compact_index, &output_index) in indices.iter().enumerate() {
        if output_index > cursor {
            segments.push(Tensor::zeros(
                [1, output_index - cursor, width],
                (&device, dtype),
            ));
        }
        segments.push(compact.clone().narrow(1, compact_index, 1));
        cursor = output_index + 1;
    }
    if cursor < sequence {
        segments.push(Tensor::zeros(
            [1, sequence - cursor, width],
            (&device, dtype),
        ));
    }
    Ok(Tensor::cat(segments, 1))
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

/// Checks that two model/state devices are identical.
fn ensure_device<D: PartialEq + core::fmt::Debug>(
    field: &str,
    actual: &D,
    expected: &D,
) -> Result<()> {
    if actual != expected {
        return Err(Error::BackendCapability(format!(
            "{field} is on {actual:?}, but Muse decode state is on {expected:?}"
        )));
    }
    Ok(())
}

#[cfg(test)]
#[path = "decode_tests.rs"]
mod tests;
