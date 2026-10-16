//! Preallocated cache-aware, batched Muse-Glimmer inference.
//!
//! # Storage
//!
//! Local and global attention layers have different physical retention needs.
//! Sliding layers share one mirrored-ring group with its own window
//! ([`NativeKvCacheSpec::with_sliding_window`]), whose tensors are shaped
//! `[batch, kv_heads, 2 * window, head_dim]`; mirroring keeps the newest
//! logical window one contiguous view after wraparound. NoPE global layers
//! share a full group shaped `[batch, kv_heads, capacity, head_dim]`. A
//! deterministic schedule maps model-layer indices to dense slots inside
//! those groups.
//!
//! # Batched execution
//!
//! A call packs every row's valid tokens into one `[B, T]` block (shared with
//! DeepSeek, see `deepseek_v2::model::block`), runs each layer once for the
//! whole batch, and scatters results back to `[B, S]` with exact zeros at
//! padding and for finished rows. Global layers write their new keys and
//! values ahead into full storage and attend over one zero-copy slot-ordered
//! view `[0, max_b (L_b + n_b))`; local layers read their ring history in
//! slot order and append the step's tokens as a second segment. Queries sit
//! at cache slots `L_b + k`; visibility (causality, the window, validity) is
//! derived from key positions, and stale invalid slots are zeroed before the
//! value product. Long cached prefills run in score-bounded query chunks:
//! global chunks score their verified causal key prefix, and local chunks
//! score the whole ring only while one of their rows can still see history,
//! then exactly their rows' window union (`schedule::ring_chunk_keys`).
//!
//! Token IDs and explicit positions are validated only at valid columns, so
//! packing overwrites padding entries with token and position zero
//! (`BlockLayout::compact_inert`) before the embedding and rotary tables.
//!
//! # Atomicity
//!
//! Every input, capacity, and cursor check runs before any model work. Ring
//! writes are recorded in the step and published with
//! [`TensorDecodeState::publish_step`] after logits succeed, so a failure
//! leaves every length and every ring byte unchanged; full-storage
//! write-aheads may leave bytes beyond a row's length, which no read can
//! reach and the next write overwrites (logical atomicity).

use core::num::NonZeroUsize;

use burn::tensor::{Int, Tensor, backend::Backend};

use crate::{
    CausalLanguageModel, CausalLmInput, CausalLmOutput, DecodeInput, Error, ForwardOptions, Result,
    architecture::DecodeState as ArchitectureDecodeState,
    cache::{
        CacheConfig, KvBatchSegment, KvCacheStorage, NativeKvCacheSpec, StepBank, TensorCacheError,
        TensorDecodeState,
    },
    models::{
        deepseek_v2::model::{
            batch::{first_uncacheable_context, rotary_contexts},
            block::BlockLayout,
        },
        shared::{
            EmptyRows, StatefulContract, StatefulRows, ensure_same_device, prepare_stateful_rows,
            select_hidden_with_last_columns, validate_logits_selection,
        },
    },
    position::{RotaryAngles, RotaryFrequencies, rotary_angles_from_frequencies},
};

use super::{
    INPUT_NAMES, MuseAttentionProjection, MuseGlimmerForCausalLm, MuseKeySlicing, RotaryTables,
    plan_attention_chunks,
};
use crate::models::muse_glimmer::{
    MuseGlimmerLayerType, MuseGlimmerTextConfig,
    schedule::{GroupFault, cache_groups},
};

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
        // Each ring group carries its own window, so the bank needs none.
        let internal_config = CacheConfig::new(requested.sequence_count(), requested.capacity())
            .map_err(|error| {
                Error::configuration("muse_glimmer.decode_state", error.to_string())
            })?;
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
    ///
    /// The model's layout is re-planned with this state's ring window. A
    /// window the model cannot use (none for a model with sliding layers)
    /// means the state was allocated for another schedule, so that planning
    /// failure is reported as the same incompatibility as any layout mismatch.
    fn validate_for_model(&self, model: &MuseGlimmerForCausalLm<B>) -> Result<()> {
        let incompatible =
            || Error::Cache("Muse-Glimmer decode state is incompatible with the model".into());
        let expected = cache_layout(model, self.layout.window).map_err(|_| incompatible())?;
        if expected != self.layout || self.tensors.main().native_kv_count() != expected.groups.len()
        {
            return Err(incompatible());
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

    /// Returns the group index of the given storage kind, if the layout has one.
    fn group_of(&self, storage: KvCacheStorage) -> Option<usize> {
        self.layout
            .schedule
            .iter()
            .find(|slot| slot.storage() == storage)
            .map(|slot| slot.coordinates().0)
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
    /// Evaluates one masked segment against external KV state in one batched pass.
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
        let Some(layout) = BlockLayout::<B>::new(&prepared, state.tensors.lengths(), &device)?
        else {
            return self.inactive_output(&prepared, options, &device);
        };
        // Masked columns are never validated, so padding entries receive the
        // valid token 0 and position 0 instead of whatever sentinel they hold.
        let tokens = layout.compact_inert(input.token_ids.clone(), 0);
        let positions = match &input.position_ids {
            Some(position_ids) => layout.compact_inert(position_ids.clone(), 0),
            None => layout.implicit_positions(),
        };
        let tables = self.cached_rotary_tables(positions, &prepared)?;
        let mut step = state.tensors.step(&prepared.token_counts)?;
        let full_group = state.group_of(KvCacheStorage::Full);
        let full_plan = full_group
            .map(|group| state.tensors.kv_read_plan_through(&step, group))
            .transpose()?;
        let sliding_group = state.group_of(KvCacheStorage::MirroredSlidingRing);
        let sliding_plan = sliding_group
            .map(|group| state.tensors.kv_read_plan(StepBank::Main, group))
            .transpose()?;
        let mut hidden = self.model.embed(tokens)?;
        let mut hidden_states = Vec::new();
        if options.hidden_states {
            hidden_states.push(layout.scatter(hidden.clone()));
        }
        for (index, layer) in self.model.layers.iter().enumerate() {
            let angles = layer
                .rope
                .as_ref()
                .map(|rope| tables.get(rope))
                .transpose()?;
            let (residual, projection) = layer.attention_projection(hidden, angles, &mut None)?;
            let slot = state.layout.schedule[index];
            let (group, group_layer) = slot.coordinates();
            let segment = match (slot, &full_plan, &sliding_plan) {
                (MuseGlimmerLayerCache::Full { .. }, Some(plan), _) => {
                    state.tensors.write_kv_ahead(
                        &mut step,
                        group,
                        group_layer,
                        projection.key.clone(),
                        projection.value.clone(),
                    )?;
                    state
                        .tensors
                        .read_kv_through(&step, group, group_layer, plan)?
                        .materialize()
                }
                (MuseGlimmerLayerCache::Sliding { .. }, _, Some(plan)) => {
                    let history = state
                        .tensors
                        .main()
                        .native_kv(group)?
                        .read_batch(group_layer, plan)?;
                    let fresh = state.tensors.step_token_segment(
                        &step,
                        projection.key.clone(),
                        projection.value.clone(),
                    )?;
                    step.push_kv_batch(
                        group,
                        group_layer,
                        projection.key.clone(),
                        projection.value.clone(),
                    );
                    history.with_segment(fresh)?.materialize()
                }
                _ => None,
            }
            .ok_or_else(|| Error::Cache(format!("Muse layer {index} read no cached keys")))?;
            let window = layer.window(state.layout.window)?;
            let context = cached_context(layer, &projection, &segment, &layout, slot, window)?;
            let attended = layer
                .self_attn
                .gated_output(context, projection.gate, None)?;
            hidden = layer.finish_block(residual, attended, None)?;
            if options.hidden_states {
                hidden_states.push(layout.scatter(hidden.clone()));
            }
        }
        let hidden = layout.scatter(self.model.norm.forward(hidden));
        if options.hidden_states {
            hidden_states.push(hidden.clone());
        }
        let selected = select_hidden_with_last_columns(
            hidden.clone(),
            prepared.last_columns(),
            &options.logits,
        )?;
        let logits = self.project_logits(selected)?;
        state.tensors.publish_step(step)?;
        Ok(CausalLmOutput {
            logits,
            final_hidden_state: options.hidden_states.then_some(hidden),
            hidden_states,
            attentions: Vec::new(),
            routers: Vec::new(),
            load_balancing_loss: None,
        })
    }

    /// Builds the exact-zero output of a decode step in which every row is finished.
    fn inactive_output(
        &self,
        prepared: &StatefulRows,
        options: &ForwardOptions,
        device: &B::Device,
    ) -> Result<CausalLmOutput<B>> {
        let hidden = Tensor::<B, 3>::zeros(
            [prepared.batch, prepared.sequence, self.config().hidden_size],
            (device, self.model.embed_tokens.weight.val().dtype()),
        );
        let selected = select_hidden_with_last_columns(
            hidden.clone(),
            prepared.last_columns(),
            &options.logits,
        )?;
        Ok(CausalLmOutput {
            logits: self.project_logits(selected)?,
            final_hidden_state: options.hidden_states.then(|| hidden.clone()),
            hidden_states: if options.hidden_states {
                vec![hidden; self.model.layers.len() + 2]
            } else {
                Vec::new()
            },
            attentions: Vec::new(),
            routers: Vec::new(),
            load_balancing_loss: None,
        })
    }

    /// Builds the block's rotary tables, per row only where frequencies differ.
    ///
    /// Static scalings use one table per configuration for every row.
    /// Context-dependent scalings evaluate each row at its own context, as an
    /// independent batch-one call would, and concatenate the tables over rows.
    fn cached_rotary_tables(
        &self,
        positions: Tensor<B, 2, Int>,
        prepared: &StatefulRows,
    ) -> Result<RotaryTables<B>> {
        // Verified: the first active context (at least one), whether every
        // active row shares it, and each row's own context.
        let contexts = rotary_contexts(&prepared.token_counts, &prepared.evaluated_contexts);
        let ropes = self.model.distinct_ropes();
        if contexts.uniform
            || ropes
                .iter()
                .all(|rope| !rope.scaling().is_context_dependent())
        {
            return self.model.rotary_tables(positions, contexts.first);
        }
        let dtype = self.model.embed_tokens.weight.val().dtype();
        let [batch, block] = positions.dims();
        let entries = ropes
            .into_iter()
            .map(|rope| {
                let mut cosine = Vec::with_capacity(batch);
                let mut sine = Vec::with_capacity(batch);
                let mut rotary_dimension = 0;
                for row in 0..batch {
                    let context = contexts.rows[row];
                    let angles = rotary_angles_from_frequencies(
                        &RotaryFrequencies::new(rope, context)?,
                        positions.clone().slice([row..row + 1, 0..block]),
                    )?
                    .to_dtype(dtype);
                    rotary_dimension = angles.rotary_dimension;
                    cosine.push(angles.cosine);
                    sine.push(angles.sine);
                }
                Ok((
                    rope,
                    RotaryAngles {
                        cosine: Tensor::cat(cosine, 0),
                        sine: Tensor::cat(sine, 0),
                        rotary_dimension,
                    },
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(RotaryTables { entries })
    }
}

/// Evaluates one layer's queries against its cached keys in score-bounded chunks.
///
/// `segment` holds slot-ordered keys and values with their positions and
/// validity; invalid slots' values are zeroed first so stale bytes cannot
/// poison attention. Full storage is position-ordered, so each query chunk
/// reads only its verified causal key prefix. A sliding layer's keys are
/// `[ring history | fresh block]`: chunks whose rows can still see history
/// read the whole ring, later chunks only their rows' window union, and the
/// plan is sized for that width (`MuseKeySlicing::RingHistory`) rather than
/// for every key. Returns the head context `[batch, query_heads, block, head_dim]`.
fn cached_context<B: Backend>(
    layer: &super::MuseGlimmerDecoderLayer<B>,
    projection: &MuseAttentionProjection<B>,
    segment: &KvBatchSegment<B>,
    layout: &BlockLayout<B>,
    slot: MuseGlimmerLayerCache,
    window: Option<usize>,
) -> Result<Tensor<B, 4>> {
    let attention = &layer.self_attn;
    let keyed = MuseAttentionProjection {
        query: projection.query.clone(),
        key: segment.keys().clone(),
        value: segment.masked_values(),
        gate: projection.gate.clone(),
    };
    let [batch, queries, keys] = attention.validate_projection(&keyed)?;
    let positional = matches!(slot, MuseGlimmerLayerCache::Full { .. });
    let visibility = layout.visibility(segment, positional).with_window(window);
    let device = keyed.query.device();
    let slicing = match (slot, window) {
        // History width `keys - queries`: the step's fresh block is the last segment.
        (MuseGlimmerLayerCache::Sliding { .. }, Some(window)) => {
            MuseKeySlicing::RingHistory { window }
        }
        _ => MuseKeySlicing::PrefixLimits {
            limits: visibility.key_limits(),
        },
    };
    let chunks = plan_attention_chunks(
        attention.effective_policy(&device),
        batch,
        attention.query_heads,
        queries,
        keys,
        slicing,
    )?;
    debug_assert_eq!(layout.block(), queries);
    attention.chunked_context(&keyed, &chunks, &visibility)
}

/// Validates shapes, devices, masks, positions, and capacity transitions.
///
/// The shared preparation reads the mask, token IDs, and explicit positions in
/// one transfer; this wrapper adds the model-device guard and the cached
/// rotary guard, which applies only once rotated keys are already cached.
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
    // A fresh prefill has no rotated keys to invalidate yet. Verified: the
    // first active row with cached keys whose context passes a rotary
    // configuration's limit, rows first and configurations second.
    let limits = model
        .model
        .distinct_ropes()
        .iter()
        .map(|rope| rope.scaling().cached_context_limit())
        .collect::<Vec<_>>();
    if let Some((context, limit)) = first_uncacheable_context(
        state.tensors.lengths(),
        &rows.token_counts,
        &rows.evaluated_contexts,
        &limits,
    ) {
        return Err(Error::BackendCapability(format!(
            "Muse cached decoding at context {context} exceeds the {limit}-token limit of \
             its context-dependent rotary scaling, which would require re-rotating cached keys"
        )));
    }
    Ok(rows)
}

/// Resolves the two homogeneous groups and dense per-layer schedule.
///
/// The sliding group stores a mirrored ring with the group's own `window`;
/// the full group stores complete history.
fn cache_layout<B: Backend>(
    model: &MuseGlimmerForCausalLm<B>,
    window: Option<usize>,
) -> Result<MuseGlimmerCacheLayout> {
    let global = model
        .config()
        .resolved_layer_types()
        .iter()
        .map(|&kind| kind == MuseGlimmerLayerType::FullAttention)
        .collect::<Vec<_>>();
    // Verified: exact group sizes, a ring (group 0) exactly with local layers
    // and a non-zero window, the full group after it exactly with global
    // layers, and every layer's (existing) group and dense slot.
    let plan = match cache_groups(&global, window) {
        Ok(plan) => plan,
        Err(GroupFault::ZeroWindow) => {
            return Err(Error::configuration(
                "muse_glimmer.decode_state.window",
                "must be non-zero",
            ));
        }
        Err(GroupFault::MissingWindow) => {
            return Err(Error::configuration(
                "muse_glimmer.decode_state.window",
                "is required by the sliding cache group",
            ));
        }
    };
    let dtype = model.model.embed_tokens.weight.val().dtype();
    let mut groups = Vec::with_capacity(2);
    if let Some(width) = plan.window.and_then(NonZeroUsize::new) {
        groups.push(
            NativeKvCacheSpec::new(
                plan.sliding_layers,
                model.config().num_key_value_heads,
                model.config().head_dim,
                KvCacheStorage::MirroredSlidingRing,
                dtype,
            )?
            .with_sliding_window(width),
        );
    }
    if plan.full_group.is_some() {
        groups.push(NativeKvCacheSpec::new(
            plan.full_layers,
            model.config().num_key_value_heads,
            model.config().head_dim,
            KvCacheStorage::Full,
            dtype,
        )?);
    }
    let schedule = global
        .iter()
        .zip(plan.groups.iter().zip(&plan.slots))
        .map(|(&is_global, (&group, &layer))| {
            if is_global {
                MuseGlimmerLayerCache::Full { group, layer }
            } else {
                MuseGlimmerLayerCache::Sliding { group, layer }
            }
        })
        .collect();
    Ok(MuseGlimmerCacheLayout {
        schedule,
        groups,
        window: plan.window,
    })
}

#[cfg(test)]
#[path = "decode_tests.rs"]
mod tests;
