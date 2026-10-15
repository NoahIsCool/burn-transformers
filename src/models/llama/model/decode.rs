//! Batched cached prefill and decode: one model pass over the whole batch per call.
//!
//! # Cache layout
//!
//! Keys and values live in at most two native-KV groups of one
//! [`TensorDecodeState`], laid out `[batch, K, time, h]`:
//!
//! * a **full** group (time = capacity) for global layers, and for sliding
//!   layers whose window `W` is at least the capacity (such a window can
//!   never exclude a key);
//! * a **ring** group (time = `W`) for sliding layers with `W` below the
//!   capacity, which then attend over at most `W` cached keys per step.
//!
//! Layers are numbered inside their group by the verified `cache_slots`.
//!
//! # One step
//!
//! [`LlamaForCausalLm::prefill`](crate::CausalLanguageModel::prefill) and
//! [`decode`](crate::CausalLanguageModel::decode) validate the call with one
//! device-to-host read (`prepare_stateful_rows`), plan the block
//! (`super::step`), then run every block once over `[batch, Q]`:
//!
//! 1. full layers write their new keys and values *ahead* into slots
//!    `[L_b, L_b + n_b)` and attend over one zero-copy view of `[0, max_b
//!    (L_b + n_b))` read through a plan built once per step;
//! 2. ring layers read their history through a per-step plan, append the
//!    step's own keys as a second segment, and record them in the step;
//! 3. one exclusion plane per group and step (key positions are shared by
//!    the group's layers) applies causality `j <= p`, the window
//!    `p - W < j`, and key validity, by replacement; invalid cached values
//!    are zeroed so stale non-finite bytes cannot leak (`0 · NaN`);
//! 4. after the forward succeeds, the step is published atomically: lengths
//!    advance and ring writes land, or nothing changes. A failure leaves the
//!    committed state untouched (write-ahead slots beyond the lengths are
//!    unreachable).
//!
//! Prefill rejects rows without valid tokens; decode accepts them as no-ops
//! with exact zero outputs. The state allocates no speculative staging, so
//! no transaction can begin on it; speculative decoding rolls back with
//! [`truncate`](crate::DecodeState::truncate) instead (ring groups only
//! within their window).
//!
//! # Positions
//!
//! Cache positions are valid-token ordinals continued from `L_b`, so padding
//! never consumes a position. Implicit rotary positions equal them; explicit
//! `position_ids` rotate instead, while masks always use cache positions (as
//! the reference uses `cache_position` for masks and `position_ids` for
//! RoPE). Context-dependent RoPE (dynamic NTK, LongRoPE) is accepted only
//! while every row stays within the original context, because cached keys
//! cannot be re-rotated.

use core::num::NonZeroUsize;

use burn::tensor::{Bool, DType, Int, Tensor, backend::Backend};

use crate::{
    CausalLanguageModel, CausalLmInput, CausalLmOutput, DecodeInput, Error, ForwardOptions,
    LogitsSelection, Result,
    architecture::DecodeState as ArchitectureDecodeState,
    attention::{AttentionProjection, GroupedQueryAttention},
    cache::{
        CacheConfig, KvBatchReadPlan, KvBatchSegment, KvCacheStorage, NativeKvCacheSpec, StepBank,
        TensorDecodeState, TensorStepWrites, TokenAlignment,
    },
    intervention::{AttentionPoints, Hooks},
    models::shared::{
        EmptyRows, StatefulContract, StatefulRows, prepare_stateful_rows,
        select_hidden_with_last_columns, validate_logits_selection,
    },
    position::RotaryAngles,
};

use super::super::{LlamaArchitecture, LlamaFamilyConfig, plan::cache_slots};
use super::{INPUT_NAMES, LayerKeys, LlamaForCausalLm, step::StepLayout};

/// Owner named when an input lives on another device than the cache.
const DECODE_STATE: &str = "Llama-family decode state";

// Cache layout.

/// Where one layer's keys and values live.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CacheSlot {
    /// The layer lives in the full-storage group.
    Full(
        /// Layer index inside the group.
        usize,
    ),
    /// The layer lives in the sliding-ring group.
    Ring(
        /// Layer index inside the group.
        usize,
    ),
}

/// One native-KV group: its index in the bank and its specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CacheGroup {
    /// Group index in the main bank.
    index: usize,
    /// Allocated geometry.
    spec: NativeKvCacheSpec,
}

/// The cache groups of a model under one cache configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
struct CacheLayout {
    /// Slot of every decoder layer, in model order.
    slots: Vec<CacheSlot>,
    /// Full-storage group, when any layer uses it.
    full: Option<CacheGroup>,
    /// Sliding-ring group, when any layer uses it.
    ring: Option<CacheGroup>,
    /// Window of the ring group.
    window: Option<NonZeroUsize>,
}

impl CacheLayout {
    /// Derives the layout of `architecture` for `capacity` tokens stored in `dtype`.
    fn plan(architecture: &LlamaArchitecture, capacity: usize, dtype: DType) -> Result<Self> {
        let ring_window = architecture
            .sliding_window()
            .filter(|&window| window < capacity)
            .and_then(NonZeroUsize::new);
        let in_ring: Vec<bool> = architecture
            .layers
            .iter()
            .map(|layer| ring_window.is_some() && layer.sliding_window().is_some())
            .collect();
        let numbering = cache_slots(&in_ring);
        let spec = |layers, storage| {
            NativeKvCacheSpec::new(
                layers,
                architecture.key_value_heads,
                architecture.head_dim,
                storage,
                dtype,
            )
        };
        let full = (numbering.full_layers > 0)
            .then(|| spec(numbering.full_layers, KvCacheStorage::Full))
            .transpose()?
            .map(|spec| CacheGroup { index: 0, spec });
        let ring = match ring_window {
            Some(window) if numbering.ring_layers > 0 => Some(CacheGroup {
                index: usize::from(full.is_some()),
                spec: spec(numbering.ring_layers, KvCacheStorage::SlidingRing)?
                    .with_sliding_window(window),
            }),
            _ => None,
        };
        let slots = in_ring
            .iter()
            .zip(&numbering.slots)
            .map(|(&ring, &slot)| {
                if ring {
                    CacheSlot::Ring(slot)
                } else {
                    CacheSlot::Full(slot)
                }
            })
            .collect();
        Ok(Self {
            slots,
            full,
            ring,
            window: ring_window,
        })
    }

    /// Allocates the groups of this layout in a fresh decode state.
    fn allocate<B: Backend>(
        &self,
        config: CacheConfig,
        batch_size: usize,
        device: &B::Device,
    ) -> Result<TensorDecodeState<B>> {
        let mut tensors = TensorDecodeState::new(config, batch_size, device)?;
        for group in [self.full, self.ring].into_iter().flatten() {
            let index = tensors.main_mut().add_native_kv(group.spec)?;
            if index != group.index {
                return Err(Error::Cache(format!(
                    "Llama-family KV group must be {}, found {index}",
                    group.index
                )));
            }
        }
        Ok(tensors)
    }
}

// The decode state.

/// External KV state of a Llama-family model.
#[derive(Debug)]
pub struct LlamaDecodeState<B: Backend> {
    /// Cache tensors and per-row committed lengths.
    tensors: TensorDecodeState<B>,
    /// Group assignment the tensors were allocated for.
    layout: CacheLayout,
}

impl<B: Backend> LlamaDecodeState<B> {
    /// Allocates the cache groups of `model` for `batch_size` rows.
    fn new(
        model: &LlamaForCausalLm<B>,
        config: CacheConfig,
        batch_size: usize,
        device: &B::Device,
    ) -> Result<Self> {
        let architecture = model.architecture();
        let capacity = config.capacity();
        if capacity > architecture.max_position_embeddings || capacity > i32::MAX as usize {
            return Err(Error::configuration(
                "llama.decode_state.capacity",
                format!(
                    "{capacity} exceeds max_position_embeddings {} or Burn's portable i32 index range",
                    architecture.max_position_embeddings
                ),
            ));
        }
        let layout = CacheLayout::plan(architecture, capacity, model.model.activation_dtype())?;
        let tensors = layout.allocate(config, batch_size, device)?;
        Ok(Self { tensors, layout })
    }

    /// Borrows the generic cache state for diagnostics and lifecycle integration.
    #[must_use]
    pub const fn tensor_state(&self) -> &TensorDecodeState<B> {
        &self.tensors
    }

    /// Returns how many layers live in a sliding ring rather than full storage.
    #[must_use]
    pub fn ring_layers(&self) -> usize {
        self.layout
            .slots
            .iter()
            .filter(|slot| matches!(slot, CacheSlot::Ring(_)))
            .count()
    }

    /// Checks that this state was allocated for `model` and is unchanged.
    fn validate_for(&self, model: &LlamaForCausalLm<B>) -> Result<()> {
        let expected = CacheLayout::plan(
            model.architecture(),
            self.tensors.capacity(),
            model.model.activation_dtype(),
        )?;
        let groups = [expected.full, expected.ring].into_iter().flatten();
        let intact = groups.clone().count() == self.tensors.main().native_kv_count()
            && groups.into_iter().all(|group| {
                self.tensors
                    .main()
                    .native_kv(group.index)
                    .is_ok_and(|cache| cache.spec() == group.spec)
            });
        if expected != self.layout || !intact {
            return Err(Error::Cache(
                "Llama-family decode state was allocated for a different model".into(),
            ));
        }
        Ok(())
    }
}

impl<B: Backend> ArchitectureDecodeState<B> for LlamaDecodeState<B> {
    /// Returns the number of rows.
    fn batch_size(&self) -> usize {
        self.tensors.batch_size()
    }

    /// Returns the logical capacity per row.
    fn capacity(&self) -> usize {
        self.tensors.capacity()
    }

    /// Returns each row's committed token count.
    fn lengths(&self) -> &[usize] {
        self.tensors.lengths()
    }

    /// Clears every row and cache group.
    fn reset(&mut self) -> Result<()> {
        self.tensors.reset().map_err(Into::into)
    }

    /// Reorders or duplicates rows across every group atomically.
    fn reorder(&mut self, indices: &[usize]) -> Result<()> {
        self.tensors.reorder(indices).map_err(Into::into)
    }

    /// Rolls rows back to shorter lengths (ring groups only within their window).
    fn truncate(&mut self, lengths: &[usize]) -> Result<()> {
        self.tensors.truncate(lengths).map_err(Into::into)
    }

    /// Returns the device holding the cache.
    fn device(&self) -> &B::Device {
        self.tensors.device()
    }
}

// The cached step's keys.

/// Keys of one batched step: cache groups, per-step plans, and shared masks.
struct CachedKeys<'s, B: Backend> {
    /// The state being extended; only this step writes it.
    state: &'s mut TensorDecodeState<B>,
    /// The step's pending writes.
    step: TensorStepWrites<B>,
    /// Layer-to-group assignment.
    layout: &'s CacheLayout,
    /// Rotary tables of the block, when any layer rotates.
    angles: Option<RotaryAngles<B>>,
    /// `[batch, Q]` cache position of every block slot.
    query_positions: Tensor<B, 2, Int>,
    /// Query heads `H`, the exclusion plane's head extent.
    heads: usize,
    /// Read plan of the full group through the step's write-ahead.
    full_plan: Option<KvBatchReadPlan<B>>,
    /// Read plan of the ring group's committed history.
    ring_plan: Option<KvBatchReadPlan<B>>,
    /// Exclusions of the full group, built by its first layer.
    full_mask: Option<Tensor<B, 4, Bool>>,
    /// Exclusions of the ring group, built by its first layer.
    ring_mask: Option<Tensor<B, 4, Bool>>,
}

impl<'s, B: Backend> CachedKeys<'s, B> {
    /// Starts a step appending `layout`'s block to `state`.
    fn new(
        state: &'s mut LlamaDecodeState<B>,
        block: &StepLayout,
        angles: Option<RotaryAngles<B>>,
        query_positions: Tensor<B, 2, Int>,
        heads: usize,
    ) -> Result<Self> {
        let LlamaDecodeState { tensors, layout } = state;
        let step = tensors
            .step(block.counts())?
            .with_alignment(TokenAlignment::Columns(block.columns().to_vec()));
        let full_plan = layout
            .full
            .map(|group| tensors.kv_read_plan_through(&step, group.index))
            .transpose()?;
        let ring_plan = layout
            .ring
            .map(|group| tensors.kv_read_plan(StepBank::Main, group.index))
            .transpose()?;
        Ok(Self {
            state: tensors,
            step,
            layout,
            angles,
            query_positions,
            heads,
            full_plan,
            ring_plan,
            full_mask: None,
            ring_mask: None,
        })
    }

    /// Returns every visible key of a full-group layer, writing the step's keys ahead.
    fn full_segment(
        &mut self,
        slot: usize,
        projection: &AttentionProjection<B>,
    ) -> Result<KvBatchSegment<B>> {
        let (group, plan) = self
            .layout
            .full
            .zip(self.full_plan.as_ref())
            .ok_or_else(|| Error::Cache("Llama-family full KV group is missing".into()))?;
        self.state.write_kv_ahead(
            &mut self.step,
            group.index,
            slot,
            projection.key.clone(),
            projection.value.clone(),
        )?;
        let read = self
            .state
            .read_kv_through(&self.step, group.index, slot, plan)?;
        read.materialize()
            .ok_or_else(|| Error::Cache("Llama-family full KV read returned no keys".into()))
    }

    /// Returns a ring layer's history followed by the step's keys, recording them.
    fn ring_segment(
        &mut self,
        slot: usize,
        projection: &AttentionProjection<B>,
    ) -> Result<KvBatchSegment<B>> {
        let (group, plan) = self
            .layout
            .ring
            .zip(self.ring_plan.as_ref())
            .ok_or_else(|| Error::Cache("Llama-family sliding KV group is missing".into()))?;
        let history = self
            .state
            .main()
            .native_kv(group.index)?
            .read_batch(slot, plan)?;
        let fresh = self.state.step_token_segment(
            &self.step,
            projection.key.clone(),
            projection.value.clone(),
        )?;
        self.step.push_kv_batch(
            group.index,
            slot,
            projection.key.clone(),
            projection.value.clone(),
        );
        history
            .with_segment(fresh)?
            .materialize()
            .ok_or_else(|| Error::Cache("Llama-family sliding KV read returned no keys".into()))
    }

    /// Returns the group's shared exclusion plane, building it on first use.
    fn exclusions(
        &mut self,
        ring: bool,
        segment: &KvBatchSegment<B>,
    ) -> Result<Tensor<B, 4, Bool>> {
        let (cached, window) = if ring {
            (&mut self.ring_mask, self.layout.window)
        } else {
            (&mut self.full_mask, None)
        };
        if let Some(mask) = cached {
            return Ok(mask.clone());
        }
        let mask = segment.exclusion_mask(&self.query_positions, self.heads, window)?;
        *cached = Some(mask.clone());
        Ok(mask)
    }

    /// Ends the borrow of the state and returns the step for publication.
    fn into_step(self) -> TensorStepWrites<B> {
        self.step
    }
}

impl<B: Backend> LayerKeys<B> for CachedKeys<'_, B> {
    /// Returns the block's rotary tables.
    fn angles(&self) -> Option<&RotaryAngles<B>> {
        self.angles.as_ref()
    }

    /// Attends over the layer's cached history plus the step's own keys.
    fn attend(
        &mut self,
        layer: usize,
        attention: &GroupedQueryAttention<B>,
        mut projection: AttentionProjection<B>,
        points: AttentionPoints,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<Tensor<B, 3>> {
        let slot = self.layout.slots.get(layer).copied().ok_or_else(|| {
            Error::Cache(format!(
                "Llama-family decode state has no slot for layer {layer}"
            ))
        })?;
        let (segment, ring) = match slot {
            CacheSlot::Full(slot) => (self.full_segment(slot, &projection)?, false),
            CacheSlot::Ring(slot) => (self.ring_segment(slot, &projection)?, true),
        };
        let mask = self.exclusions(ring, &segment)?;
        projection.key = segment.keys().clone();
        projection.value = segment.masked_values();
        attention.attend_with_hooks(projection, Some(mask), None, false, points, hooks)
    }
}

// The cached forward pass.

/// A stateful call: prefill rejects empty rows, decode accepts them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    /// Prompt processing.
    Prefill,
    /// Continuation tokens; fully masked rows are finished sequences.
    Decode,
}

impl<B: Backend> LlamaForCausalLm<B> {
    /// Runs one batched step of `input` against `state` and publishes it.
    fn forward_cached(
        &self,
        input: CausalLmInput<B>,
        state: &mut LlamaDecodeState<B>,
        options: &ForwardOptions,
        phase: Phase,
    ) -> Result<CausalLmOutput<B>> {
        if options.attentions {
            return Err(Error::BackendCapability(
                "Llama-family cached passes do not capture attention probabilities; use the stateless forward with eager attention".into(),
            ));
        }
        state.validate_for(self)?;
        let rows = self.prepare_rows(&input, state, phase)?;
        validate_logits_selection(&options.logits, rows.sequence)?;
        let context = self.cached_rotary_context(&rows)?;
        if rows.token_counts.iter().all(|&count| count == 0) {
            return self.inactive_output(&rows, options);
        }
        let layout = StepLayout::plan(&rows, state.tensors.lengths())?;
        let device = input.token_ids.device();
        let query_positions = layout.slot_positions::<B>(&device)?;
        let rotary_positions =
            layout.rotary_positions(input.position_ids.as_ref(), &query_positions)?;
        let angles = self.model.rotary_tables(rotary_positions, context)?;
        let tokens = layout.cut(&input.token_ids)?;
        let heads = self.architecture().query_heads;
        let mut keys = CachedKeys::new(state, &layout, angles, query_positions, heads)?;
        let mut hidden = self.model.embed_tokens.forward(tokens);
        let mut block_states = Vec::new();
        for (index, layer) in self.model.layers.iter().enumerate() {
            if options.hidden_states {
                block_states.push(hidden.clone());
            }
            hidden = layer.forward(hidden, index, &mut keys, &mut Hooks::new())?;
        }
        let step = keys.into_step();
        if options.hidden_states {
            block_states.push(hidden.clone());
        }
        let hidden = self.model.norm.forward(hidden);
        let output = self.cached_output(&layout, &rows, options, hidden, block_states)?;
        state.tensors.publish_step(step)?;
        Ok(output)
    }

    /// Validates the call against the state's cursors with one device read.
    fn prepare_rows(
        &self,
        input: &CausalLmInput<B>,
        state: &LlamaDecodeState<B>,
        phase: Phase,
    ) -> Result<StatefulRows> {
        let architecture = self.architecture();
        prepare_stateful_rows(
            input,
            state.tensors.device(),
            state.tensors.lengths(),
            state.tensors.capacity(),
            &StatefulContract {
                names: INPUT_NAMES,
                owner: DECODE_STATE,
                vocabulary: architecture.vocab_size,
                position_limit: Some(architecture.max_position_embeddings),
                empty_rows: match phase {
                    Phase::Prefill => EmptyRows::Reject,
                    Phase::Decode => EmptyRows::Permit,
                },
            },
        )
    }

    /// Returns the step's rotary context, rejecting one cached keys cannot follow.
    fn cached_rotary_context(&self, rows: &StatefulRows) -> Result<usize> {
        let context = rows
            .active_contexts()
            .map(|(_, context)| context)
            .max()
            .unwrap_or(1);
        let scaling = self.architecture().rope.scaling();
        match scaling.cached_context_limit() {
            Some(limit) if context > limit => Err(Error::BackendCapability(format!(
                "Llama-family cached decoding with context-dependent RoPE ({scaling:?}) is limited to the original context {limit}; this call reaches {context}"
            ))),
            _ => Ok(context),
        }
    }

    /// Returns exact zeros for a call in which no row appends a token.
    fn inactive_output(
        &self,
        rows: &StatefulRows,
        options: &ForwardOptions,
    ) -> Result<CausalLmOutput<B>> {
        let architecture = self.architecture();
        let selected = match &options.logits {
            LogitsSelection::All => rows.sequence,
            LogitsSelection::Last => 1,
            LogitsSelection::Positions(positions) => positions.len(),
        };
        let device = self.model.embed_tokens.weight.lazy_device();
        let dtype = self.model.activation_dtype();
        let zeros = |tokens, width| Tensor::zeros([rows.batch, tokens, width], (&device, dtype));
        let mut output = CausalLmOutput::from_logits(zeros(selected, architecture.vocab_size));
        if options.hidden_states {
            let states = zeros(rows.sequence, architecture.hidden_size);
            output.final_hidden_state = Some(states.clone());
            output.hidden_states = vec![states; architecture.num_layers() + 2];
        }
        Ok(output)
    }

    /// Restores requested states to the caller's columns and projects logits.
    fn cached_output(
        &self,
        layout: &StepLayout,
        rows: &StatefulRows,
        options: &ForwardOptions,
        hidden: Tensor<B, 3>,
        block_states: Vec<Tensor<B, 3>>,
    ) -> Result<CausalLmOutput<B>> {
        let selected = match &options.logits {
            LogitsSelection::Last => layout.last_states(hidden.clone())?,
            selection => select_hidden_with_last_columns(
                layout.restore(hidden.clone())?,
                rows.last_columns(),
                selection,
            )?,
        };
        let mut output = CausalLmOutput::from_logits(self.project_logits(selected)?);
        if options.hidden_states {
            let mut states = block_states
                .into_iter()
                .map(|states| layout.restore(states))
                .collect::<Result<Vec<_>>>()?;
            let final_states = layout.restore(hidden)?;
            states.push(final_states.clone());
            output.final_hidden_state = Some(final_states);
            output.hidden_states = states;
        }
        Ok(output)
    }
}

impl<B: Backend> CausalLanguageModel<B> for LlamaForCausalLm<B> {
    /// The serialized checkpoint configuration.
    type Config = LlamaFamilyConfig;
    /// External KV state.
    type State = LlamaDecodeState<B>;

    /// Delegates to the stateless forward pass.
    fn forward(
        &self,
        input: CausalLmInput<B>,
        options: &ForwardOptions,
    ) -> Result<CausalLmOutput<B>> {
        LlamaForCausalLm::forward(self, input, options)
    }

    /// Allocates full and sliding-ring KV groups for `batch_size` rows.
    fn new_decode_state(
        &self,
        config: CacheConfig,
        batch_size: usize,
        device: &B::Device,
    ) -> Result<Self::State> {
        LlamaDecodeState::new(self, config, batch_size, device)
    }

    /// Appends prompt rows (each with at least one valid token) in one batched step.
    fn prefill(
        &self,
        input: CausalLmInput<B>,
        state: &mut Self::State,
        options: &ForwardOptions,
    ) -> Result<CausalLmOutput<B>> {
        self.forward_cached(input, state, options, Phase::Prefill)
    }

    /// Appends continuation tokens in one batched step; fully masked rows are no-ops.
    fn decode(
        &self,
        input: DecodeInput<B>,
        state: &mut Self::State,
        options: &ForwardOptions,
    ) -> Result<CausalLmOutput<B>> {
        let input = CausalLmInput {
            token_ids: input.token_ids,
            token_mask: input.token_mask,
            position_ids: input.position_ids,
        };
        self.forward_cached(input, state, options, Phase::Decode)
    }
}

#[cfg(test)]
#[path = "decode_tests.rs"]
mod tests;
