//! Batched cached prefill and decode: one model pass over the whole batch per call.
//!
//! # Cache layout
//!
//! Keys and values live in at most two native-KV groups of one
//! [`TensorDecodeState`], laid out `[batch, K, time, h]`:
//!
//! * a **full** group (time = capacity) for `full_attention` layers, and for
//!   sliding layers whose window `W` is at least the capacity (such a window
//!   can never exclude a key);
//! * a **ring** group (time = `W`) for sliding layers with `W` below the
//!   capacity: gpt-oss-20b's twelve 128-token sliding layers then hold 128
//!   keys each instead of the whole context.
//!
//! Layers are numbered inside their group by the verified `cache_slots`.
//!
//! # One step
//!
//! [`prefill`](crate::CausalLanguageModel::prefill) and
//! [`decode`](crate::CausalLanguageModel::decode) validate the call with one
//! device-to-host read (`prepare_stateful_rows`), lay the batch out as one
//! right-padded `[batch, N]` block with the shared [`CompactBatch`] (row `b`'s
//! `n_b` new tokens in slots `0..n_b` at cache positions `L_b + i`), then run
//! every block once over the whole block:
//!
//! 1. full layers write their new keys and values *ahead* into slots
//!    `[L_b, L_b + n_b)` and attend over one view of `[0, max_b (L_b + n_b))`
//!    read through a plan built once per step;
//! 2. ring layers read their history through a per-step plan, append the
//!    step's own keys as a second segment, and record them in the step;
//! 3. one exclusion plane per group and step applies causality `j <= p`, the
//!    window `p - W < j`, and key validity. When every row starts at the same
//!    length and fills the block, a full read is exactly bottom-right causal,
//!    which the attention call applies itself (one-token decode: no mask).
//!    Excluded keys get probability zero (see "Clean tails" for when their
//!    values are zeroed), and the learned sinks keep every row, even a
//!    padding slot's, a finite softmax;
//! 4. the routed experts of every block dispatch all rows' tokens together;
//! 5. after the vocabulary projection succeeds, the step is published
//!    atomically: lengths advance and ring writes land, or nothing changes. A
//!    failure leaves the committed state untouched (write-ahead slots beyond
//!    the lengths are unreachable).
//!
//! # Clean tails
//!
//! Invalid keys in a full-storage read are the slots at or beyond each row's
//! end. The state tracks whether every such slot holds zero (a *clean
//! tail*): true after allocation and `reset` (which zero every tensor),
//! preserved by published steps (they fill exactly the slots they commit),
//! by `reorder` (rows move with their lengths), and by truncation to zero.
//! Truncating a row to a shorter non-zero length, or a step that fails after
//! writing ahead, leaves bytes beyond the row's end (possibly non-finite, an
//! overflowed F16 draft, say), so the flag clears and full layers zero
//! invalid values before attending (`0 · NaN` would poison the row). While
//! tails are clean that masked copy of every full layer's values (one
//! `[batch, K, T, h]` select per layer and step) is skipped: on Flex, a
//! ragged 8-row decode over about 1000 cached keys (width 256, four layers)
//! took 7.5–8.1 ms per step instead of 11.8–12.9 ms. Ring slots cycle
//! through a row's own older tokens, so ring layers always zero invalid
//! values.
//!
//! Prefill rejects rows without valid tokens; decode accepts them as no-ops
//! with exact zero outputs. Speculative transactions are not supported
//! (ring groups have no staging); roll back with
//! [`truncate`](crate::DecodeState::truncate) instead, ring groups only
//! within their window.
//!
//! # Positions
//!
//! Cache positions are valid-token ordinals continued from `L_b`, so padding
//! never consumes a position. Implicit rotary positions equal them; explicit
//! `position_ids` rotate instead, while masks always use cache positions (as
//! the reference uses `cache_position` for masks and `position_ids` for
//! RoPE). gpt-oss's YaRN frequencies do not depend on the context, so the
//! cache can grow to the model's full context.

use core::num::NonZeroUsize;

use burn::tensor::{Bool, DType, Tensor, backend::Backend};

use crate::{
    CausalLanguageModel, CausalLmInput, CausalLmOutput, DecodeInput, Error, ForwardOptions,
    LogitsSelection, Result,
    architecture::DecodeState as ArchitectureDecodeState,
    attention::{AttentionConnectivity, AttentionProjection, GroupedQueryAttention},
    cache::{
        CacheConfig, KvBatchReadPlan, KvBatchSegment, KvCacheStorage, NativeKvCacheSpec, StepBank,
        TensorCacheError, TensorDecodeState, TensorStepWrites,
    },
    intervention::{AttentionPoints, Hooks},
    models::{
        qwen3_5::model::batch::{CompactBatch, ensure_cacheable_rotary_context},
        shared::{
            EmptyRows, StatefulContract, StatefulRows, ensure_same_device, prepare_stateful_rows,
            validate_logits_selection,
        },
    },
    position::RotaryAngles,
};

use super::super::{GptOssArchitecture, GptOssConfig, plan::ring_assignment};
use super::{GptOssForCausalLm, INPUT_NAMES, LayerKeys};
use crate::models::shared::decoder_plan::{
    cache_slots, group_indices, truncation_leaves_bytes, widest_context,
};

/// Owner named when an input lives on another device than the cache.
const DECODE_STATE: &str = "gpt-oss decode state";

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
    ///
    /// The ring assignment, slot numbering, and group indices are the
    /// Verus-verified `ring_assignment`, `cache_slots`, and `group_indices`.
    fn plan(architecture: &GptOssArchitecture, capacity: usize, dtype: DType) -> Result<Self> {
        let sliding: Vec<bool> = architecture
            .layers
            .iter()
            .map(|layer| *layer != AttentionConnectivity::Global)
            .collect();
        let assignment = ring_assignment(&sliding, architecture.sliding_window(), capacity);
        let ring_window = NonZeroUsize::new(assignment.window);
        let in_ring = assignment.in_ring;
        let numbering = cache_slots(&in_ring);
        let groups = group_indices(numbering.full_layers, numbering.ring_layers);
        let spec = |layers, storage| {
            NativeKvCacheSpec::new(
                layers,
                architecture.key_value_heads,
                architecture.head_dim,
                storage,
                dtype,
            )
        };
        let full = groups
            .full
            .map(|index| {
                spec(numbering.full_layers, KvCacheStorage::Full)
                    .map(|spec| CacheGroup { index, spec })
            })
            .transpose()?;
        let ring = match (ring_window, groups.ring) {
            (Some(window), Some(index)) => Some(CacheGroup {
                index,
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
                    "gpt-oss KV group must be {}, found {index}",
                    group.index
                )));
            }
        }
        Ok(tensors)
    }
}

// The decode state.

/// External KV state of a gpt-oss model: full storage plus a sliding ring.
#[derive(Debug)]
pub struct GptOssDecodeState<B: Backend> {
    /// Cache tensors and per-row committed lengths.
    tensors: TensorDecodeState<B>,
    /// Group assignment the tensors were allocated for.
    layout: CacheLayout,
    /// Whether every full-storage slot at or beyond a row's length is zero.
    clean_tail: bool,
}

impl<B: Backend> GptOssDecodeState<B> {
    /// Allocates the cache groups of `model` for `batch_size` rows.
    fn new(
        model: &GptOssForCausalLm<B>,
        config: CacheConfig,
        batch_size: usize,
        device: &B::Device,
    ) -> Result<Self> {
        let architecture = model.architecture();
        let capacity = config.capacity();
        if capacity > architecture.max_position_embeddings || capacity > i32::MAX as usize {
            return Err(Error::configuration(
                "gpt_oss.decode_state.capacity",
                format!(
                    "{capacity} exceeds max_position_embeddings {} or Burn's portable i32 index range",
                    architecture.max_position_embeddings
                ),
            ));
        }
        let layout = CacheLayout::plan(architecture, capacity, model.model.activation_dtype())?;
        let tensors = layout.allocate(config, batch_size, device)?;
        Ok(Self {
            tensors,
            layout,
            clean_tail: true,
        })
    }

    /// Borrows the generic cache state for diagnostics and lifecycle integration.
    #[must_use]
    pub const fn tensor_state(&self) -> &TensorDecodeState<B> {
        &self.tensors
    }

    /// Returns how many layers live in the sliding ring rather than full storage.
    #[must_use]
    pub fn ring_layers(&self) -> usize {
        self.layout
            .slots
            .iter()
            .filter(|slot| matches!(slot, CacheSlot::Ring(_)))
            .count()
    }

    /// Checks that this state was allocated for `model` and is unchanged.
    fn validate_for(&self, model: &GptOssForCausalLm<B>) -> Result<()> {
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
                "gpt-oss decode state was allocated for a different model".into(),
            ));
        }
        Ok(())
    }
}

impl<B: Backend> ArchitectureDecodeState<B> for GptOssDecodeState<B> {
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

    /// Clears every row and zeroes every cache group, restoring clean tails.
    fn reset(&mut self) -> Result<()> {
        self.tensors.reset()?;
        self.clean_tail = true;
        Ok(())
    }

    /// Reorders or duplicates rows across every group atomically.
    fn reorder(&mut self, indices: &[usize]) -> Result<()> {
        self.tensors.reorder(indices).map_err(Into::into)
    }

    /// Rolls rows back to shorter lengths (the ring group only within its window).
    ///
    /// A row shortened to a non-zero length keeps its later bytes beyond the
    /// new end, so tails are no longer known to be clean.
    fn truncate(&mut self, lengths: &[usize]) -> Result<()> {
        let leaves_bytes = truncation_leaves_bytes(self.tensors.lengths(), lengths);
        self.tensors.truncate(lengths)?;
        self.clean_tail &= !leaves_bytes;
        Ok(())
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
    /// The step's compact block: slot validity and cache positions.
    batch: &'s CompactBatch<B>,
    /// Rotary tables of the block.
    angles: RotaryAngles<B>,
    /// Query heads `H`, the exclusion plane's head extent.
    heads: usize,
    /// Whether invalid full-storage slots are known to hold zero.
    clean_tail: bool,
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
    /// Starts a step appending the compact `batch` to `state`.
    ///
    /// `clean_tail` is the state's flag at the step's start; the caller
    /// clears the state's own flag for the step's duration, because from the
    /// first write-ahead until publication a failure would abandon bytes
    /// beyond the lengths.
    fn new(
        state: &'s mut TensorDecodeState<B>,
        layout: &'s CacheLayout,
        batch: &'s CompactBatch<B>,
        angles: RotaryAngles<B>,
        heads: usize,
        clean_tail: bool,
    ) -> Result<Self> {
        let step = state.step(batch.counts())?;
        let full_plan = layout
            .full
            .map(|group| state.kv_read_plan_through(&step, group.index))
            .transpose()?;
        let ring_plan = layout
            .ring
            .map(|group| state.kv_read_plan(StepBank::Main, group.index))
            .transpose()?;
        Ok(Self {
            state,
            step,
            layout,
            batch,
            angles,
            heads,
            clean_tail,
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
            .ok_or_else(|| Error::Cache("gpt-oss full KV group is missing".into()))?;
        self.state.write_kv_ahead(
            &mut self.step,
            group.index,
            slot,
            projection.key.clone(),
            projection.value.clone(),
        )?;
        self.state
            .read_kv_through(&self.step, group.index, slot, plan)?
            .materialize()
            .ok_or_else(|| Error::Cache("gpt-oss full KV read returned no keys".into()))
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
            .ok_or_else(|| Error::Cache("gpt-oss sliding KV group is missing".into()))?;
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
            .ok_or_else(|| Error::Cache("gpt-oss sliding KV read returned no keys".into()))
    }

    /// Returns the group's exclusions and whether the attention call adds causality.
    ///
    /// A full-storage read is chronological; when every key is valid and the
    /// batch is uniform (one shared length, no padding slot), bottom-right
    /// causality alone is exact. Otherwise the group's shared plane is built
    /// on first use and reused by its other layers.
    fn visibility(
        &mut self,
        ring: bool,
        segment: &KvBatchSegment<B>,
    ) -> Result<(Option<Tensor<B, 4, Bool>>, bool)> {
        if !ring && self.batch.is_uniform() && segment.key_valid().is_none() {
            return Ok((None, true));
        }
        let (cached, window) = if ring {
            (&mut self.ring_mask, self.layout.window)
        } else {
            (&mut self.full_mask, None)
        };
        if let Some(mask) = cached {
            return Ok((Some(mask.clone()), false));
        }
        let mask = segment.exclusion_mask(self.batch.cache_positions(), self.heads, window)?;
        *cached = Some(mask.clone());
        Ok((Some(mask), false))
    }
}

impl<B: Backend> LayerKeys<B> for CachedKeys<'_, B> {
    /// Returns the block's rotary tables.
    fn angles(&self) -> &RotaryAngles<B> {
        &self.angles
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
                "gpt-oss decode state has no slot for layer {layer}"
            ))
        })?;
        let (segment, ring) = match slot {
            CacheSlot::Full(slot) => (self.full_segment(slot, &projection)?, false),
            CacheSlot::Ring(slot) => (self.ring_segment(slot, &projection)?, true),
        };
        let (mask, causal) = self.visibility(ring, &segment)?;
        projection.key = segment.keys().clone();
        projection.value = if ring || !self.clean_tail {
            segment.masked_values()
        } else {
            segment.values().clone()
        };
        attention.attend_with_hooks(projection, mask, None, causal, points, hooks)
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

impl<B: Backend> GptOssForCausalLm<B> {
    /// Runs one batched step of `input` against `state` and publishes it.
    fn forward_cached(
        &self,
        input: CausalLmInput<B>,
        state: &mut GptOssDecodeState<B>,
        options: &ForwardOptions,
        phase: Phase,
    ) -> Result<CausalLmOutput<B>> {
        if options.attentions {
            return Err(Error::BackendCapability(
                "gpt-oss cached passes do not capture attention probabilities; use the stateless forward with eager attention".into(),
            ));
        }
        state.validate_for(self)?;
        if state.tensors.has_active_transaction() {
            return Err(TensorCacheError::TransactionActive {
                operation: "run ordinary gpt-oss prefill/decode",
            }
            .into());
        }
        let rows = self.prepare_rows(&input, state, phase)?;
        validate_logits_selection(&options.logits, rows.sequence)?;
        let contexts: Vec<usize> = rows.active_contexts().map(|(_, context)| context).collect();
        let context = widest_context(&contexts);
        ensure_cacheable_rotary_context(self.architecture().rope, context, "gpt-oss")?;
        let retain_router_logits = self.retains_router_logits(options);
        if rows.token_counts.iter().all(|&count| count == 0) {
            return self.inactive_output(&input, &rows, options, retain_router_logits);
        }
        let device = state.tensors.device().clone();
        let batch = CompactBatch::new(
            &rows.valid_indices,
            rows.sequence,
            state.tensors.lengths(),
            &device,
        )?;
        // Padding slots are evaluated and discarded, so they carry token 0
        // and position 0 instead of the caller's arbitrary sentinels.
        let tokens = batch.compact(&input.token_ids, 0);
        let positions = match &input.position_ids {
            Some(positions) => batch.compact(positions, 0),
            None => batch.cache_positions().clone(),
        };
        let angles = self.model.rotary_tables(positions, context)?;
        let heads = self.architecture().query_heads;
        let clean_tail = core::mem::replace(&mut state.clean_tail, false);
        let GptOssDecodeState {
            tensors, layout, ..
        } = state;
        let mut keys = CachedKeys::new(tensors, layout, &batch, angles, heads, clean_tail)?;
        let mut hidden = self.model.embed_tokens.forward(tokens);
        let mut hidden_states = Vec::new();
        let mut router_logits = Vec::new();
        for (index, layer) in self.model.layers.iter().enumerate() {
            if options.hidden_states {
                hidden_states.push(batch.restore(hidden.clone()));
            }
            let (output, logits) = layer.forward(hidden, index, &mut keys, &mut Hooks::new())?;
            hidden = output;
            if retain_router_logits {
                router_logits.push(batch.restore(logits));
            }
        }
        let (step, clean_tail) = (keys.step, keys.clean_tail);
        if options.hidden_states {
            hidden_states.push(batch.restore(hidden.clone()));
        }
        let hidden = self.model.norm.forward(hidden);
        let selected = batch.select(hidden.clone(), &options.logits, rows.last_columns())?;
        let logits = self.project_logits(selected)?;
        let final_hidden_state = options.hidden_states.then(|| batch.restore(hidden));
        if let Some(final_states) = &final_hidden_state {
            hidden_states.push(final_states.clone());
        }
        let output = self.assemble(
            logits,
            final_hidden_state,
            hidden_states,
            router_logits,
            input.token_mask.as_ref(),
            options,
        )?;
        state.tensors.publish_step(step)?;
        state.clean_tail = clean_tail;
        Ok(output)
    }

    /// Validates the call against the state's cursors with one device read.
    fn prepare_rows(
        &self,
        input: &CausalLmInput<B>,
        state: &GptOssDecodeState<B>,
        phase: Phase,
    ) -> Result<StatefulRows> {
        let architecture = self.architecture();
        ensure_same_device(
            "gpt_oss.model",
            &self.model.embed_tokens.weight.lazy_device(),
            state.tensors.device(),
            DECODE_STATE,
        )?;
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

    /// Returns exact zeros for a call in which no row appends a token.
    fn inactive_output(
        &self,
        input: &CausalLmInput<B>,
        rows: &StatefulRows,
        options: &ForwardOptions,
        retain_router_logits: bool,
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
        let states = zeros(rows.sequence, architecture.hidden_size);
        let router_logits = if retain_router_logits {
            vec![zeros(rows.sequence, architecture.experts); architecture.num_layers()]
        } else {
            Vec::new()
        };
        self.assemble(
            zeros(selected, architecture.vocab_size),
            options.hidden_states.then(|| states.clone()),
            if options.hidden_states {
                vec![states; architecture.num_layers() + 2]
            } else {
                Vec::new()
            },
            router_logits,
            input.token_mask.as_ref(),
            options,
        )
    }
}

impl<B: Backend> CausalLanguageModel<B> for GptOssForCausalLm<B> {
    /// The serialized checkpoint configuration.
    type Config = GptOssConfig;
    /// External full-storage and sliding-ring KV state.
    type State = GptOssDecodeState<B>;

    /// Delegates to the stateless forward pass.
    fn forward(
        &self,
        input: CausalLmInput<B>,
        options: &ForwardOptions,
    ) -> Result<CausalLmOutput<B>> {
        GptOssForCausalLm::forward(self, input, options)
    }

    /// Allocates full and sliding-ring KV groups for `batch_size` rows.
    fn new_decode_state(
        &self,
        config: CacheConfig,
        batch_size: usize,
        device: &B::Device,
    ) -> Result<Self::State> {
        GptOssDecodeState::new(self, config, batch_size, device)
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
