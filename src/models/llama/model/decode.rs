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
//!    `p - W < j`, and key validity, by replacement. When a full read has
//!    no invalid key and the block no padding slot, every row's queries are
//!    the last `Q` keys, so the plane is exactly bottom-right causality,
//!    which the attention call applies itself (for one-token decode: no
//!    mask at all). An excluded key gets
//!    probability exactly zero, which is harmless unless its value is
//!    non-finite (`0 · NaN = NaN`); see "Clean tails" for when invalid
//!    values are zeroed;
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
//! # Clean tails
//!
//! Invalid keys in a full-storage read are the slots at or beyond each row's
//! end. The state tracks whether every such slot holds zero (a *clean
//! tail*): true after allocation and `reset` (which zero every tensor),
//! preserved by published steps (they fill exactly the slots they commit),
//! by `reorder` (rows move with their lengths), and by truncation to zero
//! (which zeroes the row). Truncating a row to a shorter non-zero length, or
//! a step that fails after writing ahead, leaves the row's earlier bytes
//! beyond its end, possibly non-finite (an overflowed F16 draft, say), so
//! the flag clears and full layers zero invalid values before attending.
//! While tails are clean, that masked copy of every layer's values is
//! skipped; it was a third of a ragged decode step on Flex. Ring slots
//! cycle through a row's own older tokens, so ring layers always zero
//! invalid values.
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
//!
//! # Shared by every dense decoder
//!
//! The state, the cache layout, and the step are generic over the
//! crate-private `DecoderGraph` interface: [`forward_cached`] runs any
//! graph's blocks against a [`LlamaDecodeState`], which another family can
//! wrap in its own state type. Every layer's rule must be global or a
//! sliding window, and all sliding layers below the capacity must share one
//! window, because they share one ring group.

use core::num::NonZeroUsize;

use burn::tensor::{Bool, DType, Int, Tensor, backend::Backend};

use crate::{
    CausalLanguageModel, CausalLmInput, CausalLmOutput, DecodeInput, Error, ForwardOptions,
    LogitsSelection, Result,
    architecture::DecodeState as ArchitectureDecodeState,
    attention::{AttentionConnectivity, AttentionProjection, GroupedQueryAttention},
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

use super::super::{LlamaFamilyConfig, plan::cache_slots};
use super::{
    LlamaForCausalLm,
    graph::{DecoderBlock, DecoderGraph, GraphLabels, LayerKeys, RotaryTables},
    step::StepLayout,
};

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
    /// Derives the layout of `graph` for `capacity` tokens in its activation dtype.
    ///
    /// Sliding layers whose window is below the capacity share one ring
    /// group; a wider window can never exclude a cached key, so such layers
    /// live in full storage like global layers.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Configuration`] for chunked-local layers and for two
    /// different sliding windows below the capacity, which one ring group
    /// cannot hold, and cache specification errors.
    fn plan<B: Backend, G: DecoderGraph<B>>(graph: &G, capacity: usize) -> Result<Self> {
        let labels = G::LABELS;
        let geometry = graph.geometry();
        let mut ring_window: Option<NonZeroUsize> = None;
        let mut in_ring = Vec::with_capacity(graph.blocks().len());
        for (index, block) in graph.blocks().iter().enumerate() {
            let window = match block.connectivity() {
                AttentionConnectivity::Global => None,
                AttentionConnectivity::SlidingWindow { window } => {
                    NonZeroUsize::new(window).filter(|window| window.get() < capacity)
                }
                AttentionConnectivity::ChunkedLocal { .. } => {
                    return Err(Error::configuration(
                        labels.field("decode_state.layers"),
                        format!(
                            "layer {index} uses chunked-local attention, which this cache cannot store"
                        ),
                    ));
                }
            };
            if let Some(window) = window {
                if ring_window.is_some_and(|ring| ring != window) {
                    return Err(Error::configuration(
                        labels.field("decode_state.layers"),
                        format!(
                            "layer {index} slides over {window} keys, but an earlier layer over {}; one ring group holds one window",
                            ring_window.map_or(0, NonZeroUsize::get)
                        ),
                    ));
                }
                ring_window = Some(window);
            }
            in_ring.push(window.is_some());
        }
        let numbering = cache_slots(&in_ring);
        let dtype = graph.activation_dtype();
        let spec = |layers, storage| {
            NativeKvCacheSpec::new(
                layers,
                geometry.key_value_heads,
                geometry.head_dim,
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
        labels: &GraphLabels,
    ) -> Result<TensorDecodeState<B>> {
        let mut tensors = TensorDecodeState::new(config, batch_size, device)?;
        for group in [self.full, self.ring].into_iter().flatten() {
            let index = tensors.main_mut().add_native_kv(group.spec)?;
            if index != group.index {
                return Err(Error::Cache(format!(
                    "{} KV group must be {}, found {index}",
                    labels.family, group.index
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
    /// Whether every full-storage slot at or beyond a row's length is zero.
    clean_tail: bool,
}

impl<B: Backend> LlamaDecodeState<B> {
    /// Allocates the cache groups of `graph` for `batch_size` rows.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Configuration`] when the capacity exceeds the trained
    /// context or Burn's portable `i32` index range, layout errors (see
    /// `CacheLayout::plan`), and allocation errors.
    pub(crate) fn for_graph<G: DecoderGraph<B>>(
        graph: &G,
        config: CacheConfig,
        batch_size: usize,
        device: &B::Device,
    ) -> Result<Self> {
        let labels = G::LABELS;
        let limit = graph.geometry().max_position_embeddings;
        let capacity = config.capacity();
        if capacity > limit || capacity > i32::MAX as usize {
            return Err(Error::configuration(
                labels.field("decode_state.capacity"),
                format!(
                    "{capacity} exceeds max_position_embeddings {limit} or Burn's portable i32 index range"
                ),
            ));
        }
        let layout = CacheLayout::plan(graph, capacity)?;
        let tensors = layout.allocate(config, batch_size, device, &labels)?;
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

    /// Returns how many layers live in a sliding ring rather than full storage.
    #[must_use]
    pub fn ring_layers(&self) -> usize {
        self.layout
            .slots
            .iter()
            .filter(|slot| matches!(slot, CacheSlot::Ring(_)))
            .count()
    }

    /// Checks that this state was allocated for `graph` and is unchanged.
    fn validate_for<G: DecoderGraph<B>>(&self, graph: &G) -> Result<()> {
        let expected = CacheLayout::plan(graph, self.tensors.capacity())?;
        let groups = [expected.full, expected.ring].into_iter().flatten();
        let intact = groups.clone().count() == self.tensors.main().native_kv_count()
            && groups.into_iter().all(|group| {
                self.tensors
                    .main()
                    .native_kv(group.index)
                    .is_ok_and(|cache| cache.spec() == group.spec)
            });
        if expected != self.layout || !intact {
            return Err(Error::Cache(format!(
                "{} decode state was allocated for a different model",
                G::LABELS.family
            )));
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

    /// Rolls rows back to shorter lengths (ring groups only within their window).
    ///
    /// A row shortened to a non-zero length keeps its later bytes beyond the
    /// new end, so tails are no longer known to be clean.
    fn truncate(&mut self, lengths: &[usize]) -> Result<()> {
        let leaves_bytes = self
            .tensors
            .lengths()
            .iter()
            .zip(lengths)
            .any(|(&from, &to)| to != 0 && to < from);
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
    /// Rotary tables of the block's rotary layers.
    angles: RotaryTables<B>,
    /// `[batch, Q]` cache position of every block slot.
    query_positions: Tensor<B, 2, Int>,
    /// Query heads `H`, the exclusion plane's head extent.
    heads: usize,
    /// Whether every block slot holds a new token (no padding slot).
    dense_block: bool,
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
    /// Display name of the owning family, for cache errors.
    family: &'static str,
}

impl<'s, B: Backend> CachedKeys<'s, B> {
    /// Starts a step appending `block` to `state`.
    ///
    /// Clears the state's clean-tail flag for the step's duration: from the
    /// first write-ahead until publication, a failure would abandon bytes
    /// beyond the lengths. [`Self::finish`] restores it.
    fn new(
        state: &'s mut LlamaDecodeState<B>,
        block: &StepLayout,
        angles: RotaryTables<B>,
        query_positions: Tensor<B, 2, Int>,
        heads: usize,
        family: &'static str,
    ) -> Result<Self> {
        let clean_tail = core::mem::replace(&mut state.clean_tail, false);
        let LlamaDecodeState {
            tensors, layout, ..
        } = state;
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
            dense_block: block.is_dense(),
            clean_tail,
            full_plan,
            ring_plan,
            full_mask: None,
            ring_mask: None,
            family,
        })
    }

    /// Returns every visible key of a full-group layer, writing the step's keys ahead.
    fn full_segment(
        &mut self,
        slot: usize,
        projection: &AttentionProjection<B>,
    ) -> Result<KvBatchSegment<B>> {
        let family = self.family;
        let (group, plan) = self
            .layout
            .full
            .zip(self.full_plan.as_ref())
            .ok_or_else(|| Error::Cache(format!("{family} full KV group is missing")))?;
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
            .ok_or_else(|| Error::Cache(format!("{family} full KV read returned no keys")))
    }

    /// Returns a ring layer's history followed by the step's keys, recording them.
    fn ring_segment(
        &mut self,
        slot: usize,
        projection: &AttentionProjection<B>,
    ) -> Result<KvBatchSegment<B>> {
        let family = self.family;
        let (group, plan) = self
            .layout
            .ring
            .zip(self.ring_plan.as_ref())
            .ok_or_else(|| Error::Cache(format!("{family} sliding KV group is missing")))?;
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
            .ok_or_else(|| Error::Cache(format!("{family} sliding KV read returned no keys")))
    }

    /// Returns the group's exclusions and whether the attention call adds causality.
    ///
    /// A full-storage read is chronological; with every key valid and no
    /// padding slot, bottom-right causality alone is exact. Otherwise the
    /// group's shared plane is built on first use and reused by its layers.
    fn visibility(
        &mut self,
        ring: bool,
        segment: &KvBatchSegment<B>,
    ) -> Result<(Option<Tensor<B, 4, Bool>>, bool)> {
        if !ring && self.dense_block && segment.key_valid().is_none() {
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
        let mask = segment.exclusion_mask(&self.query_positions, self.heads, window)?;
        *cached = Some(mask.clone());
        Ok((Some(mask), false))
    }

    /// Ends the borrow of the state, returning the step and the tail flag to restore.
    fn finish(self) -> (TensorStepWrites<B>, bool) {
        (self.step, self.clean_tail)
    }
}

impl<B: Backend> LayerKeys<B> for CachedKeys<'_, B> {
    /// Returns layer `layer`'s rotary tables for the block.
    fn angles(&self, layer: usize) -> Option<&RotaryAngles<B>> {
        self.angles.for_layer(layer)
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
                "{} decode state has no slot for layer {layer}",
                self.family
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
pub(crate) enum Phase {
    /// Prompt processing.
    Prefill,
    /// Continuation tokens; fully masked rows are finished sequences.
    Decode,
}

/// Runs one batched step of `input` through `graph` against `state` and publishes it.
///
/// See the module documentation for the step. Prefill rejects rows without
/// valid tokens; decode treats them as no-ops with exact zero outputs.
///
/// # Errors
///
/// Returns a capability error for attention capture, foreign-state and
/// input errors, context-dependent RoPE beyond its original context, and
/// cache errors; a failed call leaves the committed state untouched.
pub(crate) fn forward_cached<B: Backend, G: DecoderGraph<B>>(
    graph: &G,
    input: CausalLmInput<B>,
    state: &mut LlamaDecodeState<B>,
    options: &ForwardOptions,
    phase: Phase,
) -> Result<CausalLmOutput<B>> {
    let labels = G::LABELS;
    if options.attentions {
        return Err(Error::BackendCapability(format!(
            "{} cached passes do not capture attention probabilities; use the stateless forward with eager attention",
            labels.family
        )));
    }
    state.validate_for(graph)?;
    let rows = prepare_rows(graph, &input, state, phase)?;
    validate_logits_selection(&options.logits, rows.sequence)?;
    let context = cached_rotary_context(graph, &rows)?;
    if rows.token_counts.iter().all(|&count| count == 0) {
        return inactive_output(graph, &rows, options);
    }
    let layout = StepLayout::plan(&rows, state.tensors.lengths())?;
    let device = input.token_ids.device();
    let query_positions = layout.slot_positions::<B>(&device)?;
    let rotary_positions =
        layout.rotary_positions(input.position_ids.as_ref(), &query_positions)?;
    let angles = graph.rotary_tables(rotary_positions, context)?;
    let tokens = layout.tokens(&input.token_ids)?;
    let heads = graph.geometry().query_heads;
    let mut keys = CachedKeys::new(
        state,
        &layout,
        angles,
        query_positions,
        heads,
        labels.family,
    )?;
    let mut hidden = graph.embed(tokens);
    let mut block_states = Vec::new();
    for (index, block) in graph.blocks().iter().enumerate() {
        if options.hidden_states {
            block_states.push(hidden.clone());
        }
        hidden = block.forward(hidden, index, &mut keys, &mut Hooks::new())?;
    }
    let (step, clean_tail) = keys.finish();
    if options.hidden_states {
        block_states.push(hidden.clone());
    }
    let hidden = graph.final_norm(hidden);
    let output = cached_output(graph, &layout, &rows, options, hidden, block_states)?;
    state.tensors.publish_step(step)?;
    state.clean_tail = clean_tail;
    Ok(output)
}

/// Validates the call against the state's cursors with one device read.
fn prepare_rows<B: Backend, G: DecoderGraph<B>>(
    graph: &G,
    input: &CausalLmInput<B>,
    state: &LlamaDecodeState<B>,
    phase: Phase,
) -> Result<StatefulRows> {
    let labels = G::LABELS;
    let geometry = graph.geometry();
    prepare_stateful_rows(
        input,
        state.tensors.device(),
        state.tensors.lengths(),
        state.tensors.capacity(),
        &StatefulContract {
            names: labels.inputs,
            owner: labels.state,
            vocabulary: geometry.vocab_size,
            position_limit: Some(geometry.max_position_embeddings),
            empty_rows: match phase {
                Phase::Prefill => EmptyRows::Reject,
                Phase::Decode => EmptyRows::Permit,
            },
        },
    )
}

/// Returns the step's rotary context, rejecting one cached keys cannot follow.
fn cached_rotary_context<B: Backend, G: DecoderGraph<B>>(
    graph: &G,
    rows: &StatefulRows,
) -> Result<usize> {
    let context = rows
        .active_contexts()
        .map(|(_, context)| context)
        .max()
        .unwrap_or(1);
    match graph.rotary_schedule().cached_context_limit() {
        Some((limit, scaling)) if context > limit => Err(Error::BackendCapability(format!(
            "{} cached decoding with context-dependent RoPE ({scaling:?}) is limited to the original context {limit}; this call reaches {context}",
            G::LABELS.family
        ))),
        _ => Ok(context),
    }
}

/// Returns exact zeros for a call in which no row appends a token.
fn inactive_output<B: Backend, G: DecoderGraph<B>>(
    graph: &G,
    rows: &StatefulRows,
    options: &ForwardOptions,
) -> Result<CausalLmOutput<B>> {
    let geometry = graph.geometry();
    let selected = match &options.logits {
        LogitsSelection::All => rows.sequence,
        LogitsSelection::Last => 1,
        LogitsSelection::Positions(positions) => positions.len(),
    };
    let device = graph.parameter_device();
    let dtype: DType = graph.activation_dtype();
    let zeros = |tokens, width| Tensor::zeros([rows.batch, tokens, width], (&device, dtype));
    let mut output = CausalLmOutput::from_logits(zeros(selected, geometry.vocab_size));
    if options.hidden_states {
        let states = zeros(rows.sequence, geometry.hidden_size);
        output.final_hidden_state = Some(states.clone());
        output.hidden_states = vec![states; graph.blocks().len() + 2];
    }
    Ok(output)
}

/// Restores requested states to the caller's columns and projects logits.
fn cached_output<B: Backend, G: DecoderGraph<B>>(
    graph: &G,
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
    let mut output = CausalLmOutput::from_logits(graph.project_logits(selected)?);
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
        LlamaDecodeState::for_graph(self, config, batch_size, device)
    }

    /// Appends prompt rows (each with at least one valid token) in one batched step.
    fn prefill(
        &self,
        input: CausalLmInput<B>,
        state: &mut Self::State,
        options: &ForwardOptions,
    ) -> Result<CausalLmOutput<B>> {
        forward_cached(self, input, state, options, Phase::Prefill)
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
        forward_cached(self, input, state, options, Phase::Decode)
    }
}

#[cfg(test)]
#[path = "decode_tests.rs"]
mod tests;
