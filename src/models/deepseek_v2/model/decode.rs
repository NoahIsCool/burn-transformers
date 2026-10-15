//! External compressed-latent cache and batched stateful DeepSeek-V2 execution.
//!
//! # Storage
//!
//! The decode state is a shared [`TensorDecodeState`](crate::cache::TensorDecodeState) with one single-plane
//! latent group ([`NativeKvCacheSpec::latent`](crate::cache::NativeKvCacheSpec::latent)): every layer stores
//! `[batch, 1, capacity, kv_rank + rope_width]`, the normalized KV latent
//! followed by the already rotated single-head positional key. A token
//! therefore costs `kv_rank + rope_width` scalars per layer instead of
//! `num_heads * (qk_width + value_width)`. Lengths, reorder, truncate, reset,
//! and speculative transactions (in-place staging, so proposals cost no extra
//! memory) come from the shared state.
//!
//! # Batched execution
//!
//! A call packs every row's valid tokens into one `[B, T]` block
//! (`T = max_b n_b`, see the `batch` module), runs each layer once for the
//! whole batch, and scatters results back to the rectangular `[B, S]` layout
//! with exact zeros at padding and for rows without tokens. Fully valid
//! inputs (unmasked prefill, one-token decode of every row) need no packing.
//! Token IDs and explicit positions are validated only at valid columns, so
//! packing overwrites padding entries with token and position zero
//! (`BlockLayout::compact_inert`) before the embedding and rotary tables.
//! Each layer writes its new latents ahead into slots `[L_b, L_b + n_b)` and
//! attends over one zero-copy slot-ordered view `[0, E)`,
//! `E = max_b (L_b + n_b)`; keys beyond a row's end are invalid and masked,
//! and their stale bytes are zeroed before use so a non-finite leftover cannot
//! poison attention. Multi-token calls use the bounded reference (unabsorbed)
//! equations; one-token decode uses absorbed MLA.
//!
//! # Atomicity
//!
//! Every input, capacity, and cursor check runs before any model work. The
//! step's cache writes are published together, after logits and requested
//! diagnostics succeed, with [`TensorDecodeState::publish_step`](crate::cache::TensorDecodeState::publish_step). A later
//! failure leaves every length and every readable cache position unchanged;
//! only slots beyond a row's length may hold the abandoned latents, which no
//! read can reach and the next write overwrites (logical atomicity).

use burn::tensor::{Int, Tensor, backend::Backend};

use crate::{
    CausalLanguageModel, CausalLmInput, CausalLmOutput, DecodeInput, Error, ForwardOptions, Result,
    architecture::DecodeState as ArchitectureDecodeState,
    cache::{
        CacheConfig, KvCacheStorage, NativeKvCacheSpec, TensorCacheError, TensorCacheMark,
        TensorDecodeState, TransactionalCache,
    },
    intervention::Layer,
    models::shared::{
        EmptyRows, StatefulContract, StatefulRows, ensure_same_device, prepare_stateful_rows,
        select_hidden_with_last_columns, validate_logits_selection,
    },
    position::{RopeScaling, RotaryAngles, RotaryFrequencies, rotary_angles_from_frequencies},
};

use super::{
    DeepseekV2ForCausalLm, INPUT_NAMES, ROUTER_DTYPE, block::BlockLayout, router_diagnostics,
};

/// Owner named when an input tensor is on another device than the cache.
const DECODE_STATE: &str = "DeepSeek decode state";

/// Index of the single latent group in the main cache bank.
const LATENT_GROUP: usize = 0;

/// MLA kernel selected by the most recent successful stateful operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MlaExecutionPath {
    /// Expanded per-head content keys and values, used for prefill/chunking.
    ReferenceUnabsorbed,
    /// Compressed-cache associative factorization, used for one-token decode.
    AbsorbedDecode,
}

/// Stateful phase whose inactive-row and MLA-path policies differ.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StatefulPhase {
    /// Requires every row to contribute tokens and uses expanded-key/value attention.
    Prefill,
    /// Allows inactive rows and selects absorbed MLA for single-token steps.
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

    /// Chooses absorbed math only for the latency-critical one-token decode case.
    const fn execution_path(self, sequence: usize) -> MlaExecutionPath {
        if matches!(self, Self::Decode) && sequence == 1 {
            MlaExecutionPath::AbsorbedDecode
        } else {
            MlaExecutionPath::ReferenceUnabsorbed
        }
    }
}

/// DeepSeek external decode state: one compressed latent group per model.
#[derive(Debug)]
pub struct DeepseekV2DecodeState<B: Backend> {
    /// Shared tensor bank, logical lengths, and transaction ledger.
    tensors: TensorDecodeState<B>,
    /// Cached normalized-latent width.
    kv_rank: usize,
    /// Cached shared positional-key width.
    rope_width: usize,
    /// Kernel selected by the most recent successful stateful operation.
    last_path: Option<MlaExecutionPath>,
}

impl<B: Backend> DeepseekV2DecodeState<B> {
    /// Allocates one `[batch, 1, capacity, kv_rank + rope_width]` latent per MLA layer.
    fn new(
        model: &DeepseekV2ForCausalLm<B>,
        config: CacheConfig,
        batch_size: usize,
        device: &B::Device,
    ) -> Result<Self> {
        if config.sliding_window().is_some() {
            return Err(Error::BackendCapability(
                "DeepSeek-V2 MLA uses full-history storage; sliding cache eviction is unsupported"
                    .into(),
            ));
        }
        let limit = model.config().max_position_embeddings;
        if config.capacity() > limit {
            return Err(Error::configuration(
                "deepseek_v2.decode_state.capacity",
                format!(
                    "{} exceeds max_position_embeddings {limit}",
                    config.capacity()
                ),
            ));
        }
        if config.capacity() > i32::MAX as usize {
            return Err(Error::configuration(
                "deepseek_v2.decode_state.capacity",
                "must fit Burn's portable signed position-index range",
            ));
        }
        let kv_rank = model.config().kv_lora_rank;
        let rope_width = model.config().qk_rope_head_dim;
        let width = kv_rank.checked_add(rope_width).ok_or_else(|| {
            Error::configuration("deepseek_v2.decode_state", "cache width overflows usize")
        })?;
        let dtype = model.model.embed_tokens.weight.val().dtype();
        if !dtype.is_float() || !B::supports_dtype(device, dtype) {
            return Err(Error::BackendCapability(format!(
                "backend cannot allocate DeepSeek MLA cache dtype {dtype:?}"
            )));
        }
        let per_layer = batch_size
            .checked_mul(config.capacity())
            .and_then(|value| value.checked_mul(width))
            .filter(|&elements| elements <= i32::MAX as usize)
            .ok_or_else(|| {
                Error::configuration(
                    "deepseek_v2.decode_state",
                    "one layer cache exceeds the portable signed tensor-index limit",
                )
            })?;
        per_layer
            .checked_mul(model.model.layers.len())
            .and_then(|elements| elements.checked_mul(dtype.size()))
            .ok_or_else(|| {
                Error::configuration(
                    "deepseek_v2.decode_state",
                    "cache byte count overflows usize",
                )
            })?;
        // In-place staging writes speculative proposals into the full-history
        // slots beyond each committed length: transactions cost no memory.
        let spec = NativeKvCacheSpec::latent(
            model.model.layers.len(),
            width,
            KvCacheStorage::Full,
            dtype,
        )?
        .with_staging_capacity(config.capacity())
        .with_in_place_staging();
        let mut tensors = TensorDecodeState::new(config, batch_size, device)?;
        let group = tensors.main_mut().add_native_kv(spec)?;
        debug_assert_eq!(group, LATENT_GROUP);
        Ok(Self {
            tensors,
            kv_rank,
            rope_width,
            last_path: None,
        })
    }

    /// Borrows the shared tensor state for memory, transaction, and cache diagnostics.
    #[must_use]
    pub const fn tensor_state(&self) -> &TensorDecodeState<B> {
        &self.tensors
    }

    /// Returns normalized-latent plus rotary-key scalars per layer/token.
    #[must_use]
    pub const fn scalars_per_layer_token(&self) -> usize {
        self.kv_rank + self.rope_width
    }

    /// Returns exact physical bytes allocated by all cache tensors.
    #[must_use]
    pub fn allocated_bytes(&self) -> usize {
        self.tensors.batch_size() * self.tensors.capacity() * self.bytes_per_token()
    }

    /// Returns bytes needed for one additional logical token across all layers.
    #[must_use]
    pub fn bytes_per_token(&self) -> usize {
        let spec = self.spec();
        self.scalars_per_layer_token() * spec.layers() * spec.dtype().size()
    }

    /// Returns the MLA path selected by the most recent successful call.
    #[must_use]
    pub const fn last_execution_path(&self) -> Option<MlaExecutionPath> {
        self.last_path
    }

    /// Returns lengths including any tokens staged by an active transaction.
    #[must_use]
    pub fn working_lengths(&self) -> &[usize] {
        self.tensors.working_lengths()
    }

    /// Returns one complete layer `[batch, capacity, kv_rank + rope_width]` for diagnostics.
    ///
    /// Slots at or beyond a row's length are unspecified: zeros, truncated
    /// tokens, or abandoned speculative and failed-step latents.
    ///
    /// # Errors
    ///
    /// Returns a cache error for an out-of-range layer.
    pub fn layer_storage(&self, layer: usize) -> Result<Tensor<B, 3>> {
        let plane = self
            .tensors
            .main()
            .native_kv(LATENT_GROUP)?
            .keys(layer)?
            .clone();
        let [batch, _, capacity, width] = plane.dims();
        Ok(plane.reshape([batch, capacity, width]))
    }

    /// Returns the latent group's validated specification.
    fn spec(&self) -> NativeKvCacheSpec {
        self.tensors
            .main()
            .native_kv(LATENT_GROUP)
            .map(|cache| cache.spec())
            .expect("a DeepSeek decode state always owns its latent group")
    }

    /// Checks that state geometry, dtype, and device belong to a model.
    fn validate_for_model(&self, model: &DeepseekV2ForCausalLm<B>) -> Result<()> {
        let spec = self.spec();
        let weight = model.model.embed_tokens.weight.val();
        if spec.layers() != model.model.layers.len()
            || self.kv_rank != model.config().kv_lora_rank
            || self.rope_width != model.config().qk_rope_head_dim
            || spec.dtype() != weight.dtype()
            || *self.tensors.device() != weight.device()
        {
            return Err(Error::Cache(
                "DeepSeek decode state is incompatible with the model".into(),
            ));
        }
        Ok(())
    }
}

impl<B: Backend> ArchitectureDecodeState<B> for DeepseekV2DecodeState<B> {
    /// Returns the immutable rectangular batch size.
    fn batch_size(&self) -> usize {
        self.tensors.batch_size()
    }

    /// Returns the preallocated token capacity of every batch row.
    fn capacity(&self) -> usize {
        self.tensors.capacity()
    }

    /// Returns committed semantic lengths for all rows.
    fn lengths(&self) -> &[usize] {
        self.tensors.lengths()
    }

    /// Clears logical lengths and every physical cache element.
    fn reset(&mut self) -> Result<()> {
        self.tensors.reset()?;
        self.last_path = None;
        Ok(())
    }

    /// Reorders or duplicates rows while preserving each row's committed prefix.
    fn reorder(&mut self, indices: &[usize]) -> Result<()> {
        self.tensors.reorder(indices).map_err(Into::into)
    }

    /// Shortens rows; the full-history latents of the kept prefix are unchanged.
    fn truncate(&mut self, requested: &[usize]) -> Result<()> {
        self.tensors.truncate(requested)?;
        self.last_path = None;
        Ok(())
    }

    /// Returns the backend device that owns all cache tensors.
    fn device(&self) -> &B::Device {
        self.tensors.device()
    }
}

impl<B: Backend> TransactionalCache for DeepseekV2DecodeState<B> {
    /// Single-use checkpoint of the committed lengths.
    type Mark = TensorCacheMark;
    /// Transaction and staging errors of the shared tensor state.
    type Error = TensorCacheError;

    /// Begins a speculative transaction; prefill and decode then stage proposals.
    fn mark(&mut self) -> core::result::Result<Self::Mark, Self::Error> {
        self.tensors.begin_transaction()
    }

    /// Commits the first `accepted_tokens` proposals of every row.
    fn commit_prefix(
        &mut self,
        mark: Self::Mark,
        accepted_tokens: usize,
    ) -> core::result::Result<(), Self::Error> {
        self.tensors.commit_uniform_prefix(mark, accepted_tokens)
    }

    /// Discards every proposal; committed latents are untouched.
    fn rollback(&mut self, mark: Self::Mark) -> core::result::Result<(), Self::Error> {
        self.tensors.rollback(mark)
    }
}

impl<B: Backend> CausalLanguageModel<B> for DeepseekV2ForCausalLm<B> {
    /// Validated family configuration associated with the causal-model interface.
    type Config = super::super::DeepseekV2Config;
    /// External per-layer compressed latents, rotary keys, and sequence metadata.
    type State = DeepseekV2DecodeState<B>;

    /// Executes the cache-free reference graph.
    fn forward(
        &self,
        input: CausalLmInput<B>,
        options: &ForwardOptions,
    ) -> Result<CausalLmOutput<B>> {
        DeepseekV2ForCausalLm::forward(self, input, options)
    }

    /// Preallocates one compressed MLA cache tensor for every decoder layer.
    fn new_decode_state(
        &self,
        config: CacheConfig,
        batch_size: usize,
        device: &B::Device,
    ) -> Result<Self::State> {
        DeepseekV2DecodeState::new(self, config, batch_size, device)
    }

    /// Writes a prompt with the bounded reference prefill equations.
    fn prefill(
        &self,
        input: CausalLmInput<B>,
        state: &mut Self::State,
        options: &ForwardOptions,
    ) -> Result<CausalLmOutput<B>> {
        self.forward_with_decode_state(input, state, options, StatefulPhase::Prefill)
    }

    /// Advances active rows, using absorbed MLA for a single-token step.
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

impl<B: Backend> DeepseekV2ForCausalLm<B> {
    /// Evaluates a masked rectangular segment against the cache in one batched pass.
    fn forward_with_decode_state(
        &self,
        input: CausalLmInput<B>,
        state: &mut DeepseekV2DecodeState<B>,
        options: &ForwardOptions,
        phase: StatefulPhase,
    ) -> Result<CausalLmOutput<B>> {
        if options.attentions {
            return Err(Error::BackendCapability(
                "cached DeepSeek attention-probability capture is unavailable".into(),
            ));
        }
        state.validate_for_model(self)?;
        let prepared = prepare_stateful_input(self, &input, state, phase)?;
        validate_logits_selection(&options.logits, prepared.sequence)?;
        let device = input.token_ids.device();
        let Some(layout) = BlockLayout::<B>::new(&prepared, state.working_lengths(), &device)?
        else {
            // Every row is finished: nothing to evaluate or append.
            return self.inactive_output(&input, &prepared, options);
        };
        let path = phase.execution_path(prepared.sequence);
        // Masked columns are never validated, so padding entries receive the
        // valid token 0 and position 0 instead of whatever sentinel they hold.
        let tokens = layout.compact_inert(input.token_ids.clone(), 0);
        let positions = match &input.position_ids {
            Some(position_ids) => layout.compact_inert(position_ids.clone(), 0),
            None => layout.implicit_positions(),
        };
        let angles = self.cached_rotary_tables(positions, &prepared)?;
        let mut step = state.tensors.step(&prepared.token_counts)?;
        let plan = state.tensors.kv_read_plan_through(&step, LATENT_GROUP)?;
        let retain_routers = options.router_logits || options.auxiliary_losses;
        let mut hidden = self.model.embed_tokens.forward(tokens);
        let mut hidden_states = Vec::new();
        if options.hidden_states {
            hidden_states.push(layout.scatter(hidden.clone()));
        }
        let mut routers = Vec::new();
        let mut visibility = None;
        let (rank, rope_width) = (state.kv_rank, state.rope_width);
        for (index, layer) in self.model.layers.iter().enumerate() {
            let residual = hidden.clone();
            let projection = layer
                .self_attn
                .project_with_angles(layer.input_layernorm.forward(hidden), &angles)?;
            let [batch, block, _] = projection.normalized_latent.dims();
            // Cache layout per token: [normalized latent | rotated shared key].
            let rotary = projection
                .rotated_key
                .clone()
                .swap_dims(1, 2)
                .reshape([batch, block, rope_width]);
            let latent = Tensor::cat(vec![projection.normalized_latent.clone(), rotary], 2);
            state
                .tensors
                .write_latent_ahead(&mut step, LATENT_GROUP, index, latent)?;
            let segment = state
                .tensors
                .read_kv_through(&step, LATENT_GROUP, index, &plan)?
                .materialize()
                .ok_or_else(|| Error::Cache("DeepSeek step read no cached keys".into()))?;
            let visibility = match &visibility {
                Some(visibility) => visibility,
                None => visibility.insert(layout.visibility(&segment, true)),
            };
            // Invalid slots may hold stale bytes; zero them so `0 * NaN` cannot occur.
            let history = segment.masked_values();
            let [_, _, keys, width] = history.dims();
            let history = history.reshape([batch, keys, width]);
            let all_latents = history.clone().narrow(2, 0, rank);
            let all_rotary = history
                .narrow(2, rank, rope_width)
                .reshape([batch, keys, 1, rope_width])
                .swap_dims(1, 2);
            let attended = match path {
                MlaExecutionPath::ReferenceUnabsorbed => layer.self_attn.mix_unabsorbed_bounded(
                    &projection,
                    all_latents,
                    all_rotary,
                    visibility,
                    self.model.execution_policy,
                )?,
                MlaExecutionPath::AbsorbedDecode => {
                    let heads = layer.self_attn.config.num_heads;
                    let exclusion = visibility.exclusion(batch, heads, 0..block, 0..keys);
                    layer
                        .self_attn
                        .mix_absorbed(&projection, all_latents, all_rotary, Some(exclusion), false)?
                        .hidden_state
                }
            };
            let output =
                layer.finish_block(residual, attended, None, Layer::decoder(index), None)?;
            hidden = output.hidden;
            if options.hidden_states {
                hidden_states.push(layout.scatter(hidden.clone()));
            }
            if let Some(routed) = output.routed
                && retain_routers
            {
                routers.push(router_diagnostics(
                    layout.scatter(routed.router_logits),
                    layout.scatter(routed.selected_experts),
                    input.token_mask.as_ref(),
                    options,
                    self.config(),
                )?);
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
        state.last_path = Some(path);
        Ok(CausalLmOutput {
            logits,
            final_hidden_state: options.hidden_states.then_some(hidden),
            hidden_states,
            attentions: Vec::new(),
            routers,
            load_balancing_loss: None,
        })
    }

    /// Builds the exact-zero output of a decode step in which every row is finished.
    ///
    /// Router placeholders use the router's F32 dtype, matching active steps.
    fn inactive_output(
        &self,
        input: &CausalLmInput<B>,
        prepared: &StatefulRows,
        options: &ForwardOptions,
    ) -> Result<CausalLmOutput<B>> {
        let device = input.token_ids.device();
        let [batch, sequence] = [prepared.batch, prepared.sequence];
        let config = self.config();
        let dtype = self.model.embed_tokens.weight.val().dtype();
        let hidden = Tensor::<B, 3>::zeros([batch, sequence, config.hidden_size], (&device, dtype));
        let sparse = self
            .model
            .layers
            .iter()
            .filter(|layer| layer.sparse_mlp.is_some())
            .count();
        let routers = if options.router_logits || options.auxiliary_losses {
            (0..sparse)
                .map(|_| {
                    router_diagnostics(
                        Tensor::zeros(
                            [batch, sequence, config.n_routed_experts],
                            (&device, ROUTER_DTYPE),
                        ),
                        Tensor::zeros([batch, sequence, config.num_experts_per_tok], &device),
                        input.token_mask.as_ref(),
                        options,
                        config,
                    )
                })
                .collect::<Result<Vec<_>>>()?
        } else {
            Vec::new()
        };
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
            routers,
            load_balancing_loss: None,
        })
    }

    /// Builds the block's rotary tables, per row only where frequencies differ.
    ///
    /// Static scalings use one table for every row. Context-dependent scalings
    /// (dynamic NTK, LongRoPE) evaluate each row at its own context, as an
    /// independent batch-one call would, and concatenate the tables over rows.
    fn cached_rotary_tables(
        &self,
        positions: Tensor<B, 2, Int>,
        prepared: &StatefulRows,
    ) -> Result<RotaryAngles<B>> {
        let contexts = prepared
            .active_contexts()
            .map(|(_, context)| context)
            .collect::<Vec<_>>();
        let first = contexts.first().copied().unwrap_or(1).max(1);
        if !self.model.rope.scaling().is_context_dependent()
            || contexts.iter().all(|&context| context == first)
        {
            return self.model.rotary_tables(positions, first);
        }
        let dtype = self.model.embed_tokens.weight.val().dtype();
        let [batch, block] = positions.dims();
        let mut cosine = Vec::with_capacity(batch);
        let mut sine = Vec::with_capacity(batch);
        let mut rotary_dimension = 0;
        for row in 0..batch {
            let context = match prepared.token_counts[row] {
                0 => first,
                _ => prepared.evaluated_contexts[row].max(1),
            };
            let frequencies = RotaryFrequencies::new(self.model.rope, context)?;
            let angles = rotary_angles_from_frequencies(
                &frequencies,
                positions.clone().slice([row..row + 1, 0..block]),
            )?
            .to_dtype(dtype);
            rotary_dimension = angles.rotary_dimension;
            cosine.push(angles.cosine);
            sine.push(angles.sine);
        }
        Ok(RotaryAngles {
            cosine: Tensor::cat(cosine, 0),
            sine: Tensor::cat(sine, 0),
            rotary_dimension,
        })
    }
}

/// Validates devices, shapes, masks, positions, and capacity before model work.
///
/// Rows append at their *working* lengths, so prefill and decode inside a
/// speculative transaction stage proposals after the earlier ones. The shared
/// preparation reads the mask, token IDs, and explicit positions in one
/// transfer; this wrapper adds the cached-rotary guard, which applies only
/// once rotated keys are already cached.
fn prepare_stateful_input<B: Backend>(
    model: &DeepseekV2ForCausalLm<B>,
    input: &CausalLmInput<B>,
    state: &DeepseekV2DecodeState<B>,
    phase: StatefulPhase,
) -> Result<StatefulRows> {
    ensure_same_device(
        "deepseek_v2.model",
        &model.model.embed_tokens.weight.lazy_device(),
        state.tensors.device(),
        DECODE_STATE,
    )?;
    let baselines = state.working_lengths();
    let rows = prepare_stateful_rows(
        input,
        state.tensors.device(),
        baselines,
        state.tensors.capacity(),
        &StatefulContract {
            names: INPUT_NAMES,
            owner: DECODE_STATE,
            vocabulary: model.config().vocab_size,
            position_limit: Some(model.config().max_position_embeddings),
            empty_rows: phase.empty_rows(),
        },
    )?;
    // A fresh prefill has no rotated keys to invalidate yet.
    for (row, context) in rows.active_contexts() {
        if baselines[row] > 0 {
            ensure_cached_rotary_context(model.model.rope.scaling(), context)?;
        }
    }
    Ok(rows)
}

/// Rejects a rotary context that would invalidate keys already in the cache.
///
/// Cached rotary keys were rotated with the frequencies of the context at
/// the time they were written. Context-dependent scalings (dynamic NTK, and
/// LongRoPE's short/long factor switch) change those frequencies beyond
/// [`RopeScaling::cached_context_limit`], so extending a non-empty cache past
/// it would mix incompatible rotations; static scalings never invalidate keys.
///
/// # Errors
///
/// Returns [`Error::BackendCapability`] for a context beyond the limit.
fn ensure_cached_rotary_context(scaling: RopeScaling, context: usize) -> Result<()> {
    match scaling.cached_context_limit() {
        Some(limit) if context > limit => Err(Error::BackendCapability(format!(
            "DeepSeek cached decoding at context {context} exceeds the {limit}-token limit of its \
             context-dependent rotary scaling, which would require re-rotating cached keys"
        ))),
        _ => Ok(()),
    }
}

#[cfg(test)]
#[path = "decode_tests.rs"]
/// Batched-versus-per-row and cached-versus-stateless parity, lifecycle, and atomicity.
mod tests;
