//! External compressed-latent cache and stateful DeepSeek-V2 execution.
//!
//! Every layer preallocates `[batch, capacity, kv_rank + rope_width]`. The first
//! section stores the normalized KV latent and the second stores the already
//! rotated single-head positional key. A decode token therefore consumes
//! `kv_rank + rope_width` scalars rather than
//! `num_heads * (qk_width + value_width)` scalars. Prefill uses the ordinary
//! expanded K/V equations as a reference path; one-token decode absorbs the K/V
//! up-projection around both matrix products.
//!
//! Writes are operation-local until logits and requested diagnostics succeed.
//! The commit first proves exact `(layer,row)` coverage and every capacity/shape
//! bound, then applies only prevalidated slice assignments and advances logical
//! lengths. A returned error therefore leaves both metadata and cache bytes
//! unchanged.

use std::collections::BTreeSet;

use burn::tensor::{Bool, DType, Int, Tensor, backend::Backend};

use crate::{
    CausalLanguageModel, CausalLmInput, CausalLmOutput, DecodeInput, Error, ForwardOptions, Result,
    architecture::DecodeState as ArchitectureDecodeState,
    cache::CacheConfig,
    models::shared::{
        EmptyRows, StatefulContract, StatefulRows, concat_rows, ensure_same_device,
        ensure_static_rotary_context, implicit_position_ids, index_tensor, prepare_stateful_rows,
        select_hidden_with_last_columns, validate_logits_selection,
    },
    moe::SparseMoeOutput,
};

use super::{
    DeepseekV2DecoderLayer, DeepseekV2ForCausalLm, DeepseekV2Model, INPUT_NAMES, ROUTER_DTYPE,
    router_diagnostics,
};

/// Owner named when an input tensor is on another device than the cache.
const DECODE_STATE: &str = "DeepSeek decode state";

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

/// DeepSeek-specific external decode state with compressed MLA storage.
#[derive(Debug)]
pub struct DeepseekV2DecodeState<B: Backend> {
    /// Validated cache capacity and initial-length metadata.
    config: CacheConfig,
    /// Fixed physical row count.
    batch_size: usize,
    /// Committed semantic prefix length for each row.
    lengths: Vec<usize>,
    /// One `[batch, capacity, kv_rank + rope_width]` tensor per decoder layer.
    layers: Vec<Tensor<B, 3>>,
    /// Cached normalized-latent width.
    kv_rank: usize,
    /// Cached shared positional-key width.
    rope_width: usize,
    /// Activation dtype used by every cache tensor.
    dtype: DType,
    /// Device shared by the model and all cache tensors.
    device: B::Device,
    /// Kernel selected by the most recent successful stateful operation.
    last_path: Option<MlaExecutionPath>,
}

impl<B: Backend> DeepseekV2DecodeState<B> {
    /// Allocates exactly one compressed tensor per MLA layer.
    fn new(
        model: &DeepseekV2ForCausalLm<B>,
        config: CacheConfig,
        batch_size: usize,
        device: &B::Device,
    ) -> Result<Self> {
        if config.sequence_count() != batch_size {
            return Err(Error::Cache(format!(
                "DeepSeek cache describes {} sequences for requested batch {batch_size}",
                config.sequence_count()
            )));
        }
        if config.sliding_window().is_some() {
            return Err(Error::BackendCapability(
                "DeepSeek-V2 MLA uses full-history storage; sliding cache eviction is unsupported"
                    .into(),
            ));
        }
        if config.capacity() > model.config().max_position_embeddings {
            return Err(Error::configuration(
                "deepseek_v2.decode_state.capacity",
                format!(
                    "{} exceeds max_position_embeddings {}",
                    config.capacity(),
                    model.config().max_position_embeddings
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
        let cache_width = kv_rank.checked_add(rope_width).ok_or_else(|| {
            Error::configuration("deepseek_v2.decode_state", "cache width overflows usize")
        })?;
        let dtype = model.model.embed_tokens.weight.val().dtype();
        if !dtype.is_float() || !B::supports_dtype(device, dtype) {
            return Err(Error::BackendCapability(format!(
                "backend cannot allocate DeepSeek MLA cache dtype {dtype:?}"
            )));
        }
        let per_layer_elements = batch_size
            .checked_mul(config.capacity())
            .and_then(|value| value.checked_mul(cache_width))
            .ok_or_else(|| {
                Error::configuration(
                    "deepseek_v2.decode_state",
                    "cache element count overflows usize",
                )
            })?;
        if per_layer_elements > i32::MAX as usize {
            return Err(Error::configuration(
                "deepseek_v2.decode_state",
                "one layer cache exceeds the portable signed tensor-index limit",
            ));
        }
        let total_elements = per_layer_elements
            .checked_mul(model.model.layers.len())
            .ok_or_else(|| {
                Error::configuration(
                    "deepseek_v2.decode_state",
                    "aggregate cache element count overflows usize",
                )
            })?;
        total_elements.checked_mul(dtype.size()).ok_or_else(|| {
            Error::configuration(
                "deepseek_v2.decode_state",
                "cache byte count overflows usize",
            )
        })?;
        let layers = (0..model.model.layers.len())
            .map(|_| {
                Tensor::zeros(
                    [batch_size, config.capacity(), cache_width],
                    (device, dtype),
                )
            })
            .collect();
        Ok(Self {
            config,
            batch_size,
            lengths: vec![0; batch_size],
            layers,
            kv_rank,
            rope_width,
            dtype,
            device: device.clone(),
            last_path: None,
        })
    }

    /// Returns normalized-latent plus rotary-key scalars per layer/token.
    #[must_use]
    pub const fn scalars_per_layer_token(&self) -> usize {
        self.kv_rank + self.rope_width
    }

    /// Returns exact physical bytes allocated by all cache tensors.
    #[must_use]
    pub fn allocated_bytes(&self) -> usize {
        self.batch_size
            * self.config.capacity()
            * self.scalars_per_layer_token()
            * self.layers.len()
            * self.dtype.size()
    }

    /// Returns bytes needed for one additional logical token across all layers.
    #[must_use]
    pub fn bytes_per_token(&self) -> usize {
        self.scalars_per_layer_token() * self.layers.len() * self.dtype.size()
    }

    /// Returns the MLA path selected by the most recent successful call.
    #[must_use]
    pub const fn last_execution_path(&self) -> Option<MlaExecutionPath> {
        self.last_path
    }

    /// Returns one complete preallocated layer tensor for diagnostics/tests.
    pub fn layer_storage(&self, layer: usize) -> Result<Tensor<B, 3>> {
        self.layers
            .get(layer)
            .cloned()
            .ok_or_else(|| Error::Cache(format!("DeepSeek cache layer {layer} is out of bounds")))
    }

    /// Checks that state geometry and device belong to a model.
    fn validate_for_model(&self, model: &DeepseekV2ForCausalLm<B>) -> Result<()> {
        if self.layers.len() != model.model.layers.len()
            || self.kv_rank != model.config().kv_lora_rank
            || self.rope_width != model.config().qk_rope_head_dim
            || self.dtype != model.model.embed_tokens.weight.val().dtype()
            || self.device != model.model.embed_tokens.weight.val().device()
        {
            return Err(Error::Cache(
                "DeepSeek decode state is incompatible with the model".into(),
            ));
        }
        Ok(())
    }

    /// Reads one row's committed compressed history.
    fn read(
        &self,
        layer: usize,
        row: usize,
        length: usize,
    ) -> Result<Option<(Tensor<B, 3>, Tensor<B, 4>)>> {
        let storage = self.layers.get(layer).ok_or_else(|| {
            Error::Cache(format!("DeepSeek cache layer {layer} is out of bounds"))
        })?;
        let committed =
            self.lengths.get(row).copied().ok_or_else(|| {
                Error::Cache(format!("DeepSeek cache row {row} is out of bounds"))
            })?;
        if length != committed {
            return Err(Error::Cache(format!(
                "DeepSeek cache read length {length} disagrees with committed row length {committed}"
            )));
        }
        if length == 0 {
            return Ok(None);
        }
        let packed =
            storage
                .clone()
                .slice([row..row + 1, 0..length, 0..self.scalars_per_layer_token()]);
        let latent = packed.clone().narrow(2, 0, self.kv_rank);
        let rotary = packed
            .narrow(2, self.kv_rank, self.rope_width)
            .reshape([1, length, 1, self.rope_width])
            .swap_dims(1, 2);
        Ok(Some((latent, rotary)))
    }

    /// Validates and atomically publishes a complete operation-local write set.
    fn commit(
        &mut self,
        writes: Vec<PendingMlaWrite<B>>,
        token_counts: &[usize],
        path: MlaExecutionPath,
    ) -> Result<()> {
        if token_counts.len() != self.batch_size {
            return Err(Error::Cache(format!(
                "DeepSeek pending counts describe {} rows for batch {}",
                token_counts.len(),
                self.batch_size
            )));
        }
        let active = token_counts.iter().filter(|&&count| count > 0).count();
        let expected = self.layers.len().checked_mul(active).ok_or_else(|| {
            Error::configuration("deepseek_v2.cache_commit", "write count overflows usize")
        })?;
        if writes.len() != expected {
            return Err(Error::Cache(format!(
                "DeepSeek pending cache coverage is {}/{expected}",
                writes.len()
            )));
        }
        let mut seen = BTreeSet::new();
        for write in &writes {
            let count = token_counts.get(write.row).copied().ok_or_else(|| {
                Error::Cache(format!("DeepSeek cache write row {} is invalid", write.row))
            })?;
            let end = write.start.checked_add(count).ok_or_else(|| {
                Error::configuration("deepseek_v2.cache_commit", "write end overflows usize")
            })?;
            if count == 0
                || write.layer >= self.layers.len()
                || write.start != self.lengths[write.row]
                || end > self.config.capacity()
                || write.data.dims() != [1, count, self.scalars_per_layer_token()]
                || write.data.dtype() != self.dtype
                || write.data.device() != self.device
                || !seen.insert((write.layer, write.row))
            {
                return Err(Error::Cache(format!(
                    "invalid or duplicate DeepSeek write for layer {}, row {}",
                    write.layer, write.row
                )));
            }
        }
        // All checks above are complete. Slice assignment is functional and has
        // no recoverable branch between the first and final publication.
        for write in writes {
            let count = token_counts[write.row];
            let end = write.start + count;
            self.layers[write.layer] = self.layers[write.layer].clone().slice_assign(
                [
                    write.row..write.row + 1,
                    write.start..end,
                    0..self.scalars_per_layer_token(),
                ],
                write.data,
            );
        }
        for (length, additional) in self.lengths.iter_mut().zip(token_counts) {
            *length += additional;
        }
        if active > 0 {
            self.last_path = Some(path);
        }
        Ok(())
    }

    /// Clears logical metadata and all physical cache bytes.
    fn reset_storage(&mut self) {
        let cache_width = self.scalars_per_layer_token();
        for layer in &mut self.layers {
            *layer = Tensor::zeros(
                [self.batch_size, self.config.capacity(), cache_width],
                (&self.device, self.dtype),
            );
        }
        self.lengths.fill(0);
        self.last_path = None;
    }
}

impl<B: Backend> ArchitectureDecodeState<B> for DeepseekV2DecodeState<B> {
    /// Returns the immutable rectangular batch size.
    fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Returns the preallocated token capacity of every batch row.
    fn capacity(&self) -> usize {
        self.config.capacity()
    }

    /// Returns committed semantic lengths for all rows.
    fn lengths(&self) -> &[usize] {
        &self.lengths
    }

    /// Clears logical lengths and every physical cache element.
    fn reset(&mut self) -> Result<()> {
        self.reset_storage();
        Ok(())
    }

    /// Reorders or duplicates rows while preserving each row's committed prefix.
    fn reorder(&mut self, indices: &[usize]) -> Result<()> {
        if indices.len() != self.batch_size || indices.iter().any(|&index| index >= self.batch_size)
        {
            return Err(Error::Cache(format!(
                "DeepSeek reorder requires {} source indices in 0..{}, found {indices:?}",
                self.batch_size, self.batch_size
            )));
        }
        let selector =
            index_tensor::<B>(indices, &self.device, "deepseek_v2.decode_state.reorder")?;
        let reordered_lengths = indices.iter().map(|&index| self.lengths[index]).collect();
        let reordered_layers = self
            .layers
            .iter()
            .map(|layer| layer.clone().select(0, selector.clone()))
            .collect();
        self.lengths = reordered_lengths;
        self.layers = reordered_layers;
        Ok(())
    }

    /// Shortens rows atomically and zeroes the discarded physical suffixes.
    fn truncate(&mut self, requested: &[usize]) -> Result<()> {
        if requested.len() != self.batch_size {
            return Err(Error::Cache(format!(
                "DeepSeek truncate requires {} lengths, found {}",
                self.batch_size,
                requested.len()
            )));
        }
        for (row, (&current, &target)) in self.lengths.iter().zip(requested).enumerate() {
            if target > current {
                return Err(Error::Cache(format!(
                    "DeepSeek truncate row {row} cannot grow from {current} to {target}"
                )));
            }
        }
        let cache_width = self.scalars_per_layer_token();
        for layer in &mut self.layers {
            let mut next = layer.clone();
            for (row, (&current, &target)) in self.lengths.iter().zip(requested).enumerate() {
                if target < current {
                    next = next.slice_assign(
                        [row..row + 1, target..current, 0..cache_width],
                        Tensor::zeros(
                            [1, current - target, cache_width],
                            (&self.device, self.dtype),
                        ),
                    );
                }
            }
            *layer = next;
        }
        self.lengths.copy_from_slice(requested);
        self.last_path = None;
        Ok(())
    }

    /// Returns the backend device that owns all cache tensors.
    fn device(&self) -> &B::Device {
        &self.device
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

    /// Writes a prompt with the reference expanded prefill equations.
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
    /// Evaluates a masked rectangular segment against committed MLA state.
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
        let path = phase.execution_path(prepared.sequence);
        let device = input.token_ids.device();
        let hidden_count = self.model.layers.len() + 2;
        let sparse_count = self
            .model
            .layers
            .iter()
            .filter(|layer| layer.sparse_mlp.is_some())
            .count();
        let retain_routers = options.router_logits || options.auxiliary_losses;
        let mut pending = Vec::new();
        let mut final_rows = Vec::with_capacity(prepared.batch);
        let mut hidden_rows: Vec<Vec<Tensor<B, 3>>> = if options.hidden_states {
            (0..hidden_count)
                .map(|_| Vec::with_capacity(prepared.batch))
                .collect()
        } else {
            Vec::new()
        };
        let mut router_rows: Vec<Vec<RowRouter<B>>> = if retain_routers {
            (0..sparse_count)
                .map(|_| Vec::with_capacity(prepared.batch))
                .collect()
        } else {
            Vec::new()
        };

        for row in 0..prepared.batch {
            let Some(plan) = prepared.compact_row::<B>(row, &device)? else {
                let empty = empty_hidden_row(self, prepared.sequence, &device);
                final_rows.push(empty.clone());
                for destination in &mut hidden_rows {
                    destination.push(empty.clone());
                }
                for destination in &mut router_rows {
                    // Router logits are FP32 regardless of the activation dtype (DSV-1).
                    destination.push(RowRouter {
                        logits: Tensor::zeros(
                            [1, prepared.sequence, self.config().n_routed_experts],
                            (&device, ROUTER_DTYPE),
                        ),
                        selected: Tensor::zeros(
                            [1, prepared.sequence, self.config().num_experts_per_tok],
                            &device,
                        ),
                    });
                }
                continue;
            };
            let tokens = plan.compact(&input.token_ids);
            let baseline = state.lengths[row];
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
                path,
                options.hidden_states,
                retain_routers,
                &mut pending,
            )?;
            final_rows.push(plan.scatter(compact.hidden)?);
            for (destination, hidden) in hidden_rows.iter_mut().zip(compact.hidden_states) {
                destination.push(plan.scatter(hidden)?);
            }
            for (destination, routed) in router_rows.iter_mut().zip(compact.routers) {
                destination.push(RowRouter {
                    logits: plan.scatter(routed.router_logits)?,
                    selected: plan.scatter(routed.selected_experts)?,
                });
            }
        }
        let hidden = concat_rows(final_rows, "deepseek_v2.hidden")?;
        let hidden_states = hidden_rows
            .into_iter()
            .map(|rows| concat_rows(rows, "deepseek_v2.hidden_states"))
            .collect::<Result<Vec<_>>>()?;
        let routers = router_rows
            .into_iter()
            .map(|rows| {
                let logits = concat_rows(
                    rows.iter().map(|row| row.logits.clone()).collect(),
                    "deepseek_v2.router_logits",
                )?;
                let selected = concat_rows(
                    rows.into_iter().map(|row| row.selected).collect(),
                    "deepseek_v2.selected_experts",
                )?;
                router_diagnostics(
                    logits,
                    selected,
                    input.token_mask.as_ref(),
                    options,
                    self.config(),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let selected = select_hidden_with_last_columns(
            hidden.clone(),
            prepared.last_columns(),
            &options.logits,
        )?;
        let logits = self.project_logits(selected)?;
        state.commit(pending, &prepared.token_counts, path)?;
        Ok(CausalLmOutput {
            logits,
            final_hidden_state: options.hidden_states.then_some(hidden),
            hidden_states,
            attentions: Vec::new(),
            routers,
            load_balancing_loss: None,
        })
    }
}

impl<B: Backend> DeepseekV2Model<B> {
    /// Runs one compact semantic row while retaining cache writes in scratch space.
    #[allow(clippy::too_many_arguments)]
    fn forward_compact_row(
        &self,
        token_ids: Tensor<B, 2, Int>,
        positions: Tensor<B, 2, Int>,
        row: usize,
        baseline: usize,
        evaluated_context: usize,
        state: &DeepseekV2DecodeState<B>,
        path: MlaExecutionPath,
        retain_hidden_states: bool,
        retain_routers: bool,
        pending: &mut Vec<PendingMlaWrite<B>>,
    ) -> Result<CompactRowOutput<B>> {
        let mut hidden = self.embed_tokens.forward(token_ids);
        let mut hidden_states = Vec::with_capacity(if retain_hidden_states {
            self.layers.len() + 2
        } else {
            0
        });
        let mut routers = Vec::new();
        if retain_hidden_states {
            hidden_states.push(hidden.clone());
        }
        for (layer_index, layer) in self.layers.iter().enumerate() {
            let output = layer.forward_cached(
                hidden,
                positions.clone(),
                self.rope,
                evaluated_context,
                row,
                baseline,
                layer_index,
                state,
                path,
                pending,
            )?;
            hidden = output.hidden;
            if retain_hidden_states {
                hidden_states.push(hidden.clone());
            }
            if retain_routers && let Some(routed) = output.routed {
                routers.push(routed);
            }
        }
        hidden = self.norm.forward(hidden);
        if retain_hidden_states {
            hidden_states.push(hidden.clone());
        }
        Ok(CompactRowOutput {
            hidden,
            hidden_states,
            routers,
        })
    }
}

impl<B: Backend> DeepseekV2DecoderLayer<B> {
    /// Evaluates one cached layer without publishing its compressed projection.
    #[allow(clippy::too_many_arguments)]
    fn forward_cached(
        &self,
        hidden: Tensor<B, 3>,
        positions: Tensor<B, 2, Int>,
        rope: crate::position::RopeConfig,
        evaluated_context: usize,
        row: usize,
        baseline: usize,
        layer_index: usize,
        state: &DeepseekV2DecodeState<B>,
        path: MlaExecutionPath,
        pending: &mut Vec<PendingMlaWrite<B>>,
    ) -> Result<super::LayerOutput<B>> {
        let residual = hidden.clone();
        let projection = self.self_attn.project_rotary(
            self.input_layernorm.forward(hidden),
            positions,
            rope,
            evaluated_context,
        )?;
        let query_sequence = projection.normalized_latent.dims()[1];
        let (all_latents, all_rotary) = match state.read(layer_index, row, baseline)? {
            Some((past_latent, past_rotary)) => (
                Tensor::cat(vec![past_latent, projection.normalized_latent.clone()], 1),
                Tensor::cat(vec![past_rotary, projection.rotated_key.clone()], 2),
            ),
            None => (
                projection.normalized_latent.clone(),
                projection.rotated_key.clone(),
            ),
        };
        let mask = cached_attention_mask::<B>(
            baseline,
            query_sequence,
            self.self_attn.config.num_heads,
            &projection.normalized_latent.device(),
        )?;
        let mixed = match path {
            MlaExecutionPath::ReferenceUnabsorbed => self.self_attn.mix_unabsorbed(
                &projection,
                all_latents,
                all_rotary,
                Some(mask),
                false,
            )?,
            MlaExecutionPath::AbsorbedDecode => self.self_attn.mix_absorbed(
                &projection,
                all_latents,
                all_rotary,
                Some(mask),
                false,
            )?,
        };
        let packed_rotary = projection.rotated_key.clone().swap_dims(1, 2).reshape([
            1,
            query_sequence,
            state.rope_width,
        ]);
        pending.push(PendingMlaWrite {
            layer: layer_index,
            row,
            start: baseline,
            data: Tensor::cat(vec![projection.normalized_latent, packed_rotary], 2),
        });
        self.finish_block(residual + mixed.hidden_state.clone(), mixed)
    }
}

/// Result from one compact semantic row.
struct CompactRowOutput<B: Backend> {
    /// Final hidden states for only the semantic tokens in one batch row.
    hidden: Tensor<B, 3>,
    /// Optional compact embedding and layer snapshots.
    hidden_states: Vec<Tensor<B, 3>>,
    /// Sparse-layer router tensors before rectangular scatter.
    routers: Vec<SparseMoeOutput<B>>,
}

/// One deferred compressed-cache append.
struct PendingMlaWrite<B: Backend> {
    /// Decoder layer owning the destination cache tensor.
    layer: usize,
    /// Batch row receiving the compact semantic tokens.
    row: usize,
    /// Committed prefix offset at which the append begins.
    start: usize,
    /// Concatenated normalized latent and rotated shared key.
    data: Tensor<B, 3>,
}

/// Rectangular router tensors reconstructed from one compact row.
struct RowRouter<B: Backend> {
    /// Full expert logits scattered back to rectangular sequence layout.
    logits: Tensor<B, 3>,
    /// Selected expert IDs scattered back to rectangular sequence layout.
    selected: Tensor<B, 3, Int>,
}

/// Validates devices, shapes, masks, positions, and capacity before model work.
///
/// The shared preparation reads the mask, token IDs, and explicit positions in
/// one transfer; this wrapper adds the dynamic-NTK guard, which DeepSeek applies
/// only once rotated keys are already cached.
fn prepare_stateful_input<B: Backend>(
    model: &DeepseekV2ForCausalLm<B>,
    input: &CausalLmInput<B>,
    state: &DeepseekV2DecodeState<B>,
    phase: StatefulPhase,
) -> Result<StatefulRows> {
    ensure_same_device(
        "deepseek_v2.model",
        &model.model.embed_tokens.weight.lazy_device(),
        &state.device,
        DECODE_STATE,
    )?;
    let rows = prepare_stateful_rows(
        input,
        &state.device,
        &state.lengths,
        state.config.capacity(),
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
        if state.lengths[row] > 0 {
            ensure_static_rotary_context(model.model.rope.scaling(), context, "DeepSeek")?;
        }
    }
    Ok(rows)
}

/// Builds `excluded(q,k) = k > baseline + q` for one compact row.
fn cached_attention_mask<B: Backend>(
    baseline: usize,
    query_sequence: usize,
    heads: usize,
    device: &B::Device,
) -> Result<Tensor<B, 4, Bool>> {
    let key_sequence = baseline.checked_add(query_sequence).ok_or_else(|| {
        Error::configuration("deepseek_v2.attention_mask", "key length overflows usize")
    })?;
    let query_start = i64::try_from(baseline)
        .map_err(|_| Error::configuration("deepseek_v2.attention_mask", "offset exceeds i64"))?;
    let end = i64::try_from(key_sequence)
        .map_err(|_| Error::configuration("deepseek_v2.attention_mask", "length exceeds i64"))?;
    let query = Tensor::<B, 1, Int>::arange(query_start..end, device)
        .reshape([1, 1, query_sequence, 1])
        .expand([1, heads, query_sequence, key_sequence]);
    let key = Tensor::<B, 1, Int>::arange(0..end, device)
        .reshape([1, 1, 1, key_sequence])
        .expand([1, heads, query_sequence, key_sequence]);
    Ok(key.greater(query))
}

/// Returns exact-zero hidden states for an inactive decode row.
fn empty_hidden_row<B: Backend>(
    model: &DeepseekV2ForCausalLm<B>,
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
/// Cached-versus-stateless parity and atomic decode-state lifecycle regressions.
mod tests;
