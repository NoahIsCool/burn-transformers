//! Cache-aware Qwen3.5/Qwen3.8 causal inference.
//!
//! Qwen's hybrid schedule has two physically different histories. A full-attention
//! layer retains every native key/value head, whereas a DeltaNet layer retains only
//! its terminal association matrix and the last `kernel - 1` convolution inputs.
//! [`Qwen3_5DecodeState`] maps model-layer indices monotonically into one cache group
//! of each kind, keeping runtime state outside Burn's parameter [`Module`] graph.
//!
//! Masked rectangular batches are compacted one row at a time before model math.
//! This is more than an optimization detail: a padding token must not shift the
//! causal-convolution history or consume a logical RoPE position. Results at masked
//! output positions are defined as zero; valid positions are scattered back to the
//! caller's rectangular shape.

use burn::tensor::{Bool, DType, ElementConversion, Int, Tensor, TensorData, backend::Backend};

use crate::{
    CausalLanguageModel, CausalLmInput, CausalLmOutput, DecodeInput, Error, ForwardOptions, Result,
    architecture::DecodeState as ArchitectureDecodeState,
    cache::{
        CacheConfig, DecodeStateError, KvCacheStorage, NativeKvCacheSpec, RecurrentStateSpec,
        TensorCacheBank, TensorCacheError, TensorDecodeState,
    },
    delta_net::{DeltaNetMode, GatedDeltaNetState},
    position::{RopeScaling, RotaryAngles, apply_rotary, derive_position_ids, rotary_angles},
};

use super::{
    Qwen3_5DecoderLayer, Qwen3_5ForCausalLm, Qwen3_5TextModel, select_hidden,
    validate_logits_selection, validate_mask, validate_token_ids,
};
use crate::models::qwen3_5::{Qwen3_5LayerType, Qwen3_5TextConfig};

const NATIVE_KV_GROUP: usize = 0;
const RECURRENT_GROUP: usize = 0;

/// Physical cache slot used by one layer in the Qwen hybrid schedule.
///
/// Slot numbers are dense within each cache family. For example, the schedule
/// `[DeltaNet, DeltaNet, Full, DeltaNet, Full]` maps to
/// `[DeltaNet(0), DeltaNet(1), Full(0), DeltaNet(2), Full(1)]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Qwen3_5LayerCache {
    /// Layer slot in the single recurrent DeltaNet cache group.
    DeltaNet(usize),
    /// Layer slot in the single native-KV full-attention cache group.
    FullAttention(usize),
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
    schedule: Vec<Qwen3_5LayerCache>,
    native_kv: Option<NativeKvCacheSpec>,
    recurrent: Option<RecurrentStateSpec>,
}

/// Qwen-specific external inference state backed by shared Burn tensor caches.
///
/// Exactly one native-KV group is allocated when the schedule contains full
/// attention, and exactly one recurrent group is allocated when it contains
/// DeltaNet. The wrapper owns only immutable layer mapping plus the generic
/// [`TensorDecodeState`]; model weights and session history never share a record.
#[derive(Debug)]
pub struct Qwen3_5DecodeState<B: Backend> {
    tensors: TensorDecodeState<B>,
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

        let layout = cache_layout(model)?;
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

    /// Verifies that cached geometry and layer ordering belong to `model`.
    fn validate_for_model(&self, model: &Qwen3_5ForCausalLm<B>) -> Result<()> {
        let expected = cache_layout(model)?;
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
        Ok(())
    }
}

impl<B: Backend> ArchitectureDecodeState<B> for Qwen3_5DecodeState<B> {
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

impl<B: Backend> CausalLanguageModel<B> for Qwen3_5ForCausalLm<B> {
    type Config = Qwen3_5TextConfig;
    type State = Qwen3_5DecodeState<B>;

    fn config(&self) -> &Self::Config {
        Qwen3_5ForCausalLm::config(self)
    }

    fn forward(
        &self,
        input: CausalLmInput<B>,
        options: &ForwardOptions,
    ) -> Result<CausalLmOutput<B>> {
        Qwen3_5ForCausalLm::forward(self, input, options)
    }

    fn new_decode_state(
        &self,
        config: CacheConfig,
        batch_size: usize,
        device: &B::Device,
    ) -> Result<Self::State> {
        Qwen3_5DecodeState::new(self, config, batch_size, device)
    }

    fn prefill(
        &self,
        input: CausalLmInput<B>,
        state: &mut Self::State,
        options: &ForwardOptions,
    ) -> Result<CausalLmOutput<B>> {
        self.forward_with_decode_state(input, state, options, DeltaNetMode::ChunkedPrefill)
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
            DeltaNetMode::Recurrent,
        )
    }
}

impl<B: Backend> Qwen3_5ForCausalLm<B> {
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
        delta_mode: DeltaNetMode,
    ) -> Result<CausalLmOutput<B>> {
        if options.attentions {
            return Err(Error::BackendCapability(
                "Qwen attention probability capture requires the reference attention path".into(),
            ));
        }
        state.validate_for_model(self)?;
        if state.tensors.has_active_transaction() {
            return Err(TensorCacheError::TransactionActive {
                operation: "run ordinary Qwen prefill/decode",
            }
            .into());
        }
        let prepared = prepare_stateful_input(self, &input, state)?;
        validate_logits_selection(&options.logits, prepared.sequence)?;
        let device = input.token_ids.device();
        let diagnostic_count = self.model.layers.len() + 2;
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
            let selector = selection_tensor::<B>(valid, &device)?;
            let tokens = input
                .token_ids
                .clone()
                .slice([sequence_index..sequence_index + 1, 0..prepared.sequence])
                .select(1, selector.clone());
            let baseline = state.tensors.lengths()[sequence_index];
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
                baseline,
                evaluated_context,
                &mut state.tensors,
                &state.layout.schedule,
                delta_mode,
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

        // Capacity was checked for every row before the first tensor mutation, so
        // this metadata publication cannot fail unless an internal invariant changed.
        state.tensors.advance(&prepared.token_counts)?;
        let hidden = Tensor::cat(final_rows, 0);
        let hidden_states = diagnostic_rows
            .into_iter()
            .map(|rows| Tensor::cat(rows, 0))
            .collect();
        let selected = select_hidden(hidden.clone(), input.token_mask.as_ref(), &options.logits)?;
        let logits = match &self.lm_head {
            Some(head) => head.forward(selected),
            None => burn::tensor::module::linear(
                selected,
                self.model.embed_tokens.weight.val().transpose(),
                None,
            ),
        };
        Ok(CausalLmOutput {
            logits,
            final_hidden_state: options.hidden_states.then_some(hidden),
            hidden_states,
            attentions: Vec::new(),
            routers: Vec::new(),
        })
    }
}

impl<B: Backend> Qwen3_5TextModel<B> {
    /// Evaluates one compacted batch row while mutating only that row's caches.
    #[allow(clippy::too_many_arguments)]
    fn forward_compact_row(
        &self,
        token_ids: Tensor<B, 2, Int>,
        positions: Tensor<B, 2, Int>,
        sequence_index: usize,
        baseline: usize,
        evaluated_context: usize,
        state: &mut TensorDecodeState<B>,
        schedule: &[Qwen3_5LayerCache],
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
            Some(rotary_angles(self.rope, positions, evaluated_context)?)
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
                baseline,
                state.main_mut(),
                slot,
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
    /// Native keys are rotated before being stored, so later decode calls never
    /// need the old hidden states or old position IDs. DeltaNet stores its FP32
    /// terminal matrix and activation-dtype convolution tail after the complete
    /// compact segment.
    #[allow(clippy::too_many_arguments)]
    fn forward_cached(
        &self,
        hidden: Tensor<B, 3>,
        rotary: Option<&RotaryAngles<B>>,
        sequence_index: usize,
        baseline: usize,
        bank: &mut TensorCacheBank<B>,
        slot: Qwen3_5LayerCache,
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
                let cache = bank.native_kv_mut(NATIVE_KV_GROUP)?;
                let history = cache.read_sequence(cache_layer, sequence_index, baseline)?;
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
                let hidden = finish_residual(self, residual, mixed);
                cache.write_sequence(
                    cache_layer,
                    sequence_index,
                    baseline,
                    new_keys,
                    new_values,
                )?;
                Ok(hidden)
            }
            Qwen3_5LayerCache::DeltaNet(cache_layer) => {
                let delta = self.linear_attn.as_ref().ok_or_else(|| {
                    Error::Checkpoint("Qwen linear layer has no DeltaNet module".into())
                })?;
                let cache = bank.recurrent_mut(RECURRENT_GROUP)?;
                let cached = cache.read_sequence(cache_layer, sequence_index)?;
                let mut recurrent = GatedDeltaNetState {
                    convolution: cached.convolution().clone(),
                    recurrent: cached.matrix().clone(),
                };
                let mixed =
                    delta.forward_with_state(normalized, None, &mut recurrent, delta_mode)?;
                let hidden = finish_residual(self, residual, mixed);
                cache.set_sequence_state(
                    cache_layer,
                    sequence_index,
                    recurrent.recurrent,
                    recurrent.convolution,
                )?;
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
) -> Tensor<B, 3> {
    let hidden = residual + mixed;
    let residual = hidden.clone();
    residual
        + layer
            .mlp
            .forward(layer.post_attention_layernorm.forward(hidden))
}

/// Builds the model-layer mapping and homogeneous cache specifications.
fn cache_layout<B: Backend>(model: &Qwen3_5ForCausalLm<B>) -> Result<Qwen3_5CacheLayout> {
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
        Some(RecurrentStateSpec::new_with_dtypes(
            delta_layers,
            config.value_heads,
            config.key_head_dim,
            config.value_head_dim,
            config.convolution_channels()?,
            convolution_width,
            DType::F32,
            activation_dtype,
        )?)
    };
    Ok(Qwen3_5CacheLayout {
        schedule,
        native_kv,
        recurrent,
    })
}

/// Validated rectangular input metadata needed before any cache mutation.
struct PreparedStatefulInput {
    batch: usize,
    sequence: usize,
    valid_indices: Vec<Vec<usize>>,
    token_counts: Vec<usize>,
    evaluated_contexts: Vec<usize>,
}

/// Validates devices, shapes, positions, masks, and all capacity transitions.
fn prepare_stateful_input<B: Backend>(
    model: &Qwen3_5ForCausalLm<B>,
    input: &CausalLmInput<B>,
    state: &Qwen3_5DecodeState<B>,
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
        validate_position_range(positions.clone(), model.config().max_position_embeddings)?;
    }
    validate_token_ids(input.token_ids.clone(), model.config().vocab_size)?;
    validate_mask(input.token_mask.as_ref(), batch, sequence)?;
    let valid_indices = semantic_indices(input.token_mask.as_ref(), batch, sequence)?;
    let token_counts = valid_indices.iter().map(Vec::len).collect::<Vec<_>>();
    let mut implicit_contexts = Vec::with_capacity(batch);
    for (sequence_index, (&baseline, &additional)) in state
        .tensors
        .lengths()
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
    let evaluated_contexts = match &input.position_ids {
        Some(positions) => {
            let device = positions.device();
            let mut contexts = Vec::with_capacity(batch);
            for (sequence_index, valid) in valid_indices.iter().enumerate() {
                let selector = selection_tensor::<B>(valid, &device)?;
                let compact = positions
                    .clone()
                    .slice([sequence_index..sequence_index + 1, 0..sequence])
                    .select(1, selector);
                contexts.push(evaluated_context(compact)?);
            }
            contexts
        }
        None => implicit_contexts,
    };
    for &context in &evaluated_contexts {
        validate_cached_rope_context(model.model.rope.scaling(), context)?;
    }
    Ok(PreparedStatefulInput {
        batch,
        sequence,
        valid_indices,
        token_counts,
        evaluated_contexts,
    })
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
    let values = mask
        .as_tensor()
        .clone()
        .into_data()
        .to_vec::<bool>()
        .map_err(|error| {
            Error::BackendCapability(format!("failed to read Qwen token mask: {error}"))
        })?;
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
fn evaluated_context<B: Backend>(positions: Tensor<B, 2, Int>) -> Result<usize> {
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
fn validate_position_range<B: Backend>(
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
mod tests {
    use burn::{
        backend::Flex,
        tensor::{
            Bool, Int, Tensor, TensorData, Tolerance,
            backend::{Backend, BackendTypes},
        },
    };

    use super::*;
    use crate::{
        DecodeState, LogitsSelection,
        architecture::TokenMask,
        cache::TensorCacheError,
        models::config::{RopeParameters, RopeType},
        models::qwen3_5::{Qwen3_5LayerType, Qwen3_5TextConfig},
    };

    type TestBackend = Flex<f32, i32>;

    /// Builds a small interleaved topology with every cache family represented.
    fn tiny_config() -> Qwen3_5TextConfig {
        Qwen3_5TextConfig {
            vocab_size: 32,
            hidden_size: 8,
            intermediate_size: 16,
            num_hidden_layers: 4,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 4,
            max_position_embeddings: 64,
            layer_types: vec![
                Qwen3_5LayerType::LinearAttention,
                Qwen3_5LayerType::LinearAttention,
                Qwen3_5LayerType::LinearAttention,
                Qwen3_5LayerType::FullAttention,
            ],
            linear_key_head_dim: 4,
            linear_value_head_dim: 4,
            linear_num_key_heads: 1,
            linear_num_value_heads: 2,
            linear_conv_kernel_dim: 3,
            partial_rotary_factor: Some(1.0),
            rope_theta: Some(10_000.0),
            mtp_num_hidden_layers: 0,
            ..Default::default()
        }
    }

    /// Initializes deterministic tiny parameters for parity comparisons.
    fn tiny_model(
        device: &<TestBackend as BackendTypes>::Device,
    ) -> Qwen3_5ForCausalLm<TestBackend> {
        TestBackend::seed(device, 211);
        Qwen3_5ForCausalLm::init(tiny_config(), device).expect("tiny Qwen model must initialize")
    }

    /// Requests every vocabulary position and the complete hidden-state trace.
    fn diagnostic_options() -> ForwardOptions {
        ForwardOptions {
            logits: LogitsSelection::All,
            hidden_states: true,
            ..Default::default()
        }
    }

    /// Compares two float tensors using tight recurrent/chunked numerical tolerance.
    fn assert_close<const D: usize>(
        actual: Tensor<TestBackend, D>,
        expected: Tensor<TestBackend, D>,
    ) {
        actual
            .into_data()
            .assert_approx_eq::<f32>(&expected.into_data(), Tolerance::rel_abs(2e-4, 2e-5));
    }

    /// The hybrid schedule maps monotonically into exactly two homogeneous groups.
    #[test]
    fn decode_state_maps_hybrid_layers_deterministically() {
        let device = Default::default();
        let model = tiny_model(&device);
        let state = model
            .new_decode_state(CacheConfig::new(2, 12).expect("cache config"), 2, &device)
            .expect("state must allocate");

        assert_eq!(state.full_attention_layer_count(), 1);
        assert_eq!(state.delta_net_layer_count(), 3);
        assert_eq!(state.layer_cache(0), Some(Qwen3_5LayerCache::DeltaNet(0)));
        assert_eq!(state.layer_cache(1), Some(Qwen3_5LayerCache::DeltaNet(1)));
        assert_eq!(state.layer_cache(2), Some(Qwen3_5LayerCache::DeltaNet(2)));
        assert_eq!(
            state.layer_cache(3),
            Some(Qwen3_5LayerCache::FullAttention(0))
        );
        assert_eq!(state.layer_cache(4), None);
        assert_eq!(state.tensor_state().main().native_kv_count(), 1);
        assert_eq!(state.tensor_state().main().recurrent_count(), 1);
        let recurrent = state
            .tensor_state()
            .main()
            .recurrent(0)
            .expect("recurrent group");
        assert_eq!(recurrent.spec().matrix_dtype(), DType::F32);
        assert_eq!(
            recurrent.spec().convolution_dtype(),
            model.model.embed_tokens.weight.val().dtype()
        );
    }

    /// A cache config must describe the requested batch and supported model context.
    #[test]
    fn decode_state_rejects_batch_and_context_mismatch() {
        let device = Default::default();
        let model = tiny_model(&device);
        let batch_error = model
            .new_decode_state(CacheConfig::new(1, 8).expect("cache config"), 2, &device)
            .expect_err("batch mismatch must fail");
        assert!(matches!(
            batch_error,
            Error::TensorCache(TensorCacheError::BatchSizeMismatch { .. })
        ));

        let context_error = model
            .new_decode_state(CacheConfig::new(1, 65).expect("cache config"), 1, &device)
            .expect_err("model context overflow must fail");
        assert!(matches!(context_error, Error::Configuration { .. }));
    }

    /// Homogeneous schedules allocate no unused cache family and remain exact.
    #[test]
    fn homogeneous_schedules_decode_without_unused_cache_groups() {
        let device = Default::default();
        for (seed, layer_type) in [
            (301, Qwen3_5LayerType::LinearAttention),
            (302, Qwen3_5LayerType::FullAttention),
        ] {
            let mut config = tiny_config();
            config.num_hidden_layers = 2;
            config.layer_types = vec![layer_type; 2];
            TestBackend::seed(&device, seed);
            let model = Qwen3_5ForCausalLm::init(config, &device).expect("homogeneous model");
            let mut state = model
                .new_decode_state(CacheConfig::new(1, 4).expect("cache config"), 1, &device)
                .expect("decode state");
            let expected = model
                .forward(
                    CausalLmInput {
                        token_ids: Tensor::<TestBackend, 2, Int>::from_ints(
                            [[1, 2, 3, 4]],
                            &device,
                        ),
                        token_mask: None,
                        position_ids: None,
                    },
                    &ForwardOptions {
                        logits: LogitsSelection::All,
                        ..Default::default()
                    },
                )
                .expect("full forward")
                .logits
                .narrow(1, 2, 2);
            model
                .prefill(
                    CausalLmInput {
                        token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[1, 2]], &device),
                        token_mask: None,
                        position_ids: None,
                    },
                    &mut state,
                    &ForwardOptions::default(),
                )
                .expect("prefill");
            let actual = model
                .decode(
                    DecodeInput {
                        token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[3, 4]], &device),
                        position_ids: None,
                        token_mask: None,
                    },
                    &mut state,
                    &ForwardOptions {
                        logits: LogitsSelection::All,
                        ..Default::default()
                    },
                )
                .expect("decode")
                .logits;

            assert_close(actual, expected);
            match layer_type {
                Qwen3_5LayerType::LinearAttention => {
                    assert_eq!(state.full_attention_layer_count(), 0);
                    assert_eq!(state.delta_net_layer_count(), 2);
                    assert_eq!(state.tensor_state().main().native_kv_count(), 0);
                    assert_eq!(state.tensor_state().main().recurrent_count(), 1);
                }
                Qwen3_5LayerType::FullAttention => {
                    assert_eq!(state.full_attention_layer_count(), 2);
                    assert_eq!(state.delta_net_layer_count(), 0);
                    assert_eq!(state.tensor_state().main().native_kv_count(), 1);
                    assert_eq!(state.tensor_state().main().recurrent_count(), 0);
                }
            }
        }
    }

    /// A one-wide convolution has no external history axis and is rejected safely.
    #[test]
    fn stateful_delta_net_rejects_unit_convolution_kernel() {
        let device = Default::default();
        let mut config = tiny_config();
        config.linear_conv_kernel_dim = 1;
        let model =
            Qwen3_5ForCausalLm::<TestBackend>::init(config, &device).expect("stateless model");
        let error = model
            .new_decode_state(CacheConfig::new(1, 4).expect("cache config"), 1, &device)
            .expect_err("zero-width convolution history must fail");
        assert!(matches!(error, Error::Configuration { .. }));
    }

    /// Context-dependent dynamic NTK cannot silently reuse stale rotated keys.
    #[test]
    fn dynamic_ntk_extension_is_rejected_before_cache_mutation() {
        let device = Default::default();
        let mut config = tiny_config();
        config.rope_parameters = Some(RopeParameters {
            rope_type: RopeType::Dynamic,
            factor: Some(2.0),
            original_max_position_embeddings: Some(2),
            ..Default::default()
        });
        let model = Qwen3_5ForCausalLm::init(config, &device).expect("dynamic-NTK model");
        let mut state = model
            .new_decode_state(CacheConfig::new(1, 4).expect("cache config"), 1, &device)
            .expect("decode state");
        let error = model
            .prefill(
                CausalLmInput {
                    token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[1, 2, 3]], &device),
                    token_mask: None,
                    position_ids: None,
                },
                &mut state,
                &ForwardOptions::default(),
            )
            .expect_err("dynamic cache extension must fail");

        assert!(matches!(error, Error::BackendCapability(_)));
        assert_eq!(state.lengths(), &[0]);
    }

    /// Invalid explicit positions and logit selections fail before state changes.
    #[test]
    fn invalid_positions_and_logit_selection_are_atomic() {
        let device = Default::default();
        let model = tiny_model(&device);
        let mut state = model
            .new_decode_state(CacheConfig::new(1, 4).expect("cache config"), 1, &device)
            .expect("decode state");
        let invalid_position = model
            .prefill(
                CausalLmInput {
                    token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[1]], &device),
                    token_mask: None,
                    position_ids: Some(Tensor::<TestBackend, 2, Int>::from_ints([[-1]], &device)),
                },
                &mut state,
                &ForwardOptions::default(),
            )
            .expect_err("negative position must fail");
        assert!(matches!(invalid_position, Error::Shape { .. }));
        assert_eq!(state.lengths(), &[0]);

        let invalid_selection = model
            .prefill(
                CausalLmInput {
                    token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[1]], &device),
                    token_mask: None,
                    position_ids: None,
                },
                &mut state,
                &ForwardOptions {
                    logits: LogitsSelection::Positions(Vec::new()),
                    ..Default::default()
                },
            )
            .expect_err("empty selection must fail");
        assert!(matches!(invalid_selection, Error::Shape { .. }));
        assert_eq!(state.lengths(), &[0]);
    }

    /// Full forward and cache-filling prefill produce the same valid graph outputs.
    #[test]
    fn prefill_matches_full_forward() {
        let device = Default::default();
        let model = tiny_model(&device);
        let input = CausalLmInput {
            token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[1, 2, 3], [4, 5, 6]], &device),
            token_mask: None,
            position_ids: None,
        };
        let options = diagnostic_options();
        let expected = model
            .forward(input.clone(), &options)
            .expect("stateless forward");
        let mut state = model
            .new_decode_state(CacheConfig::new(2, 8).expect("cache config"), 2, &device)
            .expect("decode state");
        let actual = model
            .prefill(input, &mut state, &options)
            .expect("cached prefill");

        assert_close(actual.logits, expected.logits);
        assert_close(
            actual.final_hidden_state.expect("requested hidden"),
            expected.final_hidden_state.expect("requested hidden"),
        );
        assert_eq!(actual.hidden_states.len(), expected.hidden_states.len());
        for (actual, expected) in actual.hidden_states.into_iter().zip(expected.hidden_states) {
            assert_close(actual, expected);
        }
        assert_eq!(state.lengths(), &[3, 3]);
    }

    /// A multi-token recurrent decode equals the corresponding suffix of full forward.
    #[test]
    fn multi_token_decode_matches_full_forward_suffix() {
        let device = Default::default();
        let model = tiny_model(&device);
        let options = ForwardOptions {
            logits: LogitsSelection::All,
            ..Default::default()
        };
        let expected = model
            .forward(
                CausalLmInput {
                    token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[1, 2, 3, 4, 5]], &device),
                    token_mask: None,
                    position_ids: None,
                },
                &options,
            )
            .expect("full forward")
            .logits
            .narrow(1, 2, 3);
        let mut state = model
            .new_decode_state(CacheConfig::new(1, 5).expect("cache config"), 1, &device)
            .expect("decode state");
        model
            .prefill(
                CausalLmInput {
                    token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[1, 2]], &device),
                    token_mask: None,
                    position_ids: None,
                },
                &mut state,
                &ForwardOptions::default(),
            )
            .expect("prefill");
        let actual = model
            .decode(
                DecodeInput {
                    token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[3, 4, 5]], &device),
                    position_ids: None,
                    token_mask: None,
                },
                &mut state,
                &options,
            )
            .expect("multi-token decode")
            .logits;

        assert_close(actual, expected);
        assert_eq!(state.lengths(), &[5]);
    }

    /// Ragged masks neither consume positions nor shift DeltaNet convolution state.
    #[test]
    fn ragged_prefill_and_decode_match_compacted_full_forward() {
        let device = Default::default();
        let model = tiny_model(&device);
        let mut state = model
            .new_decode_state(CacheConfig::new(2, 8).expect("cache config"), 2, &device)
            .expect("decode state");
        let last = ForwardOptions::default();

        let prefill = model
            .prefill(
                CausalLmInput {
                    token_ids: Tensor::<TestBackend, 2, Int>::from_ints(
                        [[1, 2, 3, 4], [5, 6, 0, 0]],
                        &device,
                    ),
                    token_mask: Some(TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
                        [[true, true, true, true], [true, true, false, false]].into(),
                        &device,
                    ))),
                    position_ids: None,
                },
                &mut state,
                &last,
            )
            .expect("ragged prefill");
        let reference_prefill = Tensor::cat(
            vec![
                model
                    .forward(
                        CausalLmInput {
                            token_ids: Tensor::<TestBackend, 2, Int>::from_ints(
                                [[1, 2, 3, 4]],
                                &device,
                            ),
                            token_mask: None,
                            position_ids: None,
                        },
                        &last,
                    )
                    .expect("row zero reference")
                    .logits,
                model
                    .forward(
                        CausalLmInput {
                            token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[5, 6]], &device),
                            token_mask: None,
                            position_ids: None,
                        },
                        &last,
                    )
                    .expect("row one reference")
                    .logits,
            ],
            0,
        );
        assert_close(prefill.logits, reference_prefill);
        assert_eq!(state.lengths(), &[4, 2]);

        let decoded = model
            .decode(
                DecodeInput {
                    token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[7, 8], [9, 0]], &device),
                    position_ids: None,
                    token_mask: Some(TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
                        [[true, true], [true, false]].into(),
                        &device,
                    ))),
                },
                &mut state,
                &last,
            )
            .expect("ragged decode");
        let reference_decode = Tensor::cat(
            vec![
                model
                    .forward(
                        CausalLmInput {
                            token_ids: Tensor::<TestBackend, 2, Int>::from_ints(
                                [[1, 2, 3, 4, 7, 8]],
                                &device,
                            ),
                            token_mask: None,
                            position_ids: None,
                        },
                        &last,
                    )
                    .expect("row zero continuation")
                    .logits,
                model
                    .forward(
                        CausalLmInput {
                            token_ids: Tensor::<TestBackend, 2, Int>::from_ints(
                                [[5, 6, 9]],
                                &device,
                            ),
                            token_mask: None,
                            position_ids: None,
                        },
                        &last,
                    )
                    .expect("row one continuation")
                    .logits,
            ],
            0,
        );
        assert_close(decoded.logits, reference_decode);
        assert_eq!(state.lengths(), &[6, 3]);
    }

    /// Explicit continuation positions equal the default per-sequence offsets.
    #[test]
    fn explicit_positions_match_implicit_decode_offsets() {
        let device = Default::default();
        let model = tiny_model(&device);
        let config = CacheConfig::new(2, 8).expect("cache config");
        let mut implicit = model
            .new_decode_state(config, 2, &device)
            .expect("implicit state");
        let mut explicit = model
            .new_decode_state(config, 2, &device)
            .expect("explicit state");
        let prompt = || CausalLmInput {
            token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[1, 2, 3], [4, 5, 6]], &device),
            token_mask: None,
            position_ids: None,
        };
        model
            .prefill(prompt(), &mut implicit, &ForwardOptions::default())
            .expect("implicit prefill");
        model
            .prefill(prompt(), &mut explicit, &ForwardOptions::default())
            .expect("explicit prefill");
        let implicit_output = model
            .decode(
                DecodeInput {
                    token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[7, 8], [9, 10]], &device),
                    position_ids: None,
                    token_mask: None,
                },
                &mut implicit,
                &diagnostic_options(),
            )
            .expect("implicit decode");
        let explicit_output = model
            .decode(
                DecodeInput {
                    token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[7, 8], [9, 10]], &device),
                    position_ids: Some(Tensor::<TestBackend, 2, Int>::from_ints(
                        [[3, 4], [3, 4]],
                        &device,
                    )),
                    token_mask: None,
                },
                &mut explicit,
                &diagnostic_options(),
            )
            .expect("explicit decode");
        assert_close(implicit_output.logits, explicit_output.logits);
        assert_close(
            implicit_output.final_hidden_state.expect("implicit hidden"),
            explicit_output.final_hidden_state.expect("explicit hidden"),
        );
    }

    /// Capacity failure is detected before any logical or tensor state changes.
    #[test]
    fn capacity_boundary_is_preflighted_atomically() {
        let device = Default::default();
        let model = tiny_model(&device);
        let mut state = model
            .new_decode_state(CacheConfig::new(1, 3).expect("cache config"), 1, &device)
            .expect("decode state");
        model
            .prefill(
                CausalLmInput {
                    token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[1, 2, 3]], &device),
                    token_mask: None,
                    position_ids: None,
                },
                &mut state,
                &ForwardOptions::default(),
            )
            .expect("exact-capacity prefill");
        let keys_before = state
            .tensor_state()
            .main()
            .native_kv(0)
            .expect("KV group")
            .keys(0)
            .expect("KV layer")
            .clone()
            .into_data();
        let matrix_before = state
            .tensor_state()
            .main()
            .recurrent(0)
            .expect("recurrent group")
            .matrix(0)
            .expect("recurrent layer")
            .clone()
            .into_data();
        let error = model
            .decode(
                DecodeInput {
                    token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[4]], &device),
                    position_ids: None,
                    token_mask: None,
                },
                &mut state,
                &ForwardOptions::default(),
            )
            .expect_err("capacity overflow must fail");
        assert!(matches!(
            error,
            Error::TensorCache(TensorCacheError::Metadata(
                DecodeStateError::CapacityExceeded { .. }
            ))
        ));
        assert_eq!(state.lengths(), &[3]);
        assert_eq!(
            state
                .tensor_state()
                .main()
                .native_kv(0)
                .expect("KV group")
                .keys(0)
                .expect("KV layer")
                .clone()
                .into_data(),
            keys_before
        );
        assert_eq!(
            state
                .tensor_state()
                .main()
                .recurrent(0)
                .expect("recurrent group")
                .matrix(0)
                .expect("recurrent layer")
                .clone()
                .into_data(),
            matrix_before
        );
    }

    /// Fully masked rows fail before cache mutation because they have no last logit.
    #[test]
    fn fully_masked_row_is_rejected_without_advancing_state() {
        let device = Default::default();
        let model = tiny_model(&device);
        let mut state = model
            .new_decode_state(CacheConfig::new(1, 4).expect("cache config"), 1, &device)
            .expect("decode state");
        let error = model
            .prefill(
                CausalLmInput {
                    token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[0, 0]], &device),
                    token_mask: Some(TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
                        [[false, false]].into(),
                        &device,
                    ))),
                    position_ids: None,
                },
                &mut state,
                &ForwardOptions::default(),
            )
            .expect_err("fully masked row must fail");
        assert!(matches!(error, Error::Shape { .. }));
        assert_eq!(state.lengths(), &[0]);
    }

    /// The cached mask exposes every old key and only the causal current prefix.
    #[test]
    fn cached_causal_mask_uses_bottom_right_alignment() {
        let device = Default::default();
        let mask = cached_causal_mask::<TestBackend>(2, 3, 1, &device)
            .expect("causal geometry")
            .into_data();
        assert_eq!(
            mask.as_slice::<bool>().expect("bool mask"),
            &[
                false, false, false, true, true, // query at absolute position 2
                false, false, false, false, true, // query at absolute position 3
                false, false, false, false, false, // query at absolute position 4
            ]
        );
    }

    /// Scatter semantics preserve valid order and define padding outputs as zero.
    #[test]
    fn compact_scatter_restores_rectangular_positions() {
        let device = Default::default();
        let compact = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(vec![1.0_f32, 2.0, 3.0, 4.0], [1, 2, 2]),
            &device,
        );
        let actual = scatter_compact(compact, 4, &[0, 2])
            .expect("valid scatter")
            .into_data();
        assert_eq!(
            actual.as_slice::<f32>().expect("f32 output"),
            &[1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0]
        );
    }
}
