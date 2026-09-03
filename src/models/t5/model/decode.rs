//! Cache-aware T5/FLAN decoder execution.
//!
//! T5 decoding retains two mathematically different kinds of state. Decoder
//! self-attention appends keys and values as the target prefix grows, so it uses
//! the crate's preallocated full-history KV cache. Encoder cross-attention is
//! immutable for a generation session: each decoder layer projects
//! `K = H_encoder W_K` and `V = H_encoder W_V` exactly once, then every target
//! step computes only its query projection.
//!
//! For a target chunk beginning at committed length `p`, self-attention uses
//! absolute query indices `p + i`, cached keys `0..p`, and current keys
//! `p..p+t`. Its exclusion mask is `M[i, j] = -inf` exactly when
//! `j > p + i`, and the first decoder block supplies the T5 bucketed bias
//! `b(p + i, j)` shared by later blocks. Scores intentionally remain
//! `Q K^T + b + M` without the usual `1 / sqrt(d_kv)` factor. Cross-attention
//! similarly evaluates `softmax(Q K_encoder^T + M_encoder) V_encoder`, but its
//! two encoder projections are immutable cache entries.
//!
//! Rectangular target batches are compacted row by row according to the public
//! `true == valid` mask convention. Invalid columns do not consume cache
//! capacity or relative positions. A fully masked row is accepted only during
//! decode, where it represents a finished sequence and produces exact zero
//! hidden states and logits without changing any state. All new self-attention
//! tensors remain operation-local until every layer, row, diagnostic, and
//! vocabulary projection has succeeded. The final publication validates every
//! write, advances all logical cursors atomically, then performs only infallible
//! tensor assignments.

use std::collections::BTreeSet;

use burn::{
    nn::Linear,
    tensor::{Bool, Int, Tensor, TensorData, backend::Backend},
};

use crate::{
    CausalLmInput, CausalLmOutput, DecodeInput, Error, ForwardOptions, Result,
    Seq2SeqLanguageModel,
    architecture::{
        DecodeState as ArchitectureDecodeState, EncoderOutput, Seq2SeqInput, Seq2SeqOutput,
        TokenMask,
    },
    attention::{AttentionProjection, padding_attention_mask},
    cache::{
        CacheConfig, DecodeStateError, KvCacheStorage, NativeKvCacheSpec, TensorCacheError,
        TensorDecodeState,
    },
    operations,
};

use super::{
    T5Block, T5ForConditionalGeneration, T5Stack, diagnostic_buffer, select_hidden,
    validate_encoder_output, validate_options,
};
use crate::models::t5::T5Config;

/// Index of T5's sole homogeneous decoder self-attention cache group.
const SELF_KV_GROUP: usize = 0;

/// Stateful operation whose empty-row policy differs by lifecycle phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StatefulPhase {
    /// An initial prompt must contribute at least one token in every row.
    Prefill,
    /// Finished sequences may contribute no tokens while peers continue.
    Decode,
}

impl StatefulPhase {
    /// Returns whether a fully masked input row is a valid no-op.
    const fn permits_inactive_rows(self) -> bool {
        matches!(self, Self::Decode)
    }
}

/// Encoder data and its per-layer immutable cross-attention projections.
///
/// The retained [`EncoderOutput`] defines the source binding for the lifetime of
/// the state. Subsequent decode calls accept the trait-mandated encoder argument
/// but consume this bound representation, which has already been reordered with
/// the cache during beam operations.
#[derive(Debug, Clone)]
struct T5CrossAttentionCache<B: Backend> {
    encoder: EncoderOutput<B>,
    keys: Vec<Tensor<B, 4>>,
    values: Vec<Tensor<B, 4>>,
}

impl<B: Backend> T5CrossAttentionCache<B> {
    /// Builds all decoder-layer cross projections without mutating decode state.
    fn project(model: &T5ForConditionalGeneration<B>, encoder: &EncoderOutput<B>) -> Result<Self> {
        let [batch, source_sequence, _] = encoder.hidden_state.dims();
        validate_encoder_output(encoder, batch, model.config.d_model)?;
        let mut keys = Vec::with_capacity(model.decoder.block.len());
        let mut values = Vec::with_capacity(model.decoder.block.len());
        for block in &model.decoder.block {
            let cross = block.cross_attention.as_ref().ok_or_else(|| {
                Error::configuration(
                    "t5.decoder.cross_attention",
                    "decoder block has no encoder-decoder attention module",
                )
            })?;
            let attention = &cross.enc_dec_attention.attention;
            let key = attention
                .k_proj
                .forward(encoder.hidden_state.clone())
                .reshape([
                    batch,
                    source_sequence,
                    attention.key_value_heads,
                    attention.head_dim,
                ])
                .swap_dims(1, 2);
            let value = attention
                .v_proj
                .forward(encoder.hidden_state.clone())
                .reshape([
                    batch,
                    source_sequence,
                    attention.key_value_heads,
                    attention.head_dim,
                ])
                .swap_dims(1, 2);
            keys.push(key);
            values.push(value);
        }
        Ok(Self {
            encoder: encoder.clone(),
            keys,
            values,
        })
    }

    /// Returns the number of independently projected decoder layers.
    fn layers(&self) -> usize {
        self.keys.len()
    }

    /// Returns the retained source sequence length.
    fn source_sequence(&self) -> usize {
        self.encoder.hidden_state.dims()[1]
    }

    /// Returns one row of a decoder layer's projected encoder keys and values.
    fn layer_row(&self, layer: usize, sequence: usize) -> Result<(Tensor<B, 4>, Tensor<B, 4>)> {
        let layers = self.layers();
        let key = self.keys.get(layer).ok_or_else(|| {
            Error::Cache(format!(
                "T5 cross-attention layer {layer} is outside {layers} cached layers"
            ))
        })?;
        let value = self.values.get(layer).ok_or_else(|| {
            Error::Cache(format!(
                "T5 cross-attention value layer {layer} is outside {layers} cached layers"
            ))
        })?;
        let batch = key.dims()[0];
        if sequence >= batch {
            return Err(Error::Cache(format!(
                "T5 cross-attention row {sequence} is outside batch {batch}"
            )));
        }
        Ok((
            key.clone().slice([
                sequence..sequence + 1,
                0..key.dims()[1],
                0..key.dims()[2],
                0..key.dims()[3],
            ]),
            value.clone().slice([
                sequence..sequence + 1,
                0..value.dims()[1],
                0..value.dims()[2],
                0..value.dims()[3],
            ]),
        ))
    }

    /// Returns one row of the source-validity mask, preserving gapped padding.
    fn mask_row(&self, sequence: usize) -> Option<TokenMask<B>> {
        self.encoder.token_mask.as_ref().map(|mask| {
            let width = mask.as_tensor().dims()[1];
            TokenMask::new(
                mask.as_tensor()
                    .clone()
                    .slice([sequence..sequence + 1, 0..width]),
            )
        })
    }

    /// Produces a reordered cache without mutating the current binding.
    fn reordered(&self, indices: &[usize]) -> Result<Self> {
        let selector = batch_selector::<B>(indices, &self.encoder.hidden_state.device())?;
        let encoder =
            EncoderOutput {
                hidden_state: self
                    .encoder
                    .hidden_state
                    .clone()
                    .select(0, selector.clone()),
                token_mask: self.encoder.token_mask.as_ref().map(|mask| {
                    TokenMask::new(mask.as_tensor().clone().select(0, selector.clone()))
                }),
                hidden_states: self
                    .encoder
                    .hidden_states
                    .iter()
                    .map(|hidden| hidden.clone().select(0, selector.clone()))
                    .collect(),
                attentions: self
                    .encoder
                    .attentions
                    .iter()
                    .map(|attention| attention.clone().select(0, selector.clone()))
                    .collect(),
            };
        Ok(Self {
            encoder,
            keys: self
                .keys
                .iter()
                .map(|key| key.clone().select(0, selector.clone()))
                .collect(),
            values: self
                .values
                .iter()
                .map(|value| value.clone().select(0, selector.clone()))
                .collect(),
        })
    }
}

/// T5-specific external state for self- and cross-attention decoding.
///
/// Decoder self-attention storage is allocated at construction with shape
/// `[layers, batch, heads, capacity, d_kv]`. Cross-attention storage is created
/// once at the first successful prefill because source length is not part of
/// [`CacheConfig`]; its fixed post-binding shape is
/// `[layers, batch, heads, source, d_kv]`.
#[derive(Debug)]
pub struct T5DecodeState<B: Backend> {
    tensors: TensorDecodeState<B>,
    self_spec: NativeKvCacheSpec,
    cross: Option<T5CrossAttentionCache<B>>,
}

impl<B: Backend> T5DecodeState<B> {
    /// Allocates a decode state whose layer and head geometry matches `model`.
    fn new(
        model: &T5ForConditionalGeneration<B>,
        config: CacheConfig,
        batch_size: usize,
        device: &B::Device,
    ) -> Result<Self> {
        if config.capacity() > i32::MAX as usize {
            return Err(Error::configuration(
                "t5.decode_state.capacity",
                "must fit Burn's portable signed position-index range",
            ));
        }
        let self_spec = NativeKvCacheSpec::new(
            model.config.decoder_layers(),
            model.config.num_heads,
            model.config.d_kv,
            KvCacheStorage::Full,
            model.shared.weight.val().dtype(),
        )?;
        let mut tensors = TensorDecodeState::new(config, batch_size, device)?;
        let group = tensors.main_mut().add_native_kv(self_spec)?;
        if group != SELF_KV_GROUP {
            return Err(Error::Cache(format!(
                "T5 self-KV group index must be {SELF_KV_GROUP}, found {group}"
            )));
        }
        Ok(Self {
            tensors,
            self_spec,
            cross: None,
        })
    }

    /// Borrows the generic tensor state for inspection and memory accounting.
    #[must_use]
    pub const fn tensor_state(&self) -> &TensorDecodeState<B> {
        &self.tensors
    }

    /// Reports whether a successful prefill has bound encoder projections.
    #[must_use]
    pub const fn has_encoder_cache(&self) -> bool {
        self.cross.is_some()
    }

    /// Returns the bound source length after prefill.
    #[must_use]
    pub fn encoder_sequence_length(&self) -> Option<usize> {
        self.cross
            .as_ref()
            .map(T5CrossAttentionCache::source_sequence)
    }

    /// Verifies that model topology and physical cache geometry still agree.
    fn validate_for_model(&self, model: &T5ForConditionalGeneration<B>) -> Result<()> {
        let expected = NativeKvCacheSpec::new(
            model.config.decoder_layers(),
            model.config.num_heads,
            model.config.d_kv,
            KvCacheStorage::Full,
            model.shared.weight.val().dtype(),
        )?;
        if expected != self.self_spec
            || self.tensors.main().native_kv_count() != 1
            || self.tensors.main().recurrent_count() != 0
            || self.tensors.main().native_kv(SELF_KV_GROUP)?.spec() != expected
        {
            return Err(Error::Cache(
                "T5 decode state was created for an incompatible model layout".into(),
            ));
        }
        if let Some(cross) = &self.cross
            && cross.layers() != model.config.decoder_layers()
        {
            return Err(Error::Cache(format!(
                "T5 cross cache has {} layers for a {}-layer decoder",
                cross.layers(),
                model.config.decoder_layers()
            )));
        }
        Ok(())
    }

    /// Validates that an encoder argument is structurally compatible with its binding.
    ///
    /// Burn's backend-generic tensor interface exposes no stable allocation ID.
    /// Once bound, cached K/V and the retained encoder output are therefore the
    /// source of truth; this check rejects device, batch, source-length, width,
    /// and mask-shape changes without synchronizing all encoder values to the CPU.
    fn validate_encoder_binding(&self, encoder: &EncoderOutput<B>) -> Result<()> {
        let cross = self.cross.as_ref().ok_or_else(|| {
            Error::Cache("T5 decode requires a successful decoder prefill".into())
        })?;
        let expected = cross.encoder.hidden_state.dims();
        if encoder.hidden_state.dims() != expected {
            return Err(Error::shape(
                "t5.encoder_output.hidden_state",
                format!("{expected:?} bound to decoder state"),
                format!("{:?}", encoder.hidden_state.dims()),
            ));
        }
        ensure_device(
            "t5.encoder_output.hidden_state",
            &encoder.hidden_state.device(),
            self.tensors.device(),
        )?;
        let expected_mask = cross
            .encoder
            .token_mask
            .as_ref()
            .map(|mask| mask.as_tensor().dims());
        let actual_mask = encoder
            .token_mask
            .as_ref()
            .map(|mask| mask.as_tensor().dims());
        if expected_mask != actual_mask {
            return Err(Error::shape(
                "t5.encoder_output.token_mask",
                format!("{expected_mask:?} bound to decoder state"),
                format!("{actual_mask:?}"),
            ));
        }
        if let Some(mask) = &encoder.token_mask {
            ensure_device(
                "t5.encoder_output.token_mask",
                &mask.as_tensor().device(),
                self.tensors.device(),
            )?;
        }
        Ok(())
    }
}

impl<B: Backend> ArchitectureDecodeState<B> for T5DecodeState<B> {
    /// Returns the fixed number of independently tracked target sequences.
    fn batch_size(&self) -> usize {
        self.tensors.batch_size()
    }

    /// Returns the maximum number of valid target tokens retained per row.
    fn capacity(&self) -> usize {
        self.tensors.capacity()
    }

    /// Returns the committed valid-token count of every target row.
    fn lengths(&self) -> &[usize] {
        self.tensors.lengths()
    }

    /// Clears target history and releases the immutable encoder binding.
    fn reset(&mut self) -> Result<()> {
        self.tensors.reset()?;
        self.cross = None;
        Ok(())
    }

    /// Reorders or duplicates beams across self-KV, cross-KV, and encoder data.
    fn reorder(&mut self, indices: &[usize]) -> Result<()> {
        validate_reorder(indices, self.tensors.batch_size())?;
        let reordered_cross = self
            .cross
            .as_ref()
            .map(|cross| cross.reordered(indices))
            .transpose()?;
        self.tensors.reorder(indices)?;
        self.cross = reordered_cross;
        Ok(())
    }

    /// Rolls logical target lengths backward while retaining reusable allocation.
    fn truncate(&mut self, lengths: &[usize]) -> Result<()> {
        self.tensors.truncate(lengths).map_err(Into::into)
    }

    /// Returns the device on which every state tensor is allocated.
    fn device(&self) -> &B::Device {
        self.tensors.device()
    }
}

impl<B: Backend> Seq2SeqLanguageModel<B> for T5ForConditionalGeneration<B> {
    type Config = T5Config;
    type State = T5DecodeState<B>;

    /// Evaluates the bidirectional encoder without allocating decoder state.
    fn encode(
        &self,
        source: CausalLmInput<B>,
        options: &ForwardOptions,
    ) -> Result<EncoderOutput<B>> {
        T5ForConditionalGeneration::encode(self, source, options)
    }

    /// Evaluates the complete stateless encoder-decoder graph.
    fn forward(
        &self,
        input: Seq2SeqInput<B>,
        options: &ForwardOptions,
    ) -> Result<Seq2SeqOutput<B>> {
        T5ForConditionalGeneration::forward(self, input, options)
    }

    /// Preallocates full-history self-KV storage for every decoder layer.
    fn new_decode_state(
        &self,
        config: CacheConfig,
        batch_size: usize,
        device: &B::Device,
    ) -> Result<Self::State> {
        T5DecodeState::new(self, config, batch_size, device)
    }

    /// Binds projected encoder K/V and appends the initial target prefix.
    fn prefill_decoder(
        &self,
        encoder: &EncoderOutput<B>,
        decoder: CausalLmInput<B>,
        state: &mut Self::State,
        options: &ForwardOptions,
    ) -> Result<Seq2SeqOutput<B>> {
        self.forward_with_decode_state(encoder, decoder, state, options, StatefulPhase::Prefill)
    }

    /// Appends a target segment while reusing all committed attention state.
    fn decode(
        &self,
        encoder: &EncoderOutput<B>,
        input: DecodeInput<B>,
        state: &mut Self::State,
        options: &ForwardOptions,
    ) -> Result<Seq2SeqOutput<B>> {
        self.forward_with_decode_state(
            encoder,
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

impl<B: Backend> T5ForConditionalGeneration<B> {
    /// Evaluates a masked stateless target with cache-compatible token semantics.
    ///
    /// Padding columns are removed before attention so they neither consume a
    /// relative position nor enter a causal key history. Valid results are then
    /// scattered back into the original rectangle. This matches stateful
    /// prefill for left, right, and gapped masks while preserving a fully
    /// differentiable tensor graph for all parameterized operations.
    pub(super) fn decode_stateless_masked(
        &self,
        encoder: &EncoderOutput<B>,
        decoder: CausalLmInput<B>,
        options: &ForwardOptions,
    ) -> Result<CausalLmOutput<B>> {
        let [batch, sequence] = decoder.token_ids.dims();
        let valid_indices = semantic_indices(decoder.token_mask.as_ref(), batch, sequence)?;
        let diagnostic_count = self.decoder.block.len() + 2;
        let mut final_rows = Vec::with_capacity(batch);
        let mut diagnostic_rows: Vec<Vec<Tensor<B, 3>>> = if options.hidden_states {
            (0..diagnostic_count)
                .map(|_| Vec::with_capacity(batch))
                .collect()
        } else {
            Vec::new()
        };
        for (row, valid) in valid_indices.iter().enumerate() {
            if valid.is_empty() {
                return Err(Error::shape(
                    "t5.decoder.token_mask",
                    "at least one valid token in every stateless row",
                    "a fully masked row",
                ));
            }
            let selector = sequence_selector::<B>(valid, &decoder.token_ids.device())?;
            let tokens = decoder
                .token_ids
                .clone()
                .slice([row..row + 1, 0..sequence])
                .select(1, selector);
            let hidden = self.shared.forward(tokens);
            let source = encoder.hidden_state.clone().slice([
                row..row + 1,
                0..encoder.hidden_state.dims()[1],
                0..encoder.hidden_state.dims()[2],
            ]);
            let source_mask = encoder_mask_row(encoder, row);
            let output = self.decoder.forward_decoder(
                hidden,
                None,
                source,
                source_mask.as_ref(),
                options.hidden_states,
            )?;
            final_rows.push(scatter_compact(output.hidden, sequence, valid)?);
            if options.hidden_states {
                if output.hidden_states.len() != diagnostic_count {
                    return Err(Error::Cache(format!(
                        "T5 stateless hidden-state count must be {diagnostic_count}, found {}",
                        output.hidden_states.len()
                    )));
                }
                for (destination, compact) in diagnostic_rows.iter_mut().zip(output.hidden_states) {
                    destination.push(scatter_compact(compact, sequence, valid)?);
                }
            }
        }
        let hidden = Tensor::cat(final_rows, 0);
        let selected = select_hidden(hidden.clone(), decoder.token_mask.as_ref(), &options.logits)?;
        Ok(CausalLmOutput {
            logits: self.project_logits(selected)?,
            final_hidden_state: options.hidden_states.then_some(hidden),
            hidden_states: diagnostic_rows
                .into_iter()
                .map(|rows| Tensor::cat(rows, 0))
                .collect(),
            attentions: Vec::new(),
            routers: Vec::new(),
        })
    }

    /// Evaluates one stateful target segment and publishes it atomically.
    fn forward_with_decode_state(
        &self,
        encoder: &EncoderOutput<B>,
        decoder: CausalLmInput<B>,
        state: &mut T5DecodeState<B>,
        options: &ForwardOptions,
        phase: StatefulPhase,
    ) -> Result<Seq2SeqOutput<B>> {
        validate_options(options)?;
        state.validate_for_model(self)?;
        if state.tensors.has_active_transaction() {
            return Err(TensorCacheError::TransactionActive {
                operation: "run ordinary T5 prefill/decode",
            }
            .into());
        }
        let prepared = prepare_stateful_input(self, encoder, &decoder, state, phase)?;
        validate_logits_selection(&options.logits, prepared.sequence)?;

        let pending_cross = if state.cross.is_none() {
            if phase == StatefulPhase::Decode {
                return Err(Error::Cache(
                    "T5 decode requires a successful decoder prefill".into(),
                ));
            }
            Some(T5CrossAttentionCache::project(self, encoder)?)
        } else {
            None
        };
        let cross = pending_cross
            .as_ref()
            .or(state.cross.as_ref())
            .ok_or_else(|| Error::Cache("T5 encoder cross cache is unavailable".into()))?;

        let device = decoder.token_ids.device();
        let diagnostic_count = self.decoder.block.len() + 2;
        let mut pending_self = PendingT5SelfWrites::new();
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
            let selector = sequence_selector::<B>(valid, &device)?;
            let tokens = decoder
                .token_ids
                .clone()
                .slice([sequence_index..sequence_index + 1, 0..prepared.sequence])
                .select(1, selector);
            let baseline = state.tensors.lengths()[sequence_index];
            let (compact, diagnostics) = self.decoder.forward_cached_row(
                self,
                tokens,
                sequence_index,
                baseline,
                &state.tensors,
                cross,
                &mut pending_self,
                options.hidden_states,
            )?;
            final_rows.push(scatter_compact(compact, prepared.sequence, valid)?);
            if options.hidden_states {
                if diagnostics.len() != diagnostic_count {
                    return Err(Error::Cache(format!(
                        "T5 hidden-state count must be {diagnostic_count}, found {}",
                        diagnostics.len()
                    )));
                }
                for (destination, compact_state) in diagnostic_rows.iter_mut().zip(diagnostics) {
                    destination.push(scatter_compact(compact_state, prepared.sequence, valid)?);
                }
            }
        }

        let hidden = Tensor::cat(final_rows, 0);
        let hidden_states = diagnostic_rows
            .into_iter()
            .map(|rows| Tensor::cat(rows, 0))
            .collect();
        let selected = select_hidden(hidden.clone(), decoder.token_mask.as_ref(), &options.logits)?;
        let logits = self.project_logits(selected)?;
        let bound_encoder = cross.encoder.clone();
        pending_self.commit(
            &mut state.tensors,
            self.config.decoder_layers(),
            &prepared.token_counts,
        )?;
        if let Some(cross) = pending_cross {
            state.cross = Some(cross);
        }
        Ok(Seq2SeqOutput {
            decoder: CausalLmOutput {
                logits,
                final_hidden_state: options.hidden_states.then_some(hidden),
                hidden_states,
                attentions: Vec::new(),
                routers: Vec::new(),
            },
            encoder: bound_encoder,
            cross_attentions: Vec::new(),
        })
    }
}

impl<B: Backend> T5Stack<B> {
    /// Evaluates one compact target row using cached self- and cross-attention.
    #[allow(clippy::too_many_arguments)]
    fn forward_cached_row(
        &self,
        model: &T5ForConditionalGeneration<B>,
        token_ids: Tensor<B, 2, Int>,
        sequence_index: usize,
        baseline: usize,
        state: &TensorDecodeState<B>,
        cross: &T5CrossAttentionCache<B>,
        pending: &mut PendingT5SelfWrites<B>,
        retain_hidden_states: bool,
    ) -> Result<(Tensor<B, 3>, Vec<Tensor<B, 3>>)> {
        if !self.is_decoder {
            return Err(Error::configuration(
                "t5.decoder",
                "cannot run cached decoding through an encoder stack",
            ));
        }
        let query_sequence = token_ids.dims()[1];
        let key_sequence = baseline.checked_add(query_sequence).ok_or_else(|| {
            Error::configuration("t5.decoder.position_bias", "key length overflows usize")
        })?;
        let position_bias = self
            .block
            .first()
            .ok_or_else(|| Error::configuration("t5.decoder.block", "stack must be non-empty"))?
            .self_attention
            .self_attention
            .position_bias(query_sequence, key_sequence, baseline, &token_ids.device())?;
        let mut hidden = model.shared.forward(token_ids);
        let mut hidden_states = diagnostic_buffer(self.block.len(), retain_hidden_states);
        if retain_hidden_states {
            hidden_states.push(hidden.clone());
        }
        for (layer_index, block) in self.block.iter().enumerate() {
            hidden = block.forward_cached(
                hidden,
                position_bias.clone(),
                sequence_index,
                baseline,
                state,
                cross,
                layer_index,
                pending,
            )?;
            if retain_hidden_states {
                hidden_states.push(hidden.clone());
            }
        }
        hidden = self.final_layer_norm.forward(hidden);
        if retain_hidden_states {
            hidden_states.push(hidden.clone());
        }
        Ok((hidden, hidden_states))
    }
}

impl<B: Backend> T5Block<B> {
    /// Applies cached causal self-attention, cached cross-attention, and the FFN.
    #[allow(clippy::too_many_arguments)]
    fn forward_cached(
        &self,
        hidden: Tensor<B, 3>,
        position_bias: Tensor<B, 4>,
        sequence_index: usize,
        baseline: usize,
        state: &TensorDecodeState<B>,
        cross: &T5CrossAttentionCache<B>,
        layer_index: usize,
        pending: &mut PendingT5SelfWrites<B>,
    ) -> Result<Tensor<B, 3>> {
        if !self.self_attention.causal {
            return Err(Error::configuration(
                "t5.decoder.block",
                "decoder block was initialized with bidirectional self-attention",
            ));
        }
        let residual = hidden.clone();
        let normalized = self.self_attention.layer_norm.forward(hidden);
        let attention = &self.self_attention.self_attention.attention;
        let mut projection = attention.project_self(normalized)?;
        let new_keys = projection.key.clone();
        let new_values = projection.value.clone();
        let query_sequence = projection.query.dims()[2];
        let cache = state.main().native_kv(SELF_KV_GROUP)?;
        let history = cache.read_sequence(layer_index, sequence_index, baseline)?;
        if history.retained_positions() != (0..baseline) {
            return Err(Error::Cache(
                "T5 self-attention cache evicted required global history".into(),
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
        let hidden =
            residual + attention.attend(projection, Some(mask), Some(position_bias), false)?;
        pending.writes.push(PendingT5SelfWrite {
            layer: layer_index,
            sequence: sequence_index,
            start: baseline,
            keys: new_keys,
            values: new_values,
        });

        let residual = hidden.clone();
        let cross_layer = self.cross_attention.as_ref().ok_or_else(|| {
            Error::configuration(
                "t5.decoder.cross_attention",
                "decoder block has no encoder-decoder attention module",
            )
        })?;
        let cross_attention = &cross_layer.enc_dec_attention.attention;
        let query = project_query(
            &cross_attention.q_proj,
            cross_layer.layer_norm.forward(hidden),
            cross_attention.query_heads,
            cross_attention.head_dim,
        )?;
        let (key, value) = cross.layer_row(layer_index, sequence_index)?;
        let source_mask = cross.mask_row(sequence_index);
        let cross_mask = padding_attention_mask(
            None,
            source_mask.as_ref(),
            1,
            cross_attention.query_heads,
            query_sequence,
            cross.source_sequence(),
        )?;
        let hidden = residual
            + cross_attention.attend(
                AttentionProjection {
                    query,
                    key,
                    value,
                    gate: None,
                },
                cross_mask,
                None,
                false,
            )?;
        self.feed_forward.forward(hidden)
    }
}

/// Projects T5 queries without redundantly recomputing cached encoder K/V.
fn project_query<B: Backend>(
    projection: &Linear<B>,
    hidden: Tensor<B, 3>,
    heads: usize,
    head_dim: usize,
) -> Result<Tensor<B, 4>> {
    let [batch, sequence, _] = hidden.dims();
    Ok(operations::linear(hidden, projection)?
        .reshape([batch, sequence, heads, head_dim])
        .swap_dims(1, 2))
}

/// One operation-local self-attention append.
struct PendingT5SelfWrite<B: Backend> {
    layer: usize,
    sequence: usize,
    start: usize,
    keys: Tensor<B, 4>,
    values: Tensor<B, 4>,
}

/// Deferred self-attention writes for one public prefill or decode call.
struct PendingT5SelfWrites<B: Backend> {
    writes: Vec<PendingT5SelfWrite<B>>,
}

impl<B: Backend> PendingT5SelfWrites<B> {
    /// Creates an empty write set.
    fn new() -> Self {
        Self { writes: Vec::new() }
    }

    /// Validates exact coverage and atomically publishes lengths plus tensors.
    fn commit(
        self,
        state: &mut TensorDecodeState<B>,
        layers: usize,
        token_counts: &[usize],
    ) -> Result<()> {
        self.validate_coverage(state, layers, token_counts)?;
        let cache = state.main().native_kv(SELF_KV_GROUP)?;
        let writes = self
            .writes
            .into_iter()
            .map(|write| {
                cache.prepare_sequence_write(
                    write.layer,
                    write.sequence,
                    write.start,
                    write.keys,
                    write.values,
                )
            })
            .collect::<core::result::Result<Vec<_>, _>>()?;
        let prepared = state
            .main()
            .prepare_direct_writes(Some((SELF_KV_GROUP, writes)), None)?;
        state.advance(token_counts)?;
        state.main_mut().apply_prepared_direct_writes(prepared);
        Ok(())
    }

    /// Ensures every active row produced exactly one append for every layer.
    fn validate_coverage(
        &self,
        state: &TensorDecodeState<B>,
        layers: usize,
        token_counts: &[usize],
    ) -> Result<()> {
        let batch = state.batch_size();
        if token_counts.len() != batch {
            return Err(Error::Cache(format!(
                "T5 pending writes describe {} rows for batch {batch}",
                token_counts.len()
            )));
        }
        let active = token_counts.iter().filter(|&&count| count > 0).count();
        let expected = layers.checked_mul(active).ok_or_else(|| {
            Error::configuration(
                "t5.cache_commit",
                "self-attention write count overflows usize",
            )
        })?;
        if self.writes.len() != expected {
            return Err(Error::Cache(format!(
                "T5 pending self-cache coverage is {}/{expected}",
                self.writes.len()
            )));
        }
        let mut seen = BTreeSet::new();
        for write in &self.writes {
            let count = token_counts.get(write.sequence).copied().ok_or_else(|| {
                Error::Cache(format!(
                    "T5 self-cache write references row {} outside batch {batch}",
                    write.sequence
                ))
            })?;
            if count == 0
                || write.layer >= layers
                || write.start != state.lengths()[write.sequence]
                || write.keys.dims()[2] != count
                || write.values.dims()[2] != count
                || !seen.insert((write.layer, write.sequence))
            {
                return Err(Error::Cache(format!(
                    "invalid or duplicate T5 self-cache write for layer {}, row {}",
                    write.layer, write.sequence
                )));
            }
        }
        for (sequence, &count) in token_counts.iter().enumerate() {
            if count == 0 {
                continue;
            }
            for layer in 0..layers {
                if !seen.contains(&(layer, sequence)) {
                    return Err(Error::Cache(format!(
                        "missing T5 self-cache write for layer {layer}, row {sequence}"
                    )));
                }
            }
        }
        Ok(())
    }
}

/// Validated rectangular metadata for a stateful target operation.
struct PreparedStatefulInput {
    batch: usize,
    sequence: usize,
    valid_indices: Vec<Vec<usize>>,
    token_counts: Vec<usize>,
}

/// Validates state, encoder, tokens, masks, devices, and capacity before math.
fn prepare_stateful_input<B: Backend>(
    model: &T5ForConditionalGeneration<B>,
    encoder: &EncoderOutput<B>,
    decoder: &CausalLmInput<B>,
    state: &T5DecodeState<B>,
    phase: StatefulPhase,
) -> Result<PreparedStatefulInput> {
    let [batch, sequence] = decoder.token_ids.dims();
    if batch == 0 || sequence == 0 {
        return Err(Error::shape(
            "t5.decoder.input_ids",
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
        "t5.decoder.input_ids",
        &decoder.token_ids.device(),
        state.tensors.device(),
    )?;
    ensure_device(
        "t5.model",
        &model.shared.weight.val().device(),
        state.tensors.device(),
    )?;
    if state.cross.is_some() {
        state.validate_encoder_binding(encoder)?;
    } else {
        validate_encoder_output(encoder, batch, model.config.d_model)?;
        ensure_device(
            "t5.encoder_output.hidden_state",
            &encoder.hidden_state.device(),
            state.tensors.device(),
        )?;
        if let Some(mask) = &encoder.token_mask {
            ensure_device(
                "t5.encoder_output.token_mask",
                &mask.as_tensor().device(),
                state.tensors.device(),
            )?;
        }
    }
    if let Some(mask) = &decoder.token_mask {
        ensure_device(
            "t5.decoder.token_mask",
            &mask.as_tensor().device(),
            state.tensors.device(),
        )?;
        if mask.as_tensor().dims() != [batch, sequence] {
            return Err(Error::shape(
                "t5.decoder.token_mask",
                format!("[{batch}, {sequence}]"),
                format!("{:?}", mask.as_tensor().dims()),
            ));
        }
    }
    if let Some(positions) = &decoder.position_ids {
        ensure_device(
            "t5.decoder.position_ids",
            &positions.device(),
            state.tensors.device(),
        )?;
        if positions.dims() != [batch, sequence] {
            return Err(Error::shape(
                "t5.decoder.position_ids",
                format!("[{batch}, {sequence}]"),
                format!("{:?}", positions.dims()),
            ));
        }
        return Err(Error::BackendCapability(
            "T5 decoder uses relative positions and does not accept explicit position IDs".into(),
        ));
    }
    let valid_indices = semantic_indices(decoder.token_mask.as_ref(), batch, sequence)?;
    if !phase.permits_inactive_rows() && valid_indices.iter().any(Vec::is_empty) {
        return Err(Error::shape(
            "t5.decoder.token_mask",
            "at least one valid token in every prefill row",
            "a fully masked prompt row",
        ));
    }
    let token_counts = valid_indices.iter().map(Vec::len).collect::<Vec<_>>();
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
    }
    for (row, valid) in valid_indices.iter().enumerate() {
        if valid.is_empty() {
            continue;
        }
        let selector = sequence_selector::<B>(valid, state.tensors.device())?;
        let tokens = decoder
            .token_ids
            .clone()
            .slice([row..row + 1, 0..sequence])
            .select(1, selector);
        validate_active_token_ids(tokens, model.config.vocab_size)?;
    }
    Ok(PreparedStatefulInput {
        batch,
        sequence,
        valid_indices,
        token_counts,
    })
}

/// Validates selective-logit indices before any cache mutation.
fn validate_logits_selection(selection: &crate::LogitsSelection, sequence: usize) -> Result<()> {
    if let crate::LogitsSelection::Positions(positions) = selection
        && (positions.is_empty() || positions.iter().any(|position| *position >= sequence))
    {
        return Err(Error::shape(
            "forward_options.logits.positions",
            format!("a non-empty list of indices below {sequence}"),
            format!("{positions:?}"),
        ));
    }
    Ok(())
}

/// Reads a semantic mask into stable per-row valid-column indices.
fn semantic_indices<B: Backend>(
    mask: Option<&TokenMask<B>>,
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
        .try_into_data()
        .map_err(|error| {
            Error::BackendCapability(format!(
                "failed to read T5 decoder mask from the backend: {error}"
            ))
        })?
        .to_vec::<bool>()
        .map_err(|error| {
            Error::BackendCapability(format!("failed to read T5 decoder mask: {error}"))
        })?;
    Ok(values
        .chunks_exact(sequence)
        .map(|row| {
            row.iter()
                .enumerate()
                .filter_map(|(column, &valid)| valid.then_some(column))
                .collect()
        })
        .collect())
}

/// Returns one encoder mask row without changing `true == valid` semantics.
fn encoder_mask_row<B: Backend>(
    encoder: &EncoderOutput<B>,
    sequence: usize,
) -> Option<TokenMask<B>> {
    encoder.token_mask.as_ref().map(|mask| {
        let source = mask.as_tensor().dims()[1];
        TokenMask::new(
            mask.as_tensor()
                .clone()
                .slice([sequence..sequence + 1, 0..source]),
        )
    })
}

/// Validates only active token IDs so masked sentinels remain inert.
fn validate_active_token_ids<B: Backend>(
    token_ids: Tensor<B, 2, Int>,
    vocabulary: usize,
) -> Result<()> {
    use burn::tensor::ElementConversion;

    let minimum = token_ids
        .clone()
        .min()
        .try_into_scalar()
        .map_err(|error| {
            Error::BackendCapability(format!("failed to validate T5 token IDs: {error}"))
        })?
        .elem::<i64>();
    let maximum = token_ids
        .max()
        .try_into_scalar()
        .map_err(|error| {
            Error::BackendCapability(format!("failed to validate T5 token IDs: {error}"))
        })?
        .elem::<i64>();
    if minimum < 0 || maximum as usize >= vocabulary {
        return Err(Error::shape(
            "t5.decoder.input_ids",
            format!("active token values in 0..{vocabulary}"),
            format!("observed range {minimum}..={maximum}"),
        ));
    }
    Ok(())
}

/// Creates a checked selector for target sequence columns.
fn sequence_selector<B: Backend>(
    indices: &[usize],
    device: &B::Device,
) -> Result<Tensor<B, 1, Int>> {
    let values = indices
        .iter()
        .map(|&index| {
            i64::try_from(index).map_err(|_| {
                Error::configuration(
                    "t5.decoder.token_mask",
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

/// Creates a checked selector for cache batch reordering.
fn batch_selector<B: Backend>(indices: &[usize], device: &B::Device) -> Result<Tensor<B, 1, Int>> {
    let values = indices
        .iter()
        .map(|&index| {
            i64::try_from(index).map_err(|_| {
                Error::configuration(
                    "t5.decode_state.reorder",
                    format!("batch index {index} exceeds signed tensor-index range"),
                )
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Tensor::from_data(
        TensorData::new(values, [indices.len()]),
        device,
    ))
}

/// Validates a fixed-size batch permutation or duplication vector.
fn validate_reorder(indices: &[usize], batch: usize) -> Result<()> {
    if indices.len() != batch {
        return Err(Error::Cache(format!(
            "T5 reorder requires {batch} indices, found {}",
            indices.len()
        )));
    }
    if let Some(&index) = indices.iter().find(|&&index| index >= batch) {
        return Err(Error::Cache(format!(
            "T5 reorder index {index} is outside batch {batch}"
        )));
    }
    Ok(())
}

/// Builds an explicit cached causal mask `excluded(q, k) = k > baseline + q`.
fn cached_causal_mask<B: Backend>(
    baseline: usize,
    query_sequence: usize,
    heads: usize,
    device: &B::Device,
) -> Result<Tensor<B, 4, Bool>> {
    let key_sequence = baseline.checked_add(query_sequence).ok_or_else(|| {
        Error::configuration("t5.decoder.attention_mask", "key length overflows usize")
    })?;
    let query_start = i64::try_from(baseline).map_err(|_| {
        Error::configuration("t5.decoder.attention_mask", "query offset exceeds i64")
    })?;
    let end = i64::try_from(key_sequence)
        .map_err(|_| Error::configuration("t5.decoder.attention_mask", "key length exceeds i64"))?;
    let query = Tensor::<B, 1, Int>::arange(query_start..end, device)
        .reshape([1, 1, query_sequence, 1])
        .expand([1, heads, query_sequence, key_sequence]);
    let key = Tensor::<B, 1, Int>::arange(0..end, device)
        .reshape([1, 1, 1, key_sequence])
        .expand([1, heads, query_sequence, key_sequence]);
    Ok(key.greater(query))
}

/// Returns zero hidden states for a fully inactive decode row.
fn empty_hidden_row<B: Backend>(
    model: &T5ForConditionalGeneration<B>,
    sequence: usize,
    device: &B::Device,
) -> Tensor<B, 3> {
    Tensor::zeros(
        [1, sequence, model.config.d_model],
        (device, model.shared.weight.val().dtype()),
    )
}

/// Scatters compact active states into a zero-filled rectangular row.
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
            "t5.decoder.compact_hidden",
            "compact width matching sorted, unique in-range output indices",
            format!("compact={:?}, indices={indices:?}", compact.dims()),
        ));
    }
    let width = compact.dims()[2];
    let device = compact.device();
    let dtype = compact.dtype();
    let mut segments = Vec::with_capacity(indices.len().saturating_mul(2).saturating_add(1));
    let mut cursor = 0usize;
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
    if segments.is_empty() {
        return Err(Error::shape(
            "t5.decoder.compact_hidden",
            "at least one active output position",
            "an empty compact tensor",
        ));
    }
    Ok(Tensor::cat(segments, 1))
}

/// Checks that two backend-generic device values identify the same device.
fn ensure_device<D: PartialEq>(name: &str, actual: &D, expected: &D) -> Result<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(Error::shape(
            name,
            "the decode-state device",
            "another device",
        ))
    }
}

#[cfg(test)]
mod tests {
    use burn::{
        backend::Flex,
        module::Initializer,
        tensor::{Bool, Int, Tensor, TensorData, Tolerance, backend::BackendTypes},
    };
    use proptest::prelude::*;

    use super::*;
    use crate::{
        DecodeState, LogitsSelection,
        models::t5::{T5FeedForwardProjection, T5LayerCrossAttention},
    };

    /// Deterministic CPU backend used by cache parity tests.
    type TestBackend = Flex<f32, i32>;

    /// Device selected by the deterministic test backend.
    type TestDevice = <TestBackend as BackendTypes>::Device;

    /// Builds a small topology with configurable FFN and vocabulary-head layouts.
    fn tiny_config(projection: T5FeedForwardProjection, tied: bool, layers: usize) -> T5Config {
        T5Config {
            vocab_size: 24,
            d_model: 8,
            d_kv: 4,
            d_ff: 16,
            num_layers: layers,
            num_decoder_layers: Some(layers),
            num_heads: 2,
            relative_attention_num_buckets: 8,
            relative_attention_max_distance: 16,
            dropout_rate: 0.0,
            feed_forward_proj: projection,
            tie_word_embeddings: tied,
            ..T5Config::default()
        }
    }

    /// Encodes a fixed source prompt for one decoder row.
    fn encoded_source(
        model: &T5ForConditionalGeneration<TestBackend>,
        device: &TestDevice,
    ) -> EncoderOutput<TestBackend> {
        model
            .encode(
                CausalLmInput {
                    token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[8, 9, 10]], device),
                    token_mask: None,
                    position_ids: None,
                },
                &ForwardOptions::default(),
            )
            .expect("source encoding")
    }

    /// Creates an unmasked target segment from one row of IDs.
    fn target(ids: &[i32], device: &TestDevice) -> CausalLmInput<TestBackend> {
        CausalLmInput {
            token_ids: Tensor::from_data(TensorData::new(ids.to_vec(), [1, ids.len()]), device),
            token_mask: None,
            position_ids: None,
        }
    }

    /// Compares two floating tensors at the crate's F32 golden tolerance.
    fn assert_close<const D: usize>(left: Tensor<TestBackend, D>, right: Tensor<TestBackend, D>) {
        left.into_data()
            .assert_approx_eq::<f32>(&right.into_data(), Tolerance::absolute(1e-4));
    }

    /// Materializes all physical self-KV slots, including uncommitted zero suffixes.
    fn self_cache_snapshot(state: &T5DecodeState<TestBackend>) -> Vec<Vec<f32>> {
        let cache = state
            .tensors
            .main()
            .native_kv(SELF_KV_GROUP)
            .expect("self cache");
        let mut snapshot = Vec::new();
        for layer in 0..state.self_spec.layers() {
            for row in 0..state.tensors.batch_size() {
                let (keys, values) = cache
                    .read_sequence(layer, row, state.tensors.capacity())
                    .expect("physical cache read")
                    .materialize()
                    .expect("capacity is nonzero");
                snapshot.push(keys.into_data().to_vec::<f32>().expect("F32 keys"));
                snapshot.push(values.into_data().to_vec::<f32>().expect("F32 values"));
            }
        }
        snapshot
    }

    /// Cached prefill exactly reproduces stateless logits and diagnostics.
    #[test]
    fn prefill_matches_stateless_for_dense_gated_tied_and_untied_models() {
        let device = Default::default();
        for (projection, tied) in [
            (T5FeedForwardProjection::Relu, true),
            (T5FeedForwardProjection::GatedGelu, false),
            (T5FeedForwardProjection::GatedSilu, false),
        ] {
            let model = T5ForConditionalGeneration::<TestBackend>::init(
                tiny_config(projection, tied, 2),
                &device,
            )
            .expect("tiny model");
            let encoder = encoded_source(&model, &device);
            let decoder = target(&[0, 4, 5], &device);
            let options = ForwardOptions {
                logits: LogitsSelection::All,
                hidden_states: true,
                ..ForwardOptions::default()
            };
            let stateless = model
                .decode_stateless(&encoder, decoder.clone(), &options)
                .expect("stateless decoder");
            let mut state = model
                .new_decode_state(CacheConfig::new(1, 8).unwrap(), 1, &device)
                .expect("decode state");
            let cached = model
                .prefill_decoder(&encoder, decoder, &mut state, &options)
                .expect("cached prefill")
                .decoder;

            assert_eq!(state.lengths(), [3]);
            assert!(state.has_encoder_cache());
            assert_eq!(state.encoder_sequence_length(), Some(3));
            assert_close(cached.logits, stateless.logits);
            assert_close(
                cached.final_hidden_state.expect("cached final"),
                stateless.final_hidden_state.expect("stateless final"),
            );
            assert_eq!(cached.hidden_states.len(), stateless.hidden_states.len());
            for (cached, stateless) in cached
                .hidden_states
                .into_iter()
                .zip(stateless.hidden_states)
            {
                assert_close(cached, stateless);
            }
        }
    }

    /// Multi-token and recurrent decode agree with the matching stateless suffix.
    #[test]
    fn chunked_and_recurrent_decode_match_full_forward() {
        let device = Default::default();
        let model = T5ForConditionalGeneration::<TestBackend>::init(
            tiny_config(T5FeedForwardProjection::GatedGelu, false, 2),
            &device,
        )
        .unwrap();
        let encoder = encoded_source(&model, &device);
        let options = ForwardOptions {
            logits: LogitsSelection::All,
            hidden_states: true,
            ..ForwardOptions::default()
        };
        let mut state = model
            .new_decode_state(CacheConfig::new(1, 8).unwrap(), 1, &device)
            .unwrap();
        model
            .prefill_decoder(&encoder, target(&[0, 3], &device), &mut state, &options)
            .unwrap();
        let chunk = model
            .decode(
                &encoder,
                DecodeInput {
                    token_ids: target(&[4, 5], &device).token_ids,
                    token_mask: None,
                    position_ids: None,
                },
                &mut state,
                &options,
            )
            .expect("two-token decode")
            .decoder;
        let full = model
            .decode_stateless(&encoder, target(&[0, 3, 4, 5], &device), &options)
            .unwrap();
        assert_close(chunk.logits, full.logits.narrow(1, 2, 2));
        assert_close(
            chunk.final_hidden_state.unwrap(),
            full.final_hidden_state.unwrap().narrow(1, 2, 2),
        );
        assert_eq!(state.lengths(), [4]);

        let step = model
            .decode(
                &encoder,
                DecodeInput {
                    token_ids: target(&[6], &device).token_ids,
                    token_mask: None,
                    position_ids: None,
                },
                &mut state,
                &options,
            )
            .expect("single-token decode")
            .decoder;
        let full = model
            .decode_stateless(&encoder, target(&[0, 3, 4, 5, 6], &device), &options)
            .unwrap();
        assert_close(step.logits, full.logits.narrow(1, 4, 1));
        assert_eq!(state.lengths(), [5]);
    }

    /// Ragged and gapped semantic masks have identical stateless/prefill meaning.
    #[test]
    fn ragged_gapped_prefill_matches_stateless_valid_positions() {
        let device = Default::default();
        let model = T5ForConditionalGeneration::<TestBackend>::init(
            tiny_config(T5FeedForwardProjection::GatedGelu, false, 2),
            &device,
        )
        .unwrap();
        let encoder = model
            .encode(
                CausalLmInput {
                    token_ids: Tensor::from_ints([[8, 9, 10], [11, 12, 13]], &device),
                    token_mask: Some(TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
                        [[true, false, true], [true, true, false]].into(),
                        &device,
                    ))),
                    position_ids: None,
                },
                &ForwardOptions::default(),
            )
            .unwrap();
        let decoder = CausalLmInput {
            token_ids: Tensor::from_ints([[0, 1, 4], [0, 5, 1]], &device),
            token_mask: Some(TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
                [[true, false, true], [true, true, false]].into(),
                &device,
            ))),
            position_ids: None,
        };
        let options = ForwardOptions {
            logits: LogitsSelection::All,
            hidden_states: true,
            ..ForwardOptions::default()
        };
        let stateless = model
            .decode_stateless(&encoder, decoder.clone(), &options)
            .unwrap();
        let mut state = model
            .new_decode_state(CacheConfig::new(2, 6).unwrap(), 2, &device)
            .unwrap();
        let cached = model
            .prefill_decoder(&encoder, decoder, &mut state, &options)
            .unwrap()
            .decoder;
        assert_eq!(state.lengths(), [2, 2]);
        assert_close(cached.logits, stateless.logits);
        assert_close(
            cached.final_hidden_state.unwrap(),
            stateless.final_hidden_state.unwrap(),
        );
        assert!(
            cached
                .hidden_states
                .iter()
                .all(|hidden| hidden.dims() == [2, 3, 8])
        );
    }

    /// Finished rows ignore masked sentinel IDs and preserve every physical slot.
    #[test]
    fn mixed_and_all_inactive_decode_rows_are_exact_no_ops() {
        let device = Default::default();
        let model = T5ForConditionalGeneration::<TestBackend>::init(
            tiny_config(T5FeedForwardProjection::Relu, true, 2),
            &device,
        )
        .unwrap();
        let encoder = model
            .encode(
                CausalLmInput {
                    token_ids: Tensor::from_ints([[8, 9], [10, 11]], &device),
                    token_mask: None,
                    position_ids: None,
                },
                &ForwardOptions::default(),
            )
            .unwrap();
        let mut state = model
            .new_decode_state(CacheConfig::new(2, 6).unwrap(), 2, &device)
            .unwrap();
        model
            .prefill_decoder(
                &encoder,
                CausalLmInput {
                    token_ids: Tensor::from_ints([[0, 3], [0, 99]], &device),
                    token_mask: Some(TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
                        [[true, true], [true, false]].into(),
                        &device,
                    ))),
                    position_ids: None,
                },
                &mut state,
                &ForwardOptions {
                    logits: LogitsSelection::All,
                    ..ForwardOptions::default()
                },
            )
            .unwrap();
        assert_eq!(state.lengths(), [2, 1]);
        let before = self_cache_snapshot(&state);
        let output = model
            .decode(
                &encoder,
                DecodeInput {
                    token_ids: Tensor::from_ints([[4, 5], [-1, 99]], &device),
                    token_mask: Some(TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
                        [[true, true], [false, false]].into(),
                        &device,
                    ))),
                    position_ids: None,
                },
                &mut state,
                &ForwardOptions {
                    logits: LogitsSelection::All,
                    hidden_states: true,
                    ..ForwardOptions::default()
                },
            )
            .expect("mixed decode");
        assert_eq!(state.lengths(), [4, 1]);
        let after = self_cache_snapshot(&state);
        let per_row_layer_tensors = 2;
        for layer in 0..2 {
            let row_one_start = (layer * 2 + 1) * per_row_layer_tensors;
            assert_eq!(after[row_one_start], before[row_one_start]);
            assert_eq!(after[row_one_start + 1], before[row_one_start + 1]);
        }
        assert!(
            output
                .decoder
                .logits
                .slice([1..2, 0..2, 0..24])
                .into_data()
                .to_vec::<f32>()
                .unwrap()
                .iter()
                .all(|value| *value == 0.0)
        );
        assert!(
            output
                .decoder
                .final_hidden_state
                .unwrap()
                .slice([1..2, 0..2, 0..8])
                .into_data()
                .to_vec::<f32>()
                .unwrap()
                .iter()
                .all(|value| *value == 0.0)
        );
        assert!(output.decoder.hidden_states.iter().all(|hidden| {
            hidden
                .clone()
                .slice([1..2, 0..2, 0..8])
                .into_data()
                .to_vec::<f32>()
                .unwrap()
                .iter()
                .all(|value| *value == 0.0)
        }));

        let before = self_cache_snapshot(&state);
        let lengths = state.lengths().to_vec();
        let output = model
            .decode(
                &encoder,
                DecodeInput {
                    token_ids: Tensor::from_ints([[99], [-7]], &device),
                    token_mask: Some(TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
                        [[false], [false]].into(),
                        &device,
                    ))),
                    position_ids: None,
                },
                &mut state,
                &ForwardOptions::default(),
            )
            .expect("all-inactive decode");
        assert_eq!(state.lengths(), lengths);
        assert_eq!(self_cache_snapshot(&state), before);
        assert!(
            output
                .decoder
                .logits
                .into_data()
                .to_vec::<f32>()
                .unwrap()
                .iter()
                .all(|value| *value == 0.0)
        );
    }

    /// A later-layer error leaves lengths, cross state, and all physical KV unchanged.
    #[test]
    fn later_layer_failure_is_result_atomic() {
        let device = Default::default();
        let mut model = T5ForConditionalGeneration::<TestBackend>::init(
            tiny_config(T5FeedForwardProjection::Relu, true, 2),
            &device,
        )
        .unwrap();
        let encoder = encoded_source(&model, &device);
        let mut state = model
            .new_decode_state(CacheConfig::new(1, 6).unwrap(), 1, &device)
            .unwrap();
        model
            .prefill_decoder(
                &encoder,
                target(&[0, 3], &device),
                &mut state,
                &ForwardOptions::default(),
            )
            .unwrap();
        let before = self_cache_snapshot(&state);
        let lengths = state.lengths().to_vec();
        let cross_before = state.cross.as_ref().unwrap().keys[0].clone().into_data();
        model.decoder.block[1].cross_attention = None;
        let error = model
            .decode(
                &encoder,
                DecodeInput {
                    token_ids: target(&[4], &device).token_ids,
                    token_mask: None,
                    position_ids: None,
                },
                &mut state,
                &ForwardOptions::default(),
            )
            .expect_err("corrupt later layer");
        assert!(error.to_string().contains("cross_attention"));
        assert_eq!(state.lengths(), lengths);
        assert_eq!(self_cache_snapshot(&state), before);
        assert_eq!(
            state.cross.as_ref().unwrap().keys[0].clone().into_data(),
            cross_before
        );
    }

    /// Capacity failure occurs before model execution and preserves the complete state.
    #[test]
    fn capacity_boundary_failure_preserves_state() {
        let device = Default::default();
        let model = T5ForConditionalGeneration::<TestBackend>::init(
            tiny_config(T5FeedForwardProjection::GatedGelu, false, 1),
            &device,
        )
        .unwrap();
        let encoder = encoded_source(&model, &device);
        let mut state = model
            .new_decode_state(CacheConfig::new(1, 2).unwrap(), 1, &device)
            .unwrap();
        model
            .prefill_decoder(
                &encoder,
                target(&[0, 3], &device),
                &mut state,
                &ForwardOptions::default(),
            )
            .unwrap();
        let before = self_cache_snapshot(&state);
        let error = model
            .decode(
                &encoder,
                DecodeInput {
                    token_ids: target(&[4], &device).token_ids,
                    token_mask: None,
                    position_ids: None,
                },
                &mut state,
                &ForwardOptions::default(),
            )
            .expect_err("full cache");
        assert!(error.to_string().contains("capacity"));
        assert_eq!(state.lengths(), [2]);
        assert_eq!(self_cache_snapshot(&state), before);
    }

    /// Decoder steps consume cached encoder K/V even if projection weights change.
    #[test]
    fn cross_attention_key_values_are_projected_only_during_prefill() {
        let device = Default::default();
        let mut model = T5ForConditionalGeneration::<TestBackend>::init(
            tiny_config(T5FeedForwardProjection::Relu, true, 2),
            &device,
        )
        .unwrap();
        let encoder = encoded_source(&model, &device);
        let mut state = model
            .new_decode_state(CacheConfig::new(1, 5).unwrap(), 1, &device)
            .unwrap();
        model
            .prefill_decoder(
                &encoder,
                target(&[0], &device),
                &mut state,
                &ForwardOptions::default(),
            )
            .unwrap();
        for block in &mut model.decoder.block {
            let cross = block.cross_attention.as_mut().unwrap();
            cross.enc_dec_attention.attention.k_proj.weight =
                Initializer::Constant { value: f64::NAN }.init(
                    cross.enc_dec_attention.attention.k_proj.weight.shape(),
                    &device,
                );
            cross.enc_dec_attention.attention.v_proj.weight =
                Initializer::Constant { value: f64::NAN }.init(
                    cross.enc_dec_attention.attention.v_proj.weight.shape(),
                    &device,
                );
        }
        let output = model
            .decode(
                &encoder,
                DecodeInput {
                    token_ids: target(&[4], &device).token_ids,
                    token_mask: None,
                    position_ids: None,
                },
                &mut state,
                &ForwardOptions::default(),
            )
            .expect("decode must not execute changed K/V projections");
        assert!(
            output
                .decoder
                .logits
                .into_data()
                .to_vec::<f32>()
                .unwrap()
                .iter()
                .all(|value| value.is_finite())
        );
    }

    /// Gapped encoder masks exclude cached source columns exactly as stateless attention does.
    #[test]
    fn cached_cross_attention_preserves_gapped_encoder_masks() {
        let device = Default::default();
        let model = T5ForConditionalGeneration::<TestBackend>::init(
            tiny_config(T5FeedForwardProjection::GatedSilu, false, 1),
            &device,
        )
        .unwrap();
        let mask = TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
            [[true, false, true]].into(),
            &device,
        ));
        let encoder = EncoderOutput {
            hidden_state: Tensor::from_data(
                TensorData::new(
                    vec![
                        0.1_f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 10_000.0, -10_000.0, 9_000.0,
                        -9_000.0, 8_000.0, -8_000.0, 7_000.0, -7_000.0, 0.8, 0.7, 0.6, 0.5, 0.4,
                        0.3, 0.2, 0.1,
                    ],
                    [1, 3, 8],
                ),
                &device,
            ),
            token_mask: Some(mask),
            hidden_states: Vec::new(),
            attentions: Vec::new(),
        };
        let options = ForwardOptions {
            logits: LogitsSelection::All,
            ..ForwardOptions::default()
        };
        let stateless = model
            .decode_stateless(&encoder, target(&[0, 4], &device), &options)
            .unwrap();
        let mut state = model
            .new_decode_state(CacheConfig::new(1, 4).unwrap(), 1, &device)
            .unwrap();
        let cached = model
            .prefill_decoder(&encoder, target(&[0, 4], &device), &mut state, &options)
            .unwrap();
        assert_close(cached.decoder.logits, stateless.logits);
    }

    /// Reorder, truncate, and reset coordinate both self and cross state.
    #[test]
    fn lifecycle_operations_coordinate_self_and_cross_caches() {
        let device = Default::default();
        let model = T5ForConditionalGeneration::<TestBackend>::init(
            tiny_config(T5FeedForwardProjection::Relu, true, 1),
            &device,
        )
        .unwrap();
        let encoder = model
            .encode(
                CausalLmInput {
                    token_ids: Tensor::from_ints([[8, 9], [10, 11]], &device),
                    token_mask: None,
                    position_ids: None,
                },
                &ForwardOptions::default(),
            )
            .unwrap();
        let mut state = model
            .new_decode_state(CacheConfig::new(2, 5).unwrap(), 2, &device)
            .unwrap();
        model
            .prefill_decoder(
                &encoder,
                CausalLmInput {
                    token_ids: Tensor::from_ints([[0, 3], [0, 99]], &device),
                    token_mask: Some(TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
                        [[true, true], [true, false]].into(),
                        &device,
                    ))),
                    position_ids: None,
                },
                &mut state,
                &ForwardOptions::default(),
            )
            .unwrap();
        let cross_before = state.cross.as_ref().unwrap().keys[0].clone();
        state.reorder(&[1, 0]).expect("beam reorder");
        assert_eq!(state.lengths(), [1, 2]);
        assert_close(
            state.cross.as_ref().unwrap().keys[0]
                .clone()
                .slice([0..1, 0..2, 0..2, 0..4]),
            cross_before.slice([1..2, 0..2, 0..2, 0..4]),
        );
        let first_reordered_cross = state.cross.as_ref().unwrap().keys[0].clone();
        state.reorder(&[0, 0]).expect("beam duplication");
        assert_eq!(state.lengths(), [1, 1]);
        assert_close(
            state.cross.as_ref().unwrap().keys[0]
                .clone()
                .slice([1..2, 0..2, 0..2, 0..4]),
            first_reordered_cross.slice([0..1, 0..2, 0..2, 0..4]),
        );
        state.truncate(&[0, 1]).expect("full-cache rewind");
        assert_eq!(state.lengths(), [0, 1]);
        assert!(state.has_encoder_cache());
        state.reset().expect("state reset");
        assert_eq!(state.lengths(), [0, 0]);
        assert!(!state.has_encoder_cache());
        assert!(
            self_cache_snapshot(&state)
                .into_iter()
                .flatten()
                .all(|value| value == 0.0)
        );
    }

    /// A full-KV rewind hides and overwrites the stale suffix on the next decode.
    #[test]
    fn truncate_then_decode_matches_the_shortened_prefix() {
        let device = Default::default();
        let model = T5ForConditionalGeneration::<TestBackend>::init(
            tiny_config(T5FeedForwardProjection::GatedGelu, false, 2),
            &device,
        )
        .unwrap();
        let encoder = encoded_source(&model, &device);
        let mut state = model
            .new_decode_state(CacheConfig::new(1, 6).unwrap(), 1, &device)
            .unwrap();
        model
            .prefill_decoder(
                &encoder,
                target(&[0, 3, 4], &device),
                &mut state,
                &ForwardOptions::default(),
            )
            .unwrap();
        state.truncate(&[1]).expect("full cache supports rewind");
        assert!(state.has_encoder_cache());
        let cached = model
            .decode(
                &encoder,
                DecodeInput {
                    token_ids: target(&[5], &device).token_ids,
                    token_mask: None,
                    position_ids: None,
                },
                &mut state,
                &ForwardOptions::default(),
            )
            .unwrap();
        let stateless = model
            .decode_stateless(
                &encoder,
                target(&[0, 5], &device),
                &ForwardOptions::default(),
            )
            .unwrap();
        assert_close(cached.decoder.logits, stateless.logits);
        assert_eq!(state.lengths(), [2]);
        let snapshot = self_cache_snapshot(&state);
        assert!(state.truncate(&[3]).is_err());
        assert_eq!(state.lengths(), [2]);
        assert_eq!(self_cache_snapshot(&state), snapshot);
    }

    /// Invalid lifecycle/input operations are contextual and do not initialize cross state.
    #[test]
    fn invalid_cached_inputs_fail_before_state_mutation() {
        let device = Default::default();
        let model = T5ForConditionalGeneration::<TestBackend>::init(
            tiny_config(T5FeedForwardProjection::Relu, true, 1),
            &device,
        )
        .unwrap();
        let encoder = encoded_source(&model, &device);
        let mut state = model
            .new_decode_state(CacheConfig::new(1, 4).unwrap(), 1, &device)
            .unwrap();
        let error = model
            .decode(
                &encoder,
                DecodeInput {
                    token_ids: target(&[1], &device).token_ids,
                    token_mask: None,
                    position_ids: None,
                },
                &mut state,
                &ForwardOptions::default(),
            )
            .expect_err("decode before prefill");
        assert!(error.to_string().contains("successful decoder prefill"));
        assert!(!state.has_encoder_cache());

        let error = model
            .prefill_decoder(
                &encoder,
                CausalLmInput {
                    token_ids: target(&[0], &device).token_ids,
                    token_mask: None,
                    position_ids: Some(Tensor::zeros([1, 1], &device)),
                },
                &mut state,
                &ForwardOptions::default(),
            )
            .expect_err("explicit positions are unsupported");
        assert!(matches!(error, Error::BackendCapability(_)));
        assert_eq!(state.lengths(), [0]);
        assert!(!state.has_encoder_cache());

        let error = model
            .prefill_decoder(
                &encoder,
                CausalLmInput {
                    token_ids: Tensor::from_ints([[99]], &device),
                    token_mask: None,
                    position_ids: None,
                },
                &mut state,
                &ForwardOptions::default(),
            )
            .expect_err("active invalid ID");
        assert!(error.to_string().contains("0..24"));
        assert_eq!(state.lengths(), [0]);
        assert!(!state.has_encoder_cache());

        let error = model
            .prefill_decoder(
                &encoder,
                CausalLmInput {
                    token_ids: Tensor::from_ints([[99]], &device),
                    token_mask: Some(TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
                        [[false]].into(),
                        &device,
                    ))),
                    position_ids: None,
                },
                &mut state,
                &ForwardOptions::default(),
            )
            .expect_err("prefill cannot contain an inactive row");
        assert!(error.to_string().contains("fully masked prompt row"));
        assert_eq!(state.lengths(), [0]);
        assert!(!state.has_encoder_cache());

        assert!(state.reorder(&[]).is_err());
        assert!(state.reorder(&[1]).is_err());
        assert_eq!(state.lengths(), [0]);
    }

    /// Initial cross projection is also deferred when the output head fails.
    #[test]
    fn initial_prefill_output_failure_does_not_bind_encoder() {
        let device = Default::default();
        let mut model = T5ForConditionalGeneration::<TestBackend>::init(
            tiny_config(T5FeedForwardProjection::Relu, true, 1),
            &device,
        )
        .unwrap();
        let encoder = encoded_source(&model, &device);
        let mut state = model
            .new_decode_state(CacheConfig::new(1, 4).unwrap(), 1, &device)
            .unwrap();
        model.lm_head = Some(super::super::t5_linear(8, 24, 1.0, &device));
        model
            .prefill_decoder(
                &encoder,
                target(&[0], &device),
                &mut state,
                &ForwardOptions::default(),
            )
            .expect_err("tied topology with an independent head");
        assert_eq!(state.lengths(), [0]);
        assert!(!state.has_encoder_cache());
        assert!(
            self_cache_snapshot(&state)
                .into_iter()
                .flatten()
                .all(|value| value == 0.0)
        );
    }

    /// Corrupt cross-attention topology is rejected instead of indexing cache vectors.
    #[test]
    fn incompatible_model_and_state_layout_is_rejected() {
        let device = Default::default();
        let one = T5ForConditionalGeneration::<TestBackend>::init(
            tiny_config(T5FeedForwardProjection::Relu, true, 1),
            &device,
        )
        .unwrap();
        let two = T5ForConditionalGeneration::<TestBackend>::init(
            tiny_config(T5FeedForwardProjection::Relu, true, 2),
            &device,
        )
        .unwrap();
        let encoder = encoded_source(&two, &device);
        let mut state = one
            .new_decode_state(CacheConfig::new(1, 4).unwrap(), 1, &device)
            .unwrap();
        let error = two
            .prefill_decoder(
                &encoder,
                target(&[0], &device),
                &mut state,
                &ForwardOptions::default(),
            )
            .expect_err("state belongs to another topology");
        assert!(error.to_string().contains("incompatible model layout"));
    }

    /// A semantic causal mask exposes history and only the current query prefix.
    #[test]
    fn cached_causal_mask_matches_absolute_query_positions() {
        let device = Default::default();
        let mask = cached_causal_mask::<TestBackend>(2, 3, 1, &device)
            .unwrap()
            .into_data();
        assert_eq!(
            mask.to_vec::<bool>().unwrap(),
            vec![
                false, false, false, true, true, false, false, false, false, true, false, false,
                false, false, false,
            ]
        );
    }

    proptest! {
        /// Every cached causal mask agrees with the absolute-position inequality.
        #[test]
        fn cached_causal_mask_property(
            baseline in 0_usize..8,
            query_sequence in 1_usize..6,
            heads in 1_usize..4,
        ) {
            let device = Default::default();
            let actual = cached_causal_mask::<TestBackend>(
                baseline,
                query_sequence,
                heads,
                &device,
            )
            .unwrap()
            .into_data()
            .to_vec::<bool>()
            .unwrap();
            let key_sequence = baseline + query_sequence;
            let mut expected = Vec::with_capacity(heads * query_sequence * key_sequence);
            for _head in 0..heads {
                for query in 0..query_sequence {
                    for key in 0..key_sequence {
                        expected.push(key > baseline + query);
                    }
                }
            }
            prop_assert_eq!(actual, expected);
        }
    }

    /// Test-only construction ensures a decoder block without cross attention errors cleanly.
    #[test]
    fn cross_projection_rejects_missing_decoder_component() {
        let device = Default::default();
        let mut model = T5ForConditionalGeneration::<TestBackend>::init(
            tiny_config(T5FeedForwardProjection::Relu, true, 1),
            &device,
        )
        .unwrap();
        let encoder = encoded_source(&model, &device);
        model.decoder.block[0].cross_attention = None::<T5LayerCrossAttention<TestBackend>>;
        let error = T5CrossAttentionCache::project(&model, &encoder)
            .expect_err("missing cross layer must fail");
        assert!(error.to_string().contains("cross_attention"));
    }

    /// Full-precision stateless execution retains gradients for training v2.
    #[cfg(feature = "autodiff")]
    #[test]
    fn stateless_encoder_decoder_graph_has_parameter_gradients() {
        use burn::backend::Autodiff;

        type AdBackend = Autodiff<TestBackend>;

        let device = Default::default();
        let model = T5ForConditionalGeneration::<AdBackend>::init(
            tiny_config(T5FeedForwardProjection::GatedGelu, false, 1),
            &device,
        )
        .unwrap();
        let output = model
            .forward(
                Seq2SeqInput {
                    encoder: CausalLmInput {
                        token_ids: Tensor::<AdBackend, 2, Int>::from_ints([[8, 9]], &device),
                        token_mask: None,
                        position_ids: None,
                    },
                    decoder: CausalLmInput {
                        token_ids: Tensor::<AdBackend, 2, Int>::from_ints([[0, 1, 4]], &device),
                        token_mask: Some(TokenMask::new(Tensor::<AdBackend, 2, Bool>::from_bool(
                            [[true, false, true]].into(),
                            &device,
                        ))),
                        position_ids: None,
                    },
                },
                &ForwardOptions {
                    logits: LogitsSelection::All,
                    ..ForwardOptions::default()
                },
            )
            .expect("autodiff forward");
        let gradients = burn::optim::GradientsParams::from_grads(
            output.decoder.logits.sum().backward(),
            &model,
        );
        assert!(
            gradients
                .get::<TestBackend, 2>(model.lm_head.as_ref().unwrap().weight.id)
                .is_some()
        );
        assert!(
            gradients
                .get::<TestBackend, 1>(model.decoder.final_layer_norm.weight.id)
                .is_some()
        );
        assert!(
            gradients
                .get::<TestBackend, 2>(
                    model.decoder.block[0]
                        .feed_forward
                        .gated
                        .as_ref()
                        .unwrap()
                        .down_proj
                        .weight
                        .id
                )
                .is_some()
        );
        assert!(
            gradients
                .get::<TestBackend, 2>(
                    model.encoder.block[0]
                        .self_attention
                        .self_attention
                        .attention
                        .q_proj
                        .weight
                        .id
                )
                .is_some()
        );
        assert!(
            gradients
                .get::<TestBackend, 2>(
                    model.decoder.block[0]
                        .cross_attention
                        .as_ref()
                        .unwrap()
                        .enc_dec_attention
                        .attention
                        .q_proj
                        .weight
                        .id
                )
                .is_some()
        );
        assert!(
            gradients
                .get::<TestBackend, 2>(
                    model.decoder.block[0]
                        .cross_attention
                        .as_ref()
                        .unwrap()
                        .enc_dec_attention
                        .attention
                        .k_proj
                        .weight
                        .id
                )
                .is_some()
        );
        assert!(
            gradients
                .get::<TestBackend, 2>(model.shared.weight.id)
                .is_some()
        );
    }
}
