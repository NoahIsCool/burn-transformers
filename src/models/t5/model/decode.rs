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
    tensor::{Bool, Int, Tensor, backend::Backend},
};

use crate::{
    CausalLmInput, CausalLmOutput, DecodeInput, Error, ForwardOptions, Result,
    Seq2SeqLanguageModel,
    architecture::{
        DecodeState as ArchitectureDecodeState, EncoderOutput, Seq2SeqInput, Seq2SeqOutput,
        TokenMask,
    },
    attention::{AttentionProjection, padding_attention_mask},
    cache::{CacheConfig, KvCacheStorage, NativeKvCacheSpec, TensorCacheError, TensorDecodeState},
    operations,
};

use super::{
    DECODER_INPUT, T5Block, T5ForConditionalGeneration, T5Stack, diagnostic_buffer, select_hidden,
    validate_encoder_layout, validate_encoder_output, validate_options,
};
use crate::models::shared::{
    CompactRow, EmptyRows, StatefulContract, StatefulRows, concat_rows, ensure_grid,
    ensure_same_device, index_tensor, prepare_stateful_rows, select_hidden_with_last_columns,
    validate_logits_selection,
};
use crate::models::t5::T5Config;

/// Index of T5's sole homogeneous decoder self-attention cache group.
const SELF_KV_GROUP: usize = 0;

/// Owner named when an input tensor is on another device than the cache.
const DECODE_STATE: &str = "T5 decode state";

/// Stateful operation whose empty-row policy differs by lifecycle phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StatefulPhase {
    /// An initial prompt must contribute at least one token in every row.
    Prefill,
    /// Finished sequences may contribute no tokens while peers continue.
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

/// Encoder data and its per-layer immutable cross-attention projections.
///
/// The retained [`EncoderOutput`] defines the source binding for the lifetime of
/// the state. Subsequent decode calls accept the trait-mandated encoder argument
/// but consume this bound representation, which has already been reordered with
/// the cache during beam operations.
#[derive(Debug, Clone)]
struct T5CrossAttentionCache<B: Backend> {
    /// Bound source hidden states, mask, and optional encoder diagnostics.
    encoder: EncoderOutput<B>,
    /// Per-layer projected keys with axes `[batch, heads, source, d_kv]`.
    keys: Vec<Tensor<B, 4>>,
    /// Per-layer projected values with the same axes as the keys.
    values: Vec<Tensor<B, 4>>,
}

impl<B: Backend> T5CrossAttentionCache<B> {
    /// Builds all decoder-layer cross projections without mutating decode state.
    ///
    /// The caller has already validated `encoder` (including its source-mask
    /// rows) in [`prepare_stateful_input`]; the sync-free layout check here
    /// only guards the projection reshapes.
    fn project(model: &T5ForConditionalGeneration<B>, encoder: &EncoderOutput<B>) -> Result<Self> {
        let [batch, source_sequence, _] = encoder.hidden_state.dims();
        validate_encoder_layout(
            encoder,
            batch,
            model.config.d_model,
            model.shared.weight.val().dtype(),
        )?;
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
        encoder_mask_row(&self.encoder, sequence)
    }

    /// Produces a reordered cache without mutating the current binding.
    fn reordered(&self, indices: &[usize]) -> Result<Self> {
        let selector = index_tensor::<B>(
            indices,
            &self.encoder.hidden_state.device(),
            "t5.decode_state.reorder",
        )?;
        let encoder = EncoderOutput {
            hidden_state: self
                .encoder
                .hidden_state
                .clone()
                .select(0, selector.clone()),
            token_mask: self.encoder.token_mask.as_ref().map(|mask| {
                TokenMask::new(crate::architecture::select_bool_tensor(
                    mask.as_tensor().clone(),
                    0,
                    selector.clone(),
                ))
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
    /// Decoder self-attention tensors and committed sequence lengths.
    tensors: TensorDecodeState<B>,
    /// Expected native self-cache geometry for model compatibility checks.
    self_spec: NativeKvCacheSpec,
    /// Immutable source binding installed by the first successful prefill.
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
    /// dtype, and mask-shape changes without synchronizing all encoder values to the CPU.
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
        if encoder.hidden_state.dtype() != cross.encoder.hidden_state.dtype() {
            return Err(Error::BackendCapability(format!(
                "t5.encoder_output.hidden_state dtype must match the bound encoder {:?}, found {:?}",
                cross.encoder.hidden_state.dtype(),
                encoder.hidden_state.dtype(),
            )));
        }
        ensure_same_device(
            "t5.encoder_output.hidden_state",
            &encoder.hidden_state.device(),
            self.tensors.device(),
            DECODE_STATE,
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
            ensure_same_device(
                "t5.encoder_output.token_mask",
                &mask.as_tensor().device(),
                self.tensors.device(),
                DECODE_STATE,
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
    /// T5-family encoder, decoder, and vocabulary-head architecture metadata.
    type Config = T5Config;
    /// External self-attention history and projected encoder cache.
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
    ///
    /// `valid_indices[b]` lists row `b`'s valid columns, read back together
    /// with the input validation; every row has at least one.
    pub(super) fn decode_stateless_masked(
        &self,
        encoder: &EncoderOutput<B>,
        decoder: CausalLmInput<B>,
        options: &ForwardOptions,
        valid_indices: &[Vec<usize>],
    ) -> Result<CausalLmOutput<B>> {
        let [batch, sequence] = decoder.token_ids.dims();
        debug_assert_eq!(valid_indices.len(), batch, "one column list per target row");
        let device = decoder.token_ids.device();
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
            // An empty row cannot reach this point (validation rejects fully
            // masked rows); `CompactRow::new` still reports one contextually.
            let plan = CompactRow::<B>::new(row, valid, sequence, &device)?;
            let hidden = self.shared.forward(plan.compact(&decoder.token_ids));
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
            final_rows.push(plan.scatter(output.hidden)?);
            if options.hidden_states {
                if output.hidden_states.len() != diagnostic_count {
                    return Err(Error::Cache(format!(
                        "T5 stateless hidden-state count must be {diagnostic_count}, found {}",
                        output.hidden_states.len()
                    )));
                }
                for (destination, compact) in diagnostic_rows.iter_mut().zip(output.hidden_states) {
                    destination.push(plan.scatter(compact)?);
                }
            }
        }
        let hidden = concat_rows(final_rows, "t5.decoder.hidden")?;
        let selected = select_hidden(hidden.clone(), decoder.token_mask.as_ref(), &options.logits)?;
        Ok(CausalLmOutput {
            logits: self.project_logits(selected)?,
            final_hidden_state: options.hidden_states.then_some(hidden),
            hidden_states: diagnostic_rows
                .into_iter()
                .map(|rows| concat_rows(rows, "t5.decoder.hidden_states"))
                .collect::<Result<Vec<_>>>()?,
            attentions: Vec::new(),
            routers: Vec::new(),
            load_balancing_loss: None,
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
            let Some(plan) = prepared.compact_row::<B>(sequence_index, &device)? else {
                let empty = empty_hidden_row(self, prepared.sequence, &device);
                final_rows.push(empty.clone());
                if options.hidden_states {
                    for destination in &mut diagnostic_rows {
                        destination.push(empty.clone());
                    }
                }
                continue;
            };
            let tokens = plan.compact(&decoder.token_ids);
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
            final_rows.push(plan.scatter(compact)?);
            if options.hidden_states {
                if diagnostics.len() != diagnostic_count {
                    return Err(Error::Cache(format!(
                        "T5 hidden-state count must be {diagnostic_count}, found {}",
                        diagnostics.len()
                    )));
                }
                for (destination, compact_state) in diagnostic_rows.iter_mut().zip(diagnostics) {
                    destination.push(plan.scatter(compact_state)?);
                }
            }
        }

        let hidden = concat_rows(final_rows, "t5.decoder.hidden")?;
        let hidden_states = diagnostic_rows
            .into_iter()
            .map(|rows| concat_rows(rows, "t5.decoder.hidden_states"))
            .collect::<Result<Vec<_>>>()?;
        let selected = select_hidden_with_last_columns(
            hidden.clone(),
            prepared.last_columns(),
            &options.logits,
        )?;
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
                load_balancing_loss: None,
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
    /// Decoder layer receiving this append.
    layer: usize,
    /// Batch row whose committed prefix is extended.
    sequence: usize,
    /// First cache position to write, equal to the row's current prefix length.
    start: usize,
    /// New keys with axes `[1, heads, appended_tokens, d_kv]`.
    keys: Tensor<B, 4>,
    /// New values with the same axes as the keys.
    values: Tensor<B, 4>,
}

/// Deferred self-attention writes for one public prefill or decode call.
struct PendingT5SelfWrites<B: Backend> {
    /// One deferred append per active batch row and decoder layer.
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

/// Validates state, encoder, tokens, masks, devices, and capacity before math.
///
/// The encoder binding is checked first (or validated for a fresh state);
/// explicit decoder positions are rejected because T5 uses relative buckets.
/// The shared preparation then reads the decoder mask and token IDs in one
/// transfer and plans every row's compaction and cache append.
fn prepare_stateful_input<B: Backend>(
    model: &T5ForConditionalGeneration<B>,
    encoder: &EncoderOutput<B>,
    decoder: &CausalLmInput<B>,
    state: &T5DecodeState<B>,
    phase: StatefulPhase,
) -> Result<StatefulRows> {
    let device = state.tensors.device();
    ensure_same_device(
        "t5.model",
        &model.shared.weight.lazy_device(),
        device,
        DECODE_STATE,
    )?;
    let [batch, sequence] = decoder.token_ids.dims();
    if state.cross.is_some() {
        state.validate_encoder_binding(encoder)?;
    } else {
        validate_encoder_output(
            encoder,
            batch,
            model.config.d_model,
            model.shared.weight.val().dtype(),
        )?;
        ensure_same_device(
            "t5.encoder_output.hidden_state",
            &encoder.hidden_state.device(),
            device,
            DECODE_STATE,
        )?;
        if let Some(mask) = &encoder.token_mask {
            ensure_same_device(
                "t5.encoder_output.token_mask",
                &mask.as_tensor().device(),
                device,
                DECODE_STATE,
            )?;
        }
    }
    if let Some(positions) = &decoder.position_ids {
        ensure_same_device(
            DECODER_INPUT.position_ids,
            &positions.device(),
            device,
            DECODE_STATE,
        )?;
        ensure_grid(
            positions.dims(),
            batch,
            sequence,
            DECODER_INPUT.position_ids,
        )?;
        return Err(Error::BackendCapability(
            "T5 decoder uses relative positions and does not accept explicit position IDs".into(),
        ));
    }
    prepare_stateful_rows(
        decoder,
        device,
        state.tensors.lengths(),
        state.tensors.capacity(),
        &StatefulContract {
            names: DECODER_INPUT,
            owner: DECODE_STATE,
            vocabulary: model.config.vocab_size,
            position_limit: None,
            empty_rows: phase.empty_rows(),
        },
    )
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

#[cfg(test)]
#[path = "decode_tests.rs"]
mod tests;
