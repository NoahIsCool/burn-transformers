//! Cache-aware T5/FLAN decoder execution, one batched pass per call.
//!
//! T5 decoding retains two mathematically different kinds of state. Decoder
//! self-attention appends keys and values as the target prefix grows, so it uses
//! the crate's preallocated full-history KV cache. Encoder cross-attention is
//! immutable for a generation session: each decoder layer projects
//! `K = H_encoder W_K` and `V = H_encoder W_V` exactly once, then every target
//! step computes only its query projection.
//!
//! # One pass over the whole batch
//!
//! A call appends `n_b` valid tokens to row `b`, whose committed length is
//! `L_b` (`n_b = 0` marks a finished row during decode). The valid tokens of
//! every row are packed left-aligned into one `[batch, Q]` block,
//! `Q = max_b n_b`: row `b`'s new tokens fill slots `0..n_b` in order and
//! padding fills slots `n_b..Q` (see [`super::packing`]). The whole block runs
//! through the shared block graph once:
//!
//! - each layer writes its new keys and values *ahead* into cache slots
//!   `[L_b, L_b + n_b)` and attends over the zero-copy view of slots
//!   `[0, K)`, `K = max_b (L_b + n_b)`, read through one per-step plan;
//! - query `i < n_b` of row `b` sits at absolute position `p = L_b + i` and
//!   sees exactly the keys `j <= p`; this one comparison also excludes every
//!   slot beyond the row's own history. Padded queries (`i >= n_b`) reuse the
//!   row's last position, so no row is ever fully masked unless it has no
//!   history at all; their outputs are discarded;
//! - the shared relative bias `b(j - p)` is the contiguous `[1, heads, Q, K]`
//!   plane when all active rows share one `L` (every beam- or greedy-search
//!   step), and the per-row `[batch, heads, Q, K]` positional bias otherwise;
//! - cross-attention reads the projected `[batch, heads, source, d_kv]`
//!   encoder keys and values of all rows at once.
//!
//! # Attention-ready cross keys and values
//!
//! A projection produces `[batch, source, heads * d_kv]` rows, so splitting
//! heads (`reshape` then `swap_dims(1, 2)`) yields a strided view. CubeCL
//! kernels read that view directly, but Burn Flex's attention copies every
//! non-contiguous key and value operand on every call: with the view cached,
//! each decode step re-copied all `2 * layers * batch * source * d_model`
//! cached elements (about two thirds of FLAN-T5-small's Flex decode time at
//! 512 source tokens). The cache therefore stores both tensors contiguously
//! once at prefill on backends that prefer it
//! ([`operations::contiguous_where_preferred`]), a one-time copy of the same
//! size as one decode step used to make. Values are unchanged; only the
//! physical layout differs.
//!
//! Values of invalid key slots are zeroed before attention (they may hold
//! stale bytes), and scores of excluded keys are replaced, never offset.
//! Scores remain `Q K^T + b + M` without the usual `1 / sqrt(d_kv)` factor.
//!
//! # Atomicity
//!
//! The appends form one [`TensorStepWrites`] step. Lengths advance only when
//! [`TensorDecodeState::publish_step`] succeeds, after every layer, the final
//! normalization, every diagnostic, and the vocabulary projection: a failed
//! call changes no length and no readable cache position (only slots beyond
//! the lengths may hold its abandoned write-ahead tokens). A fully masked row
//! produces exact zero hidden states and logits without changing its state.

use burn::tensor::{Bool, Int, Tensor, TensorData, backend::Backend};

use crate::{
    CausalLmInput, CausalLmOutput, DecodeInput, Error, ForwardOptions, Result,
    Seq2SeqLanguageModel,
    architecture::{
        DecodeState as ArchitectureDecodeState, EncoderOutput, Seq2SeqInput, Seq2SeqOutput,
        TokenMask,
    },
    cache::{
        CacheConfig, KvBatchReadPlan, KvCacheStorage, NativeKvCacheSpec, TensorCacheError,
        TensorDecodeState, TensorStepWrites,
    },
    intervention::Hooks,
    operations,
    position::T5RelativePositionBias,
};

use super::{
    DECODER_INPUT, T5ForConditionalGeneration, bias,
    block::{CrossAttentionPlan, CrossSource, SelfAttentionPlan, SelfKeyValues, StackPlan},
    packing::{
        BatchedRows, Packing, displacement_bound, first_out_of_range, groups_preserved, plan_rows,
        step_positions,
    },
    validate_encoder_layout, validate_encoder_output, validate_options,
};
use crate::models::shared::{
    EmptyRows, StatefulContract, StatefulRows, ensure_grid, ensure_same_device, index_tensor,
    prepare_stateful_rows, select_hidden_with_last_columns, validate_logits_selection,
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
    ///
    /// Contiguous on backends that prefer contiguous attention operands (see
    /// the module documentation); a strided head view elsewhere.
    keys: Vec<Tensor<B, 4>>,
    /// Per-layer projected values with the same axes and layout as the keys.
    values: Vec<Tensor<B, 4>>,
}

impl<B: Backend> T5CrossAttentionCache<B> {
    /// Builds all decoder-layer cross projections without mutating decode state.
    ///
    /// The caller has already validated `encoder` (including its source-mask
    /// rows) in [`prepare_stateful_input`]; the sync-free layout check here
    /// only guards the projection reshapes. Projections go through
    /// [`operations::linear`], the same contract as the stateless
    /// cross-attention projection (dtype/device agreement, packed weights).
    /// Each split-head result is stored attention-ready
    /// ([`operations::contiguous_where_preferred`]), so decode steps never
    /// re-layout it.
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
            let heads = |projection| -> Result<Tensor<B, 4>> {
                let split = operations::linear(encoder.hidden_state.clone(), projection)?
                    .reshape([
                        batch,
                        source_sequence,
                        attention.key_value_heads,
                        attention.head_dim,
                    ])
                    .swap_dims(1, 2);
                Ok(operations::contiguous_where_preferred(split))
            };
            keys.push(heads(&attention.k_proj)?);
            values.push(heads(&attention.v_proj)?);
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

    /// Reorders beams inside groups whose rows share one bound source.
    ///
    /// The caller guarantees (see the trait method) that the rows of each
    /// group hold the same encoder output, and therefore the same projected
    /// cross-attention keys and values. Moving rows inside their group then
    /// leaves the cross binding unchanged, so only the decoder self-attention
    /// history is gathered; the per-step cost no longer grows with the source
    /// length. Indices that leave their group are rejected before anything
    /// changes.
    fn reorder_within_groups(&mut self, indices: &[usize], group_size: usize) -> Result<()> {
        validate_reorder(indices, self.tensors.batch_size())?;
        if !groups_preserved(indices, group_size) {
            return Err(Error::Cache(format!(
                "T5 group-preserving reorder needs groups of {group_size} rows that divide batch {} and indices that stay inside their group",
                indices.len()
            )));
        }
        self.tensors.reorder(indices).map_err(Into::into)
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
    /// Evaluates one stateful target segment for every row in one pass and publishes it atomically.
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
        let rows = plan_rows(&prepared.valid_indices, prepared.sequence).ok_or_else(|| {
            Error::shape(
                DECODER_INPUT.token_mask,
                "sorted valid columns below the sequence length",
                format!("{:?}", prepared.valid_indices),
            )
        })?;

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
        let T5DecodeState { tensors, cross, .. } = state;
        let bound = pending_cross
            .as_ref()
            .or(cross.as_ref())
            .ok_or_else(|| Error::Cache("T5 encoder cross cache is unavailable".into()))?;

        let output = if rows.is_empty() {
            // Every row is finished: exact zeros, no model work, nothing to publish.
            let zeros = Tensor::zeros(
                [rows.batch, rows.sequence, self.config.d_model],
                (
                    &decoder.token_ids.device(),
                    self.shared.weight.val().dtype(),
                ),
            );
            BatchedOutput {
                hidden_states: if options.hidden_states {
                    vec![zeros.clone(); self.decoder.block.len() + 2]
                } else {
                    Vec::new()
                },
                hidden: zeros,
                step: None,
            }
        } else {
            self.run_batched_rows(&decoder, &rows, tensors, bound, options)?
        };
        let selected = select_hidden_with_last_columns(
            output.hidden.clone(),
            prepared.last_columns(),
            &options.logits,
        )?;
        let logits = self.project_logits(selected)?;
        let bound_encoder = bound.encoder.clone();
        if let Some(step) = output.step {
            tensors.publish_step(step)?;
        }
        if let Some(projected) = pending_cross {
            *cross = Some(projected);
        }
        Ok(Seq2SeqOutput {
            decoder: CausalLmOutput {
                logits,
                final_hidden_state: options.hidden_states.then_some(output.hidden),
                hidden_states: output.hidden_states,
                attentions: Vec::new(),
                routers: Vec::new(),
                load_balancing_loss: None,
            },
            encoder: bound_encoder,
            cross_attentions: Vec::new(),
        })
    }

    /// Runs the packed `[batch, Q]` block through every decoder layer with cached attention.
    ///
    /// Returns rectangular `[batch, S, d_model]` states (zeros at padding) and
    /// the unpublished step holding every layer's written-ahead tokens.
    fn run_batched_rows(
        &self,
        decoder: &CausalLmInput<B>,
        rows: &BatchedRows,
        tensors: &mut TensorDecodeState<B>,
        cross: &T5CrossAttentionCache<B>,
        options: &ForwardOptions,
    ) -> Result<BatchedOutput<B>> {
        let device = decoder.token_ids.device();
        let packer = TensorPacker::new(rows, decoder.token_mask.as_ref(), &device)?;
        let hidden = self
            .shared
            .forward(packer.pack_tokens(decoder.token_ids.clone()));
        let geometry = StepGeometry::new(tensors.lengths(), &rows.counts, rows.block)?;
        let step = tensors.step(&rows.counts)?;
        let plan = tensors.kv_read_plan_through(&step, SELF_KV_GROUP)?;
        let key_slots = plan.key_slots();
        let stack_plan = StackPlan {
            self_attention: SelfAttentionPlan {
                excluded: Some(geometry.exclusions(key_slots, &device)?),
                bias: Some(geometry.bias(self.decoder.relative_table()?, key_slots, &device)?),
                causal: false,
            },
            cross_attention: Some(CrossAttentionPlan::new(
                CrossSource::Projected {
                    keys: &cross.keys,
                    values: &cross.values,
                },
                cross.encoder.token_mask.as_ref(),
            )),
        };
        let mut keys = CachedSelfKeyValues {
            state: tensors,
            step,
            plan,
        };
        let output = self.decoder.run(
            hidden,
            &stack_plan,
            &mut keys,
            options.hidden_states,
            &mut Hooks::new(),
        )?;
        Ok(BatchedOutput {
            hidden: packer.unpack(output.hidden),
            hidden_states: output
                .hidden_states
                .into_iter()
                .map(|hidden| packer.unpack(hidden))
                .collect(),
            step: Some(keys.step),
        })
    }
}

/// Rectangular outputs of one batched stateful pass and its unpublished step.
struct BatchedOutput<B: Backend> {
    /// Final normalized states `[batch, S, d_model]`, zero at padding.
    hidden: Tensor<B, 3>,
    /// Requested diagnostics in the same coordinates.
    hidden_states: Vec<Tensor<B, 3>>,
    /// Written-ahead appends to publish once every output exists; `None` when no row is active.
    step: Option<TensorStepWrites<B>>,
}

/// Cached self-attention keys: write a layer's tokens ahead, then read its history.
///
/// Every layer of one call shares one step and one read plan over slots
/// `[0, max_b (L_b + n_b))`. The returned values have invalid slots zeroed
/// ([`crate::cache::KvBatchSegment::masked_values`]), so stale bytes cannot
/// reach attention through a zero probability.
struct CachedSelfKeyValues<'s, B: Backend> {
    /// Decode state receiving the write-ahead tokens.
    state: &'s mut TensorDecodeState<B>,
    /// The call's step; published by the caller after every output exists.
    step: TensorStepWrites<B>,
    /// Per-step batched read plan through the written-ahead tokens.
    plan: KvBatchReadPlan<B>,
}

impl<B: Backend> SelfKeyValues<B> for CachedSelfKeyValues<'_, B> {
    /// Writes `[batch, heads, Q, d_kv]` keys/values ahead and returns the attended history.
    fn resolve(
        &mut self,
        layer: usize,
        keys: Tensor<B, 4>,
        values: Tensor<B, 4>,
    ) -> Result<(Tensor<B, 4>, Tensor<B, 4>)> {
        self.state
            .write_kv_ahead(&mut self.step, SELF_KV_GROUP, layer, keys, values)?;
        let segment = self
            .state
            .read_kv_through(&self.step, SELF_KV_GROUP, layer, &self.plan)?
            .materialize()
            .ok_or_else(|| Error::Cache("T5 self-attention read has no key slots".into()))?;
        Ok((segment.keys().clone(), segment.masked_values()))
    }
}

/// Per-call query positions and the derived self-attention mask and bias.
#[derive(Debug, Clone, PartialEq, Eq)]
struct StepGeometry {
    /// Rows `B`.
    batch: usize,
    /// Block width `Q`.
    block: usize,
    /// Mask position of every block slot `[B, Q]`: `L_b + min(i, n_b - 1)`, or `L_b - 1` when inactive.
    mask_positions: Vec<i64>,
    /// Bias position of every block slot `[B, Q]`: `L_b + i`.
    bias_positions: Vec<i64>,
    /// The committed length shared by every active row, when there is one.
    shared_length: Option<usize>,
    /// Largest bias position, bounding query displacements.
    max_position: usize,
}

impl StepGeometry {
    /// Derives slot positions from committed lengths `L_b` and new-token counts `n_b`.
    ///
    /// The verified [`step_positions`] plans every slot; `None` means a
    /// position beyond `i64`.
    fn new(lengths: &[usize], counts: &[usize], block: usize) -> Result<Self> {
        let plan = step_positions(lengths, counts, block)
            .ok_or_else(|| Error::configuration("t5.decoder.positions", "position exceeds i64"))?;
        Ok(Self {
            batch: counts.len(),
            block,
            mask_positions: plan.mask_positions,
            bias_positions: plan.bias_positions,
            shared_length: plan.shared_length,
            max_position: plan.max_position,
        })
    }

    /// Builds the `[B, 1, Q, K]` exclusion plane `key_slot > mask_position`.
    ///
    /// Valid queries see exactly their causal history; slots beyond a row's
    /// `L_b + n_b` are excluded by the same comparison. One `B * Q * K`
    /// Boolean plane is materialized per call and shared by every layer.
    fn exclusions<B: Backend>(
        &self,
        key_slots: usize,
        device: &B::Device,
    ) -> Result<Tensor<B, 4, Bool>> {
        let keys = i64::try_from(key_slots).map_err(|_| {
            Error::configuration("t5.decoder.attention_mask", "key length exceeds i64")
        })?;
        let shape = [self.batch, 1, self.block, key_slots];
        let positions = Tensor::<B, 2, Int>::from_data(
            TensorData::new(self.mask_positions.clone(), [self.batch, self.block]),
            device,
        );
        let slots = Tensor::<B, 1, Int>::arange(0..keys, device)
            .reshape([1, 1, 1, key_slots])
            .expand(shape);
        Ok(slots.greater(
            positions
                .reshape([self.batch, 1, self.block, 1])
                .expand(shape),
        ))
    }

    /// Builds the shared relative bias `b(key_slot - bias_position)`.
    ///
    /// Active rows sharing one committed length `L` (every beam- or
    /// greedy-search step) use the contiguous `[1, heads, Q, K]` plane with
    /// query offset `L`; ragged lengths use the per-row positional bias.
    fn bias<B: Backend>(
        &self,
        table: &T5RelativePositionBias<B>,
        key_slots: usize,
        device: &B::Device,
    ) -> Result<Tensor<B, 4>> {
        if let Some(length) = self.shared_length {
            return bias::contiguous_bias(table, self.block, key_slots, length, device);
        }
        let keys = i64::try_from(key_slots)
            .map_err(|_| Error::configuration("t5.decoder.positions", "key length exceeds i64"))?;
        let queries = Tensor::<B, 2, Int>::from_data(
            TensorData::new(self.bias_positions.clone(), [self.batch, self.block]),
            device,
        );
        let keys = Tensor::<B, 1, Int>::arange(0..keys, device)
            .reshape([1, key_slots])
            .expand([self.batch, key_slots]);
        let bound = displacement_bound(self.max_position, key_slots);
        bias::positional_bias(table, queries, keys, bound)
    }
}

/// Device realization of a [`BatchedRows`] packing plan.
///
/// Index tensors are uploaded once per call and reused for the tokens, the
/// final states, and every requested diagnostic.
struct TensorPacker<B: Backend> {
    /// Rows `B`.
    batch: usize,
    /// Input columns `S`.
    sequence: usize,
    /// Block width `Q`.
    block: usize,
    /// Padded block slots `[B, Q]` (`true` = padding), when any exist.
    padded_slots: Option<Tensor<B, 2, Bool>>,
    /// Padded input columns `[B, S]` (`true` = padding), from the call's mask.
    padded_columns: Option<Tensor<B, 2, Bool>>,
    /// Flat gather indices `(compaction, scatter)` for leading or interior padding.
    gather: Option<(Tensor<B, 1, Int>, Tensor<B, 1, Int>)>,
}

impl<B: Backend> TensorPacker<B> {
    /// Uploads the plan's index and padding tensors.
    fn new(rows: &BatchedRows, mask: Option<&TokenMask<B>>, device: &B::Device) -> Result<Self> {
        let padded_slots = rows.has_padded_slots().then(|| {
            Tensor::<B, 2, Bool>::from_data(
                TensorData::new(rows.padded_slots(), [rows.batch, rows.block]),
                device,
            )
        });
        let gather = match &rows.packing {
            Packing::Prefix => None,
            Packing::Gather { compact, scatter } => Some((
                index_tensor::<B>(compact, device, "t5.decoder.packing")?,
                index_tensor::<B>(scatter, device, "t5.decoder.unpacking")?,
            )),
        };
        Ok(Self {
            batch: rows.batch,
            sequence: rows.sequence,
            block: rows.block,
            padded_slots,
            padded_columns: mask.map(|mask| mask.as_tensor().clone().bool_not()),
            gather,
        })
    }

    /// Packs `[B, S]` token IDs into the `[B, Q]` block; padded slots hold token zero.
    ///
    /// Padding columns may carry arbitrary sentinel IDs (validation covers
    /// valid columns only), so padded slots are overwritten with zero, which
    /// every vocabulary contains, before the embedding lookup.
    fn pack_tokens(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 2, Int> {
        let packed = match &self.gather {
            Some((compact, _)) => tokens
                .reshape([self.batch * self.sequence])
                .select(0, compact.clone())
                .reshape([self.batch, self.block]),
            None if self.block == self.sequence => tokens,
            None => tokens.narrow(1, 0, self.block),
        };
        match &self.padded_slots {
            Some(padded) => packed.mask_fill(padded.clone(), 0),
            None => packed,
        }
    }

    /// Unpacks `[B, Q, d]` block states into `[B, S, d]` with exact zeros at padding.
    fn unpack(&self, packed: Tensor<B, 3>) -> Tensor<B, 3> {
        let width = packed.dims()[2];
        let rectangular = match &self.gather {
            Some((_, scatter)) => packed
                .reshape([self.batch * self.block, width])
                .select(0, scatter.clone())
                .reshape([self.batch, self.sequence, width]),
            None if self.block == self.sequence => packed,
            None => {
                let padding = Tensor::zeros(
                    [self.batch, self.sequence - self.block, width],
                    (&packed.device(), packed.dtype()),
                );
                Tensor::cat(vec![packed, padding], 1)
            }
        };
        match &self.padded_columns {
            Some(padded) => rectangular.mask_fill(
                padded
                    .clone()
                    .unsqueeze_dim::<3>(2)
                    .expand([self.batch, self.sequence, width]),
                0.0,
            ),
            None => rectangular,
        }
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

/// Validates a fixed-size batch permutation or duplication vector.
fn validate_reorder(indices: &[usize], batch: usize) -> Result<()> {
    if indices.len() != batch {
        return Err(Error::Cache(format!(
            "T5 reorder requires {batch} indices, found {}",
            indices.len()
        )));
    }
    if let Some(index) = first_out_of_range(indices, batch) {
        return Err(Error::Cache(format!(
            "T5 reorder index {index} is outside batch {batch}"
        )));
    }
    Ok(())
}

#[cfg(test)]
#[path = "decode_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "decode_batch_tests.rs"]
mod batch_tests;
