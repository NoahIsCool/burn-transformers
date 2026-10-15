//! Batched external-cache execution for the Nemotron-H hybrid decoder.
//!
//! # Cache layout
//!
//! Attention and Mamba blocks occupy separate compact cache index spaces even
//! though checkpoint layers are interleaved. Attention storage is full (never
//! evicting) `[batch, kv_heads, capacity, head_dim]`; recurrent storage is
//! `[batch, mamba_heads, head_dim, state_size]` plus the final `kernel - 1`
//! convolution inputs. Dense and MoE blocks keep no state.
//!
//! # One pass per call
//!
//! A prefill or decode call evaluates every batch row in **one** model pass.
//! Each row's valid tokens are compacted into a right-padded `[B, N]` batch
//! (`N` = most valid tokens in any row; see the planning module):
//!
//! - Mamba-2 blocks advance all rows together; padding is an identity
//!   transition and each row keeps the convolution history at its own end.
//!   A row takes the reference single-step update only when it appends one
//!   token to existing state, and the chunk scan otherwise.
//! - Attention blocks write the step's keys and values into the full cache
//!   *ahead* of attention and attend over one zero-copy view `[0, L_b + n_b)`
//!   of every row. Keys are excluded by absolute position (`k > q`, outside
//!   the sliding window, or beyond a row's end); a dense batch with equal
//!   lengths needs no mask at all, only bottom-right causality.
//! - MoE blocks route only the `T = sum_b n_b` packed valid tokens, so padding
//!   never reaches a router or an expert and routing costs one host
//!   synchronization per block for the whole batch.
//!
//! Outputs are scattered back to the caller's rectangular layout with exact
//! zeros at padding, and `LogitsSelection::Last` projects each row's last
//! valid token without materializing the rectangular hidden state.
//!
//! # Atomicity
//!
//! Every cache update of a call is recorded in one
//! [`TensorStepWrites`](crate::cache::TensorStepWrites) and published only
//! after the whole forward pass and its outputs succeed. A failed call
//! therefore leaves every length and every *readable* cache position
//! unchanged; only slots beyond the committed lengths may hold the abandoned
//! write-ahead keys, which no read can reach and the next write overwrites.
//! Fully masked decode rows are exact no-ops; fully masked prefill rows are
//! rejected.
//!
//! # Speculative transactions
//!
//! Inside a transaction the same pass stages its writes. The attention cache
//! stages in place, which keeps write-ahead available, and Mamba-2 blocks
//! record their state after *every* token of a multi-token step, so a
//! verifier can evaluate a whole draft in one pass and commit any accepted
//! prefix. Per-token states come from the reference recurrent update, so a
//! multi-token transactional step requires every active row to have been
//! prefilled already.

use core::num::NonZeroUsize;

use burn::tensor::{DType, Int, Tensor, backend::Backend};

use crate::{
    CausalLanguageModel, CausalLmInput, CausalLmOutput, DecodeInput, Error, ForwardOptions,
    LogitsSelection, Result,
    architecture::{DecodeState as ArchitectureDecodeState, RouterDiagnostics},
    attention::GroupedQueryAttention,
    cache::{
        CacheConfig, DecodeStateError, KvBatchReadPlan, KvCacheStorage, NativeKvCacheSpec,
        RecurrentStateSpec, TensorCacheError, TensorCacheMark, TensorDecodeState, TensorStepWrites,
    },
    mamba2::Mamba2Row,
    models::shared::{
        EmptyRows, StatefulContract, StatefulRows, ensure_same_device, prepare_stateful_rows,
        select_hidden_with_last_columns, validate_logits_selection,
    },
};

use super::batch::CompactBatch;
use super::mamba2::{NemotronMamba2, NemotronMamba2State};
use super::plan::{
    MambaStep, TransactionalMamba, mamba_step, mtp_length_after_truncate, transactional_mamba,
};
use super::{
    BlockMixer, INPUT_NAMES, NemotronHBlockType, NemotronHFlatMoeOutput, NemotronHForCausalLm,
    NemotronMamba2Mode, nemotron_load_balancing_loss,
};

/// Owner named when an input tensor is on another device than the cache.
const DECODE_STATE: &str = "Nemotron-H decode state";

/// Native-KV group index within either the target bank or the independent MTP bank.
pub(super) const KV_GROUP: usize = 0;
/// Recurrent group index within the target bank's separate recurrent group space.
pub(super) const RECURRENT_GROUP: usize = 0;

/// Stateful operation whose empty-row policy differs intentionally.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StatefulPhase {
    /// Initial prompts require at least one semantic token per row.
    Prefill,
    /// Finished rows may remain in a rectangular batch as no-ops.
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

/// External KV, recurrent, convolution, and logical-length state.
#[derive(Debug)]
pub struct NemotronHDecodeState<B: Backend> {
    /// Owned target/MTP tensor banks, cursors, and transaction metadata.
    pub(super) tensors: TensorDecodeState<B>,
    /// Compact target attention topology, absent for attention-free schedules.
    kv_spec: Option<NativeKvCacheSpec>,
    /// Compact Mamba matrix/convolution topology, absent without recurrent blocks.
    recurrent_spec: Option<RecurrentStateSpec>,
    /// Independent auxiliary attention topology when the model has an MTP stage.
    mtp_kv_spec: Option<NativeKvCacheSpec>,
    /// Maximum tentative suffix length per row reserved for speculation.
    staging_capacity: usize,
}

impl<B: Backend> NemotronHDecodeState<B> {
    /// Allocates cache groups matching the exact interleaved schedule.
    fn new(
        model: &NemotronHForCausalLm<B>,
        config: CacheConfig,
        batch_size: usize,
        device: &B::Device,
    ) -> Result<Self> {
        Self::new_with_staging(model, config, batch_size, 0, device)
    }

    /// Allocates the main hybrid bank and independent MTP KV bank.
    ///
    /// A non-zero `staging_capacity` makes every stateful main component and
    /// the MTP attention cache transactional. The ordinary decode constructor
    /// passes zero, so reference inference pays no proposal-storage cost.
    pub(super) fn new_with_staging(
        model: &NemotronHForCausalLm<B>,
        config: CacheConfig,
        batch_size: usize,
        staging_capacity: usize,
        device: &B::Device,
    ) -> Result<Self> {
        if config.capacity() > model.config().max_position_embeddings {
            return Err(Error::configuration(
                "nemotron_h.decode_state.capacity",
                format!(
                    "{} exceeds max_position_embeddings {}",
                    config.capacity(),
                    model.config().max_position_embeddings
                ),
            ));
        }
        let (kv_spec, recurrent_spec, mtp_kv_spec) = cache_specs(model, staging_capacity)?;
        let mut tensors = TensorDecodeState::new(config, batch_size, device)?;
        if let Some(spec) = kv_spec {
            let index = tensors.main_mut().add_native_kv(spec)?;
            if index != KV_GROUP {
                return Err(Error::Cache(format!(
                    "Nemotron-H KV group must be {KV_GROUP}, found {index}"
                )));
            }
        }
        if let Some(spec) = recurrent_spec {
            let index = tensors.main_mut().add_recurrent(spec)?;
            if index != RECURRENT_GROUP {
                return Err(Error::Cache(format!(
                    "Nemotron-H recurrent group must be {RECURRENT_GROUP}, found {index}"
                )));
            }
        }
        if let Some(spec) = mtp_kv_spec {
            let index = tensors.enable_mtp()?.add_native_kv(spec)?;
            if index != KV_GROUP {
                return Err(Error::Cache(format!(
                    "Nemotron-H MTP KV group must be {KV_GROUP}, found {index}"
                )));
            }
        }
        Ok(Self {
            tensors,
            kv_spec,
            recurrent_spec,
            mtp_kv_spec,
            staging_capacity,
        })
    }

    /// Borrows the shared tensor cache for inspection and lifecycle diagnostics.
    #[must_use]
    pub const fn tensor_state(&self) -> &TensorDecodeState<B> {
        &self.tensors
    }

    /// Returns the number of cached attention blocks.
    #[must_use]
    pub fn attention_layer_count(&self) -> usize {
        self.kv_spec.map_or(0, NativeKvCacheSpec::layers)
    }

    /// Returns the number of cached Mamba2 blocks.
    #[must_use]
    pub fn mamba_layer_count(&self) -> usize {
        self.recurrent_spec.map_or(0, RecurrentStateSpec::layers)
    }

    /// Returns the number of independently cached MTP attention blocks.
    #[must_use]
    pub fn mtp_layer_count(&self) -> usize {
        self.mtp_kv_spec.map_or(0, NativeKvCacheSpec::layers)
    }

    /// Returns the maximum tentative suffix retained per sequence and bank.
    #[must_use]
    pub const fn staging_capacity(&self) -> usize {
        self.staging_capacity
    }

    /// Begins an atomic target-plus-MTP proposal suffix.
    ///
    /// # Errors
    ///
    /// Returns a cache error for a nested transaction or a state allocated
    /// without staging capacity.
    pub fn mark(&mut self) -> Result<TensorCacheMark> {
        self.tensors.begin_transaction().map_err(Into::into)
    }

    /// Retains independent prefixes of the target and shifted MTP suffixes.
    ///
    /// # Errors
    ///
    /// Returns a cache error for a stale mark or a prefix longer than the proposal.
    pub fn commit_prefixes(
        &mut self,
        mark: TensorCacheMark,
        target_tokens: usize,
        mtp_tokens: usize,
    ) -> Result<()> {
        self.tensors
            .commit_uniform_prefixes(mark, target_tokens, Some(mtp_tokens))
            .map_err(Into::into)
    }

    /// Discards every target and MTP tensor staged after `mark`.
    ///
    /// # Errors
    ///
    /// Returns a cache error for a stale mark.
    pub fn rollback(&mut self, mark: TensorCacheMark) -> Result<()> {
        self.tensors.rollback(mark).map_err(Into::into)
    }

    /// Proves that a cache was allocated for this model's exact topology.
    pub(super) fn validate_for_model(&self, model: &NemotronHForCausalLm<B>) -> Result<()> {
        let (kv, recurrent, mtp) = cache_specs(model, self.staging_capacity)?;
        if self.kv_spec != kv
            || self.recurrent_spec != recurrent
            || self.mtp_kv_spec != mtp
            || self.tensors.main().native_kv_count() != usize::from(kv.is_some())
            || self.tensors.main().recurrent_count() != usize::from(recurrent.is_some())
        {
            return Err(Error::Cache(
                "Nemotron-H decode state is incompatible with the model".into(),
            ));
        }
        if let Some(spec) = kv
            && self.tensors.main().native_kv(KV_GROUP)?.spec() != spec
        {
            return Err(Error::Cache(
                "Nemotron-H KV cache geometry is incompatible".into(),
            ));
        }
        if let Some(spec) = recurrent
            && self.tensors.main().recurrent(RECURRENT_GROUP)?.spec() != spec
        {
            return Err(Error::Cache(
                "Nemotron-H recurrent cache geometry is incompatible".into(),
            ));
        }
        match (mtp, self.tensors.mtp()) {
            (None, None) => {}
            (Some(spec), Some(bank))
                if bank.native_kv_count() == 1
                    && bank.recurrent_count() == 0
                    && bank.native_kv(KV_GROUP)?.spec() == spec => {}
            _ => {
                return Err(Error::Cache(
                    "Nemotron-H MTP cache geometry is incompatible".into(),
                ));
            }
        }
        Ok(())
    }
}

impl<B: Backend> ArchitectureDecodeState<B> for NemotronHDecodeState<B> {
    /// Returns the number of independently cached sequence rows.
    fn batch_size(&self) -> usize {
        self.tensors.batch_size()
    }

    /// Returns the preallocated logical token capacity per row.
    fn capacity(&self) -> usize {
        self.tensors.capacity()
    }

    /// Borrows committed semantic-token lengths for all rows.
    fn lengths(&self) -> &[usize] {
        self.tensors.lengths()
    }

    /// Clears logical cursors and every KV/recurrent tensor.
    fn reset(&mut self) -> Result<()> {
        self.tensors.reset().map_err(Into::into)
    }

    /// Reorders or duplicates every cache representation atomically.
    fn reorder(&mut self, indices: &[usize]) -> Result<()> {
        self.tensors.reorder(indices).map_err(Into::into)
    }

    /// Applies only cache-representation-safe logical truncations.
    ///
    /// The shifted MTP cursor follows the target: it rewinds to at most
    /// `target - 1` and never advances.
    fn truncate(&mut self, lengths: &[usize]) -> Result<()> {
        let Some(current_mtp) = self.tensors.mtp_lengths() else {
            return self.tensors.truncate(lengths).map_err(Into::into);
        };
        if lengths.len() != current_mtp.len() {
            return Err(
                TensorCacheError::Metadata(DecodeStateError::SequenceCountMismatch {
                    expected: current_mtp.len(),
                    actual: lengths.len(),
                })
                .into(),
            );
        }
        let mtp_lengths = lengths
            .iter()
            .zip(current_mtp)
            .map(|(&target, &mtp)| mtp_length_after_truncate(target, mtp))
            .collect::<Vec<_>>();
        self.tensors
            .truncate_prefixes(lengths, &mtp_lengths)
            .map_err(Into::into)
    }

    /// Returns the device shared by all cache tensors.
    fn device(&self) -> &B::Device {
        self.tensors.device()
    }
}

impl<B: Backend> CausalLanguageModel<B> for NemotronHForCausalLm<B> {
    /// Validated heterogeneous backbone and optional MTP configuration.
    type Config = super::super::NemotronHConfig;
    /// External state for attention, recurrence, convolution, and shifted MTP.
    type State = NemotronHDecodeState<B>;

    /// Runs the differentiable cache-free graph.
    fn forward(
        &self,
        input: CausalLmInput<B>,
        options: &ForwardOptions,
    ) -> Result<CausalLmOutput<B>> {
        NemotronHForCausalLm::forward(self, input, options)
    }

    /// Allocates compact cache groups for only attention and Mamba blocks.
    fn new_decode_state(
        &self,
        config: CacheConfig,
        batch_size: usize,
        device: &B::Device,
    ) -> Result<Self::State> {
        NemotronHDecodeState::new(self, config, batch_size, device)
    }

    /// Evaluates and atomically appends an initial semantic prompt.
    fn prefill(
        &self,
        input: CausalLmInput<B>,
        state: &mut Self::State,
        options: &ForwardOptions,
    ) -> Result<CausalLmOutput<B>> {
        self.forward_with_decode_state(input, state, options, StatefulPhase::Prefill)
    }

    /// Evaluates and atomically appends one or more continuation tokens.
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

/// Activations a cached pass keeps besides the final normalized states.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct Retain {
    /// Embedding, every block output, and the final normalized states.
    pub(super) hidden_states: bool,
    /// Packed router outputs of every MoE block.
    pub(super) routers: bool,
}

/// Compact result of one cached backbone pass.
pub(super) struct CachedPass<B: Backend> {
    /// Final normalized states `[B, N, hidden]`; padded slots are unspecified.
    pub(super) hidden: Tensor<B, 3>,
    /// Requested embedding, post-block, and final states, compact `[B, N, hidden]`.
    pub(super) hidden_states: Vec<Tensor<B, 3>>,
    /// Requested router outputs over the `T` packed valid tokens, in MoE block order.
    pub(super) routers: Vec<NemotronHFlatMoeOutput<B>>,
}

/// Per-call constants shared by every block of a cached pass.
struct PassContext<'a, B: Backend> {
    /// Compact layout of the call.
    batch: &'a CompactBatch<B>,
    /// Working length `W_b` of every row before this call.
    baselines: Vec<usize>,
    /// Mamba-2 row plans in ordinary execution.
    mamba_rows: Vec<Mamba2Row>,
    /// Recurrent staging policy inside a transaction; `None` outside one.
    transactional: Option<TransactionalMamba>,
    /// Batched KV read plan through this step, when the model has attention.
    kv_plan: Option<KvBatchReadPlan<B>>,
    /// Absolute query positions `[B, N]`, when exclusions are needed.
    query_positions: Option<Tensor<B, 2, Int>>,
    /// Sliding window that can exclude a key in this call, if any.
    window: Option<NonZeroUsize>,
}

impl<B: Backend> NemotronHForCausalLm<B> {
    /// Evaluates a stateful call in one batched pass and atomically publishes its cache writes.
    fn forward_with_decode_state(
        &self,
        input: CausalLmInput<B>,
        state: &mut NemotronHDecodeState<B>,
        options: &ForwardOptions,
        phase: StatefulPhase,
    ) -> Result<CausalLmOutput<B>> {
        if options.attentions {
            return Err(Error::BackendCapability(
                "Nemotron-H cached attention capture is unavailable".into(),
            ));
        }
        state.validate_for_model(self)?;
        let prepared = prepare_stateful_input(self, &input, state, phase)?;
        validate_logits_selection(&options.logits, prepared.sequence)?;
        let device = input.token_ids.device();
        let batch = CompactBatch::new(&prepared.valid_indices, prepared.sequence, &device)?;
        let retain = Retain {
            hidden_states: options.hidden_states,
            routers: options.router_logits || options.auxiliary_losses,
        };
        if batch.width() == 0 {
            // Every row is finished: no model work and no cache change.
            return self.inactive_output(&prepared, options, &device);
        }
        let tokens = batch.compact_tokens(input.token_ids)?;
        let (pass, step) = self.cached_pass(tokens, &batch, state, retain)?;
        let output = self.stateful_output(pass, &batch, &prepared, options)?;
        state.tensors.publish_step(step)?;
        Ok(output)
    }

    /// Runs the backbone over a compact batch and records its cache writes in one step.
    ///
    /// The returned step is not yet published: callers publish it once every
    /// fallible output computation has succeeded. Attention keys are already
    /// written ahead into unreachable slots (see the module documentation).
    pub(super) fn cached_pass(
        &self,
        tokens: Tensor<B, 2, Int>,
        batch: &CompactBatch<B>,
        state: &mut NemotronHDecodeState<B>,
        retain: Retain,
    ) -> Result<(CachedPass<B>, TensorStepWrites<B>)> {
        let counts = batch.counts().to_vec();
        let baselines = state.tensors.working_lengths().to_vec();
        let transactional = if state.tensors.has_active_transaction() {
            Some(transactional_mamba(&counts, &baselines).map_err(|row| {
                Error::Cache(format!(
                    "Nemotron-H row {row} must be prefilled before a multi-token speculative step"
                ))
            })?)
        } else {
            None
        };
        let mut step = state.tensors.step(&counts)?;
        let has_attention = self.model.count_blocks(NemotronHBlockType::FullAttention) > 0;
        let kv_plan = has_attention
            .then(|| state.tensors.kv_read_plan_through(&step, KV_GROUP))
            .transpose()?;
        let longest = baselines
            .iter()
            .zip(&counts)
            .map(|(&baseline, &count)| baseline + count)
            .max()
            .unwrap_or(0);
        // A window wider than every row's history can exclude nothing.
        let window = self
            .config()
            .sliding_window
            .filter(|&window| window < longest)
            .and_then(NonZeroUsize::new);
        let uniform = batch.plan().is_dense()
            && baselines.windows(2).all(|pair| pair[0] == pair[1])
            && window.is_none();
        let query_positions = (has_attention && !uniform)
            .then(|| batch.query_positions(&baselines))
            .transpose()?;
        let context = PassContext {
            batch,
            mamba_rows: counts
                .iter()
                .zip(&baselines)
                .map(|(&tokens, &baseline)| Mamba2Row {
                    tokens,
                    mode: match mamba_step(tokens, baseline) {
                        MambaStep::Recurrent => NemotronMamba2Mode::Recurrent,
                        MambaStep::Chunked => NemotronMamba2Mode::ChunkedPrefill,
                    },
                })
                .collect(),
            baselines,
            transactional,
            kv_plan,
            query_positions,
            window,
        };

        let mut hidden = self.model.embed_tokens.forward(tokens);
        let mut hidden_states = Vec::with_capacity(if retain.hidden_states {
            self.model.layers.len() + 2
        } else {
            0
        });
        if retain.hidden_states {
            hidden_states.push(hidden.clone());
        }
        let mut routers = Vec::new();
        let mut attention_index = 0;
        let mut mamba_index = 0;
        for layer in &self.model.layers {
            let residual = hidden.clone();
            let normalized = layer.norm.forward(hidden);
            let mixed = match layer.mixer()? {
                BlockMixer::Mamba2(mixer) => {
                    let mixed =
                        cached_mamba(mixer, normalized, mamba_index, state, &mut step, &context)?;
                    mamba_index += 1;
                    mixed
                }
                BlockMixer::Attention(attention) => {
                    let mixed = cached_attention(
                        attention,
                        normalized,
                        attention_index,
                        state,
                        &mut step,
                        &context,
                    )?;
                    attention_index += 1;
                    mixed
                }
                BlockMixer::Mlp(mlp) => mlp.forward(normalized)?,
                BlockMixer::Moe(moe) => {
                    let routed = moe.forward_tokens(batch.pack(normalized)?)?;
                    let mixed = batch.unpack(routed.hidden_state.clone())?;
                    if retain.routers {
                        routers.push(routed);
                    }
                    mixed
                }
            };
            hidden = residual + mixed;
            if retain.hidden_states {
                hidden_states.push(hidden.clone());
            }
        }
        hidden = self.model.norm.forward(hidden);
        if retain.hidden_states {
            hidden_states.push(hidden.clone());
        }
        Ok((
            CachedPass {
                hidden,
                hidden_states,
                routers,
            },
            step,
        ))
    }

    /// Scatters a compact pass back to the caller's rectangular layout and projects logits.
    fn stateful_output(
        &self,
        pass: CachedPass<B>,
        batch: &CompactBatch<B>,
        prepared: &StatefulRows,
        options: &ForwardOptions,
    ) -> Result<CausalLmOutput<B>> {
        let needs_rectangular =
            options.hidden_states || !matches!(options.logits, LogitsSelection::Last);
        let rectangular = needs_rectangular
            .then(|| batch.to_rectangular(pass.hidden.clone()))
            .transpose()?;
        let selected = match &rectangular {
            Some(hidden) if !matches!(options.logits, LogitsSelection::Last) => {
                select_hidden_with_last_columns(
                    hidden.clone(),
                    prepared.last_columns(),
                    &options.logits,
                )?
            }
            _ => batch.last_tokens(pass.hidden)?,
        };
        let logits = self.project_logits(selected)?.cast(DType::F32);
        let hidden_states = pass
            .hidden_states
            .into_iter()
            .map(|hidden| batch.to_rectangular(hidden))
            .collect::<Result<Vec<_>>>()?;
        let routers = pass
            .routers
            .into_iter()
            .map(|routed| {
                let experts_per_token = routed.selected_experts.dims()[1];
                let loss = options
                    .auxiliary_losses
                    .then(|| {
                        nemotron_load_balancing_loss(
                            routed.router_logits.clone().unsqueeze_dim(0),
                            experts_per_token,
                            None,
                        )
                    })
                    .transpose()?;
                let logits = options
                    .router_logits
                    .then(|| batch.to_rectangular(batch.unpack(routed.router_logits)?))
                    .transpose()?;
                Ok(RouterDiagnostics {
                    logits,
                    load_balancing_loss: loss,
                    z_loss: None,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(CausalLmOutput {
            logits,
            final_hidden_state: rectangular.filter(|_| options.hidden_states),
            hidden_states,
            attentions: Vec::new(),
            routers,
            load_balancing_loss: None,
        })
    }

    /// Builds the zero output of a call whose every row is inactive.
    ///
    /// Inactive rows produce exact zeros for logits and every requested
    /// diagnostic; their load-balancing loss over zero valid tokens is zero.
    fn inactive_output(
        &self,
        prepared: &StatefulRows,
        options: &ForwardOptions,
        device: &B::Device,
    ) -> Result<CausalLmOutput<B>> {
        let config = self.config();
        let dtype = self.model.embed_tokens.weight.val().dtype();
        let hidden = Tensor::<B, 3>::zeros(
            [prepared.batch, prepared.sequence, config.hidden_size],
            (device, dtype),
        );
        let selected = select_hidden_with_last_columns(
            hidden.clone(),
            prepared.last_columns(),
            &options.logits,
        )?;
        let logits = self.project_logits(selected)?.cast(DType::F32);
        let blocks = if options.hidden_states {
            self.model.layers.len() + 2
        } else {
            0
        };
        let moe_blocks = if options.router_logits || options.auxiliary_losses {
            self.model.count_blocks(NemotronHBlockType::Moe)
        } else {
            0
        };
        let routers = (0..moe_blocks)
            .map(|_| RouterDiagnostics {
                logits: options.router_logits.then(|| {
                    Tensor::zeros(
                        [prepared.batch, prepared.sequence, config.n_routed_experts],
                        (device, DType::F32),
                    )
                }),
                load_balancing_loss: options
                    .auxiliary_losses
                    .then(|| Tensor::zeros([1], (device, DType::F32))),
                z_loss: None,
            })
            .collect();
        Ok(CausalLmOutput {
            logits,
            final_hidden_state: options.hidden_states.then(|| hidden.clone()),
            hidden_states: vec![hidden; blocks],
            attentions: Vec::new(),
            routers,
            load_balancing_loss: None,
        })
    }
}

/// Advances one Mamba-2 block for every row and records its new state in `step`.
///
/// Outside a transaction (and for single-token transactional steps) each row
/// runs its reference mode and the terminal states are recorded for the whole
/// batch; inactive rows keep their state. A multi-token transactional step
/// runs the recurrent update for every row and records the state after each
/// token, so any accepted prefix can be committed.
fn cached_mamba<B: Backend>(
    mixer: &NemotronMamba2<B>,
    normalized: Tensor<B, 3>,
    layer: usize,
    state: &NemotronHDecodeState<B>,
    step: &mut TensorStepWrites<B>,
    context: &PassContext<'_, B>,
) -> Result<Tensor<B, 3>> {
    let read = state
        .tensors
        .main()
        .recurrent(RECURRENT_GROUP)?
        .read_working_batch(layer)?;
    let current = NemotronMamba2State {
        recurrent: read.matrix().clone(),
        convolution: read.convolution().clone(),
    };
    if context.transactional != Some(TransactionalMamba::Snapshots) {
        let (mixed, next) = mixer.forward_rows(normalized, &context.mamba_rows, &current)?;
        step.push_recurrent_batch(RECURRENT_GROUP, layer, next.recurrent, next.convolution);
        return Ok(mixed);
    }
    let counts = context.batch.counts();
    let (mixed, snapshots) = mixer.forward_rows_with_snapshots(normalized, counts, &current)?;
    for (row, &count) in counts.iter().enumerate() {
        if count > 0 {
            step.push_recurrent_row_snapshots(
                RECURRENT_GROUP,
                layer,
                row,
                snapshots
                    .recurrent
                    .clone()
                    .narrow(0, row, 1)
                    .narrow(1, 0, count),
                snapshots
                    .convolution
                    .clone()
                    .narrow(0, row, 1)
                    .narrow(1, 0, count),
            );
        }
    }
    Ok(mixed)
}

/// Writes one attention block's keys ahead and attends over every row's history.
///
/// Keys and values `[B, kv_heads, N, head_dim]` land in slots
/// `[W_b, W_b + n_b)` of the full cache; attention then reads the zero-copy
/// view `[0, max_b(W_b + n_b))`. Invalid slots (beyond a row's end) are
/// excluded and their values zeroed, so stale bytes can never reach a row.
fn cached_attention<B: Backend>(
    attention: &GroupedQueryAttention<B>,
    normalized: Tensor<B, 3>,
    layer: usize,
    state: &mut NemotronHDecodeState<B>,
    step: &mut TensorStepWrites<B>,
    context: &PassContext<'_, B>,
) -> Result<Tensor<B, 3>> {
    let mut projection = attention.project_self(normalized)?;
    let plan = context
        .kv_plan
        .as_ref()
        .ok_or_else(|| Error::Cache("Nemotron-H attention block has no KV read plan".into()))?;
    state.tensors.write_kv_ahead(
        step,
        KV_GROUP,
        layer,
        projection.key.clone(),
        projection.value.clone(),
    )?;
    let read = state.tensors.read_kv_through(step, KV_GROUP, layer, plan)?;
    let complete = read
        .retained_positions()
        .iter()
        .zip(context.baselines.iter().zip(context.batch.counts()))
        .all(|(retained, (&baseline, &count))| *retained == (0..baseline + count));
    if !complete {
        return Err(Error::Cache(
            "Nemotron-H full attention cache unexpectedly evicted history".into(),
        ));
    }
    let segment = read
        .materialize()
        .ok_or_else(|| Error::Cache("Nemotron-H attention read returned no keys".into()))?;
    projection.key = segment.keys().clone();
    projection.value = segment.masked_values();
    match &context.query_positions {
        // Dense rows of equal length without an active window: the keys are
        // `[0, W + N)` for every row, which is exactly bottom-right causality.
        None => attention.attend(projection, None, None, true),
        Some(positions) => {
            let mask = segment.exclusion_mask(positions, attention.query_heads, context.window)?;
            attention.attend(projection, Some(mask), None, false)
        }
    }
}

/// Validates devices, shapes, semantic tokens, and capacity before model work.
///
/// The shared preparation reads the mask, token IDs, and explicit positions in
/// one transfer against the working cursors (speculative transactions append
/// beyond committed lengths); this wrapper adds the model-device check and
/// Nemotron-H's bound of every appended length by `max_position_embeddings`.
fn prepare_stateful_input<B: Backend>(
    model: &NemotronHForCausalLm<B>,
    input: &CausalLmInput<B>,
    state: &NemotronHDecodeState<B>,
    phase: StatefulPhase,
) -> Result<StatefulRows> {
    ensure_same_device(
        "nemotron_h.model",
        &model.model.embed_tokens.weight.lazy_device(),
        state.tensors.device(),
        DECODE_STATE,
    )?;
    let rows = prepare_stateful_rows(
        input,
        state.tensors.device(),
        state.tensors.working_lengths(),
        state.tensors.capacity(),
        &StatefulContract {
            names: INPUT_NAMES,
            owner: DECODE_STATE,
            vocabulary: model.config().vocab_size,
            position_limit: Some(model.config().max_position_embeddings),
            empty_rows: phase.empty_rows(),
        },
    )?;
    // The append plan already proved `length + count` representable.
    let exceeds_context = state
        .tensors
        .working_lengths()
        .iter()
        .zip(&rows.token_counts)
        .any(|(&length, &count)| length + count > model.config().max_position_embeddings);
    if exceeds_context {
        return Err(Error::configuration(
            "nemotron_h.position",
            "cached length exceeds max_position_embeddings",
        ));
    }
    Ok(rows)
}

/// Derives compact cache geometry from model weights and schedule counts.
///
/// Speculative states stage attention proposals in place: full storage never
/// evicts, so proposals can live at their final slots and attention keeps its
/// zero-copy write-ahead view inside transactions.
fn cache_specs<B: Backend>(
    model: &NemotronHForCausalLm<B>,
    staging_capacity: usize,
) -> Result<(
    Option<NativeKvCacheSpec>,
    Option<RecurrentStateSpec>,
    Option<NativeKvCacheSpec>,
)> {
    let attention_layers = model.model.count_blocks(NemotronHBlockType::FullAttention);
    let mamba_layers = model
        .model
        .count_blocks(NemotronHBlockType::LinearAttention);
    let dtype = model.model.embed_tokens.weight.val().dtype();
    let staged = |spec: NativeKvCacheSpec| {
        let spec = spec.with_staging_capacity(staging_capacity);
        if staging_capacity > 0 {
            spec.with_in_place_staging()
        } else {
            spec
        }
    };
    let kv = (attention_layers > 0)
        .then(|| {
            NativeKvCacheSpec::new(
                attention_layers,
                model.config().num_key_value_heads,
                model.config().head_dim,
                KvCacheStorage::Full,
                dtype,
            )
            .map(staged)
        })
        .transpose()?;
    let recurrent = (mamba_layers > 0)
        .then(|| {
            let state_dtype = mamba_state_dtype(&model.config().mamba_ssm_cache_dtype)?;
            RecurrentStateSpec::new_with_dtypes(
                mamba_layers,
                model.config().mamba_num_heads,
                model.config().mamba_head_dim,
                model.config().ssm_state_size,
                model
                    .config()
                    .mamba_num_heads
                    .checked_mul(model.config().mamba_head_dim)
                    .and_then(|intermediate| {
                        model
                            .config()
                            .n_groups
                            .checked_mul(model.config().ssm_state_size)
                            .and_then(|state| state.checked_mul(2))
                            .and_then(|state| intermediate.checked_add(state))
                    })
                    .ok_or_else(|| {
                        Error::configuration(
                            "nemotron_h.cache",
                            "convolution channel count overflows usize",
                        )
                    })?,
                model.config().conv_kernel - 1,
                state_dtype,
                dtype,
            )
            .map(|spec| spec.with_staging_capacity(staging_capacity))
            .map_err(Error::from)
        })
        .transpose()?;
    let mtp = (staging_capacity > 0)
        .then_some(model.next_token_head.as_ref())
        .flatten()
        .map(|_| {
            NativeKvCacheSpec::new(
                1,
                model.config().num_key_value_heads,
                model.config().head_dim,
                KvCacheStorage::Full,
                dtype,
            )
            .map(|spec| spec.with_staging_capacity(staging_capacity))
        })
        .transpose()?;
    Ok((kv, recurrent, mtp))
}

/// Converts NVIDIA's serialized recurrent-cache dtype into Burn's runtime dtype.
fn mamba_state_dtype(name: &str) -> Result<DType> {
    match name {
        "float32" => Ok(DType::F32),
        "float16" => Ok(DType::F16),
        "bfloat16" => Ok(DType::BF16),
        _ => Err(Error::configuration(
            "nemotron_h.mamba_ssm_cache_dtype",
            format!("unsupported recurrent cache dtype {name:?}"),
        )),
    }
}

#[cfg(test)]
#[path = "decode_tests.rs"]
mod tests;
