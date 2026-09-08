//! Atomic external cache execution for the Nemotron-H hybrid decoder.
//!
//! Attention and Mamba blocks occupy separate compact cache index spaces even
//! though checkpoint layers are interleaved. Attention storage is
//! `[batch, kv_heads, capacity, head_dim]`; recurrent storage is
//! `[batch, mamba_heads, head_dim, state_size]` plus the final `kernel - 1`
//! convolution inputs. Dense and MoE blocks require no persistent state.
//!
//! Each call compacts semantic tokens per batch row, evaluates every block into
//! operation-local tensor handles, validates exact cache coverage, advances the
//! shared logical cursor, and only then publishes KV and recurrent tensors. A
//! returned error therefore leaves both cache representations byte-for-byte at
//! their retry baseline. Fully masked decode rows are exact no-ops; fully masked
//! prefill rows are rejected.

use std::collections::BTreeSet;

use burn::tensor::{Bool, ElementConversion, Int, Tensor, TensorData, backend::Backend};

use crate::{
    CausalLanguageModel, CausalLmInput, CausalLmOutput, DecodeInput, Error, ForwardOptions, Result,
    architecture::{DecodeState as ArchitectureDecodeState, RouterDiagnostics, TokenMask},
    cache::{
        CacheConfig, DecodeStateError, KvCacheStorage, NativeKvCacheSpec, RecurrentStateSpec,
        TensorCacheError, TensorCacheMark, TensorDecodeState,
    },
};

use super::mamba2::NemotronMamba2State;
use super::{
    NemotronHBlockType, NemotronHForCausalLm, NemotronHMoeOutput, NemotronMamba2Mode,
    nemotron_load_balancing_loss, select_hidden, validate_token_ids,
};

pub(super) const KV_GROUP: usize = 0;
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
    /// Returns whether completely inactive rows are legal.
    const fn permits_inactive_rows(self) -> bool {
        matches!(self, Self::Decode)
    }
}

/// External KV, recurrent, convolution, and logical-length state.
#[derive(Debug)]
pub struct NemotronHDecodeState<B: Backend> {
    pub(super) tensors: TensorDecodeState<B>,
    kv_spec: Option<NativeKvCacheSpec>,
    recurrent_spec: Option<RecurrentStateSpec>,
    mtp_kv_spec: Option<NativeKvCacheSpec>,
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
    pub fn mark(&mut self) -> Result<TensorCacheMark> {
        self.tensors.begin_transaction().map_err(Into::into)
    }

    /// Retains independent prefixes of the target and shifted MTP suffixes.
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
    fn truncate(&mut self, lengths: &[usize]) -> Result<()> {
        self.tensors.truncate(lengths).map_err(Into::into)
    }

    /// Returns the device shared by all cache tensors.
    fn device(&self) -> &B::Device {
        self.tensors.device()
    }
}

impl<B: Backend> CausalLanguageModel<B> for NemotronHForCausalLm<B> {
    type Config = super::super::NemotronHConfig;
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

impl<B: Backend> NemotronHForCausalLm<B> {
    /// Evaluates a compact segment and atomically publishes all cache kinds.
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
        if state.tensors.has_active_transaction()
            && (prepared.batch != 1 || prepared.token_counts != [1])
        {
            return Err(Error::Generation(
                "transactional Nemotron-H target decoding accepts exactly one active batch-one token per call"
                    .into(),
            ));
        }
        let diagnostics = self.model.layers.len() + 2;
        let moe_layers = self
            .model
            .layers
            .iter()
            .filter(|layer| layer.block_type == NemotronHBlockType::Moe)
            .count();
        let retain_routers = options.router_logits || options.auxiliary_losses;
        let mut pending = PendingHybridWrites::default();
        let mut final_rows = Vec::with_capacity(prepared.batch);
        let mut hidden_rows: Vec<Vec<Tensor<B, 3>>> = if options.hidden_states {
            (0..diagnostics)
                .map(|_| Vec::with_capacity(prepared.batch))
                .collect()
        } else {
            Vec::new()
        };
        let mut router_rows: Vec<Vec<RowRouter<B>>> = if retain_routers {
            (0..moe_layers)
                .map(|_| Vec::with_capacity(prepared.batch))
                .collect()
        } else {
            Vec::new()
        };

        for row in 0..prepared.batch {
            let valid = &prepared.valid_indices[row];
            if valid.is_empty() {
                let empty = Tensor::zeros(
                    [1, prepared.sequence, self.config().hidden_size],
                    (
                        &input.token_ids.device(),
                        self.model.embed_tokens.weight.val().dtype(),
                    ),
                );
                final_rows.push(empty.clone());
                for destination in &mut hidden_rows {
                    destination.push(empty.clone());
                }
                for destination in &mut router_rows {
                    destination.push(RowRouter {
                        logits: Tensor::zeros(
                            [1, prepared.sequence, self.config().n_routed_experts],
                            (&input.token_ids.device(), burn::tensor::DType::F32),
                        ),
                        selected: Tensor::zeros(
                            [1, prepared.sequence, self.config().num_experts_per_tok],
                            &input.token_ids.device(),
                        ),
                    });
                }
                continue;
            }
            let selector = selection_tensor::<B>(valid, state.tensors.device())?;
            let tokens = input
                .token_ids
                .clone()
                .slice([row..row + 1, 0..prepared.sequence])
                .select(1, selector);
            validate_token_ids(tokens.clone(), self.config().vocab_size)?;
            let committed_baseline = state.tensors.lengths()[row];
            let baseline = state.tensors.working_lengths()[row];
            let compact = self.forward_compact_row(
                tokens,
                row,
                committed_baseline,
                baseline,
                state.tensors.main(),
                &mut pending,
                options.hidden_states,
                retain_routers,
            )?;
            final_rows.push(scatter_compact(compact.hidden, prepared.sequence, valid)?);
            for (destination, hidden) in hidden_rows.iter_mut().zip(compact.hidden_states) {
                destination.push(scatter_compact(hidden, prepared.sequence, valid)?);
            }
            for (destination, router) in router_rows.iter_mut().zip(compact.routers) {
                destination.push(RowRouter {
                    logits: scatter_compact(router.router_logits, prepared.sequence, valid)?,
                    selected: scatter_compact_int(
                        router.selected_experts,
                        prepared.sequence,
                        valid,
                    )?,
                });
            }
        }

        let hidden = Tensor::cat(final_rows, 0);
        let hidden_states = hidden_rows
            .into_iter()
            .map(|rows| Tensor::cat(rows, 0))
            .collect();
        let routers = router_rows
            .into_iter()
            .map(|rows| {
                let logits = Tensor::cat(rows.iter().map(|row| row.logits.clone()).collect(), 0);
                let selected = Tensor::cat(rows.into_iter().map(|row| row.selected).collect(), 0);
                Ok(RouterDiagnostics {
                    logits: options.router_logits.then(|| logits.clone()),
                    load_balancing_loss: options
                        .auxiliary_losses
                        .then(|| {
                            nemotron_load_balancing_loss(
                                logits,
                                selected,
                                input.token_mask.as_ref(),
                            )
                        })
                        .transpose()?,
                    z_loss: None,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let selected = select_hidden(hidden.clone(), input.token_mask.as_ref(), &options.logits)?;
        let logits = self
            .project_logits(selected)?
            .cast(burn::tensor::DType::F32);
        pending.commit(self, state, &prepared.token_counts)?;
        Ok(CausalLmOutput {
            logits,
            final_hidden_state: options.hidden_states.then_some(hidden),
            hidden_states,
            attentions: Vec::new(),
            routers,
            load_balancing_loss: None,
        })
    }

    /// Runs one compact semantic row against unpublished cache snapshots.
    #[allow(clippy::too_many_arguments)]
    fn forward_compact_row(
        &self,
        tokens: Tensor<B, 2, Int>,
        row: usize,
        committed_baseline: usize,
        baseline: usize,
        bank: &crate::cache::TensorCacheBank<B>,
        pending: &mut PendingHybridWrites<B>,
        retain_hidden: bool,
        retain_routers: bool,
    ) -> Result<CompactRowOutput<B>> {
        let mut hidden = self.model.embed_tokens.forward(tokens);
        let mut hidden_states = Vec::with_capacity(if retain_hidden {
            self.model.layers.len() + 2
        } else {
            0
        });
        let mut routers = Vec::new();
        if retain_hidden {
            hidden_states.push(hidden.clone());
        }
        let mut attention_index = 0;
        let mut mamba_index = 0;
        for layer in &self.model.layers {
            let residual = hidden.clone();
            let normalized = layer.norm.forward(hidden);
            match layer.block_type {
                NemotronHBlockType::LinearAttention => {
                    let mixer = layer.mamba.as_ref().ok_or_else(|| {
                        Error::Cache("Nemotron-H Mamba block lost its mixer".into())
                    })?;
                    let cache = bank.recurrent(RECURRENT_GROUP)?;
                    let cached = if cache.has_active_staging() {
                        cache.read_working_sequence(mamba_index, row)?
                    } else {
                        cache.read_sequence(mamba_index, row)?
                    };
                    let state = NemotronMamba2State {
                        recurrent: cached.matrix().clone(),
                        convolution: cached.convolution().clone(),
                    };
                    let mode = if normalized.dims()[1] == 1 {
                        NemotronMamba2Mode::Recurrent
                    } else {
                        NemotronMamba2Mode::ChunkedPrefill
                    };
                    let (mixed, next) = mixer.forward_with_state(normalized, None, &state, mode)?;
                    hidden = residual + mixed;
                    pending.recurrent.push(PendingRecurrentWrite {
                        layer: mamba_index,
                        row,
                        matrix: next.recurrent,
                        convolution: next.convolution,
                    });
                    mamba_index += 1;
                }
                NemotronHBlockType::FullAttention => {
                    let attention = layer.attention.as_ref().ok_or_else(|| {
                        Error::Cache("Nemotron-H attention block lost its mixer".into())
                    })?;
                    let mut projection = attention.project_self(normalized)?;
                    let keys = projection.key.clone();
                    let values = projection.value.clone();
                    let query = projection.query.dims()[2];
                    let cache = bank.native_kv(KV_GROUP)?;
                    let history = if cache.has_active_staging() {
                        cache.read_working_sequence(attention_index, row, committed_baseline)?
                    } else {
                        cache.read_sequence(attention_index, row, baseline)?
                    };
                    if history.retained_positions() != (0..baseline) {
                        return Err(Error::Cache(
                            "Nemotron-H full cache unexpectedly evicted history".into(),
                        ));
                    }
                    if let Some((past_keys, past_values)) = history.materialize() {
                        projection.key = Tensor::cat(vec![past_keys, projection.key], 2);
                        projection.value = Tensor::cat(vec![past_values, projection.value], 2);
                    }
                    let mask = cached_attention_mask::<B>(
                        baseline,
                        query,
                        attention.query_heads,
                        self.config().sliding_window,
                        &projection.query.device(),
                    )?;
                    hidden = residual + attention.attend(projection, Some(mask), None, false)?;
                    pending.kv.push(PendingKvWrite {
                        layer: attention_index,
                        row,
                        start: baseline,
                        keys,
                        values,
                    });
                    attention_index += 1;
                }
                NemotronHBlockType::Mlp => {
                    hidden = residual
                        + layer
                            .mlp
                            .as_ref()
                            .ok_or_else(|| {
                                Error::Cache("Nemotron-H MLP block lost its mixer".into())
                            })?
                            .forward(normalized)?;
                }
                NemotronHBlockType::Moe => {
                    let routed = layer
                        .moe
                        .as_ref()
                        .ok_or_else(|| Error::Cache("Nemotron-H MoE block lost its mixer".into()))?
                        .forward(normalized)?;
                    hidden = residual + routed.hidden_state.clone();
                    if retain_routers {
                        routers.push(routed);
                    }
                }
            }
            if retain_hidden {
                hidden_states.push(hidden.clone());
            }
        }
        hidden = self.model.norm.forward(hidden);
        if retain_hidden {
            hidden_states.push(hidden.clone());
        }
        Ok(CompactRowOutput {
            hidden,
            hidden_states,
            routers,
        })
    }
}

/// One compact row's model result.
struct CompactRowOutput<B: Backend> {
    hidden: Tensor<B, 3>,
    hidden_states: Vec<Tensor<B, 3>>,
    routers: Vec<NemotronHMoeOutput<B>>,
}

/// Deferred native-KV append.
struct PendingKvWrite<B: Backend> {
    layer: usize,
    row: usize,
    start: usize,
    keys: Tensor<B, 4>,
    values: Tensor<B, 4>,
}

/// Deferred recurrent terminal-state replacement.
struct PendingRecurrentWrite<B: Backend> {
    layer: usize,
    row: usize,
    matrix: Tensor<B, 4>,
    convolution: Tensor<B, 3>,
}

/// Complete operation-local cache write set.
struct PendingHybridWrites<B: Backend> {
    kv: Vec<PendingKvWrite<B>>,
    recurrent: Vec<PendingRecurrentWrite<B>>,
}

impl<B: Backend> Default for PendingHybridWrites<B> {
    /// Creates an empty operation-local write set.
    fn default() -> Self {
        Self {
            kv: Vec::new(),
            recurrent: Vec::new(),
        }
    }
}

impl<B: Backend> PendingHybridWrites<B> {
    /// Validates complete active-row coverage then publishes both cache kinds.
    fn commit(
        self,
        model: &NemotronHForCausalLm<B>,
        state: &mut NemotronHDecodeState<B>,
        token_counts: &[usize],
    ) -> Result<()> {
        let attention_layers = model
            .model
            .layers
            .iter()
            .filter(|layer| layer.block_type == NemotronHBlockType::FullAttention)
            .count();
        let mamba_layers = model
            .model
            .layers
            .iter()
            .filter(|layer| layer.block_type == NemotronHBlockType::LinearAttention)
            .count();
        self.validate_coverage(state, token_counts, attention_layers, mamba_layers)?;
        if state.tensors.has_active_transaction() {
            // Every speculative target step is exactly one token. Keeping a
            // snapshot at each call boundary is required for Mamba2 because
            // its terminal recurrence cannot be inverted after rejection.
            if token_counts.iter().any(|&count| count > 1) {
                return Err(Error::Cache(
                    "transactional Nemotron-H hybrid decode accepts at most one token per row and call"
                        .into(),
                ));
            }
            let PendingHybridWrites { kv, recurrent } = self;
            if !kv.is_empty() {
                let cache = state.tensors.main_mut().native_kv_mut(KV_GROUP)?;
                for write in kv {
                    cache.stage_sequence(write.layer, write.row, write.keys, write.values)?;
                }
            }
            if !recurrent.is_empty() {
                let cache = state.tensors.main_mut().recurrent_mut(RECURRENT_GROUP)?;
                for write in recurrent {
                    cache.stage_sequence_snapshots(
                        write.layer,
                        write.row,
                        write.matrix.unsqueeze_dim::<5>(1),
                        write.convolution.unsqueeze_dim::<4>(1),
                    )?;
                }
            }
            state.tensors.advance(token_counts)?;
            return Ok(());
        }
        let kv = if attention_layers > 0 {
            let cache = state.tensors.main().native_kv(KV_GROUP)?;
            Some((
                KV_GROUP,
                self.kv
                    .into_iter()
                    .map(|write| {
                        cache.prepare_sequence_write(
                            write.layer,
                            write.row,
                            write.start,
                            write.keys,
                            write.values,
                        )
                    })
                    .collect::<core::result::Result<Vec<_>, _>>()?,
            ))
        } else {
            None
        };
        let recurrent = if mamba_layers > 0 {
            let cache = state.tensors.main().recurrent(RECURRENT_GROUP)?;
            Some((
                RECURRENT_GROUP,
                self.recurrent
                    .into_iter()
                    .map(|write| {
                        cache.prepare_sequence_state(
                            write.layer,
                            write.row,
                            write.matrix,
                            write.convolution,
                        )
                    })
                    .collect::<core::result::Result<Vec<_>, _>>()?,
            ))
        } else {
            None
        };
        let prepared = state.tensors.main().prepare_direct_writes(kv, recurrent)?;
        state.tensors.advance(token_counts)?;
        state
            .tensors
            .main_mut()
            .apply_prepared_direct_writes(prepared);
        Ok(())
    }

    /// Proves exact `(layer,row)` coverage for both state representations.
    fn validate_coverage(
        &self,
        state: &NemotronHDecodeState<B>,
        token_counts: &[usize],
        attention_layers: usize,
        mamba_layers: usize,
    ) -> Result<()> {
        if token_counts.len() != state.tensors.batch_size() {
            return Err(Error::Cache(
                "Nemotron-H pending token counts have the wrong batch size".into(),
            ));
        }
        let active = token_counts.iter().filter(|&&count| count > 0).count();
        if self.kv.len() != attention_layers * active
            || self.recurrent.len() != mamba_layers * active
        {
            return Err(Error::Cache(format!(
                "Nemotron-H pending coverage is kv={}/{}, recurrent={}/{}",
                self.kv.len(),
                attention_layers * active,
                self.recurrent.len(),
                mamba_layers * active
            )));
        }
        let mut kv_seen = BTreeSet::new();
        for write in &self.kv {
            let count = token_counts.get(write.row).copied().unwrap_or(0);
            if count == 0
                || write.layer >= attention_layers
                || write.start != state.tensors.working_lengths()[write.row]
                || write.keys.dims()[2] != count
                || write.values.dims()[2] != count
                || !kv_seen.insert((write.layer, write.row))
            {
                return Err(Error::Cache(format!(
                    "invalid Nemotron-H KV write at layer {}, row {}",
                    write.layer, write.row
                )));
            }
        }
        let mut recurrent_seen = BTreeSet::new();
        for write in &self.recurrent {
            let count = token_counts.get(write.row).copied().unwrap_or(0);
            if count == 0
                || write.layer >= mamba_layers
                || !recurrent_seen.insert((write.layer, write.row))
            {
                return Err(Error::Cache(format!(
                    "invalid Nemotron-H recurrent write at layer {}, row {}",
                    write.layer, write.row
                )));
            }
        }
        Ok(())
    }
}

/// Reconstructed rectangular router tensors for one row.
struct RowRouter<B: Backend> {
    logits: Tensor<B, 3>,
    selected: Tensor<B, 3, Int>,
}

/// Validated host-side state transition metadata.
struct PreparedStatefulInput {
    batch: usize,
    sequence: usize,
    valid_indices: Vec<Vec<usize>>,
    token_counts: Vec<usize>,
}

/// Validates devices, shapes, semantic tokens, and capacity before model work.
fn prepare_stateful_input<B: Backend>(
    model: &NemotronHForCausalLm<B>,
    input: &CausalLmInput<B>,
    state: &NemotronHDecodeState<B>,
    phase: StatefulPhase,
) -> Result<PreparedStatefulInput> {
    let [batch, sequence] = input.token_ids.dims();
    if batch == 0 || sequence == 0 {
        return Err(Error::shape(
            "nemotron_h.input_ids",
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
    ensure_device(&input.token_ids.device(), state.tensors.device())?;
    ensure_device(
        &model.model.embed_tokens.weight.val().device(),
        state.tensors.device(),
    )?;
    if let Some(mask) = &input.token_mask {
        ensure_device(&mask.as_tensor().device(), state.tensors.device())?;
        if mask.as_tensor().dims() != [batch, sequence] {
            return Err(Error::shape(
                "nemotron_h.token_mask",
                format!("[{batch}, {sequence}]"),
                format!("{:?}", mask.as_tensor().dims()),
            ));
        }
    }
    if let Some(positions) = &input.position_ids {
        ensure_device(&positions.device(), state.tensors.device())?;
        if positions.dims() != [batch, sequence] {
            return Err(Error::shape(
                "nemotron_h.position_ids",
                format!("[{batch}, {sequence}]"),
                format!("{:?}", positions.dims()),
            ));
        }
    }
    let valid_indices = semantic_indices(input.token_mask.as_ref(), batch, sequence)?;
    if !phase.permits_inactive_rows() && valid_indices.iter().any(Vec::is_empty) {
        return Err(Error::shape(
            "nemotron_h.token_mask",
            "at least one valid token in each prefill row",
            "a fully masked row",
        ));
    }
    let token_counts = valid_indices.iter().map(Vec::len).collect::<Vec<_>>();
    for (row, (&baseline, &additional)) in state
        .tensors
        .working_lengths()
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
        if requested > model.config().max_position_embeddings {
            return Err(Error::configuration(
                "nemotron_h.position",
                "cached length exceeds max_position_embeddings",
            ));
        }
    }
    if let Some(positions) = &input.position_ids {
        for (row, valid) in valid_indices.iter().enumerate() {
            if valid.is_empty() {
                continue;
            }
            let selector = selection_tensor::<B>(valid, state.tensors.device())?;
            let compact = positions
                .clone()
                .slice([row..row + 1, 0..sequence])
                .select(1, selector);
            let minimum = compact
                .clone()
                .min()
                .try_into_scalar()
                .map_err(|error| Error::BackendCapability(error.to_string()))?
                .elem::<i64>();
            let maximum = compact
                .max()
                .try_into_scalar()
                .map_err(|error| Error::BackendCapability(error.to_string()))?
                .elem::<i64>();
            if minimum < 0 || maximum as usize >= model.config().max_position_embeddings {
                return Err(Error::shape(
                    "nemotron_h.position_ids",
                    format!("values in 0..{}", model.config().max_position_embeddings),
                    format!("observed {minimum}..={maximum}"),
                ));
            }
        }
    }
    Ok(PreparedStatefulInput {
        batch,
        sequence,
        valid_indices,
        token_counts,
    })
}

/// Reads a true-valid mask into stable semantic source-column indices.
fn semantic_indices<B: Backend>(
    mask: Option<&TokenMask<B>>,
    batch: usize,
    sequence: usize,
) -> Result<Vec<Vec<usize>>> {
    let Some(mask) = mask else {
        let dense = (0..sequence).collect::<Vec<_>>();
        return Ok((0..batch).map(|_| dense.clone()).collect());
    };
    let values =
        crate::architecture::tensor_bool_values(mask.as_tensor().clone(), "Nemotron-H token mask")?;
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

/// Converts semantic source columns into a Burn selector.
fn selection_tensor<B: Backend>(
    indices: &[usize],
    device: &B::Device,
) -> Result<Tensor<B, 1, Int>> {
    let indices = indices
        .iter()
        .map(|&index| {
            i64::try_from(index).map_err(|_| {
                Error::configuration("nemotron_h.token_mask", "column index exceeds i64")
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let count = indices.len();
    Ok(Tensor::from_data(TensorData::new(indices, [count]), device))
}

/// Builds `excluded(q,k) = k > q || k + window <= q` for cached attention.
pub(super) fn cached_attention_mask<B: Backend>(
    baseline: usize,
    query_sequence: usize,
    heads: usize,
    window: Option<usize>,
    device: &B::Device,
) -> Result<Tensor<B, 4, Bool>> {
    let key_sequence = baseline.checked_add(query_sequence).ok_or_else(|| {
        Error::configuration("nemotron_h.attention_mask", "key length overflows usize")
    })?;
    let start = i64::try_from(baseline)
        .map_err(|_| Error::configuration("nemotron_h.attention_mask", "offset exceeds i64"))?;
    let end = i64::try_from(key_sequence)
        .map_err(|_| Error::configuration("nemotron_h.attention_mask", "length exceeds i64"))?;
    let query = Tensor::<B, 1, Int>::arange(start..end, device)
        .reshape([1, 1, query_sequence, 1])
        .expand([1, heads, query_sequence, key_sequence]);
    let key = Tensor::<B, 1, Int>::arange(0..end, device)
        .reshape([1, 1, 1, key_sequence])
        .expand([1, heads, query_sequence, key_sequence]);
    let causal = key.clone().greater(query.clone());
    Ok(match window {
        Some(window) => {
            let window = i64::try_from(window).map_err(|_| {
                Error::configuration(
                    "nemotron_h.sliding_window",
                    "window exceeds portable signed index range",
                )
            })?;
            causal.bool_or((key + window).lower_equal(query))
        }
        None => causal,
    })
}

/// Scatters compact floating rows back to rectangular input columns.
fn scatter_compact<B: Backend>(
    compact: Tensor<B, 3>,
    sequence: usize,
    indices: &[usize],
) -> Result<Tensor<B, 3>> {
    let [batch, compact_sequence, width] = compact.dims();
    if batch != 1
        || compact_sequence != indices.len()
        || indices.windows(2).any(|pair| pair[0] >= pair[1])
        || indices.last().is_some_and(|&index| index >= sequence)
    {
        return Err(Error::shape(
            "nemotron_h.compact_hidden",
            "[1, valid_columns, width] with sorted unique indices",
            format!("shape={:?}, indices={indices:?}", compact.dims()),
        ));
    }
    let device = compact.device();
    let dtype = compact.dtype();
    let mut segments = Vec::new();
    let mut cursor = 0;
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
    Ok(Tensor::cat(segments, 1))
}

/// Scatters compact discrete router assignments to rectangular columns.
fn scatter_compact_int<B: Backend>(
    compact: Tensor<B, 3, Int>,
    sequence: usize,
    indices: &[usize],
) -> Result<Tensor<B, 3, Int>> {
    let [batch, compact_sequence, width] = compact.dims();
    if batch != 1 || compact_sequence != indices.len() {
        return Err(Error::shape(
            "nemotron_h.compact_router",
            "[1, valid_columns, top_k]",
            format!("{:?}", compact.dims()),
        ));
    }
    let device = compact.device();
    let mut segments = Vec::new();
    let mut cursor = 0;
    for (compact_index, &output_index) in indices.iter().enumerate() {
        if output_index > cursor {
            segments.push(Tensor::zeros([1, output_index - cursor, width], &device));
        }
        segments.push(compact.clone().narrow(1, compact_index, 1));
        cursor = output_index + 1;
    }
    if cursor < sequence {
        segments.push(Tensor::zeros([1, sequence - cursor, width], &device));
    }
    Ok(Tensor::cat(segments, 1))
}

/// Derives compact cache geometry from model weights and schedule counts.
fn cache_specs<B: Backend>(
    model: &NemotronHForCausalLm<B>,
    staging_capacity: usize,
) -> Result<(
    Option<NativeKvCacheSpec>,
    Option<RecurrentStateSpec>,
    Option<NativeKvCacheSpec>,
)> {
    let attention_layers = model
        .model
        .layers
        .iter()
        .filter(|layer| layer.block_type == NemotronHBlockType::FullAttention)
        .count();
    let mamba_layers = model
        .model
        .layers
        .iter()
        .filter(|layer| layer.block_type == NemotronHBlockType::LinearAttention)
        .count();
    let dtype = model.model.embed_tokens.weight.val().dtype();
    let kv = (attention_layers > 0)
        .then(|| {
            NativeKvCacheSpec::new(
                attention_layers,
                model.config().num_key_value_heads,
                model.config().head_dim,
                KvCacheStorage::Full,
                dtype,
            )
            .map(|spec| spec.with_staging_capacity(staging_capacity))
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
fn mamba_state_dtype(name: &str) -> Result<burn::tensor::DType> {
    match name {
        "float32" => Ok(burn::tensor::DType::F32),
        "float16" => Ok(burn::tensor::DType::F16),
        "bfloat16" => Ok(burn::tensor::DType::BF16),
        _ => Err(Error::configuration(
            "nemotron_h.mamba_ssm_cache_dtype",
            format!("unsupported recurrent cache dtype {name:?}"),
        )),
    }
}

/// Rejects mixed-device model/input/cache combinations before dispatch.
fn ensure_device<D: PartialEq>(actual: &D, expected: &D) -> Result<()> {
    if actual != expected {
        return Err(Error::BackendCapability(
            "Nemotron-H model, input, and decode state must share a device".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use burn::{
        backend::Flex,
        tensor::{Bool, Tensor, Tolerance},
    };

    use crate::{
        DecodeState, architecture::TokenMask, models::nemotron_h::model::tests::tiny_config,
    };

    use super::*;

    type TestBackend = Flex<f32, i32>;

    /// Cached chunk/decode execution matches the full stateless continuation.
    #[test]
    fn cached_prefill_decode_matches_stateless() {
        let device = Default::default();
        let model = NemotronHForCausalLm::<TestBackend>::init(tiny_config(false), &device).unwrap();
        let full = model
            .forward(
                CausalLmInput {
                    token_ids: Tensor::from_ints([[1, 2, 3, 4]], &device),
                    token_mask: None,
                    position_ids: None,
                },
                &ForwardOptions::default(),
            )
            .unwrap()
            .logits;
        let mut state = model
            .new_decode_state(CacheConfig::new(1, 8).unwrap(), 1, &device)
            .unwrap();
        assert_eq!(state.attention_layer_count(), 1);
        assert_eq!(state.mamba_layer_count(), 1);
        model
            .prefill(
                CausalLmInput {
                    token_ids: Tensor::from_ints([[1, 2]], &device),
                    token_mask: None,
                    position_ids: None,
                },
                &mut state,
                &ForwardOptions::default(),
            )
            .unwrap();
        let cached = model
            .decode(
                DecodeInput {
                    token_ids: Tensor::from_ints([[3, 4]], &device),
                    token_mask: None,
                    position_ids: None,
                },
                &mut state,
                &ForwardOptions::default(),
            )
            .unwrap()
            .logits;
        assert_eq!(state.lengths(), &[4]);
        cached
            .into_data()
            .assert_approx_eq::<f32>(&full.into_data(), Tolerance::absolute(3e-5));
    }

    /// Sliding attention keeps full storage while stateless and cached math agree.
    #[test]
    fn sliding_window_cached_path_matches_stateless() {
        let device = Default::default();
        let mut config = tiny_config(false);
        config.sliding_window = Some(2);
        let model = NemotronHForCausalLm::<TestBackend>::init(config, &device).unwrap();
        let full = model
            .forward(
                CausalLmInput {
                    token_ids: Tensor::from_ints([[1, 2, 3, 4, 5]], &device),
                    token_mask: None,
                    position_ids: None,
                },
                &ForwardOptions::default(),
            )
            .unwrap()
            .logits;
        let mut state = model
            .new_decode_state(CacheConfig::new(1, 8).unwrap(), 1, &device)
            .unwrap();
        model
            .prefill(
                CausalLmInput {
                    token_ids: Tensor::from_ints([[1, 2, 3]], &device),
                    token_mask: None,
                    position_ids: None,
                },
                &mut state,
                &ForwardOptions::default(),
            )
            .unwrap();
        let cached = model
            .decode(
                DecodeInput {
                    token_ids: Tensor::from_ints([[4, 5]], &device),
                    token_mask: None,
                    position_ids: None,
                },
                &mut state,
                &ForwardOptions::default(),
            )
            .unwrap()
            .logits;
        cached
            .into_data()
            .assert_approx_eq::<f32>(&full.into_data(), Tolerance::absolute(3e-5));
    }

    /// Left padding does not advance recurrent or KV state for semantic tokens.
    #[test]
    fn left_padded_prefill_matches_stateless_last_token() {
        let device = Default::default();
        let model = NemotronHForCausalLm::<TestBackend>::init(tiny_config(false), &device).unwrap();
        let mask = || {
            TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
                [[false, false, true, true, true]].into(),
                &device,
            ))
        };
        let full = model
            .forward(
                CausalLmInput {
                    token_ids: Tensor::from_ints([[0, 0, 1, 2, 3]], &device),
                    token_mask: Some(mask()),
                    position_ids: None,
                },
                &ForwardOptions::default(),
            )
            .unwrap()
            .logits;
        let mut state = model
            .new_decode_state(CacheConfig::new(1, 8).unwrap(), 1, &device)
            .unwrap();
        let cached = model
            .prefill(
                CausalLmInput {
                    token_ids: Tensor::from_ints([[0, 0, 1, 2, 3]], &device),
                    token_mask: Some(mask()),
                    position_ids: None,
                },
                &mut state,
                &ForwardOptions::default(),
            )
            .unwrap()
            .logits;
        assert_eq!(state.lengths(), &[3]);
        cached
            .into_data()
            .assert_approx_eq::<f32>(&full.into_data(), Tolerance::absolute(3e-5));
    }

    /// A failed decode leaves logical, KV, recurrent, and convolution state intact.
    #[test]
    fn failed_decode_is_atomic_across_cache_kinds() {
        let device = Default::default();
        let mut model =
            NemotronHForCausalLm::<TestBackend>::init(tiny_config(false), &device).unwrap();
        let mut state = model
            .new_decode_state(CacheConfig::new(1, 4).unwrap(), 1, &device)
            .unwrap();
        model
            .prefill(
                CausalLmInput {
                    token_ids: Tensor::from_ints([[1, 2]], &device),
                    token_mask: None,
                    position_ids: None,
                },
                &mut state,
                &ForwardOptions::default(),
            )
            .unwrap();
        let kv = state
            .tensors
            .main()
            .native_kv(KV_GROUP)
            .unwrap()
            .keys(0)
            .unwrap()
            .clone()
            .into_data();
        let recurrent = state
            .tensors
            .main()
            .recurrent(RECURRENT_GROUP)
            .unwrap()
            .matrix(0)
            .unwrap()
            .clone()
            .into_data();
        let convolution = state
            .tensors
            .main()
            .recurrent(RECURRENT_GROUP)
            .unwrap()
            .convolution(0)
            .unwrap()
            .clone()
            .into_data();
        // Fail in the final MoE block, after both earlier Mamba and attention
        // blocks have produced operation-local candidate cache updates.
        model.model.layers[3].moe = None;
        assert!(
            model
                .decode(
                    DecodeInput {
                        token_ids: Tensor::from_ints([[3]], &device),
                        token_mask: None,
                        position_ids: None,
                    },
                    &mut state,
                    &ForwardOptions::default(),
                )
                .is_err()
        );
        assert_eq!(state.lengths(), &[2]);
        assert_eq!(
            state
                .tensors
                .main()
                .native_kv(KV_GROUP)
                .unwrap()
                .keys(0)
                .unwrap()
                .clone()
                .into_data(),
            kv
        );
        assert_eq!(
            state
                .tensors
                .main()
                .recurrent(RECURRENT_GROUP)
                .unwrap()
                .matrix(0)
                .unwrap()
                .clone()
                .into_data(),
            recurrent
        );
        assert_eq!(
            state
                .tensors
                .main()
                .recurrent(RECURRENT_GROUP)
                .unwrap()
                .convolution(0)
                .unwrap()
                .clone()
                .into_data(),
            convolution
        );
    }

    /// Inactive decode rows are exact no-ops while active peers advance.
    #[test]
    fn ragged_decode_and_lifecycle_operations_are_safe() {
        let device = Default::default();
        let model = NemotronHForCausalLm::<TestBackend>::init(tiny_config(false), &device).unwrap();
        let mut state = model
            .new_decode_state(CacheConfig::new(2, 8).unwrap(), 2, &device)
            .unwrap();
        model
            .prefill(
                CausalLmInput {
                    token_ids: Tensor::from_ints([[1, 2], [3, 4]], &device),
                    token_mask: None,
                    position_ids: None,
                },
                &mut state,
                &ForwardOptions::default(),
            )
            .unwrap();
        model
            .decode(
                DecodeInput {
                    token_ids: Tensor::from_ints([[0], [5]], &device),
                    token_mask: Some(TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
                        [[false], [true]].into(),
                        &device,
                    ))),
                    position_ids: None,
                },
                &mut state,
                &ForwardOptions::default(),
            )
            .unwrap();
        assert_eq!(state.lengths(), &[2, 3]);
        state.reorder(&[1, 0]).unwrap();
        assert_eq!(state.lengths(), &[3, 2]);
        assert!(state.truncate(&[2, 2]).is_err());
        state.truncate(&[0, 0]).unwrap();
        assert_eq!(state.lengths(), &[0, 0]);
        state.reset().unwrap();
    }

    /// Capacity failures happen before any terminal-state publication.
    #[test]
    fn capacity_error_preserves_lengths() {
        let device = Default::default();
        let model = NemotronHForCausalLm::<TestBackend>::init(tiny_config(false), &device).unwrap();
        let mut state = model
            .new_decode_state(CacheConfig::new(1, 2).unwrap(), 1, &device)
            .unwrap();
        model
            .prefill(
                CausalLmInput {
                    token_ids: Tensor::from_ints([[1, 2]], &device),
                    token_mask: None,
                    position_ids: None,
                },
                &mut state,
                &ForwardOptions::default(),
            )
            .unwrap();
        assert!(
            model
                .decode(
                    DecodeInput {
                        token_ids: Tensor::from_ints([[3]], &device),
                        token_mask: None,
                        position_ids: None,
                    },
                    &mut state,
                    &ForwardOptions::default(),
                )
                .is_err()
        );
        assert_eq!(state.lengths(), &[2]);
    }

    /// Cached mask construction rejects windows that cannot enter signed indices.
    #[cfg(target_pointer_width = "64")]
    #[test]
    fn oversized_cached_attention_window_is_rejected() {
        let device = Default::default();
        assert!(cached_attention_mask::<TestBackend>(0, 1, 1, Some(usize::MAX), &device).is_err());
    }
}
