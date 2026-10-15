//! External native-KV state for Qwen3 MoE prefill and decoding.
//!
//! All attention layers share one preallocated cache group laid out as
//! `[batch, kv_heads, capacity, head_dim]` per key/value tensor. A public call
//! first computes every row and layer into operation-local tensor handles. Only
//! after all fallible model work succeeds does a two-phase cache commit validate
//! every append, advance logical lengths atomically, and publish the opaque
//! prepared assignments. Thus a returned `Err` never poisons a retry baseline.
//!
//! Rectangular masks are semantic: invalid columns neither consume RoPE
//! positions nor enter the cache. Decode permits a fully masked row, defining it
//! as an exact zero-output/no-state-change operation so finished sequences can
//! remain in a heterogeneous batch. Prefill rejects empty prompt rows.

use std::collections::BTreeSet;

use burn::tensor::{Bool, Int, Tensor, backend::Backend};

use crate::{
    CausalLanguageModel, CausalLmInput, CausalLmOutput, DecodeInput, Error, ForwardOptions, Result,
    architecture::{DecodeState as ArchitectureDecodeState, RouterDiagnostics},
    cache::{CacheConfig, KvCacheStorage, NativeKvCacheSpec, TensorCacheError, TensorDecodeState},
    models::shared::{
        EmptyRows, StatefulContract, StatefulRows, concat_rows, ensure_same_device,
        implicit_position_ids, prepare_stateful_rows, select_hidden_with_last_columns,
        validate_logits_selection,
    },
    moe::SparseMoeOutput,
    position::{RopeScaling, RotaryAngles, apply_rotary, rotary_angles},
};

use super::{
    INPUT_NAMES, Qwen3MoeDecoderLayer, Qwen3MoeForCausalLm, Qwen3MoeModel,
    qwen3_moe_load_balancing_loss,
};

/// Single homogeneous native-KV cache group used by all decoder layers.
const NATIVE_KV_GROUP: usize = 0;

/// Owner named when an input tensor is on another device than the cache.
const DECODE_STATE: &str = "Qwen3 MoE decode state";

/// Stateful operation whose fully masked-row policy differs intentionally.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StatefulPhase {
    /// Initial prompts require at least one semantic token per row.
    Prefill,
    /// Finished rows may contribute no token while peers continue.
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

/// Qwen3-specific external decode state.
#[derive(Debug)]
pub struct Qwen3MoeDecodeState<B: Backend> {
    /// Shared physical cache tensors and logical per-sequence cursors.
    tensors: TensorDecodeState<B>,
    /// Validated native-head geometry used to reject incompatible models.
    spec: NativeKvCacheSpec,
}

impl<B: Backend> Qwen3MoeDecodeState<B> {
    /// Allocates one homogeneous native-KV group for all decoder layers.
    fn new(
        model: &Qwen3MoeForCausalLm<B>,
        config: CacheConfig,
        batch_size: usize,
        device: &B::Device,
    ) -> Result<Self> {
        if config.capacity() > model.config().max_position_embeddings {
            return Err(Error::configuration(
                "qwen3_moe.decode_state.capacity",
                format!(
                    "{} exceeds max_position_embeddings {}",
                    config.capacity(),
                    model.config().max_position_embeddings
                ),
            ));
        }
        if config.capacity() > i32::MAX as usize {
            return Err(Error::configuration(
                "qwen3_moe.decode_state.capacity",
                "must fit Burn's portable signed position-index range",
            ));
        }
        let spec = cache_spec(model)?;
        let mut tensors = TensorDecodeState::new(config, batch_size, device)?;
        let group = tensors.main_mut().add_native_kv(spec)?;
        if group != NATIVE_KV_GROUP {
            return Err(Error::Cache(format!(
                "Qwen3 MoE native-KV group index must be {NATIVE_KV_GROUP}, found {group}"
            )));
        }
        Ok(Self { tensors, spec })
    }

    /// Borrows the generic cache for diagnostics and lifecycle integration.
    #[must_use]
    pub const fn tensor_state(&self) -> &TensorDecodeState<B> {
        &self.tensors
    }

    /// Returns the number of attention layers in the preallocated group.
    #[must_use]
    pub const fn layer_count(&self) -> usize {
        self.spec.layers()
    }

    /// Checks that this state belongs to the supplied model and remains intact.
    fn validate_for_model(&self, model: &Qwen3MoeForCausalLm<B>) -> Result<()> {
        let expected = cache_spec(model)?;
        if self.spec != expected
            || self.tensors.main().native_kv_count() != 1
            || self.tensors.main().native_kv(NATIVE_KV_GROUP)?.spec() != expected
        {
            return Err(Error::Cache(
                "Qwen3 MoE decode state is incompatible with the model".into(),
            ));
        }
        Ok(())
    }
}

impl<B: Backend> ArchitectureDecodeState<B> for Qwen3MoeDecodeState<B> {
    /// Returns the number of logical sequence rows represented by this state.
    fn batch_size(&self) -> usize {
        self.tensors.batch_size()
    }

    /// Returns the maximum logical token count per row, excluding padding.
    fn capacity(&self) -> usize {
        self.tensors.capacity()
    }

    /// Borrows committed semantic token counts in the current batch order.
    fn lengths(&self) -> &[usize] {
        self.tensors.lengths()
    }

    /// Resets logical lengths and all cache groups through the shared state contract.
    fn reset(&mut self) -> Result<()> {
        self.tensors.reset().map_err(Into::into)
    }

    /// Reorders or duplicates sequence rows across metadata and every cache group atomically.
    fn reorder(&mut self, indices: &[usize]) -> Result<()> {
        self.tensors.reorder(indices).map_err(Into::into)
    }

    /// Shortens committed KV prefixes after validating every requested sequence length.
    fn truncate(&mut self, lengths: &[usize]) -> Result<()> {
        self.tensors.truncate(lengths).map_err(Into::into)
    }

    /// Returns the device holding this state, independent of model parameter ownership.
    fn device(&self) -> &B::Device {
        self.tensors.device()
    }
}

impl<B: Backend> CausalLanguageModel<B> for Qwen3MoeForCausalLm<B> {
    /// Architecture configuration needed to allocate and validate decode state.
    type Config = super::super::Qwen3MoeConfig;
    /// External native-KV state owned by an inference session.
    type State = Qwen3MoeDecodeState<B>;

    /// Delegates to the stateless family graph without allocating or mutating decode history.
    fn forward(
        &self,
        input: CausalLmInput<B>,
        options: &ForwardOptions,
    ) -> Result<CausalLmOutput<B>> {
        Qwen3MoeForCausalLm::forward(self, input, options)
    }

    /// Allocates external Qwen MoE cache groups after validating model/device geometry.
    fn new_decode_state(
        &self,
        config: CacheConfig,
        batch_size: usize,
        device: &B::Device,
    ) -> Result<Self::State> {
        Qwen3MoeDecodeState::new(self, config, batch_size, device)
    }

    /// Evaluates prompt rows with at least one valid token and commits all cache writes together.
    fn prefill(
        &self,
        input: CausalLmInput<B>,
        state: &mut Self::State,
        options: &ForwardOptions,
    ) -> Result<CausalLmOutput<B>> {
        self.forward_with_decode_state(input, state, options, StatefulPhase::Prefill)
    }

    /// Appends valid tokens while fully masked rows retain their state and return zero outputs.
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

impl<B: Backend> Qwen3MoeForCausalLm<B> {
    /// Evaluates a masked rectangular segment against external committed state.
    fn forward_with_decode_state(
        &self,
        input: CausalLmInput<B>,
        state: &mut Qwen3MoeDecodeState<B>,
        options: &ForwardOptions,
        phase: StatefulPhase,
    ) -> Result<CausalLmOutput<B>> {
        if options.attentions {
            return Err(Error::BackendCapability(
                "Qwen3 MoE cached attention probability capture is unavailable".into(),
            ));
        }
        state.validate_for_model(self)?;
        if state.tensors.has_active_transaction() {
            return Err(TensorCacheError::TransactionActive {
                operation: "run ordinary Qwen3 MoE prefill/decode",
            }
            .into());
        }
        let prepared = prepare_stateful_input(self, &input, state, phase)?;
        validate_logits_selection(&options.logits, prepared.sequence)?;
        let device = input.token_ids.device();
        let diagnostic_count = self.model.layers.len() + 2;
        let sparse_count = self
            .model
            .layers
            .iter()
            .filter(|layer| layer.sparse_mlp.is_some())
            .count();
        let retain_router_logits = options.router_logits || self.config().output_router_logits;
        let retain_routers = retain_router_logits || options.auxiliary_losses;
        let mut pending = PendingKvWrites::new();
        let mut final_rows = Vec::with_capacity(prepared.batch);
        let mut hidden_rows: Vec<Vec<Tensor<B, 3>>> = if options.hidden_states {
            (0..diagnostic_count)
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
                if options.hidden_states {
                    for destination in &mut hidden_rows {
                        destination.push(empty.clone());
                    }
                }
                if retain_routers {
                    for destination in &mut router_rows {
                        destination.push(RowRouter {
                            logits: Tensor::zeros(
                                [1, prepared.sequence, self.config().num_experts],
                                (&device, self.model.embed_tokens.weight.val().dtype()),
                            ),
                        });
                    }
                }
                continue;
            };
            let tokens = plan.compact(&input.token_ids);
            let baseline = state.tensors.lengths()[row];
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
                state.tensors.main(),
                &mut pending,
                options.hidden_states,
                retain_routers,
            )?;
            final_rows.push(plan.scatter(compact.hidden)?);
            if options.hidden_states {
                if compact.hidden_states.len() != diagnostic_count {
                    return Err(Error::Cache(format!(
                        "Qwen3 MoE hidden-state count must be {diagnostic_count}, found {}",
                        compact.hidden_states.len()
                    )));
                }
                for (destination, state) in hidden_rows.iter_mut().zip(compact.hidden_states) {
                    destination.push(plan.scatter(state)?);
                }
            }
            if retain_routers {
                if compact.routers.len() != sparse_count {
                    return Err(Error::Cache(format!(
                        "Qwen3 MoE router count must be {sparse_count}, found {}",
                        compact.routers.len()
                    )));
                }
                for (destination, routed) in router_rows.iter_mut().zip(compact.routers) {
                    destination.push(RowRouter {
                        logits: plan.scatter(routed.router_logits)?,
                    });
                }
            }
        }

        let hidden = concat_rows(final_rows, "qwen3_moe.hidden")?;
        let hidden_states = hidden_rows
            .into_iter()
            .map(|rows| concat_rows(rows, "qwen3_moe.hidden_states"))
            .collect::<Result<Vec<_>>>()?;
        let mut routers = Vec::with_capacity(if retain_router_logits {
            sparse_count
        } else {
            0
        });
        let mut auxiliary_router_logits = Vec::with_capacity(if options.auxiliary_losses {
            sparse_count
        } else {
            0
        });
        for rows in router_rows {
            let logits = concat_rows(
                rows.into_iter().map(|row| row.logits).collect(),
                "qwen3_moe.router_logits",
            )?;
            if options.auxiliary_losses {
                auxiliary_router_logits.push(logits.clone());
            }
            if retain_router_logits {
                routers.push(RouterDiagnostics {
                    logits: Some(logits),
                    load_balancing_loss: None,
                    z_loss: None,
                });
            }
        }
        let load_balancing_loss = options
            .auxiliary_losses
            .then(|| {
                qwen3_moe_load_balancing_loss(
                    &auxiliary_router_logits,
                    self.config().num_experts_per_tok,
                    input.token_mask.as_ref(),
                )
            })
            .transpose()?;
        let selected = select_hidden_with_last_columns(
            hidden.clone(),
            prepared.last_columns(),
            &options.logits,
        )?;
        let logits = self.project_logits(selected)?;
        pending.commit(
            &mut state.tensors,
            self.model.layers.len(),
            &prepared.token_counts,
        )?;
        Ok(CausalLmOutput {
            logits,
            final_hidden_state: options.hidden_states.then_some(hidden),
            hidden_states,
            attentions: Vec::new(),
            routers,
            load_balancing_loss,
        })
    }
}

impl<B: Backend> Qwen3MoeModel<B> {
    /// Runs one compact semantic row and retains cache writes in scratch storage.
    #[allow(clippy::too_many_arguments)]
    fn forward_compact_row(
        &self,
        token_ids: Tensor<B, 2, Int>,
        positions: Tensor<B, 2, Int>,
        row: usize,
        baseline: usize,
        evaluated_context: usize,
        bank: &crate::cache::TensorCacheBank<B>,
        pending: &mut PendingKvWrites<B>,
        retain_hidden_states: bool,
        retain_routers: bool,
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
        let rotary = rotary_angles(self.rope, positions, evaluated_context)?;
        for (layer_index, layer) in self.layers.iter().enumerate() {
            let (next, routed) = layer.forward_cached(
                hidden,
                &rotary,
                row,
                baseline,
                bank,
                layer_index,
                self.config.layer_sliding_window(layer_index),
                pending,
            )?;
            hidden = next;
            if retain_hidden_states {
                hidden_states.push(hidden.clone());
            }
            if retain_routers && let Some(routed) = routed {
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

/// Result of evaluating one compact semantic batch row.
struct CompactRowOutput<B: Backend> {
    /// Final-normalized compact states shaped `[1, valid_tokens, hidden]`.
    hidden: Tensor<B, 3>,
    /// Optional compact embedding, post-block, and normalized snapshots.
    hidden_states: Vec<Tensor<B, 3>>,
    /// Compact sparse-layer results awaiting rectangular diagnostic assembly.
    routers: Vec<SparseMoeOutput<B>>,
}

impl<B: Backend> Qwen3MoeDecoderLayer<B> {
    /// Evaluates one cached GQA/MoE block without publishing its new keys.
    #[allow(clippy::too_many_arguments)]
    fn forward_cached(
        &self,
        hidden: Tensor<B, 3>,
        rotary: &RotaryAngles<B>,
        row: usize,
        baseline: usize,
        bank: &crate::cache::TensorCacheBank<B>,
        layer_index: usize,
        sliding_window: Option<usize>,
        pending: &mut PendingKvWrites<B>,
    ) -> Result<(Tensor<B, 3>, Option<SparseMoeOutput<B>>)> {
        let residual = hidden.clone();
        let normalized = self.input_layernorm.forward(hidden);
        let mut projection = self.self_attn.project_self(normalized)?;
        (projection.query, projection.key) =
            apply_rotary(projection.query, projection.key, rotary)?;
        let keys = projection.key.clone();
        let values = projection.value.clone();
        let query_sequence = projection.query.dims()[2];
        let cache = bank.native_kv(NATIVE_KV_GROUP)?;
        let history = cache.read_sequence(layer_index, row, baseline)?;
        if history.retained_positions() != (0..baseline) {
            return Err(Error::Cache(
                "Qwen3 MoE full cache unexpectedly evicted history".into(),
            ));
        }
        if let Some((past_keys, past_values)) = history.materialize() {
            projection.key = Tensor::cat(vec![past_keys, projection.key], 2);
            projection.value = Tensor::cat(vec![past_values, projection.value], 2);
        }
        let mask = cached_attention_mask::<B>(
            baseline,
            query_sequence,
            self.self_attn.query_heads,
            sliding_window,
            &projection.query.device(),
        )?;
        let mixed = self.self_attn.attend(projection, Some(mask), None, false)?;
        let hidden = residual + mixed;
        let residual = hidden.clone();
        let normalized = self.post_attention_layernorm.forward(hidden);
        let (hidden, routed) = match (&self.sparse_mlp, &self.dense_mlp) {
            (Some(moe), None) => {
                let output = moe.forward(normalized)?;
                (residual + output.hidden_state.clone(), Some(output))
            }
            (None, Some(mlp)) => (residual + mlp.forward(normalized)?, None),
            _ => {
                return Err(Error::Checkpoint(
                    "Qwen3 MoE layer has an invalid feed-forward topology".into(),
                ));
            }
        };
        pending.writes.push(PendingKvWrite {
            layer: layer_index,
            row,
            start: baseline,
            keys,
            values,
        });
        Ok((hidden, routed))
    }
}

/// One deferred native-KV append.
struct PendingKvWrite<B: Backend> {
    /// Decoder-layer index within the homogeneous native-KV cache group.
    layer: usize,
    /// Independent batch row receiving this append.
    row: usize,
    /// Absolute logical cursor at which this suffix begins.
    start: usize,
    /// New rotated keys shaped `[1, kv_heads, suffix, head_dim]`.
    keys: Tensor<B, 4>,
    /// New values with the same native-head axes as the keys.
    values: Tensor<B, 4>,
}

/// Complete operation-local write set used for atomic publication.
struct PendingKvWrites<B: Backend> {
    /// Unpublished per-layer, per-row appends for the complete operation.
    writes: Vec<PendingKvWrite<B>>,
}

impl<B: Backend> PendingKvWrites<B> {
    /// Creates an empty scratch write set.
    fn new() -> Self {
        Self { writes: Vec::new() }
    }

    /// Validates exact layer/row coverage, then atomically publishes the call.
    fn commit(
        self,
        state: &mut TensorDecodeState<B>,
        layers: usize,
        token_counts: &[usize],
    ) -> Result<()> {
        self.validate_coverage(state, layers, token_counts)?;
        let cache = state.main().native_kv(NATIVE_KV_GROUP)?;
        let writes = self
            .writes
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
            .collect::<core::result::Result<Vec<_>, _>>()?;
        let prepared = state
            .main()
            .prepare_direct_writes(Some((NATIVE_KV_GROUP, writes)), None)?;
        state.advance(token_counts)?;
        state.main_mut().apply_prepared_direct_writes(prepared);
        Ok(())
    }

    /// Proves that every active row has exactly one append for every layer.
    fn validate_coverage(
        &self,
        state: &TensorDecodeState<B>,
        layers: usize,
        token_counts: &[usize],
    ) -> Result<()> {
        if token_counts.len() != state.batch_size() {
            return Err(Error::Cache(format!(
                "Qwen3 MoE pending counts describe {} rows for batch {}",
                token_counts.len(),
                state.batch_size()
            )));
        }
        let active = token_counts.iter().filter(|&&count| count > 0).count();
        let expected = layers.checked_mul(active).ok_or_else(|| {
            Error::configuration("qwen3_moe.cache_commit", "write count overflows usize")
        })?;
        if self.writes.len() != expected {
            return Err(Error::Cache(format!(
                "Qwen3 MoE pending cache coverage is {}/{expected}",
                self.writes.len()
            )));
        }
        let mut seen = BTreeSet::new();
        for write in &self.writes {
            let count = token_counts.get(write.row).copied().ok_or_else(|| {
                Error::Cache(format!(
                    "Qwen3 MoE cache write row {} is invalid",
                    write.row
                ))
            })?;
            if count == 0
                || write.layer >= layers
                || write.start != state.lengths()[write.row]
                || write.keys.dims()[2] != count
                || write.values.dims()[2] != count
                || !seen.insert((write.layer, write.row))
            {
                return Err(Error::Cache(format!(
                    "invalid or duplicate Qwen3 MoE write for layer {}, row {}",
                    write.layer, write.row
                )));
            }
        }
        Ok(())
    }
}

/// Rectangular router tensors reconstructed from one compact row.
struct RowRouter<B: Backend> {
    /// Router logits restored to `[1, rectangular_sequence, experts]`.
    logits: Tensor<B, 3>,
}

/// Validates devices, shapes, semantic rows, positions, and capacity transitions.
///
/// The shared preparation reads the mask, token IDs, and explicit positions in
/// one transfer; this wrapper adds the model-device and dynamic-NTK guards.
fn prepare_stateful_input<B: Backend>(
    model: &Qwen3MoeForCausalLm<B>,
    input: &CausalLmInput<B>,
    state: &Qwen3MoeDecodeState<B>,
    phase: StatefulPhase,
) -> Result<StatefulRows> {
    ensure_same_device(
        "qwen3_moe.model",
        &model.model.embed_tokens.weight.lazy_device(),
        state.tensors.device(),
        DECODE_STATE,
    )?;
    let rows = prepare_stateful_rows(
        input,
        state.tensors.device(),
        state.tensors.lengths(),
        state.tensors.capacity(),
        &StatefulContract {
            names: INPUT_NAMES,
            owner: DECODE_STATE,
            vocabulary: model.config().vocab_size,
            position_limit: Some(model.config().max_position_embeddings),
            empty_rows: phase.empty_rows(),
        },
    )?;
    for (&context, &count) in rows.evaluated_contexts.iter().zip(&rows.token_counts) {
        if count > 0 {
            validate_cached_rope_context(model.model.rope.scaling(), context)?;
        }
    }
    Ok(rows)
}

/// Rejects dynamic-NTK transitions that would invalidate cached rotated keys.
fn validate_cached_rope_context(scaling: RopeScaling, context: usize) -> Result<()> {
    if let RopeScaling::DynamicNtk(dynamic) = scaling
        && context > dynamic.original_max_position_embeddings()
    {
        return Err(Error::BackendCapability(
            "dynamic-NTK decode beyond the original context requires re-rotating cached keys"
                .into(),
        ));
    }
    Ok(())
}

/// Builds `excluded(q,k) = k > q || (window && k + window <= q)`.
fn cached_attention_mask<B: Backend>(
    baseline: usize,
    query_sequence: usize,
    heads: usize,
    window: Option<usize>,
    device: &B::Device,
) -> Result<Tensor<B, 4, Bool>> {
    let key_sequence = baseline.checked_add(query_sequence).ok_or_else(|| {
        Error::configuration("qwen3_moe.attention_mask", "key length overflows usize")
    })?;
    let query_start = i64::try_from(baseline)
        .map_err(|_| Error::configuration("qwen3_moe.attention_mask", "offset exceeds i64"))?;
    let end = i64::try_from(key_sequence)
        .map_err(|_| Error::configuration("qwen3_moe.attention_mask", "length exceeds i64"))?;
    let query = Tensor::<B, 1, Int>::arange(query_start..end, device)
        .reshape([1, 1, query_sequence, 1])
        .expand([1, heads, query_sequence, key_sequence]);
    let key = Tensor::<B, 1, Int>::arange(0..end, device)
        .reshape([1, 1, 1, key_sequence])
        .expand([1, heads, query_sequence, key_sequence]);
    let causal = key.clone().greater(query.clone());
    Ok(match window {
        Some(window) => causal.bool_or((key + window as i64).lower_equal(query)),
        None => causal,
    })
}

/// Returns exact zero hidden states for an inactive decode row.
fn empty_hidden_row<B: Backend>(
    model: &Qwen3MoeForCausalLm<B>,
    sequence: usize,
    device: &B::Device,
) -> Tensor<B, 3> {
    Tensor::zeros(
        [1, sequence, model.config().hidden_size],
        (device, model.model.embed_tokens.weight.val().dtype()),
    )
}

/// Derives the homogeneous native-KV cache specification from model weights.
fn cache_spec<B: Backend>(model: &Qwen3MoeForCausalLm<B>) -> Result<NativeKvCacheSpec> {
    NativeKvCacheSpec::new(
        model.model.layers.len(),
        model.config().num_key_value_heads,
        model.config().head_dim,
        KvCacheStorage::Full,
        model.model.embed_tokens.weight.val().dtype(),
    )
    .map_err(Into::into)
}
