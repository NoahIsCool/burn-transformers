//! External native-KV state for Qwen3 MoE prefill and decoding, one batched pass per call.
//!
//! All attention layers share one preallocated cache group laid out as
//! `[batch, kv_heads, capacity, head_dim]` per key/value tensor. Every call
//! arranges the batch in the compact right-padded layout of
//! [`CompactBatch`] (row `b`'s valid tokens first; slot `i` is cache
//! position `L_b + i`) and runs each layer once over all rows: attention
//! writes the new rotated keys and values into their slots ahead of time and
//! attends over one view of `[0, L_b + n_b)`, with sliding-window layers
//! excluding keys older than their window, and the routed experts dispatch
//! all rows' tokens together (one routing readback per sparse layer, not one
//! per row). Every layer's write is recorded in one step and published only
//! after the vocabulary projection, so a returned `Err` leaves every length
//! and readable cache position unchanged.
//!
//! Rectangular masks are semantic: invalid columns neither consume RoPE
//! positions nor enter the cache, and their outputs and router logits are
//! zero. Decode permits a fully masked row, defining it as an exact
//! zero-output/no-state-change operation so finished sequences can remain in
//! a heterogeneous batch. Prefill rejects empty prompt rows.

use core::num::NonZeroUsize;

use burn::tensor::{Int, Tensor, backend::Backend};

use crate::{
    CausalLanguageModel, CausalLmInput, CausalLmOutput, DecodeInput, Error, ForwardOptions, Result,
    architecture::DecodeState as ArchitectureDecodeState,
    cache::{
        CacheConfig, KvCacheStorage, NativeKvCacheSpec, TensorCacheError, TensorDecodeState,
        TensorStepWrites,
    },
    intervention::Layer,
    models::{
        qwen3_5::model::batch::{CachedAttention, CompactBatch, ensure_cacheable_rotary_context},
        shared::{
            EmptyRows, StatefulContract, StatefulRows, ensure_same_device, prepare_stateful_rows,
            select_hidden_with_last_columns, validate_logits_selection,
        },
    },
};

use super::{INPUT_NAMES, Qwen3MoeBackboneOutput, Qwen3MoeForCausalLm, Qwen3MoeModel};

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
    ///
    /// If row `b` has committed length `L_b`, its `i`th valid implicit
    /// position is `L_b + i` and its queries see keys `j <= L_b + i` (and, on
    /// a sliding-window layer, `j > L_b + i - window`).
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
        let rows = prepare_stateful_input(self, &input, state, phase)?;
        validate_logits_selection(&options.logits, rows.sequence)?;
        let retain_router_logits = self.retains_router_logits(options);
        let device = state.tensors.device().clone();
        if rows.token_counts.iter().all(|&count| count == 0) {
            // Every row is finished: an exact no-op with zero outputs.
            let zeros = |width: usize| {
                Tensor::<B, 3>::zeros(
                    [rows.batch, rows.sequence, width],
                    (&device, self.model.embed_tokens.weight.val().dtype()),
                )
            };
            let sparse_layers = self
                .model
                .layers
                .iter()
                .filter(|layer| layer.sparse_mlp.is_some())
                .count();
            let backbone = Qwen3MoeBackboneOutput {
                hidden: zeros(self.config().hidden_size),
                hidden_states: if options.hidden_states {
                    vec![zeros(self.config().hidden_size); self.model.layers.len() + 2]
                } else {
                    Vec::new()
                },
                router_logits: if retain_router_logits {
                    vec![zeros(self.config().num_experts); sparse_layers]
                } else {
                    Vec::new()
                },
            };
            return self.finish_cached(&input, &rows, options, backbone);
        }
        let layout = CompactBatch::new(
            &rows.valid_indices,
            rows.sequence,
            state.tensors.lengths(),
            &device,
        )?;
        // Padding slots are evaluated (and discarded), so they carry token 0
        // and position 0 instead of the caller's arbitrary sentinels.
        let tokens = layout.compact(&input.token_ids, 0);
        let positions = match &input.position_ids {
            Some(positions) => layout.compact(positions, 0),
            None => layout.cache_positions().clone(),
        };
        let context = rows
            .active_contexts()
            .map(|(_, context)| context)
            .max()
            .unwrap_or(1);
        let (compact, writes) = self.model.run_cached(
            tokens,
            positions,
            context,
            &layout,
            &mut state.tensors,
            options.hidden_states,
            retain_router_logits,
        )?;
        let backbone = Qwen3MoeBackboneOutput {
            hidden: compact.hidden,
            hidden_states: compact
                .hidden_states
                .into_iter()
                .map(|hidden| layout.restore(hidden))
                .collect(),
            router_logits: compact
                .router_logits
                .into_iter()
                .map(|logits| layout.restore(logits))
                .collect(),
        };
        // `Last` reads the compact block directly; other selections restore.
        let (selected, hidden) = match &options.logits {
            crate::LogitsSelection::Last => (
                layout.last(backbone.hidden.clone()),
                options
                    .hidden_states
                    .then(|| layout.restore(backbone.hidden.clone())),
            ),
            selection => {
                let hidden = layout.restore(backbone.hidden.clone());
                (
                    select_hidden_with_last_columns(
                        hidden.clone(),
                        rows.last_columns(),
                        selection,
                    )?,
                    options.hidden_states.then_some(hidden),
                )
            }
        };
        let output = self.finish_selected(&input, options, selected, hidden, backbone)?;
        state.tensors.publish_step(writes)?;
        Ok(output)
    }

    /// Finishes an all-inactive call from rectangular zero states.
    fn finish_cached(
        &self,
        input: &CausalLmInput<B>,
        rows: &StatefulRows,
        options: &ForwardOptions,
        backbone: Qwen3MoeBackboneOutput<B>,
    ) -> Result<CausalLmOutput<B>> {
        let selected = select_hidden_with_last_columns(
            backbone.hidden.clone(),
            rows.last_columns(),
            &options.logits,
        )?;
        let hidden = options.hidden_states.then(|| backbone.hidden.clone());
        self.finish_selected(input, options, selected, hidden, backbone)
    }

    /// Projects selected states and assembles router diagnostics and the auxiliary loss.
    ///
    /// Router logits are rectangular with zeros at padding; the stack-wide
    /// loss masks padding with the caller's token mask.
    fn finish_selected(
        &self,
        input: &CausalLmInput<B>,
        options: &ForwardOptions,
        selected: Tensor<B, 3>,
        final_hidden_state: Option<Tensor<B, 3>>,
        backbone: Qwen3MoeBackboneOutput<B>,
    ) -> Result<CausalLmOutput<B>> {
        let load_balancing_loss = options
            .auxiliary_losses
            .then(|| {
                super::qwen3_moe_load_balancing_loss(
                    &backbone.router_logits,
                    self.config().num_experts_per_tok,
                    input.token_mask.as_ref(),
                )
            })
            .transpose()?;
        let routers = if options.router_logits || self.config().output_router_logits {
            backbone
                .router_logits
                .into_iter()
                .map(super::router_diagnostics)
                .collect()
        } else {
            Vec::new()
        };
        Ok(CausalLmOutput {
            logits: self.project_logits(selected)?,
            final_hidden_state,
            hidden_states: backbone.hidden_states,
            attentions: Vec::new(),
            routers,
            load_balancing_loss,
        })
    }
}

impl<B: Backend> Qwen3MoeModel<B> {
    /// Runs every layer once over the compact batch and records its cache writes.
    ///
    /// Returns compact `[B, N, ·]` states and router logits with the
    /// unpublished step.
    #[allow(clippy::too_many_arguments)]
    fn run_cached(
        &self,
        tokens: Tensor<B, 2, Int>,
        positions: Tensor<B, 2, Int>,
        context: usize,
        layout: &CompactBatch<B>,
        tensors: &mut TensorDecodeState<B>,
        retain_hidden_states: bool,
        retain_router_logits: bool,
    ) -> Result<(Qwen3MoeBackboneOutput<B>, TensorStepWrites<B>)> {
        let mut writes = tensors.step(layout.counts())?;
        let mut hidden = self.embed_tokens.forward(tokens);
        let mut attention = CachedAttention::new(
            self.rope,
            context,
            positions,
            hidden.dtype(),
            tensors,
            &writes,
            NATIVE_KV_GROUP,
        )?;
        let mut hidden_states = Vec::with_capacity(if retain_hidden_states {
            self.layers.len() + 2
        } else {
            0
        });
        if retain_hidden_states {
            hidden_states.push(hidden.clone());
        }
        let mut router_logits = Vec::new();
        for (index, layer) in self.layers.iter().enumerate() {
            let window = self
                .config
                .layer_sliding_window(index)
                .map(|window| {
                    NonZeroUsize::new(window).ok_or_else(|| {
                        Error::configuration("qwen3_moe.sliding_window", "must be non-zero")
                    })
                })
                .transpose()?;
            let residual = hidden.clone();
            let normalized = layer.input_layernorm.forward(hidden);
            let mixed = attention.attend(
                &layer.self_attn,
                normalized,
                index,
                window,
                layout,
                tensors,
                &mut writes,
            )?;
            let hidden_after_attention = residual + mixed;
            let normalized = layer
                .post_attention_layernorm
                .forward(hidden_after_attention.clone());
            let (mlp, logits) = layer.feed_forward(normalized, Layer::decoder(index), None)?;
            hidden = hidden_after_attention + mlp;
            if retain_hidden_states {
                hidden_states.push(hidden.clone());
            }
            if retain_router_logits && let Some(logits) = logits {
                router_logits.push(logits);
            }
        }
        hidden = self.norm.forward(hidden);
        if retain_hidden_states {
            hidden_states.push(hidden.clone());
        }
        Ok((
            Qwen3MoeBackboneOutput {
                hidden,
                hidden_states,
                router_logits,
            },
            writes,
        ))
    }
}

/// Validates devices, shapes, semantic rows, positions, and capacity transitions.
///
/// The shared preparation reads the mask, token IDs, and explicit positions in
/// one transfer; this wrapper adds the model-device and context-dependent RoPE
/// guards.
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
    for (_, context) in rows.active_contexts() {
        ensure_cacheable_rotary_context(model.model.rope, context, "Qwen3 MoE")?;
    }
    Ok(rows)
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

#[cfg(test)]
#[path = "decode_tests.rs"]
mod tests;
