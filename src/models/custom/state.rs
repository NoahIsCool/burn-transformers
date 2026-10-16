//! Preallocated reference state for modular autoregressive execution.
//!
//! A heterogeneous custom schedule may mix cache representations for which no single optimized
//! state layout exists yet. This state therefore stores compact token and absolute-position
//! histories in fixed-capacity device tensors, then recomputes the stateless graph. It never
//! concatenates growing tensors, and it exercises exactly the same model function used for
//! training. The path is intentionally a correctness oracle, not an advertised low-latency KV
//! cache; concrete family runtimes provide optimized cache layouts.
//!
//! # One batched pass per step
//!
//! Every step is executed for the whole batch at once, whatever the rows' raggedness:
//!
//! 1. one device read validates the step's mask, token IDs, and explicit positions;
//! 2. the host plans the append ([`plan_history`], verified by Verus): row `r` appends its
//!    `n_r` valid tokens at slots `L_r..L_r + n_r` in input order;
//! 3. one rectangular window `[min L_r, max(L_r + n_r))` of the token and position histories is
//!    rewritten with a gather and a masked select, from the same verified plan, so the number of
//!    kernels is independent of the batch size and of each row's padding layout;
//! 4. the model evaluates the shortest rectangular history prefix in one stateless pass, with
//!    each row's evaluated length as its validity mask;
//! 5. one gather returns the appended slots in the caller's `[batch, step]` layout.
//!
//! Nothing is committed until the outputs exist, so a failing step leaves the state unchanged.
//!
//! # Inactive rows
//!
//! The state follows the stateful contract every family shares: prefill rejects a row without
//! a valid token, while decode accepts it as a no-op whose step outputs are exact zeros. A
//! row that appends nothing is evaluated over only its first history slot: its outputs are
//! masked to zero anyway, the stateless pass needs at least one valid key per row (a row that
//! has never started, such as a left-padded prompt's first chunk, has none), and a finished
//! row's long history then does not widen the evaluated prefix. That slot holds zeros after
//! allocation or reset and otherwise a token and position validated when they were written.
//! It is never committed: the row keeps its committed length, possibly zero. A step in which
//! no row is active therefore evaluates one slot per row and commits nothing.

use burn::tensor::{Element, Int, Tensor, TensorData, backend::Backend};

use crate::{
    CausalLmInput, CausalLmOutput, DecodeState as ArchitectureDecodeState, Error, ForwardOptions,
    Result,
    architecture::{DecoderSpec, EncoderOutput, RouterDiagnostics, Seq2SeqOutput, TokenMask},
    cache::{
        CacheConfig, SequenceLengths,
        tensor::{validate_repeated_allocation, validate_tensor_shape},
    },
    moe::SparseMoeOutput,
};

use crate::models::shared::{HostInput, plan, range_error};

use super::{
    ModularCausalLm, ModularSeq2SeqLm,
    common::{
        BackboneOutput, causal_output, gathered_router_diagnostics, mask_hidden, select_hidden,
        token_mask_from_values, validate_forward_options,
    },
    plan::{
        HistoryAppendError, first_position_violation, full_range, plan_history, prefix_validity,
    },
};

/// Model role encoded into state so equal-looking causal and seq2seq specs cannot be mixed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum StatePurpose {
    /// Decoder-only language-model state.
    Causal,
    /// Target-side state belonging to an encoder-decoder model.
    Seq2Seq,
}

/// External fixed-capacity history for modular reference decoding.
///
/// Tokens and positions are compacted by validity within each row. Logical lengths remain on the
/// host for checked lifecycle operations, while the actual histories stay as device tensors.
#[derive(Debug)]
pub struct ModularDecodeState<B: Backend> {
    /// Checked per-row committed lengths and immutable batch/capacity geometry.
    metadata: SequenceLengths,
    /// Compacted token IDs in preallocated `[batch, capacity]` device storage.
    tokens: Tensor<B, 2, Int>,
    /// Position IDs aligned with compacted tokens in `[batch, capacity]` device storage.
    positions: Tensor<B, 2, Int>,
    /// Checked total element count of the fixed token and position history tensors.
    allocated_integer_elements: usize,
    /// Decoder topology that must match the model consuming this state.
    model_spec: DecoderSpec,
    /// Causal or encoder-decoder role used to reject otherwise identical foreign state.
    purpose: StatePurpose,
    /// Device shared by the state histories and every accepted input tensor.
    device: B::Device,
}

impl<B: Backend> ModularDecodeState<B> {
    /// Allocates zeroed device history after checking batch and tensor geometry.
    ///
    /// Histories use the backend's integer dtype and the cache-wide portable
    /// `i32::MAX` element bound. Shape and aggregate byte overflow fail before
    /// host metadata or device tensors are allocated; available memory remains
    /// the backend's responsibility.
    pub(super) fn new(
        config: CacheConfig,
        batch_size: usize,
        model_spec: DecoderSpec,
        purpose: StatePurpose,
        device: &B::Device,
    ) -> Result<Self> {
        if config.sequence_count() != batch_size {
            return Err(Error::Cache(format!(
                "custom decode state expected {} sequences, received batch size {batch_size}",
                config.sequence_count()
            )));
        }
        let elements_per_history = validate_tensor_shape(
            "custom.decode.history",
            &[batch_size, config.capacity()],
            B::IntElem::dtype(),
        )
        .map_err(|error| Error::Cache(error.to_string()))?;
        let allocated_integer_elements = validate_repeated_allocation(
            "custom.decode.history",
            elements_per_history,
            2,
            B::IntElem::dtype(),
        )
        .map_err(|error| Error::Cache(error.to_string()))?;
        Ok(Self {
            metadata: SequenceLengths::new(config),
            tokens: Tensor::zeros([batch_size, config.capacity()], device),
            positions: Tensor::zeros([batch_size, config.capacity()], device),
            allocated_integer_elements,
            model_spec,
            purpose,
            device: device.clone(),
        })
    }

    /// Returns the number of integer elements preallocated for token and position histories.
    #[must_use]
    pub fn allocated_integer_elements(&self) -> usize {
        self.allocated_integer_elements
    }

    /// Executes an exact causal reference append and commits only after successful model math.
    pub(super) fn run_causal(
        &mut self,
        model: &ModularCausalLm<B>,
        input: CausalLmInput<B>,
        options: &ForwardOptions,
        require_each_row: bool,
    ) -> Result<CausalLmOutput<B>> {
        self.validate_model(model.spec(), StatePurpose::Causal)?;
        validate_forward_options(options)?;
        let append = self.prepare_append(input, model.spec().vocabulary_size, require_each_row)?;
        let history = append.history_input(&self.device)?;
        let backbone = model.forward_backbone(&history, options.hidden_states)?;
        let output = append.causal_output(model, backbone, options, &self.device)?;
        self.commit(append)?;
        Ok(output)
    }

    /// Executes an exact seq2seq target append and commits after the output is assembled.
    pub(super) fn run_seq2seq(
        &mut self,
        model: &ModularSeq2SeqLm<B>,
        encoder: &EncoderOutput<B>,
        input: CausalLmInput<B>,
        options: &ForwardOptions,
        require_each_row: bool,
    ) -> Result<Seq2SeqOutput<B>> {
        self.validate_model(&model.spec().decoder, StatePurpose::Seq2Seq)?;
        validate_forward_options(options)?;
        let append = self.prepare_append(
            input,
            model.spec().decoder.vocabulary_size,
            require_each_row,
        )?;
        let history = append.history_input(&self.device)?;
        let backbone = model.forward_decoder_backbone(encoder, &history, options.hidden_states)?;
        let decoder = append.seq2seq_decoder_output(model, backbone, options, &self.device)?;
        self.commit(append)?;
        Ok(Seq2SeqOutput {
            decoder,
            encoder: encoder.clone(),
            cross_attentions: Vec::new(),
        })
    }

    /// Ensures state was allocated by the exact model role and decoder topology.
    fn validate_model(&self, spec: &DecoderSpec, purpose: StatePurpose) -> Result<()> {
        if self.purpose != purpose || &self.model_spec != spec {
            return Err(Error::Cache(
                "custom decode state belongs to a different model topology or role".into(),
            ));
        }
        Ok(())
    }

    /// Builds a prospective compact history without mutating committed state.
    ///
    /// Shapes and devices are checked first; the mask, token IDs, and explicit
    /// positions are then read back once and validated on the host. Every token
    /// ID must lie in the vocabulary (padded IDs included), while explicit
    /// positions are checked over the valid columns that enter the history.
    /// The whole batch is then written with one gather/select per history.
    fn prepare_append(
        &self,
        input: CausalLmInput<B>,
        vocabulary_size: usize,
        require_each_row: bool,
    ) -> Result<PreparedAppend<B>> {
        let [batch, step] = input.token_ids.dims();
        if batch != self.batch_size() || step == 0 {
            return Err(Error::shape(
                "custom.decode.token_ids",
                format!("[{}, step>0]", self.batch_size()),
                format!("[{batch}, {step}]"),
            ));
        }
        self.validate_append_layout(&input, batch, step)?;
        let host = HostInput::read(&input, batch, step, "custom decode input")?;
        validate_decode_token_ids(&host.tokens, vocabulary_size)?;
        let validity = host.mask.unwrap_or_else(|| vec![true; batch * step]);
        let committed = self.metadata.lengths();
        // Prefill requires every row to append, so only decode can shorten rows.
        let planned_from = if require_each_row {
            committed.to_vec()
        } else {
            inactive_rows_shortened(committed, &validity, step)
        };
        // The verified planner checks one validity cell per row and column itself.
        let planned = match plan_history(
            &validity,
            step,
            &planned_from,
            self.capacity(),
            require_each_row,
        ) {
            Ok(planned) => Some(planned),
            // Decode accepts a step without any valid token as a no-op.
            Err(HistoryAppendError::EmptyStep) if !require_each_row => None,
            Err(error) => return Err(self.append_error(error)),
        };
        if let Some(positions) = &host.positions {
            validate_explicit_positions(
                positions,
                &validity,
                batch,
                step,
                self.model_spec.max_position_embeddings,
            )?;
        }
        let Some((plan, writes)) = planned else {
            // Nothing is written; every row is evaluated over its first slot.
            return Ok(PreparedAppend {
                tokens: self.tokens.clone(),
                positions: self.positions.clone(),
                absolute_indices: vec![0; batch * step],
                evaluated_lengths: planned_from,
                counts: vec![0; batch],
                input_validity: validity,
                step,
            });
        };
        let window = writes.start..writes.end;
        let width = window.len();
        let sources = Tensor::<B, 2, Int>::from_data(
            TensorData::new(to_indices(&writes.sources)?, [batch, width]),
            &self.device,
        );
        let written = crate::architecture::bool_tensor_from_values::<B, 2>(
            writes.written,
            [batch, width],
            &self.device,
        );
        let write_window = |history: &Tensor<B, 2, Int>, appended: Tensor<B, 2, Int>| {
            let current = history.clone().slice([0..batch, window.clone()]);
            history.clone().slice_assign(
                [0..batch, window.clone()],
                current.mask_where(written.clone(), appended),
            )
        };
        let tokens = write_window(&self.tokens, input.token_ids.gather(1, sources.clone()));
        let appended_positions = match input.position_ids {
            Some(supplied) => supplied.gather(1, sources),
            // Implicit positions are the slot indices themselves.
            None => Tensor::<B, 1, Int>::arange(
                to_index(window.start)?..to_index(window.end)?,
                &self.device,
            )
            .reshape([1, width])
            .expand([batch, width]),
        };
        let positions = write_window(&self.positions, appended_positions);
        Ok(PreparedAppend {
            tokens,
            positions,
            absolute_indices: to_indices(&plan.slots)?,
            evaluated_lengths: plan.lengths,
            counts: plan.counts,
            input_validity: validity,
            step,
        })
    }

    /// Converts a planning failure into this state's contextual error.
    fn append_error(&self, error: HistoryAppendError) -> Error {
        match error {
            HistoryAppendError::Layout { cells, rows, step } => Error::Cache(format!(
                "custom decode state holds {rows} rows, but the step mask has {cells} cells \
                 for {step} columns"
            )),
            HistoryAppendError::EmptyRow { .. } => Error::shape(
                "custom.prefill.token_mask",
                "at least one valid token in every prompt row",
                "a fully masked row",
            ),
            HistoryAppendError::Capacity {
                row,
                requested: None,
            } => Error::Cache(format!("custom sequence {row} length overflows usize")),
            HistoryAppendError::Capacity {
                row,
                requested: Some(requested),
            } => Error::Cache(format!(
                "custom sequence {row} requested length {requested} exceeds capacity {}",
                self.capacity()
            )),
            // Prefill rejects an empty row first, and decode answers an empty step itself and
            // plans every non-advancing row from one slot, so neither failure can reach here.
            HistoryAppendError::EmptyStep | HistoryAppendError::NeverStarted { .. } => {
                Error::Cache(
                    "custom history planner rejected a step the decode contract admits".into(),
                )
            }
        }
    }

    /// Checks mask and position extents and that every input shares the state device.
    fn validate_append_layout(
        &self,
        input: &CausalLmInput<B>,
        batch: usize,
        step: usize,
    ) -> Result<()> {
        if input.token_ids.device() != self.device {
            return Err(Error::Cache(
                "custom decode input and state must reside on the same device".into(),
            ));
        }
        if let Some(mask) = &input.token_mask {
            if mask.as_tensor().dims() != [batch, step] {
                return Err(Error::shape(
                    "custom.decode.token_mask",
                    format!("[{batch}, {step}]"),
                    format!("{:?}", mask.as_tensor().dims()),
                ));
            }
            if mask.as_tensor().device() != self.device {
                return Err(Error::Cache(
                    "custom decode mask and state must reside on the same device".into(),
                ));
            }
        }
        if let Some(positions) = &input.position_ids {
            if positions.dims() != [batch, step] {
                return Err(Error::shape(
                    "custom.decode.position_ids",
                    format!("[{batch}, {step}]"),
                    format!("{:?}", positions.dims()),
                ));
            }
            if positions.device() != self.device {
                return Err(Error::Cache(
                    "custom decode positions and state must reside on the same device".into(),
                ));
            }
        }
        Ok(())
    }

    /// Publishes a fully evaluated append and advances checked logical metadata.
    ///
    /// Rows that appended nothing keep their committed length, even when they were evaluated
    /// over a placeholder slot.
    fn commit(&mut self, append: PreparedAppend<B>) -> Result<()> {
        self.metadata
            .advance_all(&append.counts)
            .map_err(|error| Error::Cache(error.to_string()))?;
        // An advancing row ends exactly at the prefix length it was evaluated over.
        debug_assert!(
            self.metadata
                .lengths()
                .iter()
                .zip(&append.counts)
                .zip(&append.evaluated_lengths)
                .all(|((&length, &count), &evaluated)| count == 0 || length == evaluated)
        );
        self.tokens = append.tokens;
        self.positions = append.positions;
        Ok(())
    }
}

impl<B: Backend> ArchitectureDecodeState<B> for ModularDecodeState<B> {
    /// Returns the number of logical sequence rows represented by this state.
    fn batch_size(&self) -> usize {
        self.metadata.config().sequence_count()
    }

    /// Returns the maximum logical token count per row, excluding padding.
    fn capacity(&self) -> usize {
        self.metadata.config().capacity()
    }

    /// Borrows committed semantic token counts in the current batch order.
    fn lengths(&self) -> &[usize] {
        self.metadata.lengths()
    }

    /// Clears committed lengths and replaces token/position history with device-side zeros.
    fn reset(&mut self) -> Result<()> {
        self.metadata
            .reset()
            .map_err(|error| Error::Cache(error.to_string()))?;
        self.tokens = Tensor::zeros([self.batch_size(), self.capacity()], &self.device);
        self.positions = Tensor::zeros([self.batch_size(), self.capacity()], &self.device);
        Ok(())
    }

    /// Selects history rows and logical metadata together for beam expansion or reordering.
    ///
    /// Indices are validated before selection; repeated indices duplicate the
    /// same history without changing token or position semantics.
    fn reorder(&mut self, indices: &[usize]) -> Result<()> {
        let selection = plan::signed_indices(indices)
            .ok_or_else(|| Error::Cache("custom reorder index does not fit i64".into()))?;
        self.metadata
            .reorder(indices)
            .map_err(|error| Error::Cache(error.to_string()))?;
        let selection = Tensor::<B, 1, Int>::from_data(
            TensorData::new(selection, [indices.len()]),
            &self.device,
        );
        self.tokens = self.tokens.clone().select(0, selection.clone());
        self.positions = self.positions.clone().select(0, selection);
        Ok(())
    }

    /// Shortens logical histories after validating every requested prefix length.
    ///
    /// Stored tail bytes remain allocated but are invisible to subsequent
    /// prefix evaluation and may be overwritten by the next append.
    fn truncate(&mut self, lengths: &[usize]) -> Result<()> {
        self.metadata
            .truncate_all(lengths)
            .map_err(|error| Error::Cache(error.to_string()))?;
        Ok(())
    }

    /// Returns the device holding this state, independent of model parameter ownership.
    fn device(&self) -> &B::Device {
        &self.device
    }
}

/// Prospective device history plus indexing metadata for one append operation.
struct PreparedAppend<B: Backend> {
    /// Prospective `[batch, capacity]` token history, published only after successful evaluation.
    tokens: Tensor<B, 2, Int>,
    /// Prospective position history aligned with the compacted token storage.
    positions: Tensor<B, 2, Int>,
    /// Per-row history prefix evaluated by the stateless pass.
    ///
    /// An advancing row's prefix is its committed length after this append; a row that
    /// appends nothing is evaluated over one slot (see the module documentation).
    evaluated_lengths: Vec<usize>,
    /// Number of valid newly supplied tokens to commit in each row.
    counts: Vec<usize>,
    /// Row-major `[batch, step]` mask restored on newly gathered outputs.
    input_validity: Vec<bool>,
    /// Row-major compact-history indices; masked columns select index zero before output masking.
    absolute_indices: Vec<i64>,
    /// Supplied rectangular step width, including masked token columns.
    step: usize,
}

impl<B: Backend> PreparedAppend<B> {
    /// Materializes the shortest rectangular prefix containing every compact row.
    fn history_input(&self, device: &B::Device) -> Result<CausalLmInput<B>> {
        let batch = self.evaluated_lengths.len();
        // Verified: width `max L_r`, and cell `(r, c)` is valid exactly when `c < L_r`.
        let (sequence, validity) = prefix_validity(&self.evaluated_lengths)
            .ok_or_else(|| Error::Cache("custom prepared append lost its batch".into()))?;
        Ok(CausalLmInput {
            token_ids: self.tokens.clone().slice([0..batch, 0..sequence]),
            token_mask: Some(token_mask_from_values(validity, [batch, sequence], device)),
            position_ids: Some(self.positions.clone().slice([0..batch, 0..sequence])),
        })
    }

    /// Selects only the just-appended step and assembles causal diagnostics and logits.
    fn causal_output(
        &self,
        model: &ModularCausalLm<B>,
        backbone: BackboneOutput<B>,
        options: &ForwardOptions,
        device: &B::Device,
    ) -> Result<CausalLmOutput<B>> {
        let (hidden, mask) = self.gather_float(backbone.hidden, device);
        let hidden_states = backbone
            .hidden_states
            .into_iter()
            .map(|state| self.gather_float(state, device).0)
            .collect();
        let routers = self.gather_routers(backbone.routers, options, &mask, device)?;
        let selected = select_hidden(hidden.clone(), Some(&mask), &options.logits)?;
        Ok(causal_output(
            model.project(selected)?,
            hidden,
            hidden_states,
            routers,
            options,
        ))
    }

    /// Selects only the target append and applies the seq2seq vocabulary projection.
    fn seq2seq_decoder_output(
        &self,
        model: &ModularSeq2SeqLm<B>,
        backbone: BackboneOutput<B>,
        options: &ForwardOptions,
        device: &B::Device,
    ) -> Result<CausalLmOutput<B>> {
        let (hidden, mask) = self.gather_float(backbone.hidden, device);
        let hidden_states = backbone
            .hidden_states
            .into_iter()
            .map(|state| self.gather_float(state, device).0)
            .collect();
        let routers = self.gather_routers(backbone.routers, options, &mask, device)?;
        let selected = select_hidden(hidden.clone(), Some(&mask), &options.logits)?;
        Ok(causal_output(
            model.project_decoder(selected)?,
            hidden,
            hidden_states,
            routers,
            options,
        ))
    }

    /// Gathers compact absolute histories back into the caller's rectangular step shape.
    fn gather_float(
        &self,
        tensor: Tensor<B, 3>,
        device: &B::Device,
    ) -> (Tensor<B, 3>, TokenMask<B>) {
        let [batch, _, width] = tensor.dims();
        let indices = Tensor::<B, 2, Int>::from_data(
            TensorData::new(self.absolute_indices.clone(), [batch, self.step]),
            device,
        )
        .unsqueeze_dim::<3>(2)
        .expand([batch, self.step, width]);
        let mask = token_mask_from_values(self.input_validity.clone(), [batch, self.step], device);
        (
            mask_hidden(tensor.gather(1, indices), mask.as_tensor().clone()),
            mask,
        )
    }

    /// Gathers router logits and assignments so auxiliary losses describe only new tokens.
    fn gather_routers(
        &self,
        routers: Vec<SparseMoeOutput<B>>,
        options: &ForwardOptions,
        mask: &TokenMask<B>,
        device: &B::Device,
    ) -> Result<Vec<RouterDiagnostics<B>>> {
        if !options.router_logits && !options.auxiliary_losses {
            return Ok(Vec::new());
        }
        let batch = self.evaluated_lengths.len();
        let base = Tensor::<B, 2, Int>::from_data(
            TensorData::new(self.absolute_indices.clone(), [batch, self.step]),
            device,
        )
        .unsqueeze_dim::<3>(2);
        routers
            .into_iter()
            .map(|router| {
                let experts = router.router_logits.dims()[2];
                let selected = router.selected_experts.dims()[2];
                let logits = mask_hidden(
                    router
                        .router_logits
                        .gather(1, base.clone().expand([batch, self.step, experts])),
                    mask.as_tensor().clone(),
                );
                let assignments = router
                    .selected_experts
                    .gather(1, base.clone().expand([batch, self.step, selected]));
                gathered_router_diagnostics(logits, assignments, Some(mask), options)
            })
            .collect()
    }
}

/// Returns `committed` with every row that has no valid cell in this step shortened to one slot.
///
/// `validity` is the row-major `[committed.len(), step]` mask and `step > 0`. Such a row
/// appends nothing and its outputs are masked to zero, so planning it from one slot gives the
/// stateless pass a valid key even for a row that has never started, without evaluating a
/// finished row's whole history. The planner writes nothing for these rows, and commit keeps
/// their true lengths.
fn inactive_rows_shortened(committed: &[usize], validity: &[bool], step: usize) -> Vec<usize> {
    committed
        .iter()
        .zip(validity.chunks(step))
        .map(
            |(&length, cells)| {
                if cells.contains(&true) { length } else { 1 }
            },
        )
        .collect()
}

/// Rejects out-of-range decode IDs before they enter preallocated history.
///
/// Every supplied ID is checked, including padded columns, because the custom
/// decoder's contract treats the whole token tensor as caller data.
fn validate_decode_token_ids(tokens: &[i64], vocabulary: usize) -> Result<()> {
    let Some((minimum, maximum)) = full_range(tokens) else {
        return Ok(());
    };
    if plan::range_within(minimum, maximum, vocabulary) {
        Ok(())
    } else {
        Err(range_error(
            "custom.decode.token_ids",
            "token",
            vocabulary,
            minimum,
            maximum,
        ))
    }
}

/// Rejects explicit positions of valid cells outside the model context.
///
/// `positions` and `validity` are row-major `[batch, step]`. Positions
/// supplied at padded cells never enter the history and are not checked; the
/// error reports the first failing row's valid range. Implicit positions are
/// history slots below the capacity, which the model already bounds by its
/// position range, so they need no check.
fn validate_explicit_positions(
    positions: &[i64],
    validity: &[bool],
    batch: usize,
    step: usize,
    limit: usize,
) -> Result<()> {
    match first_position_violation(positions, validity, batch, step, limit) {
        Some((minimum, maximum)) => Err(range_error(
            "custom.decode.position_ids",
            "position",
            limit,
            minimum,
            maximum,
        )),
        None => Ok(()),
    }
}

/// Converts host indices into Burn's signed `i64` index type.
fn to_indices(indices: &[usize]) -> Result<Vec<i64>> {
    plan::signed_indices(indices).ok_or_else(|| {
        Error::configuration(
            "custom.decode",
            "history index exceeds the signed tensor-index range",
        )
    })
}

/// Converts a host index into Burn's signed `i64` index type.
fn to_index(index: usize) -> Result<i64> {
    i64::try_from(index).map_err(|_| {
        Error::configuration(
            "custom.decode",
            "history index exceeds the signed tensor-index range",
        )
    })
}

#[cfg(test)]
#[path = "state_tests.rs"]
/// History allocation, accounting, and batched-append regressions.
mod tests;
