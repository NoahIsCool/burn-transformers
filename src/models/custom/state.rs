//! Preallocated reference state for modular autoregressive execution.
//!
//! A heterogeneous custom schedule may mix cache representations for which no single optimized
//! state layout exists yet. This state therefore stores compact token and absolute-position
//! histories in fixed-capacity device tensors, then recomputes the stateless graph. It never
//! concatenates growing tensors, and it exercises exactly the same model function used for
//! training. The path is intentionally a correctness oracle, not an advertised low-latency KV
//! cache; concrete family runtimes provide optimized cache layouts.

use burn::tensor::{Int, Tensor, TensorData, backend::Backend};

use crate::{
    CausalLmInput, CausalLmOutput, DecodeState as ArchitectureDecodeState, Error, ForwardOptions,
    Result,
    architecture::{DecoderSpec, EncoderOutput, RouterDiagnostics, Seq2SeqOutput, TokenMask},
    cache::{CacheConfig, DecodeState as DecodeMetadata},
    moe::SparseMoeOutput,
};

use super::{
    ModularCausalLm, ModularSeq2SeqLm,
    common::{
        BackboneOutput, causal_output, gathered_router_diagnostics, mask_hidden, select_hidden,
        token_mask_from_values, validate_forward_options,
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
    metadata: DecodeMetadata,
    tokens: Tensor<B, 2, Int>,
    positions: Tensor<B, 2, Int>,
    model_spec: DecoderSpec,
    purpose: StatePurpose,
    device: B::Device,
}

impl<B: Backend> ModularDecodeState<B> {
    /// Allocates zeroed device history for one validated model topology.
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
        Ok(Self {
            metadata: DecodeMetadata::new(config),
            tokens: Tensor::zeros([batch_size, config.capacity()], device),
            positions: Tensor::zeros([batch_size, config.capacity()], device),
            model_spec,
            purpose,
            device: device.clone(),
        })
    }

    /// Returns the number of integer elements preallocated for token and position histories.
    #[must_use]
    pub fn allocated_integer_elements(&self) -> usize {
        self.batch_size() * self.capacity() * 2
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
        validate_decode_token_ids(input.token_ids.clone(), vocabulary_size)?;
        if input.token_ids.device() != self.device {
            return Err(Error::Cache(
                "custom decode input and state must reside on the same device".into(),
            ));
        }
        let validity = read_validity(input.token_mask.as_ref(), batch, step)?;
        let mut counts = vec![0usize; batch];
        for (row, cells) in validity.chunks_exact(step).enumerate() {
            counts[row] = cells.iter().filter(|&&valid| valid).count();
        }
        if require_each_row && counts.contains(&0) {
            return Err(Error::shape(
                "custom.prefill.token_mask",
                "at least one valid token in every prompt row",
                "a fully masked row",
            ));
        }
        if counts.iter().all(|&count| count == 0) {
            return Err(Error::shape(
                "custom.decode.token_mask",
                "at least one active token across the batch",
                "an entirely inactive decode step",
            ));
        }
        let old_lengths = self.metadata.lengths().to_vec();
        let mut new_lengths = Vec::with_capacity(batch);
        for (row, (&old, &count)) in old_lengths.iter().zip(&counts).enumerate() {
            let new = old.checked_add(count).ok_or_else(|| {
                Error::Cache(format!("custom sequence {row} length overflows usize"))
            })?;
            if new > self.capacity() {
                return Err(Error::Cache(format!(
                    "custom sequence {row} requested length {new} exceeds capacity {}",
                    self.capacity()
                )));
            }
            new_lengths.push(new);
        }
        if new_lengths.contains(&0) {
            return Err(Error::BackendCapability(
                "reference batched decoding cannot evaluate a never-started inactive row".into(),
            ));
        }

        let supplied_positions = match input.position_ids {
            Some(positions) => {
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
                Some(positions)
            }
            None => None,
        };
        let mut tokens = self.tokens.clone();
        let mut positions = self.positions.clone();
        let mut absolute_indices = vec![0i64; batch * step];
        for row in 0..batch {
            let valid_columns = validity[row * step..(row + 1) * step]
                .iter()
                .enumerate()
                .filter_map(|(column, &valid)| valid.then_some(column as i64))
                .collect::<Vec<_>>();
            if valid_columns.is_empty() {
                continue;
            }
            let count = valid_columns.len();
            let start = old_lengths[row];
            let end = start + count;
            let selection = Tensor::<B, 1, Int>::from_data(
                TensorData::new(valid_columns, [count]),
                &self.device,
            );
            let row_tokens = input
                .token_ids
                .clone()
                .slice([row..row + 1, 0..step])
                .select(1, selection.clone());
            tokens = tokens.slice_assign([row..row + 1, start..end], row_tokens);
            let row_positions = match &supplied_positions {
                Some(supplied) => supplied
                    .clone()
                    .slice([row..row + 1, 0..step])
                    .select(1, selection),
                None => Tensor::<B, 1, Int>::arange(start as i64..end as i64, &self.device)
                    .reshape([1, count]),
            };
            validate_decode_positions(
                row_positions.clone(),
                self.model_spec.max_position_embeddings,
            )?;
            positions = positions.slice_assign([row..row + 1, start..end], row_positions);

            let mut ordinal = 0usize;
            for column in 0..step {
                if validity[row * step + column] {
                    absolute_indices[row * step + column] = (start + ordinal) as i64;
                    ordinal += 1;
                }
            }
        }
        Ok(PreparedAppend {
            tokens,
            positions,
            new_lengths,
            counts,
            input_validity: validity,
            absolute_indices,
            step,
        })
    }

    /// Publishes a fully evaluated append and advances checked logical metadata.
    fn commit(&mut self, append: PreparedAppend<B>) -> Result<()> {
        self.metadata
            .advance_all(&append.counts)
            .map_err(|error| Error::Cache(error.to_string()))?;
        debug_assert_eq!(self.metadata.lengths(), append.new_lengths);
        self.tokens = append.tokens;
        self.positions = append.positions;
        Ok(())
    }
}

impl<B: Backend> ArchitectureDecodeState<B> for ModularDecodeState<B> {
    fn batch_size(&self) -> usize {
        self.metadata.config().sequence_count()
    }

    fn capacity(&self) -> usize {
        self.metadata.config().capacity()
    }

    fn lengths(&self) -> &[usize] {
        self.metadata.lengths()
    }

    fn reset(&mut self) -> Result<()> {
        self.metadata
            .reset()
            .map_err(|error| Error::Cache(error.to_string()))?;
        self.tokens = Tensor::zeros([self.batch_size(), self.capacity()], &self.device);
        self.positions = Tensor::zeros([self.batch_size(), self.capacity()], &self.device);
        Ok(())
    }

    fn reorder(&mut self, indices: &[usize]) -> Result<()> {
        let selection = indices
            .iter()
            .map(|&index| i64::try_from(index))
            .collect::<core::result::Result<Vec<_>, _>>()
            .map_err(|_| Error::Cache("custom reorder index does not fit i64".into()))?;
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

    fn truncate(&mut self, lengths: &[usize]) -> Result<()> {
        self.metadata
            .truncate_all(lengths)
            .map_err(|error| Error::Cache(error.to_string()))?;
        Ok(())
    }

    fn device(&self) -> &B::Device {
        &self.device
    }
}

/// Prospective device history plus indexing metadata for one append operation.
struct PreparedAppend<B: Backend> {
    tokens: Tensor<B, 2, Int>,
    positions: Tensor<B, 2, Int>,
    new_lengths: Vec<usize>,
    counts: Vec<usize>,
    input_validity: Vec<bool>,
    absolute_indices: Vec<i64>,
    step: usize,
}

impl<B: Backend> PreparedAppend<B> {
    /// Materializes the shortest rectangular prefix containing every compact row.
    fn history_input(&self, device: &B::Device) -> Result<CausalLmInput<B>> {
        let batch = self.new_lengths.len();
        let sequence = *self
            .new_lengths
            .iter()
            .max()
            .ok_or_else(|| Error::Cache("custom prepared append lost its batch".into()))?;
        let validity = (0..batch)
            .flat_map(|row| (0..sequence).map(move |column| column < self.new_lengths[row]))
            .collect::<Vec<_>>();
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
        let batch = self.new_lengths.len();
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

/// Reads a semantic input mask once for compact append planning.
fn read_validity<B: Backend>(
    mask: Option<&TokenMask<B>>,
    batch: usize,
    step: usize,
) -> Result<Vec<bool>> {
    match mask {
        Some(mask) => {
            if mask.as_tensor().dims() != [batch, step] {
                return Err(Error::shape(
                    "custom.decode.token_mask",
                    format!("[{batch}, {step}]"),
                    format!("{:?}", mask.as_tensor().dims()),
                ));
            }
            crate::architecture::tensor_bool_values(
                mask.as_tensor().clone(),
                "custom decode validity mask",
            )
        }
        None => Ok(vec![true; batch * step]),
    }
}

/// Rejects out-of-range decode IDs before they enter preallocated history.
fn validate_decode_token_ids<B: Backend>(
    token_ids: Tensor<B, 2, Int>,
    vocabulary: usize,
) -> Result<()> {
    let data = token_ids
        .try_into_data()
        .map_err(|error| {
            Error::BackendCapability(format!(
                "failed to inspect custom decode IDs on the backend: {error}"
            ))
        })?
        .convert::<i64>()
        .into_vec::<i64>()
        .map_err(|error| {
            Error::BackendCapability(format!("failed to inspect custom decode IDs: {error}"))
        })?;
    let minimum = data.iter().copied().min().unwrap_or(0);
    let maximum = data.iter().copied().max().unwrap_or(0);
    if minimum < 0 || maximum as usize >= vocabulary {
        return Err(Error::shape(
            "custom.decode.token_ids",
            format!("values in 0..{vocabulary}"),
            format!("observed range {minimum}..={maximum}"),
        ));
    }
    Ok(())
}

/// Rejects explicit cached positions outside the model context.
fn validate_decode_positions<B: Backend>(
    positions: Tensor<B, 2, Int>,
    capacity: usize,
) -> Result<()> {
    let data = positions
        .try_into_data()
        .map_err(|error| {
            Error::BackendCapability(format!(
                "failed to inspect custom decode positions on the backend: {error}"
            ))
        })?
        .convert::<i64>()
        .into_vec::<i64>()
        .map_err(|error| {
            Error::BackendCapability(format!(
                "failed to inspect custom decode positions: {error}"
            ))
        })?;
    let minimum = data.iter().copied().min().unwrap_or(0);
    let maximum = data.iter().copied().max().unwrap_or(0);
    if minimum < 0 || maximum as usize >= capacity {
        return Err(Error::shape(
            "custom.decode.position_ids",
            format!("values in 0..{capacity}"),
            format!("observed range {minimum}..={maximum}"),
        ));
    }
    Ok(())
}
