//! Scripted beam-search tests for ranking, cache reordering, sequence validity, and lifecycle failures.

use std::sync::{Arc, Mutex};

use burn::{
    backend::Flex,
    module::{Module, Param},
    tensor::{
        Bool, ElementConversion, Int, Tensor, TensorData,
        backend::{Backend, BackendTypes},
    },
};

use super::{
    BeamHypothesis, BeamSearchConfig, EarlyStopping, Seq2SeqBeamSearchRequest, completed_order,
    generate_seq2seq_beam_search,
};
use crate::generation::metadata;
use crate::{
    ArchitectureConfig, CausalLmInput, CausalLmOutput, DecodeInput, DecodeState, EncoderOutput,
    Error, FinishReason, ForwardOptions, Result, Seq2SeqInput, Seq2SeqLanguageModel, Seq2SeqOutput,
    TokenMask, cache::CacheConfig,
};

/// Deterministic F32 CPU backend used for scripted beam-search outcomes.
type TestBackend = Flex<f32, i32>;
/// Device type associated with the beam-search fixture backend.
type TestDevice = <TestBackend as BackendTypes>::Device;

/// Reads an optional `[batch, sequence]` mask to host flags; absent means all valid.
fn mask_flags<B: burn::tensor::backend::Backend>(
    mask: Option<&TokenMask<B>>,
    batch: usize,
    sequence: usize,
) -> Result<Vec<bool>> {
    metadata::read_grid(
        mask.map(|mask| mask.as_tensor().clone()),
        None,
        batch,
        sequence,
        "test mask",
    )
    .map(|grid| grid.valid)
}

/// Half-precision beam probabilities retain masking and match the F32 reference.
#[test]
fn beam_host_probabilities_respect_runtime_float_dtype() {
    use burn::tensor::DType;

    let device = Default::default();
    let logits = Tensor::<TestBackend, 2>::from_floats([[0.0, 1.0], [-1.0, 2.0]], &device);
    let reference = super::host_log_probabilities(logits.clone(), &[true, false], 2, 2)
        .expect("F32 beam reference");
    assert_eq!(reference.len(), 4);
    for dtype in [DType::F16, DType::BF16, DType::F32, DType::F64] {
        let actual =
            super::host_log_probabilities(logits.clone().cast(dtype), &[true, false], 2, 2)
                .expect("runtime dtype-aware beam conversion");
        assert_eq!(actual.len(), reference.len());
        for (actual, reference) in actual.into_iter().zip(&reference) {
            assert!(actual.is_finite() && (actual - reference).abs() < 0.02);
        }
    }
}

/// Minimal configuration for the coordinator's scripted model.
#[derive(Clone, Debug)]
struct ScriptedConfig {
    /// Number of scripted score columns emitted per decoder row.
    vocabulary_size: usize,
    /// Whether the fixture emits nonfinite logits to exercise rejection.
    emit_nonfinite: bool,
}

impl ArchitectureConfig for ScriptedConfig {
    /// Private architecture discriminator used only by the scripted fixture.
    const MODEL_TYPES: &'static [&'static str] = &["beam_search_script"];

    /// The test configuration only requires a non-empty vocabulary.
    fn validate(&self) -> Result<()> {
        if self.vocabulary_size == 0 {
            Err(Error::configuration(
                "beam_search_script.vocabulary_size",
                "must be nonzero",
            ))
        } else {
            Ok(())
        }
    }
}

/// Observable lifecycle events emitted by the scripted cache and model.
#[derive(Debug, Default)]
struct ScriptTrace {
    /// Ordered model and cache method boundaries reached by the coordinator.
    boundaries: Vec<&'static str>,
    /// Count of encoder executions, which should not repeat for each beam.
    encode_calls: usize,
    /// Count of external decoder cache allocations.
    state_allocations: usize,
    /// Count of decoder prompt prefill calls.
    prefill_calls: usize,
    /// Exact global parent indices supplied to each beam-cache reorder.
    reorders: Vec<Vec<usize>>,
    /// Token IDs passed to successive decoder calls.
    decode_tokens: Vec<Vec<i64>>,
    /// Validity flags passed with each decoder token batch.
    decode_valid: Vec<Vec<bool>>,
    /// Explicit decoder positions when the request supplies them.
    decode_positions: Vec<Option<Vec<i64>>>,
    /// Committed logical row lengths observed after each decoder call.
    decode_lengths: Vec<Vec<usize>>,
}

/// Public model/cache failures and malformed outputs at coordinator boundaries.
#[derive(Clone, Copy, Debug)]
enum ScriptFault {
    /// Return encoder output with the wrong batch extent.
    EncodeBatch,
    /// Return decoder prefill logits with invalid dimensions.
    PrefillLogits,
    /// Return one-step decoder logits with invalid dimensions.
    DecodeLogits,
    /// Fail while encoding the source batch.
    EncodeError,
    /// Fail while allocating external decoder state.
    StateError,
    /// Fail during decoder prompt prefill.
    PrefillError,
    /// Fail while reordering beam cache rows.
    ReorderError,
    /// Fail during a one-token decoder step.
    DecodeError,
    /// Corrupt the encoder validity mask returned by decoder prefill.
    PrefillEncoderMask,
    /// Corrupt encoder hidden states returned by a decoder step.
    DecodeEncoderHiddenState,
}

/// Fixed-batch logical state used to prove coordinator reorder behavior.
#[derive(Debug)]
struct ScriptedState<B: Backend> {
    /// Fixed expanded decoder batch, including every beam slot.
    batch: usize,
    /// Maximum logical decoder length accepted by the fixture.
    capacity: usize,
    /// Committed valid-token count for each expanded decoder row.
    lengths: Vec<usize>,
    /// Device shared by the scripted tensors and state.
    device: B::Device,
    /// Shared event log used to assert cache and coordinator lifecycle behavior.
    trace: Arc<Mutex<ScriptTrace>>,
    /// Optional cache or model boundary failure selected for this run.
    fault: Option<ScriptFault>,
}

impl<B: Backend> DecodeState<B> for ScriptedState<B> {
    /// Returns the configured fixed beam batch.
    fn batch_size(&self) -> usize {
        self.batch
    }

    /// Returns the configured logical target capacity.
    fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns committed decoder lengths after false-mask no-ops.
    fn lengths(&self) -> &[usize] {
        &self.lengths
    }

    /// Clears all scripted logical lengths.
    fn reset(&mut self) -> Result<()> {
        self.lengths.fill(0);
        Ok(())
    }

    /// Reorders or duplicates lengths and records exact parent indices.
    fn reorder(&mut self, indices: &[usize]) -> Result<()> {
        self.trace
            .lock()
            .expect("script trace")
            .boundaries
            .push("reorder");
        if matches!(self.fault, Some(ScriptFault::ReorderError)) {
            return Err(Error::Cache("scripted cache reorder failed".into()));
        }
        if indices.len() != self.batch {
            return Err(Error::Cache(format!(
                "scripted reorder expected {} indices, found {}",
                self.batch,
                indices.len()
            )));
        }
        let next = indices
            .iter()
            .map(|&index| {
                self.lengths.get(index).copied().ok_or_else(|| {
                    Error::Cache(format!("scripted reorder index {index} is out of bounds"))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        self.lengths = next;
        self.trace
            .lock()
            .expect("script trace mutex")
            .reorders
            .push(indices.to_vec());
        Ok(())
    }

    /// Rewinds every scripted logical length.
    fn truncate(&mut self, lengths: &[usize]) -> Result<()> {
        if lengths.len() != self.batch
            || lengths
                .iter()
                .zip(&self.lengths)
                .any(|(next, current)| next > current)
        {
            return Err(Error::Cache("invalid scripted truncation".into()));
        }
        self.lengths.clone_from_slice(lengths);
        Ok(())
    }

    /// Returns the backend allocation device.
    fn device(&self) -> &B::Device {
        &self.device
    }
}

/// Tiny deterministic seq2seq graph whose logits expose beam decisions.
#[derive(Module, Debug)]
struct ScriptedModel<B: Backend> {
    /// Minimal registered tensor satisfying Burn's derived module contract.
    dummy: Param<Tensor<B, 1>>,
    /// Vocabulary extent and nonfinite-output controls.
    #[module(skip)]
    config: ScriptedConfig,
    /// Model and cache events shared with the assertions.
    #[module(skip)]
    trace: Arc<Mutex<ScriptTrace>>,
    /// Optional malformed output or recoverable error injection.
    #[module(skip)]
    fault: Option<ScriptFault>,
    /// Optional hard vocabulary restriction, represented by valid negative-infinity logits.
    #[module(skip)]
    vocabulary_mask: Option<Vec<bool>>,
    /// Optional Markov logits table replacing the scripted five-token rows.
    #[module(skip)]
    table: Option<MarkovTable>,
}

/// First-order Markov decoder: logits depend only on the previous token.
#[derive(Clone, Debug)]
struct MarkovTable {
    /// Logits returned by decoder prefill, one row per source: a source row
    /// whose first token (its marker) is `m` receives `first[m % first.len()]`.
    first: Vec<Vec<f32>>,
    /// Row `t` holds the logits returned after decoding token `t`.
    next: Vec<Vec<f32>>,
}

impl<B: Backend> ScriptedModel<B> {
    /// Initializes one parameter so Burn can derive the full Module contract.
    fn new(config: ScriptedConfig, device: &B::Device) -> Self {
        Self {
            dummy: Param::from_tensor(Tensor::zeros([1], device)),
            config,
            trace: Arc::new(Mutex::new(ScriptTrace::default())),
            fault: None,
            vocabulary_mask: None,
            table: None,
        }
    }

    /// Creates a model output from one flat row-major logits buffer.
    fn output(
        &self,
        mut logits: Vec<f32>,
        batch: usize,
        encoder: EncoderOutput<B>,
        device: &B::Device,
    ) -> Seq2SeqOutput<B> {
        if let Some(mask) = &self.vocabulary_mask {
            assert_eq!(mask.len(), self.config.vocabulary_size);
            for row in logits.chunks_exact_mut(mask.len()) {
                for (score, &allowed) in row.iter_mut().zip(mask) {
                    if !allowed {
                        *score = f32::NEG_INFINITY;
                    }
                }
            }
        }
        Seq2SeqOutput {
            decoder: CausalLmOutput {
                logits: Tensor::from_data(
                    TensorData::new(logits, [batch, 1, self.config.vocabulary_size]),
                    device,
                ),
                final_hidden_state: None,
                hidden_states: Vec::new(),
                attentions: Vec::new(),
                routers: Vec::new(),
                load_balancing_loss: None,
            },
            encoder,
            cross_attentions: Vec::new(),
        }
    }

    /// Returns first-step logits keyed by the first encoded source token.
    fn prefill_logits(&self, encoder: &EncoderOutput<B>) -> Result<Vec<f32>> {
        let [batch, source, _] = encoder.hidden_state.dims();
        let markers = encoder
            .hidden_state
            .clone()
            .slice([0..batch, 0..1.min(source), 0..1])
            .reshape([batch])
            .into_data()
            .to_vec::<B::FloatElem>()
            .map_err(|error| Error::BackendCapability(error.to_string()))?;
        let mut logits = Vec::with_capacity(batch * self.config.vocabulary_size);
        if let Some(table) = &self.table {
            for marker in markers {
                let source = marker.elem::<f64>() as usize % table.first.len();
                logits.extend(&table.first[source]);
            }
            return Ok(logits);
        }
        for marker in markers {
            if self.config.emit_nonfinite {
                logits.extend([f32::NAN, 0.0, 0.0, 0.0, 0.0]);
            } else if marker.elem::<f64>() < 3.5 {
                // A second source marker favors a non-EOS continuation.
                logits.extend([0.0, -1.0, -1.0, 5.0, 1.0]);
            } else {
                // IDs one and two are an intentional stable tie.
                logits.extend([0.0, 4.0, 4.0, 1.0, 3.0]);
            }
        }
        Ok(logits)
    }

    /// Returns continuation logits keyed by each selected decoder token.
    fn decode_logits(&self, tokens: &[i64], valid: &[bool]) -> Vec<f32> {
        let mut logits = Vec::with_capacity(tokens.len() * self.config.vocabulary_size);
        if let Some(table) = &self.table {
            for (&token, &is_valid) in tokens.iter().zip(valid) {
                if is_valid {
                    logits.extend(&table.next[token as usize]);
                } else {
                    logits.extend(vec![0.0; self.config.vocabulary_size]);
                }
            }
            return logits;
        }
        for (&token, &is_valid) in tokens.iter().zip(valid) {
            if !is_valid {
                logits.extend([0.0; 5]);
            } else {
                match token {
                    1 => logits.extend([0.0, 0.0, 1.0, 4.0, 5.0]),
                    2 => logits.extend([0.0, 0.0, 1.0, 5.0, 4.0]),
                    _ => logits.extend([0.0, 0.0, 1.0, 5.0, 3.0]),
                }
            }
        }
        logits
    }
}

impl<B: Backend> Seq2SeqLanguageModel<B> for ScriptedModel<B> {
    /// Configuration of the deterministic scripted vocabulary.
    type Config = ScriptedConfig;
    /// Fixed-batch logical cache used to observe reorder and validity decisions.
    type State = ScriptedState<B>;

    /// Encodes source IDs as one-channel floating markers.
    fn encode(
        &self,
        source: CausalLmInput<B>,
        _options: &ForwardOptions,
    ) -> Result<EncoderOutput<B>> {
        {
            let mut trace = self.trace.lock().expect("script trace");
            trace.boundaries.push("encode");
            trace.encode_calls += 1;
        }
        if matches!(self.fault, Some(ScriptFault::EncodeError)) {
            return Err(Error::BackendCapability("scripted encoder failed".into()));
        }
        let hidden_state = source.token_ids.float().unsqueeze_dim(2);
        Ok(EncoderOutput {
            hidden_state: if matches!(self.fault, Some(ScriptFault::EncodeBatch)) {
                hidden_state.repeat_dim(0, 2)
            } else {
                hidden_state
            },
            token_mask: source.token_mask,
            hidden_states: Vec::new(),
            attentions: Vec::new(),
        })
    }

    /// Provides a complete stateless graph for the trait, unused by search.
    fn forward(
        &self,
        input: Seq2SeqInput<B>,
        options: &ForwardOptions,
    ) -> Result<Seq2SeqOutput<B>> {
        let encoder = self.encode(input.encoder, options)?;
        let batch = input.decoder.token_ids.dims()[0];
        let logits = self.prefill_logits(&encoder)?;
        let device = input.decoder.token_ids.device();
        Ok(self.output(logits, batch, encoder, &device))
    }

    /// Allocates the fixed-batch scripted lifecycle state.
    fn new_decode_state(
        &self,
        config: CacheConfig,
        batch_size: usize,
        device: &B::Device,
    ) -> Result<Self::State> {
        {
            let mut trace = self.trace.lock().expect("script trace");
            trace.boundaries.push("cache");
            trace.state_allocations += 1;
        }
        if matches!(self.fault, Some(ScriptFault::StateError)) {
            return Err(Error::Cache("scripted cache allocation failed".into()));
        }
        if config.sequence_count() != batch_size {
            return Err(Error::Cache("scripted cache batch mismatch".into()));
        }
        Ok(ScriptedState {
            batch: batch_size,
            capacity: config.capacity(),
            lengths: vec![0; batch_size],
            device: device.clone(),
            trace: Arc::clone(&self.trace),
            fault: self.fault,
        })
    }

    /// Commits initial decoder-prefix lengths and returns scripted logits.
    fn prefill_decoder(
        &self,
        encoder: &EncoderOutput<B>,
        decoder: CausalLmInput<B>,
        state: &mut Self::State,
        _options: &ForwardOptions,
    ) -> Result<Seq2SeqOutput<B>> {
        {
            let mut trace = self.trace.lock().expect("script trace");
            trace.boundaries.push("prefill");
            trace.prefill_calls += 1;
        }
        if matches!(self.fault, Some(ScriptFault::PrefillError)) {
            return Err(Error::Checkpoint("scripted decoder prefill failed".into()));
        }
        let [batch, sequence] = decoder.token_ids.dims();
        let valid = mask_flags(decoder.token_mask.as_ref(), batch, sequence)?;
        state.lengths =
            metadata::nonempty_row_lengths(&valid, batch, sequence, "scripted prefill")?;
        let logits = self.prefill_logits(encoder)?;
        let device = decoder.token_ids.device();
        let mut output = self.output(logits, batch, encoder.clone(), &device);
        if matches!(self.fault, Some(ScriptFault::PrefillLogits)) {
            output.decoder.logits = output.decoder.logits.repeat_dim(1, 2);
        }
        if matches!(self.fault, Some(ScriptFault::PrefillEncoderMask)) {
            let source = encoder.hidden_state.dims()[1];
            output.encoder.token_mask = Some(TokenMask::new(Tensor::from_data(
                TensorData::new(vec![true; batch * (source + 1)], [batch, source + 1]),
                &device,
            )));
        }
        Ok(output)
    }

    /// Records one reordered cached step and returns token-keyed logits.
    fn decode(
        &self,
        encoder: &EncoderOutput<B>,
        input: DecodeInput<B>,
        state: &mut Self::State,
        _options: &ForwardOptions,
    ) -> Result<Seq2SeqOutput<B>> {
        self.trace
            .lock()
            .expect("script trace")
            .boundaries
            .push("decode");
        if matches!(self.fault, Some(ScriptFault::DecodeError)) {
            return Err(Error::Checkpoint("scripted decoder step failed".into()));
        }
        let [batch, sequence] = input.token_ids.dims();
        if sequence != 1 || batch != state.batch {
            return Err(Error::Cache("scripted decode shape mismatch".into()));
        }
        let tokens = metadata::tensor_i64_values(input.token_ids.clone(), "scripted tokens")?;
        let valid = mask_flags(input.token_mask.as_ref(), batch, sequence)?;
        for (length, &is_valid) in state.lengths.iter_mut().zip(&valid) {
            if is_valid {
                *length += 1;
            }
        }
        let positions = input
            .position_ids
            .map(|positions| metadata::tensor_i64_values(positions, "scripted positions"))
            .transpose()?;
        {
            let mut trace = self.trace.lock().expect("script trace mutex");
            trace.decode_tokens.push(tokens.clone());
            trace.decode_valid.push(valid.clone());
            trace.decode_positions.push(positions);
            trace.decode_lengths.push(state.lengths.clone());
        }
        let logits = self.decode_logits(&tokens, &valid);
        let device = input.token_ids.device();
        let mut output = self.output(logits, batch, encoder.clone(), &device);
        if matches!(self.fault, Some(ScriptFault::DecodeLogits)) {
            output.decoder.logits = output.decoder.logits.repeat_dim(1, 2);
        }
        if matches!(self.fault, Some(ScriptFault::DecodeEncoderHiddenState)) {
            let [batch, source, hidden] = encoder.hidden_state.dims();
            output
                .encoder
                .hidden_states
                .push(Tensor::zeros([batch, source, hidden + 1], &device));
        }
        Ok(output)
    }
}

/// Constructs a valid one-source scripted request.
fn request(device: &TestDevice, search: BeamSearchConfig) -> Seq2SeqBeamSearchRequest<TestBackend> {
    let beam_batch = search.num_beams;
    Seq2SeqBeamSearchRequest {
        source: CausalLmInput {
            token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[4, 0]], device),
            token_mask: None,
            position_ids: None,
        },
        decoder_prompt: CausalLmInput {
            token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[0]], device),
            token_mask: None,
            position_ids: None,
        },
        cache: CacheConfig::new(beam_batch, search.max_new_tokens + 1).expect("script cache"),
        search,
        vocabulary_size: 5,
    }
}

/// Returns standard scripted search controls with no EOS termination.
///
/// Uses exhaustive (`"never"`) stopping so scripted traces show every step a
/// search to the length bound needs; heuristic tests opt in explicitly.
fn search(max_new_tokens: usize, beams: usize) -> BeamSearchConfig {
    BeamSearchConfig {
        max_new_tokens,
        num_beams: beams,
        num_return_sequences: beams,
        length_penalty: 1.0,
        early_stopping: EarlyStopping::Never,
        eos_token_ids: Vec::new(),
        pad_token_id: 0,
    }
}

/// Every scalar range and cross-field relationship fails preflight.
#[test]
fn beam_configuration_rejects_invalid_values() {
    let invalid = [
        BeamSearchConfig {
            max_new_tokens: 0,
            ..BeamSearchConfig::default()
        },
        BeamSearchConfig {
            num_beams: 0,
            ..BeamSearchConfig::default()
        },
        BeamSearchConfig {
            num_beams: 6,
            ..BeamSearchConfig::default()
        },
        BeamSearchConfig {
            num_beams: 2,
            num_return_sequences: 0,
            ..BeamSearchConfig::default()
        },
        BeamSearchConfig {
            num_beams: 2,
            num_return_sequences: 3,
            ..BeamSearchConfig::default()
        },
        BeamSearchConfig {
            length_penalty: f64::NAN,
            ..BeamSearchConfig::default()
        },
        BeamSearchConfig {
            length_penalty: f64::MAX,
            ..BeamSearchConfig::default()
        },
        BeamSearchConfig {
            pad_token_id: 5,
            ..BeamSearchConfig::default()
        },
        BeamSearchConfig {
            eos_token_ids: vec![5],
            ..BeamSearchConfig::default()
        },
    ];
    for config in invalid {
        assert!(config.validate(5).is_err(), "accepted {config:?}");
    }
    assert!(BeamSearchConfig::default().validate(0).is_err());
    assert!(
        BeamSearchConfig {
            length_penalty: -0.1,
            ..BeamSearchConfig::default()
        }
        .validate(5)
        .is_ok()
    );
    assert!(
        BeamSearchConfig {
            num_beams: 1,
            num_return_sequences: 1,
            ..BeamSearchConfig::default()
        }
        .validate(5)
        .is_ok()
    );
}

/// Score ties are resolved by ascending token ID and remain reproducible.
#[test]
fn one_step_stable_ties_return_tokens_in_id_order() {
    let device = Default::default();
    let model = ScriptedModel::<TestBackend>::new(
        ScriptedConfig {
            vocabulary_size: 5,
            emit_nonfinite: false,
        },
        &device,
    );
    let output = generate_seq2seq_beam_search(&model, request(&device, search(1, 2)))
        .expect("one-step beam search");
    assert_eq!(output.sequences.len(), 1);
    assert_eq!(output.sequences[0].len(), 2);
    assert_eq!(output.sequences[0][0].token_ids, vec![1]);
    assert_eq!(output.sequences[0][1].token_ids, vec![2]);
    assert_eq!(output.sequences[0][0].score, output.sequences[0][1].score);
    for sequence in &output.sequences[0] {
        assert_eq!(sequence.decoder_prompt_length, 1);
        assert_eq!(
            sequence.log_probabilities.iter().sum::<f64>(),
            sequence.cumulative_log_probability
        );
    }
    assert!(
        output.sequences[0]
            .iter()
            .all(|sequence| sequence.finish_reason == FinishReason::Length)
    );
}

/// Parent cache rows are duplicated first and gathered at every later step.
#[test]
fn cached_multistep_search_reorders_external_state() {
    let device = Default::default();
    let model = ScriptedModel::<TestBackend>::new(
        ScriptedConfig {
            vocabulary_size: 5,
            emit_nonfinite: false,
        },
        &device,
    );
    let trace = Arc::clone(&model.trace);
    let output = generate_seq2seq_beam_search(&model, request(&device, search(3, 2)))
        .expect("multistep beam search");
    assert_eq!(output.sequences[0].len(), 2);
    assert!(
        output.sequences[0]
            .iter()
            .all(|sequence| sequence.token_ids.len() == 3)
    );
    let trace = trace.lock().expect("script trace mutex");
    assert_eq!(trace.reorders.len(), 2);
    assert_eq!(trace.reorders[0], vec![0, 0]);
    assert_eq!(trace.decode_tokens.len(), 2);
    assert!(trace.decode_valid.iter().flatten().all(|valid| *valid));
}

/// Eager early stopping halts once the beam-width EOS pool is complete.
#[test]
fn eager_early_stopping_avoids_unnecessary_decode() {
    let device = Default::default();
    let model = ScriptedModel::<TestBackend>::new(
        ScriptedConfig {
            vocabulary_size: 5,
            emit_nonfinite: false,
        },
        &device,
    );
    let trace = Arc::clone(&model.trace);
    let mut config = search(4, 2);
    config.eos_token_ids = vec![1, 2];
    config.early_stopping = EarlyStopping::Eager;
    let output =
        generate_seq2seq_beam_search(&model, request(&device, config)).expect("early beam search");
    assert_eq!(output.sequences[0][0].token_ids, vec![1]);
    assert_eq!(output.sequences[0][1].token_ids, vec![2]);
    assert!(output.sequences[0].iter().all(|sequence| matches!(
        sequence.finish_reason,
        FinishReason::EndOfSequence { token_id: 1 | 2 }
    )));
    let trace = trace.lock().expect("script trace mutex");
    assert!(trace.reorders.is_empty());
    assert!(trace.decode_tokens.is_empty());
}

/// Lower-frontier EOS candidates are backups, not admissible completions.
#[test]
fn eager_stopping_ignores_eos_below_top_beam_ranks() {
    let device = Default::default();
    let model = ScriptedModel::<TestBackend>::new(
        ScriptedConfig {
            vocabulary_size: 5,
            emit_nonfinite: false,
        },
        &device,
    );
    let trace = Arc::clone(&model.trace);
    let mut config = search(4, 2);
    // The prefill ranks live IDs 1 and 2 first, followed by EOS IDs 4
    // and 3. Transformers 5.2 therefore ignores both initial EOS rows;
    // they must not fill the eager completion pool before one decode.
    config.eos_token_ids = vec![3, 4];
    config.early_stopping = EarlyStopping::Eager;
    let output = generate_seq2seq_beam_search(&model, request(&device, config))
        .expect("lower-frontier EOS beam search");
    assert_eq!(output.sequences[0].len(), 2);
    assert!(output.sequences[0].iter().all(|sequence| {
        sequence.token_ids.len() == 2
            && matches!(
                sequence.finish_reason,
                FinishReason::EndOfSequence { token_id: 3 | 4 }
            )
    }));
    let trace = trace.lock().expect("script trace mutex");
    assert_eq!(trace.reorders, vec![vec![0, 0]]);
    assert_eq!(trace.decode_tokens, vec![vec![1, 2]]);
    assert_eq!(trace.decode_valid, vec![vec![true, true]]);
}

/// Disabled eager stopping continues live non-EOS prefixes to the bound.
#[test]
fn non_eager_search_continues_after_enough_eos_hypotheses() {
    let device = Default::default();
    let model = ScriptedModel::<TestBackend>::new(
        ScriptedConfig {
            vocabulary_size: 5,
            emit_nonfinite: false,
        },
        &device,
    );
    let trace = Arc::clone(&model.trace);
    let mut config = search(3, 2);
    config.eos_token_ids = vec![1, 2];
    let _ = generate_seq2seq_beam_search(&model, request(&device, config))
        .expect("non-eager beam search");
    let trace = trace.lock().expect("script trace mutex");
    assert_eq!(trace.decode_tokens.len(), 2);
    assert_eq!(trace.reorders.len(), 2);
}

/// Finished source groups contribute false-masked PAD rows while peers run.
#[test]
fn batched_finished_rows_use_pad_noops_during_peer_decode() {
    let device = Default::default();
    let model = ScriptedModel::<TestBackend>::new(
        ScriptedConfig {
            vocabulary_size: 5,
            emit_nonfinite: false,
        },
        &device,
    );
    let trace = Arc::clone(&model.trace);
    let mut config = search(3, 1);
    config.num_return_sequences = 1;
    config.eos_token_ids = vec![1];
    config.early_stopping = EarlyStopping::Eager;
    let request = Seq2SeqBeamSearchRequest {
        source: CausalLmInput {
            token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[4, 0], [3, 0]], &device),
            token_mask: None,
            position_ids: None,
        },
        decoder_prompt: CausalLmInput {
            token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[0], [0]], &device),
            token_mask: None,
            position_ids: None,
        },
        cache: CacheConfig::new(2, 4).unwrap(),
        search: config,
        vocabulary_size: 5,
    };
    let output = generate_seq2seq_beam_search(&model, request).expect("ragged batch beam search");
    assert_eq!(output.sequences.len(), 2);
    let trace = trace.lock().expect("script trace mutex");
    assert!(!trace.decode_valid.is_empty());
    assert_eq!(trace.decode_valid[0], vec![false, true]);
    assert_eq!(trace.decode_tokens[0], vec![0, 3]);
}

/// Explicit decoder positions follow the selected parent through reorder.
#[test]
fn explicit_positions_advance_after_beam_duplication() {
    let device = Default::default();
    let model = ScriptedModel::<TestBackend>::new(
        ScriptedConfig {
            vocabulary_size: 5,
            emit_nonfinite: false,
        },
        &device,
    );
    let trace = Arc::clone(&model.trace);
    let mut request = request(&device, search(2, 2));
    request.decoder_prompt.position_ids = Some(Tensor::from_ints([[7]], &device));
    let _ = generate_seq2seq_beam_search(&model, request).expect("positioned beam search");
    let trace = trace.lock().expect("script trace mutex");
    assert_eq!(trace.decode_positions, vec![Some(vec![8, 8])]);
}

/// Positive length penalty can prefer a longer, lower-probability sequence.
#[test]
fn length_penalty_changes_finished_ranking() {
    let short = BeamHypothesis {
        token_ids: vec![1],
        log_probabilities: vec![-1.0],
        cumulative_log_probability: -1.0,
        next_position: None,
        finish_reason: Some(FinishReason::Length),
        ordinal: 0,
    };
    let long = BeamHypothesis {
        token_ids: vec![2, 3],
        log_probabilities: vec![-0.75, -0.75],
        cumulative_log_probability: -1.5,
        next_position: None,
        finish_reason: Some(FinishReason::Length),
        ordinal: 1,
    };
    assert!(completed_order(&short, &long, 0.0).is_lt());
    assert!(completed_order(&long, &short, 1.0).is_lt());
}

/// Invalid request rectangles and cache bounds fail before state allocation.
#[test]
fn request_preflight_rejects_shape_mask_and_cache_errors() {
    let device = Default::default();
    let model = ScriptedModel::<TestBackend>::new(
        ScriptedConfig {
            vocabulary_size: 5,
            emit_nonfinite: false,
        },
        &device,
    );
    let mut wrong_batch = request(&device, search(2, 2));
    wrong_batch.decoder_prompt.token_ids = Tensor::from_ints([[0], [0]], &device);
    assert!(generate_seq2seq_beam_search(&model, wrong_batch).is_err());

    let mut wrong_mask = request(&device, search(2, 2));
    wrong_mask.decoder_prompt.token_mask = Some(TokenMask::new(
        Tensor::<TestBackend, 2, Bool>::from_bool([[true, false]].into(), &device),
    ));
    assert!(generate_seq2seq_beam_search(&model, wrong_mask).is_err());

    let mut empty_prompt = request(&device, search(2, 2));
    empty_prompt.decoder_prompt.token_mask = Some(TokenMask::new(
        Tensor::<TestBackend, 2, Bool>::from_bool([[false]].into(), &device),
    ));
    assert!(generate_seq2seq_beam_search(&model, empty_prompt).is_err());

    let mut wrong_cache_batch = request(&device, search(2, 2));
    wrong_cache_batch.cache = CacheConfig::new(1, 3).unwrap();
    assert!(generate_seq2seq_beam_search(&model, wrong_cache_batch).is_err());

    // One prompt token plus two new tokens needs two slots: the final token
    // is returned but never decoded. One slot is too few.
    let mut short_cache = request(&device, search(2, 2));
    short_cache.cache = CacheConfig::new(2, 1).unwrap();
    let error = generate_seq2seq_beam_search(&model, short_cache).expect_err("one slot short");
    assert!(error.to_string().contains("= 2"), "{error}");
    let mut exact_cache = request(&device, search(2, 2));
    exact_cache.cache = CacheConfig::new(2, 2).unwrap();
    let output = generate_seq2seq_beam_search(&model, exact_cache)
        .expect("an exactly sized cache holds every fed-back token");
    assert!(
        output.sequences[0]
            .iter()
            .all(|sequence| sequence.token_ids.len() == 2)
    );
}

/// Gapped true-valid decoder masks report semantic rather than padded length.
#[test]
fn decoder_prompt_length_counts_only_valid_mask_entries() {
    let device = Default::default();
    let model = ScriptedModel::<TestBackend>::new(
        ScriptedConfig {
            vocabulary_size: 5,
            emit_nonfinite: false,
        },
        &device,
    );
    let mut request = request(&device, search(1, 2));
    request.decoder_prompt = CausalLmInput {
        token_ids: Tensor::from_ints([[0, 4, 0]], &device),
        token_mask: Some(TokenMask::new(Tensor::from_bool(
            [[true, false, true]].into(),
            &device,
        ))),
        position_ids: None,
    };
    request.cache = CacheConfig::new(2, 3).unwrap();
    let output =
        generate_seq2seq_beam_search(&model, request).expect("masked decoder prompt beam search");
    assert!(
        output.sequences[0]
            .iter()
            .all(|sequence| sequence.decoder_prompt_length == 2)
    );
}

/// NaN model scores are rejected instead of entering an unstable sort.
#[test]
fn nonfinite_model_probabilities_are_contextual_errors() {
    let device = Default::default();
    let model = ScriptedModel::<TestBackend>::new(
        ScriptedConfig {
            vocabulary_size: 5,
            emit_nonfinite: true,
        },
        &device,
    );
    let error = generate_seq2seq_beam_search(&model, request(&device, search(1, 2)))
        .expect_err("NaN scores must fail");
    assert!(error.to_string().contains("non-finite"));
}

/// Malformed requests fail before the model can encode or allocate any state.
#[test]
fn malformed_request_metadata_is_rejected_before_model_calls() {
    let device = Default::default();
    let model = ScriptedModel::<TestBackend>::new(
        ScriptedConfig {
            vocabulary_size: 5,
            emit_nonfinite: false,
        },
        &device,
    );
    let mut cases = Vec::new();
    let mut empty_source = request(&device, search(2, 2));
    empty_source.source.token_ids = Tensor::zeros([1, 0], &device);
    cases.push(empty_source);
    let mut empty_decoder = request(&device, search(2, 2));
    empty_decoder.decoder_prompt.token_ids = Tensor::zeros([1, 0], &device);
    cases.push(empty_decoder);
    let mut source_mask = request(&device, search(2, 2));
    source_mask.source.token_mask =
        Some(TokenMask::new(Tensor::from_bool([[true]].into(), &device)));
    cases.push(source_mask);
    let mut source_positions = request(&device, search(2, 2));
    source_positions.source.position_ids = Some(Tensor::from_ints([[0]], &device));
    cases.push(source_positions);
    let mut decoder_positions = request(&device, search(2, 2));
    decoder_positions.decoder_prompt.position_ids = Some(Tensor::from_ints([[0, 1]], &device));
    cases.push(decoder_positions);
    for last in [-1, i32::MAX] {
        let mut positions = request(&device, search(2, 2));
        positions.decoder_prompt.position_ids = Some(Tensor::from_ints([[last]], &device));
        cases.push(positions);
    }
    let mut invalid_options = request(&device, search(2, 2));
    invalid_options.search.max_new_tokens = 0;
    cases.push(invalid_options);
    let mut impossible_capacity = request(&device, search(2, 2));
    impossible_capacity.search.max_new_tokens = usize::MAX;
    cases.push(impossible_capacity);
    for request in cases {
        assert!(generate_seq2seq_beam_search(&model, request).is_err());
    }
    let trace = model.trace.lock().expect("trace");
    assert_eq!(
        (
            trace.encode_calls,
            trace.state_allocations,
            trace.prefill_calls
        ),
        (0, 0, 0)
    );
    assert!(trace.reorders.is_empty() && trace.decode_tokens.is_empty());
}

/// Invalid model output is rejected at its producing stage, before later mutation.
#[test]
fn malformed_model_outputs_stop_at_their_validation_boundary() {
    let device = Default::default();
    for (fault, expected_calls, expected_decodes, label) in [
        (ScriptFault::EncodeBatch, (1, 0, 0), 0, "encoder_output"),
        (ScriptFault::PrefillLogits, (1, 1, 1), 0, "model_logits"),
        (ScriptFault::DecodeLogits, (1, 1, 1), 1, "model_logits"),
    ] {
        let mut model = ScriptedModel::<TestBackend>::new(
            ScriptedConfig {
                vocabulary_size: 5,
                emit_nonfinite: false,
            },
            &device,
        );
        model.fault = Some(fault);
        let error = generate_seq2seq_beam_search(&model, request(&device, search(3, 2)))
            .expect_err("malformed output must not reach another model step");
        assert!(error.to_string().contains(label));
        let trace = model.trace.lock().expect("trace");
        assert_eq!(
            (
                trace.encode_calls,
                trace.state_allocations,
                trace.prefill_calls
            ),
            expected_calls
        );
        assert_eq!(trace.decode_tokens.len(), expected_decodes);
        assert_eq!(trace.reorders.len(), expected_decodes);
    }
}

/// Model and cache errors retain their category and stop before the next boundary.
#[test]
fn beam_contract_public_model_and_cache_failures_preserve_exact_errors() {
    let device = Default::default();
    for (fault, expected, expected_boundaries) in [
        (
            ScriptFault::EncodeError,
            Error::BackendCapability("scripted encoder failed".into()),
            vec!["encode"],
        ),
        (
            ScriptFault::StateError,
            Error::Cache("scripted cache allocation failed".into()),
            vec!["encode", "cache"],
        ),
        (
            ScriptFault::PrefillError,
            Error::Checkpoint("scripted decoder prefill failed".into()),
            vec!["encode", "cache", "prefill"],
        ),
        (
            ScriptFault::ReorderError,
            Error::Cache("scripted cache reorder failed".into()),
            vec!["encode", "cache", "prefill", "reorder"],
        ),
        (
            ScriptFault::DecodeError,
            Error::Checkpoint("scripted decoder step failed".into()),
            vec!["encode", "cache", "prefill", "reorder", "decode"],
        ),
    ] {
        let mut model = ScriptedModel::<TestBackend>::new(
            ScriptedConfig {
                vocabulary_size: 5,
                emit_nonfinite: false,
            },
            &device,
        );
        model.fault = Some(fault);
        let error = generate_seq2seq_beam_search(&model, request(&device, search(3, 2)))
            .expect_err("a failed public model boundary aborts the search");
        assert_eq!(format!("{error:?}"), format!("{expected:?}"));
        let trace = model.trace.lock().expect("trace");
        assert_eq!(trace.boundaries, expected_boundaries);
        assert!(
            trace.decode_tokens.is_empty(),
            "failed steps must not advance the script"
        );
        assert_eq!(
            trace.reorders.len(),
            usize::from(matches!(fault, ScriptFault::DecodeError))
        );
    }
}

/// Returned encoder masks are checked before a beam cache can be reordered.
#[test]
fn beam_contract_prefill_encoder_mask_failure_stops_before_reordering() {
    let device = Default::default();
    let mut model = ScriptedModel::<TestBackend>::new(
        ScriptedConfig {
            vocabulary_size: 5,
            emit_nonfinite: false,
        },
        &device,
    );
    model.fault = Some(ScriptFault::PrefillEncoderMask);
    let error = generate_seq2seq_beam_search(&model, request(&device, search(3, 2)))
        .expect_err("a model-returned source mask must match its hidden state");
    let expected = Error::shape("beam_search.encoder_output.token_mask", "[2, 2]", "[2, 3]");
    assert_eq!(format!("{error:?}"), format!("{expected:?}"));
    let trace = model.trace.lock().expect("trace");
    assert_eq!(trace.boundaries, ["encode", "cache", "prefill"]);
    assert!(trace.reorders.is_empty() && trace.decode_tokens.is_empty());
}

/// Invalid decode-time encoder diagnostics never become the next beam's inputs.
#[test]
fn beam_contract_decode_encoder_diagnostic_failure_stops_after_one_step() {
    let device = Default::default();
    let mut model = ScriptedModel::<TestBackend>::new(
        ScriptedConfig {
            vocabulary_size: 5,
            emit_nonfinite: false,
        },
        &device,
    );
    model.fault = Some(ScriptFault::DecodeEncoderHiddenState);
    let error = generate_seq2seq_beam_search(&model, request(&device, search(3, 2)))
        .expect_err("a model-returned encoder diagnostic must match its hidden state");
    let expected = Error::shape(
        "beam_search.encoder_output.hidden_states[0]",
        "[2, 2, 1]",
        "[2, 2, 2]",
    );
    assert_eq!(format!("{error:?}"), format!("{expected:?}"));
    let trace = model.trace.lock().expect("trace");
    assert_eq!(
        trace.boundaries,
        ["encode", "cache", "prefill", "reorder", "decode"]
    );
    assert_eq!(trace.reorders, [vec![0, 0]]);
    assert_eq!(trace.decode_tokens, [vec![1, 2]]);
    assert_eq!(trace.decode_valid, [vec![true, true]]);
    assert_eq!(trace.decode_lengths, [vec![2, 2]]);
}

/// An all-EOS vocabulary terminates without live successors even in non-eager mode.
#[test]
fn all_eos_frontier_and_length_boundary_preserve_eos_reason() {
    let device = Default::default();
    for maximum in [1, 4] {
        let model = ScriptedModel::<TestBackend>::new(
            ScriptedConfig {
                vocabulary_size: 5,
                emit_nonfinite: false,
            },
            &device,
        );
        let mut config = search(maximum, 2);
        config.eos_token_ids = vec![0, 1, 2, 3, 4];
        let output = generate_seq2seq_beam_search(&model, request(&device, config))
            .expect("all finite children finish");
        for (sequence, id) in output.sequences[0].iter().zip([1, 2]) {
            assert_eq!(sequence.token_ids, [id]);
            assert_eq!(
                sequence.finish_reason,
                FinishReason::EndOfSequence { token_id: id }
            );
        }
        let trace = model.trace.lock().expect("trace");
        assert!(trace.reorders.is_empty() && trace.decode_tokens.is_empty());
    }
}

/// Encoder diagnostics and all input metadata follow duplicate/reordered beam rows.
#[test]
fn beam_gathers_preserve_masks_positions_and_optional_diagnostics() {
    let device = Default::default();
    let hidden = Tensor::<TestBackend, 3>::from_floats([[[1.0], [2.0]], [[3.0], [4.0]]], &device);
    let mask = TokenMask::new(Tensor::from_bool(
        [[true, false], [false, true]].into(),
        &device,
    ));
    let encoder = EncoderOutput {
        hidden_state: hidden.clone(),
        token_mask: Some(mask.clone()),
        hidden_states: vec![hidden.mul_scalar(10.0)],
        attentions: vec![Tensor::from_data(
            TensorData::new((0..8).map(|v| v as f32).collect(), [2, 1, 2, 2]),
            &device,
        )],
    };
    super::validate_encoder_batch(&encoder, 2).expect("valid source diagnostics");
    let gathered = super::select_encoder(encoder, &[1, 0, 1]).expect("beam gather");
    super::validate_encoder_batch(&gathered, 3).expect("gathered diagnostic geometry");
    assert_eq!(
        gathered.hidden_state.into_data().to_vec::<f32>().unwrap(),
        [3.0, 4.0, 1.0, 2.0, 3.0, 4.0]
    );
    assert_eq!(
        gathered.hidden_states[0].to_data().to_vec::<f32>().unwrap(),
        [30.0, 40.0, 10.0, 20.0, 30.0, 40.0]
    );
    assert_eq!(
        gathered.attentions[0].to_data().to_vec::<f32>().unwrap(),
        [4.0, 5.0, 6.0, 7.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    );
    assert_eq!(
        mask_flags(gathered.token_mask.as_ref(), 3, 2).unwrap(),
        [false, true, true, false, false, true]
    );
    let input = CausalLmInput {
        token_ids: Tensor::from_ints([[1, 2], [3, 4]], &device),
        token_mask: Some(mask),
        position_ids: Some(Tensor::from_ints([[7, 8], [9, 10]], &device)),
    };
    let gathered = super::select_input(input, &[1, 0, 1]).expect("input gather");
    assert_eq!(
        metadata::tensor_i64_values(gathered.token_ids, "tokens").unwrap(),
        [3, 4, 1, 2, 3, 4]
    );
    assert_eq!(
        metadata::tensor_i64_values(gathered.position_ids.unwrap(), "positions").unwrap(),
        [9, 10, 7, 8, 9, 10]
    );
    assert_eq!(
        mask_flags(gathered.token_mask.as_ref(), 3, 2).unwrap(),
        [false, true, true, false, false, true]
    );
}

/// Every optional encoder rectangle is checked before backend gathers can panic.
#[test]
fn encoder_and_logit_geometry_helpers_reject_ragged_outputs() {
    let device = Default::default();
    let baseline = EncoderOutput::<TestBackend> {
        hidden_state: Tensor::zeros([2, 3, 4], &device),
        token_mask: None,
        hidden_states: Vec::new(),
        attentions: Vec::new(),
    };
    for shape in [[1, 3, 4], [2, 0, 4], [2, 3, 0]] {
        let mut encoder = baseline.clone();
        encoder.hidden_state = Tensor::zeros(shape, &device);
        assert!(super::validate_encoder_batch(&encoder, 2).is_err());
    }
    let mut encoder = baseline.clone();
    encoder.token_mask = Some(TokenMask::new(Tensor::from_bool([[true]].into(), &device)));
    assert!(
        super::validate_encoder_batch(&encoder, 2)
            .unwrap_err()
            .to_string()
            .contains("token_mask")
    );
    let mut encoder = baseline.clone();
    encoder
        .hidden_states
        .push(Tensor::zeros([2, 3, 5], &device));
    assert!(
        super::validate_encoder_batch(&encoder, 2)
            .unwrap_err()
            .to_string()
            .contains("hidden_states[0]")
    );
    for shape in [[1, 1, 3, 3], [2, 0, 3, 3], [2, 1, 2, 3], [2, 1, 3, 2]] {
        let mut encoder = baseline.clone();
        encoder.attentions.push(Tensor::zeros(shape, &device));
        assert!(
            super::validate_encoder_batch(&encoder, 2)
                .unwrap_err()
                .to_string()
                .contains("attentions[0]")
        );
    }
    for shape in [[1, 1, 5], [2, 2, 5], [2, 1, 4]] {
        assert!(
            metadata::extract_last_logits(
                Tensor::<TestBackend, 3>::zeros(shape, &device),
                2,
                5,
                "beam_search.model_logits"
            )
            .is_err()
        );
    }
    assert!(
        super::host_log_probabilities(
            Tensor::<TestBackend, 2>::zeros([2, 5], &device),
            &[true],
            2,
            5
        )
        .is_err()
    );
    assert!(
        super::host_log_probabilities(
            Tensor::<TestBackend, 2>::zeros([2, 4], &device),
            &[true, true],
            2,
            5
        )
        .is_err()
    );
    let inactive_nan = Tensor::<TestBackend, 2>::from_floats([[f32::NAN, f32::INFINITY]], &device);
    let uniform = super::host_log_probabilities(inactive_nan, &[false], 1, 2)
        .expect("inactive scores are inert");
    assert!(
        uniform
            .iter()
            .all(|value| (value + std::f64::consts::LN_2).abs() < 1e-6)
    );
}

/// Ragged masks, integer overflow, and portable position limits return errors.
#[test]
fn position_mask_and_expansion_boundaries_are_checked() {
    #[cfg(target_pointer_width = "64")]
    let device = Default::default();
    assert_eq!(
        metadata::next_positions(
            &[-99, 5, 9, 100, -9, 7],
            &[false, true, true, true, false, true],
            2,
            3,
            2,
            "decoder prompt"
        )
        .unwrap(),
        [10, 8]
    );
    // Two new tokens feed back one: positions up to `i32::MAX - 1 + 1` fit.
    assert_eq!(
        metadata::next_positions(
            &[i64::from(i32::MAX) - 1],
            &[true],
            1,
            1,
            2,
            "decoder prompt"
        )
        .unwrap(),
        [i64::from(i32::MAX)]
    );
    for (positions, valid, batch, width, maximum) in [
        (vec![0], vec![true, true], 1, 2, 1),
        (vec![0], vec![false], 1, 1, 1),
        (vec![-1], vec![true], 1, 1, 1),
        (vec![i64::MAX], vec![true], 1, 1, 1),
        (vec![i64::from(i32::MAX)], vec![true], 1, 1, 2),
        (vec![], vec![], usize::MAX, 2, 1),
        (vec![0], vec![true], 1, 1, usize::MAX),
    ] {
        assert!(
            metadata::next_positions(&positions, &valid, batch, width, maximum, "decoder prompt")
                .is_err()
        );
    }
    assert_eq!(
        metadata::nonempty_row_lengths(&[true, false, true, true], 2, 2, "mask").unwrap(),
        [1, 2]
    );
    assert!(metadata::nonempty_row_lengths(&[true], 1, 2, "mask").is_err());
    assert!(metadata::nonempty_row_lengths(&[true, false, false, false], 2, 2, "mask").is_err());
    assert!(metadata::nonempty_row_lengths(&[], usize::MAX, 2, "mask").is_err());
    assert!(mask_flags::<TestBackend>(None, usize::MAX, 2).is_err());
    assert_eq!(
        super::beam_expansion_indices(2, 3).unwrap(),
        [0, 0, 0, 1, 1, 1]
    );
    assert!(super::beam_expansion_indices(usize::MAX, 2).is_err());
    assert!(super::beam_expansion_indices(1, usize::MAX).is_err());
    #[cfg(target_pointer_width = "64")]
    assert!(matches!(
        crate::models::shared::rows::index_tensor::<TestBackend>(
            &[usize::MAX],
            &device,
            super::BEAM_ROWS
        ),
        Err(Error::Configuration { field, .. }) if field == super::BEAM_ROWS
    ));
    assert!(super::empty_completed(usize::MAX).is_err());
    #[cfg(target_pointer_width = "64")]
    assert!(
        BeamSearchConfig::default()
            .validate(u32::MAX as usize + 2)
            .is_err()
    );
    assert!(
        BeamSearchConfig {
            length_penalty: -f64::MAX,
            ..Default::default()
        }
        .validate(5)
        .is_err()
    );
}

/// Masked decoder gaps and finished source rows preserve independent positions and lengths.
#[test]
fn ragged_positioned_beams_keep_finished_cache_rows_inert() {
    let device = Default::default();
    let model = ScriptedModel::<TestBackend>::new(
        ScriptedConfig {
            vocabulary_size: 5,
            emit_nonfinite: false,
        },
        &device,
    );
    let mut config = search(3, 1);
    config.eos_token_ids = vec![1];
    config.early_stopping = EarlyStopping::Eager;
    let request = Seq2SeqBeamSearchRequest {
        source: CausalLmInput {
            token_ids: Tensor::from_ints([[4, 0], [3, 0]], &device),
            token_mask: Some(TokenMask::new(Tensor::from_bool(
                [[true, false], [true, false]].into(),
                &device,
            ))),
            position_ids: None,
        },
        decoder_prompt: CausalLmInput {
            token_ids: Tensor::from_ints([[0, 99, 0], [99, 0, 99]], &device),
            token_mask: Some(TokenMask::new(Tensor::from_bool(
                [[true, false, true], [false, true, false]].into(),
                &device,
            ))),
            position_ids: Some(Tensor::from_ints([[5, -99, 7], [-99, 10, -99]], &device)),
        },
        cache: CacheConfig::new(2, 5).unwrap(),
        search: config,
        vocabulary_size: 5,
    };
    let output = generate_seq2seq_beam_search(&model, request).expect("positioned padded beams");
    assert_eq!(output.sequences[0][0].decoder_prompt_length, 2);
    assert_eq!(output.sequences[1][0].decoder_prompt_length, 1);
    assert_eq!(output.sequences[0][0].token_ids, [1]);
    assert_eq!(output.sequences[1][0].token_ids, [3, 3, 3]);
    let trace = model.trace.lock().expect("trace");
    assert_eq!(trace.decode_valid, [vec![false, true], vec![false, true]]);
    assert_eq!(
        trace.decode_positions,
        [Some(vec![0, 11]), Some(vec![0, 12])]
    );
    assert_eq!(trace.decode_lengths, [vec![2, 2], vec![2, 3]]);
}

/// Builds a finite non-empty hypothesis with consistent per-token score accounting.
fn scored_hypothesis(tokens: &[u32], cumulative: f64, ordinal: usize) -> BeamHypothesis {
    BeamHypothesis {
        token_ids: tokens.to_vec(),
        log_probabilities: vec![cumulative / tokens.len() as f64; tokens.len()],
        cumulative_log_probability: cumulative,
        next_position: None,
        finish_reason: Some(FinishReason::Length),
        ordinal,
    }
}

/// Candidate ordering ignores dormant parents and forbids non-finite cumulative scores.
#[test]
fn candidate_frontiers_validate_scores_geometry_and_stable_ties() {
    let mut parent = scored_hypothesis(&[4], 0.0, 0);
    parent.finish_reason = None;
    let active = [Some(parent.clone()), Some(parent.clone())];
    let candidates = super::candidates_for_source(
        0,
        2,
        3,
        8,
        &active,
        &[-0.2, -0.2, f64::NEG_INFINITY, -0.2, -0.1, -0.2],
    )
    .unwrap();
    assert_eq!(
        candidates
            .iter()
            .map(|candidate| (candidate.parent_slot, candidate.token_id))
            .collect::<Vec<_>>(),
        [(1, 1), (0, 0), (0, 1), (1, 0), (1, 2)]
    );
    // A narrower frontier keeps exactly the best-ranked prefix.
    let narrow = super::candidates_for_source(
        0,
        2,
        3,
        2,
        &active,
        &[-0.2, -0.2, f64::NEG_INFINITY, -0.2, -0.1, -0.2],
    )
    .unwrap();
    assert_eq!(
        narrow
            .iter()
            .map(|candidate| (candidate.parent_slot, candidate.token_id))
            .collect::<Vec<_>>(),
        [(1, 1), (0, 0)]
    );
    assert!(
        super::candidates_for_source(0, 1, 3, 2, &[None], &[f64::NAN; 3])
            .unwrap()
            .is_empty()
    );
    assert!(
        super::candidates_for_source(0, 1, 3, 2, &active, &[f64::NEG_INFINITY; 3])
            .unwrap()
            .is_empty()
    );
    assert!(super::candidates_for_source(0, 1, 3, 2, &active, &[0.0, 0.0]).is_err());
    for invalid in [f64::NAN, f64::INFINITY] {
        assert!(super::candidates_for_source(0, 1, 1, 2, &active, &[invalid]).is_err());
    }
    parent.cumulative_log_probability = -f64::MAX;
    assert!(
        super::candidates_for_source(0, 1, 1, 2, &[Some(parent)], &[-f64::MAX])
            .unwrap_err()
            .to_string()
            .contains("overflowed cumulative")
    );
    assert!(super::candidates_for_source(usize::MAX, 2, 1, 2, &active, &[]).is_err());
    assert!(super::candidates_for_source(1, 1, usize::MAX, 2, &active, &[]).is_err());
    let three = [active[0].clone(), active[1].clone(), active[0].clone()];
    assert!(super::candidates_for_source(2, 1, usize::MAX, 2, &three, &[]).is_err());
}

/// Candidate extension is independent of its parent and fails before changing history.
#[test]
fn extending_candidates_preserves_parent_and_checks_positions() {
    let mut parent = scored_hypothesis(&[1], -0.5, 0);
    parent.finish_reason = None;
    parent.next_position = Some(10);
    let candidate = super::Candidate {
        parent_global: 0,
        parent_slot: 0,
        token_id: 2,
        log_probability: -0.25,
        cumulative_log_probability: -0.75,
    };
    let active = [Some(parent.clone())];
    let reason = FinishReason::EndOfSequence { token_id: 2 };
    let child = super::extend_candidate(&active, &candidate, Some(reason.clone()), 9).unwrap();
    assert_eq!(child.token_ids, [1, 2]);
    assert_eq!(child.log_probabilities, [-0.5, -0.25]);
    assert_eq!(child.next_position, Some(11));
    assert_eq!(child.finish_reason, Some(reason));
    assert_eq!(child.ordinal, 9);
    assert_eq!(active[0].as_ref().unwrap().token_ids, [1]);
    assert!(super::extend_candidate(&[], &candidate, None, 0).is_err());
    assert!(super::extend_candidate(&[None], &candidate, None, 0).is_err());
    parent.next_position = Some(i64::MAX);
    assert!(super::extend_candidate(&[Some(parent)], &candidate, None, 0).is_err());
}

/// Finished ordering follows score, raw probability, lexical IDs, then creation order.
#[test]
fn finished_output_applies_tie_breakers_and_rejects_incomplete_pools() {
    let short = scored_hypothesis(&[9], -1.0, 0);
    let long = scored_hypothesis(&[1, 2], -2.0, 1);
    assert!(super::completed_order(&short, &long, 1.0).is_lt());
    let lex_first = scored_hypothesis(&[1], -1.0, 9);
    assert!(super::completed_order(&lex_first, &short, 1.0).is_lt());
    let earlier = scored_hypothesis(&[1], -1.0, 2);
    assert!(super::completed_order(&earlier, &lex_first, 1.0).is_lt());
    let mut config = search(2, 3);
    config.num_return_sequences = 2;
    let output =
        super::finish_output(vec![vec![short.clone(), lex_first, earlier]], &[7], &config).unwrap();
    assert_eq!(output.sequences[0].len(), 2);
    for sequence in &output.sequences[0] {
        assert_eq!(sequence.token_ids, [1]);
        assert_eq!(sequence.decoder_prompt_length, 7);
        assert_eq!(sequence.score, -1.0);
        assert_eq!(
            sequence.cumulative_log_probability,
            sequence.log_probabilities.iter().sum::<f64>()
        );
    }
    assert!(super::finish_output(Vec::new(), &[7], &config).is_err());
    assert!(super::finish_output(vec![vec![short.clone()]], &[7], &config).is_err());
    config.num_return_sequences = 1;
    let mut unfinished = short;
    unfinished.finish_reason = None;
    assert!(super::finish_output(vec![vec![unfinished]], &[7], &config).is_err());
}

/// Finite but extreme normalization factors must not escape as infinite public scores.
#[test]
fn extreme_length_penalty_returns_contextual_error_instead_of_infinite_score() {
    let device = Default::default();
    let model = ScriptedModel::<TestBackend>::new(
        ScriptedConfig {
            vocabulary_size: 5,
            emit_nonfinite: false,
        },
        &device,
    );
    let mut config = search(2, 1);
    config.length_penalty = -1070.0;
    config
        .validate(5)
        .expect("normalizer itself is finite and nonzero");
    let error = generate_seq2seq_beam_search(&model, request(&device, config))
        .expect_err("a non-finite normalized output score is not a valid result");
    assert!(error.to_string().contains("non-finite"));
}

/// A one-beam request is the deterministic greedy boundary case.
#[test]
fn one_beam_returns_one_length_limited_sequence() {
    let device = Default::default();
    let model = ScriptedModel::<TestBackend>::new(
        ScriptedConfig {
            vocabulary_size: 5,
            emit_nonfinite: false,
        },
        &device,
    );
    let output = generate_seq2seq_beam_search(&model, request(&device, search(2, 1)))
        .expect("one-beam search");
    assert_eq!(output.sequences[0].len(), 1);
    assert_eq!(output.sequences[0][0].token_ids.len(), 2);
    assert_eq!(output.sequences[0][0].finish_reason, FinishReason::Length);
}

/// A finite frontier narrower than the beam width leaves unused cache rows as PAD no-ops.
#[test]
fn release_v2_restricted_vocabulary_keeps_unused_beam_slots_inert() {
    let device = Default::default();
    let mut model = ScriptedModel::<TestBackend>::new(
        ScriptedConfig {
            vocabulary_size: 5,
            emit_nonfinite: false,
        },
        &device,
    );
    model.vocabulary_mask = Some(vec![false, false, false, true, false]);
    let mut config = search(3, 3);
    config.num_return_sequences = 1;
    let output = generate_seq2seq_beam_search(&model, request(&device, config)).unwrap();
    assert_eq!(output.sequences.len(), 1);
    assert_eq!(output.sequences[0].len(), 1);
    let sequence = &output.sequences[0][0];
    assert_eq!(sequence.token_ids, [3, 3, 3]);
    assert_eq!(sequence.log_probabilities, [0.0; 3]);
    assert_eq!(sequence.cumulative_log_probability, 0.0);
    assert_eq!(sequence.score, 0.0);
    assert_eq!(sequence.decoder_prompt_length, 1);
    assert_eq!(sequence.finish_reason, FinishReason::Length);
    let trace = model.trace.lock().unwrap();
    assert_eq!(trace.encode_calls, 1);
    assert_eq!(trace.state_allocations, 1);
    assert_eq!(trace.prefill_calls, 1);
    assert_eq!(trace.reorders, [vec![0, 1, 2], vec![0, 1, 2]]);
    assert_eq!(trace.decode_tokens, [vec![3, 0, 0], vec![3, 0, 0]]);
    assert_eq!(
        trace.decode_valid,
        [vec![true, false, false], vec![true, false, false]]
    );
    assert_eq!(trace.decode_lengths, [vec![2, 1, 1], vec![3, 1, 1]]);
}

/// Too few reachable terminal hypotheses produce an error instead of cloned or fabricated outputs.
#[test]
fn release_v2_restricted_vocabulary_reports_insufficient_return_hypotheses() {
    let device = Default::default();
    let mut model = ScriptedModel::<TestBackend>::new(
        ScriptedConfig {
            vocabulary_size: 5,
            emit_nonfinite: false,
        },
        &device,
    );
    model.vocabulary_mask = Some(vec![false, false, false, true, false]);
    let error = generate_seq2seq_beam_search(&model, request(&device, search(2, 3)))
        .expect_err("one reachable sequence cannot supply three distinct hypotheses");
    assert!(matches!(error, Error::Generation(ref message)
        if message == "beam-search source row 0 produced 1 finished hypotheses, fewer than requested 3"));
    let trace = model.trace.lock().unwrap();
    assert_eq!(
        trace.boundaries,
        ["encode", "cache", "prefill", "reorder", "decode"]
    );
    assert_eq!(trace.decode_tokens, [vec![3, 0, 0]]);
    assert_eq!(trace.decode_valid, [vec![true, false, false]]);
    assert_eq!(trace.decode_lengths, [vec![2, 1, 1]]);
}

/// Builds a Markov-table model and a request with one source per first-step row.
///
/// Source `m` has marker token `m`, so it receives `table.first[m]`.
fn markov_fixture(
    table: MarkovTable,
    search: BeamSearchConfig,
) -> (
    ScriptedModel<TestBackend>,
    Seq2SeqBeamSearchRequest<TestBackend>,
) {
    let device = Default::default();
    let vocabulary_size = table.first[0].len();
    let sources = table.first.len();
    let mut model = ScriptedModel::<TestBackend>::new(
        ScriptedConfig {
            vocabulary_size,
            emit_nonfinite: false,
        },
        &device,
    );
    model.table = Some(table);
    let mut request = request(&device, search);
    request.vocabulary_size = vocabulary_size;
    if sources > 1 {
        let markers = (0..sources as i64).flat_map(|marker| [marker, 0]).collect();
        request.source.token_ids =
            Tensor::from_data(TensorData::new(markers, [sources, 2]), &device);
        request.decoder_prompt.token_ids = Tensor::zeros([sources, 1], &device);
        request.cache = CacheConfig::new(
            sources * request.search.num_beams,
            request.search.max_new_tokens + 1,
        )
        .expect("multi-source cache");
    }
    (model, request)
}

/// A settled finished pool stops the source without changing the result.
#[test]
fn settled_pools_stop_early_with_identical_results() {
    // EOS (token 2) dominates the first step; any live extension already
    // scores below it, so nothing can enter the one-beam pool afterwards.
    let table = MarkovTable {
        first: vec![vec![0.0, 0.0, 5.0]],
        next: vec![vec![0.0, 0.0, 0.0]; 3],
    };
    let mut config = search(10, 1);
    config.num_return_sequences = 1;
    config.eos_token_ids = vec![2];
    let (pruned_model, request) = markov_fixture(table.clone(), config.clone());
    let pruned = super::run_beam_search(&pruned_model, request.clone(), true).unwrap();
    let (exhaustive_model, request) = markov_fixture(table, config);
    let exhaustive = super::run_beam_search(&exhaustive_model, request, false).unwrap();
    assert_eq!(pruned, exhaustive);
    assert_eq!(pruned.sequences[0][0].token_ids, [2]);
    assert!(pruned_model.trace.lock().unwrap().decode_tokens.is_empty());
    assert_eq!(
        exhaustive_model.trace.lock().unwrap().decode_tokens.len(),
        9
    );
}

/// The pool-finality bound is exact at its endpoints for every penalty sign.
#[test]
fn finished_pool_finality_uses_the_length_endpoints() {
    let pool = |score: f64, length: usize| {
        vec![scored_hypothesis(
            &vec![1; length],
            score * length as f64,
            0,
        )]
    };
    // Positive penalty: c = -2 reaches at best -2 / 4 = -0.5 at the longest length.
    assert!(!super::finished_pool_is_final(
        &pool(-0.6, 1),
        1,
        -2.0,
        2,
        4,
        1.0
    ));
    assert!(super::finished_pool_is_final(
        &pool(-0.4, 1),
        1,
        -2.0,
        2,
        4,
        1.0
    ));
    // Equal scores could still win a tie breaker, so they do not settle the pool.
    assert!(!super::finished_pool_is_final(
        &pool(-0.5, 1),
        1,
        -2.0,
        2,
        4,
        1.0
    ));
    // Zero penalty compares raw cumulative scores.
    assert!(super::finished_pool_is_final(
        &[scored_hypothesis(&[1], -1.5, 0)],
        1,
        -2.0,
        2,
        4,
        0.0
    ));
    // Negative penalty multiplies by the length, so the shortest length is best.
    assert!(super::finished_pool_is_final(
        &[scored_hypothesis(&[1], -3.5, 0)],
        1,
        -2.0,
        2,
        4,
        -1.0
    ));
    assert!(!super::finished_pool_is_final(
        &[scored_hypothesis(&[1], -4.5, 0)],
        1,
        -2.0,
        2,
        4,
        -1.0
    ));
    // An incomplete pool or an exhausted length budget never settles early.
    assert!(!super::finished_pool_is_final(
        &pool(-0.4, 1),
        2,
        -2.0,
        2,
        4,
        1.0
    ));
    assert!(!super::finished_pool_is_final(
        &pool(-0.4, 1),
        1,
        -2.0,
        5,
        4,
        1.0
    ));
}

/// Whether every reorder index stays inside its destination's group of `beams` rows.
fn preserves_groups(parent_indices: &[usize], beams: usize) -> bool {
    parent_indices
        .iter()
        .enumerate()
        .all(|(destination, &parent)| parent / beams == destination / beams)
}

/// Destinations and parents built from beam rows never leave their source group.
#[test]
fn reorder_group_check_detects_cross_source_parents() {
    assert!(preserves_groups(&[0, 0, 3, 2], 2));
    assert!(!preserves_groups(&[0, 2, 2, 3], 2));
}

/// Every decode step receives the once-expanded encoder rows, unchanged.
#[test]
fn encoder_rows_are_expanded_once_and_never_regathered() {
    let device = Default::default();
    let model = ScriptedModel::<TestBackend>::new(
        ScriptedConfig {
            vocabulary_size: 5,
            emit_nonfinite: false,
        },
        &device,
    );
    let mut config = search(3, 2);
    config.num_return_sequences = 1;
    let request = Seq2SeqBeamSearchRequest {
        source: CausalLmInput {
            token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[4, 0], [3, 1]], &device),
            token_mask: None,
            position_ids: None,
        },
        decoder_prompt: CausalLmInput {
            token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[0], [0]], &device),
            token_mask: None,
            position_ids: None,
        },
        cache: CacheConfig::new(4, 3).unwrap(),
        search: config,
        vocabulary_size: 5,
    };
    let output = generate_seq2seq_beam_search(&model, request).expect("two-source search");
    assert_eq!(output.sequences.len(), 2);
    let trace = model.trace.lock().unwrap();
    assert_eq!(trace.decode_tokens.len(), 2);
    // Parent indices always stay inside their two-row source group.
    for reorder in &trace.reorders {
        assert!(preserves_groups(reorder, 2), "{reorder:?}");
    }
}

/// Full stable sort followed by truncation: the reference the bounded heap must equal.
fn reference_candidates(
    source: usize,
    beams: usize,
    vocabulary: usize,
    capacity: usize,
    active: &[Option<BeamHypothesis>],
    probabilities: &[f64],
) -> Vec<(usize, u32)> {
    let mut candidates = Vec::new();
    for slot in 0..beams {
        let row = source * beams + slot;
        let Some(parent) = &active[row] else {
            continue;
        };
        for (token, &value) in probabilities[row * vocabulary..(row + 1) * vocabulary]
            .iter()
            .enumerate()
        {
            if value.is_finite() {
                candidates.push(super::Candidate {
                    parent_global: row,
                    parent_slot: slot,
                    token_id: token as u32,
                    log_probability: value,
                    cumulative_log_probability: parent.cumulative_log_probability + value,
                });
            }
        }
    }
    candidates.sort_by(super::candidate_order);
    candidates.truncate(capacity);
    candidates
        .into_iter()
        .map(|candidate| (candidate.parent_global, candidate.token_id))
        .collect()
}

proptest::proptest! {
    #![proptest_config(proptest::test_runner::Config::with_cases(64))]

    /// The bounded heap returns exactly the prefix of a full stable sort, ties included.
    #[test]
    fn bounded_frontier_matches_full_sort_prefix(
        beams in 1_usize..4,
        vocabulary in 1_usize..8,
        capacity in 1_usize..12,
        parents in proptest::collection::vec((proptest::bool::ANY, 0_usize..3), 4),
        values in proptest::collection::vec(0_usize..4, 32),
    ) {
        let levels = [-0.5, -1.0, -2.0, f64::NEG_INFINITY];
        let active = (0..beams)
            .map(|slot| {
                let (present, score) = parents[slot];
                present.then(|| {
                    let mut parent = scored_hypothesis(&[1], -(score as f64), slot);
                    parent.finish_reason = None;
                    parent
                })
            })
            .collect::<Vec<_>>();
        let probabilities = (0..beams * vocabulary)
            .map(|index| levels[values[index % values.len()]])
            .collect::<Vec<_>>();
        let actual = super::candidates_for_source(0, beams, vocabulary, capacity, &active, &probabilities)
            .unwrap()
            .into_iter()
            .map(|candidate| (candidate.parent_global, candidate.token_id))
            .collect::<Vec<_>>();
        proptest::prop_assert_eq!(
            actual,
            reference_candidates(0, beams, vocabulary, capacity, &active, &probabilities)
        );
    }

    /// Pruning settled sources never changes any returned hypothesis, under every stopping rule.
    ///
    /// Every rank up to `num_beams` is compared, and with two sources one
    /// may be pruned while the other keeps decoding through PAD no-op rows.
    #[test]
    fn pruned_search_equals_exhaustive_search(
        vocabulary in 3_usize..6,
        beams in 1_usize..4,
        sources in 1_usize..3,
        max_new_tokens in 1_usize..6,
        eos_flags in proptest::collection::vec(proptest::bool::ANY, 6),
        penalty in proptest::sample::select(vec![-1.0, 0.0, 0.5, 1.0, 2.0]),
        early_stopping in proptest::sample::select(vec![
            EarlyStopping::Eager,
            EarlyStopping::Heuristic,
            EarlyStopping::Never,
        ]),
        logits in proptest::collection::vec(-3.0_f32..3.0, 48),
    ) {
        let beams = beams.min(vocabulary);
        let row = |offset: usize| logits[offset..offset + vocabulary].to_vec();
        let table = MarkovTable {
            first: (0..sources).map(|source| row(6 * source)).collect(),
            next: (0..vocabulary).map(|token| row(12 + 6 * token)).collect(),
        };
        // `search` returns every hypothesis of each source's finished pool.
        let mut config = search(max_new_tokens, beams);
        config.length_penalty = penalty;
        config.early_stopping = early_stopping;
        config.eos_token_ids = (0..vocabulary as u32)
            .filter(|&token| eos_flags[token as usize])
            .collect();
        let (pruned_model, request) = markov_fixture(table.clone(), config.clone());
        let pruned = super::run_beam_search(&pruned_model, request.clone(), true);
        let (exhaustive_model, request) = markov_fixture(table, config);
        let exhaustive = super::run_beam_search(&exhaustive_model, request, false);
        match (pruned, exhaustive) {
            (Ok(pruned), Ok(exhaustive)) => proptest::prop_assert_eq!(pruned, exhaustive),
            (Err(pruned), Err(exhaustive)) => {
                proptest::prop_assert_eq!(pruned.to_string(), exhaustive.to_string())
            }
            (pruned, exhaustive) => proptest::prop_assert!(
                false,
                "pruning changed the outcome: {pruned:?} vs {exhaustive:?}"
            ),
        }
        let pruned_trace = pruned_model.trace.lock().unwrap();
        proptest::prop_assert!(
            pruned_trace.decode_tokens.len()
                <= exhaustive_model.trace.lock().unwrap().decode_tokens.len()
        );
        for reorder in &pruned_trace.reorders {
            proptest::prop_assert!(preserves_groups(reorder, beams), "{:?}", reorder);
        }
    }
}

/// One settled source is pruned to PAD no-op rows while its peer keeps decoding.
#[test]
fn settled_source_is_pruned_while_its_peer_keeps_decoding() {
    // Source 0 emits EOS (token 2) at once and through token 3 one step
    // later, which fills its two-hypothesis pool; its best live beam
    // (cumulative log-probability about -5.4) can no longer beat the worst
    // finished score (about -0.66), so the source settles after step 1.
    // Source 1 starts on token 0 and its live beams cycle 0 -> 1 -> 0
    // without reaching EOS, so it keeps decoding to the length limit.
    let table = MarkovTable {
        first: vec![vec![0.0, 0.0, 5.0, 4.0], vec![4.0, 1.0, -9.0, -9.0]],
        next: vec![
            vec![0.0, 3.0, -9.0, -9.0],
            vec![3.0, 0.0, -9.0, -9.0],
            vec![0.0, 0.0, 0.0, 0.0],
            vec![-9.0, -9.0, 5.0, -9.0],
        ],
    };
    let mut config = search(4, 2);
    config.eos_token_ids = vec![2];
    let (pruned_model, request) = markov_fixture(table.clone(), config.clone());
    let pruned = super::run_beam_search(&pruned_model, request.clone(), true).unwrap();
    let (exhaustive_model, request) = markov_fixture(table, config);
    let exhaustive = super::run_beam_search(&exhaustive_model, request, false).unwrap();
    assert_eq!(pruned, exhaustive);
    assert_eq!(pruned.sequences.len(), 2);
    assert_eq!(pruned.sequences[0].len(), 2);
    assert_eq!(pruned.sequences[0][0].token_ids, [2]);
    assert_eq!(pruned.sequences[0][1].token_ids, [3, 2]);
    assert!(
        pruned.sequences[1]
            .iter()
            .all(|sequence| sequence.finish_reason == FinishReason::Length)
    );

    let pruned_trace = pruned_model.trace.lock().unwrap();
    let exhaustive_trace = exhaustive_model.trace.lock().unwrap();
    assert_eq!(pruned_trace.decode_valid.len(), 3);
    assert_eq!(exhaustive_trace.decode_valid.len(), 3);
    // Rows 0..2 belong to source 0, rows 2..4 to source 1.
    let settled_step = pruned_trace
        .decode_valid
        .iter()
        .position(|valid| !valid[0] && !valid[1])
        .expect("source 0 is pruned");
    assert_eq!(settled_step, 1);
    for valid in &pruned_trace.decode_valid[settled_step..] {
        assert_eq!(valid[..2], [false, false], "{valid:?}");
        assert_eq!(valid[2..], [true, true], "{valid:?}");
    }
    assert!(
        exhaustive_trace.decode_valid[settled_step..]
            .iter()
            .any(|valid| valid[0] || valid[1]),
        "exhaustive search keeps decoding source 0"
    );
    for reorder in &pruned_trace.reorders {
        assert!(preserves_groups(reorder, 2), "{reorder:?}");
    }
}

/// Times the bounded-heap frontier against sorting every beam-vocabulary pair.
///
/// Run with `cargo test --release --lib beam_frontier_timing -- --ignored --nocapture`.
#[test]
#[ignore = "timing benchmark; run in release mode with --ignored --nocapture"]
fn beam_frontier_timing() {
    use std::time::Instant;

    let beams = 8;
    for vocabulary in [32_128, 250_112] {
        let capacity = 2 * beams;
        let active = (0..beams)
            .map(|slot| {
                let mut parent = scored_hypothesis(&[1], -(slot as f64) * 0.1, slot);
                parent.finish_reason = None;
                Some(parent)
            })
            .collect::<Vec<_>>();
        let mut state = 0x2545_f491_4f6c_dd1d_u64;
        let probabilities = (0..beams * vocabulary)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                -((state >> 11) as f64) / (1_u64 << 53) as f64 * 20.0
            })
            .collect::<Vec<_>>();
        let repetitions = 5;
        let started = Instant::now();
        let mut heap = Vec::new();
        for _ in 0..repetitions {
            heap = super::candidates_for_source(
                0,
                beams,
                vocabulary,
                capacity,
                &active,
                &probabilities,
            )
            .unwrap()
            .into_iter()
            .map(|candidate| (candidate.parent_global, candidate.token_id))
            .collect::<Vec<_>>();
        }
        let bounded = started.elapsed();
        let started = Instant::now();
        let mut sorted = Vec::new();
        for _ in 0..repetitions {
            sorted = reference_candidates(0, beams, vocabulary, capacity, &active, &probabilities);
        }
        let full = started.elapsed();
        assert_eq!(heap, sorted);
        println!(
            "beam frontier, {beams} beams x {vocabulary} tokens: full sort {:.2} ms, bounded heap {:.2} ms ({:.1}x)",
            full.as_secs_f64() * 1e3 / repetitions as f64,
            bounded.as_secs_f64() * 1e3 / repetitions as f64,
            full.as_secs_f64() / bounded.as_secs_f64()
        );
    }
}

/// `early_stopping` reads and writes exactly Transformers' `true`/`false`/`"never"` forms.
#[test]
fn early_stopping_serializes_like_transformers() {
    for (json, rule) in [
        ("true", EarlyStopping::Eager),
        ("false", EarlyStopping::Heuristic),
        ("\"never\"", EarlyStopping::Never),
    ] {
        assert_eq!(serde_json::from_str::<EarlyStopping>(json).unwrap(), rule);
        assert_eq!(serde_json::to_string(&rule).unwrap(), json);
    }
    for invalid in ["\"eager\"", "\"Never\"", "1", "null"] {
        let error = serde_json::from_str::<EarlyStopping>(invalid).unwrap_err();
        assert!(!error.to_string().is_empty(), "{invalid}");
    }
    let error = serde_json::from_str::<EarlyStopping>("\"always\"").unwrap_err();
    assert!(
        error.to_string().contains("true, false, or \"never\""),
        "{error}"
    );
    assert_eq!(EarlyStopping::from(true), EarlyStopping::Eager);
    assert_eq!(EarlyStopping::from(false), EarlyStopping::Heuristic);
    // Transformers' default is `early_stopping=False`, the heuristic.
    assert_eq!(EarlyStopping::default(), EarlyStopping::Heuristic);
    assert_eq!(
        BeamSearchConfig::default().early_stopping,
        EarlyStopping::Heuristic
    );
    // Serialized configurations written before the enum keep their meaning.
    let config: BeamSearchConfig =
        serde_json::from_str(r#"{"early_stopping": true, "num_beams": 2}"#).unwrap();
    assert_eq!(config.early_stopping, EarlyStopping::Eager);
    let config: BeamSearchConfig = serde_json::from_str(r#"{"early_stopping": "never"}"#).unwrap();
    assert_eq!(config.early_stopping, EarlyStopping::Never);
    let round_trip: BeamSearchConfig =
        serde_json::from_str(&serde_json::to_string(&config).unwrap()).unwrap();
    assert_eq!(round_trip, config);
}

/// The heuristic stops where Transformers does; `"never"` finds the better long hypothesis.
///
/// With length penalty one, the first step finishes EOS (log p = -0.474,
/// score -0.474) and keeps token 0 live (cumulative -0.974). Transformers'
/// default heuristic scores the live beam at its current length one,
/// `-0.974 <= -0.474`, and stops. Token 0 then continues almost for free,
/// so the length-four continuation scores `-0.974 / 4 = -0.244`, which the
/// exhaustive rule finds and the heuristic does not.
#[test]
fn heuristic_stopping_follows_transformers_and_never_searches_on() {
    let table = MarkovTable {
        first: vec![vec![0.0, -10.0, 0.5]],
        next: vec![vec![10.0, -10.0, 0.0]; 3],
    };
    let run = |early_stopping, prune| {
        let mut config = search(4, 1);
        config.num_return_sequences = 1;
        config.eos_token_ids = vec![2];
        config.early_stopping = early_stopping;
        let (model, request) = markov_fixture(table.clone(), config);
        let output = super::run_beam_search(&model, request, prune).unwrap();
        let decodes = model.trace.lock().unwrap().decode_tokens.len();
        (output.sequences[0][0].clone(), decodes)
    };
    let (heuristic, decodes) = run(EarlyStopping::Heuristic, true);
    assert_eq!(heuristic.token_ids, [2]);
    assert!(matches!(
        heuristic.finish_reason,
        FinishReason::EndOfSequence { token_id: 2 }
    ));
    assert_eq!(decodes, 0, "the heuristic settles after the first step");
    // The heuristic is a rule, not an optimization: pruning does not change it.
    assert_eq!(run(EarlyStopping::Heuristic, false), (heuristic, 0));
    let (never, decodes) = run(EarlyStopping::Never, true);
    assert_eq!(never.token_ids, [0, 0, 0, 0]);
    assert_eq!(never.finish_reason, FinishReason::Length);
    assert!(
        never.score > -0.25 && never.score < -0.24,
        "{}",
        never.score
    );
    assert_eq!(decodes, 3);
    // Eager stopping returns the first full pool, here the early EOS too.
    assert_eq!(run(EarlyStopping::Eager, true).0.token_ids, [2]);
}

/// The heuristic needs a full pool and stops when `c / n^lp <= w`, ties included.
#[test]
fn heuristic_finality_scores_the_live_beam_at_its_current_length() {
    // One finished hypothesis of length 2 with score -1.0 (cumulative -2.0, lp = 1).
    let pool = [scored_hypothesis(&[1, 2], -2.0, 0)];
    // c = -3 at length 3 scores -1.0: a tie stops, as Transformers' `>` test does.
    assert!(super::heuristic_pool_is_final(&pool, 1, -3.0, 3, 1.0));
    // c = -2.9 at length 3 scores -0.967 > -1.0: the live beam may still win.
    assert!(!super::heuristic_pool_is_final(&pool, 1, -2.9, 3, 1.0));
    // A pool that is not full never stops the search.
    assert!(!super::heuristic_pool_is_final(&pool, 2, -100.0, 3, 1.0));
    assert!(!super::heuristic_pool_is_final(&[], 1, -100.0, 3, 1.0));
    // Zero penalty compares raw cumulative scores; the worst score is -2.
    assert!(super::heuristic_pool_is_final(&pool, 1, -2.0, 7, 0.0));
    assert!(!super::heuristic_pool_is_final(&pool, 1, -1.5, 7, 0.0));
    // A negative penalty multiplies by the length: -1 * 3 = -3 <= -2 * 2 = -4 is false.
    assert!(!super::heuristic_pool_is_final(&pool, 1, -1.0, 3, -1.0));
    assert!(super::heuristic_pool_is_final(&pool, 1, -2.0, 3, -1.0));
    // No generated token yet: there is no length to score at.
    assert!(!super::heuristic_pool_is_final(&pool, 1, -100.0, 0, 1.0));
}

/// The default group-preserving reorder checks groups before delegating to `reorder`.
#[test]
fn default_group_preserving_reorder_validates_groups_atomically() {
    let device = Default::default();
    let trace = Arc::new(Mutex::new(ScriptTrace::default()));
    let mut state = ScriptedState::<TestBackend> {
        batch: 4,
        capacity: 4,
        lengths: vec![1, 2, 3, 4],
        device,
        trace: Arc::clone(&trace),
        fault: None,
    };
    state.reorder_within_groups(&[1, 1, 3, 2], 2).unwrap();
    assert_eq!(state.lengths(), [2, 2, 4, 3]);
    for (indices, group) in [
        (vec![2, 0, 3, 2], 2),
        (vec![0, 1, 2, 3], 3),
        (vec![0, 1, 2, 3], 0),
    ] {
        let error = state.reorder_within_groups(&indices, group).unwrap_err();
        assert!(matches!(error, Error::Cache(_)), "{error}");
    }
    assert_eq!(state.lengths(), [2, 2, 4, 3]);
    assert_eq!(trace.lock().unwrap().reorders, [vec![1, 1, 3, 2]]);
    // One group spanning the batch admits every permutation.
    state.reorder_within_groups(&[3, 2, 1, 0], 4).unwrap();
    assert_eq!(state.lengths(), [3, 4, 2, 2]);
}

/// Builds a scripted five-token model with no faults.
fn scripted_model(device: &TestDevice) -> ScriptedModel<TestBackend> {
    ScriptedModel::<TestBackend>::new(
        ScriptedConfig {
            vocabulary_size: 5,
            emit_nonfinite: false,
        },
        device,
    )
}

/// A source batch with no rows is rejected before the encoder runs.
#[test]
fn an_empty_source_batch_is_rejected_before_encoding() {
    let device = Default::default();
    let model = scripted_model(&device);
    let mut request = request(&device, search(2, 2));
    request.source.token_ids = Tensor::<TestBackend, 2, Int>::zeros([0, 2], &device);
    let error = generate_seq2seq_beam_search(&model, request).unwrap_err();
    assert!(
        matches!(&error, Error::Shape { tensor, actual, .. }
            if tensor == "beam_search.source" && actual == "[0, 2]"),
        "{error}"
    );
    assert!(model.trace.lock().unwrap().boundaries.is_empty());
}

/// A length bound whose cache requirement overflows `usize` fails before any model call.
#[test]
fn an_overflowing_cache_requirement_is_rejected_before_encoding() {
    let device = Default::default();
    let model = scripted_model(&device);
    let mut request = request(&device, search(2, 2));
    request.search.max_new_tokens = usize::MAX;
    request.decoder_prompt.token_ids = Tensor::<TestBackend, 2, Int>::zeros([1, 2], &device);
    let error = generate_seq2seq_beam_search(&model, request).unwrap_err();
    assert!(
        matches!(&error, Error::Generation(reason)
            if reason == "beam-search cache capacity overflows usize"),
        "{error}"
    );
    assert!(model.trace.lock().unwrap().boundaries.is_empty());
}

/// A source whose every token is excluded reports that it has no finite candidate.
///
/// All-`-inf` logits are a valid model output (every token masked), not a
/// numerical failure, so the diagnostic names the empty candidate set rather
/// than the `NaN` that a log-softmax of such a row would otherwise produce.
#[test]
fn a_fully_excluded_source_row_has_no_finite_candidate() {
    let device = Default::default();
    let mut model = scripted_model(&device);
    model.vocabulary_mask = Some(vec![false; 5]);
    let error = generate_seq2seq_beam_search(&model, request(&device, search(2, 2))).unwrap_err();
    assert!(
        matches!(&error, Error::Generation(reason)
            if reason == "beam-search source row 0 has no finite candidate"),
        "{error}"
    );
}

/// A fully excluded parent beam contributes no candidates while its sibling expands.
///
/// After the first step the two live beams end in tokens 1 and 2. Every
/// continuation of token 1 is excluded (`-inf`), so its row has no
/// distribution, but the source as a whole still has finite candidates: the
/// step succeeds and every returned hypothesis descends from token 2, as
/// Hugging Face's ranking puts `-inf` scores last.
#[test]
fn a_fully_excluded_parent_is_skipped_while_its_sibling_expands() {
    let inf = f32::NEG_INFINITY;
    let table = MarkovTable {
        first: vec![vec![inf, 1.0, 0.0, inf]],
        next: vec![
            vec![0.0; 4],
            vec![inf; 4],
            vec![0.0, 0.0, 0.0, 1.0],
            vec![0.0; 4],
        ],
    };
    let (model, request) = markov_fixture(table, search(2, 2));
    let output = generate_seq2seq_beam_search(&model, request).unwrap();
    let hypotheses = &output.sequences[0];
    assert_eq!(
        hypotheses
            .iter()
            .map(|hypothesis| hypothesis.token_ids.clone())
            .collect::<Vec<_>>(),
        [vec![2, 3], vec![2, 0]]
    );
    // log(e + 1) normalizes the first step; log(e + 3) the second.
    let first = -(1.0_f64.exp() + 1.0).ln();
    let second = -(1.0_f64.exp() + 3.0).ln();
    for (hypothesis, last) in hypotheses.iter().zip([1.0 + second, second]) {
        assert_eq!(hypothesis.finish_reason, FinishReason::Length);
        assert_eq!(hypothesis.log_probabilities.len(), 2);
        for (actual, expected) in hypothesis.log_probabilities.iter().zip([first, last]) {
            assert!((actual - expected).abs() < 1e-6, "{hypotheses:?}");
        }
    }
    // Both decoded parents were live: the excluded one was skipped, not dropped early.
    assert_eq!(model.trace.lock().unwrap().decode_tokens, [vec![1, 2]]);
}

/// Zero beams can never form a final finished pool, and cannot be expanded.
#[test]
fn zero_beam_pools_are_never_final_and_cannot_expand() {
    assert!(!super::finished_pool_is_final(&[], 0, -1.0, 1, 2, 1.0));
    assert!(!super::heuristic_pool_is_final(&[], 0, -1.0, 1, 1.0));
    let error = super::beam_expansion_indices(2, 0).unwrap_err();
    assert!(
        matches!(&error, Error::Generation(reason)
            if reason == "beam expansion requires at least one beam"),
        "{error}"
    );
    assert_eq!(
        super::beam_expansion_indices(2, 3).unwrap(),
        [0, 0, 0, 1, 1, 1]
    );
}

/// The frontier wrapper's equality agrees with the candidate total order.
#[test]
fn worst_first_equality_follows_the_candidate_order() {
    let candidate = |parent_slot, token_id, cumulative_log_probability| super::Candidate {
        parent_global: parent_slot,
        parent_slot,
        token_id,
        log_probability: cumulative_log_probability,
        cumulative_log_probability,
    };
    let first = super::WorstFirst(candidate(0, 1, -0.5));
    assert!(first == super::WorstFirst(candidate(0, 1, -0.5)));
    assert!(first != super::WorstFirst(candidate(0, 2, -0.5)));
    assert!(first != super::WorstFirst(candidate(1, 1, -0.5)));
    assert!(first != super::WorstFirst(candidate(0, 1, -0.25)));
}

/// Only rows whose every logit is `-inf` stay `-inf`; NaN still poisons its row.
#[test]
fn fully_excluded_rows_stay_negative_infinity_while_nan_rows_stay_nan() {
    let device = Default::default();
    let inf = f32::NEG_INFINITY;
    let logits = Tensor::<TestBackend, 2>::from_data(
        TensorData::new(
            vec![
                inf,
                inf,
                inf,
                f32::NAN,
                inf,
                inf,
                0.0,
                0.0,
                inf,
                7.0,
                7.0,
                7.0,
            ],
            [4, 3],
        ),
        &device,
    );
    let values = super::host_log_probabilities(logits, &[true, true, true, false], 4, 3).unwrap();
    assert_eq!(&values[0..3], [f64::NEG_INFINITY; 3]);
    assert!(values[3].is_nan(), "{values:?}");
    let half = (0.5_f32).ln();
    assert_eq!(
        &values[6..9],
        [f64::from(half), f64::from(half), f64::NEG_INFINITY]
    );
    // The inactive row is zeroed before normalization: a uniform distribution.
    let third = f64::from((1.0_f32 / 3.0).ln());
    for value in &values[9..12] {
        assert!((value - third).abs() < 1e-6, "{values:?}");
    }
}
