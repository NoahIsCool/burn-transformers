//! Allocation preflight and fixed-size accounting for modular decode histories.

use burn::backend::Flex;

use super::*;
use crate::{
    CausalLanguageModel, Seq2SeqLanguageModel,
    architecture::{
        Activation, AttentionOptions, AttentionSpec, DecoderLayerSpec, DenseFeedForwardSpec,
        FeedForwardSpec, MixerSpec, NormPlacement, NormalizationSpec, PositionEncodingSpec,
        ResidualTopology, Seq2SeqSpec,
    },
    models::shared::{implicit_position_ids, rows::CompactRow},
};

/// CPU integer backend used to ensure invalid requests fail before tensor allocation.
type B = Flex<f32, i32>;

/// Builds tiny parameters with the largest portable position range and no position table.
fn spec() -> DecoderSpec {
    let norm = NormalizationSpec::Rms {
        epsilon: 1e-6,
        zero_centered: false,
    };
    DecoderSpec {
        vocabulary_size: 4,
        hidden_size: 2,
        layers: vec![DecoderLayerSpec {
            mixer: MixerSpec::Attention(AttentionSpec {
                query_heads: 1,
                key_value_heads: 1,
                head_dim: 2,
                query_key_norm: false,
                output_gate: false,
                position: PositionEncodingSpec::None,
            }),
            feed_forward: FeedForwardSpec::Dense(DenseFeedForwardSpec {
                intermediate_size: 2,
                activation: Activation::Silu,
                bias: false,
            }),
            mixer_norm: norm,
            feed_forward_norm: norm,
            norm_placement: NormPlacement::Pre,
            residual: ResidualTopology::Add,
        }],
        final_norm: norm,
        tie_word_embeddings: true,
        max_position_embeddings: i32::MAX as usize,
    }
}

/// Both public model roles reject oversized histories before allocating device or host arrays.
#[test]
fn oversized_history_geometry_fails_for_causal_and_seq2seq_models() {
    let device = Default::default();
    let causal = ModularCausalLm::<B>::init(spec(), &device).expect("tiny causal model");
    let seq2seq = ModularSeq2SeqLm::<B>::init(
        Seq2SeqSpec {
            encoder: spec(),
            decoder: spec(),
            cross_attention_heads: 1,
            cross_attention_head_dim: 2,
            cross_attention_options: AttentionOptions::default(),
        },
        &device,
    )
    .expect("tiny encoder-decoder model");
    for (batch, capacity, reason) in [
        (usize::MAX, 2, "overflow"),
        (i32::MAX as usize + 1, 1, "limit"),
        (2, i32::MAX as usize, "limit"),
    ] {
        let config = CacheConfig::new(batch, capacity).expect("positive cache geometry");
        for result in [
            causal.new_decode_state(config, batch, &device),
            seq2seq.new_decode_state(config, batch, &device),
        ] {
            let error = result.expect_err("oversized allocation must fail preflight");
            assert!(matches!(&error, Error::Cache(_)), "{error}");
            let message = error.to_string();
            assert!(message.contains("custom.decode.history"), "{message}");
            assert!(message.contains(reason), "{message}");
        }
    }
}

/// A mismatched batch returns the original shape diagnostic before allocation preflight.
#[test]
fn history_batch_mismatch_preserves_contextual_error() {
    let error = ModularDecodeState::<B>::new(
        CacheConfig::new(1, usize::MAX).expect("positive geometry"),
        2,
        spec(),
        StatePurpose::Causal,
        &Default::default(),
    )
    .expect_err("mismatched batch");
    assert!(
        error
            .to_string()
            .contains("expected 1 sequences, received batch size 2")
    );
}

/// A validity layout the planner rejects becomes a contextual cache error, not a panic.
#[test]
fn history_layout_mismatch_maps_to_a_cache_error() {
    let state = ModularDecodeState::<B>::new(
        CacheConfig::new(2, 3).expect("small positive geometry"),
        2,
        spec(),
        StatePurpose::Causal,
        &Default::default(),
    )
    .expect("small histories");
    let error = plan_history(&[true; 3], 2, state.lengths(), state.capacity(), false)
        .map_err(|error| state.append_error(error))
        .expect_err("three cells cannot cover two rows of two columns");
    assert!(matches!(error, Error::Cache(_)));
    assert!(
        error
            .to_string()
            .contains("holds 2 rows, but the step mask has 3 cells for 2 columns")
    );
}

/// Planner failures the decode contract rules out map to one internal cache error.
///
/// Prefill rejects an empty row before the planner can report an empty step,
/// and decode answers an empty step itself and plans every non-advancing row
/// from one slot, so `EmptyStep` and `NeverStarted` only arise from a direct
/// planner call like this one.
#[test]
fn planner_failures_ruled_out_by_the_decode_contract_map_to_cache_errors() {
    let state = ModularDecodeState::<B>::new(
        CacheConfig::new(2, 3).expect("small positive geometry"),
        2,
        spec(),
        StatePurpose::Causal,
        &Default::default(),
    )
    .expect("small histories");
    for (validity, failure) in [
        ([false, false], HistoryAppendError::EmptyStep),
        ([true, false], HistoryAppendError::NeverStarted { row: 1 }),
    ] {
        let planned = plan_history(&validity, 1, state.lengths(), state.capacity(), false);
        assert_eq!(planned.unwrap_err(), failure);
        let error = state.append_error(failure);
        assert!(
            matches!(&error, Error::Cache(reason)
                if reason == "custom history planner rejected a step the decode contract admits"),
            "{error:?}"
        );
    }
}

/// Decode plans every row without a valid cell from one slot and leaves active rows alone.
#[test]
fn inactive_rows_are_planned_from_one_slot() {
    // Rows: never started and idle, started and idle, never started and active, started and active.
    let validity = [false, false, false, false, true, false, false, true];
    assert_eq!(
        inactive_rows_shortened(&[0, 5, 0, 3], &validity, 2),
        [1, 1, 0, 3]
    );
    let (plan, writes) = plan_history(&validity, 2, &[1, 1, 0, 3], 8, false)
        .expect("every row now has a history slot");
    assert_eq!(plan.counts, [0, 0, 1, 1]);
    assert_eq!(plan.lengths, [1, 1, 1, 4]);
    assert_eq!(plan.slots, [0, 0, 0, 0, 0, 0, 0, 3]);
    // Only the advancing rows' slots `0..1` and `3..4` fall inside the write window.
    assert_eq!((writes.start, writes.end), (0, 4));
    assert_eq!(
        writes.written,
        [
            false, false, false, false, //
            false, false, false, false, //
            true, false, false, false, //
            false, false, false, true,
        ]
    );
}

/// Successful and rejected reorders preserve history geometry and accounting before reset.
#[test]
fn history_element_accounting_stays_fixed_across_lifecycle_operations() {
    let mut state = ModularDecodeState::<B>::new(
        CacheConfig::new(2, 3).expect("small positive geometry"),
        2,
        spec(),
        StatePurpose::Causal,
        &Default::default(),
    )
    .expect("small histories");
    assert_eq!(state.allocated_integer_elements(), 12);
    assert_eq!(state.tokens.dims(), [2, 3]);
    assert_eq!(state.positions.dims(), [2, 3]);
    // Distinct populated rows expose incorrect row selection as well as shape damage.
    state.tokens = Tensor::from_data([[1, 2, 0], [3, 2, 1]], &Default::default());
    state.positions = Tensor::from_data([[0, 1, 0], [4, 5, 6]], &Default::default());
    state.metadata.advance(0, 2).expect("first populated row");
    state.metadata.advance(1, 3).expect("second populated row");

    state.reorder(&[1, 1]).expect("duplicate row selection");
    assert_eq!(state.tokens.dims(), [2, 3]);
    assert_eq!(state.positions.dims(), [2, 3]);
    assert_eq!(
        state.allocated_integer_elements(),
        state.tokens.dims().iter().product::<usize>()
            + state.positions.dims().iter().product::<usize>(),
    );
    assert_eq!(state.lengths(), &[3, 3]);
    let reordered_tokens = state.tokens.to_data();
    let reordered_positions = state.positions.to_data();
    assert_eq!(
        reordered_tokens
            .to_vec::<i32>()
            .expect("I32 reordered tokens"),
        vec![3, 2, 1, 3, 2, 1],
    );
    assert_eq!(
        reordered_positions
            .to_vec::<i32>()
            .expect("I32 reordered positions"),
        vec![4, 5, 6, 4, 5, 6],
    );

    let error = state
        .reorder(&[0])
        .expect_err("a shorter batch must be rejected");
    assert!(matches!(error, Error::Cache(_)));
    assert_eq!(state.tokens.dims(), [2, 3]);
    assert_eq!(state.positions.dims(), [2, 3]);
    assert_eq!(state.allocated_integer_elements(), 12);
    assert_eq!(state.tokens.to_data(), reordered_tokens);
    assert_eq!(state.positions.to_data(), reordered_positions);
    assert_eq!(state.lengths(), &[3, 3]);

    state.truncate(&[0, 0]).expect("empty prefixes");
    state.reset().expect("reset histories");
    assert_eq!(state.allocated_integer_elements(), 12);
    assert_eq!(state.lengths(), &[0, 0]);
}

/// Newly allocated histories are zeroed in the backend's native integer dtype.
#[test]
fn histories_use_native_integer_dtype_and_zero_initialization() {
    let state = ModularDecodeState::<B>::new(
        CacheConfig::new(2, 3).expect("small positive geometry"),
        2,
        spec(),
        StatePurpose::Seq2Seq,
        &Default::default(),
    )
    .expect("native integer histories");
    assert_eq!(state.allocated_integer_elements(), 12);
    assert_eq!(state.tokens.dtype(), burn::tensor::DType::I32);
    assert_eq!(state.positions.dtype(), burn::tensor::DType::I32);
    assert_eq!(
        state.tokens.to_data().to_vec::<i32>().expect("I32 tokens"),
        vec![0; 6]
    );
    assert_eq!(
        state
            .positions
            .to_data()
            .to_vec::<i32>()
            .expect("I32 positions"),
        vec![0; 6]
    );
}

/// One host-side step of the history reference: token, optional position, and validity per cell.
#[derive(Clone, Debug)]
struct ReferenceStep {
    /// Row-major `[batch, step]` token IDs.
    tokens: Vec<i32>,
    /// Row-major explicit positions, or `None` for implicit slot positions.
    positions: Option<Vec<i32>>,
    /// Row-major validity flags.
    validity: Vec<bool>,
}

/// Rows of committed `(token, position)` pairs, the model of the device histories.
type ReferenceHistory = Vec<Vec<(i32, i32)>>;

/// Applies one append to the reference: valid cells append in order, implicitly at their slot.
fn reference_append(history: &mut ReferenceHistory, step: &ReferenceStep, width: usize) {
    for (row, entries) in history.iter_mut().enumerate() {
        for column in 0..width {
            let cell = row * width + column;
            if step.validity[cell] {
                let position = match &step.positions {
                    Some(positions) => positions[cell],
                    None => entries.len() as i32,
                };
                entries.push((step.tokens[cell], position));
            }
        }
    }
}

/// Reads the committed prefix of every device history row.
fn device_history(state: &ModularDecodeState<B>) -> ReferenceHistory {
    let capacity = state.capacity();
    let tokens = state.tokens.to_data().to_vec::<i32>().unwrap();
    let positions = state.positions.to_data().to_vec::<i32>().unwrap();
    state
        .lengths()
        .iter()
        .enumerate()
        .map(|(row, &length)| {
            (0..length)
                .map(|slot| {
                    (
                        tokens[row * capacity + slot],
                        positions[row * capacity + slot],
                    )
                })
                .collect()
        })
        .collect()
}

/// Builds the input tensors of one reference step.
fn step_input(step: &ReferenceStep, batch: usize, width: usize) -> CausalLmInput<B> {
    let device = Default::default();
    CausalLmInput {
        token_ids: Tensor::from_data(
            TensorData::new(step.tokens.clone(), [batch, width]),
            &device,
        ),
        token_mask: Some(crate::TokenMask::new(Tensor::from_bool(
            TensorData::new(step.validity.clone(), [batch, width]),
            &device,
        ))),
        position_ids: step.positions.clone().map(|positions| {
            Tensor::from_data(TensorData::new(positions, [batch, width]), &device)
        }),
    }
}

proptest::proptest! {
    #![proptest_config(proptest::prelude::ProptestConfig::with_cases(48))]

    /// Batched window writes reproduce a per-row host history across appends, truncation, and reorder.
    ///
    /// Each step's history is written by one gather/select per tensor; the
    /// committed prefixes must equal the reference that appends valid cells
    /// row by row, including ragged, left-, right-, and interior-padded rows,
    /// explicit positions, and slots previously exposed by truncation.
    #[test]
    fn batched_appends_match_a_per_row_history_reference(
        widths in proptest::collection::vec(1usize..4, 4),
        cells in proptest::collection::vec(proptest::bool::ANY, 48),
        values in proptest::collection::vec(0i32..64, 48),
        explicit in proptest::collection::vec(proptest::bool::ANY, 4),
        cuts in proptest::collection::vec(0usize..3, 3),
        permutation in proptest::collection::vec(0usize..3, 3),
    ) {
        /// Number of independent history rows in the generated scenario.
        const BATCH: usize = 3;
        /// Per-row history capacity; small enough that some appends overflow.
        const CAPACITY: usize = 14;
        let mut model_spec = spec();
        model_spec.vocabulary_size = 64;
        model_spec.max_position_embeddings = 64;
        let mut state = ModularDecodeState::<B>::new(
            CacheConfig::new(BATCH, CAPACITY).unwrap(),
            BATCH,
            model_spec,
            StatePurpose::Causal,
            &Default::default(),
        )
        .unwrap();
        let mut reference: ReferenceHistory = vec![Vec::new(); BATCH];
        let mut offset = 0;
        for (index, &width) in widths.iter().enumerate() {
            let count = BATCH * width;
            // The first step is a prompt in which every row needs a token.
            let validity = (0..count)
                .map(|cell| cells[offset + cell] || (index == 0 && cell % width == 0))
                .collect::<Vec<_>>();
            let step = ReferenceStep {
                tokens: values[offset..offset + count].to_vec(),
                positions: explicit[index]
                    .then(|| values[offset..offset + count].iter().map(|value| value % 40).collect()),
                validity,
            };
            offset += count;
            // A rejected plan (for example an empty decode step) must change nothing.
            if let Ok(append) = state.prepare_append(step_input(&step, BATCH, width), 64, index == 0) {
                state.commit(append).unwrap();
                reference_append(&mut reference, &step, width);
            }
            proptest::prop_assert_eq!(device_history(&state), reference.clone());
            if index == 1 {
                let lengths = reference
                    .iter()
                    .zip(&cuts)
                    .map(|(row, cut)| row.len().saturating_sub(*cut))
                    .collect::<Vec<_>>();
                state.truncate(&lengths).unwrap();
                for (row, length) in reference.iter_mut().zip(lengths) {
                    row.truncate(length);
                }
                state.reorder(&permutation).unwrap();
                reference = permutation.iter().map(|&row| reference[row].clone()).collect();
                proptest::prop_assert_eq!(device_history(&state), reference.clone());
            }
        }
    }
}

/// Median wall-clock milliseconds of `repetitions` runs of `run`, after one warm-up.
fn median_ms(repetitions: usize, mut run: impl FnMut()) -> f64 {
    run();
    let mut samples = (0..repetitions)
        .map(|_| {
            let start = std::time::Instant::now();
            run();
            start.elapsed().as_secs_f64() * 1e3
        })
        .collect::<Vec<_>>();
    samples.sort_by(f64::total_cmp);
    samples[samples.len() / 2]
}

/// Former per-row append: one selection upload, compaction, readback, and two slice assignments per row.
fn per_row_append(
    state: &ModularDecodeState<B>,
    input: &CausalLmInput<B>,
) -> (Tensor<B, 2, Int>, Tensor<B, 2, Int>) {
    let [batch, step] = input.token_ids.dims();
    let device = input.token_ids.device();
    let host = HostInput::read(input, batch, step, "benchmark").unwrap();
    let valid = host.valid_indices(batch, step);
    let mut tokens = state.tokens.clone();
    let mut positions = state.positions.clone();
    for (row, columns) in valid.iter().enumerate() {
        if columns.is_empty() {
            continue;
        }
        let plan = CompactRow::<B>::new(row, columns, step, &device).unwrap();
        let start = state.lengths()[row];
        let end = start + plan.count();
        // The former path validated each row's positions with a readback.
        implicit_position_ids::<B>(start, plan.count(), &device, "benchmark")
            .unwrap()
            .into_data();
        tokens = tokens.slice_assign([row..row + 1, start..end], plan.compact(&input.token_ids));
        positions = positions.slice_assign(
            [row..row + 1, start..end],
            implicit_position_ids(start, plan.count(), &device, "benchmark").unwrap(),
        );
    }
    (tokens, positions)
}

/// One decode step's history write for a 32-row batch: per-row loop versus one batched window.
#[test]
#[ignore = "timing measurement; run in release with --ignored --nocapture"]
fn batched_history_writes_versus_per_row_appends() {
    let device = Default::default();
    let (batch, capacity) = (32, 512);
    let mut model_spec = spec();
    model_spec.vocabulary_size = 32;
    model_spec.max_position_embeddings = capacity;
    let mut state = ModularDecodeState::<B>::new(
        crate::cache::CacheConfig::new(batch, capacity).unwrap(),
        batch,
        model_spec,
        StatePurpose::Causal,
        &device,
    )
    .unwrap();
    // Ragged committed lengths, as in beam search with finished rows.
    for row in 0..batch {
        state.metadata.advance(row, 100 + row % 7).unwrap();
    }
    let input = CausalLmInput {
        token_ids: Tensor::from_data(
            TensorData::new((0..batch as i32).map(|row| row % 32).collect(), [batch, 1]),
            &device,
        ),
        token_mask: None,
        position_ids: None,
    };
    let old = median_ms(20, || {
        let (tokens, positions) = per_row_append(&state, &input);
        tokens.into_data();
        positions.into_data();
    });
    let new = median_ms(20, || {
        let append = state.prepare_append(input.clone(), 32, false).unwrap();
        append.tokens.into_data();
        append.positions.into_data();
    });
    let append = state.prepare_append(input.clone(), 32, false).unwrap();
    let (tokens, positions) = per_row_append(&state, &input);
    append
        .tokens
        .into_data()
        .assert_eq(&tokens.into_data(), true);
    append
        .positions
        .into_data()
        .assert_eq(&positions.into_data(), true);
    println!(
        "history write, batch {batch}, capacity {capacity}, one token per row: per-row {old:.3} ms, batched {new:.3} ms ({:.1}x)",
        old / new
    );
}

/// Unwraps a shape error into `(tensor, expected, actual)`.
fn shape_parts(error: Error) -> (String, String, String) {
    match error {
        Error::Shape {
            tensor,
            expected,
            actual,
        } => (tensor, expected, actual),
        other => panic!("expected Error::Shape, found {other:?}"),
    }
}

/// Malformed decode steps fail with the offending tensor's name and commit nothing.
#[test]
fn malformed_decode_steps_are_rejected_without_commit() {
    let device = Default::default();
    let model = ModularCausalLm::<B>::init(spec(), &device).expect("tiny causal model");
    let mut state = model
        .new_decode_state(CacheConfig::new(2, 4).unwrap(), 2, &device)
        .unwrap();
    let ints = |values: Vec<i64>, shape: [usize; 2]| {
        Tensor::<B, 2, Int>::from_data(TensorData::new(values, shape), &device)
    };
    model
        .prefill(
            CausalLmInput {
                token_ids: ints(vec![1, 2, 3, 0], [2, 2]),
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .unwrap();
    let step = |token_ids, token_mask, position_ids| crate::DecodeInput {
        token_ids,
        token_mask,
        position_ids,
    };
    let cases = [
        (
            step(ints(vec![1], [1, 1]), None, None),
            (
                "custom.decode.token_ids",
                "[2, step>0]".to_owned(),
                "[1, 1]".to_owned(),
            ),
        ),
        (
            step(ints(Vec::new(), [2, 0]), None, None),
            (
                "custom.decode.token_ids",
                "[2, step>0]".to_owned(),
                "[2, 0]".to_owned(),
            ),
        ),
        (
            step(
                ints(vec![1, 2], [2, 1]),
                Some(TokenMask::new(
                    crate::architecture::bool_tensor_from_values(vec![true; 4], [2, 2], &device),
                )),
                None,
            ),
            (
                "custom.decode.token_mask",
                "[2, 1]".to_owned(),
                "[2, 2]".to_owned(),
            ),
        ),
        (
            step(ints(vec![1, 2], [2, 1]), None, Some(ints(vec![2], [1, 1]))),
            (
                "custom.decode.position_ids",
                "[2, 1]".to_owned(),
                "[1, 1]".to_owned(),
            ),
        ),
        (
            step(ints(vec![1, 4], [2, 1]), None, None),
            (
                "custom.decode.token_ids",
                "token values in 0..4".to_owned(),
                "observed range 1..=4".to_owned(),
            ),
        ),
    ];
    for (input, (tensor, expected, actual)) in cases {
        let error = model
            .decode(input, &mut state, &ForwardOptions::default())
            .unwrap_err();
        assert_eq!(shape_parts(error), (tensor.to_owned(), expected, actual));
        assert_eq!(state.lengths(), [2, 2], "a rejected step commits nothing");
    }
}

/// An encoder-decoder decode step with a foreign batch fails before the decoder runs.
#[test]
fn malformed_seq2seq_steps_are_rejected_without_commit() {
    let device = Default::default();
    let model = ModularSeq2SeqLm::<B>::init(
        Seq2SeqSpec {
            encoder: spec(),
            decoder: spec(),
            cross_attention_heads: 1,
            cross_attention_head_dim: 2,
            cross_attention_options: AttentionOptions::default(),
        },
        &device,
    )
    .expect("tiny encoder-decoder model");
    let source = CausalLmInput {
        token_ids: Tensor::<B, 2, Int>::from_ints([[1, 2], [3, 1]], &device),
        token_mask: None,
        position_ids: None,
    };
    let encoded = model.encode(source, &ForwardOptions::default()).unwrap();
    let mut state = model
        .new_decode_state(CacheConfig::new(2, 4).unwrap(), 2, &device)
        .unwrap();
    let error = model
        .prefill_decoder(
            &encoded,
            CausalLmInput {
                token_ids: Tensor::<B, 2, Int>::from_ints([[1, 2, 3]], &device),
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .unwrap_err();
    assert_eq!(
        shape_parts(error),
        (
            "custom.decode.token_ids".to_owned(),
            "[2, step>0]".to_owned(),
            "[1, 3]".to_owned()
        )
    );
    assert_eq!(state.lengths(), [0, 0]);
}
