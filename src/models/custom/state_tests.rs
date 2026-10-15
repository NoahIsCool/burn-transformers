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
        const BATCH: usize = 3;
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
