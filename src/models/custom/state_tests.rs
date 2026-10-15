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
