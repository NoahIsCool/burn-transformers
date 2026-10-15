//! The shared T5 block graph: F16 overflow guard, topology checks, and cached cross inputs.

use burn::{
    backend::Flex,
    module::{Module, ModuleMapper, Param},
    tensor::{DType, Int, Tensor, TensorData},
};

use super::*;
use crate::{
    CausalLmInput, ForwardOptions,
    architecture::Seq2SeqInput,
    models::t5::{T5Config, T5FeedForwardProjection, T5ForConditionalGeneration},
};

/// Deterministic CPU backend; it stores F16 tensors when parameters are cast.
type TestBackend = Flex<f32, i32>;

/// Builds an F16 `[1, 1, n]` tensor from F32 values (rounded to F16).
fn half(values: &[f32]) -> Tensor<TestBackend, 3> {
    Tensor::<TestBackend, 3>::from_data(
        TensorData::new(values.to_vec(), [1, 1, values.len()]),
        &Default::default(),
    )
    .cast(DType::F16)
}

/// Reads any floating tensor as F32 host values.
fn values<const D: usize>(tensor: Tensor<TestBackend, D>) -> Vec<f32> {
    tensor.cast(DType::F32).into_data().to_vec::<f32>().unwrap()
}

/// Non-F16 residuals are returned untouched, even beyond the F16 range.
#[test]
fn overflow_guard_ignores_wider_dtypes() {
    let wide = Tensor::<TestBackend, 3>::from_data(
        TensorData::new(vec![1.0e9_f32, f32::INFINITY, -3.0], [1, 1, 3]),
        &Default::default(),
    );
    assert_eq!(values(clamp_f16_residual(wide.clone())), values(wide));
}

/// Finite F16 residuals are untouched: the bound is the F16 maximum itself.
#[test]
fn overflow_guard_preserves_finite_half_values() {
    let finite = half(&[65504.0, -65504.0, 1.5, 0.0]);
    assert_eq!(values(clamp_f16_residual(finite.clone())), values(finite));
}

/// Once any value overflows, every value is clamped to the reference `max - 1000` (64512 in F16).
#[test]
fn overflow_guard_clamps_after_an_infinity() {
    let overflowed = half(&[f32::INFINITY, f32::NEG_INFINITY, 65504.0, -64600.0, 2.0]);
    assert_eq!(
        values(clamp_f16_residual(overflowed)),
        [64512.0, -64512.0, 64512.0, -64512.0, 2.0]
    );
}

/// NaN propagates through the guard, as `torch.clamp` propagates it.
#[test]
fn overflow_guard_propagates_nan() {
    let guarded = values(clamp_f16_residual(half(&[f32::NAN, f32::INFINITY, 1.0])));
    assert!(guarded[0].is_nan());
    assert_eq!(&guarded[1..], [64512.0, 1.0]);
}

/// Casts every floating parameter to F16 and amplifies FFN output projections.
struct OverflowingHalf;

impl ModuleMapper<TestBackend> for OverflowingHalf {
    /// Scales every matrix by 8 and leaves vectors (norm scales) at one.
    fn map_float<const D: usize>(
        &mut self,
        param: Param<Tensor<TestBackend, D>>,
    ) -> Param<Tensor<TestBackend, D>> {
        param.map(|tensor| {
            let shape = tensor.dims();
            let count = shape.iter().product::<usize>();
            let values = (0..count)
                .map(|index| {
                    if D == 1 {
                        1.0
                    } else {
                        ((index % 7) as f32 - 2.0) * 8.0
                    }
                })
                .collect::<Vec<_>>();
            Tensor::<TestBackend, D>::from_data(TensorData::new(values, shape), &tensor.device())
                .cast(DType::F16)
        })
    }
}

/// An F16 gated model whose FFN overflows still produces finite logits, as in the reference.
///
/// Without the guard the residual becomes infinite and the next RMS
/// normalization turns it into NaN (`inf / inf`).
#[test]
fn half_precision_overflow_keeps_logits_finite() {
    let device = Default::default();
    let config = T5Config {
        vocab_size: 16,
        d_model: 8,
        d_kv: 4,
        d_ff: 32,
        num_layers: 2,
        num_decoder_layers: Some(2),
        num_heads: 2,
        relative_attention_num_buckets: 8,
        relative_attention_max_distance: 16,
        feed_forward_proj: T5FeedForwardProjection::GatedGelu,
        tie_word_embeddings: false,
        ..T5Config::default()
    };
    let model = T5ForConditionalGeneration::<TestBackend>::init(config, &device)
        .unwrap()
        .map(&mut OverflowingHalf);
    let tokens = || CausalLmInput {
        token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[1, 2, 3, 4]], &device),
        token_mask: None,
        position_ids: None,
    };
    let output = model
        .forward(
            Seq2SeqInput {
                encoder: tokens(),
                decoder: tokens(),
            },
            &ForwardOptions {
                hidden_states: true,
                ..ForwardOptions::default()
            },
        )
        .unwrap();
    assert_eq!(output.decoder.logits.dtype(), DType::F16);
    let encoder_states = output.encoder.hidden_states.clone();
    let overflowed = encoder_states[1..encoder_states.len() - 1]
        .iter()
        .flat_map(|state| values(state.clone()))
        .any(|value| value.abs() >= 64512.0);
    assert!(overflowed, "the fixture must actually reach the F16 guard");
    assert!(
        values(output.decoder.logits)
            .into_iter()
            .all(f32::is_finite)
    );
}

/// A block refuses inputs that disagree with its encoder or decoder topology.
#[test]
fn block_topology_mismatches_are_contextual() {
    let device = Default::default();
    let config = T5Config {
        vocab_size: 16,
        d_model: 4,
        d_kv: 2,
        d_ff: 8,
        num_layers: 1,
        num_decoder_layers: Some(1),
        num_heads: 2,
        relative_attention_num_buckets: 8,
        relative_attention_max_distance: 16,
        ..T5Config::default()
    };
    let model = T5ForConditionalGeneration::<TestBackend>::init(config, &device).unwrap();
    let hidden = Tensor::<TestBackend, 3>::zeros([1, 2, 4], &device);
    let source = Tensor::<TestBackend, 3>::zeros([1, 3, 4], &device);
    let plan = SelfAttentionPlan::padded(None, None, false);
    let cross = CrossAttentionPlan::new(CrossSource::Encoder(&source), None);
    let run = |block: &crate::models::t5::T5Block<TestBackend>,
               cross: Option<&CrossAttentionPlan<'_, TestBackend>>| {
        block
            .run(
                hidden.clone(),
                plan.clone(),
                &mut FreshKeyValues,
                cross,
                Layer::encoder(0),
                &mut Hooks::new(),
            )
            .unwrap_err()
            .to_string()
    };
    assert!(run(&model.encoder.block[0], Some(&cross)).contains("t5.decoder.cross_attention"));
    assert!(run(&model.decoder.block[0], None).contains("t5.encoder.block"));
    let mut bidirectional = model.decoder.block[0].clone();
    bidirectional.self_attention.causal = false;
    assert!(run(&bidirectional, Some(&cross)).contains("bidirectional self-attention"));
}

/// Cached cross-attention reports a missing layer and rejects query norms it cannot reuse.
#[test]
fn projected_cross_attention_validates_cached_layers() {
    let device = Default::default();
    let config = T5Config {
        vocab_size: 16,
        d_model: 4,
        d_kv: 2,
        d_ff: 8,
        num_layers: 1,
        num_decoder_layers: Some(1),
        num_heads: 2,
        relative_attention_num_buckets: 8,
        relative_attention_max_distance: 16,
        ..T5Config::default()
    };
    let model = T5ForConditionalGeneration::<TestBackend>::init(config, &device).unwrap();
    let cross = model.decoder.block[0].cross_attention.as_ref().unwrap();
    let hidden = Tensor::<TestBackend, 3>::zeros([1, 1, 4], &device);
    let keys = [Tensor::<TestBackend, 4>::zeros([1, 2, 3, 2], &device)];
    let plan = CrossAttentionPlan::new(
        CrossSource::Projected {
            keys: &keys,
            values: &keys,
        },
        None,
    );
    let error = cross
        .forward_with(hidden.clone(), &plan, Layer::decoder(1), &mut Hooks::new())
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("layer 1 is outside 1 cached layers")
    );
    cross
        .forward_with(hidden.clone(), &plan, Layer::decoder(0), &mut Hooks::new())
        .expect("cached layer zero");
    let mut normalized = cross.clone();
    normalized.enc_dec_attention.attention.query_norm = Some(
        crate::normalization::RmsNormConfig::new(2)
            .init(&device)
            .unwrap(),
    );
    let error = normalized
        .forward_with(hidden, &plan, Layer::decoder(0), &mut Hooks::new())
        .unwrap_err();
    assert!(error.to_string().contains("query normalization"));
}

/// Padded target planes hide padded keys from valid queries and leave no score row empty.
///
/// Left, right, and interior padding are combined. Valid queries exclude
/// exactly the padded keys; padded queries exclude nothing, so together with
/// causality (`key <= query`) every row keeps at least key zero, including
/// the leading padded queries of a left-padded row, which precede every
/// valid key.
#[test]
fn padded_causal_plans_leave_every_query_a_visible_key() {
    let device = Default::default();
    let rows = [
        [false, false, true, true],
        [true, true, true, false],
        [true, false, true, true],
    ];
    let mask = TokenMask::new(Tensor::<TestBackend, 2, burn::tensor::Bool>::from_data(
        TensorData::new(rows.concat(), [3, 4]),
        &device,
    ));
    let plan = SelfAttentionPlan::padded_causal(&mask, None);
    assert!(plan.causal && plan.bias.is_none());
    let excluded = plan.excluded.unwrap();
    assert_eq!(excluded.dims(), [3, 1, 4, 4]);
    let excluded = excluded.into_data().to_vec::<bool>().unwrap();
    for (row, valid) in rows.iter().enumerate() {
        for query in 0..4 {
            let line = &excluded[(row * 4 + query) * 4..(row * 4 + query + 1) * 4];
            for key in 0..4 {
                assert_eq!(
                    line[key],
                    valid[query] && !valid[key],
                    "{row} {query} {key}"
                );
            }
            assert!(
                (0..=query).any(|key| !line[key]),
                "row {row} query {query} has no visible causal key"
            );
        }
    }
}

/// Exclusion planes that do not broadcast to the score shape fail with a shape error.
#[test]
fn non_broadcasting_exclusions_are_rejected() {
    let device = Default::default();
    let plane = Tensor::<TestBackend, 4, burn::tensor::Bool>::empty([2, 1, 1, 3], &device);
    assert!(expand_exclusions(Some(plane.clone()), [2, 4, 5, 3]).is_ok());
    let error = expand_exclusions(Some(plane), [1, 4, 5, 3]).unwrap_err();
    assert!(error.to_string().contains("t5.attention.mask"));
    assert!(
        expand_exclusions::<TestBackend>(None, [1, 1, 1, 1])
            .unwrap()
            .is_none()
    );
}
