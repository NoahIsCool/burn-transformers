//! Rotary composition tests against a scalar RoPE reference and position-free attention.

use burn::{
    backend::Flex,
    module::Param,
    nn::{Linear, LinearConfig},
    tensor::{Bool, Distribution, Int, Tensor, TensorData},
};

use super::*;
use crate::{
    TokenMask,
    attention::AttentionPrecisionMode,
    experiment::{AttentionPolicy, ResidualBlock},
    position::RopeScaling,
};

/// Deterministic F32 CPU backend.
type B = Flex<f32, i32>;

/// Fixed logical `[input, output]` matrix, independent of random initialization.
fn matrix(input: usize, output: usize, values: &[f32]) -> Linear<B> {
    let device = Default::default();
    let mut linear = LinearConfig::new(input, output)
        .with_bias(false)
        .init(&device);
    linear.weight = Param::from_tensor(Tensor::from_data(
        TensorData::new(values.to_vec(), [input, output]),
        &device,
    ));
    linear
}

/// One head over two channels whose Q/K/V/output projections are all the identity.
fn identity_attention(mode: AttentionPrecisionMode) -> GroupedQueryAttention<B> {
    let identity = || matrix(2, 2, &[1.0, 0.0, 0.0, 1.0]);
    let mut attention = GroupedQueryAttentionConfig::new(2, 1, 1, 2)
        .with_precision_mode(mode)
        .init::<B>(&Default::default())
        .unwrap();
    attention.q_proj = identity();
    attention.k_proj = identity();
    attention.v_proj = identity();
    attention.o_proj = identity();
    attention
}

/// Unscaled RoPE with base 10000 over `rotary` of `head` channels.
fn rope(head: usize, rotary: usize) -> RopeConfig {
    RopeConfig::new(head, rotary, 10_000.0, RopeScaling::None).unwrap()
}

/// A one-row context with explicit positions and optional padding.
fn context_at(positions: &[i32], valid: Option<&[bool]>) -> ExecutionContext<B> {
    let device = Default::default();
    let tokens = positions.len();
    let mask = valid.map(|valid| {
        TokenMask::new(Tensor::<B, 2, Bool>::from_data(
            TensorData::new(valid.to_vec(), [1, tokens]),
            &device,
        ))
    });
    ExecutionContext::new(
        Tensor::<B, 2, Int>::from_data(TensorData::new(positions.to_vec(), [1, tokens]), &device),
        mask,
    )
    .unwrap()
}

/// Runs one transform at decoder layer zero inside its own hook scope.
fn run(
    transform: &impl HiddenTransform<B>,
    hidden: Tensor<B, 3>,
    context: &ExecutionContext<B>,
) -> Vec<f32> {
    let layer = Layer::decoder(0);
    Hooks::new()
        .run(transform.activation_points(layer), |hooks| {
            transform.forward(hidden, context, layer, hooks)
        })
        .unwrap()
        .into_data()
        .to_vec()
        .unwrap()
}

/// Largest absolute elementwise difference between two equally long vectors.
fn max_difference(left: &[f32], right: &[f32]) -> f32 {
    assert_eq!(left.len(), right.len());
    left.iter()
        .zip(right)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0, f32::max)
}

/// Rotates one two-channel head by angle `position` (frequency `theta^0 = 1`).
fn rotate(x: [f64; 2], position: f64) -> [f64; 2] {
    let (sin, cos) = position.sin_cos();
    [x[0] * cos - x[1] * sin, x[1] * cos + x[0] * sin]
}

/// Causal single-head attention with identity projections, computed in F64.
///
/// Scores are `<R(p_m) x_m, R(p_n) x_n> / sqrt(2)` for `n <= m`; values are unrotated.
fn reference(tokens: &[[f64; 2]], positions: &[i32]) -> Vec<f32> {
    let mut output = Vec::new();
    for (query_index, (query, &query_position)) in tokens.iter().zip(positions).enumerate() {
        let rotated_query = rotate(*query, f64::from(query_position));
        let scores = tokens
            .iter()
            .zip(positions)
            .take(query_index + 1)
            .map(|(key, &key_position)| {
                let rotated_key = rotate(*key, f64::from(key_position));
                (rotated_query[0] * rotated_key[0] + rotated_query[1] * rotated_key[1])
                    / 2.0_f64.sqrt()
            })
            .collect::<Vec<_>>();
        let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let weights = scores
            .iter()
            .map(|score| (score - maximum).exp())
            .collect::<Vec<_>>();
        let total = weights.iter().sum::<f64>();
        for channel in 0..2 {
            let value = weights
                .iter()
                .zip(tokens)
                .map(|(weight, token)| weight / total * token[channel])
                .sum::<f64>();
            output.push(value as f32);
        }
    }
    output
}

/// Identity-projection rotary attention matches the scalar RoPE equations in both precision modes.
#[test]
fn rotary_attention_matches_a_scalar_rope_reference() {
    let tokens = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
    let hidden = Tensor::<B, 3>::from_data(
        TensorData::new(
            tokens.iter().flatten().map(|value| *value as f32).collect(),
            [1, 3, 2],
        ),
        &Default::default(),
    );
    for mode in [
        AttentionPrecisionMode::BackendDefault,
        AttentionPrecisionMode::EagerFp32Softmax,
    ] {
        let attention = RotaryAttention::new(identity_attention(mode), rope(2, 2)).unwrap();
        for positions in [[0, 1, 2], [5, 6, 7], [0, 2, 1], [3, 3, 3]] {
            let actual = run(&attention, hidden.clone(), &context_at(&positions, None));
            let expected = reference(&tokens, &positions);
            assert!(
                max_difference(&actual, &expected) < 1e-5,
                "{mode:?} {positions:?}: {actual:?} != {expected:?}"
            );
        }
    }
}

/// Query/key hook points observe the rotated tensors that form the scores.
#[test]
fn query_and_key_hooks_observe_rotated_projections() {
    let attention = RotaryAttention::new(
        identity_attention(AttentionPrecisionMode::EagerFp32Softmax),
        rope(2, 2),
    )
    .unwrap();
    let hidden = Tensor::<B, 3>::from_floats([[[1.0, 0.0], [0.0, 1.0]]], &Default::default());
    let layer = Layer::decoder(0);
    let mut hooks = Hooks::new();
    let query = hooks.capture(layer.self_attention().query());
    let key = hooks.capture(layer.self_attention().key());
    hooks
        .run(attention.activation_points(layer), |hooks| {
            attention.forward(hidden, &context_at(&[2, 5], None), layer, hooks)
        })
        .unwrap();
    let expected = [rotate([1.0, 0.0], 2.0), rotate([0.0, 1.0], 5.0)]
        .iter()
        .flatten()
        .map(|value| *value as f32)
        .collect::<Vec<_>>();
    for capture in [query, key] {
        let actual = hooks.get(&capture).unwrap();
        assert_eq!(actual.dims(), [1, 1, 2, 2]);
        let actual = actual.into_data().to_vec::<f32>().unwrap();
        assert!(max_difference(&actual, &expected) < 1e-6, "{actual:?}");
    }
}

/// Randomly initialized multi-head rotary attention is shift invariant but order sensitive.
///
/// The position-free implementation is shown alongside: it cannot distinguish
/// any of these position assignments.
#[test]
fn rotary_attention_depends_only_on_relative_positions() {
    let device = Default::default();
    B::seed(&device, 7);
    let config = GroupedQueryAttentionConfig::new(8, 4, 2, 4);
    // Rotate half of each head to also exercise the unrotated channel suffix.
    let partial = RotaryAttention::<B>::init(&config, rope(4, 2), &device).unwrap();
    let full = RotaryAttention::new(partial.attention.clone(), rope(4, 4)).unwrap();
    let hidden = Tensor::<B, 3>::random([1, 5, 8], Distribution::Normal(0.0, 1.0), &device);
    let base = context_at(&[0, 1, 2, 3, 4], None);
    let shifted = context_at(&[9, 10, 11, 12, 13], None);
    let reordered = context_at(&[0, 3, 1, 4, 2], None);
    for rotary in [&partial, &full] {
        let at_base = run(rotary, hidden.clone(), &base);
        assert!(max_difference(&at_base, &run(rotary, hidden.clone(), &shifted)) < 1e-4);
        assert!(max_difference(&at_base, &run(rotary, hidden.clone(), &reordered)) > 1e-3);
    }
    let position_free = &partial.attention;
    let at_base = run(position_free, hidden.clone(), &base);
    assert_eq!(at_base, run(position_free, hidden, &reordered));
}

/// With one shared position, rotation cancels and both attention transforms agree.
///
/// This holds with padding and an explicit policy, so the shared policy plan is
/// exercised identically by both paths.
#[test]
fn equal_positions_reduce_rotary_attention_to_position_free_attention() {
    let device = Default::default();
    B::seed(&device, 11);
    let config = GroupedQueryAttentionConfig::new(6, 2, 1, 6);
    let attention = RotaryAttention::<B>::init(&config, rope(6, 6), &device).unwrap();
    let hidden = Tensor::<B, 3>::random([1, 4, 6], Distribution::Normal(0.0, 1.0), &device);
    let allowed = Tensor::<B, 4, Bool>::from_data(
        TensorData::new(
            vec![
                true, false, true, true, true, true, false, true, true, true, true, true, true,
                true, true, true,
            ],
            [1, 1, 4, 4],
        ),
        &device,
    );
    for position in [0, 3] {
        let context = context_at(&[position; 4], Some(&[true, true, true, false]))
            .with_attention(AttentionPolicy::bidirectional().with_allowed(allowed.clone()));
        let rotary = run(&attention, hidden.clone(), &context);
        let position_free = run(&attention.attention, hidden.clone(), &context);
        assert!(max_difference(&rotary, &position_free) < 1e-5, "{position}");
    }
}

/// Construction rejects a RoPE policy built for a different head width.
#[test]
fn rotary_policy_must_match_the_attention_head_width() {
    let attention = identity_attention(AttentionPrecisionMode::BackendDefault);
    let error = RotaryAttention::new(attention, rope(4, 4)).unwrap_err();
    let message = error.to_string();
    assert!(message.contains("rotary_attention.rope"), "{message}");
    assert!(message.contains("head_dim 2"), "{message}");
}

/// A residual block composes rotary attention and exposes mixer points under both names.
#[test]
fn residual_blocks_compose_rotary_attention_with_mixer_points() {
    let device = Default::default();
    let attention = RotaryAttention::new(
        identity_attention(AttentionPrecisionMode::BackendDefault),
        rope(2, 2),
    )
    .unwrap();
    let block =
        ResidualBlock::without_normalization(2, attention, matrix(2, 2, &[0.0; 4])).unwrap();
    let layer = Layer::decoder(0);
    assert_eq!(layer.mixer_input(), layer.attention_input());
    assert_eq!(layer.mixer_output(), layer.attention_output());
    let points = block.activation_points(layer);
    assert!(points.contains(&layer.self_attention().query().info()));
    let mut hooks = Hooks::new();
    let mixed = hooks.capture(layer.mixer_output());
    let tokens = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
    let hidden = Tensor::<B, 3>::from_floats([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]], &device);
    let positions = [4, 5, 6];
    let output = block
        .run_with_hooks(
            hidden.clone(),
            &context_at(&positions, None),
            layer,
            &mut hooks,
        )
        .unwrap();
    let expected = reference(&tokens, &positions);
    let mixed = hooks
        .get(&mixed)
        .unwrap()
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    assert!(max_difference(&mixed, &expected) < 1e-5);
    let residual = (hidden + Tensor::from_data(TensorData::new(expected, [1, 3, 2]), &device))
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    let output = output.into_data().to_vec::<f32>().unwrap();
    assert!(max_difference(&output, &residual) < 1e-5);
}

/// Gradients reach every rotary-attention projection through a residual block.
#[cfg(feature = "autodiff")]
#[test]
fn rotary_attention_projections_receive_gradients() {
    use burn::backend::Autodiff;
    /// Differentiable CPU backend.
    type Ad = Autodiff<B>;
    let device = Default::default();
    let config = GroupedQueryAttentionConfig::new(4, 2, 1, 2);
    let attention = RotaryAttention::<Ad>::init(&config, rope(2, 2), &device).unwrap();
    let block = ResidualBlock::without_normalization(
        4,
        attention,
        LinearConfig::new(4, 4).init::<Ad>(&device),
    )
    .unwrap();
    let hidden = Tensor::<Ad, 3>::random([1, 3, 4], Distribution::Normal(0.0, 1.0), &device);
    let context = ExecutionContext::contiguous(1, 3, &device).unwrap();
    let loss = block
        .run(hidden, &context, Layer::decoder(0))
        .unwrap()
        .powi_scalar(2)
        .sum();
    let gradients = loss.backward();
    for projection in [
        &block.mixer.attention.q_proj,
        &block.mixer.attention.k_proj,
        &block.mixer.attention.v_proj,
        &block.mixer.attention.o_proj,
    ] {
        let gradient = projection
            .weight
            .val()
            .grad(&gradients)
            .expect("projection gradient");
        assert!(gradient.abs().sum().into_scalar() > 0.0);
    }
}
