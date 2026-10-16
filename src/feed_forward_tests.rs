//! Feed-forward tests covering activation formulas, gating, validation, and tensor layouts.

use burn::{backend::Flex, tensor::Tensor};

use crate::architecture::Activation;

use burn::{module::Param, tensor::TensorData};

use super::{
    ClampedSwiGlu, DenseFeedForwardConfig, GatedFeedForwardConfig, apply_activation,
    mlp_hook_points,
};
use crate::intervention::{Hooks, Layer};

/// Every activation preserves shape and finite values on representative inputs.
#[test]
fn activations_are_finite_and_shape_preserving() {
    let device = Default::default();
    for activation in [
        Activation::Relu,
        Activation::Silu,
        Activation::Gelu,
        Activation::GeluTanh,
        Activation::ReluSquared,
    ] {
        let input = Tensor::<Flex, 2>::from_floats([[-2.0, -0.0, 1.0, 3.0]], &device);
        let output = apply_activation(input, activation);
        assert_eq!(output.dims(), [1, 4]);
        let output = output.to_data();
        let values = output
            .as_slice::<f32>()
            .expect("Flex f32 data should expose an f32 slice");
        assert!(values.iter().all(|value| value.is_finite()));
    }
}

/// Squared ReLU exactly removes negative values before squaring.
#[test]
fn relu_squared_matches_definition() {
    let device = Default::default();
    let output = apply_activation(
        Tensor::<Flex, 1>::from_floats([-2.0, 0.0, 3.0], &device),
        Activation::ReluSquared,
    )
    .to_data();
    assert_eq!(
        output
            .as_slice::<f32>()
            .expect("Flex f32 data should expose an f32 slice"),
        &[0.0, 0.0, 9.0]
    );
}

/// Gated MLP configuration rejects zero-width projections before initialization.
#[test]
fn invalid_projection_width_is_rejected() {
    let error = DenseFeedForwardConfig::new(0, 8)
        .validate()
        .expect_err("zero residual width must fail");
    assert!(error.to_string().contains("hidden_size"));

    let error = GatedFeedForwardConfig::new(8, 0)
        .validate()
        .expect_err("zero intermediate width must fail");
    assert!(error.to_string().contains("intermediate_size"));
}

/// A valid block preserves the residual-stream shape.
#[test]
fn feed_forward_preserves_residual_shape() {
    let device = Default::default();
    let module = GatedFeedForwardConfig::new(4, 8)
        .init::<Flex>(&device)
        .expect("valid block should initialize");
    let output = module
        .forward(Tensor::<Flex, 3>::zeros([2, 3, 4], &device))
        .unwrap();
    assert_eq!(output.dims(), [2, 3, 4]);
}

/// Dense ReLU uses one input branch and zeros negative activations exactly.
#[test]
fn dense_relu_matches_two_projection_definition() {
    let device = Default::default();
    let mut module = DenseFeedForwardConfig::new(2, 2)
        .init::<Flex>(&device)
        .expect("valid dense block");
    module.up_proj.weight = Param::from_tensor(Tensor::eye(2, &device));
    module.down_proj.weight = Param::from_tensor(Tensor::eye(2, &device));

    let output = module
        .forward(Tensor::<Flex, 2>::from_floats([[-2.0, 3.0]], &device))
        .unwrap()
        .into_data();
    output.assert_approx_eq::<f32>(
        &TensorData::from([[0.0, 3.0]]),
        burn::tensor::Tolerance::absolute(1e-6),
    );
}

/// The bias builder controls both projections, including rectangular broadcasts.
#[test]
fn dense_bias_builder_matches_independent_affine_relu_squared_equations() {
    let device = Default::default();
    let inputs = [[1.0_f32, -2.0], [-3.0, 0.5]];
    let up = [[1.0_f32, -1.0], [0.5, 2.0], [-1.0, 0.25]];
    let down = [[1.0_f32, 2.0, -1.0], [-0.5, 0.25, 1.5]];
    let up_bias = [0.75_f32, -0.5, 1.25];
    let down_bias = [-0.75_f32, 0.125];
    for enabled in [false, true] {
        // Explicitly undo a previous true setting as well as enabling biases.
        let config = DenseFeedForwardConfig::new(2, 3)
            .with_activation(Activation::ReluSquared)
            .with_bias(true)
            .with_bias(enabled);
        let mut block = config
            .init::<Flex>(&device)
            .expect("rectangular dense block");
        assert_eq!(block.up_proj.bias.is_some(), enabled);
        assert_eq!(block.down_proj.bias.is_some(), enabled);
        // Direct parameter assignment uses Burn's runtime [input, output]
        // layout, whereas the scalar equations below keep [output, input].
        block.up_proj.weight = Param::from_tensor(Tensor::from_floats(up, &device).transpose());
        block.down_proj.weight = Param::from_tensor(Tensor::from_floats(down, &device).transpose());
        if enabled {
            assert_eq!(block.up_proj.bias.as_ref().unwrap().val().dims(), [3]);
            assert_eq!(block.down_proj.bias.as_ref().unwrap().val().dims(), [2]);
            block.up_proj.bias = Some(Param::from_tensor(Tensor::from_floats(up_bias, &device)));
            block.down_proj.bias =
                Some(Param::from_tensor(Tensor::from_floats(down_bias, &device)));
        }
        let expected = inputs
            .iter()
            .flat_map(|input| {
                let hidden: Vec<f64> = up
                    .iter()
                    .zip(up_bias)
                    .map(|(row, bias)| {
                        let affine = row
                            .iter()
                            .zip(input)
                            .map(|(&weight, &value)| f64::from(weight) * f64::from(value))
                            .sum::<f64>()
                            + if enabled { f64::from(bias) } else { 0.0 };
                        affine.max(0.0).powi(2)
                    })
                    .collect();
                down.iter().zip(down_bias).map(move |(row, bias)| {
                    row.iter()
                        .zip(&hidden)
                        .map(|(&weight, &value)| f64::from(weight) * value)
                        .sum::<f64>()
                        + if enabled { f64::from(bias) } else { 0.0 }
                })
            })
            .collect::<Vec<_>>();
        let input = Tensor::<Flex, 2>::from_floats(inputs, &device);
        let flat = block
            .forward(input.clone())
            .expect("matrix forward")
            .into_data();
        let batched = block
            .forward(input.reshape([1, 2, 2]))
            .expect("batched forward")
            .into_data();
        assert_eq!(flat.shape, burn::tensor::Shape::from([2, 2]));
        assert_eq!(batched.shape, burn::tensor::Shape::from([1, 2, 2]));
        for actual in [flat, batched] {
            for (actual, expected) in actual.to_vec::<f32>().unwrap().into_iter().zip(&expected) {
                assert!(actual.is_finite());
                assert!((f64::from(actual) - expected).abs() < 1e-6);
            }
        }
        assert!(expected.iter().any(|value| *value < 0.0));
        assert!((expected[0] - expected[2]).abs() > 1.0);
    }
}

/// Each activation matches its closed form at representative points, distinguishing the GELU variants.
#[test]
fn activations_match_closed_form_goldens() {
    let device = Default::default();
    let input = Tensor::<Flex, 1>::from_floats([-2.0, -1.0, 0.0, 1.0, 3.0], &device);
    let erf_gelu = |x: f64| 0.5 * x * (1.0 + erf(x / 2.0_f64.sqrt()));
    let tanh_gelu = |x: f64| {
        0.5 * x
            * (1.0 + ((2.0 / core::f64::consts::PI).sqrt() * (x + 0.044_715 * x.powi(3))).tanh())
    };
    let silu = |x: f64| x / (1.0 + (-x).exp());
    let relu = |x: f64| x.max(0.0);
    for (activation, oracle) in [
        (Activation::Relu, &relu as &dyn Fn(f64) -> f64),
        (Activation::Silu, &silu),
        (Activation::Gelu, &erf_gelu),
        (Activation::GeluTanh, &tanh_gelu),
        (Activation::ReluSquared, &|x: f64| relu(x).powi(2)),
    ] {
        let actual = apply_activation(input.clone(), activation)
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        for (x, actual) in [-2.0, -1.0, 0.0, 1.0, 3.0].into_iter().zip(actual) {
            assert!(
                (f64::from(actual) - oracle(x)).abs() < 2e-6,
                "{activation:?}({x}) = {actual}"
            );
        }
    }
    // Published references: GELU(1) = 0.8413447, tanh-GELU(1) = 0.8411920, SiLU(-2) = -0.2384058.
    assert!((erf_gelu(1.0) - 0.841_344_7).abs() < 1e-7);
    assert!((tanh_gelu(1.0) - 0.841_192_0).abs() < 1e-7);
    assert!((silu(-2.0) + 0.238_405_8).abs() < 1e-7);
    // The two GELU variants differ by more than the test tolerance at x = 1.
    assert!((erf_gelu(1.0) - tanh_gelu(1.0)).abs() > 1e-4);
}

/// Maclaurin-series error function, accurate to about 1e-12 on `[-3, 3]`.
fn erf(x: f64) -> f64 {
    let mut term = x;
    let mut sum = x;
    for n in 1..60 {
        term *= -x * x / n as f64;
        sum += term / (2 * n + 1) as f64;
    }
    sum * 2.0 / core::f64::consts::PI.sqrt()
}

/// Burn's `gelu_approximate` reproduces the previous hand-written tanh-GELU bit for bit.
#[test]
fn tanh_gelu_matches_previous_operation_graph_exactly() {
    use burn::tensor::DType;

    let device = Default::default();
    let values = (0..64)
        .map(|index| index as f32 * 0.17 - 5.0)
        .collect::<Vec<_>>();
    for dtype in [DType::F32, DType::F16] {
        let input = Tensor::<Flex, 1>::from_data(TensorData::new(values.clone(), [64]), &device)
            .cast(dtype);
        let previous = {
            let cubic = input.clone().powf_scalar(3.0);
            let inner = (input.clone() + cubic * 0.044_715) * 0.797_884_560_802_865_4;
            input.clone() * (inner.tanh() + 1.0) * 0.5
        };
        assert_eq!(
            apply_activation(input, Activation::GeluTanh).into_data(),
            previous.into_data()
        );
    }
}

/// Builds a 2-to-2 gated block with distinct gate and up weights and identity output.
fn distinct_gated_block() -> super::GatedFeedForward<Flex> {
    let device = Default::default();
    let mut block = GatedFeedForwardConfig::new(2, 2)
        .init::<Flex>(&device)
        .unwrap();
    // Runtime layout is [input, output]: gate(x) = [x0 + 2 x1, -x1], up(x) = [3 x0, x0 - x1].
    block.gate_proj.weight =
        Param::from_tensor(Tensor::from_floats([[1.0, 0.0], [2.0, -1.0]], &device));
    block.up_proj.weight =
        Param::from_tensor(Tensor::from_floats([[3.0, 1.0], [0.0, -1.0]], &device));
    block.down_proj.weight = Param::from_tensor(Tensor::eye(2, &device));
    block
}

/// The activation applies to the gate branch only and multiplies the up branch.
#[test]
fn gated_block_applies_activation_to_gate_branch_only() {
    let device = Default::default();
    let block = distinct_gated_block();
    let (x0, x1) = (0.5_f64, -1.5_f64);
    let silu = |x: f64| x / (1.0 + (-x).exp());
    let expected = [silu(x0 + 2.0 * x1) * (3.0 * x0), silu(-x1) * (x0 - x1)];
    let actual = block
        .forward(Tensor::<Flex, 2>::from_floats([[0.5, -1.5]], &device))
        .unwrap()
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    for (actual, expected) in actual.into_iter().zip(expected) {
        assert!((f64::from(actual) - expected).abs() < 1e-6);
    }
    // Exchanging the roles would compute silu(up) * gate instead.
    let swapped = silu(3.0 * x0) * (x0 + 2.0 * x1);
    assert!((swapped - expected[0]).abs() > 0.1);
}

/// Hooked forwards equal ordinary forwards and expose pre-activation and activation in order.
#[test]
fn hooked_mlp_forward_emits_canonical_points() {
    let device = Default::default();
    let layer = Layer::decoder(2);
    let input = Tensor::<Flex, 3>::from_floats([[[0.5, -1.5], [2.0, 0.25]]], &device);
    let gated = distinct_gated_block();
    let mut dense = DenseFeedForwardConfig::new(2, 2)
        .with_activation(Activation::Silu)
        .init::<Flex>(&device)
        .unwrap();
    dense.up_proj = gated.gate_proj.clone();
    dense.down_proj = gated.down_proj.clone();
    assert_eq!(
        mlp_hook_points(layer),
        [
            layer.mlp_pre_activation().info(),
            layer.mlp_activation().info()
        ]
    );
    for gated_block in [false, true] {
        let forward = |hooks: &mut Hooks<'_, Flex>| {
            if gated_block {
                gated.forward_with_hooks(input.clone(), layer, hooks)
            } else {
                dense.forward_with_hooks(input.clone(), layer, hooks)
            }
        };
        let ordinary = if gated_block {
            gated.forward(input.clone()).unwrap()
        } else {
            dense.forward(input.clone()).unwrap()
        };
        let mut hooks = Hooks::new();
        let pre_activation = hooks.capture(layer.mlp_pre_activation());
        let activation = hooks.capture(layer.mlp_activation());
        let observed = hooks.run(mlp_hook_points(layer), forward).unwrap();
        assert_eq!(observed.into_data(), ordinary.into_data());
        let pre_activation = hooks.get(&pre_activation).unwrap();
        let expected_pre = crate::operations::linear(input.clone(), &gated.gate_proj).unwrap();
        assert_eq!(pre_activation.clone().into_data(), expected_pre.into_data());
        assert_eq!(
            hooks.get(&activation).unwrap().into_data(),
            apply_activation(pre_activation, Activation::Silu).into_data()
        );
        // Ablating the activation removes the whole (bias-free) output.
        let mut ablation = Hooks::new();
        ablation.edit(layer.mlp_activation(), |activation| activation.zeros_like());
        let ablated = ablation.run(mlp_hook_points(layer), forward).unwrap();
        assert!(
            ablated
                .into_data()
                .to_vec::<f32>()
                .unwrap()
                .iter()
                .all(|value| *value == 0.0)
        );
    }
}

/// The gpt-oss clamped unit matches its closed form, including both clamps.
#[test]
fn clamped_swiglu_matches_gpt_oss_closed_form() {
    let device = Default::default();
    let glu = ClampedSwiGlu::new(7.0)
        .with_alpha(1.702)
        .with_up_offset(1.0);
    let gate = [-1.0_f64, 0.5, 9.0, 3.0];
    let up = [-8.0_f64, 2.0, 3.0, 7.5];
    let actual = glu
        .combine(
            Tensor::<Flex, 1>::from_floats(gate.map(|value| value as f32), &device),
            Tensor::<Flex, 1>::from_floats(up.map(|value| value as f32), &device),
        )
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    for ((gate, up), actual) in gate.into_iter().zip(up).zip(actual) {
        let gate = gate.min(7.0);
        let up = up.clamp(-7.0, 7.0) + 1.0;
        let expected = up * gate / (1.0 + (-1.702 * gate).exp());
        assert!(
            (f64::from(actual) - expected).abs() < 1e-5,
            "{actual} vs {expected}"
        );
    }
}

/// The clamped unit reproduces Hugging Face's gpt-oss `GptOssExperts._apply_gate`.
///
/// Pinned with transformers 5.2.0 and torch 2.10.0 (F32, CPU) on the
/// interleaved `gate_up` row below, whose even columns are gates and odd
/// columns up projections; values beyond `+-7` exercise both clamps.
#[test]
fn clamped_swiglu_matches_pinned_gpt_oss_reference() {
    let device = Default::default();
    let gate_up = [
        -9.0_f32, 8.0, -1.0, -8.0, 0.5, 2.0, 9.0, 7.5, 3.0, -0.25, 6.9, -7.1,
    ];
    let reference = [
        -1.602_542_76e-5_f64,
        0.925_225_377,
        1.051_165_34,
        55.999_626_2,
        2.236_446_62,
        -41.399_669_6,
    ];
    let gate = Tensor::<Flex, 1>::from_floats(
        core::array::from_fn::<f32, 6, _>(|index| gate_up[2 * index]),
        &device,
    );
    let up = Tensor::<Flex, 1>::from_floats(
        core::array::from_fn::<f32, 6, _>(|index| gate_up[2 * index + 1]),
        &device,
    );
    let actual = ClampedSwiGlu::new(7.0)
        .with_alpha(1.702)
        .with_up_offset(1.0)
        .combine(gate, up)
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    for (actual, expected) in actual.into_iter().zip(reference) {
        let actual = f64::from(actual);
        assert!(
            (actual - expected).abs() <= 1e-6 + 1e-6 * expected.abs(),
            "{actual} vs {expected}"
        );
    }
}

/// With alpha one and no offset, a non-binding limit reproduces ordinary SwiGLU.
#[test]
fn unclamped_unit_equals_standard_swiglu_block() {
    let device = Default::default();
    let block = distinct_gated_block();
    let input = Tensor::<Flex, 3>::from_floats([[[0.5, -1.5], [2.0, 0.25]]], &device);
    let clamped = block
        .forward_clamped(input.clone(), &ClampedSwiGlu::new(1e6))
        .unwrap();
    clamped.into_data().assert_approx_eq::<f32>(
        &block.forward(input.clone()).unwrap().into_data(),
        burn::tensor::Tolerance::absolute(1e-6),
    );
    // A binding limit changes the result.
    let tight = block
        .forward_clamped(input.clone(), &ClampedSwiGlu::new(0.5))
        .unwrap()
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    let open = block
        .forward(input)
        .unwrap()
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    assert!(
        tight
            .iter()
            .zip(&open)
            .any(|(tight, open)| (tight - open).abs() > 1e-3)
    );
}

/// Invalid clamp parameters are rejected with context and configurations round-trip.
#[test]
fn clamped_swiglu_validates_and_serializes() {
    for (glu, field) in [
        (ClampedSwiGlu::new(0.0), "limit"),
        (ClampedSwiGlu::new(f64::INFINITY), "limit"),
        (ClampedSwiGlu::new(7.0).with_alpha(0.0), "alpha"),
        (ClampedSwiGlu::new(7.0).with_alpha(f64::NAN), "alpha"),
        (
            ClampedSwiGlu::new(7.0).with_up_offset(f64::INFINITY),
            "up_offset",
        ),
    ] {
        let error = glu.validate().unwrap_err();
        assert!(error.to_string().contains(field), "{error}");
        let block = distinct_gated_block();
        assert!(
            block
                .forward_clamped(Tensor::<Flex, 2>::zeros([1, 2], &Default::default()), &glu)
                .is_err()
        );
    }
    let glu = ClampedSwiGlu::new(7.0)
        .with_alpha(1.702)
        .with_up_offset(1.0);
    let json = serde_json::to_string(&glu).unwrap();
    assert_eq!(serde_json::from_str::<ClampedSwiGlu>(&json).unwrap(), glu);
}

/// Gradients reach every projection through the clamped and hooked forwards.
#[cfg(feature = "autodiff")]
#[test]
fn clamped_and_hooked_forwards_are_differentiable() {
    use burn::backend::Autodiff;
    /// Differentiable CPU backend for the feed-forward gradient smoke test.
    type Ad = Autodiff<Flex>;
    let device = Default::default();
    let block = GatedFeedForwardConfig::new(2, 3)
        .init::<Ad>(&device)
        .unwrap();
    let input = Tensor::<Ad, 3>::from_floats([[[0.5, -1.5]]], &device);
    let glu = ClampedSwiGlu::new(7.0)
        .with_alpha(1.702)
        .with_up_offset(1.0);
    let gradients = block
        .forward_clamped(input.clone(), &glu)
        .unwrap()
        .sum()
        .backward();
    for gradient in [
        block.gate_proj.weight.grad(&gradients),
        block.up_proj.weight.grad(&gradients),
        block.down_proj.weight.grad(&gradients),
    ] {
        assert!(gradient.is_some());
    }
    let layer = Layer::decoder(0);
    let gradients = Hooks::new()
        .run(mlp_hook_points(layer), |hooks| {
            block.forward_with_hooks(input, layer, hooks)
        })
        .unwrap()
        .sum()
        .backward();
    assert!(block.gate_proj.weight.grad(&gradients).is_some());
}

/// A failing edit at either MLP point aborts the hooked forward with that point's error.
///
/// The pre-activation and activation hooks run in sequence. An observer on the
/// point that does not fail records whether that point was emitted: a failure
/// at the pre-activation point must stop before the activation point is
/// emitted, and a failure at the activation point must surface after the
/// pre-activation point was emitted exactly once. Both the dense and the gated
/// block propagate the callback's error with the failing point's name instead
/// of continuing with an unedited tensor.
#[test]
fn hooked_mlp_forward_propagates_point_failures() {
    use core::cell::Cell;

    let device = Default::default();
    let layer = Layer::decoder(1);
    let input = Tensor::<Flex, 3>::from_floats([[[0.5, -1.5]]], &device);
    let gated = distinct_gated_block();
    let mut dense = DenseFeedForwardConfig::new(2, 2)
        .with_activation(Activation::Silu)
        .init::<Flex>(&device)
        .unwrap();
    dense.up_proj = gated.gate_proj.clone();
    dense.down_proj = gated.down_proj.clone();
    for gated_block in [false, true] {
        for fail_at_activation in [false, true] {
            let emitted = Cell::new(0_usize);
            let mut hooks = Hooks::new();
            let pre_activation = hooks.capture(layer.mlp_pre_activation());
            let activation_capture = hooks.capture(layer.mlp_activation());
            let (failing, observed) = if fail_at_activation {
                (layer.mlp_activation(), layer.mlp_pre_activation())
            } else {
                (layer.mlp_pre_activation(), layer.mlp_activation())
            };
            hooks.observe(observed, |_| emitted.set(emitted.get() + 1));
            hooks.try_edit(failing.clone(), |_| {
                Err(crate::Error::configuration("test.edit", "refused"))
            });
            let error = hooks
                .run(mlp_hook_points(layer), |hooks| {
                    if gated_block {
                        gated.forward_with_hooks(input.clone(), layer, hooks)
                    } else {
                        dense.forward_with_hooks(input.clone(), layer, hooks)
                    }
                })
                .unwrap_err();
            let crate::Error::Intervention { point, reason, .. } = &error else {
                panic!("expected an intervention error, got {error:?}");
            };
            assert_eq!(*point, failing.info().name);
            assert!(reason.contains("refused"), "{reason}");
            // The earlier point ran once before the later one failed; a
            // failure at the earlier point never reaches the later one.
            assert_eq!(
                emitted.get(),
                usize::from(fail_at_activation),
                "gated={gated_block}, fail_at_activation={fail_at_activation}"
            );
            // A failed run releases every capture, including the one that ran.
            assert!(hooks.get(&pre_activation).is_err());
            assert!(hooks.get(&activation_capture).is_err());
        }
    }
}
