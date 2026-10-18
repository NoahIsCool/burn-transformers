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

/// In F32, tanh-GELU is Burn's `gelu_approximate`, the previous hand-written graph, bit for bit.
#[test]
fn tanh_gelu_matches_previous_operation_graph_exactly_in_f32() {
    let device = Default::default();
    let values = (0..64)
        .map(|index| index as f32 * 0.17 - 5.0)
        .collect::<Vec<_>>();
    let input = Tensor::<Flex, 1>::from_data(TensorData::new(values, [64]), &device);
    let previous = composed_tanh_gelu(input.clone());
    assert_eq!(
        apply_activation(input, Activation::GeluTanh).into_data(),
        previous.into_data()
    );
}

/// The hand-written tanh-GELU graph, which Burn's `gelu_approximate` reproduces in every dtype.
fn composed_tanh_gelu(input: Tensor<Flex, 1>) -> Tensor<Flex, 1> {
    let cubic = input.clone().powf_scalar(3.0);
    let inner = (input.clone() + cubic * 0.044_715) * 0.797_884_560_802_865_4;
    input * (inner.tanh() + 1.0) * 0.5
}

/// Exact tanh-GELU `0.5 x (1 + tanh(sqrt(2 / pi) (x + 0.044715 x^3)))` in F64.
fn tanh_gelu_f64(x: f64) -> f64 {
    0.5 * x * (1.0 + ((2.0 / core::f64::consts::PI).sqrt() * (x + 0.044_715 * x.powi(3))).tanh())
}

/// Returns the largest error a once-rounded F32 evaluation of tanh-GELU may have at `x` in `dtype`.
///
/// Rounding the exact value `y` to a format with `p` significand bits errs
/// by at most half a unit in its last place, `2^(floor(log2 |y|) + 1 - p)`,
/// floored at the format's smallest subnormal step. F32 evaluation adds at
/// most a few F32 roundings, about `6e-8` relative each, of `|x|` (the
/// cancellation in `1 + tanh` makes the error absolute in `x`, not relative
/// in `y`); `1e-6 (|x| + |y|)` bounds it with margin.
fn once_rounded_bound(x: f64, exact: f64, dtype: burn::tensor::DType) -> f64 {
    use burn::tensor::DType;

    let (significand_bits, smallest_step) = match dtype {
        DType::BF16 => (8, 2.0_f64.powi(-133)),
        DType::F16 => (11, 2.0_f64.powi(-24)),
        other => unreachable!("no half-precision format {other:?}"),
    };
    let unit = if exact == 0.0 {
        smallest_step
    } else {
        2.0_f64
            .powi(exact.abs().log2().floor() as i32 + 1 - significand_bits)
            .max(smallest_step)
    };
    0.5 * unit + 1e-6 * (x.abs() + exact.abs())
}

/// Reads a one-dimensional Flex tensor of any float dtype as F64 values.
fn as_f64(tensor: Tensor<Flex, 1>) -> Vec<f64> {
    tensor
        .cast(burn::tensor::DType::F32)
        .into_data()
        .to_vec::<f32>()
        .unwrap()
        .into_iter()
        .map(f64::from)
        .collect()
}

/// In F16 and BF16, tanh-GELU is evaluated in F32 and rounded once, as PyTorch's `gelu(approximate="tanh")` is.
///
/// Every input of a grid exactly representable in both formats must land
/// within half a unit in the last place of the exact value (plus F32
/// evaluation error), and must equal the F32 result rounded to the dtype.
/// The composed graph, rounded after every operation and with its constants
/// rounded to BF16, misses the bound by about twelve half-units at `x = -2`
/// (`-0.046875` against `-0.0454`), so this test detects it.
#[test]
fn half_precision_tanh_gelu_rounds_the_f32_value_once() {
    use burn::tensor::DType;

    let device = Default::default();
    // Multiples of 1/16 in [-8, 8]: exact in F16 and BF16 alike.
    let values = (-128..=128)
        .map(|step| step as f32 / 16.0)
        .collect::<Vec<_>>();
    let count = values.len();
    for dtype in [DType::BF16, DType::F16] {
        let input = Tensor::<Flex, 1>::from_data(TensorData::new(values.clone(), [count]), &device)
            .cast(dtype);
        let actual = apply_activation(input.clone(), Activation::GeluTanh);
        assert_eq!(actual.dtype(), dtype, "the output keeps the input dtype");
        let widened =
            apply_activation(input.clone().cast(DType::F32), Activation::GeluTanh).cast(dtype);
        assert_eq!(
            actual.clone().into_data(),
            widened.into_data(),
            "{dtype:?}: the F32 evaluation rounded once"
        );
        for (&x, actual) in values.iter().zip(as_f64(actual)) {
            let x = f64::from(x);
            let exact = tanh_gelu_f64(x);
            let bound = once_rounded_bound(x, exact, dtype);
            assert!(
                (actual - exact).abs() <= bound,
                "{dtype:?} tanh-GELU({x}) = {actual}, exact {exact}, bound {bound:e}"
            );
        }
    }
    // The composed graph in BF16 fails the same bound: its rounding is the defect this guards against.
    let minus_two = Tensor::<Flex, 1>::from_floats([-2.0], &device).cast(DType::BF16);
    let composed = as_f64(composed_tanh_gelu(minus_two))[0];
    let exact = tanh_gelu_f64(-2.0);
    assert_eq!(composed, -0.046_875);
    assert!((composed - exact).abs() > 10.0 * once_rounded_bound(-2.0, exact, DType::BF16));
}

proptest::proptest! {
    #![proptest_config(proptest::test_runner::Config::with_cases(256))]

    /// For any finite half-precision input, tanh-GELU lies within half a unit in the last place of the exact value.
    #[test]
    fn half_precision_tanh_gelu_is_once_rounded_for_any_input(
        x in -12.0_f32..12.0,
        bf16 in proptest::bool::ANY,
    ) {
        use burn::tensor::DType;

        let dtype = if bf16 { DType::BF16 } else { DType::F16 };
        let input = Tensor::<Flex, 1>::from_floats([x], &Default::default()).cast(dtype);
        // The input as the half-precision format holds it.
        let x = as_f64(input.clone())[0];
        let actual = as_f64(apply_activation(input, Activation::GeluTanh))[0];
        let exact = tanh_gelu_f64(x);
        let bound = once_rounded_bound(x, exact, dtype);
        proptest::prop_assert!(
            (actual - exact).abs() <= bound,
            "{:?} tanh-GELU({}) = {}, exact {}, bound {:e}", dtype, x, actual, exact, bound
        );
    }

    /// For any small shape and finite values, half-precision SiLU is the widened F32 graph rounded once.
    ///
    /// The rank-3 `[batch, token, intermediate]` layout is the one gated MLPs
    /// use; the shape and dtype survive, every element equals Burn's F32
    /// `activation::silu` rounded to the dtype, and the crate-wide helper
    /// agrees with the activation dispatch.
    #[test]
    fn half_precision_silu_is_the_widened_graph_for_any_shape(
        shape in (1_usize..4, 1_usize..5, 1_usize..9),
        seed_values in proptest::collection::vec(-40.0_f32..40.0, 96),
        bf16 in proptest::bool::ANY,
    ) {
        use burn::tensor::{DType, activation};

        let dtype = if bf16 { DType::BF16 } else { DType::F16 };
        let dims = [shape.0, shape.1, shape.2];
        let count = dims.iter().product::<usize>();
        let values = seed_values[..count].to_vec();
        let input = Tensor::<Flex, 3>::from_data(TensorData::new(values, dims), &Default::default())
            .cast(dtype);
        let actual = apply_activation(input.clone(), Activation::Silu);
        proptest::prop_assert_eq!(actual.dims(), dims);
        proptest::prop_assert_eq!(actual.dtype(), dtype);
        let widened = activation::silu(input.clone().cast(DType::F32)).cast(dtype);
        proptest::prop_assert_eq!(actual.clone().into_data(), widened.into_data());
        proptest::prop_assert_eq!(super::silu(input).into_data(), actual.into_data());
    }
}

/// Exact SiLU `x / (1 + exp(-x))` in F64.
fn silu_f64(x: f64) -> f64 {
    x / (1.0 + (-x).exp())
}

/// Returns the largest error a once-rounded F32 evaluation of SiLU may have when its exact value is `exact`, in `dtype`.
///
/// Rounding to the format errs by at most half a unit in the last place of
/// `exact` (the format's subnormal step below its normal range). The F32
/// evaluation of `x * sigmoid(x)` adds a few F32 roundings relative to `|y|`
/// (the exponential, the sum, the quotient and the product, about `6e-8`
/// each; SiLU has no cancellation, since `1 + exp(-x) >= 1`), bounded by
/// `1e-6 |y|`. Below `x = -87` the F32 `exp(x)` is subnormal and carries an
/// absolute error of at most `2^-149`, which the product scales by
/// `|x| < 104` to below `2^-142`; `2^-140` covers it. Both slacks are far
/// smaller than half a unit of either format (BF16's smallest half-step is
/// `2^-134`), so a result one unit away from the correctly rounded value
/// fails this bound.
fn silu_once_rounded_bound(exact: f64, dtype: burn::tensor::DType) -> f64 {
    use burn::tensor::DType;

    let (significand_bits, smallest_step) = match dtype {
        DType::BF16 => (8, 2.0_f64.powi(-133)),
        DType::F16 => (11, 2.0_f64.powi(-24)),
        other => unreachable!("no half-precision format {other:?}"),
    };
    let unit = if exact == 0.0 {
        smallest_step
    } else {
        2.0_f64
            .powi(exact.abs().log2().floor() as i32 + 1 - significand_bits)
            .max(smallest_step)
    };
    0.5 * unit + 1e-6 * exact.abs() + 2.0_f64.powi(-140)
}

/// Returns every finite value of a 16-bit float `dtype` as a Flex tensor of that dtype, and the values as F64.
///
/// The values are decoded from all 65,536 bit patterns, widened to F32 (an
/// exact conversion) to build the tensor, and cast to `dtype`; the cast is
/// checked to reproduce every value, so the tensor holds each finite value
/// exactly once.
fn every_finite_half(dtype: burn::tensor::DType) -> (Tensor<Flex, 1>, Vec<f64>) {
    use burn::tensor::{DType, bf16, f16};

    let widened = (0..=u16::MAX)
        .map(|bits| match dtype {
            DType::BF16 => bf16::from_bits(bits).to_f32(),
            DType::F16 => f16::from_bits(bits).to_f32(),
            other => unreachable!("no half-precision format {other:?}"),
        })
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    let count = widened.len();
    let tensor = Tensor::<Flex, 1>::from_data(
        TensorData::new(widened.clone(), [count]),
        &Default::default(),
    )
    .cast(dtype);
    assert_eq!(tensor.dtype(), dtype);
    let exact = widened.into_iter().map(f64::from).collect::<Vec<_>>();
    assert_eq!(
        as_f64(tensor.clone()),
        exact,
        "{dtype:?} holds every value exactly"
    );
    (tensor, exact)
}

/// In F32 and F64, SiLU is Burn's `activation::silu`, the two-operation graph `x * sigmoid(x)`, bit for bit.
///
/// Full-precision numerics, and therefore every F32 golden and gradient, are
/// untouched by the half-precision evaluation rule. The grid spans zero,
/// subnormal-producing negative inputs, the exponential's F32 underflow near
/// `-88` and `-104`, and large magnitudes.
#[test]
fn silu_is_burns_operation_graph_bit_for_bit_in_full_precision() {
    use burn::tensor::{DType, activation};

    let device = Default::default();
    let mut values = (0..64)
        .map(|index| index as f32 * 0.29 - 9.0)
        .collect::<Vec<_>>();
    values.extend([
        0.0, -0.0, 1e-30, -1e-30, -87.5, -90.0, -103.5, -110.0, 88.0, 1e30, -1e30,
    ]);
    let count = values.len();
    for dtype in [DType::F32, DType::F64] {
        let input = Tensor::<Flex, 1>::from_data(TensorData::new(values.clone(), [count]), &device)
            .cast(dtype);
        let actual = apply_activation(input.clone(), Activation::Silu);
        assert_eq!(actual.dtype(), dtype);
        assert_eq!(
            actual.into_data(),
            (input.clone() * activation::sigmoid(input)).into_data(),
            "{dtype:?}"
        );
    }
}

/// On every finite F16 and BF16 input, SiLU is the F32 value rounded once, within half a unit of the exact value.
///
/// This is PyTorch's `silu` convention (its CPU kernel agrees with the
/// once-rounded F32 value on every finite BF16 input and on all but one F16
/// near-tie). All 65,280 finite BF16 and 63,488 finite F16 values are checked
/// against the exact F64 SiLU with [`silu_once_rounded_bound`], so the test
/// covers zeros, subnormals, both overflow ends and every binade.
#[test]
fn half_precision_silu_is_once_rounded_on_every_finite_input() {
    use burn::tensor::{DType, activation};

    for dtype in [DType::BF16, DType::F16] {
        let (input, xs) = every_finite_half(dtype);
        let actual = apply_activation(input.clone(), Activation::Silu);
        assert_eq!(actual.dtype(), dtype, "the output keeps the input dtype");
        let widened = activation::silu(input.clone().cast(DType::F32)).cast(dtype);
        assert_eq!(
            actual.clone().into_data(),
            widened.into_data(),
            "{dtype:?}: the F32 evaluation rounded once"
        );
        let actual = as_f64(actual);
        assert_eq!(actual.len(), xs.len());
        for (&x, &actual) in xs.iter().zip(&actual) {
            let exact = silu_f64(x);
            let bound = silu_once_rounded_bound(exact, dtype);
            assert!(
                (actual - exact).abs() <= bound,
                "{dtype:?} SiLU({x:e}) = {actual:e}, exact {exact:e}, bound {bound:e}"
            );
        }
    }
}

/// Burn's composed half-precision graph `x * sigmoid(x)` misses the once-rounded bound, which the activation meets.
///
/// Rounding the sigmoid before the product moves BF16 `SiLU(2.8125)` to
/// `2.640625` and F16 `SiLU(6.6875)` to `6.67578125`, one unit below the
/// correctly rounded `2.65625` and `6.6796875` (the exact values are
/// `2.65317` and `6.67918`). The once-rounded values are PyTorch 2.10.0's
/// `F.silu` on CPU for the same inputs, and the composed ones equal its
/// `x * torch.sigmoid(x)`.
#[test]
fn composed_half_precision_silu_misses_the_once_rounded_bound() {
    use burn::tensor::{DType, activation};

    let device = Default::default();
    for (dtype, x, composed_expected, once_expected) in [
        (DType::BF16, 2.8125_f32, 2.640_625, 2.656_25),
        (DType::F16, 6.6875_f32, 6.675_781_25, 6.679_687_5),
    ] {
        let input = Tensor::<Flex, 1>::from_floats([x], &device).cast(dtype);
        let composed = as_f64(input.clone() * activation::sigmoid(input.clone()))[0];
        let once = as_f64(apply_activation(input, Activation::Silu))[0];
        let exact = silu_f64(f64::from(x));
        assert_eq!(
            (composed, once),
            (composed_expected, once_expected),
            "{dtype:?}"
        );
        let bound = silu_once_rounded_bound(exact, dtype);
        assert!((once - exact).abs() <= bound, "{dtype:?}");
        assert!(
            (composed - exact).abs() > 1.5 * bound,
            "{dtype:?}: the composed graph misses the once-rounded bound"
        );
    }
}

/// Gradients flow through the half-precision SiLU's F32 evaluation and match the derivative's closed form.
///
/// `d/dx SiLU(x) = sigmoid(x) (1 + x (1 - sigmoid(x)))`. The widening and
/// narrowing casts are differentiable, so a BF16 graph still trains through
/// the gate. The backward pass runs in F32 and its result is rounded once to
/// the BF16 input gradient, so it lies within half a BF16 unit (at most
/// `2^-8` relative) plus F32 evaluation error of the closed form.
#[cfg(feature = "autodiff")]
#[test]
fn half_precision_silu_is_differentiable() {
    use burn::{backend::Autodiff, tensor::DType};
    /// Differentiable CPU backend for the half-precision SiLU gradient check.
    type Ad = Autodiff<Flex>;

    let device = Default::default();
    let xs = [-3.0_f32, -0.5, 0.0, 0.75, 2.8125];
    let input = Tensor::<Ad, 1>::from_floats(xs, &device)
        .cast(DType::BF16)
        .require_grad();
    let gradients = apply_activation(input.clone(), Activation::Silu)
        .sum()
        .backward();
    let gradient = input.grad(&gradients).expect("SiLU input gradient");
    let gradient = gradient
        .cast(DType::F32)
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    for (x, gradient) in xs.into_iter().zip(gradient) {
        let x = f64::from(x);
        let sigmoid = 1.0 / (1.0 + (-x).exp());
        let expected = sigmoid * (1.0 + x * (1.0 - sigmoid));
        assert!(
            (f64::from(gradient) - expected).abs() <= 2.0_f64.powi(-8) * expected.abs() + 1e-6,
            "d SiLU / dx at {x} = {gradient}, expected {expected}"
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

/// Pins a Burn Flex 0.21 deviation: its F32-to-BF16 tensor cast ignores the three lowest bits of an F32 value.
///
/// Correct rounding to nearest, ties to even (PyTorch's, and `half`'s
/// `bf16::from_f32`) rounds an F32 value whose discarded low half is
/// `0x8001` to `0x8007` up: it lies above the midpoint. Flex rounds it down
/// whenever the kept half is even, as if those bits were zero and the value
/// an exact tie. Every half-precision evaluation that rounds an F32 result
/// to BF16 on Flex (`evaluate_in_f32`, the F32 norms, the F32 softmax) can
/// therefore differ from PyTorch by one unit in the last place on such
/// near-ties, about one value in 18,700 for uniformly distributed low bits.
/// Odd kept halves, F16 casts, and Flex's own elementwise BF16 kernels round
/// correctly. This test fails once Burn fixes the cast; the comparisons that
/// allow one unit for it should then be tightened.
#[test]
fn flex_bf16_cast_rounds_values_just_above_a_tie_down() {
    use burn::tensor::{DType, bf16};

    let device = Default::default();
    let values: Vec<f32> = [
        0x3efe_8001_u32,
        0x3efe_8007,
        0x3efe_8008,
        0x3eff_8001,
        0x3efe_7fff,
    ]
    .into_iter()
    .map(f32::from_bits)
    .collect();
    let count = values.len();
    let cast = Tensor::<Flex, 1>::from_data(TensorData::new(values.clone(), [count]), &device)
        .cast(DType::BF16)
        .cast(DType::F32)
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    let correct: Vec<f32> = values
        .iter()
        .map(|&value| bf16::from_f32(value).to_f32())
        .collect();
    let kept = |value: f32| value.to_bits() >> 16;
    // Even kept half, low half 0x8001 and 0x8007: Flex rounds down, correct rounding up.
    assert_eq!([kept(cast[0]), kept(cast[1])], [0x3efe, 0x3efe]);
    assert_eq!([kept(correct[0]), kept(correct[1])], [0x3eff, 0x3eff]);
    // 0x8008, an odd kept half, and a value below the midpoint round correctly.
    assert_eq!(cast[2..], correct[2..]);
    // The same value as the output of a Flex BF16 elementwise kernel rounds correctly.
    let sigmoid = burn::tensor::activation::sigmoid(
        Tensor::<Flex, 1>::from_floats([-0.011_718_75], &device).cast(DType::BF16),
    );
    let correct_sigmoid = bf16::from_f32(
        burn::tensor::activation::sigmoid(Tensor::<Flex, 1>::from_floats([-0.011_718_75], &device))
            .into_data()
            .to_vec::<f32>()
            .unwrap()[0],
    );
    assert_eq!(as_f64(sigmoid)[0], correct_sigmoid.to_f64());
}

/// On every finite F16 and BF16 input, tanh-GELU is the F32 value rounded once, within half a unit of the exact value.
///
/// This extends `half_precision_tanh_gelu_rounds_the_f32_value_once` from a
/// grid to all 65,280 finite BF16 and 63,488 finite F16 values, PyTorch's
/// `gelu(approximate="tanh")` convention (Gemma's `gelu_pytorch_tanh`).
/// Burn's own `gelu_approximate` evaluated in the half dtype misses the
/// once-rounded F32 value on 1,012 BF16 and 11,911 F16 inputs on Flex, which
/// is why the crate evaluates it in F32. The bound is [`once_rounded_bound`],
/// whose F32 slack is absolute in `x` because `1 + tanh` cancels for negative
/// inputs.
///
/// One range is pinned instead: BF16 shares F32's exponent range, and for
/// `x >= 2^127` Burn's F32 graph `x (1 + tanh(...)) / 2` overflows when it
/// doubles `x` before halving, so the result is `+inf`, where PyTorch's
/// kernel, `0.5 x (1 + tanh(...))`, halves first and returns `x`. These are
/// the 128 largest positive BF16 values, far beyond any activation a model
/// produces; F16 cannot hold them (its largest value is 65,504). Removing
/// the deviation needs a crate-owned graph with PyTorch's operation order.
#[test]
fn half_precision_tanh_gelu_is_once_rounded_on_every_finite_input() {
    use burn::tensor::{DType, activation};

    for dtype in [DType::BF16, DType::F16] {
        let (input, xs) = every_finite_half(dtype);
        let actual = apply_activation(input.clone(), Activation::GeluTanh);
        assert_eq!(actual.dtype(), dtype, "the output keeps the input dtype");
        let widened = activation::gelu_approximate(input.clone().cast(DType::F32)).cast(dtype);
        assert_eq!(
            actual.clone().into_data(),
            widened.into_data(),
            "{dtype:?}: the F32 evaluation rounded once"
        );
        let composed = as_f64(activation::gelu_approximate(input));
        let actual = as_f64(actual);
        let mut composed_misses = 0;
        let mut overflowing = 0;
        for ((&x, &actual), &composed) in xs.iter().zip(&actual).zip(&composed) {
            if x >= 2.0_f64.powi(127) {
                assert_eq!(
                    actual,
                    f64::INFINITY,
                    "{dtype:?} tanh-GELU({x:e}) overflows"
                );
                overflowing += 1;
                continue;
            }
            let exact = tanh_gelu_f64(x);
            let bound = once_rounded_bound(x, exact, dtype);
            assert!(
                (actual - exact).abs() <= bound,
                "{dtype:?} tanh-GELU({x:e}) = {actual:e}, exact {exact:e}, bound {bound:e}"
            );
            if composed.to_bits() != actual.to_bits() {
                composed_misses += 1;
            }
        }
        assert!(
            composed_misses > 0,
            "{dtype:?}: Burn's half-precision graph differs on some input"
        );
        let pinned = if dtype == DType::BF16 { 128 } else { 0 };
        assert_eq!(overflowing, pinned, "{dtype:?} inputs of at least 2^127");
    }
}
