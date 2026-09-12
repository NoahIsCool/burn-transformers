//! Feed-forward tests covering activation formulas, gating, validation, and tensor layouts.

use burn::{backend::Flex, tensor::Tensor};

use crate::architecture::Activation;

use burn::{module::Param, tensor::TensorData};

use super::{DenseFeedForwardConfig, GatedFeedForwardConfig, apply_activation};

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
