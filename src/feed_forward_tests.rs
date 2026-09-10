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
