//! Execution-validity tests for masking, failure cleanup, scope isolation, and gradients.

use burn::{
    backend::Flex,
    tensor::{Bool, Tensor},
};

use super::*;

/// Deterministic F32 CPU backend used by validity-mask fixtures.
type Cpu = Flex<f32, i32>;

/// Create a broadcast mask retaining the first token and excluding the second.
fn mask(point: Point<3>) -> PointValidity<Cpu> {
    PointValidity::new(
        point,
        Tensor::<Cpu, 3, Bool>::from_bool([[[true], [false]]].into(), &Default::default()),
    )
    .unwrap()
}

/// Read a fixture activation for exact zeroing and gradient assertions.
fn values(tensor: Tensor<Cpu, 3>) -> Vec<f32> {
    tensor.into_data().to_vec::<f32>().unwrap()
}

/// Invalid activations, including NaNs, become zero before capture and after constraints or edits.
#[test]
fn point_validity_zeroes_before_observation_and_after_each_edit() {
    let point = Layer::decoder(0).mlp_activation();
    let mut hooks = Hooks::new();
    let before = hooks.capture(&point);
    hooks.edit(&point, |tensor| tensor.ones_like());
    let after = hooks.capture(&point);
    let output = hooks
        .run_with_validity(vec![point.info()], vec![mask(point.clone())], |hooks| {
            hooks.apply_constrained(
                point.clone(),
                Tensor::<Cpu, 3>::from_floats([[[2.0, 3.0], [f32::NAN, 8.0]]], &Default::default()),
                |tensor| Ok(tensor + 2.0),
            )
        })
        .unwrap();
    assert_eq!(
        values(hooks.get(&before).unwrap()),
        vec![2.0, 3.0, 0.0, 0.0]
    );
    assert_eq!(values(hooks.get(&after).unwrap()), vec![3.0, 3.0, 0.0, 0.0]);
    assert_eq!(values(output), vec![3.0, 3.0, 0.0, 0.0]);
}

/// Validity masks run without observers and are cleared before the next unmasked execution.
#[test]
fn point_validity_applies_without_actions_and_does_not_leak_to_next_run() {
    let point = Layer::decoder(0).residual_post();
    let mut hooks = Hooks::new();
    let output = hooks
        .run_with_validity(vec![point.info()], vec![mask(point.clone())], |hooks| {
            assert!(hooks.is_empty());
            assert!(hooks.has_work());
            assert!(hooks.needs_point(point.info()));
            assert!(!hooks.is_requested(point.info()));
            hooks.apply(
                point.clone(),
                Tensor::<Cpu, 3>::ones([1, 2, 3], &Default::default()),
            )
        })
        .unwrap();
    assert_eq!(values(output), vec![1.0, 1.0, 1.0, 0.0, 0.0, 0.0]);
    assert!(!hooks.has_work());
    let output = hooks
        .run(vec![point.info()], |hooks| {
            hooks.apply(
                point.clone(),
                Tensor::<Cpu, 3>::ones([1, 2, 3], &Default::default()),
            )
        })
        .unwrap();
    assert_eq!(values(output), vec![1.0; 6]);
}

/// A nonbroadcastable mask fails the run, clears captures, and permits subsequent unmasked reuse.
#[test]
fn point_validity_errors_clear_captures_and_execution_masks() {
    let point = Layer::decoder(0).residual_post();
    let mut hooks = Hooks::new();
    let capture = hooks.capture(&point);
    let wrong = PointValidity::new(
        point.clone(),
        Tensor::<Cpu, 3, Bool>::from_bool([[[true], [false], [true]]].into(), &Default::default()),
    )
    .unwrap();
    let error = hooks
        .run_with_validity(vec![point.info()], vec![wrong], |hooks| {
            hooks.apply(
                point.clone(),
                Tensor::<Cpu, 3>::ones([1, 2, 3], &Default::default()),
            )
        })
        .unwrap_err();
    assert!(error.to_string().contains("broadcastable"));
    assert!(hooks.get(&capture).is_err());
    hooks
        .run(vec![point.info()], |hooks| {
            hooks.apply(
                point.clone(),
                Tensor::<Cpu, 3>::ones([1, 2, 3], &Default::default()),
            )
        })
        .unwrap();
    assert_eq!(values(hooks.get(&capture).unwrap()), vec![1.0; 6]);
}

/// Validity rejects unavailable points and duplicate masks while guarding against nested execution.
#[test]
fn point_validity_rejects_unknown_duplicate_and_nested_scopes() {
    let point = Layer::decoder(0).residual_post();
    let mut hooks = Hooks::<Cpu>::new();
    assert!(
        hooks
            .run_with_validity(Vec::new(), vec![mask(point.clone())], |_| Ok(()))
            .is_err()
    );
    assert!(
        hooks
            .run_with_validity(
                vec![point.info()],
                vec![mask(point.clone()), mask(point.clone())],
                |_| Ok(())
            )
            .is_err()
    );
    let output = hooks
        .run_with_validity(vec![point.info()], vec![mask(point.clone())], |hooks| {
            assert!(hooks.run(vec![point.info()], |_| Ok(())).is_err());
            hooks.apply(
                point.clone(),
                Tensor::<Cpu, 3>::ones([1, 2, 1], &Default::default()),
            )
        })
        .unwrap();
    assert_eq!(values(output), vec![1.0, 0.0]);
}

/// An execution validity mask does not permit registering actions after preflight.
#[test]
fn point_validity_keeps_mid_run_registration_guard() {
    let point = Layer::decoder(0).residual_post();
    let mut hooks = Hooks::<Cpu>::new();
    let error = hooks
        .run_with_validity(vec![point.info()], vec![mask(point.clone())], |hooks| {
            hooks.capture(&point);
            hooks.apply(
                point.clone(),
                Tensor::<Cpu, 3>::ones([1, 2, 1], &Default::default()),
            )
        })
        .unwrap_err();
    assert!(error.to_string().contains("before starting"));
}

#[cfg(feature = "autodiff")]
/// Masking removes gradient contributions at invalid tokens while retaining valid-token derivatives.
#[test]
fn point_validity_preserves_gradients_only_at_valid_positions() {
    use burn::backend::Autodiff;

    /// Differentiable CPU backend exposing gradients through valid positions.
    type Ad = Autodiff<Cpu>;
    let point = Layer::encoder(0).residual_post();
    let input = Tensor::<Ad, 3>::ones([1, 2, 2], &Default::default()).require_grad();
    let valid = PointValidity::new(
        point.clone(),
        Tensor::<Ad, 3, Bool>::from_bool([[[true], [false]]].into(), &Default::default()),
    )
    .unwrap();
    let mut hooks = Hooks::new();
    let output = hooks
        .run_with_validity(vec![point.info()], vec![valid], |hooks| {
            hooks.apply(point, input.clone())
        })
        .unwrap();
    let gradients = output.sum().backward();
    assert_eq!(
        values(input.grad(&gradients).unwrap()),
        vec![1.0, 1.0, 0.0, 0.0]
    );
}
