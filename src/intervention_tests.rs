use burn::{
    backend::Flex,
    tensor::{DType, Tensor},
};

use super::{Axis, Hooks, Layer, Point};

type B = Flex<f32, i32>;

#[test]
fn borrowed_points_observe_without_changing_output() {
    let point = Layer::decoder(0).mlp_activation();
    let mut mean = None;
    let output;
    {
        let mut hooks = Hooks::<B>::new();
        hooks.observe(point.token(0), |tensor| {
            mean = Some(tensor.mean().into_scalar());
        });
        let saved = hooks.capture(&point);
        output = hooks
            .run(vec![point.info()], |hooks| {
                hooks.apply(point, Tensor::ones([1, 2, 2], &Default::default()))
            })
            .unwrap();
        assert_eq!(values(hooks.get(&saved).unwrap()), values(output.clone()));
    }
    assert_eq!(mean, Some(1.0));
    assert_eq!(values(output), vec![1.; 4]);
}

#[test]
fn dynamic_registration_cannot_bypass_preflight() {
    let point = Layer::decoder(0).mlp_activation();
    let mut hooks = Hooks::<B>::new();
    let error = hooks
        .run(vec![point.info()], |hooks| {
            hooks.capture(point.token(0).token(1));
            hooks.apply(point, Tensor::ones([1, 2, 2], &Default::default()))
        })
        .unwrap_err();
    assert!(error.to_string().contains("before starting"));
}

#[test]
fn callback_failure_identifies_the_selected_activation() {
    let point = Layer::encoder(4).mlp_output();
    let mut hooks = Hooks::<B>::new();
    hooks.try_edit(&point, |_| {
        Err(crate::Error::configuration("adapter", "bad projection"))
    });
    let error = hooks
        .run(vec![point.info()], |hooks| {
            hooks.apply(point, Tensor::ones([1, 1, 2], &Default::default()))
        })
        .unwrap_err();
    assert!(error.to_string().contains("encoder.layers.4.mlp.output"));
    assert!(error.to_string().contains("bad projection"));
}

fn values(tensor: Tensor<B, 3>) -> Vec<f32> {
    tensor.into_data().to_vec::<f32>().unwrap()
}

#[test]
fn capture_order_and_sliced_edits_preserve_unselected_tokens() {
    let point = Layer::decoder(2).mlp_activation();
    let mut hooks = Hooks::<B>::new();
    let before = hooks.capture(point.clone().token(1).channels(1..3));
    hooks.edit(point.clone().token(1).channels(1..3), |x| x * 4.0);
    let after = hooks.capture(point.clone());
    let output = hooks
        .run(vec![point.info()], |hooks| {
            hooks.apply(point, Tensor::ones([1, 3, 4], &Default::default()))
        })
        .unwrap();
    assert_eq!(values(hooks.get(&before).unwrap()), vec![1., 1.]);
    assert_eq!(
        values(output.clone()),
        vec![1., 1., 1., 1., 1., 4., 4., 1., 1., 1., 1., 1.]
    );
    assert_eq!(values(hooks.get(&after).unwrap()), values(output));
}

#[test]
fn capture_from_clean_run_can_patch_another_input() {
    let point = Layer::encoder(0).residual_post();
    let mut clean = Hooks::<B>::new();
    let saved = clean.capture(point.clone().token(0));
    clean
        .run(vec![point.info()], |hooks| {
            hooks.apply(point.clone(), Tensor::ones([1, 2, 2], &Default::default()))
        })
        .unwrap();
    let mut patch = Hooks::new();
    patch.replace(point.clone().token(1), clean.get(&saved).unwrap());
    let output = patch
        .run(vec![point.info()], |hooks| {
            hooks.apply(point, Tensor::zeros([1, 2, 2], &Default::default()))
        })
        .unwrap();
    assert_eq!(values(output), vec![0., 0., 1., 1.]);
}

#[test]
fn custom_rank_two_points_and_borrowed_callbacks_work() {
    let point = Point::custom("probe.neurons", [Axis::Batch, Axis::Channel]);
    let multiplier = 3.0;
    let mut observed = 0;
    {
        let mut hooks = Hooks::<B>::new();
        hooks.edit(point.clone().channels(0..1), |x: Tensor<B, 2>| {
            observed += 1;
            x * multiplier
        });
        let output = hooks
            .run(vec![point.info()], |hooks| {
                hooks.apply(point, Tensor::ones([2, 2], &Default::default()))
            })
            .unwrap();
        assert_eq!(
            output.into_data().to_vec::<f32>().unwrap(),
            vec![3., 1., 3., 1.]
        );
    }
    assert_eq!(observed, 1);
}

#[test]
fn unavailable_and_impossible_selections_fail_before_computation() {
    let point = Layer::decoder(0).mlp_activation();
    let mut absent = Hooks::<B>::new();
    absent.capture(point.clone());
    let error = absent
        .run(Vec::new(), |_| -> crate::Result<()> {
            panic!("must validate before execution")
        })
        .unwrap_err();
    assert!(error.to_string().contains("unavailable"));
    let mut invalid = Hooks::<B>::new();
    invalid.capture(point.clone().head(0));
    assert!(
        invalid
            .run(vec![point.info()], |_| -> crate::Result<()> {
                panic!("must reject nonexistent axis")
            })
            .unwrap_err()
            .to_string()
            .contains("Head")
    );
    let mut duplicate = Hooks::<B>::new();
    duplicate.capture(point.clone().token(0).token(1));
    assert!(
        duplicate
            .run(vec![point.info()], |_| -> crate::Result<()> {
                panic!("must reject duplicate axis")
            })
            .unwrap_err()
            .to_string()
            .contains("duplicate")
    );
}

#[test]
fn replacements_validate_shape_dtype_and_bounds() {
    let point = Layer::decoder(0).residual_pre();
    let mut shape = Hooks::<B>::new();
    shape.edit(point.clone(), |x| x.narrow(2, 0, 1));
    let error = shape
        .run(vec![point.info()], |hooks| {
            hooks.apply(point.clone(), Tensor::ones([1, 2, 3], &Default::default()))
        })
        .unwrap_err();
    assert!(error.to_string().contains("preserve selected shape"));
    let mut dtype = Hooks::<B>::new();
    dtype.edit(point.clone(), |x| x.cast(DType::F64));
    assert!(
        dtype
            .run(vec![point.info()], |hooks| hooks.apply(
                point.clone(),
                Tensor::ones([1, 2, 3], &Default::default())
            ))
            .unwrap_err()
            .to_string()
            .contains("dtype")
    );
    let mut bounds = Hooks::<B>::new();
    bounds.capture(point.clone().token(2));
    assert!(
        bounds
            .run(vec![point.info()], |hooks| hooks
                .apply(point, Tensor::ones([1, 2, 3], &Default::default())))
            .unwrap_err()
            .to_string()
            .contains("outside extent")
    );
}

#[test]
fn run_reuse_discards_old_and_failed_partial_captures() {
    let point = Layer::decoder(0).mlp_output();
    let mut hooks = Hooks::<B>::new();
    let saved = hooks.capture(point.clone());
    assert!(hooks.get(&saved).is_err());
    hooks
        .run(vec![point.info()], |hooks| {
            hooks.apply(point.clone(), Tensor::ones([1, 1, 1], &Default::default()))
        })
        .unwrap();
    assert_eq!(values(hooks.get(&saved).unwrap()), vec![1.]);
    hooks
        .run(vec![point.info()], |hooks| {
            hooks.apply(
                point.clone(),
                Tensor::full([1, 1, 1], 2., &Default::default()),
            )
        })
        .unwrap();
    assert_eq!(values(hooks.get(&saved).unwrap()), vec![2.]);
    let failure: crate::Result<()> = hooks.run(vec![point.info()], |hooks| {
        hooks.apply(point, Tensor::ones([1, 1, 1], &Default::default()))?;
        Err(crate::Error::configuration(
            "example",
            "intentional failure",
        ))
    });
    assert!(failure.is_err());
    assert!(hooks.get(&saved).is_err());
    assert!(Hooks::<B>::new().get(&saved).is_err());
}

#[test]
fn unvisited_points_and_unscoped_execution_are_errors() {
    let point = Layer::decoder(0).mlp_activation();
    let mut hooks = Hooks::<B>::new();
    hooks.capture(point.clone());
    assert!(
        hooks
            .apply(point.clone(), Tensor::ones([1, 1, 1], &Default::default()))
            .is_err()
    );
    assert!(
        hooks
            .run(vec![point.info()], |_| Ok(()))
            .unwrap_err()
            .to_string()
            .contains("not visited")
    );
}

#[test]
fn attention_selection_distinguishes_head_and_query_axes() {
    let point = Layer::decoder(0).self_attention().pattern();
    let mut hooks = Hooks::<B>::new();
    hooks.edit(point.clone().head(1).query_token(0), |x| x.zeros_like());
    let output = hooks
        .run(vec![point.info()], |hooks| {
            hooks.apply(point, Tensor::ones([1, 2, 2, 3], &Default::default()))
        })
        .unwrap();
    assert_eq!(
        output.into_data().to_vec::<f32>().unwrap(),
        vec![1., 1., 1., 1., 1., 1., 0., 0., 0., 1., 1., 1.]
    );
}

#[cfg(feature = "autodiff")]
#[test]
fn sliced_edits_and_captures_preserve_the_requested_gradient_policy() {
    use burn::backend::Autodiff;
    type Ad = Autodiff<B>;
    let point = Layer::decoder(0).mlp_activation();
    let input = Tensor::<Ad, 3>::ones([1, 2, 2], &Default::default()).require_grad();
    let mut hooks = Hooks::<Ad>::new();
    let detached = hooks.capture(point.clone());
    let connected = hooks.capture_with_grad(point.clone());
    hooks.edit(point.clone().token(1), |x| x * 3.0);
    let output = hooks
        .run(vec![point.info()], |hooks| {
            hooks.apply(point, input.clone())
        })
        .unwrap();
    let gradients = output.sum().backward();
    assert_eq!(
        values(input.grad(&gradients).unwrap()),
        vec![1., 1., 3., 3.]
    );
    let captured_loss = hooks.get(&connected).unwrap().sum();
    assert!(input.grad(&captured_loss.backward()).is_some());
    let detached_loss = hooks.get(&detached).unwrap().sum();
    assert!(input.grad(&detached_loss.backward()).is_none());
}
