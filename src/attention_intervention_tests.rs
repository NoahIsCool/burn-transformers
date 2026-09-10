//! Observable behavior of projected-attention interventions.

use burn::{
    backend::Flex,
    tensor::{Bool, Tensor, Tolerance},
};

use super::*;
use crate::intervention::Layer;

fn values<const D: usize>(tensor: Tensor<Flex, D>) -> Vec<f32> {
    tensor.into_data().to_vec::<f32>().unwrap()
}

fn projection() -> AttentionProjection<Flex> {
    let device = Default::default();
    AttentionProjection {
        query: Tensor::from_floats([[[[1., 0.], [0., 2.]], [[0., 1.], [2., 0.]]]], &device),
        key: Tensor::from_floats([[[[1., 0.], [0., 1.]]]], &device),
        value: Tensor::from_floats([[[[1., 2.], [3., 4.]]]], &device),
        gate: None,
    }
}

#[test]
fn incompatible_projection_inputs_fail_before_observers_or_pattern_transforms() {
    use std::cell::Cell;

    use burn::module::Param;

    let device = Default::default();
    let points = Layer::decoder(0).self_attention();
    for precision in [
        AttentionPrecisionMode::BackendDefault,
        AttentionPrecisionMode::EagerFp32Softmax,
    ] {
        for field in [
            "key",
            "value",
            "gate",
            "bias",
            "mask",
            "output",
            "output_bias",
        ] {
            let mut config =
                GroupedQueryAttentionConfig::new(4, 2, 1, 2).with_precision_mode(precision);
            config.output_gate = AttentionOutputGate::Sigmoid;
            let mut module = config.init::<Flex>(&device).unwrap();
            let mut projected = projection();
            projected.gate = Some(Tensor::ones([1, 2, 4], &device));
            let mut bias = None;
            let mut mask = None;
            match field {
                "key" => projected.key = projected.key.cast(DType::F64),
                "value" => projected.value = projected.value.cast(DType::F64),
                "gate" => projected.gate = projected.gate.map(|gate| gate.cast(DType::F64)),
                "bias" => {
                    bias = Some(Tensor::<Flex, 4>::ones([1, 2, 2, 2], &device).cast(DType::F64))
                }
                "mask" => mask = Some(Tensor::zeros([1, 1, 2, 2], &device)),
                "output" => {
                    projected.query = projected.query.cast(DType::F64);
                    projected.key = projected.key.cast(DType::F64);
                    projected.value = projected.value.cast(DType::F64);
                    projected.gate = projected.gate.map(|gate| gate.cast(DType::F64));
                }
                "output_bias" => {
                    module.o_proj.bias = Some(Param::from_tensor(
                        Tensor::<Flex, 1>::ones([4], &device).cast(DType::F64),
                    ));
                }
                _ => unreachable!(),
            }
            assert!(
                module
                    .attend(projected.clone(), mask.clone(), bias.clone(), false)
                    .is_err()
            );
            let called = Cell::new(false);
            let mut hooks = Hooks::new();
            hooks.observe(points.query(), |_| called.set(true));
            if precision == AttentionPrecisionMode::EagerFp32Softmax {
                hooks.observe(points.scores(), |_| called.set(true));
            }
            let result = hooks.run(points.all(), |hooks| {
                if precision == AttentionPrecisionMode::EagerFp32Softmax {
                    module.attend_with_pattern(
                        projected,
                        mask,
                        bias,
                        false,
                        points,
                        hooks,
                        |pattern| {
                            called.set(true);
                            Ok(pattern)
                        },
                    )
                } else {
                    module.attend_with_hooks(projected, mask, bias, false, points, hooks)
                }
            });
            let error = result.unwrap_err().to_string();
            let expected_field = match field {
                "output" => "attention.output_projection".to_string(),
                "output_bias" => "attention.output_projection.bias".to_string(),
                field => format!("attention.{field}"),
            };
            assert!(error.contains(&expected_field), "{error}");
            assert!(!called.get(), "callback ran for incompatible {field}");
        }
    }
}

#[test]
fn projection_preflight_preserves_quantized_output_weights() {
    use crate::checkpoint::quantization::{
        QuantizationGroupSize, SymmetricQuantization, WeightQuantizationConfig,
        quantize_tensor_data,
    };
    use burn::{module::Param, tensor::TensorData};

    let device = Default::default();
    let mut module = GroupedQueryAttentionConfig::new(4, 1, 1, 32)
        .init::<Flex>(&device)
        .unwrap();
    let packed = quantize_tensor_data(
        &TensorData::new(vec![0.5_f32; 128], [4, 32]),
        WeightQuantizationConfig {
            scheme: SymmetricQuantization::Q4,
            group_size: QuantizationGroupSize::new(32).unwrap(),
        },
    )
    .unwrap()
    .into_burn_qfloat_data()
    .unwrap();
    module.o_proj.weight = Param::from_tensor(Tensor::from_data(packed, &device).transpose());
    let projected = AttentionProjection {
        query: Tensor::ones([1, 1, 2, 32], &device),
        key: Tensor::ones([1, 1, 2, 32], &device),
        value: Tensor::ones([1, 1, 2, 32], &device),
        gate: None,
    };
    let expected = module.attend(projected.clone(), None, None, false).unwrap();
    let points = Layer::decoder(0).self_attention();
    let mut hooks = Hooks::new();
    let saved = hooks.capture(points.query());
    let actual = hooks
        .run(points.all(), |hooks| {
            module.attend_with_hooks(projected, None, None, false, points, hooks)
        })
        .unwrap();
    assert_eq!(actual.into_data(), expected.into_data());
    assert_eq!(hooks.get(&saved).unwrap().dtype(), DType::F32);
    assert!(matches!(
        module.o_proj.weight.val().dtype(),
        DType::QFloat(_)
    ));
}

#[test]
fn attention_captures_preserve_native_and_eager_outputs() {
    let device = Default::default();
    let points = Layer::decoder(0).self_attention();
    for precision in [
        AttentionPrecisionMode::BackendDefault,
        AttentionPrecisionMode::EagerFp32Softmax,
    ] {
        let mut config =
            GroupedQueryAttentionConfig::new(4, 2, 1, 2).with_precision_mode(precision);
        config.scale = Some(0.75);
        config.softcap = Some(1.5);
        config.output_gate = AttentionOutputGate::Sigmoid;
        let module = config.init::<Flex>(&device).unwrap();
        for causal in [false, true] {
            let mut projection = projection();
            projection.gate = Some(Tensor::full([1, 2, 4], 0.4, &device));
            let bias = Tensor::from_floats(
                [[[[0.1, -0.2], [0.3, 0.4]], [[-0.1, 0.2], [-0.3, -0.4]]]],
                &device,
            );
            let expected = module
                .attend(projection.clone(), None, Some(bias.clone()), causal)
                .unwrap();
            let mut hooks = Hooks::new();
            let query = hooks.capture(points.query());
            let key = hooks.capture(points.key());
            hooks.capture(points.value());
            hooks.capture(points.head_context());
            if precision == AttentionPrecisionMode::EagerFp32Softmax {
                hooks.capture(points.scores());
                hooks.capture(points.pattern());
            }
            let actual = hooks
                .run(points.all(), |hooks| {
                    module.attend_with_hooks(
                        projection.clone(),
                        None,
                        Some(bias),
                        causal,
                        points,
                        hooks,
                    )
                })
                .unwrap();
            actual
                .into_data()
                .assert_approx_eq::<f32>(&expected.into_data(), Tolerance::rel_abs(1e-6, 1e-6));
            assert_eq!(values(hooks.get(&query).unwrap()), values(projection.query));
            assert_eq!(hooks.get(&key).unwrap().dims(), [1, 1, 2, 2]);
        }
    }
}

#[test]
fn eager_score_and_pattern_captures_preserve_reduced_precision_boundaries() {
    use burn::{module::Param, tensor::DType};
    let device = Default::default();
    let mut module = GroupedQueryAttentionConfig::new(4, 2, 1, 2)
        .with_precision_mode(AttentionPrecisionMode::EagerFp32Softmax)
        .init::<Flex>(&device)
        .unwrap();
    module.o_proj.weight = Param::from_tensor(module.o_proj.weight.val().cast(DType::F16));
    let points = Layer::decoder(0).self_attention();
    let mut projection = projection();
    projection.query = (projection.query * 1.731).cast(DType::F16);
    projection.key = (projection.key * 0.293).cast(DType::F16);
    projection.value = (projection.value * 0.657).cast(DType::F16);
    let expected = module
        .attend(projection.clone(), None, None, false)
        .unwrap();
    let mut hooks = Hooks::new();
    let scores = hooks.capture(points.scores());
    let pattern = hooks.capture(points.pattern());
    let actual = hooks
        .run(points.all(), |hooks| {
            module.attend_with_hooks(projection, None, None, false, points, hooks)
        })
        .unwrap();
    assert_eq!(actual.dtype(), DType::F16);
    assert_eq!(actual.into_data(), expected.into_data());
    assert_eq!(hooks.get(&scores).unwrap().dtype(), DType::F16);
    assert_eq!(hooks.get(&pattern).unwrap().dtype(), DType::F16);
}

#[test]
fn selected_head_edit_matches_manual_context_ablation() {
    let device = Default::default();
    let module = GroupedQueryAttentionConfig::new(4, 2, 1, 2)
        .init::<Flex>(&device)
        .unwrap();
    let points = Layer::encoder(1).self_attention();
    let mut clean_hooks = Hooks::new();
    let context = clean_hooks.capture(points.head_context());
    clean_hooks
        .run(points.all(), |hooks| {
            module.attend_with_hooks(projection(), None, None, false, points, hooks)
        })
        .unwrap();
    let clean_context = clean_hooks.get(&context).unwrap();
    let expected_context = clean_context.clone().slice_assign(
        [0..1, 1..2, 0..2, 0..2],
        Tensor::zeros([1, 1, 2, 2], &device),
    );
    let expected = operations::linear(
        expected_context.swap_dims(1, 2).reshape([1, 2, 4]),
        &module.o_proj,
    )
    .unwrap();
    let mut hooks = Hooks::new();
    hooks.edit(points.head_context().head(1), |head| head * 0.0);
    let untouched = hooks.capture(points.head_context().head(0));
    let actual = hooks
        .run(points.all(), |hooks| {
            module.attend_with_hooks(projection(), None, None, false, points, hooks)
        })
        .unwrap();
    assert_eq!(values(actual), values(expected));
    assert_eq!(
        values(hooks.get(&untouched).unwrap()),
        values(clean_context.narrow(1, 0, 1))
    );
}

#[test]
fn query_edits_feed_native_attention_without_requiring_materialized_scores() {
    let device = Default::default();
    let module = GroupedQueryAttentionConfig::new(4, 2, 1, 2)
        .init::<Flex>(&device)
        .unwrap();
    let points = Layer::decoder(0).self_attention();
    let mut expected_projection = projection();
    expected_projection.query = expected_projection.query * 2.0;
    let expected = module
        .attend(expected_projection, None, None, false)
        .unwrap();
    let mut hooks = Hooks::new();
    hooks.edit(points.query(), |query| query * 2.0);
    let actual = hooks
        .run(points.all(), |hooks| {
            module.attend_with_hooks(projection(), None, None, false, points, hooks)
        })
        .unwrap();
    assert_eq!(values(actual), values(expected));
}

#[test]
fn pattern_edits_cannot_restore_causal_or_padding_exclusions() {
    let device = Default::default();
    let module = GroupedQueryAttentionConfig::new(4, 2, 1, 2)
        .with_precision_mode(AttentionPrecisionMode::EagerFp32Softmax)
        .init::<Flex>(&device)
        .unwrap();
    let points = Layer::decoder(0).self_attention();
    let mut hooks = Hooks::new();
    hooks.edit(points.pattern().head(1), |pattern| {
        Tensor::ones(pattern.dims(), &pattern.device())
    });
    let effective_pattern = hooks.capture(points.pattern());
    let context = hooks.capture(points.head_context());
    let mask = Tensor::<Flex, 4, Bool>::from_bool(
        [[
            [[false, false], [true, true]],
            [[false, false], [true, true]],
        ]]
        .into(),
        &device,
    );
    hooks
        .run(points.all(), |hooks| {
            module.attend_with_hooks(projection(), Some(mask), None, true, points, hooks)
        })
        .unwrap();
    // First query sees only the first value; the second has no allowed keys.
    assert_eq!(
        values(hooks.get(&effective_pattern).unwrap()),
        vec![1., 0., 0., 0., 1., 0., 0., 0.]
    );
    assert_eq!(
        values(hooks.get(&context).unwrap()),
        vec![1., 2., 0., 0., 1., 2., 0., 0.]
    );
}

#[test]
fn scores_are_after_scale_and_softcap_before_additive_bias() {
    let device = Default::default();
    let mut config = GroupedQueryAttentionConfig::new(4, 2, 1, 2)
        .with_precision_mode(AttentionPrecisionMode::EagerFp32Softmax);
    config.scale = Some(0.5);
    config.softcap = Some(0.75);
    let module = config.init::<Flex>(&device).unwrap();
    let points = Layer::decoder(0).self_attention();
    let mut hooks = Hooks::new();
    let scores = hooks.capture(points.scores().head(0).query_token(0));
    hooks
        .run(points.all(), |hooks| {
            module.attend_with_hooks(
                projection(),
                None,
                Some(Tensor::full([1, 2, 2, 2], 3.0, &device)),
                true,
                points,
                hooks,
            )
        })
        .unwrap();
    let scores = values(hooks.get(&scores).unwrap());
    assert!((scores[0] - (0.5_f32 / 0.75).tanh() * 0.75).abs() < 1e-6);
    assert_eq!(
        scores[1], 0.0,
        "future score is visible before causal exclusion"
    );
}

#[test]
fn entirely_negative_infinity_scores_have_finite_zero_pattern() {
    let device = Default::default();
    let module = GroupedQueryAttentionConfig::new(4, 2, 1, 2)
        .with_precision_mode(AttentionPrecisionMode::EagerFp32Softmax)
        .init::<Flex>(&device)
        .unwrap();
    let points = Layer::encoder(0).self_attention();
    for use_bias in [false, true] {
        let mut hooks = Hooks::new();
        if !use_bias {
            hooks.edit(points.scores(), |scores| {
                Tensor::full(scores.dims(), f32::NEG_INFINITY, &scores.device())
            });
        }
        let pattern = hooks.capture(points.pattern());
        let context = hooks.capture(points.head_context());
        let bias = use_bias.then(|| Tensor::full([1, 2, 2, 2], f32::NEG_INFINITY, &device));
        if use_bias {
            let ordinary = module
                .attend(projection(), None, bias.clone(), false)
                .unwrap();
            assert!(values(ordinary).into_iter().all(|value| value == 0.0));
            let mut context_only = Hooks::new();
            context_only.capture(points.head_context());
            let output = context_only
                .run(points.all(), |hooks| {
                    module.attend_with_hooks(projection(), None, bias.clone(), false, points, hooks)
                })
                .unwrap();
            assert!(values(output).into_iter().all(|value| value == 0.0));
        }
        hooks
            .run(points.all(), |hooks| {
                module.attend_with_hooks(projection(), None, bias, false, points, hooks)
            })
            .unwrap();
        assert!(
            values(hooks.get(&pattern).unwrap())
                .into_iter()
                .all(|value| value == 0.0)
        );
        assert!(
            values(hooks.get(&context).unwrap())
                .into_iter()
                .all(|value| value == 0.0)
        );
    }
}

#[test]
fn native_score_requests_fail_before_any_callback() {
    let device = Default::default();
    let module = GroupedQueryAttentionConfig::new(4, 2, 1, 2)
        .init::<Flex>(&device)
        .unwrap();
    let points = Layer::decoder(0).self_attention();
    let mut hooks = Hooks::new();
    let query = hooks.capture(points.query());
    hooks.edit(points.scores(), |_| {
        panic!("unsupported graph must fail first")
    });
    let error = hooks
        .run(points.all(), |hooks| {
            module.attend_with_hooks(projection(), None, None, true, points, hooks)
        })
        .unwrap_err();
    assert!(error.to_string().contains("EagerFp32Softmax"));
    assert!(hooks.get(&query).is_err());
}

#[test]
fn structural_pattern_runs_without_hooks_and_is_visible_to_observers() {
    let device = Default::default();
    let module = GroupedQueryAttentionConfig::new(4, 2, 1, 2)
        .with_precision_mode(AttentionPrecisionMode::EagerFp32Softmax)
        .init::<Flex>(&device)
        .unwrap();
    let points = Layer::decoder(0).self_attention();
    let empty_output = module
        .attend_with_pattern(
            projection(),
            None,
            None,
            false,
            points,
            &mut Hooks::new(),
            |pattern| Ok(pattern * 0.0 + 1.0),
        )
        .unwrap();
    let mut hooks = Hooks::new();
    let pattern = hooks.capture(points.pattern());
    let context = hooks.capture(points.head_context());
    let observed_output = hooks
        .run(points.all(), |hooks| {
            module.attend_with_pattern(projection(), None, None, false, points, hooks, |pattern| {
                Ok(pattern * 0.0 + 1.0)
            })
        })
        .unwrap();
    assert_eq!(values(empty_output), values(observed_output));
    assert_eq!(values(hooks.get(&pattern).unwrap()), vec![1.0; 8]);
    assert_eq!(
        values(hooks.get(&context).unwrap()),
        vec![4., 6., 4., 6., 4., 6., 4., 6.]
    );
}

#[test]
fn structural_pattern_preserves_masks_and_checks_shape() {
    let device = Default::default();
    let module = GroupedQueryAttentionConfig::new(4, 2, 1, 2)
        .with_precision_mode(AttentionPrecisionMode::EagerFp32Softmax)
        .init::<Flex>(&device)
        .unwrap();
    let points = Layer::decoder(0).self_attention();
    let mut hooks = Hooks::new();
    let pattern = hooks.capture(points.pattern());
    hooks
        .run(points.all(), |hooks| {
            module.attend_with_pattern(projection(), None, None, true, points, hooks, |pattern| {
                Ok(pattern * 0.0 + 1.0)
            })
        })
        .unwrap();
    assert_eq!(
        values(hooks.get(&pattern).unwrap()),
        vec![1., 0., 1., 1., 1., 0., 1., 1.]
    );
    let error = module
        .attend_with_pattern(
            projection(),
            None,
            None,
            false,
            points,
            &mut Hooks::new(),
            |pattern| Ok(pattern.narrow(1, 0, 1)),
        )
        .unwrap_err();
    assert!(error.to_string().contains("preserve pattern shape"));
    let expected = module.attend(projection(), None, None, false).unwrap();
    let actual = module
        .attend_with_pattern(
            projection(),
            None,
            None,
            false,
            points,
            &mut Hooks::new(),
            Ok,
        )
        .unwrap();
    assert_eq!(values(actual), values(expected));
}

#[test]
fn point_validity_runs_without_observers_and_keeps_precision_explicit() {
    use crate::intervention::PointValidity;
    let device = Default::default();
    let module = GroupedQueryAttentionConfig::new(4, 2, 1, 2)
        .init::<Flex>(&device)
        .unwrap();
    let points = Layer::decoder(0).self_attention();
    let validity = PointValidity::new(
        points.query(),
        Tensor::<Flex, 4, Bool>::zeros([1, 1, 1, 1], &device),
    )
    .unwrap();
    let mut expected_projection = projection();
    expected_projection.query = expected_projection.query.zeros_like();
    let expected = module
        .attend(expected_projection, None, None, false)
        .unwrap();
    let actual = Hooks::new()
        .run_with_validity(points.all(), vec![validity], |hooks| {
            module.attend_with_hooks(projection(), None, None, false, points, hooks)
        })
        .unwrap();
    assert_eq!(values(actual), values(expected));

    let validity = || {
        PointValidity::new(
            points.pattern(),
            Tensor::<Flex, 4, Bool>::zeros([1, 1, 1, 1], &device),
        )
        .unwrap()
    };
    let error = Hooks::new()
        .run_with_validity(points.all(), vec![validity()], |hooks| {
            module.attend_with_hooks(projection(), None, None, false, points, hooks)
        })
        .unwrap_err();
    assert!(error.to_string().contains("EagerFp32Softmax"));
    let mut module = module;
    module.precision_mode = AttentionPrecisionMode::EagerFp32Softmax;
    let output = Hooks::new()
        .run_with_validity(points.all(), vec![validity()], |hooks| {
            module.attend_with_hooks(projection(), None, None, false, points, hooks)
        })
        .unwrap();
    assert_eq!(values(output), vec![0.; 8]);
}

#[cfg(feature = "autodiff")]
#[test]
fn materialized_pattern_edits_preserve_value_gradients() {
    use burn::backend::Autodiff;
    type Ad = Autodiff<Flex>;
    let device = Default::default();
    let module = GroupedQueryAttentionConfig::new(4, 2, 1, 2)
        .with_precision_mode(AttentionPrecisionMode::EagerFp32Softmax)
        .init::<Ad>(&device)
        .unwrap();
    let points = Layer::decoder(0).self_attention();
    let value = Tensor::<Ad, 4>::from_floats([[[[1., 2.], [3., 4.]]]], &device).require_grad();
    let projection = || AttentionProjection {
        query: Tensor::<Ad, 4>::zeros([1, 2, 2, 2], &device),
        key: Tensor::<Ad, 4>::zeros([1, 1, 2, 2], &device),
        value: value.clone(),
        gate: None,
    };
    let ordinary = module
        .attend(projection(), None, None, false)
        .unwrap()
        .sum()
        .backward();
    let mut hooks = Hooks::new();
    hooks.edit(points.pattern(), |pattern| pattern * 0.5);
    let modified = hooks
        .run(points.all(), |hooks| {
            module.attend_with_hooks(projection(), None, None, false, points, hooks)
        })
        .unwrap()
        .sum()
        .backward();
    let expected = value.grad(&ordinary).unwrap() * 0.5;
    value
        .grad(&modified)
        .unwrap()
        .into_data()
        .assert_approx_eq::<f32>(&expected.into_data(), Tolerance::rel_abs(1e-6, 1e-6));
}
