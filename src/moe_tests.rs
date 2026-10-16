//! Mixture-of-experts tests for routing, grouping, numerical equivalence, and invalid configurations.

use burn::{
    backend::Flex,
    module::Param,
    tensor::{DType, Int, Tensor, backend::BackendTypes},
};

use super::{
    ExpertSelection, GroupLimitedSelection, GroupScore, PrunedExpertScore, RoutedExpertPrecision,
    RoutedWeightingPrecision, RouterScoring, ScoringOrder, SparseMoe, SparseMoeConfig,
    dispatch_routed_experts_in_blocks, group_selected_experts, load_balancing_loss,
    weighted_expert_contribution,
};

/// Installs zero parameters without advancing Burn's process-wide test RNG.
fn deterministic_module(
    config: SparseMoeConfig,
    device: &<Flex as BackendTypes>::Device,
) -> SparseMoe<Flex> {
    let mut module = config.init::<Flex>(device).expect("valid MoE");
    module.router.weight = Param::from_tensor(Tensor::zeros(
        [config.hidden_size, config.routed_experts],
        device,
    ));
    for expert in module.routed.iter_mut().chain(module.shared.iter_mut()) {
        expert.gate_proj.weight = Param::from_tensor(Tensor::zeros(
            [config.hidden_size, config.intermediate_size],
            device,
        ));
        expert.up_proj.weight = Param::from_tensor(Tensor::zeros(
            [config.hidden_size, config.intermediate_size],
            device,
        ));
        expert.down_proj.weight = Param::from_tensor(Tensor::zeros(
            [config.intermediate_size, config.hidden_size],
            device,
        ));
    }
    module
}

/// Invalid top-k routing is rejected before allocating hundreds of experts.
#[test]
fn invalid_top_k_is_rejected() {
    let error = SparseMoeConfig::new(4, 8, 2, 3)
        .validate()
        .expect_err("top-k greater than expert count must fail");
    assert!(error.to_string().contains("experts_per_token"));
}

/// Serialized configs default legacy payloads to residual-dtype route weighting.
#[test]
fn weighting_precision_is_explicit_and_backward_compatible() {
    let mut config = SparseMoeConfig::new(4, 8, 3, 2);
    let mut legacy = serde_json::to_value(&config).unwrap();
    legacy
        .as_object_mut()
        .expect("configuration serializes as an object")
        .remove("weighting_precision");
    let decoded: SparseMoeConfig = serde_json::from_value(legacy).unwrap();
    assert_eq!(
        decoded.weighting_precision,
        RoutedWeightingPrecision::Residual
    );

    config.weighting_precision = RoutedWeightingPrecision::Fp32ThenResidual;
    assert_eq!(
        serde_json::to_value(config).unwrap()["weighting_precision"],
        "fp32_then_residual"
    );
}

/// Sparse routing returns stable public shapes for diagnostics and hidden states.
#[test]
fn sparse_moe_preserves_shapes() {
    let device = Default::default();
    let module = deterministic_module(SparseMoeConfig::new(4, 8, 3, 2), &device);
    let output = module
        .forward(Tensor::<Flex, 3>::zeros([2, 3, 4], &device))
        .expect("valid input must dispatch");
    assert_eq!(output.hidden_state.dims(), [2, 3, 4]);
    assert_eq!(output.router_logits.dims(), [2, 3, 3]);
    assert_eq!(output.selected_experts.dims(), [2, 3, 2]);
    assert_eq!(output.selected_weights.dims(), [2, 3, 2]);
}

/// Renormalized selected probabilities sum to the routed scale for every token.
#[test]
fn selected_weights_are_renormalized() {
    let device = Default::default();
    let mut config = SparseMoeConfig::new(4, 8, 4, 2);
    config.routed_scale = 2.5;
    let module = deterministic_module(config, &device);
    let output = module
        .forward(Tensor::<Flex, 3>::ones([1, 2, 4], &device))
        .expect("valid input must dispatch");
    let sums = output.selected_weights.sum_dim(2).to_data();
    let values = sums
        .as_slice::<f32>()
        .expect("Flex f32 data should expose an f32 slice");
    assert!(values.iter().all(|value| (*value - 2.5).abs() < 1e-5));
}

/// Disabling top-k renormalization preserves each selected global softmax probability.
#[test]
fn non_normalized_weights_preserve_global_softmax_mass() {
    let device = Default::default();
    let mut config = SparseMoeConfig::new(4, 8, 4, 2);
    config.renormalize_selected = false;
    let module = deterministic_module(config, &device);
    let output = module
        .forward(Tensor::<Flex, 3>::zeros([1, 2, 4], &device))
        .expect("tied logits should route successfully");
    let sums = output.selected_weights.sum_dim(2).to_data();
    let values = sums
        .as_slice::<f32>()
        .expect("Flex f32 data should expose an f32 slice");
    assert!(values.iter().all(|value| (*value - 0.5).abs() < 1e-5));
}

/// Invalid shapes fail through the crate error taxonomy instead of tensor panics.
#[test]
fn invalid_runtime_shape_is_rejected() {
    let device = Default::default();
    let module = SparseMoeConfig::new(4, 8, 3, 2)
        .init::<Flex>(&device)
        .expect("valid MoE should initialize");
    let error = module
        .forward(Tensor::<Flex, 3>::zeros([2, 3, 5], &device))
        .expect_err("wrong hidden width must fail");
    assert!(error.to_string().contains("moe.input"));
}

/// Dispatcher metadata is validated before casts or unchecked index kernels.
#[test]
fn invalid_dispatch_metadata_is_rejected() {
    let device = Default::default();
    let module = deterministic_module(SparseMoeConfig::new(4, 8, 3, 2), &device);
    let error = module
        .dispatch_selected_experts(
            Tensor::zeros([2, 4], &device),
            Tensor::ones([2, 1], &device),
            Tensor::<Flex, 2, Int>::from_ints([[0, 1], [1, 2]], &device),
            DType::F32,
        )
        .expect_err("route weights with the wrong top-k width must fail");
    assert!(error.to_string().contains("moe.dispatch.routes"));

    let error = module
        .dispatch_selected_experts(
            Tensor::zeros([1, 4], &device),
            Tensor::ones([1, 2], &device),
            Tensor::<Flex, 2, Int>::from_ints([[0, 1]], &device),
            DType::I32,
        )
        .expect_err("integer accumulation must fail before a tensor cast");
    assert!(error.to_string().contains("moe.dispatch.precision"));
}

/// Qwen retains an FP32 route weight until after multiplication by an F16 expert output.
#[test]
fn qwen_f16_route_weight_rounds_after_multiplication() {
    let device = Default::default();
    let expert_output = Tensor::<Flex, 2>::from_floats([[1.5]], &device).cast(DType::F16);
    let fp32_weight = Tensor::<Flex, 2>::from_floats([[0.1]], &device);

    let actual = weighted_expert_contribution(
        expert_output.clone(),
        fp32_weight.clone(),
        RoutedExpertPrecision::fp32_weights_round_each(DType::F16),
    );
    let upstream = (expert_output.clone().cast(DType::F32) * fp32_weight.clone()).cast(DType::F16);
    let weight_rounded_too_early = expert_output * fp32_weight.cast(DType::F16);

    assert_eq!(actual.dtype(), DType::F16);
    assert_eq!(actual.to_data(), upstream.to_data());
    assert_ne!(actual.to_data(), weight_rounded_too_early.to_data());
    assert_eq!(
        actual.cast(DType::F32).into_data().to_vec::<f32>().unwrap(),
        [0.150_024_41]
    );
}

/// Equal router scores select lower expert IDs first on every backend.
#[test]
fn routing_ties_are_lexicographically_deterministic() {
    let device = Default::default();
    let module = deterministic_module(SparseMoeConfig::new(4, 8, 4, 2), &device);
    let output = module
        .forward(Tensor::<Flex, 3>::zeros([1, 3, 4], &device))
        .expect("zero inputs produce tied router logits");
    assert_eq!(
        output.selected_experts.into_data(),
        Tensor::<Flex, 3, Int>::from_ints([[[0, 1], [0, 1], [0, 1]]], &device).into_data()
    );
}

/// Host grouping retains token-major order and represents empty experts.
#[test]
fn selected_expert_grouping_is_stable_and_complete() {
    let device = Default::default();
    let selected = Tensor::<Flex, 2, Int>::from_ints([[0, 2], [2, 0], [0, 2]], &device);
    let groups = group_selected_experts(selected, 4).expect("valid routing IDs");

    assert_eq!(groups[0].token_indices, [0, 1, 2]);
    assert_eq!(groups[0].weight_indices, [0, 3, 4]);
    assert!(groups[1].token_indices.is_empty());
    assert_eq!(groups[2].token_indices, [0, 1, 2]);
    assert_eq!(groups[2].weight_indices, [1, 2, 5]);
    assert!(groups[3].token_indices.is_empty());
}

/// Invalid host-materialized IDs fail before unchecked backend indexing.
#[test]
fn selected_expert_grouping_rejects_out_of_range_ids() {
    let device = Default::default();
    let selected = Tensor::<Flex, 2, Int>::from_ints([[0, -1], [1, 3]], &device);
    let error = group_selected_experts(selected, 3)
        .expect_err("negative and upper-bound expert IDs must fail");
    assert!(error.to_string().contains("expert IDs in 0..3"));
    assert!(error.to_string().contains("flattened assignment 1"));
}

/// The dispatcher invokes only experts that own at least one assignment.
#[test]
fn dispatch_counters_prove_unselected_experts_are_skipped() {
    let device = Default::default();
    let module = deterministic_module(SparseMoeConfig::new(4, 8, 4, 2), &device);
    let output = module
        .forward(Tensor::<Flex, 3>::zeros([2, 3, 4], &device))
        .expect("tied routing");
    assert_eq!(output.expert_token_counts, vec![6, 6, 0, 0]);
    assert_eq!(output.evaluated_experts, 2);
    assert_eq!(output.expert_token_counts.iter().sum::<usize>(), 12);
}

/// Host grouping preserves gradients while omitting unselected expert graphs.
#[cfg(feature = "autodiff")]
#[test]
fn selected_dispatch_retains_input_and_expert_gradients() {
    use burn::backend::Autodiff;

    /// Differentiable CPU backend used to inspect selected expert and input gradients.
    type AdBackend = Autodiff<Flex>;
    let device = Default::default();
    let mut config = SparseMoeConfig::new(4, 8, 2, 1);
    config.renormalize_selected = false;
    let mut module = config.init::<AdBackend>(&device).expect("valid MoE");
    module.router.weight = Param::from_tensor(Tensor::zeros([4, 2], &device));
    for expert in &mut module.routed {
        expert.gate_proj.weight = Param::from_tensor(Tensor::ones([4, 8], &device));
        expert.up_proj.weight = Param::from_tensor(Tensor::ones([4, 8], &device));
        expert.down_proj.weight = Param::from_tensor(Tensor::ones([8, 4], &device));
    }
    let input = Tensor::<AdBackend, 3>::ones([1, 2, 4], &device).require_grad();
    let tracked_input = input.clone();

    let output = module
        .forward(input)
        .expect("tied routing should select expert zero");
    assert_eq!(output.expert_token_counts, [2, 0]);
    let gradients = output.hidden_state.sum().backward();

    assert!(tracked_input.grad(&gradients).is_some());
    assert!(module.routed[0].gate_proj.weight.grad(&gradients).is_some());
    assert!(module.routed[0].up_proj.weight.grad(&gradients).is_some());
    assert!(module.routed[0].down_proj.weight.grad(&gradients).is_some());
    assert!(module.routed[1].gate_proj.weight.grad(&gradients).is_none());
}

/// Gradients reach the router bias and shared-expert gate but never the frozen correction bias.
///
/// Covers both orders: sigmoid group-limited routing with a selection-only
/// correction bias and a gated shared expert (DeepSeek-V3/Qwen3.5-MoE style),
/// and top-k-then-softmax routing with a router bias (gpt-oss style). Every
/// trainable routing parameter must receive a finite gradient through the
/// selected mixture weights.
#[cfg(feature = "autodiff")]
#[test]
fn routing_extensions_train_their_parameters_but_not_the_correction_bias() {
    use burn::backend::Autodiff;

    /// Differentiable CPU backend used to inspect routing-parameter gradients.
    type AdBackend = Autodiff<Flex>;
    let device = Default::default();
    let mut group_limited = SparseMoeConfig::new(4, 3, 4, 2);
    group_limited.shared_experts = 1;
    group_limited.shared_expert_gate = true;
    group_limited.routing.router_bias = true;
    group_limited.routing.correction_bias = true;
    group_limited.routing.scoring = RouterScoring::Sigmoid;
    group_limited.routing.selection = ExpertSelection::GroupLimited(GroupLimitedSelection {
        groups: 2,
        top_groups: 1,
        group_score: GroupScore::TopTwoSum,
        pruned_score: PrunedExpertScore::NegativeInfinity,
    });
    let mut top_k_then_score = SparseMoeConfig::new(4, 3, 4, 2);
    top_k_then_score.renormalize_selected = false;
    top_k_then_score.routing.order = ScoringOrder::TopKThenScore;
    top_k_then_score.routing.router_bias = true;

    let finite = |values: Vec<f32>| values.iter().all(|value| value.is_finite());
    for config in [group_limited, top_k_then_score] {
        let module = config.init::<AdBackend>(&device).expect("valid MoE");
        let input = (Tensor::<AdBackend, 1, Int>::arange(0..24, &device).float() * 0.37)
            .sin()
            .reshape([2, 3, 4]);
        let gradients = module
            .forward(input)
            .expect("finite routing")
            .hidden_state
            .sum()
            .backward();
        let weight = module
            .router
            .weight
            .grad(&gradients)
            .expect("router weight");
        assert!(finite(weight.into_data().to_vec::<f32>().unwrap()));
        let bias = module.router.bias.as_ref().expect("router bias allocated");
        let bias = bias.grad(&gradients).expect("router bias gradient");
        assert!(finite(bias.into_data().to_vec::<f32>().unwrap()));
        if let Some(gate) = &module.shared_expert_gate {
            let gate = gate.weight.grad(&gradients).expect("shared gate gradient");
            assert!(finite(gate.into_data().to_vec::<f32>().unwrap()));
        }
        if let Some(correction) = &module.correction_bias {
            assert!(correction.grad(&gradients).is_none());
        }
        assert_eq!(
            module.correction_bias.is_some(),
            config.routing.correction_bias
        );
        assert_eq!(
            module.shared_expert_gate.is_some(),
            config.shared_expert_gate
        );
    }
}

/// Load-balancing diagnostics reject mismatched masks before broadcasting.
#[test]
fn auxiliary_loss_validates_mask_shape() {
    use crate::architecture::TokenMask;
    use burn::tensor::Bool;

    let device = Default::default();
    let module = deterministic_module(SparseMoeConfig::new(4, 8, 4, 2), &device);
    let output = module
        .forward(Tensor::<Flex, 3>::zeros([1, 2, 4], &device))
        .unwrap();
    let mask = TokenMask::new(Tensor::<Flex, 2, Bool>::from_bool(
        [[true, true, true]].into(),
        &device,
    ));
    let error = output
        .load_balancing_loss(Some(&mask))
        .expect_err("wrong mask width must fail");
    assert!(error.to_string().contains("moe.token_mask"));
}

/// Public auxiliary-loss inputs reject invalid expert IDs before one-hot encoding.
#[test]
fn auxiliary_loss_validates_selected_expert_range() {
    let device = Default::default();
    let logits = Tensor::<Flex, 3>::zeros([1, 1, 4], &device);
    let selected = Tensor::<Flex, 3, Int>::from_ints([[[0, 4]]], &device);
    let error = load_balancing_loss(logits, selected, None)
        .expect_err("an out-of-range expert ID must fail cleanly");
    assert!(error.to_string().contains("expert IDs in 0..4"));
}

/// Rejects every zero cardinality and invalid routed-output scaling value.
#[test]
fn config_validates_dimensions_top_k_and_finite_positive_scale() {
    for field in ["hidden", "intermediate", "experts", "top_k"] {
        let mut config = SparseMoeConfig::new(2, 2, 2, 1);
        match field {
            "hidden" => config.hidden_size = 0,
            "intermediate" => config.intermediate_size = 0,
            "experts" => config.routed_experts = 0,
            "top_k" => config.experts_per_token = 0,
            _ => unreachable!(),
        }
        assert!(matches!(
            config.validate(),
            Err(crate::Error::Configuration { .. })
        ));
    }
    for scale in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut config = SparseMoeConfig::new(2, 2, 2, 1);
        config.routed_scale = scale;
        assert!(
            matches!(config.validate(), Err(crate::Error::Configuration { field, .. }) if field == "moe.routed_scale")
        );
    }
    let device = Default::default();
    assert!(
        SparseMoeConfig::new(usize::MAX, 2, 1, 1)
            .init::<Flex>(&device)
            .is_err()
    );
    assert!(
        matches!(SparseMoeConfig::new(2, 1, usize::MAX, 1).validate(),
        Err(crate::Error::Configuration { field, reason })
        if field == "moe.projection_shape" && reason.contains("router"))
    );
}

/// Empty routing axes and upper-bound IDs fail before host dispatch indexing.
#[test]
fn grouping_rejects_empty_metadata_and_positive_out_of_range_ids() {
    let device = Default::default();
    for (tokens, slots, experts) in [(0, 1, 2), (1, 0, 2), (1, 1, 0)] {
        let ids = Tensor::<Flex, 2, Int>::zeros([tokens, slots], &device);
        assert!(
            matches!(group_selected_experts(ids, experts), Err(crate::Error::Shape { tensor, .. }) if tensor == "moe.selected_experts")
        );
    }
    let ids = Tensor::<Flex, 2, Int>::from_ints([[0, 2]], &device);
    let error = group_selected_experts(ids, 2).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("expert ID 2 at flattened assignment 1")
    );
}

/// Dispatcher input, route axes, cardinalities, and precision fail separately.
#[test]
fn dispatcher_validates_each_independent_shape_and_expert_boundary() {
    let device = Default::default();
    let module = deterministic_module(SparseMoeConfig::new(2, 2, 2, 1), &device);
    for (input, weights, ids, field) in [
        ([0, 2], [0, 1], [0, 1], "moe.dispatch.input"),
        ([2, 3], [2, 1], [2, 1], "moe.dispatch.input"),
        ([2, 2], [1, 1], [2, 1], "moe.dispatch.routes"),
        ([2, 2], [2, 1], [1, 1], "moe.dispatch.routes"),
        ([2, 2], [2, 1], [2, 2], "moe.dispatch.routes"),
    ] {
        let error = module
            .dispatch_selected_experts(
                Tensor::zeros(input, &device),
                Tensor::zeros(weights, &device),
                Tensor::zeros(ids, &device),
                DType::F32,
            )
            .unwrap_err();
        assert!(matches!(error, crate::Error::Shape { tensor, .. } if tensor == field));
    }
    for top_k in [0, 3] {
        let mut changed = module.clone();
        changed.experts_per_token = top_k;
        let error = changed
            .dispatch_selected_experts(
                Tensor::zeros([1, 2], &device),
                Tensor::zeros([1, top_k], &device),
                Tensor::zeros([1, top_k], &device),
                DType::F32,
            )
            .unwrap_err();
        assert!(
            matches!(error, crate::Error::Configuration { field, .. } if field == "moe.dispatch.experts_per_token")
        );
    }
    let mut empty = module.clone();
    empty.routed.clear();
    assert!(
        matches!(empty.dispatch_selected_experts(Tensor::zeros([1, 2], &device),
        Tensor::zeros([1, 1], &device), Tensor::zeros([1, 1], &device), DType::F32),
        Err(crate::Error::Configuration { field, .. }) if field == "moe.dispatch.experts_per_token")
    );
    for (multiplication_dtype, accumulation_dtype) in
        [(DType::I32, DType::F32), (DType::F32, DType::I32)]
    {
        let precision = RoutedExpertPrecision {
            multiplication_dtype,
            accumulation_dtype,
        };
        assert!(
            matches!(precision.validate(), Err(crate::Error::Configuration { field, .. }) if field == "moe.dispatch.precision")
        );
    }
}

/// Manually edited modules still reject invalid routing before top-k executes.
#[test]
fn forward_rejects_empty_inputs_and_mutated_expert_cardinalities() {
    let device = Default::default();
    let module = deterministic_module(SparseMoeConfig::new(2, 2, 2, 1), &device);
    for shape in [[0, 1, 2], [1, 0, 2]] {
        assert!(matches!(module.forward(Tensor::zeros(shape, &device)),
            Err(crate::Error::Shape { tensor, .. }) if tensor == "moe.input"));
    }
    for top_k in [0, 3] {
        let mut changed = module.clone();
        changed.experts_per_token = top_k;
        assert!(matches!(changed.forward(Tensor::zeros([1, 1, 2], &device)),
            Err(crate::Error::Configuration { field, .. }) if field == "moe.experts_per_token"));
    }
}

/// Biased shared experts add independently of routed scaling and sparse choice.
#[test]
fn shared_experts_add_exact_values_while_unselected_nan_experts_are_skipped() {
    let device = Default::default();
    let mut config = SparseMoeConfig::new(1, 1, 3, 1);
    config.shared_experts = 2;
    config.expert_bias = true;
    config.activation = crate::architecture::Activation::Relu;
    config.routed_scale = 2.0;
    config.weighting_precision = RoutedWeightingPrecision::Fp32ThenResidual;
    let mut module = deterministic_module(config, &device);
    for (expert, coefficient) in module
        .routed
        .iter_mut()
        .chain(module.shared.iter_mut())
        .zip([1.0, 1.0, 1.0, 3.0, 5.0])
    {
        expert.gate_proj.weight = Param::from_tensor(Tensor::ones([1, 1], &device));
        expert.up_proj.weight = Param::from_tensor(Tensor::ones([1, 1], &device));
        expert.down_proj.weight = Param::from_tensor(Tensor::from_floats([[coefficient]], &device));
        expert.gate_proj.bias = Some(Param::from_tensor(Tensor::ones([1], &device)));
        expert.up_proj.bias = Some(Param::from_tensor(Tensor::zeros([1], &device)));
        expert.down_proj.bias = Some(Param::from_tensor(Tensor::zeros([1], &device)));
    }
    module.routed[1].up_proj.weight =
        Param::from_tensor(Tensor::from_floats([[f32::NAN]], &device));
    module.routed[2].up_proj.weight =
        Param::from_tensor(Tensor::from_floats([[f32::NAN]], &device));
    let output = module
        .forward(Tensor::from_floats([[[1.0], [2.0]]], &device))
        .unwrap();
    // Each live expert is c * ReLU(x+1) * x: (2*1 + 3 + 5) * x*(x+1).
    assert_eq!(
        output.hidden_state.into_data().to_vec::<f32>().unwrap(),
        [20.0, 60.0]
    );
    assert_eq!(
        output.selected_weights.into_data().to_vec::<f32>().unwrap(),
        [2.0, 2.0]
    );
    assert_eq!(output.expert_token_counts, [2, 0, 0]);
    assert_eq!(output.evaluated_experts, 1);
}

/// Selected/shared expert parameter errors remain contextual instead of panicking.
#[test]
fn malformed_selected_and_shared_experts_propagate_linear_shape_errors() {
    let device = Default::default();
    for shared in [false, true] {
        let mut config = SparseMoeConfig::new(2, 2, 2, 1);
        config.shared_experts = usize::from(shared);
        let mut module = deterministic_module(config, &device);
        let expert = if shared {
            &mut module.shared[0]
        } else {
            &mut module.routed[0]
        };
        expert.up_proj.weight = Param::from_tensor(Tensor::zeros([3, 2], &device));
        assert!(matches!(
            module.forward(Tensor::ones([1, 1, 2], &device)),
            Err(crate::Error::Shape { .. })
        ));
    }
}

/// Balancing loss obeys the masked means, including the all-padding zero case.
#[test]
fn load_balancing_loss_matches_two_expert_probability_oracle() {
    use crate::architecture::TokenMask;
    use burn::tensor::Bool;

    let device = Default::default();
    let logits =
        Tensor::<Flex, 3>::from_floats([[[3.0_f32.ln(), 0.0]], [[0.0, 3.0_f32.ln()]]], &device);
    let selected = Tensor::<Flex, 3, Int>::from_ints([[[0]], [[1]]], &device);
    let unmasked = load_balancing_loss(logits.clone(), selected.clone(), None)
        .unwrap()
        .into_scalar();
    assert!((unmasked - 1.0).abs() < 1e-6);
    for (mask, expected) in [([[true], [false]], 1.5), ([[false], [false]], 0.0)] {
        let mask = TokenMask::new(Tensor::<Flex, 2, Bool>::from_bool(mask.into(), &device));
        let loss = load_balancing_loss(logits.clone(), selected.clone(), Some(&mask))
            .unwrap()
            .into_scalar();
        assert!((loss - expected).abs() < 1e-6);
    }
    let all_experts = Tensor::<Flex, 3, Int>::from_ints([[[0, 1]], [[1, 0]]], &device);
    let loss = load_balancing_loss(logits, all_experts, None)
        .unwrap()
        .into_scalar();
    assert!((loss - 2.0).abs() < 1e-6);
}

/// Diagnostics reject empty axes, unequal token axes, and negative IDs.
#[test]
fn load_balancing_loss_validates_all_external_diagnostic_axes() {
    let device = Default::default();
    for (logits, ids, field) in [
        ([0, 1, 2], [0, 1, 1], "moe.router_diagnostics"),
        ([1, 0, 2], [1, 0, 1], "moe.router_diagnostics"),
        ([1, 1, 0], [1, 1, 1], "moe.router_diagnostics"),
        ([1, 1, 2], [1, 1, 0], "moe.router_diagnostics"),
        ([1, 2, 2], [2, 2, 1], "moe.selected_experts"),
        ([1, 2, 2], [1, 1, 1], "moe.selected_experts"),
    ] {
        assert!(
            matches!(load_balancing_loss(Tensor::<Flex, 3>::zeros(logits, &device),
            Tensor::<Flex, 3, Int>::zeros(ids, &device), None),
            Err(crate::Error::Shape { tensor, .. }) if tensor == field)
        );
    }
    assert!(
        matches!(load_balancing_loss(Tensor::<Flex, 3>::zeros([1, 1, 2], &device),
        Tensor::<Flex, 3, Int>::from_ints([[[-1]]], &device), None),
        Err(crate::Error::Shape { tensor, actual, .. }) if tensor == "moe.selected_experts" && actual.contains("-1"))
    );
}

/// Out-of-range I32 expert IDs return the offending value and flattened assignment index.
#[test]
fn expert_grouping_rejects_extreme_backend_i32_ids() {
    let device = Default::default();
    for id in [i32::MIN, i32::MAX] {
        let ids = Tensor::<Flex, 2, Int>::from_data(
            burn::tensor::TensorData::new(vec![0_i32, id], [1, 2]),
            &device,
        );
        let error = group_selected_experts(ids, 2).unwrap_err();
        assert!(matches!(error, crate::Error::Shape { tensor, actual, .. }
            if tensor == "moe.selected_experts" && actual.contains(&format!("expert ID {id} at flattened assignment 1"))));
    }
}

proptest::proptest! {
    #![proptest_config(proptest::test_runner::Config::with_cases(48))]

    /// Random routing assignments appear exactly once in stable token/slot order within each expert.
    #[test]
    fn expert_grouping_preserves_every_assignment_and_its_token_slot_order(
        experts in 1usize..=8,
        slots in 1usize..=5,
        input in proptest::collection::vec(0usize..100, 1..=40),
    ) {
        let device = Default::default();
        let tokens = input.len();
        // Repeated expert IDs within a token are valid metadata; no assignment may disappear.
        let ids = input.iter().flat_map(|&seed| (0..slots).map(move |slot| ((seed + slot / 2) % experts) as i64)).collect::<Vec<_>>();
        let tensor = Tensor::<Flex, 2, Int>::from_data(
            burn::tensor::TensorData::new(ids.clone(), [tokens, slots]), &device,
        );
        let groups = group_selected_experts(tensor, experts).unwrap();
        proptest::prop_assert_eq!(groups.len(), experts);
        let mut visited = vec![false; ids.len()];
        for (expert, group) in groups.iter().enumerate() {
            proptest::prop_assert_eq!(group.token_indices.len(), group.weight_indices.len());
            proptest::prop_assert!(group.weight_indices.windows(2).all(|pair| pair[0] < pair[1]));
            for (&token, &weight) in group.token_indices.iter().zip(&group.weight_indices) {
                let weight = usize::try_from(weight).unwrap();
                proptest::prop_assert!(!visited[weight]);
                visited[weight] = true;
                proptest::prop_assert_eq!(token, (weight / slots) as i64);
                proptest::prop_assert_eq!(ids[weight], expert as i64);
            }
        }
        proptest::prop_assert!(visited.into_iter().all(|seen| seen));
    }
}

/// Fills every parameter with a distinct smooth pattern so routing is non-trivial.
fn patterned_module(
    config: SparseMoeConfig,
    device: &<Flex as BackendTypes>::Device,
) -> SparseMoe<Flex> {
    let mut module = config.init::<Flex>(device).expect("valid MoE");
    let pattern = |shape: [usize; 2], seed: f32| {
        let count = shape[0] * shape[1];
        Param::from_tensor(
            (Tensor::<Flex, 1, Int>::arange(0..count as i64, device)
                .float()
                .reshape(shape)
                .mul_scalar(0.73)
                .add_scalar(seed))
            .sin()
            .mul_scalar(0.8),
        )
    };
    module.router.weight = pattern([config.hidden_size, config.routed_experts], 0.1);
    for (index, expert) in module
        .routed
        .iter_mut()
        .chain(module.shared.iter_mut())
        .enumerate()
    {
        let seed = index as f32 * 1.7;
        expert.gate_proj.weight = pattern([config.hidden_size, config.intermediate_size], seed);
        expert.up_proj.weight = pattern([config.hidden_size, config.intermediate_size], seed + 0.5);
        expert.down_proj.weight =
            pattern([config.intermediate_size, config.hidden_size], seed + 0.9);
    }
    module
}

/// The pre-refactor dispatcher: one upload pair, select, and scatter-add per expert.
fn legacy_dispatch(
    module: &SparseMoe<Flex>,
    flat_input: Tensor<Flex, 2>,
    flat_weights: Tensor<Flex, 2>,
    selected: Tensor<Flex, 2, Int>,
    precision: RoutedExpertPrecision,
) -> Tensor<Flex, 2> {
    use burn::tensor::{IndexingUpdateOp, TensorData};
    let [tokens, hidden] = flat_input.dims();
    let device = flat_input.device();
    let assignments = flat_weights.dims()[0] * flat_weights.dims()[1];
    let groups = group_selected_experts(selected, module.routed.len()).unwrap();
    let flat_weights = flat_weights.reshape([assignments]);
    let mut state =
        Tensor::<Flex, 2>::zeros([tokens, hidden], (&device, precision.accumulation_dtype));
    for (expert, group) in module.routed.iter().zip(groups) {
        let count = group.token_indices.len();
        if count == 0 {
            continue;
        }
        let tokens = Tensor::<Flex, 1, Int>::from_data(
            TensorData::new(group.token_indices, [count]),
            &device,
        );
        let weights = Tensor::<Flex, 1, Int>::from_data(
            TensorData::new(group.weight_indices, [count]),
            &device,
        );
        let weights = flat_weights
            .clone()
            .select(0, weights)
            .unsqueeze_dim::<2>(1);
        let output = expert
            .forward(flat_input.clone().select(0, tokens.clone()))
            .unwrap();
        let contribution = weighted_expert_contribution(output, weights, precision);
        state = state.select_assign(0, tokens, contribution, IndexingUpdateOp::Add);
    }
    state
}

/// Batched dispatch is bit-identical to the per-expert loop, including repeated experts and F16 sums.
#[test]
fn batched_dispatch_is_bit_identical_to_the_per_expert_loop() {
    let device = Default::default();
    let module = patterned_module(SparseMoeConfig::new(6, 5, 8, 3), &device);
    let flat_input = (Tensor::<Flex, 1, Int>::arange(0..42, &device)
        .float()
        .reshape([7, 6])
        .mul_scalar(0.41))
    .cos();
    // Token 1 repeats expert 1 in two slots; experts 4 and 7 receive nothing.
    let selected = Tensor::<Flex, 2, Int>::from_ints(
        [
            [0, 2, 5],
            [1, 1, 3],
            [6, 0, 2],
            [3, 5, 6],
            [2, 2, 0],
            [5, 3, 1],
            [0, 1, 2],
        ],
        &device,
    );
    let weights = (Tensor::<Flex, 1, Int>::arange(0..21, &device)
        .float()
        .reshape([7, 3])
        .mul_scalar(0.29))
    .sin()
    .abs();
    for precision in [
        RoutedExpertPrecision::uniform(DType::F32),
        RoutedExpertPrecision::uniform(DType::F16),
        RoutedExpertPrecision::fp32_weights_round_each(DType::F16),
    ] {
        let batched = module
            .dispatch_selected_experts_with_precision(
                flat_input.clone(),
                weights.clone(),
                selected.clone(),
                None,
                precision,
            )
            .unwrap();
        assert_eq!(batched.expert_token_counts, [4, 4, 5, 3, 0, 3, 2, 0]);
        assert_eq!(batched.evaluated_experts, 6);
        let legacy = legacy_dispatch(
            &module,
            flat_input.clone(),
            weights.clone(),
            selected.clone(),
            precision,
        );
        let legacy = legacy.into_data();
        assert_eq!(
            batched.hidden_state.into_data(),
            legacy.clone(),
            "{precision:?}"
        );
        // Memory-bounded blocks keep the summation order for every budget:
        // one expert per block (0, 1), several (5, 9), and one block (MAX).
        for budget in [0, 1, 5, 9, usize::MAX] {
            let blocked = dispatch_routed_experts_in_blocks(
                flat_input.clone(),
                weights.clone(),
                selected.clone(),
                None,
                module.routed.len(),
                precision,
                budget,
                |expert, rows| module.routed[expert].forward(rows),
            )
            .unwrap();
            assert_eq!(blocked.expert_token_counts, [4, 4, 5, 3, 0, 3, 2, 0]);
            assert_eq!(blocked.evaluated_experts, 6);
            assert_eq!(
                blocked.hidden_state.into_data(),
                legacy.clone(),
                "{precision:?}, budget {budget}"
            );
        }
    }
}

/// A NaN router score is a contextual error instead of an out-of-range gather.
#[test]
fn nan_router_scores_are_reported_with_the_token() {
    let device = Default::default();
    let mut module = deterministic_module(SparseMoeConfig::new(2, 2, 4, 2), &device);
    module.router.weight = Param::from_tensor(Tensor::from_floats(
        [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
        &device,
    ));
    let input = Tensor::<Flex, 3>::from_floats([[[1.0, 2.0], [f32::NAN, 0.0]]], &device);
    let error = module.forward(input).expect_err("NaN logits");
    assert!(
        matches!(&error, crate::Error::Shape { tensor, actual, .. }
        if tensor == "moe.router_scores" && actual.contains("flattened token 1")),
        "{error}"
    );
    let infinite = Tensor::<Flex, 3>::from_floats([[[f32::INFINITY, 0.0]]], &device);
    let error = module
        .forward(infinite.clone())
        .expect_err("softmax of +inf is NaN");
    assert!(matches!(error, crate::Error::Shape { tensor, .. } if tensor == "moe.router_scores"));

    // Top-k-then-softmax ranks raw logits, which hold +inf but no NaN; the
    // NaN appears only in the selected weights and must still be reported.
    let mut config = SparseMoeConfig::new(2, 2, 4, 2);
    config.routing.order = ScoringOrder::TopKThenScore;
    config.renormalize_selected = false;
    let mut module = deterministic_module(config, &device);
    module.router.weight = Param::from_tensor(Tensor::from_floats(
        [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
        &device,
    ));
    let error = module
        .forward(infinite)
        .expect_err("softmax over a selected +inf logit is NaN");
    assert!(
        matches!(&error, crate::Error::Shape { tensor, actual, .. }
        if tensor == "moe.router_scores" && actual.contains("flattened token 0")),
        "{error}"
    );
}

/// Sigmoid group-limited routing selects with the frozen correction bias but mixes uncorrected scores.
#[test]
fn correction_bias_steers_selection_but_not_weights() {
    let device = Default::default();
    let mut config = SparseMoeConfig::new(2, 2, 4, 1);
    config.renormalize_selected = false;
    config.routing.scoring = RouterScoring::Sigmoid;
    config.routing.correction_bias = true;
    config.routing.selection = ExpertSelection::GroupLimited(GroupLimitedSelection {
        groups: 2,
        top_groups: 1,
        group_score: GroupScore::TopTwoSum,
        pruned_score: PrunedExpertScore::NegativeInfinity,
    });
    let mut module = deterministic_module(config.clone(), &device);
    let bias = module.correction_bias.as_ref().expect("bias allocated");
    assert_eq!(bias.dims(), [4]);
    assert_eq!(bias.val().dtype(), DType::F32);
    assert!(!bias.is_require_grad());
    // Zero router weights give every expert the score sigmoid(0) = 0.5; the
    // bias alone makes expert 3 (group 1) win.
    module.correction_bias = Some(
        Param::from_tensor(Tensor::from_floats([0.0, 0.1, 0.0, 0.3], &device))
            .set_require_grad(false),
    );
    let output = module
        .forward(Tensor::<Flex, 3>::ones([1, 2, 2], &device))
        .unwrap();
    assert_eq!(
        output.selected_experts.into_data().to_vec::<i32>().unwrap(),
        [3, 3]
    );
    assert_eq!(
        output.selected_weights.into_data().to_vec::<f32>().unwrap(),
        [0.5, 0.5]
    );
    let mut missing = config;
    missing.routing.correction_bias = false;
    let mut module = deterministic_module(missing, &device);
    module.routing.correction_bias = true;
    assert!(matches!(module.forward(Tensor::ones([1, 1, 2], &device)),
        Err(crate::Error::Configuration { field, .. }) if field == "moe.router.correction_bias"));
}

/// The optional shared-expert gate multiplies the shared sum by `sigmoid(x w)`.
#[test]
fn shared_expert_gate_scales_the_shared_sum() {
    let device = Default::default();
    let mut config = SparseMoeConfig::new(1, 1, 2, 1);
    config.shared_experts = 1;
    config.shared_expert_gate = true;
    config.activation = crate::architecture::Activation::Relu;
    let mut module = deterministic_module(config.clone(), &device);
    for expert in module.routed.iter_mut().chain(module.shared.iter_mut()) {
        expert.gate_proj.weight = Param::from_tensor(Tensor::ones([1, 1], &device));
        expert.up_proj.weight = Param::from_tensor(Tensor::ones([1, 1], &device));
        expert.down_proj.weight = Param::from_tensor(Tensor::ones([1, 1], &device));
    }
    let gate = module.shared_expert_gate.as_mut().expect("gate allocated");
    gate.weight = Param::from_tensor(Tensor::from_floats([[2.0]], &device));
    let output = module
        .forward(Tensor::<Flex, 3>::from_floats([[[3.0]]], &device))
        .unwrap();
    // Each expert computes ReLU(x) * x = 9; routed weight 1; shared gated by sigmoid(6).
    let expected = 9.0 + 9.0 / (1.0 + (-6.0_f32).exp());
    let actual = output.hidden_state.into_data().to_vec::<f32>().unwrap()[0];
    assert!((actual - expected).abs() < 1e-5, "{actual} vs {expected}");

    config.shared_experts = 0;
    assert!(matches!(config.validate(),
        Err(crate::Error::Configuration { field, .. }) if field == "moe.shared_expert_gate"));
}

/// Legacy serialized configurations default the new routing fields.
#[test]
fn routing_fields_are_backward_compatible() {
    let config = SparseMoeConfig::new(4, 8, 3, 2);
    let mut legacy = serde_json::to_value(&config).unwrap();
    let object = legacy.as_object_mut().unwrap();
    object.remove("routing");
    object.remove("shared_expert_gate");
    let decoded: SparseMoeConfig = serde_json::from_value(legacy).unwrap();
    assert_eq!(decoded, config);
    assert_eq!(
        decoded.router_config().policy,
        super::RoutingPolicy::default()
    );
}

/// A single-expert bank needs no one-hot encoding and every slot counts as assigned.
#[test]
fn load_balancing_loss_supports_a_single_expert() {
    let device = Default::default();
    let loss = load_balancing_loss(
        Tensor::<Flex, 3>::zeros([1, 3, 1], &device),
        Tensor::<Flex, 3, Int>::zeros([1, 3, 1], &device),
        None,
    )
    .unwrap()
    .into_scalar();
    assert!((loss - 1.0).abs() < 1e-6);
}

/// Marks the re-executed test binary that runs the half-precision child.
const HALF_DEFAULT_CHILD: &str = "TRANSFORMERS_MOE_HALF_DEFAULT_CHILD";

/// Routing and auxiliary losses stay explicit-F32 when the device default float is F16.
///
/// Burn's default dtype is process-global per device, so the check runs in a
/// fresh child process of this test binary.
#[test]
fn routing_and_auxiliary_loss_survive_half_precision_default_dtype() {
    let output = std::process::Command::new(std::env::current_exe().expect("test binary"))
        .args([
            "--exact",
            "moe::tests::half_precision_default_child",
            "--ignored",
            "--test-threads=1",
        ])
        .env(HALF_DEFAULT_CHILD, "1")
        .output()
        .expect("spawn child test process");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success() && stdout.contains("1 passed"),
        "child failed:\n{stdout}\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

/// Child half of [`routing_and_auxiliary_loss_survive_half_precision_default_dtype`].
#[test]
#[ignore = "runs in a fresh process spawned by routing_and_auxiliary_loss_survive_half_precision_default_dtype"]
fn half_precision_default_child() {
    use crate::architecture::TokenMask;
    use burn::tensor::Bool;
    if std::env::var_os(HALF_DEFAULT_CHILD).is_none() {
        return;
    }
    let device = Default::default();
    burn::tensor::set_default_dtypes::<Flex>(&device, DType::F16, burn::tensor::IntDType::I32)
        .expect("fresh process has no tensors yet");
    let module = deterministic_module(SparseMoeConfig::new(4, 8, 3, 2), &device);
    let output = module
        .forward(Tensor::<Flex, 3>::ones([1, 2, 4], &device))
        .expect("F16 residual stream routes");
    assert_eq!(output.hidden_state.dtype(), DType::F16);
    let mask = TokenMask::new(Tensor::<Flex, 2, Bool>::from_bool(
        [[true, false]].into(),
        &device,
    ));
    let loss = output
        .load_balancing_loss(Some(&mask))
        .expect("masked loss");
    assert_eq!(loss.dtype(), DType::F32);
    assert!(loss.into_scalar().is_finite());
}

/// Measures blocked versus per-expert dispatch on prefill-sized routing problems.
///
/// The 4096-token case holds 32,768 assignments and so spans several
/// memory-bounded blocks; the others fit one block. The two dispatchers
/// alternate to cancel drift and each reports the median of seven runs. Run
/// with `cargo test --release --lib moe::tests::dispatch_timing -- --ignored --nocapture`.
#[test]
#[ignore = "timing measurement; run explicitly in release mode"]
fn dispatch_timing() {
    let device = Default::default();
    // (tokens, experts, top_k, hidden, intermediate): expert-bound and dispatch-bound shapes.
    for (tokens, experts, top_k, hidden, intermediate) in [
        (1024, 64, 8, 64, 32),
        (2048, 128, 8, 16, 4),
        (4096, 64, 8, 64, 32),
    ] {
        let module = patterned_module(
            SparseMoeConfig::new(hidden, intermediate, experts, top_k),
            &device,
        );
        let flat_input = (Tensor::<Flex, 1, Int>::arange(0..(tokens * hidden) as i64, &device)
            .float()
            .reshape([tokens, hidden])
            .mul_scalar(0.01))
        .sin();
        // Distinct experts per token keep the comparison faithful to router output.
        let ids = (0..tokens)
            .flat_map(|token| {
                let first = (token * 13) % experts;
                (0..top_k).map(move |slot| ((first + slot * 3) % experts) as i32)
            })
            .collect::<Vec<_>>();
        let selected = Tensor::<Flex, 2, Int>::from_data(
            burn::tensor::TensorData::new(ids, [tokens, top_k]),
            &device,
        );
        let weights =
            Tensor::<Flex, 2>::ones([tokens, top_k], &device).mul_scalar(1.0 / top_k as f32);
        let precision = RoutedExpertPrecision::uniform(DType::F32);
        let time = |run: &dyn Fn() -> Tensor<Flex, 2>| {
            let start = std::time::Instant::now();
            let _ = run().into_data();
            start.elapsed()
        };
        let blocked_run = || {
            module
                .dispatch_selected_experts_with_precision(
                    flat_input.clone(),
                    weights.clone(),
                    selected.clone(),
                    None,
                    precision,
                )
                .unwrap()
                .hidden_state
        };
        let legacy_run = || {
            legacy_dispatch(
                &module,
                flat_input.clone(),
                weights.clone(),
                selected.clone(),
                precision,
            )
        };
        let (_, _) = (time(&blocked_run), time(&legacy_run));
        let (mut blocked, mut legacy) = (Vec::new(), Vec::new());
        for _ in 0..7 {
            blocked.push(time(&blocked_run));
            legacy.push(time(&legacy_run));
        }
        blocked.sort();
        legacy.sort();
        let (blocked, legacy) = (blocked[3], legacy[3]);
        println!(
            "tokens={tokens} experts={experts} top_k={top_k} hidden={hidden} intermediate={intermediate}: \
             blocked {blocked:?}, per-expert {legacy:?}, speedup {:.2}x",
            legacy.as_secs_f64() / blocked.as_secs_f64()
        );
    }
}

/// Peak-memory probe: one long-prefill dispatch with blocked or single-block routing.
///
/// Routes 8192 tokens to 8 of 64 experts with a 1024-wide residual stream, so
/// all 65,536 routed rows occupy 256 MiB per F32 buffer. Set
/// `TRANSFORMERS_MOE_DISPATCH_BUDGET=all` to gather every row at once (the
/// unbounded single-block layout) or leave it unset for the default budget,
/// then compare the "maximum resident set size" of fresh processes, e.g.
/// `/usr/bin/time -l <lib test binary> --exact moe::tests::dispatch_peak_memory --ignored`.
#[test]
#[ignore = "memory measurement; run explicitly in a fresh process"]
fn dispatch_peak_memory() {
    let device = Default::default();
    let (tokens, experts, top_k, hidden) = (8192, 64, 8, 1024);
    let module = deterministic_module(SparseMoeConfig::new(hidden, 16, experts, top_k), &device);
    let flat_input = Tensor::<Flex, 2>::ones([tokens, hidden], &device);
    let ids = (0..tokens)
        .flat_map(|token| (0..top_k).map(move |slot| ((token + slot * 8) % experts) as i32))
        .collect::<Vec<_>>();
    let selected = Tensor::<Flex, 2, Int>::from_data(
        burn::tensor::TensorData::new(ids, [tokens, top_k]),
        &device,
    );
    let weights = Tensor::<Flex, 2>::ones([tokens, top_k], &device);
    let budget = match std::env::var("TRANSFORMERS_MOE_DISPATCH_BUDGET").as_deref() {
        Ok("all") => usize::MAX,
        _ => tokens.max(super::MIN_DISPATCH_BLOCK_ASSIGNMENTS),
    };
    let dispatch = dispatch_routed_experts_in_blocks(
        flat_input,
        weights,
        selected,
        None,
        experts,
        RoutedExpertPrecision::uniform(DType::F32),
        budget,
        |expert, rows| module.routed[expert].forward(rows),
    )
    .unwrap();
    let sum = dispatch.hidden_state.sum().into_scalar();
    println!("budget={budget} checksum={sum}");
}

/// Expert IDs in a non-default integer dtype are contextual errors, never panics.
///
/// Both the load-balancing diagnostics and sparse dispatch read expert IDs
/// to the host as the backend's integer element (`i32` on this Flex
/// instantiation). An `I64` ID tensor cannot be decoded as that element, so
/// each reader reports a backend-capability error naming the failed read
/// before any indexing kernel sees the IDs.
#[test]
fn expert_ids_in_a_foreign_integer_dtype_are_rejected() {
    use burn::tensor::{IntDType, TensorData};

    let device = Default::default();
    let wide_ids = |shape: [usize; 2]| {
        Tensor::<Flex, 2, Int>::from_data(
            TensorData::new(vec![0_i64; shape[0] * shape[1]], shape),
            &device,
        )
        .cast(IntDType::I64)
    };
    assert_eq!(wide_ids([1, 1]).dtype(), DType::I64);

    let error = load_balancing_loss(
        Tensor::<Flex, 3>::zeros([1, 2, 2], &device),
        wide_ids([2, 1]).reshape([1, 2, 1]),
        None,
    )
    .unwrap_err();
    assert!(
        matches!(&error, crate::Error::BackendCapability(message)
            if message.starts_with("failed to validate routed expert IDs")),
        "{error:?}"
    );

    let error = super::dispatch_routed_experts(
        Tensor::<Flex, 2>::ones([2, 3], &device),
        Tensor::<Flex, 2>::ones([2, 1], &device),
        wide_ids([2, 1]),
        None,
        2,
        RoutedExpertPrecision::uniform(DType::F32),
        |_, rows| Ok(rows),
    )
    .unwrap_err();
    assert!(
        matches!(&error, crate::Error::BackendCapability(message)
            if message.starts_with("failed to decode selected expert IDs")),
        "{error:?}"
    );
}
