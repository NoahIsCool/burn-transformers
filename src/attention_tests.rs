//! Grouped-query attention tests for numerical parity, precision, masks, and projection contracts.

use burn::{
    backend::Flex,
    tensor::{Bool, Tensor, ops::AttentionModuleOptions},
};

use crate::architecture::TokenMask;

use super::{
    AttentionExecutionPolicy, AttentionOutputGate, AttentionPrecisionMode, AttentionProjection,
    GroupedQueryAttentionConfig, bottom_right_causal_mask, dispatch_attention,
    padding_attention_mask, repeat_key_value,
};

/// The production default bounds official-width 32K score planes to 64 queries.
#[test]
fn execution_policy_plans_official_width_32k_attention() {
    let policy = AttentionExecutionPolicy::default();
    let global = policy
        .plan(1, 32, 32_768, 32_768)
        .expect("one 32K score row fits the production bound");
    assert_eq!(global.query_chunk_size(), 64);
    assert_eq!(global.chunk_count(), 512);
    assert_eq!(global.peak_score_elements(), 1 << 26);
    assert_eq!(global.score_element_limit(), Some(1 << 26));
    assert_eq!(global.query_sequence(), 32_768);
    assert_eq!(global.key_sequence_bound(), 32_768);
    policy
        .validate_chunk(1, 32, 64, 32_768)
        .expect("the exact boundary is one valid chunk");
    assert!(policy.validate_chunk(1, 32, 65, 32_768).is_err());

    let local = policy
        .plan(1, 32, 32_768, 2048 + 511)
        .expect("Muse's local key-union bound must fit");
    assert_eq!(local.query_chunk_size(), 512);
    assert_eq!(local.chunk_count(), 64);
    assert_eq!(local.peak_score_elements(), 41_926_656);
    assert_eq!(local.key_sequence_bound(), 2559);

    let oracle = policy
        .plan(1, 32, 67, 67)
        .expect("the pinned Muse oracle prompt is intentionally unchunked");
    assert_eq!(oracle.query_chunk_size(), 67);
    assert_eq!(oracle.chunk_count(), 1);
}

/// Invalid or impossible bounds fail before allocating device tensors.
#[test]
fn execution_policy_rejects_zero_and_oversized_rows() {
    let zero_queries = AttentionExecutionPolicy::query_chunked(0, 1);
    assert!(
        zero_queries
            .validate()
            .unwrap_err()
            .to_string()
            .contains("max_query_tokens")
    );
    let zero_scores = AttentionExecutionPolicy::query_chunked(1, 0);
    assert!(
        zero_scores
            .validate()
            .unwrap_err()
            .to_string()
            .contains("max_score_elements")
    );
    let too_small = AttentionExecutionPolicy::query_chunked(8, 63);
    let error = too_small
        .plan(1, 2, 8, 32)
        .expect_err("one row contains 64 score elements");
    assert!(error.to_string().contains("one attention query row"));
    assert!(
        AttentionExecutionPolicy::default()
            .plan(0, 1, 1, 1)
            .is_err()
    );
}

/// Explicit unbounded plans retain a single full-query operation.
#[test]
fn unbounded_execution_plan_is_transparent() {
    let plan = AttentionExecutionPolicy::Unbounded
        .plan(2, 3, 5, 7)
        .expect("small shape");
    assert_eq!(plan.query_chunk_size(), 5);
    assert_eq!(plan.chunk_count(), 1);
    assert_eq!(plan.peak_score_elements(), 210);
    assert_eq!(plan.score_element_limit(), None);
}

/// Backend-native attention remains the serialized and construction default.
#[test]
fn attention_precision_mode_is_explicitly_opt_in() {
    let device = Default::default();
    let default = GroupedQueryAttentionConfig::new(8, 2, 1, 4)
        .init::<Flex>(&device)
        .expect("default attention should initialize");
    assert_eq!(
        default.precision_mode,
        AttentionPrecisionMode::BackendDefault
    );
    let eager = GroupedQueryAttentionConfig::new(8, 2, 1, 4)
        .with_precision_mode(AttentionPrecisionMode::EagerFp32Softmax)
        .init::<Flex>(&device)
        .expect("eager attention should initialize");
    assert_eq!(
        eager.precision_mode,
        AttentionPrecisionMode::EagerFp32Softmax
    );
}

/// Explicit eager softmax returns zero rather than NaN for an excluded row.
#[test]
fn eager_fp32_softmax_zeroes_fully_excluded_rows() {
    let device = Default::default();
    let query = Tensor::<Flex, 4>::zeros([1, 1, 2, 2], &device);
    let key = Tensor::<Flex, 4>::zeros([1, 1, 2, 2], &device);
    let value = Tensor::<Flex, 4>::from_floats([[[[2.0, 3.0], [5.0, 7.0]]]], &device);
    let exclusion =
        Tensor::<Flex, 4, Bool>::from_bool([[[[true, true], [false, true]]]].into(), &device);
    let output = dispatch_attention(
        query,
        key,
        value,
        Some(exclusion),
        None,
        AttentionModuleOptions::default(),
        AttentionPrecisionMode::EagerFp32Softmax,
    )
    .expect("valid eager attention");
    assert_eq!(
        output
            .into_data()
            .as_slice::<f32>()
            .expect("Flex output should expose f32"),
        &[0.0, 0.0, 2.0, 3.0]
    );
}

/// Cached causal alignment anchors each query row at the key suffix.
#[test]
fn eager_causal_mask_is_bottom_right_aligned() {
    let device = Default::default();
    let suffix = bottom_right_causal_mask::<Flex>(1, 1, 2, 4, &device)
        .expect("representable suffix mask")
        .into_data();
    assert_eq!(
        suffix
            .as_slice::<bool>()
            .expect("boolean mask should expose bool"),
        &[false, false, false, true, false, false, false, false]
    );
    let longer_query = bottom_right_causal_mask::<Flex>(1, 1, 3, 2, &device)
        .expect("representable negative offset")
        .into_data();
    assert_eq!(
        longer_query
            .as_slice::<bool>()
            .expect("boolean mask should expose bool"),
        &[true, true, false, true, false, false]
    );
}

/// Qwen's projected query width may exceed its residual width.
#[test]
fn attention_accepts_non_square_query_projection() {
    GroupedQueryAttentionConfig::new(10, 3, 1, 4)
        .validate()
        .expect("query width need not equal residual width");
}

/// Projected-width arithmetic is checked before an initializer sees a shape.
#[test]
fn attention_rejects_projection_dimension_overflow() {
    let error = GroupedQueryAttentionConfig::new(8, usize::MAX, 1, 2)
        .validate()
        .expect_err("query head product must not wrap");
    assert!(error.to_string().contains("overflow"));
}

/// KV repetition duplicates each native head into one contiguous GQA group.
#[test]
fn key_value_repetition_preserves_group_order() {
    let device = Default::default();
    let tensor = Tensor::<Flex, 4>::from_floats([[[[1.0]], [[2.0]]]], &device);
    let repeated = repeat_key_value(tensor, 4)
        .expect("four query heads group over two key/value heads")
        .to_data();
    assert_eq!(
        repeated
            .as_slice::<f32>()
            .expect("Flex f32 data should expose an f32 slice"),
        &[1.0, 1.0, 2.0, 2.0]
    );
}

/// Invalid head ratios are reported instead of reaching a division or reshape panic.
#[test]
fn key_value_repetition_rejects_invalid_grouping() {
    let device = Default::default();
    let tensor = Tensor::<Flex, 4>::zeros([1, 2, 1, 3], &device);
    let error = repeat_key_value(tensor, 3).expect_err("three heads cannot group over two");
    assert!(error.to_string().contains("head_grouping"));
}

/// External projected-attention callers receive shape diagnostics before dispatch.
#[test]
fn attention_rejects_malformed_projection() {
    let device = Default::default();
    let module = GroupedQueryAttentionConfig::new(8, 4, 2, 3)
        .init::<Flex>(&device)
        .expect("valid attention should initialize");
    let projection = AttentionProjection {
        query: Tensor::zeros([1, 3, 2, 3], &device),
        key: Tensor::zeros([1, 2, 2, 3], &device),
        value: Tensor::zeros([1, 2, 2, 3], &device),
        gate: None,
    };
    let error = module
        .attend(projection, None, None, true)
        .expect_err("wrong query-head count must fail");
    assert!(error.to_string().contains("attention.query"));
}

/// Projection validates residual width before dispatching a linear backend kernel.
#[test]
fn attention_rejects_malformed_projection_input() {
    let device = Default::default();
    let module = GroupedQueryAttentionConfig::new(8, 4, 2, 3)
        .init::<Flex>(&device)
        .expect("valid attention should initialize");
    let error = module
        .project_self(Tensor::zeros([1, 2, 7], &device))
        .expect_err("wrong residual width must fail");
    assert!(error.to_string().contains("attention.query_input"));
}

/// Valid-token masks invert exactly once at the low-level attention boundary.
#[test]
fn padding_mask_marks_invalid_queries_and_keys() {
    let device = Default::default();
    let valid = TokenMask::new(Tensor::<Flex, 2, Bool>::from_bool(
        [[true, false]].into(),
        &device,
    ));
    let mask = padding_attention_mask(Some(&valid), Some(&valid), 1, 1, 2, 2)
        .expect("valid mask shape should pass")
        .expect("provided masks should produce a tensor")
        .to_data();
    assert_eq!(
        mask.as_slice::<bool>()
            .expect("boolean data should expose a bool slice"),
        &[false, true, true, true]
    );
}

/// Invalid mask extents return a contextual shape error rather than panicking.
#[test]
fn padding_mask_rejects_wrong_shape() {
    let device = Default::default();
    let valid = TokenMask::new(Tensor::<Flex, 2, Bool>::from_bool(
        [[true, true, true]].into(),
        &device,
    ));
    let error = padding_attention_mask(Some(&valid), None, 1, 1, 2, 2)
        .expect_err("mismatched query mask must fail");
    assert!(error.to_string().contains("query_mask"));
}

/// Projection, GQA dispatch, output gating, and output projection preserve shape.
#[test]
fn gated_attention_preserves_residual_shape() {
    let device = Default::default();
    let mut config = GroupedQueryAttentionConfig::new(8, 4, 2, 3);
    config.output_gate = AttentionOutputGate::Silu;
    config.query_key_norm_epsilon = Some(1e-6);
    let module = config
        .init::<Flex>(&device)
        .expect("valid attention should initialize");
    let output = module
        .forward_without_position_encoding(Tensor::<Flex, 3>::zeros([2, 5, 8], &device), None, true)
        .expect("attention should execute");
    assert_eq!(output.dims(), [2, 5, 8]);
}

/// Rejects numerical and cardinality options before parameter initialization.
#[test]
fn config_validates_zero_axes_head_ratios_and_each_numerical_option() {
    for dimensions in [
        (0, 2, 1, 2),
        (4, 0, 1, 2),
        (4, 2, 0, 2),
        (4, 2, 1, 0),
        (4, 3, 2, 2),
    ] {
        let (hidden, queries, keys, width) = dimensions;
        assert!(matches!(
            GroupedQueryAttentionConfig::new(hidden, queries, keys, width).validate(),
            Err(crate::Error::Configuration { .. })
        ));
    }
    for option in ["query_key_norm_epsilon", "scale", "softcap"] {
        for value in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut config = GroupedQueryAttentionConfig::new(4, 2, 1, 2);
            match option {
                "query_key_norm_epsilon" => config.query_key_norm_epsilon = Some(value),
                "scale" => config.scale = Some(value),
                "softcap" => config.softcap = Some(value),
                _ => unreachable!(),
            }
            assert!(
                matches!(config.validate(), Err(crate::Error::Configuration { field, .. })
                if field == format!("attention.{option}"))
            );
        }
    }
    let mut gated = GroupedQueryAttentionConfig::new(1, usize::MAX, 1, 1);
    gated.output_gate = AttentionOutputGate::Sigmoid;
    assert!(
        gated
            .validate()
            .unwrap_err()
            .to_string()
            .contains("overflow")
    );
    assert!(
        GroupedQueryAttentionConfig::new(usize::MAX, 2, 1, 1)
            .validate()
            .unwrap_err()
            .to_string()
            .contains("matrix dimensions overflow")
    );
}

/// Query chunk planning checks products without ever allocating score planes.
#[test]
fn execution_policy_covers_overflow_remaining_zero_axes_and_remainders() {
    for shape in [(1, 0, 1, 1), (1, 1, 0, 1), (1, 1, 1, 0)] {
        assert!(
            AttentionExecutionPolicy::Unbounded
                .plan(shape.0, shape.1, shape.2, shape.3)
                .is_err()
        );
    }
    for shape in [
        (usize::MAX, 2, 1, 1),
        (1, usize::MAX, 1, 2),
        (usize::MAX / 2 + 1, 1, 2, 1),
    ] {
        assert!(
            matches!(AttentionExecutionPolicy::Unbounded.plan(shape.0, shape.1, shape.2, shape.3),
            Err(crate::Error::Configuration { field, reason }) if field == "attention.execution_shape" && reason.contains("overflow"))
        );
    }
    let policy = AttentionExecutionPolicy::query_chunked(4, 30);
    let plan = policy.plan(1, 2, 7, 5).unwrap();
    assert_eq!(plan.query_chunk_size(), 3);
    assert_eq!(plan.chunk_count(), 3);
    assert_eq!(plan.peak_score_elements(), 30);
    assert_eq!(policy.maximum_query_tokens(2), 2);
    assert_eq!(policy.maximum_query_tokens(7), 4);
    assert_eq!(
        AttentionExecutionPolicy::Unbounded.maximum_query_tokens(7),
        7
    );
    policy.validate_chunk(1, 2, 1, 5).unwrap();
}

/// Constructs a rectangular projection without initializing random parameters.
fn projection(batch: usize, query: usize, key: usize) -> AttentionProjection<Flex> {
    let device = Default::default();
    AttentionProjection {
        query: Tensor::zeros([batch, 2, query, 1], &device),
        key: Tensor::zeros([batch, 1, key, 1], &device),
        value: Tensor::zeros([batch, 1, key, 1], &device),
        gate: None,
    }
}

/// Empty externally assembled projections must fail before backend kernels.
#[test]
fn projected_attention_rejects_empty_batch_query_and_key_axes() {
    let module = GroupedQueryAttentionConfig::new(2, 2, 1, 1)
        .init::<Flex>(&Default::default())
        .unwrap();
    for (batch, query, key) in [(0, 1, 1), (1, 0, 1), (1, 1, 0)] {
        assert!(
            matches!(module.attend(projection(batch, query, key), None, None, false),
                Err(crate::Error::Shape { tensor, .. }) if tensor == "attention.projection"),
            "accepted empty projected axes ({batch}, {query}, {key})"
        );
    }
}

/// Projected keys, values, gates, masks, and every bias axis are checked.
#[test]
fn projected_attention_checks_each_external_topology_boundary() {
    let device = Default::default();
    let module = GroupedQueryAttentionConfig::new(2, 2, 1, 1)
        .init::<Flex>(&device)
        .unwrap();
    for (field, shape) in [
        ("key", [1, 1, 3, 1]),
        ("key", [2, 2, 3, 1]),
        ("key", [2, 1, 3, 2]),
        ("value", [1, 1, 3, 1]),
        ("value", [2, 2, 3, 1]),
        ("value", [2, 1, 2, 1]),
        ("value", [2, 1, 3, 2]),
    ] {
        let mut projected = projection(2, 2, 3);
        if field == "key" {
            projected.key = Tensor::zeros(shape, &device);
        } else {
            projected.value = Tensor::zeros(shape, &device);
        }
        assert!(
            matches!(super::validate_projection(&module, &projected, None, None),
            Err(crate::Error::Shape { tensor, .. }) if tensor == format!("attention.{field}"))
        );
    }
    let projected = projection(2, 2, 3);
    let mask = Tensor::<Flex, 4, Bool>::zeros([1, 2, 2, 3], &device);
    assert!(
        matches!(super::validate_projection(&module, &projected, Some(&mask), None),
        Err(crate::Error::Shape { tensor, .. }) if tensor == "attention.mask")
    );
    for shape in [[3, 2, 2, 3], [1, 1, 2, 3], [1, 2, 1, 3], [1, 2, 2, 2]] {
        let bias = Tensor::zeros(shape, &device);
        assert!(
            matches!(super::validate_projection(&module, &projected, None, Some(&bias)),
            Err(crate::Error::Shape { tensor, .. }) if tensor == "attention.bias")
        );
    }
    for (gate_mode, gate_shape) in [
        (AttentionOutputGate::None, Some([2, 2, 2])),
        (AttentionOutputGate::Sigmoid, None),
        (AttentionOutputGate::Silu, Some([2, 2, 1])),
    ] {
        let mut module = module.clone();
        module.output_gate = gate_mode;
        let mut projected = projection(2, 2, 3);
        projected.gate = gate_shape.map(|shape| Tensor::zeros(shape, &device));
        assert!(
            matches!(super::validate_projection(&module, &projected, None, None),
            Err(crate::Error::Shape { tensor, .. }) if tensor == "attention.gate")
        );
    }
}

/// Cross-attention validates both independent inputs before any linear call.
#[test]
fn projection_checks_empty_query_and_independent_key_value_extents() {
    let device = Default::default();
    let module = GroupedQueryAttentionConfig::new(2, 2, 1, 1)
        .init::<Flex>(&device)
        .unwrap();
    for shape in [[0, 1, 2], [1, 0, 2]] {
        assert!(matches!(module.project_self(Tensor::zeros(shape, &device)),
            Err(crate::Error::Shape { tensor, .. }) if tensor == "attention.query_input"));
    }
    for shape in [[2, 1, 2], [1, 0, 2], [1, 1, 3]] {
        assert!(
            matches!(module.project(Tensor::zeros([1, 1, 2], &device), Tensor::zeros(shape, &device)),
            Err(crate::Error::Shape { tensor, .. }) if tensor == "attention.key_value_input")
        );
    }
}

/// Query-only and key-only masks expand on different axes and invert once.
#[test]
fn padding_masks_cover_one_sided_masks_and_invalid_key_extents() {
    let device = Default::default();
    let query = TokenMask::new(Tensor::<Flex, 2, Bool>::from_bool(
        [[false, true]].into(),
        &device,
    ));
    let key = TokenMask::new(Tensor::<Flex, 2, Bool>::from_bool(
        [[true, false, true]].into(),
        &device,
    ));
    let query_only = padding_attention_mask(Some(&query), None, 1, 2, 2, 3)
        .unwrap()
        .unwrap();
    assert_eq!(
        query_only.into_data().to_vec::<bool>().unwrap(),
        [true, true, true, false, false, false].repeat(2)
    );
    let key_only = padding_attention_mask(None, Some(&key), 1, 2, 2, 3)
        .unwrap()
        .unwrap();
    assert_eq!(
        key_only.into_data().to_vec::<bool>().unwrap(),
        [false, true, false].repeat(4)
    );
    assert!(
        padding_attention_mask::<Flex>(None, None, 1, 2, 2, 3)
            .unwrap()
            .is_none()
    );
    assert!(
        matches!(padding_attention_mask(None, Some(&query), 1, 2, 2, 3),
        Err(crate::Error::Shape { tensor, .. }) if tensor == "key_mask")
    );
    for (batch, heads, queries, keys) in [(0, 1, 1, 1), (1, 0, 1, 1), (1, 1, 0, 1), (1, 1, 1, 0)] {
        assert!(padding_attention_mask::<Flex>(None, None, batch, heads, queries, keys).is_err());
    }
}

/// Eager logits apply scale/softcap before bias and combine exclusion sources.
#[test]
fn eager_softcap_bias_and_causal_mask_match_scalar_softmax_oracle() {
    let device = Default::default();
    for causal in [false, true] {
        for excluded_middle in [false, true] {
            let query = Tensor::<Flex, 4>::ones([1, 1, 2, 1], &device);
            let key = Tensor::from_floats([[[[1.0], [2.0], [3.0]]]], &device);
            let value = Tensor::from_floats([[[[2.0], [4.0], [8.0]]]], &device);
            let mask = excluded_middle.then(|| {
                Tensor::<Flex, 4, Bool>::from_bool(
                    [[[[false, true, false], [false, true, false]]]].into(),
                    &device,
                )
            });
            let bias = Tensor::from_floats(
                [[[[0.0, 2.0_f32.ln(), 0.0], [0.0, 2.0_f32.ln(), 0.0]]]],
                &device,
            );
            let output = dispatch_attention(
                query,
                key,
                value,
                mask,
                Some(bias),
                AttentionModuleOptions {
                    scale: Some(2.0),
                    softcap: Some(1.0),
                    is_causal: causal,
                },
                AttentionPrecisionMode::EagerFp32Softmax,
            )
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap();
            for (query_index, actual) in output.into_iter().enumerate() {
                let mut numerator = 0.0_f64;
                let mut denominator = 0.0_f64;
                for (key_index, value) in [2.0, 4.0, 8.0].into_iter().enumerate() {
                    if (causal && key_index > query_index + 1)
                        || (excluded_middle && key_index == 1)
                    {
                        continue;
                    }
                    let logit = (2.0 * (key_index + 1) as f64).tanh()
                        + if key_index == 1 { 2.0_f64.ln() } else { 0.0 };
                    numerator += value * logit.exp();
                    denominator += logit.exp();
                }
                assert!((f64::from(actual) - numerator / denominator).abs() < 1e-6);
            }
        }
    }
}

/// Full/MQA projections share the same gate convention and broadcast bias.
#[test]
fn output_gates_and_broadcast_bias_produce_exact_projected_contexts() {
    use burn::module::Param;

    let device = Default::default();
    for gate_mode in [
        AttentionOutputGate::None,
        AttentionOutputGate::Sigmoid,
        AttentionOutputGate::Silu,
    ] {
        let mut config = GroupedQueryAttentionConfig::new(2, 2, 1, 1);
        config.output_gate = gate_mode;
        config.precision_mode = AttentionPrecisionMode::EagerFp32Softmax;
        let mut module = config.init::<Flex>(&device).unwrap();
        module.o_proj.weight =
            Param::from_tensor(Tensor::from_floats([[1.0, 0.0], [0.0, 1.0]], &device));
        let mut projected = projection(2, 2, 2);
        projected.value = Tensor::from_floats([[[[2.0], [4.0]]], [[[2.0], [4.0]]]], &device);
        projected.gate =
            (gate_mode != AttentionOutputGate::None).then(|| Tensor::full([2, 2, 2], 2.0, &device));
        let bias = Tensor::from_floats(
            [[
                [[0.0, 3.0_f32.ln()], [0.0, 3.0_f32.ln()]],
                [[0.0, 3.0_f32.ln()], [0.0, 3.0_f32.ln()]],
            ]],
            &device,
        );
        let output = module.attend(projected, None, Some(bias), false).unwrap();
        let expected = match gate_mode {
            AttentionOutputGate::None => 3.5,
            AttentionOutputGate::Sigmoid => 3.5 / (1.0 + (-2.0_f32).exp()),
            AttentionOutputGate::Silu => 7.0 / (1.0 + (-2.0_f32).exp()),
        };
        for value in output.into_data().to_vec::<f32>().unwrap() {
            assert!((value - expected).abs() < 1e-6);
        }
    }
}

/// Zero-centered Q/K norms affect heads while V retains its projection bias.
#[test]
fn biased_cross_projection_applies_zero_centered_qk_norms_only_to_queries_and_keys() {
    use burn::module::Param;

    let device = Default::default();
    let mut config = GroupedQueryAttentionConfig::new(2, 1, 1, 2);
    config.bias = true;
    config.query_key_norm_epsilon = Some(1e-6);
    config.query_key_norm_zero_centered = true;
    let mut module = config.init::<Flex>(&device).unwrap();
    for linear in [&mut module.q_proj, &mut module.k_proj, &mut module.v_proj] {
        linear.weight = Param::from_tensor(Tensor::from_floats([[1.0, 0.0], [0.0, 1.0]], &device));
        linear.bias = Some(Param::from_tensor(Tensor::ones([2], &device)));
    }
    let output = module
        .project(
            Tensor::from_floats([[[3.0, 4.0]]], &device),
            Tensor::from_floats([[[0.0, 0.0], [1.0, 2.0]]], &device),
        )
        .unwrap();
    let expected_query = [4.0_f32, 5.0].map(|value| value / (20.5_f32 + 1e-6).sqrt());
    for (actual, expected) in output
        .query
        .into_data()
        .to_vec::<f32>()
        .unwrap()
        .into_iter()
        .zip(expected_query)
    {
        assert!((actual - expected).abs() < 1e-6);
    }
    let expected_key = [
        1.0 / (1.0_f32 + 1e-6).sqrt(),
        1.0 / (1.0_f32 + 1e-6).sqrt(),
        2.0 / (6.5_f32 + 1e-6).sqrt(),
        3.0 / (6.5_f32 + 1e-6).sqrt(),
    ];
    for (actual, expected) in output
        .key
        .into_data()
        .to_vec::<f32>()
        .unwrap()
        .into_iter()
        .zip(expected_key)
    {
        assert!((actual - expected).abs() < 1e-6);
    }
    assert_eq!(
        output.value.into_data().to_vec::<f32>().unwrap(),
        [1.0, 1.0, 2.0, 3.0]
    );
}

/// Unrepresentable causal positions fail before an arange can allocate storage.
#[cfg(target_pointer_width = "64")]
#[test]
fn causal_mask_rejects_lengths_beyond_i64() {
    let device = Default::default();
    let oversized = i64::MAX as usize + 1;
    for (query, key, message) in [(oversized, 1, "query length"), (1, oversized, "key length")] {
        let error = bottom_right_causal_mask::<Flex>(1, 1, query, key, &device).unwrap_err();
        assert!(error.to_string().contains(message));
    }
}

/// A padded cached suffix (`Q < K`) gives the same projected output in both graphs.
#[test]
fn cached_suffix_gqa_output_agrees_between_native_and_eager_graphs() {
    use burn::tensor::{TensorData, Tolerance};

    let device = Default::default();
    // One module whose lazily initialized parameters are shared by both graphs.
    let mut module = GroupedQueryAttentionConfig::new(8, 4, 2, 2)
        .init::<Flex>(&device)
        .unwrap();
    let hidden = |sequence: usize, seed: usize| {
        let values = (0..2 * sequence * 8)
            .map(|index| ((index * 7 + seed) % 13) as f32 * 0.15 - 0.9)
            .collect::<Vec<_>>();
        Tensor::<Flex, 3>::from_data(TensorData::new(values, [2, sequence, 8]), &device)
    };
    let projection = module.project(hidden(3, 1), hidden(7, 2)).unwrap();
    let valid = TokenMask::new(Tensor::<Flex, 2, Bool>::from_bool(
        [
            [true, true, true, true, true, true, true],
            [false, false, true, true, true, true, true],
        ]
        .into(),
        &device,
    ));
    let mask = padding_attention_mask(None, Some(&valid), 2, 4, 3, 7).unwrap();
    let actual = module
        .attend(projection.clone(), mask.clone(), None, true)
        .unwrap();
    module.precision_mode = AttentionPrecisionMode::EagerFp32Softmax;
    let expected = module.attend(projection, mask, None, true).unwrap();
    actual
        .into_data()
        .assert_approx_eq::<f32>(&expected.into_data(), Tolerance::rel_abs(1e-5, 1e-5));
}

/// Planned query ranges partition the rows and stop after the final chunk.
#[test]
fn execution_plan_ranges_partition_queries() {
    let plan = AttentionExecutionPolicy::query_chunked(4, 30)
        .plan(1, 2, 7, 5)
        .unwrap();
    assert_eq!(plan.query_ranges().collect::<Vec<_>>(), [0..3, 3..6, 6..7]);
    assert_eq!(plan.query_range(2), Some(6..7));
    assert_eq!(plan.query_range(3), None);
    let unbounded = AttentionExecutionPolicy::Unbounded
        .plan(1, 1, 5, 5)
        .unwrap();
    let ranges = unbounded.query_ranges().collect::<Vec<_>>();
    assert_eq!(ranges.len(), 1);
    assert_eq!(ranges[0], 0..5);
}

/// Public sliding-window key ranges follow bottom-right alignment and reject invalid input.
#[test]
fn sliding_window_key_ranges_cover_each_chunk() {
    use super::sliding_window_key_range;
    // Rows 2..4 of 4 sit at positions 6 and 7 of 8 keys; window 3 sees 4..8.
    assert_eq!(sliding_window_key_range(4, 8, 2..4, 3), Some(4..8));
    assert_eq!(sliding_window_key_range(4, 4, 0..2, 8), Some(0..2));
    assert_eq!(sliding_window_key_range(4, 3, 0..2, 2), None);
    assert_eq!(sliding_window_key_range(4, 8, 2..2, 2), None);
}

/// Sinks are opt-in, default off in serialized configs, and initialize one zero logit per query head.
#[test]
fn attention_sinks_are_opt_in_per_query_head_parameters() {
    let device = Default::default();
    let config = GroupedQueryAttentionConfig::new(8, 4, 2, 2);
    let mut json = serde_json::to_value(&config).unwrap();
    json.as_object_mut().unwrap().remove("attention_sinks");
    let restored: GroupedQueryAttentionConfig = serde_json::from_value(json).unwrap();
    assert!(!restored.attention_sinks);
    assert!(config.init::<Flex>(&device).unwrap().sinks.is_none());
    let module = config
        .with_attention_sinks(true)
        .init::<Flex>(&device)
        .unwrap();
    let sinks = module.sinks.as_ref().unwrap().val();
    assert_eq!(sinks.dims(), [4]);
    assert_eq!(sinks.into_data().to_vec::<f32>().unwrap(), [0.0; 4]);
}

/// Backend-default sinks cannot be soft-capped, while the eager graph supports it.
#[test]
fn backend_default_sinks_reject_softcap_at_configuration() {
    let mut config = GroupedQueryAttentionConfig::new(8, 2, 1, 4).with_attention_sinks(true);
    config.softcap = Some(30.0);
    assert!(matches!(config.validate(),
        Err(crate::Error::Configuration { field, .. }) if field == "attention.attention_sinks"));
    config
        .with_precision_mode(AttentionPrecisionMode::EagerFp32Softmax)
        .validate()
        .unwrap();
}

/// Builds a GQA module with sinks and a deterministic projection for sink tests.
fn sink_module(
    precision: AttentionPrecisionMode,
    sinks: [f32; 4],
) -> (
    super::GroupedQueryAttention<Flex>,
    AttentionProjection<Flex>,
) {
    use burn::{module::Param, tensor::TensorData};

    let device = Default::default();
    let mut module = GroupedQueryAttentionConfig::new(8, 4, 2, 2)
        .with_attention_sinks(true)
        .with_precision_mode(precision)
        .init::<Flex>(&device)
        .unwrap();
    module.sinks = Some(Param::from_tensor(Tensor::from_floats(sinks, &device)));
    module.o_proj.weight = Param::from_tensor(Tensor::eye(8, &device));
    let tensor = |shape: [usize; 4], seed: usize| {
        let count = shape.iter().product::<usize>();
        let values = (0..count)
            .map(|index| ((index * 5 + seed) % 11) as f32 * 0.2 - 1.0)
            .collect::<Vec<_>>();
        Tensor::<Flex, 4>::from_data(TensorData::new(values, shape), &device)
    };
    let projection = AttentionProjection {
        query: tensor([2, 4, 3, 2], 1),
        key: tensor([2, 2, 5, 2], 2),
        value: tensor([2, 2, 5, 2], 3),
        gate: None,
    };
    (module, projection)
}

/// Native and eager sinks agree, and a vanishing sink recovers ordinary attention.
#[test]
fn module_sinks_agree_across_graphs_and_vanish_when_very_negative() {
    use burn::tensor::Tolerance;

    let sinks = [0.3, -0.7, 1.1, 0.0];
    let (native, projection) = sink_module(AttentionPrecisionMode::BackendDefault, sinks);
    let (eager, _) = sink_module(AttentionPrecisionMode::EagerFp32Softmax, sinks);
    let mut eager = eager;
    eager.o_proj = native.o_proj.clone();
    for causal in [false, true] {
        let expected = eager
            .attend(projection.clone(), None, None, causal)
            .unwrap();
        native
            .attend(projection.clone(), None, None, causal)
            .unwrap()
            .into_data()
            .assert_approx_eq::<f32>(&expected.into_data(), Tolerance::rel_abs(1e-5, 1e-5));
    }
    let (mut vanishing, _) = sink_module(AttentionPrecisionMode::BackendDefault, [-60.0; 4]);
    let with_sinks = vanishing
        .attend(projection.clone(), None, None, true)
        .unwrap();
    vanishing.sinks = None;
    let without_sinks = vanishing.attend(projection, None, None, true).unwrap();
    with_sinks
        .into_data()
        .assert_approx_eq::<f32>(&without_sinks.into_data(), Tolerance::rel_abs(1e-6, 1e-6));
}

/// Pattern rows exclude the sink column, and fully excluded rows have zero context.
#[test]
fn sink_patterns_sum_below_one_and_fully_excluded_rows_are_zero() {
    use crate::intervention::{Hooks, Layer};

    let (module, mut projection) =
        sink_module(AttentionPrecisionMode::EagerFp32Softmax, [2.0_f32.ln(); 4]);
    // Zero queries give equal scores, so with 5 keys p_key = 1 / (5 + 2).
    projection.query = projection.query.zeros_like();
    let device = Default::default();
    let mut excluded = vec![false; 2 * 4 * 3 * 5];
    excluded[..5].fill(true);
    let mask = Tensor::<Flex, 4, Bool>::from_data(
        burn::tensor::TensorData::new(excluded, [2, 4, 3, 5]),
        &device,
    );
    let points = Layer::decoder(0).self_attention();
    let mut hooks = Hooks::new();
    let pattern = hooks.capture(points.pattern());
    let context = hooks.capture(points.head_context());
    hooks
        .run(points.all(), |hooks| {
            module.attend_with_hooks(projection, Some(mask), None, false, points, hooks)
        })
        .unwrap();
    let pattern = hooks
        .get(&pattern)
        .unwrap()
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    assert!(pattern[..5].iter().all(|value| *value == 0.0));
    for value in &pattern[5..] {
        assert!((value - 1.0 / 7.0).abs() < 1e-6, "{value}");
    }
    let context = hooks
        .get(&context)
        .unwrap()
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    assert_eq!(&context[..2], &[0.0, 0.0]);
}

/// Sink parameters must match the query-head count, dtype, and device.
#[test]
fn malformed_sinks_fail_projection_validation() {
    use burn::{module::Param, tensor::DType};

    let device = Default::default();
    let (mut module, projection) = sink_module(AttentionPrecisionMode::BackendDefault, [0.0; 4]);
    module.sinks = Some(Param::from_tensor(Tensor::zeros([3], &device)));
    let error = module
        .attend(projection.clone(), None, None, false)
        .unwrap_err();
    assert!(error.to_string().contains("attention.sinks"));
    module.sinks = Some(Param::from_tensor(
        Tensor::<Flex, 1>::zeros([4], &device).cast(DType::F64),
    ));
    let error = module.attend(projection, None, None, false).unwrap_err();
    assert!(error.to_string().contains("attention.sinks"));
}

/// Sink logits receive gradients through both operation graphs.
#[cfg(feature = "autodiff")]
#[test]
fn attention_sinks_are_trainable() {
    use burn::backend::Autodiff;
    /// Differentiable CPU backend for the sink-gradient smoke test.
    type Ad = Autodiff<Flex>;
    let device = Default::default();
    for precision in [
        AttentionPrecisionMode::BackendDefault,
        AttentionPrecisionMode::EagerFp32Softmax,
    ] {
        let module = GroupedQueryAttentionConfig::new(4, 2, 1, 2)
            .with_attention_sinks(true)
            .with_precision_mode(precision)
            .init::<Ad>(&device)
            .unwrap();
        let output = module
            .forward_without_position_encoding(
                Tensor::<Ad, 3>::ones([1, 3, 4], &device),
                None,
                true,
            )
            .unwrap();
        let gradients = output.sum().backward();
        let sink_gradient = module
            .sinks
            .as_ref()
            .unwrap()
            .grad(&gradients)
            .expect("sink logits receive gradients");
        assert!(
            sink_gradient
                .into_data()
                .to_vec::<f32>()
                .unwrap()
                .iter()
                .all(|value| value.is_finite())
        );
    }
}

/// Independent bias flags allocate exactly the requested projection biases.
#[test]
fn projection_bias_flags_are_independent() {
    use super::ProjectionBias;

    let device = Default::default();
    let has_bias = |config: GroupedQueryAttentionConfig| {
        let module = config.init::<Flex>(&device).unwrap();
        [
            module.q_proj.bias.is_some(),
            module.k_proj.bias.is_some(),
            module.v_proj.bias.is_some(),
            module.o_proj.bias.is_some(),
        ]
    };
    let base = GroupedQueryAttentionConfig::new(8, 4, 2, 2);
    assert_eq!(has_bias(base.clone()), [false; 4]);
    let mut uniform = base.clone();
    uniform.bias = true;
    assert_eq!(has_bias(uniform.clone()), [true; 4]);
    // Qwen2-style biased Q/K/V with an unbiased output projection.
    let qwen2 = uniform.with_projection_bias(ProjectionBias::query_key_value_only());
    assert_eq!(has_bias(qwen2.clone()), [true, true, true, false]);
    assert_eq!(
        qwen2.effective_projection_bias(),
        ProjectionBias::query_key_value_only()
    );
    let output_only = base.with_projection_bias(ProjectionBias {
        output: true,
        ..ProjectionBias::uniform(false)
    });
    assert_eq!(has_bias(output_only), [false, false, false, true]);
    // Older serialized configs without the field keep the uniform `bias` meaning.
    let mut json = serde_json::to_value(&qwen2).unwrap();
    json.as_object_mut().unwrap().remove("projection_bias");
    let legacy: GroupedQueryAttentionConfig = serde_json::from_value(json).unwrap();
    assert_eq!(
        legacy.effective_projection_bias(),
        ProjectionBias::uniform(true)
    );
    let round_trip: GroupedQueryAttentionConfig =
        serde_json::from_str(&serde_json::to_string(&qwen2).unwrap()).unwrap();
    assert_eq!(round_trip, qwen2);
}

/// Full-width Q/K norms normalize every projected channel of a token before the head split.
#[test]
fn full_width_query_key_norm_reduces_over_all_heads() {
    use super::QueryKeyNormScope;
    use burn::{module::Param, tensor::Tolerance};

    let device = Default::default();
    for gate in [AttentionOutputGate::None, AttentionOutputGate::Sigmoid] {
        let mut config = GroupedQueryAttentionConfig::new(4, 2, 1, 2)
            .with_query_key_norm(1e-6, QueryKeyNormScope::FullWidth);
        config.output_gate = gate;
        let mut module = config.init::<Flex>(&device).unwrap();
        assert_eq!(module.query_norm.as_ref().unwrap().weight.dims(), [4]);
        assert_eq!(module.key_norm.as_ref().unwrap().weight.dims(), [2]);
        let query_width = if gate == AttentionOutputGate::None {
            4
        } else {
            8
        };
        // Identity-like projections: query channels are [1, 2, 3, 4], keys [1, 2].
        let mut query_weight = vec![0.0_f32; 4 * query_width];
        for channel in 0..4 {
            // Gated projections interleave [query_0, gate_0, query_1, gate_1] per head.
            let column = if gate == AttentionOutputGate::None {
                channel
            } else {
                (channel / 2) * 4 + channel % 2
            };
            query_weight[channel * query_width + column] = 1.0;
        }
        module.q_proj.weight = Param::from_tensor(Tensor::from_data(
            burn::tensor::TensorData::new(query_weight, [4, query_width]),
            &device,
        ));
        module.k_proj.weight = Param::from_tensor(Tensor::from_floats(
            [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]],
            &device,
        ));
        let projection = module
            .project_self(Tensor::from_floats([[[1.0, 2.0, 3.0, 4.0]]], &device))
            .unwrap();
        let rms = (30.0_f32 / 4.0 + 1e-6).sqrt();
        projection.query.into_data().assert_approx_eq::<f32>(
            &burn::tensor::TensorData::from([[[[1.0 / rms, 2.0 / rms]], [[3.0 / rms, 4.0 / rms]]]]),
            Tolerance::absolute(1e-6),
        );
        let key_rms = (5.0_f32 / 2.0 + 1e-6).sqrt();
        projection.key.into_data().assert_approx_eq::<f32>(
            &burn::tensor::TensorData::from([[[[1.0 / key_rms, 2.0 / key_rms]]]]),
            Tolerance::absolute(1e-6),
        );
    }
}
