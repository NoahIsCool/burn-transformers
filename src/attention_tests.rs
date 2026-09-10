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
