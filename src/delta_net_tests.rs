//! Delta-rule tests comparing recurrent and chunked updates, state continuation, and validation.

use burn::{
    backend::Flex,
    tensor::{Bool, DType, Int, Tensor, TensorData, Tolerance, backend::Backend},
};

use crate::architecture::TokenMask;

use super::{
    CHUNKS_PER_BLOCK, ChunkBlock, ChunkOperands, DeltaNetMode, DeltaRuleConfig, DeltaRuleInput,
    GatedDeltaNet, GatedDeltaNetConfig, GatedDeltaNetState, PreparedDeltaRule, block_chunk_ranges,
    causal_decay_matrix, chunked_gated_delta_rule, mask_hidden, prefers_recurrence,
    recurrent_gated_delta_rule, recurrent_gated_delta_rule_with_snapshots, softplus,
    trailing_unit_axis, unit_lower_inverse,
};

/// Deterministic F32 CPU backend for recurrent and chunked delta-rule comparisons.
type TestBackend = Flex<f32, i32>;

/// Supplying an all-valid mask cannot promote an F16 residual stream to F32.
#[test]
fn all_valid_mask_preserves_activation_dtype_and_values() {
    let device = Default::default();
    let hidden = Tensor::<TestBackend, 3>::from_data(
        TensorData::new(vec![0.5_f32, -1.0, 2.0, -3.0], [1, 2, 2]),
        &device,
    )
    .cast(DType::F16);
    let mask = TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
        [[true, true]].into(),
        &device,
    ));
    let masked = mask_hidden(hidden.clone(), Some(&mask));

    assert_eq!(masked.dtype(), DType::F16);
    masked.cast(DType::F32).into_data().assert_approx_eq::<f32>(
        &hidden.cast(DType::F32).into_data(),
        Tolerance::absolute(0.0),
    );
}

/// Builds deterministic nontrivial inputs spanning two heads and unequal widths.
fn fixture(sequence: usize) -> DeltaRuleInput<TestBackend> {
    let device = Default::default();
    let query = Tensor::arange(0..(2 * sequence * 2 * 3) as i64, &device)
        .float()
        .reshape([2, sequence, 2, 3])
        / 19.0
        - 0.7;
    let key = Tensor::arange(1..(2 * sequence * 2 * 3 + 1) as i64, &device)
        .float()
        .reshape([2, sequence, 2, 3])
        / 23.0
        - 0.4;
    let value = Tensor::arange(0..(2 * sequence * 2 * 2) as i64, &device)
        .float()
        .reshape([2, sequence, 2, 2])
        / 11.0
        - 0.2;
    let log_decay = Tensor::full([2, sequence, 2], -0.17, &device);
    let beta = Tensor::full([2, sequence, 2], 0.63, &device);
    DeltaRuleInput {
        query,
        key,
        value,
        log_decay,
        beta,
        initial_state: None,
        token_mask: None,
    }
}

/// Every chunk geometry matches the transparent recurrent definition.
#[test]
fn chunked_prefill_matches_recurrence() {
    for sequence in 1..=9 {
        for chunk_size in 1..=6 {
            let config = DeltaRuleConfig {
                chunk_size,
                ..Default::default()
            };
            let recurrent = recurrent_gated_delta_rule(fixture(sequence), config)
                .expect("fixture must be valid");
            let chunked =
                chunked_gated_delta_rule(fixture(sequence), config).expect("fixture must be valid");
            chunked.values.into_data().assert_approx_eq::<f32>(
                &recurrent.values.into_data(),
                Tolerance::rel_abs(1e-4, 1e-5),
            );
            chunked.final_state.into_data().assert_approx_eq::<f32>(
                &recurrent.final_state.into_data(),
                Tolerance::rel_abs(1e-4, 1e-5),
            );
        }
    }
}

/// Passing the first segment's state reproduces a single uninterrupted scan.
#[test]
fn recurrent_state_composes_across_decode_calls() {
    let complete =
        recurrent_gated_delta_rule(fixture(4), Default::default()).expect("fixture must be valid");
    let base = fixture(4);
    let first = recurrent_gated_delta_rule(
        DeltaRuleInput {
            query: base.query.clone().narrow(1, 0, 2),
            key: base.key.clone().narrow(1, 0, 2),
            value: base.value.clone().narrow(1, 0, 2),
            log_decay: base.log_decay.clone().narrow(1, 0, 2),
            beta: base.beta.clone().narrow(1, 0, 2),
            initial_state: None,
            token_mask: None,
        },
        Default::default(),
    )
    .expect("first segment must be valid");
    let second = recurrent_gated_delta_rule(
        DeltaRuleInput {
            query: base.query.narrow(1, 2, 2),
            key: base.key.narrow(1, 2, 2),
            value: base.value.narrow(1, 2, 2),
            log_decay: base.log_decay.narrow(1, 2, 2),
            beta: base.beta.narrow(1, 2, 2),
            initial_state: Some(first.final_state),
            token_mask: None,
        },
        Default::default(),
    )
    .expect("second segment must be valid");
    let expected_suffix = complete.values.narrow(1, 2, 2);
    second
        .values
        .into_data()
        .assert_approx_eq::<f32>(&expected_suffix.into_data(), Tolerance::rel_abs(1e-4, 1e-5));
    second.final_state.into_data().assert_approx_eq::<f32>(
        &complete.final_state.into_data(),
        Tolerance::rel_abs(1e-4, 1e-5),
    );
}

/// Invalid tokens emit zero and leave recurrent memory unchanged.
#[test]
fn token_mask_blocks_state_updates() {
    let device = Default::default();
    let mut masked = fixture(3);
    masked.token_mask = Some(Tensor::<TestBackend, 2, Bool>::from_bool(
        [[true, false, true], [true, false, true]].into(),
        &device,
    ));
    let output = recurrent_gated_delta_rule(masked, Default::default())
        .expect("masked fixture must be valid");
    let middle = output.values.narrow(1, 1, 1).into_data();
    middle.assert_approx_eq::<f32>(
        &TensorData::zeros::<f32, _>([2, 1, 2, 2]),
        Tolerance::absolute(1e-7),
    );
}

/// Every public tensor extent is checked before reshape or matmul dispatch.
#[test]
fn shape_validation_is_contextual() {
    let device = Default::default();
    let mut input = fixture(2);
    input.key = Tensor::zeros([2, 2, 2, 4], &device);
    assert!(
        recurrent_gated_delta_rule(input, Default::default())
            .expect_err("wrong key width must fail")
            .to_string()
            .contains("delta_net.key")
    );

    let mut input = fixture(2);
    input.beta = Tensor::zeros([2, 3, 2], &device);
    assert!(
        chunked_gated_delta_rule(input, Default::default())
            .expect_err("wrong beta sequence must fail")
            .to_string()
            .contains("delta_net.beta")
    );

    let mut input = fixture(2);
    input.initial_state = Some(Tensor::zeros([2, 2, 4, 2], &device));
    assert!(
        recurrent_gated_delta_rule(input, Default::default())
            .expect_err("wrong state key width must fail")
            .to_string()
            .contains("initial_state")
    );

    let mut input = fixture(2);
    input.token_mask = Some(Tensor::<TestBackend, 2, Int>::ones([2, 3], &device).bool());
    assert!(
        recurrent_gated_delta_rule(input, Default::default())
            .expect_err("wrong mask sequence must fail")
            .to_string()
            .contains("token_mask")
    );
}

/// Numerical and block controls reject non-finite or degenerate values.
#[test]
fn config_validation_rejects_invalid_controls() {
    assert!(
        DeltaRuleConfig {
            qk_l2_epsilon: f64::NAN,
            ..Default::default()
        }
        .validate()
        .is_err()
    );
    assert!(
        DeltaRuleConfig {
            chunk_size: 0,
            ..Default::default()
        }
        .validate()
        .is_err()
    );
}

/// The projected module's blockwise prefill equals repeated cached decode.
#[test]
fn projected_mixer_prefill_matches_decode() {
    let device = Default::default();
    TestBackend::seed(&device, 7);
    let mut config = GatedDeltaNetConfig::new(4, 1, 2, 2, 2);
    config.convolution_kernel = 3;
    config.prefill_chunk_size = 2;
    let mixer = config
        .init::<TestBackend>(&device)
        .expect("valid mixer must initialize");
    let hidden = Tensor::from_floats(
        [
            [
                [0.1, -0.2, 0.3, 0.4],
                [0.5, 0.2, -0.1, 0.7],
                [-0.4, 0.6, 0.2, -0.3],
                [0.9, -0.8, 0.1, 0.2],
                [0.3, 0.5, -0.7, 0.4],
            ],
            [
                [-0.3, 0.8, 0.4, -0.1],
                [0.2, -0.5, 0.9, 0.6],
                [0.1, 0.7, -0.6, 0.3],
                [-0.8, 0.4, 0.5, 0.2],
                [0.6, -0.2, 0.1, -0.9],
            ],
        ],
        &device,
    );

    let mut prefill_state = GatedDeltaNetState::zeros(&config, 2, &device).expect("valid state");
    let prefill = mixer
        .forward_with_state(
            hidden.clone(),
            None,
            &mut prefill_state,
            DeltaNetMode::ChunkedPrefill,
        )
        .expect("prefill must succeed");

    let mut decode_state = GatedDeltaNetState::zeros(&config, 2, &device).expect("valid state");
    let mut decode_tokens = Vec::new();
    for index in 0..hidden.dims()[1] {
        decode_tokens.push(
            mixer
                .forward_with_state(
                    hidden.clone().narrow(1, index, 1),
                    None,
                    &mut decode_state,
                    DeltaNetMode::Recurrent,
                )
                .expect("decode step must succeed"),
        );
    }
    let decode = Tensor::cat(decode_tokens, 1);
    prefill
        .into_data()
        .assert_approx_eq::<f32>(&decode.into_data(), Tolerance::rel_abs(2e-4, 2e-5));
    prefill_state.recurrent.into_data().assert_approx_eq::<f32>(
        &decode_state.recurrent.into_data(),
        Tolerance::rel_abs(2e-4, 2e-5),
    );
    prefill_state
        .convolution
        .into_data()
        .assert_approx_eq::<f32>(
            &decode_state.convolution.into_data(),
            Tolerance::rel_abs(2e-4, 2e-5),
        );
}

/// Model topology validation rejects grouping and allocation overflows.
#[test]
fn projected_config_rejects_invalid_topology() {
    assert!(GatedDeltaNetConfig::new(4, 3, 4, 2, 2).validate().is_err());
    assert!(GatedDeltaNetConfig::new(0, 1, 1, 1, 1).validate().is_err());
    assert!(
        GatedDeltaNetConfig::new(1, usize::MAX, usize::MAX, usize::MAX, usize::MAX)
            .validate()
            .is_err()
    );
}

/// External state dimensions are checked before convolution or recurrence.
#[test]
fn projected_mixer_rejects_wrong_state_shape() {
    let device = Default::default();
    let config = GatedDeltaNetConfig::new(4, 1, 2, 2, 2);
    let mixer = config
        .init::<TestBackend>(&device)
        .expect("valid mixer must initialize");
    let mut state = GatedDeltaNetState::zeros(&config, 1, &device).expect("valid state");
    state.recurrent = Tensor::zeros([1, 2, 3, 2], &device);
    let error = mixer
        .forward_with_state(
            Tensor::zeros([1, 1, 4], &device),
            None,
            &mut state,
            DeltaNetMode::Recurrent,
        )
        .expect_err("wrong recurrent width must fail");
    assert!(error.to_string().contains("state.recurrent"));
}

/// Malformed input axes fail before either nonzero state buffer is replaced.
#[test]
fn projected_mixer_shape_rejections_preserve_both_state_buffers() {
    let device = Default::default();
    let config = GatedDeltaNetConfig::new(4, 1, 2, 2, 2);
    let mixer = config
        .init::<TestBackend>(&device)
        .expect("valid mixer must initialize");
    let valid_convolution = [1, config.convolution_channels().unwrap(), 3];
    let cases = [
        ("delta_net.hidden", [0, 2, 4], None, valid_convolution),
        ("delta_net.hidden", [1, 0, 4], None, valid_convolution),
        ("delta_net.hidden", [1, 2, 3], None, valid_convolution),
        (
            "delta_net.token_mask",
            [1, 2, 4],
            Some([2, 2]),
            valid_convolution,
        ),
        (
            "delta_net.token_mask",
            [1, 2, 4],
            Some([1, 3]),
            valid_convolution,
        ),
        ("delta_net.state.convolution", [1, 2, 4], None, [2, 8, 3]),
        ("delta_net.state.convolution", [1, 2, 4], None, [1, 9, 3]),
        ("delta_net.state.convolution", [1, 2, 4], None, [1, 8, 2]),
    ];
    for mode in [DeltaNetMode::Recurrent, DeltaNetMode::ChunkedPrefill] {
        for (context, hidden_shape, mask_shape, convolution_shape) in cases {
            let mut state = GatedDeltaNetState::zeros(&config, 1, &device).expect("valid state");
            state.convolution = Tensor::full(convolution_shape, 0.25, &device);
            state.recurrent = Tensor::full([1, 2, 2, 2], -0.5, &device);
            // Own independent host bytes: a shared tensor clone alone could
            // conceal an accidental in-place mutation of the same storage.
            let convolution_before = TensorData::new(
                vec![0.25_f32; convolution_shape.iter().product()],
                convolution_shape,
            );
            let recurrent_before = TensorData::new(vec![-0.5_f32; 8], [1, 2, 2, 2]);
            let mask = mask_shape.map(|shape| {
                TokenMask::new(Tensor::<TestBackend, 2, Int>::ones(shape, &device).bool())
            });
            let error = mixer
                .forward_with_state(
                    Tensor::zeros(hidden_shape, &device),
                    mask.as_ref(),
                    &mut state,
                    mode,
                )
                .expect_err("invalid shape must fail before computation");
            let crate::Error::Shape {
                tensor,
                expected,
                actual,
            } = error
            else {
                panic!("expected contextual shape error for {context}: {error}");
            };
            assert_eq!(tensor, context);
            let (expected_shape, actual_shape) = match context {
                "delta_net.hidden" => (
                    "[batch>0, sequence>0, 4]".to_owned(),
                    format!("{hidden_shape:?}"),
                ),
                "delta_net.token_mask" => {
                    ("[1, 2]".to_owned(), format!("{:?}", mask_shape.unwrap()))
                }
                _ => (
                    format!("{valid_convolution:?}"),
                    format!("{convolution_shape:?}"),
                ),
            };
            assert_eq!(expected, expected_shape);
            assert_eq!(actual, actual_shape);
            state
                .convolution
                .into_data()
                .assert_eq(&convolution_before, true);
            state
                .recurrent
                .into_data()
                .assert_eq(&recurrent_before, true);
        }
    }
}

/// Both algorithms reject the same supplied shape before consuming caller state.
fn assert_rule_shape_contract(
    mut input: DeltaRuleInput<TestBackend>,
    context: &str,
    expected_shape: &str,
    actual_shape: &str,
) {
    let device = Default::default();
    let initial = Tensor::<TestBackend, 4>::full([2, 2, 3, 2], 0.25, &device);
    input.initial_state = Some(initial.clone());
    let expected_state = TensorData::new(vec![0.25_f32; 24], [2, 2, 3, 2]);
    for execute in [
        recurrent_gated_delta_rule::<TestBackend>,
        chunked_gated_delta_rule::<TestBackend>,
    ] {
        let error = execute(input.clone(), DeltaRuleConfig::default())
            .expect_err("invalid external geometry must fail before tensor math");
        let crate::Error::Shape {
            tensor,
            expected,
            actual,
        } = error
        else {
            panic!("expected shape context {context}, got {error}");
        };
        assert_eq!(tensor, context);
        assert_eq!(expected, expected_shape);
        assert_eq!(actual, actual_shape);
        // Compare against independent host values, not another tensor alias.
        assert_eq!(initial.dims(), [2, 2, 3, 2]);
        assert_eq!(initial.dtype(), DType::F32);
        initial.to_data().assert_eq(&expected_state, true);
    }
}

/// Empty batches are invalid, while a unit-width convolution needs no history.
#[test]
fn delta_public_contract_state_construction_checks_batch_and_zero_history() {
    let device = Default::default();
    let mut config = GatedDeltaNetConfig::new(5, 2, 6, 3, 2);
    let error = GatedDeltaNetState::<TestBackend>::zeros(&config, 0, &device)
        .expect_err("a recurrent state must have at least one batch row");
    assert!(
        matches!(error, crate::Error::Configuration { field, reason }
        if field == "delta_net.batch_size" && reason == "must be non-zero")
    );

    // Two key heads feed six value heads. The convolution holds Q, K and V,
    // hence 2*(2*3) + 6*2 = 24 channels, independent of batch and kernel width.
    for batch in [1, 3] {
        for kernel in [1, 4] {
            config.convolution_kernel = kernel;
            let state = GatedDeltaNetState::<TestBackend>::zeros(&config, batch, &device)
                .expect("valid grouped-head state");
            assert_eq!(state.convolution.dims(), [batch, 24, kernel - 1]);
            assert_eq!(state.convolution.dtype(), DType::F32);
            assert_eq!(state.recurrent.dims(), [batch, 6, 3, 2]);
            assert_eq!(state.recurrent.dtype(), DType::F32);
            assert_eq!(
                state.convolution.to_data().to_vec::<f32>().unwrap(),
                vec![0.0; batch * 24 * (kernel - 1)]
            );
            assert_eq!(
                state.recurrent.to_data().to_vec::<f32>().unwrap(),
                vec![0.0; batch * 6 * 3 * 2]
            );
        }
    }
}

/// Every empty query axis is rejected identically by recurrence and chunked prefill.
#[test]
fn delta_public_contract_rule_rejects_each_empty_query_axis() {
    let device = Default::default();
    for axis in 0..4 {
        let mut input = fixture(2);
        let mut shape = [2, 2, 2, 3];
        shape[axis] = 0;
        input.query = Tensor::from_data(TensorData::new(Vec::<f32>::new(), shape), &device);
        assert_rule_shape_contract(
            input,
            "delta_net.query",
            "non-empty [batch, sequence, heads, key_width]",
            &format!("{shape:?}"),
        );
    }
}

/// Value batch/time/head axes must align, and its independent value width is nonzero.
#[test]
fn delta_public_contract_rule_rejects_value_alignment_and_empty_width() {
    let device = Default::default();
    for shape in [[1, 2, 2, 2], [2, 1, 2, 2], [2, 2, 1, 2], [2, 2, 2, 0]] {
        let mut input = fixture(2);
        input.value = Tensor::from_data(
            TensorData::new(vec![0.5_f32; shape.iter().product()], shape),
            &device,
        );
        assert_rule_shape_contract(
            input,
            "delta_net.value",
            "[2, 2, 2, value_width>0]",
            &format!("{shape:?}"),
        );
    }
}

/// Decay axes cannot broadcast accidentally; a corrected scalar input has a host oracle.
#[test]
fn delta_public_contract_decay_shape_validation_and_scalar_recurrence() {
    let device = Default::default();
    for shape in [[1, 2, 2], [2, 1, 2], [2, 2, 1]] {
        let mut input = fixture(2);
        input.log_decay = Tensor::full(shape, -0.2, &device);
        assert_rule_shape_contract(
            input,
            "delta_net.log_decay",
            "[2, 2, 2]",
            &format!("{shape:?}"),
        );
    }

    let log_decay = 0.8_f32.ln();
    let initial = Tensor::<TestBackend, 4>::full([1, 1, 1, 1], 1.25, &device);
    let input = DeltaRuleInput {
        query: Tensor::full([1, 1, 1, 1], 2.0, &device),
        key: Tensor::full([1, 1, 1, 1], 3.0, &device),
        value: Tensor::full([1, 1, 1, 1], 4.0, &device),
        log_decay: Tensor::full([1, 1, 1], log_decay, &device),
        beta: Tensor::full([1, 1, 1], 0.5, &device),
        initial_state: Some(initial.clone()),
        token_mask: None,
    };
    // For scalar normalized q,k, S' = exp(g)S + k*beta*(v-k*exp(g)S)
    // and y=q*S'. Use host f64 arithmetic, independently of either tensor path.
    let epsilon = DeltaRuleConfig::default().qk_l2_epsilon;
    let q = 2.0 / (4.0 + epsilon).sqrt();
    let k = 3.0 / (9.0 + epsilon).sqrt();
    let decayed = f64::from(log_decay).exp() * 1.25;
    let expected_state = decayed + k * 0.5 * (4.0 - k * decayed);
    let expected_value = q * expected_state;
    for execute in [
        recurrent_gated_delta_rule::<TestBackend>,
        chunked_gated_delta_rule::<TestBackend>,
    ] {
        let output = execute(input.clone(), DeltaRuleConfig::default()).unwrap();
        assert_eq!(output.values.dims(), [1, 1, 1, 1]);
        assert_eq!(output.values.dtype(), DType::F32);
        assert_eq!(output.final_state.dims(), [1, 1, 1, 1]);
        assert_eq!(output.final_state.dtype(), DType::F32);
        for (tensor, expected) in [
            (output.values, expected_value),
            (output.final_state, expected_state),
        ] {
            let values = tensor.to_data().to_vec::<f32>().unwrap();
            assert_eq!(values.len(), 1);
            assert!(values[0].is_finite());
            assert!((f64::from(values[0]) - expected).abs() < 1e-6);
        }
        assert_eq!(initial.to_data().to_vec::<f32>().unwrap(), [1.25]);
    }
}

/// The pre-refactor chunked prefill (one triangular solve per chunk), kept as a parity oracle.
fn legacy_chunked(
    input: DeltaRuleInput<TestBackend>,
    config: DeltaRuleConfig,
) -> (Tensor<TestBackend, 4>, Tensor<TestBackend, 4>) {
    let prepared = PreparedDeltaRule::new(input, config.qk_l2_epsilon).unwrap();
    // The oracle works head-major: `[batch, heads, sequence, ·]`.
    let head_major = |tensor: &Tensor<TestBackend, 4>| tensor.clone().swap_dims(1, 2);
    let head_scalars = |tensor: &Tensor<TestBackend, 3>| tensor.clone().swap_dims(1, 2);
    let (query_all, key_all, value_all) = (
        head_major(&prepared.query),
        head_major(&prepared.key),
        head_major(&prepared.value),
    );
    let (beta_all, decay_all) = (
        head_scalars(&prepared.beta),
        head_scalars(&prepared.log_decay),
    );
    let [batch, heads, sequence, key_width] = query_all.dims();
    let value_width = value_all.dims()[3];
    let mut state = prepared
        .initial_state
        .unwrap_or_else(|| Tensor::zeros([batch, heads, key_width, value_width], &prepared.device));
    let mut outputs = Vec::new();
    for start in (0..sequence).step_by(config.chunk_size) {
        let width = config.chunk_size.min(sequence - start);
        let query = query_all.clone().narrow(2, start, width);
        let key = key_all.clone().narrow(2, start, width);
        let value = value_all.clone().narrow(2, start, width);
        let beta = beta_all.clone().narrow(2, start, width);
        let cumulative_decay = decay_all.clone().narrow(2, start, width).cumsum(2);
        let beta_expanded = beta.unsqueeze_dim::<4>(3);
        let value_beta = value.clone() * beta_expanded.clone();
        let key_beta = key.clone() * beta_expanded;
        let decay_difference = cumulative_decay.clone().unsqueeze_dim::<4>(3)
            - cumulative_decay.clone().unsqueeze_dim::<4>(2);
        let decay_matrix = decay_difference.tril(0).exp().tril(0);
        let mut inverse_delta = (-(key_beta.clone().matmul(key.clone().swap_dims(2, 3)))
            * decay_matrix.clone())
        .tril(-1);
        for row in 1..width {
            let row_values =
                inverse_delta
                    .clone()
                    .slice([0..batch, 0..heads, row..row + 1, 0..row]);
            let solved_prefix = inverse_delta
                .clone()
                .slice([0..batch, 0..heads, 0..row, 0..row]);
            let solved_row = row_values.clone() + row_values.matmul(solved_prefix);
            inverse_delta =
                inverse_delta.slice_assign([0..batch, 0..heads, row..row + 1, 0..row], solved_row);
        }
        let identity = Tensor::<TestBackend, 2>::eye(width, &prepared.device)
            .reshape([1, 1, width, width])
            .expand([batch, heads, width, width]);
        let delta_transform = inverse_delta + identity;
        let transformed_value = delta_transform.clone().matmul(value_beta);
        let decayed_key_beta = key_beta * cumulative_decay.clone().exp().unsqueeze_dim::<4>(3);
        let cumulative_key = delta_transform.matmul(decayed_key_beta);
        let corrected_value = transformed_value - cumulative_key.matmul(state.clone());
        let state_output = (query.clone() * cumulative_decay.clone().exp().unsqueeze_dim::<4>(3))
            .matmul(state.clone());
        let causal_attention = (query.matmul(key.clone().swap_dims(2, 3)) * decay_matrix).tril(0);
        outputs.push(state_output + causal_attention.matmul(corrected_value.clone()));
        let final_decay = cumulative_decay.clone().narrow(2, width - 1, 1);
        let decayed_keys = key
            * (final_decay.clone() - cumulative_decay)
                .exp()
                .unsqueeze_dim::<4>(3);
        state = state * final_decay.exp().unsqueeze_dim::<4>(3)
            + decayed_keys.swap_dims(2, 3).matmul(corrected_value);
    }
    (Tensor::cat(outputs, 2).swap_dims(1, 2), state)
}

/// Builds smooth `[batch=1, sequence, heads=2, width]` inputs for long-sequence checks.
fn long_fixture(sequence: usize) -> DeltaRuleInput<TestBackend> {
    smooth_fixture(sequence, 2, 4, 3)
}

/// Builds smooth `[1, sequence, heads, key_width | value_width]` inputs from sine patterns.
///
/// Decays lie in `[-0.25, -0.05]` and `beta = 0.7`, so every chunk exercises
/// non-trivial decay matrices and delta corrections without overflow.
fn smooth_fixture(
    sequence: usize,
    heads: usize,
    key_width: usize,
    value_width: usize,
) -> DeltaRuleInput<TestBackend> {
    let device = Default::default();
    let pattern = |scale: f64, shape: [usize; 4]| {
        let count = shape.iter().product::<usize>();
        (Tensor::<TestBackend, 1, Int>::arange(0..count as i64, &device)
            .float()
            .mul_scalar(scale))
        .sin()
        .reshape(shape)
    };
    DeltaRuleInput {
        query: pattern(0.37, [1, sequence, heads, key_width]),
        key: pattern(0.53, [1, sequence, heads, key_width]),
        value: pattern(0.71, [1, sequence, heads, value_width]),
        log_decay: (Tensor::<TestBackend, 1, Int>::arange(0..(sequence * heads) as i64, &device)
            .float()
            .mul_scalar(0.29))
        .cos()
        .mul_scalar(0.1)
        .sub_scalar(0.15)
        .reshape([1, sequence, heads]),
        beta: Tensor::full([1, sequence, heads], 0.7, &device),
        initial_state: None,
        token_mask: None,
    }
}

/// Batched blocks reproduce the per-chunk algorithm across block and chunk boundaries.
#[test]
fn batched_prefill_matches_the_per_chunk_algorithm() {
    for (sequence, chunk_size) in [(1, 4), (7, 4), (64, 4), (70, 4), (33, 1), (100, 3)] {
        let config = DeltaRuleConfig {
            chunk_size,
            ..Default::default()
        };
        let (legacy_values, legacy_state) = legacy_chunked(long_fixture(sequence), config);
        let batched = chunked_gated_delta_rule(long_fixture(sequence), config).unwrap();
        batched
            .values
            .into_data()
            .assert_approx_eq::<f32>(&legacy_values.into_data(), Tolerance::rel_abs(1e-5, 1e-6));
        batched
            .final_state
            .into_data()
            .assert_approx_eq::<f32>(&legacy_state.into_data(), Tolerance::rel_abs(1e-5, 1e-6));
    }
    // 70 tokens with C = 4 spans two blocks of up to 16 chunks plus a padded tail.
    const { assert!(70 > 4 * CHUNKS_PER_BLOCK) };
}

/// Prompts shorter than one chunk run unpadded and still match the per-chunk algorithm.
///
/// A prompt of `L < chunk_size` tokens must use `C = L`: padding it to the
/// configured chunk would run a `C - 1`-step substitution on `C x C` tiles,
/// which made short prompts several times slower than the per-chunk
/// algorithm. The operand shapes pin that, and that chunk-major operands are
/// laid out `[chunks, batch, heads, C, width]`; the parity loop pins the
/// values, including a long prompt whose last chunk is partial.
#[test]
fn short_prompts_and_partial_tails_run_unpadded() {
    let config = DeltaRuleConfig {
        chunk_size: 64,
        ..Default::default()
    };
    let prepared = PreparedDeltaRule::new(long_fixture(10), config.qk_l2_epsilon).unwrap();
    let operands = ChunkOperands::new(&prepared, config.chunk_size);
    assert_eq!((operands.chunk, operands.chunks), (10, 1));
    let block = ChunkBlock::new(&operands, 0..1);
    assert_eq!(block.attention.dims(), [1, 1, 2, 10, 10]);
    assert_eq!(block.transformed_value.dims(), [1, 1, 2, 10, 3]);
    assert_eq!(block.chunk_decay.dims(), [1, 1, 2, 1, 1]);
    // Several chunks share one padded `C`.
    let operands = ChunkOperands::new(&prepared, 4);
    assert_eq!((operands.chunk, operands.chunks), (4, 3));
    let block = ChunkBlock::new(&operands, 0..3);
    assert_eq!(block.attention.dims(), [3, 1, 2, 4, 4]);

    // 1, 10, and 63 tokens fit in one chunk; 64 * 16 + 5 = 1029 tokens end
    // with a 5-token chunk, which balanced blocks keep beside full chunks.
    for sequence in [1, 10, 63, 64 * CHUNKS_PER_BLOCK + 5] {
        let (legacy_values, legacy_state) = legacy_chunked(long_fixture(sequence), config);
        let batched = chunked_gated_delta_rule(long_fixture(sequence), config).unwrap();
        batched
            .values
            .into_data()
            .assert_approx_eq::<f32>(&legacy_values.into_data(), Tolerance::rel_abs(1e-5, 1e-6));
        batched
            .final_state
            .into_data()
            .assert_approx_eq::<f32>(&legacy_state.into_data(), Tolerance::rel_abs(1e-5, 1e-6));
    }
}

/// Multi-block prefill with a padded final chunk equals the recurrence.
#[test]
fn multi_block_prefill_matches_recurrence() {
    for (sequence, chunk_size) in [(70, 4), (37, 2)] {
        let config = DeltaRuleConfig {
            chunk_size,
            ..Default::default()
        };
        let recurrent = recurrent_gated_delta_rule(long_fixture(sequence), config).unwrap();
        let chunked = chunked_gated_delta_rule(long_fixture(sequence), config).unwrap();
        chunked.values.into_data().assert_approx_eq::<f32>(
            &recurrent.values.into_data(),
            Tolerance::rel_abs(1e-4, 1e-5),
        );
        chunked.final_state.into_data().assert_approx_eq::<f32>(
            &recurrent.final_state.into_data(),
            Tolerance::rel_abs(1e-4, 1e-5),
        );
    }
}

/// Repeated keys with `beta = 1` and slow decay stay stable under substitution.
///
/// `L` is then nearly the all-ones strict lower matrix, whose Neumann powers
/// grow binomially; forward substitution must still match the recurrence.
#[test]
fn repeated_keys_with_unit_beta_match_recurrence() {
    let device = Default::default();
    let sequence = 24;
    let mut input = long_fixture(sequence);
    input.key = Tensor::ones([1, sequence, 2, 4], &device);
    input.beta = Tensor::ones([1, sequence, 2], &device);
    input.log_decay = Tensor::full([1, sequence, 2], -1e-3, &device);
    let config = DeltaRuleConfig {
        chunk_size: 8,
        ..Default::default()
    };
    let recurrent = recurrent_gated_delta_rule(input.clone(), config).unwrap();
    let chunked = chunked_gated_delta_rule(input, config).unwrap();
    chunked.values.into_data().assert_approx_eq::<f32>(
        &recurrent.values.into_data(),
        Tolerance::rel_abs(1e-4, 1e-4),
    );
}

/// The thresholded softplus matches PyTorch on both sides of its threshold.
#[test]
fn softplus_matches_pytorch_threshold_semantics() {
    let device = Default::default();
    let inputs = [
        -100.0_f32, -20.0, -1.0, 0.0, 1.0, 19.5, 20.0, 20.5, 88.0, 100.0, 1e30,
    ];
    let actual = softplus(Tensor::<TestBackend, 1>::from_floats(inputs, &device))
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    for (input, actual) in inputs.iter().zip(actual) {
        let expected = if *input > 20.0 {
            f64::from(*input)
        } else {
            f64::from(*input).exp().ln_1p()
        };
        assert!(actual.is_finite(), "softplus({input}) = {actual}");
        assert!(
            (f64::from(actual) - expected).abs() <= 1e-6 * expected.abs().max(1e-30),
            "softplus({input}) = {actual}, expected {expected}"
        );
    }
}

/// An extreme decay input keeps chunked and recurrent mixers finite and in agreement.
#[test]
fn huge_decay_input_keeps_both_paths_finite() {
    let device = Default::default();
    let config = GatedDeltaNetConfig::new(4, 1, 2, 2, 2);
    let mut mixer = config.init::<TestBackend>(&device).unwrap();
    // a + dt_bias = 100 overflowed Burn's naive softplus to +inf before.
    mixer.dt_bias = burn::module::Param::from_tensor(Tensor::full([2], 100.0, &device));
    mixer.in_proj_a.weight = burn::module::Param::from_tensor(Tensor::zeros([4, 2], &device));
    let hidden = (Tensor::<TestBackend, 1, Int>::arange(0..20, &device).float() * 0.1)
        .sin()
        .reshape([1, 5, 4]);
    let run = |mode| {
        let mut state = GatedDeltaNetState::zeros(&config, 1, &device).unwrap();
        let output = mixer
            .forward_with_state(hidden.clone(), None, &mut state, mode)
            .unwrap();
        (output.into_data(), state.recurrent.into_data())
    };
    let (chunked, chunked_state) = run(DeltaNetMode::ChunkedPrefill);
    let (recurrent, recurrent_state) = run(DeltaNetMode::Recurrent);
    for values in [&chunked, &recurrent, &chunked_state, &recurrent_state] {
        assert!(
            values
                .to_vec::<f32>()
                .unwrap()
                .iter()
                .all(|value| value.is_finite())
        );
    }
    chunked.assert_approx_eq::<f32>(&recurrent, Tolerance::rel_abs(1e-4, 1e-5));
    chunked_state.assert_approx_eq::<f32>(&recurrent_state, Tolerance::rel_abs(1e-4, 1e-5));
}

/// A width-one convolution has no history and runs in both modes.
#[test]
fn unit_convolution_kernel_runs_without_history() {
    let device = Default::default();
    TestBackend::seed(&device, 11);
    let mut config = GatedDeltaNetConfig::new(4, 1, 2, 2, 2);
    config.convolution_kernel = 1;
    let mixer = config.init::<TestBackend>(&device).unwrap();
    let hidden = (Tensor::<TestBackend, 1, Int>::arange(0..24, &device).float() * 0.2)
        .cos()
        .reshape([1, 6, 4]);
    let stateless = mixer.forward(hidden.clone(), None).unwrap();
    let mut state = GatedDeltaNetState::zeros(&config, 1, &device).unwrap();
    let mut decoded = Vec::new();
    for index in 0..6 {
        decoded.push(
            mixer
                .forward_with_state(
                    hidden.clone().narrow(1, index, 1),
                    None,
                    &mut state,
                    DeltaNetMode::Recurrent,
                )
                .unwrap(),
        );
    }
    // 2 * (1 key head * 2) + 2 value heads * 2 = 8 convolution channels.
    assert_eq!(state.convolution.dims(), [1, 8, 0]);
    stateless.into_data().assert_approx_eq::<f32>(
        &Tensor::cat(decoded, 1).into_data(),
        Tolerance::rel_abs(1e-4, 1e-5),
    );
}

/// Right, left (with carried state), and interior padding reproduce the compacted row.
#[test]
fn padded_rows_match_their_compacted_tokens() {
    let device = Default::default();
    TestBackend::seed(&device, 5);
    let mut config = GatedDeltaNetConfig::new(4, 1, 2, 2, 2);
    config.convolution_kernel = 3;
    config.prefill_chunk_size = 2;
    let mixer = config.init::<TestBackend>(&device).unwrap();
    let tokens = (Tensor::<TestBackend, 1, Int>::arange(0..16, &device).float() * 0.3)
        .sin()
        .reshape([1, 4, 4]);
    // A nonzero incoming state from an earlier segment.
    let mut carried = GatedDeltaNetState::zeros(&config, 1, &device).unwrap();
    mixer
        .forward_with_state(
            tokens.clone().flip([1]),
            None,
            &mut carried,
            DeltaNetMode::ChunkedPrefill,
        )
        .unwrap();
    let mut reference_state = carried.clone();
    let reference = mixer
        .forward_with_state(
            tokens.clone(),
            None,
            &mut reference_state,
            DeltaNetMode::ChunkedPrefill,
        )
        .unwrap();
    // Valid positions inside a 7-token padded row for each layout.
    for valid_positions in [[0, 1, 2, 3], [3, 4, 5, 6], [0, 2, 3, 6]] {
        let mut mask = [false; 7];
        let mut padded = Tensor::<TestBackend, 3>::zeros([1, 7, 4], &device);
        for (token, &position) in valid_positions.iter().enumerate() {
            mask[position] = true;
            padded = padded.slice_assign(
                [0..1, position..position + 1, 0..4],
                tokens.clone().narrow(1, token, 1),
            );
        }
        let mask = TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
            TensorData::new(mask.to_vec(), [1, 7]),
            &device,
        ));
        for mode in [DeltaNetMode::ChunkedPrefill, DeltaNetMode::Recurrent] {
            let mut state = carried.clone();
            let output = mixer
                .forward_with_state(padded.clone(), Some(&mask), &mut state, mode)
                .unwrap();
            for (token, &position) in valid_positions.iter().enumerate() {
                output
                    .clone()
                    .narrow(1, position, 1)
                    .into_data()
                    .assert_approx_eq::<f32>(
                        &reference.clone().narrow(1, token, 1).into_data(),
                        Tolerance::rel_abs(1e-4, 1e-5),
                    );
            }
            state.recurrent.into_data().assert_approx_eq::<f32>(
                &reference_state.recurrent.clone().into_data(),
                Tolerance::rel_abs(1e-4, 1e-5),
            );
            state.convolution.into_data().assert_approx_eq::<f32>(
                &reference_state.convolution.clone().into_data(),
                Tolerance::rel_abs(1e-4, 1e-5),
            );
        }
    }
}

/// Rows with different padding in one batch each match their own unpadded run.
///
/// Token compaction gathers, scatters, and offsets the convolution history
/// per row (`valid_counts[b]`), so a row-mixing or broadcast mistake would
/// only show when rows disagree. Row 0 is right-padded, row 1 interior-padded
/// with a leading gap, and row 2 fully masked; every row carries a different
/// non-zero state. Each row must reproduce a batch-of-one, unmasked run of its
/// valid tokens from its own carried state, and the fully masked row must
/// leave its state untouched.
#[test]
fn batch_rows_with_different_padding_match_isolated_rows() {
    let device = Default::default();
    TestBackend::seed(&device, 9);
    let mut config = GatedDeltaNetConfig::new(4, 1, 2, 2, 2);
    config.convolution_kernel = 3;
    config.prefill_chunk_size = 2;
    let mixer = config.init::<TestBackend>(&device).unwrap();
    let tokens = (Tensor::<TestBackend, 1, Int>::arange(0..48, &device).float() * 0.3)
        .sin()
        .reshape([3, 4, 4]);
    let mut carried = GatedDeltaNetState::zeros(&config, 3, &device).unwrap();
    mixer
        .forward_with_state(
            tokens.clone().flip([1]),
            None,
            &mut carried,
            DeltaNetMode::ChunkedPrefill,
        )
        .unwrap();
    let row_state = |row: usize| GatedDeltaNetState {
        convolution: carried.convolution.clone().narrow(0, row, 1),
        recurrent: carried.recurrent.clone().narrow(0, row, 1),
    };

    let valid_positions: [&[usize]; 3] = [&[0, 1, 2, 3], &[1, 3, 4], &[]];
    let mut mask = vec![false; 3 * 7];
    let mut padded = Tensor::<TestBackend, 3>::zeros([3, 7, 4], &device);
    for (row, positions) in valid_positions.iter().enumerate() {
        for (token, &position) in positions.iter().enumerate() {
            mask[row * 7 + position] = true;
            padded = padded.slice_assign(
                [row..row + 1, position..position + 1, 0..4],
                tokens.clone().slice([row..row + 1, token..token + 1, 0..4]),
            );
        }
    }
    let mask = TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
        TensorData::new(mask, [3, 7]),
        &device,
    ));

    let tolerance = Tolerance::rel_abs(1e-4, 1e-5);
    for mode in [DeltaNetMode::ChunkedPrefill, DeltaNetMode::Recurrent] {
        let mut state = carried.clone();
        let output = mixer
            .forward_with_state(padded.clone(), Some(&mask), &mut state, mode)
            .unwrap();
        for (row, positions) in valid_positions.iter().enumerate() {
            let mut expected_state = row_state(row);
            if !positions.is_empty() {
                let reference = mixer
                    .forward_with_state(
                        tokens
                            .clone()
                            .slice([row..row + 1, 0..positions.len(), 0..4]),
                        None,
                        &mut expected_state,
                        DeltaNetMode::ChunkedPrefill,
                    )
                    .unwrap();
                for (token, &position) in positions.iter().enumerate() {
                    output
                        .clone()
                        .slice([row..row + 1, position..position + 1, 0..4])
                        .into_data()
                        .assert_approx_eq::<f32>(
                            &reference.clone().narrow(1, token, 1).into_data(),
                            tolerance,
                        );
                }
            }
            state
                .recurrent
                .clone()
                .narrow(0, row, 1)
                .into_data()
                .assert_approx_eq::<f32>(&expected_state.recurrent.into_data(), tolerance);
            state
                .convolution
                .clone()
                .narrow(0, row, 1)
                .into_data()
                .assert_approx_eq::<f32>(&expected_state.convolution.into_data(), tolerance);
        }
    }
}

/// The query/key and output-norm epsilons are separate, validated, and serialized compatibly.
#[test]
fn query_key_and_output_norm_epsilons_are_independent() {
    let device = Default::default();
    let mut config = GatedDeltaNetConfig::new(4, 1, 2, 2, 2);
    config.normalization_epsilon = 1e-5;
    config.qk_l2_epsilon = 1e-3;
    let mixer = config.init::<TestBackend>(&device).unwrap();
    assert_eq!(mixer.norm.epsilon, 1e-5);
    assert_eq!(mixer.rule_config.qk_l2_epsilon, 1e-3);
    assert_eq!(mixer.config(), config);
    for (field, mutate) in [
        (
            "delta_net.normalization_epsilon",
            (|config: &mut GatedDeltaNetConfig| config.normalization_epsilon = 0.0)
                as fn(&mut GatedDeltaNetConfig),
        ),
        ("delta_net.qk_l2_epsilon", |config| {
            config.qk_l2_epsilon = f64::NAN
        }),
    ] {
        let mut invalid = GatedDeltaNetConfig::new(4, 1, 2, 2, 2);
        mutate(&mut invalid);
        assert!(matches!(invalid.validate(),
            Err(crate::Error::Configuration { field: name, .. }) if name == field));
    }
    let mut legacy = serde_json::to_value(GatedDeltaNetConfig::new(4, 1, 2, 2, 2)).unwrap();
    legacy.as_object_mut().unwrap().remove("qk_l2_epsilon");
    let decoded: GatedDeltaNetConfig = serde_json::from_value(legacy).unwrap();
    assert_eq!(decoded.qk_l2_epsilon, 1e-6);
}

/// Gradients reach every chunked-rule input and the mixer's decay parameters.
#[cfg(feature = "autodiff")]
#[test]
fn chunked_rule_and_mixer_are_differentiable() {
    use burn::backend::Autodiff;
    /// Reverse-mode wrapper of the Flex test backend that records the graph.
    type AdBackend = Autodiff<TestBackend>;
    let device = Default::default();
    let base = long_fixture(9);
    let lift = |tensor: Tensor<TestBackend, 4>| {
        Tensor::<AdBackend, 4>::from_data(tensor.into_data(), &device).require_grad()
    };
    let lift3 = |tensor: Tensor<TestBackend, 3>| {
        Tensor::<AdBackend, 3>::from_data(tensor.into_data(), &device).require_grad()
    };
    let (query, key, value) = (lift(base.query), lift(base.key), lift(base.value));
    let (log_decay, beta) = (lift3(base.log_decay), lift3(base.beta));
    let output = chunked_gated_delta_rule(
        DeltaRuleInput {
            query: query.clone(),
            key: key.clone(),
            value: value.clone(),
            log_decay: log_decay.clone(),
            beta: beta.clone(),
            initial_state: None,
            token_mask: None,
        },
        DeltaRuleConfig {
            chunk_size: 2,
            ..Default::default()
        },
    )
    .unwrap();
    let gradients = (output.values.sum() + output.final_state.sum()).backward();
    for gradient in [
        query.grad(&gradients).unwrap(),
        key.grad(&gradients).unwrap(),
        value.grad(&gradients).unwrap(),
    ] {
        assert!(
            gradient
                .into_data()
                .to_vec::<f32>()
                .unwrap()
                .iter()
                .all(|value| value.is_finite())
        );
    }
    for gradient in [
        log_decay.grad(&gradients).unwrap(),
        beta.grad(&gradients).unwrap(),
    ] {
        assert!(
            gradient
                .into_data()
                .to_vec::<f32>()
                .unwrap()
                .iter()
                .all(|value| value.is_finite())
        );
    }

    // The thresholded softplus keeps gradients finite even at a + dt_bias = 100.
    let config = GatedDeltaNetConfig::new(4, 1, 2, 2, 2);
    let mut mixer: super::GatedDeltaNet<AdBackend> = config.init(&device).unwrap();
    mixer.dt_bias = burn::module::Param::from_tensor(Tensor::full([2], 100.0, &device));
    let hidden = Tensor::<AdBackend, 3>::ones([1, 3, 4], &device);
    let gradients = mixer.forward(hidden, None).unwrap().sum().backward();
    for parameter in [&mixer.dt_bias, &mixer.a_log] {
        let gradient = parameter
            .grad(&gradients)
            .expect("decay parameter gradient");
        assert!(
            gradient
                .into_data()
                .to_vec::<f32>()
                .unwrap()
                .iter()
                .all(|value| value.is_finite())
        );
    }
}

/// Marks the re-executed test binary that runs the half-precision child.
const HALF_DEFAULT_CHILD: &str = "TRANSFORMERS_DELTA_HALF_DEFAULT_CHILD";

/// Masked delta rules keep explicit F32 indicators under an F16 device default.
///
/// Burn's default dtype is process-global per device, so the check runs in a
/// fresh child process of this test binary.
#[test]
fn masked_rule_survives_half_precision_default_dtype() {
    let output = std::process::Command::new(std::env::current_exe().expect("test binary"))
        .args([
            "--exact",
            "delta_net::tests::half_precision_default_child",
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

/// Child half of [`masked_rule_survives_half_precision_default_dtype`].
#[test]
#[ignore = "runs in a fresh process spawned by masked_rule_survives_half_precision_default_dtype"]
fn half_precision_default_child() {
    if std::env::var_os(HALF_DEFAULT_CHILD).is_none() {
        return;
    }
    let device = Default::default();
    burn::tensor::set_default_dtypes::<TestBackend>(
        &device,
        DType::F16,
        burn::tensor::IntDType::I32,
    )
    .expect("fresh process has no tensors yet");
    let mut input = fixture(3);
    input.token_mask = Some(Tensor::<TestBackend, 2, Bool>::from_bool(
        [[true, false, true], [true, true, false]].into(),
        &device,
    ));
    for output in [
        recurrent_gated_delta_rule(input.clone(), Default::default()).unwrap(),
        chunked_gated_delta_rule(input, Default::default()).unwrap(),
    ] {
        assert_eq!(output.final_state.dtype(), DType::F32);
        assert!(
            output
                .values
                .cast(DType::F32)
                .into_data()
                .to_vec::<f32>()
                .unwrap()
                .iter()
                .all(|value| value.is_finite())
        );
    }
}

/// Measures the chunked, per-chunk, and recurrent rules from short prompts to long prefills.
///
/// Covers prompts below one chunk, the toy two-head geometry, and a 16-head,
/// 128-wide geometry like Qwen3.5's linear-attention layers. Inputs are built
/// outside the timed region, the algorithms alternate to cancel drift, and
/// each reports the median of nine runs; `selected` names the algorithm that
/// [`prefers_recurrence`] picks for a prefill of that shape. Run with
/// `cargo test --release --lib delta_net::tests::prefill_timing -- --ignored --nocapture`.
#[test]
#[ignore = "timing measurement; run explicitly in release mode"]
fn prefill_timing() {
    let cases = [
        (8, 2, 4, 3),
        (32, 2, 4, 3),
        (1024, 2, 4, 3),
        (2048, 2, 4, 3),
        (1, 16, 128, 128),
        (8, 16, 128, 128),
        (32, 16, 128, 128),
        (256, 16, 128, 128),
        (1024, 16, 128, 128),
    ];
    let config = DeltaRuleConfig::default();
    let median = |mut samples: Vec<std::time::Duration>| {
        samples.sort();
        samples[samples.len() / 2]
    };
    for (sequence, heads, key_width, value_width) in cases {
        let input = smooth_fixture(sequence, heads, key_width, value_width);
        let time = |run: &dyn Fn() -> Tensor<TestBackend, 4>| {
            let start = std::time::Instant::now();
            let _ = run().into_data();
            start.elapsed()
        };
        let chunked_run = || {
            chunked_gated_delta_rule(input.clone(), config)
                .unwrap()
                .values
        };
        let legacy_run = || legacy_chunked(input.clone(), config).0;
        let recurrent_run = || {
            recurrent_gated_delta_rule(input.clone(), config)
                .unwrap()
                .values
        };
        let _ = (time(&chunked_run), time(&legacy_run), time(&recurrent_run));
        let (mut chunked, mut legacy, mut recurrent) = (Vec::new(), Vec::new(), Vec::new());
        for _ in 0..9 {
            chunked.push(time(&chunked_run));
            legacy.push(time(&legacy_run));
            recurrent.push(time(&recurrent_run));
        }
        let (chunked, legacy, recurrent) = (median(chunked), median(legacy), median(recurrent));
        let selected = if prefers_recurrence(sequence, heads * key_width * value_width) {
            "recurrent"
        } else {
            "chunked"
        };
        println!(
            "sequence={sequence} heads={heads} key={key_width} value={value_width} chunk={}: \
             chunked {chunked:?}, per-chunk {legacy:?} ({:.2}x), recurrent {recurrent:?}, \
             selected {selected}",
            config.chunk_size,
            legacy.as_secs_f64() / chunked.as_secs_f64()
        );
    }
}

/// Balanced blocks partition the chunks, respect the limit, and never strand a short tail.
#[test]
fn block_ranges_are_balanced_partitions() {
    assert!(block_chunk_ranges(0, 16).is_empty());
    assert_eq!(block_chunk_ranges(16, 16), vec![0..16]);
    // 17 chunks: two blocks of 9 and 8 instead of 16 and a lone 1.
    assert_eq!(block_chunk_ranges(17, 16), vec![0..9, 9..17]);
    assert_eq!(block_chunk_ranges(5, 0), block_chunk_ranges(5, 1));
}

proptest::proptest! {
    /// Every chunk lies in exactly one block, in order, and block sizes differ by at most one.
    #[test]
    fn block_ranges_partition_every_chunk(chunks in 0usize..600, limit in 1usize..40) {
        let ranges = block_chunk_ranges(chunks, limit);
        proptest::prop_assert_eq!(ranges.len(), chunks.div_ceil(limit));
        let mut next = 0;
        for range in &ranges {
            proptest::prop_assert_eq!(range.start, next);
            proptest::prop_assert!(!range.is_empty() && range.len() <= limit);
            next = range.end;
        }
        proptest::prop_assert_eq!(next, chunks);
        if let (Some(largest), Some(smallest)) = (
            ranges.iter().map(|range| range.len()).max(),
            ranges.iter().map(|range| range.len()).min(),
        ) {
            proptest::prop_assert!(largest - smallest <= 1);
        }
    }

    /// The recurrence is chosen monotonically: shorter segments and smaller states never flip it off.
    #[test]
    fn recurrence_choice_is_monotone(sequence in 1usize..64, state in 1usize..300_000) {
        if prefers_recurrence(sequence, state) {
            proptest::prop_assert!(sequence == 1 || prefers_recurrence(sequence - 1, state));
            proptest::prop_assert!(prefers_recurrence(sequence, state / 2));
        }
        proptest::prop_assert!(prefers_recurrence(1, state));
    }
}

/// Qwen3.5-sized states always prefill with the chunked rule; tiny states use the recurrence briefly.
#[test]
fn recurrence_is_preferred_only_for_short_tiny_segments() {
    let qwen_state = 16 * 128 * 128;
    assert!(
        prefers_recurrence(1, qwen_state),
        "one token is one exact step"
    );
    assert!((2..=4096).all(|sequence| !prefers_recurrence(sequence, qwen_state)));
    let toy_state = 2 * 4 * 3;
    assert!(prefers_recurrence(1, toy_state) && prefers_recurrence(11, toy_state));
    assert!(!prefers_recurrence(13, toy_state));
    assert!(
        !prefers_recurrence(usize::MAX, toy_state),
        "overflow means chunked"
    );
}

/// Every decay entry is the single rounded subtraction, bit-identical to the direct formula.
///
/// The input is the strided heads-major view of a token-minor
/// `[chunks, batch, C, heads, 1]` tensor, as in the chunked rule, with
/// cumulative log decays reaching about `-120` so that any rounding of `G`
/// before the subtraction (as a TF32 matrix product would do) is visible
/// after `exp`.
#[test]
fn causal_decay_matrix_is_exactly_the_masked_exponential() {
    let device = Default::default();
    let (chunks, batch, width, heads) = (2, 2, 7, 3);
    let token_minor =
        Tensor::<TestBackend, 1, Int>::arange(0..(chunks * batch * width * heads) as i64, &device)
            .float()
            .mul_scalar(0.731)
            .sin()
            .abs()
            .mul_scalar(-30.0)
            .reshape([chunks, batch, width, heads, 1])
            .cumsum(2);
    let cumulative = token_minor.swap_dims(2, 3);
    let direct = (cumulative.clone() - cumulative.clone().swap_dims(3, 4))
        .tril(0)
        .exp()
        .tril(0);
    let actual = causal_decay_matrix(cumulative.clone());
    assert_eq!(actual.dims(), [chunks, batch, heads, width, width]);
    assert_eq!(
        actual.into_data().to_vec::<f32>().unwrap(),
        direct.into_data().to_vec::<f32>().unwrap()
    );
    let deepest = cumulative.min().into_scalar();
    assert!(
        deepest < -100.0,
        "fixture must exercise large |G|: {deepest}"
    );
}

/// Blocked doubling inverts every width, including non-powers of two and width one.
#[test]
fn blocked_inverse_matches_substitution_for_every_width() {
    let device = Default::default();
    for width in [1, 2, 3, 5, 8, 10, 31, 64] {
        let lower =
            Tensor::<TestBackend, 1, Int>::arange(0..(2 * 3 * width * width) as i64, &device)
                .float()
                .mul_scalar(0.37)
                .cos()
                .mul_scalar(0.4)
                .reshape([2, 1, 3, width, width])
                .tril(-1);
        let inverse = unit_lower_inverse(lower.clone());
        // Reference: row-by-row forward substitution on host values.
        let values = lower.clone().into_data().to_vec::<f32>().unwrap();
        let mut expected = vec![0.0_f64; values.len()];
        for matrix in 0..6 {
            let at = |row: usize, column: usize| matrix * width * width + row * width + column;
            for row in 0..width {
                for column in 0..=row {
                    let mut entry = if row == column { 1.0 } else { 0.0 };
                    for inner in column..row {
                        entry -= f64::from(values[at(row, inner)]) * expected[at(inner, column)];
                    }
                    expected[at(row, column)] = entry;
                }
            }
        }
        let expected = expected
            .into_iter()
            .map(|value| value as f32)
            .collect::<Vec<_>>();
        inverse.into_data().assert_approx_eq::<f32>(
            &TensorData::new(expected, [2, 1, 3, width, width]),
            Tolerance::rel_abs(1e-5, 1e-6),
        );
    }
}

/// Per-token states replay the recurrence exactly and end at its final state.
#[test]
fn recurrent_snapshots_record_every_post_token_state() {
    let config = DeltaRuleConfig::default();
    let mut input = fixture(5);
    input.token_mask = Some(Tensor::<TestBackend, 2, Bool>::from_bool(
        [
            [true, true, false, true, true],
            [false, true, true, true, false],
        ]
        .into(),
        &Default::default(),
    ));
    let reference = recurrent_gated_delta_rule(input.clone(), config).unwrap();
    let snapshots = recurrent_gated_delta_rule_with_snapshots(input.clone(), config).unwrap();
    assert_eq!(
        snapshots
            .values
            .clone()
            .into_data()
            .to_vec::<f32>()
            .unwrap(),
        reference.values.into_data().to_vec::<f32>().unwrap()
    );
    let [batch, tokens, heads, key_width, value_width] = snapshots.states.dims();
    assert_eq!(tokens, 5);
    let last = snapshots.states.clone().narrow(1, tokens - 1, 1).reshape([
        batch,
        heads,
        key_width,
        value_width,
    ]);
    assert_eq!(
        last.into_data().to_vec::<f32>().unwrap(),
        reference.final_state.into_data().to_vec::<f32>().unwrap()
    );
    // State after token t equals a fresh recurrence over the first t + 1 tokens.
    for prefix in 1..=tokens {
        let mut truncated = input.clone();
        truncated.query = truncated.query.narrow(1, 0, prefix);
        truncated.key = truncated.key.narrow(1, 0, prefix);
        truncated.value = truncated.value.narrow(1, 0, prefix);
        truncated.log_decay = truncated.log_decay.narrow(1, 0, prefix);
        truncated.beta = truncated.beta.narrow(1, 0, prefix);
        truncated.token_mask = truncated.token_mask.map(|mask| mask.narrow(1, 0, prefix));
        let expected = recurrent_gated_delta_rule(truncated, config)
            .unwrap()
            .final_state;
        let actual = snapshots.states.clone().narrow(1, prefix - 1, 1).reshape([
            batch,
            heads,
            key_width,
            value_width,
        ]);
        assert_eq!(
            actual.into_data().to_vec::<f32>().unwrap(),
            expected.into_data().to_vec::<f32>().unwrap()
        );
    }
}

/// Several batch rows share one chunk-major layout and still match row-by-row recurrences.
#[test]
fn batched_rows_match_their_recurrences() {
    let config = DeltaRuleConfig {
        chunk_size: 4,
        ..Default::default()
    };
    let rows = [smooth_fixture(19, 2, 4, 3), {
        let mut input = smooth_fixture(19, 2, 4, 3);
        input.query = input.query.flip([1]);
        input.value = input.value.mul_scalar(-0.5);
        input
    }];
    let stack = |pick: &dyn Fn(&DeltaRuleInput<TestBackend>) -> Tensor<TestBackend, 4>| {
        Tensor::cat(rows.iter().map(pick).collect(), 0)
    };
    let stack3 = |pick: &dyn Fn(&DeltaRuleInput<TestBackend>) -> Tensor<TestBackend, 3>| {
        Tensor::cat(rows.iter().map(pick).collect(), 0)
    };
    let batched = DeltaRuleInput {
        query: stack(&|input| input.query.clone()),
        key: stack(&|input| input.key.clone()),
        value: stack(&|input| input.value.clone()),
        log_decay: stack3(&|input| input.log_decay.clone()),
        beta: stack3(&|input| input.beta.clone()),
        initial_state: None,
        token_mask: None,
    };
    let chunked = chunked_gated_delta_rule(batched, config).unwrap();
    for (row, input) in rows.into_iter().enumerate() {
        let recurrent = recurrent_gated_delta_rule(input, config).unwrap();
        chunked
            .values
            .clone()
            .narrow(0, row, 1)
            .into_data()
            .assert_approx_eq::<f32>(
                &recurrent.values.into_data(),
                Tolerance::rel_abs(1e-4, 1e-5),
            );
        chunked
            .final_state
            .clone()
            .narrow(0, row, 1)
            .into_data()
            .assert_approx_eq::<f32>(
                &recurrent.final_state.into_data(),
                Tolerance::rel_abs(1e-4, 1e-5),
            );
    }
}

/// Builds `[batch, sequence, width]` smooth hidden states for mixer tests.
fn smooth_hidden(batch: usize, sequence: usize, width: usize) -> Tensor<TestBackend, 3> {
    Tensor::<TestBackend, 1, Int>::arange(0..(batch * sequence * width) as i64, &Default::default())
        .float()
        .mul_scalar(0.613)
        .sin()
        .reshape([batch, sequence, width])
}

/// A long projected prefill takes the chunked rule and still equals token-by-token decode.
#[test]
fn projected_mixer_chunked_prefill_matches_decode() {
    let device = Default::default();
    TestBackend::seed(&device, 11);
    let mut config = GatedDeltaNetConfig::new(4, 1, 2, 2, 2);
    config.convolution_kernel = 3;
    config.prefill_chunk_size = 4;
    let mixer = config.init::<TestBackend>(&device).unwrap();
    let sequence = 23;
    assert!(
        !prefers_recurrence(sequence, 2 * 2 * 2),
        "must exercise the chunked rule"
    );
    let hidden = smooth_hidden(2, sequence, 4);
    let mut prefill_state = GatedDeltaNetState::zeros(&config, 2, &device).unwrap();
    let prefill = mixer
        .forward_with_state(
            hidden.clone(),
            None,
            &mut prefill_state,
            DeltaNetMode::ChunkedPrefill,
        )
        .unwrap();
    let mut decode_state = GatedDeltaNetState::zeros(&config, 2, &device).unwrap();
    let decode = Tensor::cat(
        (0..sequence)
            .map(|index| {
                mixer
                    .forward_with_state(
                        hidden.clone().narrow(1, index, 1),
                        None,
                        &mut decode_state,
                        DeltaNetMode::Recurrent,
                    )
                    .unwrap()
            })
            .collect(),
        1,
    );
    let tolerance = Tolerance::rel_abs(2e-4, 2e-5);
    prefill
        .into_data()
        .assert_approx_eq::<f32>(&decode.into_data(), tolerance);
    prefill_state
        .recurrent
        .into_data()
        .assert_approx_eq::<f32>(&decode_state.recurrent.into_data(), tolerance);
    prefill_state
        .convolution
        .into_data()
        .assert_approx_eq::<f32>(&decode_state.convolution.into_data(), tolerance);
}

/// Mixer snapshots equal the state after feeding each column alone, padding included.
#[test]
fn mixer_snapshots_match_column_by_column_states() {
    let device = Default::default();
    TestBackend::seed(&device, 13);
    for kernel in [1, 3] {
        let mut config = GatedDeltaNetConfig::new(4, 1, 2, 2, 2);
        config.convolution_kernel = kernel;
        let mixer = config.init::<TestBackend>(&device).unwrap();
        let hidden = smooth_hidden(2, 5, 4);
        let mask = TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
            [
                [true, false, true, true, false],
                [false, true, true, false, true],
            ]
            .into(),
            &device,
        ));
        let mut start = GatedDeltaNetState::zeros(&config, 2, &device).unwrap();
        // A non-zero incoming state makes the convolution history observable.
        start.convolution = start.convolution.clone() + 0.25;
        start.recurrent = start.recurrent.clone() + 0.125;

        let mut snapshot_state = start.clone();
        let (output, snapshots) = mixer
            .forward_with_snapshots(hidden.clone(), Some(&mask), &mut snapshot_state)
            .unwrap();
        let mut recurrent_state = start.clone();
        let recurrent = mixer
            .forward_with_state(
                hidden.clone(),
                Some(&mask),
                &mut recurrent_state,
                DeltaNetMode::Recurrent,
            )
            .unwrap();
        assert_eq!(
            output.into_data().to_vec::<f32>().unwrap(),
            recurrent.into_data().to_vec::<f32>().unwrap()
        );

        let mut column_state = start.clone();
        for column in 0..5 {
            let column_mask = TokenMask::new(mask.as_tensor().clone().narrow(1, column, 1));
            mixer
                .forward_with_state(
                    hidden.clone().narrow(1, column, 1),
                    Some(&column_mask),
                    &mut column_state,
                    DeltaNetMode::Recurrent,
                )
                .unwrap();
            let [batch, _, heads, key_dim, value_dim] = snapshots.recurrent.dims();
            snapshots
                .recurrent
                .clone()
                .narrow(1, column, 1)
                .reshape([batch, heads, key_dim, value_dim])
                .into_data()
                .assert_approx_eq::<f32>(
                    &column_state.recurrent.clone().into_data(),
                    Tolerance::rel_abs(1e-5, 1e-6),
                );
            let [_, _, channels, width] = snapshots.convolution.dims();
            assert_eq!(width, kernel - 1);
            if width == 0 {
                // Burn reads a zero in a reshape target as "keep this axis".
                continue;
            }
            assert_eq!(
                snapshots
                    .convolution
                    .clone()
                    .narrow(1, column, 1)
                    .reshape([batch, channels, width])
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap(),
                column_state
                    .convolution
                    .clone()
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap()
            );
        }
        assert_eq!(
            snapshot_state
                .recurrent
                .into_data()
                .to_vec::<f32>()
                .unwrap(),
            recurrent_state
                .recurrent
                .into_data()
                .to_vec::<f32>()
                .unwrap()
        );
        assert_eq!(
            snapshot_state
                .convolution
                .into_data()
                .to_vec::<f32>()
                .unwrap(),
            recurrent_state
                .convolution
                .into_data()
                .to_vec::<f32>()
                .unwrap()
        );
    }
}

/// Every non-finite or non-positive query/key epsilon is rejected under the rule's own field name.
///
/// The error names `delta_net.rule.qk_l2_epsilon`, never the mixer's
/// `delta_net.normalization_epsilon`, which is the output-norm stabilizer.
#[test]
fn rule_config_rejects_each_invalid_epsilon() {
    for qk_l2_epsilon in [f64::NAN, f64::INFINITY, 0.0, -1e-6] {
        let error = DeltaRuleConfig {
            qk_l2_epsilon,
            ..Default::default()
        }
        .validate()
        .unwrap_err();
        assert!(
            matches!(&error, crate::Error::Configuration { field, .. }
                if field == "delta_net.rule.qk_l2_epsilon"),
            "{qk_l2_epsilon}: {error:?}"
        );
    }
}

/// The rule's query/key epsilon serializes under its new name and still reads the old key.
///
/// `DeltaRuleConfig::normalization_epsilon` was renamed to `qk_l2_epsilon`
/// because the mixer config uses `normalization_epsilon` for the output RMS
/// norm. JSON written before the rename must load into the renamed field, the
/// new spelling must round-trip, and the deprecated accessor must keep
/// returning the same value for source compatibility.
#[test]
fn rule_config_reads_the_legacy_epsilon_key_and_writes_the_new_one() {
    let legacy: DeltaRuleConfig =
        serde_json::from_str(r#"{"normalization_epsilon": 0.001, "chunk_size": 32}"#).unwrap();
    assert_eq!(
        legacy,
        DeltaRuleConfig {
            qk_l2_epsilon: 1e-3,
            chunk_size: 32,
        }
    );
    let written = serde_json::to_value(legacy).unwrap();
    assert_eq!(written["qk_l2_epsilon"], 1e-3);
    assert!(written.get("normalization_epsilon").is_none());
    assert_eq!(
        serde_json::from_value::<DeltaRuleConfig>(written).unwrap(),
        legacy
    );
    #[allow(deprecated)]
    let renamed = legacy.normalization_epsilon();
    assert_eq!(renamed, legacy.qk_l2_epsilon);
    let config = GatedDeltaNetConfig::new(4, 1, 2, 2, 2);
    assert_eq!(config.rule_config().qk_l2_epsilon, config.qk_l2_epsilon);
}

/// Without a token mask, snapshots and state equal those of an all-valid mask.
///
/// The unmasked path counts valid columns as `1..=sequence` instead of a
/// cumulative sum over the mask, and equal key and value head counts skip
/// the key-head repetition; both must reproduce the masked computation.
#[test]
fn unmasked_snapshots_equal_all_valid_masked_snapshots() {
    let device = Default::default();
    TestBackend::seed(&device, 21);
    let mut config = GatedDeltaNetConfig::new(4, 2, 2, 2, 2);
    config.convolution_kernel = 3;
    let mixer = config.init::<TestBackend>(&device).unwrap();
    let hidden = smooth_hidden(2, 4, 4);
    let all_valid = TokenMask::new(Tensor::<TestBackend, 2, Int>::ones([2, 4], &device).bool());
    let mut start = GatedDeltaNetState::zeros(&config, 2, &device).unwrap();
    start.convolution = start.convolution.clone() + 0.25;
    start.recurrent = start.recurrent.clone() - 0.125;

    let mut unmasked_state = start.clone();
    let (unmasked, unmasked_snapshots) = mixer
        .forward_with_snapshots(hidden.clone(), None, &mut unmasked_state)
        .unwrap();
    let mut masked_state = start;
    let (masked, masked_snapshots) = mixer
        .forward_with_snapshots(hidden, Some(&all_valid), &mut masked_state)
        .unwrap();
    let tolerance = Tolerance::absolute(1e-6);
    unmasked
        .into_data()
        .assert_approx_eq::<f32>(&masked.into_data(), tolerance);
    unmasked_snapshots
        .recurrent
        .into_data()
        .assert_approx_eq::<f32>(&masked_snapshots.recurrent.into_data(), tolerance);
    unmasked_snapshots
        .convolution
        .into_data()
        .assert_approx_eq::<f32>(&masked_snapshots.convolution.into_data(), tolerance);
    unmasked_state
        .recurrent
        .into_data()
        .assert_approx_eq::<f32>(&masked_state.recurrent.into_data(), tolerance);
}

/// One stateful mixer entry point together with the delta-rule kernel it selects.
#[derive(Clone, Copy, Debug)]
enum StatefulCall {
    /// `GatedDeltaNet::forward_with_state` with the given mode and segment length.
    State(
        /// Requested execution mode.
        DeltaNetMode,
        /// Tokens in the single-row segment, which decides the prefill kernel.
        usize,
    ),
    /// `GatedDeltaNet::forward_with_snapshots` over a segment of this many tokens.
    Snapshots(
        /// Tokens in the single-row segment.
        usize,
    ),
}

impl StatefulCall {
    /// Runs this entry point on a smooth single-row input and returns only its error.
    fn error(
        self,
        mixer: &GatedDeltaNet<TestBackend>,
        state: &mut GatedDeltaNetState<TestBackend>,
    ) -> crate::Error {
        match self {
            Self::State(mode, tokens) => mixer
                .forward_with_state(smooth_hidden(1, tokens, 4), None, state, mode)
                .unwrap_err(),
            Self::Snapshots(tokens) => mixer
                .forward_with_snapshots(smooth_hidden(1, tokens, 4), None, state)
                .unwrap_err(),
        }
    }
}

/// Every stateful kernel path of the `[4, 1, 2, 2, 2]` test mixer.
///
/// With `2 * 2 * 2 = 8` state elements, chunked prefill keeps the recurrence
/// for 3 tokens and switches to the chunked rule at 16; the snapshot kernel
/// always runs the recurrence. The test asserts that kernel choice so a
/// later cost-model change cannot silently drop the chunked case.
fn stateful_calls() -> [StatefulCall; 4] {
    assert!(prefers_recurrence(3, 8));
    assert!(!prefers_recurrence(16, 8));
    [
        StatefulCall::State(DeltaNetMode::Recurrent, 3),
        StatefulCall::State(DeltaNetMode::ChunkedPrefill, 3),
        StatefulCall::State(DeltaNetMode::ChunkedPrefill, 16),
        StatefulCall::Snapshots(3),
    ]
}

/// Asserts that `call` fails as `expected` describes and leaves `state` as supplied.
fn assert_failure_preserves_state(
    mixer: &GatedDeltaNet<TestBackend>,
    config: &GatedDeltaNetConfig,
    call: StatefulCall,
    expected: impl Fn(&crate::Error) -> bool,
) {
    let device = Default::default();
    let mut state = GatedDeltaNetState::zeros(config, 1, &device).unwrap();
    state.convolution = state.convolution.clone() + 0.25;
    state.recurrent = state.recurrent.clone() - 0.5;
    let convolution_before = state.convolution.clone().into_data();
    let recurrent_before = state.recurrent.clone().into_data();
    let error = call.error(mixer, &mut state);
    assert!(expected(&error), "{call:?}: {error:?}");
    state
        .convolution
        .into_data()
        .assert_eq(&convolution_before, true);
    state
        .recurrent
        .into_data()
        .assert_eq(&recurrent_before, true);
}

/// A failing output projection leaves both caller state buffers untouched.
///
/// The output projection is the last fallible step of the mixer. A manually
/// assembled module whose `out_proj` does not accept the value width must
/// report the projection error without having published the advanced
/// convolution history or recurrent state, so a caller can retry or roll
/// back from exactly the state it supplied. The guarantee is checked on the
/// recurrent kernel, on chunked prefill at a length that keeps the
/// recurrence and at one that selects the chunked rule, and on the snapshot
/// kernel that speculative rollback depends on.
#[test]
fn failing_output_projection_preserves_caller_state() {
    use burn::{module::Param, nn::LinearConfig};

    let device = Default::default();
    let config = GatedDeltaNetConfig::new(4, 1, 2, 2, 2);
    let mut mixer = config.init::<TestBackend>(&device).unwrap();
    mixer.out_proj = LinearConfig::new(3, 4).init(&device);
    mixer.out_proj.weight = Param::from_tensor(Tensor::ones([3, 4], &device));
    for call in stateful_calls() {
        assert_failure_preserves_state(
            &mixer,
            &config,
            call,
            |error| matches!(error, crate::Error::Shape { tensor, .. } if tensor == "linear.weight"),
        );
    }
}

/// An invalid manually set `rule_config` fails in every kernel without touching state.
///
/// The delta-rule kernels validate their controls before any tensor work, so
/// a zero chunk size or a non-positive query/key epsilon is reported as a
/// configuration error on every path, the recurrent ones included, and the
/// caller's buffers stay as supplied.
#[test]
fn invalid_rule_config_fails_every_kernel_and_preserves_state() {
    let device = Default::default();
    let config = GatedDeltaNetConfig::new(4, 1, 2, 2, 2);
    // The kernels name the `DeltaRuleConfig` field; the stateless forward
    // validates the rebuilt `GatedDeltaNetConfig`, which names its own field.
    for (rule_config, kernel_field, kernel_reason, config_field) in [
        (
            DeltaRuleConfig {
                chunk_size: 0,
                ..DeltaRuleConfig::default()
            },
            "delta_net.chunk_size",
            "must be non-zero",
            "delta_net.chunk_size",
        ),
        (
            DeltaRuleConfig {
                qk_l2_epsilon: 0.0,
                ..DeltaRuleConfig::default()
            },
            "delta_net.rule.qk_l2_epsilon",
            "must be finite and greater than zero",
            "delta_net.qk_l2_epsilon",
        ),
    ] {
        let mut mixer = config.init::<TestBackend>(&device).unwrap();
        mixer.rule_config = rule_config;
        for call in stateful_calls() {
            assert_failure_preserves_state(&mixer, &config, call, |error| {
                matches!(error, crate::Error::Configuration { field, reason }
                    if field == kernel_field && reason == kernel_reason)
            });
        }
        // The stateless forward rejects the rebuilt configuration while
        // allocating its zero state.
        let error = mixer.forward(smooth_hidden(1, 3, 4), None).unwrap_err();
        assert!(
            matches!(&error, crate::Error::Configuration { field, .. } if field == config_field),
            "{error:?}"
        );
    }
}

/// The stateless forward reports an empty batch while allocating its zero state.
#[test]
fn stateless_forward_rejects_an_empty_batch() {
    let device = Default::default();
    let mixer = GatedDeltaNetConfig::new(4, 1, 2, 2, 2)
        .init::<TestBackend>(&device)
        .unwrap();
    let error = mixer
        .forward(Tensor::zeros([0, 2, 4], &device), None)
        .unwrap_err();
    assert!(
        matches!(&error, crate::Error::Configuration { field, .. } if field == "delta_net.batch_size"),
        "{error:?}"
    );
}

/// A strided per-head tensor gains its trailing unit axis without moving any value.
///
/// The input is a transposed view, so its strides are not row-major; every
/// token's `[batch, heads, 1, 1]` slice, as the recurrence takes it, must
/// still hold that token's per-head values of every row.
#[test]
fn trailing_unit_axis_keeps_every_row_of_a_strided_tensor() {
    let device = Default::default();
    let [batch, sequence] = [3, 5];
    for heads in [1, 2] {
        let values = (0..batch * heads * sequence)
            .map(|index| index as f32)
            .collect::<Vec<_>>();
        // `[batch, heads, sequence]` storage viewed as `[batch, sequence, heads]`.
        let strided = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(values, [batch, heads, sequence]),
            &device,
        )
        .swap_dims(1, 2);
        let expected = strided.clone().into_data().to_vec::<f32>().unwrap();
        let lifted = trailing_unit_axis(strided);
        assert_eq!(lifted.dims(), [batch, sequence, heads, 1]);
        assert_eq!(
            lifted.clone().into_data().to_vec::<f32>().unwrap(),
            expected
        );
        for token in 0..sequence {
            let factors = lifted.clone().narrow(1, token, 1).swap_dims(1, 2);
            assert_eq!(factors.dims(), [batch, heads, 1, 1]);
            let actual = factors.into_data().to_vec::<f32>().unwrap();
            let wanted = (0..batch)
                .flat_map(|row| (0..heads).map(move |head| (row, head)))
                .map(|(row, head)| ((row * heads + head) * sequence + token) as f32)
                .collect::<Vec<_>>();
            assert_eq!(actual, wanted, "{heads} heads, token {token}");
        }
    }
}
