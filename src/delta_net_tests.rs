//! Delta-rule tests comparing recurrent and chunked updates, state continuation, and validation.

use burn::{
    backend::Flex,
    tensor::{Bool, DType, Int, Tensor, TensorData, Tolerance, backend::Backend},
};

use crate::architecture::TokenMask;

use super::{
    DeltaNetMode, DeltaRuleConfig, DeltaRuleInput, GatedDeltaNetConfig, GatedDeltaNetState,
    chunked_gated_delta_rule, mask_hidden, recurrent_gated_delta_rule,
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
            normalization_epsilon: f64::NAN,
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
    let epsilon = DeltaRuleConfig::default().normalization_epsilon;
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
