//! Mamba-2 recurrence, chunked-scan, masking, and reduced-precision oracle tests.

use burn::{
    backend::Flex,
    module::Param,
    tensor::{Tolerance, bf16, f16},
};

use super::*;

/// CPU Flex backend for full-precision fixtures and explicit reduced-precision casts.
type TestBackend = Flex<f32, i32>;

/// Small grouped topology that exercises two chunks and head repetition.
fn config() -> Mamba2Config {
    Mamba2Config {
        hidden_size: 8,
        num_heads: 4,
        head_dim: 2,
        state_size: 3,
        groups: 2,
        convolution_kernel: 3,
        chunk_size: 2,
        time_step_clamp: TimeStepClamp::chunked_only(TimeStepLimits::at_least(0.001)),
        normalization_epsilon: 1e-5,
        projection_bias: false,
        convolution_bias: true,
    }
}

/// Chunk algebra and the defining token recurrence agree from non-zero state.
#[test]
fn chunked_and_recurrent_paths_match() {
    let device = Default::default();
    let mixer = config().init::<TestBackend>(&device).unwrap();
    let input = Tensor::from_floats(
        [[
            [0.1, 0.2, -0.3, 0.4, 0.5, -0.6, 0.7, 0.8],
            [0.3, -0.1, 0.2, 0.6, -0.4, 0.5, 0.9, -0.2],
            [-0.5, 0.7, 0.1, -0.8, 0.2, 0.4, -0.6, 0.3],
            [0.8, 0.1, -0.2, 0.5, 0.6, -0.7, 0.4, -0.3],
            [0.2, -0.4, 0.6, 0.1, -0.5, 0.3, 0.7, 0.9],
        ]],
        &device,
    );
    let mut state = Mamba2State::zeros(&config(), 1, DType::F32, &device).unwrap();
    state.recurrent = Tensor::ones([1, 4, 2, 3], &device) * 0.07;
    state.convolution = Tensor::ones([1, 20, 2], &device) * -0.03;
    let (reference, reference_state) = mixer
        .forward_with_state(input.clone(), None, &state, Mamba2Mode::Recurrent)
        .unwrap();
    let (chunked, chunked_state) = mixer
        .forward_with_state(input, None, &state, Mamba2Mode::ChunkedPrefill)
        .unwrap();
    chunked
        .into_data()
        .assert_approx_eq::<f32>(&reference.into_data(), Tolerance::absolute(2e-5));
    chunked_state.recurrent.into_data().assert_approx_eq::<f32>(
        &reference_state.recurrent.into_data(),
        Tolerance::absolute(2e-5),
    );
    chunked_state
        .convolution
        .into_data()
        .assert_approx_eq::<f32>(
            &reference_state.convolution.into_data(),
            Tolerance::absolute(1e-6),
        );
}

/// Segmenting prompt prefill preserves convolution and recurrent cache state.
///
/// The split is intentionally not aligned to the configured scan chunk width,
/// so this covers both the `kernel - 1` convolution history convention and
/// the algebra that carries an FP32 terminal matrix between arbitrary calls.
#[test]
fn segmented_prefill_matches_one_shot_cache_and_outputs() {
    let device = Default::default();
    let mixer = config().init::<TestBackend>(&device).unwrap();
    let input = Tensor::from_floats(
        [[
            [0.1, 0.2, -0.3, 0.4, 0.5, -0.6, 0.7, 0.8],
            [0.3, -0.1, 0.2, 0.6, -0.4, 0.5, 0.9, -0.2],
            [-0.5, 0.7, 0.1, -0.8, 0.2, 0.4, -0.6, 0.3],
            [0.8, 0.1, -0.2, 0.5, 0.6, -0.7, 0.4, -0.3],
            [0.2, -0.4, 0.6, 0.1, -0.5, 0.3, 0.7, 0.9],
        ]],
        &device,
    );
    let mut initial = Mamba2State::zeros(&config(), 1, DType::F32, &device).unwrap();
    initial.recurrent = Tensor::ones([1, 4, 2, 3], &device) * 0.037;
    initial.convolution = Tensor::ones([1, 20, 2], &device) * -0.019;

    let (one_shot, one_shot_state) = mixer
        .forward_with_state(input.clone(), None, &initial, Mamba2Mode::ChunkedPrefill)
        .unwrap();
    let (prefix, prefix_state) = mixer
        .forward_with_state(
            input.clone().narrow(1, 0, 3),
            None,
            &initial,
            Mamba2Mode::ChunkedPrefill,
        )
        .unwrap();
    let (suffix, segmented_state) = mixer
        .forward_with_state(
            input.narrow(1, 3, 2),
            None,
            &prefix_state,
            Mamba2Mode::ChunkedPrefill,
        )
        .unwrap();
    let segmented = Tensor::cat(vec![prefix, suffix], 1);

    segmented
        .into_data()
        .assert_approx_eq::<f32>(&one_shot.into_data(), Tolerance::absolute(3e-5));
    segmented_state
        .recurrent
        .into_data()
        .assert_approx_eq::<f32>(
            &one_shot_state.recurrent.into_data(),
            Tolerance::absolute(3e-5),
        );
    assert_eq!(
        segmented_state.convolution.to_data(),
        one_shot_state.convolution.to_data()
    );
}

/// A one-head scalar recurrence matches a hand-computed selective-SSM oracle.
#[test]
fn recurrent_scan_matches_scalar_golden_values() {
    let device = Default::default();
    let x = Tensor::<TestBackend, 4>::from_floats([[[[3.0]], [[-2.0]]]], &device);
    let b = Tensor::<TestBackend, 4>::from_floats([[[[4.0]], [[0.5]]]], &device);
    let c = Tensor::<TestBackend, 4>::from_floats([[[[5.0]], [[3.0]]]], &device);
    let dt = Tensor::<TestBackend, 3>::from_floats([[[0.1], [0.2]]], &device);
    let a = Tensor::<TestBackend, 1>::from_floats([-1.0], &device);
    let d = Tensor::<TestBackend, 1>::from_floats([0.5], &device);
    let initial = Tensor::<TestBackend, 4>::from_floats([[[[2.0]]]], &device);
    let (output, terminal) = recurrent_scan(x, b, c, dt, a, d, initial).unwrap();
    output.into_data().assert_approx_eq::<f32>(
        &TensorData::from([[[[16.548_374]], [[5.792_34]]]]),
        Tolerance::absolute(1e-5),
    );
    terminal.into_data().assert_approx_eq::<f32>(
        &TensorData::from([[[[2.264_113_4]]]]),
        Tolerance::absolute(1e-6),
    );
}

/// Segmented decay retains local increments after an enormous earlier prefix.
#[test]
fn chunk_transition_sum_avoids_prefix_cancellation() {
    let device = Default::default();
    let transitions = stable_decay_transitions(Tensor::<TestBackend, 3>::from_floats(
        [[[-100_000_000.0], [-1.0], [-2.0]]],
        &device,
    ));
    transitions.into_data().assert_approx_eq::<f32>(
        &TensorData::from([[
            [[1.0], [0.0], [0.0]],
            [[(-1.0_f32).exp()], [1.0], [0.0]],
            [[(-3.0_f32).exp()], [(-2.0_f32).exp()], [1.0]],
        ]]),
        Tolerance::absolute(1e-6),
    );
}

/// Grouped gated normalization follows gate-then-normalize ordering.
#[test]
fn grouped_gate_is_applied_before_rms_normalization() {
    let device = Default::default();
    let norm = GatedGroupRmsNorm::<TestBackend> {
        weight: Param::from_tensor(Tensor::ones([4], &device)),
        groups: 2,
        epsilon: 1e-6,
    };
    let values = Tensor::from_floats([[[1.0, 2.0, 3.0, 4.0]]], &device);
    let gate = Tensor::ones([1, 1, 4], &device);
    let output = norm.forward(values, gate).unwrap();
    let data = output.into_data();
    let values = data.as_slice::<f32>().unwrap();
    assert!(values.iter().all(|value| value.is_finite()));
    assert!((values[0] / values[1] - 0.5).abs() < 1e-5);
    assert!((values[2] / values[3] - 0.75).abs() < 1e-5);
}

/// Grouped normalization retains Zamba2's multiply-by-inverse-root graph.
///
/// These literals were generated with `Zamba2RMSNormGated.forward` from
/// Transformers revision `cfa309cfa34b5497a32fd50af49aa97aac94c054`.
/// A saturated gate isolates RMS application from backend-specific fused
/// SiLU approximations. The first component is a one-ULP discriminator:
/// replacing multiplication by reciprocal-sqrt with division produces
/// `0xbfaa`, not `0xbfa9`.
#[test]
fn grouped_norm_matches_pinned_reduced_precision_oracles() {
    let device = Default::default();
    let norm = GatedGroupRmsNorm::<TestBackend> {
        weight: Param::from_tensor(
            Tensor::from_floats(
                [-1.934_570_3, 0.954_101_56, -0.644_531_25, -2.945_312_5],
                &device,
            )
            .cast(DType::F16),
        ),
        groups: 2,
        epsilon: 1e-6,
    };
    let values = Tensor::from_floats(
        [[[6.406_25, -6.535_156_2, 7.539_062_5, -5.781_25]]],
        &device,
    )
    .cast(DType::F16);
    let gate = Tensor::full([1, 1, 4], 80.0, (&device, DType::F16));

    let actual = norm.forward(values.clone(), gate.clone()).unwrap();
    let actual_bits = actual
        .into_data()
        .to_vec::<f16>()
        .unwrap()
        .into_iter()
        .map(f16::to_bits)
        .collect::<Vec<_>>();
    assert_eq!(actual_bits, [0xbfa9, 0xbbb5, 0xb9c9, 0x4111]);

    let gated = values.cast(DType::F32) * activation::silu(gate.cast(DType::F32));
    let grouped = gated.reshape([1, 1, 2, 2]);
    let divided = (grouped.clone() / (grouped.square().mean_dim(3) + norm.epsilon).sqrt())
        .reshape([1, 1, 4])
        .cast(DType::F16)
        * norm.weight.val().reshape([1, 1, 4]);
    let divided_bits = divided
        .into_data()
        .to_vec::<f16>()
        .unwrap()
        .into_iter()
        .map(f16::to_bits)
        .collect::<Vec<_>>();
    assert_eq!(divided_bits[0], 0xbfaa);

    let bf16_norm = GatedGroupRmsNorm::<TestBackend> {
        weight: Param::from_tensor(
            Tensor::from_floats(
                [-1.934_570_3, 0.954_101_56, -0.644_531_25, -2.945_312_5],
                &device,
            )
            .cast(DType::BF16),
        ),
        groups: 2,
        epsilon: 1e-6,
    };
    let bf16_output = bf16_norm
        .forward(
            Tensor::from_floats(
                [[[6.406_25, -6.535_156_2, 7.539_062_5, -5.781_25]]],
                &device,
            )
            .cast(DType::BF16),
            Tensor::full([1, 1, 4], 80.0, (&device, DType::BF16)),
        )
        .unwrap();
    let bf16_bits = bf16_output
        .into_data()
        .to_vec::<bf16>()
        .unwrap()
        .into_iter()
        .map(bf16::to_bits)
        .collect::<Vec<_>>();
    assert_eq!(bf16_bits, [0xbff6, 0xbf76, 0xbf3a, 0x4022]);
}

/// Softplus and bias addition occur in projection dtype before scan promotion.
///
/// The expected F16 value is from the pinned fallback. Promoting before
/// softplus yields adjacent value `0x0675`, so the literal guards the exact
/// mixed-precision boundary as well as the result dtype.
#[test]
fn discretization_rounds_before_fp32_scan_math() {
    let device = Default::default();
    let raw = Tensor::<TestBackend, 3>::from_floats([[[-9.726_562_5]]], &device).cast(DType::F16);
    let bias = Tensor::<TestBackend, 1>::from_floats([0.501_464_84], &device).cast(DType::F16);
    let valid = Tensor::<TestBackend, 2, Bool>::from_bool([[true]].into(), &device);
    let actual = discretization_steps(raw.clone(), bias.clone(), None, valid);
    assert_eq!(actual.dtype(), DType::F16);
    let actual_bits = actual
        .into_data()
        .to_vec::<f16>()
        .unwrap()
        .into_iter()
        .map(f16::to_bits)
        .collect::<Vec<_>>();
    assert_eq!(actual_bits, [0x0673]);

    let promoted_biased =
        raw.clone().cast(DType::F32) + bias.clone().cast(DType::F32).reshape([1, 1, 1]);
    let promoted = (promoted_biased.clone().clamp_min(0.0)
        + (-promoted_biased.abs()).exp().log1p())
    .cast(DType::F16);
    let promoted_bits = promoted
        .into_data()
        .to_vec::<f16>()
        .unwrap()
        .into_iter()
        .map(f16::to_bits)
        .collect::<Vec<_>>();
    assert_eq!(promoted_bits, [0x0675]);

    let all_valid = || Tensor::<TestBackend, 2, Bool>::from_bool([[true]].into(), &device);
    let clamp = TimeStepClamp::chunked_only(TimeStepLimits {
        lower: 0.001,
        upper: Some(0.01),
    });
    let recurrent = discretization_steps(
        raw.clone(),
        bias.clone(),
        clamp.for_mode(Mamba2Mode::Recurrent),
        all_valid(),
    );
    let chunked = discretization_steps(
        raw,
        bias,
        clamp.for_mode(Mamba2Mode::ChunkedPrefill),
        all_valid(),
    );
    assert_eq!(
        recurrent
            .into_data()
            .to_vec::<f16>()
            .unwrap()
            .into_iter()
            .map(f16::to_bits)
            .collect::<Vec<_>>(),
        [0x0673]
    );
    assert_eq!(
        chunked.into_data().to_vec::<f16>().unwrap(),
        [f16::from_f32(0.001)]
    );

    let bf16_dt = discretization_steps(
        Tensor::<TestBackend, 3>::from_floats([[[-9.726_562_5]]], &device).cast(DType::BF16),
        Tensor::<TestBackend, 1>::from_floats([0.501_464_84], &device).cast(DType::BF16),
        None,
        all_valid(),
    );
    let bf16_bits = bf16_dt
        .into_data()
        .to_vec::<bf16>()
        .unwrap()
        .into_iter()
        .map(bf16::to_bits)
        .collect::<Vec<_>>();
    assert_eq!(bf16_bits, [0x38ca]);
}

/// Recurrent decode preserves the fallback's low-precision update and output graph.
///
/// The oracle comes from `mamba2_selective_state_update` at pinned revision
/// `cfa309cfa34b5497a32fd50af49aa97aac94c054`. Computing `dt * x * B` and
/// the state-to-`C` contraction entirely in FP32 changes the second output
/// from `0x480e` to `0x480d`.
#[test]
fn recurrent_scan_matches_pinned_f16_operation_order() {
    let device = Default::default();
    let x = Tensor::<TestBackend, 4>::from_floats([[[[-2.974_609_4, -2.005_859_4]]]], &device)
        .cast(DType::F16);
    let b = Tensor::<TestBackend, 4>::from_floats([[[[-0.223_266_6, -0.369_873_05]]]], &device)
        .cast(DType::F16);
    let c = Tensor::<TestBackend, 4>::from_floats([[[[2.910_156_2, 3.505_859_4]]]], &device)
        .cast(DType::F16);
    let dt = Tensor::<TestBackend, 3>::from_floats([[[0.836_914_06]]], &device).cast(DType::F16);
    let a = Tensor::<TestBackend, 1>::from_floats([-1.803_729_5], &device);
    let d = Tensor::<TestBackend, 1>::from_floats([-1.289_062_5], &device).cast(DType::F16);
    let initial = Tensor::<TestBackend, 4>::from_floats(
        [[[[2.960_324_3, -0.659_675_4], [2.499_563_9, 0.827_825_8]]]],
        &device,
    );

    let (output, terminal) = recurrent_scan(x, b, c, dt, a, d, initial).unwrap();
    assert_eq!(output.dtype(), DType::F16);
    let bits = output
        .into_data()
        .to_vec::<f16>()
        .unwrap()
        .into_iter()
        .map(f16::to_bits)
        .collect::<Vec<_>>();
    assert_eq!(bits, [0x4909, 0x480e]);
    terminal.into_data().assert_approx_eq::<f32>(
        &TensorData::from([[[[1.210_400_9_f32, 0.775_106_43], [0.927_173_8, 0.804_048]]]]),
        Tolerance::absolute(2e-6),
    );
}

/// Chunked prefill keeps FP32 outputs while recurrent decode restores model dtype.
///
/// This asymmetry is present in the pinned fallback: `mamba2_chunk_scan`
/// promotes its scan tensors, whereas `mamba2_selective_state_update` casts
/// its result back to the input activation dtype before grouped normalization.
#[test]
fn recurrent_and_chunked_scans_preserve_reference_output_dtypes() {
    let device = Default::default();
    let x = Tensor::<TestBackend, 4>::from_floats([[[[0.7, -0.2]], [[-0.4, 1.1]]]], &device)
        .cast(DType::F16);
    let b = Tensor::<TestBackend, 4>::from_floats([[[[0.3]], [[-0.8]]]], &device).cast(DType::F16);
    let c = Tensor::<TestBackend, 4>::from_floats([[[[1.2]], [[0.6]]]], &device).cast(DType::F16);
    let dt = Tensor::<TestBackend, 3>::from_floats([[[0.2], [0.9]]], &device).cast(DType::F16);
    let a = Tensor::<TestBackend, 1>::from_floats([-0.75], &device);
    let d = Tensor::<TestBackend, 1>::from_floats([0.4], &device).cast(DType::F16);
    let initial = Tensor::<TestBackend, 4>::from_floats([[[[0.25], [-0.5]]]], &device);

    let (recurrent, recurrent_state) = recurrent_scan(
        x.clone(),
        b.clone(),
        c.clone(),
        dt.clone(),
        a.clone(),
        d.clone(),
        initial.clone(),
    )
    .unwrap();
    let (chunked, chunked_state) =
        chunked_scan(x, b, c, dt, a, d.cast(DType::F32), initial, 2).unwrap();
    assert_eq!(recurrent.dtype(), DType::F16);
    assert_eq!(chunked.dtype(), DType::F32);
    assert_eq!(recurrent_state.dtype(), DType::F32);
    assert_eq!(chunked_state.dtype(), DType::F32);
    recurrent
        .cast(DType::F32)
        .into_data()
        .assert_approx_eq::<f32>(&chunked.into_data(), Tolerance::absolute(3e-3));

    let bf16_x = Tensor::<TestBackend, 4>::ones([1, 1, 1, 1], (&device, DType::BF16));
    let bf16_b = Tensor::<TestBackend, 4>::ones([1, 1, 1, 1], (&device, DType::BF16));
    let bf16_c = Tensor::<TestBackend, 4>::ones([1, 1, 1, 1], (&device, DType::BF16));
    let bf16_dt = Tensor::<TestBackend, 3>::ones([1, 1, 1], (&device, DType::BF16));
    let bf16_d = Tensor::<TestBackend, 1>::ones([1], (&device, DType::BF16));
    let bf16_initial = Tensor::<TestBackend, 4>::zeros([1, 1, 1, 1], &device);
    let (bf16_recurrent, _) = recurrent_scan(
        bf16_x.clone(),
        bf16_b.clone(),
        bf16_c.clone(),
        bf16_dt.clone(),
        Tensor::from_floats([-1.0], &device),
        bf16_d.clone(),
        bf16_initial.clone(),
    )
    .unwrap();
    let (bf16_chunked, _) = chunked_scan(
        bf16_x,
        bf16_b,
        bf16_c,
        bf16_dt,
        Tensor::from_floats([-1.0], &device),
        bf16_d.cast(DType::F32),
        bf16_initial,
        1,
    )
    .unwrap();
    assert_eq!(bf16_recurrent.dtype(), DType::BF16);
    assert_eq!(bf16_chunked.dtype(), DType::F32);
}

/// Reduced-precision recurrent caches round between consecutive tokens.
///
/// A multi-token recurrent call is defined as repeated decode updates. This
/// exact equality prevents an optimization from accidentally retaining an
/// unpublished FP32 state between tokens when the configured cache is F16.
#[test]
fn f16_recurrent_cache_matches_repeated_single_token_updates() {
    let device = Default::default();
    let x = Tensor::<TestBackend, 4>::from_floats([[[[1.3, -0.7]], [[-0.2, 0.9]]]], &device)
        .cast(DType::F16);
    let b = Tensor::<TestBackend, 4>::from_floats([[[[0.6]], [[1.1]]]], &device).cast(DType::F16);
    let c = Tensor::<TestBackend, 4>::from_floats([[[[-0.4]], [[0.8]]]], &device).cast(DType::F16);
    let dt = Tensor::<TestBackend, 3>::from_floats([[[0.35], [0.55]]], &device).cast(DType::F16);
    let a = Tensor::<TestBackend, 1>::from_floats([-1.25], &device);
    let d = Tensor::<TestBackend, 1>::from_floats([0.75], &device).cast(DType::F16);
    let initial =
        Tensor::<TestBackend, 4>::from_floats([[[[0.33], [-0.27]]]], &device).cast(DType::F16);

    let (combined_output, combined_state) = recurrent_scan(
        x.clone(),
        b.clone(),
        c.clone(),
        dt.clone(),
        a.clone(),
        d.clone(),
        initial.clone(),
    )
    .unwrap();
    let (first_output, first_state) = recurrent_scan(
        x.clone().narrow(1, 0, 1),
        b.clone().narrow(1, 0, 1),
        c.clone().narrow(1, 0, 1),
        dt.clone().narrow(1, 0, 1),
        a.clone(),
        d.clone(),
        initial,
    )
    .unwrap();
    let (second_output, repeated_state) = recurrent_scan(
        x.narrow(1, 1, 1),
        b.narrow(1, 1, 1),
        c.narrow(1, 1, 1),
        dt.narrow(1, 1, 1),
        a,
        d,
        first_state,
    )
    .unwrap();
    assert_eq!(
        combined_output.to_data(),
        Tensor::cat(vec![first_output, second_output], 1).to_data()
    );
    assert_eq!(combined_state.to_data(), repeated_state.to_data());
}

/// Semantic invalid tokens produce identity recurrent transitions.
///
/// Transformers' padding helper leaves `dt` active and would decay the
/// state here. This crate intentionally zeroes it because stateful model
/// execution compacts true-valid tokens before invoking Mamba.
#[test]
fn semantic_mask_prevents_padding_decay() {
    let device = Default::default();
    let valid = Tensor::<TestBackend, 2, Bool>::from_bool([[false]].into(), &device);
    let dt = discretization_steps(
        Tensor::zeros([1, 1, 1], &device),
        Tensor::ones([1], &device),
        Some(TimeStepLimits::at_least(0.001)),
        valid,
    );
    assert_eq!(dt.to_data(), TensorData::zeros::<f32, _>([1, 1, 1]));

    let initial = Tensor::<TestBackend, 4>::from_floats([[[[3.25]]]], &device);
    let (_, terminal) = recurrent_scan(
        Tensor::zeros([1, 1, 1, 1], &device),
        Tensor::zeros([1, 1, 1, 1], &device),
        Tensor::zeros([1, 1, 1, 1], &device),
        dt,
        Tensor::from_floats([-7.0], &device),
        Tensor::ones([1], &device),
        initial.clone(),
    )
    .unwrap();
    assert_eq!(terminal.to_data(), initial.to_data());
}

/// Gated normalization rejects every invalid grouping or tensor geometry.
#[test]
fn grouped_norm_rejects_invalid_geometry() {
    let device = Default::default();
    let values = Tensor::<TestBackend, 3>::ones([1, 2, 4], &device);
    let gate = Tensor::<TestBackend, 3>::ones([1, 2, 4], &device);
    let norm = GatedGroupRmsNorm::<TestBackend> {
        weight: Param::from_tensor(Tensor::ones([4], &device)),
        groups: 3,
        epsilon: 1e-6,
    };
    assert!(norm.forward(values.clone(), gate.clone()).is_err());

    let norm = GatedGroupRmsNorm::<TestBackend> {
        weight: Param::from_tensor(Tensor::ones([3], &device)),
        groups: 2,
        epsilon: 1e-6,
    };
    assert!(norm.forward(values.clone(), gate).is_err());

    let norm = GatedGroupRmsNorm::<TestBackend> {
        weight: Param::from_tensor(Tensor::ones([4], &device)),
        groups: 2,
        epsilon: 1e-6,
    };
    assert!(
        norm.forward(values, Tensor::ones([1, 1, 4], &device))
            .is_err()
    );

    let norm = GatedGroupRmsNorm::<TestBackend> {
        weight: Param::from_tensor(Tensor::ones([4], &device)),
        groups: 2,
        epsilon: f64::NAN,
    };
    assert!(
        norm.forward(
            Tensor::ones([1, 2, 4], &device),
            Tensor::ones([1, 2, 4], &device),
        )
        .is_err()
    );
}

/// Invalid groups and state dtypes return contextual errors before kernels.
#[test]
fn invalid_configuration_and_state_are_rejected() {
    let device = Default::default();
    let mut invalid = config();
    invalid.groups = 3;
    assert!(invalid.validate().is_err());

    let mixer = config().init::<TestBackend>(&device).unwrap();
    let mut state = Mamba2State::zeros(&config(), 1, DType::F32, &device).unwrap();
    state.recurrent = state.recurrent.cast(DType::F64);
    let error = mixer
        .forward_with_state(
            Tensor::zeros([1, 1, 8], &device),
            None,
            &state,
            Mamba2Mode::Recurrent,
        )
        .unwrap_err();
    assert!(error.to_string().contains("recurrent"));
}

/// Low-precision terminal storage is promoted for scan math and restored afterward.
#[test]
fn recurrent_storage_dtype_round_trips_across_a_segment() {
    let device = Default::default();
    let mixer = config().init::<TestBackend>(&device).unwrap();
    let state =
        Mamba2State::zeros_with_dtypes(&config(), 1, DType::F16, DType::F32, &device).unwrap();
    let (_, next) = mixer
        .forward_with_state(
            Tensor::zeros([1, 2, 8], &device),
            None,
            &state,
            Mamba2Mode::ChunkedPrefill,
        )
        .unwrap();
    assert_eq!(next.recurrent.dtype(), DType::F16);
    assert_eq!(next.convolution.dtype(), DType::F32);
    assert!(
        Mamba2State::<TestBackend>::zeros_with_dtypes(
            &config(),
            1,
            DType::I32,
            DType::F32,
            &device,
        )
        .is_err()
    );
}

/// F32 discretization uses PyTorch's thresholded `softplus` graph exactly.
///
/// PyTorch evaluates `x > 20 ? x : log1p(exp(x))`. The former stable identity
/// `max(x, 0) + log1p(exp(-|x|))` is equal in exact arithmetic but rounds
/// differently for positive `x`; the sweep both pins the reference graph bit
/// for bit and proves the two differ somewhere on this backend, so the test
/// would fail for the old formula.
#[test]
fn discretization_softplus_matches_the_reference_graph() {
    let device = Default::default();
    let inputs = (0..2_048)
        .map(|index| -30.0 + index as f32 * 0.029_3)
        .collect::<Vec<_>>();
    let count = inputs.len();
    let raw = Tensor::<TestBackend, 3>::from_data(
        TensorData::new(inputs.clone(), [1, count, 1]),
        &device,
    );
    let valid = Tensor::<TestBackend, 2, Bool>::ones([1, count], &device);
    let actual = discretization_steps(raw.clone(), Tensor::zeros([1], &device), None, valid)
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    let reference = raw
        .clone()
        .exp()
        .log1p()
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    let identity = (raw.clone().clamp_min(0.0) + (-raw.abs()).exp().log1p())
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    for ((&input, &actual), &reference) in inputs.iter().zip(&actual).zip(&reference) {
        let expected = if input > 20.0 { input } else { reference };
        assert_eq!(actual.to_bits(), expected.to_bits(), "softplus({input})");
    }
    assert!(
        actual
            .iter()
            .zip(&identity)
            .any(|(actual, identity)| actual.to_bits() != identity.to_bits()),
        "the sweep no longer distinguishes the reference graph from the stable identity"
    );
    // Large arguments stay finite instead of overflowing `exp`.
    let large = discretization_steps(
        Tensor::<TestBackend, 3>::from_floats([[[100.0], [1.0e30]]], &device),
        Tensor::zeros([1], &device),
        None,
        Tensor::ones([1, 2], &device),
    )
    .into_data()
    .to_vec::<f32>()
    .unwrap();
    assert_eq!(large, [100.0, 1.0e30]);
}

/// One token observes the prefill clamp only in chunked mode.
///
/// With `dt_bias = -12`, `softplus` gives about `6e-6`; chunked prefill raises
/// it to `time_step_min = 1e-3`, the recurrent update keeps it, so the
/// terminal states differ by roughly that ratio. Callers must therefore pick
/// the chunk scan for a first prompt token, as the reference does.
#[test]
fn one_token_clamp_differs_between_modes() {
    let device = Default::default();
    let mut mixer = config().init::<TestBackend>(&device).unwrap();
    mixer.dt_bias = mixer.dt_bias.clone().map(|bias| bias.ones_like() * -12.0);
    let input = Tensor::<TestBackend, 3>::from_floats(
        [[[0.1, 0.2, -0.3, 0.4, 0.5, -0.6, 0.7, 0.8]]],
        &device,
    );
    let state = Mamba2State::zeros(&config(), 1, DType::F32, &device).unwrap();
    let norm = |state: Mamba2State<TestBackend>| state.recurrent.abs().max().into_scalar();
    let (_, chunked) = mixer
        .forward_with_state(input.clone(), None, &state, Mamba2Mode::ChunkedPrefill)
        .unwrap();
    let (_, recurrent) = mixer
        .forward_with_state(input, None, &state, Mamba2Mode::Recurrent)
        .unwrap();
    let ratio = norm(chunked) / norm(recurrent);
    assert!(ratio > 100.0 && ratio < 200.0, "{ratio}");
}
