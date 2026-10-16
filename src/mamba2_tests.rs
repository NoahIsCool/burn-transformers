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
    let (output, terminal, _) = recurrent_scan(x, b, c, dt, a, d, initial, false).unwrap();
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
    let transitions = SegmentMasks::new(3, &device).transitions(
        Tensor::<TestBackend, 4>::from_floats([[[[-100_000_000.0, -1.0, -2.0]]]], &device),
    );
    transitions.into_data().assert_approx_eq::<f32>(
        &TensorData::from([[[[
            [1.0, 0.0, 0.0],
            [(-1.0_f32).exp(), 1.0, 0.0],
            [(-3.0_f32).exp(), (-2.0_f32).exp(), 1.0],
        ]]]]),
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
    let actual = discretization_steps(raw.clone(), bias.clone(), None);
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

    let clamp = TimeStepClamp::chunked_only(TimeStepLimits {
        lower: 0.001,
        upper: Some(0.01),
    });
    let recurrent = discretization_steps(
        raw.clone(),
        bias.clone(),
        clamp.for_mode(Mamba2Mode::Recurrent),
    );
    let chunked = discretization_steps(raw, bias, clamp.for_mode(Mamba2Mode::ChunkedPrefill));
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

    let (output, terminal, _) = recurrent_scan(x, b, c, dt, a, d, initial, false).unwrap();
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

    let (recurrent, recurrent_state, _) = recurrent_scan(
        x.clone(),
        b.clone(),
        c.clone(),
        dt.clone(),
        a.clone(),
        d.clone(),
        initial.clone(),
        false,
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
    let (bf16_recurrent, _, _) = recurrent_scan(
        bf16_x.clone(),
        bf16_b.clone(),
        bf16_c.clone(),
        bf16_dt.clone(),
        Tensor::from_floats([-1.0], &device),
        bf16_d.clone(),
        bf16_initial.clone(),
        false,
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

    let (combined_output, combined_state, _) = recurrent_scan(
        x.clone(),
        b.clone(),
        c.clone(),
        dt.clone(),
        a.clone(),
        d.clone(),
        initial.clone(),
        false,
    )
    .unwrap();
    let (first_output, first_state, _) = recurrent_scan(
        x.clone().narrow(1, 0, 1),
        b.clone().narrow(1, 0, 1),
        c.clone().narrow(1, 0, 1),
        dt.clone().narrow(1, 0, 1),
        a.clone(),
        d.clone(),
        initial,
        false,
    )
    .unwrap();
    let (second_output, repeated_state, _) = recurrent_scan(
        x.narrow(1, 1, 1),
        b.narrow(1, 1, 1),
        c.narrow(1, 1, 1),
        dt.narrow(1, 1, 1),
        a,
        d,
        first_state,
        false,
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
    let invalid = Tensor::<TestBackend, 2, Bool>::from_bool([[true]].into(), &device);
    let dt = zero_invalid(
        discretization_steps(
            Tensor::zeros([1, 1, 1], &device),
            Tensor::ones([1], &device),
            Some(TimeStepLimits::at_least(0.001)),
        ),
        Some(&invalid),
    );
    assert_eq!(dt.to_data(), TensorData::zeros::<f32, _>([1, 1, 1]));
    // Replacement, not multiplication: non-finite padding is cleared too.
    let cleared = zero_invalid(
        Tensor::<TestBackend, 3>::from_floats([[[f32::NAN, f32::INFINITY]]], &device),
        Some(&invalid),
    );
    assert_eq!(cleared.to_data(), TensorData::zeros::<f32, _>([1, 1, 2]));

    let initial = Tensor::<TestBackend, 4>::from_floats([[[[3.25]]]], &device);
    let (_, terminal, _) = recurrent_scan(
        Tensor::zeros([1, 1, 1, 1], &device),
        Tensor::zeros([1, 1, 1, 1], &device),
        Tensor::zeros([1, 1, 1, 1], &device),
        dt,
        Tensor::from_floats([-7.0], &device),
        Tensor::ones([1], &device),
        initial.clone(),
        false,
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

    for epsilon in [f64::NAN, 0.0, -1e-6] {
        let norm = GatedGroupRmsNorm::<TestBackend> {
            weight: Param::from_tensor(Tensor::ones([4], &device)),
            groups: 2,
            epsilon,
        };
        assert_configuration(
            norm.forward(
                Tensor::ones([1, 2, 4], &device),
                Tensor::ones([1, 2, 4], &device),
            ),
            "mamba2.gated_norm.epsilon",
            "greater than zero",
        );
    }
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
    let actual = discretization_steps(raw.clone(), Tensor::zeros([1], &device), None)
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
/// the chunk scan for a first prompt token, as the reference does. The input
/// projection is fixed and small so every head's step stays near `6e-6`
/// (randomly initialized projections could move a head above the clamp).
#[test]
fn one_token_clamp_differs_between_modes() {
    let device = Default::default();
    let mut mixer = config().init::<TestBackend>(&device).unwrap();
    let [inputs, outputs] = mixer.in_proj.weight.dims();
    mixer.in_proj.weight =
        Param::from_tensor(waves([1, inputs, outputs], 3.0, 0.01).reshape([inputs, outputs]));
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

/// Deterministic `[batch, tokens, width]` values with varied signs and magnitudes.
fn waves(shape: [usize; 3], salt: f32, scale: f32) -> Tensor<TestBackend, 3> {
    let count = shape.iter().product();
    let values = (0..count)
        .map(|index| ((index as f32 + salt) * 0.611).sin() * scale)
        .collect::<Vec<_>>();
    Tensor::from_data(TensorData::new(values, shape), &Default::default())
}

/// Deterministic four-dimensional values for direct scan fixtures.
fn waves4(shape: [usize; 4], salt: f32, scale: f32) -> Tensor<TestBackend, 4> {
    let count = shape.iter().product();
    let values = (0..count)
        .map(|index| ((index as f32 + salt) * 0.437).cos() * scale)
        .collect::<Vec<_>>();
    Tensor::from_data(TensorData::new(values, shape), &Default::default())
}

/// The former per-head, chunk-sequential SSD scan (test oracle only).
///
/// It repeats `B/C` to every head and processes one chunk per iteration,
/// exactly as the mixer did before the grouped block-vectorized scan.
#[allow(clippy::too_many_arguments)]
fn legacy_chunked_scan(
    x: Tensor<TestBackend, 4>,
    b: Tensor<TestBackend, 4>,
    c: Tensor<TestBackend, 4>,
    dt: Tensor<TestBackend, 3>,
    a: Tensor<TestBackend, 1>,
    d: Tensor<TestBackend, 1>,
    mut state: Tensor<TestBackend, 4>,
    chunk_size: usize,
) -> (Tensor<TestBackend, 4>, Tensor<TestBackend, 4>) {
    let [batch, sequence, heads, _] = x.dims();
    let b = repeat(b, heads);
    let c = repeat(c, heads);
    let mut chunks = Vec::new();
    for start in (0..sequence).step_by(chunk_size) {
        let width = (sequence - start).min(chunk_size);
        let x = x.clone().narrow(1, start, width).cast(DType::F32);
        let b = b.clone().narrow(1, start, width).cast(DType::F32);
        let c = c.clone().narrow(1, start, width).cast(DType::F32);
        let dt = dt.clone().narrow(1, start, width).cast(DType::F32);
        let log_decay = dt.clone() * a.clone().reshape([1, 1, heads]);
        let cumulative = log_decay.clone().cumsum(1);
        let transitions = SegmentMasks::new(width, &x.device())
            .transitions(log_decay.swap_dims(1, 2).unsqueeze_dim::<4>(2))
            .reshape([batch, heads, width, width])
            .permute([0, 2, 3, 1]);
        let kernel = c
            .clone()
            .permute([0, 2, 1, 3])
            .matmul(b.clone().permute([0, 2, 3, 1]))
            .permute([0, 2, 3, 1]);
        let input = x.clone() * dt.unsqueeze_dim::<4>(3);
        let within = (kernel * transitions)
            .permute([0, 3, 1, 2])
            .matmul(input.clone().permute([0, 2, 1, 3]))
            .permute([0, 2, 1, 3]);
        let from_initial = state
            .clone()
            .matmul(c.permute([0, 2, 3, 1]))
            .permute([0, 3, 1, 2])
            * cumulative.clone().exp().unsqueeze_dim::<4>(3);
        chunks.push(within + from_initial + x * d.clone().reshape([1, 1, heads, 1]));
        let terminal = cumulative
            .clone()
            .narrow(1, width - 1, 1)
            .reshape([batch, heads]);
        let decay = (terminal.clone().reshape([batch, 1, heads]) - cumulative).exp();
        let updates = (input * decay.unsqueeze_dim::<4>(3))
            .permute([0, 2, 3, 1])
            .matmul(b.permute([0, 2, 1, 3]));
        state = state * terminal.exp().reshape([batch, heads, 1, 1]) + updates;
    }
    (Tensor::cat(chunks, 1), state)
}

/// Repeats `[b, t, g, n]` group values to `[b, t, h, n]` heads (test oracle only).
fn repeat(tensor: Tensor<TestBackend, 4>, heads: usize) -> Tensor<TestBackend, 4> {
    let [batch, sequence, groups, state] = tensor.dims();
    tensor
        .unsqueeze_dim::<5>(3)
        .expand([batch, sequence, groups, heads / groups, state])
        .reshape([batch, sequence, heads, state])
}

/// Direct scan operands `(x, b, c, dt, a, d, initial)` for a grouped fixture.
#[allow(clippy::type_complexity)]
fn scan_operands(
    batch: usize,
    sequence: usize,
    heads: usize,
    groups: usize,
) -> (
    Tensor<TestBackend, 4>,
    Tensor<TestBackend, 4>,
    Tensor<TestBackend, 4>,
    Tensor<TestBackend, 3>,
    Tensor<TestBackend, 1>,
    Tensor<TestBackend, 1>,
    Tensor<TestBackend, 4>,
) {
    let device = Default::default();
    let (head_dim, state) = (2, 3);
    let dt = waves([batch, sequence, heads], 3.0, 0.4).abs() + 0.05;
    let a = Tensor::<TestBackend, 1>::from_data(
        TensorData::new(
            (0..heads)
                .map(|head| -0.3 - 0.4 * head as f32)
                .collect::<Vec<_>>(),
            [heads],
        ),
        &device,
    );
    let d = Tensor::<TestBackend, 1>::from_data(
        TensorData::new(
            (0..heads)
                .map(|head| 0.5 + 0.1 * head as f32)
                .collect::<Vec<_>>(),
            [heads],
        ),
        &device,
    );
    (
        waves4([batch, sequence, heads, head_dim], 1.0, 0.9),
        waves4([batch, sequence, groups, state], 7.0, 0.8),
        waves4([batch, sequence, groups, state], 11.0, 0.7),
        dt,
        a,
        d,
        waves4([batch, heads, head_dim, state], 5.0, 0.3),
    )
}

/// Grouped block scans equal the former per-head sequential scan and the recurrence.
///
/// Budgets force one chunk per block, two chunks per block with a ragged
/// final block, and the whole segment in one block; groups are shared by two
/// heads, so the per-group content kernel and broadcast state contractions
/// are exercised.
#[test]
fn grouped_block_scan_matches_sequential_and_recurrent_oracles() {
    let (x, b, c, dt, a, d, initial) = scan_operands(2, 11, 4, 2);
    let (legacy, legacy_state) = legacy_chunked_scan(
        x.clone(),
        b.clone(),
        c.clone(),
        dt.clone(),
        a.clone(),
        d.clone(),
        initial.clone(),
        3,
    );
    let (recurrent, recurrent_state, _) = recurrent_scan(
        x.clone(),
        b.clone(),
        c.clone(),
        dt.clone(),
        a.clone(),
        d.clone(),
        initial.clone(),
        false,
    )
    .unwrap();
    for budget in [1, 2 * 2 * 4 * 9, usize::MAX] {
        let (output, state) = chunked_scan_with_budget(
            x.clone(),
            b.clone(),
            c.clone(),
            dt.clone(),
            a.clone(),
            d.clone(),
            initial.clone(),
            3,
            budget,
        )
        .unwrap();
        for expected in [&legacy, &recurrent] {
            output
                .clone()
                .into_data()
                .assert_approx_eq::<f32>(&expected.clone().into_data(), Tolerance::absolute(2e-5));
        }
        for expected in [&legacy_state, &recurrent_state] {
            state
                .clone()
                .into_data()
                .assert_approx_eq::<f32>(&expected.clone().into_data(), Tolerance::absolute(2e-5));
        }
    }
}

/// A single-chunk prompt keeps the former scan's arithmetic for its outputs.
#[test]
fn single_chunk_outputs_match_the_former_scan_closely() {
    let (x, b, c, dt, a, d, initial) = scan_operands(1, 4, 4, 2);
    let (legacy, _) = legacy_chunked_scan(
        x.clone(),
        b.clone(),
        c.clone(),
        dt.clone(),
        a.clone(),
        d.clone(),
        initial.clone(),
        8,
    );
    let (output, _) = chunked_scan(x, b, c, dt, a, d, initial, 8).unwrap();
    output
        .into_data()
        .assert_approx_eq::<f32>(&legacy.into_data(), Tolerance::absolute(1e-6));
}

/// Ragged right-padded rows equal independent compact runs; empty rows keep state exactly.
#[test]
fn ragged_rows_match_independent_compact_runs() {
    let device = Default::default();
    let mixer = config().init::<TestBackend>(&device).unwrap();
    let hidden = waves([3, 5, 8], 2.0, 0.6);
    let mut state = Mamba2State::zeros(&config(), 3, DType::F32, &device).unwrap();
    state.recurrent = waves4([3, 4, 2, 3], 9.0, 0.2);
    state.convolution = waves([3, 20, 2], 4.0, 0.3);
    let counts = [5, 2, 0];
    let rows = counts.map(|tokens| Mamba2Row {
        tokens,
        mode: Mamba2Mode::ChunkedPrefill,
    });
    let (output, next) = mixer.forward_rows(hidden.clone(), &rows, &state).unwrap();
    for (row, &count) in counts.iter().enumerate() {
        let row_state = Mamba2State {
            recurrent: state.recurrent.clone().narrow(0, row, 1),
            convolution: state.convolution.clone().narrow(0, row, 1),
        };
        let row_matrix = next.recurrent.clone().narrow(0, row, 1);
        let row_convolution = next.convolution.clone().narrow(0, row, 1);
        if count == 0 {
            assert_eq!(row_matrix.into_data(), row_state.recurrent.into_data());
            assert_eq!(
                row_convolution.into_data(),
                row_state.convolution.into_data()
            );
            continue;
        }
        let (expected, expected_state) = mixer
            .forward_with_state(
                hidden.clone().narrow(0, row, 1).narrow(1, 0, count),
                None,
                &row_state,
                Mamba2Mode::ChunkedPrefill,
            )
            .unwrap();
        output
            .clone()
            .narrow(0, row, 1)
            .narrow(1, 0, count)
            .into_data()
            .assert_approx_eq::<f32>(&expected.into_data(), Tolerance::absolute(2e-6));
        row_matrix.into_data().assert_approx_eq::<f32>(
            &expected_state.recurrent.into_data(),
            Tolerance::absolute(2e-6),
        );
        row_convolution.into_data().assert_approx_eq::<f32>(
            &expected_state.convolution.into_data(),
            Tolerance::absolute(1e-6),
        );
    }
}

/// Rows in different modes run as separate groups and return in batch order.
#[test]
fn mixed_mode_rows_match_per_row_execution() {
    let device = Default::default();
    let mixer = config().init::<TestBackend>(&device).unwrap();
    let hidden = waves([3, 3, 8], 6.0, 0.5);
    let mut state = Mamba2State::zeros(&config(), 3, DType::F32, &device).unwrap();
    state.recurrent = waves4([3, 4, 2, 3], 1.0, 0.25);
    state.convolution = waves([3, 20, 2], 8.0, 0.2);
    let rows = [
        Mamba2Row {
            tokens: 1,
            mode: Mamba2Mode::Recurrent,
        },
        Mamba2Row {
            tokens: 3,
            mode: Mamba2Mode::ChunkedPrefill,
        },
        Mamba2Row {
            tokens: 1,
            mode: Mamba2Mode::Recurrent,
        },
    ];
    assert_eq!(
        mode_groups(&rows).groups,
        vec![
            (Mamba2Mode::Recurrent, vec![0, 2]),
            (Mamba2Mode::ChunkedPrefill, vec![1]),
        ]
    );
    let (output, next) = mixer.forward_rows(hidden.clone(), &rows, &state).unwrap();
    for (row, plan) in rows.iter().enumerate() {
        let row_state = Mamba2State {
            recurrent: state.recurrent.clone().narrow(0, row, 1),
            convolution: state.convolution.clone().narrow(0, row, 1),
        };
        let (expected, expected_state) = mixer
            .forward_with_state(
                hidden.clone().narrow(0, row, 1).narrow(1, 0, plan.tokens),
                None,
                &row_state,
                plan.mode,
            )
            .unwrap();
        output
            .clone()
            .narrow(0, row, 1)
            .narrow(1, 0, plan.tokens)
            .into_data()
            .assert_approx_eq::<f32>(&expected.into_data(), Tolerance::absolute(2e-6));
        next.recurrent
            .clone()
            .narrow(0, row, 1)
            .into_data()
            .assert_approx_eq::<f32>(
                &expected_state.recurrent.into_data(),
                Tolerance::absolute(2e-6),
            );
        next.convolution
            .clone()
            .narrow(0, row, 1)
            .into_data()
            .assert_approx_eq::<f32>(
                &expected_state.convolution.into_data(),
                Tolerance::absolute(1e-6),
            );
    }
    // Inactive rows never split the batch into a second group.
    let idle = [
        Mamba2Row {
            tokens: 0,
            mode: Mamba2Mode::ChunkedPrefill,
        },
        Mamba2Row {
            tokens: 1,
            mode: Mamba2Mode::Recurrent,
        },
    ];
    assert_eq!(
        mode_groups(&idle).groups,
        vec![(Mamba2Mode::Recurrent, vec![0, 1])]
    );
}

/// Snapshot `t` equals the state after `t + 1` single-token decode calls.
///
/// The scan itself is the same recurrence; only the batched projections
/// (`[2, 4]` tokens here against one) may round differently, so values agree
/// to F32 rounding and, for an F16 state, to one F16 ULP.
#[test]
fn snapshots_equal_repeated_single_token_updates() {
    let device = Default::default();
    for (state_dtype, tolerance) in [(DType::F32, 2e-6), (DType::F16, 5e-4)] {
        let mixer = config().init::<TestBackend>(&device).unwrap();
        let hidden = waves([2, 4, 8], 3.0, 0.7);
        let mut state =
            Mamba2State::zeros_with_dtypes(&config(), 2, state_dtype, DType::F32, &device).unwrap();
        state.recurrent = waves4([2, 4, 2, 3], 2.0, 0.3).cast(state_dtype);
        state.convolution = waves([2, 20, 2], 5.0, 0.4);
        let counts = [4, 2];
        let (output, snapshots) = mixer
            .forward_rows_with_snapshots(hidden.clone(), &counts, &state)
            .unwrap();
        assert_eq!(snapshots.recurrent.dims(), [2, 4, 4, 2, 3]);
        assert_eq!(snapshots.convolution.dims(), [2, 4, 20, 2]);
        assert_eq!(snapshots.recurrent.dtype(), state_dtype);
        let close = |actual: TensorData, expected: TensorData| {
            actual.convert::<f32>().assert_approx_eq::<f32>(
                &expected.convert::<f32>(),
                Tolerance::absolute(tolerance),
            );
        };
        for (row, &count) in counts.iter().enumerate() {
            let mut current = Mamba2State {
                recurrent: state.recurrent.clone().narrow(0, row, 1),
                convolution: state.convolution.clone().narrow(0, row, 1),
            };
            for token in 0..count {
                let (expected, next) = mixer
                    .forward_with_state(
                        hidden.clone().narrow(0, row, 1).narrow(1, token, 1),
                        None,
                        &current,
                        Mamba2Mode::Recurrent,
                    )
                    .unwrap();
                close(
                    output
                        .clone()
                        .narrow(0, row, 1)
                        .narrow(1, token, 1)
                        .into_data(),
                    expected.into_data(),
                );
                close(
                    snapshots
                        .recurrent
                        .clone()
                        .narrow(0, row, 1)
                        .narrow(1, token, 1)
                        .reshape([1, 4, 2, 3])
                        .into_data(),
                    next.recurrent.clone().into_data(),
                );
                close(
                    snapshots
                        .convolution
                        .clone()
                        .narrow(0, row, 1)
                        .narrow(1, token, 1)
                        .reshape([1, 20, 2])
                        .into_data(),
                    next.convolution.clone().into_data(),
                );
                current = next;
            }
        }
    }
}

/// Row plans must describe every row and fit the padded width.
#[test]
fn invalid_row_plans_are_rejected() {
    let device = Default::default();
    let mixer = config().init::<TestBackend>(&device).unwrap();
    let state = Mamba2State::zeros(&config(), 2, DType::F32, &device).unwrap();
    let hidden = waves([2, 3, 8], 1.0, 0.5);
    let row = |tokens| Mamba2Row {
        tokens,
        mode: Mamba2Mode::ChunkedPrefill,
    };
    assert!(
        mixer
            .forward_rows(hidden.clone(), &[row(3)], &state)
            .is_err()
    );
    assert!(
        mixer
            .forward_rows(hidden.clone(), &[row(3), row(4)], &state)
            .is_err()
    );
    assert!(
        mixer
            .forward_rows_with_snapshots(hidden, &[1, 4], &state)
            .is_err()
    );
}

/// The block scan stays differentiable: gradients reach the input projection.
#[cfg(feature = "autodiff")]
#[test]
fn chunked_prefill_is_differentiable() {
    use burn::backend::Autodiff;

    /// Autodiff wrapper used to check gradient flow through the block scan.
    type AdBackend = Autodiff<TestBackend>;
    let device = Default::default();
    let mixer = config().init::<AdBackend>(&device).unwrap();
    let hidden = Tensor::<AdBackend, 3>::from_data(waves([1, 5, 8], 2.0, 0.6).into_data(), &device);
    let loss = mixer
        .forward(hidden, None, Mamba2Mode::ChunkedPrefill)
        .unwrap()
        .sum();
    let gradients = loss.backward();
    assert!(mixer.in_proj.weight.grad(&gradients).is_some());
    assert!(mixer.a_log.grad(&gradients).is_some());
}

proptest::proptest! {
    #![proptest_config(proptest::test_runner::Config::with_cases(24))]

    /// Every chunk width, block budget, grouping, and length agrees with the recurrence.
    #[test]
    fn block_scan_matches_recurrence_for_any_partition(
        sequence in 1usize..14,
        chunk_size in 1usize..6,
        grouping in 0usize..3,
        budget in 1usize..400,
    ) {
        let (heads, groups) = [(2, 1), (4, 2), (4, 4)][grouping];
        let (x, b, c, dt, a, d, initial) = scan_operands(1, sequence, heads, groups);
        let (expected, expected_state, _) = recurrent_scan(
            x.clone(), b.clone(), c.clone(), dt.clone(), a.clone(), d.clone(), initial.clone(), false,
        ).unwrap();
        let (actual, state) = chunked_scan_with_budget(
            x, b, c, dt, a, d, initial, chunk_size, budget,
        ).unwrap();
        let actual = actual.into_data().to_vec::<f32>().unwrap();
        let expected = expected.into_data().to_vec::<f32>().unwrap();
        for (actual, expected) in actual.iter().zip(&expected) {
            proptest::prop_assert!((actual - expected).abs() <= 3e-5, "{} != {}", actual, expected);
        }
        let state = state.into_data().to_vec::<f32>().unwrap();
        let expected_state = expected_state.into_data().to_vec::<f32>().unwrap();
        for (actual, expected) in state.iter().zip(&expected_state) {
            proptest::prop_assert!((actual - expected).abs() <= 3e-5, "{} != {}", actual, expected);
        }
    }
}

/// Release-mode timing of the grouped block scan against the former sequential scan.
///
/// Run with `cargo test --release --lib mamba2::tests::block_scan_timing --
/// --ignored --nocapture`. Shapes follow a scaled-down Lightning layer
/// (8 heads per group, 128-token chunks).
#[test]
#[ignore = "release-mode timing benchmark"]
fn block_scan_timing() {
    use std::time::Instant;
    let device = Default::default();
    let (batch, sequence, heads, groups, head_dim, state, chunk) = (1, 2048, 16, 2, 64, 128, 128);
    let x = waves4([batch, sequence, heads, head_dim], 1.0, 0.5);
    let b = waves4([batch, sequence, groups, state], 2.0, 0.5);
    let c = waves4([batch, sequence, groups, state], 3.0, 0.5);
    let dt = waves([batch, sequence, heads], 4.0, 0.05).abs() + 0.001;
    let a = Tensor::<TestBackend, 1>::ones([heads], &device) * -0.5;
    let d = Tensor::<TestBackend, 1>::ones([heads], &device);
    let initial = Tensor::<TestBackend, 4>::zeros([batch, heads, head_dim, state], &device);
    let time = |label: &str, run: &dyn Fn() -> Tensor<TestBackend, 4>| {
        let _ = run().into_data();
        let started = Instant::now();
        for _ in 0..3 {
            let _ = run().into_data();
        }
        let seconds = started.elapsed().as_secs_f64() / 3.0;
        eprintln!("mamba2 scan timing: {label}: {:.1} ms", seconds * 1e3);
        seconds
    };
    let legacy = time("former per-head sequential scan", &|| {
        legacy_chunked_scan(
            x.clone(),
            b.clone(),
            c.clone(),
            dt.clone(),
            a.clone(),
            d.clone(),
            initial.clone(),
            chunk,
        )
        .0
    });
    let blocked = time("grouped block scan", &|| {
        chunked_scan(
            x.clone(),
            b.clone(),
            c.clone(),
            dt.clone(),
            a.clone(),
            d.clone(),
            initial.clone(),
            chunk,
        )
        .unwrap()
        .0
    });
    eprintln!("mamba2 scan timing: speedup {:.2}x", legacy / blocked);
}

/// Exact bit patterns of a floating tensor of any rank, read in its own dtype.
fn bits<const D: usize>(tensor: Tensor<TestBackend, D>) -> Vec<u32> {
    let data = tensor.into_data();
    match data.dtype {
        DType::F32 => data
            .to_vec::<f32>()
            .unwrap()
            .into_iter()
            .map(f32::to_bits)
            .collect(),
        DType::F16 => data
            .to_vec::<f16>()
            .unwrap()
            .into_iter()
            .map(|value| u32::from(value.to_bits()))
            .collect(),
        DType::BF16 => data
            .to_vec::<bf16>()
            .unwrap()
            .into_iter()
            .map(|value| u32::from(value.to_bits()))
            .collect(),
        other => panic!("unexpected dtype {other:?}"),
    }
}

/// The matmul and broadcast forms of the decode update round identically.
///
/// The update `(dt · B) ⊗ x` has a single summand per element, so the batched
/// outer product used on Flex must reproduce the reference's two roundings
/// bit for bit in F32, F16, and BF16, for a batch above one and values of
/// both signs.
#[test]
fn input_update_forms_are_bit_identical() {
    let (batch, heads, head_dim, state) = (3, 4, 5, 6);
    for dtype in [DType::F32, DType::F16, DType::BF16] {
        let dt = (waves([batch, heads, 1], 1.0, 0.3).abs() + 0.01)
            .reshape([batch, heads])
            .cast(dtype);
        let b = waves([batch, heads, state], 2.0, 1.7).cast(dtype);
        let x = waves([batch, heads, head_dim], 3.0, 2.3).cast(dtype);
        let matmul = input_update(dt.clone(), b.clone(), x.clone(), true);
        let broadcast = input_update(dt, b, x, false);
        assert_eq!(matmul.dims(), [batch, heads, head_dim, state]);
        assert_eq!(matmul.dtype(), dtype);
        assert_eq!(bits(matmul), bits(broadcast), "{dtype:?}");
    }
}

/// Recurrent scans with either update form return identical outputs, states, and snapshots.
///
/// Covers grouped heads (`heads != groups`), a batch above one, a non-zero
/// initial state, several tokens with per-token snapshots, and the F16/BF16
/// state dtypes whose per-token rounding makes the snapshots observable.
#[test]
fn recurrent_scan_update_forms_are_bit_identical() {
    let device = Default::default();
    let (batch, sequence, heads, groups, head_dim, state) = (2, 3, 4, 2, 3, 5);
    for dtype in [DType::F32, DType::F16, DType::BF16] {
        let x = waves4([batch, sequence, heads, head_dim], 1.0, 0.9).cast(dtype);
        let b = waves4([batch, sequence, groups, state], 2.0, 0.7).cast(dtype);
        let c = waves4([batch, sequence, groups, state], 3.0, 0.8).cast(dtype);
        let dt = (waves([batch, sequence, heads], 4.0, 0.2).abs() + 0.01).cast(dtype);
        let a = Tensor::<TestBackend, 1>::from_floats([-0.5, -1.0, -0.25, -2.0], &device);
        let d = Tensor::<TestBackend, 1>::from_floats([1.0, 0.5, -0.5, 2.0], &device);
        let initial = waves4([batch, heads, head_dim, state], 5.0, 0.1).cast(dtype);
        let run = |as_matmul| {
            recurrent_scan_with(
                x.clone(),
                b.clone(),
                c.clone(),
                dt.clone(),
                a.clone(),
                d.clone(),
                initial.clone(),
                true,
                as_matmul,
            )
            .unwrap()
        };
        let (matmul_output, matmul_state, matmul_snapshots) = run(true);
        let (broadcast_output, broadcast_state, broadcast_snapshots) = run(false);
        assert_eq!(bits(matmul_output), bits(broadcast_output), "{dtype:?}");
        assert_eq!(bits(matmul_state), bits(broadcast_state), "{dtype:?}");
        assert_eq!(
            bits(matmul_snapshots.unwrap()),
            bits(broadcast_snapshots.unwrap()),
            "{dtype:?}"
        );
    }
}

/// Release-mode timing of the decode update's two forms and of one recurrent decode token.
///
/// Shapes follow Nemotron-H Lightning's Mamba-2 layers (64 heads of width 64,
/// state 128, 8 groups) at batch 1 and 8. The forms are interleaved and the
/// median of 31 samples is reported. Run with `cargo test --release --lib
/// mamba2::tests::recurrent_update_timing -- --ignored --nocapture`.
#[test]
#[ignore = "release-mode timing benchmark"]
fn recurrent_update_timing() {
    use std::time::Instant;
    let device = Default::default();
    let (heads, groups, head_dim, state) = (64, 8, 64, 128);
    let median = |samples: &mut Vec<f64>| {
        samples.sort_by(f64::total_cmp);
        samples[samples.len() / 2] * 1e6
    };
    for batch in [1, 8] {
        let dt = waves([batch, heads, 1], 1.0, 0.3)
            .abs()
            .reshape([batch, heads]);
        let b = waves([batch, heads, state], 2.0, 1.0);
        let x = waves([batch, heads, head_dim], 3.0, 1.0);
        let (mut broadcast, mut matmul) = (Vec::new(), Vec::new());
        for _ in 0..31 {
            for (as_matmul, samples) in [(false, &mut broadcast), (true, &mut matmul)] {
                let started = Instant::now();
                let _ = input_update(dt.clone(), b.clone(), x.clone(), as_matmul).into_data();
                samples.push(started.elapsed().as_secs_f64());
            }
        }
        let (before, after) = (median(&mut broadcast), median(&mut matmul));
        eprintln!(
            "mamba2 decode update [{batch}, {heads}, {head_dim}, {state}]: broadcast {before:.0} us, matmul {after:.0} us ({:.1}x)",
            before / after
        );

        let x = waves4([batch, 1, heads, head_dim], 1.0, 0.9);
        let b = waves4([batch, 1, groups, state], 2.0, 0.7);
        let c = waves4([batch, 1, groups, state], 3.0, 0.8);
        let dt = waves([batch, 1, heads], 4.0, 0.2).abs() + 0.01;
        let a = Tensor::<TestBackend, 1>::ones([heads], &device) * -0.5;
        let d = Tensor::<TestBackend, 1>::ones([heads], &device);
        let initial = waves4([batch, heads, head_dim, state], 5.0, 0.1);
        let (mut broadcast, mut matmul) = (Vec::new(), Vec::new());
        for _ in 0..31 {
            for (as_matmul, samples) in [(false, &mut broadcast), (true, &mut matmul)] {
                let started = Instant::now();
                let (output, terminal, _) = recurrent_scan_with(
                    x.clone(),
                    b.clone(),
                    c.clone(),
                    dt.clone(),
                    a.clone(),
                    d.clone(),
                    initial.clone(),
                    false,
                    as_matmul,
                )
                .unwrap();
                let _ = (output.into_data(), terminal.into_data());
                samples.push(started.elapsed().as_secs_f64());
            }
        }
        let (before, after) = (median(&mut broadcast), median(&mut matmul));
        eprintln!(
            "mamba2 one-token recurrent_scan, batch {batch}: broadcast {before:.0} us, matmul {after:.0} us ({:.1}x)",
            before / after
        );
    }
}

/// The former segment transitions: two `mask_fill`s over expanded Boolean masks (test oracle).
fn legacy_transitions(log_decay: Tensor<TestBackend, 4>) -> Tensor<TestBackend, 5> {
    let [batch, heads, chunks, width] = log_decay.dims();
    let device = log_decay.device();
    let rows = Tensor::<TestBackend, 1, Int>::arange(0..width as i64, &device)
        .reshape([1, 1, 1, width, 1]);
    let columns = Tensor::<TestBackend, 1, Int>::arange(0..width as i64, &device)
        .reshape([1, 1, 1, 1, width]);
    let mask_shape = [1, 1, 1, width, width];
    let outside = columns
        .clone()
        .expand(mask_shape)
        .greater_equal(rows.clone().expand(mask_shape));
    let future = columns.expand(mask_shape).greater(rows.expand(mask_shape));
    let shape = [batch, heads, chunks, width, width];
    log_decay
        .unsqueeze_dim::<5>(4)
        .expand(shape)
        .mask_fill(outside.expand(shape), 0.0)
        .cumsum(3)
        .mask_fill(future.expand(shape), f32::NEG_INFINITY)
        .exp()
}

proptest::proptest! {
    #![proptest_config(proptest::test_runner::Config::with_cases(64))]

    /// Table-based transitions equal the former `mask_fill` formulation bit for bit.
    ///
    /// Log decays are finite and non-positive, as `dt · A` is, and include
    /// exact `+0.0` and `-0.0` (padded steps), at widths 1, 2, a chunk width,
    /// and the `chunks + 1` width of the carry matrix.
    #[test]
    fn segment_transitions_match_the_mask_fill_formulation(
        width in proptest::sample::select(vec![1_usize, 2, 3, 8, 17]),
        batch in 1_usize..3,
        heads in 1_usize..3,
        chunks in 1_usize..3,
        seed in proptest::collection::vec((0_u8..4, -30.0_f32..0.0), 1..64),
    ) {
        let count = batch * heads * chunks * width;
        let values = (0..count)
            .map(|index| match seed[index % seed.len()] {
                (0, _) => 0.0,
                (1, _) => -0.0,
                (_, value) => value * (1.0 + index as f32 * 0.01),
            })
            .collect::<Vec<_>>();
        let log_decay = Tensor::<TestBackend, 4>::from_data(
            TensorData::new(values, [batch, heads, chunks, width]),
            &Default::default(),
        );
        let actual = SegmentMasks::new(width, &Default::default()).transitions(log_decay.clone());
        let expected = legacy_transitions(log_decay);
        proptest::prop_assert_eq!(bits(actual), bits(expected));
    }
}

/// Release-mode timing of the segment transitions against the former `mask_fill` form.
///
/// Shapes are one Lightning Mamba-2 block (64 heads, 128-token chunks) with
/// 4 and 16 chunks. Run with `cargo test --release --lib
/// mamba2::tests::segment_transition_timing -- --ignored --nocapture`.
#[test]
#[ignore = "release-mode timing benchmark"]
fn segment_transition_timing() {
    use std::time::Instant;
    let (heads, width) = (64, 128);
    let masks = SegmentMasks::<TestBackend>::new(width, &Default::default());
    for chunks in [4, 16] {
        let log_decay = -waves4([1, heads, chunks, width], 1.0, 0.05).abs();
        let (mut legacy, mut tables) = (Vec::new(), Vec::new());
        for _ in 0..15 {
            let started = Instant::now();
            let _ = legacy_transitions(log_decay.clone()).into_data();
            legacy.push(started.elapsed().as_secs_f64());
            let started = Instant::now();
            let _ = masks.transitions(log_decay.clone()).into_data();
            tables.push(started.elapsed().as_secs_f64());
        }
        let median = |samples: &mut Vec<f64>| {
            samples.sort_by(f64::total_cmp);
            samples[samples.len() / 2] * 1e3
        };
        let (before, after) = (median(&mut legacy), median(&mut tables));
        eprintln!(
            "mamba2 segment transitions [1, {heads}, {chunks}, {width}, {width}]: mask_fill {before:.1} ms, tables {after:.1} ms ({:.2}x)",
            before / after
        );
    }
}

/// Asserts that `result` is a configuration error for `field` whose reason contains `reason`.
fn assert_configuration<T: core::fmt::Debug>(result: Result<T>, field: &str, reason: &str) {
    match result {
        Err(Error::Configuration {
            field: actual,
            reason: message,
        }) => {
            assert_eq!(actual, field);
            assert!(message.contains(reason), "{field}: {message}");
        }
        other => panic!("expected a {field} configuration error, got {other:?}"),
    }
}

/// Asserts that `result` is a shape error for `tensor`.
fn assert_shape<T: core::fmt::Debug>(result: Result<T>, tensor: &str) {
    match result {
        Err(Error::Shape { tensor: actual, .. }) => assert_eq!(actual, tensor),
        other => panic!("expected a {tensor} shape error, got {other:?}"),
    }
}

/// Clamp constructors select the documented paths and every invalid interval is named.
///
/// `unclamped` leaves both paths free, `both` clamps both, and validation
/// checks each configured path independently: a non-finite or negative lower
/// bound, and a non-finite or inverted upper bound, are rejected under the
/// field of the path that carries them.
#[test]
fn time_step_clamps_select_paths_and_reject_invalid_intervals() {
    let limits = TimeStepLimits {
        lower: 0.001,
        upper: Some(0.1),
    };
    let unclamped = TimeStepClamp::unclamped();
    assert_eq!(unclamped.for_mode(Mamba2Mode::Recurrent), None);
    assert_eq!(unclamped.for_mode(Mamba2Mode::ChunkedPrefill), None);
    let both = TimeStepClamp::both(limits);
    assert_eq!(both.for_mode(Mamba2Mode::Recurrent), Some(limits));
    assert_eq!(both.for_mode(Mamba2Mode::ChunkedPrefill), Some(limits));
    for clamp in [unclamped, both] {
        let mut valid = config();
        valid.time_step_clamp = clamp;
        valid.validate().expect("valid clamp");
    }

    let invalid = [
        (
            TimeStepLimits::at_least(f64::NAN),
            "lower discretization bound",
        ),
        (
            TimeStepLimits::at_least(-1e-3),
            "lower discretization bound",
        ),
        (
            TimeStepLimits {
                lower: 0.0,
                upper: Some(f64::INFINITY),
            },
            "upper discretization bound",
        ),
        (
            TimeStepLimits {
                lower: 0.5,
                upper: Some(0.25),
            },
            "upper discretization bound",
        ),
    ];
    for (bad, reason) in invalid {
        for (clamp, field) in [
            (
                TimeStepClamp::chunked_only(bad),
                "mamba2.time_step_clamp.chunked",
            ),
            (
                TimeStepClamp {
                    chunked: None,
                    recurrent: Some(bad),
                },
                "mamba2.time_step_clamp.recurrent",
            ),
        ] {
            let mut config = config();
            config.time_step_clamp = clamp;
            assert_configuration(config.validate(), field, reason);
        }
    }
}

/// Zero widths, a short kernel, a bad epsilon, and every overflowing width are rejected.
#[test]
fn configuration_validation_names_each_invalid_geometry() {
    for change in [
        (|config: &mut Mamba2Config| config.hidden_size = 0) as fn(&mut Mamba2Config),
        |config| config.num_heads = 0,
        |config| config.head_dim = 0,
        |config| config.state_size = 0,
        |config| config.groups = 0,
        |config| config.convolution_kernel = 1,
        |config| config.chunk_size = 0,
    ] {
        let mut invalid = config();
        change(&mut invalid);
        assert_configuration(invalid.validate(), "mamba2", "must be non-zero");
    }
    for epsilon in [0.0, -1e-6, f64::NAN, f64::INFINITY] {
        let mut invalid = config();
        invalid.normalization_epsilon = epsilon;
        assert_configuration(
            invalid.validate(),
            "mamba2.normalization_epsilon",
            "greater than zero",
        );
    }
    // One head group keeps the geometry valid while each width overflows in turn.
    for (heads, head_dim, state_size, width) in [
        (2, usize::MAX / 2 + 1, 1, "intermediate width"),
        (1, 1, usize::MAX / 2 + 1, "B/C channel width"),
        (1, usize::MAX - 1, 1, "convolution width"),
        (1, usize::MAX / 2, 1, "projection width"),
    ] {
        let mut invalid = config();
        invalid.num_heads = heads;
        invalid.head_dim = head_dim;
        invalid.state_size = state_size;
        invalid.groups = 1;
        assert_configuration(
            invalid.validate(),
            "mamba2",
            &format!("{width} overflows usize"),
        );
    }
}

/// State allocation rejects an empty batch and non-floating convolution storage.
#[test]
fn state_allocation_rejects_empty_batches_and_integer_storage() {
    let device = Default::default();
    assert_configuration(
        Mamba2State::<TestBackend>::zeros(&config(), 0, DType::F32, &device),
        "mamba2.batch_size",
        "must be non-zero",
    );
    assert_configuration(
        Mamba2State::<TestBackend>::zeros(&config(), 1, DType::I32, &device),
        "mamba2.state_dtype",
        "convolution storage must be floating-point",
    );
    // The stateless entry point allocates zero state, so an empty batch fails there.
    let mixer = config().init::<TestBackend>(&device).unwrap();
    let empty = Tensor::<TestBackend, 3>::zeros([0, 2, 8], &device);
    assert_configuration(
        mixer.forward(empty, None, Mamba2Mode::Recurrent),
        "mamba2.batch_size",
        "must be non-zero",
    );
}

/// A manually zeroed chunk size fails stateless allocation and chunked prefill, not decode.
///
/// `Mamba2::config` reconstructs the topology without validating it, so the
/// stateless forward rejects the zero chunk size while allocating its zero
/// state. With supplied state only chunked prefill uses the chunk width: it
/// reports the scan's `mamba2.chunk` error, while the recurrent path, which
/// never chunks, still runs.
#[test]
fn manual_zero_chunk_size_fails_only_where_chunks_are_used() {
    let device = Default::default();
    let mut mixer = config().init::<TestBackend>(&device).unwrap();
    mixer.chunk_size = 0;
    let state = Mamba2State::zeros(&config(), 1, DType::F32, &device).unwrap();
    let hidden = waves([1, 3, 8], 0.5, 0.4);
    assert_configuration(
        mixer.forward(hidden.clone(), None, Mamba2Mode::Recurrent),
        "mamba2",
        "must be non-zero",
    );
    assert_configuration(
        mixer.forward_with_state(hidden.clone(), None, &state, Mamba2Mode::ChunkedPrefill),
        "mamba2.chunk",
        "non-zero chunk width",
    );
    let (output, _) = mixer
        .forward_with_state(hidden, None, &state, Mamba2Mode::Recurrent)
        .expect("the recurrent path never chunks");
    assert_eq!(output.dims(), [1, 3, 8]);
}

/// Row-wise entry points reject mismatched external state on every grouping path.
///
/// `forward_rows` validates the hidden tensor and the row counts itself and
/// leaves the state check to each mixer pass, so a state whose matrix has the
/// wrong `state_size` must surface the pass's `mamba2.state.recurrent` shape
/// error whether all rows share one mode (one pass), the modes differ (one
/// pass per mode group over the selected state rows), or the snapshot entry
/// point runs. A matching state then succeeds on the same rows.
#[test]
fn row_entry_points_reject_mismatched_state_on_every_grouping_path() {
    let device = Default::default();
    let mixer = config().init::<TestBackend>(&device).unwrap();
    let hidden = waves([2, 3, 8], 2.0, 0.4);
    let mut state = Mamba2State::zeros(&config(), 2, DType::F32, &device).unwrap();
    state.recurrent = Tensor::zeros([2, 4, 2, 4], &device);
    let row = |tokens, mode| Mamba2Row { tokens, mode };
    let uniform = [row(3, Mamba2Mode::Recurrent), row(2, Mamba2Mode::Recurrent)];
    let mixed = [
        row(3, Mamba2Mode::ChunkedPrefill),
        row(2, Mamba2Mode::Recurrent),
    ];
    for rows in [uniform, mixed] {
        assert_shape(
            mixer.forward_rows(hidden.clone(), &rows, &state),
            "mamba2.state.recurrent",
        );
    }
    assert_shape(
        mixer.forward_rows_with_snapshots(hidden.clone(), &[3, 2], &state),
        "mamba2.state.recurrent",
    );
    let valid = Mamba2State::zeros(&config(), 2, DType::F32, &device).unwrap();
    let (output, _) = mixer
        .forward_rows(hidden, &mixed, &valid)
        .expect("matching state");
    assert_eq!(output.dims(), [2, 3, 8]);
}

/// Hidden states, token masks, and external state must match the mixer geometry.
#[test]
fn mismatched_inputs_and_state_are_shape_errors() {
    let device = Default::default();
    let mixer = config().init::<TestBackend>(&device).unwrap();
    let state = Mamba2State::zeros(&config(), 1, DType::F32, &device).unwrap();
    for shape in [[0, 2, 8], [1, 0, 8], [1, 2, 7]] {
        assert_shape(
            mixer.forward_with_state(
                Tensor::zeros(shape, &device),
                None,
                &state,
                Mamba2Mode::Recurrent,
            ),
            "mamba2.hidden",
        );
    }
    let hidden = waves([1, 3, 8], 0.5, 0.4);
    let short_mask = TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
        TensorData::new(vec![true; 2], [1, 2]),
        &device,
    ));
    assert_shape(
        mixer.forward_with_state(
            hidden.clone(),
            Some(&short_mask),
            &state,
            Mamba2Mode::ChunkedPrefill,
        ),
        "mamba2.token_mask",
    );
    let mut wrong_matrix = state.clone();
    wrong_matrix.recurrent = Tensor::zeros([1, 4, 2, 2], &device);
    assert_shape(
        mixer.forward_with_state(hidden.clone(), None, &wrong_matrix, Mamba2Mode::Recurrent),
        "mamba2.state.recurrent",
    );
    let mut short_history = state.clone();
    short_history.convolution = Tensor::zeros([1, 20, 1], &device);
    assert_shape(
        mixer.forward_with_state(hidden.clone(), None, &short_history, Mamba2Mode::Recurrent),
        "mamba2.state.convolution",
    );
    // The history must use the activation dtype, since it is concatenated with the inputs.
    let mut reduced_history = state;
    reduced_history.convolution = reduced_history.convolution.cast(DType::F16);
    assert_shape(
        mixer.forward_with_state(hidden, None, &reduced_history, Mamba2Mode::Recurrent),
        "mamba2.state.convolution",
    );
}

/// Trailing masked tokens are identity transitions and do not change earlier outputs.
///
/// A token mask zeroes the masked inputs, their convolution outputs, and their
/// discretization steps, so the selective state decays by `exp(0) = 1` and
/// receives no update. The terminal matrix state of a padded segment therefore
/// equals the state after only its valid prefix, and causality keeps the
/// valid outputs identical, in both execution modes. The convolution history
/// is the physical tail, which includes the zeroed padding.
#[test]
fn masked_suffix_tokens_leave_state_and_valid_outputs_unchanged() {
    let device = Default::default();
    let mixer = config().init::<TestBackend>(&device).unwrap();
    let mut state = Mamba2State::zeros(&config(), 1, DType::F32, &device).unwrap();
    state.recurrent = waves4([1, 4, 2, 3], 0.3, 0.2);
    let hidden = waves([1, 5, 8], 2.0, 0.5);
    let mask = TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
        TensorData::new(vec![true, true, true, false, false], [1, 5]),
        &device,
    ));
    for mode in [Mamba2Mode::Recurrent, Mamba2Mode::ChunkedPrefill] {
        let (padded, padded_state) = mixer
            .forward_with_state(hidden.clone(), Some(&mask), &state, mode)
            .unwrap();
        let (prefix, prefix_state) = mixer
            .forward_with_state(hidden.clone().narrow(1, 0, 3), None, &state, mode)
            .unwrap();
        padded
            .narrow(1, 0, 3)
            .into_data()
            .assert_approx_eq::<f32>(&prefix.into_data(), Tolerance::absolute(1e-6));
        padded_state.recurrent.into_data().assert_approx_eq::<f32>(
            &prefix_state.recurrent.into_data(),
            Tolerance::absolute(1e-6),
        );
        // The physical history ends in two zeroed padding columns.
        let history = padded_state
            .convolution
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        assert!(history.iter().all(|&value| value == 0.0), "{mode:?}");
    }
}

/// The scan kernels reject operands that the mixer would never pass them.
///
/// `recurrent_scan` checks its terminal state against the operand geometry,
/// which catches a mismatched initial state when no token updates it, and
/// `chunked_scan_with_budget` needs tokens, a non-zero chunk, and groups that
/// divide the heads.
#[test]
fn scan_kernels_reject_inconsistent_operands() {
    let device = Default::default();
    let empty = |shape: [usize; 4]| {
        Tensor::<TestBackend, 4>::from_data(TensorData::new(Vec::<f32>::new(), shape), &device)
    };
    let result = recurrent_scan(
        empty([1, 0, 2, 2]),
        empty([1, 0, 1, 3]),
        empty([1, 0, 1, 3]),
        Tensor::from_data(TensorData::new(Vec::<f32>::new(), [1, 0, 2]), &device),
        Tensor::from_floats([-1.0, -1.0], &device),
        Tensor::ones([2], &device),
        Tensor::zeros([1, 2, 2, 2], &device),
        false,
    );
    assert_shape(
        result.map(|(output, _, _)| output.dims()),
        "mamba2.scan_state",
    );

    let operands = |tokens: usize, groups: usize| {
        (
            waves4([1, tokens, 4, 2], 0.1, 0.3),
            waves4([1, tokens, groups, 3], 0.2, 0.3),
            waves4([1, tokens, groups, 3], 0.3, 0.3),
            waves([1, tokens, 4], 0.4, 0.1).abs(),
        )
    };
    for (tokens, groups, chunk) in [(0, 2, 2), (3, 2, 0), (3, 3, 2)] {
        let (x, b, c, dt) = if tokens == 0 {
            (
                empty([1, 0, 4, 2]),
                empty([1, 0, groups, 3]),
                empty([1, 0, groups, 3]),
                Tensor::from_data(TensorData::new(Vec::<f32>::new(), [1, 0, 4]), &device),
            )
        } else {
            operands(tokens, groups)
        };
        let result = chunked_scan_with_budget(
            x,
            b,
            c,
            dt,
            Tensor::from_floats([-1.0, -0.5, -0.25, -2.0], &device),
            Tensor::ones([4], &device),
            Tensor::zeros([1, 4, 2, 3], &device),
            chunk,
            SCAN_BLOCK_ELEMENTS,
        );
        assert_configuration(
            result.map(|(output, _)| output.dims()),
            "mamba2.chunk",
            "groups dividing the heads",
        );
    }
}
