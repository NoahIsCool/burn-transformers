//! Execution-context and module-composition tests with explicit numerical and masking references.

use super::*;
use burn::{
    backend::Flex,
    tensor::{Bool, Int, Tensor, TensorData},
};

/// Deterministic F32 CPU backend used by execution and composition fixtures.
type B = Flex<f32, i32>;

/// Build one three-token execution context with positions zero through two and no padding.
fn context() -> ExecutionContext<B> {
    ExecutionContext::new(
        Tensor::<B, 2, Int>::from_data([[0, 1, 2]], &Default::default()),
        None,
    )
    .unwrap()
}

/// Broadcast attention permissions are intersected with valid query/key positions before execution.
#[test]
fn policy_broadcasts_allowed_links_and_intersects_padding() {
    let device = Default::default();
    // Dynamic Boolean import intentionally exercises backend-native conversion.
    let allowed = Tensor::<B, 4, Bool>::from_data(
        TensorData::new(vec![true, false, true], [1, 1, 1, 3]),
        &device,
    );
    let padding = Tensor::<B, 4, Int>::from_data(
        TensorData::new(vec![0, 0, 1, 0, 1, 0], [1, 1, 2, 3]),
        &device,
    )
    .bool();
    let policy = AttentionPolicy::causal().with_allowed(allowed);
    let mask = policy
        .exclusion_mask(Some(padding), [1, 2, 2, 3], &device)
        .unwrap()
        .unwrap();
    assert_eq!(mask.dims(), [1, 2, 2, 3]);
    assert_eq!(
        mask.int().into_data().to_vec::<i32>().unwrap(),
        vec![0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0]
    );
    assert!(policy.causal);
}

/// Attention bias broadcasts over supported dimensions and rejects incompatible policy extents.
#[test]
fn policy_bias_broadcast_and_invalid_extents_are_explicit() {
    let device = Default::default();
    let policy =
        AttentionPolicy::<B>::bidirectional().with_bias(Tensor::full([1, 1, 1, 1], 0.5, &device));
    let bias = policy
        .expanded_bias([2, 3, 4, 5], &device)
        .unwrap()
        .unwrap();
    assert_eq!(bias.dims(), [2, 3, 4, 5]);
    assert!(
        bias.into_data()
            .to_vec::<f32>()
            .unwrap()
            .iter()
            .all(|x| *x == 0.5)
    );
    let invalid =
        AttentionPolicy::<B>::custom(Tensor::<B, 4, Int>::zeros([1, 2, 1, 1], &device).bool());
    assert!(
        invalid
            .exclusion_mask(None, [1, 3, 4, 5], &device)
            .unwrap_err()
            .to_string()
            .contains("attention.allowed")
    );
    assert!(policy.validate([0, 1, 1, 1], &device).is_err());
}

/// A layer-specific policy override leaves other layers and cross-attention configuration unchanged.
#[test]
fn layer_overrides_do_not_change_other_layers_or_cross_attention() {
    let layer = Layer::decoder(1);
    let context = context()
        .with_attention(AttentionPolicy::bidirectional())
        .with_layer_attention(layer, AttentionPolicy::causal())
        .with_cross_attention(AttentionPolicy::bidirectional())
        .with_layer_cross_attention(layer, AttentionPolicy::causal());
    assert!(context.attention_for(layer, false).causal);
    assert!(!context.attention_for(Layer::decoder(0), true).causal);
    assert!(context.cross_attention_for(layer).causal);
    assert!(!context.cross_attention_for(Layer::decoder(0)).causal);
    let context = context.with_layer_attention(layer, AttentionPolicy::bidirectional());
    assert_eq!(context.layer_attention.len(), 1);
    assert!(!context.attention_for(layer, true).causal);
}

/// Staged execution rejects skipped or repeated layers and ranges beyond final normalization.
#[test]
fn state_rejects_skipped_repeated_and_normalized_layer_ranges() {
    let mut state =
        HiddenState::<B>::new(Tensor::ones([1, 3, 4], &Default::default()), context()).unwrap();
    assert!(state.validate_range(0..0, 3).is_ok());
    assert!(state.validate_range(0..2, 3).is_ok());
    state.next_layer = 2;
    assert!(state.validate_range(2..3, 3).is_ok());
    for range in [0..2, 1..3, 2..4] {
        assert!(state.validate_range(range, 3).is_err());
    }
    state.normalized = true;
    assert!(state.validate_range(2..3, 3).is_err());
}

/// Resuming an edited state validates its residual width, position geometry, and token mask.
#[test]
fn state_validation_rejects_mutated_width_positions_and_mask() {
    let device = Default::default();
    let mut state = HiddenState::<B>::new(Tensor::ones([1, 3, 4], &device), context()).unwrap();
    assert!(state.validate(5).is_err());
    state.context.positions = Tensor::from_data([[0, 1, 3]], &device);
    assert!(
        state
            .validate(4)
            .unwrap_err()
            .to_string()
            .contains("positions")
    );
    state.context.positions = Tensor::from_data([[0, 1, 2]], &device);
    state.context.token_mask = Some(TokenMask::new(
        Tensor::<B, 2, Int>::ones([1, 2], &device).bool(),
    ));
    assert!(
        state
            .validate(4)
            .unwrap_err()
            .to_string()
            .contains("token_mask")
    );
    assert!(ExecutionContext::<B>::new(Tensor::from_data([[-1, 0]], &device), None).is_err());
}

/// Exported encoder state retains source positions, validity, and its normalization boundary.
#[test]
fn encoder_export_preserves_positions_validity_and_normalization_boundary() {
    let device = Default::default();
    let context = ExecutionContext::new(
        Tensor::<B, 2, Int>::from_data([[2, 3, 4]], &device),
        Some(TokenMask::new(
            Tensor::<B, 2, Int>::from_data([[1, 0, 1]], &device).bool(),
        )),
    )
    .unwrap();
    let mut state = HiddenState::<B>::new(Tensor::ones([1, 3, 4], &device), context).unwrap();
    assert!(state.clone().into_encoder_context().is_err());
    state.normalized = true;
    let output = state.clone().into_encoder_output().unwrap();
    let source = state.into_encoder_context().unwrap();
    assert_eq!(output.hidden_state.into_data(), source.hidden.into_data());
    assert_eq!(
        source.positions.into_data().to_vec::<i32>().unwrap(),
        vec![2, 3, 4]
    );
    assert_eq!(
        source
            .token_mask
            .unwrap()
            .into_tensor()
            .int()
            .into_data()
            .to_vec::<i32>()
            .unwrap(),
        vec![1, 0, 1]
    );
}

/// Cross-attention accepts independent source lengths while rejecting negative source positions.
#[test]
fn source_positions_are_independent_of_target_extent_but_never_negative() {
    let device = Default::default();
    let mut context = context();
    context.encoder = Some(EncoderContext {
        hidden: Tensor::ones([1, 2, 4], &device),
        positions: Tensor::from_ints([[30, 31]], &device),
        token_mask: None,
    });
    assert!(context.validate([1, 3], &device).is_ok());
    context.encoder.as_mut().unwrap().positions = Tensor::from_ints([[-1, 0]], &device);
    assert!(
        context
            .validate([1, 3], &device)
            .unwrap_err()
            .to_string()
            .contains("execution.encoder.positions")
    );
}

/// Fixed logical [input, output] matrices make composition expectations independent of initialization.
fn matrix(input: usize, output: usize, values: &[f32]) -> burn::nn::Linear<B> {
    let device = Default::default();
    let mut linear = burn::nn::LinearConfig::new(input, output)
        .with_bias(false)
        .init(&device);
    linear.weight = burn::module::Param::from_tensor(Tensor::from_data(
        TensorData::new(values.to_vec(), [input, output]),
        &device,
    ));
    linear
}

/// Compare an actual public execution result with a scalar mathematical oracle.
fn composition_close(tensor: Tensor<B, 3>, expected: &[f32]) {
    let actual = tensor.into_data().to_vec::<f32>().unwrap();
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() < 3e-6,
            "element {index}: {actual} != {expected}"
        );
    }
}

/// Each supported normalization placement agrees with an explicit scalar residual-block reference.
#[test]
fn composition_normalization_placements_follow_scalar_residual_equations() {
    use crate::architecture::{NormPlacement, NormalizationSpec};
    let device = Default::default();
    let input = Tensor::<B, 3>::from_floats([[[1.0, 3.0], [-2.0, 1.0], [4.0, -3.0]]], &device);
    for placement in [
        NormPlacement::Pre,
        NormPlacement::Post,
        NormPlacement::Sandwich,
    ] {
        let block = ResidualBlock::new(
            2,
            matrix(2, 2, &[0.25, 0.0, 0.0, 0.25]),
            matrix(2, 2, &[0.5, 0.0, 0.0, 0.5]),
            NormalizationSpec::Rms {
                epsilon: 0.25,
                zero_centered: false,
            },
            placement,
            &device,
        )
        .unwrap();
        let normalize = |x: [f64; 2]| {
            let scale = ((x[0] * x[0] + x[1] * x[1]) / 2.0 + 0.25).sqrt();
            [x[0] / scale, x[1] / scale]
        };
        let mut expected = Vec::new();
        for mut x in [[1.0, 3.0], [-2.0, 1.0], [4.0, -3.0]] {
            for factor in [0.25, 0.5] {
                x = match placement {
                    NormPlacement::Pre => {
                        let n = normalize(x);
                        [x[0] + factor * n[0], x[1] + factor * n[1]]
                    }
                    NormPlacement::Post => {
                        normalize([x[0] * (1.0 + factor), x[1] * (1.0 + factor)])
                    }
                    NormPlacement::Sandwich => {
                        let n = normalize(x);
                        let n = normalize([factor * n[0], factor * n[1]]);
                        [x[0] + n[0], x[1] + n[1]]
                    }
                };
            }
            expected.extend(x.map(|v| v as f32));
        }
        composition_close(
            block
                .run(input.clone(), &context(), Layer::decoder(0))
                .unwrap(),
            &expected,
        );
        let mut hooks = Hooks::new();
        let output = hooks.capture(Layer::decoder(0).residual_post());
        block
            .run_with_hooks(input.clone(), &context(), Layer::decoder(0), &mut hooks)
            .unwrap();
        composition_close(hooks.get(&output).unwrap(), &expected);
    }
    let norm = NormalizationSpec::Rms {
        epsilon: 0.25,
        zero_centered: false,
    };
    assert!(
        ResidualBlock::new(
            0,
            matrix(2, 2, &[1.0; 4]),
            matrix(2, 2, &[1.0; 4]),
            norm,
            NormPlacement::Pre,
            &device
        )
        .is_err()
    );
    let invalid = NormalizationSpec::Rms {
        epsilon: f64::NAN,
        zero_centered: false,
    };
    assert!(
        ResidualBlock::new(
            2,
            matrix(2, 2, &[1.0; 4]),
            matrix(2, 2, &[1.0; 4]),
            invalid,
            NormPlacement::Pre,
            &device
        )
        .is_err()
    );
    assert!(
        ResidualBlock::without_normalization(0, matrix(2, 2, &[1.0; 4]), matrix(2, 2, &[1.0; 4]))
            .is_err()
    );
}

/// Dense and gated branches expose activation values at the configured nonlinear computation point.
#[test]
fn composition_dense_and_gated_branches_expose_the_actual_nonlinearity() {
    use crate::{
        architecture::Activation,
        feed_forward::{DenseFeedForward, GatedFeedForward},
    };
    let device = Default::default();
    let input = Tensor::<B, 3>::from_floats([[[-2.0, 3.0], [4.0, -1.0], [0.0, 2.0]]], &device);
    let identity = || matrix(2, 2, &[1.0, 0.0, 0.0, 1.0]);
    let dense = DenseFeedForward {
        up_proj: identity(),
        down_proj: identity(),
        activation: Activation::Relu,
    };
    let block = ResidualBlock::without_normalization(2, matrix(2, 2, &[0.0; 4]), dense).unwrap();
    composition_close(
        block
            .run(input.clone(), &context(), Layer::decoder(0))
            .unwrap(),
        &[-2.0, 6.0, 8.0, -1.0, 0.0, 4.0],
    );
    let gated = GatedFeedForward {
        gate_proj: identity(),
        up_proj: identity(),
        down_proj: identity(),
        activation: Activation::Relu,
    };
    let block = ResidualBlock::without_normalization(2, matrix(2, 2, &[0.0; 4]), gated).unwrap();
    composition_close(
        block
            .run(input.clone(), &context(), Layer::decoder(0))
            .unwrap(),
        &[-2.0, 12.0, 20.0, -1.0, 0.0, 6.0],
    );
    let mut hooks = Hooks::new();
    let before = hooks.capture(Layer::decoder(0).mlp_activation());
    hooks.edit(Layer::decoder(0).mlp_activation(), |activation| {
        activation.zeros_like()
    });
    let output = block
        .run_with_hooks(input.clone(), &context(), Layer::decoder(0), &mut hooks)
        .unwrap();
    composition_close(hooks.get(&before).unwrap(), &[0.0, 3.0, 4.0, 0.0, 0.0, 2.0]);
    assert_eq!(output.into_data(), input.into_data());
}

/// Composed grouped-query attention respects causal policy, bias, padding, and empty allowed rows.
#[test]
fn composition_gqa_obeys_causality_bias_padding_and_fully_masked_rows() {
    use crate::attention::{AttentionPrecisionMode, GroupedQueryAttentionConfig};
    let device = Default::default();
    let input = Tensor::<B, 3>::from_floats([[[2.0, 1.0], [6.0, 3.0], [10.0, 5.0]]], &device);
    for mode in [
        AttentionPrecisionMode::BackendDefault,
        AttentionPrecisionMode::EagerFp32Softmax,
    ] {
        let mut attention = GroupedQueryAttentionConfig::new(2, 2, 1, 1)
            .with_precision_mode(mode)
            .init::<B>(&device)
            .unwrap();
        attention.q_proj = matrix(2, 2, &[0.0; 4]);
        attention.k_proj = matrix(2, 1, &[0.0; 2]);
        attention.v_proj = matrix(2, 1, &[1.0, 0.0]);
        attention.o_proj = matrix(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        let adapter = WidthAdapter::new(attention, 2, 2, 2, &device).unwrap();
        // Equal zero scores give prefix means; both query heads share the same values.
        composition_close(
            adapter
                .run(input.clone(), &context(), Layer::decoder(0))
                .unwrap(),
            &[2.0, 2.0, 4.0, 4.0, 6.0, 6.0],
        );
        composition_close(
            adapter
                .run(input.clone(), &context(), Layer::encoder(0))
                .unwrap(),
            &[6.0; 6],
        );
        let allowed =
            Tensor::<B, 4, Int>::from_data([[[[1, 1, 1], [0, 0, 0], [1, 1, 1]]]], &device).bool();
        let mut masked = context().with_attention(
            AttentionPolicy::custom(allowed).with_bias(Tensor::zeros([1, 1, 1, 1], &device)),
        );
        masked.token_mask = Some(TokenMask::new(
            Tensor::<B, 2, Int>::from_data([[1, 1, 0]], &device).bool(),
        ));
        let mut hooks = Hooks::new();
        let pattern = (mode == AttentionPrecisionMode::EagerFp32Softmax)
            .then(|| hooks.capture(Layer::decoder(0).self_attention().pattern()));
        composition_close(
            adapter
                .run_with_hooks(input.clone(), &masked, Layer::decoder(0), &mut hooks)
                .unwrap(),
            &[4.0, 4.0, 0.0, 0.0, 0.0, 0.0],
        );
        if let Some(pattern) = pattern {
            let values = hooks
                .get(&pattern)
                .unwrap()
                .into_data()
                .to_vec::<f32>()
                .unwrap();
            assert_eq!(
                values,
                vec![
                    0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0
                ]
            );
        }
        // Bias log(3) triples the first key's weight: (3*2 + 6 + 10) / 5 = 4.4.
        let biased = context().with_attention(
            AttentionPolicy::bidirectional()
                .with_bias(Tensor::from_data([[[[3.0_f32.ln(), 0.0, 0.0]]]], &device)),
        );
        composition_close(
            adapter
                .run(input.clone(), &biased, Layer::decoder(0))
                .unwrap(),
            &[4.4; 6],
        );
        let invalid = context().with_attention(
            AttentionPolicy::bidirectional().with_bias(Tensor::zeros([1, 3, 1, 1], &device)),
        );
        assert!(
            adapter
                .run(input.clone(), &invalid, Layer::decoder(0))
                .unwrap_err()
                .to_string()
                .contains("attention.bias")
        );
        let mismatched = context().with_attention(AttentionPolicy::bidirectional().with_bias(
            Tensor::<B, 4>::zeros([1, 1, 1, 1], &device).cast(burn::tensor::DType::F64),
        ));
        assert!(
            adapter
                .run(input.clone(), &mismatched, Layer::decoder(0))
                .unwrap_err()
                .to_string()
                .contains("residual_block.attention.bias")
        );
    }
}

/// A registered consumer module intentionally violating one output contract.
#[derive(burn::module::Module, Debug)]
struct BadComposition<B: burn::tensor::backend::Backend> {
    /// Registered parameter module retained so the fixture obeys Burn's module contract.
    parameter: burn::nn::Linear<B>,
    /// Choose a dtype violation instead of truncating the token axis.
    #[module(skip)]
    wrong_dtype: bool,
}

impl<B: burn::tensor::backend::Backend> HiddenTransform<B> for BadComposition<B> {
    /// Deliberately return an incompatible dtype or token extent to test composition validation.
    fn forward(
        &self,
        hidden: Tensor<B, 3>,
        _: &ExecutionContext<B>,
        _: Layer,
        _: &mut Hooks<'_, B>,
    ) -> Result<Tensor<B, 3>> {
        Ok(if self.wrong_dtype {
            hidden.cast(burn::tensor::DType::F64)
        } else {
            hidden.narrow(1, 0, 1)
        })
    }
}

/// Width adapters match hand-computed matrix products and reject invalid custom module outputs.
#[test]
fn composition_width_adapters_have_explicit_matrix_products_and_reject_bad_modules() {
    let device = Default::default();
    let input = Tensor::<B, 3>::from_floats([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]], &device);
    // [a,b] -> [a,b,a+b] -> [2a,3b,4(a+b)] -> [6a+4b,4a+7b].
    let mut adapter = WidthAdapter::new(
        matrix(3, 3, &[2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0]),
        2,
        3,
        2,
        &device,
    )
    .unwrap();
    adapter.input_projection = Some(matrix(2, 3, &[1.0, 0.0, 1.0, 0.0, 1.0, 1.0]));
    adapter.output_projection = Some(matrix(3, 2, &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0]));
    composition_close(
        adapter
            .run(input.clone(), &context(), Layer::decoder(0))
            .unwrap(),
        &[14.0, 18.0, 34.0, 40.0, 54.0, 62.0],
    );
    for dimensions in [[0, 2, 2], [2, 0, 2], [2, 2, 0]] {
        assert!(
            WidthAdapter::new(
                matrix(2, 2, &[1.0; 4]),
                dimensions[0],
                dimensions[1],
                dimensions[2],
                &device
            )
            .is_err()
        );
    }
    for wrong_dtype in [false, true] {
        let bad = || BadComposition {
            parameter: matrix(2, 2, &[1.0; 4]),
            wrong_dtype,
        };
        let adapter = WidthAdapter::new(bad(), 2, 2, 2, &device).unwrap();
        assert!(
            adapter
                .run(input.clone(), &context(), Layer::decoder(0))
                .unwrap_err()
                .to_string()
                .contains("width_adapter.transform")
        );
        let block =
            ResidualBlock::without_normalization(2, bad(), matrix(2, 2, &[0.0; 4])).unwrap();
        assert!(
            block
                .run(input.clone(), &context(), Layer::decoder(0))
                .unwrap_err()
                .to_string()
                .contains("residual_block.mixer")
        );
        let block =
            ResidualBlock::without_normalization(2, matrix(2, 2, &[0.0; 4]), bad()).unwrap();
        assert!(
            block
                .run(input.clone(), &context(), Layer::decoder(0))
                .unwrap_err()
                .to_string()
                .contains("residual_block.feed_forward")
        );
    }
    let block =
        ResidualBlock::without_normalization(3, matrix(2, 2, &[0.0; 4]), matrix(2, 2, &[0.0; 4]))
            .unwrap();
    assert!(
        block
            .run(input, &context(), Layer::decoder(0))
            .unwrap_err()
            .to_string()
            .contains("residual_block.input")
    );
}

/// Composition preflight rejects empty extents and incompatible source, position, or padding metadata.
#[test]
fn composition_context_rejects_invalid_extent_source_batch_positions_and_padding() {
    let device = Default::default();
    let mut changed = context();
    changed.evaluated_context = 0;
    assert!(
        changed
            .validate([1, 3], &device)
            .unwrap_err()
            .to_string()
            .contains("evaluated_context")
    );
    assert!(context().validate([2, 3], &device).is_err());
    assert!(context().validate([1, 0], &device).is_err());
    let source = || EncoderContext {
        hidden: Tensor::<B, 3>::ones([1, 2, 4], &device),
        positions: Tensor::from_ints([[0, 1]], &device),
        token_mask: None,
    };
    let mut changed = context();
    changed.encoder = Some(source());
    changed.encoder.as_mut().unwrap().hidden = Tensor::ones([2, 2, 4], &device);
    assert!(
        changed
            .validate([1, 3], &device)
            .unwrap_err()
            .to_string()
            .contains("execution.encoder")
    );
    changed.encoder = Some(source());
    changed.encoder.as_mut().unwrap().positions = Tensor::from_ints([[0, 1, 2]], &device);
    assert!(
        changed
            .validate([1, 3], &device)
            .unwrap_err()
            .to_string()
            .contains("execution.encoder")
    );
    changed.encoder = Some(source());
    changed.encoder.as_mut().unwrap().token_mask = Some(TokenMask::new(
        Tensor::<B, 2, Int>::ones([1, 3], &device).bool(),
    ));
    assert!(
        changed
            .validate([1, 3], &device)
            .unwrap_err()
            .to_string()
            .contains("execution.encoder.token_mask")
    );
    changed.encoder = Some(source());
    changed.encoder.as_mut().unwrap().token_mask = Some(TokenMask::new(
        Tensor::<B, 2, Int>::from_ints([[1, 0]], &device).bool(),
    ));
    assert!(changed.validate([1, 3], &device).is_ok());
}

// Validation cost: position values once per pass, layouts once per block.

/// Contiguous contexts equal explicit `0..T` positions and know their extent without a readback.
#[test]
fn contiguous_context_matches_explicit_positions_and_rejects_empty_extents() {
    let device = Default::default();
    let contiguous = ExecutionContext::<B>::contiguous(2, 3, &device).unwrap();
    let explicit =
        ExecutionContext::<B>::new(Tensor::from_ints([[0, 1, 2], [0, 1, 2]], &device), None)
            .unwrap();
    assert_eq!(
        contiguous.positions.to_data().to_vec::<i32>().unwrap(),
        explicit.positions.to_data().to_vec::<i32>().unwrap()
    );
    assert_eq!(contiguous.evaluated_context, explicit.evaluated_context);
    assert!(contiguous.token_mask.is_none());
    contiguous.validate([2, 3], &device).unwrap();
    for (batch, tokens) in [(0, 3), (2, 0)] {
        let error = ExecutionContext::<B>::contiguous(batch, tokens, &device).unwrap_err();
        assert!(error.to_string().contains("execution.positions"), "{error}");
    }
}

/// Layout checks never read position values; full validation catches edited public fields.
#[test]
fn layout_validation_is_value_free_while_full_validation_rejects_edited_positions() {
    let device = Default::default();
    let mut edited = context();
    edited.positions = Tensor::from_ints([[0, -1, 2]], &device);
    edited.validate_layout([1, 3], &device).unwrap();
    let error = edited.validate([1, 3], &device).unwrap_err().to_string();
    assert!(
        error.contains("execution.positions") && error.contains("-1..=2"),
        "{error}"
    );

    let mut shrunk = context();
    shrunk.evaluated_context = 2;
    shrunk.validate_layout([1, 3], &device).unwrap();
    let error = shrunk.validate([1, 3], &device).unwrap_err().to_string();
    assert!(error.contains("0..2"), "{error}");

    // Public composition entry points validate values once before any block runs.
    let input = Tensor::<B, 3>::ones([1, 3, 2], &device);
    let block =
        ResidualBlock::without_normalization(2, matrix(2, 2, &[0.0; 4]), matrix(2, 2, &[0.0; 4]))
            .unwrap();
    let error = block
        .run(input.clone(), &edited, Layer::decoder(0))
        .unwrap_err();
    assert!(error.to_string().contains("execution.positions"), "{error}");
    let adapter = WidthAdapter::new(matrix(2, 2, &[1.0, 0.0, 0.0, 1.0]), 2, 2, 2, &device).unwrap();
    let error = adapter.run(input, &shrunk, Layer::decoder(0)).unwrap_err();
    assert!(error.to_string().contains("execution.positions"), "{error}");
}

/// A run rejected by context validation clears the previous run's captures.
///
/// Validation happens inside the hook scope, so `get` cannot return
/// activations from an earlier successful run after a failed one.
#[test]
fn rejected_contexts_fail_inside_the_hook_scope_and_clear_old_captures() {
    let device = Default::default();
    let input = Tensor::<B, 3>::ones([1, 3, 2], &device);
    let identity = || matrix(2, 2, &[1.0, 0.0, 0.0, 1.0]);
    let mut edited = context();
    edited.positions = Tensor::from_ints([[0, -1, 2]], &device);

    let block = ResidualBlock::without_normalization(2, identity(), identity()).unwrap();
    let mut hooks = Hooks::new();
    let output = hooks.capture(Layer::decoder(0).residual_post());
    block
        .run_with_hooks(input.clone(), &context(), Layer::decoder(0), &mut hooks)
        .unwrap();
    assert!(hooks.get(&output).is_ok());
    let error = block
        .run_with_hooks(input.clone(), &edited, Layer::decoder(0), &mut hooks)
        .unwrap_err();
    assert!(error.to_string().contains("execution.positions"), "{error}");
    assert!(
        hooks.get(&output).is_err(),
        "stale capture survived a failed run"
    );
    // A width error is also raised inside the scope.
    block
        .run_with_hooks(input.clone(), &context(), Layer::decoder(0), &mut hooks)
        .unwrap();
    let narrow = Tensor::<B, 3>::ones([1, 3, 1], &device);
    assert!(
        block
            .run_with_hooks(narrow, &context(), Layer::decoder(0), &mut hooks)
            .is_err()
    );
    assert!(hooks.get(&output).is_err());

    // The adapter exposes its inner block's points.
    let adapter = WidthAdapter::new(block, 2, 2, 2, &device).unwrap();
    let mut hooks = Hooks::new();
    let output = hooks.capture(Layer::decoder(0).residual_post());
    adapter
        .run_with_hooks(input.clone(), &context(), Layer::decoder(0), &mut hooks)
        .unwrap();
    assert!(hooks.get(&output).is_ok());
    assert!(
        adapter
            .run_with_hooks(input, &edited, Layer::decoder(0), &mut hooks)
            .is_err()
    );
    assert!(
        hooks.get(&output).is_err(),
        "stale capture survived a failed run"
    );
}

/// Attaching an encoder checks its layout and nonnegative source positions together.
#[test]
fn with_encoder_validates_source_layout_and_positions() {
    let device = Default::default();
    let source = |positions: [[i32; 2]; 1]| EncoderContext {
        hidden: Tensor::<B, 3>::ones([1, 2, 4], &device),
        positions: Tensor::from_ints(positions, &device),
        token_mask: None,
    };
    let attached = context().with_encoder(source([[5, 9]])).unwrap();
    assert_eq!(attached.encoder.unwrap().positions.dims(), [1, 2]);
    let error = context().with_encoder(source([[-3, 0]])).unwrap_err();
    assert!(
        error.to_string().contains("execution.encoder.positions"),
        "{error}"
    );
    let mut wide = source([[0, 1]]);
    wide.hidden = Tensor::ones([2, 2, 4], &device);
    let error = context().with_encoder(wide).unwrap_err();
    assert!(error.to_string().contains("execution.encoder"), "{error}");
}

/// One readback returns every scalar in order, widening I32 storage to `i64`.
#[test]
fn batched_integer_readback_preserves_order_and_handles_empty_input() {
    let device = Default::default();
    let values = Tensor::<B, 1, Int>::from_ints([4, -7, 2], &device);
    let read = read_int_scalars(
        vec![values.clone().max(), values.clone().min(), values.sum()],
        "fixture scalars",
    )
    .unwrap();
    assert_eq!(read, [4, -7, -1]);
    assert!(
        read_int_scalars::<B>(Vec::new(), "nothing")
            .unwrap()
            .is_empty()
    );
}

/// Reports per-call validation cost; run with `--ignored --nocapture`.
///
/// Compares the sync-free layout check, the one-readback value check, and the
/// previous scheme of two separate scalar readbacks. On Flex a readback is a
/// cheap host copy; on Metal/CUDA every readback also drains the device queue.
#[test]
#[ignore = "timing report, not a correctness check"]
fn validation_cost_report() {
    use std::time::Instant;
    let device = Default::default();
    let context = ExecutionContext::<B>::contiguous(8, 512, &device).unwrap();
    let iterations = 2_000;
    let time = |label: &str, run: &mut dyn FnMut()| {
        let start = Instant::now();
        for _ in 0..iterations {
            run();
        }
        let nanoseconds = start.elapsed().as_nanos() / iterations as u128;
        println!("{label}: {nanoseconds} ns/call");
    };
    time("validate_layout (0 readbacks)", &mut || {
        context.validate_layout([8, 512], &device).unwrap()
    });
    time("validate (1 batched readback)", &mut || {
        context.validate([8, 512], &device).unwrap()
    });
    time("previous scheme (2 scalar readbacks)", &mut || {
        let low = context.positions.clone().min().into_scalar();
        let high = context.positions.clone().max().into_scalar();
        assert!(low <= high);
    });
}

/// Explicit connectivity without padding excludes exactly the disallowed links.
#[test]
fn allowed_links_without_padding_exclude_only_disallowed_pairs() {
    let device = Default::default();
    // Query 0 may see key 0 only; query 1 may see both keys.
    let allowed = Tensor::<B, 4, Int>::from_ints([[[[1, 0], [1, 1]]]], &device).bool();
    let mask = AttentionPolicy::custom(allowed)
        .exclusion_mask(None, [2, 1, 2, 2], &device)
        .unwrap()
        .expect("connectivity always produces a mask");
    assert_eq!(mask.dims(), [2, 1, 2, 2]);
    assert_eq!(
        mask.int().into_data().to_vec::<i32>().unwrap(),
        [0, 1, 0, 0, 0, 1, 0, 0]
    );
    assert!(
        AttentionPolicy::<B>::bidirectional()
            .exclusion_mask(None, [1, 1, 2, 2], &device)
            .unwrap()
            .is_none()
    );
}

/// Contexts and representations reject empty axes with the field a caller must repair.
#[test]
fn empty_positions_sources_and_representations_are_rejected() {
    let device = Default::default();
    let error =
        ExecutionContext::<B>::new(Tensor::<B, 2, Int>::zeros([1, 0], &device), None).unwrap_err();
    assert_eq!(
        error.to_string(),
        "invalid configuration at execution.positions: positions must be nonempty"
    );
    for [tokens, width] in [[0, 4], [2, 0]] {
        let mut changed = context();
        changed.encoder = Some(EncoderContext {
            hidden: Tensor::<B, 3>::ones([1, tokens, width], &device),
            positions: Tensor::<B, 2, Int>::zeros([1, tokens], &device),
            token_mask: None,
        });
        let error = changed.validate_layout([1, 3], &device).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("matching nonempty batch/token axes"),
            "{tokens}x{width}: {error}"
        );
    }
    let state = HiddenState::<B>::new(Tensor::ones([1, 3, 4], &device), context()).unwrap();
    assert!(
        state
            .validate(0)
            .unwrap_err()
            .to_string()
            .contains("execution.hidden")
    );
    for shape in [[0, 3, 4], [1, 0, 4]] {
        let mut empty = state.clone();
        empty.hidden = Tensor::ones(shape, &device);
        assert!(
            empty
                .validate(4)
                .unwrap_err()
                .to_string()
                .contains("execution.hidden"),
            "{shape:?}"
        );
    }
}

/// Encoder output, like encoder context, is exported only after final normalization.
#[test]
fn encoder_output_requires_a_normalized_source_stack() {
    let device = Default::default();
    let state = HiddenState::<B>::new(Tensor::ones([1, 3, 4], &device), context()).unwrap();
    assert_eq!(
        state.into_encoder_output().unwrap_err().to_string(),
        "invalid configuration at execution.encoder: normalize the source stack before exporting encoder output"
    );
}

/// Residual blocks reject empty batch or token axes before running any branch.
#[test]
fn residual_blocks_reject_empty_batch_and_token_axes() {
    let device = Default::default();
    let block =
        ResidualBlock::without_normalization(2, matrix(2, 2, &[0.0; 4]), matrix(2, 2, &[0.0; 4]))
            .unwrap();
    for shape in [[0, 3, 2], [1, 0, 2]] {
        let mut hooks = Hooks::new();
        let error = block
            .forward(
                Tensor::<B, 3>::ones(shape, &device),
                &context(),
                Layer::decoder(0),
                &mut hooks,
            )
            .unwrap_err();
        assert!(
            error.to_string().contains("residual_block.input"),
            "{shape:?}: {error}"
        );
    }
}

/// A failing edit before the feed-forward nonlinearity aborts dense and gated branches.
#[test]
fn pre_activation_edit_failures_abort_dense_and_gated_feed_forwards() {
    use crate::{
        architecture::Activation,
        feed_forward::{DenseFeedForward, GatedFeedForward},
    };
    let device = Default::default();
    let input = Tensor::<B, 3>::ones([1, 3, 2], &device);
    let identity = || matrix(2, 2, &[1.0, 0.0, 0.0, 1.0]);
    let failing_hooks = || {
        let mut hooks = Hooks::new();
        hooks.try_edit(Layer::decoder(0).mlp_pre_activation(), |_| {
            Err(Error::configuration("test.edit", "rejected pre-activation"))
        });
        hooks
    };
    let dense = ResidualBlock::without_normalization(
        2,
        matrix(2, 2, &[0.0; 4]),
        DenseFeedForward {
            up_proj: identity(),
            down_proj: identity(),
            activation: Activation::Relu,
        },
    )
    .unwrap();
    let gated = ResidualBlock::without_normalization(
        2,
        matrix(2, 2, &[0.0; 4]),
        GatedFeedForward {
            gate_proj: identity(),
            up_proj: identity(),
            down_proj: identity(),
            activation: Activation::Relu,
        },
    )
    .unwrap();
    let errors = [
        dense
            .run_with_hooks(
                input.clone(),
                &context(),
                Layer::decoder(0),
                &mut failing_hooks(),
            )
            .unwrap_err(),
        gated
            .run_with_hooks(input, &context(), Layer::decoder(0), &mut failing_hooks())
            .unwrap_err(),
    ];
    for error in errors {
        let message = error.to_string();
        assert!(
            message.contains("edit callback returned an error"),
            "{message}"
        );
        assert!(matches!(error, Error::Intervention { .. }), "{error:?}");
    }
}
