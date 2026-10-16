//! Public state boundaries preserve the graph while allowing deliberate edits.

use burn::{
    backend::Flex,
    tensor::{Bool, DType, Int, Tensor, Tolerance},
};

use super::*;
use crate::architecture::TokenMask;

/// CPU Flex backend for reproducible execution-boundary comparisons.
type B = Flex<f32, i32>;

/// Builds a tiny tied-embedding hybrid with one DeltaNet block and one full-attention block.
fn config() -> Qwen3_5TextConfig {
    Qwen3_5TextConfig {
        vocab_size: 16,
        hidden_size: 8,
        intermediate_size: 12,
        num_hidden_layers: 2,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: 4,
        max_position_embeddings: 32,
        layer_types: vec![
            Qwen3_5LayerType::LinearAttention,
            Qwen3_5LayerType::FullAttention,
        ],
        linear_key_head_dim: 4,
        linear_value_head_dim: 4,
        linear_num_key_heads: 1,
        linear_num_value_heads: 2,
        linear_conv_kernel_dim: 3,
        partial_rotary_factor: Some(1.0),
        rope_theta: Some(10_000.0),
        tie_word_embeddings: true,
        mtp_num_hidden_layers: 0,
        ..Default::default()
    }
}

/// Builds one three-token prefix with a variable last token and implicit positions.
fn input(last: i32) -> CausalLmInput<B> {
    CausalLmInput {
        token_ids: Tensor::<B, 2, Int>::from_ints([[1, 2, last]], &Default::default()),
        token_mask: None,
        position_ids: None,
    }
}

/// Resuming consecutive layer ranges preserves masked tokens, explicit positions, and final states.
#[test]
fn split_and_resume_preserve_explicit_positions_padding_and_full_backbone_values() {
    let device = Default::default();
    let model = Qwen3_5TextModel::<B>::init(config(), &device).unwrap();
    let mut input = input(3);
    input.position_ids = Some(Tensor::<B, 2, Int>::from_ints([[0, 4, 5]], &device));
    input.token_mask = Some(TokenMask::new(Tensor::<B, 2, Bool>::from_bool(
        [[false, true, true]].into(),
        &device,
    )));
    let (expected, _) = model.forward(&input, false).unwrap();
    let state = model.embed(&input).unwrap();
    assert_eq!(state.context.evaluated_context, 6);
    let state = model.run_layers(state, 0..1).unwrap();
    assert_eq!(state.next_layer, 1);
    assert!(!state.normalized);
    let state = model.run_layers(state, 1..2).unwrap();
    let state = model.normalize(state).unwrap();
    assert!(state.normalized);
    assert_eq!(state.hidden.to_data(), expected.to_data());
    assert_eq!(
        state.context.positions.to_data(),
        input.position_ids.unwrap().to_data()
    );
}

/// Hooks retain original layer addresses during continuation and reject points outside the range.
#[test]
fn scoped_range_hooks_keep_original_addresses_and_affect_the_continuation() {
    let device = Default::default();
    let model = Qwen3_5TextModel::<B>::init(config(), &device).unwrap();
    let state = model.embed(&input(3)).unwrap();
    let prefix = model.run_layers(state, 0..1).unwrap();
    let expected = model.run_layers(prefix.clone(), 1..2).unwrap();
    let mut hooks = Hooks::new();
    hooks.edit(Layer::decoder(1).residual_post(), |hidden| hidden * 0.0);
    let changed = model
        .run_layers_with_hooks(prefix.clone(), 1..2, &mut hooks)
        .unwrap();
    assert!(
        changed
            .hidden
            .to_data()
            .to_vec::<f32>()
            .unwrap()
            .iter()
            .all(|value| *value == 0.0)
    );
    assert_ne!(changed.hidden.to_data(), expected.hidden.to_data());
    let mut invalid = Hooks::new();
    invalid.capture(Layer::decoder(0).mlp_activation());
    assert!(
        model
            .run_layers_with_hooks(prefix, 1..2, &mut invalid)
            .is_err()
    );
}

/// Execution rejects skipped or repeated layers and forbids resuming an already normalized state.
#[test]
fn block_ranges_and_normalization_reject_implicit_repetition_or_skipping() {
    let model = Qwen3_5TextModel::<B>::init(config(), &Default::default()).unwrap();
    let embedded = model.embed(&input(3)).unwrap();
    assert!(model.run_layers(embedded.clone(), 1..2).is_err());
    assert!(model.run_layers(embedded.clone(), 0..3).is_err());
    let prefix = model.run_layers(embedded, 0..1).unwrap();
    assert!(model.run_layers(prefix.clone(), 0..1).is_err());
    let normalized = model.normalize(prefix).unwrap();
    assert!(model.normalize(normalized.clone()).is_err());
    assert!(model.run_layers(normalized, 1..2).is_err());
}

/// A latent F64 replacement fails before callbacks and clears captures from the earlier valid run.
#[test]
fn changed_latent_dtype_fails_before_callbacks_and_clears_previous_captures() {
    use std::cell::Cell;

    let model = Qwen3_5TextModel::<B>::init(config(), &Default::default()).unwrap();
    let embedded = model.embed(&input(3)).unwrap();
    let called = Cell::new(false);
    let mut hooks = Hooks::new();
    let capture = hooks.capture(Layer::decoder(0).residual_pre());
    hooks.edit(Layer::decoder(0).residual_pre(), |hidden| {
        called.set(true);
        hidden
    });
    model
        .run_layers_with_hooks(embedded.clone(), 0..1, &mut hooks)
        .unwrap();
    assert!(hooks.get(&capture).is_ok());
    called.set(false);
    let mut changed = embedded;
    changed.hidden = changed.hidden.cast(DType::F64);
    let error = model
        .run_layers_with_hooks(changed.clone(), 0..1, &mut hooks)
        .unwrap_err();
    assert!(error.to_string().contains("qwen.execution.hidden dtype"));
    assert!(!called.get());
    assert!(hooks.get(&capture).is_err());

    let error = model.layers[0]
        .forward(
            changed.hidden.clone(),
            changed.context.positions,
            None,
            model.layer_rope[0],
            changed.context.evaluated_context,
        )
        .unwrap_err();
    assert!(error.to_string().contains("qwen.block.hidden dtype"));
    let error = model
        .tied_vocabulary_projection(changed.hidden)
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("qwen.tied_vocabulary_projection dtype")
    );
}

/// Final normalization rejects a changed latent dtype and discards its previous capture.
#[test]
fn normalization_rejects_changed_latent_dtype_and_clears_previous_capture() {
    let model = Qwen3_5TextModel::<B>::init(config(), &Default::default()).unwrap();
    let state = model.embed(&input(3)).unwrap();
    let mut hooks = Hooks::new();
    let capture = hooks.capture(Stack::Decoder.final_norm());
    model
        .normalize_with_hooks(state.clone(), &mut hooks)
        .unwrap();
    assert!(hooks.get(&capture).is_ok());
    let mut changed = state;
    changed.hidden = changed.hidden.cast(DType::F64);
    let error = model.normalize_with_hooks(changed, &mut hooks).unwrap_err();
    assert!(error.to_string().contains("qwen.execution.hidden dtype"));
    assert!(hooks.get(&capture).is_err());
}

/// An F64 attention bias fails input validation before a residual callback can run.
#[test]
fn changed_attention_bias_dtype_fails_before_callbacks() {
    use std::cell::Cell;

    let device = Default::default();
    let model = full_attention_model();
    let mut state = model.embed(&input(3)).unwrap();
    state.context.attention = Some(
        AttentionPolicy::causal()
            .with_bias(Tensor::<B, 4>::zeros([1, 1, 1, 1], &device).cast(DType::F64)),
    );
    let called = Cell::new(false);
    let mut hooks = Hooks::new();
    hooks.edit(Layer::decoder(0).residual_pre(), |hidden| {
        called.set(true);
        hidden
    });
    let error = model
        .run_layers_with_hooks(state, 0..1, &mut hooks)
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("qwen.execution.attention.bias dtype")
    );
    assert!(!called.get());
}

/// Builds a single full-attention block to isolate configurable connectivity from DeltaNet.
fn full_attention_model() -> Qwen3_5TextModel<B> {
    let mut config = config();
    config.num_hidden_layers = 1;
    config.layer_types = vec![Qwen3_5LayerType::FullAttention];
    Qwen3_5TextModel::init(config, &Default::default()).unwrap()
}

/// Changing the last token affects the first token only under explicitly bidirectional attention.
#[test]
fn full_attention_policy_allows_future_token_influence_only_when_requested() {
    let model = full_attention_model();
    let run = |last, bidirectional| {
        let mut state = model.embed(&input(last)).unwrap();
        if bidirectional {
            state.context.attention = Some(AttentionPolicy::bidirectional());
        }
        model
            .run_layers(state, 0..1)
            .unwrap()
            .hidden
            .narrow(1, 0, 1)
            .to_data()
    };
    assert_eq!(run(3, false), run(4, false));
    assert_ne!(run(3, true), run(4, true));
}

/// A broadcast custom mask excluding every key produces an exactly zero attention output.
#[test]
fn custom_connectivity_broadcasts_and_fully_excluded_attention_rows_are_zero() {
    let device = Default::default();
    let model = full_attention_model();
    let mut state = model.embed(&input(3)).unwrap();
    let allowed = Tensor::<B, 4, Bool>::from_bool([[[[false, false, false]]]].into(), &device);
    state.context.attention = Some(
        AttentionPolicy::custom(allowed).with_bias(Tensor::<B, 4>::zeros([1, 1, 1, 1], &device)),
    );
    let mut hooks = Hooks::new();
    let attention = hooks.capture(Layer::decoder(0).attention_output());
    model
        .run_layers_with_hooks(state, 0..1, &mut hooks)
        .unwrap();
    assert!(
        hooks
            .get(&attention)
            .unwrap()
            .to_data()
            .to_vec::<f32>()
            .unwrap()
            .iter()
            .all(|value| *value == 0.0)
    );
}

/// Bidirectional policy on a hybrid containing DeltaNet fails before executing block callbacks.
#[test]
fn unsupported_deltanet_policies_fail_before_any_block_callback() {
    use std::cell::Cell;

    let model = Qwen3_5TextModel::<B>::init(config(), &Default::default()).unwrap();
    let mut state = model.embed(&input(3)).unwrap();
    state.context.attention = Some(AttentionPolicy::bidirectional());
    let called = Cell::new(false);
    let mut hooks = Hooks::new();
    hooks.edit(Layer::decoder(0).residual_pre(), |hidden| {
        called.set(true);
        hidden
    });
    assert!(
        model
            .run_layers_with_hooks(state, 0..2, &mut hooks)
            .is_err()
    );
    assert!(!called.get());
}

/// An independent head initially matches tied projection and survives later embedding replacement.
#[test]
fn explicit_independent_vocabulary_head_starts_equal_and_then_can_diverge() {
    let device = Default::default();
    let mut model = Qwen3_5TextModel::<B>::init(config(), &device).unwrap();
    let hidden = Tensor::<B, 3>::ones([1, 2, 8], &device);
    let head = model.independent_vocabulary_head();
    assert_ne!(head.weight.id, model.embed_tokens.weight.id);
    let storage: burn::backend::flex::FlexTensor = head.weight.val().into_primitive().tensor();
    assert!(
        storage.is_contiguous(),
        "inference Flex stores the head contiguously"
    );
    assert_eq!(
        head.weight.val().into_data(),
        model.embed_tokens.weight.val().transpose().into_data()
    );
    let original = operations::linear(hidden.clone(), &head).unwrap().to_data();
    // The weights above are equal exactly. The tied projection reads the
    // embedding through a transposed view while the head is contiguous, so an
    // x86 AVX GEMM may sum the eight products in another order. The logits are
    // small sums of partly cancelling O(0.02) terms, so reordering moves them
    // by a few 1e-7 absolutely; a head that differed from the embedding would
    // move them by O(1e-2).
    model
        .tied_vocabulary_projection(hidden.clone())
        .unwrap()
        .to_data()
        .assert_approx_eq::<f32>(&original, Tolerance::absolute(1e-5));
    model.embed_tokens.weight = Param::from_tensor(Tensor::zeros([16, 8], &device));
    assert_eq!(
        operations::linear(hidden.clone(), &head).unwrap().to_data(),
        original
    );
    assert_ne!(
        model.tied_vocabulary_projection(hidden).unwrap().to_data(),
        original
    );
}

/// Backpropagating through an independent head reaches its weight but not the source embedding.
#[cfg(feature = "autodiff")]
#[test]
fn independent_vocabulary_head_uses_a_new_gradient_leaf() {
    use burn::backend::Autodiff;

    /// Autodiff wrapper used to distinguish the independent head gradient leaf.
    type Ad = Autodiff<B>;
    let device = Default::default();
    let model = Qwen3_5TextModel::<Ad>::init(config(), &device).unwrap();
    let head = model.independent_vocabulary_head();
    let gradients = operations::linear(Tensor::<Ad, 3>::ones([1, 2, 8], &device), &head)
        .unwrap()
        .sum()
        .backward();
    assert!(head.weight.grad(&gradients).is_some());
    assert!(model.embed_tokens.weight.grad(&gradients).is_none());
}

/// Message carried by the deliberately failing hook callbacks below.
const HOOK_FAILURE: &str = "deliberate hook failure";

/// Registers one edit at `point` whose callback fails with [`HOOK_FAILURE`].
fn failing_hooks(point: &PointInfo) -> Hooks<'static, B> {
    /// Builds the callback error returned at every point.
    fn failure() -> Error {
        Error::configuration("test.hook", HOOK_FAILURE)
    }
    let mut hooks = Hooks::new();
    match point.rank() {
        3 => hooks.try_edit(
            crate::intervention::Point::<3>::custom(
                point.name.clone(),
                point.axes.clone().try_into().unwrap(),
            ),
            |_| Err(failure()),
        ),
        4 => hooks.try_edit(
            crate::intervention::Point::<4>::custom(
                point.name.clone(),
                point.axes.clone().try_into().unwrap(),
            ),
            |_| Err(failure()),
        ),
        rank => panic!("{}: unexpected rank {rank}", point.name),
    }
    hooks
}

/// Asserts that a hooked call failed with the deliberate callback error.
fn assert_hook_failure<T: core::fmt::Debug>(result: Result<T>, point: &PointInfo) {
    let error = result.expect_err(&point.name);
    assert!(
        error.to_string().contains(HOOK_FAILURE),
        "{}: {error}",
        point.name
    );
}

/// A failing callback at any catalog point aborts the fused and staged passes with its error.
///
/// Every DeltaNet, attention (with eager scores and pattern), MLP, residual,
/// embedding, and final-norm point is emitted on both paths, and the user's
/// error is returned rather than swallowed or replaced.
#[test]
fn a_failing_edit_at_any_point_aborts_with_its_error() {
    let device = Default::default();
    let mut model = Qwen3_5ForCausalLm::<B>::init(config(), &device).unwrap();
    model.model.layers[1]
        .self_attn
        .as_mut()
        .unwrap()
        .precision_mode = crate::attention::AttentionPrecisionMode::EagerFp32Softmax;
    let points = model.activation_points();
    assert!(
        points
            .iter()
            .any(|point| point.name == "decoder.layers.1.attention.pattern")
    );
    for point in &points {
        let mut hooks = failing_hooks(point);
        assert_hook_failure(
            model
                .forward_with_hooks(input(3), &ForwardOptions::default(), &mut hooks)
                .map(|_| ()),
            point,
        );
    }
    let backbone = &model.model;
    for point in &points {
        let mut hooks = failing_hooks(point);
        let staged = if *point == Stack::Decoder.embedding().info() {
            backbone.embed_with_hooks(&input(3), &mut hooks).map(|_| ())
        } else if *point == Stack::Decoder.final_norm().info() {
            let state = backbone.embed(&input(3)).unwrap();
            let state = backbone.run_layers(state, 0..2).unwrap();
            backbone.normalize_with_hooks(state, &mut hooks).map(|_| ())
        } else {
            let state = backbone.embed(&input(3)).unwrap();
            backbone
                .run_layers_with_hooks(state, 0..2, &mut hooks)
                .map(|_| ())
        };
        assert_hook_failure(staged, point);
    }
}

/// The hooked headless backbone equals the plain backbone and the generic staging trait.
#[test]
fn hooked_backbone_matches_the_plain_backbone() {
    use crate::architecture::StagedModel;

    let model = Qwen3_5TextModel::<B>::init(config(), &Default::default()).unwrap();
    let (plain, plain_states) = model.forward(&input(3), true).unwrap();
    let mut hooks = Hooks::new();
    let residual = hooks.capture(Layer::decoder(0).residual_post());
    let (hooked, hooked_states) = model
        .forward_with_hooks(&input(3), true, &mut hooks)
        .unwrap();
    assert_eq!(hooked.to_data(), plain.to_data());
    assert_eq!(hooked_states.len(), 4);
    assert_eq!(plain_states.len(), 4);
    assert_eq!(
        hooks.get(&residual).unwrap().to_data(),
        plain_states[1].to_data()
    );
    assert_eq!(StagedModel::stack(&model), Stack::Decoder);
    assert_eq!(StagedModel::num_layers(&model), 2);
}

/// Staged execution rejects contexts and module trees the hybrid cannot evaluate.
#[test]
fn staged_execution_rejects_unusable_contexts_and_modules() {
    let device = Default::default();
    let message =
        |result: Result<HiddenState<B>>| result.expect_err("the staged call must fail").to_string();
    let model = Qwen3_5TextModel::<B>::init(config(), &device).unwrap();
    let fresh = || model.embed(&input(3)).unwrap();

    let mut cross = fresh();
    cross.context.cross_attention = Some(AttentionPolicy::causal());
    assert!(message(model.run_layers(cross, 0..2)).contains("cross-attention"));
    let mut layer_cross = fresh();
    layer_cross
        .context
        .layer_cross_attention
        .push((Layer::decoder(1), AttentionPolicy::causal()));
    assert!(message(model.run_layers(layer_cross, 0..2)).contains("cross-attention"));
    let mut encoded = fresh();
    encoded.context.encoder = Some(crate::experiment::EncoderContext {
        hidden: Tensor::zeros([1, 2, 8], &device),
        positions: Tensor::<B, 2, Int>::from_ints([[0, 1]], &device),
        token_mask: None,
    });
    assert!(message(model.run_layers(encoded, 0..2)).contains("cross-attention"));

    // DeltaNet keeps its causal recurrence: neither connectivity nor a bias applies.
    for policy in [
        AttentionPolicy::causal().with_bias(Tensor::<B, 4>::zeros([1, 1, 3, 3], &device)),
        AttentionPolicy {
            allowed: Some(Tensor::<B, 4, Bool>::from_bool(
                [[[[true, true, true]]]].into(),
                &device,
            )),
            ..AttentionPolicy::causal()
        },
    ] {
        let mut delta = fresh();
        delta.context = delta
            .context
            .with_layer_attention(Layer::decoder(0), policy);
        assert!(message(model.run_layers(delta, 0..2)).contains("causal recurrence"));
    }

    let mut beyond = fresh();
    beyond.next_layer = 3;
    assert!(message(model.normalize(beyond)).contains("valid block boundary"));

    let mut stray = fresh();
    stray.context = stray
        .context
        .with_layer_attention(Layer::decoder(2), AttentionPolicy::causal());
    assert!(message(model.run_layers(stray, 0..2)).contains("unavailable Qwen decoder layer"));

    let mut misshaped = fresh();
    misshaped.context = misshaped.context.with_layer_attention(
        Layer::decoder(1),
        AttentionPolicy::custom(Tensor::<B, 4, Bool>::from_bool(
            [[[[true, true]]]].into(),
            &device,
        )),
    );
    assert!(message(model.run_layers(misshaped, 0..2)).contains("attention.allowed"));

    // One rotary configuration is required per block, staged or fused.
    let mut short_rope = Qwen3_5TextModel::<B>::init(config(), &device).unwrap();
    short_rope.layer_rope.pop();
    let state = short_rope.embed(&input(3)).unwrap();
    assert!(message(short_rope.run_layers(state, 0..2)).contains("one rotary configuration"));
    let error = short_rope.forward(&input(3), false).unwrap_err();
    assert!(
        error.to_string().contains("one rotary configuration"),
        "{error}"
    );

    // A block whose mixer was removed fails on both paths.
    let mut no_attention = Qwen3_5TextModel::<B>::init(config(), &device).unwrap();
    no_attention.layers[1].self_attn = None;
    let state = no_attention.embed(&input(3)).unwrap();
    assert!(message(no_attention.run_layers(state, 0..2)).contains("lacks its mixer"));
    let error = no_attention.forward(&input(3), false).unwrap_err();
    assert!(error.to_string().contains("no attention module"), "{error}");
    let mut no_delta = Qwen3_5TextModel::<B>::init(config(), &device).unwrap();
    no_delta.layers[0].linear_attn = None;
    let error = no_delta.forward(&input(3), false).unwrap_err();
    assert!(error.to_string().contains("no DeltaNet module"), "{error}");
}

/// One block called directly validates its hidden dtype and token-mask extent.
#[test]
fn single_block_validates_dtype_and_mask_extent() {
    let device = Default::default();
    let model = Qwen3_5TextModel::<B>::init(config(), &device).unwrap();
    let hidden = model
        .embed_tokens
        .forward(Tensor::<B, 2, Int>::from_ints([[1, 2, 3]], &device));
    let positions = Tensor::<B, 2, Int>::from_ints([[0, 1, 2]], &device);
    let run = |hidden: Tensor<B, 3>, mask: Option<&TokenMask<B>>| {
        model.layers[1].forward(hidden, positions.clone(), mask, model.layer_rope[1], 3)
    };
    let error = run(hidden.clone().cast(DType::F16), None).unwrap_err();
    assert!(
        error.to_string().contains("qwen.block.hidden dtype"),
        "{error}"
    );
    let wide_mask = TokenMask::new(Tensor::<B, 2, Bool>::from_bool(
        [[true, true, true, true]].into(),
        &device,
    ));
    let error = run(hidden.clone(), Some(&wide_mask)).unwrap_err();
    assert!(error.to_string().contains("token_mask"), "{error}");
    let full_mask = TokenMask::new(Tensor::<B, 2, Bool>::from_bool(
        [[true, true, true]].into(),
        &device,
    ));
    assert_eq!(
        run(hidden.clone(), Some(&full_mask)).unwrap().to_data(),
        run(hidden, None).unwrap().to_data()
    );
}

/// The tied readout rejects empty or mis-sized hidden states.
#[test]
fn tied_vocabulary_projection_rejects_malformed_hidden_states() {
    let device = Default::default();
    let model = Qwen3_5TextModel::<B>::init(config(), &device).unwrap();
    for shape in [[0, 1, 8], [1, 0, 8], [1, 1, 7]] {
        let error = model
            .tied_vocabulary_projection(Tensor::zeros(shape, &device))
            .unwrap_err();
        assert!(
            matches!(&error, Error::Shape { tensor, .. } if tensor == "qwen.tied_vocabulary_projection"),
            "{error}"
        );
    }
}

/// Attention-probability capture is unavailable on both stateless forward paths.
#[test]
fn attention_probability_capture_is_rejected() {
    let model = Qwen3_5ForCausalLm::<B>::init(config(), &Default::default()).unwrap();
    let options = ForwardOptions {
        attentions: true,
        ..Default::default()
    };
    for error in [
        model.forward(input(3), &options).unwrap_err(),
        model
            .forward_with_hooks(input(3), &options, &mut Hooks::new())
            .unwrap_err(),
    ] {
        assert!(matches!(error, Error::BackendCapability(_)));
        assert!(
            error.to_string().contains("reference attention path"),
            "{error}"
        );
    }
}

/// An ungated full-attention block projects only queries and still runs.
#[test]
fn ungated_attention_projects_queries_only() {
    let device = Default::default();
    let ungated = Qwen3_5TextConfig {
        attn_output_gate: false,
        ..config()
    };
    let model = Qwen3_5ForCausalLm::<B>::init(ungated, &device).unwrap();
    let attention = model.model.layers[1].self_attn.as_ref().unwrap();
    assert_eq!(attention.q_proj.weight.dims(), [8, 2 * 4]);
    assert_eq!(
        model
            .forward(input(3), &ForwardOptions::default())
            .unwrap()
            .logits
            .dims(),
        [1, 1, 16]
    );
}

/// Builds the two-block test topology with one auxiliary MTP layer.
fn mtp_model() -> Qwen3_5ForCausalLm<B> {
    let config = Qwen3_5TextConfig {
        mtp_num_hidden_layers: 1,
        ..config()
    };
    Qwen3_5ForCausalLm::init(config, &Default::default()).unwrap()
}

/// The stateless MTP path requires its graph, non-empty tokens, and matching positions.
#[test]
fn stateless_mtp_validates_graph_tokens_and_positions() {
    let device = Default::default();
    let hidden = || Tensor::<B, 3>::zeros([1, 1, 8], &device);
    let positions = || Tensor::<B, 2, Int>::from_ints([[0]], &device);
    let base = Qwen3_5ForCausalLm::<B>::init(config(), &device).unwrap();
    let error = base
        .forward_mtp(Tensor::from_ints([[1]], &device), hidden(), positions(), 0)
        .unwrap_err();
    assert!(
        matches!(&error, Error::Configuration { field, .. } if field == "qwen3_5.mtp_num_hidden_layers"),
        "{error}"
    );

    let model = mtp_model();
    for shape in [[1, 0], [0, 1]] {
        let error = model
            .forward_mtp(
                Tensor::<B, 2, Int>::zeros(shape, &device),
                Tensor::zeros([shape[0], shape[1], 8], &device),
                Tensor::<B, 2, Int>::zeros(shape, &device),
                0,
            )
            .unwrap_err();
        assert!(
            matches!(&error, Error::Shape { tensor, .. } if tensor == "qwen.mtp.next_token_ids"),
            "{error}"
        );
    }
    let error = model
        .forward_mtp(
            Tensor::from_ints([[1]], &device),
            hidden(),
            Tensor::<B, 2, Int>::from_ints([[0, 1]], &device),
            0,
        )
        .unwrap_err();
    assert!(
        matches!(&error, Error::Shape { tensor, .. } if tensor == "qwen.mtp.position_ids"),
        "{error}"
    );
    // Step `s` selects layer `s mod 1`, so every step uses the same layer.
    let step = |step| {
        model
            .forward_mtp(
                Tensor::from_ints([[1]], &device),
                hidden(),
                positions(),
                step,
            )
            .unwrap()
            .logits
            .to_data()
    };
    assert_eq!(step(0), step(5));
}

/// MTP construction rejects absent layers, dedicated embeddings, and an unrepresentable width.
#[test]
fn mtp_initialization_rejects_unsupported_topologies() {
    let device = Default::default();
    let cases = [
        (
            Qwen3_5TextConfig {
                mtp_num_hidden_layers: 0,
                ..config()
            },
            "qwen3_5.mtp_num_hidden_layers",
        ),
        (
            Qwen3_5TextConfig {
                mtp_num_hidden_layers: 1,
                mtp_use_dedicated_embeddings: true,
                ..config()
            },
            "qwen3_5.mtp_use_dedicated_embeddings",
        ),
        (
            Qwen3_5TextConfig {
                mtp_num_hidden_layers: 1,
                hidden_size: usize::MAX / 2 + 1,
                ..config()
            },
            "qwen3_5.hidden_size",
        ),
    ];
    for (config, expected) in cases {
        let error = Qwen3_5Mtp::<B>::init(&config, &device).unwrap_err();
        assert!(
            matches!(&error, Error::Configuration { field, .. } if field == expected),
            "{error}"
        );
    }
}

/// The MTP graph rejects mismatched, empty, or mis-sized inputs and an empty layer list.
#[test]
fn mtp_graph_validates_its_inputs_and_layers() {
    let device = Default::default();
    let mut model = mtp_model();
    let rope = model.model.rope;
    let mtp = model.mtp.as_mut().unwrap();
    let embedding = |shape: [usize; 3]| Tensor::<B, 3>::zeros(shape, &device);
    let shape_error = |error: Error, expected: &str| {
        assert!(
            matches!(&error, Error::Shape { tensor, .. } if tensor == expected),
            "{error}"
        );
    };
    shape_error(
        mtp.fuse(embedding([1, 2, 8]), embedding([1, 1, 8]))
            .unwrap_err(),
        "qwen.mtp.target_hidden_state",
    );
    for shape in [[0, 1, 8], [1, 0, 8], [1, 1, 7]] {
        shape_error(
            mtp.fuse(embedding(shape), embedding(shape)).unwrap_err(),
            "qwen.mtp.inputs",
        );
    }
    shape_error(
        mtp.forward(
            embedding([1, 2, 8]),
            embedding([1, 2, 8]),
            Tensor::<B, 2, Int>::from_ints([[0]], &device),
            0,
            rope,
            2,
        )
        .unwrap_err(),
        "qwen.mtp.position_ids",
    );
    mtp.layers.clear();
    let error = mtp
        .forward(
            embedding([1, 1, 8]),
            embedding([1, 1, 8]),
            Tensor::<B, 2, Int>::from_ints([[0]], &device),
            0,
            rope,
            1,
        )
        .unwrap_err();
    assert!(
        matches!(&error, Error::Configuration { field, .. } if field == "qwen3_5.mtp"),
        "{error}"
    );
}
