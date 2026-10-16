//! Public state boundaries preserve the graph while allowing deliberate edits.

use burn::{
    backend::Flex,
    tensor::{Bool, DType, Int, Tensor},
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
    assert_eq!(
        model
            .tied_vocabulary_projection(hidden.clone())
            .unwrap()
            .to_data(),
        original
    );
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
