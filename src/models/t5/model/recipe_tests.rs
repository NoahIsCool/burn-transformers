//! Public T5 recipe and continuation contracts, using tiny deterministic graphs.

use burn::{
    backend::Flex,
    tensor::{Int, Tensor},
};

use super::*;
use crate::experiment::AttentionPolicy;

type TestBackend = Flex<f32, i32>;

fn config() -> T5Config {
    T5Config {
        vocab_size: 16,
        d_model: 4,
        d_kv: 2,
        d_ff: 8,
        num_layers: 3,
        num_decoder_layers: Some(1),
        num_heads: 2,
        relative_attention_num_buckets: 8,
        relative_attention_max_distance: 32,
        dropout_rate: 0.0,
        feed_forward_proj: T5FeedForwardProjection::GatedGelu,
        tie_word_embeddings: true,
        ..T5Config::default()
    }
}

fn input<B: Backend>(device: &B::Device) -> CausalLmInput<B> {
    CausalLmInput {
        token_ids: Tensor::<B, 2, Int>::from_ints([[1, 2, 3]], device),
        token_mask: None,
        position_ids: None,
    }
}

fn values(tensor: Tensor<TestBackend, 3>) -> Vec<f32> {
    tensor.into_data().to_vec::<f32>().unwrap()
}

#[test]
fn unchanged_t5_recipe_matches_family_encoder_exactly_with_same_parameters() {
    let device = Default::default();
    let model = T5ForConditionalGeneration::<TestBackend>::init(config(), &device).unwrap();
    let expected = model
        .encode(input(&device), &ForwardOptions::default())
        .unwrap();
    let shared_id = model.shared.weight.id;
    let extracted = model.into_encoder().unwrap();
    assert_eq!(extracted.shared.weight.id, shared_id);
    let encoder = T5EncoderRecipe::from_config(config())
        .unwrap()
        .init(&device)
        .unwrap()
        .load_record(extracted.into_record());
    assert_eq!(encoder.bias_sources, vec![0, 0, 0]);
    assert!(
        encoder.encoder.block[0]
            .self_attention
            .self_attention
            .relative_attention_bias
            .is_some()
    );
    assert!(
        encoder.encoder.block[1]
            .self_attention
            .self_attention
            .relative_attention_bias
            .is_none()
    );
    let actual = encoder
        .encode(input(&device), &ForwardOptions::default())
        .unwrap();
    assert_eq!(values(actual.hidden_state), values(expected.hidden_state));
}

#[test]
fn t5_recipe_split_resume_and_final_normalization_are_exact() {
    let device = Default::default();
    let encoder = T5EncoderRecipe::from_config(config())
        .unwrap()
        .init::<TestBackend>(&device)
        .unwrap();
    let expected = encoder
        .encode(input(&device), &ForwardOptions::default())
        .unwrap();
    let state = encoder.embed(input(&device)).unwrap();
    assert_eq!(state.next_layer, 0);
    assert!(!state.normalized);
    let state = encoder.run_layers(state, 0..1).unwrap();
    assert_eq!(state.next_layer, 1);
    let mut hooks = Hooks::new();
    let capture = hooks.capture(Layer::encoder(1).mlp_activation().token(0));
    let state = encoder
        .run_layers_with_hooks(state, 1..3, &mut hooks)
        .unwrap();
    assert_eq!(hooks.get(&capture).unwrap().dims(), [1, 1, 8]);
    let state = encoder.normalize(state).unwrap();
    assert!(state.normalized);
    assert_eq!(
        values(state.into_encoder_output().unwrap().hidden_state),
        values(expected.hidden_state)
    );
}

#[test]
fn heterogeneous_t5_recipe_shares_bias_only_between_compatible_head_counts() {
    let device = Default::default();
    let mut recipe = T5EncoderRecipe::from_config(config()).unwrap();
    recipe.layers[0].attention_heads = 1;
    recipe.layers[0].head_dim = 4;
    recipe.layers[1].feed_forward_width = 12;
    recipe.layers[2].attention_heads = 1;
    recipe.layers[2].head_dim = 3;
    recipe.layers[2].feed_forward_width = 5;
    recipe.layers[2].feed_forward_projection = T5FeedForwardProjection::Relu;
    let encoder = recipe.init::<TestBackend>(&device).unwrap();
    assert_eq!(encoder.bias_sources, vec![0, 1, 0]);
    assert!(
        encoder.encoder.block[1]
            .self_attention
            .self_attention
            .relative_attention_bias
            .is_some()
    );
    assert!(
        encoder.encoder.block[2]
            .self_attention
            .self_attention
            .relative_attention_bias
            .is_none()
    );
    let mut hooks = Hooks::new();
    let middle = hooks.capture(Layer::encoder(1).mlp_activation());
    let last = hooks.capture(Layer::encoder(2).mlp_activation());
    let output = encoder
        .encode_with_hooks(input(&device), &ForwardOptions::default(), &mut hooks)
        .unwrap();
    assert_eq!(output.hidden_state.dims(), [1, 3, 4]);
    assert_eq!(hooks.get(&middle).unwrap().dims(), [1, 3, 12]);
    assert_eq!(hooks.get(&last).unwrap().dims(), [1, 3, 5]);
}

#[test]
fn t5_recipe_attention_policy_changes_future_visibility() {
    let device = Default::default();
    let encoder = T5EncoderRecipe::from_config(config())
        .unwrap()
        .init::<TestBackend>(&device)
        .unwrap();
    let execute = |future, causal| {
        let mut source = input(&device);
        source.token_ids = Tensor::<TestBackend, 2, Int>::from_ints([[1, 2, future]], &device);
        let mut state = encoder.embed(source).unwrap();
        if causal {
            state.context = state.context.with_attention(AttentionPolicy::causal());
        }
        let state = encoder.run_layers(state, 0..3).unwrap();
        values(encoder.normalize(state).unwrap().hidden.narrow(1, 0, 1))
    };
    assert_eq!(execute(3, true), execute(4, true));
    assert_ne!(execute(3, false), execute(4, false));
}

#[test]
fn t5_recipe_relative_positions_are_explicit_and_translation_invariant() {
    let device = Default::default();
    let encoder = T5EncoderRecipe::from_config(config())
        .unwrap()
        .init::<TestBackend>(&device)
        .unwrap();
    let execute = |positions| {
        let mut source = input(&device);
        source.position_ids = Some(Tensor::<TestBackend, 2, Int>::from_ints(
            [positions],
            &device,
        ));
        values(
            encoder
                .encode(source, &ForwardOptions::default())
                .unwrap()
                .hidden_state,
        )
    };
    assert_eq!(execute([0, 1, 2]), execute([10, 11, 12]));
    assert_ne!(execute([0, 1, 2]), execute([0, 10, 11]));
}

#[test]
fn t5_recipe_rejects_skipped_repeated_and_normalized_execution() {
    let device = Default::default();
    let encoder = T5EncoderRecipe::from_config(config())
        .unwrap()
        .init::<TestBackend>(&device)
        .unwrap();
    assert!(
        encoder
            .run_layers(encoder.embed(input(&device)).unwrap(), 1..2)
            .is_err()
    );
    let state = encoder
        .run_layers(encoder.embed(input(&device)).unwrap(), 0..1)
        .unwrap();
    assert!(encoder.run_layers(state.clone(), 0..1).is_err());
    let normalized = encoder.normalize(state).unwrap();
    assert!(encoder.run_layers(normalized.clone(), 1..3).is_err());
    assert!(encoder.normalize(normalized).is_err());
}

#[test]
fn t5_recipe_zero_layer_control_and_invalid_dimensions_are_explicit() {
    let device = Default::default();
    let mut recipe = T5EncoderRecipe::from_config(config()).unwrap();
    recipe.layers.clear();
    let encoder = recipe.init::<TestBackend>(&device).unwrap();
    assert_eq!(
        encoder
            .encode(input(&device), &ForwardOptions::default())
            .unwrap()
            .hidden_state
            .dims(),
        [1, 3, 4]
    );
    let mut recipe = T5EncoderRecipe::from_config(config()).unwrap();
    recipe.layers[1].attention_heads = 0;
    let error = recipe.init::<TestBackend>(&device).unwrap_err();
    assert!(error.to_string().contains("layers.1"));
}

#[cfg(feature = "autodiff")]
#[test]
fn t5_recipe_staged_inserted_module_preserves_upstream_gradients() {
    use burn::{backend::Autodiff, nn::LinearConfig};

    type AdBackend = Autodiff<TestBackend>;
    let device = Default::default();
    let encoder = T5EncoderRecipe::from_config(config())
        .unwrap()
        .init::<AdBackend>(&device)
        .unwrap();
    let inserted = LinearConfig::new(4, 4).init::<AdBackend>(&device);
    let mut state = encoder
        .run_layers(encoder.embed(input(&device)).unwrap(), 0..1)
        .unwrap();
    state.hidden = inserted.forward(state.hidden);
    let state = encoder.run_layers(state, 1..3).unwrap();
    let loss = encoder.normalize(state).unwrap().hidden.square().sum();
    let grads = loss.backward();
    assert!(inserted.weight.val().grad(&grads).is_some());
    assert!(encoder.shared.weight.val().grad(&grads).is_some());
}

#[test]
fn t5_recipe_parameter_transfer_preserves_unchanged_graph_and_reports_new_shapes() {
    use crate::transfer::{ShapeChangePolicy, TransferPlan};

    let device = Default::default();
    let source = T5ForConditionalGeneration::<TestBackend>::init(config(), &device).unwrap();
    let recipe = T5EncoderRecipe::from_config(config()).unwrap();
    let target = recipe.init::<TestBackend>(&device).unwrap();
    let plan = TransferPlan::new()
        .copy_prefix("shared", "shared")
        .copy_prefix("encoder", "encoder");
    let (copied, report) = plan.apply(&source, &target).unwrap();
    assert!(report.newly_initialized.is_empty());
    assert!(
        report
            .omitted
            .iter()
            .any(|parameter| parameter.name.starts_with("decoder."))
    );
    assert_eq!(
        values(
            copied
                .encode(input(&device), &ForwardOptions::default())
                .unwrap()
                .hidden_state
        ),
        values(
            source
                .encode(input(&device), &ForwardOptions::default())
                .unwrap()
                .hidden_state
        ),
    );

    let mut changed = recipe;
    changed.layers[1].attention_heads = 1;
    changed.layers[1].feed_forward_width = 12;
    let changed = changed.init::<TestBackend>(&device).unwrap();
    let (changed, report) = TransferPlan::new()
        .copy_prefix("shared", "shared")
        .copy_prefix("encoder", "encoder")
        .on_shape_change(ShapeChangePolicy::Reinitialize)
        .apply(&source, &changed)
        .unwrap();
    assert!(
        report
            .reused
            .iter()
            .any(|parameter| parameter.target == "shared.weight")
    );
    assert!(
        report
            .newly_initialized
            .iter()
            .any(|parameter| parameter.name.contains("block.1"))
    );
    assert_eq!(
        changed
            .encode(input(&device), &ForwardOptions::default())
            .unwrap()
            .hidden_state
            .dims(),
        [1, 3, 4]
    );
}

#[test]
fn t5_invalid_continuation_clears_previous_hook_captures() {
    let device = Default::default();
    let encoder = T5EncoderRecipe::from_config(config())
        .unwrap()
        .init::<TestBackend>(&device)
        .unwrap();
    let mut hooks = Hooks::new();
    let saved = hooks.capture(Layer::encoder(0).mlp_activation());
    encoder
        .run_layers_with_hooks(encoder.embed(input(&device)).unwrap(), 0..1, &mut hooks)
        .unwrap();
    assert!(hooks.get(&saved).is_ok());
    assert!(
        encoder
            .run_layers_with_hooks(
                encoder.embed(input(&device)).unwrap(),
                0..usize::MAX,
                &mut hooks
            )
            .is_err()
    );
    assert!(hooks.get(&saved).is_err());
}

#[test]
fn t5_embedding_hooks_capture_and_edit_before_layers_and_clear_on_failure() {
    use burn::nn::LinearConfig;

    let device = Default::default();
    let encoder = T5EncoderRecipe::from_config(config())
        .unwrap()
        .init::<TestBackend>(&device)
        .unwrap();
    let external = LinearConfig::new(4, 4).init::<TestBackend>(&device);
    let original = encoder.embed(input(&device)).unwrap();
    let expected = external.forward(original.hidden.clone());
    let mut hooks = Hooks::new();
    let saved = hooks.capture(Stack::Encoder.embedding());
    hooks.edit(Stack::Encoder.embedding(), |hidden| {
        external.forward(hidden)
    });
    let state = encoder
        .embed_with_hooks(input(&device), &mut hooks)
        .unwrap();
    assert_eq!(values(hooks.get(&saved).unwrap()), values(original.hidden));
    assert_eq!(values(state.hidden), values(expected));
    assert_eq!(state.next_layer, 0);
    assert!(!state.normalized);

    let mut invalid = input(&device);
    invalid.token_ids = Tensor::<TestBackend, 2, Int>::from_ints([[99, 1, 2]], &device);
    assert!(encoder.embed_with_hooks(invalid, &mut hooks).is_err());
    assert!(hooks.get(&saved).is_err());
}
