//! Public T5 recipe and continuation contracts, using tiny deterministic graphs.

use burn::{
    backend::Flex,
    tensor::{Int, Tensor},
};

use super::*;
use crate::experiment::AttentionPolicy;

/// CPU backend used to compare complete and staged encoder execution.
type TestBackend = Flex<f32, i32>;

/// Builds three small gated-GELU encoder layers with shared relative-position bias.
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

/// Supplies one unpadded three-token sequence with implicit relative positions.
fn input<B: Backend>(device: &B::Device) -> CausalLmInput<B> {
    CausalLmInput {
        token_ids: Tensor::<B, 2, Int>::from_ints([[1, 2, 3]], device),
        token_mask: None,
        position_ids: None,
    }
}

/// Materializes F32 activations for exact comparisons of the same parameter graph.
fn values(tensor: Tensor<TestBackend, 3>) -> Vec<f32> {
    tensor.into_data().to_vec::<f32>().unwrap()
}

/// Extracted encoder records preserve shared parameter identity, bias ownership, and outputs.
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

/// Splitting layer execution preserves cursor progression, captures, and final normalized outputs.
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

/// Different head counts allocate separate bias owners while edited MLP widths retain residual shape.
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

/// Causal policy hides a changed future token that bidirectional attention can observe.
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

/// Translating every position preserves outputs, while changing relative distances affects them.
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

/// Continuation rejects skipped or repeated layers and already normalized states.
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

/// Empty recipes support embedding controls and invalid head counts name their layer.
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

/// A linear module inserted between layer ranges remains connected to upstream embeddings.
#[cfg(feature = "autodiff")]
#[test]
fn t5_recipe_staged_inserted_module_preserves_upstream_gradients() {
    use burn::{backend::Autodiff, nn::LinearConfig};

    /// Autodiff wrapper used to inspect inserted-module and embedding gradients.
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

/// Encoder transfer preserves unchanged outputs and reports reinitialization for edited shapes.
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

/// A rejected layer range clears captures retained from a previous successful continuation.
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

/// Embedding hooks capture before edits, preserve the initial cursor, and clear on invalid tokens.
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

// Properties vary execution partitions and geometry, while comparing one fixed
// parameter graph against itself. These check semantics rather than RNG values.
proptest::proptest! {
    #![proptest_config(proptest::test_runner::Config {
        cases: 16,
        rng_seed: proptest::test_runner::RngSeed::Fixed(0x5435_5245_4349_5045),
        ..Default::default()
    })]

    /// Arbitrary heterogeneous layer partitions preserve translated-position outputs and final captures.
    #[test]
    fn t5_recipe_partitions_preserve_heterogeneous_graphs_and_relative_translation(
        geometry in proptest::collection::vec((1usize..4, 1usize..5, 1usize..16), 1..5),
        cut_seed in 0usize..5,
        offset in 0i32..64,
        causal in proptest::bool::ANY,
    ) {
        let device = Default::default();
        let mut recipe = T5EncoderRecipe::from_config(config()).unwrap();
        let original = recipe.layers[0].clone();
        recipe.layers = geometry.iter().map(|&(heads, width, feed_forward)| {
            let mut layer = original.clone();
            layer.attention_heads = heads;
            layer.head_dim = width;
            layer.feed_forward_width = feed_forward;
            layer
        }).collect();
        let encoder = recipe.init::<TestBackend>(&device).unwrap();
        let depth = geometry.len();
        let cut = cut_seed % (depth + 1);
        let state = |translation| {
            let mut source = input(&device);
            source.position_ids = Some(Tensor::<TestBackend, 2, Int>::from_ints(
                [[translation, translation + 2, translation + 7]], &device,
            ));
            let mut state = encoder.embed(source).unwrap();
            state.context = state.context.with_attention(if causal {
                AttentionPolicy::causal()
            } else {
                AttentionPolicy::bidirectional()
            });
            state
        };
        let expected = encoder.normalize(encoder.run_layers(state(0), 0..depth).unwrap()).unwrap();
        let prefix = encoder.run_layers(state(offset), 0..cut).unwrap();
        proptest::prop_assert_eq!(prefix.next_layer, cut);
        proptest::prop_assert!(!prefix.normalized);
        let completed = encoder.run_layers(prefix, cut..depth).unwrap();
        let mut hooks = Hooks::new();
        let final_norm = hooks.capture(Stack::Encoder.final_norm());
        let actual = encoder.normalize_with_hooks(completed, &mut hooks).unwrap();
        proptest::prop_assert_eq!(actual.next_layer, depth);
        proptest::prop_assert_eq!(values(actual.hidden.clone()), values(expected.hidden));
        proptest::prop_assert_eq!(values(hooks.get(&final_norm).unwrap()), values(actual.hidden));
    }
}

/// Invalid dtypes, cursors, and attention policies fail before any hook callback runs.
#[test]
fn t5_recipe_rejects_incompatible_staged_contexts_before_callbacks() {
    use burn::tensor::DType;

    let device = Default::default();
    let encoder = T5EncoderRecipe::from_config(config())
        .unwrap()
        .init::<TestBackend>(&device)
        .unwrap();
    let base = encoder.embed(input(&device)).unwrap();
    let mut cases = Vec::new();
    let mut state = base.clone();
    state.hidden = state.hidden.cast(DType::F64);
    cases.push((state, "dtype"));
    let mut state = base.clone();
    state.next_layer = 4;
    cases.push((state, "cursor"));
    let mut state = base.clone();
    state.context.cross_attention = Some(AttentionPolicy::bidirectional());
    cases.push((state, "cross-attention"));
    for layer in [Layer::decoder(0), Layer::encoder(3)] {
        let mut state = base.clone();
        state.context = state
            .context
            .with_layer_attention(layer, AttentionPolicy::causal());
        cases.push((state, "outside this encoder"));
    }
    let mut state = base.clone();
    state.context =
        state
            .context
            .with_attention(AttentionPolicy::bidirectional().with_bias(
                Tensor::<TestBackend, 4>::zeros([1, 1, 1, 1], &device).cast(DType::F64),
            ));
    cases.push((state, "bias dtype"));
    let mut state = base;
    state.context = state.context.with_attention(
        AttentionPolicy::bidirectional().with_bias(Tensor::zeros([2, 1, 1, 1], &device)),
    );
    cases.push((state, "bias"));

    for (state, diagnostic) in cases {
        let called = std::cell::Cell::new(false);
        let mut hooks = Hooks::new();
        hooks.edit(Layer::encoder(0).residual_pre(), |hidden| {
            called.set(true);
            hidden
        });
        let error = encoder
            .run_layers_with_hooks(state, 0..3, &mut hooks)
            .unwrap_err();
        assert!(error.to_string().contains(diagnostic), "{error}");
        assert!(
            !called.get(),
            "invalid state invoked a callback: {diagnostic}"
        );
    }
}

/// Missing learned biases and encoder depth or role mismatches produce topology diagnostics.
#[test]
fn t5_recipe_rejects_missing_bias_owners_and_inconsistent_encoder_extraction() {
    let device = Default::default();
    let mut encoder = T5EncoderRecipe::from_config(config())
        .unwrap()
        .init::<TestBackend>(&device)
        .unwrap();
    encoder.encoder.block[0]
        .self_attention
        .self_attention
        .relative_attention_bias = None;
    let error = encoder
        .run_layers(encoder.embed(input(&device)).unwrap(), 0..1)
        .unwrap_err();
    assert!(error.to_string().contains("learned bias owner"));

    for wrong_role in [false, true] {
        let mut model = T5ForConditionalGeneration::<TestBackend>::init(config(), &device).unwrap();
        if wrong_role {
            model.encoder.is_decoder = true;
        } else {
            model.encoder.block.pop();
        }
        let error = model.into_encoder().unwrap_err();
        assert!(error.to_string().contains("topology"));
    }
}

/// Empty input axes fail validation and requested diagnostics include embeddings, blocks, and normalization.
#[test]
fn t5_recipe_rejects_empty_inputs_and_reports_full_hidden_diagnostics() {
    let device = Default::default();
    let encoder = T5EncoderRecipe::from_config(config())
        .unwrap()
        .init::<TestBackend>(&device)
        .unwrap();
    for shape in [[0, 3], [1, 0]] {
        let mut source = input(&device);
        source.token_ids = Tensor::<TestBackend, 2, Int>::zeros(shape, &device);
        assert!(
            encoder
                .embed(source)
                .unwrap_err()
                .to_string()
                .contains("non-empty")
        );
    }
    let output = encoder
        .encode(
            input(&device),
            &ForwardOptions {
                hidden_states: true,
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(output.hidden_states.len(), encoder.encoder.block.len() + 2);
    assert!(
        output
            .hidden_states
            .iter()
            .all(|hidden| hidden.dims() == [1, 3, 4])
    );
}
