//! Nemotron-H activation catalogs, hooked execution, and staged execution.

use burn::{
    backend::Flex,
    tensor::{Bool, Tensor, Tolerance},
};

use crate::{
    architecture::TokenMask, experiment::AttentionPolicy,
    models::nemotron_h::model::tests::tiny_config,
};

use super::*;

/// CPU Flex backend with F32 activations and I32 indices.
type TestBackend = Flex<f32, i32>;

/// A padded two-row batch over the tiny four-block schedule.
fn padded_input() -> CausalLmInput<TestBackend> {
    let device = Default::default();
    CausalLmInput {
        token_ids: Tensor::from_ints([[0, 3, 4, 5], [6, 7, 8, 9]], &device),
        token_mask: Some(TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
            [[false, true, true, true], [true, true, true, true]].into(),
            &device,
        ))),
        position_ids: None,
    }
}

/// Options retaining the final hidden state and every block output.
fn hidden_options() -> ForwardOptions {
    ForwardOptions {
        hidden_states: true,
        logits: crate::LogitsSelection::All,
        ..Default::default()
    }
}

/// Every block kind advertises exactly its own points, in emission order.
#[test]
fn activation_points_follow_block_kinds() {
    let device = Default::default();
    let model = NemotronHForCausalLm::<TestBackend>::init(tiny_config(false), &device).unwrap();
    let names = model
        .activation_points()
        .into_iter()
        .map(|point| point.name)
        .collect::<Vec<_>>();
    let expected = [
        "decoder.embedding",
        "decoder.layers.0.residual.pre",
        "decoder.layers.0.attention.input",
        "decoder.layers.0.attention.output",
        "decoder.layers.0.residual.post",
        "decoder.layers.1.residual.pre",
        "decoder.layers.1.attention.input",
        "decoder.layers.1.attention.query",
        "decoder.layers.1.attention.key",
        "decoder.layers.1.attention.value",
        "decoder.layers.1.attention.head_context",
        "decoder.layers.1.attention.output",
        "decoder.layers.1.residual.post",
        "decoder.layers.2.residual.pre",
        "decoder.layers.2.mlp.input",
        "decoder.layers.2.mlp.pre_activation",
        "decoder.layers.2.mlp.activation",
        "decoder.layers.2.mlp.output",
        "decoder.layers.2.residual.post",
        "decoder.layers.3.residual.pre",
        "decoder.layers.3.mlp.input",
        "decoder.layers.3.mlp.output",
        "decoder.layers.3.residual.post",
        "decoder.final_norm",
    ];
    assert_eq!(names, expected);
    assert_eq!(
        HookedModel::<TestBackend>::activation_points(&model)
            .unwrap()
            .len(),
        expected.len()
    );
}

/// Empty hooks leave every output bit-identical to the plain forward pass.
#[test]
fn empty_hooks_match_forward_bit_for_bit() {
    let device = Default::default();
    let model = NemotronHForCausalLm::<TestBackend>::init(tiny_config(false), &device).unwrap();
    let plain = model.forward(padded_input(), &hidden_options()).unwrap();
    let hooked = model
        .forward_with_hooks(padded_input(), &hidden_options(), &mut Hooks::new())
        .unwrap();
    assert_eq!(plain.logits.into_data(), hooked.logits.into_data());
    for (plain, hooked) in plain.hidden_states.iter().zip(&hooked.hidden_states) {
        assert_eq!(plain.to_data(), hooked.to_data());
    }
}

/// Every advertised point is visited, captures match the returned states, and edits propagate.
#[test]
fn captures_cover_every_point_and_edits_propagate() {
    let device = Default::default();
    let model = NemotronHForCausalLm::<TestBackend>::init(tiny_config(false), &device).unwrap();
    let mut hooks = Hooks::new();
    let handles = model
        .activation_points()
        .into_iter()
        .map(|point| hooks.capture_dynamic(point))
        .collect::<Vec<_>>();
    let output = model
        .forward_with_hooks(padded_input(), &hidden_options(), &mut hooks)
        .unwrap();
    for handle in &handles {
        assert!(hooks.get_dynamic(handle).is_ok(), "{}", handle.point().name);
    }
    // The block-two residual capture equals the returned block-two hidden state.
    let mut hooks = Hooks::new();
    let residual = hooks.capture(Layer::decoder(2).residual_post());
    model
        .forward_with_hooks(padded_input(), &hidden_options(), &mut hooks)
        .unwrap();
    assert_eq!(
        hooks.get(&residual).unwrap().into_data(),
        output.hidden_states[3].clone().into_data()
    );

    // Zeroing the attention block's output changes every later activation.
    let mut hooks = Hooks::new();
    hooks.edit(
        Layer::decoder(1).mixer_output(),
        |tensor: Tensor<TestBackend, 3>| tensor.zeros_like(),
    );
    let ablated = model
        .forward_with_hooks(padded_input(), &hidden_options(), &mut hooks)
        .unwrap();
    assert_eq!(
        ablated.hidden_states[2].clone().into_data(),
        output.hidden_states[1].clone().into_data()
    );
    assert_ne!(ablated.logits.into_data(), output.logits.into_data());

    // A point that does not exist on this schedule is rejected before execution.
    let mut hooks = Hooks::new();
    let _ = hooks.capture(Layer::decoder(0).mlp_input());
    assert!(
        model
            .forward_with_hooks(padded_input(), &hidden_options(), &mut hooks)
            .is_err()
    );
}

/// Embedding, two block ranges, and final normalization reproduce the forward pass.
#[test]
fn staged_execution_matches_forward() {
    let device = Default::default();
    let model = NemotronHForCausalLm::<TestBackend>::init(tiny_config(false), &device).unwrap();
    let expected = model
        .forward(padded_input(), &hidden_options())
        .unwrap()
        .final_hidden_state
        .unwrap();
    let state = model.model.embed(&padded_input()).unwrap();
    let state = model.model.run_layers(state, 0..2).unwrap();
    assert_eq!(state.next_layer, 2);
    assert!(model.model.run_layers(state.clone(), 0..2).is_err());
    let state = model.model.run_layers(state, 2..4).unwrap();
    let state = model.model.normalize(state).unwrap();
    assert!(model.model.normalize(state.clone()).is_err());
    state
        .hidden
        .into_data()
        .assert_approx_eq::<f32>(&expected.clone().into_data(), Tolerance::absolute(1e-6));

    // The generic trait surface runs the same stack.
    let generic = StagedModel::run_stack(&model, padded_input()).unwrap();
    generic
        .hidden
        .into_data()
        .assert_approx_eq::<f32>(&expected.into_data(), Tolerance::absolute(1e-6));
}

/// Attention honors staged policies; recurrent blocks reject them; overrides must exist.
#[test]
fn staged_policies_apply_only_where_they_are_defined() {
    let device = Default::default();
    let model = NemotronHForCausalLm::<TestBackend>::init(tiny_config(false), &device).unwrap();
    let state = model.model.embed(&padded_input()).unwrap();
    let causal = model.model.run_layers(state.clone(), 0..4).unwrap();

    let mut bidirectional = state.clone();
    bidirectional.context = bidirectional
        .context
        .with_layer_attention(Layer::decoder(1), AttentionPolicy::bidirectional());
    let bidirectional = model.model.run_layers(bidirectional, 0..4).unwrap();
    assert_ne!(
        bidirectional.hidden.into_data(),
        causal.hidden.clone().into_data()
    );

    let mut recurrent = state.clone();
    recurrent.context = recurrent
        .context
        .with_layer_attention(Layer::decoder(0), AttentionPolicy::bidirectional());
    assert!(matches!(
        model.model.run_layers(recurrent, 0..4),
        Err(Error::BackendCapability(_))
    ));

    let mut missing = state;
    missing.context = missing
        .context
        .with_layer_attention(Layer::decoder(7), AttentionPolicy::causal());
    assert!(model.model.run_layers(missing, 0..4).is_err());
}

/// Staged hooks accept only the points of the executed range.
#[test]
fn staged_hooks_are_scoped_to_their_range() {
    let device = Default::default();
    let model = NemotronHForCausalLm::<TestBackend>::init(tiny_config(false), &device).unwrap();
    let state = model.model.embed(&padded_input()).unwrap();
    let mut hooks = Hooks::new();
    let inside = hooks.capture(Layer::decoder(1).residual_post());
    let state = model
        .model
        .run_layers_with_hooks(state, 0..2, &mut hooks)
        .unwrap();
    assert_eq!(
        hooks.get(&inside).unwrap().into_data(),
        state.hidden.clone().into_data()
    );
    let mut outside = Hooks::new();
    let _ = outside.capture(Layer::decoder(0).residual_post());
    assert!(
        model
            .model
            .run_layers_with_hooks(state, 2..4, &mut outside)
            .is_err()
    );
}

/// Hooked edits stay differentiable: gradients reach the embedding through an edit.
#[cfg(feature = "autodiff")]
#[test]
fn hooked_edits_are_differentiable() {
    use burn::{backend::Autodiff, tensor::TensorData};

    /// Autodiff wrapper for gradient checks.
    type AdBackend = Autodiff<TestBackend>;
    let device = Default::default();
    let model = NemotronHForCausalLm::<AdBackend>::init(tiny_config(false), &device).unwrap();
    let mut hooks = Hooks::new();
    hooks.edit(
        Layer::decoder(2).mlp_output(),
        |tensor: Tensor<AdBackend, 3>| tensor * 0.5,
    );
    let output = model
        .forward_with_hooks(
            CausalLmInput {
                token_ids: Tensor::from_data(TensorData::new(vec![1_i64, 2, 3], [1, 3]), &device),
                token_mask: None,
                position_ids: None,
            },
            &ForwardOptions::default(),
            &mut hooks,
        )
        .unwrap();
    let gradients = output.logits.sum().backward();
    assert!(model.model.embed_tokens.weight.grad(&gradients).is_some());
}
