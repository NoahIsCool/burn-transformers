//! Muse-Glimmer activation hooks: identity parity, interventions, staging, and policies.

use burn::{
    backend::Flex,
    module::{Module, ModuleMapper, Param},
    tensor::{Bool, Int, Tensor, TensorData, Tolerance},
};

use super::super::{gate_point, post_attention_norm_point};
use super::*;
use crate::{
    LogitsSelection,
    architecture::TokenMask,
    checkpoint::TensorSink,
    experiment::AttentionPolicy,
    models::muse_glimmer::{MuseGlimmerCheckpointOptions, MuseGlimmerTextConfig},
};

/// Deterministic F32 CPU backend with I32 indices.
type TestBackend = Flex<f32, i32>;

/// Multiplies every floating parameter by a constant so activations are O(1).
struct Scale(
    /// Factor applied to every floating parameter.
    f32,
);

impl ModuleMapper<TestBackend> for Scale {
    /// Scales one parameter.
    fn map_float<const D: usize>(
        &mut self,
        param: Param<Tensor<TestBackend, D>>,
    ) -> Param<Tensor<TestBackend, D>> {
        let factor = self.0;
        param.map(|tensor| tensor * factor)
    }
}

/// A four-layer local/local/local/global miniature with deterministic weights.
fn model() -> MuseGlimmerForCausalLm<TestBackend> {
    let config = MuseGlimmerTextConfig {
        vocab_size: 32,
        hidden_size: 8,
        intermediate_size: 12,
        num_hidden_layers: 4,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: 4,
        max_position_embeddings: 32,
        sliding_window: Some(3),
        bos_token_id: None,
        eos_token_id: None,
        ..Default::default()
    };
    let device = Default::default();
    let contract = config
        .checkpoint_contract(MuseGlimmerCheckpointOptions::default())
        .unwrap();
    let mut model = MuseGlimmerForCausalLm::init(config, &device).unwrap();
    for (index, expected) in contract.expected_tensors().iter().enumerate() {
        let shape = expected
            .shape
            .iter()
            .map(|&dimension| dimension as usize)
            .collect::<Vec<_>>();
        let values = (0..shape.iter().product())
            .map(|element: usize| ((element + index * 11) % 31) as f32 * 0.002 - 0.03)
            .collect::<Vec<_>>();
        TensorSink::write_tensor(&mut model, &expected.name, TensorData::new(values, shape))
            .unwrap();
    }
    model.map(&mut Scale(20.0))
}

/// A padded two-row prompt.
fn input() -> CausalLmInput<TestBackend> {
    let device = Default::default();
    CausalLmInput {
        token_ids: Tensor::<TestBackend, 2, Int>::from_ints(
            [[3, 17, 5, 9, 30], [0, 7, 1, 22, 4]],
            &device,
        ),
        token_mask: Some(TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
            [[true; 5], [false, true, true, true, true]].into(),
            &device,
        ))),
        position_ids: None,
    }
}

/// Requests every logit and the final hidden state.
fn options() -> ForwardOptions {
    ForwardOptions {
        logits: LogitsSelection::All,
        hidden_states: true,
        ..ForwardOptions::default()
    }
}

/// The catalog lists every block's gate and sandwich-norm points in emission order.
#[test]
fn activation_points_cover_gates_and_sandwich_norms() {
    let names = model()
        .activation_points()
        .into_iter()
        .map(|point| point.name)
        .collect::<Vec<_>>();
    assert_eq!(names.len(), 2 + 4 * 16);
    let gate = names
        .iter()
        .position(|name| name == "decoder.layers.2.attention.gate_logits")
        .unwrap();
    assert_eq!(names[gate - 1], "decoder.layers.2.attention.head_context");
    assert_eq!(names[gate + 1], "decoder.layers.2.attention.gate");
    assert_eq!(names[gate + 2], "decoder.layers.2.attention.output");
    assert_eq!(names[gate + 3], "decoder.layers.2.post_attention_norm");
    assert!(names.contains(&"decoder.layers.3.post_feedforward_norm".to_owned()));
}

/// Capturing every point leaves logits bit-identical to the unhooked forward.
#[test]
fn identity_hooks_reproduce_the_unhooked_forward() {
    let model = model();
    let expected = model.forward(input(), &options()).unwrap();
    let mut hooks = Hooks::new();
    let captures = model
        .activation_points()
        .into_iter()
        .map(|point| hooks.capture_dynamic(point))
        .collect::<Vec<_>>();
    let actual = model
        .forward_with_hooks(input(), &options(), &mut hooks)
        .unwrap();
    actual
        .logits
        .into_data()
        .assert_approx_eq::<f32>(&expected.logits.into_data(), Tolerance::absolute(0.0));
    for capture in &captures {
        let activation = hooks.get_dynamic(capture).unwrap();
        assert_eq!(activation.shape.len(), capture.point().rank());
    }
    // The final-norm capture is the final hidden state.
    let final_norm = hooks
        .get_dynamic(captures.last().unwrap())
        .unwrap()
        .into_tensor::<3>()
        .unwrap();
    final_norm.into_data().assert_approx_eq::<f32>(
        &expected.final_hidden_state.unwrap().into_data(),
        Tolerance::absolute(0.0),
    );
}

/// Saturating a layer's gate shut removes its attention branch exactly.
#[test]
fn closing_the_gate_removes_the_attention_branch() {
    let model = model();
    let layer = Layer::decoder(1);
    let mut hooks = Hooks::new();
    hooks.edit(gate_point(layer), |gate: Tensor<TestBackend, 3>| {
        gate.zeros_like()
    });
    let attended = hooks.capture(layer.attention_output());
    let normalized = hooks.capture(post_attention_norm_point(layer));
    model
        .forward_with_hooks(input(), &options(), &mut hooks)
        .unwrap();
    // `o_proj` has no bias, so a closed gate yields an exact zero branch.
    for capture in [&attended, &normalized] {
        assert!(
            hooks
                .get(capture)
                .unwrap()
                .into_data()
                .iter::<f32>()
                .all(|value| value == 0.0)
        );
    }
}

/// Staged execution equals the fused forward; a residual patch equals a hook edit.
#[test]
fn staged_execution_matches_forward_and_supports_patching() {
    let model = model();
    let expected = model
        .forward(input(), &options())
        .unwrap()
        .final_hidden_state
        .unwrap();
    let state = model.embed(input()).unwrap();
    let state = model.run_layers(state, 0..2).unwrap();
    let state = model.run_layers(state, 2..4).unwrap();
    let state = model.normalize(state).unwrap();
    state
        .hidden
        .into_data()
        .assert_approx_eq::<f32>(&expected.into_data(), Tolerance::absolute(1e-6));

    let donor = Tensor::<TestBackend, 3>::ones([2, 5, 8], &Default::default()) * 0.5;
    let mut state = model
        .run_layers(model.embed(input()).unwrap(), 0..3)
        .unwrap();
    state.hidden = donor.clone();
    let patched = model
        .normalize(model.run_layers(state, 3..4).unwrap())
        .unwrap()
        .hidden;
    let mut hooks = Hooks::new();
    hooks.replace(Layer::decoder(3).residual_pre(), donor);
    let hooked = model
        .forward_with_hooks(input(), &options(), &mut hooks)
        .unwrap()
        .final_hidden_state
        .unwrap();
    patched
        .into_data()
        .assert_approx_eq::<f32>(&hooked.into_data(), Tolerance::absolute(1e-6));
}

/// An override replaces causality and windows; bidirectional attention changes outputs.
#[test]
fn staged_policies_override_layer_connectivity() {
    let model = model();
    let run = |edit: &dyn Fn(&mut ExecutionContext<TestBackend>)| {
        let mut state = model.embed(input()).unwrap();
        edit(&mut state.context);
        model
            .normalize(model.run_layers(state, 0..4).unwrap())
            .unwrap()
            .hidden
    };
    let default = run(&|_| {});
    // Causal override on the global layer reproduces its default connectivity.
    let causal_global = run(&|context| {
        context.layer_attention = vec![(Layer::decoder(3), AttentionPolicy::causal())];
    });
    causal_global
        .into_data()
        .assert_approx_eq::<f32>(&default.clone().into_data(), Tolerance::absolute(1e-6));
    let bidirectional = run(&|context| {
        context.layer_attention = vec![(Layer::decoder(3), AttentionPolicy::bidirectional())];
    });
    assert!((bidirectional - default).abs().max().into_scalar() > 1e-5);
}

/// Invalid stage boundaries, foreign layers, out-of-scope hooks, and double norms fail.
#[test]
fn staged_execution_rejects_invalid_requests() {
    let model = model();
    let state = model.embed(input()).unwrap();
    assert!(model.run_layers(state.clone(), 1..2).is_err());
    assert!(model.run_layers(state.clone(), 0..5).is_err());
    let mut foreign = state.clone();
    foreign.context.layer_attention = vec![(Layer::decoder(7), AttentionPolicy::causal())];
    assert!(model.run_layers(foreign, 0..4).is_err());
    let normalized = model.normalize(state.clone()).unwrap();
    assert!(model.normalize(normalized).is_err());
    let mut hooks = Hooks::new();
    let _ = hooks.capture(Layer::decoder(2).residual_pre());
    let error = model
        .run_layers_with_hooks(state, 0..1, &mut hooks)
        .unwrap_err();
    assert!(error.to_string().contains("unavailable"), "{error}");
    assert!(matches!(
        model.forward_with_hooks(
            input(),
            &ForwardOptions {
                attentions: true,
                ..ForwardOptions::default()
            },
            &mut Hooks::new()
        ),
        Err(Error::BackendCapability(_))
    ));
}

/// The family is usable through the generic hook and staging traits.
#[test]
fn generic_traits_dispatch_to_the_family() {
    /// Returns the number of points through the generic trait.
    fn count<B: Backend, M: HookedModel<B>>(model: &M) -> usize {
        model.activation_points().unwrap().len()
    }
    let model = model();
    assert_eq!(count(&model), model.activation_points().len());
    assert_eq!(StagedModel::<TestBackend>::num_layers(&model), 4);
    assert_eq!(model.run_stack(input()).unwrap().hidden.dims(), [2, 5, 8]);
}

/// Gate edits stay differentiable: gradients reach the embedding table.
#[cfg(feature = "autodiff")]
#[test]
fn hooked_forward_is_differentiable() {
    use burn::backend::Autodiff;

    /// Reverse-mode backend over the deterministic CPU backend.
    type AdBackend = Autodiff<TestBackend>;
    let device = Default::default();
    let config = MuseGlimmerTextConfig {
        vocab_size: 32,
        hidden_size: 8,
        intermediate_size: 12,
        num_hidden_layers: 4,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: 4,
        max_position_embeddings: 32,
        sliding_window: Some(3),
        bos_token_id: None,
        eos_token_id: None,
        ..Default::default()
    };
    let model = MuseGlimmerForCausalLm::<AdBackend>::init(config, &device).unwrap();
    let mut hooks = Hooks::new();
    hooks.edit(
        gate_point(Layer::decoder(0)),
        |gate: Tensor<AdBackend, 3>| gate * 0.5,
    );
    let logits = model
        .forward_with_hooks(
            CausalLmInput {
                token_ids: Tensor::<AdBackend, 2, Int>::from_ints([[3, 17, 5]], &device),
                token_mask: None,
                position_ids: None,
            },
            &ForwardOptions::default(),
            &mut hooks,
        )
        .unwrap()
        .logits;
    let gradients = logits.sum().backward();
    assert!(model.model.embed_tokens.weight.grad(&gradients).is_some());
}
