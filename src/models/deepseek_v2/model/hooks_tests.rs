//! DeepSeek-V2 activation hooks: identity parity, interventions, staging, and policies.

use burn::{
    backend::Flex,
    tensor::{Bool, Int, Tensor, TensorData, Tolerance},
};

use super::super::tests_support;
use super::*;
use crate::{LogitsSelection, architecture::TokenMask};

/// Deterministic F32 CPU backend with I32 indices.
type TestBackend = Flex<f32, i32>;

/// The low-rank-query HF reference topology (one dense and one sparse block).
fn model() -> DeepseekV2ForCausalLm<TestBackend> {
    tests_support::reference_model(tests_support::reference_v3_config(), &Default::default())
}

/// A padded two-row prompt with explicit in-range positions.
fn input() -> CausalLmInput<TestBackend> {
    let device = Default::default();
    CausalLmInput {
        token_ids: Tensor::<TestBackend, 2, Int>::from_ints(
            [[3, 17, 5, 9], [0, 7, 1, 22]],
            &device,
        ),
        token_mask: Some(TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
            [[true; 4], [false, true, true, true]].into(),
            &device,
        ))),
        position_ids: None,
    }
}

/// Requests every logit.
fn all_logits() -> ForwardOptions {
    ForwardOptions {
        logits: LogitsSelection::All,
        ..ForwardOptions::default()
    }
}

/// The catalog names each block's MLA, router, and branch points in emission order.
#[test]
fn activation_points_cover_every_block_in_emission_order() {
    let model = model();
    let names = model
        .activation_points()
        .into_iter()
        .map(|point| point.name)
        .collect::<Vec<_>>();
    assert_eq!(names.first().unwrap(), "decoder.embedding");
    assert_eq!(names.last().unwrap(), "decoder.final_norm");
    let sparse = names
        .iter()
        .position(|name| name == "decoder.layers.1.mlp.router_logits")
        .unwrap();
    assert_eq!(names[sparse + 1], "decoder.layers.1.mlp.routed_output");
    assert_eq!(names[sparse + 2], "decoder.layers.1.mlp.shared_output");
    assert!(names.contains(&"decoder.layers.0.mlp.pre_activation".to_owned()));
    assert!(!names.contains(&"decoder.layers.1.mlp.pre_activation".to_owned()));
    let latent = names
        .iter()
        .position(|name| name.ends_with("layers.0.attention.query_latent"))
        .expect("q-LoRA exposes its latent");
    assert!(names[latent + 1].ends_with("layers.0.attention.kv_latent"));
    // Every name is unique, so each point is addressable.
    let mut sorted = names.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(sorted.len(), names.len());
}

/// Capturing every point (forcing full-plane attention) leaves logits unchanged.
#[test]
fn identity_hooks_reproduce_the_unhooked_forward() {
    let model = model();
    let expected = model.forward(input(), &all_logits()).unwrap().logits;
    let mut hooks = Hooks::new();
    let captures = model
        .activation_points()
        .into_iter()
        .map(|point| hooks.capture_dynamic(point))
        .collect::<Vec<_>>();
    let actual = model
        .forward_with_hooks(input(), &all_logits(), &mut hooks)
        .unwrap()
        .logits;
    actual
        .into_data()
        .assert_approx_eq::<f32>(&expected.into_data(), Tolerance::absolute(1e-6));
    for capture in &captures {
        let activation = hooks.get_dynamic(capture).unwrap();
        assert_eq!(activation.shape.len(), capture.point().rank());
    }
    // Hooks with no actions keep the bounded path and are bit-identical.
    let mut empty = Hooks::new();
    let quiet = model
        .forward_with_hooks(input(), &all_logits(), &mut empty)
        .unwrap()
        .logits;
    quiet.into_data().assert_approx_eq::<f32>(
        &model
            .forward(input(), &all_logits())
            .unwrap()
            .logits
            .into_data(),
        Tolerance::absolute(0.0),
    );
}

/// Router-logit edits change routing, and branch captures sum to the block output.
#[test]
fn router_and_branch_points_are_live() {
    let model = model();
    let baseline = model.forward(input(), &all_logits()).unwrap().logits;
    let layer = Layer::decoder(1);
    let mut hooks = Hooks::new();
    let routed = hooks.capture(super::super::moe::routed_output_point(layer));
    let shared = hooks.capture(super::super::moe::shared_output_point(layer));
    let output = hooks.capture(layer.mlp_output());
    model
        .forward_with_hooks(input(), &all_logits(), &mut hooks)
        .unwrap();
    (hooks.get(&routed).unwrap() + hooks.get(&shared).unwrap())
        .into_data()
        .assert_approx_eq::<f32>(
            &hooks.get(&output).unwrap().into_data(),
            Tolerance::absolute(1e-6),
        );

    let mut hooks = Hooks::new();
    // Make expert 7 dominate every token's selection.
    hooks.edit(super::super::moe::router_logits_point(layer), |logits| {
        let [batch, tokens, experts] = logits.dims();
        let boost = Tensor::<TestBackend, 1>::from_data(
            TensorData::new(
                (0..experts)
                    .map(|e| if e == 7 { 50.0 } else { 0.0 })
                    .collect(),
                [experts],
            ),
            &logits.device(),
        );
        logits
            + boost
                .reshape([1, 1, experts])
                .expand([batch, tokens, experts])
    });
    let edited = model
        .forward_with_hooks(input(), &all_logits(), &mut hooks)
        .unwrap()
        .logits;
    let difference = (edited - baseline).abs().max().into_scalar();
    assert!(
        difference > 1e-3,
        "router edit had no effect ({difference})"
    );
}

/// Zeroing the KV latent of block 0 changes every later logit.
#[test]
fn kv_latent_ablation_reaches_the_logits() {
    let model = model();
    let baseline = model.forward(input(), &all_logits()).unwrap().logits;
    let mut hooks = Hooks::new();
    hooks.edit(
        Layer::decoder(0).self_attention().kv_latent(),
        |latent: Tensor<TestBackend, 3>| latent.zeros_like(),
    );
    let ablated = model
        .forward_with_hooks(input(), &all_logits(), &mut hooks)
        .unwrap()
        .logits;
    assert!((ablated - baseline).abs().max().into_scalar() > 1e-4);
}

/// Probabilities and attention-internal points are mutually exclusive.
#[test]
fn probabilities_with_internal_points_are_rejected() {
    let model = model();
    let mut hooks = Hooks::new();
    let _pattern = hooks.capture(Layer::decoder(0).self_attention().pattern());
    let error = model
        .forward_with_hooks(
            input(),
            &ForwardOptions {
                attentions: true,
                ..ForwardOptions::default()
            },
            &mut hooks,
        )
        .unwrap_err();
    assert!(matches!(error, Error::BackendCapability(message) if message.contains("pattern")));
}

/// Staged execution equals the fused forward, and a residual patch equals a hook edit.
#[test]
fn staged_execution_matches_forward_and_supports_residual_patching() {
    let model = model();
    let expected = model
        .forward(
            input(),
            &ForwardOptions {
                hidden_states: true,
                ..ForwardOptions::default()
            },
        )
        .unwrap()
        .final_hidden_state
        .unwrap();
    let state = model.embed(input()).unwrap();
    let state = model.run_layers(state, 0..1).unwrap();
    let state = model.run_layers(state, 1..2).unwrap();
    let state = model.normalize(state).unwrap();
    assert!(state.normalized && state.next_layer == 2);
    state
        .hidden
        .into_data()
        .assert_approx_eq::<f32>(&expected.into_data(), Tolerance::absolute(1e-6));

    let donor = Tensor::<TestBackend, 3>::ones([2, 4, 8], &Default::default()) * 0.25;
    let state = model.embed(input()).unwrap();
    let mut state = model.run_layers(state, 0..1).unwrap();
    state.hidden = donor.clone();
    let patched = model
        .normalize(model.run_layers(state, 1..2).unwrap())
        .unwrap()
        .hidden;
    let mut hooks = Hooks::new();
    hooks.replace(Layer::decoder(1).residual_pre(), donor);
    let hooked = model
        .forward_with_hooks(
            input(),
            &ForwardOptions {
                hidden_states: true,
                ..ForwardOptions::default()
            },
            &mut hooks,
        )
        .unwrap()
        .final_hidden_state
        .unwrap();
    patched
        .into_data()
        .assert_approx_eq::<f32>(&hooked.into_data(), Tolerance::absolute(1e-6));
}

/// Custom policies replace causality; an explicit causal `allowed` plane changes nothing.
#[test]
fn staged_attention_policies_are_honored() {
    let model = model();
    let device = Default::default();
    let run = |context_edit: &dyn Fn(&mut ExecutionContext<TestBackend>)| {
        let mut state = model.embed(input()).unwrap();
        context_edit(&mut state.context);
        model
            .normalize(model.run_layers(state, 0..2).unwrap())
            .unwrap()
            .hidden
    };
    let causal = run(&|_| {});
    let lower = Tensor::<TestBackend, 2, Bool>::from_bool(
        [
            [true, false, false, false],
            [true, true, false, false],
            [true, true, true, false],
            [true, true, true, true],
        ]
        .into(),
        &device,
    )
    .reshape([1, 1, 4, 4]);
    let explicit = run(&|context| {
        context.attention = Some(AttentionPolicy::custom(lower.clone()));
    });
    explicit
        .into_data()
        .assert_approx_eq::<f32>(&causal.clone().into_data(), Tolerance::absolute(1e-6));
    let bidirectional = run(&|context| {
        context.layer_attention = vec![(Layer::decoder(0), AttentionPolicy::bidirectional())];
    });
    assert!((bidirectional - causal).abs().max().into_scalar() > 1e-5);
}

/// Invalid stage boundaries, foreign layers, and double normalization are rejected.
#[test]
fn staged_execution_rejects_invalid_requests() {
    let model = model();
    let state = model.embed(input()).unwrap();
    assert!(
        model.run_layers(state.clone(), 1..2).is_err(),
        "skipped block"
    );
    assert!(
        model.run_layers(state.clone(), 0..3).is_err(),
        "past the stack"
    );
    let mut foreign = state.clone();
    foreign.context.layer_attention = vec![(Layer::decoder(9), AttentionPolicy::causal())];
    assert!(model.run_layers(foreign, 0..2).is_err());
    let normalized = model.normalize(state.clone()).unwrap();
    assert!(model.normalize(normalized.clone()).is_err());
    assert!(model.run_layers(normalized, 0..1).is_err());
    // A hook on a block outside the staged range is not in scope.
    let mut hooks = Hooks::new();
    let _ = hooks.capture(Layer::decoder(1).residual_pre());
    let error = model
        .run_layers_with_hooks(state, 0..1, &mut hooks)
        .unwrap_err();
    assert!(error.to_string().contains("unavailable"), "{error}");
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
    assert_eq!(StagedModel::<TestBackend>::num_layers(&model), 2);
    assert_eq!(
        StagedModel::<TestBackend>::layer(&model, 1),
        Layer::decoder(1)
    );
    let hidden = model.run_stack(input()).unwrap().hidden;
    assert_eq!(hidden.dims(), [2, 4, 8]);
}

/// Edits stay differentiable: gradients reach the embedding through an edited point.
#[cfg(feature = "autodiff")]
#[test]
fn hooked_forward_is_differentiable() {
    use burn::backend::Autodiff;

    /// Reverse-mode backend over the deterministic CPU backend.
    type AdBackend = Autodiff<TestBackend>;
    let device = Default::default();
    let model =
        tests_support::reference_model::<AdBackend>(tests_support::reference_v3_config(), &device);
    let mut hooks = Hooks::new();
    hooks.edit(
        Layer::decoder(0).self_attention().kv_latent(),
        |latent: Tensor<AdBackend, 3>| latent * 0.5,
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
