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
    let routed = hooks.capture(layer.routed_output());
    let shared = hooks.capture(layer.shared_output());
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
    hooks.edit(layer.router_logits(), |logits| {
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

/// Message carried by the deliberately failing hook callbacks below.
const HOOK_FAILURE: &str = "deliberate hook failure";

/// Registers one edit at `point` whose callback fails with [`HOOK_FAILURE`].
fn failing_hooks(point: &PointInfo) -> Hooks<'static, TestBackend> {
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
/// Every embedding, MLA-internal, router, branch, residual, and final-norm
/// point is emitted on both paths, and the user's error is returned rather
/// than swallowed or replaced.
#[test]
fn a_failing_edit_at_any_point_aborts_with_its_error() {
    let model = model();
    let points = model.activation_points();
    for point in &points {
        let mut hooks = failing_hooks(point);
        assert_hook_failure(
            model.forward_with_hooks(input(), &all_logits(), &mut hooks),
            point,
        );
    }
    for point in &points {
        let mut hooks = failing_hooks(point);
        let staged = if *point == Stack::Decoder.embedding().info() {
            model.embed_with_hooks(input(), &mut hooks).map(|_| ())
        } else if *point == Stack::Decoder.final_norm().info() {
            let state = model.embed(input()).unwrap();
            let state = model.run_layers(state, 0..2).unwrap();
            model.normalize_with_hooks(state, &mut hooks).map(|_| ())
        } else {
            let state = model.embed(input()).unwrap();
            model
                .run_layers_with_hooks(state, 0..2, &mut hooks)
                .map(|_| ())
        };
        assert_hook_failure(staged, point);
    }
}

/// Staged blocks reject reduced-precision latents and cross-attention sources.
#[test]
fn staged_execution_rejects_foreign_dtypes_and_cross_attention() {
    let model = model();
    let state = model.embed(input()).unwrap();
    let mut half = state.clone();
    half.hidden = half.hidden.cast(burn::tensor::DType::F16);
    for error in [
        model.run_layers(half.clone(), 0..1).unwrap_err(),
        model.normalize(half).unwrap_err(),
    ] {
        assert!(
            matches!(&error, Error::Configuration { field, reason }
                if field == "deepseek_v2.execution.hidden" && reason.contains("model dtype")),
            "{error}"
        );
    }
    let mut cross = state.clone();
    cross.context.cross_attention = Some(AttentionPolicy::causal());
    let mut layer_cross = state.clone();
    layer_cross
        .context
        .layer_cross_attention
        .push((Layer::decoder(0), AttentionPolicy::causal()));
    let mut encoded = state;
    let device = Default::default();
    encoded.context.encoder = Some(crate::experiment::EncoderContext {
        hidden: Tensor::zeros([2, 3, 8], &device),
        positions: Tensor::<TestBackend, 1, Int>::arange(0..3, &device)
            .reshape([1, 3])
            .expand([2, 3]),
        token_mask: None,
    });
    for source in [cross, layer_cross, encoded] {
        let error = model.run_layers(source, 0..1).unwrap_err();
        assert!(error.to_string().contains("cross-attention"), "{error}");
    }
}

/// A causal staged policy with an all-zero score bias reproduces the default block.
///
/// The bias moves the block onto the custom-connectivity graph; adding zero
/// to every score must not change the result beyond reduction rounding.
/// Every valid query is compared: all four positions of the unpadded row and
/// positions 1..3 of the row whose first token is padding, where the custom
/// graph must also exclude the padded key exactly as the default mask does.
#[test]
fn zero_bias_causal_policy_matches_the_default_block() {
    let model = model();
    let state = model.embed(input()).unwrap();
    let plain = model.run_layers(state.clone(), 0..2).unwrap().hidden;
    let mut biased = state;
    biased.context.attention =
        Some(AttentionPolicy::causal().with_bias(Tensor::zeros([1, 1, 4, 4], &Default::default())));
    let biased = model.run_layers(biased, 0..2).unwrap().hidden;
    // The padded query (row 1, position 0) has no valid key on either path,
    // so its output is unspecified; every other query must agree.
    for (row, positions) in [(0..1, 0..4), (1..2, 1..4)] {
        biased
            .clone()
            .slice([row.clone(), positions.clone()])
            .into_data()
            .assert_approx_eq::<f32>(
                &plain.clone().slice([row, positions]).into_data(),
                Tolerance::rel_abs(1e-5, 1e-6),
            );
    }
}

/// The generic hooked trait delegates to the family's hooked forward.
#[test]
fn hooked_trait_forward_matches_the_family_method() {
    let model = model();
    let mut family = Hooks::new();
    let family_latent = family.capture(Layer::decoder(1).self_attention().kv_latent());
    let direct = model
        .forward_with_hooks(input(), &all_logits(), &mut family)
        .unwrap();
    let mut generic = Hooks::new();
    let generic_latent = generic.capture(Layer::decoder(1).self_attention().kv_latent());
    let through_trait =
        HookedModel::forward_with_hooks(&model, input(), &all_logits(), &mut generic).unwrap();
    assert_eq!(through_trait.logits.to_data(), direct.logits.to_data());
    assert_eq!(
        generic.get(&generic_latent).unwrap().to_data(),
        family.get(&family_latent).unwrap().to_data()
    );
}

/// The reference fixture builder honors a tied head by aliasing instead of writing it.
///
/// The tied model owns no `lm_head`, and every `[batch, position]` logit row
/// equals the final normalized hidden state times the transposed embedding
/// table, `logits = h W_e^T`, the same oracle as
/// `tied_readout_matches_reshaping_linear_oracle`. The fixture's embedding is
/// nonzero, so agreement shows the readout really is the embedding.
#[test]
fn tied_reference_fixture_shares_the_embedding_readout() {
    let mut config = tests_support::reference_v3_config();
    config.tie_word_embeddings = true;
    let model = tests_support::reference_model::<TestBackend>(config, &Default::default());
    assert!(model.lm_head.is_none());
    let output = model
        .forward(
            input(),
            &ForwardOptions {
                hidden_states: true,
                ..all_logits()
            },
        )
        .unwrap();
    let logits = output.logits;
    assert_eq!(logits.dims()[..2], [2, 4]);
    let embedding = model.model.embed_tokens.weight.val();
    assert!(
        embedding
            .clone()
            .into_data()
            .iter::<f32>()
            .any(|value| value != 0.0)
    );
    let final_hidden = output
        .final_hidden_state
        .expect("hidden states were requested");
    let oracle = burn::tensor::module::linear(final_hidden, embedding.transpose(), None);
    logits
        .into_data()
        .assert_approx_eq::<f32>(&oracle.into_data(), Tolerance::rel_abs(1e-5, 1e-6));
}
