//! GPT-2 through the family-generic hooked and staged experiment interfaces.

use burn::{
    backend::Flex,
    tensor::{Bool, Int, Tensor},
};

use super::*;
use crate::{experiment::AttentionPolicy, intervention::DynamicActivation};

/// Deterministic CPU backend; staged and complete passes must agree bit for bit.
type TestBackend = Flex<f32, i32>;

/// A small randomly initialized causal language model with two blocks.
fn model() -> Gpt2ForCausalLm<TestBackend> {
    Gpt2ForCausalLm::init(
        Gpt2Config {
            vocab_size: 32,
            n_embd: 12,
            n_layer: 2,
            n_head: 3,
            n_positions: 16,
            n_inner: Some(20),
            ..Gpt2Config::default()
        },
        &Default::default(),
    )
    .unwrap()
}

/// A right-padded two-row batch, optionally with explicit positions.
fn input(ids: [[i64; 4]; 2], positions: Option<[[i64; 4]; 2]>) -> CausalLmInput<TestBackend> {
    let device = Default::default();
    CausalLmInput {
        token_ids: Tensor::<TestBackend, 2, Int>::from_ints(ids, &device),
        token_mask: Some(TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
            [[true, true, true, true], [true, true, true, false]].into(),
            &device,
        ))),
        position_ids: positions
            .map(|positions| Tensor::<TestBackend, 2, Int>::from_ints(positions, &device)),
    }
}

/// Every captured point of one run, in catalog order.
type ActivationCache<B> = Vec<(PointInfo, DynamicActivation<B>)>;

/// Generic activation cache: captures every listed point of any hooked model.
fn run_with_cache<B: Backend, M: HookedModel<B>>(
    model: &M,
    input: M::Input,
) -> Result<(M::Output, ActivationCache<B>)> {
    let mut hooks = Hooks::new();
    let handles = model
        .activation_points()?
        .into_iter()
        .map(|point| hooks.capture_dynamic(point))
        .collect::<Vec<_>>();
    let output = model.forward_with_hooks(input, &ForwardOptions::default(), &mut hooks)?;
    let cache = handles
        .iter()
        .map(|handle| Ok((handle.point().clone(), hooks.get_dynamic(handle)?)))
        .collect::<Result<_>>()?;
    Ok((output, cache))
}

/// The hooked trait runs the complete pass and captures every listed point.
#[test]
fn hooked_trait_runs_the_inherent_pass() {
    let model = model();
    let (output, cache) =
        run_with_cache(&model, input([[1, 2, 3, 4], [5, 6, 7, 0]], None)).unwrap();
    let expected = model
        .forward(
            input([[1, 2, 3, 4], [5, 6, 7, 0]], None),
            &ForwardOptions::default(),
        )
        .unwrap();
    assert_eq!(output.logits.into_data(), expected.logits.into_data());
    assert_eq!(cache.len(), model.activation_points().len());
    assert_eq!(
        HookedModel::activation_points(&model).unwrap(),
        model.activation_points()
    );
}

/// Staged execution reproduces the complete pass, resumes across ranges, and patches residuals.
#[test]
fn staged_trait_reproduces_resumes_and_patches_the_complete_pass() {
    let model = model();
    let positions = Some([[0, 1, 2, 3], [3, 4, 5, 6]]);
    for positions in [None, positions] {
        let expected = model
            .model
            .forward(&input([[1, 2, 3, 4], [5, 6, 7, 0]], positions), false)
            .unwrap()
            .last_hidden_state;
        let whole = model
            .run_stack(input([[1, 2, 3, 4], [5, 6, 7, 0]], positions))
            .unwrap();
        assert!(whole.normalized);
        assert_eq!(whole.hidden.into_data(), expected.clone().into_data());
        let state =
            StagedModel::embed(&model, input([[1, 2, 3, 4], [5, 6, 7, 0]], positions)).unwrap();
        assert_eq!(
            state.context.evaluated_context,
            if positions.is_some() { 7 } else { 4 }
        );
        let state = StagedModel::run_layers(&model, state, 0..1).unwrap();
        let state = StagedModel::run_layers(&model, state, 1..2).unwrap();
        let split = StagedModel::normalize(&model, state).unwrap();
        assert_eq!(split.hidden.into_data(), expected.into_data());
    }

    // Patching the whole residual entering block 1 with another input's
    // residual reproduces that input's output (same mask and positions).
    let donor = StagedModel::run_layers(
        &model,
        StagedModel::embed(&model, input([[9, 8, 7, 6], [5, 4, 3, 2]], None)).unwrap(),
        0..1,
    )
    .unwrap();
    let state = StagedModel::run_layers(
        &model,
        StagedModel::embed(&model, input([[1, 2, 3, 4], [5, 6, 7, 0]], None)).unwrap(),
        0..1,
    )
    .unwrap();
    let patched = HiddenState {
        hidden: donor.hidden,
        ..state
    };
    let patched = StagedModel::normalize(
        &model,
        StagedModel::run_layers(&model, patched, 1..2).unwrap(),
    )
    .unwrap();
    let other = model
        .model
        .forward(&input([[9, 8, 7, 6], [5, 4, 3, 2]], None), false)
        .unwrap()
        .last_hidden_state;
    assert_eq!(patched.hidden.into_data(), other.into_data());
}

/// Each staged scope emits only its own points, so hooks act on exactly one stage.
#[test]
fn staged_hook_scopes_emit_only_their_stage_points() {
    let model = model();
    let mut hooks = Hooks::new();
    let embedding = hooks.capture(Stack::Decoder.embedding());
    let state = model
        .embed_with_hooks(input([[1, 2, 3, 4], [5, 6, 7, 0]], None), &mut hooks)
        .unwrap();
    assert_eq!(hooks.get(&embedding).unwrap().dims(), [2, 4, 12]);

    let mut hooks = Hooks::new();
    let block = hooks.capture(Layer::decoder(1).mlp_output());
    hooks.edit(Layer::decoder(1).residual_post(), |residual| residual * 0.0);
    let state = model
        .run_layers_with_hooks(state, 0..2, &mut hooks)
        .unwrap();
    assert_eq!(hooks.get(&block).unwrap().dims(), [2, 4, 12]);
    assert!(
        state
            .hidden
            .clone()
            .into_data()
            .to_vec::<f32>()
            .unwrap()
            .iter()
            .all(|&value| value == 0.0)
    );
    // A point outside the scope's blocks is rejected inside the scope.
    let mut hooks = Hooks::new();
    hooks.capture(Layer::decoder(0).mlp_output());
    let fresh = StagedModel::embed(&model, input([[1, 2, 3, 4], [5, 6, 7, 0]], None)).unwrap();
    let fresh = StagedModel::run_layers(&model, fresh, 0..1).unwrap();
    assert!(
        model
            .run_layers_with_hooks(fresh, 1..2, &mut hooks)
            .is_err()
    );
}

/// Staged calls reject skipped ranges, repeated normalization, and policies GPT-2 cannot apply.
#[test]
fn staged_trait_rejects_invalid_cursors_and_attention_policies() {
    let model = model();
    let state = StagedModel::embed(&model, input([[1, 2, 3, 4], [5, 6, 7, 0]], None)).unwrap();
    assert!(StagedModel::run_layers(&model, state.clone(), 1..2).is_err());
    assert!(StagedModel::normalize(&model, state.clone()).is_err());
    let mut policed = state.clone();
    policed.context = policed
        .context
        .with_attention(AttentionPolicy::bidirectional());
    let error = StagedModel::run_layers(&model, policed, 0..2).unwrap_err();
    assert!(matches!(error, Error::BackendCapability(_)), "{error}");
    let finished = StagedModel::normalize(
        &model,
        StagedModel::run_layers(&model, state, 0..2).unwrap(),
    )
    .unwrap();
    assert!(StagedModel::normalize(&model, finished.clone()).is_err());
    assert!(StagedModel::run_layers(&model, finished, 2..2).is_err());
}
