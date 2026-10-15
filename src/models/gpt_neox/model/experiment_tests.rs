//! GPT-NeoX through the family-generic hooked and staged experiment interfaces.

use burn::{
    backend::Flex,
    tensor::{Bool, Int, Tensor},
};

use super::*;
use crate::{experiment::AttentionPolicy, intervention::DynamicActivation};

/// Deterministic CPU backend; staged and complete passes must agree bit for bit.
type TestBackend = Flex<f32, i32>;

/// A small randomly initialized model with two blocks and half-head RoPE.
fn model() -> GptNeoXForCausalLm<TestBackend> {
    let device = Default::default();
    TestBackend::seed(&device, 3);
    GptNeoXForCausalLm::init(
        GptNeoXConfig {
            vocab_size: 13,
            hidden_size: 8,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            intermediate_size: 12,
            rotary_pct: 0.5,
            max_position_embeddings: 16,
            initializer_range: 0.5,
            ..Default::default()
        },
        &device,
    )
    .unwrap()
}

/// A right-padded two-row batch, optionally with explicit RoPE positions.
fn input(ids: [[i64; 3]; 2], positions: Option<[[i64; 3]; 2]>) -> CausalLmInput<TestBackend> {
    let device = Default::default();
    CausalLmInput {
        token_ids: Tensor::<TestBackend, 2, Int>::from_ints(ids, &device),
        token_mask: Some(TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
            [[true, true, true], [true, true, false]].into(),
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

/// The hooked trait runs the complete causal pass and captures every listed point.
#[test]
fn hooked_trait_runs_the_inherent_pass() {
    let model = model();
    let (output, cache) = run_with_cache(&model, input([[1, 4, 2], [3, 5, 0]], None)).unwrap();
    let expected = model
        .forward(
            input([[1, 4, 2], [3, 5, 0]], None),
            &ForwardOptions::default(),
        )
        .unwrap();
    assert_eq!(output.logits.into_data(), expected.logits.into_data());
    assert_eq!(cache.len(), model.activation_points().len());
    assert_eq!(cache[0].0, Stack::Decoder.embedding().info());
    assert_eq!(cache.last().unwrap().0, Stack::Decoder.final_norm().info());
}

/// Staged execution reproduces the complete pass, resumes across ranges, and patches residuals.
#[test]
fn staged_trait_reproduces_resumes_and_patches_the_complete_pass() {
    let model = model();
    for positions in [None, Some([[0, 2, 5], [1, 3, 9]])] {
        let expected = model
            .forward_hidden(input([[1, 4, 2], [3, 5, 0]], positions))
            .unwrap();
        let whole = model
            .run_stack(input([[1, 4, 2], [3, 5, 0]], positions))
            .unwrap();
        assert!(whole.normalized);
        assert_eq!(whole.hidden.into_data(), expected.clone().into_data());
        let state = StagedModel::embed(&model, input([[1, 4, 2], [3, 5, 0]], positions)).unwrap();
        let context = if positions.is_some() { 10 } else { 3 };
        assert_eq!(state.context.evaluated_context, context);
        let state = StagedModel::run_layers(&model, state, 0..1).unwrap();
        let state = StagedModel::run_layers(&model, state, 1..2).unwrap();
        let split = StagedModel::normalize(&model, state).unwrap();
        assert_eq!(split.hidden.into_data(), expected.into_data());
    }

    // Patching the whole residual entering block 1 with another input's
    // residual reproduces that input's output (same mask and positions).
    let donor = StagedModel::run_layers(
        &model,
        StagedModel::embed(&model, input([[7, 8, 9], [10, 11, 12]], None)).unwrap(),
        0..1,
    )
    .unwrap();
    let state = StagedModel::run_layers(
        &model,
        StagedModel::embed(&model, input([[1, 4, 2], [3, 5, 0]], None)).unwrap(),
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
        .forward_hidden(input([[7, 8, 9], [10, 11, 12]], None))
        .unwrap();
    assert_eq!(patched.hidden.into_data(), other.into_data());
}

/// Staged calls reject bad cursors, out-of-range edited positions, and attention policies.
#[test]
fn staged_trait_rejects_invalid_states_and_attention_policies() {
    let model = model();
    let state = StagedModel::embed(&model, input([[1, 4, 2], [3, 5, 0]], None)).unwrap();
    assert!(StagedModel::run_layers(&model, state.clone(), 1..2).is_err());
    assert!(StagedModel::normalize(&model, state.clone()).is_err());

    let mut policed = state.clone();
    policed.context = policed
        .context
        .with_attention(AttentionPolicy::bidirectional());
    let error = StagedModel::run_layers(&model, policed, 0..2).unwrap_err();
    assert!(matches!(error, Error::BackendCapability(_)), "{error}");

    // Positions beyond the RoPE table are caught even when edited after embedding.
    let mut moved = state.clone();
    moved.context.positions =
        Tensor::<TestBackend, 2, Int>::from_ints([[0, 1, 20], [0, 1, 2]], &Default::default());
    moved.context.evaluated_context = 21;
    let error = StagedModel::run_layers(&model, moved, 0..2).unwrap_err();
    assert!(error.to_string().contains("position"), "{error}");

    let finished = StagedModel::normalize(
        &model,
        StagedModel::run_layers(&model, state, 0..2).unwrap(),
    )
    .unwrap();
    assert!(StagedModel::normalize(&model, finished.clone()).is_err());
    assert!(StagedModel::run_layers(&model, finished, 2..2).is_err());
}
