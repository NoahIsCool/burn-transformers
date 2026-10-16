//! Stateless Gemma graph: shapes, the embedding scale, soft-caps, per-layer
//! RoPE sets, input validation, padding invariance, attention precision
//! modes, hooks, staged execution, and gradients.

use burn::tensor::Tensor;

use super::test_support::{
    B, VOCABULARY, assert_close, host, input, ints, mask, max_gap, model_from, tiny_config,
    tiny_model,
};
use super::*;
use crate::{
    Error, LogitsSelection,
    architecture::{HookedModel, StagedModel},
    experiment::AttentionPolicy,
    intervention::Stack,
    models::gemma::{GemmaConfig, GemmaVariant},
};

/// Absolute tolerance between two F32 evaluations of the same mathematics.
const TOLERANCE: f32 = 2e-5;

/// Requests every column's logits and every hidden state.
fn everything() -> ForwardOptions {
    ForwardOptions {
        logits: LogitsSelection::All,
        hidden_states: true,
        ..ForwardOptions::default()
    }
}

/// Logits of `[[1..=6]]` under `model`.
fn six_token_logits(model: &GemmaForCausalLm<B>) -> Vec<f32> {
    host(
        model
            .forward(input(ints([[1, 2, 3, 4, 5, 6]]), None, None), &everything())
            .unwrap()
            .logits,
    )
}

/// Both variants allocate their head and produce `[batch, token, V]` with `L + 2` hidden states.
#[test]
fn every_variant_builds_and_reports_its_diagnostics() {
    for variant in GemmaVariant::ALL {
        let model = tiny_model(variant, 1);
        assert_eq!(model.lm_head.is_none(), model.config().tie_word_embeddings);
        let output = model
            .forward(
                input(ints([[1, 2, 3], [4, 5, 6]]), None, None),
                &everything(),
            )
            .unwrap();
        assert_eq!(output.logits.dims(), [2, 3, VOCABULARY], "{variant:?}");
        assert_eq!(output.hidden_states.len(), 4 + 2, "{variant:?}");
        assert!(output.routers.is_empty() && output.load_balancing_loss.is_none());
        assert_close(
            output.final_hidden_state.unwrap(),
            output.hidden_states[5].clone(),
            0.0,
            "final hidden state",
        );
        let last = model
            .forward(
                input(ints([[1, 2, 3], [4, 5, 6]]), None, None),
                &ForwardOptions::default(),
            )
            .unwrap();
        assert_close(
            last.logits,
            output.logits.narrow(1, 2, 1),
            TOLERANCE,
            "last logits",
        );
    }
    let untied = model_from(
        GemmaConfig {
            tie_word_embeddings: false,
            ..tiny_config(GemmaVariant::Gemma3Text)
        },
        1,
    );
    assert!(untied.lm_head.is_some());
    assert_eq!(six_token_logits(&untied).len(), 6 * VOCABULARY);
}

/// The residual stream starts at the embedding times `sqrt(d) = 4`, exactly.
#[test]
fn the_embedding_is_scaled_by_the_rounded_root_of_the_width() {
    let model = tiny_model(GemmaVariant::Gemma3Text, 2);
    let tokens = ints([[3, 9, 27]]);
    let output = model
        .forward(input(tokens.clone(), None, None), &everything())
        .unwrap();
    let raw = model.model.embed_tokens.forward(tokens);
    assert_close(
        output.hidden_states[0].clone(),
        raw * 4.0,
        0.0,
        "scaled embedding",
    );
}

/// The final soft-cap is `c · tanh(z / c)` of the uncapped logits.
#[test]
fn the_final_soft_cap_bounds_logits_exactly() {
    let capped = tiny_model(GemmaVariant::Gemma2, 3);
    let mut uncapped = capped.clone();
    uncapped.model.architecture.final_logit_softcap = None;
    let logits = six_token_logits(&capped);
    let raw = six_token_logits(&uncapped);
    assert!(logits.iter().all(|value| value.abs() < 6.0));
    assert!(
        raw.iter().any(|value| value.abs() > 1.0),
        "logits are informative"
    );
    let expected: Vec<f32> = raw.iter().map(|z| 6.0 * (z / 6.0).tanh()).collect();
    assert!(max_gap(&logits, &expected) < 1e-5);
}

/// Attention and final soft-caps, the rounded scale, and per-type RoPE all change outputs.
#[test]
fn every_gemma_mechanism_reaches_the_logits() {
    let gemma2 = tiny_model(GemmaVariant::Gemma2, 4);
    let reference = six_token_logits(&gemma2);
    let mut uncapped = gemma2.clone();
    for layer in &mut uncapped.model.layers {
        layer.self_attn.softcap = None;
    }
    assert!(
        max_gap(&six_token_logits(&uncapped), &reference) > 1e-4,
        "attention soft-cap"
    );
    let mut unscaled = gemma2.clone();
    for layer in &mut unscaled.model.layers {
        layer.self_attn.scale = None;
    }
    assert!(
        max_gap(&six_token_logits(&unscaled), &reference) > 1e-4,
        "query_pre_attn_scalar"
    );
    let gemma3 = tiny_model(GemmaVariant::Gemma3Text, 4);
    let reference = six_token_logits(&gemma3);
    let mut one_base = gemma3.clone();
    one_base.model.architecture.full_rope = one_base.model.architecture.sliding_rope;
    assert!(
        max_gap(&six_token_logits(&one_base), &reference) > 1e-4,
        "global RoPE set"
    );
    let schedule = gemma3.rotary_schedule();
    assert_eq!(schedule.sets.len(), 2);
    assert_eq!(schedule.layers, [Some(0), Some(0), Some(1), Some(0)]);
    assert_eq!(
        tiny_model(GemmaVariant::Gemma2, 4)
            .rotary_schedule()
            .sets
            .len(),
        1
    );
}

/// A sliding layer never sees a key `W` or more positions back.
#[test]
fn sliding_layers_forget_tokens_beyond_the_window() {
    let config = GemmaConfig {
        layer_types: Some(vec!["sliding_attention".into(); 4]),
        ..tiny_config(GemmaVariant::Gemma3Text)
    };
    let model = model_from(config, 5);
    let run = |first: i64| {
        let logits = model
            .forward(
                input(ints([[first, 2, 3, 4, 5, 6]]), None, None),
                &everything(),
            )
            .unwrap()
            .logits;
        host(logits.narrow(1, 5, 1))
    };
    // Four sliding layers of window 3 reach back at most 4 · 2 = 8 > 5
    // positions, so the first token still matters at the last position...
    assert!(max_gap(&run(1), &run(7)) > 1e-6);
    // ...but only through intermediate positions: a single layer would not.
    let single = model_from(
        GemmaConfig {
            num_hidden_layers: 1,
            layer_types: Some(vec!["sliding_attention".into()]),
            ..tiny_config(GemmaVariant::Gemma3Text)
        },
        5,
    );
    let run = |first: i64| {
        host(
            single
                .forward(
                    input(ints([[first, 2, 3, 4, 5, 6]]), None, None),
                    &everything(),
                )
                .unwrap()
                .logits
                .narrow(1, 5, 1),
        )
    };
    assert_eq!(max_gap(&run(1), &run(7)), 0.0);
}

/// Left padding with implicit ordinals reproduces the unpadded sequence exactly in position.
#[test]
fn left_padding_does_not_shift_positions() {
    for variant in GemmaVariant::ALL {
        let model = tiny_model(variant, 2);
        let padded = model
            .forward(
                input(
                    ints([[0, 0, 7, 3, 9, 11]]),
                    Some(mask([[false, false, true, true, true, true]])),
                    None,
                ),
                &everything(),
            )
            .unwrap();
        let plain = model
            .forward(input(ints([[7, 3, 9, 11]]), None, None), &everything())
            .unwrap();
        assert_close(
            padded.logits.narrow(1, 2, 4),
            plain.logits,
            TOLERANCE,
            &format!("{variant:?}"),
        );
        assert!(
            host(padded.hidden_states[2].clone())
                .iter()
                .all(|value| value.is_finite())
        );
    }
}

/// Invalid inputs fail before any computation with Gemma-named errors.
#[test]
fn invalid_inputs_are_rejected() {
    let model = tiny_model(GemmaVariant::Gemma3Text, 3);
    let options = ForwardOptions::default();
    let cases = [
        (input(ints([[1, 99]]), None, None), "gemma.input_ids"),
        (
            input(ints([[1, 2]]), Some(mask([[false, false]])), None),
            "gemma.token_mask",
        ),
        (
            input(ints([[1, 2]]), None, Some(ints([[0, 64]]))),
            "gemma.position_ids",
        ),
        (
            input(ints([[1, 2]]), None, Some(ints([[0, 1, 2]]))),
            "gemma.position_ids",
        ),
    ];
    for (bad, name) in cases {
        let error = model.forward(bad, &options).unwrap_err();
        assert!(error.to_string().contains(name), "{error}");
    }
    let selection = ForwardOptions {
        logits: LogitsSelection::Positions(vec![2]),
        ..ForwardOptions::default()
    };
    assert!(
        model
            .forward(input(ints([[1, 2]]), None, None), &selection)
            .is_err()
    );
}

/// The eager graph soft-caps scores like the fused kernel and exposes window-limited patterns.
#[test]
fn eager_attention_matches_fused_attention_and_captures_patterns() {
    let fused = tiny_model(GemmaVariant::Gemma2, 4);
    let eager = fused
        .clone()
        .with_attention_precision(AttentionPrecisionMode::EagerFp32Softmax);
    let tokens = ints([[0, 5, 6, 7, 8, 9]]);
    let token_mask = mask([[false, true, true, true, true, true]]);
    let options = ForwardOptions {
        logits: LogitsSelection::All,
        attentions: true,
        ..ForwardOptions::default()
    };
    let error = fused
        .forward(
            input(tokens.clone(), Some(token_mask.clone()), None),
            &options,
        )
        .unwrap_err();
    assert!(matches!(error, Error::BackendCapability(_)), "{error}");
    assert!(error.to_string().contains("Gemma"), "{error}");
    let captured = eager
        .forward(
            input(tokens.clone(), Some(token_mask.clone()), None),
            &options,
        )
        .unwrap();
    let plain = fused
        .forward(input(tokens, Some(token_mask), None), &everything())
        .unwrap();
    let valid_logits = |logits: Tensor<B, 3>| logits.narrow(1, 1, 5);
    assert_close(
        valid_logits(captured.logits),
        valid_logits(plain.logits),
        TOLERANCE,
        "eager vs fused",
    );
    assert_eq!(captured.attentions.len(), 4);
    for (layer, pattern) in captured.attentions.into_iter().enumerate() {
        assert_eq!(pattern.dims(), [1, 4, 6, 6]);
        let sliding = layer % 2 == 0;
        let values = host(pattern);
        for head in 0..4 {
            for query in 1..6 {
                let row = &values[(head * 6 + query) * 6..(head * 6 + query + 1) * 6];
                assert_eq!(row[0], 0.0, "padded key");
                for (key, &probability) in row.iter().enumerate().skip(1) {
                    let hidden = key > query || (sliding && key + 3 <= query);
                    assert_eq!(
                        probability == 0.0,
                        hidden,
                        "layer {layer} query {query} key {key}"
                    );
                }
                let total: f32 = row.iter().sum();
                assert!((total - 1.0).abs() < 1e-5);
            }
        }
    }
}

/// Captured scores are exactly `c · tanh(q·kᵀ · s / c)` of the captured rotated queries and keys.
///
/// With `c = 0.2` over a quarter of the raw scores lie beyond `2c`, where
/// tanh saturates, so a cap applied with the wrong constant, before the
/// scale, after the mask, or not at all would miss by far more than F32
/// rounding. Query head `h` reads key/value head `h / 2` (four query heads
/// over two KV heads). The fused and eager graphs must also agree on the
/// saturated logits.
#[test]
fn saturated_attention_scores_are_soft_capped_exactly() {
    /// The soft-cap `c`, small enough that tanh saturates for many scores.
    const CAP: f32 = 0.2;
    let mut fused = tiny_model(GemmaVariant::Gemma2, 11);
    for layer in &mut fused.model.layers {
        layer.self_attn.softcap = Some(f64::from(CAP));
    }
    let eager = fused
        .clone()
        .with_attention_precision(AttentionPrecisionMode::EagerFp32Softmax);
    let tokens = ints([[1, 2, 3, 4, 5, 6]]);
    let attention = Layer::decoder(1).self_attention();
    let mut hooks = Hooks::new();
    let query = hooks.capture(attention.query());
    let key = hooks.capture(attention.key());
    let scores = hooks.capture(attention.scores());
    let options = ForwardOptions {
        logits: LogitsSelection::All,
        ..ForwardOptions::default()
    };
    let eager_logits = eager
        .forward_with_hooks(input(tokens.clone(), None, None), &options, &mut hooks)
        .unwrap()
        .logits;
    let [query, key, scores] = [query, key, scores].map(|capture| hooks.get(&capture).unwrap());
    assert_eq!(query.dims(), [1, 4, 6, 4]);
    assert_eq!(key.dims(), [1, 2, 6, 4]);
    assert_eq!(scores.dims(), [1, 4, 6, 6]);
    let (query, key, scores) = (host(query), host(key), host(scores));
    let scale = 8.0_f32.powf(-0.5);
    let mut saturated = 0;
    for head in 0..4 {
        for row in 0..6 {
            for column in 0..6 {
                let dot: f32 = (0..4)
                    .map(|channel| {
                        query[(head * 6 + row) * 4 + channel]
                            * key[((head / 2) * 6 + column) * 4 + channel]
                    })
                    .sum();
                let raw = dot * scale;
                saturated += usize::from(raw.abs() > 2.0 * CAP);
                let expected = CAP * (raw / CAP).tanh();
                let actual = scores[(head * 6 + row) * 6 + column];
                assert!(
                    (actual - expected).abs() < 1e-6,
                    "head {head} query {row} key {column}: {actual} vs {expected}"
                );
                assert!(actual.abs() <= CAP, "a capped score is bounded by c");
            }
        }
    }
    assert!(
        saturated > 4 * 6 * 6 / 4,
        "only {saturated} saturated scores"
    );
    let fused_logits = fused
        .forward(input(tokens, None, None), &options)
        .unwrap()
        .logits;
    assert_close(fused_logits, eager_logits, TOLERANCE, "fused vs eager");
}

/// The catalog lists every point in emission order, with scores only under eager attention.
#[test]
fn activation_points_follow_emission_order() {
    let model = tiny_model(GemmaVariant::Gemma3Text, 5);
    let points = model.activation_points();
    assert_eq!(points.len(), 2 + 4 * 12);
    assert_eq!(points.first(), Some(&Stack::Decoder.embedding().info()));
    assert_eq!(points.last(), Some(&Stack::Decoder.final_norm().info()));
    let layer = Layer::decoder(1);
    let attention = layer.self_attention();
    let expected = [
        layer.residual_pre().info(),
        layer.attention_input().info(),
        attention.query().info(),
        attention.key().info(),
        attention.value().info(),
        attention.head_context().info(),
        layer.attention_output().info(),
        layer.mlp_input().info(),
        layer.mlp_pre_activation().info(),
        layer.mlp_activation().info(),
        layer.mlp_output().info(),
        layer.residual_post().info(),
    ];
    assert_eq!(model.model.layers[1].activation_points(1), expected);
    let start = points
        .iter()
        .position(|point| *point == expected[0])
        .unwrap();
    assert_eq!(points[start..start + expected.len()], expected);
    assert!(!points.contains(&attention.scores().info()));
    let eager = model.with_attention_precision(AttentionPrecisionMode::EagerFp32Softmax);
    assert!(
        eager
            .activation_points()
            .contains(&attention.pattern().info())
    );
    assert_eq!(
        HookedModel::activation_points(&eager).unwrap(),
        eager.activation_points()
    );
}

/// Points sit after the sandwich norms: the output points are exactly the residual branches.
#[test]
fn hook_points_observe_the_sandwich_norms() {
    let model = tiny_model(GemmaVariant::Gemma2, 6);
    let tokens = ints([[3, 4, 5, 6]]);
    let layer = Layer::decoder(1);
    let mut hooks = Hooks::new();
    let pre = hooks.capture(layer.residual_pre());
    let attention = hooks.capture(layer.attention_output());
    let mlp = hooks.capture(layer.mlp_output());
    let post = hooks.capture(layer.residual_post());
    let mlp_input = hooks.capture(layer.mlp_input());
    model
        .forward_with_hooks(
            input(tokens, None, None),
            &ForwardOptions::default(),
            &mut hooks,
        )
        .unwrap();
    let [pre, attention, mlp, post, mlp_input] =
        [pre, attention, mlp, post, mlp_input].map(|capture| hooks.get(&capture).unwrap());
    assert_close(
        post,
        pre.clone() + attention.clone() + mlp,
        TOLERANCE,
        "residual sum",
    );
    let expected_input = model.model.layers[1]
        .pre_feedforward_layernorm
        .forward(pre + attention);
    assert_close(mlp_input, expected_input, TOLERANCE, "pre-feedforward norm");
}

/// Identity hooks preserve outputs; an edit changes them; captures see the edit.
#[test]
fn hooks_capture_and_edit_activations() {
    let model = tiny_model(GemmaVariant::Gemma3Text, 6);
    let tokens = ints([[3, 4, 5, 6]]);
    let options = ForwardOptions::default();
    let expected = model
        .forward(input(tokens.clone(), None, None), &options)
        .unwrap();
    let mut empty = Hooks::new();
    let same = model
        .forward_with_hooks(input(tokens.clone(), None, None), &options, &mut empty)
        .unwrap();
    assert_close(same.logits, expected.logits.clone(), 0.0, "empty hooks");

    let point = Layer::decoder(2).mlp_activation();
    let mut hooks = Hooks::new();
    let before = hooks.capture(point.clone());
    hooks.edit(point.clone(), |tensor| tensor * 0.0);
    let after = hooks.capture(point.clone());
    let edited = model
        .forward_with_hooks(input(tokens, None, None), &options, &mut hooks)
        .unwrap();
    assert!(max_gap(&host(edited.logits), &host(expected.logits)) > 1e-4);
    assert_eq!(hooks.get(&before).unwrap().dims(), [1, 4, 24]);
    assert!(
        host(hooks.get(&after).unwrap())
            .iter()
            .all(|&value| value == 0.0)
    );
}

/// Padded entries read as zero at every point and edits cannot revive them.
#[test]
fn padded_points_are_zero_and_cannot_be_revived() {
    let model = tiny_model(GemmaVariant::Gemma2, 7);
    let tokens = ints([[0, 4, 5, 6]]);
    let token_mask = mask([[false, true, true, true]]);
    let residual = Layer::decoder(1).residual_pre();
    let mut hooks = Hooks::new();
    let captured = hooks.capture(residual.clone());
    hooks.edit(residual.clone(), |tensor| tensor + 1.0e3);
    let revived = hooks.capture(residual);
    model
        .forward_with_hooks(
            input(tokens, Some(token_mask), None),
            &ForwardOptions::default(),
            &mut hooks,
        )
        .unwrap();
    for capture in [&captured, &revived] {
        let residual = host(hooks.get(capture).unwrap());
        assert!(
            residual[..16].iter().all(|&value| value == 0.0),
            "padded column"
        );
    }
    let shifted = host(hooks.get(&revived).unwrap());
    assert!(
        shifted[16..].iter().all(|&value| value > 100.0),
        "valid columns shifted"
    );
}

/// Staged embedding, block ranges, and normalization reproduce the hidden states.
#[test]
fn staged_execution_reproduces_the_stateless_pass() {
    for variant in GemmaVariant::ALL {
        let model = tiny_model(variant, 8);
        let tokens = ints([[0, 9, 10, 11, 12], [13, 14, 15, 16, 17]]);
        let token_mask = mask([[false, true, true, true, true], [true; 5]]);
        let full = model
            .forward(
                input(tokens.clone(), Some(token_mask.clone()), None),
                &everything(),
            )
            .unwrap();
        let state = StagedModel::embed(&model, input(tokens, Some(token_mask), None)).unwrap();
        assert_close(
            state.hidden.clone().narrow(1, 1, 4),
            full.hidden_states[0].clone().narrow(1, 1, 4),
            0.0,
            "scaled embedding",
        );
        let state = StagedModel::run_layers(&model, state, 0..3).unwrap();
        assert_close(
            state.hidden.clone().narrow(1, 1, 4),
            full.hidden_states[3].clone().narrow(1, 1, 4),
            TOLERANCE,
            "after layer 2",
        );
        let state = StagedModel::run_layers(&model, state, 3..4).unwrap();
        let state = StagedModel::normalize(&model, state).unwrap();
        assert_close(
            state.hidden.narrow(1, 1, 4),
            full.final_hidden_state.unwrap().narrow(1, 1, 4),
            TOLERANCE,
            &format!("{variant:?} staged"),
        );
    }
}

/// Staged ranges and states are validated with Gemma-named errors.
#[test]
fn staged_misuse_is_rejected() {
    let model = tiny_model(GemmaVariant::Gemma3Text, 9);
    let state = model
        .embed_with_hooks(input(ints([[1, 2]]), None, None), &mut Hooks::new())
        .unwrap();
    assert!(StagedModel::run_layers(&model, state.clone(), 1..2).is_err());
    assert!(StagedModel::run_layers(&model, state.clone(), 0..5).is_err());
    let normalized = StagedModel::normalize(&model, state.clone()).unwrap();
    let error = StagedModel::normalize(&model, normalized.clone()).unwrap_err();
    assert!(error.to_string().contains("gemma.normalize"), "{error}");
    assert!(StagedModel::run_layers(&model, normalized, 0..1).is_err());
    let mut stretched = state.clone();
    stretched.context.evaluated_context = 65;
    let error = StagedModel::run_layers(&model, stretched, 0..1).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("gemma.execution.evaluated_context"),
        "{error}"
    );
    let mut foreign = state;
    foreign.context.layer_attention = vec![(Layer::decoder(9), AttentionPolicy::causal())];
    let error = StagedModel::run_layers(&model, foreign, 0..1).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("gemma.execution.layer_attention"),
        "{error}"
    );
    assert_eq!(StagedModel::num_layers(&model), 4);
    assert_eq!(model.layer(3), Layer::decoder(3));
}

/// A staged attention policy replaces a sliding layer's window.
#[test]
fn staged_attention_policies_replace_the_sliding_rule() {
    let model = tiny_model(GemmaVariant::Gemma2, 10);
    let tokens = ints([[1, 2, 3, 4, 5, 6]]);
    let windowed = StagedModel::run_stack(&model, input(tokens.clone(), None, None)).unwrap();
    let mut state = StagedModel::embed(&model, input(tokens, None, None)).unwrap();
    state.context = state
        .context
        .with_layer_attention(Layer::decoder(0), AttentionPolicy::causal());
    let state = StagedModel::run_layers(&model, state, 0..4).unwrap();
    let global = StagedModel::normalize(&model, state).unwrap();
    let last = |hidden: Tensor<B, 3>| host(hidden.narrow(1, 5, 1));
    let early = |hidden: Tensor<B, 3>| host(hidden.narrow(1, 0, 3));
    // Positions inside the window agree; the last one now sees beyond it.
    assert!(
        max_gap(
            &early(global.hidden.clone()),
            &early(windowed.hidden.clone())
        ) < TOLERANCE
    );
    assert!(max_gap(&last(global.hidden), &last(windowed.hidden)) > 1e-5);
}

/// The stateless graph is differentiable end to end, soft-caps and norms included.
#[cfg(feature = "autodiff")]
#[test]
fn gradients_reach_every_parameter_kind() {
    use burn::{backend::Autodiff, tensor::Int};
    /// Autodiff over the deterministic F32 CPU backend.
    type A = Autodiff<B>;
    for variant in GemmaVariant::ALL {
        let device = Default::default();
        let model = GemmaForCausalLm::<A>::init(tiny_config(variant), &device).unwrap();
        let tokens = Tensor::<A, 2, Int>::from_ints([[1, 2, 3, 4, 5]], &device);
        let logits = model
            .forward(
                CausalLmInput {
                    token_ids: tokens,
                    token_mask: None,
                    position_ids: None,
                },
                &ForwardOptions {
                    logits: LogitsSelection::All,
                    ..ForwardOptions::default()
                },
            )
            .unwrap()
            .logits;
        let gradients = logits.sum().backward();
        let layer = &model.model.layers[0];
        let mut sums = vec![
            layer
                .mlp
                .gate_proj
                .weight
                .grad(&gradients)
                .expect("MLP")
                .sum(),
            layer
                .self_attn
                .q_proj
                .weight
                .grad(&gradients)
                .expect("query")
                .sum(),
            layer
                .input_layernorm
                .weight
                .grad(&gradients)
                .expect("input norm")
                .sum(),
            layer
                .post_feedforward_layernorm
                .weight
                .grad(&gradients)
                .expect("post-feedforward norm")
                .sum(),
            model
                .model
                .norm
                .weight
                .grad(&gradients)
                .expect("final norm")
                .sum(),
            model
                .model
                .embed_tokens
                .weight
                .grad(&gradients)
                .expect("embedding")
                .sum(),
        ];
        if let Some(norm) = &layer.self_attn.query_norm {
            sums.push(norm.weight.grad(&gradients).expect("query norm").sum());
        }
        assert_eq!(
            sums.len(),
            6 + usize::from(variant == GemmaVariant::Gemma3Text)
        );
        for gradient in sums {
            assert!(gradient.into_scalar().is_finite(), "{variant:?}");
        }
    }
}

/// A failing edit at any point of either variant aborts the complete and the staged passes.
///
/// Eager attention adds the score and pattern points; right padding runs the
/// validity-masked scope.
#[test]
fn every_variant_propagates_edit_failures() {
    use crate::models::shared::test_support::{
        assert_edit_failures_propagate, assert_staged_edit_failures_propagate,
    };
    for variant in GemmaVariant::ALL {
        let model = tiny_model(variant, 31)
            .with_attention_precision(AttentionPrecisionMode::EagerFp32Softmax);
        let padded = || {
            input(
                ints([[1, 2, 3], [4, 5, 0]]),
                Some(mask([[true, true, true], [true, true, false]])),
                None,
            )
        };
        let points = model.activation_points();
        assert_edit_failures_propagate(&points, |hooks| {
            model.forward_with_hooks(padded(), &ForwardOptions::default(), hooks)
        });
        assert_staged_edit_failures_propagate(&model, &padded(), &points);
    }
}

/// A replaced vocabulary head of the wrong input width fails at projection.
#[test]
fn replaced_heads_are_checked_at_projection() {
    let mut model = tiny_model(GemmaVariant::Gemma2, 32);
    model.lm_head = Some(burn::nn::LinearConfig::new(15, VOCABULARY).init(&Default::default()));
    let error = model
        .forward(
            input(ints([[1, 2]]), None, None),
            &ForwardOptions::default(),
        )
        .unwrap_err();
    assert!(
        matches!(&error, Error::Shape { tensor, expected, actual }
            if tensor == "linear.weight" && expected == "[16, nonzero_output]" && actual == "[15, 32]"),
        "{error:?}"
    );
}
