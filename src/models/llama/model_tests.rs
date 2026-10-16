//! Stateless graph: shapes, input validation, padding invariance, attention
//! precision modes, hooks, staged execution, and gradients.

use burn::tensor::{Int, Tensor};

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
    models::llama::{LlamaFamilyConfig, LlamaVariant},
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

/// Every variant allocates its tied or untied head and produces `[batch, token, V]`.
#[test]
fn every_variant_builds_and_reports_its_diagnostics() {
    for variant in LlamaVariant::ALL {
        let model = tiny_model(variant, 1);
        assert_eq!(model.lm_head.is_none(), model.config().tie_word_embeddings);
        let output = model
            .forward(
                input(ints([[1, 2, 3], [4, 5, 6]]), None, None),
                &everything(),
            )
            .unwrap();
        assert_eq!(output.logits.dims(), [2, 3, VOCABULARY], "{variant:?}");
        assert_eq!(output.hidden_states.len(), 3 + 2, "{variant:?}");
        assert!(output.routers.is_empty() && output.load_balancing_loss.is_none());
        assert_close(
            output.final_hidden_state.unwrap(),
            output.hidden_states[4].clone(),
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
}

/// Left padding with implicit ordinals reproduces the unpadded sequence exactly in position.
#[test]
fn left_padding_does_not_shift_positions() {
    for variant in LlamaVariant::ALL {
        let model = tiny_model(variant, 2);
        let padded = model
            .forward(
                input(
                    ints([[0, 0, 7, 3, 9]]),
                    Some(mask([[false, false, true, true, true]])),
                    None,
                ),
                &everything(),
            )
            .unwrap();
        let plain = model
            .forward(input(ints([[7, 3, 9]]), None, None), &everything())
            .unwrap();
        assert_close(
            padded.logits.narrow(1, 2, 3),
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

/// An all-valid token mask takes the unpadded path: logits are bit-identical to no mask.
///
/// Validation already proves every token valid, so no exclusion plane is
/// built and global layers keep the attention call's own causality; sliding
/// layers build the same window plane either way (ordinals equal columns).
#[test]
fn an_all_valid_mask_is_the_unpadded_pass() {
    for variant in LlamaVariant::ALL {
        let model = tiny_model(variant, 4);
        let tokens = ints([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]);
        let run = |token_mask| {
            model
                .forward(input(tokens.clone(), token_mask, None), &everything())
                .unwrap()
                .logits
        };
        let plain = run(None);
        let all_valid = run(Some(mask([[true; 5]; 2])));
        assert_close(all_valid, plain, 0.0, &format!("{variant:?}"));
    }
}

/// Invalid inputs fail before any computation with contextual errors.
#[test]
fn invalid_inputs_are_rejected() {
    let model = tiny_model(LlamaVariant::Qwen3, 3);
    let options = ForwardOptions::default();
    let cases = [
        (input(ints([[1, 99]]), None, None), "llama.input_ids"),
        (
            input(ints([[1, 2]]), Some(mask([[false, false]])), None),
            "llama.token_mask",
        ),
        (
            input(ints([[1, 2]]), None, Some(ints([[0, 64]]))),
            "llama.position_ids",
        ),
        (
            input(ints([[1, 2]]), None, Some(ints([[0, 1, 2]]))),
            "llama.position_ids",
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

/// The eager F32-softmax graph equals the fused graph in F32 and exposes attention probabilities.
#[test]
fn eager_attention_matches_fused_attention_and_captures_patterns() {
    let fused = tiny_model(LlamaVariant::Mistral, 4);
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
    assert_eq!(captured.attentions.len(), 3);
    for pattern in captured.attentions {
        assert_eq!(pattern.dims(), [1, 4, 6, 6]);
        let values = host(pattern);
        // The padded key (column 0) and future keys carry no probability;
        // a window of 3 excludes keys three or more positions back.
        for head in 0..4 {
            for query in 1..6 {
                let row = &values[(head * 6 + query) * 6..(head * 6 + query + 1) * 6];
                assert_eq!(row[0], 0.0);
                for (key, &probability) in row.iter().enumerate().skip(1) {
                    if key > query || key + 3 <= query {
                        assert_eq!(probability, 0.0, "query {query} key {key}");
                    }
                }
                let total: f32 = row.iter().sum();
                assert!((total - 1.0).abs() < 1e-5);
            }
        }
    }
}

/// The catalog lists every point in emission order, with scores only under eager attention.
#[test]
fn activation_points_follow_emission_order() {
    let model = tiny_model(LlamaVariant::Olmo2, 5);
    let points = model.activation_points();
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

/// Identity hooks preserve outputs; an edit changes them; captures see the edit.
#[test]
fn hooks_capture_and_edit_activations() {
    let model = tiny_model(LlamaVariant::Qwen2, 6);
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

    let point = Layer::decoder(1).mlp_activation();
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
    let model = tiny_model(LlamaVariant::Llama, 7);
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

/// Poisoning a padded value cannot reach valid outputs.
#[test]
fn padded_value_edits_do_not_change_valid_outputs() {
    let model = tiny_model(LlamaVariant::Qwen3, 12);
    let tokens = ints([[0, 4, 5, 6]]);
    let token_mask = mask([[false, true, true, true]]);
    let options = ForwardOptions {
        logits: LogitsSelection::All,
        ..ForwardOptions::default()
    };
    let expected = model
        .forward(
            input(tokens.clone(), Some(token_mask.clone()), None),
            &options,
        )
        .unwrap();
    let mut hooks = Hooks::new();
    hooks.edit(
        Layer::decoder(0).self_attention().value(),
        |tensor: Tensor<B, 4>| {
            let [batch, heads, _, width] = tensor.dims();
            let poison = Tensor::full([batch, heads, 1, width], f32::NAN, &tensor.device());
            tensor.slice_assign([0..batch, 0..heads, 0..1, 0..width], poison)
        },
    );
    let edited = model
        .forward_with_hooks(input(tokens, Some(token_mask), None), &options, &mut hooks)
        .unwrap();
    let valid = |logits: Tensor<B, 3>| logits.narrow(1, 1, 3);
    assert_close(
        valid(edited.logits),
        valid(expected.logits),
        TOLERANCE,
        "poisoned padded value",
    );
}

/// Staged embedding, block ranges, and normalization reproduce the hidden states.
#[test]
fn staged_execution_reproduces_the_stateless_pass() {
    for variant in LlamaVariant::ALL {
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
        let state = StagedModel::run_layers(&model, state, 0..1).unwrap();
        assert_close(
            state.hidden.clone().narrow(1, 1, 4),
            full.hidden_states[1].clone().narrow(1, 1, 4),
            TOLERANCE,
            "after layer 0",
        );
        let state = StagedModel::run_layers(&model, state, 1..3).unwrap();
        let state = StagedModel::normalize(&model, state).unwrap();
        assert_close(
            state.hidden.narrow(1, 1, 4),
            full.final_hidden_state.unwrap().narrow(1, 1, 4),
            TOLERANCE,
            &format!("{variant:?} staged"),
        );
    }
}

/// Staged ranges and states are validated.
#[test]
fn staged_misuse_is_rejected() {
    let model = tiny_model(LlamaVariant::Qwen3, 9);
    let state = model.embed_with_hooks(input(ints([[1, 2]]), None, None), &mut Hooks::new());
    let state = state.unwrap();
    assert!(StagedModel::run_layers(&model, state.clone(), 1..2).is_err());
    assert!(StagedModel::run_layers(&model, state.clone(), 0..4).is_err());
    let normalized = StagedModel::normalize(&model, state.clone()).unwrap();
    assert!(StagedModel::normalize(&model, normalized.clone()).is_err());
    assert!(StagedModel::run_layers(&model, normalized, 0..1).is_err());
    let mut far = state.clone();
    far.context.positions = ints([[0, 64]]);
    far.context.evaluated_context = 65;
    assert!(StagedModel::run_layers(&model, far, 0..1).is_err());
    // In-range positions under an evaluated context beyond the trained one
    // would silently change context-dependent RoPE frequencies.
    let mut stretched = state;
    stretched.context.evaluated_context = 65;
    let error = StagedModel::run_layers(&model, stretched, 0..1).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("llama.execution.evaluated_context"),
        "{error}"
    );
    assert_eq!(StagedModel::num_layers(&model), 3);
    assert_eq!(model.layer(2), Layer::decoder(2));
}

/// A staged attention policy replaces the layer's causal rule.
#[test]
fn staged_attention_policies_replace_the_causal_rule() {
    let model = tiny_model(LlamaVariant::Qwen3, 10);
    let tokens = ints([[1, 2, 3, 4]]);
    let causal = StagedModel::run_stack(&model, input(tokens.clone(), None, None)).unwrap();
    let mut state = StagedModel::embed(&model, input(tokens.clone(), None, None)).unwrap();
    state.context = state.context.with_attention(AttentionPolicy::causal());
    let state = StagedModel::run_layers(&model, state, 0..3).unwrap();
    let explicit = StagedModel::normalize(&model, state).unwrap();
    assert_close(
        explicit.hidden,
        causal.hidden.clone(),
        TOLERANCE,
        "explicit causal policy",
    );
    let mut state = StagedModel::embed(&model, input(tokens, None, None)).unwrap();
    state.context = state
        .context
        .with_layer_attention(Layer::decoder(1), AttentionPolicy::bidirectional());
    let state = StagedModel::run_layers(&model, state, 0..3).unwrap();
    let bidirectional = StagedModel::normalize(&model, state).unwrap();
    let first = |hidden: Tensor<B, 3>| host(hidden.narrow(1, 0, 1));
    assert!(max_gap(&first(bidirectional.hidden), &first(causal.hidden)) > 1e-4);
}

/// RoPE translation reaches the graph: llama3 scaling changes long-range logits.
#[test]
fn rope_scaling_is_executed() {
    let scaled: crate::models::config::RopeParameters = serde_json::from_value(serde_json::json!({
        "rope_type": "llama3", "rope_theta": 10000.0, "factor": 8.0,
        "low_freq_factor": 1.0, "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8,
    }))
    .unwrap();
    let tokens: Vec<i64> = (0..20).map(|index| index % 30 + 1).collect();
    let tokens =
        Tensor::<B, 1, Int>::from_ints(tokens.as_slice(), &Default::default()).unsqueeze_dim(0);
    let run = |config: LlamaFamilyConfig| {
        model_from(config, 11)
            .forward(
                input(tokens.clone(), None, None),
                &ForwardOptions::default(),
            )
            .unwrap()
            .logits
    };
    let plain = run(tiny_config(LlamaVariant::Qwen3));
    let llama3 = run(LlamaFamilyConfig {
        rope_parameters: Some(scaled),
        ..tiny_config(LlamaVariant::Qwen3)
    });
    assert!(max_gap(&host(plain), &host(llama3)) > 1e-5);
}

/// The stateless graph is differentiable end to end.
#[cfg(feature = "autodiff")]
#[test]
fn gradients_reach_every_parameter_kind() {
    use burn::backend::Autodiff;
    /// Autodiff over the deterministic F32 CPU backend.
    type A = Autodiff<B>;
    for variant in LlamaVariant::ALL {
        let device = Default::default();
        let model = LlamaForCausalLm::<A>::init(tiny_config(variant), &device).unwrap();
        let tokens = Tensor::<A, 2, Int>::from_ints([[1, 2, 3, 4]], &device);
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
        let gate = layer
            .mlp
            .gate_proj
            .weight
            .grad(&gradients)
            .expect("MLP gradient");
        let query = layer
            .self_attn
            .q_proj
            .weight
            .grad(&gradients)
            .expect("query gradient");
        let norm = layer
            .attention_norm
            .weight
            .grad(&gradients)
            .expect("norm gradient");
        let embedding = model
            .model
            .embed_tokens
            .weight
            .grad(&gradients)
            .expect("embedding gradient");
        for gradient in [gate.sum(), query.sum(), norm.sum(), embedding.sum()] {
            assert!(gradient.into_scalar().is_finite(), "{variant:?}");
        }
    }
}

/// A failing edit at any point of any variant aborts the complete and the staged passes.
///
/// Eager attention adds the score and pattern points, and right padding runs
/// the validity-masked scope.
#[test]
fn every_variant_propagates_edit_failures() {
    use crate::models::shared::test_support::{
        assert_edit_failures_propagate, assert_staged_edit_failures_propagate,
    };
    for variant in LlamaVariant::ALL {
        let model = tiny_model(variant, 21)
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

/// Each block lists exactly its own slice of the model catalog.
#[test]
fn block_catalogs_partition_the_model_catalog() {
    let model = tiny_model(LlamaVariant::Olmo2, 22)
        .with_attention_precision(AttentionPrecisionMode::EagerFp32Softmax);
    let mut expected = vec![Stack::Decoder.embedding().info()];
    for (index, block) in model.model.layers.iter().enumerate() {
        expected.extend(block.activation_points(index));
    }
    expected.push(Stack::Decoder.final_norm().info());
    assert_eq!(model.activation_points(), expected);
}

/// The hooked pass rejects attention diagnostics and empty grids inside its scope.
#[test]
fn hooked_passes_reject_diagnostics_and_empty_grids() {
    let model = tiny_model(LlamaVariant::Llama, 23);
    let mut hooks = Hooks::new();
    let embedding = hooks.capture(Stack::Decoder.embedding());
    let error = model
        .forward_with_hooks(
            input(ints([[1, 2]]), None, None),
            &ForwardOptions {
                attentions: true,
                ..ForwardOptions::default()
            },
            &mut hooks,
        )
        .unwrap_err();
    assert!(
        matches!(&error, Error::BackendCapability(reason) if reason.contains("attention.pattern hook point")),
        "{error:?}"
    );
    let empty = input(
        Tensor::<B, 2, Int>::zeros([1, 0], &Default::default()),
        Some(crate::TokenMask::new(
            Tensor::<B, 2, burn::tensor::Bool>::empty([1, 0], &Default::default()),
        )),
        None,
    );
    let error = model
        .forward_with_hooks(empty, &ForwardOptions::default(), &mut hooks)
        .unwrap_err();
    assert!(matches!(&error, Error::Shape { .. }), "{error:?}");
    assert!(
        hooks.get(&embedding).is_err(),
        "no successful run left a capture"
    );
}

/// Staged states must keep the embedding dtype, stay decoder-only, and carry valid policies.
#[test]
fn staged_states_reject_dtypes_sources_and_misshapen_policies() {
    use crate::experiment::EncoderContext;
    let model = tiny_model(LlamaVariant::Qwen3, 24);
    let state = StagedModel::embed(&model, input(ints([[1, 2, 3]]), None, None)).unwrap();
    let mut half = state.clone();
    half.hidden = half.hidden.cast(burn::tensor::DType::F16);
    let error = StagedModel::run_layers(&model, half, 0..3).unwrap_err();
    assert!(
        matches!(&error, Error::Configuration { field, .. } if field == "llama.execution.hidden"),
        "{error:?}"
    );
    let mut sourced = state.clone();
    sourced.context.encoder = Some(EncoderContext {
        hidden: Tensor::zeros([1, 2, 16], &Default::default()),
        positions: ints([[0, 1]]),
        token_mask: None,
    });
    let error = StagedModel::normalize(&model, sourced).unwrap_err();
    assert!(
        matches!(&error, Error::Configuration { field, reason }
            if field == "llama.execution.cross_attention" && reason.contains("decoder-only")),
        "{error:?}"
    );
    let mut biased = state;
    biased.context = biased.context.with_attention(
        AttentionPolicy::causal().with_bias(Tensor::zeros([1, 1, 2, 2], &Default::default())),
    );
    let error = StagedModel::run_layers(&model, biased, 0..3).unwrap_err();
    assert!(matches!(&error, Error::Shape { .. }), "{error:?}");
}
