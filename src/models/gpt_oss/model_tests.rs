//! The stateless graph: shapes, the routed experts against a scalar
//! reference, padding and position semantics, router diagnostics and the
//! load-balancing objective, eager attention, hooks, staged execution, and
//! gradients.

use burn::tensor::{Tensor, TensorData};

use super::test_support::{
    B, VOCABULARY, assert_close, host, input, ints, mask, max_gap, model_from, tiny_config,
    tiny_model,
};
use super::*;
use crate::{
    LogitsSelection,
    architecture::{HookedModel, StagedModel},
    experiment::AttentionPolicy,
};

/// Absolute tolerance between F32 graphs that differ only in operation order.
const TOLERANCE: f32 = 2e-5;

/// Eight tokens per row; row 1 is left-padded by three.
const TOKENS: [[i64; 8]; 2] = [[3, 17, 8, 25, 11, 4, 29, 6], [0, 0, 0, 12, 5, 30, 9, 21]];

/// Validity of [`TOKENS`].
const VALID: [[bool; 8]; 2] = [
    [true; 8],
    [false, false, false, true, true, true, true, true],
];

/// Requests every column's logits.
fn all_logits() -> ForwardOptions {
    ForwardOptions {
        logits: LogitsSelection::All,
        ..ForwardOptions::default()
    }
}

/// Initialization allocates the documented parameter shapes.
#[test]
fn init_allocates_reference_shapes() {
    let model = tiny_model(1);
    assert_eq!(model.model.layers.len(), 3);
    let layer = &model.model.layers[0];
    assert_eq!(layer.mlp.experts.len(), 4);
    let expert = &layer.mlp.experts[0];
    assert_eq!(expert.gate_proj.weight.val().dims(), [16, 12]);
    assert_eq!(expert.up_proj.weight.val().dims(), [16, 12]);
    assert_eq!(expert.down_proj.weight.val().dims(), [12, 16]);
    assert_eq!(layer.mlp.router.weight.val().dims(), [16, 4]);
    assert_eq!(layer.self_attn.sinks.as_ref().unwrap().val().dims(), [4]);
    assert_eq!(
        layer.connectivity,
        AttentionConnectivity::SlidingWindow { window: 3 }
    );
    assert_eq!(
        model.model.layers[1].connectivity,
        AttentionConnectivity::Global
    );
    assert_eq!(model.config(), &tiny_config());
    assert_eq!(model.architecture().experts, 4);
    assert!(model.lm_head.is_some());
    assert!(
        GptOssForCausalLm::<B>::init(
            GptOssConfig {
                hidden_size: 0,
                ..tiny_config()
            },
            &Default::default()
        )
        .is_err()
    );
}

/// Computes one token's routed MLP with scalar arithmetic, as the reference does.
///
/// `z = W x + b`; the top two logits (smallest ID on ties) are softmax-weighted;
/// each expert computes `down((clamp(u, ±7) + 1) · g σ(1.702 g)) + b` with
/// `g = min(gate, 7)` from input-major weights.
fn reference_mlp(mlp: &GptOssSparseMlp<B>, token: &[f32]) -> Vec<f32> {
    let router = host(mlp.router.weight.val()); // [d, E] (column layout, transposed view)
    let bias = host(mlp.router.bias.as_ref().unwrap().val());
    let experts = bias.len();
    let hidden = token.len();
    let logits: Vec<f32> = (0..experts)
        .map(|expert| {
            bias[expert]
                + (0..hidden)
                    .map(|input| router[input * experts + expert] * token[input])
                    .sum::<f32>()
        })
        .collect();
    let mut order: Vec<usize> = (0..experts).collect();
    order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]).then(a.cmp(&b)));
    let selected = &order[..2];
    let peak = logits[selected[0]];
    let exps: Vec<f32> = selected.iter().map(|&e| (logits[e] - peak).exp()).collect();
    let total: f32 = exps.iter().sum();
    let mut output = vec![0.0_f32; hidden];
    for (slot, &expert) in selected.iter().enumerate() {
        let weights = &mlp.experts[expert];
        let gate_weight = host(weights.gate_proj.weight.val());
        let up_weight = host(weights.up_proj.weight.val());
        let down_weight = host(weights.down_proj.weight.val());
        let gate_bias = host(weights.gate_proj.bias.as_ref().unwrap().val());
        let up_bias = host(weights.up_proj.bias.as_ref().unwrap().val());
        let down_bias = host(weights.down_proj.bias.as_ref().unwrap().val());
        let width = gate_bias.len();
        let activated: Vec<f32> = (0..width)
            .map(|j| {
                let project = |w: &[f32], b: &[f32]| {
                    b[j] + (0..hidden)
                        .map(|i| w[i * width + j] * token[i])
                        .sum::<f32>()
                };
                let gate = project(&gate_weight, &gate_bias).min(7.0);
                let up = project(&up_weight, &up_bias).clamp(-7.0, 7.0);
                (up + 1.0) * gate / (1.0 + (-1.702 * gate).exp())
            })
            .collect();
        let weight = exps[slot] / total;
        for (out, value) in output.iter_mut().enumerate() {
            let projected = down_bias[out]
                + (0..width)
                    .map(|j| down_weight[j * hidden + out] * activated[j])
                    .sum::<f32>();
            *value += weight * projected;
        }
    }
    output
}

/// The routed MLP equals the scalar reference, including clamped gates.
#[test]
fn routed_mlp_matches_a_scalar_reference() {
    let model = tiny_model(3);
    let mlp = &model.model.layers[0].mlp;
    // Large inputs drive gate and up projections beyond the clamp of 7.
    let values: Vec<f32> = (0..2 * 3 * 16)
        .map(|index| ((index * 7 % 23) as f32 - 11.0) * 0.9)
        .collect();
    let input = Tensor::<B, 3>::from_data(
        TensorData::new(values.clone(), [2, 3, 16]),
        &Default::default(),
    );
    let (output, logits) = mlp
        .forward_with_hooks(input, Layer::decoder(0), &mut Hooks::new())
        .unwrap();
    assert_eq!(logits.dims(), [2, 3, 4]);
    let output = host(output);
    let mut clamped = false;
    for token in 0..6 {
        let row = &values[token * 16..(token + 1) * 16];
        let expected = reference_mlp(mlp, row);
        let gap = max_gap(&output[token * 16..(token + 1) * 16], &expected);
        let scale = expected.iter().fold(1.0_f32, |m, v| m.max(v.abs()));
        assert!(gap <= 1e-5 * scale, "token {token}: {gap:e}");
        clamped |= expected.iter().any(|v| v.abs() > 7.0);
    }
    assert!(clamped, "the inputs should exercise the clamp");
}

/// Logits, hidden states, and selections have the documented shapes.
#[test]
fn forward_returns_requested_outputs() {
    let model = tiny_model(5);
    let output = model
        .forward(
            input(ints(TOKENS), Some(mask(VALID)), None),
            &ForwardOptions {
                logits: LogitsSelection::All,
                hidden_states: true,
                ..ForwardOptions::default()
            },
        )
        .unwrap();
    assert_eq!(output.logits.dims(), [2, 8, VOCABULARY]);
    assert_eq!(output.hidden_states.len(), 3 + 2);
    assert_eq!(output.final_hidden_state.unwrap().dims(), [2, 8, 16]);
    assert!(output.routers.is_empty());
    assert!(output.load_balancing_loss.is_none());
    let last = model
        .forward(
            input(ints(TOKENS), Some(mask(VALID)), None),
            &ForwardOptions::default(),
        )
        .unwrap();
    assert_eq!(last.logits.dims(), [2, 1, VOCABULARY]);
}

/// Left padding neither shifts positions nor leaks into valid tokens.
#[test]
fn left_padding_matches_the_unpadded_row() {
    let model = tiny_model(7);
    let padded = model
        .forward(input(ints(TOKENS), Some(mask(VALID)), None), &all_logits())
        .unwrap()
        .logits;
    let alone = model
        .forward(input(ints([[12, 5, 30, 9, 21]]), None, None), &all_logits())
        .unwrap()
        .logits;
    assert_close(
        padded.narrow(0, 1, 1).narrow(1, 3, 5),
        alone,
        TOLERANCE,
        "padded row",
    );
}

/// Implicit positions equal explicit valid-token ordinals.
#[test]
fn implicit_positions_are_valid_token_ordinals() {
    let model = tiny_model(9);
    let implicit = model
        .forward(input(ints(TOKENS), Some(mask(VALID)), None), &all_logits())
        .unwrap()
        .logits;
    let ordinals = ints([[0, 1, 2, 3, 4, 5, 6, 7], [0, 0, 0, 0, 1, 2, 3, 4]]);
    let explicit = model
        .forward(
            input(ints(TOKENS), Some(mask(VALID)), Some(ordinals)),
            &all_logits(),
        )
        .unwrap()
        .logits;
    assert_close(implicit, explicit, 0.0, "positions");
}

/// Invalid inputs are rejected before computation.
#[test]
fn invalid_inputs_are_rejected() {
    let model = tiny_model(11);
    let options = ForwardOptions::default();
    assert!(
        model
            .forward(input(ints([[1, 99]]), None, None), &options)
            .is_err()
    );
    assert!(
        model
            .forward(
                input(ints([[1, 2]]), Some(mask([[false, false]])), None),
                &options
            )
            .is_err()
    );
    assert!(
        model
            .forward(input(ints([[1, 2]]), None, Some(ints([[0, 64]]))), &options)
            .is_err()
    );
    // Attention capture needs the eager graph.
    let capture = ForwardOptions {
        attentions: true,
        ..ForwardOptions::default()
    };
    assert!(matches!(
        model.forward(input(ints([[1, 2]]), None, None), &capture),
        Err(Error::BackendCapability(_))
    ));
}

/// The eager sink graph equals the fused one and returns sink-discounted patterns.
#[test]
fn eager_attention_matches_fused_attention() {
    let fused = tiny_model(13);
    let eager = tiny_model(13).with_attention_precision(AttentionPrecisionMode::EagerFp32Softmax);
    let options = ForwardOptions {
        logits: LogitsSelection::All,
        attentions: true,
        ..ForwardOptions::default()
    };
    let expected = fused
        .forward(input(ints(TOKENS), Some(mask(VALID)), None), &all_logits())
        .unwrap()
        .logits;
    let output = eager
        .forward(input(ints(TOKENS), Some(mask(VALID)), None), &options)
        .unwrap();
    // Capturing patterns runs a hooked pass, which zeroes padded positions.
    let keep = mask(VALID)
        .as_tensor()
        .clone()
        .float()
        .unsqueeze_dim::<3>(2);
    assert_close(
        output.logits * keep.clone(),
        expected * keep,
        TOLERANCE,
        "eager vs fused",
    );
    assert_eq!(output.attentions.len(), 3);
    // The sliding layer sees at most three keys; sinks keep every row below one.
    let pattern = host(output.attentions[0].clone());
    for (row, chunk) in pattern.chunks(8).enumerate() {
        let sum: f32 = chunk.iter().sum();
        assert!(sum < 1.0, "row {row}: {sum}");
        assert!(chunk.iter().filter(|&&p| p > 0.0).count() <= 3, "row {row}");
    }
}

/// Router logits and the reference load-balancing objective are returned on request.
#[test]
fn router_diagnostics_and_load_balancing_loss() {
    let model = tiny_model(15);
    let options = ForwardOptions {
        router_logits: true,
        auxiliary_losses: true,
        ..ForwardOptions::default()
    };
    let output = model
        .forward(input(ints(TOKENS), Some(mask(VALID)), None), &options)
        .unwrap();
    assert_eq!(output.routers.len(), 3);
    let logits: Vec<_> = output
        .routers
        .iter()
        .map(|router| router.logits.clone().unwrap())
        .collect();
    assert!(logits.iter().all(|logits| logits.dims() == [2, 8, 4]));
    let loss = output.load_balancing_loss.unwrap();
    // The scalar reference: one softmax over all layers' logits, top-2, masked means.
    let valid: Vec<bool> = VALID.concat();
    let tokens: Vec<Vec<f32>> = logits
        .iter()
        .flat_map(|logits| {
            host(logits.clone())
                .chunks(4)
                .map(<[f32]>::to_vec)
                .collect::<Vec<_>>()
        })
        .collect();
    let mut assigned = [0.0_f64; 4];
    let mut probability = [0.0_f64; 4];
    let mut counted = 0.0_f64;
    for (index, token) in tokens.iter().enumerate() {
        if !valid[index % 16] {
            continue;
        }
        let peak = token.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f64> = token.iter().map(|z| f64::from((z - peak).exp())).collect();
        let total: f64 = exps.iter().sum();
        let mut order: Vec<usize> = (0..4).collect();
        order.sort_by(|&a, &b| exps[b].total_cmp(&exps[a]).then(a.cmp(&b)));
        for &expert in &order[..2] {
            assigned[expert] += 1.0;
        }
        for expert in 0..4 {
            probability[expert] += exps[expert] / total;
        }
        counted += 1.0;
    }
    let expected: f64 = 4.0
        * (0..4)
            .map(|e| (assigned[e] / counted) * (probability[e] / counted))
            .sum::<f64>();
    let loss = host(loss)[0];
    assert!(
        (f64::from(loss) - expected).abs() < 1e-5,
        "{loss} vs {expected}"
    );
    // Configured output_router_logits returns logits without being asked.
    let configured = model_from(
        GptOssConfig {
            output_router_logits: true,
            ..tiny_config()
        },
        15,
    );
    let output = configured
        .forward(input(ints(TOKENS), None, None), &ForwardOptions::default())
        .unwrap();
    assert_eq!(output.routers.len(), 3);
    assert!(gpt_oss_load_balancing_loss::<B>(&[], 2, None).is_err());
    assert!(gpt_oss_load_balancing_loss(&logits, 5, None).is_err());
}

/// A tied model projects with its embedding table.
#[test]
fn tied_models_project_with_the_embedding() {
    let model = model_from(
        GptOssConfig {
            tie_word_embeddings: true,
            ..tiny_config()
        },
        17,
    );
    assert!(model.lm_head.is_none());
    let output = model
        .forward(
            input(ints([[1, 2, 3]]), None, Some(ints([[0, 1, 2]]))),
            &ForwardOptions {
                hidden_states: true,
                ..ForwardOptions::default()
            },
        )
        .unwrap();
    let hidden = output.final_hidden_state.unwrap().narrow(1, 2, 1);
    let table = model.model.embed_tokens.weight.val();
    let expected = burn::tensor::module::linear(hidden, table.transpose(), None);
    assert_close(output.logits, expected, 1e-6, "tied head");
}

/// Hooks expose every listed point; captures and edits act on the forward pass.
#[test]
fn hooks_capture_and_edit_listed_points() {
    let model = tiny_model(19);
    let points = HookedModel::activation_points(&model).unwrap();
    assert_eq!(points.first().unwrap().name, "decoder.embedding");
    assert_eq!(points.last().unwrap().name, "decoder.final_norm");
    // 1 + 3 · 12 + 1 points: no score/pattern points under fused attention.
    assert_eq!(points.len(), 2 + 3 * 12);
    let names: Vec<&str> = points.iter().map(|point| point.name.as_str()).collect();
    assert!(names.contains(&"decoder.layers.1.mlp.router_logits"));
    assert!(names.contains(&"decoder.layers.1.mlp.router_weights"));
    let eager = tiny_model(19).with_attention_precision(AttentionPrecisionMode::EagerFp32Softmax);
    assert_eq!(eager.activation_points().len(), 2 + 3 * 14);

    // Captured router logits equal the diagnostics of an unhooked pass.
    let mut hooks = Hooks::new();
    let capture = hooks.capture(router_logits_point(Layer::decoder(2)));
    let hooked = model
        .forward_with_hooks(
            input(ints(TOKENS), Some(mask(VALID)), None),
            &ForwardOptions::default(),
            &mut hooks,
        )
        .unwrap();
    let plain = model
        .forward(
            input(ints(TOKENS), Some(mask(VALID)), None),
            &ForwardOptions {
                router_logits: true,
                ..ForwardOptions::default()
            },
        )
        .unwrap();
    assert_close(
        hooked.logits.clone(),
        plain.logits.clone(),
        0.0,
        "empty edits",
    );
    let captured = hooks.get(&capture).unwrap();
    // Padded positions read as zero in captures.
    let reference = plain.routers[2].logits.clone().unwrap();
    let valid = mask(VALID)
        .as_tensor()
        .clone()
        .float()
        .unsqueeze_dim::<3>(2);
    assert_close(captured, reference * valid, 0.0, "router capture");

    // Zeroing every expert weight removes the MLP branch of that layer.
    let mut edit = Hooks::new();
    edit.edit(router_weights_point(Layer::decoder(0)), |weights| {
        weights.zeros_like()
    });
    let edited = model
        .forward_with_hooks(
            input(ints(TOKENS), None, None),
            &ForwardOptions::default(),
            &mut edit,
        )
        .unwrap();
    assert!(
        max_gap(
            &host(edited.logits),
            &host(
                model
                    .forward(input(ints(TOKENS), None, None), &ForwardOptions::default())
                    .unwrap()
                    .logits
            )
        ) > 1e-3
    );
    // Attention capture inside a hooked pass is refused.
    let refuse = ForwardOptions {
        attentions: true,
        ..ForwardOptions::default()
    };
    assert!(
        model
            .forward_with_hooks(input(ints(TOKENS), None, None), &refuse, &mut Hooks::new())
            .is_err()
    );
}

/// Staged execution reproduces the stateless pass and accepts residual patches.
#[test]
fn staged_execution_matches_the_stateless_pass() {
    let model = tiny_model(21);
    let full = model
        .forward(
            input(ints(TOKENS), Some(mask(VALID)), None),
            &ForwardOptions {
                hidden_states: true,
                ..ForwardOptions::default()
            },
        )
        .unwrap();
    let staged =
        StagedModel::run_stack(&model, input(ints(TOKENS), Some(mask(VALID)), None)).unwrap();
    assert!(staged.normalized);
    let valid = mask(VALID)
        .as_tensor()
        .clone()
        .float()
        .unsqueeze_dim::<3>(2);
    assert_close(
        staged.hidden * valid.clone(),
        full.final_hidden_state.unwrap() * valid,
        0.0,
        "staged",
    );
    // Splitting the range anywhere gives the same states.
    let state = model.embed(input(ints(TOKENS), None, None)).unwrap();
    let state = model.run_layers(state, 0..2).unwrap();
    assert_eq!(state.next_layer, 2);
    let state = model.run_layers(state, 2..3).unwrap();
    let state = model.normalize(state).unwrap();
    assert!(
        model.normalize(state.clone()).is_err(),
        "double normalization"
    );
    let once = model.run_stack(input(ints(TOKENS), None, None)).unwrap();
    assert_close(state.hidden, once.hidden, 0.0, "split ranges");
    assert_eq!(StagedModel::num_layers(&model), 3);
    assert_eq!(model.stack(), Stack::Decoder);
}

/// Staged ranges and edited states are validated before any layer runs.
///
/// Every refusal names its configuration path, so callers can tell a
/// malformed state (dtype, cross-attention, an override for a missing
/// layer, an evaluated context beyond the trained one) from a bad range.
#[test]
fn staged_misuse_is_rejected() {
    use crate::intervention::Layer;

    let model = tiny_model(24);
    let state = model.embed(input(ints([[1, 2, 3]]), None, None)).unwrap();
    let refusal = |state: crate::experiment::HiddenState<B>, range, path: &str| {
        let error = model.run_layers(state, range).unwrap_err();
        assert!(error.to_string().contains(path), "{path}: {error}");
    };
    assert!(
        model.run_layers(state.clone(), 1..2).is_err(),
        "skipped layer"
    );
    assert!(
        model.run_layers(state.clone(), 0..4).is_err(),
        "past the stack"
    );
    let normalized = model.normalize(state.clone()).unwrap();
    assert!(
        model.run_layers(normalized, 0..1).is_err(),
        "after the norm"
    );

    let mut wide = state.clone();
    wide.hidden = wide.hidden.cast(burn::tensor::DType::F64);
    refusal(wide, 0..1, "gpt_oss.execution.hidden");
    let mut crossed = state.clone();
    crossed.context = crossed
        .context
        .with_cross_attention(AttentionPolicy::bidirectional());
    refusal(crossed, 0..1, "gpt_oss.execution.cross_attention");
    for layer in [Layer::decoder(3), Layer::encoder(0)] {
        let mut overridden = state.clone();
        overridden.context = overridden
            .context
            .with_layer_attention(layer, AttentionPolicy::causal());
        refusal(overridden, 0..1, "gpt_oss.execution.layer_attention");
    }
    let mut stretched = state.clone();
    stretched.context.evaluated_context = model.architecture().max_position_embeddings + 1;
    refusal(stretched, 0..1, "gpt_oss.execution.evaluated_context");

    // The trait's hooked forward is the inherent one.
    let options = ForwardOptions::default();
    let through_trait = HookedModel::forward_with_hooks(
        &model,
        input(ints([[1, 2, 3]]), None, None),
        &options,
        &mut crate::intervention::Hooks::new(),
    )
    .unwrap();
    let direct = model
        .forward(input(ints([[1, 2, 3]]), None, None), &options)
        .unwrap();
    assert_close(through_trait.logits, direct.logits, 0.0, "trait delegation");
}

/// An attention policy replaces a layer's rule in staged execution.
#[test]
fn staged_attention_policies_override_connectivity() {
    let model = tiny_model(23);
    let mut state = model
        .embed(input(ints([[1, 2, 3, 4, 5, 6]]), None, None))
        .unwrap();
    let baseline = model.run_layers(state.clone(), 0..3).unwrap().hidden;
    // A causal policy without a window widens the sliding layers to global attention.
    state.context.attention = Some(AttentionPolicy::causal());
    let widened = model.run_layers(state, 0..3).unwrap().hidden;
    assert!(max_gap(&host(baseline), &host(widened)) > 1e-4);
}

/// Gradients reach the router, the experts, and the sinks.
#[cfg(feature = "autodiff")]
#[test]
fn gradients_reach_router_experts_and_sinks() {
    use burn::backend::Autodiff;

    let model: GptOssForCausalLm<Autodiff<B>> =
        GptOssForCausalLm::init(tiny_config(), &Default::default()).unwrap();
    let tokens = Tensor::<Autodiff<B>, 2, burn::tensor::Int>::from_data(
        TensorData::new(vec![1_i64, 5, 9, 2, 7], [1, 5]),
        &Default::default(),
    );
    let output = model
        .forward(
            CausalLmInput {
                token_ids: tokens,
                token_mask: None,
                position_ids: None,
            },
            &all_logits(),
        )
        .unwrap();
    let gradients = output.logits.square().sum().backward();
    let layer = &model.model.layers[0];
    assert!(layer.mlp.router.weight.grad(&gradients).is_some());
    assert!(
        layer
            .self_attn
            .sinks
            .as_ref()
            .unwrap()
            .grad(&gradients)
            .is_some()
    );
    let reached = layer
        .mlp
        .experts
        .iter()
        .filter(|expert| expert.gate_proj.weight.grad(&gradients).is_some())
        .count();
    assert!(reached > 0, "some routed expert receives a gradient");
}
