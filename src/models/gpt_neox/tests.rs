//! Deterministic GPT-NeoX graph and public activation-interface regressions.

use super::{GptNeoXConfig, GptNeoXModel};
use crate::{
    CausalLmInput, ForwardOptions, LogitsSelection,
    architecture::TokenMask,
    intervention::{Hooks, Layer, Stack},
};
use burn::{
    backend::Flex,
    module::Param,
    nn::{EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig},
    tensor::{Tensor, TensorData},
};

type B = Flex<f32, i32>;

fn config() -> GptNeoXConfig {
    GptNeoXConfig {
        vocab_size: 13,
        hidden_size: 8,
        num_hidden_layers: 2,
        num_attention_heads: 2,
        intermediate_size: 12,
        rotary_pct: 0.5,
        max_position_embeddings: 16,
        ..Default::default()
    }
}

fn tensor<const D: usize>(shape: [usize; D], offset: f64, scale: f64) -> Tensor<B, D> {
    let values = (0..shape.iter().product())
        .map(|i| ((i as f64 * 0.17 + offset).sin() * scale) as f32)
        .collect::<Vec<_>>();
    Tensor::from_data(TensorData::new(values, shape), &Default::default())
}

fn linear(module: &mut Linear<B>, input: usize, output: usize, offset: f64) {
    module.weight = Param::from_tensor(tensor([output, input], offset, 0.15).transpose());
    module.bias = Some(Param::from_tensor(tensor([output], offset + 0.4, 0.03)));
}

fn norm(module: &mut LayerNorm<B>, offset: f64) {
    let scales = (0..8)
        .map(|i| (0.7 + i as f64 * 0.05) as f32)
        .collect::<Vec<_>>();
    module.gamma = Param::from_tensor(Tensor::from_data(
        TensorData::new(scales, [8]),
        &Default::default(),
    ));
    module.beta = Some(Param::from_tensor(tensor([8], offset, 0.03)));
}

fn model() -> GptNeoXModel<B> {
    let mut model = GptNeoXModel::init(config(), &Default::default()).unwrap();
    model.embed_in.weight = Param::from_tensor(tensor([13, 8], 0.1, 0.15));
    for (i, layer) in model.layers.iter_mut().enumerate() {
        let offset = 1.0 + i as f64 * 10.0;
        norm(&mut layer.input_layernorm, offset);
        norm(&mut layer.post_attention_layernorm, offset + 0.5);
        linear(&mut layer.attention.q_proj, 8, 8, offset);
        linear(&mut layer.attention.k_proj, 8, 8, offset + 1.0);
        linear(&mut layer.attention.v_proj, 8, 8, offset + 2.0);
        linear(&mut layer.attention.o_proj, 8, 8, offset + 3.0);
        linear(&mut layer.mlp.up_proj, 8, 12, offset + 4.0);
        linear(&mut layer.mlp.down_proj, 12, 8, offset + 5.0);
    }
    norm(&mut model.final_layer_norm, 30.0);
    model.embed_out.weight = Param::from_tensor(tensor([13, 8], 31.0, 0.15).transpose());
    model
}

fn input() -> CausalLmInput<B> {
    CausalLmInput {
        token_ids: Tensor::from_data([[1, 4, 2]], &Default::default()),
        token_mask: None,
        position_ids: Some(Tensor::from_data([[0, 2, 5]], &Default::default())),
    }
}

fn options() -> ForwardOptions {
    ForwardOptions {
        logits: LogitsSelection::All,
        hidden_states: true,
        ..Default::default()
    }
}

fn difference(left: Tensor<B, 3>, right: Tensor<B, 3>) -> f32 {
    (left - right).abs().max().into_scalar()
}

fn assert_fixture(actual: Tensor<B, 3>, key: &str) {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/oracle.json")).unwrap();
    let values = fixture[key]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect::<Vec<_>>();
    let expected = Tensor::from_data(TensorData::new(values, actual.dims()), &Default::default());
    assert!(
        difference(actual, expected) < 2e-6,
        "PyTorch oracle for {key}"
    );
}

#[test]
fn matches_pytorch_partial_rope_parallel_residual_and_lens_boundaries() {
    let model = model();
    let mut hooks = Hooks::new();
    let first_block = hooks.capture(Layer::decoder(0).residual_post());
    let last_block = hooks.capture(Layer::decoder(1).residual_post());
    let embedding = hooks.capture(Stack::Decoder.embedding());
    let output = model
        .forward_with_hooks(input(), &options(), &mut hooks)
        .unwrap();
    assert_fixture(output.logits.clone(), "logits");
    assert_fixture(hooks.get(&first_block).unwrap(), "block0");
    assert_fixture(output.final_hidden_state.clone().unwrap(), "final");
    assert_eq!(output.hidden_states.len(), 3);
    assert!(
        difference(
            output.hidden_states[0].clone(),
            hooks.get(&embedding).unwrap()
        ) < 1e-7
    );
    assert!(
        difference(
            output.hidden_states[1].clone(),
            hooks.get(&first_block).unwrap()
        ) < 1e-7
    );
    assert!(
        difference(
            output.logits,
            model
                .project_hidden(hooks.get(&last_block).unwrap())
                .unwrap()
        ) < 1e-7
    );
}

#[test]
fn activation_ablation_and_ordered_restoration_change_downstream_logits() {
    let model = model();
    let point = Layer::decoder(0).mlp_activation().token(1);
    let mut observe = Hooks::new();
    let saved = observe.capture(&point);
    let clean = model
        .forward_with_hooks(input(), &options(), &mut observe)
        .unwrap();
    let original = observe.get(&saved).unwrap();
    assert_eq!(original.dims(), [1, 1, 12]);
    let mut ablate = Hooks::new();
    ablate.edit(&point, |x| x.zeros_like());
    let changed = model
        .forward_with_hooks(input(), &options(), &mut ablate)
        .unwrap();
    assert!(difference(clean.logits.clone(), changed.logits) > 1e-5);
    ablate.replace(&point, original);
    let restored = model
        .forward_with_hooks(input(), &options(), &mut ablate)
        .unwrap();
    assert!(difference(clean.logits, restored.logits) < 1e-7);
}

#[test]
fn attention_visibility_and_padding_remain_explicit() {
    let model = model();
    let changed = CausalLmInput {
        token_ids: Tensor::from_data([[1, 4, 12]], &Default::default()),
        ..input()
    };
    let clean_causal = model.forward(input(), &options()).unwrap().logits;
    let changed_causal = model.forward(changed.clone(), &options()).unwrap().logits;
    assert!(difference(clean_causal.narrow(1, 0, 1), changed_causal.narrow(1, 0, 1)) < 1e-7);
    let clean_bidirectional = model
        .forward_bidirectional_with_hooks(input(), &options(), &mut Hooks::new())
        .unwrap()
        .logits;
    let changed_bidirectional = model
        .forward_bidirectional_with_hooks(changed, &options(), &mut Hooks::new())
        .unwrap()
        .logits;
    assert!(
        difference(
            clean_bidirectional.narrow(1, 0, 1),
            changed_bidirectional.narrow(1, 0, 1)
        ) > 1e-5
    );
    let masked = CausalLmInput {
        token_mask: Some(TokenMask::new(Tensor::from_data(
            [[true, true, false]],
            &Default::default(),
        ))),
        ..input()
    };
    let all = model.forward(masked.clone(), &options()).unwrap();
    let last = model.forward(masked, &ForwardOptions::default()).unwrap();
    assert!(difference(last.logits, all.logits.narrow(1, 1, 1)) < 1e-7);
}

#[test]
fn invalid_inputs_and_unsupported_choices_return_contextual_errors() {
    let model = model();
    let bad_ids = CausalLmInput {
        token_ids: Tensor::from_data([[1, -1, 2]], &Default::default()),
        ..input()
    };
    assert!(
        model
            .forward(bad_ids, &options())
            .unwrap_err()
            .to_string()
            .contains("token_ids")
    );
    let bad_position = CausalLmInput {
        position_ids: Some(Tensor::from_data([[0, 2, 16]], &Default::default())),
        ..input()
    };
    assert!(
        model
            .forward(bad_position, &options())
            .unwrap_err()
            .to_string()
            .contains("position_ids")
    );
    let empty_mask = CausalLmInput {
        token_mask: Some(TokenMask::new(Tensor::from_data(
            [[false, false, false]],
            &Default::default(),
        ))),
        ..input()
    };
    assert!(
        model
            .forward(empty_mask, &options())
            .unwrap_err()
            .to_string()
            .contains("valid token")
    );
    let mut missing = Hooks::new();
    missing.capture(Layer::decoder(2).residual_post());
    assert!(
        model
            .forward_with_hooks(input(), &options(), &mut missing)
            .is_err()
    );
    assert!(
        GptNeoXConfig {
            hidden_size: 9,
            ..config()
        }
        .validate()
        .is_err()
    );
    assert!(
        GptNeoXConfig {
            rotary_pct: 0.25,
            ..config()
        }
        .validate()
        .is_err()
    );
    assert!(
        GptNeoXConfig {
            hidden_act: "gelu_new".into(),
            ..config()
        }
        .validate()
        .is_err()
    );
    assert!(
        GptNeoXConfig {
            tie_word_embeddings: true,
            ..config()
        }
        .validate()
        .is_err()
    );
}

#[test]
fn invalid_public_module_replacements_fail_before_backend_operations() {
    let mut changed = model();
    changed.embed_in = EmbeddingConfig::new(2, 8).init(&Default::default());
    assert!(
        changed
            .forward(input(), &options())
            .unwrap_err()
            .to_string()
            .contains("embed_in")
    );
    let mut changed = model();
    changed.final_layer_norm = LayerNormConfig::new(7).init(&Default::default());
    assert!(
        changed
            .forward(input(), &options())
            .unwrap_err()
            .to_string()
            .contains("final_layer_norm")
    );
    let mut changed = model();
    changed.layers[0].mlp.down_proj = LinearConfig::new(12, 7).init(&Default::default());
    assert!(
        changed
            .forward(input(), &options())
            .unwrap_err()
            .to_string()
            .contains("down_proj")
    );
}

#[test]
fn replacing_attention_derives_partial_rope_from_the_new_head_width() {
    use crate::{
        attention::GroupedQueryAttentionConfig,
        position::{RopeConfig, RopeScaling, apply_rotary, rotary_angles},
    };
    let mut model = model();
    model.layers[0].attention = GroupedQueryAttentionConfig::new(8, 1, 1, 8)
        .init(&Default::default())
        .unwrap();
    let mut hooks = Hooks::new();
    let normalized = hooks.capture(Layer::decoder(0).attention_input());
    let query = hooks.capture(Layer::decoder(0).self_attention().query());
    model
        .forward_with_hooks(input(), &options(), &mut hooks)
        .unwrap();
    let projection = model.layers[0]
        .attention
        .project_self(hooks.get(&normalized).unwrap())
        .unwrap();
    let angles = rotary_angles(
        RopeConfig::new(8, 4, 10_000.0, RopeScaling::None).unwrap(),
        input().position_ids.unwrap(),
        16,
    )
    .unwrap();
    let (expected, _) = apply_rotary(projection.query, projection.key, &angles).unwrap();
    let actual = hooks.get(&query).unwrap();
    assert!((actual - expected).abs().max().into_scalar() < 1e-7);
}
