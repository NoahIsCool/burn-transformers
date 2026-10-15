//! Automatic-family activation catalogs, identity hooks, and unsupported-hook cleanup.

use super::*;
use crate::intervention::{Hooks, Layer};

/// Qwen registry hooks expose token-scoped MLP captures while empty and identity hooks preserve logits.
#[test]
fn automatic_qwen_exposes_points_and_preserves_identity_outputs() {
    let device = Default::default();
    let model =
        AutoCausalLm::from(Qwen3_5ForCausalLm::<TestBackend>::init(tiny_qwen(), &device).unwrap());
    let point = Layer::decoder(1).mlp_activation();
    assert!(model.activation_points().unwrap().contains(&point.info()));
    let input = registry_tokens([[1, 2, 3], [4, 5, 6]], &device);
    let options = ForwardOptions::default();
    let expected = model.forward(input.clone(), &options).unwrap();

    let empty = model
        .forward_with_hooks(input.clone(), &options, &mut Hooks::new())
        .unwrap();
    assert_registry_logits(empty.logits, expected.logits.clone());

    let mut hooks = Hooks::new();
    let captured = hooks.capture(point.clone().token(1));
    hooks.edit(point, |tensor| tensor);
    let actual = model
        .forward_with_hooks(input, &options, &mut hooks)
        .unwrap();
    assert_registry_logits(actual.logits, expected.logits);
    assert_eq!(hooks.get(&captured).unwrap().dims(), [2, 1, 16]);
}

/// T5 registry hooks capture both stacks and preserve encoder states and decoder logits under identity edits.
#[test]
fn automatic_t5_exposes_both_stacks_and_preserves_identity_outputs() {
    let device = Default::default();
    let model = AutoSeq2SeqLm::from(
        T5ForConditionalGeneration::<TestBackend>::init(tiny_t5(), &device).unwrap(),
    );
    let encoder_point = Layer::encoder(0).mlp_activation();
    let decoder_point = Layer::decoder(0).mlp_activation();
    let points = model.activation_points().unwrap();
    assert!(points.contains(&encoder_point.info()));
    assert!(points.contains(&decoder_point.info()));
    let input = Seq2SeqInput {
        encoder: registry_tokens([[1, 2, 3], [4, 5, 6]], &device),
        decoder: registry_tokens([[0, 4], [0, 5]], &device),
    };
    let options = ForwardOptions::default();
    let expected = model.forward(input.clone(), &options).unwrap();

    let empty = model
        .forward_with_hooks(input.clone(), &options, &mut Hooks::new())
        .unwrap();
    assert_registry_logits(empty.decoder.logits, expected.decoder.logits.clone());

    let mut hooks = Hooks::new();
    let captured_encoder = hooks.capture(encoder_point.token(1));
    let captured_decoder = hooks.capture(decoder_point.clone().token(0));
    hooks.edit(decoder_point, |tensor| tensor);
    let actual = model
        .forward_with_hooks(input, &options, &mut hooks)
        .unwrap();
    assert_registry_logits(actual.decoder.logits, expected.decoder.logits);
    assert_registry_logits(actual.encoder.hidden_state, expected.encoder.hidden_state);
    assert_eq!(hooks.get(&captured_encoder).unwrap().dims(), [2, 1, 16]);
    assert_eq!(hooks.get(&captured_decoder).unwrap().dims(), [2, 1, 16]);
}

/// Llama-family registry hooks expose the family catalog and preserve identity outputs.
#[test]
fn automatic_llama_exposes_points_and_preserves_identity_outputs() {
    use crate::models::llama::{LlamaFamilyConfig, LlamaForCausalLm, LlamaVariant};
    let device = Default::default();
    let config = LlamaFamilyConfig {
        vocab_size: 32,
        hidden_size: 16,
        intermediate_size: 24,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dim: 4,
        max_position_embeddings: 32,
        eos_token_id: None,
        pad_token_id: None,
        bos_token_id: None,
        ..LlamaFamilyConfig::new(LlamaVariant::Qwen3)
    };
    let model = AutoCausalLm::from(LlamaForCausalLm::<TestBackend>::init(config, &device).unwrap());
    let point = Layer::decoder(1).mlp_activation();
    assert!(model.activation_points().unwrap().contains(&point.info()));
    let input = registry_tokens([[1, 2, 3], [4, 5, 6]], &device);
    let options = ForwardOptions::default();
    let expected = model.forward(input.clone(), &options).unwrap();
    let mut hooks = Hooks::new();
    let captured = hooks.capture(point.clone().token(1));
    hooks.edit(point, |tensor| tensor);
    let actual = model
        .forward_with_hooks(input, &options, &mut hooks)
        .unwrap();
    assert_registry_logits(actual.logits, expected.logits);
    assert_eq!(hooks.get(&captured).unwrap().dims(), [2, 1, 24]);
}

/// gpt-oss registry hooks expose the router points, preserve identity outputs, and apply real edits.
///
/// Zeroing one layer's router weights removes that block's routed experts
/// from the residual stream, so the logits must move: the edit reaches the
/// family's hooked pass rather than an unsupported or ordinary forward.
#[test]
fn automatic_gpt_oss_exposes_router_points_and_applies_edits() {
    use crate::models::{
        config::RopeParameters,
        gpt_oss::{GptOssConfig, GptOssForCausalLm},
        qwen3_moe::{router_logits_point, router_weights_point},
    };
    let device = Default::default();
    let config = GptOssConfig {
        vocab_size: 32,
        hidden_size: 16,
        intermediate_size: 12,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dim: 4,
        num_local_experts: 4,
        num_experts_per_tok: 2,
        sliding_window: Some(2),
        layer_types: GptOssConfig::default_layer_types(2),
        max_position_embeddings: 32,
        rope_parameters: RopeParameters {
            factor: Some(4.0),
            original_max_position_embeddings: Some(8),
            ..GptOssConfig::default_rope_parameters()
        },
        ..GptOssConfig::default()
    };
    let model =
        AutoCausalLm::from(GptOssForCausalLm::<TestBackend>::init(config, &device).unwrap());
    let layer = Layer::decoder(1);
    let points = model.activation_points().unwrap();
    for point in [
        router_logits_point(layer).info(),
        router_weights_point(layer).info(),
        layer.mlp_output().info(),
    ] {
        assert!(points.contains(&point), "missing {}", point.name);
    }
    let input = registry_tokens([[1, 2, 3], [4, 5, 6]], &device);
    let options = ForwardOptions::default();
    let expected = model.forward(input.clone(), &options).unwrap();

    let mut hooks = Hooks::new();
    let logits = hooks.capture(router_logits_point(layer));
    hooks.edit(router_weights_point(layer), |tensor| tensor);
    let identity = model
        .forward_with_hooks(input.clone(), &options, &mut hooks)
        .unwrap();
    assert_registry_logits(identity.logits, expected.logits.clone());
    assert_eq!(hooks.get(&logits).unwrap().dims(), [2, 3, 4]);

    let mut hooks = Hooks::new();
    hooks.edit(router_weights_point(layer), |tensor| tensor.zeros_like());
    let edited = model
        .forward_with_hooks(input, &options, &mut hooks)
        .unwrap();
    let moved = (edited.logits - expected.logits).abs().max().into_scalar();
    assert!(
        moved > 1e-4,
        "zeroed router weights left the logits unchanged"
    );
}

/// Unsupported MoE hooks return family-specific capability errors and clear captures before any callback.
#[test]
fn unsupported_automatic_hooks_fail_before_callbacks_and_clear_stale_captures() {
    let device = Default::default();
    let model = AutoCausalLm::from(
        Qwen3MoeForCausalLm::<TestBackend>::init(tiny_qwen_moe(), &device).unwrap(),
    );
    let catalog_error = model.activation_points().unwrap_err();
    assert!(matches!(catalog_error, Error::BackendCapability(_)));
    assert!(catalog_error.to_string().contains("qwen3_moe"));

    let point = Layer::decoder(0).mlp_activation();
    let mut hooks = Hooks::new();
    let saved = hooks.capture(point.clone());
    hooks
        .run(vec![point.info()], |hooks| {
            hooks.apply(point.clone(), Tensor::ones([1, 1, 2], &device))
        })
        .unwrap();
    assert!(hooks.get(&saved).is_ok());
    hooks.edit(point, |_| {
        panic!("unsupported model must fail before callbacks")
    });

    let input = registry_tokens([[1, 2], [3, 4]], &device);
    let options = ForwardOptions::default();
    let error = model
        .forward_with_hooks(input.clone(), &options, &mut hooks)
        .unwrap_err();
    assert!(matches!(error, Error::BackendCapability(_)));
    assert!(error.to_string().contains("qwen3_moe"));
    assert!(hooks.get(&saved).is_err());
    assert!(matches!(
        model.forward_with_hooks(input.clone(), &options, &mut Hooks::new()),
        Err(Error::BackendCapability(_))
    ));
    assert!(model.forward(input, &options).is_ok());
}
