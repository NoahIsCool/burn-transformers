use super::*;
use crate::intervention::{Hooks, Layer};

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
