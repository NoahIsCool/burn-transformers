//! Automatic-family activation catalogs, hooked and staged dispatch, and capability failures.
//!
//! Every registered causal family is exercised through [`AutoCausalLm`]: a
//! family whose registry entry is `experiments: supported` must capture and
//! edit activations and stage its stack exactly like the concrete family, and
//! a `pending` family must fail every experiment call with a capability error
//! before any callback runs. The table is exhaustive over
//! [`AutoCausalConfig`], so a new registry entry needs a fixture here.

use super::*;
use crate::{
    architecture::{HookedModel, StagedModel},
    intervention::{Hooks, Layer, Stack},
    models::{
        config::RopeParameters,
        deepseek_v2::model::tests_support as deepseek,
        gemma::{GemmaConfig, GemmaForCausalLm, GemmaVariant},
        gpt_oss::{GptOssConfig, GptOssForCausalLm},
        llama::{LlamaFamilyConfig, LlamaForCausalLm, LlamaVariant},
    },
};

/// Tiny Gemma 3 text configuration with one local and one global layer.
fn tiny_gemma() -> GemmaConfig {
    GemmaConfig {
        vocab_size: 32,
        hidden_size: 16,
        intermediate_size: 24,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dim: 4,
        max_position_embeddings: 32,
        sliding_window: 4,
        pad_token_id: None,
        bos_token_id: None,
        eos_token_id: None,
        ..GemmaConfig::new(GemmaVariant::Gemma3Text)
    }
}

/// Tiny gpt-oss configuration: two alternating sliding/full layers, four experts, top-2.
fn tiny_gpt_oss() -> GptOssConfig {
    GptOssConfig {
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
    }
}

/// Tiny Qwen3 configuration of the Llama family with two layers.
fn tiny_llama() -> LlamaFamilyConfig {
    LlamaFamilyConfig {
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
    }
}

/// Builds a tiny randomly initialized model of `family`, wrapped in the automatic enum.
///
/// The match is exhaustive, so registering a family without a fixture here
/// fails to compile instead of silently skipping it.
fn tiny_family_model(
    family: AutoCausalConfig,
    device: &RegistryDevice,
) -> AutoCausalLm<TestBackend> {
    match family {
        AutoCausalConfig::DeepseekV2 => {
            deepseek::initialized_model(deepseek::tiny_config(), device).into()
        }
        AutoCausalConfig::MuseGlimmer => MuseGlimmerForCausalLm::init(tiny_muse(), device)
            .expect("tiny Muse model")
            .into(),
        AutoCausalConfig::NemotronH => NemotronHForCausalLm::init(tiny_nemotron(), device)
            .expect("tiny Nemotron-H model")
            .into(),
        AutoCausalConfig::Qwen3_5 => Qwen3_5ForCausalLm::init(tiny_qwen(), device)
            .expect("tiny Qwen3.5 model")
            .into(),
        AutoCausalConfig::Qwen3Moe => Qwen3MoeForCausalLm::init(tiny_qwen_moe(), device)
            .expect("tiny Qwen3-MoE model")
            .into(),
        AutoCausalConfig::Llama => LlamaForCausalLm::init(tiny_llama(), device)
            .expect("tiny Llama-family model")
            .into(),
        AutoCausalConfig::Gemma => GemmaForCausalLm::init(tiny_gemma(), device)
            .expect("tiny Gemma model")
            .into(),
        AutoCausalConfig::GptOss => GptOssForCausalLm::init(tiny_gpt_oss(), device)
            .expect("tiny gpt-oss model")
            .into(),
    }
}

/// Largest absolute elementwise difference between two equally shaped tensors.
fn max_difference(left: Tensor<TestBackend, 3>, right: Tensor<TestBackend, 3>) -> f32 {
    (left - right).abs().max().into_scalar()
}

/// Captures every advertised point through a generic [`HookedModel`] bound.
///
/// This is the TransformerLens-style activation cache a family-generic tool
/// writes once; it must work for the automatic enum and a loaded bundle alike.
/// Returns the number of captured points.
fn capture_every_point<M: HookedModel<TestBackend, Input = CausalLmInput<TestBackend>>>(
    model: &M,
    input: CausalLmInput<TestBackend>,
) -> usize {
    let mut hooks = Hooks::new();
    let handles = model
        .activation_points()
        .expect("catalog")
        .into_iter()
        .map(|point| hooks.capture_dynamic(point))
        .collect::<Vec<_>>();
    model
        .forward_with_hooks(input, &ForwardOptions::default(), &mut hooks)
        .expect("hooked pass capturing every point");
    for handle in &handles {
        let activation = hooks.get_dynamic(handle).expect("captured activation");
        assert_eq!(activation.shape.len(), handle.point().rank());
    }
    handles.len()
}

/// Checks hooks and staged execution of one `experiments: supported` family.
fn assert_supported_experiments(model: &AutoCausalLm<TestBackend>, device: &RegistryDevice) {
    let family = model.family();
    let input = registry_tokens([[1, 2, 3], [4, 5, 6]], device);
    let options = ForwardOptions::default();
    let expected = model.forward(input.clone(), &options).unwrap();

    // The catalog is the family's own, through either the inherent or the trait path.
    let points = model.activation_points().unwrap();
    assert_eq!(
        points,
        HookedModel::<TestBackend>::activation_points(model).unwrap()
    );
    let residual = Layer::decoder(0).residual_post();
    for point in [
        Stack::Decoder.embedding().info(),
        residual.info(),
        Stack::Decoder.final_norm().info(),
    ] {
        assert!(points.contains(&point), "{family:?} lacks {}", point.name);
    }

    // Empty hooks and identity edits reproduce the stateless forward.
    let empty = model
        .forward_with_hooks(input.clone(), &options, &mut Hooks::new())
        .unwrap();
    assert_registry_logits(empty.logits, expected.logits.clone());
    let mut identity = Hooks::new();
    let captured = identity.capture(residual.clone().token(1));
    identity.edit(residual.clone(), |tensor| tensor);
    let unchanged = model
        .forward_with_hooks(input.clone(), &options, &mut identity)
        .unwrap();
    assert_registry_logits(unchanged.logits, expected.logits.clone());
    let [batch, tokens, _] = identity.get(&captured).unwrap().dims();
    assert_eq!([batch, tokens], [2, 1], "{family:?}");

    // Ablating the first block's residual output changes the logits.
    let mut ablate = Hooks::new();
    ablate.edit(residual, |tensor| tensor.zeros_like());
    let ablated = model.forward_with_hooks(input.clone(), &options, &mut ablate);
    let change = max_difference(ablated.unwrap().logits, expected.logits);
    assert!(change > 1e-6, "{family:?}: ablation changed nothing");

    // A generic activation cache sees every advertised point.
    assert_eq!(capture_every_point(model, input.clone()), points.len());

    // Staged execution composes to the complete stack, in any split.
    let layers = StagedModel::<TestBackend>::num_layers(model);
    assert!(layers > 0, "{family:?} stages no blocks");
    assert_eq!(StagedModel::<TestBackend>::stack(model), Stack::Decoder);
    let whole = model.run_stack(input.clone()).unwrap();
    assert!(whole.normalized && whole.next_layer == layers);
    let state = model.embed(input.clone()).unwrap();
    let state = model.run_layers(state, 0..1).unwrap();
    let state = model.run_layers(state, 1..layers).unwrap();
    let split = model.normalize(state).unwrap();
    assert_eq!(
        max_difference(split.hidden, whole.hidden.clone()),
        0.0,
        "{family:?}: staged splits disagree"
    );
    let with_states = model
        .forward(
            input.clone(),
            &ForwardOptions {
                hidden_states: true,
                ..ForwardOptions::default()
            },
        )
        .unwrap();
    if let Some(final_hidden) = with_states.final_hidden_state {
        assert_registry_logits(whole.hidden, final_hidden);
    }

    // A staged scope emits only its own stage's points.
    let mut embedding_hooks = Hooks::new();
    let embedding = embedding_hooks.capture(Stack::Decoder.embedding());
    model.embed_with_hooks(input, &mut embedding_hooks).unwrap();
    assert_eq!(embedding_hooks.get(&embedding).unwrap().dims()[0], 2);
}

/// Checks that a `pending` family fails every experiment call before callbacks run.
fn assert_pending_experiments(model: &AutoCausalLm<TestBackend>, device: &RegistryDevice) {
    let model_type = model.config().model_type();
    let capability = |error: Error| {
        assert!(matches!(error, Error::BackendCapability(_)), "{error:?}");
        assert!(error.to_string().contains(model_type), "{error}");
    };
    capability(model.activation_points().unwrap_err());

    // A stale capture from an earlier scope is cleared, and no callback runs.
    let point = Layer::decoder(0).residual_post();
    let mut hooks = Hooks::new();
    let saved = hooks.capture(point.clone());
    hooks
        .run(vec![point.info()], |hooks| {
            hooks.apply(point.clone(), Tensor::ones([1, 1, 2], device))
        })
        .unwrap();
    assert!(hooks.get(&saved).is_ok());
    hooks.edit(point, |_| {
        panic!("a pending family must fail before callbacks")
    });
    let input = registry_tokens([[1, 2], [3, 4]], device);
    let options = ForwardOptions::default();
    capability(
        model
            .forward_with_hooks(input.clone(), &options, &mut hooks)
            .unwrap_err(),
    );
    assert!(hooks.get(&saved).is_err());
    capability(
        model
            .forward_with_hooks(input.clone(), &options, &mut Hooks::new())
            .unwrap_err(),
    );

    // Staged execution reports the same capability gap.
    assert_eq!(StagedModel::<TestBackend>::num_layers(model), 0);
    capability(model.embed(input.clone()).unwrap_err());
    capability(model.run_stack(input.clone()).unwrap_err());

    // Ordinary inference is unaffected.
    assert!(model.forward(input, &options).is_ok());
}

/// Every registered causal family captures, edits, and stages through the automatic enum.
#[test]
fn every_registered_causal_family_dispatches_experiments_through_the_auto_enum() {
    let device = Default::default();
    for &family in AutoCausalConfig::ALL {
        let model = tiny_family_model(family, &device);
        assert_eq!(model.family(), family);
        if family.supports_experiments() {
            assert_supported_experiments(&model, &device);
        } else {
            assert_pending_experiments(&model, &device);
        }
    }
}

/// The families whose hook implementations have landed are registered as supported.
#[test]
fn registry_declares_the_families_with_hook_implementations() {
    let supported = AutoCausalConfig::ALL
        .iter()
        .filter(|family| family.supports_experiments())
        .map(|family| family.model_type())
        .collect::<Vec<_>>();
    for model_type in ["nemotron_h", "qwen3_5_text", "qwen3_moe", "llama"] {
        assert!(supported.contains(&model_type), "{model_type}");
    }
}

/// A loaded bundle is itself a hooked and staged model, delegating to its model.
#[test]
fn loaded_bundles_forward_experiment_traits_to_their_model() {
    let device = Default::default();
    let model = tiny_family_model(AutoCausalConfig::Llama, &device);
    let directory = tempfile::tempdir().expect("temporary source root");
    let source = crate::ModelSourceMetadata::local(directory.path(), "llama").unwrap();
    let loaded = LoadedModel::new(
        model,
        None,
        GenerationConfig::default(),
        source,
        crate::checkpoint::LoadReport::new(),
        32,
    )
    .unwrap();
    let input = registry_tokens([[1, 2, 3], [4, 5, 6]], &device);
    assert_eq!(
        HookedModel::<TestBackend>::activation_points(&loaded).unwrap(),
        loaded.model().activation_points().unwrap()
    );
    assert_eq!(
        capture_every_point(&loaded, input.clone()),
        loaded.model().activation_points().unwrap().len()
    );
    assert_eq!(
        StagedModel::<TestBackend>::num_layers(&loaded),
        StagedModel::<TestBackend>::num_layers(loaded.model())
    );
    assert_eq!(StagedModel::<TestBackend>::stack(&loaded), Stack::Decoder);
    let through_bundle = loaded.run_stack(input.clone()).unwrap();
    let through_model = loaded.model().run_stack(input).unwrap();
    assert_eq!(
        max_difference(through_bundle.hidden, through_model.hidden),
        0.0
    );
}

/// T5 registry hooks capture both stacks, preserve identity outputs, and apply edits.
#[test]
fn automatic_t5_exposes_both_stacks_and_preserves_identity_outputs() {
    let device = Default::default();
    let model = AutoSeq2SeqLm::from(
        T5ForConditionalGeneration::<TestBackend>::init(tiny_t5(), &device).unwrap(),
    );
    let encoder_point = Layer::encoder(0).mlp_activation();
    let decoder_point = Layer::decoder(0).mlp_activation();
    let points = model.activation_points().unwrap();
    assert_eq!(
        points,
        HookedModel::<TestBackend>::activation_points(&model).unwrap()
    );
    assert!(points.contains(&encoder_point.info()));
    assert!(points.contains(&decoder_point.info()));
    let input = Seq2SeqInput {
        encoder: registry_tokens([[1, 2, 3], [4, 5, 6]], &device),
        decoder: registry_tokens([[0, 4], [0, 5]], &device),
    };
    let options = ForwardOptions::default();
    let expected = model.forward(input.clone(), &options).unwrap();

    let empty = HookedModel::<TestBackend>::forward_with_hooks(
        &model,
        input.clone(),
        &options,
        &mut Hooks::new(),
    )
    .unwrap();
    assert_registry_logits(empty.decoder.logits, expected.decoder.logits.clone());

    let mut hooks = Hooks::new();
    let captured_encoder = hooks.capture(encoder_point.token(1));
    let captured_decoder = hooks.capture(decoder_point.clone().token(0));
    hooks.edit(decoder_point.clone(), |tensor| tensor);
    let actual = model
        .forward_with_hooks(input.clone(), &options, &mut hooks)
        .unwrap();
    assert_registry_logits(actual.decoder.logits, expected.decoder.logits.clone());
    assert_registry_logits(actual.encoder.hidden_state, expected.encoder.hidden_state);
    assert_eq!(hooks.get(&captured_encoder).unwrap().dims(), [2, 1, 16]);
    assert_eq!(hooks.get(&captured_decoder).unwrap().dims(), [2, 1, 16]);

    // Editing a decoder activation changes the decoder logits.
    let mut ablate = Hooks::new();
    ablate.edit(decoder_point, |tensor| tensor.zeros_like());
    let ablated = model
        .forward_with_hooks(input, &options, &mut ablate)
        .unwrap();
    assert!(max_difference(ablated.decoder.logits, expected.decoder.logits) > 1e-6);
}

/// gpt-oss registry hooks expose the router points, preserve identity outputs, and apply real edits.
///
/// Zeroing one layer's router weights removes that block's routed experts
/// from the residual stream, so the logits must move: the edit reaches the
/// family's hooked pass rather than an unsupported or ordinary forward.
#[test]
fn automatic_gpt_oss_exposes_router_points_and_applies_edits() {
    let device = Default::default();
    let config = tiny_gpt_oss();
    let model =
        AutoCausalLm::from(GptOssForCausalLm::<TestBackend>::init(config, &device).unwrap());
    let layer = Layer::decoder(1);
    let points = model.activation_points().unwrap();
    for point in [
        layer.router_logits().info(),
        layer.router_weights().info(),
        layer.mlp_output().info(),
    ] {
        assert!(points.contains(&point), "missing {}", point.name);
    }
    let input = registry_tokens([[1, 2, 3], [4, 5, 6]], &device);
    let options = ForwardOptions::default();
    let expected = model.forward(input.clone(), &options).unwrap();

    let mut hooks = Hooks::new();
    let logits = hooks.capture(layer.router_logits());
    hooks.edit(layer.router_weights(), |tensor| tensor);
    let identity = model
        .forward_with_hooks(input.clone(), &options, &mut hooks)
        .unwrap();
    assert_registry_logits(identity.logits, expected.logits.clone());
    assert_eq!(hooks.get(&logits).unwrap().dims(), [2, 3, 4]);

    let mut hooks = Hooks::new();
    hooks.edit(layer.router_weights(), |tensor| tensor.zeros_like());
    let edited = model
        .forward_with_hooks(input, &options, &mut hooks)
        .unwrap();
    let moved = (edited.logits - expected.logits).abs().max().into_scalar();
    assert!(
        moved > 1e-4,
        "zeroed router weights left the logits unchanged"
    );
}
