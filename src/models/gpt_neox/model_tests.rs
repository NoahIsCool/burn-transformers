//! GPT-NeoX parameter validation, residual wiring, precision modes, and initialization.

use burn::{backend::Flex, module::Param, tensor::Bool};

use super::*;
use crate::LogitsSelection;

/// Small CPU backend used to test metadata validation without large allocations.
type B = Flex<f32, i32>;

/// Initializes ordinary tiny parameters before tests edit only attention metadata.
fn model() -> GptNeoXForCausalLm<B> {
    GptNeoXForCausalLm::init(
        GptNeoXConfig {
            vocab_size: 13,
            hidden_size: 8,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            intermediate_size: 12,
            rotary_pct: 0.5,
            max_position_embeddings: 16,
            ..Default::default()
        },
        &Default::default(),
    )
    .expect("the tiny topology must initialize")
}

/// Zero, incompatible, or residual-changing heads return contextual configuration errors.
#[test]
fn invalid_attention_geometry_is_rejected_before_projection_validation() {
    for (hidden_size, query_heads, key_value_heads, head_dim) in [
        (7, 2, 2, 4),
        (8, 0, 2, 4),
        (8, 2, 0, 4),
        (8, 2, 2, 0),
        (8, 3, 2, 4),
    ] {
        let mut model = model();
        let attention = &mut model.layers[0].attention;
        attention.hidden_size = hidden_size;
        attention.query_heads = query_heads;
        attention.key_value_heads = key_value_heads;
        attention.head_dim = head_dim;
        let error = model.validate_parameters().expect_err("invalid geometry");
        assert!(error.to_string().contains("gpt_neox.layers.0.attention"));
        assert!(error.to_string().contains("nonzero, compatible"));
    }
}

/// Multiplication overflow in query or key/value geometry returns an error without allocation.
#[test]
fn projection_width_overflow_is_reported_from_edited_head_counts() {
    for key_value_heads in [1, usize::MAX] {
        let mut model = model();
        model.layers[0].attention.query_heads = usize::MAX;
        model.layers[0].attention.key_value_heads = key_value_heads;
        let error = model.validate_parameters().expect_err("overflowing width");
        assert!(error.to_string().contains("gpt_neox.layers.0.attention"));
        assert!(error.to_string().contains("projection widths overflow"));
    }
}

/// A fitting ungated query width can still overflow when the output gate doubles it.
#[test]
fn gated_projection_width_overflow_is_reported_separately() {
    let mut model = model();
    let attention = &mut model.layers[0].attention;
    attention.query_heads = usize::MAX / 2 + 1;
    attention.key_value_heads = 1;
    attention.head_dim = 1;
    attention.output_gate = AttentionOutputGate::Sigmoid;
    let error = model
        .validate_parameters()
        .expect_err("gated width overflow");
    assert!(error.to_string().contains("gpt_neox.layers.0"));
    assert!(error.to_string().contains("gated query width overflows"));
}

/// An unchanged initialized model satisfies all projection and normalization contracts.
#[test]
fn initialized_attention_parameters_pass_validation() {
    model()
        .validate_parameters()
        .expect("valid tiny parameters");
}

/// Parameter labels render their dotted paths only when an error needs them.
#[test]
fn parameter_labels_render_dotted_paths() {
    assert_eq!(Label::Model("embed_out").render(), "gpt_neox.embed_out");
    assert_eq!(
        Label::Layer(3, "attention.q_proj").render(),
        "gpt_neox.layers.3.attention.q_proj"
    );
    assert_eq!(Label::Layer(3, "").render(), "gpt_neox.layers.3");
}

/// A two-layer model with nontrivial random parameters for wiring checks.
fn two_layer(parallel: bool) -> GptNeoXForCausalLm<B> {
    let device = Default::default();
    B::seed(&device, 11);
    GptNeoXForCausalLm::init(
        GptNeoXConfig {
            vocab_size: 13,
            hidden_size: 8,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            intermediate_size: 12,
            rotary_pct: 0.5,
            max_position_embeddings: 16,
            use_parallel_residual: parallel,
            initializer_range: 0.5,
            ..Default::default()
        },
        &device,
    )
    .unwrap()
}

/// Three tokens with a padded last column.
fn padded_input() -> CausalLmInput<B> {
    let device = Default::default();
    CausalLmInput {
        token_ids: Tensor::<B, 2, Int>::from_ints([[1, 4, 2], [3, 5, 0]], &device),
        token_mask: Some(TokenMask::new(Tensor::<B, 2, Bool>::from_bool(
            [[true, true, true], [true, true, false]].into(),
            &device,
        ))),
        position_ids: None,
    }
}

/// Largest absolute elementwise difference between two equal-shaped tensors.
fn gap(left: Tensor<B, 3>, right: Tensor<B, 3>) -> f32 {
    (left - right).abs().max().into_scalar()
}

/// Parallel and sequential residual modes wire attention and MLP exactly as documented.
///
/// For every block, `y = x + attention + mlp`. In parallel mode the MLP
/// normalizes the block input, `LN_m(x)`; in sequential mode it normalizes
/// the attention-updated residual, `LN_m(x + attention)`. The checks use the
/// block's actual tensors captured through hooks.
#[test]
fn parallel_and_sequential_residuals_compose_as_documented() {
    let mut logits = Vec::new();
    for parallel in [true, false] {
        let model = two_layer(parallel);
        let mut hooks = Hooks::new();
        let captures: Vec<_> = (0..2)
            .map(|index| {
                let layer = Layer::decoder(index);
                (
                    hooks.capture(layer.residual_pre()),
                    hooks.capture(layer.attention_output()),
                    hooks.capture(layer.mlp_input()),
                    hooks.capture(layer.mlp_output()),
                    hooks.capture(layer.residual_post()),
                )
            })
            .collect();
        let output = model
            .forward_with_hooks(padded_input(), &ForwardOptions::default(), &mut hooks)
            .unwrap();
        for (index, (pre, attention, mlp_input, mlp, post)) in captures.iter().enumerate() {
            let pre = hooks.get(pre).unwrap();
            let attention = hooks.get(attention).unwrap();
            let mlp = hooks.get(mlp).unwrap();
            let expected_post = pre.clone() + attention.clone() + mlp;
            assert!(
                gap(hooks.get(post).unwrap(), expected_post) < 1e-5,
                "layer {index}"
            );
            let normalized = if parallel { pre } else { pre + attention };
            let expected_input = model.layers[index]
                .post_attention_layernorm
                .forward(normalized);
            assert!(
                gap(hooks.get(mlp_input).unwrap(), expected_input) < 1e-6,
                "layer {index} parallel={parallel}"
            );
        }
        logits.push(output.logits);
    }
    assert!(
        gap(logits[0].clone(), logits[1].clone()) > 1e-3,
        "modes must differ"
    );
}

/// Backend-default attention matches the eager graph and drops only score and pattern points.
#[test]
fn backend_default_attention_matches_eager_and_drops_score_points() {
    let eager = two_layer(true);
    let options = ForwardOptions {
        logits: LogitsSelection::All,
        hidden_states: true,
        ..ForwardOptions::default()
    };
    let expected = eager.forward(padded_input(), &options).unwrap();
    let fused = eager
        .clone()
        .with_attention_precision(AttentionPrecisionMode::BackendDefault);
    let actual = fused.forward(padded_input(), &options).unwrap();
    assert!(gap(actual.logits, expected.logits) < 2e-5);
    for (actual, expected) in actual.hidden_states.into_iter().zip(expected.hidden_states) {
        assert!(gap(actual, expected) < 2e-5);
    }
    let points = fused.activation_points();
    let attention = Layer::decoder(1).self_attention();
    assert!(!points.contains(&attention.scores().info()));
    assert!(!points.contains(&attention.pattern().info()));
    assert_eq!(points.len() + 4, eager.activation_points().len());
    let mut restored = fused;
    restored.set_attention_precision(AttentionPrecisionMode::EagerFp32Softmax);
    assert_eq!(restored.activation_points(), eager.activation_points());
}

/// Hidden states are the embedding, every block output, and the final normalized state.
#[test]
fn hidden_states_follow_the_crate_convention() {
    let model = two_layer(true);
    let mut hooks = Hooks::new();
    let embedding = hooks.capture(Stack::Decoder.embedding());
    let blocks: Vec<_> = (0..2)
        .map(|index| hooks.capture(Layer::decoder(index).residual_post()))
        .collect();
    let output = model
        .forward_with_hooks(
            padded_input(),
            &ForwardOptions {
                hidden_states: true,
                ..ForwardOptions::default()
            },
            &mut hooks,
        )
        .unwrap();
    assert_eq!(output.hidden_states.len(), model.layers.len() + 2);
    assert_eq!(
        output.hidden_states[0].clone().into_data(),
        hooks.get(&embedding).unwrap().into_data()
    );
    for (index, block) in blocks.iter().enumerate() {
        assert_eq!(
            output.hidden_states[index + 1].clone().into_data(),
            hooks.get(block).unwrap().into_data()
        );
    }
    assert_eq!(
        output.hidden_states[3].clone().into_data(),
        output.final_hidden_state.unwrap().into_data()
    );
}

/// The constructor and [`GptNeoXConfig::init_policy`] describe the same distributions.
#[test]
fn initialization_follows_the_policy_and_hugging_face() {
    use crate::initialization::{ParameterInit, ParameterRole};

    let config = GptNeoXConfig {
        vocab_size: 256,
        hidden_size: 64,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        intermediate_size: 128,
        max_position_embeddings: 64,
        initializer_range: 0.03,
        ..Default::default()
    };
    let device = Default::default();
    B::seed(&device, 5);
    let model = GptNeoXForCausalLm::<B>::init(config.clone(), &device).unwrap();
    let spread = |tensor: Tensor<B, 2>| {
        let values = tensor.into_data().to_vec::<f32>().unwrap();
        let count = values.len() as f64;
        let mean = values.iter().map(|&value| f64::from(value)).sum::<f64>() / count;
        (values
            .iter()
            .map(|&value| (f64::from(value) - mean).powi(2))
            .sum::<f64>()
            / count)
            .sqrt()
    };
    let near = |actual: f64| (actual / 0.03 - 1.0).abs() < 0.1;
    assert!(near(spread(model.embed_in.weight.val())));
    assert!(near(spread(model.embed_out.weight.val())));
    for layer in &model.layers {
        for projection in [
            &layer.attention.q_proj,
            &layer.attention.o_proj,
            &layer.mlp.up_proj,
            &layer.mlp.down_proj,
        ] {
            assert!(near(spread(projection.weight.val())));
            let bias = projection.bias.as_ref().unwrap().val();
            assert!(
                bias.into_data()
                    .to_vec::<f32>()
                    .unwrap()
                    .iter()
                    .all(|&value| value == 0.0)
            );
        }
    }
    let policy = config.init_policy();
    policy.validate_for(&model).unwrap();
    let (_, report) = policy.apply(model).unwrap();
    for parameter in &report.parameters {
        let expected = match parameter.role {
            ParameterRole::Embedding | ParameterRole::Projection => {
                ParameterInit::Normal { std: 0.03 }
            }
            ParameterRole::Bias => ParameterInit::Zeros,
            ParameterRole::ResidualProjection | ParameterRole::Other => ParameterInit::Keep,
        };
        assert_eq!(parameter.init, expected, "{}", parameter.path);
    }
}

/// Applying [`GptNeoXConfig::init_policy`] re-samples projections but keeps LayerNorm parameters.
///
/// Pins the documented limit of the policy: it cannot express LayerNorm's
/// ones, so re-initializing an edited model leaves its norms untouched.
#[test]
fn applying_the_initialization_policy_keeps_layer_norm_parameters() {
    use burn::module::Param;

    let device = Default::default();
    let mut model = model();
    let width = model.config.hidden_size;
    model.final_layer_norm.gamma = Param::from_tensor(Tensor::full([width], 2.0, &device));
    model.final_layer_norm.beta = Some(Param::from_tensor(Tensor::full([width], 0.5, &device)));
    let projection_before = model.layers[0].attention.q_proj.weight.val();
    let policy = model.config.init_policy();
    let (model, _) = policy.apply(model).unwrap();

    let values = |tensor: Tensor<B, 1>| tensor.into_data().to_vec::<f32>().unwrap();
    let norm = &model.final_layer_norm;
    assert!(values(norm.gamma.val()).iter().all(|&value| value == 2.0));
    assert!(
        values(norm.beta.as_ref().unwrap().val())
            .iter()
            .all(|&value| value == 0.5)
    );
    assert_ne!(
        projection_before.into_data().to_vec::<f32>().unwrap(),
        model.layers[0]
            .attention
            .q_proj
            .weight
            .val()
            .into_data()
            .to_vec::<f32>()
            .unwrap()
    );
}

/// GPT-NeoX configurations load through the shared `ModelConfig` API with field-level errors.
#[test]
fn configurations_load_through_the_shared_model_config_api() {
    use crate::models::config::{ConfigLoadError, ModelConfig, Validate};

    let json = r#"{"model_type": "gpt_neox", "vocab_size": 13, "hidden_size": 8,
                   "num_hidden_layers": 1, "num_attention_heads": 2,
                   "intermediate_size": 12, "rotary_pct": 0.5,
                   "max_position_embeddings": 16}"#;
    let config = GptNeoXConfig::from_json_str(json).unwrap();
    assert_eq!(config.hidden_size, 8);
    assert_eq!(<GptNeoXConfig as ModelConfig>::MODEL_TYPES, ["gpt_neox"]);
    crate::ArchitectureConfig::validate(&config).unwrap();
    for (edit, field) in [
        (("\"hidden_size\": 8", "\"hidden_size\": 9"), "hidden_size"),
        (
            ("\"rotary_pct\": 0.5", "\"rotary_pct\": 0.25"),
            "rotary_pct",
        ),
        (
            ("\"model_type\": \"gpt_neox\"", "\"model_type\": \"gpt2\""),
            "model_type",
        ),
    ] {
        match GptNeoXConfig::from_json_str(&json.replace(edit.0, edit.1)).unwrap_err() {
            ConfigLoadError::Validation(error) => assert_eq!(error.field, field),
            other => panic!("expected a validation error, found {other}"),
        }
    }
    let invalid = GptNeoXConfig {
        hidden_size: 9,
        ..config
    };
    assert_eq!(
        Validate::validate(&invalid).unwrap_err().field,
        "hidden_size"
    );
    assert!(
        invalid
            .validate()
            .unwrap_err()
            .to_string()
            .contains("gpt_neox.hidden_size")
    );
}

/// Unwraps a shape error into `(tensor, expected, actual)`.
fn shape_parts(error: Error) -> (String, String, String) {
    match error {
        Error::Shape {
            tensor,
            expected,
            actual,
        } => (tensor, expected, actual),
        other => panic!("expected Error::Shape, found {other:?}"),
    }
}

/// Unwraps a configuration error into `(field, reason)`.
fn configuration_parts(error: Error) -> (String, String) {
    match error {
        Error::Configuration { field, reason } => (field, reason),
        other => panic!("expected Error::Configuration, found {other:?}"),
    }
}

/// A failing edit at any emitted point aborts causal, bidirectional, and hidden-only passes.
#[test]
fn every_activation_point_propagates_edit_failures() {
    use crate::models::shared::test_support::assert_edit_failures_propagate;
    for parallel in [true, false] {
        let model = two_layer(parallel);
        let points = model.activation_points();
        assert_edit_failures_propagate(&points, |hooks| {
            model.forward_with_hooks(padded_input(), &ForwardOptions::default(), hooks)
        });
        assert_edit_failures_propagate(&points, |hooks| {
            model.forward_bidirectional_with_hooks(
                padded_input(),
                &ForwardOptions::default(),
                hooks,
            )
        });
        assert_edit_failures_propagate(&points, |hooks| {
            model.forward_hidden_with_hooks(padded_input(), hooks)
        });
    }
}

/// Bidirectional attention lets earlier tokens read later ones; both entry points agree.
#[test]
fn bidirectional_forward_reads_future_tokens() {
    let model = two_layer(true);
    let options = ForwardOptions {
        logits: LogitsSelection::All,
        ..ForwardOptions::default()
    };
    let plain = model
        .forward_bidirectional(padded_input(), &options)
        .unwrap();
    let hooked = model
        .forward_bidirectional_with_hooks(padded_input(), &options, &mut Hooks::new())
        .unwrap();
    assert_eq!(plain.logits.clone().into_data(), hooked.logits.into_data());
    let causal = model.forward(padded_input(), &options).unwrap();
    // The first token reads later keys only without the causal exclusion.
    assert!(gap(plain.logits.narrow(1, 0, 1), causal.logits.narrow(1, 0, 1)) > 1e-4);
    // In one block, a row's last token already sees every key in both modes.
    let mut single = model;
    single.layers.truncate(1);
    let bidirectional = single
        .forward_bidirectional(padded_input(), &options)
        .unwrap();
    let causal = single.forward(padded_input(), &options).unwrap();
    let last = |logits: Tensor<B, 3>| logits.narrow(0, 0, 1).narrow(1, 2, 1);
    assert!(gap(last(bidirectional.logits), last(causal.logits)) < 1e-5);
}

/// Attention-probability requests fail like GPT-2's: a backend capability pointing at pattern hooks.
///
/// GPT-NeoX's attention kernel never materializes probabilities, so the request
/// is rejected before any work in every pass direction, with the same error
/// variant GPT-2 uses for the same request.
#[test]
fn attention_probability_requests_point_to_pattern_hooks() {
    let model = two_layer(true);
    let options = ForwardOptions {
        attentions: true,
        ..ForwardOptions::default()
    };
    for error in [
        model.forward(padded_input(), &options).unwrap_err(),
        model
            .forward_bidirectional(padded_input(), &options)
            .unwrap_err(),
    ] {
        match error {
            Error::BackendCapability(message) => assert_eq!(
                message,
                "GPT-NeoX attention probabilities are available through \
                 self_attention().pattern() hooks"
            ),
            other => panic!("expected Error::BackendCapability, found {other:?}"),
        }
    }
}

/// Router-logit and auxiliary-loss requests on the dense model succeed with empty diagnostics.
///
/// Every GPT-NeoX block is dense, so there is nothing to route: the request
/// leaves `routers` empty and `load_balancing_loss` absent, and the logits equal
/// those of a default pass, as for every other dense family.
#[test]
fn router_requests_return_empty_dense_diagnostics() {
    let model = two_layer(true);
    let plain = model
        .forward(padded_input(), &ForwardOptions::default())
        .unwrap();
    for options in [
        ForwardOptions {
            router_logits: true,
            ..ForwardOptions::default()
        },
        ForwardOptions {
            auxiliary_losses: true,
            ..ForwardOptions::default()
        },
        ForwardOptions {
            router_logits: true,
            auxiliary_losses: true,
            ..ForwardOptions::default()
        },
    ] {
        let output = model.forward(padded_input(), &options).unwrap();
        assert!(output.routers.is_empty());
        assert!(output.load_balancing_loss.is_none());
        assert!(output.attentions.is_empty());
        assert_eq!(output.logits.to_data(), plain.logits.to_data());
    }
}

/// One family-generic router request succeeds on every dense family with empty diagnostics.
///
/// A [`HookedModel`] experiment written once must not fail on one dense family
/// and succeed on another: GPT-2, GPT-NeoX, Llama, Gemma, Qwen3.5 (one
/// full-attention block), and a modular dense decoder all accept
/// `router_logits` and `auxiliary_losses`, return no router entries, and no
/// load-balancing loss.
#[test]
fn dense_families_answer_router_requests_identically() {
    use crate::{
        architecture::{
            Activation, AttentionSpec, DecoderLayerSpec, DecoderSpec, DenseFeedForwardSpec,
            FeedForwardSpec, HookedModel, MixerSpec, NormPlacement, NormalizationSpec,
            PositionEncodingSpec, ResidualTopology,
        },
        models::{
            custom::ModularCausalLm,
            gemma::{GemmaConfig, GemmaForCausalLm, GemmaVariant},
            gpt2::{Gpt2Config, Gpt2ForCausalLm},
            llama::{LlamaFamilyConfig, LlamaForCausalLm, LlamaVariant},
            qwen3_5::{Qwen3_5ForCausalLm, Qwen3_5LayerType, Qwen3_5TextConfig},
        },
    };

    /// Runs the shared router request and checks the dense-model answer.
    fn assert_empty_routers<M>(family: &str, model: &M)
    where
        M: HookedModel<B, Input = CausalLmInput<B>, Output = CausalLmOutput<B>>,
    {
        let options = ForwardOptions {
            router_logits: true,
            auxiliary_losses: true,
            ..ForwardOptions::default()
        };
        let input = CausalLmInput {
            token_ids: Tensor::<B, 2, Int>::from_ints([[1, 4, 2]], &Default::default()),
            token_mask: None,
            position_ids: None,
        };
        let output = model
            .forward_with_hooks(input, &options, &mut Hooks::new())
            .unwrap_or_else(|error| panic!("{family} rejected a dense router request: {error}"));
        assert!(output.routers.is_empty(), "{family} invented routers");
        assert!(
            output.load_balancing_loss.is_none(),
            "{family} invented a load-balancing loss"
        );
    }

    let device = Default::default();
    assert_empty_routers("GPT-NeoX", &two_layer(false));
    let gpt2 = Gpt2ForCausalLm::<B>::init(
        Gpt2Config {
            vocab_size: 13,
            n_embd: 8,
            n_layer: 1,
            n_head: 2,
            n_positions: 16,
            ..Gpt2Config::default()
        },
        &device,
    )
    .unwrap();
    assert_empty_routers("GPT-2", &gpt2);
    let llama = LlamaForCausalLm::<B>::init(
        LlamaFamilyConfig {
            vocab_size: 13,
            hidden_size: 8,
            intermediate_size: 12,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 4,
            max_position_embeddings: 16,
            pad_token_id: None,
            bos_token_id: None,
            eos_token_id: None,
            ..LlamaFamilyConfig::new(LlamaVariant::Llama)
        },
        &device,
    )
    .unwrap();
    assert_empty_routers("Llama", &llama);
    let gemma = GemmaForCausalLm::<B>::init(
        GemmaConfig {
            vocab_size: 13,
            hidden_size: 8,
            intermediate_size: 12,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 4,
            max_position_embeddings: 16,
            query_pre_attn_scalar: 4.0,
            sliding_window: 2,
            pad_token_id: None,
            bos_token_id: None,
            eos_token_id: None,
            ..GemmaConfig::new(GemmaVariant::Gemma2)
        },
        &device,
    )
    .unwrap();
    assert_empty_routers("Gemma", &gemma);
    let qwen = Qwen3_5ForCausalLm::<B>::init(
        Qwen3_5TextConfig {
            vocab_size: 13,
            hidden_size: 8,
            intermediate_size: 12,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 4,
            max_position_embeddings: 16,
            layer_types: vec![Qwen3_5LayerType::FullAttention],
            linear_key_head_dim: 4,
            linear_value_head_dim: 4,
            linear_num_key_heads: 1,
            linear_num_value_heads: 2,
            linear_conv_kernel_dim: 3,
            partial_rotary_factor: Some(1.0),
            rope_theta: Some(10_000.0),
            mtp_num_hidden_layers: 0,
            tie_word_embeddings: false,
            bos_token_id: None,
            eos_token_id: None,
            pad_token_id: None,
            ..Qwen3_5TextConfig::default()
        },
        &device,
    )
    .unwrap();
    assert_empty_routers("Qwen3.5", &qwen);
    let rms = NormalizationSpec::Rms {
        epsilon: 1e-6,
        zero_centered: false,
    };
    let modular = ModularCausalLm::<B>::init(
        DecoderSpec {
            vocabulary_size: 13,
            hidden_size: 8,
            layers: vec![DecoderLayerSpec {
                mixer: MixerSpec::Attention(AttentionSpec {
                    query_heads: 2,
                    key_value_heads: 1,
                    head_dim: 4,
                    query_key_norm: false,
                    output_gate: false,
                    position: PositionEncodingSpec::None,
                }),
                feed_forward: FeedForwardSpec::Dense(DenseFeedForwardSpec {
                    intermediate_size: 12,
                    activation: Activation::Silu,
                    bias: false,
                }),
                mixer_norm: rms,
                feed_forward_norm: rms,
                norm_placement: NormPlacement::Pre,
                residual: ResidualTopology::Add,
            }],
            final_norm: rms,
            tie_word_embeddings: false,
            max_position_embeddings: 16,
        },
        &device,
    )
    .unwrap();
    assert_empty_routers("modular", &modular);
}

/// Inputs longer than the position table or with no rows fail before embedding lookup.
#[test]
fn input_extent_errors_name_the_position_limit() {
    let model = two_layer(true);
    let device = Default::default();
    for dims in [[1, 17], [0, 3]] {
        let input = CausalLmInput {
            token_ids: Tensor::<B, 2, Int>::zeros(dims, &device),
            token_mask: None,
            position_ids: None,
        };
        assert_eq!(
            shape_parts(
                model
                    .forward(input, &ForwardOptions::default())
                    .unwrap_err()
            ),
            (
                "gpt_neox.token_ids".into(),
                "nonempty [batch, sequence<=16]".into(),
                format!("[{}, {}]", dims[0], dims[1]),
            )
        );
    }
}

/// Lens projections reject empty or mis-sized states, foreign dtypes, and an edited final norm.
#[test]
fn lens_projection_validates_states_and_the_final_norm() {
    let mut model = two_layer(true);
    let device = Default::default();
    for dims in [[0, 1, 8], [1, 0, 8], [1, 2, 7]] {
        assert_eq!(
            shape_parts(
                model
                    .project_hidden(Tensor::<B, 3>::zeros(dims, &device))
                    .unwrap_err()
            ),
            (
                "gpt_neox.hidden".into(),
                "nonempty [batch, token, 8]".into(),
                format!("{dims:?}"),
            )
        );
    }
    let half = Tensor::<B, 3>::zeros([1, 2, 8], &device).cast(DType::F16);
    assert_eq!(
        configuration_parts(model.normalize_hidden(half).unwrap_err()),
        (
            "gpt_neox.hidden".into(),
            "dtype and device must match the model parameters".into()
        )
    );
    let projected = model
        .project_hidden(Tensor::<B, 3>::ones([1, 2, 8], &device))
        .unwrap();
    assert_eq!(projected.dims(), [1, 2, 13]);
    model.final_layer_norm.beta = Some(Param::from_tensor(Tensor::zeros([7], &device)));
    assert_eq!(
        shape_parts(
            model
                .normalize_hidden(Tensor::<B, 3>::ones([1, 2, 8], &device))
                .unwrap_err()
        ),
        (
            "gpt_neox.final_layer_norm".into(),
            "[8]".into(),
            "[7]".into()
        )
    );
}

/// Every edited norm and projection is checked against the residual width, by label.
#[test]
fn edited_parameter_shapes_are_reported_by_label() {
    let device = Default::default();
    let wrong_linear =
        |input: usize, output: usize| LinearConfig::new(input, output).init::<B>(&device);
    /// Installs a replacement projection at one parameter slot of the model.
    type Edit = fn(&mut GptNeoXForCausalLm<B>, Linear<B>);
    let cases: [(&str, usize, usize, Edit, &str); 6] = [
        (
            "attention.q_proj",
            8,
            7,
            |m, l| m.layers[1].attention.q_proj = l,
            "[8, 8]",
        ),
        (
            "attention.k_proj",
            8,
            7,
            |m, l| m.layers[1].attention.k_proj = l,
            "[8, 8]",
        ),
        (
            "attention.v_proj",
            8,
            7,
            |m, l| m.layers[1].attention.v_proj = l,
            "[8, 8]",
        ),
        (
            "attention.o_proj",
            7,
            8,
            |m, l| m.layers[1].attention.o_proj = l,
            "[8, 8]",
        ),
        (
            "mlp.up_proj",
            7,
            12,
            |m, l| m.layers[1].mlp.up_proj = l,
            "[8, 12]",
        ),
        (
            "mlp.down_proj",
            12,
            7,
            |m, l| m.layers[1].mlp.down_proj = l,
            "[12, 8]",
        ),
    ];
    for (label, input, output, edit, expected) in cases {
        let mut model = two_layer(true);
        edit(&mut model, wrong_linear(input, output));
        let (tensor, shape, _) = shape_parts(model.validate_parameters().unwrap_err());
        assert_eq!(tensor, format!("gpt_neox.layers.1.{label}"));
        assert_eq!(shape, expected);
    }
    let mut model = two_layer(true);
    model.embed_out = wrong_linear(8, 12);
    let (tensor, shape, actual) = shape_parts(model.validate_parameters().unwrap_err());
    assert_eq!(
        (tensor.as_str(), shape.as_str(), actual.as_str()),
        ("gpt_neox.embed_out", "[8, 13]", "[8, 12]")
    );
    for norm in ["input_layernorm", "post_attention_layernorm"] {
        let mut model = two_layer(true);
        let replacement = LayerNormConfig::new(7).init(&device);
        if norm == "input_layernorm" {
            model.layers[1].input_layernorm = replacement;
        } else {
            model.layers[1].post_attention_layernorm = replacement;
        }
        let (tensor, shape, actual) = shape_parts(model.validate_parameters().unwrap_err());
        assert_eq!(
            (tensor.as_str(), shape.as_str(), actual.as_str()),
            (format!("gpt_neox.layers.1.{norm}").as_str(), "[8]", "[7]")
        );
    }
}

/// Norm betas may be absent; query/key norms, MLP width, and dtype agreement are enforced.
#[test]
fn edited_norm_widths_and_dtypes_are_enforced() {
    use crate::normalization::RmsNormConfig;
    let device = Default::default();
    let mut model = two_layer(true);
    model.layers[0].input_layernorm.beta = None;
    model.final_layer_norm.beta = None;
    model
        .validate_parameters()
        .expect("affine norms without a bias remain valid");
    model
        .forward(padded_input(), &ForwardOptions::default())
        .expect("a bias-free norm still executes");

    let mut model = two_layer(true);
    model.layers[0].attention.key_norm = Some(RmsNormConfig::new(3).init(&device).unwrap());
    assert_eq!(
        shape_parts(model.validate_parameters().unwrap_err()),
        (
            "gpt_neox.layers.0.attention.query_key_norm".into(),
            "[4]".into(),
            "[3]".into()
        )
    );

    let mut model = two_layer(true);
    model.layers[0].mlp.up_proj = LinearConfig::new(8, 0).init(&device);
    assert_eq!(
        configuration_parts(model.validate_parameters().unwrap_err()),
        (
            "gpt_neox.layers.0.mlp".into(),
            "intermediate width must be nonzero".into()
        )
    );

    let mut model = two_layer(true);
    model.layers[1].attention.o_proj.weight = model.layers[1]
        .attention
        .o_proj
        .weight
        .clone()
        .map(|weight| weight.cast(DType::F16));
    assert_eq!(
        configuration_parts(model.validate_parameters().unwrap_err()),
        (
            "gpt_neox.layers.1.attention.o_proj".into(),
            "parameters must share device and dtype".into()
        )
    );
}

/// A replacement head width whose rotary prefix is odd fails while building RoPE tables.
#[test]
fn replacement_head_width_with_an_odd_rotary_prefix_is_rejected() {
    let mut model = two_layer(true);
    // hidden 8 = 4 heads of width 2; the configured rotary fraction 0.5 rotates one channel.
    let mut attention = crate::attention::GroupedQueryAttentionConfig::new(8, 4, 4, 2)
        .init(&Default::default())
        .unwrap();
    attention.precision_mode = AttentionPrecisionMode::EagerFp32Softmax;
    model.layers[1].attention = attention;
    model
        .validate_parameters()
        .expect("the geometry itself is consistent");
    let (field, _) = configuration_parts(
        model
            .forward(padded_input(), &ForwardOptions::default())
            .unwrap_err(),
    );
    assert_eq!(field, "gpt_neox.layers.1.rotary");
}

/// Times eager and backend-default causal forwards (release builds only).
///
/// A Pythia-shaped model (hidden 256, 8 heads, 6 layers, quarter-head RoPE)
/// on a batch of 8 sequences of 256 tokens. The eager timing also runs on
/// earlier sources, so it compares implementations. Run with
/// `cargo test --release --lib neox_forward_benchmark -- --ignored --nocapture`.
#[test]
#[ignore = "timing benchmark; run explicitly in release mode"]
fn neox_forward_benchmark() {
    let device = Default::default();
    let mut model = GptNeoXForCausalLm::<B>::init(
        GptNeoXConfig {
            vocab_size: 512,
            hidden_size: 256,
            num_hidden_layers: 6,
            num_attention_heads: 8,
            intermediate_size: 1024,
            max_position_embeddings: 512,
            ..Default::default()
        },
        &device,
    )
    .unwrap();
    let (batch, sequence) = (8_usize, 256_usize);
    let ids: Vec<i32> = (0..batch * sequence)
        .map(|index| (index * 7 % 500) as i32)
        .collect();
    let input = || CausalLmInput {
        token_ids: Tensor::<B, 2, Int>::from_data(
            burn::tensor::TensorData::new(ids.clone(), [batch, sequence]),
            &device,
        ),
        token_mask: None,
        position_ids: None,
    };
    for mode in [
        AttentionPrecisionMode::EagerFp32Softmax,
        AttentionPrecisionMode::BackendDefault,
    ] {
        for layer in &mut model.layers {
            layer.attention.precision_mode = mode;
        }
        let mut best = f64::INFINITY;
        for _ in 0..5 {
            let started = std::time::Instant::now();
            model
                .forward(input(), &ForwardOptions::default())
                .unwrap()
                .logits
                .into_data();
            best = best.min(started.elapsed().as_secs_f64());
        }
        eprintln!("GPT-NeoX forward {mode:?} batch={batch} sequence={sequence}: best={best:.4}s");
    }
}
