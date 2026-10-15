//! GPT-NeoX parameter validation, residual wiring, precision modes, and initialization.

use burn::{backend::Flex, tensor::Bool};

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

/// The former name is an alias of the renamed causal language model.
#[test]
fn former_model_name_aliases_the_causal_language_model() {
    let model: GptNeoXModel<B> = model();
    let renamed: &GptNeoXForCausalLm<B> = &model;
    assert_eq!(renamed.layers.len(), 1);
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
