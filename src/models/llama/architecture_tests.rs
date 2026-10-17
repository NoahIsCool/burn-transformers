//! The translation table: per-variant flags, layer schedules, RoPE, and every
//! contextual rejection, plus property tests of the schedule rules.

use proptest::prelude::*;

use super::super::plan::rotary_by_interval;
use super::*;
use crate::{models::config::TokenIds, position::RopeScaling};

/// A small valid configuration of `variant` with no generation tokens.
fn small(variant: LlamaVariant) -> LlamaFamilyConfig {
    LlamaFamilyConfig {
        vocab_size: 64,
        hidden_size: 32,
        intermediate_size: 48,
        num_hidden_layers: 4,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dim: 8,
        max_position_embeddings: 128,
        pad_token_id: None,
        bos_token_id: None,
        eos_token_id: None,
        ..LlamaFamilyConfig::new(variant)
    }
}

/// Returns the error field of an invalid configuration.
fn rejected(config: &LlamaFamilyConfig) -> &'static str {
    config
        .architecture()
        .expect_err("configuration must be rejected")
        .field
}

/// Each variant's flags follow the module's translation table.
#[test]
fn flags_follow_the_translation_table() {
    let llama = small(LlamaVariant::Llama).architecture().unwrap();
    assert_eq!(llama.projection_bias, ProjectionBias::uniform(false));
    assert_eq!(llama.query_key_norm, None);
    assert_eq!(llama.sublayer_norm, SublayerNorm::Input);
    assert_eq!(llama.norm_rounding, NormRounding::CastThenScale);
    assert_eq!(llama.rotary_precision, RotaryPrecision::ActivationDtype);

    let mistral = small(LlamaVariant::Mistral).architecture().unwrap();
    assert_eq!(mistral.projection_bias, ProjectionBias::uniform(false));
    assert_eq!(mistral.sliding_window(), Some(4096));

    let qwen2 = small(LlamaVariant::Qwen2).architecture().unwrap();
    assert_eq!(
        qwen2.projection_bias,
        ProjectionBias::query_key_value_only()
    );
    assert_eq!(
        qwen2.sliding_window(),
        None,
        "use_sliding_window gates the window"
    );

    let qwen3 = small(LlamaVariant::Qwen3).architecture().unwrap();
    assert_eq!(qwen3.query_key_norm, Some(QueryKeyNormScope::PerHead));
    assert!(!qwen3.mlp_bias);

    let olmo2 = small(LlamaVariant::Olmo2).architecture().unwrap();
    assert_eq!(olmo2.query_key_norm, Some(QueryKeyNormScope::FullWidth));
    assert_eq!(olmo2.sublayer_norm, SublayerNorm::Output);
    assert_eq!(olmo2.norm_rounding, NormRounding::ScaleThenCast);
    assert_eq!(olmo2.rotary_precision, RotaryPrecision::Float32);
    assert!(!olmo2.norm_rounding.round_before_scale());
    assert_eq!(
        olmo2.rotary_precision.rotation_dtype(DType::F16),
        DType::F32
    );
    assert_eq!(
        llama.rotary_precision.rotation_dtype(DType::F16),
        DType::F16
    );

    let biased = LlamaFamilyConfig {
        attention_bias: Some(true),
        mlp_bias: Some(true),
        ..small(LlamaVariant::SmolLm3)
    }
    .architecture()
    .unwrap();
    assert_eq!(biased.projection_bias, ProjectionBias::uniform(true));
    assert!(biased.mlp_bias);
    assert_eq!(biased.num_layers(), 4);
    assert_eq!(
        biased.attention_config().effective_projection_bias(),
        ProjectionBias::uniform(true)
    );
    assert!(biased.mlp_config().bias);
}

/// References that hard-code a bias reject a contradicting serialized flag.
#[test]
fn contradicting_bias_flags_are_rejected() {
    let mistral = LlamaFamilyConfig {
        attention_bias: Some(true),
        ..small(LlamaVariant::Mistral)
    };
    assert_eq!(rejected(&mistral), "attention_bias");
    let qwen2 = LlamaFamilyConfig {
        attention_bias: Some(false),
        ..small(LlamaVariant::Qwen2)
    };
    assert_eq!(rejected(&qwen2), "attention_bias");
    for variant in [
        LlamaVariant::Mistral,
        LlamaVariant::Qwen2,
        LlamaVariant::Qwen3,
        LlamaVariant::Olmo2,
    ] {
        let config = LlamaFamilyConfig {
            mlp_bias: Some(true),
            ..small(variant)
        };
        assert_eq!(rejected(&config), "mlp_bias", "{variant:?}");
    }
}

/// Mistral slides every layer unless its window is null.
#[test]
fn mistral_windows_apply_to_every_layer() {
    let sliding = LlamaFamilyConfig {
        sliding_window: Some(16),
        ..small(LlamaVariant::Mistral)
    }
    .architecture()
    .unwrap();
    assert!(
        sliding
            .layers
            .iter()
            .all(|layer| layer.sliding_window() == Some(16))
    );
    let global = LlamaFamilyConfig {
        sliding_window: None,
        ..small(LlamaVariant::Mistral)
    }
    .architecture()
    .unwrap();
    assert!(
        global
            .layers
            .iter()
            .all(|layer| layer.sliding_window().is_none())
    );
    let ministral = LlamaFamilyConfig {
        layer_types: Some(vec!["full_attention".into(); 4]),
        ..small(LlamaVariant::Mistral)
    };
    assert_eq!(rejected(&ministral), "layer_types");
}

/// Qwen slides layers from `max_window_layers` when enabled, or follows `layer_types`.
#[test]
fn qwen_schedules_follow_max_window_layers_and_layer_types() {
    let derived = LlamaFamilyConfig {
        use_sliding_window: true,
        sliding_window: Some(8),
        max_window_layers: Some(2),
        ..small(LlamaVariant::Qwen3)
    }
    .architecture()
    .unwrap();
    let windows: Vec<_> = derived
        .layers
        .iter()
        .map(|layer| layer.sliding_window())
        .collect();
    assert_eq!(windows, [None, None, Some(8), Some(8)]);

    let explicit = LlamaFamilyConfig {
        use_sliding_window: true,
        sliding_window: Some(8),
        layer_types: Some(
            [
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention",
            ]
            .map(String::from)
            .to_vec(),
        ),
        ..small(LlamaVariant::Qwen2)
    }
    .architecture()
    .unwrap();
    let windows: Vec<_> = explicit
        .layers
        .iter()
        .map(|layer| layer.sliding_window())
        .collect();
    assert_eq!(windows, [Some(8), None, Some(8), None]);

    let windowless = LlamaFamilyConfig {
        layer_types: Some(vec!["sliding_attention".into(); 4]),
        ..small(LlamaVariant::Qwen2)
    };
    assert_eq!(rejected(&windowless), "layer_types");
    let short = LlamaFamilyConfig {
        layer_types: Some(vec!["full_attention".into(); 3]),
        ..small(LlamaVariant::Qwen2)
    };
    assert_eq!(rejected(&short), "layer_types");
    let unknown = LlamaFamilyConfig {
        layer_types: Some(vec!["chunked_attention".into(); 4]),
        ..small(LlamaVariant::Qwen2)
    };
    assert_eq!(rejected(&unknown), "layer_types");
    let zero = LlamaFamilyConfig {
        use_sliding_window: true,
        sliding_window: Some(0),
        ..small(LlamaVariant::Qwen2)
    };
    assert_eq!(rejected(&zero), "sliding_window");
}

/// Llama and OLMo 2 accept only all-full `layer_types`.
#[test]
fn global_references_reject_sliding_layer_types() {
    for variant in [LlamaVariant::Llama, LlamaVariant::Olmo2] {
        let full = LlamaFamilyConfig {
            layer_types: Some(vec!["full_attention".into(); 4]),
            ..small(variant)
        };
        assert!(full.architecture().is_ok(), "{variant:?}");
        let sliding = LlamaFamilyConfig {
            layer_types: Some(vec!["sliding_attention".into(); 4]),
            sliding_window: Some(8),
            ..small(variant)
        };
        assert_eq!(rejected(&sliding), "layer_types", "{variant:?}");
    }
}

/// SmolLM3 NoPE layers follow `no_rope_layers` or the interval; NoPE layers may slide.
#[test]
fn smollm3_nope_and_sliding_schedules() {
    let interval = small(LlamaVariant::SmolLm3).architecture().unwrap();
    let rotary: Vec<bool> = interval.layers.iter().map(|layer| layer.rotary).collect();
    assert_eq!(rotary, [true, true, true, false]);

    let explicit = LlamaFamilyConfig {
        no_rope_layers: Some(vec![1, 0, 1, 0]),
        use_sliding_window: true,
        sliding_window: Some(8),
        ..small(LlamaVariant::SmolLm3)
    }
    .architecture()
    .unwrap();
    let schedule: Vec<(bool, Option<usize>)> = explicit
        .layers
        .iter()
        .map(|layer| (layer.rotary, layer.sliding_window()))
        .collect();
    assert_eq!(
        schedule,
        [
            (true, None),
            (false, Some(8)),
            (true, None),
            (false, Some(8))
        ]
    );

    for flags in [vec![1, 0, 1], vec![1, 2, 1, 0]] {
        let invalid = LlamaFamilyConfig {
            no_rope_layers: Some(flags),
            ..small(LlamaVariant::SmolLm3)
        };
        assert_eq!(rejected(&invalid), "no_rope_layers");
    }
    let zero = LlamaFamilyConfig {
        no_rope_layer_interval: Some(0),
        ..small(LlamaVariant::SmolLm3)
    };
    assert_eq!(rejected(&zero), "no_rope_layer_interval");
    let elsewhere = LlamaFamilyConfig {
        no_rope_layers: Some(vec![1, 0, 1, 1]),
        ..small(LlamaVariant::Llama)
    };
    assert_eq!(rejected(&elsewhere), "no_rope_layers");
    let all_rotary = LlamaFamilyConfig {
        no_rope_layers: Some(vec![1; 4]),
        ..small(LlamaVariant::Qwen3)
    };
    assert!(all_rotary.architecture().is_ok());
}

/// RoPE fields go through the shared translator with the variant's default theta.
#[test]
fn rope_is_translated_by_the_shared_translator() {
    let smollm3 = small(LlamaVariant::SmolLm3).architecture().unwrap();
    assert_eq!(smollm3.rope.theta(), 2_000_000.0);
    assert_eq!(smollm3.rope.rotary_dimension(), 8);
    let legacy = LlamaFamilyConfig {
        rope_theta: Some(500_000.0),
        ..small(LlamaVariant::Llama)
    }
    .architecture()
    .unwrap();
    assert_eq!(legacy.rope.theta(), 500_000.0);
    let llama3 = LlamaFamilyConfig {
        rope_parameters: Some(
            serde_json::from_value(serde_json::json!({
                "rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
                "high_freq_factor": 4.0, "original_max_position_embeddings": 32,
            }))
            .unwrap(),
        ),
        ..small(LlamaVariant::Llama)
    }
    .architecture()
    .unwrap();
    assert!(matches!(llama3.rope.scaling(), RopeScaling::Llama3(_)));
    let partial = LlamaFamilyConfig {
        partial_rotary_factor: Some(0.5),
        ..small(LlamaVariant::Llama)
    };
    assert_eq!(rejected(&partial), "rope_parameters");
    let missing_factor = LlamaFamilyConfig {
        rope_parameters: Some(
            serde_json::from_value(serde_json::json!({ "rope_type": "linear" })).unwrap(),
        ),
        ..small(LlamaVariant::Llama)
    };
    assert_eq!(rejected(&missing_factor), "rope_parameters");
}

/// Geometry, dropout, activation, and token errors name their field.
#[test]
fn invalid_geometry_and_options_are_rejected() {
    let cases: [(LlamaFamilyConfig, &str); 9] = [
        (
            LlamaFamilyConfig {
                vocab_size: 0,
                ..small(LlamaVariant::Llama)
            },
            "vocab_size",
        ),
        (
            LlamaFamilyConfig {
                head_dim: 0,
                ..small(LlamaVariant::Llama)
            },
            "head_dim",
        ),
        (
            LlamaFamilyConfig {
                num_hidden_layers: 0,
                ..small(LlamaVariant::Llama)
            },
            "num_hidden_layers",
        ),
        (
            LlamaFamilyConfig {
                num_key_value_heads: 3,
                ..small(LlamaVariant::Llama)
            },
            "num_attention_heads",
        ),
        (
            LlamaFamilyConfig {
                rms_norm_eps: 0.0,
                ..small(LlamaVariant::Llama)
            },
            "rms_norm_eps",
        ),
        (
            LlamaFamilyConfig {
                attention_dropout: 0.1,
                ..small(LlamaVariant::Llama)
            },
            "attention_dropout",
        ),
        (
            LlamaFamilyConfig {
                hidden_act: "tanh".into(),
                ..small(LlamaVariant::Llama)
            },
            "hidden_act",
        ),
        (
            LlamaFamilyConfig {
                pad_token_id: Some(TokenIds::One(64)),
                ..small(LlamaVariant::Llama)
            },
            "pad_token_id",
        ),
        (
            LlamaFamilyConfig {
                model_type: "gpt2".into(),
                ..small(LlamaVariant::Llama)
            },
            "model_type",
        ),
    ];
    for (config, field) in cases {
        assert_eq!(rejected(&config), field);
    }
    for (name, expected) in [
        ("swish", Activation::Silu),
        ("gelu", Activation::Gelu),
        ("gelu_new", Activation::GeluTanh),
        ("relu", Activation::Relu),
    ] {
        let config = LlamaFamilyConfig {
            hidden_act: name.into(),
            ..small(LlamaVariant::Llama)
        };
        assert_eq!(config.architecture().unwrap().activation, expected);
    }
}

/// The family RMSNorm multiplies by the reciprocal root and rounds per variant.
#[test]
fn rms_norm_conventions_follow_the_variant() {
    let llama = small(LlamaVariant::Llama)
        .architecture()
        .unwrap()
        .rms_norm(32);
    assert!(llama.round_before_scale);
    assert_eq!(llama.root_mode, RmsNormRootMode::MultiplyByReciprocalSqrt);
    assert_eq!(llama.epsilon, 1e-6);
    let olmo = small(LlamaVariant::Olmo2)
        .architecture()
        .unwrap()
        .rms_norm(32);
    assert!(!olmo.round_before_scale);
    assert_eq!(olmo.epsilon, 1e-5);
}

proptest! {
    /// The NoPE rule equals Transformers' `int((l + 1) % k != 0)`.
    #[test]
    fn nope_interval_matches_the_reference_formula(layer in 0usize..10_000, interval in 1usize..64) {
        prop_assert_eq!(rotary_by_interval(layer, interval), (layer + 1) % interval != 0);
    }

    /// Derived Qwen schedules slide exactly the layers at or past `max_window_layers`.
    #[test]
    fn qwen_derived_schedules_match_the_reference(
        layers in 1usize..12,
        first in 0usize..14,
        enabled in any::<bool>(),
    ) {
        let config = LlamaFamilyConfig {
            num_hidden_layers: layers,
            use_sliding_window: enabled,
            sliding_window: Some(4),
            max_window_layers: Some(first),
            ..small(LlamaVariant::Qwen2)
        };
        let architecture = config.architecture().unwrap();
        for (layer, schedule) in architecture.layers.iter().enumerate() {
            prop_assert_eq!(schedule.sliding_window().is_some(), enabled && layer >= first);
        }
    }
}

/// Dynamic NTK is inert in the Llama family: it equals unscaled RoPE wherever positions are accepted.
///
/// Review finding NUM-4. Transformers rescales the dynamic-NTK base only for
/// a pass longer than `max_position_embeddings`, and every built-in family
/// refuses positions at or beyond that limit, so a translated
/// `rope_type: dynamic` never changes a frequency here (the module
/// documentation says so). This pins that behavior with the Transformers
/// golden of the default-RoPE `llama-bias-tied` fixture: its 12-token pass is
/// unscaled under dynamic NTK in Transformers too, so the same weights under
/// `{dynamic, factor 3}` must reproduce it, while a pass that would reach the
/// scaling branch is refused instead of silently computed without it.
#[test]
fn dynamic_ntk_is_inert_within_the_accepted_positions() {
    use crate::{
        CausalLmInput, ForwardOptions, LogitsSelection,
        architecture::TokenMask,
        models::{AutoCausalLm, AutoLoadOptions, config::ModelConfig},
    };
    use burn::{
        backend::Flex,
        tensor::{Int, Tensor, TensorData},
    };

    /// CPU backend with F32 values and I32 indices.
    type B = Flex<f32, i32>;
    let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/tiny_goldens/llama-bias-tied");
    let mut config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(fixture.join("config.json")).unwrap())
            .unwrap();
    config["rope_parameters"] = serde_json::json!({
        "rope_type": "dynamic", "factor": 3.0, "rope_theta": 10000.0
    });
    let parsed = LlamaFamilyConfig::from_json_str(&config.to_string()).unwrap();
    assert!(matches!(
        parsed.architecture().unwrap().rope.scaling(),
        RopeScaling::DynamicNtk(_)
    ));
    let root = tempfile::tempdir().unwrap();
    std::fs::write(root.path().join("config.json"), config.to_string()).unwrap();
    std::fs::copy(
        fixture.join("model.safetensors"),
        root.path().join("model.safetensors"),
    )
    .unwrap();
    let model = AutoCausalLm::<B>::load_local(
        root.path(),
        &AutoLoadOptions {
            load_tokenizer: false,
            ..AutoLoadOptions::default()
        },
        &Default::default(),
    )
    .unwrap()
    .into_model();

    let golden: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(fixture.join("golden.json")).unwrap())
            .unwrap();
    let matrix = |key: &str| serde_json::from_value::<Vec<Vec<i64>>>(golden[key].clone()).unwrap();
    let ints = |rows: &[Vec<i64>]| {
        Tensor::<B, 2, Int>::from_data(
            TensorData::new(rows.concat(), [rows.len(), rows[0].len()]),
            &Default::default(),
        )
    };
    let (input_ids, mask) = (matrix("input_ids"), matrix("attention_mask"));
    let logits = model
        .forward(
            CausalLmInput {
                token_ids: ints(&input_ids),
                token_mask: Some(TokenMask::new(ints(&mask).equal_elem(1))),
                position_ids: Some(ints(&matrix("position_ids"))),
            },
            &ForwardOptions {
                logits: LogitsSelection::All,
                ..ForwardOptions::default()
            },
        )
        .unwrap()
        .logits
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    let expected = serde_json::from_value::<Vec<f64>>(golden["logits"].clone()).unwrap();
    let vocabulary = expected.len() / (input_ids.len() * input_ids[0].len());
    // Every position's logits are compared, not a prefix that `zip` would truncate to.
    assert_eq!(logits.len(), expected.len());
    for (index, (actual, expected)) in logits.iter().zip(&expected).enumerate() {
        let position = index / vocabulary;
        if mask[position / input_ids[0].len()][position % input_ids[0].len()] == 1 {
            assert!(
                (actual - *expected as f32).abs() <= 1e-5,
                "logit {index}: {actual} vs {expected}"
            );
        }
    }
    // The first pass long enough for Transformers to rescale is refused.
    let limit = parsed.max_position_embeddings;
    let error = model
        .forward(
            CausalLmInput {
                token_ids: ints(&[vec![3; limit + 1]]),
                token_mask: None,
                position_ids: None,
            },
            &ForwardOptions::default(),
        )
        .map(drop)
        .unwrap_err();
    assert!(error.to_string().contains("position"), "{error}");
}
