use super::*;

#[test]
fn derives_the_documented_hybrid_schedule() {
    let config = Qwen3_5TextConfig {
        num_hidden_layers: 8,
        layer_types: Vec::new(),
        ..Qwen3_5TextConfig::default()
    };
    assert_eq!(
        config.resolved_layer_types(),
        vec![
            Qwen3_5LayerType::LinearAttention,
            Qwen3_5LayerType::LinearAttention,
            Qwen3_5LayerType::LinearAttention,
            Qwen3_5LayerType::FullAttention,
            Qwen3_5LayerType::LinearAttention,
            Qwen3_5LayerType::LinearAttention,
            Qwen3_5LayerType::LinearAttention,
            Qwen3_5LayerType::FullAttention,
        ]
    );
}

#[test]
fn explicit_schedule_overrides_the_derived_interval() {
    let config = Qwen3_5TextConfig {
        num_hidden_layers: 4,
        layer_types: vec![Qwen3_5LayerType::FullAttention; 4],
        ..Qwen3_5TextConfig::default()
    };
    config.validate().unwrap();
}

#[test]
fn rejects_an_explicit_schedule_with_the_wrong_length() {
    let config = Qwen3_5TextConfig {
        num_hidden_layers: 4,
        layer_types: vec![Qwen3_5LayerType::FullAttention; 3],
        ..Qwen3_5TextConfig::default()
    };
    assert!(config.validate().is_err());
}

#[test]
fn gated_query_projection_has_twice_the_attention_width() {
    let config = Qwen3_5TextConfig {
        num_attention_heads: 24,
        head_dim: 256,
        attn_output_gate: true,
        ..Qwen3_5TextConfig::default()
    };
    assert_eq!(config.query_projection_width(), 12_288);
}

/// Serialized aliases for the implemented SiLU DeltaNet gate remain compatible.
#[test]
fn accepts_only_implemented_delta_gate_spellings() {
    for gate in [None, Some("silu"), Some("swish")] {
        let config = Qwen3_5TextConfig {
            output_gate_type: gate.map(str::to_owned),
            ..Qwen3_5TextConfig::default()
        };
        config.validate().expect("SiLU spelling must validate");
    }

    let config = Qwen3_5TextConfig {
        output_gate_type: Some("sigmoid".into()),
        ..Qwen3_5TextConfig::default()
    };
    let error = config
        .validate()
        .expect_err("an unimplemented gate must fail closed");
    assert_eq!(error.field, "output_gate_type");
}

/// Runtime-affecting legacy fields cannot be accepted and then ignored.
#[test]
fn rejects_unimplemented_recurrent_and_layer_topologies() {
    let cases = [
        Qwen3_5TextConfig {
            mamba_ssm_dtype: Some("float16".into()),
            ..Qwen3_5TextConfig::default()
        },
        Qwen3_5TextConfig {
            mlp_only_layers: vec![0],
            ..Qwen3_5TextConfig::default()
        },
        Qwen3_5TextConfig {
            mtp_use_dedicated_embeddings: true,
            ..Qwen3_5TextConfig::default()
        },
    ];
    let expected_fields = [
        "mamba_ssm_dtype",
        "mlp_only_layers",
        "mtp_use_dedicated_embeddings",
    ];

    for (config, expected_field) in cases.into_iter().zip(expected_fields) {
        let error = config
            .validate()
            .expect_err("unimplemented topology must fail closed");
        assert_eq!(error.field, expected_field);
    }
}
