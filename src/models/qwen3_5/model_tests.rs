use burn::{
    backend::Flex,
    tensor::{Bool, Int, Tensor, backend::Backend},
};

use super::*;
use crate::models::config::ModelConfig;

type TestBackend = Flex<f32, i32>;

/// Builds a tiny hybrid model that exercises both mixer families.
fn tiny_config() -> Qwen3_5TextConfig {
    Qwen3_5TextConfig {
        vocab_size: 32,
        hidden_size: 8,
        intermediate_size: 16,
        num_hidden_layers: 4,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: 4,
        max_position_embeddings: 64,
        layer_types: vec![
            Qwen3_5LayerType::LinearAttention,
            Qwen3_5LayerType::LinearAttention,
            Qwen3_5LayerType::LinearAttention,
            Qwen3_5LayerType::FullAttention,
        ],
        linear_key_head_dim: 4,
        linear_value_head_dim: 4,
        linear_num_key_heads: 1,
        linear_num_value_heads: 2,
        linear_conv_kernel_dim: 3,
        partial_rotary_factor: Some(1.0),
        rope_theta: Some(10_000.0),
        mtp_num_hidden_layers: 0,
        ..Default::default()
    }
}

/// Tiny stateless inference returns only requested logits and diagnostics.
#[test]
fn tiny_hybrid_forward_respects_output_selection() {
    let device = Default::default();
    TestBackend::seed(&device, 11);
    let model = Qwen3_5ForCausalLm::<TestBackend>::init(tiny_config(), &device)
        .expect("tiny model must initialize");
    let input = CausalLmInput {
        token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[1, 2, 3], [4, 5, 6]], &device),
        token_mask: None,
        position_ids: None,
    };
    let output = model
        .forward(
            input,
            &ForwardOptions {
                logits: LogitsSelection::Positions(vec![0, 2]),
                hidden_states: true,
                ..Default::default()
            },
        )
        .expect("forward must succeed");
    assert_eq!(output.logits.dims(), [2, 2, 32]);
    assert_eq!(output.hidden_states.len(), 6);
    assert_eq!(
        output.final_hidden_state.expect("requested").dims(),
        [2, 3, 8]
    );
}

/// Tied checkpoints project with the embedding table and allocate no second matrix.
#[test]
fn tied_embedding_head_runs_without_duplicate_parameter() {
    let device = Default::default();
    TestBackend::seed(&device, 17);
    let mut config = tiny_config();
    config.tie_word_embeddings = true;
    let model = Qwen3_5ForCausalLm::<TestBackend>::init(config, &device)
        .expect("tied tiny model must initialize");
    assert!(model.lm_head.is_none());

    let output = model
        .forward(
            CausalLmInput {
                token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[1, 2]], &device),
                token_mask: None,
                position_ids: None,
            },
            &ForwardOptions::default(),
        )
        .expect("embedding-backed vocabulary projection must run");
    assert_eq!(output.logits.dims(), [1, 1, 32]);
}

/// A gradient must reach its leaf parameter and contain a finite learning signal.
#[cfg(feature = "autodiff")]
fn assert_finite_nonzero_gradient<const D: usize>(
    gradient: Option<Tensor<TestBackend, D>>,
    parameter: &str,
) {
    let values = gradient
        .unwrap_or_else(|| panic!("{parameter} is disconnected from the stateless loss"))
        .into_data()
        .to_vec::<f32>()
        .expect("Flex gradients are F32");
    assert!(
        values.iter().all(|value| value.is_finite()),
        "{parameter} has a non-finite gradient"
    );
    assert!(
        values.iter().any(|value| *value != 0.0),
        "{parameter} has an entirely zero gradient"
    );
}

/// The complete 3:1 hybrid graph propagates gradients through both mixer families.
///
/// A tied vocabulary head matches the 0.8B checkpoint; the untied case also
/// proves that embedding gradients pass through the backbone rather than
/// only its shared output head. Partial RoPE exercises both rotated and
/// unrotated attention channels. Four causal
/// positions make the DeltaNet decay parameters affect retained history;
/// a single-token example could not prove that recurrent learning signal.
/// Squared logit loss is used only to probe connectivity and finite values,
/// not to prescribe an eventual training objective.
#[cfg(feature = "autodiff")]
#[test]
fn stateless_hybrid_graph_has_finite_nonzero_parameter_gradients() {
    use burn::backend::Autodiff;

    type AdBackend = Autodiff<TestBackend>;
    let device = Default::default();
    for tied in [false, true] {
        AdBackend::seed(&device, 29);
        let mut config = tiny_config();
        config.tie_word_embeddings = tied;
        config.partial_rotary_factor = Some(0.5);
        let model = Qwen3_5ForCausalLm::<AdBackend>::init(config, &device)
            .expect("tiny differentiable hybrid model initializes");
        assert_eq!(model.lm_head.is_none(), tied);
        let output = model
            .forward(
                CausalLmInput {
                    token_ids: Tensor::<AdBackend, 2, Int>::from_ints([[1, 2, 3, 4]], &device),
                    token_mask: None,
                    position_ids: None,
                },
                &ForwardOptions {
                    logits: LogitsSelection::All,
                    ..ForwardOptions::default()
                },
            )
            .expect("stateless hybrid autodiff forward succeeds");
        let gradients = output.logits.square().mean().backward();
        assert_finite_nonzero_gradient(
            model.model.embed_tokens.weight.grad(&gradients),
            "embedding and tied vocabulary head",
        );
        assert_finite_nonzero_gradient(model.model.norm.weight.grad(&gradients), "final norm");
        if let Some(head) = &model.lm_head {
            assert_finite_nonzero_gradient(head.weight.grad(&gradients), "untied vocabulary head");
        }

        for (index, layer) in model.model.layers.iter().enumerate() {
            for (name, projection) in [
                ("ffn.gate", &layer.mlp.gate_proj),
                ("ffn.up", &layer.mlp.up_proj),
                ("ffn.down", &layer.mlp.down_proj),
            ] {
                assert_finite_nonzero_gradient(
                    projection.weight.grad(&gradients),
                    &format!("layer {index} {name}"),
                );
            }
            for (name, norm) in [
                ("input norm", &layer.input_layernorm),
                ("post-attention norm", &layer.post_attention_layernorm),
            ] {
                assert_finite_nonzero_gradient(
                    norm.weight.grad(&gradients),
                    &format!("layer {index} {name}"),
                );
            }
            if let Some(delta) = &layer.linear_attn {
                for (name, projection) in [
                    ("delta.qkv", &delta.in_proj_qkv),
                    ("delta.output_gate", &delta.in_proj_z),
                    ("delta.beta", &delta.in_proj_b),
                    ("delta.decay", &delta.in_proj_a),
                    ("delta.output", &delta.out_proj),
                ] {
                    assert_finite_nonzero_gradient(
                        projection.weight.grad(&gradients),
                        &format!("layer {index} {name}"),
                    );
                }
                assert_finite_nonzero_gradient(
                    delta.conv1d.weight.grad(&gradients),
                    &format!("layer {index} delta.convolution"),
                );
                for (name, parameter) in [
                    ("delta.dt_bias", &delta.dt_bias),
                    ("delta.a_log", &delta.a_log),
                    ("delta.norm", &delta.norm.weight),
                ] {
                    assert_finite_nonzero_gradient(
                        parameter.grad(&gradients),
                        &format!("layer {index} {name}"),
                    );
                }
            }
            if let Some(attention) = &layer.self_attn {
                for (name, projection) in [
                    ("attention.query_and_gate", &attention.q_proj),
                    ("attention.key", &attention.k_proj),
                    ("attention.value", &attention.v_proj),
                    ("attention.output", &attention.o_proj),
                ] {
                    assert_finite_nonzero_gradient(
                        projection.weight.grad(&gradients),
                        &format!("layer {index} {name}"),
                    );
                }
                for (name, norm) in [
                    ("attention.query_norm", attention.query_norm.as_ref()),
                    ("attention.key_norm", attention.key_norm.as_ref()),
                ] {
                    assert_finite_nonzero_gradient(
                        norm.expect("Qwen enables Q/K normalization")
                            .weight
                            .grad(&gradients),
                        &format!("layer {index} {name}"),
                    );
                }
            }
        }
    }
}

/// Last-logit selection follows each row's last valid token under padding.
#[test]
fn last_selection_uses_semantic_mask() {
    let device = Default::default();
    let hidden = Tensor::from_floats(
        [
            [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
            [[4.0, 40.0], [5.0, 50.0], [6.0, 60.0]],
        ],
        &device,
    );
    let mask = TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
        [[true, true, false], [true, false, false]].into(),
        &device,
    ));
    let selected = select_hidden(hidden, Some(&mask), &LogitsSelection::Last)
        .expect("valid selection")
        .into_data()
        .to_vec::<f32>()
        .expect("f32 values");
    assert_eq!(selected, vec![2.0, 20.0, 4.0, 40.0]);
}

/// Invalid IDs, masks, positions, and explicit selections return errors.
#[test]
fn input_validation_precedes_backend_indexing() {
    let device = Default::default();
    assert!(
        validate_token_ids(
            Tensor::<TestBackend, 2, Int>::from_ints([[32]], &device),
            32
        )
        .is_err()
    );
    let empty = TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
        [[false, false]].into(),
        &device,
    ));
    assert!(validate_mask(Some(&empty), 1, 2).is_err());
    assert!(
        select_hidden(
            Tensor::<TestBackend, 3>::zeros([1, 2, 3], &device),
            None,
            &LogitsSelection::Positions(vec![2]),
        )
        .is_err()
    );
}

/// Stateless execution enforces the same declared context as cached decoding.
#[test]
fn stateless_positions_fail_before_embedding_or_attention_outside_context() {
    let device = Default::default();
    let model = Qwen3_5ForCausalLm::<TestBackend>::init(tiny_config(), &device)
        .expect("tiny model must initialize");
    let token_ids = Tensor::<TestBackend, 2, Int>::from_ints([[1, 2]], &device);

    for positions in [[0, -1], [0, 64]] {
        let error = model
            .forward(
                CausalLmInput {
                    token_ids: token_ids.clone(),
                    token_mask: None,
                    position_ids: Some(Tensor::from_ints([positions], &device)),
                },
                &ForwardOptions::default(),
            )
            .expect_err("invalid stateless positions must be recoverable");
        assert!(matches!(error, Error::Shape { .. }));
        assert!(error.to_string().contains("qwen.position_ids"));
    }
}

/// Stateless device preflight identifies every independently supplied tensor.
#[test]
fn stateless_device_affinity_errors_are_contextual() {
    assert!(ensure_model_device("qwen.input_ids", &0_u8, &0_u8).is_ok());
    for tensor in [
        "qwen.input_ids",
        "qwen.token_mask",
        "qwen.position_ids",
        "qwen.mtp.target_hidden_state",
        "qwen.lm_head",
    ] {
        let error = ensure_model_device(tensor, &1_u8, &0_u8)
            .expect_err("different device identities must fail preflight");
        assert!(error.to_string().contains(tensor));
        assert!(error.to_string().contains("model parameters"));
    }
}

/// The official reduced Qwen3.8 fixture constructs all lazy parameter shapes.
#[test]
fn official_qwen38_topology_initializes_lazily() {
    let device = Default::default();
    let config =
        Qwen3_5TextConfig::from_json_str(include_str!("../../../tests/fixtures/qwen3_8_27b.json"))
            .expect("pinned fixture must parse");
    let model = Qwen3_5ForCausalLm::<TestBackend>::init(config, &device)
        .expect("official topology must initialize lazily");
    assert_eq!(model.model.layers.len(), 64);
    assert_eq!(
        model.mtp.as_ref().expect("official MTP graph").layers.len(),
        1
    );
    assert_eq!(model.config().vocab_size, 248_320);
    assert_eq!(model.model.rope.rotary_dimension(), 64);
}
