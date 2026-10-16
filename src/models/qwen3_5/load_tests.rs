//! Synthetic Qwen parameter installation, tied aliases, and malformed payload tests.

use burn::{
    backend::Flex,
    tensor::{Int, Tensor, TensorData},
};

use crate::checkpoint::quantization::{
    QuantizationEligibility, QuantizationGroupSize, WeightQuantizationConfig,
    classify_weight_for_quantization, quantize_tensor_data,
};
use crate::{
    ForwardOptions, architecture::CausalLmInput, checkpoint::TensorSink,
    models::config::ModelConfig,
};

use super::*;
use crate::models::qwen3_5::{
    Qwen3_5CheckpointOptions, Qwen3_5CheckpointSource, Qwen3_5TextConfig,
};

/// CPU Flex backend with F32 activations and portable I32 indices.
type TestBackend = Flex<f32, i32>;

/// Loads every base-graph target from the synthetic contract and runs forward.
#[test]
fn tiny_contract_dispatches_every_executable_parameter() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../tests/fixtures/qwen3_5_checkpoint/tiny_text_inventory.json"
    ))
    .unwrap();
    let config =
        Qwen3_5TextConfig::from_json_str(&fixture.get("config").unwrap().to_string()).unwrap();
    let contract = config
        .checkpoint_contract(Qwen3_5CheckpointOptions {
            source: Qwen3_5CheckpointSource::TextOnly,
            include_mtp: false,
        })
        .unwrap();
    let device = Default::default();
    let mut model = Qwen3_5ForCausalLm::<TestBackend>::init(config, &device).unwrap();

    for expected in contract.expected_tensors() {
        if expected.name == "lm_head.weight" {
            continue;
        }
        let shape = expected
            .shape
            .iter()
            .map(|dimension| usize::try_from(*dimension).unwrap())
            .collect::<Vec<_>>();
        model
            .write_tensor(&expected.name, TensorData::zeros::<f32, _>(shape))
            .unwrap_or_else(|error| panic!("{}: {error}", expected.name));
    }
    model
        .alias_tensor("lm_head.weight", "model.embed_tokens.weight")
        .unwrap();

    let output = model
        .forward(
            CausalLmInput {
                token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[1, 2]], &device),
                token_mask: None,
                position_ids: None,
            },
            &ForwardOptions::default(),
        )
        .unwrap();
    assert_eq!(output.logits.dims(), [1, 1, 32]);
    assert!(
        output
            .logits
            .to_data()
            .as_slice::<f32>()
            .unwrap()
            .iter()
            .all(|value| *value == 0.0)
    );
}

/// Unknown, malformed, schedule-mismatched, and premature aliases are typed.
#[test]
fn invalid_dispatch_paths_are_contextual_and_non_panicking() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../tests/fixtures/qwen3_5_checkpoint/tiny_text_inventory.json"
    ))
    .unwrap();
    let config =
        Qwen3_5TextConfig::from_json_str(&fixture.get("config").unwrap().to_string()).unwrap();
    let device = Default::default();
    let mut model = Qwen3_5ForCausalLm::<TestBackend>::init(config, &device).unwrap();

    let alias = model
        .alias_tensor("lm_head.weight", "model.embed_tokens.weight")
        .unwrap_err();
    assert!(matches!(alias, Qwen3_5LoadError::InvalidAlias { .. }));

    let malformed = model
        .write_tensor(
            "model.layers.not-an-index.weight",
            TensorData::from([0.0_f32]),
        )
        .unwrap_err();
    assert!(matches!(malformed, Qwen3_5LoadError::InvalidLayer { .. }));

    let wrong_mixer = model
        .write_tensor(
            "model.layers.0.self_attn.q_proj.weight",
            TensorData::zeros::<f32, _>([16, 8]),
        )
        .unwrap_err();
    assert!(matches!(
        wrong_mixer,
        Qwen3_5LoadError::MissingComponent { .. }
    ));

    let unknown = model
        .write_tensor("mtp.not_a_parameter", TensorData::zeros::<f32, _>([8, 16]))
        .unwrap_err();
    assert!(matches!(unknown, Qwen3_5LoadError::UnknownTarget { .. }));
}

/// The strict MTP contract has a concrete sink for every declared tensor.
#[test]
fn mtp_contract_dispatches_all_fifteen_parameters() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../tests/fixtures/qwen3_5_checkpoint/tiny_text_inventory.json"
    ))
    .unwrap();
    let config =
        Qwen3_5TextConfig::from_json_str(&fixture.get("config").unwrap().to_string()).unwrap();
    let contract = config
        .checkpoint_contract(Qwen3_5CheckpointOptions {
            source: Qwen3_5CheckpointSource::TextOnly,
            include_mtp: true,
        })
        .unwrap();
    let mtp = contract
        .expected_tensors()
        .iter()
        .filter(|tensor| tensor.name.starts_with("mtp."))
        .collect::<Vec<_>>();
    assert_eq!(mtp.len(), 15);

    let device = Default::default();
    let mut model = Qwen3_5ForCausalLm::<TestBackend>::init(config, &device).unwrap();
    for expected in contract.expected_tensors() {
        if expected.name == "lm_head.weight" {
            continue;
        }
        let shape = expected
            .shape
            .iter()
            .map(|dimension| usize::try_from(*dimension).unwrap())
            .collect::<Vec<_>>();
        model
            .write_tensor(&expected.name, TensorData::zeros::<f32, _>(shape))
            .unwrap_or_else(|error| panic!("{}: {error}", expected.name));
    }
    model
        .alias_tensor("lm_head.weight", "model.embed_tokens.weight")
        .unwrap();

    let prediction = model
        .forward_mtp(
            Tensor::<TestBackend, 2, Int>::from_ints([[2]], &device),
            Tensor::zeros([1, 1, 8], &device),
            Tensor::<TestBackend, 2, Int>::from_ints([[0]], &device),
            0,
        )
        .expect("fully loaded MTP graph must execute");
    assert_eq!(prediction.logits.dims(), [1, 1, 32]);
    assert_eq!(prediction.hidden_state.dims(), [1, 1, 8]);
}

/// Every eligible Qwen projection executes when installed as native QFloat.
#[test]
fn qfloat_qwen_graph_matches_its_full_precision_reference() {
    let config = Qwen3_5TextConfig {
        vocab_size: 32,
        hidden_size: 32,
        intermediate_size: 32,
        num_hidden_layers: 2,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: 16,
        max_position_embeddings: 32,
        partial_rotary_factor: Some(1.0),
        layer_types: vec![
            crate::models::qwen3_5::Qwen3_5LayerType::LinearAttention,
            crate::models::qwen3_5::Qwen3_5LayerType::FullAttention,
        ],
        linear_conv_kernel_dim: 3,
        linear_key_head_dim: 16,
        linear_value_head_dim: 16,
        linear_num_key_heads: 2,
        linear_num_value_heads: 2,
        mtp_num_hidden_layers: 0,
        tie_word_embeddings: false,
        attention_bias: false,
        attn_output_gate: true,
        ..Qwen3_5TextConfig::default()
    };
    let contract = config
        .checkpoint_contract(Qwen3_5CheckpointOptions {
            source: Qwen3_5CheckpointSource::TextOnly,
            include_mtp: false,
        })
        .unwrap();
    let device = Default::default();
    let mut reference = Qwen3_5ForCausalLm::<TestBackend>::init(config.clone(), &device)
        .expect("valid aligned reference graph");
    let mut packed = Qwen3_5ForCausalLm::<TestBackend>::init(config, &device)
        .expect("valid aligned packed graph");
    let quantization = WeightQuantizationConfig {
        group_size: QuantizationGroupSize::new(32).unwrap(),
        ..WeightQuantizationConfig::default()
    };
    let mut packed_count = 0;
    // Dequantized artifacts `[out, in]` of two checked projections.
    let mut artifacts = std::collections::BTreeMap::new();

    for expected in contract.expected_tensors() {
        let shape = expected
            .shape
            .iter()
            .map(|dimension| usize::try_from(*dimension).unwrap())
            .collect::<Vec<_>>();
        let elements = shape.iter().product::<usize>();
        let seed = expected
            .name
            .bytes()
            .fold(0_usize, |state, byte| state.wrapping_add(byte as usize));
        let values = (0..elements)
            .map(|index| (((index + seed) % 31) as f32 - 15.0) * 0.002)
            .collect::<Vec<_>>();
        let source = TensorData::new(values, shape.clone());
        reference
            .write_tensor(&expected.name, source.clone())
            .unwrap_or_else(|error| panic!("reference {}: {error}", expected.name));
        let packed_data = match classify_weight_for_quantization(&expected.name, &shape) {
            QuantizationEligibility::Eligible => {
                packed_count += 1;
                let quantized = quantize_tensor_data(&source, quantization)
                    .unwrap_or_else(|error| panic!("quantize {}: {error}", expected.name));
                artifacts.insert(expected.name.clone(), quantized.dequantize_f32().unwrap());
                quantized
                    .into_burn_qfloat_data()
                    .unwrap_or_else(|error| panic!("adapt {}: {error}", expected.name))
            }
            QuantizationEligibility::Excluded(_) => source,
        };
        packed
            .write_tensor(&expected.name, packed_data)
            .unwrap_or_else(|error| panic!("packed {}: {error}", expected.name));
    }
    assert_eq!(packed_count, 16);
    // Packed overrides install the artifact exactly: as QFloat where the
    // backend keeps the column layout's blocks (Metal), as the exact
    // dequantized values otherwise (Flex), never a requantized approximation.
    for (name, weight) in [
        (
            "model.layers.1.self_attn.q_proj.weight",
            packed.model.layers[1]
                .self_attn
                .as_ref()
                .unwrap()
                .q_proj
                .weight
                .val(),
        ),
        (
            "lm_head.weight",
            packed.lm_head.as_ref().unwrap().weight.val(),
        ),
    ] {
        assert_exact_artifact(weight, &artifacts[name]);
    }

    let input = CausalLmInput {
        token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[1, 2, 3]], &device),
        token_mask: None,
        position_ids: None,
    };
    let embedded = packed.model.embed(&input).unwrap();
    let mut individual = embedded.clone();
    for (index, block) in packed.model.layers.iter().enumerate() {
        individual.hidden = block
            .forward(
                individual.hidden,
                individual.context.positions.clone(),
                individual.context.token_mask.as_ref(),
                packed.model.layer_rope[index],
                individual.context.evaluated_context,
            )
            .unwrap();
        individual.next_layer = index + 1;
    }
    let ranged = packed
        .model
        .run_layers(embedded, 0..packed.model.layers.len())
        .unwrap();
    assert_eq!(
        packed.model.normalize(individual).unwrap().hidden.to_data(),
        packed.model.normalize(ranged).unwrap().hidden.to_data()
    );
    let expected = reference
        .forward(input.clone(), &ForwardOptions::default())
        .expect("reference forward")
        .logits
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    let actual = packed
        .forward(input, &ForwardOptions::default())
        .expect("packed forward")
        .logits
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    assert!(actual.iter().all(|value| value.is_finite()));
    let dot = expected
        .iter()
        .zip(&actual)
        .map(|(left, right)| left * right)
        .sum::<f32>();
    let left_norm = expected
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    let right_norm = actual.iter().map(|value| value * value).sum::<f32>().sqrt();
    let cosine = dot / (left_norm * right_norm);
    assert!(cosine >= 0.995, "Q8 cosine similarity was {cosine}");
}

/// Asserts a column-layout `[in, out]` linear weight holds exactly the dequantized `[out, in]` artifact.
fn assert_exact_artifact(weight: Tensor<TestBackend, 2>, artifact: &TensorData) {
    let installed = match weight.dtype() {
        burn::tensor::DType::QFloat(_) => weight.dequantize(),
        _ => weight,
    };
    let [inputs, outputs] = installed.dims();
    let values = installed.into_data().to_vec::<f32>().unwrap();
    let expected = artifact.to_vec::<f32>().unwrap();
    assert_eq!(artifact.shape.to_vec(), vec![outputs, inputs]);
    for input in 0..inputs {
        for output in 0..outputs {
            assert_eq!(
                values[input * outputs + output],
                expected[output * inputs + input],
                "weight[{input}, {output}]"
            );
        }
    }
}

/// The tiny hybrid fixture: DeltaNet block 0, gated full-attention block 1, one MTP layer, tied.
fn fixture_config() -> Qwen3_5TextConfig {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../tests/fixtures/qwen3_5_checkpoint/tiny_text_inventory.json"
    ))
    .unwrap();
    Qwen3_5TextConfig::from_json_str(&fixture.get("config").unwrap().to_string()).unwrap()
}

/// One `[1]`-shaped F32 payload; every rejection below fails before its shape matters.
fn scalar() -> TensorData {
    TensorData::from([0.0_f32])
}

/// Returns the internal target named by a sink error.
fn error_target(error: &Qwen3_5LoadError) -> String {
    match error {
        Qwen3_5LoadError::UnknownTarget { target }
        | Qwen3_5LoadError::MissingComponent { target, .. }
        | Qwen3_5LoadError::InvalidLayer { target, .. }
        | Qwen3_5LoadError::Parameter { target, .. }
        | Qwen3_5LoadError::InvalidAlias { target, .. } => target.clone(),
    }
}

/// Malformed names, out-of-range indices, and schedule-mismatched mixers produce typed errors.
///
/// Block 0 is DeltaNet and block 1 full attention without projection biases;
/// block 1's Q/K norms are removed to exercise their topology guards.
#[test]
fn sink_reports_every_target_rejection_with_its_reason() {
    let device = Default::default();
    let mut model = Qwen3_5ForCausalLm::<TestBackend>::init(fixture_config(), &device).unwrap();
    let attention = model.model.layers[1].self_attn.as_mut().unwrap();
    attention.query_norm = None;
    attention.key_norm = None;
    let invalid = |target: &str, reason: &str| Qwen3_5LoadError::InvalidLayer {
        target: target.into(),
        reason: reason.into(),
    };
    let missing = |target: &str, component: &'static str| Qwen3_5LoadError::MissingComponent {
        target: target.into(),
        component,
    };
    let unknown = |target: &str| Qwen3_5LoadError::UnknownTarget {
        target: target.into(),
    };
    let not_decimal =
        |index: &str| format!("`{index}` is not a decimal usize: invalid digit found in string");
    let cases = [
        unknown("model.bogus.weight"),
        unknown("mtp.bogus.weight"),
        unknown("model.layers.0.bogus.weight"),
        unknown("model.layers.1.self_attn.bogus.weight"),
        unknown("model.layers.0.linear_attn.bogus.weight"),
        invalid(
            "model.layers.0",
            "expected `model.layers.<index>.<parameter>`",
        ),
        invalid("model.layers.0.", "parameter suffix must not be empty"),
        invalid(
            "model.layers.2.input_layernorm.weight",
            "index 2 is outside 0..2",
        ),
        invalid("mtp.layers.0", "expected `mtp.layers.<index>.<parameter>`"),
        invalid("mtp.layers.0.", "parameter suffix must not be empty"),
        invalid("mtp.layers.x.input_layernorm.weight", &not_decimal("x")),
        invalid(
            "mtp.layers.1.input_layernorm.weight",
            "MTP index 1 is outside 0..1",
        ),
        missing(
            "model.layers.0.self_attn.q_proj.weight",
            "full-attention mixer",
        ),
        missing(
            "model.layers.1.linear_attn.in_proj_qkv.weight",
            "Gated DeltaNet mixer",
        ),
        missing(
            "model.layers.1.self_attn.q_proj.bias",
            "linear projection bias",
        ),
        missing("model.layers.1.self_attn.q_norm.weight", "query RMSNorm"),
        missing("model.layers.1.self_attn.k_norm.weight", "key RMSNorm"),
        missing(
            "lm_head.weight",
            "independent lm_head for an untied checkpoint",
        ),
    ];
    for expected in cases {
        let target = error_target(&expected);
        assert_eq!(
            model.write_tensor(&target, scalar()),
            Err(expected),
            "{target}"
        );
    }

    // A wrong shape is reported by the shared installer with the target.
    assert!(matches!(
        model.write_tensor("model.embed_tokens.weight", scalar()),
        Err(Qwen3_5LoadError::Parameter { target, .. }) if target == "model.embed_tokens.weight"
    ));

    // Without the auxiliary graph every `mtp.*` target names the missing graph.
    model.mtp = None;
    assert_eq!(
        model.write_tensor("mtp.fc.weight", scalar()),
        Err(missing(
            "mtp.fc.weight",
            "configured multi-token-prediction graph"
        ))
    );
}

/// The tied alias is the only one accepted, and only by a tied causal model.
#[test]
fn aliases_follow_the_tied_head_contract() {
    let device = Default::default();
    let alias = |target: &str, source: &str, reason: &str| Qwen3_5LoadError::InvalidAlias {
        target: target.into(),
        source_target: source.into(),
        reason: reason.into(),
    };
    let mut tied = Qwen3_5ForCausalLm::<TestBackend>::init(fixture_config(), &device).unwrap();
    for (target, source) in [
        ("lm_head.weight", "model.norm.weight"),
        ("model.norm.weight", "model.embed_tokens.weight"),
    ] {
        assert_eq!(
            tied.alias_tensor(target, source),
            Err(alias(
                target,
                source,
                "only the tied embedding-to-LM-head alias is supported"
            ))
        );
    }
    let untied_config = Qwen3_5TextConfig {
        tie_word_embeddings: false,
        ..fixture_config()
    };
    let mut untied = Qwen3_5ForCausalLm::<TestBackend>::init(untied_config, &device).unwrap();
    assert_eq!(
        untied.alias_tensor("lm_head.weight", "model.embed_tokens.weight"),
        Err(alias(
            "lm_head.weight",
            "model.embed_tokens.weight",
            "the configured checkpoint declares an independent LM head"
        ))
    );
    untied
        .write_tensor("lm_head.weight", TensorData::zeros::<f32, _>([32, 8]))
        .expect("an untied model owns its head");

    let mut headless = Qwen3_5TextModel::<TestBackend>::init(fixture_config(), &device).unwrap();
    assert_eq!(
        headless.alias_tensor("lm_head.weight", "model.embed_tokens.weight"),
        Err(alias(
            "lm_head.weight",
            "model.embed_tokens.weight",
            "the headless backbone stores only the embedding owner; vocabulary views are explicit execution operations"
        ))
    );
}

/// The headless backbone installs every base target and rejects layers it does not own.
#[test]
fn headless_backbone_installs_its_base_contract() {
    let config = fixture_config();
    let contract = config
        .checkpoint_contract(Qwen3_5CheckpointOptions {
            source: Qwen3_5CheckpointSource::TextOnly,
            include_mtp: false,
        })
        .unwrap();
    let device = Default::default();
    let mut model = Qwen3_5TextModel::<TestBackend>::init(config, &device).unwrap();
    for expected in contract.expected_tensors() {
        if expected.name == "lm_head.weight" {
            continue;
        }
        let shape = expected
            .shape
            .iter()
            .map(|dimension| usize::try_from(*dimension).unwrap())
            .collect::<Vec<_>>();
        model
            .write_tensor(&expected.name, TensorData::zeros::<f32, _>(shape))
            .unwrap_or_else(|error| panic!("{}: {error}", expected.name));
    }
    assert_eq!(
        model.write_tensor("model.layers.2.input_layernorm.weight", scalar()),
        Err(Qwen3_5LoadError::InvalidLayer {
            target: "model.layers.2.input_layernorm.weight".into(),
            reason: "index 2 is outside 0..2".into(),
        })
    );
    let (hidden, _) = model
        .forward(
            &CausalLmInput {
                token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[1, 2]], &device),
                token_mask: None,
                position_ids: None,
            },
            false,
        )
        .unwrap();
    assert!(hidden.into_data().iter::<f32>().all(|value| value == 0.0));
}

/// Distinct marker for a full-attention projection bias target, `None` otherwise.
///
/// The marker is `kind + 10 * owner`, with `kind` 1..=4 for the q, k, v, and
/// o projections and `owner` 1 for the MTP graph (0 for the base backbone),
/// so a bias installed into the wrong projection or graph is visible.
fn projection_bias_marker(name: &str) -> Option<f32> {
    let kind = ["q_proj", "k_proj", "v_proj", "o_proj"]
        .iter()
        .position(|projection| name.ends_with(&format!("self_attn.{projection}.bias")))?;
    let owner = if name.starts_with("mtp.") { 10.0 } else { 0.0 };
    Some(kind as f32 + 1.0 + owner)
}

/// Biased full attention installs each of the four projection biases into its own field.
///
/// Every bias target carries a distinct constant from
/// [`projection_bias_marker`]. In Qwen3.5 `k_proj.bias` and `v_proj.bias`
/// share the `[kv_heads * head_dim]` shape, so only distinct payloads can
/// show that the dispatcher neither swaps them nor routes an MTP bias into
/// the base block. Reading every Burn bias back proves each landed intact.
#[test]
fn biased_attention_projections_install_their_biases() {
    let config = Qwen3_5TextConfig {
        attention_bias: true,
        ..fixture_config()
    };
    let contract = config
        .checkpoint_contract(Qwen3_5CheckpointOptions {
            source: Qwen3_5CheckpointSource::TextOnly,
            include_mtp: true,
        })
        .unwrap();
    let device = Default::default();
    let mut model = Qwen3_5ForCausalLm::<TestBackend>::init(config, &device).unwrap();
    let mut biases = 0;
    for expected in contract.expected_tensors() {
        if expected.name == "lm_head.weight" {
            continue;
        }
        let shape = expected
            .shape
            .iter()
            .map(|dimension| usize::try_from(*dimension).unwrap())
            .collect::<Vec<_>>();
        let data = match projection_bias_marker(&expected.name) {
            Some(marker) => {
                biases += 1;
                TensorData::full(shape, marker)
            }
            None => TensorData::zeros::<f32, _>(shape),
        };
        model
            .write_tensor(&expected.name, data)
            .unwrap_or_else(|error| panic!("{}: {error}", expected.name));
    }
    // One full-attention base block and one MTP block, four biases each.
    assert_eq!(biases, 8);

    let base = model
        .model
        .layers
        .iter()
        .filter_map(|layer| layer.self_attn.as_ref())
        .map(|attention| (0.0, attention));
    let mtp = model
        .mtp
        .as_ref()
        .expect("the contract includes the MTP graph")
        .layers
        .iter()
        .filter_map(|layer| layer.self_attn.as_ref())
        .map(|attention| (10.0, attention));
    let mut checked = 0;
    for (owner, attention) in base.chain(mtp) {
        let projections = [
            &attention.q_proj,
            &attention.k_proj,
            &attention.v_proj,
            &attention.o_proj,
        ];
        for (kind, projection) in projections.into_iter().enumerate() {
            let expected = kind as f32 + 1.0 + owner;
            let values = projection
                .bias
                .as_ref()
                .expect("a biased projection owns a bias")
                .val()
                .into_data()
                .to_vec::<f32>()
                .unwrap();
            assert!(!values.is_empty());
            assert!(
                values.iter().all(|&value| value == expected),
                "projection {kind} of owner {owner} holds {values:?}, expected {expected}"
            );
            checked += 1;
        }
    }
    assert_eq!(checked, biases);
}

/// A tied configuration whose model nevertheless owns an independent head rejects the alias.
#[test]
fn tied_alias_rejects_a_model_that_owns_a_head() {
    let device = Default::default();
    let mut tied = Qwen3_5ForCausalLm::<TestBackend>::init(fixture_config(), &device).unwrap();
    let untied = Qwen3_5ForCausalLm::<TestBackend>::init(
        Qwen3_5TextConfig {
            tie_word_embeddings: false,
            ..fixture_config()
        },
        &device,
    )
    .unwrap();
    tied.lm_head = untied.lm_head;
    tied.write_tensor(
        "model.embed_tokens.weight",
        TensorData::zeros::<f32, _>([32, 8]),
    )
    .unwrap();
    assert_eq!(
        tied.alias_tensor("lm_head.weight", "model.embed_tokens.weight"),
        Err(Qwen3_5LoadError::InvalidAlias {
            target: "lm_head.weight".into(),
            source_target: "model.embed_tokens.weight".into(),
            reason: "the configured checkpoint declares an independent LM head".into(),
        })
    );
}
