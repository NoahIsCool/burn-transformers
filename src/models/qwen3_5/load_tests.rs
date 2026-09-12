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
                quantize_tensor_data(&source, quantization)
                    .unwrap_or_else(|error| panic!("quantize {}: {error}", expected.name))
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
    assert!(matches!(
        packed.model.layers[1]
            .self_attn
            .as_ref()
            .unwrap()
            .q_proj
            .weight
            .val()
            .dtype(),
        burn::tensor::DType::QFloat(_)
    ));
    assert!(matches!(
        packed.lm_head.as_ref().unwrap().weight.val().dtype(),
        burn::tensor::DType::QFloat(_)
    ));

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
