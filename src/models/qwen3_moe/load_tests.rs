//! Synthetic checkpoint tests for streamed Qwen3 MoE installation and malformed artifacts.

use std::{fs::File, io::Write, path::Path};

use burn::{
    backend::Flex,
    module::{Module, ModuleVisitor, Param},
    nn::Linear,
    tensor::{Int, Tensor, TensorData},
};
use serde_json::{Map, Value, json};
use tempfile::TempDir;

use crate::{
    CausalLanguageModel, CausalLmInput, CausalLmOutput, DecodeInput, DecodeState, ForwardOptions,
    LogitsSelection,
    cache::CacheConfig,
    checkpoint::{CheckpointError, TensorOperation},
    models::config::{RopeParameters, RopeType},
    position::{RopeScaling, inverse_frequencies, rotary_angles},
};

use super::*;
use crate::models::qwen3_moe::Qwen3MoeConfig;

/// CPU Flex backend with F32 activations and portable I32 indices.
type TestBackend = Flex<f32, i32>;

/// Minimal routed topology used by sink and streaming tests.
fn tiny_config(tied: bool) -> Qwen3MoeConfig {
    Qwen3MoeConfig {
        vocab_size: 16,
        hidden_size: 4,
        intermediate_size: 8,
        num_hidden_layers: 1,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: 2,
        max_position_embeddings: 16,
        max_window_layers: 1,
        decoder_sparse_step: 1,
        moe_intermediate_size: 3,
        num_experts: 3,
        num_experts_per_tok: 2,
        norm_topk_prob: true,
        tie_word_embeddings: tied,
        ..Default::default()
    }
}

/// Writes a complete zero-valued F32 SafeTensors artifact.
fn write_checkpoint(
    root: &Path,
    contract: &super::super::Qwen3MoeCheckpointContract,
    mutate: impl FnOnce(&mut Vec<(String, Vec<u64>)>),
) {
    let mut tensors = contract
        .mappings()
        .operations()
        .iter()
        .filter_map(|operation| match operation {
            TensorOperation::Map { source, target, .. } => {
                let shape = contract
                    .expected_tensors()
                    .iter()
                    .find(|expected| &expected.name == target)
                    .unwrap()
                    .shape
                    .clone();
                Some((source.clone(), shape))
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    mutate(&mut tensors);

    let mut offset = 0_u64;
    let mut header = Map::new();
    for (name, shape) in tensors {
        let elements = shape.iter().copied().product::<u64>();
        let end = offset + elements * 4;
        header.insert(
            name,
            json!({"dtype": "F32", "shape": shape, "data_offsets": [offset, end]}),
        );
        offset = end;
    }
    let mut header = serde_json::to_vec(&Value::Object(header)).unwrap();
    while !header.len().is_multiple_of(8) {
        header.push(b' ');
    }
    let mut file = File::create(root.join("model.safetensors")).unwrap();
    file.write_all(&(header.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(&header).unwrap();
    file.write_all(&vec![0; offset as usize]).unwrap();
}

/// Every official sparse target dispatches to its exact Burn field.
#[test]
fn complete_contract_installs_and_runs() {
    let config = tiny_config(false);
    let contract = config
        .checkpoint_contract(Qwen3MoeCheckpointOptions::default())
        .unwrap();
    let device = Default::default();
    let mut model = Qwen3MoeForCausalLm::<TestBackend>::init(config, &device).unwrap();
    for expected in contract.expected_tensors() {
        let shape = expected
            .shape
            .iter()
            .map(|&dimension| dimension as usize)
            .collect::<Vec<_>>();
        model
            .write_tensor(&expected.name, TensorData::zeros::<f32, _>(shape))
            .unwrap_or_else(|error| panic!("{}: {error}", expected.name));
    }
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
    assert!(
        output
            .logits
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .iter()
            .all(|&value| value == 0.0)
    );
}

/// Streamed local loading opens one shard and applies every parameter once.
#[test]
fn local_loader_streams_tied_and_untied_checkpoints() {
    let device = Default::default();
    for tied in [false, true] {
        let config = tiny_config(tied);
        let contract = config
            .checkpoint_contract(Qwen3MoeCheckpointOptions::default())
            .unwrap();
        let temp = TempDir::new().unwrap();
        write_checkpoint(temp.path(), &contract, |_| {});
        let mut model = Qwen3MoeForCausalLm::<TestBackend>::init(config, &device).unwrap();
        let outcome = model
            .load_checkpoint(temp.path(), &LoadOptions::default())
            .unwrap();
        assert_eq!(
            outcome.report.applied.len(),
            contract.expected_tensors().len()
        );
        assert_eq!(outcome.stats.source.shard_opens, 1);
        assert_eq!(outcome.stats.source.max_open_shards, 1);
        assert_eq!(outcome.stats.aliases_applied, usize::from(tied));
    }
}

/// Planning rejects malformed inventories before initializing a parameter.
#[test]
fn missing_shape_and_unexpected_fail_before_writes() {
    let device = Default::default();
    let config = tiny_config(false);
    let contract = config
        .checkpoint_contract(Qwen3MoeCheckpointOptions::default())
        .unwrap();

    let missing = TempDir::new().unwrap();
    write_checkpoint(missing.path(), &contract, |tensors| {
        tensors.retain(|(name, _)| name != "model.layers.0.mlp.gate.weight");
    });
    let mut model = Qwen3MoeForCausalLm::<TestBackend>::init(config.clone(), &device).unwrap();
    let error = model
        .load_checkpoint(missing.path(), &LoadOptions::default())
        .unwrap_err();
    assert!(matches!(
        error,
        Qwen3MoeCheckpointLoadError::Checkpoint(CheckpointError::MissingMappingSource { .. })
    ));
    assert!(!model.model.embed_tokens.weight.is_initialized());

    let unexpected = TempDir::new().unwrap();
    write_checkpoint(unexpected.path(), &contract, |tensors| {
        tensors.push(("unexpected.weight".into(), vec![1]));
    });
    let mut model = Qwen3MoeForCausalLm::<TestBackend>::init(config.clone(), &device).unwrap();
    let error = model
        .load_checkpoint(unexpected.path(), &LoadOptions::default())
        .unwrap_err();
    assert!(matches!(
        error,
        Qwen3MoeCheckpointLoadError::Checkpoint(CheckpointError::StrictValidation {
            unexpected: 1,
            ..
        })
    ));
    assert!(!model.model.embed_tokens.weight.is_initialized());

    let wrong = TempDir::new().unwrap();
    write_checkpoint(wrong.path(), &contract, |tensors| {
        tensors
            .iter_mut()
            .find(|(name, _)| name == "model.embed_tokens.weight")
            .unwrap()
            .1 = vec![15, 4];
    });
    let mut model = Qwen3MoeForCausalLm::<TestBackend>::init(config, &device).unwrap();
    let error = model
        .load_checkpoint(wrong.path(), &LoadOptions::default())
        .unwrap_err();
    assert!(matches!(
        error,
        Qwen3MoeCheckpointLoadError::Checkpoint(CheckpointError::ShapeMismatch { .. })
    ));
    assert!(!model.model.embed_tokens.weight.is_initialized());
}

/// Wrong dtype, malformed names, expert bounds, and topology mismatch are typed.
#[test]
fn sink_rejects_invalid_targets_and_payloads() {
    let device = Default::default();
    let mut model = Qwen3MoeForCausalLm::<TestBackend>::init(tiny_config(false), &device).unwrap();
    assert!(matches!(
        model.write_tensor(
            "model.embed_tokens.weight",
            TensorData::zeros::<i32, _>([16, 4])
        ),
        Err(Qwen3MoeLoadError::Parameter { .. })
    ));
    assert!(matches!(
        model.write_tensor("model.layers.bad.weight", TensorData::from([0.0_f32])),
        Err(Qwen3MoeLoadError::InvalidIndex { .. })
    ));
    assert!(matches!(
        model.write_tensor(
            "model.layers.0.sparse_mlp.routed.9.gate_proj.weight",
            TensorData::zeros::<f32, _>([3, 4])
        ),
        Err(Qwen3MoeLoadError::InvalidIndex { .. })
    ));
    model.model.layers[0].sparse_mlp = None;
    assert!(matches!(
        model.write_tensor(
            "model.layers.0.sparse_mlp.router.weight",
            TensorData::zeros::<f32, _>([3, 4])
        ),
        Err(Qwen3MoeLoadError::MissingComponent { .. })
    ));
}

/// Mixes one dense block with one routed block and explicit, biased GQA projections.
fn mixed_projection_config(rope_type: RopeType) -> Qwen3MoeConfig {
    Qwen3MoeConfig {
        hidden_size: 8,
        intermediate_size: 12,
        moe_intermediate_size: 6,
        num_hidden_layers: 2,
        head_dim: 4,
        num_experts: 4,
        mlp_only_layers: vec![0],
        attention_bias: true,
        max_position_embeddings: 32,
        // The nested theta is authoritative; a disagreeing legacy top-level
        // theta would be rejected by the shared translator.
        rope_theta: None,
        rope_parameters: Some(RopeParameters {
            rope_type,
            rope_theta: Some(100.0),
            factor: Some(2.0),
            // Linear RoPE reads no original context, so only YaRN carries one.
            original_max_position_embeddings: (rope_type == RopeType::Yarn).then_some(8),
            attention_factor: (rope_type == RopeType::Yarn).then_some(1.25),
            beta_fast: (rope_type == RopeType::Yarn).then_some(4.0),
            beta_slow: (rope_type == RopeType::Yarn).then_some(1.0),
            ..Default::default()
        }),
        ..tiny_config(false)
    }
}

/// Produces stable named payloads, including every dormant expert, without an RNG.
fn mixed_projection_values(target: &str, elements: usize) -> Vec<f32> {
    let seed = target.bytes().map(usize::from).sum::<usize>();
    (0..elements)
        .map(|index| {
            if target.ends_with("norm.weight") {
                1.0
            } else {
                ((index * 7 + seed) % 31) as f32 * 0.01 - 0.15
            }
        })
        .collect()
}

/// Loads a complete deterministic local artifact through the public strict family loader.
fn mixed_projection_model(rope_type: RopeType) -> Qwen3MoeForCausalLm<TestBackend> {
    let config = mixed_projection_config(rope_type);
    let contract = config.checkpoint_contract(Default::default()).unwrap();
    let mut header = Map::new();
    let mut payload = Vec::new();
    for operation in contract.mappings().operations() {
        let TensorOperation::Map { source, target, .. } = operation else {
            panic!("untied miniature must contain only direct mappings");
        };
        let expected = contract
            .expected_tensors()
            .iter()
            .find(|tensor| &tensor.name == target)
            .unwrap();
        let start = payload.len();
        for value in
            mixed_projection_values(target, expected.shape.iter().product::<u64>() as usize)
        {
            payload.extend_from_slice(&value.to_le_bytes());
        }
        assert!(header.insert(source.clone(), json!({
            "dtype": "F32", "shape": expected.shape, "data_offsets": [start, payload.len()]
        })).is_none());
    }
    let mut header = serde_json::to_vec(&header).unwrap();
    while !header.len().is_multiple_of(8) {
        header.push(b' ');
    }
    let temp = TempDir::new().unwrap();
    let mut file = File::create(temp.path().join("model.safetensors")).unwrap();
    file.write_all(&(header.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(&header).unwrap();
    file.write_all(&payload).unwrap();
    drop(file);
    let mut model = Qwen3MoeForCausalLm::init(config, &Default::default()).unwrap();
    let outcome = model
        .load_checkpoint(temp.path(), &LoadOptions::default())
        .unwrap();
    let mut applied = outcome.report.applied.clone();
    applied.sort();
    let mut expected = contract
        .expected_tensors()
        .iter()
        .map(|tensor| tensor.name.clone())
        .collect::<Vec<_>>();
    expected.sort();
    assert_eq!(applied, expected);
    assert!(outcome.report.missing.is_empty() && outcome.report.unexpected.is_empty());
    assert_eq!(outcome.stats.source.shard_opens, 1);
    assert_eq!(outcome.stats.source.max_open_shards, 1);
    assert_eq!(outcome.stats.aliases_applied, 0);
    model
}

/// Audits complete loading without accidentally initializing a skipped lazy parameter.
struct MixedParameterAudit(
    /// Number of visited float parameters whose checkpoint payload is already installed.
    usize,
);

impl ModuleVisitor<TestBackend> for MixedParameterAudit {
    /// Requires an installed payload before counting a model parameter.
    fn visit_float<const D: usize>(&mut self, parameter: &Param<Tensor<TestBackend, D>>) {
        assert!(parameter.is_initialized());
        self.0 += 1;
    }
}

/// Compares installed column-layout weights to their checkpoint row-major payloads.
fn assert_mixed_projection(
    linear: &Linear<TestBackend>,
    target: &str,
    input: usize,
    output: usize,
    biased: bool,
) {
    assert_eq!(linear.weight.val().dims(), [input, output]);
    assert_eq!(
        linear
            .weight
            .val()
            .transpose()
            .to_data()
            .to_vec::<f32>()
            .unwrap(),
        mixed_projection_values(&format!("{target}.weight"), input * output)
    );
    assert_eq!(linear.bias.is_some(), biased);
    if let Some(bias) = &linear.bias {
        assert_eq!(bias.val().dims(), [output]);
        assert_eq!(
            bias.val().to_data().to_vec::<f32>().unwrap(),
            mixed_projection_values(&format!("{target}.bias"), output)
        );
    }
}

/// Strict mixed loading installs dense SwiGLU, sparse experts, and both layers' GQA biases.
#[test]
fn mixed_projections_load_dense_sparse_and_biased_attention_exactly() {
    let model = mixed_projection_model(RopeType::Linear);
    let contract = model
        .config()
        .checkpoint_contract(Default::default())
        .unwrap();
    for projection in ["gate_proj", "up_proj", "down_proj"] {
        let source = format!("model.layers.0.mlp.{projection}.weight");
        let target = format!("model.layers.0.dense_mlp.{projection}.weight");
        assert!(
            contract
                .mappings()
                .operations()
                .iter()
                .any(|operation| matches!(operation,
            TensorOperation::Map { source: actual_source, target: actual_target, .. }
                if actual_source == &source && actual_target == &target))
        );
    }
    assert!(contract.expected_tensors().iter().all(|tensor| {
        !tensor.name.starts_with("model.layers.0.sparse_mlp.")
            && !tensor.name.starts_with("model.layers.1.dense_mlp.")
    }));
    let mut audit = MixedParameterAudit(0);
    model.visit(&mut audit);
    assert_eq!(audit.0, contract.expected_tensors().len());
    assert!(model.model.layers[0].sparse_mlp.is_none());
    assert!(model.model.layers[1].dense_mlp.is_none());
    let dense = model.model.layers[0].dense_mlp.as_ref().unwrap();
    assert_mixed_projection(
        &dense.gate_proj,
        "model.layers.0.dense_mlp.gate_proj",
        8,
        12,
        false,
    );
    assert_mixed_projection(
        &dense.up_proj,
        "model.layers.0.dense_mlp.up_proj",
        8,
        12,
        false,
    );
    assert_mixed_projection(
        &dense.down_proj,
        "model.layers.0.dense_mlp.down_proj",
        12,
        8,
        false,
    );
    for (layer, block) in model.model.layers.iter().enumerate() {
        for (projection, name, output) in [
            (&block.self_attn.q_proj, "q_proj", 8),
            (&block.self_attn.k_proj, "k_proj", 4),
            (&block.self_attn.v_proj, "v_proj", 4),
            (&block.self_attn.o_proj, "o_proj", 8),
        ] {
            assert_mixed_projection(
                projection,
                &format!("model.layers.{layer}.self_attn.{name}"),
                8,
                output,
                true,
            );
        }
    }
    let sparse = model.model.layers[1].sparse_mlp.as_ref().unwrap();
    assert_mixed_projection(
        &sparse.router,
        "model.layers.1.sparse_mlp.router",
        8,
        4,
        false,
    );
    for (expert, feed_forward) in sparse.routed.iter().enumerate() {
        let prefix = format!("model.layers.1.sparse_mlp.routed.{expert}");
        assert_mixed_projection(
            &feed_forward.gate_proj,
            &format!("{prefix}.gate_proj"),
            8,
            6,
            false,
        );
        assert_mixed_projection(
            &feed_forward.up_proj,
            &format!("{prefix}.up_proj"),
            8,
            6,
            false,
        );
        assert_mixed_projection(
            &feed_forward.down_proj,
            &format!("{prefix}.down_proj"),
            6,
            8,
            false,
        );
    }
}

/// Serialized scaling overrides select independently known four-channel frequencies and amplitudes.
#[test]
fn mixed_projections_scaled_positions_match_scalar_oracles() {
    for rope_type in [RopeType::Linear, RopeType::Yarn] {
        let model = mixed_projection_model(rope_type);
        let rope = model.model.rope;
        assert_eq!(rope.theta(), 100.0); // The nested checkpoint theta is used.
        assert_eq!((rope.head_dimension(), rope.rotary_dimension()), (4, 4));
        let (frequencies, amplitude) = match rope.scaling() {
            RopeScaling::Linear(linear) => {
                assert_eq!(rope_type, RopeType::Linear);
                assert_eq!(linear.factor(), 2.0);
                // theta^(-2i/4)/factor gives [1/2, 1/20].
                ([0.5_f64, 0.05], 1.0)
            }
            RopeScaling::Yarn(yarn) => {
                assert_eq!(rope_type, RopeType::Yarn);
                assert_eq!(yarn.factor(), 2.0);
                assert_eq!(yarn.original_max_position_embeddings(), 8);
                assert_eq!((yarn.beta_fast(), yarn.beta_slow()), (4.0, 1.0));
                assert_eq!(yarn.attention_factor(), 1.25);
                // d=4, theta=100, L=8, beta=(4,1) gives correction bounds [0,1].
                // Thus the first frequency extrapolates; the second interpolates by two.
                ([1.0_f64, 0.05], 1.25)
            }
            other => panic!("unexpected initialized scaling {other:?}"),
        };
        let actual = inverse_frequencies(rope, 12).unwrap();
        assert_eq!(actual.len(), 2);
        for (actual, expected) in actual.into_iter().zip(frequencies) {
            assert!(actual.is_finite());
            assert!((f64::from(actual) - expected).abs() <= 1e-6);
        }
        let positions = [0_i32, 3, 11];
        let angles = rotary_angles::<TestBackend>(
            rope,
            Tensor::from_ints([positions], &Default::default()),
            12,
        )
        .unwrap();
        assert_eq!(angles.cosine.dims(), [1, 1, 3, 4]);
        assert_eq!(angles.sine.dims(), [1, 1, 3, 4]);
        let cosine = angles.cosine.to_data().to_vec::<f32>().unwrap();
        let sine = angles.sine.to_data().to_vec::<f32>().unwrap();
        for (row, position) in positions.into_iter().enumerate() {
            for channel in 0..4 {
                let phase = f64::from(position) * frequencies[channel % 2];
                for (actual, expected) in [
                    (cosine[row * 4 + channel], amplitude * phase.cos()),
                    (sine[row * 4 + channel], amplitude * phase.sin()),
                ] {
                    assert!(actual.is_finite() && expected.is_finite());
                    assert!((f64::from(actual) - expected).abs() <= 1e-6);
                }
            }
        }
    }
}

/// Compares equally shaped finite tensors with an absolute F32 tolerance.
fn mixed_projection_close(actual: Tensor<TestBackend, 3>, expected: Tensor<TestBackend, 3>) {
    assert_eq!(actual.dims(), expected.dims());
    let actual = actual.to_data().to_vec::<f32>().unwrap();
    let expected = expected.to_data().to_vec::<f32>().unwrap();
    assert!(expected.iter().any(|value| value.abs() > 1e-3));
    for (actual, expected) in actual.into_iter().zip(expected) {
        assert!(actual.is_finite() && expected.is_finite());
        assert!((actual - expected).abs() <= 1e-4, "{actual} != {expected}");
    }
}

/// Computes a scalar stable softmax without using Burn router or auxiliary-loss helpers.
fn mixed_host_softmax(values: &[f64]) -> Vec<f64> {
    let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exponents = values
        .iter()
        .map(|value| (value - maximum).exp())
        .collect::<Vec<_>>();
    let sum = exponents.iter().sum::<f64>();
    exponents.into_iter().map(|value| value / sum).collect()
}

/// One routed layer and one new token have loss `E * sum(top_k(softmax(softmax(raw))))`.
fn assert_mixed_single_token_router(output: &CausalLmOutput<TestBackend>) {
    assert_eq!(output.routers.len(), 1);
    let router = &output.routers[0];
    assert!(router.load_balancing_loss.is_none() && router.z_loss.is_none());
    let logits = router.logits.as_ref().unwrap();
    assert_eq!(logits.dims(), [1, 1, 4]);
    let values = logits.to_data().to_vec::<f32>().unwrap();
    assert!(values.iter().all(|value| value.is_finite()));
    let probabilities = mixed_host_softmax(&values.into_iter().map(f64::from).collect::<Vec<_>>());
    let mut recorded = mixed_host_softmax(&probabilities);
    recorded.sort_by(|left, right| right.total_cmp(left));
    let expected = 4.0 * (recorded[0] + recorded[1]);
    let actual = f64::from(
        output
            .load_balancing_loss
            .as_ref()
            .unwrap()
            .clone()
            .into_scalar(),
    );
    assert!(actual.is_finite() && expected.is_finite());
    assert!((actual - expected).abs() <= 1e-4, "{actual} != {expected}");
}

/// Linear and YaRN mixed graphs retain exactly one router and agree across cached continuations.
#[test]
fn mixed_projections_scaled_cached_forward_and_router_parity() {
    let device = Default::default();
    let options = ForwardOptions {
        logits: LogitsSelection::All,
        hidden_states: true,
        router_logits: true,
        auxiliary_losses: true,
        ..Default::default()
    };
    for rope_type in [RopeType::Linear, RopeType::Yarn] {
        let model = mixed_projection_model(rope_type);
        let reference = model
            .forward(
                CausalLmInput {
                    token_ids: Tensor::from_ints([[1, 2, 3, 4]], &device),
                    token_mask: None,
                    position_ids: Some(Tensor::from_ints([[1, 4, 9, 13]], &device)),
                },
                &options,
            )
            .unwrap();
        assert_eq!(reference.routers.len(), 1);
        assert_eq!(reference.hidden_states.len(), 4);
        let mut state = model
            .new_decode_state(CacheConfig::new(1, 8).unwrap(), 1, &device)
            .unwrap();
        assert_eq!(state.layer_count(), 2);
        let prefill = model
            .prefill(
                CausalLmInput {
                    token_ids: Tensor::from_ints([[1, 2]], &device),
                    token_mask: None,
                    position_ids: Some(Tensor::from_ints([[1, 4]], &device)),
                },
                &mut state,
                &options,
            )
            .unwrap();
        assert_eq!(prefill.routers.len(), 1);
        mixed_projection_close(prefill.logits, reference.logits.clone().narrow(1, 0, 2));
        for (token, position, column) in [(3, 9, 2), (4, 13, 3)] {
            let actual = model
                .decode(
                    DecodeInput {
                        token_ids: Tensor::from_ints([[token]], &device),
                        token_mask: None,
                        position_ids: Some(Tensor::from_ints([[position]], &device)),
                    },
                    &mut state,
                    &options,
                )
                .unwrap();
            assert_mixed_single_token_router(&actual);
            mixed_projection_close(
                actual.routers[0].logits.as_ref().unwrap().clone(),
                reference.routers[0]
                    .logits
                    .as_ref()
                    .unwrap()
                    .clone()
                    .narrow(1, column, 1),
            );
            mixed_projection_close(actual.logits, reference.logits.clone().narrow(1, column, 1));
        }
        assert_eq!(state.lengths(), &[4]);
    }
}
