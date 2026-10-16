//! Nemotron-H streamed installation, optional parameter, and scalar-oracle tests.

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
    CausalLanguageModel, CausalLmInput, DecodeInput, DecodeState, ForwardOptions, LogitsSelection,
    cache::CacheConfig,
    checkpoint::{CheckpointError, TensorOperation},
};

use super::*;
use crate::models::nemotron_h::{NemotronHBlockType, NemotronHCheckpointContract, NemotronHConfig};

/// CPU Flex backend with F32 activations and I32 token indices for loader fixtures.
type TestBackend = Flex<f32, i32>;

/// Reduced hybrid topology retaining every sink-dispatch branch and MTP.
fn tiny_config(tied: bool) -> NemotronHConfig {
    NemotronHConfig {
        vocab_size: 16,
        hidden_size: 8,
        layers_block_type: vec![
            NemotronHBlockType::LinearAttention,
            NemotronHBlockType::FullAttention,
            NemotronHBlockType::Mlp,
            NemotronHBlockType::Moe,
        ],
        num_hidden_layers: Some(4),
        tie_word_embeddings: tied,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: 4,
        max_position_embeddings: 16,
        intermediate_size: 12,
        ssm_state_size: 3,
        mamba_num_heads: 4,
        mamba_head_dim: 2,
        n_groups: 2,
        conv_kernel: 3,
        chunk_size: 2,
        n_routed_experts: 4,
        n_shared_experts: 1,
        moe_intermediate_size: 6,
        moe_shared_expert_intermediate_size: 8,
        num_experts_per_tok: 2,
        n_group: 2,
        topk_group: 1,
        num_nextn_predict_layers: 1,
        ..Default::default()
    }
}

/// Writes one complete, zero-valued F32 SafeTensors fixture.
fn write_checkpoint(
    root: &Path,
    contract: &NemotronHCheckpointContract,
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
            TensorOperation::Alias { .. }
            | TensorOperation::Split { .. }
            | TensorOperation::Concatenate { .. }
            | TensorOperation::DequantizeMxfp4 { .. } => None,
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

/// Every base and MTP contract target reaches the exact ranked Burn field.
#[test]
fn complete_contract_installs_and_runs() {
    let config = tiny_config(false);
    let contract = config
        .checkpoint_contract(NemotronHCheckpointOptions::default())
        .unwrap();
    let device = Default::default();
    let mut model = NemotronHForCausalLm::<TestBackend>::init(config, &device).unwrap();
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

/// Pass two opens one local shard and applies tied and untied contracts.
#[test]
fn local_loader_streams_tied_and_untied_checkpoints() {
    let device = Default::default();
    for tied in [false, true] {
        let config = tiny_config(tied);
        let contract = config
            .checkpoint_contract(NemotronHCheckpointOptions::default())
            .unwrap();
        let temp = TempDir::new().unwrap();
        write_checkpoint(temp.path(), &contract, |_| {});
        let mut model = NemotronHForCausalLm::<TestBackend>::init(config, &device).unwrap();
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

/// Metadata mismatches fail before the first lazy parameter is initialized.
#[test]
fn strict_mismatches_fail_before_writes() {
    let device = Default::default();
    let config = tiny_config(false);
    let contract = config
        .checkpoint_contract(NemotronHCheckpointOptions::default())
        .unwrap();

    let missing = TempDir::new().unwrap();
    write_checkpoint(missing.path(), &contract, |tensors| {
        tensors.retain(|(name, _)| name != "backbone.layers.0.mixer.A_log");
    });
    let mut model = NemotronHForCausalLm::<TestBackend>::init(config.clone(), &device).unwrap();
    let error = model
        .load_checkpoint(missing.path(), &LoadOptions::default())
        .unwrap_err();
    assert!(matches!(
        error,
        NemotronHCheckpointLoadError::Checkpoint(CheckpointError::MissingMappingSource { .. })
    ));
    assert!(!model.model.embed_tokens.weight.is_initialized());

    let unexpected = TempDir::new().unwrap();
    write_checkpoint(unexpected.path(), &contract, |tensors| {
        tensors.push(("unexpected.weight".into(), vec![1]));
    });
    let mut model = NemotronHForCausalLm::<TestBackend>::init(config.clone(), &device).unwrap();
    let error = model
        .load_checkpoint(unexpected.path(), &LoadOptions::default())
        .unwrap_err();
    assert!(matches!(
        error,
        NemotronHCheckpointLoadError::Checkpoint(CheckpointError::StrictValidation {
            unexpected: 1,
            ..
        })
    ));
    assert!(!model.model.embed_tokens.weight.is_initialized());

    let wrong_shape = TempDir::new().unwrap();
    write_checkpoint(wrong_shape.path(), &contract, |tensors| {
        tensors
            .iter_mut()
            .find(|(name, _)| name == "backbone.embeddings.weight")
            .unwrap()
            .1 = vec![15, 8];
    });
    let mut model = NemotronHForCausalLm::<TestBackend>::init(config, &device).unwrap();
    let error = model
        .load_checkpoint(wrong_shape.path(), &LoadOptions::default())
        .unwrap_err();
    assert!(matches!(
        error,
        NemotronHCheckpointLoadError::Checkpoint(CheckpointError::ShapeMismatch { .. })
    ));
    assert!(!model.model.embed_tokens.weight.is_initialized());
}

/// Malformed ranks, indexes, aliases, and absent mixers return typed errors.
#[test]
fn sink_rejects_invalid_targets_and_payloads() {
    let device = Default::default();
    let mut model = NemotronHForCausalLm::<TestBackend>::init(tiny_config(false), &device).unwrap();
    assert!(matches!(
        model.write_tensor(
            "model.embed_tokens.weight",
            TensorData::zeros::<i32, _>([16, 8])
        ),
        Err(NemotronHLoadError::Parameter { .. })
    ));
    assert!(matches!(
        model.write_tensor("model.layers.bad.weight", TensorData::from([0.0_f32])),
        Err(NemotronHLoadError::InvalidIndex { .. })
    ));
    assert!(matches!(
        model.write_tensor(
            "model.layers.3.moe.routed.9.up_proj.weight",
            TensorData::zeros::<f32, _>([6, 8])
        ),
        Err(NemotronHLoadError::InvalidIndex { .. })
    ));
    assert!(matches!(
        model.write_tensor(
            "model.layers.0.attention.q_proj.weight",
            TensorData::zeros::<f32, _>([8, 8])
        ),
        Err(NemotronHLoadError::MissingComponent { .. })
    ));
    assert!(matches!(
        model.alias_tensor("other.weight", "model.embed_tokens.weight"),
        Err(NemotronHLoadError::InvalidAlias { .. })
    ));
}

/// Enables the public bias and latent-expert variants on the existing miniature.
fn optional_projection_config() -> NemotronHConfig {
    NemotronHConfig {
        attention_bias: true,
        use_bias: true,
        mlp_bias: true,
        moe_latent_size: Some(4),
        n_group: 1,
        topk_group: 1,
        norm_topk_prob: true,
        routed_scaling_factor: 1.7,
        ..tiny_config(false)
    }
}

/// Generates named, nonzero F32 payloads without initialization order or RNG state.
fn optional_projection_values(target: &str, elements: usize) -> Vec<f32> {
    let seed = target.bytes().map(usize::from).sum::<usize>();
    (0..elements)
        .map(|index| {
            if target.ends_with("correction_bias") {
                index as f32 * 0.25
            } else if target.ends_with("norm.weight") {
                1.0
            } else {
                ((index * 7 + seed) % 29) as f32 * 0.01 - 0.14
            }
        })
        .collect()
}

/// Writes a complete named F32 artifact, including dormant experts and MTP parameters.
fn optional_projection_checkpoint(root: &Path) -> NemotronHCheckpointContract {
    let contract = optional_projection_config()
        .checkpoint_contract(NemotronHCheckpointOptions::default())
        .unwrap();
    let mut header = Map::new();
    let mut payload = Vec::new();
    for operation in contract.mappings().operations() {
        let TensorOperation::Map { source, target, .. } = operation else {
            panic!("the untied miniature must contain only direct mappings");
        };
        let expected = contract
            .expected_tensors()
            .iter()
            .find(|tensor| &tensor.name == target)
            .unwrap();
        let start = payload.len();
        for value in
            optional_projection_values(target, expected.shape.iter().product::<u64>() as usize)
        {
            payload.extend_from_slice(&value.to_le_bytes());
        }
        assert!(
            header
                .insert(
                    source.clone(),
                    json!({
                        "dtype": "F32", "shape": expected.shape,
                        "data_offsets": [start, payload.len()]
                    })
                )
                .is_none()
        );
    }
    let mut header = serde_json::to_vec(&header).unwrap();
    while !header.len().is_multiple_of(8) {
        header.push(b' ');
    }
    let mut file = File::create(root.join("model.safetensors")).unwrap();
    file.write_all(&(header.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(&header).unwrap();
    file.write_all(&payload).unwrap();
    contract
}

/// Loads all optional parameters through the same public two-pass entry point as users.
fn optional_projection_model() -> NemotronHForCausalLm<TestBackend> {
    let temp = TempDir::new().unwrap();
    let contract = optional_projection_checkpoint(temp.path());
    let mut model =
        NemotronHForCausalLm::init(optional_projection_config(), &Default::default()).unwrap();
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
    assert_eq!(outcome.stats.aliases_applied, 0);
    model
}

/// Inspects initialization without materializing a missing parameter as a side effect.
struct OptionalParameterAudit(
    /// Count of visited parameters whose payloads were already initialized.
    usize,
);

impl ModuleVisitor<TestBackend> for OptionalParameterAudit {
    /// Counts only parameters whose payload was already installed by the loader.
    fn visit_float<const D: usize>(&mut self, parameter: &Param<Tensor<TestBackend, D>>) {
        assert!(parameter.is_initialized());
        self.0 += 1;
    }
}

/// Checks checkpoint `[out, in]` values against Burn's installed column layout and bias.
fn assert_optional_linear(linear: &Linear<TestBackend>, target: &str, input: usize, output: usize) {
    let weight = linear.weight.val();
    assert_eq!(weight.dims(), [input, output]);
    assert_eq!(
        weight.transpose().to_data().to_vec::<f32>().unwrap(),
        optional_projection_values(&format!("{target}.weight"), input * output)
    );
    let bias = linear.bias.as_ref().unwrap().val();
    assert_eq!(bias.dims(), [output]);
    assert_eq!(
        bias.to_data().to_vec::<f32>().unwrap(),
        optional_projection_values(&format!("{target}.bias"), output)
    );
}

/// Optional biases and rectangular latent maps reach base and MTP fields exactly.
#[test]
fn optional_projections_load_exact_bias_and_latent_payloads() {
    let model = optional_projection_model();
    let contract = model
        .config()
        .checkpoint_contract(NemotronHCheckpointOptions::default())
        .unwrap();
    for (source, target) in [
        (
            "backbone.layers.0.mixer.in_proj.bias",
            "model.layers.0.mamba.in_proj.bias",
        ),
        (
            "backbone.layers.1.mixer.q_proj.bias",
            "model.layers.1.attention.q_proj.bias",
        ),
        (
            "backbone.layers.2.mixer.up_proj.bias",
            "model.layers.2.mlp.up_proj.bias",
        ),
        (
            "backbone.layers.3.mixer.fc1_latent_proj.weight",
            "model.layers.3.moe.fc1_latent_proj.weight",
        ),
        (
            "mtp.layers.0.mixer.o_proj.bias",
            "next_token_head.attention.o_proj.bias",
        ),
        (
            "mtp.layers.1.mixer.fc2_latent_proj.bias",
            "next_token_head.moe.fc2_latent_proj.bias",
        ),
    ] {
        assert!(
            contract
                .mappings()
                .operations()
                .iter()
                .any(|operation| matches!(operation,
            TensorOperation::Map { source: actual_source, target: actual_target, .. }
                if actual_source == source && actual_target == target))
        );
    }
    let mut audit = OptionalParameterAudit(0);
    model.visit(&mut audit);
    assert_eq!(audit.0, contract.expected_tensors().len());
    let mamba = model.model.layers[0].mamba.as_ref().unwrap();
    assert_optional_linear(&mamba.in_proj, "model.layers.0.mamba.in_proj", 8, 32);
    assert_optional_linear(&mamba.out_proj, "model.layers.0.mamba.out_proj", 8, 8);
    let dense = model.model.layers[2].mlp.as_ref().unwrap();
    assert_optional_linear(&dense.up_proj, "model.layers.2.mlp.up_proj", 8, 12);
    assert_optional_linear(&dense.down_proj, "model.layers.2.mlp.down_proj", 12, 8);
    let mtp = model.next_token_head.as_ref().unwrap();
    for (attention, prefix) in [
        (
            model.model.layers[1].attention.as_ref().unwrap(),
            "model.layers.1.attention",
        ),
        (&mtp.attention, "next_token_head.attention"),
    ] {
        for (projection, name, output) in [
            (&attention.q_proj, "q_proj", 8),
            (&attention.k_proj, "k_proj", 4),
            (&attention.v_proj, "v_proj", 4),
            (&attention.o_proj, "o_proj", 8),
        ] {
            assert_optional_linear(projection, &format!("{prefix}.{name}"), 8, output);
        }
    }
    for (moe, prefix) in [
        (
            model.model.layers[3].moe.as_ref().unwrap(),
            "model.layers.3.moe",
        ),
        (&mtp.moe, "next_token_head.moe"),
    ] {
        assert_optional_linear(
            moe.fc1_latent_proj.as_ref().unwrap(),
            &format!("{prefix}.fc1_latent_proj"),
            8,
            4,
        );
        assert_optional_linear(
            moe.fc2_latent_proj.as_ref().unwrap(),
            &format!("{prefix}.fc2_latent_proj"),
            4,
            8,
        );
        for (index, expert) in moe.routed.iter().enumerate() {
            assert_optional_linear(
                &expert.up_proj,
                &format!("{prefix}.routed.{index}.up_proj"),
                4,
                6,
            );
            assert_optional_linear(
                &expert.down_proj,
                &format!("{prefix}.routed.{index}.down_proj"),
                6,
                4,
            );
        }
        assert_optional_linear(
            &moe.shared.up_proj,
            &format!("{prefix}.shared.up_proj"),
            8,
            8,
        );
        assert_optional_linear(
            &moe.shared.down_proj,
            &format!("{prefix}.shared.down_proj"),
            8,
            8,
        );
    }
}

/// Evaluates checkpoint row-major matrices in f64 without Burn projection or routing helpers.
fn optional_host_linear(input: &[f64], target: &str, output: usize, biased: bool) -> Vec<f64> {
    let weight = optional_projection_values(&format!("{target}.weight"), input.len() * output);
    let bias = optional_projection_values(&format!("{target}.bias"), output);
    weight
        .chunks_exact(input.len())
        .enumerate()
        .map(|(row, weights)| {
            weights
                .iter()
                .zip(input)
                .map(|(&weight, &value)| f64::from(weight) * value)
                .sum::<f64>()
                + if biased { f64::from(bias[row]) } else { 0.0 }
        })
        .collect()
}

/// Computes a biased ReLU-squared expert as `W_down max(W_up x+b_up,0)^2+b_down`.
fn optional_host_expert(input: &[f64], target: &str, intermediate: usize) -> Vec<f64> {
    let hidden = optional_host_linear(input, &format!("{target}.up_proj"), intermediate, true)
        .into_iter()
        .map(|value| value.max(0.0).powi(2))
        .collect::<Vec<_>>();
    optional_host_linear(&hidden, &format!("{target}.down_proj"), input.len(), true)
}

/// Low-rank routing projects before experts and restores width after the weighted sum.
#[test]
fn optional_projections_latent_moe_matches_independent_scalar_oracle() {
    let model = optional_projection_model();
    let moe = model.model.layers[3].moe.as_ref().unwrap();
    let rows = [
        [0.2_f32, -0.3, 0.4, 0.1, -0.2, 0.5, 0.3, -0.1],
        [-0.4, 0.1, 0.2, 0.3, 0.5, -0.2, 0.1, 0.4],
    ];
    let output = moe
        .forward(Tensor::from_data(
            TensorData::new(rows.concat(), [1, 2, 8]),
            &Default::default(),
        ))
        .unwrap();
    assert_eq!(output.hidden_state.dims(), [1, 2, 8]);
    let actual = output.hidden_state.to_data().to_vec::<f32>().unwrap();
    let selected = output.selected_experts.to_data().to_vec::<i32>().unwrap();
    let selected_weights = output.selected_weights.to_data().to_vec::<f32>().unwrap();
    let mut expected = Vec::new();
    for (token, row) in rows.iter().enumerate() {
        let input = row.iter().copied().map(f64::from).collect::<Vec<_>>();
        let logits = optional_host_linear(&input, "model.layers.3.moe.router", 4, false);
        let scores = logits
            .iter()
            .map(|&value| 1.0 / (1.0 + (-value).exp()))
            .collect::<Vec<_>>();
        let mut order = [0, 1, 2, 3];
        order.sort_by(|&left, &right| {
            (scores[right] + right as f64 * 0.25).total_cmp(&(scores[left] + left as f64 * 0.25))
        });
        assert_eq!(order, [3, 2, 1, 0]);
        assert!(scores[2] + 0.5 - (scores[1] + 0.25) > 0.1);
        assert_eq!(&selected[token * 2..token * 2 + 2], &[3, 2]);
        let latent = optional_host_linear(&input, "model.layers.3.moe.fc1_latent_proj", 4, true);
        let mut sum = [0.0; 4];
        for (slot, expert) in [3, 2].into_iter().enumerate() {
            let weight = scores[expert] / (scores[3] + scores[2]) * 1.7;
            assert!((f64::from(selected_weights[token * 2 + slot]) - weight).abs() <= 1e-4);
            let contribution =
                optional_host_expert(&latent, &format!("model.layers.3.moe.routed.{expert}"), 6);
            for (sum, value) in sum.iter_mut().zip(contribution) {
                *sum += weight * value;
            }
        }
        let restored = optional_host_linear(&sum, "model.layers.3.moe.fc2_latent_proj", 8, true);
        let shared = optional_host_expert(&input, "model.layers.3.moe.shared", 8);
        expected.extend(
            restored
                .into_iter()
                .zip(shared)
                .map(|(routed, shared)| routed + shared),
        );
    }
    assert_eq!(actual.len(), expected.len());
    assert!(expected.iter().any(|value| value.abs() > 1e-3));
    for (actual, expected) in actual.into_iter().zip(expected) {
        assert!(actual.is_finite() && expected.is_finite());
        assert!(
            (f64::from(actual) - expected).abs() <= 1e-4,
            "{actual} != {expected}"
        );
    }
    assert_eq!(output.expert_token_counts, vec![0, 0, 2, 2]);
    assert_eq!(output.evaluated_experts, 2);
}

/// Compares finite nonzero logits without relative tolerances that hide small outputs.
fn optional_projection_close(actual: Tensor<TestBackend, 3>, expected: Tensor<TestBackend, 3>) {
    assert_eq!(actual.dims(), expected.dims());
    let actual = actual.to_data().to_vec::<f32>().unwrap();
    let expected = expected.to_data().to_vec::<f32>().unwrap();
    assert!(expected.iter().any(|value| value.abs() > 1e-3));
    for (actual, expected) in actual.into_iter().zip(expected) {
        assert!(actual.is_finite() && expected.is_finite());
        assert!((actual - expected).abs() <= 1e-4, "{actual} != {expected}");
    }
}

/// A loaded biased/latent hybrid preserves cached continuation and has an executable MTP head.
#[test]
fn optional_projections_loaded_cache_and_mtp_outputs_are_nonzero() {
    let model = optional_projection_model();
    let device = Default::default();
    let options = ForwardOptions {
        logits: LogitsSelection::All,
        hidden_states: true,
        ..Default::default()
    };
    let reference = model
        .forward(
            CausalLmInput {
                token_ids: Tensor::from_ints([[1, 2, 3, 4]], &device),
                token_mask: None,
                position_ids: None,
            },
            &options,
        )
        .unwrap();
    let mut state = model
        .new_decode_state(CacheConfig::new(1, 8).unwrap(), 1, &device)
        .unwrap();
    assert_eq!(
        (
            state.attention_layer_count(),
            state.mamba_layer_count(),
            state.mtp_layer_count()
        ),
        (1, 1, 0)
    );
    let prefill = model
        .prefill(
            CausalLmInput {
                token_ids: Tensor::from_ints([[1, 2]], &device),
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &options,
        )
        .unwrap();
    optional_projection_close(prefill.logits, reference.logits.clone().narrow(1, 0, 2));
    for token in [3, 4] {
        let output = model
            .decode(
                DecodeInput {
                    token_ids: Tensor::from_ints([[token]], &device),
                    token_mask: None,
                    position_ids: None,
                },
                &mut state,
                &options,
            )
            .unwrap();
        optional_projection_close(
            output.logits,
            reference.logits.clone().narrow(1, token as usize - 1, 1),
        );
    }
    assert_eq!(state.lengths(), &[4]);
    let bank = state.tensor_state().main();
    assert_eq!(
        bank.native_kv(0).unwrap().keys(0).unwrap().dims(),
        [1, 1, 8, 4]
    );
    assert_eq!(
        bank.recurrent(0).unwrap().matrix(0).unwrap().dims(),
        [1, 4, 2, 3]
    );
    assert_eq!(
        bank.recurrent(0).unwrap().convolution(0).unwrap().dims(),
        [1, 20, 2]
    );
    let mtp = model
        .predict_next_token_logits(
            Tensor::from_ints([[2, 3, 4, 5]], &device),
            reference.final_hidden_state.unwrap(),
        )
        .unwrap();
    assert_eq!(mtp.dims(), [1, 4, 16]);
    let values = mtp.to_data().to_vec::<f32>().unwrap();
    assert!(values.iter().all(|value| value.is_finite()));
    assert!(values.iter().any(|value| value.abs() > 1e-3));
}

/// Writes `target` with a one-element payload and returns the sink's typed error.
fn sink_error(model: &mut NemotronHForCausalLm<TestBackend>, target: &str) -> NemotronHLoadError {
    model
        .write_tensor(target, TensorData::from([0.0_f32]))
        .expect_err(target)
}

/// Every malformed, out-of-range, unknown, or schedule-absent target names its cause.
#[test]
fn sink_dispatch_rejects_every_malformed_or_absent_target_with_its_cause() {
    let device = Default::default();
    let mut model = NemotronHForCausalLm::<TestBackend>::init(tiny_config(false), &device).unwrap();
    let unknown = [
        "model.unknown.weight",
        "model.layers.0.unknown.weight",
        "model.layers.0.mamba.unknown",
        "model.layers.1.attention.unknown",
        "model.layers.2.mlp.unknown",
        "model.layers.3.moe.unknown",
        "model.layers.3.moe.shared.unknown",
        "model.layers.3.moe.routed.0.unknown",
        "model.layers.3.moe.routed.0.",
        "next_token_head.unknown",
        "next_token_head.attention.unknown",
        "next_token_head.moe.unknown",
    ];
    for target in unknown {
        assert_eq!(
            sink_error(&mut model, target),
            NemotronHLoadError::UnknownTarget {
                target: target.into()
            }
        );
    }
    let invalid = [
        ("model.layers.4.norm.weight", "block 4 is outside 0..4"),
        (
            "model.layers.0",
            "expected `model.layers.<index>.<parameter>`",
        ),
        ("model.layers.0.", "parameter suffix must not be empty"),
        (
            "model.layers.-1.norm.weight",
            "`-1` is not a decimal usize: invalid digit found in string",
        ),
        (
            "model.layers.3.moe.routed.0",
            "expected `routed.<index>.<parameter>`",
        ),
        (
            "model.layers.3.moe.routed.x.up_proj.weight",
            "invalid expert index `x`: invalid digit found in string",
        ),
        (
            "model.layers.3.moe.routed.4.up_proj.weight",
            "expert 4 is outside 0..4",
        ),
        (
            "next_token_head.moe.routed.9.down_proj.weight",
            "expert 9 is outside 0..4",
        ),
    ];
    for (target, reason) in invalid {
        assert_eq!(
            sink_error(&mut model, target),
            NemotronHLoadError::InvalidIndex {
                target: target.into(),
                reason: reason.into()
            }
        );
    }
    let missing = [
        (
            "model.layers.1.mamba.in_proj.weight",
            "Mamba2 mixer selected by the block schedule",
        ),
        (
            "model.layers.0.attention.q_proj.weight",
            "attention mixer selected by the block schedule",
        ),
        (
            "model.layers.3.mlp.up_proj.weight",
            "dense MLP selected by the block schedule",
        ),
        (
            "model.layers.2.moe.router.weight",
            "routed MoE selected by the block schedule",
        ),
        (
            "model.layers.0.mamba.in_proj.bias",
            "configured linear bias",
        ),
        (
            "model.layers.0.mamba.out_proj.bias",
            "configured linear bias",
        ),
        (
            "model.layers.1.attention.o_proj.bias",
            "configured linear bias",
        ),
        ("model.layers.2.mlp.up_proj.bias", "configured linear bias"),
        (
            "model.layers.3.moe.fc1_latent_proj.weight",
            "first routed-expert latent projection",
        ),
        (
            "model.layers.3.moe.fc2_latent_proj.weight",
            "second routed-expert latent projection",
        ),
    ];
    for (target, component) in missing {
        assert_eq!(
            sink_error(&mut model, target),
            NemotronHLoadError::MissingComponent {
                target: target.into(),
                component
            }
        );
    }
    // No rejected target initialized a parameter as a side effect.
    assert!(!model.model.embed_tokens.weight.is_initialized());
    assert!(!model.model.layers[0].norm.weight.is_initialized());
}

/// Optional components absent from a configuration are reported, not silently skipped.
#[test]
fn sink_rejects_components_the_configuration_omits() {
    let device = Default::default();
    // A tied head, no MTP stage, and an unbiased convolution.
    let config = NemotronHConfig {
        num_nextn_predict_layers: 0,
        use_conv_bias: false,
        ..tiny_config(true)
    };
    let mut model = NemotronHForCausalLm::<TestBackend>::init(config, &device).unwrap();
    for (target, component) in [
        (
            "lm_head.weight",
            "independent LM head for an untied checkpoint",
        ),
        (
            "next_token_head.eh_proj.weight",
            "checkpoint-declared next-token head",
        ),
        (
            "model.layers.0.mamba.conv1d.bias",
            "Mamba depthwise-convolution bias",
        ),
    ] {
        assert_eq!(
            sink_error(&mut model, target),
            NemotronHLoadError::MissingComponent {
                target: target.into(),
                component
            }
        );
    }
    // A latent projection exposes only `weight` and `bias`.
    let mut latent =
        NemotronHForCausalLm::<TestBackend>::init(optional_projection_config(), &device).unwrap();
    for target in [
        "model.layers.3.moe.fc1_latent_proj.scale",
        "next_token_head.moe.fc2_latent_proj.scale",
    ] {
        assert_eq!(
            sink_error(&mut latent, target),
            NemotronHLoadError::UnknownTarget {
                target: target.into()
            }
        );
    }
}

/// Only an installed embedding may alias into a tied head; diagnostics name the cause.
#[test]
fn tied_alias_requires_the_tied_topology_and_an_installed_owner() {
    let device = Default::default();
    let (head, embedding) = ("lm_head.weight", "model.embed_tokens.weight");
    let alias_error = |target: &str, source: &str, reason: &str| NemotronHLoadError::InvalidAlias {
        target: target.into(),
        source_target: source.into(),
        reason: reason.into(),
    };
    let mut untied =
        NemotronHForCausalLm::<TestBackend>::init(tiny_config(false), &device).unwrap();
    assert_eq!(
        untied.alias_tensor(head, "model.norm.weight").unwrap_err(),
        alias_error(
            head,
            "model.norm.weight",
            "only the tied embedding-to-LM-head alias is supported"
        )
    );
    assert_eq!(
        untied.alias_tensor(head, embedding).unwrap_err(),
        alias_error(
            head,
            embedding,
            "the configured model declares an independent LM head"
        )
    );
    let mut tied = NemotronHForCausalLm::<TestBackend>::init(tiny_config(true), &device).unwrap();
    let error = tied.alias_tensor(head, embedding).unwrap_err();
    assert_eq!(
        error,
        alias_error(
            head,
            embedding,
            "the embedding storage owner has not been installed"
        )
    );
    assert_eq!(
        error.to_string(),
        "invalid Nemotron-H alias `lm_head.weight` -> `model.embed_tokens.weight`: \
         the embedding storage owner has not been installed"
    );
    tied.write_tensor(embedding, TensorData::zeros::<f32, _>([16, 8]))
        .unwrap();
    tied.alias_tensor(head, embedding).unwrap();
    // A tied configuration whose graph regained an independent head is refused.
    let mut regrown = untied.clone();
    regrown.model.config.tie_word_embeddings = true;
    assert_eq!(
        regrown.alias_tensor(head, embedding).unwrap_err(),
        alias_error(
            head,
            embedding,
            "the configured model declares an independent LM head"
        )
    );
}

/// Sink diagnostics render the target, component, and parameter cause.
#[test]
fn sink_error_messages_name_target_and_cause() {
    let device = Default::default();
    let mut model = NemotronHForCausalLm::<TestBackend>::init(tiny_config(false), &device).unwrap();
    assert_eq!(
        sink_error(&mut model, "model.other").to_string(),
        "unknown Nemotron-H parameter target `model.other`"
    );
    assert_eq!(
        sink_error(&mut model, "model.layers.9.norm.weight").to_string(),
        "invalid Nemotron-H target `model.layers.9.norm.weight`: block 9 is outside 0..4"
    );
    assert_eq!(
        sink_error(&mut model, "model.layers.0.mamba.in_proj.bias").to_string(),
        "Nemotron-H target `model.layers.0.mamba.in_proj.bias` requires missing component \
         `configured linear bias`"
    );
    let error = sink_error(&mut model, "model.norm.weight");
    assert!(
        matches!(&error, NemotronHLoadError::Parameter { target, .. } if target == "model.norm.weight")
    );
    assert!(
        error
            .to_string()
            .starts_with("cannot install Nemotron-H parameter `model.norm.weight`: "),
        "{error}"
    );
    assert!(std::error::Error::source(&error).is_some());
}

/// Configuration, planning, and pass-two failures are distinct, and only pass two reports progress.
#[test]
fn checkpoint_load_errors_distinguish_configuration_planning_and_execution() {
    let device = Default::default();
    let config = tiny_config(false);
    let contract = config
        .checkpoint_contract(NemotronHCheckpointOptions::default())
        .unwrap();
    let temp = TempDir::new().unwrap();
    write_checkpoint(temp.path(), &contract, |_| {});

    // A configuration edited after initialization cannot describe a checkpoint.
    let mut model = NemotronHForCausalLm::<TestBackend>::init(config.clone(), &device).unwrap();
    model.model.config.vocab_size = 0;
    let error = model
        .load_checkpoint(temp.path(), &LoadOptions::default())
        .unwrap_err();
    assert!(
        matches!(&error, NemotronHCheckpointLoadError::Config(config)
        if config.field == "vocab_size")
    );
    assert!(error.execution_failure().is_none());
    assert_eq!(error.to_string(), "nemotron_h.vocab_size: must be nonzero");

    // A missing root fails discovery before any write.
    let mut model = NemotronHForCausalLm::<TestBackend>::init(config.clone(), &device).unwrap();
    let error = model
        .load_checkpoint(temp.path().join("absent"), &LoadOptions::default())
        .unwrap_err();
    assert!(
        matches!(&error, NemotronHCheckpointLoadError::Checkpoint(CheckpointError::Io {
            operation, path, ..
        }) if *operation == "inspect checkpoint root" && path.ends_with("absent")),
        "{error}"
    );
    assert!(error.execution_failure().is_none());

    // A graph whose head was removed after planning fails in pass two with progress.
    let mut model = NemotronHForCausalLm::<TestBackend>::init(config, &device).unwrap();
    model.lm_head = None;
    let error = model
        .load_checkpoint(temp.path(), &LoadOptions::default())
        .unwrap_err();
    let failure = error
        .execution_failure()
        .expect("pass two reached the sink");
    assert_eq!(failure.failed_target.as_deref(), Some("lm_head.weight"));
    assert!(matches!(
        &failure.cause,
        crate::checkpoint::ExecutionFailureCause::Sink(NemotronHLoadError::MissingComponent {
            component: "independent LM head for an untied checkpoint",
            ..
        })
    ));
    assert!(
        !failure
            .report
            .applied
            .iter()
            .any(|name| name == "lm_head.weight")
    );
}

/// `load_checkpoint_with` applies explicit provenance; the 0.1.0 name forwards to it unchanged.
///
/// Provenance that names no reviewed NVIDIA artifact selects the generic
/// strict policy, so all three entry points produce the same audited report.
#[test]
#[allow(deprecated)]
fn load_checkpoint_with_is_the_options_entry_point_and_the_former_name_forwards() {
    use crate::models::nemotron_h::NemotronHCheckpointProvenance;

    let device = Default::default();
    let config = tiny_config(false);
    let contract = config
        .checkpoint_contract(NemotronHCheckpointOptions::default())
        .unwrap();
    let temp = TempDir::new().unwrap();
    write_checkpoint(temp.path(), &contract, |_| {});
    let explicit = NemotronHCheckpointOptions::default().with_provenance(
        NemotronHCheckpointProvenance::new("example/tiny-nemotron", "fixture-revision"),
    );
    let load = |entry: usize| {
        let mut model = NemotronHForCausalLm::<TestBackend>::init(config.clone(), &device).unwrap();
        match entry {
            0 => model.load_checkpoint(temp.path(), &LoadOptions::default()),
            1 => model.load_checkpoint_with(temp.path(), &LoadOptions::default(), explicit.clone()),
            _ => model.load_checkpoint_with_architecture_options(
                temp.path(),
                &LoadOptions::default(),
                explicit.clone(),
            ),
        }
        .unwrap()
        .report
    };
    let default = load(0);
    assert!(default.is_strictly_valid());
    assert_eq!(load(1), default);
    assert_eq!(load(2), default);
}

/// Packed sidecar payloads install exactly the artifact, never a requantized approximation.
///
/// The sink's one installer must keep a packed column-layout weight exact on
/// every backend (as exact dequantized values on Burn Flex, whose quantized
/// transpose requantizes) and leave float payloads unchanged, as the
/// certified Qwen3-MoE and Muse sinks do.
#[test]
fn packed_payloads_install_the_artifact_exactly() {
    crate::checkpoint::packed_sink_support::assert_install_keeps_packed_weights_exact(
        |parameter, data| install("model.layers.0.mixer.out_proj.weight", parameter, data),
    );
}
