//! Strict loading: synthetic dense and MXFP4 checkpoints install every
//! parameter exactly (stacked experts split and de-interleaved), format
//! mismatches fail during planning, and the sink rejects targets the graph
//! does not own.

use std::path::Path;

use burn::tensor::{Tensor, TensorData};
use safetensors::{Dtype, tensor::TensorView};

use super::*;
use crate::{
    CausalLmInput, ForwardOptions,
    checkpoint::DTypePolicy,
    models::gpt_oss::{
        GptOssConfig,
        model::test_support::{B, VOCABULARY, host, ints, tiny_config},
    },
};

/// One synthetic checkpoint tensor: name, SafeTensors dtype, shape, and LE bytes.
type Fixture = (String, Dtype, Vec<usize>, Vec<u8>);

/// Returns `count` distinct, exactly representable F32 values derived from `salt`.
fn values(count: usize, salt: usize) -> Vec<f32> {
    (0..count)
        .map(|index| ((index * 37 + salt * 101) % 997) as f32 / 512.0 - 0.9)
        .collect()
}

/// Encodes F32 values little-endian.
fn le_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

/// Writes `config` and `tensors` as a single-file checkpoint.
fn write(root: &Path, config: &GptOssConfig, tensors: &[Fixture]) {
    std::fs::write(
        root.join("config.json"),
        serde_json::to_string(config).unwrap(),
    )
    .unwrap();
    let views: Vec<(String, TensorView<'_>)> = tensors
        .iter()
        .map(|(name, dtype, shape, bytes)| {
            (
                name.clone(),
                TensorView::new(*dtype, shape.clone(), bytes).unwrap(),
            )
        })
        .collect();
    std::fs::write(
        root.join("model.safetensors"),
        safetensors::serialize(views, None).unwrap(),
    )
    .unwrap();
}

/// Builds an F32 tensor for every target of `config`'s dense contract, with distinct values.
fn dense_tensors(config: &GptOssConfig) -> Vec<Fixture> {
    let contract = config
        .checkpoint_contract(GptOssCheckpointOptions::default())
        .unwrap();
    contract
        .expected_tensors()
        .iter()
        .filter(|tensor| !(config.tie_word_embeddings && tensor.name == "lm_head.weight"))
        .enumerate()
        .map(|(salt, tensor)| {
            let shape: Vec<usize> = tensor.shape.iter().map(|&axis| axis as usize).collect();
            let count = shape.iter().product();
            (
                tensor.name.clone(),
                Dtype::F32,
                shape,
                le_bytes(&values(count, salt)),
            )
        })
        .collect()
}

/// Returns the F32 values of fixture `name`.
fn fixture_values(tensors: &[Fixture], name: &str) -> Vec<f32> {
    let (_, _, _, bytes) = tensors
        .iter()
        .find(|(fixture, ..)| fixture == name)
        .unwrap_or_else(|| panic!("no fixture `{name}`"));
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

/// Allocates a fresh graph for `config`.
fn graph(config: &GptOssConfig) -> GptOssForCausalLm<B> {
    GptOssForCausalLm::init(config.clone(), &Default::default()).unwrap()
}

/// Reads a parameter as host F32 values.
fn param<const D: usize>(tensor: Tensor<B, D>) -> Vec<f32> {
    host(tensor)
}

/// Asserts expert `expert` of `layer` holds the de-interleaved stacked fixtures.
fn assert_expert(
    model: &GptOssForCausalLm<B>,
    layer: usize,
    expert: usize,
    gate_up: &[f32],
    gate_up_bias: &[f32],
    down: &[f32],
    down_bias: &[f32],
) {
    let (hidden, width) = (16, 12);
    let mlp = &model.model.layers[layer].mlp.experts[expert];
    let gate = param(mlp.gate_proj.weight.val());
    let up = param(mlp.up_proj.weight.val());
    for row in 0..hidden {
        for column in 0..width {
            let base = (expert * hidden + row) * 2 * width + 2 * column;
            assert_eq!(
                gate[row * width + column],
                gate_up[base],
                "gate[{row}, {column}]"
            );
            assert_eq!(
                up[row * width + column],
                gate_up[base + 1],
                "up[{row}, {column}]"
            );
        }
    }
    let gate_bias = param(mlp.gate_proj.bias.as_ref().unwrap().val());
    let up_bias = param(mlp.up_proj.bias.as_ref().unwrap().val());
    for column in 0..width {
        assert_eq!(
            gate_bias[column],
            gate_up_bias[expert * 2 * width + 2 * column]
        );
        assert_eq!(
            up_bias[column],
            gate_up_bias[expert * 2 * width + 2 * column + 1]
        );
    }
    let span = width * hidden;
    assert_eq!(
        param(mlp.down_proj.weight.val()),
        down[expert * span..(expert + 1) * span]
    );
    assert_eq!(
        param(mlp.down_proj.bias.as_ref().unwrap().val()),
        down_bias[expert * hidden..(expert + 1) * hidden]
    );
}

/// A dense checkpoint installs every parameter exactly, splitting stacked experts.
#[test]
fn dense_checkpoints_install_every_parameter_exactly() {
    let config = tiny_config();
    let tensors = dense_tensors(&config);
    let directory = tempfile::tempdir().unwrap();
    write(directory.path(), &config, &tensors);
    let mut model = graph(&config);
    let outcome = model
        .load_checkpoint(directory.path(), &LoadOptions::default())
        .unwrap();
    assert!(outcome.report.is_strictly_valid());
    assert_eq!(
        param(model.model.embed_tokens.weight.val()),
        fixture_values(&tensors, "model.embed_tokens.weight")
    );
    // Column-layout projections hold the transposed `[out, in]` checkpoint matrix.
    let query = model.model.layers[1]
        .self_attn
        .q_proj
        .weight
        .val()
        .transpose();
    assert_eq!(
        param(query),
        fixture_values(&tensors, "model.layers.1.self_attn.q_proj.weight")
    );
    assert_eq!(
        param(
            model.model.layers[2]
                .self_attn
                .sinks
                .as_ref()
                .unwrap()
                .val()
        ),
        fixture_values(&tensors, "model.layers.2.self_attn.sinks")
    );
    assert_eq!(
        param(
            model.model.layers[0]
                .mlp
                .router
                .bias
                .as_ref()
                .unwrap()
                .val()
        ),
        fixture_values(&tensors, "model.layers.0.mlp.router.bias")
    );
    for layer in 0..3 {
        let prefix = format!("model.layers.{layer}.mlp.experts");
        let gate_up = fixture_values(&tensors, &format!("{prefix}.gate_up_proj"));
        let gate_up_bias = fixture_values(&tensors, &format!("{prefix}.gate_up_proj_bias"));
        let down = fixture_values(&tensors, &format!("{prefix}.down_proj"));
        let down_bias = fixture_values(&tensors, &format!("{prefix}.down_proj_bias"));
        for expert in 0..4 {
            assert_expert(
                &model,
                layer,
                expert,
                &gate_up,
                &gate_up_bias,
                &down,
                &down_bias,
            );
        }
    }
    let head = model.lm_head.as_ref().unwrap().weight.val().transpose();
    assert_eq!(param(head), fixture_values(&tensors, "lm_head.weight"));
    // The loaded model runs.
    let output = model
        .forward(
            CausalLmInput {
                token_ids: ints([[1, 2, 3]]),
                token_mask: None,
                position_ids: None,
            },
            &ForwardOptions::default(),
        )
        .unwrap();
    assert_eq!(output.logits.dims(), [1, 1, VOCABULARY]);
}

/// Loads every floating tensor, including nominally BF16 decoded experts, as F32.
fn f32_options() -> LoadOptions {
    LoadOptions {
        dtype_policy: DTypePolicy::Cast(crate::checkpoint::TensorDType::F32),
        ..LoadOptions::default()
    }
}

/// Returns a config whose expert widths are MXFP4-blockable (multiples of 32).
fn blockable_config() -> GptOssConfig {
    GptOssConfig {
        hidden_size: 32,
        intermediate_size: 32,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dim: 8,
        num_hidden_layers: 2,
        layer_types: GptOssConfig::default_layer_types(2),
        quantization_config: Some(serde_json::json!({"quant_method": "mxfp4"})),
        ..tiny_config()
    }
}

/// Packs seeded MXFP4 blocks/scales for `[experts, rows, columns]` decoded rows.
fn mxfp4_pair(experts: usize, rows: usize, columns: usize, salt: u8) -> (Vec<u8>, Vec<u8>) {
    let groups = columns / 32;
    let blocks = (0..experts * rows * groups * 16)
        .map(|index| (index as u8).wrapping_mul(29).wrapping_add(salt))
        .collect();
    let scales = (0..experts * rows * groups)
        .map(|index| 120 + ((index + usize::from(salt)) % 12) as u8)
        .collect();
    (blocks, scales)
}

/// Decodes one MXFP4 value of row `row`, column `column` independently of the loader.
fn decoded(
    blocks: &[u8],
    scales: &[u8],
    rows: usize,
    columns: usize,
    expert: usize,
    row: usize,
    column: usize,
) -> f32 {
    /// FP4 E2M1 values by code: sign bit 3, then the eight magnitudes.
    const FP4: [f32; 16] = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ];
    let groups = columns / 32;
    let block = (expert * rows + row) * groups + column / 32;
    let byte = blocks[block * 16 + (column % 32) / 2];
    let code = if column.is_multiple_of(2) {
        byte & 15
    } else {
        byte >> 4
    };
    FP4[usize::from(code)] * 2.0_f32.powi(i32::from(scales[block]) - 127)
}

/// MXFP4 expert tensors decode exactly into every expert's projections.
#[test]
fn mxfp4_checkpoints_decode_experts_exactly() {
    let config = blockable_config();
    let (experts, hidden, width) = (4, 32, 32);
    let mut tensors: Vec<Fixture> = dense_tensors(&config)
        .into_iter()
        .filter(|(name, ..)| {
            !name.ends_with("experts.gate_up_proj") && !name.ends_with("experts.down_proj")
        })
        .collect();
    let mut pairs = Vec::new();
    for layer in 0..2 {
        let prefix = format!("model.layers.{layer}.mlp.experts");
        // gate_up: blocks [E, 2f, d/32, 16] decode to [E, d, 2f]; down: [E, d, f/32, 16] to [E, f, d].
        for (name, rows, columns, salt) in [
            ("gate_up_proj", 2 * width, hidden, 3 + layer as u8),
            ("down_proj", hidden, width, 11 + layer as u8),
        ] {
            let (blocks, scales) = mxfp4_pair(experts, rows, columns, salt);
            tensors.push((
                format!("{prefix}.{name}_blocks"),
                Dtype::U8,
                vec![experts, rows, columns / 32, 16],
                blocks.clone(),
            ));
            tensors.push((
                format!("{prefix}.{name}_scales"),
                Dtype::U8,
                vec![experts, rows, columns / 32],
                scales.clone(),
            ));
            pairs.push((layer, name, rows, columns, blocks, scales));
        }
    }
    let directory = tempfile::tempdir().unwrap();
    write(directory.path(), &config, &tensors);
    let mut model = graph(&config);
    // Decoded experts are nominally BF16; the dense fixtures are F32.
    let outcome = model
        .load_checkpoint(directory.path(), &f32_options())
        .unwrap();
    assert!(
        outcome
            .report
            .transformed
            .iter()
            .any(|record| record.operation == "dequantize_mxfp4")
    );
    for (layer, name, rows, columns, blocks, scales) in &pairs {
        for expert in 0..experts {
            let mlp = &model.model.layers[*layer].mlp.experts[expert];
            match *name {
                "gate_up_proj" => {
                    // Decoded rows are output features 2j (gate) and 2j + 1 (up).
                    let gate = param(mlp.gate_proj.weight.val());
                    let up = param(mlp.up_proj.weight.val());
                    for input in 0..hidden {
                        for output in 0..width {
                            let value =
                                |row| decoded(blocks, scales, *rows, *columns, expert, row, input);
                            assert_eq!(gate[input * width + output], value(2 * output));
                            assert_eq!(up[input * width + output], value(2 * output + 1));
                        }
                    }
                }
                _ => {
                    let down = param(mlp.down_proj.weight.val());
                    for input in 0..width {
                        for output in 0..hidden {
                            assert_eq!(
                                down[input * hidden + output],
                                decoded(blocks, scales, *rows, *columns, expert, output, input)
                            );
                        }
                    }
                }
            }
        }
    }
}

/// A checkpoint whose expert format disagrees with the config fails during planning.
#[test]
fn format_mismatches_fail_during_planning() {
    let config = blockable_config();
    let dense = GptOssConfig {
        quantization_config: None,
        ..config.clone()
    };
    let directory = tempfile::tempdir().unwrap();
    // Dense tensors under an MXFP4 config: blocks and scales are missing.
    write(directory.path(), &config, &dense_tensors(&dense));
    let error = graph(&config)
        .load_checkpoint(directory.path(), &LoadOptions::default())
        .unwrap_err();
    assert!(
        matches!(error, GptOssCheckpointLoadError::Checkpoint(_)),
        "{error}"
    );
    assert!(error.execution_failure().is_none());
}

/// A tied checkpoint stores no head; the head aliases the embedding.
#[test]
fn tied_checkpoints_alias_the_embedding() {
    let config = GptOssConfig {
        tie_word_embeddings: true,
        ..tiny_config()
    };
    let tensors = dense_tensors(&config);
    assert!(tensors.iter().all(|(name, ..)| name != "lm_head.weight"));
    let directory = tempfile::tempdir().unwrap();
    write(directory.path(), &config, &tensors);
    let mut model = graph(&config);
    assert!(model.lm_head.is_none());
    let outcome = model
        .load_checkpoint(directory.path(), &LoadOptions::default())
        .unwrap();
    assert_eq!(outcome.report.aliased.len(), 1);
}

/// MXFP4 experts decode straight into a requested BF16 dtype, exactly.
#[test]
fn mxfp4_checkpoints_load_into_bf16_exactly() {
    let config = blockable_config();
    let mut tensors: Vec<Fixture> = dense_tensors(&config)
        .into_iter()
        .filter(|(name, ..)| {
            !name.ends_with("experts.gate_up_proj") && !name.ends_with("experts.down_proj")
        })
        .collect();
    for layer in 0..2 {
        for (name, rows, columns) in [("gate_up_proj", 64, 32), ("down_proj", 32, 32)] {
            let (blocks, scales) = mxfp4_pair(4, rows, columns, 5);
            let prefix = format!("model.layers.{layer}.mlp.experts.{name}");
            tensors.push((
                format!("{prefix}_blocks"),
                Dtype::U8,
                vec![4, rows, 1, 16],
                blocks,
            ));
            tensors.push((
                format!("{prefix}_scales"),
                Dtype::U8,
                vec![4, rows, 1],
                scales,
            ));
        }
    }
    let directory = tempfile::tempdir().unwrap();
    write(directory.path(), &config, &tensors);
    let options = LoadOptions {
        dtype_policy: DTypePolicy::Cast(crate::checkpoint::TensorDType::BF16),
        ..LoadOptions::default()
    };
    let mut bf16 = graph(&config);
    bf16.load_checkpoint(directory.path(), &options).unwrap();
    let mut f32 = graph(&config);
    f32.load_checkpoint(directory.path(), &f32_options())
        .unwrap();
    // Under the default policy the decoded experts keep their nominal BF16.
    let mut preserved = graph(&config);
    preserved
        .load_checkpoint(directory.path(), &LoadOptions::default())
        .unwrap();
    assert_eq!(
        preserved.model.layers[0].mlp.experts[0]
            .gate_proj
            .weight
            .val()
            .dtype(),
        burn::tensor::DType::BF16
    );
    let weight =
        |model: &GptOssForCausalLm<B>| model.model.layers[1].mlp.experts[2].down_proj.weight.val();
    assert_eq!(weight(&bf16).dtype(), burn::tensor::DType::BF16);
    // Every decoded value is exact in BF16, so widening it back loses nothing.
    assert_eq!(
        host(weight(&bf16).cast(burn::tensor::DType::F32)),
        host(weight(&f32))
    );
}

/// The sink rejects unknown, malformed, out-of-range, and unallocated targets.
#[test]
fn sink_rejects_targets_the_graph_does_not_own() {
    let config = GptOssConfig {
        attention_bias: false,
        ..tiny_config()
    };
    let mut model = graph(&config);
    let data = || TensorData::new(vec![0.0_f32; 16], [16]);
    for (target, expected) in [
        ("model.unknown", "unknown"),
        ("model.layers.x.input_layernorm.weight", "invalid"),
        ("model.layers.3.input_layernorm.weight", "invalid"),
        ("model.layers.0", "invalid"),
        ("model.layers.0.mlp.unknown", "unknown"),
        ("model.layers.0.self_attn.q_proj.bias", "missing"),
    ] {
        let error = model.write_tensor(target, data()).unwrap_err();
        let matches = match expected {
            "unknown" => matches!(error, GptOssLoadError::UnknownTarget { .. }),
            "invalid" => matches!(error, GptOssLoadError::InvalidIndex { .. }),
            _ => matches!(error, GptOssLoadError::MissingComponent { .. }),
        };
        assert!(matches, "{target}: {error}");
        assert!(!error.to_string().is_empty());
    }
    // A stacked expert tensor must match the block's expert geometry.
    let error = model
        .write_tensor(
            "model.layers.0.mlp.experts.down_proj",
            TensorData::new(vec![0.0_f32; 3 * 12 * 16], [3, 12, 16]),
        )
        .unwrap_err();
    assert!(
        matches!(error, GptOssLoadError::ExpertShape { .. }),
        "{error}"
    );
    let error = model
        .write_tensor(
            "model.layers.0.mlp.experts.gate_up_proj_bias",
            TensorData::new(vec![0.0_f32; 4 * 12], [4, 12]),
        )
        .unwrap_err();
    assert!(
        matches!(error, GptOssLoadError::ExpertShape { .. }),
        "{error}"
    );
    // A stacked payload whose bytes disagree with its shape is rejected, never read.
    let (count, hidden, width) = {
        let experts = &model.model.layers[0].mlp.experts;
        let [hidden, width] = experts[0].gate_proj.weight.lazy_shape().dims();
        (experts.len(), hidden, width)
    };
    for (target, shape) in [
        (
            "model.layers.0.mlp.experts.gate_up_proj",
            vec![count, hidden, 2 * width],
        ),
        (
            "model.layers.0.mlp.experts.down_proj",
            vec![count, width, hidden],
        ),
    ] {
        let short = TensorData::from_bytes_vec(vec![0; 4], shape, burn::tensor::DType::F32);
        let error = model.write_tensor(target, short).unwrap_err();
        assert!(
            matches!(
                error,
                GptOssLoadError::Parameter {
                    source: ParameterInstallError::PayloadLength { actual: 4, .. },
                    ..
                }
            ),
            "{target}: {error}"
        );
    }
    // A wrongly ranked parameter reports the install failure.
    let error = model
        .write_tensor(
            "model.norm.weight",
            TensorData::new(vec![0.0_f32; 4], [2, 2]),
        )
        .unwrap_err();
    assert!(
        matches!(error, GptOssLoadError::Parameter { .. }),
        "{error}"
    );
}

/// Only the embedding-to-head alias of a tied, installed embedding is accepted.
#[test]
fn sink_accepts_only_the_tied_head_alias() {
    let mut untied = graph(&tiny_config());
    assert!(matches!(
        untied.alias_tensor("lm_head.weight", "model.embed_tokens.weight"),
        Err(GptOssLoadError::InvalidAlias { .. })
    ));
    assert!(matches!(
        untied.alias_tensor("model.norm.weight", "model.embed_tokens.weight"),
        Err(GptOssLoadError::InvalidAlias { .. })
    ));
    let mut tied = graph(&GptOssConfig {
        tie_word_embeddings: true,
        ..tiny_config()
    });
    assert!(
        tied.write_tensor(
            "lm_head.weight",
            TensorData::new(vec![0.0_f32; VOCABULARY * 16], [VOCABULARY, 16])
        )
        .is_err()
    );
    // Burn initializes parameters lazily; an uninstalled embedding is not an owner.
    let installed = tied.write_tensor(
        "model.embed_tokens.weight",
        TensorData::new(vec![0.5_f32; VOCABULARY * 16], [VOCABULARY, 16]),
    );
    assert!(installed.is_ok());
    assert!(
        tied.alias_tensor("lm_head.weight", "model.embed_tokens.weight")
            .is_ok()
    );
}

/// A block without routed experts cannot receive stacked expert tensors.
#[test]
fn stacked_experts_require_routed_experts() {
    let mut model = graph(&tiny_config());
    model.model.layers[0].mlp.experts.clear();
    for target in [
        "model.layers.0.mlp.experts.gate_up_proj",
        "model.layers.0.mlp.experts.down_proj",
    ] {
        let error = model
            .write_tensor(target, TensorData::new(vec![0.0_f32; 4], [1, 2, 2]))
            .unwrap_err();
        assert!(
            matches!(&error, GptOssLoadError::MissingComponent { target: t, component }
                if t == target && *component == "routed experts"),
            "{target}: {error}"
        );
    }
}

/// The tied alias needs an installed embedding owner.
#[test]
fn the_tied_alias_needs_an_installed_embedding() {
    let mut tied = graph(&GptOssConfig {
        tie_word_embeddings: true,
        ..tiny_config()
    });
    let error = tied
        .alias_tensor("lm_head.weight", "model.embed_tokens.weight")
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("the embedding storage owner has not been installed"),
        "{error}"
    );
}

/// A graph edited after planning fails in the sink and reports the partial load.
#[test]
fn sink_failures_after_planning_keep_the_partial_report() {
    let config = tiny_config();
    let directory = tempfile::tempdir().unwrap();
    write(directory.path(), &config, &dense_tensors(&config));
    let mut model = graph(&config);
    model.model.layers[1].mlp.experts.clear();
    let error = model
        .load_checkpoint(directory.path(), &LoadOptions::default())
        .unwrap_err();
    let failure = error
        .execution_failure()
        .expect("planning succeeded, so execution reports the failure");
    assert!(
        failure
            .failed_target
            .as_deref()
            .is_some_and(|target| target.starts_with("model.layers.1.mlp.experts.")),
        "{:?}",
        failure.failed_target
    );
    assert!(matches!(
        &failure.cause,
        crate::checkpoint::ExecutionFailureCause::Sink(GptOssLoadError::MissingComponent { .. })
    ));
}

/// `load_checkpoint_with` under the (empty) family options equals `load_checkpoint`.
#[test]
fn load_checkpoint_with_default_options_matches_load_checkpoint() {
    let config = tiny_config();
    let tensors = dense_tensors(&config);
    let directory = tempfile::tempdir().unwrap();
    write(directory.path(), &config, &tensors);
    let detected = graph(&config)
        .load_checkpoint(directory.path(), &LoadOptions::default())
        .unwrap();
    let explicit = graph(&config)
        .load_checkpoint_with(
            directory.path(),
            &LoadOptions::default(),
            GptOssCheckpointOptions::default(),
        )
        .unwrap();
    assert!(explicit.report.is_strictly_valid());
    assert_eq!(explicit.report, detected.report);
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
        |parameter, data| install("model.layers.0.self_attn.q_proj.weight", parameter, data),
    );
}
