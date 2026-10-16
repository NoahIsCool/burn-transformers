//! Tests T5 checkpoint installation, streaming loads, failure reports, and reference execution.

use std::{
    fs::{self, File},
    io::Write,
    path::{Path, PathBuf},
};

use burn::{
    backend::Flex,
    tensor::{Int, Tensor, TensorData, Tolerance},
};
use serde_json::{Map, Value, json};
use tempfile::TempDir;

use crate::{
    DecodeInput, DecodeState, ForwardOptions, Seq2SeqLanguageModel,
    architecture::{CausalLmInput, Seq2SeqInput},
    cache::CacheConfig,
    checkpoint::{
        CheckpointError, ExecutionFailureCause, LoadOptions, TensorDType, TensorOperation,
        TensorTransform,
    },
    models::config::ModelConfig,
};

use super::*;
use crate::models::t5::{T5Config, T5FeedForwardProjection};

/// CPU backend used by checkpoint installation and execution tests.
type TestBackend = Flex<f32, i32>;

/// Returns a minimal but complete asymmetric encoder-decoder topology.
fn tiny(projection: T5FeedForwardProjection, tied: bool) -> T5Config {
    T5Config {
        vocab_size: 13,
        d_model: 4,
        d_kv: 2,
        d_ff: 6,
        num_layers: 2,
        num_decoder_layers: Some(1),
        num_heads: 2,
        relative_attention_num_buckets: 8,
        relative_attention_max_distance: 16,
        feed_forward_proj: projection,
        dense_act_fn: None,
        is_gated_act: None,
        tie_word_embeddings: tied,
        ..T5Config::default()
    }
}

/// Serializes zero-valued tensors in the minimal SafeTensors wire format.
fn write_checkpoint(
    root: &Path,
    contract: &super::super::T5CheckpointContract,
    mutate: impl FnOnce(&mut Vec<(String, Vec<u64>)>),
) {
    let mut tensors = contract
        .mappings()
        .operations()
        .iter()
        .filter_map(|operation| {
            let TensorOperation::Map {
                source,
                target,
                transform,
            } = operation
            else {
                return None;
            };
            let expected = contract
                .expected_tensors()
                .iter()
                .find(|expected| expected.name == *target)
                .unwrap();
            let mut source_shape = expected.shape.clone();
            if *transform == TensorTransform::Transpose2d {
                source_shape.swap(0, 1);
            }
            Some((source.clone(), source_shape))
        })
        .collect::<Vec<_>>();
    mutate(&mut tensors);

    let mut offset = 0_u64;
    let mut header = Map::new();
    for (name, shape) in &tensors {
        let elements = shape
            .iter()
            .try_fold(1_u64, |product, dimension| product.checked_mul(*dimension));
        let bytes = elements.unwrap().checked_mul(4).unwrap();
        let end = offset.checked_add(bytes).unwrap();
        header.insert(
            name.clone(),
            json!({
                "dtype": "F32",
                "shape": shape,
                "data_offsets": [offset, end],
            }),
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
    file.write_all(&vec![0_u8; offset as usize]).unwrap();
}

/// Writes every storage-owning target directly through the model sink.
fn install_contract(model: &mut T5ForConditionalGeneration<TestBackend>, config: &T5Config) {
    let contract = config.checkpoint_contract().unwrap();
    for expected in contract.expected_tensors() {
        if expected.name == "lm_head.weight" && config.tie_word_embeddings {
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
    if config.tie_word_embeddings {
        model
            .alias_tensor("lm_head.weight", "shared.weight")
            .unwrap();
    }
}

/// Every projection and head-tying combination accepts all checkpoint targets.
#[test]
fn every_dense_gated_tied_and_untied_target_dispatches() {
    use T5FeedForwardProjection::*;

    let device = Default::default();
    for projection in [Relu, Gelu, GatedGelu, GatedGeluNew, GatedSilu] {
        for tied in [false, true] {
            let config = tiny(projection, tied);
            let mut model =
                T5ForConditionalGeneration::<TestBackend>::init(config.clone(), &device).unwrap();
            install_contract(&mut model, &config);
        }
    }
}

/// Installed zero weights produce correctly shaped zero logits through both model stacks.
#[test]
fn zero_checkpoint_runs_a_complete_stateless_forward_pass() {
    let device = Default::default();
    let config = tiny(T5FeedForwardProjection::GatedGelu, false);
    let mut model =
        T5ForConditionalGeneration::<TestBackend>::init(config.clone(), &device).unwrap();
    install_contract(&mut model, &config);

    let output = model
        .forward(
            Seq2SeqInput {
                encoder: CausalLmInput {
                    token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[2, 3]], &device),
                    token_mask: None,
                    position_ids: None,
                },
                decoder: CausalLmInput {
                    token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[0, 4]], &device),
                    token_mask: None,
                    position_ids: None,
                },
            },
            &ForwardOptions::default(),
        )
        .unwrap();
    assert_eq!(output.decoder.logits.dims(), [1, 1, 13]);
    assert!(
        output
            .decoder
            .logits
            .to_data()
            .as_slice::<f32>()
            .unwrap()
            .iter()
            .all(|value| *value == 0.0)
    );
}

/// Invalid aliases, target paths, components, and shapes return structured installation errors.
#[test]
fn invalid_dispatch_paths_are_contextual_and_non_panicking() {
    let device = Default::default();
    let mut tied = T5ForConditionalGeneration::<TestBackend>::init(
        tiny(T5FeedForwardProjection::GatedSilu, true),
        &device,
    )
    .unwrap();

    assert!(matches!(
        tied.alias_tensor("lm_head.weight", "shared.weight"),
        Err(T5LoadError::InvalidAlias { .. })
    ));
    assert!(matches!(
        tied.write_tensor("adapter.weight", TensorData::from([0.0_f32])),
        Err(T5LoadError::UnknownTarget { .. })
    ));
    assert!(matches!(
        tied.write_tensor(
            "encoder.block.nope.layer.0.layer_norm.weight",
            TensorData::zeros::<f32, _>([4])
        ),
        Err(T5LoadError::InvalidBlock { .. })
    ));
    assert!(matches!(
        tied.write_tensor(
            "decoder.block.9.layer.0.layer_norm.weight",
            TensorData::zeros::<f32, _>([4])
        ),
        Err(T5LoadError::InvalidBlock { .. })
    ));
    assert!(matches!(
        tied.write_tensor(
            "encoder.block.1.layer.0.SelfAttention.relative_attention_bias.weight",
            TensorData::zeros::<f32, _>([8, 2])
        ),
        Err(T5LoadError::MissingComponent { .. })
    ));
    assert!(matches!(
        tied.write_tensor(
            "encoder.block.0.layer.1.DenseReluDense.wi.weight",
            TensorData::zeros::<f32, _>([6, 4])
        ),
        Err(T5LoadError::MissingComponent { .. })
    ));
    assert!(matches!(
        tied.write_tensor("shared.weight", TensorData::zeros::<f32, _>([12, 4])),
        Err(T5LoadError::Parameter {
            source: ParameterInstallError::Shape { .. },
            ..
        })
    ));

    tied.write_tensor("shared.weight", TensorData::zeros::<f32, _>([13, 4]))
        .unwrap();
    assert!(matches!(
        tied.alias_tensor("decoder.embed_tokens.weight", "shared.weight"),
        Err(T5LoadError::InvalidAlias { .. })
    ));

    let mut untied = T5ForConditionalGeneration::<TestBackend>::init(
        tiny(T5FeedForwardProjection::Relu, false),
        &device,
    )
    .unwrap();
    untied
        .write_tensor("shared.weight", TensorData::zeros::<f32, _>([13, 4]))
        .unwrap();
    assert!(matches!(
        untied.alias_tensor("lm_head.weight", "shared.weight"),
        Err(T5LoadError::InvalidAlias { .. })
    ));
    assert!(matches!(
        untied.write_tensor(
            "decoder.block.0.layer.2.DenseReluDense.wi_0.weight",
            TensorData::zeros::<f32, _>([6, 4])
        ),
        Err(T5LoadError::MissingComponent { .. })
    ));
}

/// Local loads report expected tensor transforms, aliases, and single-shard streaming statistics.
#[test]
fn local_loader_plans_then_streams_tied_and_untied_checkpoints() {
    let device = Default::default();
    for tied in [false, true] {
        let config = tiny(T5FeedForwardProjection::GatedGelu, tied);
        let contract = config.checkpoint_contract().unwrap();
        let temp = TempDir::new().unwrap();
        write_checkpoint(temp.path(), &contract, |_| {});

        let planned = config
            .plan_checkpoint(temp.path(), &LoadOptions::default())
            .unwrap();
        assert_eq!(planned.load_plan().parameter_count(), 38);
        assert_eq!(planned.manifest().tensors.len(), if tied { 37 } else { 38 });
        assert_eq!(
            planned.load_plan().report.transformed.len(),
            if tied { 25 } else { 26 }
        );
        let (_manifest, generic_plan) = planned.into_parts();
        assert_eq!(generic_plan.parameter_count(), 38);

        let mut model = T5ForConditionalGeneration::<TestBackend>::init(config, &device).unwrap();
        let outcome = model
            .load_checkpoint(temp.path(), &LoadOptions::default())
            .unwrap();
        assert_eq!(outcome.report.applied.len(), 38);
        assert_eq!(outcome.report.transformed.len(), if tied { 25 } else { 26 });
        assert_eq!(outcome.stats.tensors_applied, if tied { 37 } else { 38 });
        assert_eq!(outcome.stats.aliases_applied, usize::from(tied));
        assert_eq!(outcome.stats.source.shard_opens, 1);
        assert_eq!(outcome.stats.source.max_open_shards, 1);
    }
}

/// Missing tensors, unexpected tensors, and shape errors leave model parameters uninitialized.
#[test]
fn planning_failures_precede_all_model_parameter_writes() {
    let device = Default::default();
    let config = tiny(T5FeedForwardProjection::GatedGelu, false);
    let contract = config.checkpoint_contract().unwrap();

    let missing = TempDir::new().unwrap();
    write_checkpoint(missing.path(), &contract, |tensors| {
        tensors.retain(|(name, _)| name != "encoder.final_layer_norm.weight");
    });
    let mut missing_model =
        T5ForConditionalGeneration::<TestBackend>::init(config.clone(), &device).unwrap();
    let error = missing_model
        .load_checkpoint(missing.path(), &LoadOptions::default())
        .unwrap_err();
    assert!(matches!(
        error,
        T5CheckpointLoadError::Checkpoint(CheckpointError::MissingMappingSource { .. })
    ));
    assert!(!missing_model.shared.weight.is_initialized());

    let unexpected = TempDir::new().unwrap();
    write_checkpoint(unexpected.path(), &contract, |tensors| {
        tensors.push(("unexpected.weight".into(), vec![1]));
    });
    let mut unexpected_model =
        T5ForConditionalGeneration::<TestBackend>::init(config.clone(), &device).unwrap();
    let error = unexpected_model
        .load_checkpoint(unexpected.path(), &LoadOptions::default())
        .unwrap_err();
    assert!(matches!(
        error,
        T5CheckpointLoadError::Checkpoint(CheckpointError::StrictValidation { unexpected: 1, .. })
    ));
    assert!(!unexpected_model.shared.weight.is_initialized());

    let wrong_shape = TempDir::new().unwrap();
    write_checkpoint(wrong_shape.path(), &contract, |tensors| {
        tensors
            .iter_mut()
            .find(|(name, _)| name == "shared.weight")
            .unwrap()
            .1 = vec![12, 4];
    });
    let mut wrong_shape_model =
        T5ForConditionalGeneration::<TestBackend>::init(config, &device).unwrap();
    let error = wrong_shape_model
        .load_checkpoint(wrong_shape.path(), &LoadOptions::default())
        .unwrap_err();
    assert!(matches!(
        error,
        T5CheckpointLoadError::Checkpoint(CheckpointError::ShapeMismatch { tensor, .. })
            if tensor == "shared.weight"
    ));
    assert!(!wrong_shape_model.shared.weight.is_initialized());
}

/// A corrupted topology yields a sink error with the failed target and partial application report.
#[test]
fn execution_failure_preserves_partial_report_and_sink_cause() {
    let device = Default::default();
    let config = tiny(T5FeedForwardProjection::Relu, true);
    let contract = config.checkpoint_contract().unwrap();
    let temp = TempDir::new().unwrap();
    write_checkpoint(temp.path(), &contract, |_| {});
    let planned = config
        .plan_checkpoint(temp.path(), &LoadOptions::default())
        .unwrap();
    let mut model = T5ForConditionalGeneration::<TestBackend>::init(config, &device).unwrap();

    // Corrupt a public topology field after pass one so the architecture sink,
    // rather than metadata planning, supplies the audited failure.
    model.encoder.block[0].feed_forward.dense = None;
    let error =
        execute_local_load_plan(planned.manifest(), planned.load_plan(), &mut model).unwrap_err();
    assert!(matches!(
        error.cause,
        ExecutionFailureCause::Sink(T5LoadError::MissingComponent { .. })
    ));
    assert!(error.failed_target.is_some());
    assert!(!error.report.applied.is_empty());
    assert!(error.report.applied.len() < planned.load_plan().parameter_count());
}

/// Checkpoint planning errors expose no runtime execution-failure report.
#[test]
fn checkpoint_error_helpers_expose_only_runtime_partial_reports() {
    let checkpoint = T5CheckpointLoadError::Checkpoint(CheckpointError::InvalidMapping {
        message: "test".into(),
    });
    assert!(checkpoint.execution_failure().is_none());
}

/// F32 storage accounts for four bytes per element and the temporary fixture supports writes.
#[test]
fn synthetic_writer_uses_expected_float_storage_size() {
    assert_eq!(TensorDType::F32.storage_bytes(13 * 4).unwrap(), 208);
    let temp = TempDir::new().unwrap();
    fs::write(temp.path().join("sentinel"), b"ok").unwrap();
    assert_eq!(fs::read(temp.path().join("sentinel")).unwrap(), b"ok");
}

/// The sink accepts only tensors already transformed into Burn's linear layout.
#[test]
fn linear_sink_rejects_checkpoint_layout_and_accepts_runtime_layout() {
    let device = Default::default();
    let mut config = tiny(T5FeedForwardProjection::Relu, false);
    config.d_kv = 3;
    let mut model = T5ForConditionalGeneration::<TestBackend>::init(config, &device).unwrap();

    let q = "encoder.block.0.layer.0.SelfAttention.q.weight";
    assert!(matches!(
        model.write_tensor(q, TensorData::zeros::<f32, _>([6, 4])),
        Err(T5LoadError::Parameter {
            source: ParameterInstallError::Shape { .. },
            ..
        })
    ));
    model
        .write_tensor(q, TensorData::zeros::<f32, _>([4, 6]))
        .unwrap();

    assert!(matches!(
        model.write_tensor("lm_head.weight", TensorData::zeros::<f32, _>([13, 4])),
        Err(T5LoadError::Parameter {
            source: ParameterInstallError::Shape { .. },
            ..
        })
    ));
    model
        .write_tensor("lm_head.weight", TensorData::zeros::<f32, _>([4, 13]))
        .unwrap();
}

/// Loads the pinned official FLAN-T5-small artifact and executes real weights.
///
/// Reference probes were produced by Transformers 5.2.0 in CPU F32 at the
/// pinned `0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab` revision. The same test
/// then proves cached prefill/decode parity against the loaded stateless graph.
///
/// The test is ignored because it requires a 308 MB artifact outside Git and
/// allocates the complete F32 model. Set `TRANSFORMERS_FLAN_T5_SMALL_DIR` when
/// the repository's `models` symlink is unavailable from a worktree.
#[test]
#[ignore = "requires pinned google/flan-t5-small SafeTensors artifacts"]
fn pinned_flan_t5_small_loads_strictly_and_runs_forward() {
    let root = std::env::var_os("TRANSFORMERS_FLAN_T5_SMALL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("models/google/flan-t5-small-0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab")
        });
    let config = T5Config::from_json_str(
        &fs::read_to_string(root.join("config.json"))
            .unwrap_or_else(|error| panic!("cannot read pinned FLAN config: {error}")),
    )
    .unwrap();
    assert_eq!(config.feed_forward_proj, T5FeedForwardProjection::GatedGelu);
    assert!(!config.tie_word_embeddings);

    let plan = config
        .plan_checkpoint(&root, &LoadOptions::default())
        .unwrap();
    assert_eq!(plan.manifest().tensors.len(), 190);
    assert_eq!(plan.load_plan().parameter_count(), 190);
    assert_eq!(plan.load_plan().report.transformed.len(), 145);
    assert!(plan.load_plan().report.allowlisted_ignored.is_empty());
    assert_eq!(plan.load_plan().memory.checkpoint_tensor_bytes, 307_844_608);

    let device = Default::default();
    let mut model = T5ForConditionalGeneration::<TestBackend>::init(config, &device).unwrap();
    let outcome = model
        .load_checkpoint(&root, &LoadOptions::default())
        .unwrap();
    assert_eq!(outcome.report.applied.len(), 190);
    assert_eq!(outcome.report.transformed.len(), 145);
    assert_eq!(outcome.stats.source.shard_opens, 1);
    assert_eq!(outcome.stats.source.max_open_shards, 1);

    let output = model
        .forward(
            Seq2SeqInput {
                encoder: CausalLmInput {
                    token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[71, 8, 57]], &device),
                    token_mask: None,
                    position_ids: None,
                },
                decoder: CausalLmInput {
                    token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[0]], &device),
                    token_mask: None,
                    position_ids: None,
                },
            },
            &ForwardOptions::default(),
        )
        .unwrap();
    assert_eq!(output.decoder.logits.dims(), [1, 1, 32_128]);
    let reference_logits = output.decoder.logits.to_data();
    let reference_logits = reference_logits.as_slice::<f32>().unwrap();
    let mut ranked = reference_logits
        .iter()
        .copied()
        .enumerate()
        .collect::<Vec<_>>();
    ranked.sort_unstable_by(|left, right| right.1.total_cmp(&left.1));
    assert_eq!(
        ranked
            .iter()
            .take(10)
            .map(|(token, _)| *token)
            .collect::<Vec<_>>(),
        vec![71, 37, 2588, 1609, 10129, 6357, 290, 282, 389, 6776]
    );
    for (token, expected) in [
        (0, -60.822_26_f32),
        (1, -3.995_400_7),
        (2, -14.725_48),
        (3, -4.768_744_5),
        (8, -7.057_482_2),
        (71, 3.299_487_6),
        (100, -3.746_507_4),
        (999, -16.090_796),
        (32_000, -61.027_275),
        (32_127, -60.742_435),
    ] {
        assert!(
            (reference_logits[token] - expected).abs() <= 1e-3,
            "token {token}: expected {expected}, found {}",
            reference_logits[token]
        );
    }
    assert!(
        output
            .decoder
            .logits
            .to_data()
            .as_slice::<f32>()
            .unwrap()
            .iter()
            .all(|value| value.is_finite())
    );

    let encoder = model
        .encode(
            CausalLmInput {
                token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[71, 8, 57]], &device),
                token_mask: None,
                position_ids: None,
            },
            &ForwardOptions::default(),
        )
        .unwrap();
    let mut state = model
        .new_decode_state(CacheConfig::new(1, 4).unwrap(), 1, &device)
        .unwrap();
    let cached = model
        .prefill_decoder(
            &encoder,
            CausalLmInput {
                token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[0]], &device),
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .unwrap();
    cached.decoder.logits.into_data().assert_approx_eq::<f32>(
        &output.decoder.logits.into_data(),
        Tolerance::absolute(1e-4),
    );

    let cached_step = model
        .decode(
            &encoder,
            DecodeInput {
                token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[3]], &device),
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .unwrap();
    let stateless_step = model
        .forward(
            Seq2SeqInput {
                encoder: CausalLmInput {
                    token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[71, 8, 57]], &device),
                    token_mask: None,
                    position_ids: None,
                },
                decoder: CausalLmInput {
                    token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[0, 3]], &device),
                    token_mask: None,
                    position_ids: None,
                },
            },
            &ForwardOptions::default(),
        )
        .unwrap();
    cached_step
        .decoder
        .logits
        .into_data()
        .assert_approx_eq::<f32>(
            &stateless_step.decoder.logits.into_data(),
            Tolerance::absolute(1e-4),
        );
    assert_eq!(state.lengths(), [2]);
}

/// Malformed block targets and unknown parameters name the precise sink failure.
#[test]
fn malformed_targets_and_missing_components_are_specific() {
    let device = Default::default();
    let mut gated = T5ForConditionalGeneration::<TestBackend>::init(
        tiny(T5FeedForwardProjection::GatedSilu, false),
        &device,
    )
    .unwrap();
    let vector = || TensorData::zeros::<f32, _>([4]);
    let invalid_block =
        |model: &mut T5ForConditionalGeneration<TestBackend>, target: &str| match model
            .write_tensor(target, vector())
            .unwrap_err()
        {
            T5LoadError::InvalidBlock { target: t, reason } => {
                assert_eq!(t, target);
                reason
            }
            other => panic!("expected an invalid block for {target}, found {other:?}"),
        };
    assert_eq!(
        invalid_block(&mut gated, "encoder.block.0"),
        "expected `<stack>.block.<index>.<parameter>`"
    );
    assert_eq!(
        invalid_block(&mut gated, "encoder.block..layer.0.layer_norm.weight"),
        "block index and parameter suffix must be non-empty"
    );
    assert_eq!(
        invalid_block(&mut gated, "decoder.block.0."),
        "block index and parameter suffix must be non-empty"
    );
    for target in [
        "encoder.block.0.layer.9.layer_norm.weight",
        "encoder.block.0.layer.1.EncDecAttention.q.weight",
        "decoder.block.0.layer.0.SelfAttention.bias",
        "decoder.block.0.layer.2.DenseReluDense.wx.weight",
    ] {
        assert_eq!(
            gated.write_tensor(target, vector()),
            Err(T5LoadError::UnknownTarget {
                target: target.into()
            }),
            "{target}"
        );
    }

    // Edited public modules no longer match the validated checkpoint topology.
    gated.lm_head = None;
    gated.decoder.block[0].cross_attention = None;
    gated.decoder.block[0].feed_forward.gated = None;
    gated.encoder.block[0].feed_forward.gated = None;
    for (target, component) in [
        (
            "lm_head.weight",
            "independent LM head for an untied checkpoint",
        ),
        (
            "decoder.block.0.layer.1.EncDecAttention.k.weight",
            "decoder encoder-decoder attention",
        ),
        (
            "decoder.block.0.layer.1.layer_norm.weight",
            "decoder encoder-decoder attention",
        ),
        (
            "encoder.block.0.layer.1.DenseReluDense.wi_1.weight",
            "gated `wi_1` feed-forward projection",
        ),
        (
            "encoder.block.0.layer.1.DenseReluDense.wo.weight",
            "feed-forward output projection matching the configured topology",
        ),
    ] {
        assert_eq!(
            gated.write_tensor(target, vector()),
            Err(T5LoadError::MissingComponent {
                target: target.into(),
                component,
            }),
            "{target}"
        );
    }
}

/// A sink failure during streaming surfaces through `load_checkpoint` with its partial report.
#[test]
fn load_checkpoint_returns_execution_failures_with_partial_reports() {
    let device = Default::default();
    let config = tiny(T5FeedForwardProjection::Relu, false);
    let contract = config.checkpoint_contract().unwrap();
    let temp = TempDir::new().unwrap();
    write_checkpoint(temp.path(), &contract, |_| {});
    let mut model = T5ForConditionalGeneration::<TestBackend>::init(config, &device).unwrap();
    model.lm_head = None;
    let error = model
        .load_checkpoint(temp.path(), &LoadOptions::default())
        .unwrap_err();
    let failure = error
        .execution_failure()
        .expect("planning succeeded, so execution reports the failure");
    assert_eq!(failure.failed_target.as_deref(), Some("lm_head.weight"));
    assert!(matches!(
        failure.cause,
        ExecutionFailureCause::Sink(T5LoadError::MissingComponent { .. })
    ));
    assert!(matches!(error, T5CheckpointLoadError::Execution(_)));
}
