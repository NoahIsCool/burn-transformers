//! Local automatic loading, artifact containment, and generation-default precedence regressions.

use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    io::Write,
    path::Path,
};

use burn::{
    backend::Flex,
    tensor::{DType, Int, Tensor, TensorData, Tolerance},
};
use serde_json::{Map, Value, json};

use super::*;
use crate::{
    CausalLmInput, DecodeInput, ForwardOptions, LogitsSelection, Seq2SeqInput,
    cache::CacheConfig,
    checkpoint::{
        CheckpointManifest, ExpectedTensor, LoadReport, MappingPlan, TensorOperation,
        TensorTransform,
        quantization::{
            BURNPACK_QUANTIZATION_FORMAT_VERSION_V2, PackedExecutionSupport, QuantizationExclusion,
            QuantizationGroupSize, QuantizedBurnpackArtifact, QuantizedCheckpointOptions,
            SymmetricQuantization, WeightQuantizationConfig, quantize_local_checkpoint_to_burnpack,
        },
    },
    models::{
        AutoCausalConfig,
        deepseek_v2::{
            DeepseekV2CheckpointOptions, model::tests_support::tiny_config as tiny_deepseek_config,
        },
        muse_glimmer::{
            MuseGlimmerCheckpointOptions, MuseGlimmerCheckpointSource, MuseGlimmerLayerType,
            MuseGlimmerTextConfig,
        },
        nemotron_h::{NemotronHBlockType, NemotronHCheckpointOptions, NemotronHConfig},
        qwen3_5::{
            Qwen3_5CheckpointOptions, Qwen3_5CheckpointSource, Qwen3_5LayerType, Qwen3_5TextConfig,
        },
        qwen3_moe::{Qwen3MoeCheckpointOptions, Qwen3MoeConfig},
        t5::T5FeedForwardProjection,
    },
};

/// CPU backend with F32 tensor values and I32 token indices for deterministic fixtures.
type TestBackend = Flex<f32, i32>;

/// Writes one complete zero-valued F32 checkpoint from a family contract.
///
/// The contract remains the source of truth for both external names and
/// shapes. Alias operations have no independent checkpoint payload; the
/// automatic-loader fixtures use untied embeddings so every logical target
/// is represented by one physical `Map` operation.
fn write_contract_checkpoint(root: &Path, expected: &[ExpectedTensor], mappings: &MappingPlan) {
    let mut tensors = Vec::new();
    for operation in mappings.operations() {
        match operation {
            TensorOperation::Map {
                source,
                target,
                transform,
            } => {
                let target_shape = expected
                    .iter()
                    .find(|tensor| tensor.name == *target)
                    .unwrap_or_else(|| panic!("mapping target `{target}` must be expected"))
                    .shape
                    .clone();
                let source_shape = match transform {
                    TensorTransform::Identity => target_shape,
                    TensorTransform::Transpose2d => {
                        assert_eq!(target_shape.len(), 2, "transpose fixture must be rank two");
                        vec![target_shape[1], target_shape[0]]
                    }
                };
                tensors.push((source.clone(), source_shape));
            }
            TensorOperation::Alias { .. } => {}
            TensorOperation::Split { .. } | TensorOperation::Concatenate { .. } => {
                panic!("automatic family fixture supports map and alias contracts only")
            }
        }
    }

    let mut offset = 0_u64;
    let mut header = Map::new();
    for (name, shape) in tensors {
        let elements = shape.iter().copied().product::<u64>();
        let end = offset
            .checked_add(elements.checked_mul(4).expect("fixture byte size"))
            .expect("fixture data offset");
        assert!(
            header
                .insert(
                    name.clone(),
                    json!({"dtype": "F32", "shape": shape, "data_offsets": [offset, end]}),
                )
                .is_none(),
            "fixture source `{name}` must be unique"
        );
        offset = end;
    }
    let mut header = serde_json::to_vec(&Value::Object(header)).expect("fixture header");
    while !header.len().is_multiple_of(8) {
        header.push(b' ');
    }
    let mut file = File::create(root.join("model.safetensors")).expect("fixture checkpoint");
    file.write_all(&(header.len() as u64).to_le_bytes())
        .expect("fixture header length");
    file.write_all(&header).expect("fixture header bytes");
    file.write_all(&vec![0; offset as usize])
        .expect("fixture tensor bytes");
}

/// Writes a nonzero, quantization-exact F32 checkpoint for a Muse miniature.
///
/// Projection groups contain only `{-7/256, 0, 7/256}`. Both symmetric Q4
/// and Q8 therefore reconstruct every matrix value exactly: Q4 maps the
/// extrema to `+-7`, while Q8 maps them to `+-127`. This makes differences
/// between full-precision and packed execution evidence of wiring or cache
/// errors rather than expected quantization noise.
fn write_patterned_muse_checkpoint(
    root: &Path,
    expected: &[ExpectedTensor],
    mappings: &MappingPlan,
) {
    let mut tensors = Vec::new();
    let mut offset = 0_u64;
    let mut header = Map::new();
    for operation in mappings.operations() {
        let TensorOperation::Map {
            source,
            target,
            transform,
        } = operation
        else {
            continue;
        };
        assert_eq!(
            *transform,
            TensorTransform::Identity,
            "Muse patterned fixture expects identity source layouts"
        );
        let shape = expected
            .iter()
            .find(|tensor| tensor.name == *target)
            .unwrap_or_else(|| panic!("mapping target `{target}` must be expected"))
            .shape
            .clone();
        let elements = shape.iter().copied().product::<u64>();
        let end = offset
            .checked_add(elements.checked_mul(4).expect("fixture byte size"))
            .expect("fixture data offset");
        assert!(
            header
                .insert(
                    source.clone(),
                    json!({"dtype": "F32", "shape": shape, "data_offsets": [offset, end]}),
                )
                .is_none(),
            "fixture source `{source}` must be unique"
        );
        tensors.push((
            target.clone(),
            usize::try_from(elements).expect("host fixture size"),
        ));
        offset = end;
    }

    let mut header = serde_json::to_vec(&Value::Object(header)).expect("fixture header");
    while !header.len().is_multiple_of(8) {
        header.push(b' ');
    }
    let mut file = File::create(root.join("model.safetensors")).expect("fixture checkpoint");
    file.write_all(&(header.len() as u64).to_le_bytes())
        .expect("fixture header length");
    file.write_all(&header).expect("fixture header bytes");
    for (target, elements) in tensors {
        for index in 0..elements {
            let value = if target == "model.norm.weight" {
                1.0_f32
            } else if target.contains("layernorm") || target.ends_with(".bias") {
                0.0
            } else if target == "model.embed_tokens.weight" {
                ((index % 11) as f32 - 5.0) / 64.0
            } else {
                match index % 3 {
                    0 => -7.0 / 256.0,
                    1 => 0.0,
                    _ => 7.0 / 256.0,
                }
            };
            file.write_all(&value.to_le_bytes())
                .expect("fixture tensor value");
        }
    }
}

/// Returns a two-layer Muse topology whose every matrix row aligns to G64.
fn packed_muse_config() -> MuseGlimmerTextConfig {
    MuseGlimmerTextConfig {
        vocab_size: 128,
        hidden_size: 64,
        intermediate_size: 64,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dim: 16,
        max_position_embeddings: 16,
        sliding_window: Some(4),
        layer_types: vec![
            MuseGlimmerLayerType::SlidingAttention,
            MuseGlimmerLayerType::FullAttention,
        ],
        layer_rope_theta: vec![10_000.0, 0.0],
        bos_token_id: Some(TokenIds::One(1)),
        eos_token_id: Some(TokenIds::One(5)),
        pad_token_id: Some(TokenIds::One(0)),
        tie_word_embeddings: false,
        attention_bias: true,
        ..MuseGlimmerTextConfig::default()
    }
}

/// Executes stateless, cached-prefill, and one-token recurrent Muse paths.
fn execute_muse_paths(model: &AutoCausalLm<TestBackend>) -> [TensorData; 4] {
    let device = Default::default();
    let options = ForwardOptions {
        logits: LogitsSelection::All,
        ..ForwardOptions::default()
    };
    let prompt = CausalLmInput {
        token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[1, 2, 3]], &device),
        token_mask: None,
        position_ids: None,
    };
    let stateless = model
        .forward(prompt.clone(), &options)
        .expect("stateless Muse fixture")
        .logits
        .into_data();
    let mut state = model
        .new_decode_state(
            CacheConfig::new(1, 8).expect("fixture cache geometry"),
            1,
            &device,
        )
        .expect("Muse fixture decode state");
    let prefill = model
        .prefill(prompt, &mut state, &options)
        .expect("cached Muse fixture prefill")
        .logits
        .into_data();
    let decoded = model
        .decode(
            DecodeInput {
                token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[4]], &device),
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &options,
        )
        .expect("recurrent Muse fixture decode")
        .logits
        .into_data();
    let extended = model
        .forward(
            CausalLmInput {
                token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[1, 2, 3, 4]], &device),
                token_mask: None,
                position_ids: None,
            },
            &ForwardOptions::default(),
        )
        .expect("extended stateless Muse fixture")
        .logits
        .into_data();
    [stateless, prefill, decoded, extended]
}

/// Persists one direct architecture config plus deterministic generation metadata.
fn write_auto_metadata<T: serde::Serialize>(root: &Path, config: &T) {
    std::fs::write(
        root.join("config.json"),
        serde_json::to_vec(config).expect("serializable family config"),
    )
    .expect("config fixture");
    std::fs::write(
        root.join("generation_config.json"),
        br#"{"max_new_tokens":5,"do_sample":false}"#,
    )
    .expect("generation fixture");
}

/// Returns provenance options that also avoid requiring a tokenizer fixture.
fn fixture_options(repository: &str, revision: &str) -> AutoLoadOptions {
    AutoLoadOptions {
        load_tokenizer: false,
        repository: Some(repository.into()),
        revision: Some(revision.into()),
        ..AutoLoadOptions::default()
    }
}

/// Verifies exhaustive strict accounting for one untied map/alias contract.
fn assert_strict_report(report: &LoadReport, expected: &[ExpectedTensor], mappings: &MappingPlan) {
    let mut expected_names = expected
        .iter()
        .map(|tensor| tensor.name.clone())
        .collect::<Vec<_>>();
    expected_names.sort();
    assert_eq!(report.applied, expected_names);
    assert!(report.is_strictly_valid());
    assert!(report.missing.is_empty());
    assert!(report.unexpected.is_empty());
    assert!(report.allowlisted_missing.is_empty());
    assert!(report.allowlisted_ignored.is_empty());
    assert!(report.aliased.is_empty());
    assert!(report.dtype_converted.is_empty());
    assert!(report.quantized.is_empty());

    let expected_transforms = mappings
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
            match transform {
                TensorTransform::Identity if source != target => Some((target.as_str(), "rename")),
                TensorTransform::Transpose2d => Some((target.as_str(), "transpose_2d")),
                TensorTransform::Identity => None,
            }
        })
        .collect::<BTreeMap<_, _>>();
    let observed_transforms = report
        .transformed
        .iter()
        .map(|record| (record.target.as_str(), record.operation.as_str()))
        .collect::<BTreeMap<_, _>>();
    assert_eq!(observed_transforms, expected_transforms);
    let memory = report.memory.expect("strict plan must estimate memory");
    assert!(memory.checkpoint_tensor_bytes > 0);
    assert!(memory.target_parameter_bytes > 0);
}

/// Verifies source identity and normalized host-side generation defaults.
fn assert_bundle_metadata(
    loaded: &LoadedModel<AutoCausalLm<TestBackend>>,
    root: &Path,
    model_type: &str,
    repository: &str,
    revision: &str,
    eos_token_ids: &[u32],
    pad_token_id: u32,
) {
    assert_eq!(
        loaded.source().root(),
        root.canonicalize().expect("canonical fixture root")
    );
    assert_eq!(loaded.source().model_type(), model_type);
    assert_eq!(loaded.source().repository(), Some(repository));
    assert_eq!(loaded.source().revision(), Some(revision));
    assert_eq!(loaded.generation_defaults().max_new_tokens, 5);
    assert_eq!(
        loaded.generation_defaults().strategy,
        SamplingStrategy::Greedy
    );
    assert_eq!(loaded.generation_defaults().eos_token_ids, eos_token_ids);
    assert_eq!(
        loaded.generation_defaults().pad_token_id,
        Some(pad_token_id)
    );
    assert!(loaded.tokenizer().is_none());
    assert_execution_audit(loaded);
}

/// Requires internally consistent, bounded local checkpoint execution counters.
fn assert_execution_audit<M>(loaded: &LoadedModel<M>) {
    let stats = loaded
        .execution_stats()
        .expect("automatic loads must retain execution counters");
    assert_eq!(
        stats.tensors_applied + stats.aliases_applied,
        loaded.load_report().applied.len()
    );
    assert_eq!(stats.source_reads, stats.source.tensor_reads);
    assert_eq!(stats.source_payload_bytes, stats.source.payload_bytes_read);
    assert!(stats.source_reads > 0);
    assert!(stats.source_payload_bytes > 0);
    assert_eq!(stats.source.shard_opens, 1);
    assert_eq!(stats.source.max_open_shards, 1);
}

/// Runs one automatic causal variant and requires finite vocabulary logits.
fn assert_finite_forward(loaded: &LoadedModel<AutoCausalLm<TestBackend>>, vocabulary_size: usize) {
    let device = Default::default();
    let output = loaded
        .model()
        .forward(
            CausalLmInput {
                token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[1, 2]], &device),
                token_mask: None,
                position_ids: None,
            },
            &ForwardOptions::default(),
        )
        .expect("loaded automatic model must execute");
    let [batch, selected, vocabulary] = output.logits.dims();
    assert_eq!([batch, selected, vocabulary], [1, 1, vocabulary_size]);
    assert!(
        output
            .logits
            .into_data()
            .as_slice::<f32>()
            .expect("F32 reference backend")
            .iter()
            .all(|value| value.is_finite())
    );
}

/// Automatic loading is base-only unless MTP consumption is explicit.
#[test]
fn automatic_options_keep_mtp_opt_in() {
    assert!(!AutoLoadOptions::default().qwen_mtp);
}

/// Automatic packed loading rejects families without a completed backend gate.
#[test]
fn automatic_packed_loading_rejects_uncertified_causal_families() {
    let root = tempfile::tempdir().expect("temporary model root");
    std::fs::write(
        root.path().join("config.json"),
        serde_json::to_vec(&tiny_deepseek_config()).expect("serializable config"),
    )
    .expect("config fixture");
    let error = AutoCausalLm::<TestBackend>::load_local(
        root.path(),
        &AutoLoadOptions {
            load_tokenizer: false,
            quantized_burnpack: Some(PathBuf::from("weights.bpk")),
            ..AutoLoadOptions::default()
        },
        &Default::default(),
    )
    .expect_err("uncertified family must fail before checkpoint discovery");
    assert!(matches!(error, Error::BackendCapability(_)));
}

/// Qwen3-MoE uses the public automatic path for a provenance-bound packed overlay.
#[test]
fn automatic_qwen3_moe_loads_v2_quantized_burnpack() {
    let root = tempfile::tempdir().expect("temporary model root");
    let config = Qwen3MoeConfig {
        vocab_size: 64,
        hidden_size: 64,
        intermediate_size: 64,
        num_hidden_layers: 1,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dim: 16,
        max_position_embeddings: 16,
        max_window_layers: 1,
        decoder_sparse_step: 1,
        moe_intermediate_size: 64,
        num_experts: 2,
        num_experts_per_tok: 1,
        norm_topk_prob: true,
        eos_token_id: Some(TokenIds::One(3)),
        pad_token_id: Some(TokenIds::One(0)),
        ..Qwen3MoeConfig::default()
    };
    let contract = config
        .checkpoint_contract(Qwen3MoeCheckpointOptions::default())
        .expect("valid packed Qwen3-MoE contract");
    write_auto_metadata(root.path(), &config);
    write_contract_checkpoint(
        root.path(),
        contract.expected_tensors(),
        contract.mappings(),
    );
    let load_options = LoadOptions::default();
    let checkpoint = CheckpointManifest::discover(root.path(), &load_options)
        .expect("discover tiny Qwen3-MoE checkpoint");
    let plan = contract
        .build_load_plan(&checkpoint, &load_options)
        .expect("plan tiny Qwen3-MoE checkpoint");
    let burnpack = root.path().join("weights.bpk");
    let converted = quantize_local_checkpoint_to_burnpack(
        &checkpoint,
        &plan,
        &burnpack,
        QuantizedCheckpointOptions::default(),
    )
    .expect("convert tiny Qwen3-MoE Burnpack");
    assert_eq!(converted.report.quantized.len(), 11);
    let expected_packed_bytes = converted
        .report
        .quantized
        .iter()
        .map(|record| record.bytes)
        .sum::<u64>();
    let artifact =
        QuantizedBurnpackArtifact::open(&burnpack).expect("open tiny Qwen3-MoE Burnpack");
    assert_eq!(
        artifact.manifest().format_version,
        BURNPACK_QUANTIZATION_FORMAT_VERSION_V2
    );
    assert!(artifact.manifest().provenance.is_some());
    drop(artifact);

    let repository = "Qwen/tiny-qwen3-moe-packed-fixture";
    let revision = "qwen3-moe-packed-fixture-revision";
    let loaded = AutoCausalLm::<TestBackend>::load_local(
        root.path(),
        &AutoLoadOptions {
            quantized_burnpack: Some(PathBuf::from("weights.bpk")),
            ..fixture_options(repository, revision)
        },
        &Default::default(),
    )
    .expect("automatic packed Qwen3-MoE load");

    let model = match loaded.model() {
        AutoCausalLm::Qwen3Moe(model) => model,
        model => panic!("expected Qwen3-MoE automatic variant, found {model:?}"),
    };
    assert!(matches!(
        model
            .lm_head
            .as_ref()
            .expect("untied fixture LM head")
            .weight
            .val()
            .dtype(),
        DType::QFloat(_)
    ));
    assert_eq!(model.model.embed_tokens.weight.val().dtype(), DType::F32);
    assert!(loaded.load_report().is_strictly_valid());
    assert_eq!(
        loaded.load_report().applied.len(),
        contract.expected_tensors().len()
    );
    assert_eq!(loaded.load_report().quantized.len(), 11);
    assert!(
        loaded
            .load_report()
            .quantized
            .iter()
            .all(|record| { record.scheme == "burn-qfloat-symmetric-q8-g64-block-1xg" })
    );
    let execution = loaded
        .execution_stats()
        .expect("automatic packed load retains checkpoint counters");
    assert_eq!(execution.source_reads, 7);
    assert_eq!(execution.source_payload_bytes, 17_792);
    assert_eq!(execution.source.shard_opens, 1);
    assert_eq!(execution.source.max_open_shards, 1);
    let quantized = loaded
        .quantized_load_audit()
        .expect("automatic packed load retains sidecar counters");
    assert_eq!(
        quantized.format_version(),
        BURNPACK_QUANTIZATION_FORMAT_VERSION_V2
    );
    assert_eq!(quantized.preflight().passes_started, 1);
    assert_eq!(quantized.preflight().passes_completed, 1);
    assert_eq!(quantized.preflight().tensor_reads, 11);
    assert_eq!(quantized.preflight().payload_bytes, expected_packed_bytes);
    assert!(quantized.preflight().peak_owned_payload_bytes > 0);
    assert_eq!(quantized.packed().tensor_reads, 11);
    assert_eq!(quantized.packed().payload_bytes, expected_packed_bytes);
    assert_eq!(quantized.packed().tensors_applied, 11);
    let before = quantized
        .provenance()
        .pre_execution
        .expect("v2 load hashes checkpoint before execution");
    let after = quantized
        .provenance()
        .post_execution
        .expect("v2 load hashes checkpoint after execution");
    assert_eq!(before.passes_completed, 1);
    assert_eq!(after.passes_completed, 1);
    assert_eq!(before.shards_hashed, checkpoint.shards.len());
    assert_eq!(after.shards_hashed, checkpoint.shards.len());
    assert_eq!(before.bytes_hashed, checkpoint.artifact_bytes);
    assert_eq!(after.bytes_hashed, checkpoint.artifact_bytes);
    assert_bundle_metadata(
        &loaded,
        root.path(),
        "qwen3_moe",
        repository,
        revision,
        &[3],
        0,
    );
    assert_finite_forward(&loaded, config.vocab_size);
}

/// Model tokens provide defaults and local sampling metadata overrides them.
#[test]
fn generation_defaults_are_normalized_from_local_metadata() {
    let root = tempfile::tempdir().expect("temporary model root");
    std::fs::write(
        root.path().join("generation_config.json"),
        br#"{
                "max_new_tokens": 17,
                "do_sample": true,
                "temperature": 0.7,
                "top_k": 0,
                "top_p": 0.9,
                "repetition_penalty": 1.1,
                "presence_penalty": 0.2,
                "seed": 42,
                "eos_token_id": [9, 8, 9],
                "pad_token_id": 7,
                "unknown_future_field": true
            }"#,
    )
    .expect("generation fixture");
    let config = normalize_generation_defaults(
        root.path(),
        Some(&TokenIds::Many(vec![3, 2, 3])),
        Some(&TokenIds::One(1)),
        16,
        None,
    )
    .expect("valid normalized defaults");
    assert_eq!(config.max_new_tokens, 17);
    assert_eq!(config.strategy, SamplingStrategy::Sample);
    assert_eq!(config.temperature, 0.7);
    assert_eq!(config.top_k, None);
    assert_eq!(config.top_p, Some(0.9));
    assert_eq!(config.repetition_penalty, 1.1);
    assert_eq!(config.presence_penalty, 0.2);
    assert_eq!(config.seed, 42);
    assert_eq!(config.eos_token_ids, [8, 9]);
    assert_eq!(config.pad_token_id, Some(7));
}

/// Greedy local defaults discard inactive sampling warpers and explicit config wins.
#[test]
fn explicit_defaults_take_precedence_and_greedy_ignores_warpers() {
    let root = tempfile::tempdir().expect("temporary model root");
    std::fs::write(
        root.path().join("generation_config.json"),
        br#"{"do_sample":false,"temperature":0.0,"top_k":50,"top_p":0.8}"#,
    )
    .expect("generation fixture");
    let greedy = normalize_generation_defaults(root.path(), None, None, 64, None)
        .expect("inactive invalid temperature is normalized away");
    assert_eq!(greedy.strategy, SamplingStrategy::Greedy);
    assert_eq!(greedy.temperature, 1.0);
    assert_eq!(greedy.top_k, None);
    assert_eq!(greedy.top_p, None);

    let explicit = GenerationConfig {
        max_new_tokens: 3,
        eos_token_ids: vec![5],
        ..GenerationConfig::default()
    };
    let normalized = normalize_generation_defaults(
        root.path(),
        Some(&TokenIds::One(7)),
        None,
        64,
        Some(&explicit),
    )
    .expect("explicit defaults");
    assert_eq!(normalized, explicit);
}

/// Invalid local metadata, token bounds, and ambiguous padding are rejected.
#[test]
fn generation_defaults_reject_invalid_artifacts_and_tokens() {
    let root = tempfile::tempdir().expect("temporary model root");
    std::fs::write(root.path().join("generation_config.json"), b"{")
        .expect("invalid generation fixture");
    assert!(normalize_generation_defaults(root.path(), None, None, 8, None).is_err());
    std::fs::remove_file(root.path().join("generation_config.json"))
        .expect("remove invalid fixture");
    assert!(
        normalize_generation_defaults(root.path(), Some(&TokenIds::One(8)), None, 8, None).is_err()
    );
    assert!(
        normalize_generation_defaults(
            root.path(),
            None,
            Some(&TokenIds::Many(vec![1, 2])),
            8,
            None
        )
        .is_err()
    );
}

/// Every registered causal family reaches its own strict loader, never the fallback.
#[test]
fn every_registered_family_reaches_its_strict_loader() {
    for family in AutoCausalConfig::ALL {
        let root = tempfile::tempdir().expect("temporary model root");
        std::fs::write(
            root.path().join("config.json"),
            json!({"model_type": family.model_type(), "vocab_size": "not an integer"}).to_string(),
        )
        .unwrap();
        let error = AutoCausalLm::<TestBackend>::load_local(
            root.path(),
            &fixture_options("fixture/registry", "registry"),
            &Default::default(),
        )
        .unwrap_err();
        assert!(
            matches!(&error, Error::Configuration { field, .. } if field == "auto_causal.config"),
            "{family:?}: {error}"
        );
    }
}

/// Packed-sidecar certification names only registered causal families.
#[test]
fn packed_certification_names_registered_families() {
    for model_type in PACKED_CAUSAL_MODEL_TYPES {
        assert!(
            AutoCausalConfig::from_model_type(model_type).is_some(),
            "{model_type}"
        );
    }
}

/// Required config reading rejects missing files and non-directory roots.
#[test]
fn required_config_is_bounded_and_regular() {
    let root = tempfile::tempdir().expect("temporary model root");
    assert!(read_required_config(root.path()).is_err());
    let file = root.path().join("not-a-directory");
    std::fs::write(&file, b"{}").expect("fixture file");
    assert!(read_required_config(&file).is_err());
    std::fs::create_dir(root.path().join("config.json")).expect("fixture directory");
    assert!(read_required_config(root.path()).is_err());
}

/// Fixed local artifacts preserve exact size limits and contextual I/O errors.
#[test]
fn contained_artifacts_enforce_boundaries_and_requiredness() {
    let directory = tempfile::tempdir().expect("temporary artifact root");
    let root = directory.path().canonicalize().expect("canonical root");
    assert_eq!(
        contained_regular_file(&root, "absent.json", 2, false).unwrap(),
        None
    );
    let error = contained_regular_file(&root, "absent.json", 2, true).unwrap_err();
    assert!(matches!(error, Error::Io { path, source }
        if path == root.join("absent.json") && source.kind() == std::io::ErrorKind::NotFound));

    let path = root.join("config.json");
    std::fs::write(&path, b"{}").expect("bounded fixture");
    assert_eq!(
        contained_regular_file(&root, "config.json", 2, true).unwrap(),
        Some(path.clone())
    );
    assert_eq!(
        read_required_config(&root).unwrap(),
        (root.clone(), "{}".into())
    );
    assert!(
        matches!(contained_regular_file(&root, "config.json", 1, true).unwrap_err(),
        Error::Artifact { path: actual, reason }
        if actual == path && reason == "config.json exceeds the 1-byte limit")
    );

    let generation_path = root.join("generation_config.json");
    File::create(&generation_path)
        .unwrap()
        .set_len(MAX_GENERATION_CONFIG_BYTES + 1)
        .unwrap();
    assert!(matches!(read_generation_config(&root).unwrap_err(),
        Error::Artifact { path: actual, reason }
        if actual == generation_path && reason == format!("generation_config.json exceeds the {MAX_GENERATION_CONFIG_BYTES}-byte limit")));
    File::create(&path)
        .unwrap()
        .set_len(MAX_CONFIG_BYTES + 1)
        .unwrap();
    assert!(matches!(read_required_config(&root).unwrap_err(),
        Error::Artifact { path: actual, reason }
        if actual == path && reason == format!("config.json exceeds the {MAX_CONFIG_BYTES}-byte limit")));
}

/// Artifact type and text decoding errors identify the actual offending path.
#[test]
fn required_config_preserves_artifact_type_and_encoding_errors() {
    let directory = tempfile::tempdir().expect("temporary artifact root");
    let root = directory.path().canonicalize().expect("canonical root");
    let missing = root.join("missing-root");
    assert!(matches!(read_required_config(&missing).unwrap_err(),
        Error::Io { path, source } if path == missing && source.kind() == std::io::ErrorKind::NotFound));
    let config_path = root.join("config.json");
    std::fs::write(&config_path, [0xff]).unwrap();
    assert!(matches!(read_required_config(&root).unwrap_err(),
        Error::Io { path, source } if path == config_path && source.kind() == std::io::ErrorKind::InvalidData));
    assert!(matches!(read_required_config(&config_path).unwrap_err(),
        Error::Artifact { path, reason } if path == config_path && reason == "model source root must be a directory"));
    let generation_path = root.join("generation_config.json");
    std::fs::create_dir(&generation_path).unwrap();
    assert!(matches!(read_generation_config(&root).unwrap_err(),
        Error::Artifact { path, reason } if path == generation_path && reason == "generation_config.json must be a regular file"));
}

/// In-root aliases are valid, but neither required nor optional files may escape.
#[cfg(unix)]
#[test]
fn automatic_metadata_resolves_symlinks_without_escaping_model_root() {
    use std::os::unix::fs::symlink;

    let directory = tempfile::tempdir().expect("temporary artifact root");
    let outside = tempfile::tempdir().expect("temporary external root");
    let root = directory.path().canonicalize().expect("canonical root");
    let config = root.join("config-data.json");
    std::fs::write(&config, b"{}").unwrap();
    symlink("config-data.json", root.join("config.json")).unwrap();
    assert_eq!(
        contained_regular_file(&root, "config.json", 2, true).unwrap(),
        Some(config)
    );
    assert_eq!(
        read_required_config(&root).unwrap(),
        (root.clone(), "{}".into())
    );

    let external = outside.path().join("generation.json");
    std::fs::write(&external, b"{}").unwrap();
    let external = external.canonicalize().unwrap();
    symlink(&external, root.join("generation_config.json")).unwrap();
    assert!(matches!(read_generation_config(&root).unwrap_err(),
        Error::Artifact { path, reason }
        if path == external && reason == "generation_config.json resolves outside the model source root"));
    symlink(&external, root.join("foreign.json")).unwrap();
    assert!(
        matches!(contained_regular_file(&root, "foreign.json", 2, true).unwrap_err(),
        Error::Artifact { path, reason }
        if path == external && reason == "foreign.json resolves outside the model source root")
    );
}

/// Source options remain independently optional and malformed provenance fails early.
#[test]
fn automatic_source_metadata_validates_optional_provenance() {
    let directory = tempfile::tempdir().expect("temporary artifact root");
    for (repository, revision) in [
        (None, None),
        (Some("owner/model"), None),
        (None, Some("revision")),
    ] {
        let options = AutoLoadOptions {
            repository: repository.map(str::to_owned),
            revision: revision.map(str::to_owned),
            ..AutoLoadOptions::default()
        };
        let source = source_metadata(directory.path(), "qwen3_5_text", &options).unwrap();
        assert_eq!(source.root(), directory.path().canonicalize().unwrap());
        assert_eq!(source.model_type(), "qwen3_5_text");
        assert_eq!(source.repository(), repository);
        assert_eq!(source.revision(), revision);
    }
    for (options, expected_field, expected_reason) in [
        (
            AutoLoadOptions {
                repository: Some(" owner/model".into()),
                ..AutoLoadOptions::default()
            },
            "model_source.repository",
            "must be non-empty and contain no surrounding whitespace",
        ),
        (
            AutoLoadOptions {
                revision: Some("rev\u{0000}ision".into()),
                ..AutoLoadOptions::default()
            },
            "model_source.revision",
            "must not contain control characters",
        ),
    ] {
        assert!(
            matches!(source_metadata(directory.path(), "qwen3_5_text", &options).unwrap_err(),
            Error::Configuration { field, reason } if field == expected_field && reason == expected_reason)
        );
    }
}

/// Explicit generation controls bypass malformed local metadata but remain validated.
#[test]
fn explicit_generation_defaults_bypass_artifacts_and_validate_the_override() {
    let root = tempfile::tempdir().expect("temporary model root");
    std::fs::write(root.path().join("generation_config.json"), b"not JSON").unwrap();
    let explicit = GenerationConfig {
        max_new_tokens: 2,
        ..GenerationConfig::default()
    };
    assert_eq!(
        normalize_generation_defaults(
            root.path(),
            Some(&TokenIds::One(99)),
            Some(&TokenIds::Many(vec![98, 99])),
            8,
            Some(&explicit)
        )
        .unwrap(),
        explicit
    );
    let invalid = GenerationConfig {
        max_new_tokens: 0,
        ..explicit
    };
    assert!(
        matches!(normalize_generation_defaults(root.path(), None, None, 8, Some(&invalid)).unwrap_err(),
        Error::Generation(reason) if reason == "max_new_tokens must be greater than zero")
    );
    let error = normalize_generation_defaults(root.path(), None, None, 8, None).unwrap_err();
    assert!(matches!(error, Error::Json { path, source }
        if path == root.path().canonicalize().unwrap().join("generation_config.json") && source.is_syntax()));
}

/// Token normalization keeps EOS sets and scalar padding distinct at their boundaries.
#[test]
fn generation_special_tokens_preserve_field_context_and_set_semantics() {
    assert_eq!(
        normalize_token_list(None, 8, "eos_token_id").unwrap(),
        Vec::<u32>::new()
    );
    assert_eq!(
        normalize_pad_token(Some(&TokenIds::Many(vec![])), 8).unwrap(),
        None
    );
    assert_eq!(
        normalize_pad_token(Some(&TokenIds::Many(vec![7, 7])), 8).unwrap(),
        Some(7)
    );
    for (tokens, field, expected) in [
        (
            TokenIds::One(8),
            "eos_token_id",
            "token 8 is outside vocabulary size 8",
        ),
        (
            TokenIds::One(8),
            "generation_config.eos_token_id",
            "token 8 is outside vocabulary size 8",
        ),
    ] {
        assert!(
            matches!(normalize_token_list(Some(&tokens), 8, field).unwrap_err(),
            Error::Configuration { field: actual, reason } if actual == field && reason == expected)
        );
    }
    assert!(
        matches!(normalize_pad_token_named(Some(&TokenIds::Many(vec![1, 2])), 8,
        "generation_config.pad_token_id").unwrap_err(), Error::Configuration { field, reason }
        if field == "generation_config.pad_token_id" && reason == "padding requires exactly one token ID")
    );
}

/// Wider host vocabularies cannot smuggle unrepresentable IDs into the u32 API.
#[cfg(target_pointer_width = "64")]
#[test]
fn generation_special_tokens_reject_ids_wider_than_runtime() {
    let oversized = u32::MAX as usize + 1;
    assert_eq!(
        normalize_token_list(
            Some(&TokenIds::One(oversized - 1)),
            oversized,
            "eos_token_id"
        )
        .unwrap(),
        [u32::MAX]
    );
    assert!(
        matches!(normalize_token_list(Some(&TokenIds::One(oversized)), oversized + 1,
        "eos_token_id").unwrap_err(), Error::Configuration { field, reason }
        if field == "eos_token_id" && reason == format!("token {oversized} does not fit the u32 runtime"))
    );
}

/// Local overrides validate active sampling values and retain generation field names.
#[test]
fn local_generation_defaults_validate_active_fields() {
    let root = tempfile::tempdir().expect("temporary model root");
    let path = root.path().join("generation_config.json");
    for (json, expected_reason) in [
        (
            json!({"do_sample": true, "temperature": 0.0}),
            "temperature must be finite and greater than zero",
        ),
        (
            json!({"do_sample": true, "top_p": 0.0}),
            "top_p must be finite and in (0, 1], found 0",
        ),
        (
            json!({"max_new_tokens": 0}),
            "max_new_tokens must be greater than zero",
        ),
    ] {
        std::fs::write(&path, serde_json::to_vec(&json).unwrap()).unwrap();
        assert!(
            matches!(normalize_generation_defaults(root.path(), None, None, 8, None).unwrap_err(),
            Error::Generation(reason) if reason == expected_reason)
        );
    }
    for (json, expected_field, expected_reason) in [
        (
            json!({"eos_token_id": 8}),
            "generation_config.eos_token_id",
            "token 8 is outside vocabulary size 8",
        ),
        (
            json!({"pad_token_id": [1, 2]}),
            "generation_config.pad_token_id",
            "padding requires exactly one token ID",
        ),
    ] {
        std::fs::write(&path, serde_json::to_vec(&json).unwrap()).unwrap();
        assert!(
            matches!(normalize_generation_defaults(root.path(), None, None, 8, None).unwrap_err(),
            Error::Configuration { field, reason } if field == expected_field && reason == expected_reason)
        );
    }
    std::fs::write(
        &path,
        br#"{"do_sample":true,"top_k":2,"eos_token_id":[],"pad_token_id":[]}"#,
    )
    .unwrap();
    let normalized = normalize_generation_defaults(
        root.path(),
        Some(&TokenIds::One(7)),
        Some(&TokenIds::One(0)),
        8,
        None,
    )
    .unwrap();
    assert_eq!(normalized.strategy, SamplingStrategy::Sample);
    assert_eq!(normalized.temperature, 1.0);
    assert_eq!(normalized.top_k, Some(2));
    assert!(normalized.eos_token_ids.is_empty());
    assert_eq!(normalized.pad_token_id, None);
}

/// Absent and explicitly disabled tokenizers do not require decodable artifacts.
#[test]
fn optional_tokenizer_respects_absence_and_explicit_disable() {
    let root = tempfile::tempdir().expect("temporary model root");
    assert!(
        load_optional_tokenizer(root.path(), true)
            .unwrap()
            .is_none()
    );
    std::fs::write(root.path().join("tokenizer.json"), b"not JSON").unwrap();
    assert!(
        load_optional_tokenizer(root.path(), false)
            .unwrap()
            .is_none()
    );
    #[cfg(not(feature = "tokenizers"))]
    assert!(
        load_optional_tokenizer(root.path(), true)
            .unwrap()
            .is_none()
    );
    #[cfg(feature = "tokenizers")]
    assert!(
        matches!(load_optional_tokenizer(root.path(), true).unwrap_err(), Error::Tokenizer(reason)
        if reason.contains("failed to decode") && reason.contains("tokenizer.json"))
    );
}

/// Automatic loading must not suppress decoder errors or follow escaping tokenizer files.
#[cfg(all(feature = "tokenizers", unix))]
#[test]
fn enabled_tokenizer_rejects_escaping_artifact() {
    let root = tempfile::tempdir().expect("temporary model root");
    let outside = tempfile::NamedTempFile::new().expect("external tokenizer fixture");
    let external = outside.path().canonicalize().unwrap();
    std::os::unix::fs::symlink(&external, root.path().join("tokenizer.json")).unwrap();
    assert!(
        matches!(load_optional_tokenizer(root.path(), true).unwrap_err(),
        Error::Artifact { path, reason } if path == external && reason == "tokenizer.json resolves outside the model source root")
    );
    assert!(
        load_optional_tokenizer(root.path(), false)
            .unwrap()
            .is_none()
    );
}

/// Invalid topologies fail in family parsing before tokenizer or tensor discovery.
#[test]
fn automatic_loads_preserve_family_config_error_context() {
    let root = tempfile::tempdir().expect("temporary model root");
    for (model_type, family) in [
        ("deepseek_v2", "deepseek_v2"),
        ("muse_glimmer_text", "muse_glimmer"),
        ("nemotron_h", "nemotron_h"),
        ("qwen3_5_text", "qwen3_5"),
        ("qwen3_moe", "qwen3_moe"),
    ] {
        write_auto_metadata(
            root.path(),
            &json!({"model_type": model_type, "vocab_size": 0}),
        );
        let error = AutoCausalLm::<TestBackend>::load_local(
            root.path(),
            &AutoLoadOptions::default(),
            &Default::default(),
        )
        .unwrap_err();
        assert!(matches!(error, Error::Configuration { field, reason }
            if field == "auto_causal.config" && reason == format!("{family}.vocab_size: must be nonzero")));
    }
    write_auto_metadata(root.path(), &json!({"model_type": "t5", "vocab_size": 0}));
    assert!(
        matches!(AutoSeq2SeqLm::<TestBackend>::load_local(root.path(), &AutoLoadOptions::default(),
        &Default::default()).unwrap_err(), Error::Configuration { field, reason }
        if field == "auto_seq2seq.config" && reason == "t5.vocab_size: must be nonzero")
    );
}

/// Cross-family options fail before touching otherwise missing model artifacts.
#[test]
fn automatic_load_rejects_wrong_family_options_and_registry() {
    let root = tempfile::tempdir().expect("temporary model root");
    write_auto_metadata(root.path(), &json!({"model_type": "t5"}));
    assert!(
        matches!(AutoCausalLm::<TestBackend>::load_local(root.path(), &AutoLoadOptions::default(),
        &Default::default()).unwrap_err(), Error::UnsupportedArchitecture(reason)
        if reason == format!("causal model_type `t5` is not executable; supported values: {}", AutoCausalLm::<TestBackend>::MODEL_TYPES.join(", ")))
    );
    let mtp = AutoLoadOptions {
        qwen_mtp: true,
        ..AutoLoadOptions::default()
    };
    assert!(
        matches!(AutoCausalLm::<TestBackend>::load_local(root.path(), &mtp,
        &Default::default()).unwrap_err(), Error::Configuration { field, reason }
        if field == "auto_causal.qwen_mtp" && reason == "Qwen MTP is defined only for qwen3_5_text checkpoints")
    );
    write_auto_metadata(root.path(), &json!({"model_type": "qwen3_moe"}));
    assert!(
        matches!(AutoSeq2SeqLm::<TestBackend>::load_local(root.path(), &AutoLoadOptions::default(),
        &Default::default()).unwrap_err(), Error::UnsupportedArchitecture(reason)
        if reason == "seq2seq model_type `qwen3_moe` is not executable; supported values: t5")
    );
    let packed = AutoLoadOptions {
        quantized_burnpack: Some("missing.bpk".into()),
        ..AutoLoadOptions::default()
    };
    assert!(
        matches!(AutoSeq2SeqLm::<TestBackend>::load_local(root.path().join("missing-root"), &packed,
        &Default::default()).unwrap_err(), Error::BackendCapability(reason)
        if reason == "packed Burnpack loading is currently certified only for causal model types: muse_glimmer_text, qwen3_5_text, qwen3_moe")
    );
}

/// Returns one full-attention Qwen layer with a checkpoint-declared MTP body.
fn tiny_automatic_qwen_config() -> Qwen3_5TextConfig {
    Qwen3_5TextConfig {
        vocab_size: 16,
        hidden_size: 8,
        intermediate_size: 16,
        num_hidden_layers: 1,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: 4,
        max_position_embeddings: 16,
        layer_types: vec![Qwen3_5LayerType::FullAttention],
        linear_key_head_dim: 4,
        linear_value_head_dim: 4,
        linear_num_key_heads: 1,
        linear_num_value_heads: 2,
        linear_conv_kernel_dim: 3,
        partial_rotary_factor: Some(1.0),
        rope_theta: Some(10_000.0),
        mtp_num_hidden_layers: 1,
        tie_word_embeddings: false,
        bos_token_id: Some(TokenIds::One(1)),
        eos_token_id: Some(TokenIds::One(2)),
        pad_token_id: Some(TokenIds::One(0)),
        ..Qwen3_5TextConfig::default()
    }
}

/// Base-only loading cannot expose an unconsumed MTP graph in either Qwen namespace.
#[test]
fn automatic_qwen_namespaces_preserve_explicit_mtp_consumption() {
    let config = tiny_automatic_qwen_config();
    for source in [
        Qwen3_5CheckpointSource::TextOnly,
        Qwen3_5CheckpointSource::Multimodal,
    ] {
        let root = tempfile::tempdir().expect("temporary Qwen checkpoint");
        let contract = config
            .checkpoint_contract(Qwen3_5CheckpointOptions {
                source,
                include_mtp: true,
            })
            .unwrap();
        write_auto_metadata(root.path(), &config);
        write_contract_checkpoint(
            root.path(),
            contract.expected_tensors(),
            contract.mappings(),
        );
        for include_mtp in [false, true] {
            let options = AutoLoadOptions {
                load_tokenizer: false,
                qwen_mtp: include_mtp,
                ..AutoLoadOptions::default()
            };
            let loaded =
                AutoCausalLm::<TestBackend>::load_local(root.path(), &options, &Default::default())
                    .unwrap();
            let AutoCausalLm::Qwen3_5(model) = loaded.model() else {
                panic!("Qwen automatic variant")
            };
            assert_eq!(model.mtp.is_some(), include_mtp);
            assert!(loaded.load_report().is_strictly_valid());
            assert!(loaded.load_report().missing.is_empty());
            assert!(loaded.load_report().unexpected.is_empty());
            if include_mtp {
                assert_strict_report(
                    loaded.load_report(),
                    contract.expected_tensors(),
                    contract.mappings(),
                );
            } else {
                assert_eq!(loaded.load_report().allowlisted_ignored.len(), 15);
                assert!(
                    loaded
                        .load_report()
                        .allowlisted_ignored
                        .iter()
                        .all(|name| name.starts_with("mtp."))
                );
            }
        }
    }
}

/// Namespace errors and sidecar paths are preserved by the complete automatic loader.
#[test]
fn automatic_causal_load_preserves_namespace_and_sidecar_errors() {
    for (config, family) in [
        (
            serde_json::to_value(tiny_automatic_qwen_config()).unwrap(),
            "Qwen",
        ),
        (serde_json::to_value(packed_muse_config()).unwrap(), "Muse"),
    ] {
        let root = tempfile::tempdir().expect("temporary model root");
        write_auto_metadata(root.path(), &config);
        for (names, expected) in [
            (
                vec!["foreign.weight"],
                format!(
                    "{family} checkpoint contains neither official language parameter namespace"
                ),
            ),
            (
                vec![
                    "model.embed_tokens.weight",
                    "model.language_model.embed_tokens.weight",
                ],
                format!(
                    "{family} checkpoint ambiguously contains wrapped and text-only language namespaces"
                ),
            ),
        ] {
            let expected_tensors = names
                .iter()
                .map(|name| ExpectedTensor::new(*name, vec![1]))
                .collect::<Vec<_>>();
            let mut mappings = MappingPlan::default();
            for name in names {
                mappings.map(name, name);
            }
            write_contract_checkpoint(root.path(), &expected_tensors, &mappings);
            assert!(
                matches!(AutoCausalLm::<TestBackend>::load_local(root.path(), &AutoLoadOptions::default(),
                &Default::default()).unwrap_err(), Error::Checkpoint(reason) if reason == expected)
            );
        }
    }

    let root = tempfile::tempdir().expect("temporary Qwen checkpoint");
    let config = tiny_automatic_qwen_config();
    let contract = config
        .checkpoint_contract(Qwen3_5CheckpointOptions {
            source: Qwen3_5CheckpointSource::TextOnly,
            include_mtp: false,
        })
        .unwrap();
    write_auto_metadata(root.path(), &config);
    write_contract_checkpoint(
        root.path(),
        contract.expected_tensors(),
        contract.mappings(),
    );
    for supplied in [
        PathBuf::from("missing.bpk"),
        root.path()
            .canonicalize()
            .unwrap()
            .join("absolute-missing.bpk"),
    ] {
        let resolved = if supplied.is_absolute() {
            supplied.clone()
        } else {
            root.path().canonicalize().unwrap().join(&supplied)
        };
        let options = AutoLoadOptions {
            load_tokenizer: false,
            quantized_burnpack: Some(supplied),
            ..AutoLoadOptions::default()
        };
        assert!(
            matches!(AutoCausalLm::<TestBackend>::load_local(root.path(), &options, &Default::default()).unwrap_err(),
            Error::Artifact { path, reason } if path == resolved && !reason.is_empty())
        );
    }
}

/// The DeepSeek branch performs strict discovery, installation, and execution.
#[test]
fn automatic_deepseek_loads_complete_strict_fixture() {
    let root = tempfile::tempdir().expect("temporary model root");
    let mut config = tiny_deepseek_config();
    config.eos_token_id = Some(TokenIds::Many(vec![3, 2, 3]));
    config.pad_token_id = Some(TokenIds::One(0));
    let contract = config
        .checkpoint_contract(DeepseekV2CheckpointOptions::default())
        .expect("valid DeepSeek contract");
    write_auto_metadata(root.path(), &config);
    write_contract_checkpoint(
        root.path(),
        contract.expected_tensors(),
        contract.mappings(),
    );

    let repository = "deepseek-ai/tiny-deepseek-v2-fixture";
    let revision = "deepseek-fixture-revision";
    let loaded = AutoCausalLm::<TestBackend>::load_local(
        root.path(),
        &fixture_options(repository, revision),
        &Default::default(),
    )
    .expect("automatic DeepSeek load");

    assert!(matches!(loaded.model(), AutoCausalLm::DeepseekV2(_)));
    assert!(loaded.quantized_load_audit().is_none());
    assert_strict_report(
        loaded.load_report(),
        contract.expected_tensors(),
        contract.mappings(),
    );
    assert_bundle_metadata(
        &loaded,
        root.path(),
        "deepseek_v2",
        repository,
        revision,
        &[2, 3],
        0,
    );
    assert_finite_forward(&loaded, config.vocab_size);
}

/// The Muse text-only branch classifies its namespace and streams every tensor.
#[test]
fn automatic_muse_glimmer_loads_complete_strict_fixture() {
    let root = tempfile::tempdir().expect("temporary model root");
    let config = MuseGlimmerTextConfig {
        vocab_size: 16,
        hidden_size: 4,
        intermediate_size: 8,
        num_hidden_layers: 2,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: 2,
        max_position_embeddings: 16,
        sliding_window: Some(4),
        layer_types: vec![
            MuseGlimmerLayerType::SlidingAttention,
            MuseGlimmerLayerType::FullAttention,
        ],
        layer_rope_theta: vec![10_000.0, 0.0],
        bos_token_id: Some(TokenIds::One(1)),
        eos_token_id: Some(TokenIds::Many(vec![5, 4, 5])),
        pad_token_id: Some(TokenIds::One(0)),
        tie_word_embeddings: false,
        ..MuseGlimmerTextConfig::default()
    };
    let family_options = MuseGlimmerCheckpointOptions {
        source: MuseGlimmerCheckpointSource::TextOnly,
    };
    let contract = config
        .checkpoint_contract(family_options)
        .expect("valid Muse contract");
    write_auto_metadata(root.path(), &config);
    write_contract_checkpoint(
        root.path(),
        contract.expected_tensors(),
        contract.mappings(),
    );

    let repository = "meta-models/tiny-muse-glimmer-fixture";
    let revision = "muse-fixture-revision";
    let loaded = AutoCausalLm::<TestBackend>::load_local(
        root.path(),
        &fixture_options(repository, revision),
        &Default::default(),
    )
    .expect("automatic Muse load");

    assert!(matches!(loaded.model(), AutoCausalLm::MuseGlimmer(_)));
    assert_strict_report(
        loaded.load_report(),
        contract.expected_tensors(),
        contract.mappings(),
    );
    assert_bundle_metadata(
        &loaded,
        root.path(),
        "muse_glimmer_text",
        repository,
        revision,
        &[4, 5],
        0,
    );
    assert_finite_forward(&loaded, config.vocab_size);
}

/// Q8 and Q4 install every Muse projection through the packed execution path.
#[test]
fn automatic_muse_glimmer_loads_q8_q4_and_preserves_all_execution_paths() {
    let root = tempfile::tempdir().expect("temporary model root");
    let config = packed_muse_config();
    let family_options = MuseGlimmerCheckpointOptions {
        source: MuseGlimmerCheckpointSource::TextOnly,
    };
    let contract = config
        .checkpoint_contract(family_options)
        .expect("valid packed Muse contract");
    write_auto_metadata(root.path(), &config);
    write_patterned_muse_checkpoint(
        root.path(),
        contract.expected_tensors(),
        contract.mappings(),
    );
    let checkpoint = CheckpointManifest::discover(root.path(), &LoadOptions::default())
        .expect("discover patterned Muse checkpoint");
    let plan = contract
        .build_load_plan(&checkpoint, &LoadOptions::default())
        .expect("plan patterned Muse checkpoint");
    let expected_quantized = (0..config.num_hidden_layers)
        .flat_map(|layer| {
            [
                format!("model.layers.{layer}.self_attn.q_proj.weight"),
                format!("model.layers.{layer}.self_attn.k_proj.weight"),
                format!("model.layers.{layer}.self_attn.v_proj.weight"),
                format!("model.layers.{layer}.self_attn.gate_proj.weight"),
                format!("model.layers.{layer}.self_attn.o_proj.weight"),
                format!("model.layers.{layer}.mlp.gate_proj.weight"),
                format!("model.layers.{layer}.mlp.up_proj.weight"),
                format!("model.layers.{layer}.mlp.down_proj.weight"),
            ]
        })
        .chain(["lm_head.weight".to_owned()])
        .collect::<BTreeSet<_>>();
    assert_eq!(expected_quantized.len(), 17);

    let full = AutoCausalLm::<TestBackend>::load_local(
        root.path(),
        &fixture_options("meta-models/tiny-muse", "full-precision"),
        &Default::default(),
    )
    .expect("load source-precision Muse fixture");
    let full_paths = execute_muse_paths(full.model());
    full_paths[0].assert_approx_eq::<f32>(&full_paths[1], Tolerance::absolute(3e-4));
    full_paths[2].assert_approx_eq::<f32>(&full_paths[3], Tolerance::absolute(3e-4));

    for scheme in [SymmetricQuantization::Q8, SymmetricQuantization::Q4] {
        let label = match scheme {
            SymmetricQuantization::Q8 => "q8",
            SymmetricQuantization::Q4 => "q4",
        };
        let burnpack = root.path().join(format!("muse-{label}.bpk"));
        let converted = quantize_local_checkpoint_to_burnpack(
            &checkpoint,
            &plan,
            &burnpack,
            QuantizedCheckpointOptions {
                weights: WeightQuantizationConfig {
                    scheme,
                    group_size: QuantizationGroupSize::new(64).expect("supported G64"),
                },
                ..QuantizedCheckpointOptions::default()
            },
        )
        .unwrap_or_else(|error| panic!("convert Muse {label}: {error}"));
        assert!(converted.report.is_strictly_valid());
        assert_eq!(converted.report.applied.len(), 35);
        assert_eq!(converted.report.quantized.len(), 17);
        assert_eq!(converted.excluded.len(), 18);
        assert_eq!(
            converted
                .report
                .quantized
                .iter()
                .map(|record| record.target.clone())
                .collect::<BTreeSet<_>>(),
            expected_quantized
        );
        let exclusion_count = |reason| {
            converted
                .excluded
                .iter()
                .filter(|record| record.reason == reason)
                .count()
        };
        assert_eq!(exclusion_count(QuantizationExclusion::Embedding), 1);
        assert_eq!(exclusion_count(QuantizationExclusion::Normalization), 9);
        assert_eq!(exclusion_count(QuantizationExclusion::Bias), 8);
        for pass in [
            converted.provenance.pre_execution,
            converted.provenance.post_execution,
        ] {
            let pass = pass.expect("v2 conversion hashes both sides of pass two");
            assert_eq!(pass.passes_completed, 1);
            assert_eq!(pass.shards_hashed, 1);
            assert_eq!(pass.bytes_hashed, checkpoint.artifact_bytes);
        }
        let artifact = QuantizedBurnpackArtifact::open(&burnpack)
            .unwrap_or_else(|error| panic!("open Muse {label}: {error}"));
        assert_eq!(
            artifact.manifest().packed_execution,
            PackedExecutionSupport::BurnQFloatMatmul
        );
        assert_eq!(artifact.manifest().tensors.len(), 17);
        drop(artifact);

        let loaded = AutoCausalLm::<TestBackend>::load_local(
            root.path(),
            &AutoLoadOptions {
                quantized_burnpack: Some(burnpack.clone()),
                ..fixture_options("meta-models/tiny-muse", label)
            },
            &Default::default(),
        )
        .unwrap_or_else(|error| panic!("automatically load Muse {label}: {error}"));
        let model = match loaded.model() {
            AutoCausalLm::MuseGlimmer(model) => model,
            model => panic!("expected Muse automatic variant, found {model:?}"),
        };
        assert_eq!(model.model.embed_tokens.weight.val().dtype(), DType::F32);
        assert_eq!(model.model.norm.weight.val().dtype(), DType::F32);
        for (layer_index, layer) in model.model.layers.iter().enumerate() {
            for (projection, linear) in [
                ("attention query", &layer.self_attn.q_proj),
                ("attention key", &layer.self_attn.k_proj),
                ("attention value", &layer.self_attn.v_proj),
                ("attention gate", &layer.self_attn.gate_proj),
                ("attention output", &layer.self_attn.o_proj),
                ("FFN gate", &layer.mlp.gate_proj),
                ("FFN up", &layer.mlp.up_proj),
                ("FFN down", &layer.mlp.down_proj),
            ] {
                assert!(
                    matches!(linear.weight.val().dtype(), DType::QFloat(_)),
                    "Muse layer {layer_index} {projection} must retain packed storage"
                );
            }
            for (projection, linear) in [
                ("query", &layer.self_attn.q_proj),
                ("key", &layer.self_attn.k_proj),
                ("value", &layer.self_attn.v_proj),
                ("output", &layer.self_attn.o_proj),
            ] {
                assert_eq!(
                    linear
                        .bias
                        .as_ref()
                        .unwrap_or_else(|| panic!("configured {projection} bias"))
                        .val()
                        .dtype(),
                    DType::F32
                );
            }
            for norm in [
                &layer.input_layernorm,
                &layer.post_attention_layernorm,
                &layer.pre_feedforward_layernorm,
                &layer.post_feedforward_layernorm,
            ] {
                assert_eq!(norm.weight.val().dtype(), DType::F32);
            }
        }
        assert!(matches!(
            model
                .lm_head
                .as_ref()
                .expect("independent Muse LM head")
                .weight
                .val()
                .dtype(),
            DType::QFloat(_)
        ));
        assert!(loaded.load_report().is_strictly_valid());
        assert_eq!(loaded.load_report().applied.len(), 35);
        assert_eq!(loaded.load_report().quantized.len(), 17);
        let execution = loaded
            .execution_stats()
            .expect("packed Muse load retains source counters");
        assert_eq!(execution.tensors_applied, 35);
        assert_eq!(execution.source_reads, 18);
        assert_eq!(execution.source.shard_opens, 1);
        assert_eq!(execution.source.max_open_shards, 1);
        let audit = loaded
            .quantized_load_audit()
            .expect("packed Muse load retains sidecar counters");
        assert_eq!(
            audit.format_version(),
            BURNPACK_QUANTIZATION_FORMAT_VERSION_V2
        );
        assert_eq!(audit.preflight().passes_completed, 1);
        assert_eq!(audit.preflight().tensor_reads, 17);
        assert_eq!(audit.packed().tensors_applied, 17);
        assert_eq!(audit.packed().tensor_reads, 17);
        assert!(audit.provenance().pre_execution.is_some());
        assert!(audit.provenance().post_execution.is_some());

        let packed_paths = execute_muse_paths(loaded.model());
        packed_paths[0].assert_approx_eq::<f32>(&packed_paths[1], Tolerance::absolute(3e-4));
        packed_paths[2].assert_approx_eq::<f32>(&packed_paths[3], Tolerance::absolute(3e-4));
        for (packed, source) in packed_paths.iter().zip(&full_paths) {
            packed.assert_approx_eq::<f32>(source, Tolerance::absolute(5e-4));
        }
    }
}

/// A sidecar whose groups cross Muse matrix rows is reference-only, not executable.
#[test]
fn automatic_muse_glimmer_rejects_unaligned_reference_only_burnpack() {
    let root = tempfile::tempdir().expect("temporary model root");
    let config = MuseGlimmerTextConfig {
        vocab_size: 64,
        hidden_size: 32,
        intermediate_size: 32,
        num_hidden_layers: 1,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: 16,
        max_position_embeddings: 8,
        sliding_window: Some(4),
        layer_types: vec![MuseGlimmerLayerType::FullAttention],
        layer_rope_theta: vec![0.0],
        tie_word_embeddings: false,
        bos_token_id: None,
        eos_token_id: None,
        ..MuseGlimmerTextConfig::default()
    };
    let family_options = MuseGlimmerCheckpointOptions {
        source: MuseGlimmerCheckpointSource::TextOnly,
    };
    let contract = config
        .checkpoint_contract(family_options)
        .expect("valid unaligned Muse contract");
    write_auto_metadata(root.path(), &config);
    write_patterned_muse_checkpoint(
        root.path(),
        contract.expected_tensors(),
        contract.mappings(),
    );
    let checkpoint = CheckpointManifest::discover(root.path(), &LoadOptions::default())
        .expect("discover unaligned Muse checkpoint");
    let plan = contract
        .build_load_plan(&checkpoint, &LoadOptions::default())
        .expect("plan unaligned Muse checkpoint");
    let burnpack = root.path().join("unaligned.bpk");
    quantize_local_checkpoint_to_burnpack(
        &checkpoint,
        &plan,
        &burnpack,
        QuantizedCheckpointOptions::default(),
    )
    .expect("reference conversion remains valid");
    let artifact = QuantizedBurnpackArtifact::open(&burnpack).expect("open reference sidecar");
    assert_eq!(
        artifact.manifest().packed_execution,
        PackedExecutionSupport::Unavailable
    );
    drop(artifact);

    let error = AutoCausalLm::<TestBackend>::load_local(
        root.path(),
        &AutoLoadOptions {
            load_tokenizer: false,
            quantized_burnpack: Some(burnpack),
            ..AutoLoadOptions::default()
        },
        &Default::default(),
    )
    .expect_err("automatic execution must reject row-unaligned sidecars");
    assert!(
        matches!(&error, Error::Checkpoint(message) if message.contains("packed quantized execution is unavailable")),
        "unexpected error: {error}"
    );
}

/// Builds the tiny Nemotron-H schedule used by the automatic-load fixtures.
fn loadable_nemotron_config() -> NemotronHConfig {
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
        tie_word_embeddings: false,
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
        num_nextn_predict_layers: 0,
        bos_token_id: Some(TokenIds::One(1)),
        eos_token_id: Some(TokenIds::Many(vec![7, 6, 7])),
        pad_token_id: Some(TokenIds::One(0)),
        ..NemotronHConfig::default()
    }
}

/// The Nemotron branch loads a complete heterogeneous hybrid schedule.
#[test]
fn automatic_nemotron_h_loads_complete_strict_fixture() {
    let root = tempfile::tempdir().expect("temporary model root");
    let config = loadable_nemotron_config();
    let contract = config
        .checkpoint_contract(NemotronHCheckpointOptions::default())
        .expect("valid Nemotron-H contract");
    write_auto_metadata(root.path(), &config);
    write_contract_checkpoint(
        root.path(),
        contract.expected_tensors(),
        contract.mappings(),
    );

    let repository = "nvidia/tiny-nemotron-h-fixture";
    let revision = "nemotron-fixture-revision";
    let loaded = AutoCausalLm::<TestBackend>::load_local(
        root.path(),
        &fixture_options(repository, revision),
        &Default::default(),
    )
    .expect("automatic Nemotron-H load");

    assert!(matches!(loaded.model(), AutoCausalLm::NemotronH(_)));
    assert_strict_report(
        loaded.load_report(),
        contract.expected_tensors(),
        contract.mappings(),
    );
    assert_bundle_metadata(
        &loaded,
        root.path(),
        "nemotron_h",
        repository,
        revision,
        &[6, 7],
        0,
    );
    assert_finite_forward(&loaded, config.vocab_size);
}

/// The encoder-decoder registry retains the same pass-two audit as causal loads.
#[test]
fn automatic_t5_load_retains_execution_stats() {
    let root = tempfile::tempdir().expect("temporary model root");
    let config = T5Config {
        vocab_size: 13,
        d_model: 4,
        d_kv: 2,
        d_ff: 6,
        num_layers: 2,
        num_decoder_layers: Some(1),
        num_heads: 2,
        relative_attention_num_buckets: 8,
        relative_attention_max_distance: 16,
        feed_forward_proj: T5FeedForwardProjection::GatedGelu,
        dense_act_fn: None,
        is_gated_act: None,
        tie_word_embeddings: false,
        ..T5Config::default()
    };
    let contract = config
        .checkpoint_contract()
        .expect("valid tiny T5 contract");
    write_auto_metadata(root.path(), &config);
    write_contract_checkpoint(
        root.path(),
        contract.expected_tensors(),
        contract.mappings(),
    );

    let loaded = AutoSeq2SeqLm::<TestBackend>::load_local(
        root.path(),
        &fixture_options("google/tiny-flan-fixture", "flan-fixture-revision"),
        &Default::default(),
    )
    .expect("automatic T5 load");

    assert!(matches!(loaded.model(), AutoSeq2SeqLm::T5(_)));
    assert_strict_report(
        loaded.load_report(),
        contract.expected_tensors(),
        contract.mappings(),
    );
    assert_execution_audit(&loaded);
    assert!(loaded.quantized_load_audit().is_none());
    assert_eq!(loaded.source().model_type(), "t5");
    assert_eq!(loaded.generation_defaults().eos_token_ids, [1]);
    assert_eq!(loaded.generation_defaults().pad_token_id, Some(0));
}

/// Loads the pinned real FLAN checkpoint through the automatic public API.
#[test]
#[ignore = "requires pinned google/flan-t5-small SafeTensors artifacts"]
fn automatic_flan_load_bundles_and_executes_real_weights() {
    let root = std::env::var_os("TRANSFORMERS_FLAN_T5_SMALL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("models/google/flan-t5-small-0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab")
        });
    let options = AutoLoadOptions {
        repository: Some("google/flan-t5-small".into()),
        revision: Some("0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab".into()),
        ..AutoLoadOptions::default()
    };
    let device = Default::default();
    let loaded = AutoSeq2SeqLm::<TestBackend>::load_local(&root, &options, &device)
        .expect("automatic FLAN load must be strict");
    assert_eq!(loaded.source().repository(), Some("google/flan-t5-small"));
    assert_eq!(
        loaded.source().revision(),
        Some("0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab")
    );
    assert_eq!(loaded.load_report().applied.len(), 190);
    assert_eq!(loaded.generation_defaults().eos_token_ids, [1]);
    assert_eq!(loaded.generation_defaults().pad_token_id, Some(0));
    #[cfg(feature = "tokenizers")]
    assert!(loaded.tokenizer().is_some());
    #[cfg(not(feature = "tokenizers"))]
    assert!(loaded.tokenizer().is_none());

    let output = loaded
        .model()
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
        .expect("automatic FLAN model must execute");
    assert_eq!(output.decoder.logits.dims(), [1, 1, 32_128]);
    assert!(
        output
            .decoder
            .logits
            .to_data()
            .as_slice::<f32>()
            .expect("F32 reference backend")
            .iter()
            .all(|value| value.is_finite())
    );
}

/// NVIDIA's legacy bare `Infinity` literal loads through the automatic registry (AUTO-1).
#[test]
fn automatic_nemotron_h_accepts_legacy_infinity_literal() {
    let root = tempfile::tempdir().expect("temporary model root");
    let config = loadable_nemotron_config();
    let contract = config
        .checkpoint_contract(NemotronHCheckpointOptions::default())
        .expect("valid Nemotron-H contract");
    write_auto_metadata(root.path(), &config);
    write_contract_checkpoint(
        root.path(),
        contract.expected_tensors(),
        contract.mappings(),
    );
    let config_path = root.path().join("config.json");
    let json = std::fs::read_to_string(&config_path).unwrap();
    let legacy = json.replace(
        "\"time_step_limit\":[0.0,null]",
        "\"time_step_limit\":[0.0,Infinity]",
    );
    assert_ne!(legacy, json, "fixture must contain the unbounded limit");
    std::fs::write(&config_path, legacy).unwrap();

    let loaded = AutoCausalLm::<TestBackend>::load_local(
        root.path(),
        &fixture_options("nvidia/tiny-nemotron-h-fixture", "legacy-literal"),
        &Default::default(),
    )
    .expect("legacy Nemotron-H config must load like its family parser accepts it");
    assert!(matches!(loaded.model(), AutoCausalLm::NemotronH(_)));
    let crate::models::AutoCausalConfigRef::NemotronH(parsed) = loaded.model().config() else {
        panic!("Nemotron-H model must expose its configuration");
    };
    assert_eq!(parsed.time_step_limit, [Some(0.0), None]);
}

/// Sampling defaults follow the pinned Hugging Face `GenerationConfig` semantics (AUTO-2).
#[test]
fn sampling_defaults_follow_hugging_face_generation() {
    let root = tempfile::tempdir().expect("temporary model root");
    let path = root.path().join("generation_config.json");
    for (json, vocabulary_size, top_k, top_p) in [
        // Unset top_k means Hugging Face's default of 50.
        (
            json!({"do_sample": true, "temperature": 0.3, "top_p": 0.95}),
            64,
            Some(50),
            Some(0.95),
        ),
        // The default and explicit values are clamped to the vocabulary.
        (json!({"do_sample": true}), 16, Some(16), None),
        (json!({"do_sample": true, "top_k": 9}), 8, Some(8), None),
        // Zero disables top-k; top_p >= 1 disables nucleus filtering.
        (
            json!({"do_sample": true, "top_k": 0, "top_p": 1.0}),
            64,
            None,
            None,
        ),
        (
            json!({"do_sample": true, "top_k": null, "top_p": 1.5}),
            64,
            Some(50),
            None,
        ),
        // Greedy decoding ignores every warper.
        (
            json!({"do_sample": false, "top_k": 5, "top_p": 0.5}),
            64,
            None,
            None,
        ),
        (json!({}), 64, None, None),
    ] {
        std::fs::write(&path, serde_json::to_vec(&json).unwrap()).unwrap();
        let config = normalize_generation_defaults(root.path(), None, None, vocabulary_size, None)
            .unwrap_or_else(|error| panic!("{json}: {error}"));
        assert_eq!((config.top_k, config.top_p), (top_k, top_p), "{json}");
    }
    // Python's `json` writes NaN as a bare literal, which is not JSON: the file
    // fails to parse before any sampling setting is interpreted.
    std::fs::write(&path, br#"{"do_sample": true, "top_p": NaN}"#).unwrap();
    let error = normalize_generation_defaults(root.path(), None, None, 8, None).unwrap_err();
    assert!(matches!(error, Error::Json { .. }), "{error}");
}

/// A NaN `top_p` is kept so validation rejects it, rather than silently disabling nucleus sampling.
///
/// JSON cannot spell NaN, so the parsed configuration is built directly; this
/// exercises the `top_p < 1` filter in `apply_local_generation_config`.
#[test]
fn nan_top_p_is_rejected_instead_of_disabling_nucleus_sampling() {
    let applied = |top_p: f64| {
        let local = LocalGenerationConfig {
            do_sample: Some(true),
            top_p: Some(top_p),
            ..LocalGenerationConfig::default()
        };
        let mut config = GenerationConfig::default();
        apply_local_generation_config(&mut config, &local, 64).unwrap();
        config
    };
    let config = applied(f64::NAN);
    assert!(config.top_p.is_some_and(f64::is_nan));
    let error = config.validate(64).unwrap_err();
    assert!(
        error.to_string().contains("top_p must be finite"),
        "{error}"
    );
    // Values at or above one disable nucleus filtering, as Hugging Face's
    // warper condition `top_p < 1.0` does.
    for top_p in [1.0, 1.5, f64::INFINITY] {
        let config = applied(top_p);
        assert_eq!(config.top_p, None, "{top_p}");
        config.validate(64).unwrap();
    }
    // Non-positive values are kept so validation rejects them, as Hugging Face's warper does.
    for top_p in [0.0, -0.5, f64::NEG_INFINITY] {
        let config = applied(top_p);
        assert_eq!(config.top_p, Some(top_p));
        assert!(config.validate(64).is_err(), "{top_p}");
    }
}

/// Behavior-changing settings without an equivalent are reported, neutral ones are not.
#[test]
fn unsupported_generation_settings_are_reported_not_dropped() {
    let root = tempfile::tempdir().expect("temporary model root");
    let path = root.path().join("generation_config.json");
    std::fs::write(
        &path,
        serde_json::to_vec(&json!({
            "do_sample": false,
            "num_beams": 4,
            "max_length": 512,
            "no_repeat_ngram_size": 3,
            "min_p": 0.1,
            "typical_p": 0.5,
            "length_penalty": 1,
            "bad_words_ids": [],
            "suppress_tokens": null,
            "renormalize_logits": false,
            "num_return_sequences": 1,
            "unknown_future_field": true
        }))
        .unwrap(),
    )
    .unwrap();
    let resolved = resolve_generation_defaults(root.path(), None, None, 64, None).unwrap();
    assert_eq!(
        resolved.unsupported,
        BTreeMap::from([
            ("max_length".to_owned(), json!(512)),
            ("no_repeat_ngram_size".to_owned(), json!(3)),
            ("num_beams".to_owned(), json!(4)),
        ])
    );
    // Sampling-only warpers matter once sampling is enabled.
    std::fs::write(
        &path,
        br#"{"do_sample": true, "min_p": 0.1, "typical_p": 1.0, "epsilon_cutoff": 0.0003}"#,
    )
    .unwrap();
    let resolved = resolve_generation_defaults(root.path(), None, None, 64, None).unwrap();
    assert_eq!(
        resolved.unsupported.keys().collect::<Vec<_>>(),
        ["epsilon_cutoff", "min_p"]
    );
    // Explicit defaults bypass the file and report nothing.
    let explicit = GenerationConfig::default();
    let resolved =
        resolve_generation_defaults(root.path(), None, None, 64, Some(&explicit)).unwrap();
    assert!(resolved.unsupported.is_empty());
}

/// Automatic loads attach the unsupported-settings report to the bundle.
#[test]
fn automatic_loads_attach_unsupported_generation_settings() {
    let root = tempfile::tempdir().expect("temporary model root");
    let config = loadable_nemotron_config();
    let contract = config
        .checkpoint_contract(NemotronHCheckpointOptions::default())
        .expect("valid Nemotron-H contract");
    write_auto_metadata(root.path(), &config);
    write_contract_checkpoint(
        root.path(),
        contract.expected_tensors(),
        contract.mappings(),
    );
    std::fs::write(
        root.path().join("generation_config.json"),
        br#"{"max_new_tokens":5,"num_beams":2}"#,
    )
    .unwrap();
    let loaded = AutoCausalLm::<TestBackend>::load_local(
        root.path(),
        &fixture_options("nvidia/tiny-nemotron-h-fixture", "beams"),
        &Default::default(),
    )
    .unwrap();
    assert_eq!(
        loaded.unsupported_generation_settings(),
        &BTreeMap::from([("num_beams".to_owned(), json!(2))])
    );
}

/// Every Llama-family `model_type` loads strictly through the one registry entry.
#[test]
fn llama_family_aliases_load_through_one_registry_entry() {
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny_goldens");
    for (case, model_type) in [
        ("llama3", "llama"),
        ("mistral-sliding", "mistral"),
        ("qwen2", "qwen2"),
        ("qwen3", "qwen3"),
        ("smollm3", "smollm3"),
        ("olmo2", "olmo2"),
    ] {
        let loaded = AutoCausalLm::<TestBackend>::load_local(
            fixtures.join(case),
            &fixture_options("fixture/tiny-goldens", case),
            &Default::default(),
        )
        .unwrap_or_else(|error| panic!("{case}: {error}"));
        assert_eq!(loaded.source().model_type(), model_type);
        assert!(loaded.load_report().is_strictly_valid(), "{case}");
        let model = loaded.model();
        assert_eq!(model.family(), AutoCausalConfig::Llama);
        assert_eq!(model.config().model_type(), "llama");
        let crate::models::AutoCausalConfigRef::Llama(config) = model.config() else {
            panic!("{case}: wrong registry variant");
        };
        assert_eq!(config.model_type, model_type);
        let mut state = model
            .new_decode_state(CacheConfig::new(1, 4).unwrap(), 1, &Default::default())
            .unwrap();
        let input = CausalLmInput {
            token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[3, 4]], &Default::default()),
            token_mask: None,
            position_ids: None,
        };
        let logits = model
            .prefill(input, &mut state, &ForwardOptions::default())
            .unwrap()
            .logits;
        assert_eq!(logits.dims(), [1, 1, 96]);
    }
}

proptest::proptest! {
    /// The sampling top-k is disabled exactly by zero and otherwise lies in `1..=vocabulary`.
    #[test]
    fn sampling_top_k_is_disabled_by_zero_and_clamped_otherwise(
        top_k in proptest::option::of(0_usize..200),
        vocabulary_size in 1_usize..100,
    ) {
        let resolved = sampling_top_k(top_k, vocabulary_size);
        proptest::prop_assert_eq!(resolved.is_none(), top_k == Some(0));
        if let Some(resolved) = resolved {
            proptest::prop_assert!((1..=vocabulary_size).contains(&resolved));
            proptest::prop_assert_eq!(resolved, top_k.unwrap_or(50).min(vocabulary_size));
        }
    }
}
