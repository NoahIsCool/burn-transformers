//! Tests for the family-specific planning steps of strict automatic causal loading.

use std::{fs::File, io::Write, path::Path};

use burn::backend::Flex;

use super::*;
use crate::checkpoint::{CheckpointError, MappingPlan, TensorOperation};

/// Deterministic CPU backend; planning never allocates parameters.
type B = Flex<f32, i32>;

/// Writes a one-tensor SafeTensors shard whose only name is `lm_head.weight`.
fn write_namespace_free_checkpoint(root: &Path) {
    let header = br#"{"lm_head.weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
    let mut file = File::create(root.join("model.safetensors")).unwrap();
    file.write_all(&(header.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(header).unwrap();
    file.write_all(&[0; 4]).unwrap();
}

/// Typed planning failures convert into [`Error::CheckpointPlan`] with their complete message.
#[test]
fn typed_planning_errors_convert_into_checkpoint_plan() {
    for (error, expected) in [
        (
            CheckpointError::LanguageNamespaceNotFound { family: "Qwen" },
            "Qwen checkpoint contains neither official language parameter namespace",
        ),
        (
            CheckpointError::AmbiguousLanguageNamespace { family: "Muse" },
            "Muse checkpoint ambiguously contains wrapped and text-only language namespaces",
        ),
    ] {
        let error = Error::from(error);
        assert_eq!(error.to_string(), format!("checkpoint error: {expected}"));
        assert!(matches!(&error, Error::CheckpointPlan(cause) if cause.to_string() == expected));
    }
}

/// Qwen3.5 and Muse planning reject a checkpoint with neither language namespace by family name.
///
/// Both families probe the shared classifier before building a contract, so
/// the failure names the family and neither namespace, not a missing tensor.
/// One layer keeps the configurations within the one-tensor checkpoint's
/// budget, so detection is reached.
#[test]
fn namespace_detection_failures_name_the_family() {
    let root = tempfile::TempDir::new().unwrap();
    write_namespace_free_checkpoint(root.path());
    let options = AutoLoadOptions::default();

    let qwen = <Qwen3_5ForCausalLm<B> as CausalFamily<B>>::plan_checkpoint(
        &Qwen3_5TextConfig {
            num_hidden_layers: 1,
            ..Qwen3_5TextConfig::default()
        },
        root.path(),
        &options,
    )
    .err()
    .expect("a namespace-free Qwen checkpoint must fail");
    assert!(
        matches!(
            qwen,
            Error::CheckpointPlan(CheckpointError::LanguageNamespaceNotFound { family: "Qwen" })
        ),
        "{qwen}"
    );

    let muse = <MuseGlimmerForCausalLm<B> as CausalFamily<B>>::plan_checkpoint(
        &MuseGlimmerTextConfig {
            num_hidden_layers: 1,
            ..MuseGlimmerTextConfig::default()
        },
        root.path(),
        &options,
    )
    .err()
    .expect("a namespace-free Muse checkpoint must fail");
    assert!(
        matches!(
            muse,
            Error::CheckpointPlan(CheckpointError::LanguageNamespaceNotFound { family: "Muse" })
        ),
        "{muse}"
    );
}

/// Parses a committed tiny-golden `config.json` into a family configuration.
fn golden_config<C: ModelConfig>(case: &str) -> C {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/tiny_goldens")
        .join(case)
        .join("config.json");
    C::from_json_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

/// Counts the distinct checkpoint tensors a mapping plan consumes.
fn distinct_sources(mappings: &MappingPlan) -> usize {
    let mut sources = std::collections::BTreeSet::new();
    for operation in mappings.operations() {
        match operation {
            TensorOperation::Map { source, .. } | TensorOperation::Split { source, .. } => {
                sources.insert(source.as_str());
            }
            TensorOperation::Concatenate { sources: parts, .. } => {
                sources.extend(parts.iter().map(String::as_str));
            }
            TensorOperation::DequantizeMxfp4 { blocks, scales, .. } => {
                sources.extend([blocks.as_str(), scales.as_str()]);
            }
            TensorOperation::Alias { .. } => {}
        }
    }
    sources.len()
}

/// Checks one family's bound against its contract: never above the sources
/// the contract consumes, and equal to them when the family claims one-to-one.
fn assert_lower_bound(label: &str, minimum: usize, mappings: &MappingPlan, exact: bool) {
    let sources = distinct_sources(mappings);
    assert!(
        minimum <= sources,
        "{label}: bound {minimum} exceeds {sources} sources"
    );
    if exact {
        assert_eq!(minimum, sources, "{label}: one-to-one bound must be exact");
    }
}

/// Every family's pre-contract bound is a true lower bound on its contract's sources.
///
/// A bound above the sources would reject a loadable checkpoint, so each
/// family is checked against the contract it actually builds; the
/// one-to-one families (Qwen3-MoE, DeepSeek-V2, Nemotron-H) must match it
/// exactly with tied and untied heads.
#[test]
fn minimum_checkpoint_tensors_never_exceed_the_contract_sources() {
    use crate::models::{
        gemma::GemmaCheckpointOptions, gpt_oss::GptOssCheckpointOptions,
        llama::LlamaCheckpointOptions,
    };

    for case in ["llama3", "llama-bias-tied", "qwen2", "olmo2"] {
        let config: LlamaFamilyConfig = golden_config(case);
        let contract = config
            .checkpoint_contract(LlamaCheckpointOptions::default())
            .unwrap();
        let minimum = <LlamaForCausalLm<B> as CausalFamily<B>>::minimum_checkpoint_tensors(&config);
        assert_lower_bound(case, minimum, contract.mappings(), false);
    }
    for case in ["gemma2", "gemma3-text"] {
        let config: GemmaConfig = golden_config(case);
        let contract = config
            .checkpoint_contract(GemmaCheckpointOptions::default())
            .unwrap();
        let minimum = <GemmaForCausalLm<B> as CausalFamily<B>>::minimum_checkpoint_tensors(&config);
        assert_lower_bound(case, minimum, contract.mappings(), false);
    }
    let config: GptOssConfig = golden_config("gpt-oss");
    let contract = config
        .checkpoint_contract(GptOssCheckpointOptions::default())
        .unwrap();
    let minimum = <GptOssForCausalLm<B> as CausalFamily<B>>::minimum_checkpoint_tensors(&config);
    assert_lower_bound("gpt-oss", minimum, contract.mappings(), false);

    let config = Qwen3_5TextConfig::default();
    for source in [
        Qwen3_5CheckpointSource::Multimodal,
        Qwen3_5CheckpointSource::TextOnly,
    ] {
        let contract = config
            .checkpoint_contract(Qwen3_5CheckpointOptions {
                source,
                include_mtp: false,
            })
            .unwrap();
        let minimum =
            <Qwen3_5ForCausalLm<B> as CausalFamily<B>>::minimum_checkpoint_tensors(&config);
        assert_lower_bound("qwen3_5", minimum, contract.mappings(), false);
    }
    let config = MuseGlimmerTextConfig::default();
    for source in [
        MuseGlimmerCheckpointSource::Multimodal,
        MuseGlimmerCheckpointSource::TextOnly,
    ] {
        let contract = config
            .checkpoint_contract(MuseGlimmerCheckpointOptions { source })
            .unwrap();
        let minimum =
            <MuseGlimmerForCausalLm<B> as CausalFamily<B>>::minimum_checkpoint_tensors(&config);
        assert_lower_bound("muse", minimum, contract.mappings(), false);
    }

    for tied in [false, true] {
        let config = Qwen3MoeConfig {
            tie_word_embeddings: tied,
            ..Qwen3MoeConfig::default()
        };
        let contract = config
            .checkpoint_contract(Qwen3MoeCheckpointOptions::default())
            .unwrap();
        let minimum =
            <Qwen3MoeForCausalLm<B> as CausalFamily<B>>::minimum_checkpoint_tensors(&config);
        // The bound subtracts the LM head even when it is stored, so an
        // untied head is the one source it does not count.
        assert_lower_bound(
            "qwen3_moe",
            minimum + usize::from(!tied),
            contract.mappings(),
            true,
        );

        let config = DeepseekV2Config {
            tie_word_embeddings: tied,
            ..DeepseekV2Config::default()
        };
        let contract = config
            .checkpoint_contract(DeepseekV2CheckpointOptions::default())
            .unwrap();
        let minimum =
            <DeepseekV2ForCausalLm<B> as CausalFamily<B>>::minimum_checkpoint_tensors(&config);
        assert_lower_bound(
            "deepseek_v2",
            minimum + usize::from(!tied),
            contract.mappings(),
            true,
        );

        let config = NemotronHConfig {
            tie_word_embeddings: tied,
            ..NemotronHConfig::default()
        };
        let contract = config
            .checkpoint_contract(NemotronHCheckpointOptions::default())
            .unwrap();
        let minimum =
            <NemotronHForCausalLm<B> as CausalFamily<B>>::minimum_checkpoint_tensors(&config);
        assert_lower_bound(
            "nemotron_h",
            minimum + usize::from(!tied),
            contract.mappings(),
            true,
        );
    }
}

/// Plans `config` for family `F` against the one-tensor checkpoint at `root`.
fn plan_against<F: CausalFamily<B>>(
    config: &F::Config,
    root: &Path,
    options: &AutoLoadOptions,
) -> Error {
    F::plan_checkpoint(config, root, options)
        .err()
        .expect("a one-tensor checkpoint cannot hold a full configuration")
}

/// A configuration deeper than its checkpoint fails before the contract is built.
///
/// Every family reports [`CheckpointError::TooFewTensors`] with its own bound
/// against the one discovered tensor; building the contract first would
/// instead end in a strict-validation or namespace error. With a
/// missing-tensor allowlist the bound proves nothing, so planning proceeds to
/// the contract and fails there instead.
#[test]
fn configurations_larger_than_the_checkpoint_fail_before_their_contract() {
    let root = tempfile::TempDir::new().unwrap();
    write_namespace_free_checkpoint(root.path());
    let options = AutoLoadOptions::default();

    /// Asserts the budget error for one family and returns its bound.
    fn too_few<F: CausalFamily<B>>(config: &F::Config, root: &Path, options: &AutoLoadOptions) {
        let required = F::minimum_checkpoint_tensors(config);
        assert!(required > 1);
        let error = plan_against::<F>(config, root, options);
        assert!(
            matches!(
                &error,
                Error::CheckpointPlan(CheckpointError::TooFewTensors { required: r, available: 1 })
                    if *r == required
            ),
            "{error}"
        );
    }

    too_few::<LlamaForCausalLm<B>>(&golden_config("llama3"), root.path(), &options);
    too_few::<GemmaForCausalLm<B>>(&golden_config("gemma2"), root.path(), &options);
    too_few::<GptOssForCausalLm<B>>(&golden_config("gpt-oss"), root.path(), &options);
    too_few::<Qwen3_5ForCausalLm<B>>(&Qwen3_5TextConfig::default(), root.path(), &options);
    too_few::<MuseGlimmerForCausalLm<B>>(&MuseGlimmerTextConfig::default(), root.path(), &options);
    too_few::<Qwen3MoeForCausalLm<B>>(&Qwen3MoeConfig::default(), root.path(), &options);
    too_few::<DeepseekV2ForCausalLm<B>>(&DeepseekV2Config::default(), root.path(), &options);
    too_few::<NemotronHForCausalLm<B>>(&NemotronHConfig::default(), root.path(), &options);
    // Pinned provenance changes Nemotron's discovery policy, not the bound.
    too_few::<NemotronHForCausalLm<B>>(
        &NemotronHConfig::default(),
        root.path(),
        &AutoLoadOptions {
            repository: Some("nvidia/fixture".into()),
            revision: Some("fixture-revision".into()),
            ..AutoLoadOptions::default()
        },
    );

    let lenient = AutoLoadOptions {
        checkpoint: crate::checkpoint::LoadOptions {
            allow_missing: vec![crate::checkpoint::KeyMatcher::Prefix("model.".into())],
            ..crate::checkpoint::LoadOptions::default()
        },
        ..AutoLoadOptions::default()
    };
    let error =
        plan_against::<LlamaForCausalLm<B>>(&golden_config("llama3"), root.path(), &lenient);
    assert!(
        !matches!(
            error,
            Error::CheckpointPlan(CheckpointError::TooFewTensors { .. })
        ),
        "{error}"
    );
}
