//! Strict loading: committed tiny checkpoints load exactly from every
//! published namespace, the vision tower is reported as ignored, malformed
//! checkpoints fail before any payload is installed, the sink rejects
//! targets the configured graph does not own, and Gemma 3's declared F16
//! range hazard refuses F16 loads unless the caller accepts the overflow.

use std::path::{Path, PathBuf};

use burn::tensor::{DType, TensorData, f16};
use safetensors::{Dtype, SafeTensors, tensor::TensorView};

use super::*;
use crate::{
    CausalLmInput, ForwardOptions, LogitsSelection,
    checkpoint::{DTypePolicy, RangeHazardPolicy, TensorDType},
    models::{
        config::ModelConfig,
        gemma::{
            GemmaConfig, GemmaVariant,
            model::test_support::{B, host, ints, model_from, tiny_config, tiny_model},
        },
    },
};

/// Returns a committed tiny-golden fixture directory.
fn fixture(case: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/tiny_goldens")
        .join(case)
}

/// Loads a fixture's configuration and allocates its graph.
fn graph(root: &Path) -> GemmaForCausalLm<B> {
    let config = std::fs::read_to_string(root.join("config.json")).unwrap();
    let config = GemmaConfig::from_json_str(&config).unwrap();
    GemmaForCausalLm::init(config, &Default::default()).unwrap()
}

/// Copies `case` into a temporary directory, rewriting its tensors with `edit`.
///
/// `edit` receives owned `(name, dtype, shape, bytes)` entries and may
/// rename, drop, or add tensors.
fn rewritten(
    case: &str,
    edit: impl FnOnce(&mut Vec<(String, Dtype, Vec<usize>, Vec<u8>)>),
) -> tempfile::TempDir {
    let source = fixture(case);
    let directory = tempfile::tempdir().unwrap();
    std::fs::copy(
        source.join("config.json"),
        directory.path().join("config.json"),
    )
    .unwrap();
    let bytes = std::fs::read(source.join("model.safetensors")).unwrap();
    let tensors = SafeTensors::deserialize(&bytes).unwrap();
    let mut entries: Vec<(String, Dtype, Vec<usize>, Vec<u8>)> = tensors
        .tensors()
        .into_iter()
        .map(|(name, view)| {
            (
                name,
                view.dtype(),
                view.shape().to_vec(),
                view.data().to_vec(),
            )
        })
        .collect();
    edit(&mut entries);
    let views: Vec<(String, TensorView<'_>)> = entries
        .iter()
        .map(|(name, dtype, shape, data)| {
            (
                name.clone(),
                TensorView::new(*dtype, shape.clone(), data).unwrap(),
            )
        })
        .collect();
    let payload = safetensors::serialize(views, None).unwrap();
    std::fs::write(directory.path().join("model.safetensors"), payload).unwrap();
    directory
}

/// Renames every text-only `model.` tensor under `prefix` and adds a small vision tower.
fn multimodal(case: &str, prefix: &str, vision: &[&str]) -> tempfile::TempDir {
    let prefix = prefix.to_owned();
    let vision: Vec<String> = vision.iter().map(|name| (*name).to_owned()).collect();
    rewritten(case, move |entries| {
        for (name, ..) in entries.iter_mut() {
            *name = format!("{prefix}.{}", name.strip_prefix("model.").unwrap());
        }
        for name in vision {
            entries.push((name, Dtype::F32, vec![2], vec![0; 8]));
        }
    })
}

/// Last-position logits of a fixed prompt.
fn prompt_logits(model: &GemmaForCausalLm<B>) -> Vec<f32> {
    let output = model
        .forward(
            CausalLmInput {
                token_ids: ints([[5, 17, 3, 29, 8]]),
                token_mask: None,
                position_ids: None,
            },
            &ForwardOptions {
                logits: LogitsSelection::Last,
                ..ForwardOptions::default()
            },
        )
        .unwrap();
    host(output.logits)
}

/// Every committed fixture loads with zero unexplained tensors.
#[test]
fn committed_fixtures_load_strictly() {
    for case in ["gemma2", "gemma3-text"] {
        let root = fixture(case);
        let mut model = graph(&root);
        let outcome = model
            .load_checkpoint(&root, &LoadOptions::default())
            .unwrap();
        assert!(outcome.report.is_strictly_valid(), "{case}");
        assert!(outcome.report.allowlisted_ignored.is_empty(), "{case}");
    }
}

/// Both multimodal namespaces load the same text tower and report the omitted vision tensors.
#[test]
fn multimodal_namespaces_load_the_text_tower_and_report_the_vision_tower() {
    let root = fixture("gemma3-text");
    let mut text_only = graph(&root);
    text_only
        .load_checkpoint(&root, &LoadOptions::default())
        .unwrap();
    let reference = prompt_logits(&text_only);
    for (prefix, vision) in [
        (
            "model.language_model",
            [
                "model.vision_tower.vision_model.post_layernorm.weight",
                "model.multi_modal_projector.mm_soft_emb_norm.weight",
            ],
        ),
        (
            "language_model.model",
            [
                "vision_tower.vision_model.post_layernorm.weight",
                "multi_modal_projector.mm_soft_emb_norm.weight",
            ],
        ),
    ] {
        let directory = multimodal("gemma3-text", prefix, &vision);
        let mut model = graph(directory.path());
        let outcome = model
            .load_checkpoint(directory.path(), &LoadOptions::default())
            .unwrap();
        assert!(outcome.report.is_strictly_valid(), "{prefix}");
        let mut ignored = outcome.report.allowlisted_ignored.clone();
        ignored.sort();
        let mut expected: Vec<String> = vision.iter().map(|name| (*name).to_owned()).collect();
        expected.sort();
        assert_eq!(ignored, expected, "{prefix}");
        assert_eq!(prompt_logits(&model), reference, "{prefix}");
    }
}

/// A multimodal checkpoint with a final soft-cap fails before any payload is read.
#[test]
fn multimodal_checkpoints_refuse_a_final_soft_cap() {
    let directory = multimodal("gemma3-text", "model.language_model", &[]);
    let config = std::fs::read_to_string(directory.path().join("config.json")).unwrap();
    let capped = GemmaConfig {
        final_logit_softcapping: Some(30.0),
        ..GemmaConfig::from_json_str(&config).unwrap()
    };
    let mut model = GemmaForCausalLm::<B>::init(capped.clone(), &Default::default()).unwrap();
    let error = model
        .load_checkpoint(directory.path(), &LoadOptions::default())
        .unwrap_err();
    assert!(
        matches!(&error, GemmaCheckpointLoadError::Config(config) if config.field == "final_logit_softcapping"),
        "{error}"
    );
    assert!(error.execution_failure().is_none(), "nothing was installed");
    // The same configuration loads the text-only namespace, whose reference caps.
    let mut text_only = GemmaForCausalLm::<B>::init(capped, &Default::default()).unwrap();
    let root = fixture("gemma3-text");
    assert!(
        text_only
            .load_checkpoint(&root, &LoadOptions::default())
            .is_ok()
    );
}

/// Missing, unexpected, misshapen, stray-vision, and ambiguous checkpoints fail before installation.
#[test]
fn malformed_checkpoints_fail_before_installation() {
    let missing = rewritten("gemma3-text", |entries| {
        entries.retain(|(name, ..)| name != "model.layers.1.self_attn.k_norm.weight");
    });
    let stored_tied_head = rewritten("gemma2", |entries| {
        let embedding = entries
            .iter()
            .find(|(name, ..)| name == "model.embed_tokens.weight")
            .cloned()
            .unwrap();
        entries.push((
            "lm_head.weight".into(),
            embedding.1,
            embedding.2,
            embedding.3,
        ));
    });
    let misshapen = rewritten("gemma3-text", |entries| {
        let entry = entries
            .iter_mut()
            .find(|(name, ..)| name == "model.layers.0.self_attn.q_norm.weight")
            .unwrap();
        entry.2 = vec![4];
        entry.3.truncate(16);
    });
    let vision_in_text_only = rewritten("gemma2", |entries| {
        entries.push((
            "vision_tower.vision_model.post_layernorm.weight".into(),
            Dtype::F32,
            vec![2],
            vec![0; 8],
        ));
    });
    let stray_in_multimodal = multimodal(
        "gemma3-text",
        "model.language_model",
        &["model.audio_tower.weight"],
    );
    for (directory, context) in [
        (missing, "missing"),
        (stored_tied_head, "stored tied head"),
        (misshapen, "misshapen Q norm"),
        (
            vision_in_text_only,
            "vision tensor in a text-only checkpoint",
        ),
        (stray_in_multimodal, "non-vision stray tensor"),
    ] {
        let mut model = graph(directory.path());
        let error = model
            .load_checkpoint(directory.path(), &LoadOptions::default())
            .unwrap_err();
        assert!(
            matches!(error, GemmaCheckpointLoadError::Checkpoint(_)),
            "{context}: {error}"
        );
        assert!(error.execution_failure().is_none(), "{context}");
    }
    let ambiguous = rewritten("gemma2", |entries| {
        let embedding = entries
            .iter()
            .find(|(name, ..)| name == "model.embed_tokens.weight")
            .cloned()
            .unwrap();
        entries.push((
            "model.language_model.embed_tokens.weight".into(),
            embedding.1,
            embedding.2,
            embedding.3,
        ));
    });
    let mut model = graph(ambiguous.path());
    let error = model
        .load_checkpoint(ambiguous.path(), &LoadOptions::default())
        .unwrap_err();
    assert!(
        matches!(
            &error,
            GemmaCheckpointLoadError::Checkpoint(CheckpointError::AmbiguousLanguageEmbedding {
                family: "Gemma",
                embeddings,
            }) if embeddings.len() == 2
        ),
        "{error}"
    );
    assert!(error.to_string().contains("ambiguously"), "{error}");
}

/// The sink rejects targets outside the configured graph.
#[test]
fn the_sink_rejects_foreign_targets() {
    let mut gemma2 = tiny_model(GemmaVariant::Gemma2, 1);
    let scalar = || TensorData::new(vec![0.0_f32; 4], [4]);
    for (target, fragment) in [
        ("model.unknown.weight", "unknown"),
        ("model.layers.7.input_layernorm.weight", "outside"),
        ("model.layers.x.input_layernorm.weight", "decimal"),
        ("model.layers.0", "expected"),
        ("model.layers.0.", "expected"),
        ("model.layers.0.self_attn.o_proj.bias", "projection bias"),
        ("model.layers.0.mlp.gate_proj.bias", "unknown"),
        ("model.layers.0.self_attn.rotary.weight", "unknown"),
        (
            "model.layers.0.self_attn.query_norm.weight",
            "query RMSNorm",
        ),
        ("model.layers.0.self_attn.key_norm.weight", "key RMSNorm"),
        ("lm_head.weight", "independent LM head"),
    ] {
        let error = gemma2.write_tensor(target, scalar()).unwrap_err();
        assert!(error.to_string().contains(fragment), "{target}: {error}");
    }
    let shape = gemma2
        .write_tensor("model.norm.weight", scalar())
        .unwrap_err();
    assert!(matches!(shape, GemmaLoadError::Parameter { .. }), "{shape}");
    let mut gemma3 = tiny_model(GemmaVariant::Gemma3Text, 1);
    gemma3
        .write_tensor("model.layers.3.self_attn.key_norm.weight", scalar())
        .unwrap();
    gemma3
        .write_tensor("model.layers.3.post_feedforward_layernorm.weight", {
            TensorData::new(vec![0.0_f32; 16], [16])
        })
        .unwrap();
}

/// Only a tied model accepts the embedding-to-head alias.
#[test]
fn the_tied_head_alias_is_validated() {
    let mut tied = tiny_model(GemmaVariant::Gemma2, 2);
    tied.alias_tensor("lm_head.weight", "model.embed_tokens.weight")
        .unwrap();
    assert!(
        tied.alias_tensor("lm_head.weight", "model.norm.weight")
            .is_err()
    );
    let mut untied = crate::models::gemma::model::test_support::model_from(
        GemmaConfig {
            tie_word_embeddings: false,
            ..tiny_config(GemmaVariant::Gemma3Text)
        },
        2,
    );
    let error = untied
        .alias_tensor("lm_head.weight", "model.embed_tokens.weight")
        .unwrap_err();
    assert!(error.to_string().contains("independent LM head"), "{error}");
    untied
        .write_tensor(
            "lm_head.weight",
            TensorData::new(vec![0.0_f32; 32 * 16], [32, 16]),
        )
        .unwrap();
    let mut lazy =
        GemmaForCausalLm::<B>::init(tiny_config(GemmaVariant::Gemma2), &Default::default())
            .unwrap();
    let error = lazy
        .alias_tensor("lm_head.weight", "model.embed_tokens.weight")
        .unwrap_err();
    assert!(error.to_string().contains("not been installed"), "{error}");
}

/// Biased attention installs projection biases; a graph edited after planning reports the partial load.
#[test]
fn biases_install_and_sink_failures_keep_the_partial_report() {
    let mut biased = model_from(
        GemmaConfig {
            attention_bias: true,
            ..tiny_config(GemmaVariant::Gemma3Text)
        },
        2,
    );
    let values = (0..16).map(|value| value as f32).collect::<Vec<_>>();
    biased
        .write_tensor(
            "model.layers.0.self_attn.o_proj.bias",
            TensorData::new(values.clone(), [16]),
        )
        .unwrap();
    let installed = biased.model.layers[0]
        .self_attn
        .o_proj
        .bias
        .as_ref()
        .unwrap()
        .val()
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    assert_eq!(installed, values);

    let root = fixture("gemma3-text");
    let mut model = graph(&root);
    model.model.layers[2].self_attn.key_norm = None;
    let error = model
        .load_checkpoint(&root, &LoadOptions::default())
        .unwrap_err();
    let failure = error
        .execution_failure()
        .expect("planning succeeded, so execution reports the failure");
    assert_eq!(
        failure.failed_target.as_deref(),
        Some("model.layers.2.self_attn.key_norm.weight")
    );
    assert!(matches!(
        &failure.cause,
        crate::checkpoint::ExecutionFailureCause::Sink(GemmaLoadError::MissingComponent { .. })
    ));
}

/// `load_checkpoint_with` installs the namespace it is given instead of detecting one.
///
/// The detected text-only source reproduces `load_checkpoint` exactly; forcing
/// a multimodal namespace on the same checkpoint is a strict planning error.
#[test]
fn load_checkpoint_with_uses_the_given_namespace() {
    let root = fixture("gemma3-text");
    let detected = graph(&root)
        .load_checkpoint(&root, &LoadOptions::default())
        .unwrap();
    let explicit = graph(&root)
        .load_checkpoint_with(
            &root,
            &LoadOptions::default(),
            GemmaCheckpointOptions {
                source: GemmaCheckpointSource::TextOnly,
            },
        )
        .unwrap();
    assert_eq!(explicit.report, detected.report);
    for source in [
        GemmaCheckpointSource::Multimodal,
        GemmaCheckpointSource::LegacyMultimodal,
    ] {
        let error = graph(&root)
            .load_checkpoint_with(
                &root,
                &LoadOptions::default(),
                GemmaCheckpointOptions { source },
            )
            .unwrap_err();
        assert!(
            matches!(error, GemmaCheckpointLoadError::Checkpoint(_)),
            "{source:?}: {error}"
        );
    }
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

// Range hazards: Gemma 3 refuses F16 unless the overflow is accepted.

/// Load options selecting `dtype_policy` under the range-hazard `policy`.
fn hazard_options(dtype_policy: DTypePolicy, policy: RangeHazardPolicy) -> LoadOptions {
    LoadOptions {
        dtype_policy,
        range_hazards: policy,
        ..LoadOptions::default()
    }
}

/// Copies `case` with every tensor stored in F16, each F32 value rounded once.
fn stored_in_f16(case: &str) -> tempfile::TempDir {
    rewritten(case, |entries| {
        for (name, dtype, _, bytes) in entries.iter_mut() {
            assert_eq!(*dtype, Dtype::F32, "{name}: the fixtures store F32");
            *bytes = bytes
                .as_chunks::<4>()
                .0
                .iter()
                .flat_map(|chunk| f16::from_f32(f32::from_le_bytes(*chunk)).to_le_bytes())
                .collect();
            *dtype = Dtype::F16;
        }
    })
}

/// Rewrites a multimodal copy's text `config.json` as the `gemma3` wrapper Transformers writes.
fn wrap_as_gemma3(directory: &Path) {
    let text: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(directory.join("config.json")).unwrap())
            .unwrap();
    let wrapper = serde_json::json!({
        "architectures": ["Gemma3ForConditionalGeneration"],
        "model_type": "gemma3",
        "text_config": text,
        "vision_config": {"model_type": "siglip_vision_model"},
    });
    std::fs::write(directory.join("config.json"), wrapper.to_string()).unwrap();
}

/// Counts the storage-owning parameters strict planning installs for the checkpoint at `root`.
///
/// Plans with the hazard accepted, so F16 storage is counted too.
fn planned_parameters(root: &Path) -> usize {
    let config = std::fs::read_to_string(root.join("config.json")).unwrap();
    let config = GemmaConfig::from_json_str(&config).unwrap();
    let manifest = CheckpointManifest::discover(root, &LoadOptions::default()).unwrap();
    let source =
        GemmaCheckpointSource::detect(manifest.tensors.keys().map(String::as_str)).unwrap();
    config
        .checkpoint_contract(GemmaCheckpointOptions { source })
        .unwrap()
        .build_load_plan(
            &manifest,
            &hazard_options(DTypePolicy::Auto, RangeHazardPolicy::Accept),
        )
        .unwrap()
        .tensors
        .len()
}

/// Asserts that `error` is Gemma 3's F16 refusal naming all `parameters`, the embedding first.
fn assert_gemma3_f16_refusal(error: &CheckpointError, parameters: usize, context: &str) {
    let CheckpointError::RangeHazard {
        hazard,
        target,
        parameters: refused,
    } = error
    else {
        panic!("{context}: expected Gemma 3's F16 refusal, got {error}");
    };
    assert_eq!(
        *hazard,
        GemmaVariant::Gemma3Text.range_hazards()[0],
        "{context}"
    );
    assert_eq!(target, "model.embed_tokens.weight", "{context}");
    assert_eq!(*refused, parameters, "{context}");
}

/// Last-position logits of [`prompt_logits`]'s prompt, read as F32 whatever the model's dtype.
fn prompt_logits_as_f32(model: &GemmaForCausalLm<B>) -> Vec<f32> {
    let output = model
        .forward(
            CausalLmInput {
                token_ids: ints([[5, 17, 3, 29, 8]]),
                token_mask: None,
                position_ids: None,
            },
            &ForwardOptions::default(),
        )
        .unwrap();
    host(output.logits.cast(DType::F32))
}

/// Gemma 3 refuses F16 parameters, cast or stored, in every published namespace, before installing any.
///
/// The refusal is a planning error: no execution failure, no installed
/// parameter, and the count names every storage-owning tensor of the plan.
#[test]
fn gemma3_refuses_f16_parameters_before_installing_any() {
    let text_only = fixture("gemma3-text");
    let current = multimodal(
        "gemma3-text",
        "model.language_model",
        &["model.vision_tower.vision_model.post_layernorm.weight"],
    );
    let legacy = multimodal(
        "gemma3-text",
        "language_model.model",
        &["vision_tower.vision_model.post_layernorm.weight"],
    );
    let wrapped = multimodal(
        "gemma3-text",
        "model.language_model",
        &["model.vision_tower.vision_model.post_layernorm.weight"],
    );
    wrap_as_gemma3(wrapped.path());
    let stored = stored_in_f16("gemma3-text");
    let cast = DTypePolicy::Cast(TensorDType::F16);
    for (root, policy, context) in [
        (text_only.as_path(), cast, "gemma3_text, cast"),
        (current.path(), cast, "multimodal namespace, cast"),
        (legacy.path(), cast, "legacy multimodal namespace, cast"),
        (wrapped.path(), cast, "gemma3 wrapper, cast"),
        (stored.path(), DTypePolicy::Auto, "F16 storage under Auto"),
        (
            stored.path(),
            DTypePolicy::Preserve,
            "F16 storage under Preserve",
        ),
    ] {
        let mut model = graph(root);
        let error = model
            .load_checkpoint(root, &hazard_options(policy, RangeHazardPolicy::default()))
            .unwrap_err();
        assert!(error.execution_failure().is_none(), "{context}");
        let GemmaCheckpointLoadError::Checkpoint(error) = &error else {
            panic!("{context}: expected a planning error, got {error}");
        };
        assert_gemma3_f16_refusal(error, planned_parameters(root), context);
        assert!(
            !model.model.embed_tokens.weight.is_initialized(),
            "{context}: nothing was installed"
        );
    }
}

/// Accepting the overflow loads the faithful F16 graph, cast or stored, and it runs in F16.
///
/// The tiny fixture stays far inside F16's range, so its logits are finite
/// and close to the F32 graph's; a real Gemma 3 checkpoint's are not.
#[test]
fn accepting_the_overflow_loads_the_faithful_f16_graph() {
    let root = fixture("gemma3-text");
    let mut reference = graph(&root);
    reference
        .load_checkpoint(&root, &LoadOptions::default())
        .unwrap();
    let reference = prompt_logits(&reference);
    let stored = stored_in_f16("gemma3-text");
    for (root, policy, context) in [
        (
            root.as_path(),
            DTypePolicy::Cast(TensorDType::F16),
            "cast to F16",
        ),
        (stored.path(), DTypePolicy::Auto, "stored in F16"),
    ] {
        let mut model = graph(root);
        let outcome = model
            .load_checkpoint(root, &hazard_options(policy, RangeHazardPolicy::Accept))
            .unwrap_or_else(|error| panic!("{context}: {error}"));
        assert!(outcome.report.is_strictly_valid(), "{context}");
        assert_eq!(
            model.model.embed_tokens.weight.val().dtype(),
            DType::F16,
            "{context}"
        );
        let logits = prompt_logits_as_f32(&model);
        assert!(logits.iter().all(|logit| logit.is_finite()), "{context}");
        let gap = crate::models::gemma::model::test_support::max_gap(&logits, &reference);
        assert!(gap < 0.05, "{context}: F16 logits deviate by {gap}");
    }
}

/// F32 and BF16 Gemma 3 loads, cast or kept from storage, are unaffected by the refusing default.
#[test]
fn f32_and_bf16_gemma3_loads_are_unaffected() {
    let root = fixture("gemma3-text");
    let stored = stored_in_f16("gemma3-text");
    for (root, policy, dtype) in [
        (root.as_path(), DTypePolicy::Auto, DType::F32),
        (
            root.as_path(),
            DTypePolicy::Cast(TensorDType::F32),
            DType::F32,
        ),
        (
            root.as_path(),
            DTypePolicy::Cast(TensorDType::BF16),
            DType::BF16,
        ),
        (
            stored.path(),
            DTypePolicy::Cast(TensorDType::F32),
            DType::F32,
        ),
        (
            stored.path(),
            DTypePolicy::Cast(TensorDType::BF16),
            DType::BF16,
        ),
    ] {
        let mut model = graph(root);
        let outcome = model
            .load_checkpoint(root, &hazard_options(policy, RangeHazardPolicy::Refuse))
            .unwrap_or_else(|error| panic!("{policy:?}: {error}"));
        assert!(outcome.report.is_strictly_valid(), "{policy:?}");
        assert_eq!(
            model.model.embed_tokens.weight.val().dtype(),
            dtype,
            "{policy:?}"
        );
    }
}

/// Gemma 2 declares no hazard (none was measured), so its F16 loads proceed under the refusing default.
#[test]
fn gemma2_declares_no_hazard_and_loads_in_f16() {
    assert!(GemmaVariant::Gemma2.range_hazards().is_empty());
    let root = fixture("gemma2");
    let stored = stored_in_f16("gemma2");
    for (root, policy) in [
        (root.as_path(), DTypePolicy::Cast(TensorDType::F16)),
        (stored.path(), DTypePolicy::Auto),
    ] {
        let mut model = graph(root);
        let outcome = model
            .load_checkpoint(root, &hazard_options(policy, RangeHazardPolicy::Refuse))
            .unwrap_or_else(|error| panic!("{policy:?}: {error}"));
        assert!(outcome.report.is_strictly_valid(), "{policy:?}");
        assert_eq!(model.model.embed_tokens.weight.val().dtype(), DType::F16);
    }
}

/// The refusal names Gemma 3, F16, the measured overflow and the unmeasured sizes, the alternatives, and the opt-in.
#[test]
fn the_gemma3_refusal_names_the_overflow_the_alternatives_and_the_opt_in() {
    let root = fixture("gemma3-text");
    let message = graph(&root)
        .load_checkpoint(
            &root,
            &hazard_options(
                DTypePolicy::Cast(TensorDType::F16),
                RangeHazardPolicy::Refuse,
            ),
        )
        .unwrap_err()
        .to_string();
    for fragment in [
        // The checkpoint family and the dtype.
        "Gemma 3 (model_type gemma3 or gemma3_text) checkpoints are known to overflow F16",
        "planned parameters in F16 (first `model.embed_tokens.weight`)",
        // The measured overflow, its reference, and what was not measured.
        "google/gemma-3-4b-it (revision 093f9f38)",
        "block 5's MLP reaches 73,509.6 at the BOS position, channel 443",
        "beyond F16's largest finite value 65,504",
        "Transformers 5.2.0's CPU and MPS F16 graphs",
        "gemma-3-12b-it and gemma-3-27b-it",
        "Unmeasured: other sizes (such as 1B and 270M) and fine-tunes",
        "acceptance/measurements/half-precision-envelope-20261018.json",
        // The alternatives.
        "F32 (`DTypePolicy::Cast(TensorDType::F32)`)",
        "BF16 (`DTypePolicy::Cast(TensorDType::BF16)`) on a backend that can store it \
         (Burn 0.21's Metal backend cannot)",
        "packed Q4/Q8 weights with F32 activations",
        // The opt-in and its consequence.
        "set `LoadOptions::range_hazards` to `RangeHazardPolicy::Accept`",
        "its logits are expected to be non-finite",
    ] {
        assert!(
            message.contains(fragment),
            "missing {fragment:?} in {message}"
        );
    }
}

/// Automatic loading refuses F16 `gemma3` and `gemma3_text` checkpoints, honors the opt-in, and leaves the rest alone.
///
/// The refusal arrives as [`crate::Error::CheckpointPlan`] before the graph
/// is allocated; the multimodal wrapper still reports its vision tensor as
/// intentionally ignored once accepted.
#[test]
fn automatic_loading_applies_the_gemma3_policy() {
    let text_only = fixture("gemma3-text");
    let vision = "model.vision_tower.vision_model.post_layernorm.weight";
    let wrapped = multimodal("gemma3-text", "model.language_model", &[vision]);
    wrap_as_gemma3(wrapped.path());
    let gemma2 = fixture("gemma2");
    let load = |root: &Path, dtype: TensorDType, policy: RangeHazardPolicy| {
        let options = crate::AutoLoadOptions {
            checkpoint: hazard_options(DTypePolicy::Cast(dtype), policy),
            load_tokenizer: false,
            ..crate::AutoLoadOptions::default()
        };
        crate::AutoCausalLm::<B>::load_local(root, &options, &Default::default())
    };
    for (root, ignored, context) in [
        (text_only.as_path(), &[][..], "gemma3_text"),
        (wrapped.path(), &[vision][..], "gemma3"),
    ] {
        let Err(crate::Error::CheckpointPlan(error)) =
            load(root, TensorDType::F16, RangeHazardPolicy::Refuse)
        else {
            panic!("{context}: an F16 load must be refused while planning");
        };
        assert_gemma3_f16_refusal(&error, planned_parameters(root), context);
        let accepted = load(root, TensorDType::F16, RangeHazardPolicy::Accept)
            .unwrap_or_else(|error| panic!("{context}: {error}"));
        assert_eq!(
            accepted.load_report().allowlisted_ignored,
            ignored,
            "{context}"
        );
        for dtype in [TensorDType::F32, TensorDType::BF16] {
            load(root, dtype, RangeHazardPolicy::Refuse)
                .unwrap_or_else(|error| panic!("{context} in {dtype}: {error}"));
        }
    }
    load(&gemma2, TensorDType::F16, RangeHazardPolicy::Refuse)
        .unwrap_or_else(|error| panic!("gemma2 in F16: {error}"));
}
