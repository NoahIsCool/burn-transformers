//! Deterministic regression seeds for the configuration fuzz harness.

use std::path::{Path, PathBuf};

use super::*;

/// Directory of the committed configuration corpus.
fn seed_directory() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("seeds/configuration")
}

/// Reads one committed seed.
fn seed(name: &str) -> String {
    std::fs::read_to_string(seed_directory().join(name)).unwrap()
}

/// Every valid family seed, the family that must accept it, and its bounded count fields.
const FAMILY_SEEDS: &[(&str, &str, &[&str])] = &[
    ("llama.json", "llama", &["num_hidden_layers"]),
    ("llama-bias-tied.json", "llama", &["num_hidden_layers"]),
    ("mistral.json", "llama", &["num_hidden_layers"]),
    ("qwen2.json", "llama", &["num_hidden_layers"]),
    ("qwen3.json", "llama", &["num_hidden_layers"]),
    ("smollm3.json", "llama", &["num_hidden_layers"]),
    ("olmo2.json", "llama", &["num_hidden_layers"]),
    ("gemma2.json", "gemma", &["num_hidden_layers"]),
    ("gemma3-text.json", "gemma", &["num_hidden_layers"]),
    (
        "gpt-oss.json",
        "gpt_oss",
        &["num_hidden_layers", "num_local_experts"],
    ),
    ("gpt2.json", "gpt2", &["n_layer"]),
    ("gpt-neox.json", "gpt_neox", &["num_hidden_layers"]),
    ("qwen.json", "qwen3_5", &["num_hidden_layers"]),
    (
        "qwen3-moe.json",
        "qwen3_moe",
        &["num_hidden_layers", "num_experts"],
    ),
    (
        "deepseek-v2.json",
        "deepseek_v2",
        &["num_hidden_layers", "n_routed_experts"],
    ),
    ("muse-glimmer.json", "muse_glimmer", &["num_hidden_layers"]),
    ("nemotron-h.json", "nemotron_h", &["num_hidden_layers"]),
    (
        "nemotron-h-moe.json",
        "nemotron_h",
        &["num_hidden_layers", "n_routed_experts"],
    ),
    ("t5.json", "t5", &["num_layers", "num_decoder_layers"]),
];

/// Writes `value` into the text-model object of `document` (`text_config` when present).
fn with_count(document: &str, field: &str, value: usize) -> String {
    // The Nemotron seed keeps NVIDIA's bare `Infinity`, which plain JSON rejects.
    let normalized = document.replace("Infinity", "null");
    let mut document: Value = serde_json::from_str(&normalized).unwrap();
    let text = if document.get("text_config").is_some() {
        &mut document["text_config"]
    } else {
        &mut document
    };
    text[field] = Value::from(value);
    document.to_string()
}

/// Every committed seed runs through the harness, and each family seed is accepted by its family.
#[test]
fn committed_seeds_replay_and_cover_every_family() {
    let mut names = std::fs::read_dir(seed_directory())
        .unwrap()
        .map(|entry| entry.unwrap().file_name().into_string().unwrap())
        .collect::<Vec<_>>();
    names.sort();
    for name in &names {
        exercise(seed(name).as_bytes());
    }
    let mut covered = std::collections::BTreeSet::new();
    for (name, family, _) in FAMILY_SEEDS {
        let accepted = every_family(&seed(name));
        assert!(
            accepted.contains(family),
            "{name}: accepted by {accepted:?}"
        );
        covered.insert(*family);
    }
    assert_eq!(covered.len(), 11, "one seed per registered family parser");
    // Each over-bound seed is a variant of a family seed that no parser accepts.
    let huge = names
        .iter()
        .filter(|name| name.contains("-huge-"))
        .collect::<Vec<_>>();
    assert!(huge.len() >= 11, "{huge:?}");
    for name in huge {
        assert_eq!(every_family(&seed(name)), Vec::<&str>::new(), "{name}");
    }
}

/// Absurd counts are rejected by every family that reads them, without panicking.
///
/// Regression for FUZZ-CFG-1: the harness skipped every input whose layer
/// count exceeded 128, which hid the capacity-overflow panics of the Llama,
/// gpt-oss, Qwen3.5, and Muse validators at `usize::MAX` (CFG-COUNT-1).
#[test]
fn absurd_counts_are_rejected_by_every_family_that_reads_them() {
    for (name, family, fields) in FAMILY_SEEDS {
        for field in *fields {
            let bound = if field.contains("expert") {
                MAX_EXPERTS
            } else {
                MAX_LAYERS
            };
            for value in [usize::MAX, bound + 1] {
                let patched = with_count(&seed(name), field, value);
                exercise(patched.as_bytes());
                assert!(
                    !every_family(&patched).contains(family),
                    "{name}: {family} accepted {field} = {value}"
                );
            }
        }
    }
}

/// `usize::MAX` depths on an otherwise default document are rejected by all eleven parsers.
///
/// Every family reads at least one of the four depth spellings, and the
/// families that default their `model_type` would otherwise accept the
/// document.
#[test]
fn maximal_depth_is_rejected_by_all_parsers() {
    let depth = usize::MAX;
    let json = format!(
        r#"{{"num_hidden_layers":{depth},"num_layers":{depth},"num_decoder_layers":{depth},"n_layer":{depth}}}"#
    );
    assert_eq!(every_family(&json), Vec::<&str>::new());
    let defaults = every_family("{}");
    assert!(defaults.len() >= 7, "{defaults:?}");
}

/// Malformed, wrapped, and legacy documents stay panic-free.
#[test]
fn malformed_and_legacy_documents_never_panic() {
    exercise(br#"{"model_type":"t5","num_layers":2,"num_decoder_layers":2}"#);
    exercise(br#"{"text_config":{"model_type":"qwen3_5_text"}}"#);
    exercise(br#"{"time_step_limit":[0.0,Infinity]}"#);
    exercise(br#"{"time_step_limit":[0.0,{"__float__":"Infinity"}]}"#);
    exercise(br#"{"time_step_limit":[0.0,{"__float__":"NaN"}]}"#);
    exercise(b"\xff");
    exercise(&[b' '; crate::MAX_INPUT_BYTES + 1]);
}

/// Seed mutations from a fixed generator stay panic-free and respect the count bounds.
#[test]
fn mutated_seeds_respect_the_oracles() {
    let mut state: u64 = 0x5eed_c0f1_9000_0001;
    let mut next = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        state >> 33
    };
    /// Replacement bytes; digits keep most mutations inside numeric literals.
    const ALPHABET: &[u8] = b"0123456789-.e\"{}[],:";
    for (name, _, _) in FAMILY_SEEDS {
        let original = seed(name).into_bytes();
        for _ in 0..16 {
            let mut mutated = original.clone();
            for _ in 0..1 + next() % 4 {
                let position = (next() as usize) % mutated.len();
                mutated[position] = ALPHABET[(next() as usize) % ALPHABET.len()];
            }
            exercise(&mutated);
        }
    }
}
