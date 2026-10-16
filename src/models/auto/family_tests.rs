//! Tests for the family-specific planning steps of strict automatic causal loading.

use std::{fs::File, io::Write, path::Path};

use burn::backend::Flex;

use super::*;

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

/// Typed checkpoint failures keep their complete message in [`Error::Checkpoint`].
#[test]
fn checkpoint_errors_keep_their_complete_message() {
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
        assert!(matches!(
            checkpoint_error(error),
            Error::Checkpoint(message) if message == expected
        ));
    }
}

/// Qwen3.5 and Muse planning reject a checkpoint with neither language namespace by family name.
///
/// Both families probe the shared classifier before building a contract, so
/// the failure names the family and neither namespace, not a missing tensor.
#[test]
fn namespace_detection_failures_name_the_family() {
    let root = tempfile::TempDir::new().unwrap();
    write_namespace_free_checkpoint(root.path());
    let options = AutoLoadOptions::default();

    let qwen = <Qwen3_5ForCausalLm<B> as CausalFamily<B>>::plan_checkpoint(
        &Qwen3_5TextConfig::default(),
        root.path(),
        &options,
    )
    .err()
    .expect("a namespace-free Qwen checkpoint must fail");
    assert!(matches!(
        qwen,
        Error::Checkpoint(message)
            if message == "Qwen checkpoint contains neither official language parameter namespace"
    ));

    let muse = <MuseGlimmerForCausalLm<B> as CausalFamily<B>>::plan_checkpoint(
        &MuseGlimmerTextConfig::default(),
        root.path(),
        &options,
    )
    .err()
    .expect("a namespace-free Muse checkpoint must fail");
    assert!(matches!(
        muse,
        Error::Checkpoint(message)
            if message == "Muse checkpoint contains neither official language parameter namespace"
    ));
}
