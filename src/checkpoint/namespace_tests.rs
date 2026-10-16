//! Tests for the shared language-namespace classifier used by family checkpoint loaders.

use proptest::prelude::*;

use super::*;

/// Exactly one official namespace is accepted; neither or both is a typed, family-named error.
#[test]
fn language_namespace_classification_is_total_and_exclusive() {
    assert_eq!(
        LanguageNamespace::classify(["model.language_model.embed_tokens.weight"], "Qwen").unwrap(),
        LanguageNamespace::Wrapped
    );
    assert_eq!(
        LanguageNamespace::classify(["model.embed_tokens.weight", "lm_head.weight"], "Muse")
            .unwrap(),
        LanguageNamespace::TextOnly
    );
    for names in [vec!["lm_head.weight"], vec![]] {
        let error = LanguageNamespace::classify(names, "Qwen").unwrap_err();
        assert!(matches!(
            error,
            CheckpointError::LanguageNamespaceNotFound { family: "Qwen" }
        ));
        assert_eq!(
            error.to_string(),
            "Qwen checkpoint contains neither official language parameter namespace"
        );
    }
    let error = LanguageNamespace::classify(
        [
            "model.language_model.norm.weight",
            "model.embed_tokens.weight",
        ],
        "Qwen",
    )
    .unwrap_err();
    assert!(matches!(
        error,
        CheckpointError::AmbiguousLanguageNamespace { family: "Qwen" }
    ));
    assert_eq!(
        error.to_string(),
        "Qwen checkpoint ambiguously contains wrapped and text-only language namespaces"
    );
    // Near misses of either namespace are not mistaken for it.
    assert!(matches!(
        LanguageNamespace::classify(
            [
                "model.language_modelx.weight",
                "model.embed_tokens.weight.extra"
            ],
            "Muse"
        ),
        Err(CheckpointError::LanguageNamespaceNotFound { family: "Muse" })
    ));
}

/// A discovered manifest is classified by its complete tensor inventory.
#[test]
fn manifest_namespace_reads_every_discovered_tensor_name() {
    use std::{fs::File, io::Write};

    let temp = tempfile::TempDir::new().unwrap();
    let header = br#"{"lm_head.weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"model.embed_tokens.weight":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}"#;
    let mut file = File::create(temp.path().join("model.safetensors")).unwrap();
    file.write_all(&(header.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(header).unwrap();
    file.write_all(&[0; 8]).unwrap();
    let manifest =
        super::super::discover_checkpoint(temp.path(), &super::super::LoadOptions::default())
            .unwrap();
    assert_eq!(
        LanguageNamespace::of_manifest(&manifest, "Muse").unwrap(),
        LanguageNamespace::TextOnly
    );
}

proptest! {
    /// Wrapped iff some name has the wrapped prefix and none is the text-only embedding.
    #[test]
    fn classification_matches_its_definition(
        names in proptest::collection::vec(
            prop_oneof![
                Just("model.language_model.layers.0.weight".to_owned()),
                Just("model.embed_tokens.weight".to_owned()),
                Just("lm_head.weight".to_owned()),
                "[a-z._]{0,24}",
            ],
            0..8,
        )
    ) {
        let wrapped = names.iter().any(|name| name.starts_with("model.language_model."));
        let text_only = names.iter().any(|name| name == "model.embed_tokens.weight");
        let result = LanguageNamespace::classify(names.iter().map(String::as_str), "Fixture");
        match (wrapped, text_only) {
            (true, false) => prop_assert_eq!(result.unwrap(), LanguageNamespace::Wrapped),
            (false, true) => prop_assert_eq!(result.unwrap(), LanguageNamespace::TextOnly),
            (false, false) => {
                let missing = matches!(
                    result,
                    Err(CheckpointError::LanguageNamespaceNotFound { family: "Fixture" })
                );
                prop_assert!(missing);
            }
            (true, true) => {
                let ambiguous = matches!(
                    result,
                    Err(CheckpointError::AmbiguousLanguageNamespace { family: "Fixture" })
                );
                prop_assert!(ambiguous);
            }
        }
    }
}
