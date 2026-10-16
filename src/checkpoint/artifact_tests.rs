//! Unit tests for discovery's private parsing, diagnostics, and header arithmetic.
//!
//! Discovery-level behavior (index reconciliation, aliases, and path
//! containment) is exercised through the public API in `checkpoint_tests.rs`;
//! these tests pin the private serde visitors and error classifiers whose
//! failure branches the public entry points guard before reaching them.

use std::{fs, path::Path};

use tempfile::TempDir;

use super::*;

/// A non-object SafeTensors header names the expected top-level shape.
///
/// Discovery rejects a header that does not begin with `{` before Serde runs,
/// so the visitor's `expecting` diagnostic is reachable only through a direct
/// parse. It must still describe the contract rather than a generic map.
#[test]
fn raw_header_visitor_names_the_expected_object() {
    for json in ["[]", "7", "\"header\"", "null"] {
        let error = serde_json::from_str::<RawHeader>(json).unwrap_err();
        assert!(
            error.to_string().contains("a SafeTensors header object"),
            "{json}: {error}"
        );
    }
    let header = serde_json::from_str::<RawHeader>(
        r#"{"__metadata__":{"format":"pt"},"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
    )
    .unwrap();
    assert_eq!(header.metadata["format"], "pt");
    assert_eq!(header.tensors["w"].data_offsets, [0, 4]);
}

/// Only a backticked duplicate-key diagnostic becomes `DuplicateJsonKey`.
///
/// A parser message that merely contains the word "duplicate" (for example,
/// inside an invalid string value) carries no key, so it must remain an
/// ordinary contextual `InvalidJson` error with the parser text preserved.
#[test]
fn json_error_classification_requires_a_quoted_duplicate_key() {
    let path = Path::new("model.safetensors");
    let untagged = serde_json::from_str::<u8>("\"duplicate\"").unwrap_err();
    let message = untagged.to_string();
    assert!(message.contains("duplicate") && !message.contains('`'));
    let CheckpointError::InvalidJson {
        kind,
        path: reported,
        message: preserved,
    } = classify_json_error("SafeTensors header", path, untagged)
    else {
        panic!("an untagged message must stay an InvalidJson diagnostic");
    };
    assert_eq!(kind, "SafeTensors header");
    assert_eq!(reported, path);
    assert_eq!(preserved, message);

    let duplicate = serde_json::from_str::<UniqueStringMap>(r#"{"k":"a","k":"b"}"#).unwrap_err();
    assert!(matches!(
        classify_json_error("checkpoint index", path, duplicate),
        CheckpointError::DuplicateJsonKey {
            kind: "checkpoint index",
            key,
            ..
        } if key == "k"
    ));
}

/// A header length whose data start overflows `u64` is an arithmetic error.
///
/// The default header limit rejects such a prefix first; with the limit raised
/// to `u64::MAX`, the verified data-section arithmetic must still refuse to
/// wrap `8 + header_bytes` rather than inventing a data section.
#[test]
fn overflowing_header_length_is_rejected_before_reading_the_header() {
    let temp = TempDir::new().unwrap();
    for header_bytes in [u64::MAX, u64::MAX - 7] {
        let mut bytes = header_bytes.to_le_bytes().to_vec();
        bytes.extend_from_slice(b"{}");
        fs::write(temp.path().join("model.safetensors"), &bytes).unwrap();
        let mut options = LoadOptions::default();
        options.limits.max_header_bytes = u64::MAX;
        assert!(
            matches!(
                discover_checkpoint(temp.path(), &options),
                Err(CheckpointError::ArithmeticOverflow {
                    context: "computing SafeTensors data start"
                })
            ),
            "{header_bytes}"
        );
    }
    // The largest non-overflowing prefix is an ordinary impossible length.
    let mut bytes = (u64::MAX - 8).to_le_bytes().to_vec();
    bytes.extend_from_slice(b"{}");
    fs::write(temp.path().join("model.safetensors"), &bytes).unwrap();
    let mut options = LoadOptions::default();
    options.limits.max_header_bytes = u64::MAX;
    assert!(matches!(
        discover_checkpoint(temp.path(), &options),
        Err(CheckpointError::InvalidHeaderLength {
            header_bytes,
            file_bytes: 10,
            ..
        }) if header_bytes == u64::MAX - 8
    ));
}
