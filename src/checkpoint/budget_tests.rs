//! Tests for the pre-contract tensor budget of checkpoint planning.

use std::{fs::File, io::Write, path::Path};

use proptest::prelude::*;

use super::*;
use crate::checkpoint::{KeyMatcher, discover_checkpoint};

/// Writes a single-file checkpoint with `count` one-element F32 tensors and discovers it.
fn manifest_with(root: &Path, count: usize) -> CheckpointManifest {
    let entries = (0..count)
        .map(|index| {
            format!(
                r#""tensor.{index}":{{"dtype":"F32","shape":[1],"data_offsets":[{},{}]}}"#,
                4 * index,
                4 * (index + 1)
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    let header = format!("{{{entries}}}");
    let mut file = File::create(root.join("model.safetensors")).unwrap();
    file.write_all(&(header.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(header.as_bytes()).unwrap();
    file.write_all(&vec![0; 4 * count]).unwrap();
    discover_checkpoint(root, &LoadOptions::default()).unwrap()
}

/// A bound up to the discovered count passes; one more fails and names both counts.
#[test]
fn the_budget_admits_exactly_the_discovered_count() {
    let root = tempfile::TempDir::new().unwrap();
    let manifest = manifest_with(root.path(), 3);
    let options = LoadOptions::default();
    for required in 0..=3 {
        require_source_tensors(&manifest, &options, required).unwrap();
    }
    let error = require_source_tensors(&manifest, &options, 4).unwrap_err();
    assert!(matches!(
        error,
        CheckpointError::TooFewTensors {
            required: 4,
            available: 3
        }
    ));
    assert_eq!(
        error.to_string(),
        "the configuration needs at least 4 checkpoint tensors, but the checkpoint provides \
         only 3 (tensors plus index aliases)"
    );
}

/// A missing-tensor allowlist can excuse any number of tensors, so the bound is not applied.
#[test]
fn a_missing_tensor_allowlist_disables_the_budget() {
    let root = tempfile::TempDir::new().unwrap();
    let manifest = manifest_with(root.path(), 1);
    let options = LoadOptions {
        allow_missing: vec![KeyMatcher::Prefix("model.layers.".into())],
        ..LoadOptions::default()
    };
    require_source_tensors(&manifest, &options, usize::MAX).unwrap();
}

/// Resolved index-only aliases count as available sources.
#[test]
fn index_aliases_count_toward_the_available_sources() {
    let root = tempfile::TempDir::new().unwrap();
    let mut manifest = manifest_with(root.path(), 2);
    manifest
        .index_aliases
        .push(crate::checkpoint::ResolvedIndexTensorAlias::new(
            "lm_head.weight",
            "tensor.0",
            "model.safetensors",
        ));
    let options = LoadOptions::default();
    require_source_tensors(&manifest, &options, 3).unwrap();
    assert!(require_source_tensors(&manifest, &options, 4).is_err());
}

proptest! {
    /// The verdict is `required <= available`, and a rejection reports both counts.
    #[test]
    fn the_budget_is_a_comparison(required in any::<usize>(), available in any::<usize>()) {
        match tensor_budget(required, available) {
            Ok(()) => prop_assert!(required <= available),
            Err(CheckpointError::TooFewTensors { required: r, available: a }) => {
                prop_assert!(required > available);
                prop_assert_eq!((r, a), (required, available));
            }
            Err(other) => prop_assert!(false, "unexpected error {other}"),
        }
    }
}
