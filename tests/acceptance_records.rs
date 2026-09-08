//! Structural gates for the provenance records committed with this crate.
//!
//! Acceptance JSON is executable release evidence rather than prose. These
//! checks prevent malformed records, mutable-looking revisions, filename/
//! revision drift, and accidental promotion of a checkpoint while the README
//! still describes the crate as pre-release.

use std::{fs, path::PathBuf};

use serde_json::Value;

/// Returns every committed acceptance record in deterministic filename order.
fn acceptance_records() -> Vec<PathBuf> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("acceptance");
    let mut records = fs::read_dir(&root)
        .unwrap_or_else(|error| panic!("read {}: {error}", root.display()))
        .map(|entry| entry.expect("read acceptance directory entry").path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "json")
        })
        .collect::<Vec<_>>();
    records.sort();
    records
}

/// Every pre-release record is valid JSON and binds an immutable Git revision.
#[test]
fn committed_acceptance_records_are_pinned_and_not_advertised() {
    let records = acceptance_records();
    assert!(
        records.len() >= 7,
        "the known v1 evidence set must not silently disappear"
    );

    for path in records {
        let bytes = fs::read(&path)
            .unwrap_or_else(|error| panic!("read acceptance record {}: {error}", path.display()));
        let record: Value = serde_json::from_slice(&bytes)
            .unwrap_or_else(|error| panic!("parse acceptance record {}: {error}", path.display()));
        assert_eq!(record["schema_version"], 1, "{}", path.display());
        assert_eq!(record["advertised_supported"], false, "{}", path.display());

        let repository = record["repository"]
            .as_str()
            .unwrap_or_else(|| panic!("{} has no repository", path.display()));
        assert!(
            repository.contains('/'),
            "{repository} is not a Hub repository"
        );
        let revision = record["revision"]
            .as_str()
            .unwrap_or_else(|| panic!("{} has no revision", path.display()));
        assert_eq!(
            revision.len(),
            40,
            "{} has a non-SHA revision",
            path.display()
        );
        assert!(
            revision.bytes().all(|byte| byte.is_ascii_hexdigit()),
            "{} has a non-hex revision",
            path.display()
        );
        let filename = path
            .file_name()
            .and_then(|name| name.to_str())
            .expect("acceptance filenames must be UTF-8");
        assert!(
            filename.contains(revision),
            "{} does not bind its revision in its filename",
            path.display()
        );
        assert!(
            record["phase"]
                .as_str()
                .is_some_and(|phase| !phase.is_empty()),
            "{} has no evidence phase",
            path.display()
        );
        assert!(
            record["checks"].is_object(),
            "{} has no check inventory",
            path.display()
        );
    }
}
