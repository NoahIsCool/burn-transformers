//! Strict reader for the pinned Nemotron-H official precision bundle.
//!
//! `scripts/generate_nemotron_h_precision_reference.py` evaluates the same
//! official class and checkout as the MPS/F16 oracle on the CPU, on the
//! oracle's prompt and recorded greedy tokens, in F32 (BF16-resident blocks
//! widened while they run, which is exact), BF16, and F16. Its F32 rows are the
//! numerical truth the corresponding-path gate compares Burn with; its F16 rows
//! are a second official F16 implementation. Every byte is authenticated by a
//! pinned SHA-256 before any value is used, and the manifest's provenance must
//! name the oracle, checkout, checkpoint, and passed widening self-check this
//! gate relies on.

#![allow(dead_code)]

use super::{
    REVISION, SMOKE_TOKEN_IDS,
    nemotron_h_base_reference::{
        OfficialExecutionPath, REFERENCE_DECISIONS, REFERENCE_VOCABULARY_SIZE,
    },
};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::{Path, PathBuf},
};

/// Environment variable selecting the external immutable bundle.
const PRECISION_DIR_ENV: &str = "TRANSFORMERS_NEMOTRON_H_PRECISION_DIR";

/// Exact create-only directory emitted by the generator.
const BUNDLE_DIRNAME: &str = "nemotron-3.5-lightning-30b-a3b-a9904d24-precision.bundle";

/// Pinned SHA-256 of the bundle manifest.
const MANIFEST_SHA256: &str = "5da4427df4c882d3c4d42b5c2d37c2c61a4837597c944b9c67a23903a7ce73f9";

/// Official Transformers revision evaluated.
const TRANSFORMERS_REVISION: &str = "cfa309cfa34b5497a32fd50af49aa97aac94c054";

/// Exact config identity of the evaluated checkpoint.
const CONFIG_SHA256: &str = "a3827a0f5e311547b40943dc081e3ff2f8a277466e8c1a3df2291e8db8a7617c";

/// Pinned SHA-256 of the MPS/F16 oracle manifest the evaluation read its inputs from.
const ORACLE_MANIFEST_SHA256: &str =
    "bd3abb417f2fee82d1bef8bb654ca846fa4816351b1d9d7d0d057105f92de196";

/// Official Nemotron-H sources executed, by checkout-relative path.
const SOURCE_SHA256: [(&str, &str); 2] = [
    (
        "src/transformers/models/nemotron_h/configuration_nemotron_h.py",
        "00a4c60bd302d796b25c6ae25c608a5ee7e27f9a0adea92911b0c45470232052",
    ),
    (
        "src/transformers/models/nemotron_h/modeling_nemotron_h.py",
        "d42067429e3268d1163579f5d257fe0baa9a2b02edba5fcb175fb8d42c9b6af4",
    ),
];

/// One evaluation of the bundle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Precision {
    /// BF16-resident blocks widened to F32 while they run: the truth.
    Float32,
    /// The official class loaded and evaluated in F16 on the CPU.
    Float16,
    /// The official class loaded and evaluated in BF16 on the CPU.
    Bfloat16,
}

impl Precision {
    /// The bundle's label for this evaluation.
    const fn label(self) -> &'static str {
        match self {
            Self::Float32 => "float32",
            Self::Float16 => "float16",
            Self::Bfloat16 => "bfloat16",
        }
    }
}

/// One row family recorded for every evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum RowKind {
    /// The stateless prompt row (`logits_to_keep=1` on the prompt alone).
    Prompt,
    /// The 32 teacher-forced rows of one stateless chunk.
    Teacher,
    /// Cached prefill plus 31 one-token decodes.
    Cached,
}

impl RowKind {
    /// The bundle's label for this row family.
    const fn label(self) -> &'static str {
        match self {
            Self::Prompt => "prompt",
            Self::Teacher => "teacher",
            Self::Cached => "cached",
        }
    }

    /// Number of full-vocabulary rows recorded.
    pub(super) const fn rows(self) -> usize {
        match self {
            Self::Prompt => 1,
            Self::Teacher | Self::Cached => REFERENCE_DECISIONS,
        }
    }

    /// The official Mamba execution stream that produced these rows.
    pub(super) const fn path(self) -> OfficialExecutionPath {
        match self {
            Self::Prompt | Self::Teacher => OfficialExecutionPath::Stateless,
            Self::Cached => OfficialExecutionPath::Cached,
        }
    }
}

/// Pinned SHA-256 of every logit file, by (evaluation, row family).
const FILES: [(Precision, RowKind, &str); 9] = [
    (
        Precision::Float32,
        RowKind::Prompt,
        "568f634d35d1e3a88fde59ee80a9c26caf5d59c2ac5eefbed6350acf16f8e588",
    ),
    (
        Precision::Float32,
        RowKind::Teacher,
        "981f2f984f807012786103510ff8caaba79def48201bd0736bcdf38b48c62c88",
    ),
    (
        Precision::Float32,
        RowKind::Cached,
        "5b1dd446712ca764bf655fe324776e4468579e83f1b719e3437af96d817224f0",
    ),
    (
        Precision::Float16,
        RowKind::Prompt,
        "d9fd6aee0ed30f0252b7141fd951e90cd2cb5aacd034c5422f833b3a949ac21f",
    ),
    (
        Precision::Float16,
        RowKind::Teacher,
        "6dc5e0a295f9bb47c01f65b52a1026855b1866332a94d845e463545b2107f867",
    ),
    (
        Precision::Float16,
        RowKind::Cached,
        "d01d94bbbdce744e6621c717707f5a527a9870459115c91557e69104c772c7f6",
    ),
    (
        Precision::Bfloat16,
        RowKind::Prompt,
        "c84aa9bc69b26358a41dedbbfd5394452d733284a77bc7cc03143c79cbe55ea4",
    ),
    (
        Precision::Bfloat16,
        RowKind::Teacher,
        "b888d62713d068185e26a35c9e82a6b77f3a141f6045dd812483ec32130d4633",
    ),
    (
        Precision::Bfloat16,
        RowKind::Cached,
        "cd38799625f1c5901e68df54b7ca8887becdc803c9ede6e93000b86b362c736e",
    ),
];

/// One authenticated F32-truth row bound to its execution stream and decision.
#[derive(Debug, Clone, Copy)]
pub(super) struct TruthRow<'a> {
    /// Official stream that produced the row.
    path: OfficialExecutionPath,
    /// Row family the row belongs to.
    kind: RowKind,
    /// Zero-based decision (always 0 for the prompt row).
    decision: usize,
    /// Complete vocabulary logits in token-ID order.
    values: &'a [f32],
}

impl<'a> TruthRow<'a> {
    /// Returns the official execution stream that produced this row.
    pub(super) const fn path(&self) -> OfficialExecutionPath {
        self.path
    }

    /// Returns the row family.
    pub(super) const fn kind(&self) -> RowKind {
        self.kind
    }

    /// Returns the zero-based decision represented by this row.
    pub(super) const fn decision(&self) -> usize {
        self.decision
    }

    /// Borrows all vocabulary logits in token-ID order.
    pub(super) const fn values(&self) -> &'a [f32] {
        self.values
    }

    /// Builds a synthetic row for pure contract tests.
    #[cfg(test)]
    pub(super) const fn synthetic(kind: RowKind, decision: usize, values: &'a [f32]) -> Self {
        Self {
            path: kind.path(),
            kind,
            decision,
            values,
        }
    }
}

/// The authenticated precision bundle: every row of every evaluation.
pub(super) struct ValidatedPrecisionReference {
    /// Row matrices in `FILES` order, row-major `[rows, vocabulary]`.
    rows: Vec<(Precision, RowKind, Vec<f32>)>,
    /// The manifest's recorded generation tokens, equal to the oracle's.
    pub(super) generated_ids: Vec<u32>,
}

impl ValidatedPrecisionReference {
    /// Borrows the complete `[rows, vocabulary]` matrix of one evaluation and row family.
    pub(super) fn matrix(&self, precision: Precision, kind: RowKind) -> &[f32] {
        self.rows
            .iter()
            .find(|(candidate, family, _)| *candidate == precision && *family == kind)
            .map(|(_, _, values)| values.as_slice())
            .expect("every evaluation and row family is loaded")
    }

    /// Borrows one row of an evaluation as plain values.
    pub(super) fn row(&self, precision: Precision, kind: RowKind, decision: usize) -> &[f32] {
        assert!(
            decision < kind.rows(),
            "{} has no decision {decision}",
            kind.label()
        );
        let start = decision * REFERENCE_VOCABULARY_SIZE;
        &self.matrix(precision, kind)[start..start + REFERENCE_VOCABULARY_SIZE]
    }

    /// Returns the F32 truth row of `kind` at `decision`, bound to its stream.
    pub(super) fn truth(&self, kind: RowKind, decision: usize) -> TruthRow<'_> {
        TruthRow {
            path: kind.path(),
            kind,
            decision,
            values: self.row(Precision::Float32, kind, decision),
        }
    }
}

/// Hex SHA-256 of exact bytes.
fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// Reads `path`, refusing symlinks, and checks its pinned digest.
fn authenticated_bytes(path: &Path, expected_sha256: &str) -> Vec<u8> {
    let metadata = fs::symlink_metadata(path)
        .unwrap_or_else(|error| panic!("inspect {}: {error}", path.display()));
    assert!(
        metadata.is_file() && !metadata.file_type().is_symlink(),
        "{} must be a regular non-symlink file",
        path.display()
    );
    let bytes = fs::read(path).unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    let actual = sha256_hex(&bytes);
    assert_eq!(
        actual,
        expected_sha256,
        "{} changed: SHA-256 {actual}",
        path.display()
    );
    bytes
}

/// Decodes little-endian binary32 values, requiring the exact row geometry and finite values.
fn decode_rows(label: &str, bytes: &[u8], rows: usize) -> Vec<f32> {
    assert_eq!(
        bytes.len(),
        rows * REFERENCE_VOCABULARY_SIZE * 4,
        "{label} must hold {rows} full-vocabulary rows"
    );
    let values = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| f32::from_le_bytes(*word))
        .collect::<Vec<_>>();
    assert!(
        values.iter().all(|value| value.is_finite()),
        "{label} contains a non-finite value"
    );
    values
}

/// Reads one manifest field, failing with its JSON path.
fn field<'a>(value: &'a Value, path: &[&str]) -> &'a Value {
    path.iter().fold(value, |node, key| {
        node.get(key)
            .unwrap_or_else(|| panic!("precision manifest lacks {}", path.join(".")))
    })
}

/// Checks the manifest's identity, inputs, evaluations, provenance, and file anchors.
fn validate_manifest(manifest: &Value, oracle_generated_ids: &[u32]) {
    assert_eq!(*field(manifest, &["schema_version"]), 1);
    assert_eq!(*field(manifest, &["revision"]), REVISION);
    let prompt = SMOKE_TOKEN_IDS
        .iter()
        .map(|&id| Value::from(id))
        .collect::<Vec<_>>();
    assert_eq!(field(manifest, &["prompt_ids"]).as_array(), Some(&prompt));
    let generated = oracle_generated_ids
        .iter()
        .map(|&id| Value::from(id))
        .collect::<Vec<_>>();
    assert_eq!(
        field(manifest, &["generated_ids"]).as_array(),
        Some(&generated)
    );
    let evaluations = field(manifest, &["evaluations"])
        .as_array()
        .expect("evaluations array");
    let expected = [
        ("bfloat16", "bfloat16", "bfloat16"),
        ("float32", "bfloat16", "float32"),
        ("float16", "float16", "float16"),
    ];
    assert_eq!(evaluations.len(), expected.len(), "evaluation list changed");
    for (evaluation, (label, load, run)) in evaluations.iter().zip(expected) {
        assert_eq!(*field(evaluation, &["label"]), label);
        assert_eq!(*field(evaluation, &["load_dtype"]), load);
        assert_eq!(*field(evaluation, &["evaluation_dtype"]), run);
        assert_eq!(*field(evaluation, &["device"]), "cpu");
    }
    let provenance = field(manifest, &["provenance"]);
    assert_eq!(*field(provenance, &["config_sha256"]), CONFIG_SHA256);
    assert_eq!(*field(provenance, &["attention_implementation"]), "eager");
    assert_eq!(
        *field(provenance, &["transformers_checkout", "revision"]),
        TRANSFORMERS_REVISION
    );
    assert_eq!(
        *field(provenance, &["transformers_checkout", "clean"]),
        true
    );
    for (relative, digest) in SOURCE_SHA256 {
        assert_eq!(
            *field(
                provenance,
                &["transformers_checkout", "sources_sha256", relative]
            ),
            digest
        );
    }
    assert_eq!(
        *field(
            provenance,
            &[
                "oracle_sha256",
                "nemotron-3.5-lightning-30b-a3b-a9904d24-reference.json"
            ]
        ),
        ORACLE_MANIFEST_SHA256
    );
    assert_eq!(
        *field(
            provenance,
            &["widening_self_check", "bitwise_equal", "resident"]
        ),
        true
    );
    assert_eq!(
        *field(
            provenance,
            &["widening_self_check", "bitwise_equal", "offloaded"]
        ),
        true
    );
    for (precision, kind, digest) in FILES {
        let name = format!("{}-{}-logits.f32le", precision.label(), kind.label());
        assert_eq!(*field(manifest, &["files_sha256", name.as_str()]), digest);
    }
}

/// Resolves the bundle directory from the environment, requiring its exact name.
fn bundle_root() -> PathBuf {
    let supplied = PathBuf::from(std::env::var_os(PRECISION_DIR_ENV).unwrap_or_else(|| {
        panic!("set {PRECISION_DIR_ENV} to the pinned {BUNDLE_DIRNAME} directory")
    }));
    let metadata = fs::symlink_metadata(&supplied)
        .unwrap_or_else(|error| panic!("inspect {}: {error}", supplied.display()));
    assert!(
        metadata.is_dir() && !metadata.file_type().is_symlink(),
        "the precision bundle must be a non-symlink directory"
    );
    assert_eq!(
        supplied.file_name().and_then(|name| name.to_str()),
        Some(BUNDLE_DIRNAME),
        "the precision bundle directory name changed"
    );
    supplied
}

/// Authenticates the bundle named by the environment and loads every row.
pub(super) fn load_and_validate_precision_bundle(
    oracle_generated_ids: &[u32],
) -> ValidatedPrecisionReference {
    let root = bundle_root();
    let manifest_bytes = authenticated_bytes(&root.join("reference.json"), MANIFEST_SHA256);
    let manifest: Value =
        serde_json::from_slice(&manifest_bytes).expect("the precision manifest must be JSON");
    validate_manifest(&manifest, oracle_generated_ids);
    let rows = FILES
        .iter()
        .map(|&(precision, kind, digest)| {
            let name = format!("{}-{}-logits.f32le", precision.label(), kind.label());
            let bytes = authenticated_bytes(&root.join(&name), digest);
            (precision, kind, decode_rows(&name, &bytes, kind.rows()))
        })
        .collect();
    ValidatedPrecisionReference {
        rows,
        generated_ids: oracle_generated_ids.to_vec(),
    }
}

#[cfg(test)]
mod tests {
    //! Synthetic tests for row families, payload decoding, and fail-closed manifest validation.

    use super::*;

    /// Row families map to the official streams and carry their decision counts.
    #[test]
    fn row_families_bind_their_streams_and_counts() {
        assert_eq!(RowKind::Prompt.path(), OfficialExecutionPath::Stateless);
        assert_eq!(RowKind::Teacher.path(), OfficialExecutionPath::Stateless);
        assert_eq!(RowKind::Cached.path(), OfficialExecutionPath::Cached);
        assert_eq!(RowKind::Prompt.rows(), 1);
        assert_eq!(RowKind::Teacher.rows(), REFERENCE_DECISIONS);
        assert_eq!(FILES.len(), 9);
        let mut names = FILES
            .iter()
            .map(|(precision, kind, _)| format!("{}-{}", precision.label(), kind.label()))
            .collect::<Vec<_>>();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), 9, "every evaluation and family appears once");
    }

    /// Decoding requires the exact geometry and finite values.
    #[test]
    fn rows_decode_exactly_and_reject_malformed_payloads() {
        let mut bytes = Vec::new();
        for index in 0..REFERENCE_VOCABULARY_SIZE {
            bytes.extend_from_slice(&(index as f32 * 0.5).to_le_bytes());
        }
        let values = decode_rows("prompt", &bytes, 1);
        assert_eq!(values[3], 1.5);
        assert!(std::panic::catch_unwind(|| decode_rows("prompt", &bytes[4..], 1)).is_err());
        let mut infinite = bytes.clone();
        infinite[..4].copy_from_slice(&f32::INFINITY.to_le_bytes());
        assert!(std::panic::catch_unwind(|| decode_rows("prompt", &infinite, 1)).is_err());
    }

    /// The manifest validator accepts the recorded shape and rejects any changed identity.
    #[test]
    fn manifest_validation_is_fail_closed() {
        let generated = (0..REFERENCE_DECISIONS as u32).collect::<Vec<_>>();
        let mut files = serde_json::Map::new();
        for (precision, kind, digest) in FILES {
            files.insert(
                format!("{}-{}-logits.f32le", precision.label(), kind.label()),
                Value::from(digest),
            );
        }
        let sources = SOURCE_SHA256
            .iter()
            .map(|(path, digest)| ((*path).to_owned(), Value::from(*digest)))
            .collect::<serde_json::Map<_, _>>();
        let manifest = serde_json::json!({
            "schema_version": 1,
            "revision": REVISION,
            "prompt_ids": SMOKE_TOKEN_IDS,
            "generated_ids": generated,
            "evaluations": [
                {"label": "bfloat16", "load_dtype": "bfloat16", "evaluation_dtype": "bfloat16", "device": "cpu"},
                {"label": "float32", "load_dtype": "bfloat16", "evaluation_dtype": "float32", "device": "cpu"},
                {"label": "float16", "load_dtype": "float16", "evaluation_dtype": "float16", "device": "cpu"},
            ],
            "files_sha256": files,
            "provenance": {
                "config_sha256": CONFIG_SHA256,
                "attention_implementation": "eager",
                "transformers_checkout": {"revision": TRANSFORMERS_REVISION, "clean": true, "sources_sha256": sources},
                "oracle_sha256": {"nemotron-3.5-lightning-30b-a3b-a9904d24-reference.json": ORACLE_MANIFEST_SHA256},
                "widening_self_check": {"bitwise_equal": {"resident": true, "offloaded": true}},
            },
        });
        validate_manifest(&manifest, &generated);
        let mut changed_tokens = generated.clone();
        changed_tokens[5] += 1;
        assert!(
            std::panic::catch_unwind(|| validate_manifest(&manifest, &changed_tokens)).is_err()
        );
        for pointer in [
            "/revision",
            "/provenance/transformers_checkout/revision",
            "/provenance/widening_self_check/bitwise_equal/offloaded",
            "/evaluations/1/evaluation_dtype",
            "/files_sha256/float32-teacher-logits.f32le",
        ] {
            let mut tampered = manifest.clone();
            *tampered.pointer_mut(pointer).expect("fixture pointer") = Value::from("tampered");
            assert!(
                std::panic::catch_unwind(|| validate_manifest(&tampered, &generated)).is_err(),
                "{pointer} must be checked"
            );
        }
    }
}
