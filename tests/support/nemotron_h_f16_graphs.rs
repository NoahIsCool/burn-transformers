//! Strict reader for the pinned official Nemotron-H F16 graph bundles of the precision envelope.
//!
//! The precision gate bounds each path by the envelope of four official F16
//! graphs (see `nemotron_h_parity`): CPU eager (the precision bundle's
//! `float16` rows), MPS eager (the oracle), and the two bundles read here,
//! which `scripts/generate_nemotron_h_f16_graph_reference.py` wrote by
//! evaluating the same official class and checkout in float16 with SDPA
//! attention on the CPU and on MPS, on the oracle's prompt and greedy tokens.
//! Each new graph differs from its device's existing graph only in the
//! attention implementation (the CPU keeps the precision tool's default
//! experts request, which resolves to `grouped_mm`; MPS keeps the oracle's
//! `eager` experts). Both ran with the precision tool's 16 GiB resident
//! budget: the other 23 decoder blocks were loaded from disk onto the
//! evaluating device for their forward, which the generator's tiny-model
//! self-check proves bitwise neutral on that device. A resident MPS load
//! swapped this 96 GiB host beyond the measurement watchdog's limit.
//!
//! `TRANSFORMERS_NEMOTRON_H_F16_GRAPHS_DIR` names a non-symlink directory of
//! the exact name [`GRAPHS_DIRNAME`] that holds exactly the two pinned bundle
//! directories, each holding exactly its manifest and three row files. Every
//! byte is authenticated by a pinned SHA-256 before any value is used, and each
//! manifest must name the graph, the precision bundle, the oracle, the
//! checkout, the checkpoint config, the generator and the precision tool it
//! reused, and the passed self-check this gate relies on.

#![allow(dead_code)]

use super::{
    REVISION, SMOKE_TOKEN_IDS,
    nemotron_h_parity::OfficialF16Graph,
    nemotron_h_precision_reference::{
        CONFIG_SHA256, MANIFEST_SHA256 as PRECISION_MANIFEST_SHA256, ORACLE_MANIFEST_SHA256,
        RowKind, SOURCE_SHA256, TRANSFORMERS_REVISION, authenticated_bytes, decode_rows, field,
    },
};
use serde_json::Value;
use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
};

/// Environment variable selecting the external directory of graph bundles.
const GRAPHS_DIR_ENV: &str = "TRANSFORMERS_NEMOTRON_H_F16_GRAPHS_DIR";

/// Exact name of that directory.
pub(super) const GRAPHS_DIRNAME: &str = "nemotron-3.5-lightning-30b-a3b-a9904d24-f16-graphs";

/// SHA-256 of the generator that wrote both bundles.
const GENERATOR_SHA256: &str = "81f5bd7735e79fde9e0d10bab5049bafa6803e8e469e3f0d9b44d0898e97b48f";

/// SHA-256 of the precision tool whose helpers the generator reused.
const PRECISION_TOOL_SHA256: &str =
    "7d846c4b51cd31acedbdd6f6ac37467f803ccbce99145ec676e1a59d897b759f";

/// Name of each bundle's manifest.
const MANIFEST: &str = "reference.json";

/// One pinned graph bundle.
#[derive(Debug, Clone, Copy)]
pub(super) struct PinnedGraph {
    /// Graph the bundle evaluates.
    pub(super) graph: OfficialF16Graph,
    /// Device the official class ran on.
    device: &'static str,
    /// Attention implementation requested and resolved.
    attention: &'static str,
    /// Experts implementation requested (`None`: Transformers' default request).
    experts_requested: Option<&'static str>,
    /// Experts implementation the request resolved to.
    experts_resolved: &'static str,
    /// Bundle directory name inside [`GRAPHS_DIRNAME`].
    pub(super) dirname: &'static str,
    /// Pinned SHA-256 of the bundle's manifest.
    manifest_sha256: &'static str,
    /// Pinned SHA-256 of each row file.
    files: [(RowKind, &'static str); 3],
}

/// Resident budget, in GiB, both bundles ran with (the precision tool's).
const RESIDENT_GIB: f64 = 16.0;

/// Decoder blocks that budget offloads to disk, of 52.
const OFFLOADED_BLOCKS: u64 = 23;

/// The two pinned bundles, in [`OfficialF16Graph::ALL`] order.
pub(super) const PINNED: [PinnedGraph; 2] = [
    PinnedGraph {
        graph: OfficialF16Graph::CpuSdpa,
        device: "cpu",
        attention: "sdpa",
        experts_requested: None,
        experts_resolved: "grouped_mm",
        dirname: "f16-cpu-sdpa.bundle",
        manifest_sha256: "96253a039c3fb3580cf9fdd2bc12cfb88d6dad265f445f76f6559c40cb2d7f53",
        files: [
            (
                RowKind::Prompt,
                "2693474fbaa46efa994df55c7d0385ed53897944c0a6b83f45e93f02a60edef6",
            ),
            (
                RowKind::Teacher,
                "ac2ea1d0ef480472d23583cddcd5e1fcac833a9b853e7857d4d86608e22cff4a",
            ),
            (
                RowKind::Cached,
                "de2f89005459473d73a3c504f6a547e36b8084116c97428abd8d34d170f174e5",
            ),
        ],
    },
    PinnedGraph {
        graph: OfficialF16Graph::MpsSdpa,
        device: "mps",
        attention: "sdpa",
        experts_requested: Some("eager"),
        experts_resolved: "eager",
        dirname: "f16-mps-sdpa.bundle",
        manifest_sha256: "4298f895b7ed5f174d15008398359e7240c238f89112915860897618b974fba7",
        files: [
            (
                RowKind::Prompt,
                "250f59e53528cf60fc7fd68f9cd45e82ebabf373e4b94b82b0c62cf213071af3",
            ),
            (
                RowKind::Teacher,
                "328f593f52d21c875e7e1688f689f6195028a90107cbaf63aadc054b8413e4e6",
            ),
            (
                RowKind::Cached,
                "d4cb3139faa109835c00b83be79c1174663ff4d68555dcdd4e6b16c9aad2ec8f",
            ),
        ],
    },
];

/// Row file name of one family.
fn row_file(kind: RowKind) -> String {
    format!("float16-{}-logits.f32le", kind.label())
}

/// The authenticated rows of both new graphs.
pub(super) struct ValidatedF16Graphs {
    /// Row matrices by (graph, family), row-major `[rows, vocabulary]`.
    rows: Vec<(OfficialF16Graph, RowKind, Vec<f32>)>,
}

impl ValidatedF16Graphs {
    /// Borrows the complete `[rows, vocabulary]` matrix of one new graph and family.
    ///
    /// # Panics
    ///
    /// For a graph this reader does not hold (CPU eager and MPS eager come from
    /// the precision bundle and the oracle).
    pub(super) fn matrix(&self, graph: OfficialF16Graph, kind: RowKind) -> &[f32] {
        self.rows
            .iter()
            .find(|(candidate, family, _)| *candidate == graph && *family == kind)
            .map(|(_, _, values)| values.as_slice())
            .unwrap_or_else(|| panic!("{} rows are not in the graph bundles", graph.label()))
    }
}

/// Checks one manifest's identity, inputs, graph, provenance, and file anchors.
fn validate_manifest(manifest: &Value, pinned: &PinnedGraph, oracle_generated_ids: &[u32]) {
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
    let graph = field(manifest, &["graph"]);
    assert_eq!(*field(graph, &["dtype"]), "float16");
    assert_eq!(*field(graph, &["device"]), pinned.device);
    assert_eq!(
        *field(graph, &["attention_implementation"]),
        pinned.attention
    );
    assert_eq!(
        *field(graph, &["experts_implementation_requested"]),
        pinned.experts_requested.map_or(Value::Null, Value::from)
    );
    assert_eq!(
        *field(graph, &["resolved", "attention_implementation"]),
        pinned.attention
    );
    assert_eq!(
        *field(graph, &["resolved", "experts_implementation"]),
        pinned.experts_resolved
    );
    for (kind, digest) in pinned.files {
        assert_eq!(
            *field(manifest, &["files_sha256", row_file(kind).as_str()]),
            digest
        );
    }
    let provenance = field(manifest, &["provenance"]);
    assert_eq!(*field(provenance, &["script_sha256"]), GENERATOR_SHA256);
    assert_eq!(
        *field(provenance, &["precision_tool_sha256"]),
        PRECISION_TOOL_SHA256
    );
    assert_eq!(
        *field(provenance, &["precision_manifest_sha256"]),
        PRECISION_MANIFEST_SHA256
    );
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
    assert_eq!(*field(provenance, &["config_sha256"]), CONFIG_SHA256);
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
    assert_eq!(*field(provenance, &["torch"]), "2.10.0");
    assert_eq!(*field(provenance, &["transformers"]), "5.17.0.dev0");
    assert_eq!(
        *field(provenance, &["environment", "PYTORCH_ENABLE_MPS_FALLBACK"]),
        "0"
    );
    let check = field(provenance, &["self_check"]);
    assert_eq!(*field(check, &["harness_bitwise_equal"]), true);
    assert_eq!(*field(check, &["finite"]), true);
    assert_eq!(*field(check, &["offloaded_bitwise_equal"]), true);
    assert_eq!(*field(provenance, &["max_resident_gib"]), RESIDENT_GIB);
    assert_eq!(*field(provenance, &["offloaded_blocks"]), OFFLOADED_BLOCKS);
    assert_eq!(
        field(provenance, &["offload_execution_devices"]).as_array(),
        Some(&vec![Value::from(pinned.device)])
    );
    assert_eq!(
        *field(check, &["graph", "attention_implementation"]),
        pinned.attention
    );
    assert_eq!(
        *field(check, &["graph", "experts_implementation"]),
        pinned.experts_resolved
    );
}

/// Requires `path` to be a non-symlink directory whose entries are exactly `expected`.
fn exact_directory(path: &Path, expected: &BTreeSet<String>) {
    let metadata = fs::symlink_metadata(path)
        .unwrap_or_else(|error| panic!("inspect {}: {error}", path.display()));
    assert!(
        metadata.is_dir() && !metadata.file_type().is_symlink(),
        "{} must be a non-symlink directory",
        path.display()
    );
    let entries = fs::read_dir(path)
        .unwrap_or_else(|error| panic!("list {}: {error}", path.display()))
        .map(|entry| {
            entry
                .expect("directory entry")
                .file_name()
                .into_string()
                .expect("UTF-8 entry name")
        })
        .collect::<BTreeSet<_>>();
    assert_eq!(
        &entries,
        expected,
        "{} must hold exactly its pinned members",
        path.display()
    );
}

/// Resolves the graphs directory from the environment, requiring its exact name and members.
fn graphs_root() -> PathBuf {
    let supplied = PathBuf::from(std::env::var_os(GRAPHS_DIR_ENV).unwrap_or_else(|| {
        panic!("set {GRAPHS_DIR_ENV} to the pinned {GRAPHS_DIRNAME} directory")
    }));
    assert_eq!(
        supplied.file_name().and_then(|name| name.to_str()),
        Some(GRAPHS_DIRNAME),
        "the F16 graphs directory name changed"
    );
    exact_directory(
        &supplied,
        &PINNED
            .iter()
            .map(|pinned| pinned.dirname.to_owned())
            .collect(),
    );
    supplied
}

/// Authenticates both pinned bundles named by the environment and loads every row.
pub(super) fn load_and_validate_f16_graphs(oracle_generated_ids: &[u32]) -> ValidatedF16Graphs {
    let root = graphs_root();
    let mut rows = Vec::new();
    for pinned in &PINNED {
        let bundle = root.join(pinned.dirname);
        let mut members = RowKind::ALL.map(row_file).to_vec();
        members.push(MANIFEST.to_owned());
        exact_directory(&bundle, &members.into_iter().collect());
        let manifest_bytes = authenticated_bytes(&bundle.join(MANIFEST), pinned.manifest_sha256);
        let manifest: Value =
            serde_json::from_slice(&manifest_bytes).expect("an F16 graph manifest must be JSON");
        validate_manifest(&manifest, pinned, oracle_generated_ids);
        for (kind, digest) in pinned.files {
            let name = row_file(kind);
            let bytes = authenticated_bytes(&bundle.join(&name), digest);
            rows.push((pinned.graph, kind, decode_rows(&name, &bytes, kind.rows())));
        }
    }
    ValidatedF16Graphs { rows }
}

#[cfg(test)]
mod tests {
    //! Synthetic tests for the pinned inventory and fail-closed manifest validation.

    use super::*;
    use crate::nemotron_h_base_reference::REFERENCE_DECISIONS;

    /// A manifest exactly as the generator writes it for `pinned`.
    fn manifest(pinned: &PinnedGraph, generated: &[u32]) -> Value {
        let mut files = serde_json::Map::new();
        for (kind, digest) in pinned.files {
            files.insert(row_file(kind), Value::from(digest));
        }
        let sources = SOURCE_SHA256
            .iter()
            .map(|(path, digest)| ((*path).to_owned(), Value::from(*digest)))
            .collect::<serde_json::Map<_, _>>();
        serde_json::json!({
            "schema_version": 1,
            "revision": REVISION,
            "prompt_ids": SMOKE_TOKEN_IDS,
            "generated_ids": generated,
            "graph": {
                "dtype": "float16",
                "device": pinned.device,
                "attention_implementation": pinned.attention,
                "experts_implementation_requested": pinned.experts_requested,
                "resolved": {"attention_implementation": pinned.attention,
                             "experts_implementation": pinned.experts_resolved},
            },
            "files_sha256": files,
            "provenance": {
                "script_sha256": GENERATOR_SHA256,
                "precision_tool_sha256": PRECISION_TOOL_SHA256,
                "precision_manifest_sha256": PRECISION_MANIFEST_SHA256,
                "oracle_sha256": {"nemotron-3.5-lightning-30b-a3b-a9904d24-reference.json": ORACLE_MANIFEST_SHA256},
                "config_sha256": CONFIG_SHA256,
                "transformers_checkout": {"revision": TRANSFORMERS_REVISION, "clean": true, "sources_sha256": sources},
                "torch": "2.10.0",
                "transformers": "5.17.0.dev0",
                "environment": {"PYTORCH_ENABLE_MPS_FALLBACK": "0"},
                "max_resident_gib": RESIDENT_GIB,
                "offloaded_blocks": OFFLOADED_BLOCKS,
                "offload_execution_devices": [pinned.device],
                "self_check": {
                    "harness_bitwise_equal": true,
                    "finite": true,
                    "offloaded_bitwise_equal": true,
                    "graph": {"attention_implementation": pinned.attention,
                              "experts_implementation": pinned.experts_resolved},
                },
            },
        })
    }

    /// The inventory pins the two new graphs once each, with distinct directories and files.
    #[test]
    fn inventory_pins_the_two_new_graphs() {
        assert_eq!(
            PINNED.map(|pinned| pinned.graph),
            [OfficialF16Graph::CpuSdpa, OfficialF16Graph::MpsSdpa]
        );
        assert_ne!(PINNED[0].dirname, PINNED[1].dirname);
        for pinned in &PINNED {
            assert_eq!(pinned.files.map(|(kind, _)| kind), RowKind::ALL);
        }
        assert_eq!(row_file(RowKind::Teacher), "float16-teacher-logits.f32le");
    }

    /// Each manifest validates as written and fails on any changed identity.
    #[test]
    fn manifest_validation_is_fail_closed() {
        let generated = (0..REFERENCE_DECISIONS as u32).collect::<Vec<_>>();
        for pinned in &PINNED {
            let manifest = manifest(pinned, &generated);
            validate_manifest(&manifest, pinned, &generated);
            let other = &PINNED[usize::from(pinned.device == "cpu")];
            assert!(
                std::panic::catch_unwind(|| validate_manifest(&manifest, other, &generated))
                    .is_err()
            );
            let mut changed_tokens = generated.clone();
            changed_tokens[3] += 1;
            assert!(
                std::panic::catch_unwind(|| validate_manifest(&manifest, pinned, &changed_tokens))
                    .is_err()
            );
            for pointer in [
                "/revision",
                "/graph/dtype",
                "/graph/attention_implementation",
                "/graph/experts_implementation_requested",
                "/graph/resolved/experts_implementation",
                "/files_sha256/float16-cached-logits.f32le",
                "/provenance/script_sha256",
                "/provenance/precision_tool_sha256",
                "/provenance/precision_manifest_sha256",
                "/provenance/config_sha256",
                "/provenance/transformers_checkout/clean",
                "/provenance/torch",
                "/provenance/environment/PYTORCH_ENABLE_MPS_FALLBACK",
                "/provenance/self_check/harness_bitwise_equal",
                "/provenance/self_check/offloaded_bitwise_equal",
                "/provenance/max_resident_gib",
                "/provenance/offloaded_blocks",
                "/provenance/offload_execution_devices",
            ] {
                let mut tampered = manifest.clone();
                *tampered.pointer_mut(pointer).expect("fixture pointer") = Value::from("tampered");
                assert!(
                    std::panic::catch_unwind(|| validate_manifest(&tampered, pinned, &generated))
                        .is_err(),
                    "{pointer} must be checked"
                );
            }
        }
    }

    /// A directory must hold exactly its pinned members and must not be a symlink.
    #[test]
    fn directories_must_hold_exactly_their_members() {
        let root = std::env::temp_dir().join(format!(
            "nemotron-f16-graphs-{}-{}",
            std::process::id(),
            line!()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("a")).expect("fixture directory");
        let expected = ["a".to_owned()].into_iter().collect::<BTreeSet<_>>();
        exact_directory(&root, &expected);
        fs::write(root.join("extra"), b"").expect("fixture file");
        assert!(std::panic::catch_unwind(|| exact_directory(&root, &expected)).is_err());
        fs::remove_file(root.join("extra")).expect("remove fixture file");
        #[cfg(unix)]
        {
            let link = root.with_extension("link");
            let _ = fs::remove_file(&link);
            std::os::unix::fs::symlink(&root, &link).expect("fixture symlink");
            assert!(std::panic::catch_unwind(|| exact_directory(&link, &expected)).is_err());
            fs::remove_file(&link).expect("remove fixture symlink");
        }
        fs::remove_dir_all(&root).expect("remove fixture");
    }
}
