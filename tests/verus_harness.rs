//! Structural checks for the in-place Verus proofs described in `verus/README.md`.
//!
//! Verus itself is an external tool with a pinned toolchain, so ordinary
//! `cargo test` cannot run the proofs. These tests keep the proof surface
//! honest between Verus runs: every harness must include real source files,
//! proof code must not use trusted escape hatches, and any source file that
//! carries Verus specifications must actually be checked by some harness.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

/// Tokens that would weaken a proof into an assumption.
const FORBIDDEN_PROOF_TOKENS: &[&str] = &[
    "admit(",
    "assume(",
    "assume_specification",
    "external_body",
    "external_fn_specification",
];

/// Returns the repository root containing `Cargo.toml`.
fn repository_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Lists every Verus harness root, i.e. each `verus/*.rs` file.
fn harnesses(root: &Path) -> Vec<PathBuf> {
    let mut files = fs::read_dir(root.join("verus"))
        .expect("read verus directory")
        .map(|entry| entry.expect("read verus entry").path())
        .filter(|path| path.extension().is_some_and(|extension| extension == "rs"))
        .collect::<Vec<_>>();
    files.sort();
    files
}

/// Extracts `#[path = "…"]` targets and normalizes them to repository paths.
///
/// Harnesses may nest inline modules to mirror crate paths, which changes how
/// Rust resolves relative `#[path]` attributes. Every included file must live
/// under `src/`, so normalization keeps the suffix starting at `src/`.
fn included_sources(harness: &Path) -> Vec<String> {
    let text = fs::read_to_string(harness).expect("read harness");
    let mut sources = Vec::new();
    for line in text.lines() {
        let Some(start) = line.find("#[path = \"") else {
            continue;
        };
        let rest = &line[start + "#[path = \"".len()..];
        let target = rest.split('"').next().expect("closing quote");
        let normalized = target
            .find("src/")
            .map(|offset| target[offset..].to_owned())
            .unwrap_or_else(|| {
                panic!(
                    "{}: #[path] must name a file under src/: {target}",
                    harness.display()
                )
            });
        sources.push(normalized);
    }
    sources
}

/// Lists every Rust file under `src/` that carries a Verus specification.
fn annotated_sources(root: &Path) -> BTreeSet<String> {
    /// Recursively visits one source directory.
    fn visit(root: &Path, directory: &Path, found: &mut BTreeSet<String>) {
        for entry in fs::read_dir(directory).expect("read source directory") {
            let path = entry.expect("read source entry").path();
            if path.is_dir() {
                visit(root, &path, found);
            } else if path.extension().is_some_and(|extension| extension == "rs")
                && fs::read_to_string(&path)
                    .expect("read source file")
                    .contains("verus_spec(")
            {
                let relative = path.strip_prefix(root).expect("source below root");
                found.insert(relative.to_string_lossy().replace('\\', "/"));
            }
        }
    }
    let mut found = BTreeSet::new();
    visit(root, &root.join("src"), &mut found);
    found
}

/// Every harness has module documentation and includes at least one real source file.
#[test]
fn harnesses_document_and_include_existing_sources() {
    let root = repository_root();
    let harnesses = harnesses(&root);
    assert!(
        !harnesses.is_empty(),
        "verus/ must contain at least one harness"
    );
    for harness in harnesses {
        let text = fs::read_to_string(&harness).expect("read harness");
        assert!(
            text.trim_start().starts_with("//!"),
            "{} must begin with module documentation",
            harness.display()
        );
        let sources = included_sources(&harness);
        assert!(
            !sources.is_empty(),
            "{} must verify real source files through #[path]",
            harness.display()
        );
        for source in sources {
            assert!(
                root.join(&source).is_file(),
                "{} includes missing source {source}",
                harness.display()
            );
        }
    }
}

/// Harnesses and the sources they verify contain no trusted proof escape hatches.
#[test]
fn proofs_do_not_use_trusted_escape_hatches() {
    let root = repository_root();
    for harness in harnesses(&root) {
        let mut files = vec![harness.clone()];
        files.extend(
            included_sources(&harness)
                .into_iter()
                .map(|source| root.join(source)),
        );
        for file in files {
            let text = fs::read_to_string(&file).expect("read proof input");
            for token in FORBIDDEN_PROOF_TOKENS {
                assert!(
                    !text.contains(token),
                    "{} uses forbidden proof escape hatch `{token}`",
                    file.display()
                );
            }
        }
    }
}

/// A source specification that no harness includes would never be checked.
#[test]
fn every_annotated_source_is_verified_by_a_harness() {
    let root = repository_root();
    let included = harnesses(&root)
        .iter()
        .flat_map(|harness| included_sources(harness))
        .collect::<BTreeSet<_>>();
    let unchecked = annotated_sources(&root)
        .difference(&included)
        .cloned()
        .collect::<Vec<_>>();
    assert!(
        unchecked.is_empty(),
        "source files carry Verus specifications but no verus/*.rs harness includes them: {unchecked:?}"
    );
}
