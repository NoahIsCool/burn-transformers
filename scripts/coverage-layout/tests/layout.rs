//! Adversarial source fixtures for conservative classification, not gate tests.

use std::{
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

use sha2::{Digest, Sha256};
use transformers_coverage_layout::{Classification, FileReport, Report, classify};

/// Monotonic suffix prevents fixture collisions within one test process.
static NEXT: AtomicU64 = AtomicU64::new(0);

/// An isolated, explicitly owned temporary source tree.
struct Fixture(PathBuf);

impl Fixture {
    /// Creates a unique fixture root without reusing another test's directory.
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "tfm-coverage-layout-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }

    /// Writes test-owned source bytes and creates only fixture-local parents.
    fn put(&self, relative: &str, content: impl AsRef<[u8]>) {
        let path = self.0.join(relative);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, content).unwrap();
    }

    /// Classifies this fixture's explicit root module.
    fn report(&self) -> Report {
        classify(&self.0, &[PathBuf::from("lib.rs")]).unwrap()
    }
}

impl Drop for Fixture {
    /// Removes only the unique directory created by this fixture.
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).unwrap();
    }
}

/// Finds one physical file by its root-relative report name.
fn file<'a>(report: &'a Report, name: &str) -> &'a FileReport {
    report.files.iter().find(|file| file.path == name).unwrap()
}

/// Resolves a physical line from the complete compact run representation.
fn line(file: &FileReport, number: usize) -> Classification {
    file.lines
        .iter()
        .find(|run| run.start_line <= number && number <= run.end_line)
        .unwrap()
        .classification
}

/// Collects a file's full physical-line classification for order comparisons.
fn expanded(file: &FileReport) -> Vec<Classification> {
    (1..=file.line_count)
        .map(|number| line(file, number))
        .collect()
}

#[test]
fn cfg_proofs_do_not_guess_other_atoms_or_module_names() {
    let fixture = Fixture::new();
    fixture.put("lib.rs", "#[cfg(all(test, feature = \"x\"))] fn a() {}\n#[cfg(any(test, feature = \"x\"))] fn b() {}\n#[cfg(not(test))] fn c() {}\nmod tests { fn production() {} }\n#[cfg(any())] fn disabled_everywhere() {}\n#[test] fn unproven_builtin_or_macro_attribute() {}\n");
    let report = fixture.report();
    assert_eq!(
        expanded(file(&report, "lib.rs")),
        vec![
            Classification::TestOnly,
            Classification::Retained,
            Classification::Retained,
            Classification::Retained,
            Classification::Retained,
            Classification::Retained
        ]
    );
}

#[test]
fn multiline_cfg_nested_inheritance_and_multiple_attributes() {
    let fixture = Fixture::new();
    fixture.put("lib.rs", "#[cfg(\n all(\n  test,\n  any(feature = \"a\", feature = \"b\"),\n ),\n)]\nmod outer {\n #[cfg(not(feature = \"c\"))]\n fn inherited() {}\n}\n#[cfg(feature = \"a\")]\n#[cfg(test)]\nfn double() {}\n");
    let report = fixture.report();
    assert!(
        expanded(file(&report, "lib.rs"))
            .iter()
            .all(|value| *value == Classification::TestOnly)
    );
}

#[test]
fn cfg_attr_implication_is_conservative_and_recursive() {
    let fixture = Fixture::new();
    fixture.put("lib.rs", "#[cfg_attr(feature = \"a\", cfg(test))] fn unknown() {}\n#[cfg_attr(not(test), cfg(test))] fn conditional() {}\n#[cfg_attr(all(), cfg_attr(all(), cfg(test)))] fn nested() {}\n#[cfg_attr(test, cfg(not(test)))] fn live() {}\n");
    let report = fixture.report();
    assert_eq!(
        expanded(file(&report, "lib.rs")),
        vec![
            Classification::Retained,
            Classification::TestOnly,
            Classification::TestOnly,
            Classification::Retained
        ]
    );
}

#[test]
fn malformed_and_unsupported_cfg_remain_visible_and_retained() {
    let fixture = Fixture::new();
    fixture.put("lib.rs", "#[cfg(test)] #[cfg(not(test, unix))] fn malformed() {}\n#[cfg(test)] #[cfg(mystery(test))] fn unsupported() {}\n#[cfg_attr(test)] fn missing_attribute() {}\n");
    let report = fixture.report();
    assert_eq!(
        report
            .diagnostics
            .iter()
            .filter(|issue| issue.code == "unresolved_cfg")
            .count(),
        3
    );
    assert!(
        expanded(file(&report, "lib.rs"))
            .iter()
            .all(|value| *value == Classification::Retained)
    );
}

#[test]
fn raw_strings_comments_and_character_literals_cannot_forge_syntax() {
    let fixture = Fixture::new();
    fixture.put(
        "lib.rs",
        r####"const TEXT: &str = r###"#[cfg(test)] mod fake { " } //"###;
// #[cfg(test)] mod fake;
/* outer /* #[cfg(test)] */ mod fake; */
fn live() { let _brace = '}'; }
#[cfg(test)] fn real_test_helper() {}
"####,
    );
    let report = fixture.report();
    assert_eq!(report.files.len(), 1);
    assert_eq!(
        expanded(file(&report, "lib.rs")),
        vec![
            Classification::Retained,
            Classification::Retained,
            Classification::Retained,
            Classification::Retained,
            Classification::TestOnly
        ]
    );
}

#[test]
fn same_line_items_are_mixed_and_never_exclusively_test_only() {
    let fixture = Fixture::new();
    fixture.put(
        "lib.rs",
        "fn live() {} #[cfg(test)] fn helper() {}\nmod inner { #[cfg(test)] fn helper() {} }\n",
    );
    let report = fixture.report();
    assert_eq!(
        expanded(file(&report, "lib.rs")),
        vec![Classification::Mixed, Classification::Mixed]
    );
    assert!(
        report
            .diagnostics
            .iter()
            .any(|issue| issue.code == "mixed_production_test_lines")
    );
}

#[test]
fn external_test_modules_inherit_test_only_and_inner_file_cfg() {
    let fixture = Fixture::new();
    fixture.put("lib.rs", "#[cfg(test)] mod test_helpers;\nmod inner_cfg;\n");
    fixture.put("test_helpers.rs", "fn helper() {}\nmod nested;\n");
    fixture.put("test_helpers/nested.rs", "fn nested_helper() {}\n");
    fixture.put("inner_cfg.rs", "#![cfg(test)]\nfn helper() {}\n");
    let report = fixture.report();
    for path in ["test_helpers.rs", "test_helpers/nested.rs", "inner_cfg.rs"] {
        assert!(
            expanded(file(&report, path))
                .iter()
                .all(|value| *value == Classification::TestOnly)
        );
    }
}

#[test]
fn shared_files_merge_production_and_test_contexts_in_both_orders() {
    let fixture = Fixture::new();
    fixture.put(
        "shared.rs",
        "fn shared() {}\n#[cfg(test)] fn exclusively_test() {}\n",
    );
    let declarations = [
        "#[cfg(test)] #[path = \"shared.rs\"] mod tests;",
        "#[path = \"shared.rs\"] mod live;",
    ];
    fixture.put("lib.rs", declarations.join("\n"));
    let first = fixture.report();
    fixture.put(
        "lib.rs",
        declarations
            .into_iter()
            .rev()
            .collect::<Vec<_>>()
            .join("\n"),
    );
    let second = fixture.report();
    assert_eq!(
        expanded(file(&first, "shared.rs")),
        expanded(file(&second, "shared.rs"))
    );
    assert_eq!(
        expanded(file(&first, "shared.rs")),
        vec![Classification::Mixed, Classification::TestOnly]
    );
    assert_eq!(file(&first, "shared.rs").contexts.len(), 2);
}

#[test]
fn equivalent_aliases_are_reported_but_not_mistaken_for_cycles() {
    let fixture = Fixture::new();
    fixture.put(
        "lib.rs",
        "#[path = \"shared.rs\"] mod a;\n#[path = \"shared.rs\"] mod b;\n",
    );
    fixture.put("shared.rs", "fn shared() {}\n");
    let report = fixture.report();
    assert!(
        report
            .diagnostics
            .iter()
            .any(|issue| issue.code == "duplicate_context")
    );
    assert!(
        !report
            .diagnostics
            .iter()
            .any(|issue| issue.code == "module_cycle")
    );
    assert_eq!(file(&report, "shared.rs").contexts.len(), 2);
    assert_eq!(
        line(file(&report, "shared.rs"), 1),
        Classification::Retained
    );
}

#[test]
fn active_path_cycle_is_explicit_and_bounded() {
    let fixture = Fixture::new();
    fixture.put("lib.rs", "mod first;\n");
    fixture.put("first.rs", "#[path = \"second.rs\"] mod second;\n");
    fixture.put("second.rs", "#[path = \"first.rs\"] mod first;\n");
    let report = fixture.report();
    assert_eq!(report.files.len(), 3);
    assert_eq!(
        report
            .diagnostics
            .iter()
            .filter(|issue| issue.code == "module_cycle")
            .count(),
        1
    );
}

#[test]
fn default_mod_rs_and_non_mod_rs_nested_paths_follow_rust_layout() {
    let fixture = Fixture::new();
    fixture.put("lib.rs", "mod flat;\nmod directory;\n");
    fixture.put("flat.rs", "mod child;\n");
    fixture.put("flat/child.rs", "fn flat_child() {}\n");
    fixture.put("directory/mod.rs", "mod child;\n");
    fixture.put("directory/child.rs", "fn directory_child() {}\n");
    let report = fixture.report();
    assert_eq!(report.files.len(), 5);
    assert!(report.diagnostics.is_empty());
}

#[test]
fn explicit_paths_respect_inline_and_noninline_base_directories() {
    let fixture = Fixture::new();
    fixture.put(
        "lib.rs",
        "mod flat;\n#[path = \"custom\"] mod inline { #[path = \"leaf.rs\"] mod leaf; }\n",
    );
    fixture.put(
        "flat.rs",
        "#[path = \"sibling.rs\"] mod sibling;\nmod inline { #[path = \"deep.rs\"] mod deep; }\n",
    );
    fixture.put("sibling.rs", "fn sibling() {}\n");
    fixture.put("flat/inline/deep.rs", "fn deep() {}\n");
    fixture.put("custom/leaf.rs", "fn custom() {}\n");
    let report = fixture.report();
    assert_eq!(report.files.len(), 5);
    assert!(report.diagnostics.is_empty());
}

#[test]
fn conditional_paths_never_choose_one_feature_assignment() {
    let fixture = Fixture::new();
    fixture.put(
        "lib.rs",
        "#[cfg(test)] #[cfg_attr(feature = \"x\", path = \"alternate.rs\")] mod module;\n",
    );
    fixture.put("alternate.rs", "fn alternate() {}\n");
    fixture.put("module.rs", "fn default() {}\n");
    let report = fixture.report();
    assert!(
        report
            .diagnostics
            .iter()
            .any(|issue| issue.code == "ambiguous_module_path")
    );
    for path in ["alternate.rs", "module.rs"] {
        assert_eq!(line(file(&report, path), 1), Classification::Retained);
    }
}

#[test]
fn conflicting_default_module_files_are_both_retained() {
    let fixture = Fixture::new();
    fixture.put("lib.rs", "#[cfg(test)] mod conflict;\n");
    fixture.put("conflict.rs", "fn flat() {}\n");
    fixture.put("conflict/mod.rs", "fn nested() {}\n");
    let report = fixture.report();
    assert!(
        report
            .diagnostics
            .iter()
            .any(|issue| issue.code == "ambiguous_module_files")
    );
    for path in ["conflict.rs", "conflict/mod.rs"] {
        assert_eq!(line(file(&report, path), 1), Classification::Retained);
    }
}

#[test]
fn missing_and_traversing_paths_are_unresolved_without_outside_reads() {
    let outer = Fixture::new();
    outer.put("outside.rs", "fn outside() {}\n");
    outer.put(
        "root/lib.rs",
        "mod missing;\n#[path = \"../outside.rs\"] mod escape;\n",
    );
    let report = classify(&outer.0.join("root"), &[PathBuf::from("lib.rs")]).unwrap();
    assert_eq!(report.files.len(), 1);
    assert_eq!(
        report
            .diagnostics
            .iter()
            .filter(|issue| issue.code == "unresolved_module")
            .count(),
        2
    );
}

#[cfg(unix)]
#[test]
fn canonical_symlinks_cannot_escape_the_source_root() {
    let outer = Fixture::new();
    outer.put("outside.rs", "fn outside() {}\n");
    outer.put("root/lib.rs", "mod linked;\n");
    std::os::unix::fs::symlink(outer.0.join("outside.rs"), outer.0.join("root/linked.rs")).unwrap();
    let report = classify(&outer.0.join("root"), &[PathBuf::from("lib.rs")]).unwrap();
    assert_eq!(report.files.len(), 1);
    assert!(
        report
            .diagnostics
            .iter()
            .any(|issue| issue.code == "unresolved_module")
    );
}

#[test]
fn parse_errors_and_macro_expansion_remain_visible_without_generated_exclusions() {
    let fixture = Fixture::new();
    fixture.put("lib.rs", "mod broken;\ninclude!(\"generated.rs\");\nmacro_rules! make { () => { mod generated; } }\nmake!();\n");
    fixture.put("broken.rs", "#[cfg(test)] fn broken(\n");
    fixture.put(
        "generated.rs",
        "fn should_not_be_read_without_expansion() {}\n",
    );
    let report = fixture.report();
    assert_eq!(report.files.len(), 2);
    assert!(
        report
            .diagnostics
            .iter()
            .any(|issue| issue.code == "parse_error")
    );
    assert_eq!(
        report
            .diagnostics
            .iter()
            .filter(|issue| issue.code == "opaque_macro_or_verbatim")
            .count(),
        3
    );
    assert_eq!(
        line(file(&report, "broken.rs"), 1),
        Classification::Retained
    );
}

#[test]
fn unicode_crlf_bom_shebang_coordinates_hash_exact_original_bytes() {
    let fixture = Fixture::new();
    let source = "\u{feff}#!/usr/bin/env rustx\r\nconst TEXT: &str = \"é🙂\"; #[cfg(test)] fn helper() {}\r\n";
    fixture.put("lib.rs", source);
    let report = fixture.report();
    let output = file(&report, "lib.rs");
    assert_eq!(
        output.sha256,
        format!("{:x}", Sha256::digest(source.as_bytes()))
    );
    assert_eq!(output.byte_length, source.len());
    assert_eq!(output.line_count, 2);
    let region = output.contexts[0]
        .regions
        .iter()
        .find(|region| region.kind == "fn")
        .unwrap();
    assert_eq!(
        &source[region.span.start.byte..region.span.end.byte],
        "#[cfg(test)] fn helper() {}"
    );
    assert_eq!(region.span.start.line, 2);
    // The prefix before `#` is 29 UTF-8 bytes but 25 Unicode scalars;
    // both public column conventions are one-based.
    assert_eq!(region.span.start.column, 30);
    assert_eq!(region.span.start.unicode_column, 26);
    assert_eq!(line(output, 2), Classification::Mixed);
}

#[test]
fn empty_blank_and_unterminated_files_have_complete_physical_line_runs() {
    let fixture = Fixture::new();
    for (source, expected) in [
        ("", vec![]),
        (
            "\n \t\r\n",
            vec![Classification::Blank, Classification::Blank],
        ),
        ("fn last() {}", vec![Classification::Retained]),
    ] {
        fixture.put("lib.rs", source);
        let report = fixture.report();
        assert_eq!(expanded(file(&report, "lib.rs")), expected);
    }
}

#[test]
fn associated_items_and_nested_local_items_have_independent_ownership() {
    let fixture = Fixture::new();
    fixture.put("lib.rs", "struct S;\nimpl S {\n #[cfg(test)] fn test_helper() {}\n fn live() {}\n}\ntrait T {\n #[cfg(test)] fn test_helper() {}\n fn live();\n}\nfn outer() {\n #[cfg(test)] fn local() {}\n}\n");
    let report = fixture.report();
    let output = file(&report, "lib.rs");
    for number in [3, 7, 11] {
        assert_eq!(line(output, number), Classification::TestOnly);
    }
    for number in [1, 2, 4, 5, 6, 8, 9, 10, 12] {
        assert_eq!(line(output, number), Classification::Retained);
    }
}

#[test]
fn multiple_explicit_entries_merge_shared_physical_sources() {
    let fixture = Fixture::new();
    fixture.put(
        "lib.rs",
        "#[cfg(test)] #[path = \"shared.rs\"] mod shared;\n",
    );
    fixture.put("main.rs", "#[path = \"shared.rs\"] mod shared;\n");
    fixture.put("shared.rs", "fn shared() {}\n");
    let report = classify(
        &fixture.0,
        &[PathBuf::from("lib.rs"), PathBuf::from("main.rs")],
    )
    .unwrap();
    assert_eq!(line(file(&report, "shared.rs"), 1), Classification::Mixed);
}

#[test]
fn invalid_inputs_fail_without_synthesizing_a_successful_layout() {
    let fixture = Fixture::new();
    fixture.put("lib.rs", [0xff, 0xfe]);
    assert!(
        classify(&fixture.0, &[PathBuf::from("lib.rs")])
            .unwrap_err()
            .contains("UTF-8")
    );
    assert!(classify(&fixture.0, &[]).is_err());
    assert!(
        classify(
            Path::new("/this/path/does/not/exist"),
            &[PathBuf::from("lib.rs")]
        )
        .is_err()
    );
    assert!(classify(&fixture.0, &[PathBuf::from("missing.rs")]).is_err());
}

#[test]
fn report_json_is_versioned_and_never_encodes_a_coverage_gate_result() {
    let fixture = Fixture::new();
    fixture.put("lib.rs", "#[cfg(test)] fn helper() {}\n");
    let json = serde_json::to_value(fixture.report()).unwrap();
    assert_eq!(json["schema_version"], 1);
    assert_eq!(
        json["purpose"],
        "supplementary_source_classification_not_a_coverage_gate"
    );
    assert!(json.get("coverage_percent").is_none());
    assert!(json.get("gate_passed").is_none());
    assert_eq!(json["qualified_for_coverage_exclusions"], false);
    assert_eq!(json["macro_expansion_resolved"], false);
    assert_eq!(json["files"][0]["lines"][0]["classification"], "test_only");
}

/// A parsed test-only reference cannot certify files also reachable via include!.
#[test]
fn opaque_production_edges_keep_candidate_reports_explicitly_unqualified() {
    let fixture = Fixture::new();
    fixture.put(
        "lib.rs",
        "#[cfg(test)] #[path = \"shared.rs\"] mod tests;\ninclude!(\"shared.rs\");\n",
    );
    fixture.put("shared.rs", "fn shared() {}\n");
    let report = fixture.report();
    // This is an observed-graph candidate, not permission to exclude the line.
    assert_eq!(
        line(file(&report, "shared.rs"), 1),
        Classification::TestOnly
    );
    assert!(!report.qualified_for_coverage_exclusions);
    assert!(!report.macro_expansion_resolved);
    assert!(
        report
            .diagnostics
            .iter()
            .any(|issue| issue.code == "opaque_macro_or_verbatim")
    );
}
