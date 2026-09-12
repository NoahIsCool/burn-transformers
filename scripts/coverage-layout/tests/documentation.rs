//! Adversarial fixtures for authored documentation, test scope, and macro boundaries.

use std::{
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};
use transformers_coverage_layout::{
    DocumentationReport, DocumentationSite, audit_documentation, audit_standards,
};

/// Unique fixture suffix; no test reuses or removes another test's source tree.
static NEXT: AtomicU64 = AtomicU64::new(0);

/// A temporary source tree whose exact bytes are owned by one test.
struct Fixture(
    /// Unique directory owned by this fixture.
    PathBuf,
);

impl Fixture {
    /// Creates a bounded test-owned directory.
    fn new(source: &str) -> Self {
        let root = std::env::temp_dir().join(format!(
            "tfm-documentation-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&root).unwrap();
        let fixture = Self(root);
        fixture.put("lib.rs", source);
        fixture
    }

    /// Writes a source buffer without modifying repository fixtures.
    fn put(&self, name: &str, source: &str) {
        let path = self.0.join(name);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, source).unwrap();
    }

    /// Runs the same source-only documentation API used by the CLI.
    fn audit(&self) -> DocumentationReport {
        audit_documentation(&self.0, &[PathBuf::from("lib.rs")]).unwrap()
    }

    /// Runs the comprehensive authored-item policy over the same source graph.
    fn standards(&self) -> DocumentationReport {
        audit_standards(&self.0, &[PathBuf::from("lib.rs")]).unwrap()
    }
}

impl Drop for Fixture {
    /// Removes only the uniquely named test directory allocated above.
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).unwrap();
    }
}

/// Finds one named parsed site without relying on line-number heuristics.
fn site<'a>(report: &'a DocumentationReport, name: &str) -> &'a DocumentationSite {
    report
        .files
        .iter()
        .flat_map(|file| &file.sites)
        .find(|site| site.name == name)
        .unwrap()
}

/// Public, private, trait, and foreign functions all require their own production docs.
#[test]
fn private_functions_trait_implementations_and_default_methods_are_required() {
    let fixture = Fixture::new(
        r#"//! Module theory.
/// Public function.
pub fn public() {}
fn private() {}
struct Model;
trait Contract {
    /// Required signature.
    fn required(&self);
    fn default_method(&self) {}
}
impl Contract for Model {
    fn required(&self) {}
}
impl Model {
    /// Private implementation helper.
    fn helper(&self) {}
}
unsafe extern "C" { fn foreign(); }
"#,
    );
    let report = fixture.audit();
    assert!(report.complete);
    assert!(!report.passed);
    assert!(site(&report, "public").style_present);
    assert!(!site(&report, "private").style_present);
    assert!(!site(&report, "default_method").style_present);
    assert!(site(&report, "helper").style_present);
    assert!(!site(&report, "foreign").style_present);
    let required: Vec<_> = report.files[0]
        .sites
        .iter()
        .filter(|site| site.name == "required")
        .collect();
    assert_eq!(required.len(), 2);
    assert!(
        required
            .iter()
            .any(|site| site.kind == "trait_method" && site.style_present)
    );
    assert!(
        required
            .iter()
            .any(|site| site.kind == "impl_method" && !site.style_present)
    );
}

/// Equivalent and computed rustdoc are reported without satisfying line-doc style.
#[test]
fn literal_doc_style_is_distinct_from_equivalent_block_and_attribute_docs() {
    let fixture = Fixture::new(
        r##"#![doc = "Equivalent module docs."]
#[doc = "Equivalent function docs."]
fn attribute() {}
/** Equivalent block docs. */
fn block() {}
///
fn empty() {}
#[doc(hidden)]
fn hidden() {}
#[doc = include_str!("explanation.md")]
fn computed() {}
#[cfg_attr(feature = "docs", doc = "Conditional docs.")]
fn conditional() {}
/// Ordinary required spelling.
fn literal() {}
"##,
    );
    let report = fixture.audit();
    assert!(!report.passed);
    for name in ["lib.rs", "attribute", "block"] {
        assert!(site(&report, name).rustdoc_present, "{name}");
        assert!(!site(&report, name).style_present, "{name}");
    }
    for name in ["empty", "hidden"] {
        assert!(!site(&report, name).rustdoc_present);
        assert!(!site(&report, name).style_present);
    }
    assert!(site(&report, "computed").computed_documentation);
    assert!(site(&report, "conditional").computed_documentation);
    assert!(site(&report, "literal").style_present);
}

/// Outer module docs cannot substitute for the required inner module description.
#[test]
fn inline_production_modules_need_inner_docs_even_when_outer_docs_exist() {
    let fixture = Fixture::new(
        "//! Root module.\n/// Outer public docs.\nmod outer_only { /// Helper.\nfn helper() {} }\nmod documented { //! Inner module theory.\n/// Child.\nfn child() {} }\n",
    );
    let report = fixture.audit();
    assert!(site(&report, "outer_only").rustdoc_present);
    assert!(!site(&report, "outer_only").style_present);
    assert!(site(&report, "documented").style_present);
    assert!(report.files[0].inline_test_bodies.is_empty());
}

/// Structural test ownership exempts legacy functions; names and uncertain cfg do not.
#[test]
fn test_cfg_and_harness_bodies_are_exempt_but_test_names_are_not() {
    let fixture = Fixture::new(
        r#"//! Root.
#[cfg(test)] #[path = "checks.rs"] mod checks;
#[test] fn harness() { fn test_helper() {} }
#[cfg(all(test, feature = "x"))] fn test_cfg() {}
#[cfg(any(test, feature = "x"))] fn possibly_production() {}
mod tests { fn production_despite_name() {} }
/// Production outer function.
fn outer() { #[cfg(test)] { fn nested_test_helper() {} } }
"#,
    );
    fixture.put("checks.rs", "fn helper_in_test_file() {}\n");
    let report = fixture.audit();
    for name in [
        "harness",
        "test_helper",
        "test_cfg",
        "nested_test_helper",
        "helper_in_test_file",
    ] {
        assert!(site(&report, name).test_only, "{name}");
    }
    assert!(!site(&report, "possibly_production").test_only);
    assert!(!site(&report, "production_despite_name").test_only);
    assert!(site(&report, "checks.rs").test_only);
}

/// A shared physical file retains production requirements despite a test alias.
#[test]
fn a_production_alias_prevents_test_filename_exemption() {
    let fixture = Fixture::new(
        "//! Root.\n#[cfg(test)] #[path=\"shared_tests.rs\"] mod checks;\n#[path=\"shared_tests.rs\"] mod production;\n",
    );
    fixture.put("shared_tests.rs", "fn reused() {}\n");
    let report = fixture.audit();
    assert!(!site(&report, "reused").test_only);
    assert!(!site(&report, "shared_tests.rs").test_only);
}

/// Legacy tests remain doc-exempt while inline bodies still violate sibling placement.
#[test]
fn inline_test_bodies_require_sibling_placement_without_requiring_test_docs() {
    let fixture =
        Fixture::new("//! Root.\n#[cfg(test)] mod checks { #[test] fn test_case() {} }\n");
    let report = fixture.audit();
    assert!(report.complete);
    assert!(site(&report, "test_case").test_only);
    assert_eq!(report.files[0].inline_test_bodies.len(), 1);
    assert!(!report.passed);
    let external = Fixture::new("//! Root.\n#[cfg(test)] #[path=\"checks.rs\"] mod checks;\n");
    external.put("checks.rs", "#[test] fn test_case() {}\n");
    assert!(external.audit().passed);
}

/// Fixed and metavariable function templates retain literal source documentation spans.
#[test]
fn macro_authored_item_fragments_and_metavariable_prefixes_use_real_doc_spans() {
    let fixture = Fixture::new(
        r#"//! Macro behavior.
macro_rules! fixed { () => {
    /// Authored fixed function.
    fn generated_name() {}
}; }
macro_rules! named { ($name:ident) => {
    /// Authored template documentation.
    #[doc = $description]
    pub fn $name() {}
}; }
macro_rules! missing { ($name:ident) => { fn $name() {} }; }
macro_rules! pointer { () => { type Callback = fn(i32) -> i32; }; }
"#,
    );
    let report = fixture.audit();
    assert!(report.complete);
    assert!(!report.macro_expansion_resolved);
    assert!(site(&report, "generated_name").style_present);
    let templates: Vec<_> = report.files[0]
        .sites
        .iter()
        .filter(|site| site.kind == "macro_function_template")
        .collect();
    assert_eq!(templates.len(), 2);
    assert!(
        templates
            .iter()
            .any(|site| site.style_present && site.computed_documentation)
    );
    assert_eq!(
        templates.iter().filter(|site| site.style_present).count(),
        1
    );
    assert!(
        report.files[0]
            .macros
            .iter()
            .all(|boundary| boundary.unresolved_functions.is_empty())
    );
}

/// Opaque function prefixes block completeness while comments and strings remain inert.
#[test]
fn truly_opaque_prefixes_are_review_blockers_and_strings_do_not_invent_functions() {
    let fixture = Fixture::new(
        r#"//! Opaque source.
macro_rules! opaque { ($visibility:tt, $name:ident) => { $visibility fn $name() {} }; }
const TEXT: &str = "fn fake() {} #[test] /// fake docs";
// fn another_fake() {}
"#,
    );
    let report = fixture.audit();
    assert!(!report.complete);
    assert!(!report.passed);
    assert_eq!(report.files[0].sites.len(), 1);
    assert_eq!(
        report.files[0]
            .macros
            .iter()
            .map(|boundary| boundary.unresolved_functions.len())
            .sum::<usize>(),
        1
    );
}

/// Parse failures block completeness and CRLF/BOM bytes preserve original coordinates.
#[test]
fn malformed_sources_fail_closed_and_crlf_spelling_retains_exact_lines() {
    let invalid = Fixture::new("//! Module.\nfn broken(\n");
    assert!(!invalid.audit().complete);
    let valid =
        Fixture::new("\u{feff}//! Unicode module.\r\n/// Unicode λ helper.\r\nfn correct() {}\r\n");
    let report = valid.audit();
    assert!(report.passed);
    assert_eq!(site(&report, "correct").span.start.line, 2);
}

/// Standards mode adds every named declaration and member without changing legacy sites.
#[test]
fn standards_require_private_declarations_and_members_independently() {
    let fixture = Fixture::new(
        r#"//! Root.
struct Record { member: u8 }
struct Tuple(u8, u16);
enum Choice { Unit, Tuple(u8), Record { value: u8 } }
union Bits { number: u32 }
trait Contract { const LIMIT: usize; type Output; }
trait Alias = Contract;
type Count = usize;
const LIMIT: usize = 2;
static FLAG: bool = false;
impl Contract for Record { const LIMIT: usize = 2; type Output = u8; }
unsafe extern "C" { static EXTERNAL: u8; type Opaque; }
"#,
    );
    let legacy = fixture.audit();
    assert!(legacy.passed);
    assert_eq!(legacy.schema_version, 1);
    assert_eq!(legacy.files[0].sites.len(), 1);
    let report = fixture.standards();
    assert!(report.complete);
    assert!(!report.passed);
    assert_eq!(report.schema_version, 2);
    for kind in [
        "struct",
        "enum",
        "union",
        "trait",
        "trait_alias",
        "type_alias",
        "constant",
        "static",
        "impl_constant",
        "impl_type",
        "trait_constant",
        "trait_type",
        "foreign_static",
        "foreign_type",
        "variant",
        "field",
        "tuple_field",
    ] {
        assert!(
            report.files[0]
                .sites
                .iter()
                .any(|site| site.kind == kind && !site.style_present),
            "{kind}"
        );
    }
    let tuple_fields: Vec<_> = report.files[0]
        .sites
        .iter()
        .filter(|site| site.kind == "tuple_field")
        .collect();
    assert_eq!(tuple_fields.len(), 3);
    assert_eq!(
        tuple_fields
            .iter()
            .map(|site| site.name.as_str())
            .collect::<Vec<_>>(),
        ["0", "1", "0"]
    );
}

/// Parent prose never documents children, and equivalent/computed docs remain visible.
#[test]
fn standards_preserve_doc_placement_and_alternative_form_evidence() {
    let fixture = Fixture::new(
        r#"//! Root.
/// Container.
struct Record { absent: u8, /** Equivalent. */ block: u8,
    #[doc = include_str!("field.md")] computed: u8,
    /// Exact member documentation.
    documented: u8 }
/// Choices.
enum Choice { Undocumented, /// Documented variant.
    Present(/// Documented tuple member.
        u8) }
"#,
    );
    let report = fixture.standards();
    assert!(report.complete);
    assert!(site(&report, "Record").style_present);
    assert!(!site(&report, "absent").style_present);
    assert!(site(&report, "block").rustdoc_present);
    assert!(!site(&report, "block").style_present);
    assert!(site(&report, "computed").computed_documentation);
    assert!(site(&report, "documented").style_present);
    assert!(!site(&report, "Undocumented").style_present);
    assert!(site(&report, "Present").style_present);
    assert!(site(&report, "0").style_present);
}

/// Test functions, helpers, and physical modules need docs only in the opt-in policy.
#[test]
fn standards_include_test_owned_sources_without_erasing_test_evidence() {
    let fixture = Fixture::new("//! Root.\n#[cfg(test)] #[path = \"checks.rs\"] mod checks;\n");
    fixture.put("checks.rs", "#[test] fn scenario() {}\nfn helper() {}\n");
    assert!(fixture.audit().passed);
    let report = fixture.standards();
    assert!(report.complete);
    assert!(!report.passed);
    for name in ["checks.rs", "scenario", "helper"] {
        assert!(site(&report, name).test_only, "{name}");
        assert!(!site(&report, name).style_present, "{name}");
    }
    fixture.put("checks.rs", "//! Test scenarios.\n/// Checks the successful scenario.\n#[test] fn scenario() {}\n/// Builds test input.\nfn helper() {}\n");
    assert!(fixture.standards().passed);
}

/// Authored macro items are inspected, while opaque declaration templates block completeness.
#[test]
fn standards_macro_boundaries_cover_types_and_tests_without_expansion_claims() {
    let fixture = Fixture::new(
        r#"//! Root.
/// Defines a fixed record.
macro_rules! fixed { () => { /// Fixed record.
    struct Record { member: u8 } }; }
/// Defines a named record.
macro_rules! named { ($name:ident) => { struct $name { member: u8 } }; }
#[cfg(test)]
/// Defines a test-owned template.
macro_rules! test_named { ($name:ident) => { type $name = u8; }; }
"#,
    );
    assert!(fixture.audit().complete);
    let report = fixture.standards();
    assert!(!report.complete);
    assert!(!report.macro_expansion_resolved);
    assert!(site(&report, "fixed").style_present);
    assert!(site(&report, "Record").style_present);
    assert!(!site(&report, "member").style_present);
    assert_eq!(
        report.files[0]
            .macros
            .iter()
            .flat_map(|boundary| boundary.unresolved_items.as_ref().unwrap())
            .count(),
        2
    );
    let legacy_json = serde_json::to_value(fixture.audit()).unwrap();
    assert!(
        legacy_json["files"][0]["macros"]
            .as_array()
            .unwrap()
            .iter()
            .all(|boundary| boundary.get("unresolved_items").is_none())
    );
}

/// Generic const parameters inside parseable templates do not become named-item findings.
#[test]
fn standards_parseable_generic_templates_do_not_invent_const_item_requirements() {
    let fixture = Fixture::new(
        r#"//! Root.
/// Produces a generic record.
macro_rules! fixed { () => { /// Generic record.
    struct Record<const N: usize> { /// Fixed-size payload.
        bytes: [u8; N] } }; }
"#,
    );
    let report = fixture.standards();
    assert!(report.complete);
    assert!(report.passed);
    assert!(
        !report.files[0]
            .sites
            .iter()
            .any(|site| site.kind == "constant")
    );
}

/// Local aliases within macro test bodies keep real spans despite non-Rust strategies.
#[test]
fn standards_visit_local_items_in_parseable_macro_statement_bodies() {
    let fixture = Fixture::new(
        r#"//! Root.
proptest! {
    #![proptest_config(Config::with_cases(24))]
    /// Checks a generated value.
    #[test]
    fn scenario(value in 0..4) {
        type Count = usize;
        let _: Count = value;
    }
}

"#,
    );
    let report = fixture.standards();
    assert!(report.complete);
    assert!(!report.passed);
    let alias = site(&report, "Count");
    assert_eq!(alias.kind, "type_alias");
    assert_eq!(alias.span.start.line, 7);
    assert!(!alias.style_present);
    assert!(site(&report, "scenario").style_present);
}

/// Integration-only nested tests stay in their harness; production aliases retain relocation checks.
#[test]
fn standards_harness_placement_exemption_requires_every_explicit_entry_context() {
    let fixture = Fixture::new("//! Production.\n#[path = \"tests/support.rs\"] mod reused;\n");
    fixture.put(
        "tests/harness.rs",
        "//! Integration harness.\n#[path = \"support.rs\"] mod support;\n",
    );
    fixture.put("tests/support.rs", "//! Harness helpers.\n#[cfg(test)] mod checks { //! Scenarios.\n/// Checks a boundary.\n#[test] fn scenario() {} }\n");
    let harness = audit_standards(&fixture.0, &[PathBuf::from("tests/harness.rs")]).unwrap();
    assert!(harness.passed);
    let support = harness
        .files
        .iter()
        .find(|file| file.path == "tests/support.rs")
        .unwrap();
    assert_eq!(support.test_harness_only, Some(true));
    assert!(support.inline_test_bodies.is_empty());
    let mixed = audit_standards(
        &fixture.0,
        &[PathBuf::from("lib.rs"), PathBuf::from("tests/harness.rs")],
    )
    .unwrap();
    assert!(mixed.complete);
    assert!(!mixed.passed);
    let support = mixed
        .files
        .iter()
        .find(|file| file.path == "tests/support.rs")
        .unwrap();
    assert_eq!(support.test_harness_only, Some(false));
    assert_eq!(support.inline_test_bodies.len(), 1);
}

/// The real CLI distinguishes report generation from invocation failures in every mode.
#[test]
fn raw_cli_preserves_versioned_json_and_report_only_exit_codes() {
    let fixture = Fixture::new("fn undocumented() {}\n");
    let help = std::process::Command::new(env!("CARGO_BIN_EXE_transformers-coverage-layout"))
        .arg("--help")
        .output()
        .expect("run CLI help");
    assert!(help.status.success());
    assert!(String::from_utf8_lossy(&help.stdout).starts_with("usage:"));
    assert!(help.stderr.is_empty());
    for (mode, schema) in [
        (None, 1),
        (Some("--documentation"), 1),
        (Some("--standards"), 2),
    ] {
        let mut command =
            std::process::Command::new(env!("CARGO_BIN_EXE_transformers-coverage-layout"));
        if let Some(mode) = mode {
            command.arg(mode);
        }
        let output = command
            .arg(&fixture.0)
            .arg("lib.rs")
            .output()
            .expect("run real CLI");
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(output.stdout.last(), Some(&b'\n'));
        let report: serde_json::Value =
            serde_json::from_slice(&output.stdout).expect("one JSON report");
        assert_eq!(report["schema_version"], schema);
        if mode.is_some() {
            assert_eq!(report["passed"], false);
        }
    }
    for args in [
        &[][..],
        &["--standards"],
        &["--unknown"],
        &["--standards", "missing-root", "lib.rs"],
    ] {
        let output = std::process::Command::new(env!("CARGO_BIN_EXE_transformers-coverage-layout"))
            .current_dir(&fixture.0)
            .args(args)
            .output()
            .expect("run invalid CLI");
        assert_eq!(output.status.code(), Some(2));
        assert!(output.stdout.is_empty());
        assert!(String::from_utf8_lossy(&output.stderr).starts_with("coverage-layout:"));
    }
}
