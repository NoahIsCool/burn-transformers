//! Adversarial fixtures for authored documentation, test scope, and macro boundaries.

use std::{
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};
use transformers_coverage_layout::{DocumentationReport, DocumentationSite, audit_documentation};

/// Unique fixture suffix; no test reuses or removes another test's source tree.
static NEXT: AtomicU64 = AtomicU64::new(0);

/// A temporary source tree whose exact bytes are owned by one test.
struct Fixture(PathBuf);

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
