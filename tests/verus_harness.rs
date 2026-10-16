//! Structural checks for the in-place Verus proofs described in `verus/README.md`.
//!
//! Verus itself is an external tool with a pinned toolchain, so ordinary
//! `cargo test` cannot run the proofs. These tests keep the proof surface
//! honest between Verus runs:
//!
//! * every harness includes real source files;
//! * proof code uses no trusted escape hatch, including the ones that
//!   `--no-cheating` still accepts (`exec_allows_no_decreases_clause`,
//!   `verifier::external`, and the crate's own `verus_verify(external)`);
//! * every source file compiled under `verus_keep_ghost` is checked by some
//!   harness;
//! * the proof-surface figures the documentation quotes (harness count,
//!   per-harness sources and verified counts, the verified total, and the
//!   number of distinct source files) agree with the tree and with each
//!   other, so they cannot silently drift.
//!
//! The verified counts themselves come from Verus: `scripts/check_release.py
//! --gate verus` runs every harness and compares each count with the table in
//! `verus/README.md`. These tests check that the documents agree with that
//! table and with the harness files.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

/// Tokens that would weaken a proof into an assumption or remove an item from verification.
///
/// `--no-cheating` rejects `admit`, `assume`, `external_body`,
/// `assume_specification`, and `assume_termination`, but it still accepts
/// `exec_allows_no_decreases_clause`, under which a body that never returns
/// satisfies any postcondition, and `verifier::external`, which drops an item
/// (contract included) from verification without a diagnostic. The external
/// specification forms trust a hand-written contract for code Verus never
/// checks, and `inline_air_stmt` injects unchecked solver input. All of them
/// are rejected here, in harnesses and in every included source file. The
/// attribute spelling the crate itself uses, `verus_verify(external)`, is not
/// a fixed substring, so [`escape_hatches`] checks it separately (see
/// [`FORBIDDEN_VERUS_VERIFY_ARGUMENT`]).
const FORBIDDEN_PROOF_TOKENS: &[&str] = &[
    "admit(",
    "assume(",
    "assume_specification",
    "assume_termination",
    "exec_allows_no_decreases_clause",
    "external_body",
    "external_fn_specification",
    "external_trait_extension",
    "external_trait_specification",
    "external_type_specification",
    "inline_air_stmt",
    "verifier(external)",
    "verifier::external",
];

/// The `verus_verify(...)` argument that removes an item from verification.
///
/// Every verified function carries `verus_verify`, and with the pinned prover
/// `#[cfg_attr(verus_keep_ghost, verus_verify(external))]` behaves like
/// `verifier::external` under `--no-cheating`: a function with a false
/// `ensures` reports `0 verified, 0 errors`. The argument may sit anywhere in
/// the list and be padded with whitespace, so [`verus_verify_arguments`]
/// parses the list instead of matching a substring.
/// (`verus_verify(external_body)` is reported as `external_body`.)
const FORBIDDEN_VERUS_VERIFY_ARGUMENT: &str = "external";

/// How [`escape_hatches`] reports a `verus_verify` attribute whose arguments
/// include [`FORBIDDEN_VERUS_VERIFY_ARGUMENT`].
const VERUS_VERIFY_EXTERNAL: &str = "verus_verify(external)";

/// Documents, besides the changelog, that quote the proof surface's size.
///
/// The changelog is checked only in its `[Unreleased]` section (see
/// [`unreleased_changelog`]); released sections keep their historical figures.
/// `.github/workflows/ci.yml` is checked when present; packaged archives omit it.
const COUNTING_DOCUMENTS: &[&str] = &[
    "README.md",
    "ROADMAP.md",
    "RELEASE_GATES.md",
    "verus/README.md",
    ".github/workflows/ci.yml",
];

/// Returns the repository root containing `Cargo.toml`.
fn repository_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Lists every Verus harness root, i.e. each `verus/*.rs` file, in file-name order.
fn harnesses(root: &Path) -> Vec<PathBuf> {
    let mut files = fs::read_dir(root.join("verus"))
        .expect("read verus directory")
        .map(|entry| entry.expect("read verus entry").path())
        .filter(|path| path.extension().is_some_and(|extension| extension == "rs"))
        .collect::<Vec<_>>();
    files.sort();
    files
}

/// Returns a harness's file name, such as `cache.rs`.
fn harness_name(harness: &Path) -> String {
    harness
        .file_name()
        .expect("harness file name")
        .to_string_lossy()
        .into_owned()
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

/// Returns the distinct source files that the harnesses include.
fn distinct_included_sources(root: &Path) -> BTreeSet<String> {
    harnesses(root)
        .iter()
        .flat_map(|harness| included_sources(harness))
        .collect()
}

/// Lists every Rust file under `src/` that carries Verus-only code.
///
/// Every verified file needs `verus_keep_ghost`, the configuration Verus
/// compiles with: proof attributes, `verus!` blocks of specifications and
/// lemmas, and ghost-only reference functions all sit behind it. Keying on it
/// rather than on `verus_spec(` also catches a file whose proof content is
/// only lemmas or `verus_verify` items.
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
                    .contains("verus_keep_ghost")
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

/// Whether `character` can continue a Rust identifier.
fn continues_identifier(character: char) -> bool {
    character.is_ascii_alphanumeric() || character == '_'
}

/// Returns the trimmed top-level arguments of every `verus_verify(...)` attribute in `text`.
///
/// `verus_verify` must begin an identifier, so `helper_verus_verify(` does
/// not count, and whitespace may separate it from `(`. The argument list ends
/// at the matching `)`, and only commas outside nested parentheses separate
/// arguments, so `rlimit(10)` is one argument. An unclosed list runs to the
/// end of the text, so a truncated attribute is still scanned.
fn verus_verify_arguments(text: &str) -> Vec<&str> {
    /// The attribute whose argument lists are returned.
    const NAME: &str = "verus_verify";
    let mut arguments = Vec::new();
    for (start, _) in text.match_indices(NAME) {
        if text[..start]
            .chars()
            .next_back()
            .is_some_and(continues_identifier)
        {
            continue;
        }
        let Some(list) = text[start + NAME.len()..].trim_start().strip_prefix('(') else {
            continue;
        };
        let mut depth = 0_usize;
        let mut argument_start = 0;
        let mut end = list.len();
        for (offset, character) in list.char_indices() {
            match character {
                '(' => depth += 1,
                ')' if depth == 0 => {
                    end = offset;
                    break;
                }
                ')' => depth -= 1,
                ',' if depth == 0 => {
                    arguments.push(list[argument_start..offset].trim());
                    argument_start = offset + 1;
                }
                _ => {}
            }
        }
        arguments.push(list[argument_start..end].trim());
    }
    arguments
}

/// Returns the escape hatches that occur in `text`: the forbidden proof
/// tokens in list order, then [`VERUS_VERIFY_EXTERNAL`] when some
/// `verus_verify(...)` attribute has the argument `external`.
///
/// A token that ends in an identifier character must also end the identifier
/// or path it occurs in, so `verifier::external` does not match inside
/// `verifier::external_body` (which `external_body` reports on its own).
fn escape_hatches(text: &str) -> Vec<&'static str> {
    let mut found = FORBIDDEN_PROOF_TOKENS
        .iter()
        .copied()
        .filter(|token| {
            let bounded = token.ends_with(continues_identifier);
            text.match_indices(token).any(|(start, _)| {
                !bounded
                    || !text[start + token.len()..]
                        .chars()
                        .next()
                        .is_some_and(continues_identifier)
            })
        })
        .collect::<Vec<_>>();
    if verus_verify_arguments(text).contains(&FORBIDDEN_VERUS_VERIFY_ARGUMENT) {
        found.push(VERUS_VERIFY_EXTERNAL);
    }
    found
}

/// Parses a decimal count written with optional thousands commas, such as `1,877`.
fn parse_count(word: &str) -> Option<usize> {
    let digits = word.replace(',', "");
    if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    digits.parse().ok()
}

/// Splits prose into whitespace-separated words without surrounding punctuation.
fn words(text: &str) -> Vec<&str> {
    text.split_whitespace()
        .map(|word| word.trim_matches(|character: char| !character.is_ascii_alphanumeric()))
        .collect()
}

/// Returns every count written directly before `noun`, optionally with the
/// `qualifier` word between them (`25 harnesses`, `25 Verus harnesses`).
///
/// Words that are not numerals (`twelve`, `N`, `their`) are not counts.
fn counts_before(text: &str, noun: &str, qualifier: Option<&str>) -> Vec<usize> {
    let words = words(text);
    let mut counts = Vec::new();
    for (index, word) in words.iter().enumerate() {
        if *word != noun || index == 0 {
            continue;
        }
        let mut previous = index - 1;
        if Some(words[previous]) == qualifier && previous > 0 {
            previous -= 1;
        }
        if let Some(count) = parse_count(words[previous]) {
            counts.push(count);
        }
    }
    counts
}

/// Returns the body of the level-two Markdown section titled `heading`, up to the next one.
fn section<'a>(text: &'a str, heading: &str) -> &'a str {
    let marker = format!("## {heading}\n");
    let start = text
        .find(&marker)
        .unwrap_or_else(|| panic!("missing `## {heading}` section"));
    let rest = &text[start + marker.len()..];
    let end = rest.find("\n## ").unwrap_or(rest.len());
    &rest[..end]
}

/// Returns the `[Unreleased]` section of `CHANGELOG.md`, which describes the current tree.
fn unreleased_changelog(text: &str) -> &str {
    section(text, "[Unreleased]")
}

/// Expands shell-style brace groups: `src/{a,b}/c.rs` gives `src/a/c.rs` and `src/b/c.rs`.
fn expand_braces(pattern: &str) -> Vec<String> {
    let Some(open) = pattern.find('{') else {
        return vec![pattern.to_owned()];
    };
    let close = open + pattern[open..].find('}').expect("closing brace");
    pattern[open + 1..close]
        .split(',')
        .flat_map(|choice| {
            expand_braces(&format!(
                "{}{choice}{}",
                &pattern[..open],
                &pattern[close + 1..]
            ))
        })
        .collect()
}

/// Returns the code spans (text between backticks) of a Markdown table cell.
fn code_spans(cell: &str) -> Vec<&str> {
    cell.split('`').skip(1).step_by(2).collect()
}

/// One row of the harness table in `verus/README.md`.
struct HarnessRow {
    /// Harness file name, such as `cache.rs`.
    name: String,
    /// Source files the row lists, with brace groups expanded.
    sources: BTreeSet<String>,
    /// The `verified` count Verus reported for the harness in the recorded run.
    verified: usize,
}

/// Parses the table in the `Harnesses` section of `verus/README.md`, whose rows read
/// `| harness | sources | verified | what the proofs establish |`.
fn harness_table(readme: &str) -> Vec<HarnessRow> {
    section(readme, "Harnesses")
        .lines()
        .filter(|line| line.starts_with("| `"))
        .map(|line| {
            let cells = line.split('|').collect::<Vec<_>>();
            assert!(cells.len() >= 5, "malformed harness table row: {line}");
            let name = code_spans(cells[1])
                .first()
                .copied()
                .unwrap_or_else(|| panic!("harness table row without a name: {line}"))
                .to_owned();
            let sources = code_spans(cells[2])
                .into_iter()
                .flat_map(expand_braces)
                .collect();
            let verified = parse_count(cells[3].trim())
                .unwrap_or_else(|| panic!("harness table row without a verified count: {line}"));
            HarnessRow {
                name,
                sources,
                verified,
            }
        })
        .collect()
}

/// Returns the recorded total, written as `**N verified, 0 errors**` in the
/// `Harnesses` section of `verus/README.md`.
fn recorded_total(readme: &str) -> usize {
    let harnesses = section(readme, "Harnesses");
    let end = harnesses
        .find(" verified, 0 errors**")
        .expect("verus/README.md records `**N verified, 0 errors**`");
    let start = harnesses[..end].rfind("**").expect("bold total") + 2;
    parse_count(&harnesses[start..end]).expect("numeric verified total")
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
            let found = escape_hatches(&text);
            assert!(
                found.is_empty(),
                "{} uses forbidden proof escape hatches {found:?}",
                file.display()
            );
        }
    }
}

/// The scanner reports every trusted construct, in each attribute spelling.
#[test]
fn escape_hatch_scanner_rejects_every_trusted_construct() {
    let cases = [
        ("proof! { admit(); }", "admit("),
        ("proof! { assume(x > 0); }", "assume("),
        (
            "pub assume_specification[ f ](x: u8);",
            "assume_specification",
        ),
        (
            "#[cfg_attr(verus_keep_ghost, verifier::assume_termination)]",
            "assume_termination",
        ),
        (
            "#[cfg_attr(verus_keep_ghost, verifier::exec_allows_no_decreases_clause)]\n\
             fn spin() -> u8 { loop {} }",
            "exec_allows_no_decreases_clause",
        ),
        ("#[verifier::external_body]", "external_body"),
        (
            "#[verifier::external_fn_specification]",
            "external_fn_specification",
        ),
        (
            "#[verifier::external_trait_extension]",
            "external_trait_extension",
        ),
        (
            "#[verifier::external_trait_specification]",
            "external_trait_specification",
        ),
        (
            "#[verifier::external_type_specification]",
            "external_type_specification",
        ),
        ("inline_air_stmt(\"(assume false)\")", "inline_air_stmt"),
        ("#[verifier(external)]", "verifier(external)"),
        ("#[verifier::external]", "verifier::external"),
        (
            "#[cfg_attr(verus_keep_ghost, verifier::external)]",
            "verifier::external",
        ),
        (
            "#[cfg_attr(verus_keep_ghost, verifier::external, verus_verify)]",
            "verifier::external",
        ),
        (
            "#[cfg_attr(verus_keep_ghost, verus_verify(external))]",
            "verus_verify(external)",
        ),
        ("#[verus_verify(external)]", "verus_verify(external)"),
        ("#[verus_verify( external )]", "verus_verify(external)"),
        (
            "#[verus_verify(external, rlimit(10))]",
            "verus_verify(external)",
        ),
        (
            "#[verus_verify(rlimit(10), external)]",
            "verus_verify(external)",
        ),
        (
            "#[cfg_attr(verus_keep_ghost, verus_verify (\n    external,\n))]",
            "verus_verify(external)",
        ),
    ];
    for (text, token) in cases {
        assert!(
            escape_hatches(text).contains(&token),
            "`{token}` not reported in {text:?}"
        );
    }
    assert_eq!(
        escape_hatches("#[verifier::external_body]"),
        ["external_body"],
        "`verifier::external` must not match inside `verifier::external_body`"
    );
    assert_eq!(
        escape_hatches("#[cfg_attr(verus_keep_ghost, verus_verify(external_body))]"),
        ["external_body"],
        "`verus_verify(external_body)` is reported once, as `external_body`"
    );
}

/// The crate's `verus_verify` attribute passes the scanner unless an argument is `external`.
#[test]
fn verus_verify_arguments_other_than_external_pass_the_scanner() {
    for text in [
        "#[cfg_attr(verus_keep_ghost, verus_verify)]",
        "#[cfg_attr(verus_keep_ghost, verus_verify(dual_spec))]",
        "#[verus_verify(rlimit(10))]",
        "#[verus_verify(dual_spec, rlimit(external))]",
        "fn helper_verus_verify(external: u8) {}",
        "// verus_verify",
    ] {
        assert_eq!(
            escape_hatches(text),
            Vec::<&str>::new(),
            "{text:?} must pass the scanner"
        );
    }
}

/// `verus_verify` argument lists split at top-level commas and end at the matching parenthesis.
#[test]
fn verus_verify_argument_parser_respects_nesting_and_bounds() {
    assert_eq!(
        verus_verify_arguments("#[verus_verify(a, rlimit(1, 2), b)] fn f() {}"),
        ["a", "rlimit(1, 2)", "b"]
    );
    assert_eq!(
        verus_verify_arguments("#[verus_verify] #[verus_verify (x)] #[verus_verify()]"),
        ["x", ""]
    );
    assert_eq!(
        verus_verify_arguments("#[verus_verify(external"),
        ["external"],
        "an unclosed list is scanned to the end of the text"
    );
    assert_eq!(
        verus_verify_arguments("my_verus_verify(external); verus_verify_x(external)"),
        Vec::<&str>::new(),
        "only the `verus_verify` identifier itself introduces arguments"
    );
}

/// Ordinary proof code, including loop invariants and decreases clauses, passes the scanner.
#[test]
fn escape_hatch_scanner_accepts_ordinary_proof_code() {
    let text = r#"
#[cfg(verus_keep_ghost)]
verus! {
pub proof fn lemma_square(n: int) requires n >= 0 ensures n * n >= 0 {
    assert(n * n >= 0) by (nonlinear_arith) requires n >= 0;
}
}
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(r => requires d > 0 ensures r == n / d))]
fn quotient(n: u64, d: u64) -> u64 {
    #[cfg_attr(verus_keep_ghost, verus_spec(invariant i <= n, decreases n - i))]
    while i < n { i += 1; }
    #[cfg_attr(verus_keep_ghost, verifier::loop_isolation(false))]
    n / d
}
"#;
    assert_eq!(escape_hatches(text), Vec::<&str>::new());
}

/// A source specification that no harness includes would never be checked.
#[test]
fn every_annotated_source_is_verified_by_a_harness() {
    let root = repository_root();
    let unchecked = annotated_sources(&root)
        .difference(&distinct_included_sources(&root))
        .cloned()
        .collect::<Vec<_>>();
    assert!(
        unchecked.is_empty(),
        "source files carry Verus-only code but no verus/*.rs harness includes them: {unchecked:?}"
    );
}

/// The harness table in `verus/README.md` lists every harness once, with its exact sources.
#[test]
fn verus_readme_table_lists_every_harness_with_its_sources() {
    let root = repository_root();
    let readme = fs::read_to_string(root.join("verus/README.md")).expect("read verus/README.md");
    let rows = harness_table(&readme);
    let harnesses = harnesses(&root);
    assert_eq!(
        rows.iter().map(|row| row.name.clone()).collect::<Vec<_>>(),
        harnesses
            .iter()
            .map(|harness| harness_name(harness))
            .collect::<Vec<_>>(),
        "verus/README.md must list every verus/*.rs harness once, in file-name order"
    );
    for (row, harness) in rows.iter().zip(&harnesses) {
        let included = included_sources(harness)
            .into_iter()
            .collect::<BTreeSet<_>>();
        assert_eq!(
            row.sources, included,
            "verus/README.md lists different verified sources for {}",
            row.name
        );
    }
    assert_eq!(
        rows.iter().map(|row| row.verified).sum::<usize>(),
        recorded_total(&readme),
        "the per-harness verified counts in verus/README.md must sum to its bold total"
    );
}

/// `verus/README.md` states the number of distinct verified source files correctly.
#[test]
fn verus_readme_counts_distinct_sources() {
    let root = repository_root();
    let readme = fs::read_to_string(root.join("verus/README.md")).expect("read verus/README.md");
    assert_eq!(
        counts_before(&readme, "distinct", None),
        [distinct_included_sources(&root).len()],
        "verus/README.md must state how many distinct source files the harnesses include"
    );
}

/// Every document that quotes the proof surface quotes the current harness count and total.
#[test]
fn documents_quote_the_current_proof_surface() {
    let root = repository_root();
    let harness_count = harnesses(&root).len();
    let total = recorded_total(
        &fs::read_to_string(root.join("verus/README.md")).expect("read verus/README.md"),
    );
    let mut documents = COUNTING_DOCUMENTS
        .iter()
        .filter_map(|path| {
            fs::read_to_string(root.join(path))
                .ok()
                .map(|text| ((*path).to_owned(), text))
        })
        .collect::<Vec<_>>();
    let changelog = fs::read_to_string(root.join("CHANGELOG.md")).expect("read CHANGELOG.md");
    documents.push((
        "CHANGELOG.md [Unreleased]".to_owned(),
        unreleased_changelog(&changelog).to_owned(),
    ));
    let mut quoted = 0;
    for (path, text) in &documents {
        for count in counts_before(text, "harnesses", Some("Verus")) {
            quoted += 1;
            assert_eq!(
                count, harness_count,
                "{path} quotes {count} Verus harnesses; verus/ has {harness_count}"
            );
        }
        for count in counts_before(text, "verified", None) {
            assert_eq!(
                count, total,
                "{path} quotes {count} verified items; verus/README.md records {total}"
            );
        }
    }
    assert!(
        quoted > 0,
        "no document quotes the harness count, so this check would be vacuous"
    );
}

/// The count parser reads the spellings the documents use and ignores other words.
#[test]
fn count_parser_reads_documented_spellings() {
    let text = "The 25 harnesses include 87 distinct source files: twelve fuzz \
                targets, and 25\n  Verus harnesses (**1,877 verified, 0 errors**). \
                Integration-test harnesses and `N verified` stay unchecked.";
    assert_eq!(counts_before(text, "harnesses", Some("Verus")), [25, 25]);
    assert_eq!(counts_before(text, "verified", None), [1_877]);
    assert_eq!(counts_before(text, "distinct", None), [87]);
    assert_eq!(parse_count("1,877"), Some(1_877));
    assert_eq!(parse_count("N"), None);
    assert_eq!(parse_count(""), None);
    assert_eq!(
        expand_braces("src/{a,b}/x/{c,d}.rs"),
        [
            "src/a/x/c.rs",
            "src/a/x/d.rs",
            "src/b/x/c.rs",
            "src/b/x/d.rs"
        ]
    );
    assert_eq!(expand_braces("src/lib.rs"), ["src/lib.rs"]);
    assert_eq!(
        code_spans(" `src/{a,b}.rs`, `src/c.rs` "),
        ["src/{a,b}.rs", "src/c.rs"]
    );
    assert_eq!(
        unreleased_changelog("# Log\n## [Unreleased]\nnew\n## [0.1.0]\nold\n"),
        "new"
    );
    assert_eq!(
        section("## A\n| `x.rs` |\n### B\nb\n## C\nc", "A"),
        "| `x.rs` |\n### B\nb"
    );
    assert_eq!(section("## Last\ntail", "Last"), "tail");
}

/// A file whose only Verus content is a `verus!` block of lemmas still needs a harness.
#[test]
fn lemma_only_sources_count_as_annotated() {
    let root = tempfile::tempdir().expect("temporary crate");
    let source = root.path().join("src/proofs");
    fs::create_dir_all(&source).expect("create src/proofs");
    fs::write(
        source.join("lemmas.rs"),
        "#[cfg(verus_keep_ghost)]\nverus! {\npub proof fn lemma_zero() ensures 0int == 0 {}\n}\n",
    )
    .expect("write lemma file");
    fs::write(
        root.path().join("src/plain.rs"),
        "pub fn value() -> u8 { 7 }\n",
    )
    .expect("write plain file");
    assert_eq!(
        annotated_sources(root.path()),
        BTreeSet::from(["src/proofs/lemmas.rs".to_owned()])
    );
}
