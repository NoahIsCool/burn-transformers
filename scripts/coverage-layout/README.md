# Supplementary Rust source-layout classifier

This standalone development tool parses Rust source with pinned Syn and emits
versioned JSON. It adds no dependencies to the transformer runtime. It does not
read coverage profiles, change a denominator, or decide whether a release passes.
It is repository-only, like the independent fuzz package: Cargo omits nested
standalone packages from the transformer runtime archive. Full source/MSRV audit
groups require a repository checkout; runtime archive verification is separate.

```sh
cargo +stable run --manifest-path scripts/coverage-layout/Cargo.toml --locked --offline -- . src/lib.rs
cargo +stable test --manifest-path scripts/coverage-layout/Cargo.toml --locked --offline
cargo +stable clippy --manifest-path scripts/coverage-layout/Cargo.toml --locked --offline --all-targets -- -D warnings
```

Run these commands from the transformer crate. The first positional argument is
the containment root; the remaining arguments are explicit crate entry files.
The selected `stable` alias must resolve to Rust 1.92 or newer. The tool has its
own lockfile. Redirect stdout to a fresh audit artifact if retaining a report.
`--help` prints the command syntax. Place `--` before the root to preserve a
dash-prefixed path; native operating-system path bytes are retained. Missing
entries and unknown leading options fail before any source reads or JSON output.

## Meaning of the report

Every read source buffer carries its exact SHA-256, byte length, and physical-line
count. Byte spans are half-open; lines and UTF-8 byte columns are one-based.
Unicode scalar columns are also retained. BOM, CRLF, and shebang bytes participate
in hashing and coordinates. Comments and strings are parsed as Rust syntax, not
searched for text resembling attributes.

Configuration predicates are evaluated once with `test=false` and once with
`test=true`. All other atoms remain unknown. Ancestor conditions are conjoined;
only a condition false in production and possibly true in tests yields a
test-only **candidate**. All observed references to one canonical source path
are combined, so a production reference prevents exclusion based on an earlier
test reference. Same-line production/test ownership is reported as mixed and
retained. Unknown predicates and ambiguous paths remain visible and retained.

The tool follows external `mod` declarations and literal `#[path]` attributes,
including nested inline modules. Aliases and active recursion cycles are separate
cases. Bounds are 4,096 reference contexts, 128 traversal levels, and 8 MiB per
source buffer. Missing/escaping files, parse errors, and traversal limits are
diagnostics or fatal input errors, never inferred empty sources.

## Qualification still required

This is supplementary syntax classification, **not proof of complete compiler
reachability**. It neither expands procedural/declarative macros nor follows
`include!` or arbitrary macro-generated module edges. An opaque production macro
could introduce another reference to a file observed only through a test module.
Candidate test-only spans therefore must not be used for exclusions until those
possible references have been resolved against authenticated compiler evidence.
No generated code is automatically excluded. A diagnostic-free report is not
an acceptance certificate, and the process's zero exit status only means a report
was produced.

Before gate integration, bind the report to the exact source inventory, command,
features, toolchain, profile, and LLVM export; verify source bytes before and
after execution. Reports from older source revisions cannot qualify edited
files. Canonical-path containment and one-read snapshots assume ordinary local
filesystem concurrency, not an adversary replacing files between checks.

## Versioned structural test audit

Each file additionally records `test_audit.schema_version = 1`. This independent
parsed-attribute visitor distinguishes bodyless external module declarations
from inline modules and records bare `#[test]`/`#[bench]`, test-sensitive cfg, and
conditional attributes on files, items, associated items, expressions, and
statements. It does **not** expand macro tokens or change any exclusion candidate.
Pure production-only cfg and ordinary feature cfg introduce no test-body claim;
test-dependent uncertainty remains an explicit blocker. Conditional/expanding
external declarations are not treated as harmless.

`complete` means non-opaque parsed syntax was visited within span/depth bounds.
It is false on parse, verbatim, depth, or coordinate failure; it never attests
macro expansion, generated syntax, compiler reachability, or tool build origin.
The consumer joins this audit with the independently authenticated compiler
capture. Missing/old audits are blockers. Even a complete empty audit remains a
diagnostic: opaque macros can emit tests/helpers from original source locations
that cannot be distinguished from runtime code by this parsed audit alone.

## Authored documentation check

The separate `--documentation` mode checks contributor source style. It does not
change coverage classification or require compiler coverage evidence. Build the
helper once, then run the Python wrapper from the repository root:

```sh
cargo +stable build --release --locked --manifest-path scripts/coverage-layout/Cargo.toml --target-dir target
python3 scripts/check_documentation.py --classifier target/release/transformers-coverage-layout --output target/documentation.json
```

Use `--offline` on Cargo only when dependencies are already cached. The wrapper
never builds or executes transformer code. It inventories tracked and untracked
nonignored Rust sources under `src/`, `examples/`, `scripts/`, and `fuzz/`, selects
crate roots, and conservatively audits unreferenced authored sources as additional
entries. Retained evidence, integration-test roots, and fixture directories are
excluded from discovery; a production module reference still brings their source
into the audit. A `*_tests.rs` filename alone never proves test ownership.

Every authored function needs a nonempty literal `///` comment, including private
helpers, trait declarations/default methods, trait implementation methods, and
foreign declarations. Every physical or inline production module needs a
nonempty inner `//!` comment. Bodyless `mod name;` declarations are checked through
their destination files. Equivalent block comments and `#[doc = "..."]` are
reported as existing rustdoc but do not satisfy these exact spelling rules.
Computed documentation is reported separately. Actual `#[test]`/`#[bench]` and
test-only cfg bodies are exempt; uncertain feature/cfg ownership remains included.
Inline test modules containing functions are separately reported for relocation
to sibling test files. Duplicate module references preserve all contexts and
produce informational diagnostics, rather than an incomplete audit.

Macro boundaries remain visible. The checker visits parseable authored item
fragments and checks real attribute spans on simple `fn $name` templates. A
template's `#[doc = $description]` cannot substitute for literal `///`; an
accompanying literal comment can satisfy the style. Unresolvable authored function
prefixes block completion. Macro expansion, generated derive methods (including
Burn, Serde, and thiserror), and externally generated source are outside this
authored-source claim: `macro_expansion_resolved` always remains false. This is
not a universal expansion or compiler-reachability proof.

JSON includes exact source and helper hashes, source spans, test exclusions,
alternative doc forms, unresolved macro functions, and stable component counts.
The wrapper verifies source inventory and bytes before/after parsing. Exit codes
are **0** for success, **1** for missing style or misplaced inline tests, and
**2** for incomplete inspection or invalid input. `--inventory-only` tolerates
style findings, but still fails on incomplete inspection. The raw Rust CLI only
reports JSON; use the wrapper for enforcement. Neither form judges whether prose
is accurate or useful; that remains a review responsibility.

Focused regression checks:

```sh
cargo +stable test --locked --manifest-path scripts/coverage-layout/Cargo.toml --test documentation
python3 -m unittest discover -s scripts/tests -p test_check_documentation.py -v
```

## Comprehensive authored-item standards

The opt-in `--standards ROOT ENTRY [ENTRY ...]` mode emits schema 2. It retains
the exact source hashes, spans, literal documentation requirements, and macro
expansion limitations above. The existing `--documentation` mode continues to
emit schema 1 with its original function/module scope and test exemptions.

Schema 2 additionally requires documentation on private and public structs,
enums, unions, traits, trait/type aliases, constants, statics, associated
constants/types, foreign statics/types, macro definitions, enum variants, and
each named or tuple field. Tuple fields use their zero-based position as their
reported name. Test functions, helpers, and module bodies require documentation
as well; `test_only` remains ownership evidence rather than an exemption.
Imports, implementation blocks, generic parameters, and local bindings introduce
no separate documentation requirement. File/module documentation explains those
contexts; documentation on a container cannot satisfy its members' requirements.

Parseable authored macro fragments retain their original source spans. Opaque
named declaration templates appear in `macros[].unresolved_items`, including in
test-owned macros, and block completion. A syntax audit cannot evaluate generated
fields or declarations introduced by macro expansion. `complete` describes the
parsed authored scope, and `macro_expansion_resolved` remains false.

Run the raw mode against explicit crate roots, for example:

```sh
cargo +stable run --locked --manifest-path scripts/coverage-layout/Cargo.toml -- --standards . src/lib.rs scripts/coverage-layout/src/lib.rs scripts/coverage-layout/src/main.rs
```

As in the legacy raw CLI, exit 0 means JSON was written successfully; inspect
`complete`, `passed`, and per-site `style_present` before treating an audit as
successful. This mode checks documentation presence and placement, not prose
quality, behavioral correctness, test coverage, or formal verification.

For a source-inventory-bound repository audit, opt the wrapper into the same
policy. It additionally discovers `tests/` and nested tooling test harnesses,
selects their crate roots, and adds any unreferenced source as an explicit entry.
It retains fixture and historical-evidence discovery exclusions visibly in JSON;
an actual module reference still brings excluded sources into the parsed graph.

```sh
python3 scripts/check_documentation.py --standards --classifier target/debug/transformers-coverage-layout --output target/standards.json
```

Schema 2 inventories group authored sites and missing documentation by declaration
kind, with test ownership recorded separately. Missing entries include the path,
one-based line, name, kind, expected spelling, and observed alternative doc forms.
`--inventory-only` permits missing prose for an exploratory report while still
failing incomplete inspection. The default wrapper invocation retains its legacy
schema, discovery scope, summaries, and exit codes. The repository source gate
explicitly selects `--standards` so contributor validation enforces the expanded
requirements.

The sibling-test rule applies to unit tests embedded in production modules.
Schema 2 exempts nested tests from relocation when every observed reference to
their physical source originates in an explicit integration entry directly under
`tests/` or a tooling package's `scripts/.../tests/` directory. The report records
that placement policy as `test_harness_only`; it neither changes structural
`test_only` evidence nor exempts any documentation. A production reference to the
same source prevents the relocation exemption. This is a repository layout
convention over the observed graph, not a compiler test-harness proof.
