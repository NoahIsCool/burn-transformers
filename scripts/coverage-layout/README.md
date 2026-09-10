# Supplementary Rust source-layout classifier

This standalone development tool parses Rust source with pinned Syn and emits
versioned JSON. It adds no dependencies to the transformer runtime. It does not
read coverage profiles, change a denominator, or decide whether a release passes.

```sh
cargo +stable run --manifest-path scripts/coverage-layout/Cargo.toml --locked --offline -- . src/lib.rs
cargo +stable test --manifest-path scripts/coverage-layout/Cargo.toml --locked --offline
cargo +stable clippy --manifest-path scripts/coverage-layout/Cargo.toml --locked --offline --all-targets -- -D warnings
```

Run these commands from the transformer crate. The first positional argument is
the containment root; the remaining arguments are explicit crate entry files.
The selected `stable` alias must resolve to Rust 1.92 or newer. The tool has its
own lockfile. Redirect stdout to a fresh audit artifact if retaining a report.

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
