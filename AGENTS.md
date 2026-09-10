# Repository Guidelines

These are required development standards, not optional release polish. Apply
them throughout implementation while following the current priorities in
`ROADMAP.md`.

## Project Structure & Module Organization

This Rust/Burn crate targets transformer experimentation. Prioritize activation inspection, intervention, and composition; inference validates implementations. Follow `ROADMAP.md`: interfaces in v0.2, training in v0.3. Shared mathematics lives in `src/attention.rs`, `src/cache/`, and related modules; family wiring belongs in `src/models/`. Keep reusable operations free of model-name branches and decode caches external to parameters.

Unit tests use sibling `*_tests.rs` files. Integration tests and fixtures live in `tests/`; runnable experiments live in `examples/`. Document consumer workflows in `EXPERIMENTS.md` and `TRAINING.md`. `scripts/` contains release and reference tooling, `fuzz/` contains fuzz targets, and `acceptance/` stores measurement evidence. `models/` and `target/` are ignored local storage.

## Implementation Workflow & Dependency Policy

- Build production-quality, independently testable components. Keep shared
  mathematical features separate from family-specific configuration translation,
  layer wiring, and checkpoint mappings. Do not introduce model-name branches
  into reusable operations or put decode caches into model parameters.
- Before implementing or changing an architecture, inspect its official inference
  code and checkpoint configuration. Pin exact reference-code and checkpoint
  revisions, record tensor-level compatibility and required transformations,
  and identify which features can be shared with other families.
- Implement and validate shared features incrementally, then compose them into
  models. Run focused tests and fix errors early, before building more features
  on top of an unverified component. Validate tiny synthetic configurations
  before attempting large checkpoints.
- Keep dependencies small and centered on Burn. Reuse Burn primitives when their
  semantics match and established libraries for standard functionality, such as
  Serde for serialization. Do not reimplement major existing facilities without
  a documented architectural need, and do not use `burn-transformers` as a
  substitute for implementing this crate's architectures.
- Update architecture and support documentation, including
  `TRANSFORMER_INNOVATIONS.md`, as research and implementation progress. Correct
  discovered inaccuracies against pinned sources; distinguish implemented,
  experimentally validated, and advertised-supported capabilities.

## Build, Test, and Development Commands

Run from the repository root. Use Rust 1.92 or newer; `Cargo.toml` pins Burn 0.21. Default features enable `std` and Flex.

- `cargo build --locked`: build with the committed dependency lockfile.
- `cargo test --locked`: run ordinary Rust tests and doctests.
- `cargo test --locked normalization`: run tests matching a focused name filter.
- `cargo fmt --check`: verify rustfmt formatting.
- `cargo clippy --locked --all-targets --all-features -- -D warnings`: enforce warning-free linting on an equipped build host.
- `python3 -m unittest discover -s scripts/tests -v`: test Python tooling.
- `python3 scripts/check_release.py --gate source --offline`: run the consolidated source checks with cached dependencies.
- `cargo run --release --features autodiff --example t5_classifier -- --train`: train a small classifier locally.

Before compiling, verify at least 20 GiB of local free space and use Cargo's
normal target directory when that preflight passes. Document any necessary
storage fallback. Keep downloaded artifacts on the ignored external `models/`
volume and verify its availability and free space before each pinned download.

## Coding Style & Naming Conventions

Use Rust 2024 conventions, four-space indentation, and rustfmt. Use `snake_case` for modules/functions and `UpperCamelCase` for types. Preserve explicit backend, dtype, shape, and cache contracts; return contextual errors for unsupported capabilities.

Document every production function, including private helpers, with standard
Rust `///` comments. Use module-level `//!` documentation and readable section
comments to explain each component: not only what the code does, but also the
mathematical theory and equations, tensor axes, numerical precision assumptions,
mask/cache invariants, and algorithmic tradeoffs. Keep public documentation
enforced with `#![deny(missing_docs)]`.

## Testing Guidelines

Use Rust `#[test]`, existing `proptest` patterns, and deterministic Flex fixtures. Name tests after observable behavior. Cover numerical parity, invalid inputs, and cache boundaries when relevant. Hardware and real-checkpoint tests require their documented features and local artifacts.

- Each shared feature must have an expansive unit-test suite covering normal
  behavior, invalid configurations, shapes, dtypes, masks, and relevant edge
  cases. Include padding and fully masked rows, offsets, capacity boundaries,
  reset/reorder/truncate/rollback, ring wraparound, and routing ties where those
  operations apply.
- Use golden mathematical comparisons against pinned official references and
  the documented F32/F16 parity tolerances. Test full-forward versus
  prefill/decode and optimized/chunked versus reference/recurrent paths where
  applicable. Test loaders, checkpoint transformations, quantization, tokenizers,
  and generation with both valid fixtures and malformed/adversarial inputs.
- Add property tests and relevant fuzz targets, rather than relying only on
  hand-picked examples. Keep full-precision stateless graphs autodiff-safe and
  include smoke tests proving gradients exist. Quantized inference does not
  imply training support.
- Aim for near-100% meaningful reachable branches in pure configuration,
  mathematical helpers, cache-state, and artifact-validation code. Enforce at
  least 95% shared/core and 90% crate-wide line coverage under
  `RELEASE_GATES.md`. Do not weaken thresholds or add exclusions merely to make
  a gate green; literal 100% crate-wide coverage is not a substitute for useful
  parity, property, fuzz, and hardware tests.
- Keep deterministic Flex tests as the ordinary baseline. Run opt-in Metal F16
  tests serially and only with their required backend and local artifacts.
- Before advertising checkpoint support, require strict real-weight loading
  with zero unexplained missing/unexpected tensors, reference-logit and greedy
  generation parity, cached/uncached agreement, and the applicable memory,
  context-boundary, quantization-quality, and performance acceptance gates.
  Successful loading or synthetic tests alone do not establish support.

Continuously run the checks relevant to changed code. CI/release qualification
must include formatting, warning-denied Clippy, Rust tests and doctests, MSRV and
stable builds, Python tooling tests, dependency advisory/license checks, and
CUDA/ROCm compile checks on equipped hosts. Follow `RELEASE_GATES.md` for exact
commands, tolerances, coverage measurement rules, and additional hardware gates.
Report failed, skipped, or unavailable checks explicitly; never present unrun
checks or old-source measurements as current validation.

## Commit & Pull Request Guidelines

- Keep `codex/transformers` as the integration branch. Use sibling branches such
  as `codex/tfm-core` or `codex/tfm-checkpoint` for parallel work. Do not create
  `codex/transformers/...` descendants: the integration ref prevents that naming
  hierarchy.
- Use subagents in separate Git worktrees for independent parallel components,
  with explicit, non-overlapping ownership. One integrator owns Cargo metadata,
  `src/lib.rs`, public traits, and the model registry; coordinate changes to
  those shared surfaces before integration.
- Make focused, conventionally named commits at every major milestone, rather
  than accumulating the implementation into one final commit. Follow
  `type(transformers): imperative summary`, such as
  `fix(transformers): validate cache capacity`.
- Cherry-pick or merge only green, tested milestones. Re-run relevant checks on
  the integrated result and record which source snapshot and commands were
  validated. Do not silently carry failing tests into the next milestone.
- Preserve unrelated user changes and other agents' work. Stage explicit paths;
  never use `git add -A` or a broad staging command that sweeps in unrelated
  work. Do not destructively reset, overwrite, or clean existing changes.
- Keep commits local unless pushing, publishing, remote creation, or tagging is
  separately authorized.

PRs should describe the problem, changed behavior, relevant issues, and
validation commands/results; identify unrun hardware checks. Update acceptance
records and support documentation when claims change. Preserve authenticated
historical evidence and record new measurements separately with exact source
and artifact identities.

Keep weights, generated Burnpacks, credentials, and build outputs out of commits. The runtime loads local artifacts only; successful loading alone does not establish checkpoint support.
