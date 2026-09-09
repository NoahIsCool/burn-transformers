# Release checklist

This checklist prepares a source release of the standalone `transformers`
crate. Completing it does not publish to a registry, create a Git tag, or make
an unsupported checkpoint claim.

## 1. Establish the release scope

- Work from the intended integration commit with no unreviewed transformer
  changes.
- Confirm `package.version`, `package.rust-version`, and the pinned Burn version
  in `Cargo.toml`.
- Add repository, homepage, or documentation URLs only after a canonical public
  location exists; do not invent metadata merely to silence Cargo's warning.
- Date the matching `CHANGELOG.md` entry and leave later work under
  `Unreleased`.
- Confirm both `LICENSE-MIT` and `LICENSE-APACHE` are present.
- Keep model weights, tokenizer artifacts, generated Burnpacks, build output,
  and local model symlinks outside Git and the Cargo package.

## 2. Reconcile support claims

- Treat an architecture implementation as experimental unless its exact
  checkpoint record says `advertised_supported: true`.
- Check that every advertised record names an immutable revision, backend,
  dtype, tokenizer evidence, cache behavior, generation evidence, and known
  limits.
- Reconcile the README support matrix and architecture documentation with those
  records. An acceptance result must never be inferred from successful loading
  alone.
- Re-run every non-CI hardware or large-artifact gate whose recorded inputs or
  execution code changed since its evidence was captured.

## 3. Run local source gates

Run commands from the repository root with the standalone manifest:

```text
cargo fmt --manifest-path crates/transformers/Cargo.toml --check
cargo clippy --manifest-path crates/transformers/Cargo.toml \
  --all-targets --all-features -- -D warnings
cargo test --manifest-path crates/transformers/Cargo.toml
cargo check --manifest-path crates/transformers/Cargo.toml --no-default-features
cargo check --manifest-path crates/transformers/Cargo.toml \
  --no-default-features --features std
cargo doc --manifest-path crates/transformers/Cargo.toml \
  --no-default-features --features std,flex,tokenizers,chat-templates --no-deps
```

Run every public example with the feature set and local checkpoint revision
documented by that example. Accelerator acceptance runs must use the certified
backend and dtype rather than substituting a portable test backend.

## 4. Verify the distributable crate

First inspect the exact allow-listed file set, then build and verify the source
archive:

```text
cargo +stable package --manifest-path crates/transformers/Cargo.toml --locked --list
cargo +stable package --manifest-path crates/transformers/Cargo.toml --locked
```

The list may contain source, examples, compact test fixtures, reference scripts,
documentation, and acceptance metadata. It must not contain `models/`,
`target/`, SafeTensors or PyTorch weights, Burnpack sidecars, authentication
material, or generated reference payloads intended for external storage.

Package from the clean, committed transformer source tree. Preserve unrelated
changes outside this standalone crate. Record the commit ID and archive checksum
beside any separately distributed release
artifact. Publishing, tagging, or pushing remains a distinct, explicitly
authorized operation.

## 5. Complete executable and hardware release gates

[RELEASE_GATES.md](RELEASE_GATES.md) defines the command runner, exact MSRV
verification, failing coverage thresholds, dependency policy, separate backend
compile checks, and the remaining fuzz/hardware/performance requirements.
Passing the source/package commands above does not waive those gates.
