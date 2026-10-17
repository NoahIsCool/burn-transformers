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
- Reconcile the [model status page](ARCHITECTURE_COMPATIBILITY.md), the README
  family table, and the architecture documentation with those records. An
  acceptance result must never be inferred from successful loading alone.
- Re-run every non-CI hardware or large-artifact gate whose recorded inputs or
  execution code changed since its evidence was captured.

## 3. Run local source gates

Run commands from the repository root with the standalone manifest:

```text
cargo fmt --manifest-path Cargo.toml --check
cargo clippy --manifest-path Cargo.toml \
  --all-targets --all-features -- -D warnings
cargo test --manifest-path Cargo.toml
cargo check --manifest-path Cargo.toml --no-default-features
cargo check --manifest-path Cargo.toml \
  --no-default-features --features std
cargo doc --manifest-path Cargo.toml \
  --no-default-features --features std,flex,tokenizers,chat-templates --no-deps
```

Then build the documentation exactly as docs.rs does, from
`[package.metadata.docs.rs]`: nightly rustdoc with `--cfg docsrs`, which also
labels feature-gated items with the Cargo feature they require.

```text
RUSTDOCFLAGS="--cfg docsrs -D warnings" cargo +nightly doc \
  --manifest-path Cargo.toml --locked --no-deps --lib \
  --no-default-features --features std,flex,tokenizers,chat-templates
```

Run every public example with the feature set and local checkpoint revision
documented by that example. Accelerator acceptance runs must use the certified
backend and dtype rather than substituting a portable test backend.

## 4. Verify the distributable crate

First inspect the exact allow-listed file set, then build and verify the source
archive:

```text
cargo +stable package --manifest-path Cargo.toml --locked --list
cargo +stable package --manifest-path Cargo.toml --locked
```

The list may contain source, examples, compact test fixtures, reference scripts,
documentation, and the published acceptance records. It must not contain
`models/`, `target/`, SafeTensors or PyTorch weights, Burnpack sidecars,
authentication material, generated reference payloads intended for external
storage, or repository-only evidence: measurement campaign directories other
than the published compact summaries, and the mechanistic studies' recorded
figures, result records, and captured runs (see
[`scripts/package_contents.py`](scripts/package_contents.py)). The compressed
`.crate` must stay below the package gate's 8 MiB budget, under crates.io's
10 MiB upload limit. `python3 scripts/check_release.py --gate package --offline`
checks both and runs the archive's Rust and Python tests. Finally, confirm that
the registry would accept the archive without uploading it:

```text
cargo +stable publish --manifest-path Cargo.toml --locked --dry-run
```

Package from the clean, committed transformer source tree. Preserve unrelated
changes outside this standalone crate. Record the commit ID and archive checksum
beside any separately distributed release
artifact. Publishing, tagging, or pushing remains a distinct, explicitly
authorized operation.

## 5. Complete executable and hardware release gates

[RELEASE_GATES.md](RELEASE_GATES.md) defines the command runner, exact MSRV
verification, coverage thresholds and qualification, dependency policy, separate backend
compile checks, and the remaining fuzz/hardware/performance requirements.
Passing the source/package commands above does not waive those gates.
