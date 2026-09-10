# Standalone repository migration

On September 10, 2026, `crates/transformers` was extracted from the local
`hmm_compression` repository into `/Users/nover/repos/transformers`. The parent
checkout removes the crate entirely: no submodule, path dependency, or
compatibility symlink remains. The integration branch stays `codex/transformers`.
No remote, registry publication, or release tag is created by this migration.

## Preserved history and artifacts

The extraction filters every original local branch to `crates/transformers/`
and moves that subtree to the new repository root. Commit IDs necessarily
change because trees and parent IDs change. Authors, committers, dates, messages,
and each retained historical crate tree are preserved. Commits unrelated to
the crate may be pruned. The original HMM history is not rewritten; removal
is a normal new commit in that repository. All 431 source commits were retained,
including 16 reflog-only commits protected under
`refs/archive/transformers/<original-commit>`. The 100 local branches and stash
were extracted; 17 app-specific tree snapshots stay only in the original
repository and backup. The complete map verifies parent order as well as trees
and commit metadata. Old auxiliary HMM worktrees remain untouched; create new
worktrees from this standalone repository for future transformer development.

The `migration/` directory records the extraction tool/version, source cut,
verification results, and original-to-extracted commit/ref maps. The local
`.git/migration/hmm-compression-original.bundle` additionally preserves the
unfiltered original branches for reproducing Git-bound historical tools. That
backup is not tracked, packaged, or published. It contains the original
monorepo history, so review it before sharing. The extracted repository does
not depend on the original checkout, this bundle, or shared Git object storage
for ordinary development. `history-verification.json` describes the extracted
cut before the subsequent migration-only documentation commit; its paths are
the locations used during verification. The checked-in verifier can audit a
fresh extraction against an original source mirror at that cut.

The original working directory is moved intact. Its ignored `models` symlink
still points to `/Volumes/bag of holding/models`; downloaded weights are not
copied or removed. Its ignored `target/` build tree and external cache links
are retained. Cargo may rebuild path-sensitive artifacts at the new location.
Unrelated HMM edits remain in the original checkout.

## Current development commands

From the standalone repository root:

```sh
cargo fmt --manifest-path Cargo.toml --check
cargo test --manifest-path Cargo.toml --locked --offline
python3 scripts/check_release.py --gate source --offline
```

The manifest, registry dependencies, nested fuzz package, and coverage helper
package are already standalone. Consumer path dependencies must point at this
checkout, for example `../transformers` from a sibling project. HMM itself has
no dependency on this crate.

## Historical reproduction

Acceptance JSON, producer hashes, embedded reference fixtures, and previously
qualified example sources remain byte-identical. Old absolute paths and commit
IDs in those records describe the original execution, not the current checkout.
The commit map translates historical source identity; it does not retroactively
qualify new executions or rewrite authenticated evidence. Some source comments
also retain historical commands; use README commands for current examples.

These five frozen producers derive the monorepo root from their original
directory depth and **must not be run directly from the flattened checkout**:

| Producer | Original producer commit |
| --- | --- |
| `scripts/generate_qwen35_08b_reference.py` | `06f39e77758a6eaf94b1583de60c531b94de872c` |
| `scripts/generate_qwen3_moe_reference.py` | Use the original commit recorded in the specific authenticated reference bundle. |
| `scripts/generate_qwen38_reference.py` | `b7aab93d8825f0b0458265fa11f5a2d68cb226bc` |
| `scripts/generate_flan_t5_small_reference.py` | `2dcf44c283dbc48377249b79c82f5483b912b2a7` |
| `scripts/generate_flan_t5_xxl_reference.py` | `163c9495b4a2f378edc1cd436cb3717aece81463` |

A wrong ancestor may itself be a Git repository and silently supply unrelated
provenance. These producers intentionally remain unchanged because their exact
bytes authenticate existing official-reference bundles. Ordinary tests and
reference consumers remain usable in the standalone checkout. Muse's Rust
acceptance harness now resolves either layout explicitly and still rejects
dirty or stale source.

To reproduce an old bundle, clone the unfiltered backup into a separate,
previously nonexistent directory and check out the original producer commit:

```sh
git clone /Users/nover/repos/transformers/.git/migration/hmm-compression-original.bundle \
  /private/tmp/transformers-original-layout
git -C /private/tmp/transformers-original-layout fetch \
  /Users/nover/repos/transformers/.git/migration/hmm-compression-original.bundle \
  'refs/archive/transformers/*:refs/archive/transformers/*'
git -C /private/tmp/transformers-original-layout switch --detach \
  06f39e77758a6eaf94b1583de60c531b94de872c
```

The example selects the original Qwen0.8B producer. For another bundle, select
its own authenticated original commit. Run its unchanged producer under
`crates/transformers/scripts/` from that original-layout checkout after
verifying HEAD, clean source, pinned source/helper hashes, Python package/wheel
identities, and checkpoint revision. Use a fresh external output directory;
never overwrite existing reference bundles. A source archive without Git, a
symlink to the flattened scripts, or substituted rewritten commit IDs does not
preserve this provenance contract.
