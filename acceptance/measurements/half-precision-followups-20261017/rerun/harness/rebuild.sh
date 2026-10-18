#!/bin/bash
# Scratch (rerun at 8181605): forces a rebuild of the gate binaries from the worktree checkout (not committed).
# The no-op build (logs/build-0-noop.log) found the f9f78b5 export's binaries fresh, so this updates
# src/lib.rs's modification time only (no content change; git status stays clean) and rebuilds.
set -u
R=/private/tmp/claude-501/-Users-nover-repos-transformers/804c4b01-7ddc-4f18-a02c-672082928230/scratchpad/followups/rerun/8181605
cd /Users/nover/repos/transformers-worktrees/tfm-half-followups2 || exit 2
export CARGO_TARGET_DIR="/Volumes/bag of holding/cargo-targets/tfm-half-followups2-gates"
L=$R/logs/build.log
{
echo "== source $(git rev-parse HEAD) dirty=$(git status --porcelain | wc -l | tr -d ' ')"
echo "== checkout $(pwd) CARGO_TARGET_DIR=$CARGO_TARGET_DIR"
echo "== start $(date -u +%Y-%m-%dT%H:%M:%S+00:00)"
echo "== src/lib.rs sha256 before touch $(shasum -a 256 src/lib.rs | cut -d' ' -f1)"
touch src/lib.rs
echo "== touched src/lib.rs (mtime only); sha256 after $(shasum -a 256 src/lib.rs | cut -d' ' -f1); git blob $(git hash-object src/lib.rs) HEAD blob $(git rev-parse HEAD:src/lib.rs)"
echo "== cmd cargo test --locked --release --features metal,tokenizers --test llama_family_artifacts --test gemma_artifacts --no-run"
} > $L
cargo test --locked --release --features metal,tokenizers --test llama_family_artifacts --test gemma_artifacts --no-run >> $L 2>&1
status=$?
{
echo "== exit=$status end $(date -u +%Y-%m-%dT%H:%M:%S+00:00)"
for b in $(grep -o 'release/deps/[a-z_]*-[0-9a-f]*' $L | sort -u); do echo "== binary $CARGO_TARGET_DIR/$b sha256 $(shasum -a 256 "$CARGO_TARGET_DIR/$b" | cut -d' ' -f1)"; done
echo "== source after build $(git rev-parse HEAD) dirty=$(git status --porcelain | wc -l | tr -d ' ')"
} >> $L
exit $status
