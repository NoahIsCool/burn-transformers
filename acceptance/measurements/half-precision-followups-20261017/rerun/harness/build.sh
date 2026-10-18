#!/bin/bash
# Scratch (rerun at 8181605): builds the release gate binaries in the worktree checkout itself (not committed).
set -u
R=/private/tmp/claude-501/-Users-nover-repos-transformers/804c4b01-7ddc-4f18-a02c-672082928230/scratchpad/followups/rerun/8181605
cd /Users/nover/repos/transformers-worktrees/tfm-half-followups2 || exit 2
export CARGO_TARGET_DIR="/Volumes/bag of holding/cargo-targets/tfm-half-followups2-gates"
L=$R/logs/build.log
{
echo "== source $(git rev-parse HEAD) dirty=$(git status --porcelain | wc -l | tr -d ' ')"
echo "== checkout $(pwd) CARGO_TARGET_DIR=$CARGO_TARGET_DIR"
echo "== start $(date -u +%Y-%m-%dT%H:%M:%S+00:00)"
echo "== cmd cargo test --locked --release --features metal,tokenizers --test llama_family_artifacts --test gemma_artifacts --no-run"
} > $L
cargo test --locked --release --features metal,tokenizers --test llama_family_artifacts --test gemma_artifacts --no-run >> $L 2>&1
status=$?
{
echo "== exit=$status end $(date -u +%Y-%m-%dT%H:%M:%S+00:00)"
echo "== source after build $(git rev-parse HEAD) dirty=$(git status --porcelain | wc -l | tr -d ' ')"
} >> $L
exit $status
