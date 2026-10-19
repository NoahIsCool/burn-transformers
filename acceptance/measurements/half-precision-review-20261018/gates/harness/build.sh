#!/bin/bash
# Review round: builds the release gate and library test binaries in the worktree checkout itself, at the round's code head.
set -u
G=/private/tmp/claude-501/-Users-nover-repos-transformers/804c4b01-7ddc-4f18-a02c-672082928230/scratchpad/review3/gates
cd /Users/nover/repos/transformers-worktrees/tfm-half-followups2 || exit 2
export CARGO_TARGET_DIR="/Volumes/bag of holding/cargo-targets/tfm-half-followups2-gates"
L=$G/logs/build.log
CMD="cargo test --locked --release --features metal,tokenizers --test gemma_artifacts --lib --no-run"
{
echo "== source $(git rev-parse HEAD) dirty=$(git status --porcelain | wc -l | tr -d ' ')"
echo "== checkout $(pwd) CARGO_TARGET_DIR=$CARGO_TARGET_DIR"
echo "== start $(date -u +%Y-%m-%dT%H:%M:%S+00:00)"
echo "== cmd $CMD"
} > $L
$CMD >> $L 2>&1
status=$?
{
echo "== exit=$status end $(date -u +%Y-%m-%dT%H:%M:%S+00:00)"
for b in $(grep -o 'release/deps/[a-z_]*-[0-9a-f]*' $L | sort -u); do echo "== binary $CARGO_TARGET_DIR/$b sha256 $(shasum -a 256 "$CARGO_TARGET_DIR/$b" | cut -d' ' -f1)"; done
echo "== source after build $(git rev-parse HEAD) dirty=$(git status --porcelain | wc -l | tr -d ' ')"
} >> $L
exit $status
