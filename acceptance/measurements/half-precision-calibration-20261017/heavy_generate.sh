#!/bin/bash
# Heavy-run wrapper for the BF16 reference generation (scratch, not committed).
# Waits for source gates, takes the shared heavy-model lock, checks the tree is clean,
# logs the source identity, runs the driver, and always releases the lock.
set -u
WT=/Users/nover/repos/transformers-worktrees/tfm-polish-half
LOCK=/Users/nover/repos/transformers-worktrees/.heavy-model.lock
LOG="$1"; shift
while pgrep -f full_gate.sh >/dev/null; do sleep 60; done
until mkdir "$LOCK" 2>/dev/null; do sleep 60; done
trap 'rmdir "$LOCK"' EXIT
cd "$WT" || exit 2
dirty=$(git status --porcelain | wc -l | tr -d ' ')
echo "== source $(git rev-parse HEAD) dirty=$dirty" >> "$LOG"
if [ "$dirty" != 0 ]; then echo "== refusing: dirty tree" >> "$LOG"; exit 3; fi
echo "== $(date -u +%Y-%m-%dT%H:%M:%SZ) $(python3 -I -c 'import sys,torch,transformers;print("python",sys.version.split()[0],"torch",torch.__version__,"transformers",transformers.__version__)')" >> "$LOG"
python3 -I /private/tmp/claude-501/-Users-nover-repos-transformers/804c4b01-7ddc-4f18-a02c-672082928230/scratchpad/generate_bf16_bundles.py "$WT" "$LOG" "$@"
status=$?
echo "== driver exit=$status" >> "$LOG"
exit $status
