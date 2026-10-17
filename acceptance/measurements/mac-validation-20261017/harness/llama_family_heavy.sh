#!/bin/bash
# Heavy-run wrapper for the Llama-family gates (scratch, not committed).
# Waits for source gates, takes the shared heavy-model lock, refuses a dirty tree,
# runs the per-test driver, and always releases the lock.
set -u
WT=/Users/nover/repos/transformers-worktrees/tfm-polish-half
LOCK=/Users/nover/repos/transformers-worktrees/.heavy-model.lock
S=/private/tmp/claude-501/-Users-nover-repos-transformers/804c4b01-7ddc-4f18-a02c-672082928230/scratchpad
OUT="$S/mac-validation/llama-family"
BIN="$1"; shift
while pgrep -f full_gate.sh >/dev/null; do sleep 60; done
until mkdir "$LOCK" 2>/dev/null; do echo "$(date -u +%FT%TZ) waiting for $LOCK" >> "$OUT/summary.log"; sleep 60; done
trap 'rmdir "$LOCK"' EXIT
cd "$WT" || exit 2
dirty=$(git status --porcelain | wc -l | tr -d ' ')
echo "== source $(git rev-parse HEAD) dirty=$dirty" >> "$OUT/summary.log"
echo "== $(date -u +%FT%TZ) lock acquired; binary $BIN" >> "$OUT/summary.log"
if [ "$dirty" != 0 ]; then echo "== refusing: dirty tree" >> "$OUT/summary.log"; exit 3; fi
python3 -I "$S/run_llama_family_gates.py" "$BIN" "$OUT" "$@"
status=$?
echo "== $(date -u +%FT%TZ) driver exit=$status; releasing lock" >> "$OUT/summary.log"
exit $status
