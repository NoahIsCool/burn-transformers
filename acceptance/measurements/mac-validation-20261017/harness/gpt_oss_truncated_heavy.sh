#!/bin/bash
# Heavy-run wrapper for the truncated gpt-oss-20b F32 gates (scratch, not committed).
# Waits for source gates, takes the shared heavy-model lock, runs the gates, and
# always releases the lock.
set -u
G=/private/tmp/claude-501/-Users-nover-repos-transformers/804c4b01-7ddc-4f18-a02c-672082928230/scratchpad/mac-validation/gpt-oss-truncated
LOCK=/Users/nover/repos/transformers-worktrees/.heavy-model.lock
BIN="/Volumes/bag of holding/cargo-targets/tfm-polish-half-gates/release/deps/gpt_oss_artifacts-98a177a8f6a4ce38"
while pgrep -f full_gate.sh >/dev/null; do sleep 60; done
until mkdir "$LOCK" 2>/dev/null; do sleep 60; done
trap 'rmdir "$LOCK"; echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) lock released" >> "$G/heavy.log"' EXIT
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) lock acquired" >> "$G/heavy.log"
python3 -I /private/tmp/claude-501/-Users-nover-repos-transformers/804c4b01-7ddc-4f18-a02c-672082928230/scratchpad/run_gpt_oss_truncated_gates.py "$BIN" "$G" "$@"
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) runner exit=$?" >> "$G/heavy.log"
