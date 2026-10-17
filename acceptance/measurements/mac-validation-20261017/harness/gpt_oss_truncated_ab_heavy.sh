#!/bin/bash
# Heavy-run wrapper for the A/B rerun of the truncated gpt-oss-20b F32 gates at
# c15f24e's parent 9a9c730, from a verified git-archive export (scratch, not committed).
set -u
S=/private/tmp/claude-501/-Users-nover-repos-transformers/804c4b01-7ddc-4f18-a02c-672082928230/scratchpad
G=$S/mac-validation/gpt-oss-truncated/ab-9a9c730
LOCK=/Users/nover/repos/transformers-worktrees/.heavy-model.lock
BIN="/Volumes/bag of holding/cargo-targets/tfm-polish-half-gates-ab/release/deps/gpt_oss_artifacts-98a177a8f6a4ce38"
while pgrep -f full_gate.sh >/dev/null; do sleep 60; done
until mkdir "$LOCK" 2>/dev/null; do sleep 60; done
trap 'rmdir "$LOCK"; echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) lock released" >> "$G/heavy.log"' EXIT
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) lock acquired" >> "$G/heavy.log"
GATE_EXPORT_DIR=$S/gptoss-ab/src-9a9c730 GATE_EXPORT_SHA=9a9c7308cf233f90ccbee80c0156efdf4cceea27 \
  python3 -I $S/run_gpt_oss_truncated_gates.py "$BIN" "$G" "$@"
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) runner exit=$?" >> "$G/heavy.log"
