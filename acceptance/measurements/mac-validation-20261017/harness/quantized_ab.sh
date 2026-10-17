#!/bin/bash
# Alternating same-session A/B of the packed Muse smokes: control 9a9c730 (c15f24e's parent,
# verified git-archive export, separate target dir) against the head worktree (scratch, not committed).
set -u
S=/private/tmp/claude-501/-Users-nover-repos-transformers/804c4b01-7ddc-4f18-a02c-672082928230/scratchpad
G=$S/mac-validation/quantized
O=$G/ab-9a9c730
ctl() { GATE_EXPORT_DIR=$S/gptoss-ab/src-9a9c730 GATE_EXPORT_SHA=9a9c7308cf233f90ccbee80c0156efdf4cceea27 GATE_LOG_SUFFIX=".$1-control" python3 -I $S/run_quantized_gates.py $O/binaries-control.json $O "$2"; }
hd() { GATE_LOG_SUFFIX=".$1-head" python3 -I $S/run_quantized_gates.py $G/binaries.json $O "$2"; }
ctl r1 muse-q8-smoke; hd r1 muse-q8-smoke; ctl r2 muse-q8-smoke; hd r2 muse-q8-smoke
ctl r1 muse-q4-smoke; hd r1 muse-q4-smoke
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) ab exit" >> $O/summary.log
