#!/bin/bash
# Scratch (half2 S9): runs the remaining heavy work one item at a time after the large-checkpoint driver ends (not committed).
# Every step takes the shared heavy-model lock itself (run_heavy.py, run_extra.py, run_gates.py).
set -u
H=/private/tmp/claude-501/-Users-nover-repos-transformers/804c4b01-7ddc-4f18-a02c-672082928230/scratchpad/half2
G=$H/gates
L=$H/chain.log
log() { echo "$(date -u +%FT%TZ) $*" >> $L; }
for v in "/Volumes/bag of holding" /Volumes/T9; do mount | grep -q " on $v " || { log "STOP: $v is not mounted"; exit 5; }; done
log "waiting for driver B"
until grep -q "done qwen35 qwen35-bench muse deepseek qwen3moe nemotron qwen38$" $G/logs/driver.log; do sleep 30; done
log "driver B done"
step() { log "start $1"; shift; "$@"; log "end status=$?"; }
step control python3 -I $H/silu-cost/run_heavy.py $G/control/plan.json
step memory python3 -I $H/silu-cost/run_heavy.py $G/memory/plan.json
step muse-official bash -c "cd $G && python3 -I run_extra.py binaries-extra.json muse-official"
step probe python3 -I $H/silu-cost/run_heavy.py $H/probe-evidence/plan.json
step diag-burn python3 -I $H/silu-cost/run_heavy.py $H/diag/plan-burn.json
step diag-torch python3 -I $H/silu-cost/run_heavy.py $H/diag/plan-torch.json
step qwen38-quantized bash -c "cd $G && python3 -I run_gates.py binaries-all.json qwen38-quantized"
step silu-cost python3 -I $H/silu-cost/run_heavy.py $H/silu-cost/plan.json
log "chain done"
