#!/bin/bash
# Scratch (half2 S9): runs the reworked Burn per-block dumps after the Nemotron attribution ends (not committed).
H=/private/tmp/claude-501/-Users-nover-repos-transformers/804c4b01-7ddc-4f18-a02c-672082928230/scratchpad/half2
until grep -q "end nemotron attribution" $H/chain.log; do sleep 30; done
echo "$(date -u +%FT%TZ) start diag-burn2" >> $H/chain.log
python3 -I $H/silu-cost/run_heavy.py $H/diag/plan-burn2.json
echo "$(date -u +%FT%TZ) end diag-burn2 status=$?" >> $H/chain.log
