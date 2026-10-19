#!/bin/bash
# Scratch (half2 S9): runs the Nemotron attribution plan after the heavy chain ends (not committed).
H=/private/tmp/claude-501/-Users-nover-repos-transformers/804c4b01-7ddc-4f18-a02c-672082928230/scratchpad/half2
until grep -q "chain done" $H/chain.log; do sleep 30; done
echo "$(date -u +%FT%TZ) start nemotron attribution" >> $H/chain.log
python3 -I $H/silu-cost/run_heavy.py $H/nemotron/plan.json
echo "$(date -u +%FT%TZ) end nemotron attribution status=$?" >> $H/chain.log
