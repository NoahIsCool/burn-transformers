#!/bin/bash
# Runs the post-load footprint probe for each model under both load orders, alternating, under the lock.
set -u
S=/private/tmp/claude-501/-Users-nover-repos-transformers/66ab4e94-8aa9-472d-b244-dc221f41f2e6/scratchpad
LOCK=/Users/nover/repos/transformers-worktrees/.heavy-model.lock
B="/Volumes/bag of holding/models"
OUT=$S/probe/results.txt
probe() {
  local model=$1; shift
  for order in source target; do
    until mkdir "$LOCK" 2>/dev/null; do sleep 15; done
    line=$(cd /Users/nover/repos/transformers-worktrees/tfm-qwen38-bisect && env "$@" python3 $S/footprint_trace.py $S/probe/trace-$model-$order.tsv -- $S/probe/probe-$order-order probe_post_load_footprint --ignored --exact --nocapture --test-threads 1 2>&1 | grep -E "probe loaded|peak=" | tr '\n' ' ')
    rmdir "$LOCK"
    plateau=$(python3 -c "
import sys
rows=[tuple(map(float,l.split())) for l in open('$S/probe/trace-$model-$order.tsv')]
end=rows[-1][0]; tail=[v for t,v in rows if end-2.5<=t<=end-0.5]
print(f'{max(tail)/1e9:.3f}' if tail else 'na')")
    echo "$model $order plateau_GB=$plateau | $line" >> $OUT
  done
}
probe qwen3.8 TRANSFORMERS_PROBE_DIR="$B/Qwen/Qwen3.8-27B-1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
probe qwen3-moe TRANSFORMERS_PROBE_DIR="$B/Qwen/Qwen3-30B-A3B-ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
probe muse TRANSFORMERS_PROBE_DIR="$B/meta-models/Muse-Glimmer-30B-a4e59da52a7bc87ae7251dd5545c0dd437c44b68" TRANSFORMERS_PROBE_REPOSITORY=meta-models/Muse-Glimmer-30B TRANSFORMERS_PROBE_REVISION=a4e59da52a7bc87ae7251dd5545c0dd437c44b68
probe nemotron TRANSFORMERS_PROBE_DIR="$B/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16-a9904d24bcc1d289a1950fa9d2b978c47cf903b9" TRANSFORMERS_PROBE_PHYSICAL=1 TRANSFORMERS_PROBE_REPOSITORY=nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 TRANSFORMERS_PROBE_REVISION=a9904d24bcc1d289a1950fa9d2b978c47cf903b9
probe flan-t5-xxl TRANSFORMERS_PROBE_DIR="$B/google/flan-t5-xxl" TRANSFORMERS_PROBE_KIND=seq2seq TRANSFORMERS_PROBE_DTYPE=preserve TRANSFORMERS_PROBE_REPOSITORY=google/flan-t5-xxl TRANSFORMERS_PROBE_REVISION=ae7c9136adc7555eeccc78cdd960dfd60fb346ce
echo probes-done >> $OUT
