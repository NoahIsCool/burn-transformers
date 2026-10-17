#!/bin/bash
# Runs the official BF16-vs-F32/F64 calibration tests (bundles only; no checkpoint, no Burn model).
# Scratch helper, not committed. Usage: run_calibration.sh LOG
set -u
WT=/Users/nover/repos/transformers-worktrees/tfm-polish-half
LOG="$1"
R=/Volumes/T9/transformers-references
export CARGO_TARGET_DIR="/Volumes/bag of holding/cargo-targets/tfm-polish-half"
cd "$WT" || exit 2
dirty=$(git status --porcelain | wc -l | tr -d ' ')
echo "== source $(git rev-parse HEAD) dirty=$dirty" >> "$LOG"
[ "$dirty" = 0 ] || { echo "== refusing: dirty tree" >> "$LOG"; exit 3; }
echo "== $(date -u +%Y-%m-%dT%H:%M:%SZ) $(rustc --version)" >> "$LOG"
bin() { cargo test --locked --test "$1" --no-run 2>&1 | grep Executable | sed 's/.*(\(.*\))/\1/'; }
L=$(bin llama_family_artifacts); G=$(bin gemma_artifacts)
echo "== binaries $(shasum -a 256 "$L" | cut -c1-64) $L; $(shasum -a 256 "$G" | cut -c1-64) $G (debug test profile, default features)" >> "$LOG"
for n in SmolLM-135M-1d461723 Qwen3-4B-531c80e2 SmolLM3-3B-1fa6fa28 Qwen2-7B-453ed157-f64 Llama-3.1-8B-d04e592b; do
  echo "== calibration $n: TRANSFORMERS_LLAMA_FAMILY_REFERENCE_DIR=$R/$n TRANSFORMERS_LLAMA_FAMILY_BF16_REFERENCE_DIR=$R/$n-bf16 llama_family_artifacts --ignored --exact official_bf16_reference_calibrates_the_half_precision_bounds --nocapture" >> "$LOG"
  TRANSFORMERS_LLAMA_FAMILY_REFERENCE_DIR=$R/$n TRANSFORMERS_LLAMA_FAMILY_BF16_REFERENCE_DIR=$R/$n-bf16 \
    /usr/bin/time -l "$L" --ignored --exact official_bf16_reference_calibrates_the_half_precision_bounds --nocapture --test-threads 1 >> "$LOG" 2>&1
  echo "== exit=$?" >> "$LOG"
done
echo "== calibration gemma-3-4b-it-093f9f38 and -long: TRANSFORMERS_GEMMA_REFERENCE_DIR=$R/gemma-3-4b-it-093f9f38 TRANSFORMERS_GEMMA_BF16_REFERENCE_DIR=$R/gemma-3-4b-it-093f9f38-bf16 TRANSFORMERS_GEMMA_LONG_REFERENCE_DIR=$R/gemma-3-4b-it-093f9f38-long TRANSFORMERS_GEMMA_LONG_BF16_REFERENCE_DIR=$R/gemma-3-4b-it-093f9f38-long-bf16 gemma_artifacts --ignored --exact official_bf16_references_calibrate_the_half_precision_bounds --nocapture" >> "$LOG"
TRANSFORMERS_GEMMA_REFERENCE_DIR=$R/gemma-3-4b-it-093f9f38 TRANSFORMERS_GEMMA_BF16_REFERENCE_DIR=$R/gemma-3-4b-it-093f9f38-bf16 \
TRANSFORMERS_GEMMA_LONG_REFERENCE_DIR=$R/gemma-3-4b-it-093f9f38-long TRANSFORMERS_GEMMA_LONG_BF16_REFERENCE_DIR=$R/gemma-3-4b-it-093f9f38-long-bf16 \
  /usr/bin/time -l "$G" --ignored --exact official_bf16_references_calibrate_the_half_precision_bounds --nocapture --test-threads 1 >> "$LOG" 2>&1
echo "== exit=$?" >> "$LOG"
