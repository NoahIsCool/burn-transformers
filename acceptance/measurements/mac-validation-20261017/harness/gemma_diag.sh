#!/bin/bash
# Gemma half-precision diagnostics (scratch, not committed). Waits for source gates, takes the shared
# heavy-model lock once, runs each step in its own process under /usr/bin/time -l with its own log
# (source identity first), refuses a dirty worktree before each step, and always releases the lock.
# Usage: gemma_diag.sh STEP...   (steps are the case labels below)
set -u
S=/private/tmp/claude-501/-Users-nover-repos-transformers/804c4b01-7ddc-4f18-a02c-672082928230/scratchpad
WT=/Users/nover/repos/transformers-worktrees/tfm-polish-half
LOCK=/Users/nover/repos/transformers-worktrees/.heavy-model.lock
G=$S/mac-validation/gemma/diagnostics
D=$G/data
MODEL=/Volumes/T9/models/google/gemma-3-4b-it
R=/Volumes/T9/transformers-references
TRUTH=$R/gemma-3-4b-it-093f9f38
TWIN=$R/gemma-3-4b-it-093f9f38-bf16
LTRUTH=$R/gemma-3-4b-it-093f9f38-long
LTWIN=$R/gemma-3-4b-it-093f9f38-long-bf16
REL="/Volumes/bag of holding/cargo-targets/tfm-polish-half-gates/release"
DUMP="$REL/burn-dump-gemma"
DUMPP="$REL/burn-dump-gemma-patched"
GATE=$S/mac-validation/gemma/bin/gemma_artifacts-53f167b
GATEP=$S/mac-validation/gemma/bin/gemma_artifacts-patched-gelu
P0=2,818,5279,529,7001,563
P1=2,2063,10779,78113,236769,236749,1473,107,140,12234,13293,506,538,236772,594,123466,1548,29738,107
P2=2,21313,835,2939,106703,41193,6658,5149,1547
GREEDY=9079,236761,108,50429,563,496,4185,3988,573,1610,236764,8013,236764,228183,532,6540,236761,1030,563,2033,531,1551,9639,70836,236764,2440,506,94648,25822,236764,506,132440
mkdir -p "$D"
while pgrep -f '[f]ull_gate.sh' >/dev/null; do sleep 60; done
until mkdir "$LOCK" 2>/dev/null; do echo "$(date -u +%FT%TZ) lock held; waiting"; sleep 60; done
trap 'rmdir "$LOCK"' EXIT
echo "$(date -u +%FT%TZ) lock acquired"

# run LOG [ENV=VALUE...] -- COMMAND...
run() {
  local log="$1"; shift
  local envs=()
  while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  local dirty; dirty=$(git -C "$WT" status --porcelain | wc -l | tr -d ' ')
  {
    echo "== source $(git -C "$WT" rev-parse HEAD) dirty=$dirty"
    echo "== start $(date -u +%FT%TZ) load $(sysctl -n vm.loadavg)"
    for e in ${envs[@]+"${envs[@]}"}; do echo "== env $e"; done
    printf '== cmd'; printf ' %q' "$@"; echo
  } > "$log"
  if [ "$dirty" != 0 ]; then echo "== refusing: dirty tree" >> "$log"; return 3; fi
  local start; start=$(date +%s)
  env -u TRANSFORMERS_GEMMA_DTYPE ${envs[@]+"${envs[@]}"} /usr/bin/time -l "$@" >> "$log" 2>&1
  local status=$?
  echo "== exit=$status wall=$(( $(date +%s) - start ))s" >> "$log"
  echo "$(date -u +%FT%TZ) $(basename "$log") exit=$status"
}

GENV=(TRANSFORMERS_GEMMA_DIR=$MODEL TRANSFORMERS_GEMMA_REFERENCE_DIR=$TRUTH TRANSFORMERS_GEMMA_BF16_REFERENCE_DIR=$TWIN
      TRANSFORMERS_GEMMA_LONG_REFERENCE_DIR=$LTRUTH TRANSFORMERS_GEMMA_LONG_BF16_REFERENCE_DIR=$LTWIN)

for step in "$@"; do
  case "$step" in
    official-variants)
      run "$G/official-variants.log" -- python3 -I "$S/gemma_official_variants.py" "$MODEL" "$TRUTH" "$TWIN" "$D/official" \
        cpu-float32-eager cpu-bfloat16-eager cpu-bfloat16-eager-f32gelu cpu-bfloat16-eager-burngelu \
        cpu-bfloat16-sdpa mps-bfloat16-eager mps-bfloat16-sdpa mps-bfloat16-eager-burngelu ;;
    official-f16)
      run "$G/official-f16-variants.log" -- python3 -I "$S/gemma_official_variants.py" "$MODEL" "$TRUTH" "$TWIN" "$D/official-f16" \
        cpu-float16-eager mps-float16-eager ;;
    official-decode)
      run "$G/official-decode.log" -- python3 -I "$S/official_decode.py" "$MODEL" "$TRUTH" "$D/official-decode" \
        cpu-float32-eager cpu-bfloat16-eager cpu-bfloat16-sdpa mps-bfloat16-eager mps-bfloat16-sdpa ;;
    burn-dumps)
      for cfg in "flex bf16 fused $DUMP" "flex bf16 eager $DUMP" "flex bf16 fused-f32gelu $DUMPP" \
                 "flex f32 fused $DUMP" "metal f16 fused $DUMP" "flex f16 fused $DUMP" "metal f16 fused-f32gelu $DUMPP"; do
        read -r backend dtype graph bin <<< "$cfg"
        attention=${graph%%-*}
        for i in 0 1 2; do
          eval "tokens=\$P$i"
          run "$G/burn-dump.$backend-$dtype-$graph.prompt$i.log" -- "$bin" "$backend" "$dtype" "$MODEL" "$tokens" \
            "$D/burn/$backend-$dtype-$graph/prompt$i" "$attention"
        done
      done ;;
    burn-decode)
      for cfg in "flex bf16 fused $DUMP" "flex bf16 eager $DUMP" "flex bf16 fused-f32gelu $DUMPP" "flex f32 fused $DUMP"; do
        read -r backend dtype graph bin <<< "$cfg"
        attention=${graph%%-*}
        run "$G/burn-decode.$backend-$dtype-$graph.log" -- "$bin" "$backend" "$dtype" "$MODEL" "$P0,$GREEDY" \
          "$D/burn-decode/$backend-$dtype-$graph" "$attention" 6
      done ;;
    patched-gates)
      for t in prompt_logits_match_the_reference cached_greedy_matches_the_reference batched_prefill_matches_the_reference \
               long_context_logits_cross_the_sliding_window; do
        run "$G/patched-f32gelu.bf16.flex.$t.log" "${GENV[@]}" TRANSFORMERS_GEMMA_DTYPE=bf16 -- \
          "$GATEP" --ignored --exact --nocapture --test-threads 1 "flex_$t"
      done
      run "$G/patched-f32gelu.bf16.flex.long_context_cached_decode_wraps_the_rings.log" "${GENV[@]}" TRANSFORMERS_GEMMA_DTYPE=bf16 \
        TRANSFORMERS_GEMMA_MIN_GREEDY=0 -- "$GATEP" --ignored --exact --nocapture --test-threads 1 flex_long_context_cached_decode_wraps_the_rings ;;
    *) echo "unknown step $step" ;;
  esac
done
echo "$(date -u +%FT%TZ) done"
