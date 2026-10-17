#!/bin/bash
# Decodes each bundle's longest prompt after a quarter prefill with burn-dump (Flex/BF16, fused
# attention) under the heavy-model lock, then approximates the gate's unreached decoded-prompt check.
set -u
S=/private/tmp/claude-501/-Users-nover-repos-transformers/804c4b01-7ddc-4f18-a02c-672082928230/scratchpad
G=$S/mac-validation/llama-family/diagnostics
R=/Volumes/T9/transformers-references
BD="/Volumes/bag of holding/cargo-targets/tfm-polish-half-gates/release/burn-dump"
for spec in "Qwen3-4B|/Volumes/T9/models/Qwen/Qwen3-4B|Qwen3-4B-531c80e2" \
            "SmolLM3-3B|/Volumes/T9/models/HuggingFaceTB/SmolLM3-3B|SmolLM3-3B-1fa6fa28"; do
  IFS='|' read -r name model truth <<< "$spec"
  read -r index tokens prefill < <(python3 -I "$S/longest_prompt.py" "$R/$truth")
  out="$G/data/$name-burn-longest-decode/flex-bf16-fused"
  bash "$S/heavy_diag.sh" "$G/$name.burn-longest-decode.flex-bf16-fused.log" "$BD" flex bf16 "$model" \
    "$tokens" "$out" fused "$prefill" || echo "FAIL $name"
  python3 -I "$S/judge_longest_decode.py" "$R/$truth" "$R/$truth-bf16" "$out" "$index" "$prefill" \
    | tee -a "$G/$name.burn-longest-decode.flex-bf16-fused.log"
done
