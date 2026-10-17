#!/bin/bash
# Real-checkpoint gates for the final head (validation helper, not committed).
# Usage: final_gates.sh <checkout> ; each test runs in its own process to bound memory.
set -u
cd "$1" || exit 2
export CARGO_TARGET_DIR="/Volumes/bag of holding/cargo-targets/tfm-polish-gates"
M="/Volumes/bag of holding/models"; A="$M/transformers-acceptance"; R=/Volumes/T9/transformers-references
echo "== source $(git rev-parse HEAD) dirty=$(git status --porcelain | wc -l | tr -d ' ')"
build() { cargo test --locked --release "$@" --no-run 2>&1 | grep -E "^error|Executable" | sed 's/.*(\(.*\))/\1/'; }
run_each() { # binary tests...
  local bin="$1"; shift
  for t in "$@"; do
    /usr/bin/time -l "$bin" --ignored --exact --nocapture --test-threads 1 "$t" 2>&1 \
      | grep -E "^test |test result|max \||greedy|cosine|failures|panicked|peak memory|tok/s|matches|mismatch" | head -20
  done
}
echo "== FLAN-T5-small Flex + Metal"
export TRANSFORMERS_FLAN_T5_SMALL_DIR="$M/google/flan-t5-small-0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab" TRANSFORMERS_FLAN_T5_SMALL_REFERENCE_DIR="$A/flan-t5-small-0fc9ddf7-reference.bundle"
B=$(build --no-default-features --features std,flex,metal,tokenizers --test flan_t5_small | tail -1)
run_each "$B" flan_t5_small_flex_matches_official_reference flan_t5_small_metal_matches_official_reference
echo "== Qwen3.5-0.8B Metal/F16"
export TRANSFORMERS_QWEN35_DIR="$M/qwen3.5-0.8b-2fc06364715b967f1860aea9cf38778875588b17" TRANSFORMERS_QWEN35_REFERENCE_DIR="$A"
B=$(build --no-default-features --features std,metal,tokenizers,chat-templates --test local_artifacts | tail -1)
run_each "$B" loads_pinned_qwen35_weights_and_runs_metal_inference
B=$(build --features metal,tokenizers --test llama_family_artifacts | tail -1)
LT="flex_prompt_logits_match_the_reference flex_cached_greedy_matches_the_reference flex_batched_prefill_matches_the_reference metal::metal_prompt_logits_match_the_reference metal::metal_cached_greedy_matches_the_reference metal::metal_batched_prefill_matches_the_reference"
for pair in "SmolLM-135M-1d461723:$M/HuggingFaceTB/SmolLM-135M-1d461723eec654e65efdc40cf49301c89c0c92f4" \
            "Qwen3-4B-531c80e2:/Volumes/T9/models/Qwen/Qwen3-4B" \
            "SmolLM3-3B-1fa6fa28:/Volumes/T9/models/HuggingFaceTB/SmolLM3-3B" \
            "Qwen2-7B-453ed157-f64:/Volumes/T9/models/Qwen/Qwen2-7B" \
            "Llama-3.1-8B-d04e592b:/Volumes/T9/models/meta-llama/Llama-3.1-8B"; do
  name=${pair%%:*}; dir=${pair#*:}
  echo "== Llama family $name"
  [ -d "$R/$name" ] || { echo "SKIP: no bundle $R/$name"; continue; }
  tol=1e-3; case "$name" in Qwen2-7B*) tol=2.5e-3;; esac
  TRANSFORMERS_LLAMA_FAMILY_TOLERANCE=$tol TRANSFORMERS_LLAMA_FAMILY_DIR="$dir" TRANSFORMERS_LLAMA_FAMILY_REFERENCE_DIR="$R/$name" run_each "$B" $LT
done
echo "== Gemma 3 4B it"
B=$(build --features metal,tokenizers --test gemma_artifacts | tail -1)
TRANSFORMERS_GEMMA_DIR=/Volumes/T9/models/google/gemma-3-4b-it TRANSFORMERS_GEMMA_REFERENCE_DIR="$R/gemma-3-4b-it-093f9f38" TRANSFORMERS_GEMMA_LONG_REFERENCE_DIR="$R/gemma-3-4b-it-093f9f38-long" \
  run_each "$B" flex_prompt_logits_match_the_reference flex_cached_greedy_matches_the_reference flex_batched_prefill_matches_the_reference flex_long_context_logits_cross_the_sliding_window flex_long_context_cached_decode_wraps_the_rings metal::metal_prompt_logits_match_the_reference metal::metal_cached_greedy_matches_the_reference metal::metal_batched_prefill_matches_the_reference metal::metal_long_context_logits_cross_the_sliding_window metal::metal_long_context_cached_decode_wraps_the_rings
echo "== gpt-oss-20b truncated references"
B=$(build --features metal,tokenizers --test gpt_oss_artifacts | tail -1)
for bundle in gpt-oss-20b-6cee5e81-1layer-f32-reference.bundle gpt-oss-20b-6cee5e81-4layer-f32-reference.bundle; do
  echo "-- $bundle"
  TRANSFORMERS_GPT_OSS_DIR=/Volumes/T9/models/openai/gpt-oss-20b TRANSFORMERS_GPT_OSS_REFERENCE_DIR="$A/$bundle" \
    run_each "$B" flex_prompt_logits_match_the_reference flex_cached_greedy_matches_the_reference flex_batched_prefill_matches_the_reference flex_cached_long_prompt_matches_the_reference metal::metal_prompt_logits_match_the_reference metal::metal_cached_greedy_matches_the_reference metal::metal_batched_prefill_matches_the_reference metal::metal_cached_long_prompt_matches_the_reference
done
echo "== DeepSeek-V2-Lite metadata + Metal F16 smoke"
B=$(build --no-default-features --features std,metal,tokenizers,chat-templates --test deepseek_v2_artifacts | tail -1)
TRANSFORMERS_DEEPSEEK_V2_DIR="$M/deepseek-ai/DeepSeek-V2-Lite-604d5664dddd88a0433dbae533b7fe9472482de0" \
  run_each "$B" validates_pinned_deepseek_v2_lite_config_headers_plan_memory_and_tokenizer loads_pinned_deepseek_v2_lite_f16_and_runs_metal_smoke
echo "== done"
