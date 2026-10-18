#!/bin/bash
# Scratch (half2 S9): builds the release gate binaries from the clean export of the integrated head.
set -u
G=/private/tmp/claude-501/-Users-nover-repos-transformers/804c4b01-7ddc-4f18-a02c-672082928230/scratchpad/half2/gates
export CARGO_TARGET_DIR="/Volumes/bag of holding/cargo-targets/tfm-half-followups2-gates"
cd $G/src || exit 2
b() { local label="$1"; shift; echo "== build $label: cargo test --locked --release $* --no-run" >> $G/logs/build.log; cargo test --locked --release "$@" --no-run >> $G/logs/build.log 2>&1; echo "== $label exit=$?" >> $G/logs/build.log; }
b A --features metal,tokenizers --test llama_family_artifacts --test gemma_artifacts
b B --no-default-features --features std,metal,tokenizers,chat-templates --test local_artifacts --test qwen35_benchmarks --test muse_glimmer_artifacts --test qwen3_moe_artifacts --test deepseek_v2_artifacts --test nemotron_h_artifacts
b C --no-default-features --features std,metal,tokenizers,kernel-tracing --test muse_glimmer_artifacts --test qwen3_moe_artifacts
b D --no-default-features --features std,metal,tokenizers --test muse_glimmer_artifacts
echo "== done" >> $G/logs/build.log
