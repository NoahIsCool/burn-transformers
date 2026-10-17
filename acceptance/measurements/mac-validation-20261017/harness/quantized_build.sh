#!/bin/bash
# Builds the release gate binaries for the "quantized" validation group (scratch, not committed).
set -u
cd /Users/nover/repos/transformers-worktrees/tfm-polish-half || exit 2
export CARGO_TARGET_DIR="/Volumes/bag of holding/cargo-targets/tfm-polish-half-gates"
G=/private/tmp/claude-501/-Users-nover-repos-transformers/804c4b01-7ddc-4f18-a02c-672082928230/scratchpad/mac-validation/quantized
echo "== source $(git rev-parse HEAD) dirty=$(git status --porcelain | wc -l | tr -d ' ')"
echo "== rustc $(rustc -V) cargo $(cargo -V)"
df -h "/Volumes/bag of holding" | tail -1
b() { # label, cargo args...
  local label="$1"; shift
  echo "== build $label: cargo test --locked --release $* --no-run  ($(date -u +%FT%TZ))"
  cargo test --locked --release "$@" --no-run 2>&1 | grep -E "^(error|warning)|Executable|Finished" 
  echo "== build $label exit=${PIPESTATUS[0]} ($(date -u +%FT%TZ))"
}
b A --no-default-features --features std,kernel-tracing --test packed_metal_kernels
b B --features metal --lib --test packed_metal_kernels
b C --no-default-features --features std,tokenizers --test muse_glimmer_artifacts
b D --no-default-features --features std,metal,tokenizers,kernel-tracing --test muse_glimmer_artifacts --test qwen3_moe_artifacts
b E --no-default-features --features std,metal,tokenizers,chat-templates --test qwen3_moe_artifacts
echo "== source after $(git rev-parse HEAD) dirty=$(git status --porcelain | wc -l | tr -d ' ')"
