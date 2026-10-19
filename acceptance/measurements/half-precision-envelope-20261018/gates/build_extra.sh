#!/bin/bash
# Scratch (half2 S9): builds the extra binaries at low priority (not committed):
#  E  qwen35_benchmarks at the base 7bd057e (git archive export), for the controlled cold-token rerun;
#  M  muse_glimmer_artifacts at f9f78b5 from a clean Git clone (the official gate records Git provenance);
#  T  the per-block diagnosis tool against the gate export.
set -u
H=/private/tmp/claude-501/-Users-nover-repos-transformers/804c4b01-7ddc-4f18-a02c-672082928230/scratchpad/half2
L=$H/gates/logs/build-extra.log
b() { local label="$1" dir="$2" target="$3"; shift 3
  echo "== build $label in $dir (CARGO_TARGET_DIR=$target): cargo $*" >> $L
  (cd "$dir" && CARGO_TARGET_DIR="$target" nice -n 19 cargo "$@" >> $L 2>&1); echo "== $label exit=$?" >> $L; }
b E $H/bench-base/src "/Volumes/bag of holding/cargo-targets/tfm-half-followups2-bench-base" test --locked --release --no-default-features --features std,metal,tokenizers,chat-templates --test qwen35_benchmarks --no-run -j 4
b M $H/gates/muse-git "/Volumes/bag of holding/cargo-targets/tfm-half-followups2-gates-git" test --locked --release --no-default-features --features std,metal,tokenizers --test muse_glimmer_artifacts --no-run -j 4
b T $H/diag/burn-tool "/Volumes/bag of holding/cargo-targets/tfm-half-followups2-diag" build --release --offline -j 4
echo "== done" >> $L
