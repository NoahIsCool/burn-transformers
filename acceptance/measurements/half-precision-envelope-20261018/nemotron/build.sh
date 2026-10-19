#!/bin/bash
# Scratch (half2 S9 Nemotron diagnosis): builds the base and single-site-revert Nemotron gate binaries at low priority (not committed).
set -u
H=/private/tmp/claude-501/-Users-nover-repos-transformers/804c4b01-7ddc-4f18-a02c-672082928230/scratchpad/half2
L=$H/nemotron/logs/build.log
b() { local label="$1" dir="$2" target="$3"; shift 3
  echo "== build $label in $dir (CARGO_TARGET_DIR=$target): cargo $*" >> $L
  (cd "$dir" && CARGO_TARGET_DIR="$target" nice -n 19 cargo "$@" >> $L 2>&1); echo "== $label exit=$?" >> $L; }
b N0 $H/bench-base/src "/Volumes/bag of holding/cargo-targets/tfm-half-followups2-bench-base" test --locked --release --no-default-features --features std,metal,tokenizers,chat-templates --test nemotron_h_artifacts --no-run -j 4
b N1 $H/nemotron/isolate/src "/Volumes/bag of holding/cargo-targets/tfm-half-followups2-nemotron-isolate" test --locked --release --no-default-features --features std,metal,tokenizers,chat-templates --test nemotron_h_artifacts --no-run -j 4
echo "== done" >> $L
