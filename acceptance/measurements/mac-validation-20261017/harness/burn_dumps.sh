#!/bin/bash
# Runs burn-dump for every prompt of a bundle in several backend/dtype/attention configurations,
# one process at a time under the heavy-model lock (scratch, not committed).
# Usage: burn_dumps.sh NAME MODEL_DIR BUNDLE_DIR "backend dtype attention" ...
set -u
S=/private/tmp/claude-501/-Users-nover-repos-transformers/804c4b01-7ddc-4f18-a02c-672082928230/scratchpad
G=$S/mac-validation/llama-family/diagnostics
BD="${BURN_DUMP:-/Volumes/bag of holding/cargo-targets/tfm-polish-half-gates/release/burn-dump}"
TAG_SUFFIX="${TAG_SUFFIX:-}"
NAME="$1"; MODEL="$2"; BUNDLE="$3"; shift 3
mapfile -t PROMPTS < <(python3 -I -c "
import json, sys
d = json.load(open(sys.argv[1] + '/reference.json'))
for p in d['prompts']:
    print(','.join(map(str, p['token_ids'])))" "$BUNDLE")
for cfg in "$@"; do
  read -r backend dtype attention <<< "$cfg"
  for i in "${!PROMPTS[@]}"; do
    tag="$backend-$dtype-$attention$TAG_SUFFIX"
    bash "$S/heavy_diag.sh" "$G/$NAME.burn-dump.$tag.prompt$i.log" "$BD" "$backend" "$dtype" "$MODEL" \
      "${PROMPTS[$i]}" "$G/data/$NAME-burn/$tag/prompt$i" "$attention" || echo "FAIL $NAME $tag prompt $i"
  done
done
