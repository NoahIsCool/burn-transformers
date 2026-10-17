#!/bin/bash
# Runs burn-dump over prompt 0 plus the bundle's greedy tokens with a prefill of prompt 0 and
# token-by-token decoding of the rest, one process at a time under the heavy-model lock (scratch).
# Usage: burn_decode_dumps.sh NAME MODEL_DIR BUNDLE_DIR "backend dtype attention" ...
set -u
S=/private/tmp/claude-501/-Users-nover-repos-transformers/804c4b01-7ddc-4f18-a02c-672082928230/scratchpad
G=$S/mac-validation/llama-family/diagnostics
BD="${BURN_DUMP:-/Volumes/bag of holding/cargo-targets/tfm-polish-half-gates/release/burn-dump}"
TAG_SUFFIX="${TAG_SUFFIX:-}"
NAME="$1"; MODEL="$2"; BUNDLE="$3"; shift 3
read -r TOKENS PREFILL < <(python3 -I -c "
import json, sys
d = json.load(open(sys.argv[1] + '/reference.json'))
p = d['prompts'][0]['token_ids']
s = p + [step['token'] for step in d['greedy']['steps']]
print(','.join(map(str, s)), len(p))" "$BUNDLE")
for cfg in "$@"; do
  read -r backend dtype attention <<< "$cfg"
  tag="$backend-$dtype-$attention$TAG_SUFFIX"
  bash "$S/heavy_diag.sh" "$G/$NAME.burn-decode.$tag.log" "$BD" "$backend" "$dtype" "$MODEL" \
    "$TOKENS" "$G/data/$NAME-burn-decode/$tag" "$attention" "$PREFILL" || echo "FAIL $NAME $tag"
done
