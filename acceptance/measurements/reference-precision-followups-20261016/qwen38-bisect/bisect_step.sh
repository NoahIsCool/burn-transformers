#!/bin/bash
# git bisect run step: bad when the Qwen3.8 F16 smoke's sampled peak footprint exceeds 67.35 GB.
# Exit 125 skips commits that do not build or lack the test.
set -u
S=/private/tmp/claude-501/-Users-nover-repos-transformers/66ab4e94-8aa9-472d-b244-dc221f41f2e6/scratchpad
LOCK=/Users/nover/repos/transformers-worktrees/.heavy-model.lock
SEED=/Users/nover/repos/transformers-worktrees/tfm-checkpoint-replay/Cargo.lock
LOG=$S/logs/bisect.log
commit=$(git rev-parse --short HEAD)
seeded=0
if ! git ls-files --error-unmatch Cargo.lock >/dev/null 2>&1; then cp "$SEED" Cargo.lock; seeded=1; fi
cleanup() {
  if [ $seeded = 1 ]; then rm -f Cargo.lock; else git checkout -q -- Cargo.lock 2>/dev/null; fi
}
trap cleanup EXIT
binary=$(CARGO_TARGET_DIR=/Volumes/T9/cargo-targets/tfm-qwen38-base cargo test --offline --release \
  --no-default-features --features std,metal,tokenizers,chat-templates --test local_artifacts --no-run \
  --message-format=json 2>/dev/null | python3 -c '
import json, sys
for line in sys.stdin:
    try:
        d = json.loads(line)
    except ValueError:
        continue
    if d.get("reason") == "compiler-artifact" and d.get("executable") and d["target"]["kind"] == ["test"]:
        print(d["executable"])
')
if [ -z "$binary" ]; then echo "$commit build-failed skip" >> "$LOG"; exit 125; fi
if ! "$binary" --list --ignored 2>/dev/null | grep -q "loads_pinned_qwen38_text_and_mtp_and_runs_metal"; then
  echo "$commit no-test skip" >> "$LOG"; exit 125
fi
until mkdir "$LOCK" 2>/dev/null; do sleep 15; done
result=$(TRANSFORMERS_QWEN38_DIR="/Volumes/bag of holding/models/Qwen/Qwen3.8-27B-1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0" \
  python3 "$S/footprint_trace.py" "$S/logs/bisect-$commit.tsv" -- "$binary" loads_pinned_qwen38_text_and_mtp_and_runs_metal \
  --ignored --exact --nocapture --test-threads 1 2>&1 | grep -E "peak=|stateless_seconds" | sed -E 's/mask_cosine.*(load_seconds)/\1/' | tr '\n' ' ')
rmdir "$LOCK"
peak=$(echo "$result" | sed -nE 's/.*peak=([0-9]+).*/\1/p')
if [ -z "$peak" ] || ! echo "$result" | grep -q "exit=0"; then echo "$commit run-failed skip $result" >> "$LOG"; exit 125; fi
if [ "$peak" -gt 67350000000 ]; then verdict=bad; code=1; else verdict=good; code=0; fi
echo "$commit $verdict $(git log -1 --format=%s HEAD | cut -c1-80) | $result" >> "$LOG"
exit $code
