#!/bin/bash
# Build the Qwen3.5-0.8B benchmark binary for one source tree with rustc 1.92.0
# (ded5c06cf), the toolchain of the September 10 series
# (small-model-performance-20260910/metal-build.json), and the advertised
# profile's features std,metal,tokenizers,chat-templates.
# Usage: build.sh <label> <source-tree> <target-dir> <output-dir>
# The target directory must be fresh or belong to this source tree alone, so
# cargo cannot judge another tree's artifacts fresh.
set -uo pipefail
label="$1"; tree="$2"; target="$3"; out="$4"
mkdir -p "$out"
for v in "/Volumes/bag of holding" /Volumes/T9; do
  mount | grep -q " on $v (" || { echo "MISSING VOLUME $v" | tee "$out/preflight.txt"; exit 3; }
done
avail_t=$(df -k "/Volumes/bag of holding" | awk 'NR==2{print $4}')
{
  echo "== label $label"
  echo "== date $(date -u +%FT%TZ)"
  echo "== tree $tree"
  echo "== commit $(git -C "$tree" rev-parse HEAD)"
  echo "== dirty $(git -C "$tree" status --porcelain --untracked-files=all | wc -l | tr -d ' ')"
  echo "== src_tree $(git -C "$tree" rev-parse HEAD:src)"
  echo "== cargo_lock_sha256 $(shasum -a 256 "$tree/Cargo.lock" | cut -d' ' -f1)"
  echo "== cargo_toml_sha256 $(shasum -a 256 "$tree/Cargo.toml" | cut -d' ' -f1)"
  echo "== target $target"
  echo "== free_kib_target_volume $avail_t"
  mount | grep -E " on /Volumes/(T9|bag of holding) \("
  rustc +1.92.0 -vV
  cargo +1.92.0 -V
} > "$out/preflight.txt"
if [ "$avail_t" -lt $((20*1024*1024)) ]; then
  echo "PREFLIGHT FAILED: less than 20 GiB free" | tee -a "$out/preflight.txt"; exit 4
fi
cmd=(cargo +1.92.0 test --manifest-path "$tree/Cargo.toml" --locked --offline --release
     --no-default-features --features std,metal,tokenizers,chat-templates
     --no-run --test qwen35_benchmarks --message-format=json)
printf '%q ' env CARGO_TARGET_DIR="$target" "${cmd[@]}" > "$out/command.txt"; echo >> "$out/command.txt"
start=$(date +%s)
CARGO_TARGET_DIR="$target" "${cmd[@]}" > "$out/cargo.jsonl" 2> "$out/cargo.stderr"
rc=$?
end=$(date +%s)
exe=$(python3 -I -c "
import json,sys
for line in open(sys.argv[1]):
    try: m=json.loads(line)
    except Exception: continue
    if m.get('reason')=='compiler-artifact' and m.get('executable') and m['target']['name']=='qwen35_benchmarks':
        print(m['executable'])
" "$out/cargo.jsonl" | tail -1)
fresh=$(python3 -I -c "
import json,sys
n=0
for line in open(sys.argv[1]):
    try: m=json.loads(line)
    except Exception: continue
    if m.get('reason')=='compiler-artifact' and m['target']['name'] in ('transformers','qwen35_benchmarks'):
        print(m['target']['name'], 'fresh' if m.get('fresh') else 'compiled')
" "$out/cargo.jsonl")
{
  echo "== exit $rc seconds $((end-start))"
  echo "$fresh" | sed 's/^/== artifact /'
  echo "== executable $exe"
  echo "== sha256 $(shasum -a 256 "$exe" | cut -d' ' -f1)"
  echo "== bytes $(stat -f %z "$exe")"
  echo "== commit_after $(git -C "$tree" rev-parse HEAD)"
  echo "== dirty_after $(git -C "$tree" status --porcelain --untracked-files=all | wc -l | tr -d ' ')"
} > "$out/result.txt"
cat "$out/result.txt"
exit $rc
