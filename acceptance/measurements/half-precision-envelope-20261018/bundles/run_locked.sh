#!/bin/bash
# Scratch (half2 S2): runs make_bundles.py for each KEY[:VARIANT,...] under the heavy-model lock,
# released between checkpoints and on any exit, including failure. Both volumes must be mounted.
set -u
B=/private/tmp/claude-501/-Users-nover-repos-transformers/804c4b01-7ddc-4f18-a02c-672082928230/scratchpad/half2/bundles
LOCK=/Users/nover/repos/transformers-worktrees/.heavy-model.lock
held=0
release() { if [ "$held" = 1 ]; then rmdir "$LOCK" && echo "$(date -u +%FT%TZ) lock released"; held=0; fi; }
trap 'release; exit 130' INT TERM HUP
trap 'release' EXIT
for volume in "/Volumes/bag of holding" /Volumes/T9; do
  mount | grep -q " on $volume " || { echo "$(date -u +%FT%TZ) STOP: $volume is not mounted"; exit 5; }
done
for spec in "$@"; do
  key=${spec%%:*}; variants=""
  [ "$spec" != "$key" ] && variants=$(echo "${spec#*:}" | tr ',' ' ')
  echo "$(date -u +%FT%TZ) waiting for the lock ($key)"
  until mkdir "$LOCK" 2>/dev/null; do sleep 60; done
  held=1; echo "$(date -u +%FT%TZ) lock acquired for $key"
  for volume in "/Volumes/bag of holding" /Volumes/T9; do
    mount | grep -q " on $volume " || { echo "$(date -u +%FT%TZ) STOP: $volume is not mounted"; exit 5; }
  done
  python3 -I "$B/make_bundles.py" "$key" $variants || echo "$(date -u +%FT%TZ) $key had a failure"
  release
done
echo "$(date -u +%FT%TZ) done"
