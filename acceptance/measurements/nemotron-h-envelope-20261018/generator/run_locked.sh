#!/bin/bash
# Campaign driver: runs run_graph.py for each LABEL under the heavy-model lock,
# released between graphs and on any exit, including failure. Both external
# volumes must be mounted, before the lock and again once it is held.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
EXPORT=$1; shift
LOCK=/Users/nover/repos/transformers-worktrees/.heavy-model.lock
held=0
release() { if [ "$held" = 1 ]; then rmdir "$LOCK" && echo "$(date -u +%FT%TZ) lock released"; held=0; fi; }
trap 'release; exit 130' INT TERM HUP
trap 'release' EXIT
mounted() {
  for volume in "/Volumes/bag of holding" /Volumes/T9; do
    mount | grep -q " on $volume " || { echo "$(date -u +%FT%TZ) STOP: $volume is not mounted"; return 1; }
  done
}
mounted || exit 5
for label in "$@"; do
  echo "$(date -u +%FT%TZ) waiting for the lock ($label)"
  until mkdir "$LOCK" 2>/dev/null; do sleep 60; done
  held=1; echo "$(date -u +%FT%TZ) lock acquired for $label"
  mounted || exit 5
  python3 -I "$HERE/run_graph.py" "$label" "$EXPORT"; status=$?
  echo "$(date -u +%FT%TZ) $label exit=$status"
  release
done
echo "$(date -u +%FT%TZ) done"
