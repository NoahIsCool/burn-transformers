#!/bin/bash
# Campaign driver: runs one run_gate.py invocation under the heavy-model lock,
# released on any exit, including failure. Both external volumes must be
# mounted, before the lock and again once it is held.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
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
echo "$(date -u +%FT%TZ) waiting for the lock ($3)"
until mkdir "$LOCK" 2>/dev/null; do sleep 60; done
held=1; echo "$(date -u +%FT%TZ) lock acquired for $3"
mounted || exit 5
python3 -I "$HERE/run_gate.py" "$@"; status=$?
echo "$(date -u +%FT%TZ) $3 exit=$status"
release
exit $status
