#!/bin/bash
# Heavy-run wrapper for one diagnostic process (scratch, not committed).
# Usage: heavy_diag.sh LOG COMMAND... ; waits for source gates, takes the shared
# heavy-model lock, refuses a dirty tree, logs the source identity and command,
# runs the command under /usr/bin/time -l, and always releases the lock.
set -u
WT=/Users/nover/repos/transformers-worktrees/tfm-polish-half
LOCK=/Users/nover/repos/transformers-worktrees/.heavy-model.lock
LOG="$1"; shift
while pgrep -f full_gate.sh >/dev/null; do sleep 60; done
until mkdir "$LOCK" 2>/dev/null; do sleep 60; done
trap 'rmdir "$LOCK"' EXIT
dirty=$(git -C "$WT" status --porcelain | wc -l | tr -d ' ')
{
  echo "== source $(git -C "$WT" rev-parse HEAD) dirty=$dirty"
  echo "== start $(date -u +%FT%TZ) load $(sysctl -n vm.loadavg)"
  printf '== cmd'; printf ' %q' "$@"; echo
} > "$LOG"
if [ "$dirty" != 0 ]; then echo "== refusing: dirty tree" >> "$LOG"; exit 3; fi
start=$(date +%s)
/usr/bin/time -l "$@" >> "$LOG" 2>&1
status=$?
echo "== exit=$status wall=$(( $(date +%s) - start ))s" >> "$LOG"
exit $status
