#!/bin/bash
# Hold the heavy-model lock for one Qwen3.5-0.8B benchmark campaign and run it.
# Usage: driver.sh CAMPAIGN_DIR PROTOCOL_SCRIPT
# The runner checks both volumes, AC power and the host identity before every
# sample and stops (resumably) when one fails; this driver also refuses to
# start, or to take the lock, when a volume is missing.
set -u
C="$1"; P="$2"
LOCK=/Users/nover/repos/transformers-worktrees/.heavy-model.lock
log="$C/driver.log"
volumes_mounted() {
  for v in "/Volumes/T9" "/Volumes/bag of holding"; do
    mount | grep -q " on $v (" || { echo "$(date -u +%FT%TZ) MISSING VOLUME $v; $1" >> "$log"; return 1; }
  done
}
volumes_mounted "not starting" || exit 3
echo "$(date -u +%FT%TZ) waiting for $LOCK ($(uptime))" >> "$log"
until mkdir "$LOCK" 2>/dev/null; do sleep 60; done
trap 'rmdir "$LOCK" 2>/dev/null; echo "$(date -u +%FT%TZ) lock released" >> "$log"' EXIT
trap 'exit 130' INT TERM HUP
echo "$(date -u +%FT%TZ) lock acquired ($(uptime))" >> "$log"
volumes_mounted "stopping" || exit 3
echo "$(date -u +%FT%TZ) protocol script sha256 $(shasum -a 256 "$P" | cut -d' ' -f1)" >> "$log"
python3 -I "$P" run "$C" >> "$C/runner.log" 2>&1
rc=$?
echo "$(date -u +%FT%TZ) runner exit $rc" >> "$log"
exit $rc
