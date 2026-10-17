#!/bin/bash
# Waits for source gates, takes the shared heavy-model lock, runs the Gemma gates, and always releases the lock.
# Usage: gemma_heavy.sh BINARY OUT_DIR [ONLY_PREFIX...]
set -u
S=/private/tmp/claude-501/-Users-nover-repos-transformers/804c4b01-7ddc-4f18-a02c-672082928230/scratchpad
LOCK=/Users/nover/repos/transformers-worktrees/.heavy-model.lock
while pgrep -f '[f]ull_gate.sh' >/dev/null; do sleep 60; done
until mkdir "$LOCK" 2>/dev/null; do echo "$(date -u +%FT%TZ) lock held; waiting"; sleep 60; done
trap 'rmdir "$LOCK"' EXIT
echo "$(date -u +%FT%TZ) lock acquired"
python3 -I "$S/run_gemma_gates.py" "$@"
echo "$(date -u +%FT%TZ) runner exit=$?"
