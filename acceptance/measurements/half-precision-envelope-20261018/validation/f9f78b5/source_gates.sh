#!/bin/bash
# Scratch (half2): source validation of the worktree head, one check after another (not committed).
set -u
WT=/Users/nover/repos/transformers-worktrees/tfm-half-followups2
V=/private/tmp/claude-501/-Users-nover-repos-transformers/804c4b01-7ddc-4f18-a02c-672082928230/scratchpad/half2/validate
export CARGO_TARGET_DIR="/Volumes/bag of holding/cargo-targets/tfm-half-followups2"
cd "$WT" || exit 2
echo "== source $(git rev-parse HEAD) dirty=$(git status --porcelain | wc -l | tr -d ' ') $(date -u +%FT%TZ)" > $V/summary.log
run() { # name command...
  local name="$1"; shift
  local start=$(date +%s)
  { echo "== source $(git rev-parse HEAD) dirty=$(git status --porcelain | wc -l | tr -d ' ')"; echo "== cmd $*"; "$@"; echo "== exit=$?"; } > "$V/$name.log" 2>&1
  local status=$(tail -1 "$V/$name.log" | sed 's/== exit=//')
  echo "$name exit=$status wall=$(( $(date +%s) - start ))s $(date -u +%FT%TZ)" >> $V/summary.log
}
run fmt cargo fmt --check
run clippy-nightly cargo clippy --locked --all-targets --all-features -- -D warnings
run clippy-1.99.1 cargo +1.99.1 clippy --locked --all-targets --all-features -- -D warnings
run clippy-1.99.1-cpu cargo +1.99.1 clippy --locked --all-targets --no-default-features --features std,flex,cpu,cuda,rocm,tokenizers,chat-templates,autodiff,train -- -D warnings
run test-default cargo test --locked
run test-features cargo test --locked --no-default-features --features std,flex,tokenizers,chat-templates,autodiff,train
run msrv-1.92.0 cargo +1.92.0 check --locked --all-targets
run rustdoc-private env RUSTDOCFLAGS="-D warnings" cargo doc --locked --no-deps --document-private-items
run rustdoc-public env RUSTDOCFLAGS="-D warnings" cargo doc --locked --no-deps
run doc-audit python3 scripts/check_documentation.py --standards --classifier "/Volumes/bag of holding/cargo-targets/tfm-polish-layout/debug/transformers-coverage-layout" --output /private/tmp/claude-501/-Users-nover-repos-transformers/804c4b01-7ddc-4f18-a02c-672082928230/scratchpad/followups-docs.json
run python-tests python3 -m unittest discover -s scripts/tests
echo "== done $(date -u +%FT%TZ)" >> $V/summary.log
