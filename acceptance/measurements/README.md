# Measurement records

These files supplement the revision-named model acceptance records one directory
above. They are not model support declarations and do not use that schema.
Each run identifies its checkpoint, executable, scope, and unmeasured gates.
The primary model record links to the applicable run using an acceptance-root
relative path. A run's `raw_report.file` is relative to this directory; its
`primary_acceptance` or `checkpoint.source_record` is relative to `acceptance/`.

Keep complete raw timing samples and content hashes. A whole-process memory
high-water mark is not a phase-specific peak above idle, and a finite-state
long-context stress pass is not language-quality or official-logit parity.
The source-check runner executes Python consistency tests for these records.

The [September 10 small-model refresh](small-model-performance-20260910/manifest.json)
uses compact summaries rather than committing duplicated tensor lists and large
memory traces. The manifest authenticates each committed summary; an execution's
raw-report hash identifies the separate original local JSON, not the compact file.
All repeated timing samples are retained. The Qwen historical timing failure
remains unresolved despite close controlled rebuilt-old/current medians.

The later [v0.1 release decision](v0_1-release-20260910.json) records explicit
user authorization to adopt the controlled timing baseline while retaining
that historical slowdown as an unexplained limitation. It names the three
promoted execution profiles; the primary checkpoint records remain the
authoritative support declarations. Earlier measurement flags and results are
not rewritten by this policy decision.

The [v1](whole-test-file-coverage-v1-20260910.json) and
[v2](whole-test-file-coverage-v2-20260910.json) coverage records retain distinct
source snapshots and exclusion policies. Neither is qualified production-only
coverage; v2 exposes a numerical gap after removing 78 verified test-only files.

The [October 16 real-checkpoint replay](real-checkpoint-replay-20261016.json)
records the final polish-cycle source `f4c56c4` against every locally available
reference: FLAN-T5-small and Qwen3.5-0.8B (the v0.1 profiles), the Llama-family
checkpoints, gemma-3-4b-it, truncated gpt-oss-20b, and the DeepSeek-V2-Lite
smoke, 61 tests on Flex and Metal, each in its own process. Its log is committed
with its SHA-256, and `scripts/tests/test_real_checkpoint_replay_record.py`
checks that every summarized run appears in it. It promotes no checkpoint.

## Published and repository-only records

The crates.io package carries only part of this directory, because the
complete evidence exceeds crates.io's 10 MiB upload limit. It publishes the
top-level summary record of every campaign (`*.json` here), this README, and
the [September 10 small-model refresh](small-model-performance-20260910/manifest.json)
summaries that the v0.1 release record names as its timing baseline. The
revision-named checkpoint records one directory above are published too.
Every other campaign directory, with its raw logs, coverage exports, replay
reports, copied lockfiles, and runner scripts, exists only in the Git
repository, so a published record's link into one resolves only there.
[`scripts/package_contents.py`](../../scripts/package_contents.py) defines the
split and the package gate enforces it (see [RELEASE_GATES.md](../../RELEASE_GATES.md)).
Run from the extracted package, the record tests that need a repository-only
file skip with a message naming it; in a Git checkout a missing file fails.
