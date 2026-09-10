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

The [v1](whole-test-file-coverage-v1-20260910.json) and
[v2](whole-test-file-coverage-v2-20260910.json) coverage records retain distinct
source snapshots and exclusion policies. Neither is qualified production-only
coverage; v2 exposes a numerical gap after removing 78 verified test-only files.
