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
