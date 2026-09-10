# Experiment interface property and fuzz evidence

The new public-interface properties cover semantic-axis selection, malformed ranges,
broadcast activation validity, attention exclusions after structural and consumer
edits, transfer aliasing and failure atomicity, and stable categorical losses and
gradients at tied maxima. Nine properties passed with 48 deterministic cases each.
The six ordinary fuzz-oracle tests and all-target warning-denied fuzz Clippy also
passed. Their exact commands, storage preflights and logs are under
[`normal-validation/`](normal-validation/).

The separate AddressSanitizer campaign ran the new `experiment_interfaces` target
for exactly **10,000 inputs**, using seed `20260910`, 13 copied seed files and an
initially empty writable corpus. Native exit was zero, with no timeout, crash
artifact or warning/error/sanitizer diagnostic. Native elapsed time was **2.079 s**;
libFuzzer reported **249 MiB peak RSS** and 308 new corpus units. These are bounded
campaign diagnostics, not comparative benchmarks or line-coverage measurements.

[`manifest.json`](manifest.json) links command arrays, raw logs, tool identities,
binary hashes, normal checks and source inventories. The native execution interval
preserved all 168 inventoried production/protocol inputs, the binary, and both seed
inventories. This non-test build excludes sibling `cfg(test)` files from that
inventory. The separate ordinary-test input snapshot was captured after validation.
The source snapshot includes untracked files; its integration commit alone does not
identify all pending source changes.

The integrator reused the existing external ASan dependency cache through normal
`fuzz/target`; the reviewed storage choice and both volume preflights are retained
in [`fuzz-storage.json`](fuzz-storage.json) and the command records. No artifacts
were downloaded and the native campaign performed no build. Verbose build evidence
confirms address sanitization and sanitizer coverage. The prebuilt standard/native
dependency closure is not claimed to be hermetically rebuilt. No accelerator,
checkpoint support, packaging or exhaustive-correctness qualification is implied.
