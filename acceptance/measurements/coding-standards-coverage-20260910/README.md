# Coding standards coverage evidence

This records the frozen-source baseline, an intermediate failed attempt, and the final integrated test run separately. The baseline passed the existing aggregate gate but failed the separate whole-test-file diagnostic. The intermediate attempt passed crate-wide at 90.2508% but reached only 94.8014% shared/core. The final measurements are in `record.json`; thresholds remain 90% crate-wide and 95% shared/core.

The final **qualified whole-test-file diagnostic** passes at **90.406140% crate-wide** and **95.154505% shared/core**. The existing native aggregate independently passes at 93.378818% and 96.146017%.

| Measured revision | Native aggregate: crate / shared | Whole-file diagnostic: crate / shared |
| --- | --- | --- |
| `530b8ea` baseline | 92.845653% / 95.307139% | 89.472241% / 93.452257% — both fail |
| `575073f` intermediate | 93.299370% / 95.997419% | 90.250811% / 94.801350% — shared fails |
| `af95066` final | 93.378818% / 96.146017% | 90.406140% / 95.154505% — both pass |

The coverage profile is `std,flex,tokenizers,chat-templates,autodiff`, with library and ordinary integration tests. Shared/core means `src/` outside `src/models/`. Model implementations remain in the crate denominator. Hardware-specific paths, examples, and branches are not measured by these counters.

The legacy aggregate includes unit-test and generated mappings. The separate physical-line diagnostic removes only independently qualified whole test files. Its parsed structural audit cannot resolve arbitrary generated test code or procedural-macro inputs. **Neither measurement is qualified as production-only or branch coverage.** No exclusion policy or threshold was relaxed.

Reproduce the native capture on an equipped host with at least 20 GiB free:

```sh
python3 scripts/check_release.py --gate coverage --toolchain nightly --offline --coverage-compiler-evidence
```

Use the exact pinned compiler and LLVM tools described in `scripts/COVERAGE_EVIDENCE.md`. Before changing captured source, follow `scripts/WHOLE_FILE_COVERAGE.md` to produce the separately qualified whole-file diagnostic. The recorded source revision identifies the measured code; later evidence-only commits do not retroactively qualify modified code.

Each measurement retains original source snapshots, compiler records and dep-info, native export invocation, logs, gate results, classifier invocation, and the whole-file ledger. The final native coverage JSON and all three classifier reports use lossless XZ compression; `record.json` records both compressed and decompressed hashes. The final native report permits portable recounting against the recorded source. The two earlier native exports remain in the explicitly listed ignored local directories with exact hashes; they are not included in the tracked bundle, limiting package growth. Cargo JSON remains in each ignored local run directory as a metadata-bearing rewrite, so its bytes are not claimed equal to native LLVM output.

Compiled binaries, profiles, and tool executables are deliberately untracked. Their identities were authenticated during each capture, but independent replay of that authentication requires the original ignored artifacts or a fresh run. This is local trusted-tool evidence, not compiler attestation. Historical records remain unchanged, and no checkpoint support or release promotion is inferred.
