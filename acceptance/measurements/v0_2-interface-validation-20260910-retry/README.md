# v0.2 interface validation

**Passed: all 29 recorded steps, with unchanged inputs.** This is the final
local implementation check for the [v0.2 contract](../../../V0_2_INTERFACE.md),
completed on September 10, 2026. The package remains unreleased at version 0.1.0.

The [manifest](manifest.json) contains exact commands, statuses, timings,
toolchain identities, test counts, binary hashes, and links to complete logs.
The [before](source-inventory-before.json) and [after](source-inventory-after.json)
inventories agree at SHA-256
`398eef1639e7fd7d2739f036e73332193d6fc4d655668c77922ead15dba56a0d`.
They include untracked implementation files, fixtures, protocol inputs, Cargo
metadata, and historical records consumed by tests. Markdown, generated study
outputs, and ignored build/weight storage are excluded; study executables
authenticate their prepared artifacts separately.

## Checks

| Check | Result |
| --- | --- |
| Rust tests, all targets | 1,368 passed; 32 opt-in tests ignored |
| Rust doctests | 3 passed |
| Repository Python tests | 359 passed; 14 skipped |
| Tuned Lens / Greater-Than Python tests | 7 / 5 passed |
| Clippy, all targets and features | Passed with warnings denied |
| rustfmt and rustdoc | Passed |
| Minimal and `std` configurations | Passed |
| Exact Rust 1.92.0, all targets | Passed |
| Fresh example build and local replay | All ten passed |

Rust execution uses `std,flex,tokenizers,chat-templates,autodiff`; all-feature
validation is compilation/linting. Cargo commands use the committed lockfile,
offline dependencies, and release builds because the local debug target is an
external symlink. The complete run took 243.00 seconds, including example
execution. [validation_runner.py](validation_runner.py) preserves the executed
one-off runner; it requires the already prepared studies and writes a new
directory for each attempt.

## Example replay

Each executable was hashed after the final example build and remained unchanged.
Numerical assertions passed. These are single-host CPU observations after
compilation, not performance regression or hardware qualification.

| Example | Wall seconds | Peak process RSS, bytes |
| --- | ---: | ---: |
| Activation workbench | 0.160 | 2,981,888 |
| Attention-head attribution | 0.115 | 2,310,144 |
| Custom composition | 0.121 | 2,670,592 |
| T5 classifier | 0.119 | 2,752,512 |
| MLA encoder-decoder | 0.120 | 2,572,288 |
| Random Qwen prefix | 0.129 | 3,424,256 |
| Grokking Fourier analysis | 0.433 | 22,708,224 |
| Clock and Pizza | 1.950 | 153,468,928 |
| Tuned Lens | 25.822 | 1,284,308,992 |
| Greater-Than | 195.690 | 874,725,376 |

Memory is macOS `/usr/bin/time -l` maximum resident set size for the child
process, in bytes; it is not system memory footprint. The manifest preserves
the actual measurement wrapper and raw stderr. The three generated study
reports are copied under [replayed_results/](replayed_results/); Grokking and
the other examples retain their numerical output in the command logs.

## Scientific and checkpoint scope

The [example guide](../../../examples/README.md) distinguishes Grokking's
Table 1 reproduction, Clock and Pizza's published geometric measurements,
and the bounded Tuned Lens and Greater-Than adaptations. Clock and Pizza's
unresolved published accuracy discrepancy remains documented. No training
dynamics, original large-sample lens curve, or complete GPT-2 circuit is claimed.

Separate real-checkpoint records validate [selective Qwen prefix loading](../qwen-prefix-v02-20260910.json)
and [FLAN-T5 encoder extraction/staged hooks](../v0_2-flan-t5-small-flex-20260910/manifest.json).
This runner links and hashes those records; it does not relabel them as new runs.
Coverage, accelerator execution, package verification, and expanded checkpoint
support require their own [release gates](../../../RELEASE_GATES.md).

## Preserved earlier attempt

The [first consolidated attempt](../v0_2-interface-validation-20260910/manifest.json)
passed every execution and numerical check, but its inventory changed when a
generated `plot_validation.json` was added during the last example. No existing
source or protocol file changed. The recorder correctly rejected that attempt;
its [audit and numerical output](../v0_2-interface-validation-20260910/post_run_audit.json)
remain available. This second run repeated all 29 steps with every report file
frozen and passed without changing acceptance thresholds or implementation code.
