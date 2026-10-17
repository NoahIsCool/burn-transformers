# 17. CubeCL CPU: the DeepSeek-V2 parity case dies of SIGFPE on x86-64

| | |
|---|---|
| Upstream | [tracel-ai/cubecl](https://github.com/tracel-ai/cubecl), CPU runtime (`cubecl-cpu`, MLIR/LLVM code generation), reached through `burn-cubecl` |
| Version reproduced | burn 0.21.0 (this crate's pin), on GitHub's `ubuntu-latest` x86-64 runner; not reproduced on arm64 |
| Kind | **Crash** (the process is killed by SIGFPE) |
| Upstream status, checked 2026-10-17 | **Not isolated; do not file yet.** The kernel that divides by zero has not been identified |
| Repro | `cpu_deepseek_v2_matches_flex` in `tests/backend_parity.rs`, on an x86-64 Linux host: `cargo test --locked --no-default-features --features std,flex,cpu --test backend_parity cpu_deepseek_v2_matches_flex -- --include-ignored --nocapture` |
| Patch | None |
| Workarounds an upstream fix removes | None in the crate. CI's `cpu-parity` job runs this case as an expected failure, each test in its own process |

## Observation

CI's `cpu-parity` job ran the CubeCL CPU parity suite on GitHub's x86-64
`ubuntu-latest` runner twice, at `76ae7ab` (run 39036489525) and `04d9881`
(run 39038281588). Both times, `cpu_hybrid_registry_families_match_flex`
(which then checked Qwen3.5, Qwen3-MoE, DeepSeek-V2, and Muse Glimmer in
turn) printed the Qwen3.5 and Qwen3-MoE lines, and about 30 s later the test
process was killed:

```text
process didn't exit successfully: `.../backend_parity-315dddfa45552c3b cpu --include-ignored
--nocapture --test-threads 1 --skip ...` (signal: 8, SIGFPE: erroneous arithmetic operation)
```

Each family's line is printed (unbuffered, with `--nocapture`) after its
checks complete, so the crash happened while the tiny DeepSeek-V2 case was
running, in the CubeCL CPU backend's generated code. Every case before it in
that process, and the CubeCL CPU golden suites in their own processes,
passed with the deviations measured on arm64.

On x86-64, SIGFPE from ordinary code is an integer division (or remainder)
by zero, or `i32::MIN / -1`: `div` and `idiv` trap. On arm64, `udiv` and
`sdiv` by zero return zero without trapping. That explains why the same case
matches Flex to `1.2e-7` on an Apple M3 Max, and suggests that some kernel
there divides by zero and the zero quotient happens not to matter.

## Ruled out on arm64

Both checks use the case on arm64 with instrumented copies of the pinned crates
(`[patch.crates-io]`, scratch only):

- **cubecl-std's perpendicular copy** (`copy_perpendicular`, the kernel of
  [report 15](15-cubecl-cpu-reshape-copy.md)) divides by `shape(last) /
  vector_size` and by `output.stride(axis) / vector_size`. A probe in
  `launch_copy_perpendicular_ref` reporting either quotient as zero never
  fired, in any family of the hybrid case or in Llama, Gemma, gpt-oss, the
  modular encoder-decoder, or T5.
- **Zero-sized tensors.** A probe in `burn-cubecl`'s `CubeTensor::new` and
  `new_contiguous` reporting any zero dimension fired only for the
  `[0, 0, 0, 0]` placeholders that `Tensor::inplace` swaps in during
  `operations::assign_region`. Every family creates them, including Qwen3.5
  and Qwen3-MoE, which pass on x86-64, and DeepSeek-V2 creates no other.

Views are not covered by the second probe: a zero-length slice keeps its
parent's handle, so a zero-sized view would not appear there.

## Next step

Isolate the kernel on an x86-64 host, for example the SCC: run the case
under `gdb` (or with core dumps) to get the faulting instruction, and log
each CubeCL launch's kernel name and shapes just before the crash. Then
reduce it to a repro against `cubecl-cpu` alone, and check cubecl 0.11
(Burn 0.22) before filing.
