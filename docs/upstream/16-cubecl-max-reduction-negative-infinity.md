# 16. cubek-reduce: `max_dim` of a row of `-inf` returns `f32::MIN` on every CubeCL backend

| | |
|---|---|
| Upstream | [tracel-ai/cubek](https://github.com/tracel-ai/cubek), crate `cubek-reduce` (`components::instructions`), reached through `burn-cubecl`'s `max_dim`/`min_dim` |
| Version reproduced | burn 0.21.0, cubek-reduce 0.2.0 (this crate's pin), on the CubeCL CPU backend and on Metal |
| Kind | **Correctness** (a finite maximum for a row with no finite entry; an out-of-range `argmax`) |
| Upstream status, checked 2026-10-17 | **Fixed in v0.22.0; do not file.** [cubek#457](https://github.com/tracel-ai/cubek/pull/457) ("fix(reduce): propagate NaNs in extrema reductions", merged 2026-08-04, `09e10a5`, following burn#4814) uses negative and positive infinity as the floating-point identities of `Max` and `Min` (`instructions/extrema.rs`, `max_identity`/`min_identity`), which in 0.3.0 also compute `argmax`/`argmin` (an index output of the same instruction). It ships in cubek-reduce 0.3.0, which burn-cubecl 0.22.0 uses through cubek 0.3.0. Checked by reading the 0.3.0 sources; the repro was not run on 0.22.0 (its `cpu` feature builds against another prebuilt LLVM, tracel-llvm 23, that this host has not downloaded). |
| Related upstream issues | burn#4814; cubek#430 / #457 |
| Repro | [`repros/cubecl-max-reduction-negative-infinity`](repros/cubecl-max-reduction-negative-infinity) (`--features metal` adds Metal) |
| Patch | None drafted: fixed upstream before this report, and a 0.21.0 patch would have to rebuild the 0.3.0 identity helpers, which need `elem_type_of` from cubecl 0.11 |
| Patch verified | Not applicable |
| Workarounds an upstream fix removes | The "Exhausted rows" rule of `deterministic_top_k` in `src/moe/router.rs`; its pin `cubecl_max_reduction_of_negative_infinities_justifies_the_exhausted_row_rule` (`src/moe/router_tests.rs`, `--features cpu`) fails once the reduction is fixed |

## Issue draft (not to be filed: fixed in v0.22.0)

Kept as the record of the 0.21.0 behaviour this crate works around.

**Title:** CubeCL `max_dim` of an all-`-inf` row returns `-3.4028235e38` (`min_dim` of `+inf` returns `3.4028235e38`)

### Describe the bug

`Tensor::max_dim` on a CubeCL backend seeds the reduction with `min_value()`, the lowest
*finite* value of the element type, rather than `-inf`. A row whose every entry is `-inf`
(a fully masked score row) therefore reduces to `f32::MIN`, which is not one of its
entries; `min_dim` of a row of `+inf` likewise returns `f32::MAX`. Code that finds the
maximal entries by comparing a row with its own maximum (`x.equal(x.max_dim(d))`, the
usual tie-aware argmax or top-k) finds none. `argmax` of such a row returns `-1` (the
`u32::MAX` index sentinel read as `i32`). Flex returns `-inf`, `+inf`, and index 0.

### To reproduce

Standalone project: `repros/cubecl-max-reduction-negative-infinity` (`burn = "=0.21.0"`,
features `cpu`, `flex`; `--features metal` adds Metal).

### Expected behavior

`max_dim` of `[-inf, -inf, -inf, -inf]` is `-inf`; `min_dim` of `[+inf; 4]` is `+inf`;
`argmax` of a row of equal values is 0.

### Actual behavior (Apple M3 Max)

```text
flex (control):
  row 0: max            -inf min            -inf  (expected -inf, -inf); argmax 0; some entry equals the max: true
  row 1: max             inf min             inf  (expected inf, inf); argmax 0; some entry equals the max: true
cubecl-cpu:
  row 0: max   -3.4028235e38 min            -inf  (expected -inf, -inf); argmax -1; some entry equals the max: false   <-- wrong
  row 1: max             inf min    3.4028235e38  (expected inf, inf); argmax 0; some entry equals the max: true   <-- wrong
metal:
  row 0: max   -3.4028235e38 min            -inf  (expected -inf, -inf); argmax -1; some entry equals the max: false   <-- wrong
  row 1: max             inf min    3.4028235e38  (expected inf, inf); argmax 0; some entry equals the max: true   <-- wrong
4 wrong reduction values
```

The finite control row is correct on every backend.

### Root cause

cubek-reduce 0.2.0, `src/components/instructions/max.rs`:

```rust
fn null_input(_this: &Self) -> Vector<P::EI, P::SI> {
    Vector::empty().fill(P::EI::min_value())
}
```

with the same `min_value()` in `null_accumulator` and `to_output_parallel`; `min.rs` uses
`max_value()`, and `argmax.rs`/`argmin.rs` seed their value accumulators the same way.
`Float::min_value()` is the lowest finite value, and the reduction keeps the accumulator
whenever `accumulator > item`, which holds for every `-inf` item.

### Proposed fix

Use the infinities as floating-point identities, as cubek#457 did
(`max_identity`/`min_identity` in cubek-reduce 0.3.0's `instructions/extrema.rs`).

### Environment

burn 0.21.0 (features `cpu`, `flex`, `metal`), cubek-reduce 0.2.0, Rust 1.94 nightly
(2025-12-22), Apple M3 Max, macOS 27. CUDA and ROCm share the kernel source but were not
run.

## Notes for this crate

- Found through report [15](15-cubecl-cpu-reshape-copy.md): on the CubeCL CPU backend,
  Nemotron-H's wrong group mask left rows whose eligible experts were all `-inf`, and the
  router's `deterministic_top_k` returned expert ID `E` for them, which dispatch rejected
  ("expert ID 4 at flattened assignment 29"). The kernel promises in-range, distinct IDs
  for any input, including NaN router logits (`RouteDecision::non_finite_tokens` then flags
  the token); on every CubeCL backend (measured on Metal and the CubeCL CPU backend; CUDA
  and ROCm run the same kernel source) it broke that promise for every row whose remaining
  keys were all `-inf` or NaN.
- Fixed in the crate, exactly and for every backend: `deterministic_top_k` treats a row
  whose available keys are all `-inf` as one tie and picks the smallest available index,
  without consulting the maximum (`src/moe/router.rs`, "Exhausted rows"). Flex results
  are unchanged. `cubecl_cpu_top_k_is_total_for_non_finite_rows` runs the existing
  totality test on the CubeCL CPU backend; without the rule it returns ID 4 for the
  all-NaN and all-`-inf` rows.
- Other `max_dim` users were checked: `generation.rs`'s min-p filter and
  `training::log_softmax` give the same results with `f32::MIN` as with `-inf` for a row of
  `-inf` (both mask or produce NaN there). Not changed: greedy `select` in
  `generation.rs` takes `argmax`, which returns `-1` on CubeCL for a row of `-inf`; only a
  row whose hard constraints ban every token reaches it.
