# 08. burn-core: a mapped parameter (`LinearLayout::Col` weight) is not an autodiff leaf

| | |
|---|---|
| Upstream | [tracel-ai/burn](https://github.com/tracel-ai/burn), crate `burn-core` (`module::param`), seen through `burn-nn`'s `LinearConfig` |
| Version reproduced | 0.21.0 (this crate's pin) |
| Kind | Correctness (first optimizer step skips the weight; `load_record` panics) |
| Upstream status, checked 2026-10-16 | `init_mapper`: **fixed in v0.22.0** (it detaches the base value, maps it, then applies the gradient policy). The `load_record` panic (`transform_for_load` maps a tracked tensor before `set_require_grad`) was not confirmed on `main` because that code moved; re-run the repro on v0.22.0 and file only if it still panics. |
| Repro | [`repros/linear-col-layout-gradient`](repros/linear-col-layout-gradient) |
| Patch | [`patches/08-linear-col-layout-gradient.patch`](patches/08-linear-col-layout-gradient.patch) (`crates/burn-core/src/module/param/{base,tensor}.rs`) |
| Patch verified | Yes: column layout — first SGD step changes the weight false → true, gradient on passes 1..3 `[false, true, true]` → `[true, true, true]`, `load_record` from a live module panics → loads |
| Workarounds an upstream fix removes | `operations::init_linear` (`src/operations.rs`), which rebuilds every column-layout weight as a detached lazy leaf and detaches in its load mapper; `t5_linear` in `src/models/t5/model.rs`, which uses `LinearLayout::Row` to avoid the problem; the "untracked until the layout mapper finishes" ordering in `checkpoint::runtime::install_float_parameter` |

## Issue draft

**Title:** `LinearLayout::Col` weights get no gradient on the first backward pass, and `load_record` of a live autodiff record panics ("Can't convert a non leaf tensor into a tracked tensor")

### Describe the bug

`LinearConfig::new(d_in, d_out).with_layout(LinearLayout::Col).init::<Autodiff<B>>()`
draws the weight as `[d_out, d_in]` and registers an `init_mapper` that transposes it.
`Param::init_mapper` applies that function to the value the initializer returns, and the
initializer has already called `require_grad()`. The stored parameter is therefore the
transpose *node*, not a leaf:

1. Backward consumes a non-leaf's gradient, so `weight.grad(&grads)` is `None` after the
   first backward pass and `GradientsParams::from_grads` gives the optimizer nothing for
   the weight. The first optimizer step silently skips every column-layout weight. (Later
   passes happen to work, because the first one consumed the transpose step.)
2. `load_record` of a record taken from a live autodiff module (`model.into_record()`,
   e.g. to copy weights into a target network) applies the `load_mapper` transpose to the
   tracked record tensor and then calls `set_require_grad(true)` on the resulting non-leaf,
   which panics.

### To reproduce

Standalone project: `repros/linear-col-layout-gradient` (`burn = "=0.21.0"`, features
`flex`, `autodiff`). For both layouts it reports whether the first SGD step changes the
weight, whether the weight has a gradient on three consecutive passes, and whether
`config.init().load_record(other.into_record())` succeeds.

### Expected behavior

Both layouts behave identically: gradients on every pass, the first step updates the
weight, and loading a record works.

### Actual behavior

```text
Row layout
  first SGD step changes the weight: true
  weight gradient on passes 1..3: [true, true, true]
  load_record from a live module: ok, gradients [true, true, true]
Col layout
  first SGD step changes the weight: false
  weight gradient on passes 1..3: [false, true, true]
  load_record from a live module: panicked   (Can't convert a non leaf tensor into a tracked tensor)
```

### Root cause

- `crates/burn-core/src/module/param/base.rs`, `Param::init_mapper` composes
  `func(prev(device, require_grad))`, so the mapping is recorded after the leaf is
  created.
- `crates/burn-core/src/module/param/tensor.rs`, `Param::<Tensor>::transform_for_load`
  runs `mapper.on_load(tensor)` on a possibly tracked tensor and then
  `set_require_grad(expected_require_grad)` on the mapped non-leaf.

### Proposed fix

Map untracked values and make the mapped value the leaf:

```rust
// init_mapper
value.init = new_init_fn(move |device, require_grad| {
    let tensor = prev(device, false);
    func(tensor).set_require_grad(require_grad)
});

// transform_for_load
new_tensor = mapper.on_load(new_tensor).detach();
new_tensor = new_tensor.set_require_grad(expected_require_grad);
```

A loaded parameter is a new leaf by definition (the device-transfer branch already
detaches). Measured with the patch: every line of the column-layout output matches the
row layout.

### Environment

burn 0.21.0, Flex backend with `Autodiff`, Rust 1.94 nightly, Apple M3 Max, macOS 27.0.1.

## Notes for this crate

- v0.22.0's `init_mapper` matches the first half of the patch; once the crate upgrades,
  `init_linear`'s lazy re-wrapping for autodiff becomes unnecessary (its contiguous
  re-layout for inference Flex is a separate, performance-only concern), and T5 can use
  the column layout again.
