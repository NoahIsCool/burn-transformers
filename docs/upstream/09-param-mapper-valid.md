# 09. burn-core: `AutodiffModule::valid()` and `from_inner()` drop parameter save/load mappers

| | |
|---|---|
| Upstream | [tracel-ai/burn](https://github.com/tracel-ai/burn), crate `burn-core` (`module::param`) |
| Version reproduced | 0.21.0 (this crate's pin) |
| Kind | Correctness (records saved after `valid()` have the wrong layout; square projections reload transposed without any error) |
| Upstream status, checked 2026-10-16 | Appears **fixed on `main`**: `Param` conversion now passes `param_mapper` through (`valid()` / `train()` keep it). Re-run the repro on v0.22.0; file only if it still fails. |
| Repro | [`repros/param-mapper-valid`](repros/param-mapper-valid) |
| Patch | [`patches/09-param-mapper-valid.patch`](patches/09-param-mapper-valid.patch) (`crates/burn-core/src/module/param/{base,tensor}.rs`) |
| Patch verified | Yes: square column-layout projection saved after `valid()` and reloaded, max output error 0.71-0.83 → 0; saved weight shape `[in, out]` → `[out, in]` |
| Workarounds an upstream fix removes | The rule in `src/training.rs` (module docs) to save records only from the original training owner, never from a `valid()` or `from_inner()` copy |

## Issue draft

**Title:** `valid()` / `from_inner()` drop `Param` save/load mappers, so records saved afterwards have the wrong layout (`LinearLayout::Col` weights saved transposed)

### Describe the bug

`impl AutodiffModule<B> for Param<Tensor<B, D>>` rebuilds the parameter with
`Param::initialized(id, tensor)` in both directions, which resets `param_mapper` to the
default. A `LinearLayout::Col` weight relies on its mappers: it is held as `[in, out]`,
saved as `[out, in]`, and transposed back on load. After `valid()` (evaluation) or
`from_inner()` the module saves `[in, out]`, and loading that record into a column-layout
module transposes it again. A non-square projection then has the wrong shape; a square one
silently computes with `Wᵀ`.

### To reproduce

Standalone project: `repros/param-mapper-valid` (`burn = "=0.21.0"`, features `flex`,
`autodiff`). It saves a column-layout `Linear` through `BinBytesRecorder` from the training
owner, from `owner.valid()`, and from `Linear::from_inner(owner.valid())`, reloads each
into a fresh inference module, and compares outputs with the owner's.

### Expected behavior

All three records reload to the same function; the saved weight has the checkpoint layout
`[out, in]`.

### Actual behavior

```text
Col-layout 4 -> 4 projection, saved and reloaded into a fresh inference module:
  training owner (Autodiff) record           max |y - expected| = 0.00e0
  owner.valid() record                       max |y - expected| = 7.07e-1
  Linear::from_inner(owner.valid()) record   max |y - expected| = 7.07e-1
Col-layout 4 -> 3 projection, saved weight shape (checkpoint layout is [out, in] = [3, 4]):
  training owner record                      [3, 4]
  owner.valid() record                       [4, 3]
  Linear::from_inner(owner.valid()) record   [4, 3]
```

### Root cause

`crates/burn-core/src/module/param/tensor.rs`, `AutodiffModule::valid` and `from_inner`
for `Param<Tensor<B, D>>` construct a new `Param` without the mapper. The mapper closures
are typed on `Tensor<B, D>` versus `Tensor<B::InnerBackend, D>`, so they cannot be moved
across as is.

### Proposed fix

Lift the mappers through the backend conversion:

```rust
impl<T: Parameter + 'static> ParamMapper<T> {
    pub(crate) fn lift<U: Parameter + 'static>(&self, into: fn(U) -> T, back: fn(T) -> U) -> ParamMapper<U> {
        let lift = |mapper: &Mapper<T>| -> Mapper<U> {
            let mapper = mapper.clone();
            new_mapper(move |value: U| back(mapper(into(value))))
        };
        ParamMapper { load: self.load.as_ref().map(lift), save: self.save.as_ref().map(lift) }
    }
}

// valid():
param.param_mapper = self.param_mapper.lift(Tensor::from_inner, Tensor::inner);
// from_inner():
param.param_mapper = module.param_mapper.lift(Tensor::inner, Tensor::from_inner);
```

Mappers run only at save and load time, where passing through `from_inner`/`inner`
(detaching) is harmless. Measured with the patch: all three routes reload with error 0 and
save `[3, 4]`.

### Environment

burn 0.21.0, Flex backend with `Autodiff`, Rust 1.94 nightly, Apple M3 Max, macOS 27.0.1.

## Notes for this crate

- The same lifting would apply to `Param<Tensor<B, D, Int>>` and `Bool` if they ever carry
  mappers; the patch covers float parameters, the only ones with layout mappers in Burn.
