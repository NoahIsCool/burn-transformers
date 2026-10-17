# 02. burn-autodiff: backward silently drops gradients after a graph merge re-maps a merged origin

| | |
|---|---|
| Upstream | [tracel-ai/burn](https://github.com/tracel-ai/burn), crate `burn-autodiff` |
| Version reproduced | 0.21.0 (this crate's pin) |
| Kind | Correctness (silent, nondeterministic loss of parameter gradients) |
| Upstream status, checked 2026-10-16 | **Not fixed** in v0.22.0 or on `main`: `GraphLocator::merge_two` still does not register the merged origin. File this one. |
| Repro | [`repros/autodiff-graph-merge-lost-gradient`](repros/autodiff-graph-merge-lost-gradient) |
| Patch | [`patches/02-autodiff-graph-merge-lost-gradient.patch`](patches/02-autodiff-graph-merge-lost-gradient.patch) (one line plus comment, `crates/burn-autodiff/src/runtime/graph.rs`) |
| Patch verified | Yes: 75-269 of 1,000 trials lose the gradient unpatched (varies per run), 0 of 1,000 patched; this crate's old `module::linear` router fails its regression test at step 1 unpatched and passes with the patched `burn-autodiff` |
| Workarounds an upstream fix removes | The FP32 router projection in `src/moe/router.rs` (`router_logits`) avoids `burn::tensor::module::linear` only to dodge this bug; regression tests `router_projection_receives_gradients_on_every_logits_only_backward_pass` (`src/moe_tests.rs`) and `auxiliary_loss_alone_reaches_the_router_on_every_pass` (`src/models/deepseek_v2/model/moe_tests.rs`) should stay |

## Issue draft

**Title:** Autodiff: gradients silently missing after graph merges (`GraphLocator::merge_two` does not re-map the merged graph's origin)

### Describe the bug

On `Autodiff<B>`, `Tensor::grad` sometimes returns `None` for a parameter that the loss
clearly depends on, so an optimizer silently skips it. It is nondeterministic (it depends
on `HashMap` iteration order inside the graph locator) and needs no unusual API: it
happens to an op whose parents' graphs were already cleaned up by an earlier backward
pass (for example the first op on a parameter in a training step after the first), once
that op's graph has been merged twice.

We hit it in a mixture-of-experts router: `module::linear(x, w, Some(b))` produced the
router logits, the routed dispatch merged the logits' graph with the experts' graphs, and
a loss on the logits alone (auxiliary or z-loss) gave `w` a gradient on the first step
and on only a fraction of later steps.

### To reproduce

Standalone project: `repros/autodiff-graph-merge-lost-gradient` (`burn = "=0.21.0"`,
features `flex`, `autodiff`).

```rust
type B = Autodiff<Flex>;
let w = Tensor::<B, 2>::random([4, 4], Distribution::Default, &device).require_grad();
let _ = w.clone().sum().backward();           // consumes w's leaf step

let a = w.clone() * 2.0;                      // no registered parent graph: new graph, origin `a`
let p = Tensor::<B, 2>::random([4, 4], Distribution::Default, &device).require_grad();
let q = Tensor::<B, 2>::random([4, 4], Distribution::Default, &device).require_grad();
let _side = (a.clone() * p) * q;              // merges a's graph twice

let grads = a.sum().backward();
assert!(w.grad(&grads).is_some());            // fails in ~1 of 4 trials
```

### Expected behavior

`w.grad(&grads)` is `Some` on every trial.

### Actual behavior

`cargo run --release` (1,000 trials per run) printed 189, 163, 75, 269 and 160 lost
gradients in five runs on an Apple M3 Max.

### Root cause

`crates/burn-autodiff/src/runtime/graph.rs`, `GraphLocator::merge_two`:

```rust
// Re-map merged origin to the main graph
self.graphs.insert(merged.origin, main.clone());

// Move all keys (node IDs) from the merged graph to the main graph
if let Some(locator_keys) = self.keys.remove(&merged.origin) {
    for k in locator_keys.iter() {
        self.graphs.insert(*k, main.clone());
    }
    ...
    locator_keys_main.extend(locator_keys);
}
```

`keys[origin]` never contains the origin itself (`register_key` skips `origin == key`), so
after `merge_two` the merged origin is mapped to `main` in `graphs` but is not one of
`main`'s keys. When `main` is later merged into a third graph, every key of `main` is
re-mapped, but the old merged origin keeps pointing at `main`'s `Graph` object, whose
state was just moved out. The next op on that origin tensor registers its step in the
emptied graph, and a backward pass started from it cannot reach the origin's step
(`BreadthFirstSearch::traverse` skips parents without a step), so every leaf behind it
gets no gradient. Tracing node ids in the MoE case showed exactly this: the logits node
(`137`) was the origin of a graph merged into `45`, `45` was merged into `100`, and the
loss ops registered into the dead graph `45`.

Which graph becomes `main` in `merge` depends on `HashMap::drain` order, hence the
nondeterminism.

### Proposed fix

Record the merged origin as a key of the main graph, so later merges re-map it too:

```diff
         // Re-map merged origin to the main graph
         self.graphs.insert(merged.origin, main.clone());
+        self.register_key(main.origin, merged.origin);
```

With the patch: 0 of 1,000 trials lose the gradient (repeated three times), and the
original MoE regression test (24 consecutive logits-only and load-balancing backward
passes through `module::linear`) passes. `remove_entry` already removes a node from its
graph's key set, so the extra key is cleaned up like any other.

### Environment

burn 0.21.0 (also present on `main` at the time of writing), Flex backend, Rust 1.94
nightly, Apple M3 Max, macOS 27.0.1.

## Notes for this crate

- The router workaround avoids the trigger (its logits are no longer a graph origin), not
  the bug. Any other graph whose origin is reused after two merges can still lose
  gradients on Burn 0.21/0.22; consider applying this one-line patch via `[patch]` in
  training experiments until upstream fixes it.
- Verification of the crate-level symptom used a scratch copy of the crate with the old
  `module::linear` FP32 router restored: the regression test failed at step 1 against
  crates.io `burn-autodiff` and passed against the patched copy.
