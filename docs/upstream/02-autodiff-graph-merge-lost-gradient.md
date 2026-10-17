# 02. burn-autodiff: a graph merge leaves the merged origin pointing at an emptied graph

| | |
|---|---|
| Upstream | [tracel-ai/burn](https://github.com/tracel-ai/burn), crate `burn-autodiff` |
| Version reproduced | 0.21.0 (this crate's pin) and **0.22.0** |
| Kind | Correctness: 0.21.0 silently returns no gradient; 0.22.0 panics in a valid `backward()` |
| Upstream status, checked 2026-10-16 | **Open.** v0.22.0 reproduces it (measured below); `crates/burn-autodiff/src/runtime/graph.rs` on `main` at `df0812e` is identical to v0.22.0. **File this one first.** |
| Related upstream issues | [burn#5573](https://github.com/tracel-ai/burn/issues/5573) (a concurrent `backward()` frees another thread's steps; closed by [#5647](https://github.com/tracel-ai/burn/pull/5647), which v0.22.0 includes) has the same 0.21 symptom but a different cause; this one is single-threaded and independent of `cleanup_orphaned_entries`. [burn#5879](https://github.com/tracel-ai/burn/issues/5879) (closed) produced the same "tape has already been consumed" panic through a same-device `to_device`, a different trigger. |
| Repro | [`repros/v0.22.0/autodiff-graph-merge-lost-gradient`](repros/v0.22.0/autodiff-graph-merge-lost-gradient) (to file); [`repros/autodiff-graph-merge-lost-gradient`](repros/autodiff-graph-merge-lost-gradient) (0.21.0) |
| Patch | [`patches/02-autodiff-graph-merge-lost-gradient.patch`](patches/02-autodiff-graph-merge-lost-gradient.patch): one line plus a comment; applies unchanged to v0.21.0, v0.22.0 and `main` |
| Patch verified | Yes, both versions, 1,000 trials per creation order. v0.22.0: 1,000 of 1,000 backward passes panic with the side leaves created first → 0 panics, 0 lost gradients. v0.21.0: 255 and 218 of 1,000 lose the gradient (two orders, one run) → 0 and 0. This crate's old `module::linear` router regression test fails without the patch and passes with it (0.21.0). |
| Workarounds an upstream fix removes | The FP32 router projection in `src/moe/router.rs` (`router_logits`) avoids `burn::tensor::module::linear` only to dodge this bug; keep the regression tests `router_projection_receives_gradients_on_every_logits_only_backward_pass` (`src/moe_tests.rs`) and `auxiliary_loss_alone_reaches_the_router_on_every_pass` (`src/models/deepseek_v2/model/moe_tests.rs`) |

## Issue draft

**Title:** Autodiff: valid `backward()` panics with "graph tape has already been consumed" after two graph merges (`GraphLocator::merge_two` does not re-key the merged origin)

### Describe the bug

A single-threaded program can make `backward()` panic with

```text
Cannot run backward: graph tape has already been consumed at NodeId(..). Burn does not
support retain_graph. ...
```

although no tape was reused. It takes an op whose parents' steps were already consumed by
an earlier backward pass (for example the first op on a parameter in the second training
step), followed by two graph merges in which that op's graph is the merged one both times.
In 0.21.0 the same situation did not panic: `Tensor::grad` silently returned `None` for the
parameter, and an optimizer skipped it.

We hit it in a mixture-of-experts router: `module::linear(x, w, Some(b))` produced the
router logits, the routed dispatch merged the logits' graph with the experts' graphs, and
a loss on the logits alone (auxiliary or z-loss) gave `w` a gradient on the first step and
on only some later ones.

### To Reproduce

Standalone project `repros/v0.22.0/autodiff-graph-merge-lost-gradient` (`burn =
"=0.22.0"`, features `flex`, `autodiff`), `cargo run --release`:

```rust
let device = Device::flex().autodiff();
let leaf = || Tensor::<2>::random([4, 4], Distribution::Default, &device).require_grad();
let w = leaf();
let q = leaf(); // created before `a`, so these graph origins have smaller node ids
let p = leaf();
let _ = w.clone().sum().backward(); // consumes w's leaf step

let a = w.clone() * 2.0;         // no registered parent: a new graph with origin `a`
let _side = (a.clone() * p) * q; // a's graph is merged into p's, then p's into q's

let grads = a.sum().backward();  // panics
assert!(w.grad(&grads).is_some());
```

The repro runs 1,000 trials with the side leaves created after `a` and 1,000 with them
created before it, catching panics.

### Expected behavior

Every backward pass succeeds and `w.grad(&grads)` is `Some`.

### Actual behavior (Apple M3 Max, burn 0.22.0)

```text
side leaves created after `a`: 0 of 1000 backward passes returned no gradient for `w`, 0 panicked
side leaves created before `a`: 0 of 1000 backward passes returned no gradient for `w`, 1000 panicked
```

On 0.21.0, where `merge` picked the main graph in `HashMap` order, the same program lost
the gradient without panicking in 7-27% of trials per 1,000-trial run (eight runs over
both creation orders).

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
after `merge_two` the merged origin maps to `main` in `graphs` but is not one of `main`'s
keys. When `main` is later merged into a third graph, every key of `main` is re-mapped,
but the old merged origin keeps pointing at `main`'s `Graph`, whose state was just moved
out. The next op on that origin tensor (`a.sum()` above) registers its step in the
emptied graph, so the backward pass started there cannot find `a`'s step: 0.22.0's tape
check reports it as consumed, and 0.21.0's traversal skipped it.

In the MoE case, tracing node ids showed exactly this: the logits node (137) was the
origin of a graph merged into graph 45, graph 45 was merged into graph 100, and the loss
ops registered in the emptied graph 45.

Since v0.22.0 `merge` keeps the graph with the smallest origin id (`BTreeMap`), so the
outcome is deterministic: it fails whenever the reused origin is younger than the two
graphs it is merged into.

### Proposed fix

Record the merged origin as a key of the main graph, so later merges re-map it too:

```diff
         // Re-map merged origin to the main graph
         self.graphs.insert(merged.origin, main.clone());
+        self.register_key(main.origin, merged.origin);
```

`remove_entry` already removes a node from its graph's key set, so the extra key is
cleaned up like any other. With the patch, v0.22.0: 0 panics and 0 missing gradients in
2 × 1,000 trials; 0.21.0: 0 missing gradients. Burn's own test suites were not run (the
crates.io package of burn-autodiff ships no unit tests).

### Environment

burn 0.22.0 (also 0.21.0), Flex backend, Rust 1.99.1, Apple M3 Max, macOS 27.0.1. Single
threaded.

## Notes for this crate

- The router workaround avoids the trigger (its logits are no longer a graph origin), not
  the bug. Any other graph whose origin is reused after two merges can still lose
  gradients on 0.21.0 (or panic on 0.22.0); the one-line patch can be applied via `[patch]`
  in training experiments until upstream fixes it.
- Crate-level check (0.21.0): a scratch copy of the crate with the old `module::linear` FP32
  router restored failed the router regression test against crates.io `burn-autodiff` and
  passed against the patched copy.
