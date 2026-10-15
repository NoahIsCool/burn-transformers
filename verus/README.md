# Verus proofs of runtime code

The crate proves selected pure functions **in place**: the specification is
attached to the function that ships in `src/`, and Verus checks that exact body.
There is no separately maintained model to keep in sync, and the production
crate gains no dependency.

## How it works

Verus compiles its input with `--cfg verus_keep_ghost`. Ordinary `cargo`,
`rustc`, `rustfmt`, Clippy, and rust-analyzer never set that configuration, so
every proof attribute below disappears from normal builds:

```rust
#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

/// Splits a retained suffix into at most two physical ring intervals.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(split =>
    requires
        capacity > 0,
        count <= capacity,
    ensures
        split.first_count + split.second_count == count,
        split.physical_start + split.first_count <= capacity,
))]
fn split_ring_write(absolute_start: usize, count: usize, capacity: usize) -> RingWriteSplit {
    // ordinary Rust
}
```

- `verus_verify` marks a function or type for verification. Types returned by
  or passed to a verified function also need `verus_verify`.
- `verus_spec(result => requires … ensures …)` states the contract. Loop
  invariants use the same attribute on a `while` statement:
  `#[cfg_attr(verus_keep_ghost, verus_spec(invariant …, decreases …))]`.
- Items without `verus_verify` are treated as trusted external code. Keep
  verified cores free of Burn tensors, floating point, and allocator-dependent
  behavior; put such work in thin unverified wrappers around a verified core.
- Verus checks every machine-integer operation for overflow and underflow, so
  a verified body is also a proof that its arithmetic never panics.
- `usize::from(bool)`, iterator adaptors, and some standard helpers have no
  Verus specification. Prefer `if`, index `while` loops, and `min`/`max`.
- Loop invariants are attributes on `while` statements, which needs
  `#![cfg_attr(verus_keep_ghost, feature(proc_macro_hygiene))]` at the harness
  root. Only the Verus compiler sees it; ordinary builds are unaffected.
- Verus compiles verified bodies with an older edition, so they cannot use
  let-chains. Associated functions without a `self` receiver cannot carry
  `verus_spec`; make the verified core a free function and call it.
- Ghost snapshots of mutable state inside a body use `proof_decl!` under the
  same `cfg`; lemmas live in a `#[cfg(verus_keep_ghost)] verus! { … }` block
  next to the code they justify.

## Harnesses

Each `verus/*.rs` file other than this guide is an independent Verus crate root
for one area. It includes the real source files with `#[path = "../src/…"]`
attributes, mirroring their crate paths when they refer to one another. Areas
therefore verify independently and can be extended in parallel.

| Harness | Verified source | What the proofs establish |
| --- | --- | --- |
| `attention.rs` | `src/attention/plan.rs`, `src/attention/widths.rs` | Score-bounded query chunks are maximal, partition the queries, and respect both limits; sliding-window key ranges contain every visible key; chunk frames keep absolute chunk indices; grouped-query projection widths are exact and representable. |
| `cache.rs` | `src/cache/{ring,lengths,retention,rows}.rs` | Ring writes and reads land in slot `p mod capacity`; the length ledger keeps `committed <= working <= capacity` across advance, begin, commit, rollback, truncate, reorder, and reset; rewind predicates are sound and complete; row plans stay in bounds. |
| `checkpoint.rs` | `src/checkpoint/layout.rs`, `src/checkpoint/quantization/packing.rs` | Shape products and storage sizes are exact or rejected on overflow; accepted SafeTensors byte intervals tile the data section exactly; packed Q4/Q8 code reads decode the stored nibble or byte. |
| `experiment.rs` | `src/intervention/selection.rs`, `src/experiment/bounds.rs` | Activation selections expand to exactly the requested in-bounds ranges; position, label, broadcast, and staged-block bounds are overflow-free; the pairwise loss reduction terminates. |
| `generation.rs` | `src/generation/bookkeeping.rs` | Stop matching and finish precedence, prefill chunking, beam-row group preservation, the beam frontier, position preflights, speculative accepted-prefix arithmetic, and prompt-lookup drafting. |
| `models.rs` | `src/models/shared/plan.rs` | Padded-row compaction and scatter plans restore the rectangular layout; logits selection, implicit cached positions, rotary extents, value ranges, and per-row cache appends are exact and bounded. |
| `moe.rs` | `src/moe/assignment.rs` | The expert counting sort partitions every routed assignment by expert in stable order; memory-bounded dispatch blocks are maximal runs of whole experts. |
| `position.rs` | `src/position/{index_plan,t5,ids}.rs` | MRoPE axis selection, T5 displacement tables and bucket indices, and batched position IDs are in bounds and exact. |
| `tokenization.rs` | `src/tokenization/{layout,calendar}.rs` | Truncation budgets reserve exactly the template width; padding splits and masks are exact; calendar arithmetic for `strftime_now` is correct. |

Proof code must not use `admit`, `assume`, `assume_specification`,
`external_body`, or `external_fn_specification`; the `verus_harness`
integration test rejects them in harnesses and in every included source file.
Verification runs with `--no-cheating`, which rejects the same escape hatches.

## Running

Verus releases pin their own Rust toolchain and ship Z3. With the release
recorded below installed and its toolchain present (`rustup toolchain install
1.98.1-aarch64-apple-darwin --profile minimal`), run every harness from the
repository root:

```sh
for harness in verus/*.rs; do
  "$VERUS" --crate-type lib --no-cheating --triggers-mode silent "$harness" || exit 1
done
```

`VERUS` is the `verus` binary from the release archive. Verification performs
no Cargo build, downloads nothing, and writes no target directory.

| Item | Version |
| --- | --- |
| Verus | `0.2026.10.04.426d8b0` (`release/0.2026.10.04.426d8b0`, arm64 macOS) |
| Rust used by Verus | `1.98.1-aarch64-apple-darwin` |
| Z3 | `4.16.0` (bundled) |

## What is and is not proved

Each harness documents its verified functions, their postconditions, and the
runtime behavior that remains covered only by tests. A proof covers the integer
and indexing logic of the annotated body for all admissible inputs. It does not
cover Burn tensor kernels, floating-point numerics, allocation failure, or
unannotated callers; those remain the responsibility of the Rust test suite.
