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

## Harnesses

Each `verus/*.rs` file other than this guide is an independent Verus crate root
for one area. It includes the real source files with `#[path = "../src/…"]`
attributes, mirroring their crate paths when they refer to one another. Areas
therefore verify independently and can be extended in parallel.

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
