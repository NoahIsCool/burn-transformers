# Ring-write arithmetic verification

`cache_ring.rs` is a standalone Verus model of the scalar formulas in
`src/cache/ring.rs`, reviewed against source commit
`a9f1dc64799ec004f6fee46e1935d821d2d1f2fa`. It changes no runtime code or Cargo
dependencies. Recheck this correspondence when the production formulas change.

## What is verified

`prepare_retained_input` reproduces the checked append arithmetic from
`plan_ring_write`. For start position `a`, token count `n`, and positive ring
capacity `c`, it returns `None` exactly when `a + n` exceeds `usize::MAX`.
Otherwise it retains `k = min(n, c)` newest tokens, with input start `n - k`
and absolute start `a + n - k`.

`split_ring_write` reproduces `write_segments` using these formulas:

```text
p = absolute_start % c
f = min(k, c - p)
s = k - f

first input:   [input_start, input_start + f)
second input:  [input_start + f, input_start + k)
first output:  [p, p + f)
second output: [0, s)
```

The proof requires `c > 0`, `k <= c`, and
`input_start + k <= usize::MAX`, with the last sum interpreted mathematically.
The verified executable body establishes that all machine additions,
subtractions, and the remainder operation are valid. Its postconditions show:

- The input intervals are adjacent and partition the entire retained suffix.
- Each input interval has the same length as its destination interval.
- Both destinations fit inside the ring. Since `s <= p`, they are disjoint.
- Zero count gives two zero-length intervals. Positive count gives a nonempty
  first interval; a nonempty second interval implies the first reaches `c`.
- When `k = c`, the destinations together cover every physical slot exactly
  once, including wrapped writes.

`checked_write_partition` verifies composition of these two model functions.
For every non-overflowing append, the retained input ends at `n` and its
destinations are bounded and disjoint. The proof applies to all admissible
`usize` inputs, including zero tokens at `usize::MAX` and a full write with
`n = c = usize::MAX`; it does not allocate those many elements.

## Runtime correspondence and trusted boundary

| Verus model | Production code | Correspondence |
| --- | --- | --- |
| `prepare_retained_input` | `plan_ring_write` | Same checked sum, minimum, and suffix subtractions; `None` models `PositionOverflow`. |
| `split_ring_write` | `write_segments` | Same remainder, minimum, segment lengths, and range endpoints. The model spells out `usize::min` as a conditional. |
| `checked_write_partition` | Their composition in `plan_ring_write` | Shows that successful public-planner arithmetic establishes the split's preconditions. |

This is a separately maintained exact-integer model, not a proof of the
compiled library. Correspondence with the production functions is a manual
review obligation. It does not verify production `Vec` construction, omission
of empty intervals, allocator behavior, the correspondence of each destination
with every token's absolute position modulo capacity, mirrored-ring offsets,
cache transaction coordination, Burn tensor operations, or floating-point
behavior. Those runtime behaviors require separate library tests and reviews;
this proof does not replace them or claim their results. The model also does
not verify the fields of the production overflow-error payload.

The proof contains no `assume`, `admit`, or `external_body`. It relies on Verus,
the preverified `vstd` specifications of Rust machine arithmetic, and Z3.

## Reproduce the verification

Verified on 2026-09-12 with:

- Verus `0.2026.09.06.8dea4a2`, release profile, `macos_aarch64`.
- Rust `1.98.0 (88d9e12ae 2026-08-18)`, `aarch64-apple-darwin`.
- Z3 `4.16.0`, 64-bit.

From the repository root, the exact local command was:

```sh
RUSTUP_TOOLCHAIN=1.98.0 \
VERUS_Z3_PATH=/private/tmp/z3-4.16.0-arm64-osx/z3-4.16.0-arm64-osx-15.7.3/bin/z3 \
/private/tmp/verus-0.2026.09.06.8dea4a2/verus-arm64-macos/verus \
  --crate-type lib --no-cheating verus/cache_ring.rs
```

Recorded result for this artifact:

```text
verification results:: 3 verified, 0 errors
```

The temporary installation paths are local to that run. For another machine,
use the same installed versions and substitute its Verus and Z3 paths.

Formatting and whitespace checks:

```sh
rustup run 1.98.0 rustfmt --edition 2024 --check verus/cache_ring.rs
git diff --check
```

This file is not a Cargo test target and therefore has no ordinary Rust test
harness. Verus checks its three executable model functions for all admissible
inputs, including both branches of the minimum and checked-add failure. No
line or branch coverage percentage was measured or inferred from that proof.
The verification-only invocation omits `--compile`, performs no Cargo build or
download, and produces no binary or target directory. This bounded check used
the existing installations with 14 GiB free locally; the repository's 20 GiB
preflight remains required before normal builds.
