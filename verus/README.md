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
- Quantifiers over one array at a time (`forall|i| … a[i] …`) verify more
  reliably than quantifiers relating several arrays; state cross-array facts
  through a spec function of one index.

## Harnesses

Each `verus/*.rs` file other than this guide is an independent Verus crate root
for one area. It includes the real source files with `#[path = "../src/…"]`
attributes, mirroring their crate paths when they refer to one another. Areas
therefore verify independently and can be extended in parallel. Each
harness's module documentation lists its verified functions, their
postconditions, and what remains covered only by Rust tests.

The counts are the verified obligations Verus reported for each harness at
source `33f620f` with the release recorded below (408 in total, 0 errors).

| Harness | Verified source | Verified | What the proofs establish |
| --- | --- | ---: | --- |
| `attention.rs` | `src/attention/{plan,widths}.rs` | 35 | Score-bounded query chunks are maximal, partition the queries, and respect both limits; sliding-window key ranges contain every visible key; chunk frames keep absolute chunk indices; grouped-query projection widths are exact and representable. |
| `cache.rs` | `src/cache/{ring,lengths,retention,rows}.rs` | 56 | Ring writes and reads land in slot `p mod capacity`; the length ledger keeps `committed <= working <= capacity` across advance, begin, commit, rollback, truncate, reorder, and reset; rewind predicates are sound and complete; row runs are ordered, disjoint, and cover exactly the rows with new tokens. |
| `checkpoint.rs` | `src/checkpoint/layout.rs`, `src/checkpoint/quantization/packing.rs` | 35 | Shape products and storage sizes are exact or rejected on overflow; accepted SafeTensors byte intervals tile the data section exactly; packed Q4/Q8 code reads decode the stored nibble or byte. |
| `custom.rs` | `src/models/custom/plan.rs` | 9 | The clamped T5 displacement table has radius `min(max_distance, C - 1)`, its device index is in range and inverts slot to displacement; a row append fills slots `L..L + n` in order and reports overflow. |
| `deepseek_muse.rs` | `src/models/deepseek_v2/model/batch.rs`, `src/models/muse_glimmer/schedule.rs` | 18 | DeepSeek batched-row gathers, key extents, monotone per-chunk key limits, and cache slots; Muse's backward-counted global-layer rule, dense per-kind cache slots, and cached ring chunk key slices that contain every visible key and at most `H + W - 1 + rows` keys. |
| `experiment.rs` | `src/intervention/selection.rs`, `src/experiment/bounds.rs` | 15 | Activation selections expand to exactly the requested in-bounds ranges; position, label, broadcast, and staged-block bounds are overflow-free; the pairwise loss reduction terminates. |
| `gemma.rs` | `src/models/gemma/plan.rs` | 5 | The default local/global schedule equals `(l + 1) mod p != 0` for every period; every layer's rotary set index names its layer type's set and is in range. |
| `generation.rs` | `src/generation/bookkeeping.rs` | 22 | Stop matching and finish precedence, prefill chunking, beam-row group preservation, the beam frontier, position preflights, speculative accepted-prefix arithmetic, and prompt-lookup drafting. |
| `gpt_oss.rs` | `src/checkpoint/mxfp4.rs`, `src/models/gpt_oss/plan.rs` | 46 | MXFP4 shape validation is sound and complete (`Ok` exactly for valid `[E, R, G, 16]` blocks with `[E, R, G]` scales), offsets are in bounds, nibbles split correctly, and each decoded value's binary64 bits are exact; the sliding/full schedule, cache slots, and interleaved `gate_up` offsets are exact. |
| `integration.rs` | `src/models/auto/defaults.rs`, `src/models/custom/mamba2_geometry.rs` | 4 | Hugging Face's sampling `top_k` default (an explicit zero disables it; otherwise `min(k or 50, vocabulary)`, always in `1..=vocabulary`); Mamba-2 head grouping exists exactly when the group size divides the heads. |
| `llama.rs` | `src/models/llama/plan.rs` | 20 | SmolLM3 NoPE and Qwen sliding-layer rules, dense cache-group slots, contiguous-run detection, batched-step cache positions, and the compaction and restoring gathers. |
| `models.rs` | `src/models/shared/plan.rs` | 24 | Padded-row compaction and scatter plans restore the rectangular layout; logits selection, implicit cached positions, rotary extents, value ranges, and per-row cache appends are exact and bounded. |
| `moe.rs` | `src/moe/assignment.rs` | 18 | The expert counting sort partitions every routed assignment by expert in stable order; memory-bounded dispatch blocks are maximal runs of whole experts. |
| `nemotron.rs` | `src/models/shared/plan.rs`, `src/models/nemotron_h/model/plan.rs` | 51 | Nemotron-H compact-batch gather, scatter, pack, unpack, and last-token maps; query positions that keep every padded slot inside its row's keys; the Mamba-2 path choice, transactional staging policy, shifted MTP cursor, and one-pass verification length. |
| `perf.rs` | `src/checkpoint/runtime/tiles.rs` | — | The multi-threaded tiled transpose splits columns into whole-block bands that cover every column exactly once, never use more threads than requested, and never index past the matrix; ceiling division is overflow-free. |
| `position.rs` | `src/position/{index_plan,t5,ids}.rs` | 12 | MRoPE axis selection, T5 displacement tables and bucket indices, and batched position IDs are in bounds and exact. `relative_table_index` is a verified specification of the device T5 index, linked to it by a unit test. |
| `qwen.rs` | `src/models/qwen3_5/plan.rs`, `src/models/qwen3_moe/plan.rs` | 8 | The hybrid DeltaNet/attention schedule rule, dense per-kind cache slots, the compact right-padded row plan, the MTP cursor after truncation, and the Qwen3-MoE sparse-layer predicate. |
| `t5_gpt.rs` | `src/models/t5/model/packing.rs` | 12 | Batched T5 decoding's left-aligned compaction and scatter gathers stay in bounds, every block slot's causal visibility and bias position equal their formulas, and `groups_preserved` characterizes the beam reorders that may leave encoder state in place. |
| `tokenization.rs` | `src/tokenization/{layout,calendar}.rs` | 18 | Truncation budgets reserve exactly the template width; padding splits and masks are exact; calendar fields for `strftime_now` are overflow-free and in range. |

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
no Cargo build, downloads nothing, and writes no target directory. The whole
loop takes about 15 seconds on an Apple M3 Max.

Without Verus installed, `cargo test --locked --test verus_harness` still
checks the harness structure: every harness begins with module documentation
and includes existing sources, no harness or included source uses an escape
hatch, and every source file carrying Verus specifications is included by some
harness.

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
