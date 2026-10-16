# Verus proofs of runtime code

The crate proves selected pure functions **in place**: the specification is
attached to the function that ships in `src/`, and Verus checks that exact
body. There is no separately maintained model to keep in sync, and the
production crate gains no dependency.

## How it works

Verus compiles its input with `--cfg verus_keep_ghost`. Ordinary `cargo`,
`rustc`, `rustfmt`, Clippy, and rust-analyzer never set that configuration,
so every proof attribute, ghost block, and lemma below disappears from
normal builds. `src/checkpoint/division.rs`, abridged, shows the shape of a
verified source file:

```rust
#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

#[cfg(verus_keep_ghost)]
verus! {

/// Specification: the exact rounded-up quotient `⌈dividend / divisor⌉` for `divisor > 0`.
pub open spec fn ceiling(dividend: int, divisor: int) -> int {
    if dividend % divisor == 0 { dividend / divisor } else { dividend / divisor + 1 }
}

/// `⌈n / d⌉` is the least `q` with `n <= q · d`; it never exceeds `n` and is positive for `n > 0`.
pub proof fn lemma_ceiling_bounds(dividend: int, divisor: int)
    requires
        dividend >= 0,
        divisor >= 1,
    ensures
        dividend <= ceiling(dividend, divisor) * divisor,
        0 <= ceiling(dividend, divisor) <= dividend,
        // ...
{
    vstd::arithmetic::div_mod::lemma_fundamental_div_mod(dividend, divisor);
    // ... `assert(...) by (nonlinear_arith) requires ...;` steps
}

} // verus!

/// Returns `⌈value / divisor⌉` without forming the overflow-prone `value + divisor - 1`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(quotient =>
    requires
        divisor > 0,
    ensures
        quotient == ceiling(value as int, divisor as int),
        value <= quotient * divisor,
        quotient == 0 || (quotient - 1) * divisor < value,
        value > 0 ==> quotient > 0,
        quotient <= value,
))]
pub(crate) const fn ceil_div_u64(value: u64, divisor: u64) -> u64 {
    #[cfg(verus_keep_ghost)]
    proof! {
        lemma_ceiling_bounds(value as int, divisor as int);
    }
    // `div_ceil` has no Verus specification, so the rounding is spelled out.
    let remainder = value % divisor;
    value / divisor + if remainder == 0 { 0 } else { 1 }
}
```

The proofs are written with five constructs:

- **`verus_verify`** marks an item for verification: a function, method,
  constant, struct, or enum. Types returned by or passed to a verified
  function, and constants it reads, also need it.
- **`verus_spec(result => requires … ensures …)`** states a function's
  contract on the shipped body. Loop invariants use the same attribute on a
  `while` statement:
  `#[cfg_attr(verus_keep_ghost, verus_spec(invariant …, decreases …))]`.
- **`proof! { … }`** blocks, under `#[cfg(verus_keep_ghost)]`, hold ghost
  steps inside an executable body: lemma calls, `assert(…)`, and
  `assert(…) by (nonlinear_arith)`, `by (bit_vector)`, or
  `by (compute_only)` for facts the default solver needs help with.
  **`proof_decl!`** declares ghost variables there, such as a snapshot of
  mutable state before a loop (`let ghost before = flags@;`).
- **`verus! { … }`** blocks, under `#[cfg(verus_keep_ghost)]`, define the
  `spec fn` vocabulary that contracts are written in (reference formulas
  such as `spec_bucket` or `lexer_state`) and the `proof fn` lemmas that
  bodies call.
- **`#[cfg(any(test, verus_keep_ghost))]`** keeps an executable reference
  function out of every non-test build, debug or release: only `cargo test`
  and Verus compile it, so Verus proves it and unit tests compare the
  production path against it (`transposed_offset` in
  `src/checkpoint/mxfp4.rs`, `displacement_slot` in
  `src/models/custom/plan.rs`).

Rules that follow from Verus's semantics:

- Crate items without `verus_verify` are outside the proof: Verus ignores
  them, and verified code cannot call them; standard-library calls need a
  `vstd` specification. Keep verified cores free of Burn
  tensors, floating point, I/O, and allocator-dependent behavior; put such
  work in thin unverified wrappers around a verified core.
- Verus checks every machine-integer operation for overflow and underflow
  and every slice index for bounds, so a verified body is also a proof that
  its arithmetic and indexing never panic.
- `usize::from(bool)`, iterator adaptors, `div_ceil`, `is_ascii_*`,
  `RangeInclusive::contains`, and some other standard helpers have no Verus
  specification. Prefer `if`, index `while` loops, explicit comparisons, and
  `min`/`max`; the affected Clippy lints are allowed locally with a comment
  saying why. `is_multiple_of` does have one (`n.is_multiple_of(d)` is
  `n % d == 0`, and `n == 0` for `d == 0`), so new verified code can use it.
  Older bodies, including the `ceil_div_u64` excerpt above, still compare a
  remainder with zero, and some comments and three
  `clippy::manual_is_multiple_of` allowances still claim it has no
  specification. Those predate this correction and are being replaced.
- Loop invariants are attributes on `while` statements, which needs
  `#![cfg_attr(verus_keep_ghost, feature(proc_macro_hygiene))]` at the
  harness root. Only the Verus compiler sees it; ordinary builds are
  unaffected.
- Verus compiles verified bodies with an older edition, so they cannot use
  let-chains. Associated functions without a `self` receiver cannot carry
  `verus_spec`; make the verified core a free function and call it (for
  example `empty_ledger` behind `LengthLedger::new`).
- Quantifiers over one sequence at a time (`forall|i| … a[i] …`) verify more
  reliably than quantifiers relating several sequences; state cross-sequence
  facts through a spec function of one index, and prefer explicit
  `#[trigger]` annotations to automatic trigger selection.

## Harnesses

Each `verus/*.rs` file other than this guide is an independent Verus crate
root for one area. It includes the real source files with
`#[path = "../src/…"]` attributes; when an included file refers to another
by its crate path (`crate::checkpoint::division`,
`crate::models::shared::plan`), the harness re-exports the included module
under that path. Areas therefore verify independently and can be extended
in parallel. Two sources are shared and re-verified by both of their
harnesses: `src/checkpoint/division.rs` (`checkpoint.rs`, `perf.rs`) and
`src/models/shared/plan.rs` (`models.rs`, `nemotron.rs`). Each harness's
module documentation lists its verified functions, their postconditions,
and what remains covered only by Rust tests.

The 25 harnesses include 87 distinct source files, which are exactly the
files under `src/` that contain Verus-only code (`verus_keep_ghost`). The
counts are the "verified" totals Verus reported for each harness on source
revision `fbcdb12` with the release recorded below: **1,877 verified, 0 errors**.
`scripts/check_release.py --gate verus` fails when a harness reports a
different count, so update this table with the proofs.

| Harness | Verified sources | Verified | What the proofs establish |
| --- | --- | ---: | --- |
| `attention.rs` | `src/attention/{geometry,grouping,plan,rules,widths}.rs`, `src/feed_forward/widths.rs` | 106 | Operand and projected-layer shape validators accept exactly the documented shapes and name the first failed check; grouped-query head groups are exact and folding `[b, H, rows, x]` to `[b, G, g·rows, x]` is a bijection that keeps every flat index; differential attention reads each head's reference value pair; query chunks have an exact peak size, respect both limits, are the largest admitted, and partition the rows; sliding-window key ranges are exactly the union of their rows' visible keys; reduced window and chunk rules exclude exactly the absolute rule's links; projection widths are exact or rejected in check order. |
| `cache.rs` | `src/cache/{lengths,retention,ring,rows}.rs`, `src/cache/tensor/{plan,recurrent/plan,kv/plan}.rs`, `src/cache/tensor/kv/batch/{layout,slots}.rs`, `src/cache/tensor/state/step/coverage.rs` | 199 | Every ring write, mirror, and read uses slot `p mod capacity`, and writes keep exactly the newest `min(n, capacity)` tokens; the length ledger keeps `committed <= working <= capacity` across advance, begin, commit, rollback, truncate, reorder, and reset, rejecting exactly rule violations with the first offending row; rewind predicates are sound and complete; row runs are ordered, disjoint, and cover exactly the rows with new tokens; allocation counts, KV read/write segments, batched layouts, slot tables, snapshot indices, and step coverage are exact, so an accepted step writes every active row exactly once. |
| `checkpoint.rs` | `src/checkpoint/{division,layout,lexical}.rs`, `src/checkpoint/dtype/scalar.rs`, `src/checkpoint/mapping/{aliases,ledger}.rs`, `src/checkpoint/runtime/{permutation,rounding,schedule}.rs`, `src/checkpoint/quantization/{counts,packing,staging}.rs` | 168 | Shape products and storage sizes are exact or rejected on overflow; accepted SafeTensors byte intervals tile the data section exactly (sound and complete); slice, section, and concatenation blocks stay in bounds; untrusted paths cannot climb out of their directory; alias chains terminate and reach their owner, each target is claimed at most once, and execution groups partition the plan; rounding is nearest-even; Q4/Q8 packing then reading is the identity; group, packed, and scale counts tile each payload; staging budgets never wrap. |
| `config.rs` | `src/models/config/plan.rs` | 26 | Multimodal RoPE sections, grouped-head divisibility, token-ID fields, and the trained RoPE context are accepted exactly under their rules, with faults in check order; the legacy `Infinity` lexer lists exactly the token starts outside JSON strings (sound and complete), increasing and at least eight bytes apart. |
| `custom.rs` | `src/models/custom/plan.rs` | 44 | The clamped T5 displacement table has radius `min(max_distance, C - 1)`; its device index lies in the table and slot-to-displacement inverts it; a row append gives valid column `c` slot `L + rank(c)`, filling `L..L + n` in order; batch appends report the first policy error in order; the write window is the tightest one, and each written cell gathers the unique column of its rank. |
| `deepseek_muse.rs` | `src/models/deepseek_v2/model/batch.rs`, `src/models/deepseek_v2/plan.rs`, `src/models/muse_glimmer/schedule.rs` | 61 | Batched-row gathers and scatters, attained block and key extents, nondecreasing key limits that cover every causal prefix, and cache slots are exact; the DeepSeek sparse-layer rule, eligible experts, and latent-cache byte accounting are exact; Muse's backward-counted global layers number exactly `⌈n / 4⌉`, cache slots are dense, and ring chunk key slices contain every visible key within `H + W - 1 + rows` keys. |
| `experiment.rs` | `src/intervention/selection.rs`, `src/experiment/{bounds,precedence}.rs`, `src/architecture/constraints.rs` | 70 | Semantic selections expand to exactly the requested in-bounds ranges; position, label, broadcast, and staged-block bounds are exact and overflow-free, and staged ranges run every block once in order; each pairwise-reduction level reads every column exactly once and the reduction takes `⌈log₂ w⌉` levels; the last matching override wins; architecture constraints are decided exactly in precedence order. |
| `gemma.rs` | `src/models/gemma/plan.rs` | 21 | The default local/global schedule equals `(l + 1) mod p != 0` for every period and has exactly `⌊L / p⌋` global layers; every layer's rotary set holds its own layer type's configuration; namespace detection selects exactly a unique namespace; checkpoint extents are exact. |
| `generation.rs` | `src/generation/{bookkeeping,layout,ngram,random}.rs` | 85 | Stop matching and finish precedence; prefill chunking and row spans; beam-row group preservation and the finalized/live frontier partition; position and cache-capacity preflights; speculative accepted-prefix arithmetic; prompt-lookup drafting at the largest matching n-gram size; scatter layouts that keep every entry in its row's segment; n-gram follower lists that stay strictly increasing; SplitMix64 draws that are injective in the counter. |
| `gpt_oss.rs` | `src/checkpoint/mxfp4.rs`, `src/models/gpt_oss/plan.rs` | 154 | MXFP4 shape validation is sound and complete, with exactly one error class per violated rule; block, scale, and value counts are `16·E·R·G`, `E·R·G`, and `E·R·32G`; offsets are in bounds; the decode table index is a bijection with `(scale, code)`; each decoded value's binary64 bits are exact; the thread partition covers every unit exactly once with at most the planned threads, and every unit owns a contiguous output slice whose value slots are written once; the sliding/full schedule, cache slots, and interleaved `gate_up` offsets (a bijection) are exact. |
| `integration.rs` | `src/models/auto/defaults.rs`, `src/models/custom/mamba2_geometry.rs` | 14 | Hugging Face's sampling `top_k` (explicit zero disables it, otherwise `min(k or 50, vocabulary)` in `1..=vocabulary`); the clamped `min_new_tokens`; canonical EOS and padding ID lists that are exactly the distinct in-vocabulary IDs, sorted, with balanced recursion; Mamba-2 head grouping exists exactly when the group size divides the heads. |
| `llama.rs` | `src/models/llama/plan.rs` | 67 | SmolLM3 NoPE and Qwen sliding-layer schedules; ring assignment, dense cache-group slots (an order-preserving bijection), and group numbering; used rotary sets and the tightest cached-context limit; batched-step plans whose slices or gathers place every valid column at its slot, with exact cache positions and restoring gathers. |
| `loaders.rs` | `src/models/gpt_neox/load/packed.rs`, `src/models/{qwen3_5,qwen3_moe,deepseek_v2,muse_glimmer,nemotron_h,t5}/checkpoint/dimensions.rs`, `src/models/qwen3_5/checkpoint/mtp.rs`, `src/models/auto/load/report.rs` | 61 | GPT-NeoX's packed query/key/value split is a bijection onto the payload and its staging peak is exact; every family's checkpoint widths are exact or name the first term that overflows, in each contract's checking order; the Qwen3.5 MTP inventory and the automatic generation-default reports are exact. |
| `mixers.rs` | `src/delta_net/plan.rs`, `src/mamba2/plan.rs` | 116 | DeltaNet chunk plans cover the tokens with less than one chunk of padding, balanced blocks and doubling-inverse levels tile their extents, widths and channel slices are exact, and the recurrence choice equals its cost comparison; Mamba-2 widths, slices, groups, padding flags, history and window columns, and scan blocks are exact, and the mode partition with its restore permutation is a two-sided inverse. |
| `mla.rs` | `src/mla/geometry.rs` | 23 | Multi-head latent attention extents are checked in field order; every projection width is exact and validation succeeds exactly when every width and matrix size fits `usize`; suffix key bounds are exactly each row's causal prefix; visibility metadata, projected components, and bias broadcasts are accepted exactly under their shapes. |
| `models.rs` | `src/models/shared/{plan,load_audit}.rs` | 62 | Padded-row compaction and scatter plans restore the rectangular layout; host row planning returns exactly the specified outcome with errors in precedence order; logits selections, implicit cached positions, rotary extents, value ranges, and per-row cache appends are exact and bounded; load-audit payload totals and equivalence reads are exact or report overflow. |
| `moe.rs` | `src/moe/{assignment,geometry}.rs` | 57 | The expert counting sort is a stable bijection that partitions every routed assignment by expert; dispatch blocks are maximal runs of whole experts that partition the experts; row slices, signed indices, and routed-ID flattening are exact; expert-bank, router, and group-limited routing geometry is accepted exactly. |
| `nemotron.rs` | `src/models/shared/plan.rs`, `src/models/nemotron_h/model/{plan,cycle}.rs`, `src/models/nemotron_h/routing.rs` | 137 | Compact-batch gather, scatter, pack, unpack, and last-token maps equal their equations; padded query slots keep every row's keys visible; the Mamba-2 path choice, transactional staging policy, shifted MTP cursor, one-pass verification length, per-kind cache slots, speculative append bounds, and group-limited routing are exact (the shared plan is re-verified here). |
| `perf.rs` | `src/checkpoint/division.rs`, `src/checkpoint/runtime/tiles.rs`, `src/checkpoint/quantization/hash_workers.rs` | 31 | The tiled transpose's bands are whole blocks that cover every column exactly once, never outnumber the planned threads, and never exceed the matrix; ceiling division is exact without overflow; the provenance hash worker count is a power of two that divides the buffer budget, whose shares sum exactly to it. |
| `position.rs` | `src/position/{index_plan,t5,ids,rope,rope_scaling}.rs` | 40 | MRoPE axis selection gives each section that fits the interleaving exactly its frequency count; T5 displacement tables and lookup indices are exact and in bounds; T5 buckets equal the reference formula and are monotone in distance; batched position IDs are exact; RoPE dimension, scaling-width, and LongRoPE table checks accept exactly their rules. |
| `qwen.rs` | `src/models/qwen3_5/plan.rs`, `src/models/qwen3_moe/plan.rs` | 61 | The hybrid DeltaNet/attention schedule has exactly `⌊L / interval⌋` full-attention layers; per-family cache slots are dense; the compact right-padded row plan, slot validity, and slot positions are exact; MTP cursors after truncation and speculative-cycle preflights are exact; the Qwen3-MoE sparse-layer predicate is exact. |
| `robustness.rs` | `src/tokenization/chat_template/limits.rs` | 19 | Every sandbox limit is its formula or saturates; the render budget's counter never exceeds its budget or wraps; the text allowance accepts exactly the reservations that succeed; repetition sizes, copy charges, replacement and indentation bounds, the bounded-buffer write check, and cleanup fuel are exact. |
| `t5_gpt.rs` | `src/models/t5/model/packing.rs`, `src/models/t5/buckets.rs`, `src/models/gpt2/extents.rs`, `src/models/gpt_neox/widths.rs` | 70 | Batched T5 compaction and scatter gathers equal their formulas and stay in bounds; every block slot's causal visibility and bias position are exact; `groups_preserved` characterizes the beam reorders that may leave encoder state in place; bucket geometry, relative-bias owners, GPT-2 extents, and GPT-NeoX widths are exact. |
| `tokenization.rs` | `src/tokenization/{calendar,layout,vocabulary}.rs`, `src/tokenization/pycompat/decimal.rs`, `src/tokenization/chat_template/scan.rs`, `src/tokenization/chat_template/guard/lexer.rs` | 166 | Truncation budgets, padding splits, masks, and row-major flattening are exact; calendar fields equal Python's `datetime` for years 1-9999; special-token IDs reconcile exactly; the integer core of Python's `round` is exact; the chat-template tag scanner equals a reference model of MiniJinja's lexer, and the guard lexer's tokenization is the body's unique one. |
| `transfer.rs` | `src/transfer/plan.rs` | 19 | Target resolution follows the documented rule precedence; target ties are accepted exactly when consistent; the fan-out check names the smallest path reaching two tie classes; class leaders identify entries exactly by class. |

## Rules for proof code

Proof code must not trust anything Verus has not checked. The
`verus_harness` integration test rejects these constructs in harnesses and in
every included source file:

| Rejected | Why |
| --- | --- |
| `admit(`, `assume(`, `assume_specification`, `assume_termination`, `external_body` | Assume a fact, a contract, or termination without proof. `--no-cheating` rejects these too. |
| `exec_allows_no_decreases_clause` | Lets an executable loop or recursion skip its termination proof, so a body that never returns satisfies any postcondition. `--no-cheating` accepts it. |
| `verifier::external`, `verifier(external)`, and `external` as any argument of `verus_verify(...)` | Removes an item, contract included, from verification without a diagnostic. `--no-cheating` accepts it, so even a false `ensures` under `verus_verify(external)` produces no error. |
| `external_fn_specification`, `external_type_specification`, `external_trait_specification`, `external_trait_extension` | Trust a hand-written contract for code Verus never checks. |
| `inline_air_stmt` | Injects unchecked solver input. |

Verification also runs with `--no-cheating`, which rejects the first row's
constructs in the prover itself.

A contract can verify and still be too weak, so specifications are
checked by mutation testing. Introduce a deliberate defect into a scratch
copy of a verified body (a swapped branch, an index shifted by one, two
checks in the other order, a constant result), verify the copy through a
scratch harness whose `#[path]` points at it, and confirm that Verus
rejects it. A mutant that verifies exposes a missing postcondition, and
the contract is strengthened until the mutant fails: for example, a guard
lexer that accepted `1.5.2` as one float once verified, so the number
contract now states the literal's shape, and `stacked_layout` now states
that `rows * columns` fits `usize`. Mutants never enter the repository; the
committed harnesses include only the real sources.

## Running

Verus releases pin their own Rust toolchain and ship Z3. With the release
recorded below installed and its toolchain present (`rustup toolchain
install 1.98.1-aarch64-apple-darwin --profile minimal`), run every harness
from the repository root:

```sh
for harness in verus/*.rs; do
  "$VERUS" --crate-type lib --no-cheating --triggers-mode silent "$harness" || exit 1
done
```

`VERUS` is the `verus` binary from the release archive. Each harness prints
one `verification results:: N verified, 0 errors` line. Verification
performs no Cargo build, downloads nothing, and writes no target directory.
The whole loop takes 35 to 55 seconds on an Apple M3 Max, depending on
concurrent load; a single harness can be run alone while working on its
area.

The release gate runs the same loop and records the evidence:

```sh
VERUS=/path/to/verus python3 scripts/check_release.py --gate verus
```

It checks that `$VERUS` is the pinned release, runs every harness with the
flags above, requires each harness's `N verified, 0 errors` to match the
table above, and writes the per-harness counts, the total, and the SHA-256
of every harness and included source to `verus.json` in its evidence
directory. CI runs this gate on every push and pull request that changes
`src/`, `verus/`, or the gate itself.

Without Verus installed, `cargo test --locked --test verus_harness` still
checks the harness structure: every harness begins with module
documentation and includes existing sources, no harness or included source
uses an escape hatch, every source file with Verus-only code is included by
some harness, and the table above, its totals, and the counts quoted in the
top-level documents agree with the harness files.

| Item | Version |
| --- | --- |
| Verus | `0.2026.10.04.426d8b0` (`release/0.2026.10.04.426d8b0`, arm64 macOS) |
| Rust used by Verus | `1.98.1-aarch64-apple-darwin` |
| Z3 | `4.16.0` (bundled) |

## What is and is not proved

Each harness documents its verified functions, their postconditions, and
the runtime behavior that remains covered only by tests. A proof covers
the integer and indexing logic of the annotated body for all admissible
inputs. It does not cover Burn tensor kernels, floating-point numerics,
allocation failure, or unannotated callers; those remain the
responsibility of the Rust test suite. Where a proof states that a
function equals a reference formula or model (a Python `datetime` helper,
MiniJinja's lexer, a Transformers schedule), the correspondence between
that model and the external implementation is checked by the Rust tests,
not by Verus.
