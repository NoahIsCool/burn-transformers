//! Verus harness for the performance package's pure work-partition arithmetic.
//!
//! Covers the tiled transpose's thread bands and the provenance hash pass's
//! worker count and buffer shares.
//!
//! Verifies, in place, the specifications attached to
//! `src/checkpoint/runtime/tiles.rs` and the division core it uses:
//!
//! - the shared `ceil_div_usize` and `ceil_div_u64` (`src/division.rs`,
//!   also checked by `verus/checkpoint.rs`, `verus/gpt_oss.rs`, and
//!   `verus/mixers.rs`) return exactly `⌈n / d⌉` without forming
//!   `n + d - 1`: the least `q` with `n <= q · d < n + d`, never above `n`,
//!   and positive for positive `n`;
//! - `transpose_plan` equals its documented formulas exactly: `t = 1` below the
//!   parallel threshold and `min(available, ⌈columns / b⌉)` above it,
//!   `w = ⌈⌈columns / b⌉ / t⌉ · b` (or every column when that covers all blocks),
//!   and `n = ⌈columns / w⌉`. It splits a `[rows, columns]` transpose into cache
//!   blocks of `min(columns, 32)` columns and per-thread bands that are whole blocks
//!   (unless one band spans every column), cover every column exactly once
//!   (`(n - 1) · w < columns <= n · w`), number at most the planned threads,
//!   which never exceed the caller's available parallelism and are one below the
//!   parallel threshold, and whose per-band target slice of `rows · w` scalars
//!   never exceeds the whole matrix (`band_width_bounds`,
//!   `band_count_bounds`); all of its arithmetic is overflow-free.
//!
//! It also verifies `src/checkpoint/quantization/hash_workers.rs`, the
//! concurrent provenance hash pass's partition: `hash_worker_count` returns
//! the largest power of two (1, 2, 4, or 8) not above
//! `max(1, min(parallelism, 8, shards))`, so it uses more than half of the
//! usable workers and divides the hash-buffer budget; `worker_share` gives
//! each worker exactly `B / max(w, 1)` bytes, and for those counts the
//! shares sum to the budget `B`.
//!
//! See `verus/README.md` for the method, toolchain, and command. The tiled copy
//! that consumes the plan, the storage-order policy, and backend matrix products
//! remain covered by tests and measured benchmarks rather than proofs.

// Loop invariants are attributes on `while` expressions, which rustc accepts
// only with this feature; it applies solely under Verus.
#![cfg_attr(verus_keep_ghost, feature(proc_macro_hygiene))]
#![allow(unused)]

#[path = "../src/division.rs"]
pub mod division;

#[path = "../src/checkpoint/runtime/tiles.rs"]
pub mod tiles;

#[path = "../src/checkpoint/quantization/hash_workers.rs"]
pub mod hash_workers;
