//! Verus harness for the performance package's pure work-partition arithmetic.
//!
//! Verifies, in place, the specifications attached to
//! `src/checkpoint/runtime/tiles.rs`:
//!
//! - `ceiling_division` returns `⌈n / d⌉` without forming `n + d - 1`, never
//!   exceeds `n`, and is positive for positive `n`;
//! - `transpose_plan` splits a `[rows, columns]` transpose into cache blocks of
//!   `min(columns, 32)` columns and per-thread bands that are whole blocks
//!   (unless one band spans every column), cover every column exactly once
//!   (`(n - 1) · w < columns <= n · w`), number at most the planned threads,
//!   which never exceed the caller's available parallelism and are one below the
//!   parallel threshold, and whose per-band target slice of `rows · w` scalars
//!   never exceeds the whole matrix; all of its arithmetic is overflow-free.
//!
//! See `verus/README.md` for the method, toolchain, and command. The tiled copy
//! that consumes the plan, the storage-order policy, and backend matrix products
//! remain covered by tests and measured benchmarks rather than proofs.

// Loop invariants are attributes on `while` expressions, which rustc accepts
// only with this feature; it applies solely under Verus.
#![cfg_attr(verus_keep_ghost, feature(proc_macro_hygiene))]
#![allow(unused)]

#[path = "../src/checkpoint/runtime/tiles.rs"]
pub mod tiles;
