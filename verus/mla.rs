//! Verus harness for multi-head latent attention's width and key-prefix arithmetic.
//!
//! Verifies, in place, the specifications attached to `src/mla/geometry.rs`:
//!
//! - configuration extents are checked in field order, then rotary parity,
//!   then the optional query rank, with each fault exactly when every earlier
//!   check passed;
//! - projection widths are exact (`d_n + d_r`, `r + d_r`, `H * (d_n + d_v)`,
//!   `H * (d_n + d_r)`, `H * d_v`), and validation succeeds exactly when every
//!   width and every projection matrix's element count fits `usize`, naming
//!   the first overflowing matrix otherwise;
//! - bottom-right suffix key bounds are exactly each row's causal prefix
//!   `K - Q + i + 1` and satisfy the prefix contract; the bound validator
//!   accepts exactly contract-satisfying bounds; a chunk's key prefix covers
//!   every row before its end; and window widths saturate exactly at
//!   `i64::MAX`;
//! - visibility metadata is accepted exactly when the window is non-zero,
//!   positions and validity masks have their `[batch or 1, tokens]` and
//!   `[batch, tokens]` extents, and the key bounds satisfy the prefix
//!   contract; projected components are accepted exactly when they match the
//!   configured non-empty layout; and a bias broadcasts exactly when every
//!   axis is one or equal to the score plane.
//!
//! Not proved here: the Burn tensor projections, latent expansion, rotary
//! application, softmax, and absorbed GEMMs that consume these widths; the
//! MLA unit tests cover those. Loop invariants are attributes on `while`
//! statements, which needs `proc_macro_hygiene` under the Verus compiler only.
//!
//! See `verus/README.md` for the method, toolchain, and command.

#![allow(unused)]
#![feature(proc_macro_hygiene)]

#[path = "../src/mla/geometry.rs"]
pub mod geometry;
