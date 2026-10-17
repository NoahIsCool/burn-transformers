//! Verus harness for multi-head latent attention's width and key-prefix
//! arithmetic.
//!
//! Verifies, in place, the specifications attached to `src/mla/geometry.rs`:
//!
//! - `mla_axis_fault` checks configuration extents in field order, then rotary
//!   parity, then the optional query rank, with each fault exactly when every
//!   earlier check passed;
//! - projection widths are exact: `qk_head_width` (`d_n + d_r`),
//!   `compressed_width` (`r + d_r`), and `expanded_width` (`H * (d_n + d_v)`,
//!   per-head sum checked first), each naming its overflowing term;
//!   `mla_widths` also derives `H * (d_n + d_r)` and `H * d_v` and succeeds
//!   exactly when every width and every projection matrix's element count
//!   fits `usize`, naming the first overflowing matrix otherwise; the public
//!   width helpers and validation share one verified derivation per width,
//!   and initialization allocates from the validated widths;
//! - `suffix_key_limits` returns bottom-right suffix key bounds that are
//!   exactly each row's causal prefix `K - Q + i + 1` and satisfy the prefix
//!   contract (one bound per row, each in `1..=K`, nondecreasing);
//!   `key_limits_valid` accepts exactly contract-satisfying bounds;
//!   `chunk_key_end` returns a chunk's key prefix `limits[end - 1]`, which
//!   covers every row before its end; and `saturating_window` converts window
//!   widths exactly, saturating at `i64::MAX`;
//! - `visibility_fault` accepts visibility metadata exactly when the window is
//!   non-zero, positions and validity masks have their `[batch or 1, tokens]`
//!   and `[batch, tokens]` extents, and the key bounds satisfy the prefix
//!   contract; `projection_matches` accepts projected components exactly when
//!   they match the configured non-empty layout; and `broadcasts_over` holds
//!   exactly when every axis of a bias is one or equal to the score plane.
//!
//! Not proved here: the Burn tensor projections, latent expansion, rotary
//! application, softmax, and absorbed GEMMs that consume these widths; the
//! MLA unit tests cover those. Loop invariants are attributes on `while`
//! statements, which needs `proc_macro_hygiene` under the Verus compiler only.
//!
//! See `verus/README.md` for the method, toolchain, and command.

#![allow(unused)]
#![cfg_attr(verus_keep_ghost, feature(proc_macro_hygiene))]

#[path = "../src/mla/geometry.rs"]
pub mod geometry;
