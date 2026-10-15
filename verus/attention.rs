//! Verus harness for attention's pure planning and shape arithmetic.
//!
//! Verifies, in place, the specifications attached to:
//!
//! - `src/attention/geometry.rs`: grouped operand and projected-layer shape
//!   validators accept exactly the documented shapes and report the first
//!   failed check; a skipped causal mask (one query row) excludes no link;
//!   and native routing passes the backend's causal flag only for square
//!   planes, where top-left and bottom-right alignment coincide;
//! - `src/attention/grouping.rs`: grouped-query head counts are exact;
//!   folding `[b, H, rows, x]` to `[b, G, g * rows, x]` keeps every flat index
//!   and maps `(head, token)` bijectively onto `(kv head, folded row)`, with
//!   unfolding as its inverse and the sink selector's member index in range;
//!   differential attention's value pairing gives every query head its
//!   reference pair `[V_kv(i) ; V_kv(i + H/2)]` natively, after repetition,
//!   or for one shared head;
//! - `src/attention/plan.rs`: query-chunk planning against score-element
//!   limits (exact peak size, both limits respected, the largest chunk they
//!   admit, `ceil(Q / c)` chunks with no overflow, and each error exactly when
//!   every earlier check passed), chunk ranges that partition the query rows,
//!   sliding-window key ranges that are exactly the union of their rows'
//!   visible keys, the row cap and window key bound of bounded execution
//!   (with a lemma that every planned chunk re-plans as one call), and chunk
//!   frames that preserve every position's absolute chunk index with small,
//!   overflow-free parameters;
//! - `src/attention/rules.rs`: the window and chunk exclusion oracles equal
//!   their mathematical definitions; reducing a sliding window or a chunked
//!   rule to local positions of a span excludes exactly the links the
//!   absolute rule excludes on every local pair, with parameters below the
//!   span (so `Unrestricted` collapse is exact); mask extents are validated in
//!   a fixed order; and the ordinal query offset `K - Q` cannot underflow
//!   once the alignment check passes;
//! - `src/attention/widths.rs`: grouped-query projection widths, whose
//!   success implies exact widths, head grouping, and representable element
//!   counts for every projection matrix, with each error exactly when every
//!   earlier check passed, and differential widths that add the even
//!   query-head requirement last.
//!
//! See `verus/README.md` for the method, toolchain, and command.

#![allow(unused)]

#[path = "../src/attention/geometry.rs"]
pub mod geometry;

#[path = "../src/attention/grouping.rs"]
pub mod grouping;

#[path = "../src/attention/plan.rs"]
pub mod plan;

#[path = "../src/attention/rules.rs"]
pub mod rules;

#[path = "../src/attention/widths.rs"]
pub mod widths;
