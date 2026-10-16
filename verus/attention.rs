//! Verus harness for attention's pure planning and shape arithmetic.
//!
//! Verifies, in place, the specifications attached to:
//!
//! - `src/attention/geometry.rs`: grouped operand and projected-layer shape
//!   validators accept exactly the documented shapes and report the first
//!   failed check; a skipped causal mask (one query row) excludes no link;
//!   native routing passes the backend's causal flag only for square
//!   planes, where top-left and bottom-right alignment coincide; and mask
//!   unions broadcast to the larger extent on every axis;
//! - `src/attention/grouping.rs`: grouped-query head counts are exact, and
//!   repeating K/V by expand-and-reshape makes repeated head `m` read source
//!   head `m / g`; folding `[b, H, rows, x]` to `[b, G, g * rows, x]` keeps
//!   every flat index and maps `(head, token)` bijectively onto
//!   `(kv head, folded row)`, with unfolding as its inverse and the sink
//!   selector's member index in range; differential attention's native value
//!   pairing gives every query head its reference pair
//!   `[V_kv(i) ; V_kv(i + H/2)]`, one shared head pairs with itself, and the
//!   repeated pairing reads valid repeated heads `(h mod H/2, h mod H/2 +
//!   H/2)`, whose source heads are the reference pair by definition;
//! - `src/attention/plan.rs`: query-chunk planning against score-element
//!   limits (exact peak size, both limits respected, the largest chunk they
//!   admit, `ceil(Q / c)` chunks with no overflow, and each error exactly when
//!   every earlier check passed), chunk ranges that partition the query rows,
//!   sliding-window key ranges that are exactly the union of their rows'
//!   visible keys, the row cap and window key bound of bounded execution
//!   (with lemmas that every planned chunk has a window key range within the
//!   bound and re-plans as one call), and chunk frames that preserve every
//!   position's absolute chunk index with small, overflow-free parameters
//!   and exist exactly when the span crosses a chunk boundary;
//! - `src/attention/rules.rs`: the window and chunk exclusion oracles equal
//!   their mathematical definitions; reducing a sliding window or a chunked
//!   rule to local positions of a span excludes exactly the links the
//!   absolute rule excludes on every local pair, with parameters below the
//!   span, and collapses to `Unrestricted` exactly when the rule excludes
//!   nothing in the span; the largest position value a
//!   reduced rule forms is exact and below `2 * K - 1`; mask extents are
//!   validated in a fixed order; and the ordinal query offset `K - Q` cannot
//!   underflow once the alignment check passes;
//! - `src/attention/widths.rs`: grouped-query projection widths, whose
//!   success implies exact widths, head grouping, and representable element
//!   counts for every projection matrix, with each error exactly when every
//!   earlier check passed, and differential widths that add the even
//!   query-head requirement last.
//!
//! It also verifies `src/feed_forward/widths.rs`, which shares this area:
//! feed-forward validation accepts exactly non-empty residual and
//! intermediate widths and names the residual width first.
//!
//! Not proved here: Burn tensor kernels, floating-point softmax and masking
//! numerics, and the unannotated wrappers that call these cores; the attention
//! and feed-forward unit tests cover those.
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

#[path = "../src/feed_forward/widths.rs"]
pub mod feed_forward_widths;
