//! Verus harness for attention's pure planning and shape arithmetic.
//!
//! Verifies, in place, the specifications attached to:
//!
//! - `src/attention/geometry.rs`: `operand_geometry` (grouped `[B, H, Q, d]`
//!   query, `[B, G, K, d]` key, and `[B, G, K, d_v]` value operands with
//!   optional mask, bias, and sinks) and `projection_fault` (a projected
//!   layer's tensors against its configured layout) accept exactly the
//!   documented shapes and otherwise return the first failed check in variant
//!   order, each fault exactly when every earlier check passed; the helpers
//!   `mismatched` and `misfit_bias` decide extent equality and `[1 or B, H,
//!   Q, K]` bias broadcasting exactly. `HeadGeometry::groups` is the exact
//!   quotient (`groups * G == H`, at least one for non-empty heads);
//!   `causal_mask_needed` is exactly `causal && Q != 1`, and a skipped causal
//!   mask (one query row) excludes no link; `native_route` passes the
//!   backend's causal flag only for square planes, where top-left and
//!   bottom-right alignment exclude exactly the same links; and
//!   `broadcast_shape` takes the larger extent on every axis.
//! - `src/attention/grouping.rs`: `head_groups` returns the exact group size
//!   `g` (`g * G == H`) exactly when both counts are non-zero and `G` divides
//!   `H`, and repeating K/V by expand-and-reshape makes repeated head `m` read
//!   source head `m / g`; `folded_rows` folds `[b, H, rows, x]` to
//!   `[b, G, g * rows, x]` keeping every flat index and maps `(head, token)`
//!   bijectively onto `(kv head, folded row)` (`lemma_fold_index`,
//!   `lemma_unfold_index`), with every member index below `g`, and
//!   `unfolded_rows` is its inverse; `value_pairing` gives every query head of
//!   differential attention its reference pair `[V_kv(i) ; V_kv(i + H/2)]`:
//!   natively for an even key/value count (`lemma_native_pairing`), as head
//!   zero twice for one shared head, and otherwise through valid repeated
//!   heads `(h mod H/2, h mod H/2 + H/2)` whose source heads are the reference
//!   pair.
//! - `src/attention/plan.rs`: `plan_query_chunks` satisfies
//!   `chunk_plan_contract`: an exact peak score size, both limits respected,
//!   the largest chunk they admit, `ceil(Q / c)` chunks that cover the rows,
//!   and each error exactly when every earlier check passed;
//!   `query_chunk_range` returns ranges that partition the query rows in
//!   order; `sliding_window_key_range` returns exactly the union of its rows'
//!   visible keys, at most `rows + W - 1` wide; `capped_query_rows` is
//!   `min(cap, Q)` and `windowed_key_bound` is `min(cap + W - 1, K)` without
//!   overflow, and the lemmas `lemma_window_chunk_admitted` and
//!   `lemma_planned_chunk_fits_one_call` prove that every planned chunk of
//!   bounded execution has a window key range within that bound and re-plans
//!   as one call; `chunk_frame` exists exactly when the span crosses a chunk
//!   boundary and then preserves every position's absolute chunk index with
//!   `phase < width <= span`.
//! - `src/attention/rules.rs`: `window_excludes` and `chunk_excludes` equal
//!   their mathematical definitions; `reduce_window` and `reduce_chunk`
//!   reduce a sliding window or a chunked rule to local positions of a span,
//!   excluding exactly the links the absolute rule excludes on every local
//!   pair, with parameters below the span, and collapse to `Unrestricted`
//!   exactly when the rule excludes nothing in the span; `position_peak` and
//!   `aligned_position_peak` return the exact largest position value formed
//!   (below `2 * span - 1` for a reduced rule); `mask_geometry_fault` validates
//!   mask extents in a fixed order; `rejects_short_keys` is exactly the
//!   `K < Q` rejection rule, and `position_source` chooses columns or
//!   ordinals, whose offset `K - Q` cannot underflow once the alignment check
//!   passes (`lemma_alignment_admits_ordinals`).
//! - `src/attention/widths.rs`: `projection_widths` succeeds exactly for
//!   non-empty, grouped layouts whose widths and projection-matrix element
//!   counts are representable, returning the exact widths, with each error
//!   exactly when every earlier check passed; `paired_projection_widths` adds
//!   the even query-head requirement last.
//!
//! It also verifies `src/feed_forward/widths.rs`, which shares this area:
//! `width_fault` accepts exactly non-empty residual and intermediate widths
//! and names the residual width first.
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
