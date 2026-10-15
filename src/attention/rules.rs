//! Pure integer rules of attention connectivity, checked in place by Verus.
//!
//! # Exclusion rules
//!
//! With query position `p` and key position `j` (absolute token positions),
//! the connectivity rules of [`super::AttentionConnectivity`] exclude a link
//! when
//!
//! | rule | excluded when |
//! | --- | --- |
//! | causal | `j > p` |
//! | sliding window `W` | `W = 0` or `j + W <= p` |
//! | chunked local `C` | `C = 0` or `floor(p / C) != floor(j / C)` |
//!
//! [`window_excludes`] and [`chunk_excludes`] are the scalar oracles the
//! tensor builders vectorize; the window rule compares the gap `p - j` so it
//! never overflows.
//!
//! # Local reduction
//!
//! Tensor masks compare *local* positions `q, k in [0, K)` of a span of `K`
//! keys whose first key sits at absolute position `key_start`. A local rule
//! must exclude exactly the links its absolute rule excludes:
//! `local(q, k) = absolute(key_start + q, key_start + k)` for every `q, k < K`.
//!
//! - Windows depend only on the gap `p - j`, so `key_start` cancels. A gap
//!   between local positions never reaches `K`, so a window `W >= K` excludes
//!   nothing and collapses to [`LocalRule::Unrestricted`]
//!   ([`reduce_window`]); a narrower window keeps `W < K`.
//! - Chunks become a frame `(width, phase)` with `phase < width <= K`
//!   ([`super::plan::chunk_frame`]): absolute chunks of `key_start + q` agree
//!   exactly when `floor((q + phase) / width)` agree. A span inside one chunk
//!   collapses to [`LocalRule::Unrestricted`] ([`reduce_chunk`]).
//!
//! Every surviving parameter is below `K` (or a width of at most `K`), so the
//! largest value the tensor arithmetic forms, `(K - 1) + parameter`, stays
//! below `2 * K`.
//!
//! # Geometry and positions
//!
//! [`mask_geometry_fault`] validates one mask call's extents,
//! [`position_peak`] bounds the largest position value a rule forms, and
//! [`rejects_short_keys`] and [`position_source`] decide which position
//! vectors a mask needs. Queries are the bottom-right suffix of the keys, so
//! valid-token ordinals of the queries start `K - Q` columns into the keys;
//! the planner proves that this offset never underflows once the alignment
//! check has passed.
//!
//! # Verified properties
//!
//! Verus checks every function here through `verus/attention.rs`: the oracles
//! equal their mathematical definitions, both reductions preserve the
//! exclusion relation on every local position pair with parameters below the
//! span, geometry faults are reported exactly and in order, and the ordinal
//! offset is exact and in range.

#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

#[cfg(verus_keep_ghost)]
use super::plan::chunk_of;

use super::plan::chunk_frame;

#[cfg(verus_keep_ghost)]
verus! {

/// The sliding window `window` excludes key position `key` for query position `query`.
pub open spec fn window_excluded(query: int, key: int, window: int) -> bool {
    key + window <= query
}

/// Chunked-local attention with chunks of `chunk` positions excludes `(query, key)`.
pub open spec fn chunk_excluded(query: int, key: int, chunk: int) -> bool {
    chunk_of(query, chunk) != chunk_of(key, chunk)
}

/// More queries than keys are rejected unless only unpadded or non-causal global attention runs.
pub open spec fn short_keys_rejected(
    global: bool,
    causal: bool,
    key_masked: bool,
    query_sequence: int,
    key_sequence: int,
) -> bool {
    key_sequence < query_sequence && (!global || (causal && key_masked))
}

/// Passing the alignment check admits ordinal positions: their offset `K - Q` cannot underflow.
///
/// Ordinals are needed when a key mask is present and either causality or a
/// restricting rule applies. A rejected `K < Q` is excused only for global
/// connectivity that is not causally padded; global connectivity is
/// unrestricted, so such a call needs no positions at all.
pub proof fn lemma_alignment_admits_ordinals(
    global: bool,
    unrestricted: bool,
    causal: bool,
    key_masked: bool,
    query_sequence: int,
    key_sequence: int,
)
    requires
        global ==> unrestricted,
        !short_keys_rejected(global, causal, key_masked, query_sequence, key_sequence),
    ensures
        key_masked && (causal || !unrestricted) ==> query_sequence <= key_sequence,
{
}

/// What a reduced local rule excludes for local query position `query` and key `key`.
pub open spec fn local_excluded(rule: LocalRule, query: int, key: int) -> bool {
    match rule {
        LocalRule::Unrestricted => false,
        LocalRule::Window(window) => window_excluded(query, key, window as int),
        LocalRule::Chunk { width, phase } => (query + phase) / (width as int) != (key
            + phase) / (width as int),
    }
}

/// The reduced rule's parameters are small: windows below the span, frames inside it.
pub open spec fn local_rule_fits(rule: LocalRule, span: int) -> bool {
    match rule {
        LocalRule::Unrestricted => true,
        LocalRule::Window(window) => window < span,
        LocalRule::Chunk { width, phase } => 1 <= width <= span && phase < width,
    }
}

} // verus!

/// Whether a sliding window of `window` positions excludes `key` for `query`.
///
/// Equals `window == 0 || key + window <= query` over unbounded integers: a
/// zero window excludes every key, and otherwise only the `window` newest
/// positions up to the query (including keys after it) stay visible. The gap
/// form never overflows.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(excluded =>
    ensures
        excluded == (window == 0 || window_excluded(query as int, key as int, window as int)),
))]
pub const fn window_excludes(query: usize, key: usize, window: usize) -> bool {
    window == 0 || (query >= key && query - key >= window)
}

/// Whether chunked-local attention with `chunk` positions excludes `key` for `query`.
///
/// Equals `chunk == 0 || floor(query / chunk) != floor(key / chunk)`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(excluded =>
    ensures
        excluded == (chunk == 0 || chunk_excluded(query as int, key as int, chunk as int)),
))]
pub const fn chunk_excludes(query: usize, key: usize, chunk: usize) -> bool {
    chunk == 0 || query / chunk != key / chunk
}

/// A connectivity rule rewritten for local positions `[0, span)` with small parameters.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LocalRule {
    /// The rule excludes no link inside the span.
    Unrestricted,
    /// Exclude `key + window <= query`, with `window < span`.
    Window(usize),
    /// Exclude links whose `floor((position + phase) / width)` differ.
    Chunk {
        /// Frame width, `1 <= width <= span`.
        width: usize,
        /// Frame phase, `phase < width`.
        phase: usize,
    },
}

/// Reduces a sliding window to local positions of a `span`-key timeline.
///
/// The result excludes exactly the local pairs the window excludes; these are
/// also the absolute pairs `(key_start + q, key_start + k)` for any
/// `key_start`, because `j + W <= p` depends only on the gap `p - j`. Windows of at least
/// `span` positions exclude nothing and become [`LocalRule::Unrestricted`];
/// narrower windows keep their width.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(rule =>
    requires
        span >= 1,
    ensures
        rule == LocalRule::Unrestricted <==> window >= span,
        window < span ==> rule == LocalRule::Window(window),
        local_rule_fits(rule, span as int),
        forall|query: int, key: int|
            0 <= query < span && 0 <= key < span ==> #[trigger] local_excluded(rule, query, key)
                == window_excluded(query, key, window as int),
))]
pub fn reduce_window(window: usize, span: usize) -> LocalRule {
    // No local gap reaches `span`, so a window that wide excludes nothing.
    if window >= span {
        LocalRule::Unrestricted
    } else {
        LocalRule::Window(window)
    }
}

/// Reduces chunked-local attention over absolute positions `key_start + [0, span)`.
///
/// The result excludes local pair `(q, k)` exactly when the absolute
/// positions `key_start + q` and `key_start + k` lie in different chunks of
/// `chunk` positions. A span inside one chunk becomes
/// [`LocalRule::Unrestricted`]; otherwise the frame of
/// [`super::plan::chunk_frame`] keeps `phase < width <= span`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(rule =>
    requires
        chunk >= 1,
        span >= 1,
    ensures
        !(rule is Window),
        local_rule_fits(rule, span as int),
        forall|query: int, key: int|
            0 <= query < span && 0 <= key < span ==> #[trigger] local_excluded(rule, query, key)
                == chunk_excluded(key_start + query, key_start + key, chunk as int),
))]
pub fn reduce_chunk(chunk: usize, key_start: usize, span: usize) -> LocalRule {
    match chunk_frame(chunk, key_start, span) {
        None => LocalRule::Unrestricted,
        Some((width, phase)) => {
            #[cfg(verus_keep_ghost)]
            proof! {
                assert forall|query: int, key: int| 0 <= query < span && 0 <= key < span implies
                    #[trigger] local_excluded(LocalRule::Chunk { width, phase }, query, key)
                        == chunk_excluded(key_start + query, key_start + key, chunk as int) by {
                    assert(chunk_of(key_start + query, chunk as int) == key_start / chunk + (query
                        + phase) / (width as int));
                    assert(chunk_of(key_start + key, chunk as int) == key_start / chunk + (key
                        + phase) / (width as int));
                }
            }
            LocalRule::Chunk { width, phase }
        }
    }
}

/// Why the extents of one exclusion mask are unusable.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MaskGeometryFault {
    /// Batch, heads, query sequence, or key sequence is zero.
    EmptyAxis,
    /// The query sequence does not fit `i64` position arithmetic.
    QueryTooLong,
    /// The key sequence does not fit `i64` position arithmetic.
    KeyTooLong,
    /// The absolute position `key_start + key_sequence` overflows `usize`.
    KeyStartOverflow,
}

/// Validates one mask call's extents, reporting the first failed check.
///
/// Checks, in order: every axis is non-empty, both sequences fit `i64`, and
/// absolute key positions `key_start + [0, key_sequence)` fit `usize`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(fault =>
    ensures
        match fault {
            None => {
                &&& batch >= 1 && heads >= 1 && query_sequence >= 1 && key_sequence >= 1
                &&& query_sequence as int <= i64::MAX as int
                &&& key_sequence as int <= i64::MAX as int
                &&& key_start + key_sequence <= usize::MAX
            },
            Some(MaskGeometryFault::EmptyAxis) =>
                batch == 0 || heads == 0 || query_sequence == 0 || key_sequence == 0,
            Some(MaskGeometryFault::QueryTooLong) => {
                &&& batch >= 1 && heads >= 1 && query_sequence >= 1 && key_sequence >= 1
                &&& query_sequence as int > i64::MAX as int
            },
            Some(MaskGeometryFault::KeyTooLong) => {
                &&& batch >= 1 && heads >= 1 && query_sequence >= 1 && key_sequence >= 1
                &&& query_sequence as int <= i64::MAX as int
                &&& key_sequence as int > i64::MAX as int
            },
            Some(MaskGeometryFault::KeyStartOverflow) => {
                &&& batch >= 1 && heads >= 1 && query_sequence >= 1 && key_sequence >= 1
                &&& query_sequence as int <= i64::MAX as int
                &&& key_sequence as int <= i64::MAX as int
                &&& key_start + key_sequence > usize::MAX
            },
        }
))]
pub fn mask_geometry_fault(
    batch: usize,
    heads: usize,
    query_sequence: usize,
    key_sequence: usize,
    key_start: usize,
) -> Option<MaskGeometryFault> {
    if batch == 0 || heads == 0 || query_sequence == 0 || key_sequence == 0 {
        return Some(MaskGeometryFault::EmptyAxis);
    }
    // Widening to `u64` is lossless for every `usize`, unlike narrowing `i64::MAX`.
    if query_sequence as u64 > i64::MAX as u64 {
        return Some(MaskGeometryFault::QueryTooLong);
    }
    if key_sequence as u64 > i64::MAX as u64 {
        return Some(MaskGeometryFault::KeyTooLong);
    }
    if key_start.checked_add(key_sequence).is_none() {
        return Some(MaskGeometryFault::KeyStartOverflow);
    }
    None
}

/// Whether a mask with more queries than keys must be rejected.
///
/// Queries are the bottom-right suffix of the keys, so `Q > K` leaves the
/// leading rows with no key at all. Only unpadded global attention, or global
/// attention without causality, tolerates that (its leading rows then see
/// nothing); local rules and ordinal positions need `K >= Q`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(rejected =>
    ensures
        rejected == short_keys_rejected(
            global,
            causal,
            key_masked,
            query_sequence as int,
            key_sequence as int,
        ),
))]
pub const fn rejects_short_keys(
    global: bool,
    causal: bool,
    key_masked: bool,
    query_sequence: usize,
    key_sequence: usize,
) -> bool {
    key_sequence < query_sequence && (!global || (causal && key_masked))
}

/// Which position vectors one exclusion mask compares.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PositionSource {
    /// Neither causality nor the rule needs positions.
    Unneeded,
    /// Column indices: keys `0..K`, queries bottom-right aligned `K - Q..K`.
    Columns,
    /// Valid-token ordinals of the keys; the queries' ordinals start at column `query_offset`.
    Ordinals {
        /// First key column of the query suffix, `K - Q`.
        query_offset: usize,
    },
}

/// Chooses the positions a mask needs, after the alignment check has passed.
///
/// Positions are needed for causal masks and for every rule that is not
/// [`LocalRule::Unrestricted`]; a key mask turns them into valid-token
/// ordinals whose query suffix starts at key column `K - Q`. The precondition
/// `K >= Q` for ordinal positions follows from the alignment check
/// (`lemma_alignment_admits_ordinals`), because global connectivity always
/// reduces to an unrestricted rule.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(source =>
    requires
        key_masked && (causal || !unrestricted) ==> query_sequence <= key_sequence,
    ensures
        !(causal || !unrestricted) ==> source == PositionSource::Unneeded,
        (causal || !unrestricted) && !key_masked ==> source == PositionSource::Columns,
        (causal || !unrestricted) && key_masked ==> source == (PositionSource::Ordinals {
            query_offset: (key_sequence - query_sequence) as usize,
        }),
        source is Ordinals ==> query_sequence <= key_sequence,
))]
pub fn position_source(
    unrestricted: bool,
    causal: bool,
    key_masked: bool,
    query_sequence: usize,
    key_sequence: usize,
) -> PositionSource {
    if !causal && unrestricted {
        return PositionSource::Unneeded;
    }
    if key_masked {
        PositionSource::Ordinals {
            query_offset: key_sequence - query_sequence,
        }
    } else {
        PositionSource::Columns
    }
}

/// Largest position bottom-right-aligned query and key vectors hold: `max(Q, K) - 1`.
///
/// Queries take positions `K - Q..K` (formed as `0..Q` plus `K - Q`) and keys
/// `0..K`, so every formed value is at most `max(Q, K) - 1`, which is attained.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(peak =>
    requires
        queries >= 0,
        keys >= 0,
    ensures
        peak == (if queries >= keys { queries } else { keys }) - 1,
        queries - 1 <= peak,
        (keys - queries) + (queries - 1) <= peak,
))]
pub fn aligned_position_peak(queries: i64, keys: i64) -> i64 {
    queries.max(keys) - 1
}

/// Largest value a reduced rule's tensor arithmetic forms over `span` local positions.
///
/// Positions are below `span`, so the largest is `(span - 1) + extent`, where
/// `extent` is the window width or frame phase (zero when unrestricted).
/// Returns `None` when that sum leaves `i64`. Reduced parameters are below the
/// span, so a successful peak is below `2 * span - 1`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(peak =>
    requires
        span >= 0,
    ensures
        match peak {
            Some(peak) => peak == span - 1 + extent,
            None => span - 1 + extent > i64::MAX || span - 1 + extent < i64::MIN,
        },
        0 <= extent < span ==> match peak {
            Some(peak) => peak < 2 * span - 1,
            None => span - 1 + extent > i64::MAX,
        },
))]
pub fn position_peak(span: i64, extent: i64) -> Option<i64> {
    (span - 1).checked_add(extent)
}

#[cfg(test)]
#[path = "rules_tests.rs"]
mod tests;
