//! Dependency-free Muse-Glimmer layer schedules and cached key geometry, verified in place by Verus.
//!
//! # Layer schedule
//!
//! Muse counts its hybrid schedule *backward* from the last layer: of `n`
//! layers, layer `i` is a global (full-attention, NoPE) layer exactly when
//! `(n - 1 - i) mod 4 = 0`, so the final layer is always global and three
//! local (sliding, RoPE) layers precede each global one.
//!
//! [`derived_global_layers`](crate::models::muse_glimmer::schedule::derived_global_layers)
//! builds the whole derived schedule, which holds exactly `ceil(n / 4)`
//! global layers.
//!
//! Decode caches store each kind in one homogeneous group, so every layer
//! needs its dense *slot* inside its group: the number of earlier layers of
//! the same kind. [`group_slots`](crate::models::muse_glimmer::schedule::group_slots) computes that map; its proof establishes
//! that each group's slots are exactly `0, 1, …` in layer order, hence
//! injective and dense.
//!
//! # Ring key slices
//!
//! A batched cached step evaluates a `[B, T]` block whose query `k` of row
//! `b` sits at position `L_b + k`. Sliding layers read their ring history in
//! physical slot order (not chronologically) and append the step's fresh
//! block, so the keys are `[history H | block T]` and fresh entry `j` of
//! *every* row is key column `H + j`. Under a window of `W` positions, query
//! `k` sees `L_b + k - W < p <= L_b + k`: fresh entries `k - W < j <= k`
//! and, only while `k + 1 < W`, some history slots. A query chunk `s..e`
//! therefore needs exactly
//!
//! ```text
//! keys = H + s + 1 - W .. H + e     if s + 1 >= W   (no history)
//!      = 0             .. H + e     otherwise       (the whole ring)
//! ```
//!
//! which is row-independent and at most `H + W - 1 + (e - s)` keys wide
//! ([`ring_chunk_keys`](crate::models::muse_glimmer::schedule::ring_chunk_keys)),
//! instead of every key of the step. Every fresh key of the slice is visible
//! to some row of the chunk, so the slice is exactly the chunk's window
//! union whenever it skips the history.
//!
//! # Chunk planning
//!
//! The attention policy sizes query chunks with a key bound per slicing
//! rule ([`planning_bound`](crate::models::muse_glimmer::schedule::planning_bound)),
//! which the proof shows is never exceeded by a realized ring or prefix
//! slice of a chunk it admits; position-ordered cache reads take their key
//! prefix from verified limits
//! ([`prefix_chunk_limit`](crate::models::muse_glimmer::schedule::prefix_chunk_limit)),
//! and every realized plane's element count is an exact checked product
//! ([`plane_elements`](crate::models::muse_glimmer::schedule::plane_elements)).
//!
//! # Cache groups
//!
//! Decode caches hold at most two homogeneous groups: a sliding ring (index
//! `0` when present) and a full-history group after it
//! ([`cache_groups`](crate::models::muse_glimmer::schedule::cache_groups)).
//!
//! All functions are checked through `verus/deepseek_muse.rs`.

#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

#[cfg(verus_keep_ghost)]
verus! {

/// Number of entries of `kinds` equal to `kind`.
///
/// Twin of `count_kind` in `src/models/qwen3_5/plan.rs` (checked by
/// `verus/qwen.rs`); keep the two definitions identical.
pub open spec fn count_kind(kinds: Seq<bool>, kind: bool) -> nat
    decreases kinds.len(),
{
    if kinds.len() == 0 {
        0
    } else {
        count_kind(kinds.drop_last(), kind) + if kinds.last() == kind { 1nat } else { 0nat }
    }
}

/// Extending a prefix by one entry adds one exactly when the entry matches.
///
/// Twin of `lemma_count_extend` in `src/models/qwen3_5/plan.rs`.
pub proof fn lemma_count_extend(kinds: Seq<bool>, end: int, kind: bool)
    requires
        0 <= end < kinds.len(),
    ensures
        count_kind(kinds.subrange(0, end + 1), kind) == count_kind(kinds.subrange(0, end), kind)
            + if kinds[end] == kind { 1nat } else { 0nat },
{
    assert(kinds.subrange(0, end + 1).drop_last() =~= kinds.subrange(0, end));
}

/// Appending an entry adds one to its kind's count.
///
/// Twin of `lemma_count_push` in `src/models/qwen3_5/plan.rs`.
pub proof fn lemma_count_push(kinds: Seq<bool>, entry: bool, kind: bool)
    ensures
        count_kind(kinds.push(entry), kind) == count_kind(kinds, kind)
            + if entry == kind { 1nat } else { 0nat },
{
    assert(kinds.push(entry).drop_last() =~= kinds);
}

/// Multiples of four below `m` gain one exactly when `m - 1` is one.
///
/// With `ceil(m / 4) = (m + 3) / 4`: `(m + 3) / 4 = (m + 2) / 4 + [(m - 1) mod 4 = 0]`.
pub proof fn lemma_quarter_step(m: int)
    requires
        m >= 1,
    ensures
        (m + 3) / 4 == (m + 2) / 4 + if (m - 1) % 4 == 0 { 1int } else { 0int },
{
    vstd::arithmetic::div_mod::lemma_fundamental_div_mod(m - 1, 4);
    let quotient = (m - 1) / 4;
    let remainder = (m - 1) % 4;
    vstd::arithmetic::div_mod::lemma_fundamental_div_mod_converse(m + 3, 4, quotient + 1, remainder);
    if remainder == 0 {
        vstd::arithmetic::div_mod::lemma_fundamental_div_mod_converse(m + 2, 4, quotient, 3);
    } else {
        vstd::arithmetic::div_mod::lemma_fundamental_div_mod_converse(
            m + 2,
            4,
            quotient + 1,
            remainder - 1,
        );
    }
}

/// A sequence containing an entry of some kind counts at least one entry of that kind.
pub proof fn lemma_count_positive(kinds: Seq<bool>, index: int)
    requires
        0 <= index < kinds.len(),
    ensures
        count_kind(kinds, kinds[index]) > 0,
    decreases kinds.len(),
{
    if index < kinds.len() - 1 {
        lemma_count_positive(kinds.drop_last(), index);
        assert(kinds.drop_last()[index] == kinds[index]);
    }
}

/// A prefix holds at most as many matching entries as it has entries.
pub proof fn lemma_count_bounded(kinds: Seq<bool>, kind: bool)
    ensures
        count_kind(kinds, kind) <= kinds.len(),
    decreases kinds.len(),
{
    if kinds.len() > 0 {
        lemma_count_bounded(kinds.drop_last(), kind);
    }
}

/// Fresh block entry `j` is visible to block query `k` under a `window`-position window.
///
/// Both sit in the same row at positions `L_b + j` and `L_b + k`, so the
/// causal and window rules reduce to `k - window < j <= k`.
pub open spec fn fresh_visible(k: int, j: int, window: int) -> bool {
    0 <= j <= k && k < j + window
}

/// Fresh key `j` is visible to some query row `start <= k < end`.
pub open spec fn seen_by_chunk(start: int, end: int, j: int, window: int) -> bool {
    exists|k: int| start <= k < end && #[trigger] fresh_visible(k, j, window)
}

/// First key of the ring slice of query rows starting at `start` (module documentation).
pub open spec fn ring_lo(history: int, start: int, window: int) -> int {
    if start + 1 < window {
        0
    } else {
        history + start + 1 - window
    }
}

/// Keys of the ring slice of query rows `start..end` over `history` ring slots.
pub open spec fn ring_width(history: int, start: int, end: int, window: int) -> int {
    history + end - ring_lo(history, start, window)
}

/// The widest aligned window union of `rows` query rows: `rows + W - 1` keys, at most `K`.
pub open spec fn aligned_width(rows: int, window: int, keys: int) -> int {
    min_int(rows + window - 1, keys)
}

/// `a + b`, saturated at `usize::MAX`.
pub open spec fn sat_add(a: int, b: int) -> int {
    if a + b > usize::MAX {
        usize::MAX as int
    } else {
        a + b
    }
}

/// `a - b`, saturated at zero.
pub open spec fn sat_sub(a: int, b: int) -> int {
    if a < b {
        0
    } else {
        a - b
    }
}

/// The smaller of two integers, `vstd::math::min`.
pub open spec fn min_int(a: int, b: int) -> int {
    vstd::math::min(a, b)
}

} // verus!

/// Returns whether layer `index` of `layers` is a global (full-attention) layer.
///
/// Global layers sit at `(layers - 1 - index) mod 4 = 0`, counted backward
/// so the last layer is global.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(global =>
    requires
        index < layers,
    ensures
        global == ((layers - 1 - index) % 4 == 0),
        index + 1 == layers ==> global,
))]
pub(crate) fn is_global_layer(layers: usize, index: usize) -> bool {
    (layers - 1 - index).is_multiple_of(4)
}

/// Returns the derived backward-counted schedule of `layers` layers (`true` = global).
///
/// Entry `i` is [`is_global_layer`]`(layers, i)`. Proved: every entry is
/// exactly that rule, and the schedule holds exactly `ceil(layers / 4)`
/// global layers — one closing each group of four counted back from the
/// last layer (13 of the released 52).
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(global =>
    ensures
        global.len() == layers,
        forall|i: int| 0 <= i < layers ==> #[trigger] global@[i] == ((layers - 1 - i) % 4 == 0),
        count_kind(global@, true) == (layers + 3) / 4,
))]
pub(crate) fn derived_global_layers(layers: usize) -> Vec<bool> {
    let mut global: Vec<bool> = Vec::with_capacity(layers);
    let mut index: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            index <= layers,
            global.len() == index,
            forall|i: int| 0 <= i < index ==> #[trigger] global@[i] == ((layers - 1 - i) % 4 == 0),
            count_kind(global@, true) == (layers + 3) / 4 - ((layers - index) as int + 3) / 4,
        decreases layers - index,
    ))]
    while index < layers {
        let kind = is_global_layer(layers, index);
        #[cfg(verus_keep_ghost)]
        proof! {
            lemma_count_push(global@, kind, true);
            lemma_quarter_step((layers - index) as int);
        }
        global.push(kind);
        index += 1;
    }
    global
}

/// Returns each layer's dense slot within its kind's cache group.
///
/// `global[i]` marks layer `i` as global; the slot of layer `i` is the number
/// of earlier layers of the same kind, so each group's slots run `0, 1, …`
/// in layer order.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(slots =>
    ensures
        slots.len() == global.len(),
        forall|i: int| 0 <= i < global.len()
            ==> slots@[i] == count_kind(global@.subrange(0, i), global@[i]),
        forall|i: int, j: int| 0 <= i < j < global.len() && global@[i] == global@[j]
            ==> slots@[i] < slots@[j],
))]
pub(crate) fn group_slots(global: &[bool]) -> Vec<usize> {
    let mut slots: Vec<usize> = Vec::with_capacity(global.len());
    let mut next_global: usize = 0;
    let mut next_local: usize = 0;
    let mut index: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            index <= global.len(),
            slots.len() == index,
            next_global == count_kind(global@.subrange(0, index as int), true),
            next_local == count_kind(global@.subrange(0, index as int), false),
            forall|i: int| 0 <= i < index
                ==> slots@[i] == count_kind(global@.subrange(0, i), global@[i]),
            forall|i: int, j: int| 0 <= i < j < index && global@[i] == global@[j]
                ==> slots@[i] < slots@[j],
            forall|i: int| 0 <= i < index
                ==> slots@[i] < count_kind(global@.subrange(0, index as int), global@[i]),
        decreases global.len() - index,
    ))]
    while index < global.len() {
        #[cfg(verus_keep_ghost)]
        proof! {
            lemma_count_extend(global@, index as int, true);
            lemma_count_extend(global@, index as int, false);
            lemma_count_bounded(global@.subrange(0, index as int), true);
            lemma_count_bounded(global@.subrange(0, index as int), false);
        }
        if global[index] {
            slots.push(next_global);
            next_global += 1;
        } else {
            slots.push(next_local);
            next_local += 1;
        }
        index += 1;
    }
    slots
}

/// Returns the key slice `(lo, hi)` that block query rows `start..end` need over `[history | block]`.
///
/// See the module documentation: rows at or past `window - 1` skip the ring
/// history and read exactly the union of their windows; earlier rows read
/// the whole history (its slots are in ring order, so no narrower contiguous
/// slice is exact); and every slice ends at the chunk's last fresh entry.
/// Requires `start < end` and `window > 0`; returns `None` only when
/// `history + end` overflows `usize`. Proved: the slice contains every
/// visible fresh key, starts exactly at `ring_lo`, is at most
/// `H + W - 1 + (end - start)` keys wide, and every fresh key it covers is
/// visible to some row of the chunk (the window union is tight).
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(range =>
    requires
        start < end,
        window > 0,
    ensures
        match range {
            Some((lo, hi)) => {
                &&& hi == history + end
                &&& lo <= hi
                &&& forall|k: int, j: int| start <= k < end && #[trigger] fresh_visible(k, j, window as int)
                        ==> lo <= history + j && history + j < hi
                &&& lo == ring_lo(history as int, start as int, window as int)
                &&& (start + 1 < window ==> lo == 0)
                &&& (start + 1 >= window ==> history <= lo && lo == history + start + 1 - window)
                &&& hi - lo <= history + window - 1 + (end - start)
                // Tightness: every fresh key of the slice is visible to a row of the chunk.
                &&& forall|j: int| 0 <= j && lo <= history + j < hi
                    ==> #[trigger] seen_by_chunk(start as int, end as int, j, window as int)
            },
            None => history + end > usize::MAX,
        },
))]
pub(crate) fn ring_chunk_keys(
    history: usize,
    start: usize,
    end: usize,
    window: usize,
) -> Option<(usize, usize)> {
    let hi = history.checked_add(end)?;
    // `start < end` keeps `start + 1` in range, and `lo <= history + start < hi`.
    let lo = if start >= window - 1 {
        history + (start + 1 - window)
    } else {
        0
    };
    #[cfg(verus_keep_ghost)]
    proof! {
        // Row `max(j, start)` sees fresh key `j`: it is causal, and the
        // window reaches back to `start + 1 - W <= j` (or to before the block).
        assert forall|j: int| 0 <= j && lo <= history + j < hi implies
            #[trigger] seen_by_chunk(start as int, end as int, j, window as int) by {
            let k = if j >= start { j } else { start as int };
            assert(fresh_visible(k, j, window as int));
        }
    }
    Some((lo, hi))
}

/// Integer view of a Muse key-slicing rule (see `MuseKeySlicing`).
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum KeyRule {
    /// Keys aligned with the queries under a column window of `window` keys.
    AlignedWindow {
        /// Local window in key columns.
        window: usize,
    },
    /// Keys aligned with the queries; a chunk reads its causal prefix.
    AlignedPrefix,
    /// `[ring history | fresh block]` under a window of `window` positions.
    Ring {
        /// Local window in positions.
        window: usize,
    },
    /// Position-ordered keys with precomputed causal prefix limits.
    Limits,
}

/// Returns the widest key slice of any chunk of at most `maximum_query` rows.
///
/// Aligned windows need `W + q - 1` keys, ring chunks at most
/// `(K - Q) + W - 1 + q`, and causal prefixes up to all `K` keys; every
/// bound saturates and is clamped to `K`. Proved: the result is exactly that
/// formula, and it is sound: every ring slice ([`ring_chunk_keys`]) of a
/// chunk with at most `maximum_query` rows fits within it, as does every
/// aligned window union of `rows <= maximum_query` rows, which spans at most
/// `min(rows + W - 1, K)` keys (the bound `sliding_window_key_range` is
/// proved to respect). Prefix and limit
/// slices lie within `0..K`, the bound itself. So the planner's audit of
/// realized slices never rejects a chunk the policy admitted.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(bound =>
    ensures
        bound <= keys,
        match rule {
            KeyRule::AlignedWindow { window } => {
                &&& bound == min_int(sat_add(window as int, sat_sub(maximum_query as int, 1)), keys as int)
                &&& forall|rows: int| 1 <= rows <= maximum_query
                    ==> #[trigger] aligned_width(rows, window as int, keys as int) <= bound
            },
            KeyRule::AlignedPrefix => bound == keys,
            KeyRule::Limits => bound == keys,
            KeyRule::Ring { window } => {
                &&& bound == min_int(sat_add(sat_add(sat_sub(keys as int, queries as int),
                    sat_sub(window as int, 1)), maximum_query as int), keys as int)
                &&& (window > 0 && queries <= keys ==> forall|start: int, end: int|
                    0 <= start < end <= queries && end - start <= maximum_query
                    ==> #[trigger] ring_width((keys - queries) as int, start, end, window as int) <= bound)
            },
        },
))]
pub(crate) fn planning_bound(
    rule: KeyRule,
    queries: usize,
    keys: usize,
    maximum_query: usize,
) -> usize {
    let bound = match rule {
        KeyRule::AlignedWindow { window } => window.saturating_add(maximum_query.saturating_sub(1)),
        KeyRule::AlignedPrefix | KeyRule::Limits => keys,
        KeyRule::Ring { window } => keys
            .saturating_sub(queries)
            .saturating_add(window.saturating_sub(1))
            .saturating_add(maximum_query),
    };
    bound.min(keys)
}

/// Returns the key prefix `0..λ` that position-ordered query rows ending at `end` read.
///
/// `limits[i]` bounds the keys rows `0..=i` need. Proved: `Some(λ)` exactly
/// when the limits cover all `queries` rows and `λ = limits[end - 1]` lies
/// in `1..=keys`; `None` otherwise.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(limit =>
    requires
        end >= 1,
    ensures
        match limit {
            Some(limit) => {
                &&& limits.len() == queries
                &&& end - 1 < limits.len()
                &&& limit == limits@[end - 1]
                &&& 1 <= limit <= keys
            },
            None => !(limits.len() == queries && end - 1 < limits.len()
                && 1 <= limits@[end - 1] <= keys),
        },
))]
pub(crate) fn prefix_chunk_limit(
    limits: &[usize],
    queries: usize,
    keys: usize,
    end: usize,
) -> Option<usize> {
    if end > limits.len() {
        return None;
    }
    let limit = limits[end - 1];
    if limits.len() == queries && 1 <= limit && limit <= keys {
        Some(limit)
    } else {
        None
    }
}

/// Returns the element count of a `[batch, heads, rows, keys]` score plane, or `None` on overflow.
///
/// Multiplies left to right like the planner's checked fold. Proved:
/// `Some(n)` exactly when no partial product overflows, with
/// `n = batch * heads * rows * keys`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(elements =>
    ensures
        match elements {
            Some(elements) => {
                &&& batch * heads <= usize::MAX
                &&& batch * heads * rows <= usize::MAX
                &&& elements == batch * heads * rows * keys
            },
            None => batch * heads > usize::MAX || batch * heads * rows > usize::MAX
                || batch * heads * rows * keys > usize::MAX,
        },
))]
pub(crate) fn plane_elements(
    batch: usize,
    heads: usize,
    rows: usize,
    keys: usize,
) -> Option<usize> {
    let heads_total = batch.checked_mul(heads)?;
    let rows_total = heads_total.checked_mul(rows)?;
    rows_total.checked_mul(keys)
}

/// Why the decode cache groups cannot be planned.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GroupFault {
    /// Sliding layers exist but the window is zero.
    ZeroWindow,
    /// Sliding layers exist but no window is configured.
    MissingWindow,
}

/// Decode cache groups of a Muse schedule (see [`cache_groups`]).
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct CacheGroups {
    /// Number of local (sliding) layers.
    pub(crate) sliding_layers: usize,
    /// Number of global (full-history) layers.
    pub(crate) full_layers: usize,
    /// The ring's retained window; `None` without sliding layers.
    pub(crate) window: Option<usize>,
    /// Bank index of the sliding ring, present exactly with sliding layers.
    pub(crate) sliding_group: Option<usize>,
    /// Bank index of the full-history group, present exactly with global layers.
    pub(crate) full_group: Option<usize>,
    /// Bank group of every layer.
    pub(crate) groups: Vec<usize>,
    /// Dense slot of every layer inside its group.
    pub(crate) slots: Vec<usize>,
}

/// Plans the decode cache groups and every layer's `(group, slot)`.
///
/// `global[i]` marks layer `i` as global. The sliding ring (when any layer
/// is local) is group `0` and the full-history group follows it. Proved:
/// the counts are exact, a sliding group exists exactly with local layers
/// and then needs a non-zero window (each error names the missing piece),
/// the full group exists exactly with global layers, every layer belongs to
/// its kind's (existing) group, and slots are [`group_slots`] — so no layer
/// can be mapped to an absent group.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(plan =>
    ensures
        ({
            let sliding = count_kind(global@, false);
            let full = count_kind(global@, true);
            match plan {
                Ok(plan) => {
                    &&& plan.sliding_layers == sliding
                    &&& plan.full_layers == full
                    &&& plan.sliding_layers + plan.full_layers == global.len()
                    &&& (sliding == 0 ==> plan.window.is_none() && plan.sliding_group.is_none())
                    &&& (sliding > 0 ==> plan.window == window && window.is_some()
                        && window.unwrap() > 0 && plan.sliding_group == Some(0usize))
                    &&& (full == 0 ==> plan.full_group.is_none())
                    &&& (full > 0 ==> plan.full_group == Some(if sliding > 0 { 1usize } else { 0usize }))
                    &&& plan.groups.len() == global.len()
                    &&& plan.slots.len() == global.len()
                    &&& forall|i: int| 0 <= i < global.len() && global@[i]
                        ==> plan.full_group == Some(#[trigger] plan.groups@[i])
                    &&& forall|i: int| 0 <= i < global.len() && !global@[i]
                        ==> plan.sliding_group == Some(#[trigger] plan.groups@[i])
                    &&& forall|i: int| 0 <= i < global.len()
                        ==> #[trigger] plan.slots@[i] == count_kind(global@.subrange(0, i), global@[i])
                },
                Err(GroupFault::ZeroWindow) => sliding > 0 && window == Some(0usize),
                Err(GroupFault::MissingWindow) => sliding > 0 && window.is_none(),
            }
        }),
))]
pub(crate) fn cache_groups(
    global: &[bool],
    window: Option<usize>,
) -> Result<CacheGroups, GroupFault> {
    let mut sliding_layers: usize = 0;
    let mut full_layers: usize = 0;
    let mut index: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            index <= global.len(),
            sliding_layers + full_layers == index,
            sliding_layers == count_kind(global@.subrange(0, index as int), false),
            full_layers == count_kind(global@.subrange(0, index as int), true),
        decreases global.len() - index,
    ))]
    while index < global.len() {
        #[cfg(verus_keep_ghost)]
        proof! {
            lemma_count_extend(global@, index as int, true);
            lemma_count_extend(global@, index as int, false);
        }
        if global[index] {
            full_layers += 1;
        } else {
            sliding_layers += 1;
        }
        index += 1;
    }
    #[cfg(verus_keep_ghost)]
    proof! {
        assert(global@.subrange(0, global.len() as int) =~= global@);
    }
    let window = if sliding_layers == 0 {
        None
    } else {
        match window {
            Some(0) => return Err(GroupFault::ZeroWindow),
            Some(window) => Some(window),
            None => return Err(GroupFault::MissingWindow),
        }
    };
    let sliding_group = if sliding_layers > 0 { Some(0) } else { None };
    let full_group = if full_layers == 0 {
        None
    } else if sliding_layers > 0 {
        Some(1)
    } else {
        Some(0)
    };
    let slots = group_slots(global);
    let mut groups: Vec<usize> = Vec::with_capacity(global.len());
    let mut layer: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            layer <= global.len(),
            groups.len() == layer,
            sliding_layers == count_kind(global@, false),
            full_layers == count_kind(global@, true),
            sliding_group == (if sliding_layers > 0 { Some(0usize) } else { None }),
            full_group == (if full_layers == 0 { None } else if sliding_layers > 0 {
                Some(1usize) } else { Some(0usize) }),
            forall|i: int| 0 <= i < layer && global@[i] ==> full_group == Some(#[trigger] groups@[i]),
            forall|i: int| 0 <= i < layer && !global@[i] ==> sliding_group == Some(#[trigger] groups@[i]),
        decreases global.len() - layer,
    ))]
    while layer < global.len() {
        #[cfg(verus_keep_ghost)]
        proof! {
            // A layer of either kind makes its kind's count positive.
            lemma_count_positive(global@, layer as int);
        }
        groups.push(if global[layer] {
            if sliding_layers > 0 { 1 } else { 0 }
        } else {
            0
        });
        layer += 1;
    }
    Ok(CacheGroups {
        sliding_layers,
        full_layers,
        window,
        sliding_group,
        full_group,
        groups,
        slots,
    })
}

#[cfg(test)]
#[path = "schedule_tests.rs"]
/// Schedule and slot properties against brute-force oracles.
mod tests;
