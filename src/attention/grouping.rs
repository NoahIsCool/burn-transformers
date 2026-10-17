//! Pure head-grouping arithmetic for grouped-query and differential attention.
//!
//! # Group folding
//!
//! Grouped-query attention maps `H` query heads onto `G` key/value heads with
//! `g = H / G` query heads per group; query head `h` reads key/value head
//! `kv(h) = floor(h / g)` and is member `j = h mod g` of that group.
//! Folding reshapes `[batch, H, rows, x]` to `[batch, G, g * rows, x]`, so
//! token `t` of head `h` becomes folded row `j * rows + t` of head `kv(h)`.
//! Because `H = G * g`, the flat (row-major) index of every element is
//! unchanged:
//!
//! `(b * H + h) * rows + t = (b * G + kv(h)) * (g * rows) + j * rows + t`,
//!
//! so the reshape is a view of a contiguous tensor, and every folded row
//! `r < g * rows` of head `k` comes from exactly one query head
//! `k * g + floor(r / rows)` and token `r mod rows`. The one-hot sink selector
//! uses the same member index `floor(r / rows)`.
//!
//! # Differential pairing
//!
//! Differential attention pairs query head `i < H / 2` with `i + H / 2`; both
//! heads use the paired value `[V_kv(i) ; V_kv(i + H/2)]`. With `n` value
//! heads (`n = G` natively, `n = H` after repeating K/V) and `half = n / 2`,
//! the evaluation builds `W_m = [V_(m mod half) ; V_(m mod half + half)]` for
//! value head `m`, and query head `h` reads `W_(h / (H / n))`. That is the
//! reference pair exactly when, for `i = h mod (H / 2)`,
//!
//! `kv(h) mod half = kv(i)` and `kv(h) mod half + half = kv(i + H / 2)`,
//!
//! which holds for every even `G`. After repetition `n = H` and value head
//! `m` is a copy of source head `m / (H / G)`, so the identity holds by
//! definition for every `G`. A single key/value head pairs with itself.
//!
//! # Verified properties
//!
//! Verus checks every function here through `verus/attention.rs`: group
//! counts are exact, repetition reads source head `m / g` for repeated head
//! `m`, folding preserves every flat index and is a bijection between
//! `(head, token)` and `(kv head, folded row)`, unfolding inverts it, and
//! native pairing reproduces the reference pairs for every query head. That
//! Burn's `reshape` is row-major and `expand` broadcasts is trusted and
//! covered by tests.

#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

#[cfg(verus_keep_ghost)]
verus! {

/// Query heads per key/value head, `query_heads / key_value_heads`.
pub open spec fn group_size(query_heads: usize, key_value_heads: usize) -> int {
    (query_heads / key_value_heads) as int
}

/// Row-major flat index of `[batch, head, row]` in a `[_, heads, rows]` layout
/// (trailing axes fixed).
pub open spec fn flat_index(batch: int, head: int, row: int, heads: int, rows: int) -> int {
    (batch * heads + head) * rows + row
}

/// Row-major flat index of `[batch, kv_head, member, row]` in a
/// `[_, kv_heads, groups, rows]` layout.
pub open spec fn expanded_index(
    batch: int,
    kv_head: int,
    member: int,
    row: int,
    kv_heads: int,
    groups: int,
    rows: int,
) -> int {
    ((batch * kv_heads + kv_head) * groups + member) * rows + row
}

/// Key/value head read by query head `head` when `groups` query heads share one.
pub open spec fn kv_head_of(head: int, groups: int) -> int {
    head / groups
}

/// Folded row of token `row` of query head `head`: member `head mod groups`
/// times `rows`, plus `row`.
pub open spec fn folded_row_of(head: int, row: int, groups: int, rows: int) -> int {
    (head % groups) * rows + row
}

/// Group member that folded row `folded` belongs to: `floor(folded / rows)`
/// (the sink selector channel).
pub open spec fn member_of(folded: int, rows: int) -> int {
    folded / rows
}

/// Query head that folded row `folded` of key/value head `kv_head` came from.
pub open spec fn unfolded_head_of(kv_head: int, folded: int, groups: int, rows: int) -> int {
    kv_head * groups + member_of(folded, rows)
}

/// Value pair of query head `head` as built: `(kv(h) mod half, kv(h) mod half + half)`.
pub open spec fn built_pair(head: int, groups: int, half: int) -> (int, int) {
    (kv_head_of(head, groups) % half, kv_head_of(head, groups) % half + half)
}

/// Reference value pair of query head `head`: heads of `i = h mod pairs` and `i + pairs`.
pub open spec fn reference_pair(head: int, groups: int, pairs: int) -> (int, int) {
    (kv_head_of(head % pairs, groups), kv_head_of(head % pairs + pairs, groups))
}

/// An exact quotient of grouped heads: `q = kv * (q / kv)`, positive when `q` is.
proof fn lemma_group_quotient(query_heads: int, key_value_heads: int)
    requires
        key_value_heads >= 1,
        query_heads >= 0,
        query_heads % key_value_heads == 0,
    ensures
        query_heads == key_value_heads * (query_heads / key_value_heads),
        query_heads / key_value_heads >= 0,
        query_heads >= 1 ==> query_heads / key_value_heads >= 1,
{
    vstd::arithmetic::div_mod::lemma_fundamental_div_mod(query_heads, key_value_heads);
    let groups = query_heads / key_value_heads;
    assert(groups >= 0) by (nonlinear_arith)
        requires
            query_heads == key_value_heads * groups,
            key_value_heads >= 1,
            query_heads >= 0;
    if query_heads >= 1 {
        assert(groups >= 1) by (nonlinear_arith)
            requires
                query_heads == key_value_heads * groups,
                query_heads >= 1,
                groups >= 0;
    }
}

/// Folding `head = kv * groups + member` keeps the flat index of every element.
proof fn lemma_fold_index(
    batch: int,
    query_heads: int,
    key_value_heads: int,
    groups: int,
    rows: int,
    head: int,
    row: int,
)
    requires
        groups >= 1,
        rows >= 1,
        query_heads == key_value_heads * groups,
        0 <= head < query_heads,
        0 <= row < rows,
        batch >= 0,
    ensures
        0 <= kv_head_of(head, groups) < key_value_heads,
        0 <= folded_row_of(head, row, groups, rows) < groups * rows,
        flat_index(batch, head, row, query_heads, rows) == flat_index(
            batch,
            kv_head_of(head, groups),
            folded_row_of(head, row, groups, rows),
            key_value_heads,
            groups * rows,
        ),
{
    let kv = head / groups;
    let member = head % groups;
    vstd::arithmetic::div_mod::lemma_fundamental_div_mod(head, groups);
    vstd::arithmetic::div_mod::lemma_mod_pos_bound(head, groups);
    assert(0 <= kv < key_value_heads) by (nonlinear_arith)
        requires
            head == groups * kv + member,
            0 <= member < groups,
            0 <= head < key_value_heads * groups,
            groups >= 1;
    assert(0 <= member * rows + row < groups * rows) by (nonlinear_arith)
        requires
            0 <= member < groups,
            0 <= row < rows;
    assert((batch * query_heads + head) * rows + row
        == (batch * key_value_heads + kv) * (groups * rows) + member * rows + row)
        by (nonlinear_arith)
        requires
            head == groups * kv + member,
            query_heads == key_value_heads * groups;
}

/// Every folded row of every key/value head comes from one query head and token.
proof fn lemma_unfold_index(
    key_value_heads: int,
    groups: int,
    rows: int,
    kv_head: int,
    folded: int,
)
    requires
        groups >= 1,
        rows >= 1,
        0 <= kv_head < key_value_heads,
        0 <= folded < groups * rows,
    ensures
        0 <= member_of(folded, rows) < groups,
        0 <= unfolded_head_of(kv_head, folded, groups, rows) < key_value_heads * groups,
        kv_head_of(unfolded_head_of(kv_head, folded, groups, rows), groups) == kv_head,
        folded_row_of(unfolded_head_of(kv_head, folded, groups, rows), folded % rows, groups, rows)
            == folded,
{
    let member = folded / rows;
    let head = kv_head * groups + member;
    vstd::arithmetic::div_mod::lemma_fundamental_div_mod(folded, rows);
    vstd::arithmetic::div_mod::lemma_mod_pos_bound(folded, rows);
    assert(0 <= member < groups) by (nonlinear_arith)
        requires
            folded == rows * member + folded % rows,
            0 <= folded % rows < rows,
            0 <= folded < groups * rows;
    assert(0 <= head < key_value_heads * groups) by (nonlinear_arith)
        requires
            0 <= member < groups,
            0 <= kv_head < key_value_heads,
            head == kv_head * groups + member;
    vstd::arithmetic::div_mod::lemma_fundamental_div_mod_converse(head, groups, kv_head, member);
    assert(member * rows + folded % rows == folded) by (nonlinear_arith)
        requires folded == rows * member + folded % rows;
}

/// Native pairing over an even number of key/value heads reproduces the reference pair.
proof fn lemma_native_pairing(query_heads: int, key_value_heads: int, groups: int, head: int)
    requires
        groups >= 1,
        key_value_heads >= 2,
        key_value_heads % 2 == 0,
        query_heads == key_value_heads * groups,
        0 <= head < query_heads,
    ensures
        built_pair(head, groups, key_value_heads / 2) == reference_pair(
            head,
            groups,
            query_heads / 2,
        ),
{
    let half = key_value_heads / 2;
    let pairs = half * groups;
    assert(key_value_heads == 2 * half);
    assert(query_heads == 2 * pairs) by (nonlinear_arith)
        requires
            query_heads == key_value_heads * groups,
            key_value_heads == 2 * half,
            pairs == half * groups;
    assert(query_heads / 2 == pairs);
    let kv = head / groups;
    vstd::arithmetic::div_mod::lemma_fundamental_div_mod(head, groups);
    vstd::arithmetic::div_mod::lemma_mod_pos_bound(head, groups);
    let member = head % groups;
    assert(0 <= kv < 2 * half) by (nonlinear_arith)
        requires
            head == groups * kv + member,
            0 <= member < groups,
            0 <= head < 2 * half * groups,
            groups >= 1;
    if head < pairs {
        // Low half: the head is its own `i`, and `i + pairs` lies `half` KV heads later.
        vstd::arithmetic::div_mod::lemma_small_mod(head as nat, pairs as nat);
        assert(kv < half) by (nonlinear_arith)
            requires
                head == groups * kv + member,
                0 <= member,
                head < half * groups,
                groups >= 1;
        vstd::arithmetic::div_mod::lemma_small_mod(kv as nat, half as nat);
        assert(head + pairs == groups * (kv + half) + member) by (nonlinear_arith)
            requires
                head == groups * kv + member,
                pairs == half * groups;
        vstd::arithmetic::div_mod::lemma_fundamental_div_mod_converse(
            head + pairs,
            groups,
            kv + half,
            member,
        );
    } else {
        // High half: `i = head - pairs` lies `half` KV heads earlier.
        let low = head - pairs;
        assert(0 <= low < pairs);
        vstd::arithmetic::div_mod::lemma_fundamental_div_mod_converse(head, pairs, 1, low);
        assert(kv >= half) by (nonlinear_arith)
            requires
                head == groups * kv + member,
                member < groups,
                head >= half * groups,
                groups >= 1;
        assert(low == groups * (kv - half) + member) by (nonlinear_arith)
            requires
                head == groups * kv + member,
                pairs == half * groups,
                low == head - pairs;
        vstd::arithmetic::div_mod::lemma_fundamental_div_mod_converse(
            low,
            groups,
            kv - half,
            member,
        );
        vstd::arithmetic::div_mod::lemma_fundamental_div_mod_converse(kv, half, 1, kv - half);
        assert(low + pairs == head);
    }
}

} // verus!

/// Query heads per key/value head, or `None` unless both counts are non-zero and `kv` divides `q`.
///
/// Verified: the group size is exact, and repeating key/value heads by
/// expanding `[b, G, T, x]` to `[b, G, g, T, x]` and reshaping to
/// `[b, H, T, x]` (as `repeat_key_value` does) places element `[b, m, t]` at
/// expanded position `[b, m / g, m mod g, t]`. Expansion copies source head
/// `m / g` into every member, so repeated head `m` reads source head `m / g`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(groups =>
    ensures
        match groups {
            Some(groups) => {
                &&& query_heads >= 1 && key_value_heads >= 1
                &&& groups >= 1
                &&& groups * key_value_heads == query_heads
                &&& forall|batch: int, head: int, rows: int, row: int|
                    0 <= batch && 0 <= head < query_heads && 0 <= row < rows
                        ==> #[trigger] flat_index(batch, head, row, query_heads as int, rows)
                        == expanded_index(
                        batch,
                        kv_head_of(head, groups as int),
                        head % (groups as int),
                        row,
                        key_value_heads as int,
                        groups as int,
                        rows,
                    )
            },
            None => query_heads == 0 || key_value_heads == 0 || query_heads % key_value_heads != 0,
        }
))]
pub fn head_groups(query_heads: usize, key_value_heads: usize) -> Option<usize> {
    if key_value_heads == 0 || query_heads == 0 {
        return None;
    }
    if !query_heads.is_multiple_of(key_value_heads) {
        return None;
    }
    let groups = query_heads / key_value_heads;
    #[cfg(verus_keep_ghost)]
    proof! {
        lemma_group_quotient(query_heads as int, key_value_heads as int);
        assert(groups * key_value_heads == key_value_heads * groups) by (nonlinear_arith);
        assert forall|batch: int, head: int, rows: int, row: int|
            0 <= batch && 0 <= head < query_heads && 0 <= row < rows implies #[trigger]
            flat_index(batch, head, row, query_heads as int, rows) == expanded_index(
                batch,
                kv_head_of(head, groups as int),
                head % (groups as int),
                row,
                key_value_heads as int,
                groups as int,
                rows,
            ) by {
            lemma_fold_index(
                batch,
                query_heads as int,
                key_value_heads as int,
                groups as int,
                rows,
                head,
                row,
            );
            let kv = kv_head_of(head, groups as int);
            let member = head % (groups as int);
            // The folded layout `[_, G, g * rows]` is the expanded `[_, G, g, rows]` layout.
            assert(flat_index(batch, kv, member * rows + row, key_value_heads as int, groups * rows)
                == expanded_index(
                batch,
                kv,
                member,
                row,
                key_value_heads as int,
                groups as int,
                rows,
            )) by (nonlinear_arith);
        }
    }
    Some(groups)
}

/// Rows of one folded key/value head: `(query_heads / key_value_heads) * rows`.
///
/// Requires grouped heads and a representable `query_heads * rows` (every
/// `[batch, query_heads, rows, x]` tensor satisfies it). Folding query head
/// `h`, token `t` to key/value head `h / g`, row `(h mod g) * rows + t`
/// preserves every flat index, and every folded row comes from exactly one
/// query head and token (see the module documentation).
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(folded =>
    requires
        key_value_heads >= 1,
        query_heads % key_value_heads == 0,
        query_heads * rows <= usize::MAX,
    ensures
        folded == group_size(query_heads, key_value_heads) * rows,
        forall|head: int|
            0 <= head < query_heads ==> 0 <= #[trigger] kv_head_of(
                head,
                group_size(query_heads, key_value_heads),
            ) < key_value_heads,
        forall|head: int, row: int|
            0 <= head < query_heads && 0 <= row < rows ==> 0 <= #[trigger] folded_row_of(
                head,
                row,
                group_size(query_heads, key_value_heads),
                rows as int,
            ) < folded,
        forall|batch: int, head: int, row: int|
            0 <= batch && 0 <= head < query_heads && 0 <= row < rows ==> #[trigger] flat_index(
                batch,
                head,
                row,
                query_heads as int,
                rows as int,
            ) == flat_index(
                batch,
                kv_head_of(head, group_size(query_heads, key_value_heads)),
                folded_row_of(head, row, group_size(query_heads, key_value_heads), rows as int),
                key_value_heads as int,
                folded as int,
            ),
        forall|row: int|
            0 <= row < folded ==> 0 <= #[trigger] member_of(row, rows as int) < group_size(
                query_heads,
                key_value_heads,
            ),
        forall|kv_head: int, row: int|
            0 <= kv_head < key_value_heads && 0 <= row < folded ==> {
                let head = #[trigger] unfolded_head_of(
                    kv_head,
                    row,
                    group_size(query_heads, key_value_heads),
                    rows as int,
                );
                &&& 0 <= head < query_heads
                &&& kv_head_of(head, group_size(query_heads, key_value_heads)) == kv_head
                &&& folded_row_of(
                    head,
                    row % (rows as int),
                    group_size(query_heads, key_value_heads),
                    rows as int,
                ) == row
            },
))]
pub fn folded_rows(query_heads: usize, key_value_heads: usize, rows: usize) -> usize {
    let groups = query_heads / key_value_heads;
    #[cfg(verus_keep_ghost)]
    proof! {
        lemma_group_quotient(query_heads as int, key_value_heads as int);
        assert(groups * rows <= query_heads * rows) by (nonlinear_arith)
            requires
                query_heads as int == key_value_heads as int * groups as int,
                key_value_heads >= 1,
                groups >= 0;
        assert forall|batch: int, head: int, row: int|
            0 <= batch && 0 <= head < query_heads && 0 <= row < rows implies #[trigger]
            flat_index(batch, head, row, query_heads as int, rows as int) == flat_index(
                batch,
                kv_head_of(head, groups as int),
                folded_row_of(head, row, groups as int, rows as int),
                key_value_heads as int,
                groups * rows,
            ) by {
            lemma_fold_index(
                batch,
                query_heads as int,
                key_value_heads as int,
                groups as int,
                rows as int,
                head,
                row,
            );
        }
        let size = group_size(query_heads, key_value_heads);
        assert(size == groups as int);
        assert forall|head: int, row: int| 0 <= head < query_heads && 0 <= row < rows implies
            0 <= #[trigger] folded_row_of(head, row, size, rows as int) < groups * rows by {
            lemma_fold_index(
                0,
                query_heads as int,
                key_value_heads as int,
                groups as int,
                rows as int,
                head,
                row,
            );
        }
        assert forall|head: int| 0 <= head < query_heads implies 0 <= #[trigger] kv_head_of(
            head,
            size,
        ) < key_value_heads by {
            lemma_fold_index(
                0,
                query_heads as int,
                key_value_heads as int,
                groups as int,
                1,
                head,
                0,
            );
        }
        assert forall|row: int| 0 <= row < groups * rows implies 0 <= #[trigger] member_of(
            row,
            rows as int,
        ) < size by {
            assert(groups >= 1 && rows >= 1) by (nonlinear_arith)
                requires
                    0 <= row < groups * rows,
                    groups >= 0,
                    rows >= 0;
            lemma_unfold_index(1, groups as int, rows as int, 0, row);
        }
        assert forall|kv_head: int, row: int|
            0 <= kv_head < key_value_heads && 0 <= row < groups * rows implies {
                let head = #[trigger] unfolded_head_of(kv_head, row, size, rows as int);
                &&& 0 <= head < query_heads
                &&& kv_head_of(head, size) == kv_head
                &&& folded_row_of(head, row % (rows as int), size, rows as int) == row
            } by {
            assert(groups >= 1 && rows >= 1) by (nonlinear_arith)
                requires
                    0 <= row < groups * rows,
                    groups >= 0,
                    rows >= 0;
            lemma_unfold_index(key_value_heads as int, groups as int, rows as int, kv_head, row);
        }
    }
    groups * rows
}

/// Rows per query head after unfolding `folded` rows: `folded / (query_heads / key_value_heads)`.
///
/// Inverts [`folded_rows`]: unfolding `(query_heads / key_value_heads) * rows`
/// folded rows yields exactly `rows`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(rows =>
    requires
        key_value_heads >= 1,
        query_heads >= 1,
        query_heads % key_value_heads == 0,
    ensures
        rows == folded / (query_heads / key_value_heads),
        folded % (query_heads / key_value_heads) == 0 ==> rows * (query_heads / key_value_heads)
            == folded,
))]
pub fn unfolded_rows(query_heads: usize, key_value_heads: usize, folded: usize) -> usize {
    let groups = query_heads / key_value_heads;
    #[cfg(verus_keep_ghost)]
    proof! {
        vstd::arithmetic::div_mod::lemma_fundamental_div_mod(
            query_heads as int,
            key_value_heads as int,
        );
        assert(groups >= 1) by (nonlinear_arith)
            requires
                query_heads as int == key_value_heads as int * groups as int,
                query_heads >= 1;
        vstd::arithmetic::div_mod::lemma_fundamental_div_mod(folded as int, groups as int);
    }
    folded / groups
}

/// How differential attention builds `2d`-wide paired values (see the module documentation).
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValuePairing {
    /// One key/value head pairs with itself: `[V_0 ; V_0]`.
    Shared,
    /// Even native key/value heads: `W_m = [V_(m mod half) ; V_(m mod half + half)]`.
    Native {
        /// Half the key/value heads.
        half: usize,
    },
    /// An odd count above one: K/V repeat to query heads, which then pair natively.
    Repeated {
        /// Half the query heads.
        half: usize,
    },
}

/// Chooses how paired values are built for `query_heads` over `key_value_heads`.
///
/// Requires an even, grouped head layout. Every query head then reads exactly
/// the reference pair `[V_kv(i) ; V_kv(i + H/2)]` of its pair index
/// `i = h mod (H / 2)`. Verified: natively, for an even key/value count, this
/// is a theorem about `kv(h) mod half`; for one shared head both reference
/// heads are head zero. After repetition (an odd count above one), query
/// head `h` reads repeated heads `h mod half` and `h mod half + half`, both
/// valid, with `half = H / 2`; repeated head `m` reads source head `m / g`
/// (the layout [`head_groups`] proves for `repeat_key_value`), so the source
/// pair is the reference pair by definition.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(pairing =>
    requires
        key_value_heads >= 1,
        query_heads >= 2,
        query_heads % 2 == 0,
        query_heads % key_value_heads == 0,
    ensures
        pairing == ValuePairing::Shared <==> key_value_heads == 1,
        match pairing {
            ValuePairing::Shared => forall|head: int| 0 <= head < query_heads ==> #[trigger]
                reference_pair(head, query_heads as int, (query_heads / 2) as int) == (0int, 0int),
            ValuePairing::Native { half } => {
                let groups = (query_heads / key_value_heads) as int;
                let pairs = (query_heads / 2) as int;
                &&& key_value_heads % 2 == 0
                &&& half * 2 == key_value_heads
                &&& forall|head: int| 0 <= head < query_heads ==> #[trigger] built_pair(
                    head,
                    groups,
                    half as int,
                ) == reference_pair(head, groups, pairs)
            },
            ValuePairing::Repeated { half } => {
                let groups = (query_heads / key_value_heads) as int;
                let pairs = (query_heads / 2) as int;
                &&& key_value_heads % 2 == 1 && key_value_heads > 1
                &&& half * 2 == query_heads
                // Repeated heads (one per query head) pair as `(h mod half, h mod half + half)`.
                &&& forall|head: int| 0 <= head < query_heads ==> {
                    let repeated = #[trigger] built_pair(head, 1, half as int);
                    &&& 0 <= repeated.0 < half
                    &&& repeated.1 == repeated.0 + half
                    &&& (kv_head_of(repeated.0, groups), kv_head_of(repeated.1, groups))
                        == reference_pair(head, groups, pairs)
                }
            },
        },
))]
pub fn value_pairing(query_heads: usize, key_value_heads: usize) -> ValuePairing {
    if key_value_heads == 1 {
        #[cfg(verus_keep_ghost)]
        proof! {
            assert forall|head: int| 0 <= head < query_heads implies #[trigger] reference_pair(
                head,
                query_heads as int,
                (query_heads / 2) as int,
            ) == (0int, 0int) by {
                vstd::arithmetic::div_mod::lemma_mod_pos_bound(head, (query_heads / 2) as int);
                vstd::arithmetic::div_mod::lemma_basic_div(
                    head % ((query_heads / 2) as int),
                    query_heads as int,
                );
                vstd::arithmetic::div_mod::lemma_basic_div(
                    head % ((query_heads / 2) as int) + (query_heads / 2) as int,
                    query_heads as int,
                );
            }
        }
        return ValuePairing::Shared;
    }
    let parity = key_value_heads % 2;
    if parity == 0 {
        #[cfg(verus_keep_ghost)]
        proof! {
            let groups = query_heads / key_value_heads;
            vstd::arithmetic::div_mod::lemma_fundamental_div_mod(
                query_heads as int,
                key_value_heads as int,
            );
            assert(groups >= 1) by (nonlinear_arith)
                requires
                    query_heads as int == key_value_heads as int * groups as int,
                    query_heads >= 1;
            assert forall|head: int| 0 <= head < query_heads implies #[trigger] built_pair(
                head,
                groups as int,
                (key_value_heads / 2) as int,
            ) == reference_pair(head, groups as int, (query_heads / 2) as int) by {
                lemma_native_pairing(
                    query_heads as int,
                    key_value_heads as int,
                    groups as int,
                    head,
                );
            }
        }
        return ValuePairing::Native {
            half: key_value_heads / 2,
        };
    }
    #[cfg(verus_keep_ghost)]
    proof! {
        let half = (query_heads / 2) as int;
        assert forall|head: int| 0 <= head < query_heads implies {
            let repeated = #[trigger] built_pair(head, 1, half);
            &&& 0 <= repeated.0 < half
            &&& repeated.1 == repeated.0 + half
        } by {
            vstd::arithmetic::div_mod::lemma_mod_pos_bound(head, half);
            assert(head / 1 == head);
        }
    }
    ValuePairing::Repeated {
        half: query_heads / 2,
    }
}

#[cfg(test)]
#[path = "grouping_tests.rs"]
mod tests;
