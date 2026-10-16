//! Pure width validation and key-prefix planning for multi-head latent attention.
//!
//! # Projection widths
//!
//! With `H` heads, content width `d_n`, rotary width `d_r`, value width `d_v`,
//! KV latent rank `r`, optional query rank `r_q`, and hidden width `D`, MLA
//! projects queries to `H * (d_n + d_r)` channels (directly, or through
//! `D -> r_q -> H * (d_n + d_r)`), the joint KV-A projection to `r + d_r`, the
//! KV-B expansion to `H * (d_n + d_v)`, and the output from `H * d_v` back to
//! `D`. Burn allocates parameters lazily, so every width and every matrix
//! element count is checked here before allocation, and allocation uses the
//! widths validation returns.
//!
//! # Key prefixes
//!
//! Positional visibility keeps host bounds `limits[i]`: query rows `0..=i`
//! see only keys `0..limits[i]`. Valid bounds are non-empty, at most `K`, and
//! nondecreasing, so a chunk of rows `[s, e)` needs only the key prefix
//! `0..limits[e - 1]`. For `Q` queries bottom-right aligned to `K` unpadded
//! keys, row `i` sits at position `K - Q + i` and sees keys `j <= K - Q + i`,
//! so its bound is exactly `K - Q + i + 1`.
//!
//! # Verified properties
//!
//! Verus checks every function here through `verus/mla.rs`: widths are exact
//! and every matrix element count fits `usize` exactly when validation
//! succeeds, faults are reported in the documented order, suffix bounds are
//! exactly the causal prefixes and satisfy the prefix contract, the
//! validator accepts exactly contract-satisfying bounds, and a chunk's key
//! prefix covers every row of the chunk.

#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

#[cfg(verus_keep_ghost)]
verus! {

/// Bottom-right causality: query row `row` of `Q` sits at key position
/// `K - Q + row` and sees `key`.
pub open spec fn causally_visible(row: int, key: int, queries: int, keys: int) -> bool {
    key <= keys - queries + row
}

/// Bounds never decrease: an earlier row's bound is at most a later row's.
pub open spec fn nondecreasing(limits: Seq<usize>) -> bool {
    forall|first: int, second: int|
        0 <= first <= second < limits.len()
            ==> #[trigger] limits[first] <= #[trigger] limits[second]
}

/// Key-prefix bounds for `queries` rows over `keys` keys: one per row, in `1..=K`, nondecreasing.
pub open spec fn key_limits_contract(limits: Seq<usize>, queries: int, keys: int) -> bool {
    &&& limits.len() == queries
    &&& forall|row: int| 0 <= row < limits.len() ==> 1 <= #[trigger] limits[row] <= keys
    &&& nondecreasing(limits)
}

/// A row-major element count `input * output` fits `usize`.
pub open spec fn fits(input: int, output: int) -> bool {
    input * output <= usize::MAX
}

} // verus!

/// A required MLA extent, named after its configuration field.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MlaAxis {
    /// `hidden_size`.
    HiddenSize,
    /// `num_heads`.
    NumHeads,
    /// `qk_nope_head_dim`.
    QkNopeHeadDim,
    /// `qk_rope_head_dim`.
    QkRopeHeadDim,
    /// `value_head_dim`.
    ValueHeadDim,
    /// `kv_lora_rank`.
    KvLoraRank,
}

/// Why MLA extents are unusable before any width arithmetic.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MlaAxisFault {
    /// A required extent is zero.
    Zero(
        /// The first zero extent in field order.
        MlaAxis,
    ),
    /// The rotary width is odd, but RoPE rotates channel pairs.
    OddRotary,
    /// A present query rank is zero.
    ZeroQueryRank,
}

/// Checks MLA extents in field order, then rotary parity, then the optional query rank.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(fault =>
    ensures
        ({
            let nonzero = hidden_size >= 1 && num_heads >= 1 && nope >= 1 && rope >= 1
                && value >= 1 && kv_rank >= 1;
            match fault {
                None => nonzero && rope % 2 == 0 && query_rank != Some(0usize),
                Some(MlaAxisFault::Zero(MlaAxis::HiddenSize)) => hidden_size == 0,
                Some(MlaAxisFault::Zero(MlaAxis::NumHeads)) => hidden_size >= 1 && num_heads == 0,
                Some(MlaAxisFault::Zero(MlaAxis::QkNopeHeadDim)) =>
                    hidden_size >= 1 && num_heads >= 1 && nope == 0,
                Some(MlaAxisFault::Zero(MlaAxis::QkRopeHeadDim)) =>
                    hidden_size >= 1 && num_heads >= 1 && nope >= 1 && rope == 0,
                Some(MlaAxisFault::Zero(MlaAxis::ValueHeadDim)) =>
                    hidden_size >= 1 && num_heads >= 1 && nope >= 1 && rope >= 1 && value == 0,
                Some(MlaAxisFault::Zero(MlaAxis::KvLoraRank)) => {
                    &&& hidden_size >= 1 && num_heads >= 1 && nope >= 1
                    &&& rope >= 1 && value >= 1 && kv_rank == 0
                },
                Some(MlaAxisFault::OddRotary) => nonzero && rope % 2 == 1,
                Some(MlaAxisFault::ZeroQueryRank) =>
                    nonzero && rope % 2 == 0 && query_rank == Some(0usize),
            }
        }),
))]
pub(crate) fn mla_axis_fault(
    hidden_size: usize,
    num_heads: usize,
    nope: usize,
    rope: usize,
    value: usize,
    kv_rank: usize,
    query_rank: Option<usize>,
) -> Option<MlaAxisFault> {
    if hidden_size == 0 {
        return Some(MlaAxisFault::Zero(MlaAxis::HiddenSize));
    }
    if num_heads == 0 {
        return Some(MlaAxisFault::Zero(MlaAxis::NumHeads));
    }
    if nope == 0 {
        return Some(MlaAxisFault::Zero(MlaAxis::QkNopeHeadDim));
    }
    if rope == 0 {
        return Some(MlaAxisFault::Zero(MlaAxis::QkRopeHeadDim));
    }
    if value == 0 {
        return Some(MlaAxisFault::Zero(MlaAxis::ValueHeadDim));
    }
    if kv_rank == 0 {
        return Some(MlaAxisFault::Zero(MlaAxis::KvLoraRank));
    }
    // A bound remainder (not `is_multiple_of`, which Verus does not specify).
    let parity = rope % 2;
    if parity != 0 {
        return Some(MlaAxisFault::OddRotary);
    }
    let zero_rank = match query_rank {
        Some(rank) => rank == 0,
        None => false,
    };
    if zero_rank {
        return Some(MlaAxisFault::ZeroQueryRank);
    }
    None
}

/// One MLA projection matrix, in validation order.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MlaMatrix {
    /// Direct query projection `q_proj`, `D -> H * (d_n + d_r)`.
    Query,
    /// Query down-projection `q_a_proj`, `D -> r_q`.
    QueryDown,
    /// Query up-projection `q_b_proj`, `r_q -> H * (d_n + d_r)`.
    QueryUp,
    /// Joint KV-A projection `kv_a_proj_with_mqa`, `D -> r + d_r`.
    JointKeyValue,
    /// KV-B expansion `kv_b_proj`, `r -> H * (d_n + d_v)`.
    KeyValueExpansion,
    /// Output projection `o_proj`, `H * d_v -> D`.
    Output,
}

/// Validated MLA projection widths.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct MlaWidths {
    /// Per-head query/key width `d_n + d_r`.
    pub(crate) qk_head_dim: usize,
    /// Joint KV-A output `r + d_r`.
    pub(crate) compressed_width: usize,
    /// KV-B output `H * (d_n + d_v)`.
    pub(crate) expanded_width: usize,
    /// Query projection output `H * (d_n + d_r)`.
    pub(crate) query_width: usize,
    /// Output projection input `H * d_v`.
    pub(crate) value_width: usize,
}

/// Why MLA widths are not representable.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MlaWidthFault {
    /// `d_n + d_r` overflows.
    QkHeadDim,
    /// `r + d_r` overflows.
    CompressedWidth,
    /// `d_n + d_v` overflows.
    ExpandedPerHead,
    /// `H * (d_n + d_v)` overflows.
    ExpandedHeads,
    /// `H * (d_n + d_r)` overflows.
    QueryWidth,
    /// A projection matrix's element count `input * output` overflows.
    Matrix {
        /// The first overflowing matrix in validation order.
        matrix: MlaMatrix,
        /// Its input width.
        input: usize,
        /// Its output width.
        output: usize,
    },
}

/// Per-head query/key width `d_n + d_r`, or [`MlaWidthFault::QkHeadDim`] when it overflows.
///
/// Verified exact: the shared derivation behind
/// `MultiLatentAttentionConfig::qk_head_dim` and [`mla_widths`].
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(width =>
    ensures
        match width {
            Ok(width) => width == nope + rope,
            Err(fault) => fault == MlaWidthFault::QkHeadDim && nope + rope > usize::MAX,
        },
))]
pub(crate) fn qk_head_width(nope: usize, rope: usize) -> Result<usize, MlaWidthFault> {
    match nope.checked_add(rope) {
        Some(width) => Ok(width),
        None => Err(MlaWidthFault::QkHeadDim),
    }
}

/// Joint KV-A output width `r + d_r`, or [`MlaWidthFault::CompressedWidth`] when it overflows.
///
/// Verified exact: the shared derivation behind
/// `MultiLatentAttentionConfig::compressed_projection_width` and [`mla_widths`].
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(width =>
    ensures
        match width {
            Ok(width) => width == kv_rank + rope,
            Err(fault) => fault == MlaWidthFault::CompressedWidth && kv_rank + rope > usize::MAX,
        },
))]
pub(crate) fn compressed_width(kv_rank: usize, rope: usize) -> Result<usize, MlaWidthFault> {
    match kv_rank.checked_add(rope) {
        Some(width) => Ok(width),
        None => Err(MlaWidthFault::CompressedWidth),
    }
}

/// KV-B expansion width `H * (d_n + d_v)`, naming the per-head sum or the head product on overflow.
///
/// Verified exact, with the per-head sum checked first: the shared derivation
/// behind `MultiLatentAttentionConfig::expanded_kv_width` and [`mla_widths`].
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(width =>
    ensures
        match width {
            Ok(width) => nope + value <= usize::MAX && width == num_heads * (nope + value),
            Err(MlaWidthFault::ExpandedPerHead) => nope + value > usize::MAX,
            Err(MlaWidthFault::ExpandedHeads) => {
                &&& nope + value <= usize::MAX
                &&& num_heads * (nope + value) > usize::MAX
            },
            Err(_) => false,
        },
))]
pub(crate) fn expanded_width(
    num_heads: usize,
    nope: usize,
    value: usize,
) -> Result<usize, MlaWidthFault> {
    let Some(per_head) = nope.checked_add(value) else {
        return Err(MlaWidthFault::ExpandedPerHead);
    };
    match num_heads.checked_mul(per_head) {
        Some(width) => Ok(width),
        None => Err(MlaWidthFault::ExpandedHeads),
    }
}

/// Computes MLA widths and checks every projection matrix's element count.
///
/// Widths are checked in the order `d_n + d_r`, `r + d_r`, `d_n + d_v`,
/// `H * (d_n + d_v)`, `H * (d_n + d_r)`; matrices in [`MlaMatrix`] order,
/// skipping the query projections the optional rank does not use. The output
/// input width `H * d_v` never overflows once the expansion fits.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(result =>
    ensures
        ({
            let qk = nope + rope;
            let compressed = kv_rank + rope;
            let per_head = nope + value;
            let expanded = num_heads * per_head;
            let query = num_heads * qk;
            let value_width = num_heads * value;
            let widths_fit = qk <= usize::MAX && compressed <= usize::MAX
                && per_head <= usize::MAX && expanded <= usize::MAX && query <= usize::MAX;
            let query_fits = match query_rank {
                None => fits(hidden_size as int, query),
                Some(rank) => fits(hidden_size as int, rank as int) && fits(rank as int, query),
            };
            let kv_a_fits = fits(hidden_size as int, compressed);
            let kv_b_fits = fits(kv_rank as int, expanded);
            let output_fits = fits(value_width, hidden_size as int);
            match result {
                Ok(widths) => {
                    &&& widths_fit && query_fits && kv_a_fits && kv_b_fits && output_fits
                    &&& widths.qk_head_dim == qk
                    &&& widths.compressed_width == compressed
                    &&& widths.expanded_width == expanded
                    &&& widths.query_width == query
                    &&& widths.value_width == value_width
                },
                Err(MlaWidthFault::QkHeadDim) => qk > usize::MAX,
                Err(MlaWidthFault::CompressedWidth) => qk <= usize::MAX && compressed > usize::MAX,
                Err(MlaWidthFault::ExpandedPerHead) => {
                    &&& qk <= usize::MAX && compressed <= usize::MAX
                    &&& per_head > usize::MAX
                },
                Err(MlaWidthFault::ExpandedHeads) => {
                    &&& qk <= usize::MAX && compressed <= usize::MAX
                    &&& per_head <= usize::MAX && expanded > usize::MAX
                },
                Err(MlaWidthFault::QueryWidth) => {
                    &&& qk <= usize::MAX && compressed <= usize::MAX
                    &&& per_head <= usize::MAX && expanded <= usize::MAX
                    &&& query > usize::MAX
                },
                Err(MlaWidthFault::Matrix { matrix, input, output }) => {
                    &&& widths_fit
                    &&& !fits(input as int, output as int)
                    &&& match matrix {
                        MlaMatrix::Query =>
                            query_rank is None && input == hidden_size && output == query,
                        MlaMatrix::QueryDown => query_rank == Some(output) && input == hidden_size,
                        MlaMatrix::QueryUp => {
                            &&& query_rank == Some(input) && output == query
                            &&& fits(hidden_size as int, input as int)
                        },
                        MlaMatrix::JointKeyValue =>
                            query_fits && input == hidden_size && output == compressed,
                        MlaMatrix::KeyValueExpansion => {
                            &&& query_fits && kv_a_fits
                            &&& input == kv_rank && output == expanded
                        },
                        MlaMatrix::Output => {
                            &&& query_fits && kv_a_fits && kv_b_fits
                            &&& input == value_width && output == hidden_size
                        },
                    }
                },
            }
        }),
))]
pub(crate) fn mla_widths(
    hidden_size: usize,
    num_heads: usize,
    nope: usize,
    rope: usize,
    value: usize,
    kv_rank: usize,
    query_rank: Option<usize>,
) -> Result<MlaWidths, MlaWidthFault> {
    let qk_head_dim = qk_head_width(nope, rope)?;
    let compressed_width = compressed_width(kv_rank, rope)?;
    let expanded_width = expanded_width(num_heads, nope, value)?;
    let Some(query_width) = num_heads.checked_mul(qk_head_dim) else {
        return Err(MlaWidthFault::QueryWidth);
    };
    #[cfg(verus_keep_ghost)]
    proof! {
        assert(num_heads * value <= num_heads * (nope + value)) by (nonlinear_arith);
    }
    // The checked expansion bounds `H * d_v <= H * (d_n + d_v)`.
    let value_width = num_heads * value;
    match query_rank {
        None => {
            if hidden_size.checked_mul(query_width).is_none() {
                return Err(MlaWidthFault::Matrix {
                    matrix: MlaMatrix::Query,
                    input: hidden_size,
                    output: query_width,
                });
            }
        }
        Some(rank) => {
            if hidden_size.checked_mul(rank).is_none() {
                return Err(MlaWidthFault::Matrix {
                    matrix: MlaMatrix::QueryDown,
                    input: hidden_size,
                    output: rank,
                });
            }
            if rank.checked_mul(query_width).is_none() {
                return Err(MlaWidthFault::Matrix {
                    matrix: MlaMatrix::QueryUp,
                    input: rank,
                    output: query_width,
                });
            }
        }
    }
    if hidden_size.checked_mul(compressed_width).is_none() {
        return Err(MlaWidthFault::Matrix {
            matrix: MlaMatrix::JointKeyValue,
            input: hidden_size,
            output: compressed_width,
        });
    }
    if kv_rank.checked_mul(expanded_width).is_none() {
        return Err(MlaWidthFault::Matrix {
            matrix: MlaMatrix::KeyValueExpansion,
            input: kv_rank,
            output: expanded_width,
        });
    }
    if value_width.checked_mul(hidden_size).is_none() {
        return Err(MlaWidthFault::Matrix {
            matrix: MlaMatrix::Output,
            input: value_width,
            output: hidden_size,
        });
    }
    Ok(MlaWidths {
        qk_head_dim,
        compressed_width,
        expanded_width,
        query_width,
        value_width,
    })
}

/// Key-prefix bounds of `queries` rows bottom-right aligned to `keys` unpadded keys.
///
/// Row `i` sits at key position `K - Q + i`, so `limits[i] = K - Q + i + 1`:
/// it sees exactly the keys below its bound. The bounds satisfy the prefix
/// contract (one per row, in `1..=K`, nondecreasing).
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(limits =>
    requires
        queries <= keys,
    ensures
        limits.len() == queries,
        forall|row: int|
            0 <= row < queries ==> #[trigger] limits[row] == keys - queries + row + 1,
        forall|row: int, key: int|
            0 <= row < queries && 0 <= key ==> (#[trigger] causally_visible(
                row,
                key,
                queries as int,
                keys as int,
            ) <==> key < limits[row]),
        key_limits_contract(limits@, queries as int, keys as int),
))]
pub(crate) fn suffix_key_limits(queries: usize, keys: usize) -> Vec<usize> {
    let offset = keys - queries;
    let mut limits = Vec::with_capacity(queries);
    let mut row = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            row <= queries,
            queries <= keys,
            offset == keys - queries,
            limits.len() == row,
            forall|done: int| 0 <= done < row ==> #[trigger] limits[done] == offset + done + 1,
        decreases queries - row,
    ))]
    while row < queries {
        limits.push(offset + row + 1);
        row += 1;
    }
    limits
}

/// Whether `limits` satisfy the prefix contract for `queries` rows over `keys` keys.
///
/// Accepts exactly one bound per row, each in `1..=keys`, in nondecreasing order.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(valid =>
    ensures
        valid == key_limits_contract(limits@, queries as int, keys as int),
))]
pub(crate) fn key_limits_valid(limits: &[usize], queries: usize, keys: usize) -> bool {
    if limits.len() != queries {
        return false;
    }
    let mut row = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            row <= limits.len(),
            limits.len() == queries,
            forall|done: int| 0 <= done < row ==> 1 <= #[trigger] limits[done] <= keys,
            forall|first: int, second: int|
                0 <= first <= second <= row && second < limits.len() ==> #[trigger] limits[first]
                    <= #[trigger] limits[second],
        decreases limits.len() - row,
    ))]
    while row < limits.len() {
        let limit = limits[row];
        if limit == 0 || limit > keys {
            return false;
        }
        if row + 1 < limits.len() && limit > limits[row + 1] {
            return false;
        }
        row += 1;
    }
    true
}

/// Key prefix a chunk of query rows ending before `end` may need: `limits[end - 1]`.
///
/// For nondecreasing bounds it covers every row `0..end` (so every row of the
/// chunk); under the prefix contract it also lies in `1..=K`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(key_end =>
    requires
        nondecreasing(limits@),
        1 <= end <= limits.len(),
    ensures
        key_end == limits[end - 1],
        forall|row: int| 0 <= row < end ==> #[trigger] limits[row] <= key_end,
))]
pub(crate) fn chunk_key_end(limits: &[usize], end: usize) -> usize {
    limits[end - 1]
}

/// A window width as an `i64` position offset, saturating at `i64::MAX`.
///
/// Positions never reach `i64::MAX`, so a saturated window still keeps every key.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(width =>
    ensures
        width as int == if window as int <= i64::MAX as int {
            window as int
        } else {
            i64::MAX as int
        },
))]
pub(crate) fn saturating_window(window: usize) -> i64 {
    // Widening to `u64` is lossless for every `usize`.
    if window as u64 > i64::MAX as u64 {
        i64::MAX
    } else {
        window as i64
    }
}

/// Why positional visibility metadata disagrees with one `[batch, query, key]` geometry.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum VisibilityFault {
    /// A sliding window of zero positions would exclude every key.
    ZeroWindow,
    /// Positions, validity, or key bounds have the wrong extents or values.
    Shape,
}

/// Validates visibility metadata: a non-zero window, then every extent and the key bounds.
///
/// Position tensors are `[batch or 1, tokens]`, validity masks `[batch,
/// tokens]`, and the key bounds must satisfy the prefix contract for
/// `queries` rows over `keys` keys.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(fault =>
    ensures
        ({
            let rows_ok = (query_positions[0] == batch || query_positions[0] == 1)
                && (key_positions[0] == batch || key_positions[0] == 1);
            let columns_ok = query_positions[1] == queries && key_positions[1] == keys;
            let query_valid_ok = query_valid is None
                || (query_valid->Some_0[0] == batch && query_valid->Some_0[1] == queries);
            let key_valid_ok = key_valid is None
                || (key_valid->Some_0[0] == batch && key_valid->Some_0[1] == keys);
            let valid_ok = query_valid_ok && key_valid_ok;
            let shape_ok = rows_ok && columns_ok && valid_ok && key_limits_contract(
                limits@,
                queries as int,
                keys as int,
            );
            match fault {
                None => window != Some(0usize) && shape_ok,
                Some(VisibilityFault::ZeroWindow) => window == Some(0usize),
                Some(VisibilityFault::Shape) => window != Some(0usize) && !shape_ok,
            }
        }),
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn visibility_fault(
    window: Option<usize>,
    query_positions: [usize; 2],
    key_positions: [usize; 2],
    query_valid: Option<[usize; 2]>,
    key_valid: Option<[usize; 2]>,
    limits: &[usize],
    batch: usize,
    queries: usize,
    keys: usize,
) -> Option<VisibilityFault> {
    let limits_ok = key_limits_valid(limits, queries, keys);
    let zero_window = match window {
        Some(window) => window == 0,
        None => false,
    };
    if zero_window {
        return Some(VisibilityFault::ZeroWindow);
    }
    let query_valid_ok = match query_valid {
        Some(valid) => valid[0] == batch && valid[1] == queries,
        None => true,
    };
    let key_valid_ok = match key_valid {
        Some(valid) => valid[0] == batch && valid[1] == keys,
        None => true,
    };
    if !(query_positions[0] == batch || query_positions[0] == 1)
        || !(key_positions[0] == batch || key_positions[0] == 1)
        || query_positions[1] != queries
        || key_positions[1] != keys
        || !query_valid_ok
        || !key_valid_ok
        || !limits_ok
    {
        return Some(VisibilityFault::Shape);
    }
    None
}

/// Configured head layout of one MLA layer.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct MlaHeads {
    /// Attention heads `H`.
    pub(crate) heads: usize,
    /// Content query/key width `d_n`.
    pub(crate) content: usize,
    /// Rotary query/key width `d_r`.
    pub(crate) rotary: usize,
    /// KV latent rank `r`.
    pub(crate) rank: usize,
}

/// Whether projected MLA components match the configured layout.
///
/// Content queries are `[B, H, Q, d_n]`, rotary queries `[B, H, Q, d_r]`, the
/// latent `[B, K, r]`, and the shared rotary key `[B, 1, K, d_r]`, with `B`,
/// `Q`, and `K` non-zero and read from the content queries and the latent.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(matches =>
    ensures
        matches == ({
            let batch = query_content[0];
            let queries = query_content[2];
            let keys = latent[1];
            &&& batch >= 1 && queries >= 1 && keys >= 1
            &&& query_content[1] == layout.heads && query_content[3] == layout.content
            &&& latent[0] == batch && latent[2] == layout.rank
            &&& query_rotary[0] == batch && query_rotary[1] == layout.heads
            &&& query_rotary[2] == queries && query_rotary[3] == layout.rotary
            &&& rotated_key[0] == batch && rotated_key[1] == 1
            &&& rotated_key[2] == keys && rotated_key[3] == layout.rotary
        }),
))]
pub(crate) fn projection_matches(
    layout: MlaHeads,
    query_content: [usize; 4],
    query_rotary: [usize; 4],
    latent: [usize; 3],
    rotated_key: [usize; 4],
) -> bool {
    let batch = query_content[0];
    let queries = query_content[2];
    let keys = latent[1];
    batch != 0
        && queries != 0
        && keys != 0
        && query_content[1] == layout.heads
        && query_content[3] == layout.content
        && latent[0] == batch
        && latent[2] == layout.rank
        && query_rotary[0] == batch
        && query_rotary[1] == layout.heads
        && query_rotary[2] == queries
        && query_rotary[3] == layout.rotary
        && rotated_key[0] == batch
        && rotated_key[1] == 1
        && rotated_key[2] == keys
        && rotated_key[3] == layout.rotary
}

/// Whether `extent` broadcasts over `shape`: every axis is one or equal.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(broadcasts =>
    ensures
        broadcasts == forall|axis: int| 0 <= axis < 4 ==> (#[trigger] extent[axis] == 1
            || extent[axis] == shape[axis]),
))]
pub(crate) fn broadcasts_over(extent: [usize; 4], shape: [usize; 4]) -> bool {
    let mut axis = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            axis <= 4,
            forall|done: int|
                0 <= done < axis ==> (#[trigger] extent[done] == 1 || extent[done] == shape[done]),
        decreases 4 - axis,
    ))]
    while axis < 4 {
        if extent[axis] != 1 && extent[axis] != shape[axis] {
            return false;
        }
        axis += 1;
    }
    true
}

#[cfg(test)]
#[path = "geometry_tests.rs"]
mod tests;
