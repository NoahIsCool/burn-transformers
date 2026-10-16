//! Dependency-free integer index plans for multimodal RoPE and T5 relative bias.
//!
//! These helpers contain only machine-integer arithmetic so that Verus can
//! check them in place (`verus/position.rs`, see `verus/README.md`):
//!
//! * [`mrope_frequency_axis`] assigns each complex frequency of Qwen's
//!   multimodal RoPE to exactly one of the temporal, height, or width axes.
//! * [`relative_table_length`] and [`relative_table_index`] describe T5's
//!   displacement table. A bias for query `i` and key `j` depends only on the
//!   displacement `j - i`, so a `[query, key]` bias needs just
//!   `query + key - 1` distinct buckets. Entry `t` of that table holds the
//!   bucket for the anchor pair `(query_length - 1, t)`; the pair `(i, j)` reads
//!   entry `j + (query_length - 1 - i)`, which has the same displacement.
//!
//! * [`plan_relative_table`] validates a `[query, key]` bias geometry with a
//!   query offset and returns the table length and the anchor query position.
//! * [`relative_bias_shape_fault`] validates the learned `[buckets, heads]`
//!   table shape, and [`mrope_section_total`], [`rotary_operand_fits`], and
//!   [`split_half_width_fits`] validate rotary tensor geometry.
//!
//! The proofs establish that no arithmetic overflows, every index lies inside
//! its table, and each lookup preserves the key-minus-query displacement. For
//! an offset query block, the pair `(i, j)` reads the entry whose displacement
//! from the anchor is `j - (query_offset + i)`, the true key-minus-query
//! displacement, and every table entry is read by some pair. Under Qwen's
//! interleaving, the multimodal axes receive exactly
//! `min(height, (half + 1) / 3)` height and `min(width, half / 3)` width
//! frequencies, so sections that fit receive exactly their configured counts.
//! [`mrope_frequency_axis`] and the validators run as written at runtime.
//! [`relative_table_index`] is the scalar specification of the T5 index that
//! the bias evaluates for all pairs at once as a device tensor expression; the
//! proof applies to this function, and a unit test checks that every device
//! entry equals it.

#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

#[cfg(verus_keep_ghost)]
verus! {

/// The multimodal axis of frequency `index`: height, width, or temporal (see [`mrope_frequency_axis`]).
pub open spec fn spec_mrope_axis(index: int, height_sections: int, width_sections: int) -> int {
    if index % 3 == 1 && index / 3 < height_sections {
        1
    } else if index % 3 == 2 && index / 3 < width_sections {
        2
    } else {
        0
    }
}

/// Number of frequencies in `0..half` assigned to `axis`.
pub open spec fn mrope_axis_count(half: int, height_sections: int, width_sections: int, axis: int) -> int
    decreases half,
{
    if half <= 0 {
        0
    } else {
        mrope_axis_count(half - 1, height_sections, width_sections, axis) + if spec_mrope_axis(
            half - 1,
            height_sections,
            width_sections,
        ) == axis {
            1int
        } else {
            0int
        }
    }
}

/// The smaller of two integers.
pub open spec fn spec_min(left: int, right: int) -> int {
    if left <= right {
        left
    } else {
        right
    }
}

/// Qwen's interleaving gives height `min(h, (n + 1) / 3)` and width `min(w, n / 3)` of `n` frequencies.
///
/// Height owns indices `1, 4, 7, ...` (there are `(n + 1) / 3` below `n`) up
/// to group `h`, width owns `2, 5, 8, ...` (`n / 3` below `n`) up to group
/// `w`, and temporal owns the rest.
pub proof fn lemma_mrope_axis_counts(half: int, height_sections: int, width_sections: int)
    requires
        half >= 0,
        height_sections >= 0,
        width_sections >= 0,
    ensures
        mrope_axis_count(half, height_sections, width_sections, 1) == spec_min(
            height_sections,
            (half + 1) / 3,
        ),
        mrope_axis_count(half, height_sections, width_sections, 2) == spec_min(
            width_sections,
            half / 3,
        ),
        mrope_axis_count(half, height_sections, width_sections, 0) == half - spec_min(
            height_sections,
            (half + 1) / 3,
        ) - spec_min(width_sections, half / 3),
    decreases half,
{
    if half > 0 {
        let last = half - 1;
        lemma_mrope_axis_counts(last, height_sections, width_sections);
        // Write `last = 3 g + r`; the new index is height iff `r == 1`, width iff `r == 2`.
        let group = last / 3;
        let phase = last % 3;
        assert(last == 3 * group + phase && 0 <= phase < 3) by {
            vstd::arithmetic::div_mod::lemma_fundamental_div_mod(last, 3);
        }
        // `half / 3` and `(half + 1) / 3` advance past `g` exactly when the new
        // index completes a width or height slot.
        vstd::arithmetic::div_mod::lemma_fundamental_div_mod_converse(
            last + 1,
            3,
            if phase == 2 { group + 1 } else { group },
            if phase == 2 { 0 } else { phase + 1 },
        );
        vstd::arithmetic::div_mod::lemma_fundamental_div_mod_converse(
            last + 2,
            3,
            if phase == 0 { group } else { group + 1 },
            if phase == 0 { 2 } else { phase - 1 },
        );
    }
}

/// Every table entry `t < query_length + key_length - 1` is read by some in-range pair.
///
/// The anchor query `query_length - 1` reads entry `t = key` for every key,
/// covering `t < key_length`; the last key `key_length - 1` with query
/// `query_length + key_length - 2 - t` covers the remaining entries.
pub proof fn lemma_relative_table_entries_are_read(entry: int, query_length: int, key_length: int)
    requires
        query_length >= 1,
        key_length >= 1,
        0 <= entry < query_length + key_length - 1,
    ensures
        exists|query: int, key: int|
            0 <= query < query_length && 0 <= key < key_length && #[trigger] (key + (query_length
                - 1 - query)) == entry,
{
    if entry < key_length {
        let query = query_length - 1;
        assert(entry + (query_length - 1 - query) == entry);
    } else {
        let key = key_length - 1;
        let query = query_length + key_length - 2 - entry;
        assert(key + (query_length - 1 - query) == entry);
    }
}

} // verus!

/// Selects the position axis (0 = temporal, 1 = height, 2 = width) of one frequency.
///
/// Qwen interleaves axes with period three: frequency `i` belongs to height
/// when `i % 3 == 1` and its group `i / 3` is among the first
/// `height_sections` groups, to width when `i % 3 == 2` and `i / 3 <
/// width_sections`, and to temporal otherwise. Comparing groups instead of
/// `i < 3 * sections` avoids overflow for any section count.
///
/// Verified: the axis equals `spec_mrope_axis`; with `lemma_mrope_axis_counts`
/// this fixes how many of the first `half` frequencies each axis receives.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(axis =>
    ensures
        axis < 3,
        axis == spec_mrope_axis(index as int, height_sections as int, width_sections as int),
        (axis == 1) == (index % 3 == 1 && index / 3 < height_sections),
        (axis == 2) == (index % 3 == 2 && index / 3 < width_sections),
))]
pub fn mrope_frequency_axis(index: usize, height_sections: usize, width_sections: usize) -> usize {
    let group = index / 3;
    let phase = index % 3;
    if phase == 1 && group < height_sections {
        1
    } else if phase == 2 && group < width_sections {
        2
    } else {
        0
    }
}

/// Returns the number of distinct displacements `query_length + key_length - 1`.
///
/// `None` signals an empty axis or an unrepresentable length; callers report
/// it as a contextual shape error.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(length =>
    ensures
        length.is_some() == (query_length > 0 && key_length > 0
            && query_length + key_length - 1 <= usize::MAX),
        length.is_some() ==> length.unwrap() == query_length + key_length - 1,
))]
pub fn relative_table_length(query_length: usize, key_length: usize) -> Option<usize> {
    if query_length == 0 || key_length == 0 {
        return None;
    }
    // `key_length - 1` cannot underflow because `key_length > 0`.
    query_length.checked_add(key_length - 1)
}

/// Returns the displacement-table entry read by local query `query` and key `key`.
///
/// The entry is `key + (query_length - 1) - query`. It preserves the
/// key-minus-query displacement relative to the anchor query
/// `query_length - 1`, and it is at most `key + query_length - 1`, so every key
/// `key < key_length` reads inside the table of length
/// `query_length + key_length - 1`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(entry =>
    requires
        query < query_length,
        key + query_length - 1 <= usize::MAX,
    ensures
        entry + query == key + query_length - 1,
        entry + 1 <= key + query_length,
))]
pub fn relative_table_index(query: usize, key: usize, query_length: usize) -> usize {
    // `query < query_length` makes the subtraction non-negative, and the sum is
    // at most `key + query_length - 1`, which fits by precondition.
    key + (query_length - 1 - query)
}

/// Returns the total `temporal + height + width` of multimodal RoPE sections.
///
/// `None` signals that the sum overflows `usize`; callers report it as a
/// configuration error before comparing the total with `rotary_dimension / 2`.
///
/// Verified: the result is the exact sum when it is representable and `None`
/// exactly when it is not.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(total =>
    ensures
        total.is_some() == (sections[0] + sections[1] + sections[2] <= usize::MAX),
        total.is_some() ==> total.unwrap() == sections[0] + sections[1] + sections[2],
))]
pub(crate) fn mrope_section_total(sections: [usize; 3]) -> Option<usize> {
    let Some(partial) = sections[0].checked_add(sections[1]) else {
        return None;
    };
    partial.checked_add(sections[2])
}

/// Whether rotary tables fit one `[batch, heads, sequence, width]` operand.
///
/// The cosine table must be `[batch, 1, sequence, rotary]`, the sine table
/// must have the same shape, and `rotary` must be even and at most `width`.
///
/// Verified: the result is exactly that conjunction; when it holds, the
/// rotated prefix splits into `rotary / 2` complex pairs and the unrotated
/// suffix has width `width - rotary`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(fits =>
    ensures
        fits == (cosine[0] == operand[0] && cosine[1] == 1 && cosine[2] == operand[2]
            && cosine[3] == rotary_dimension && sine[0] == cosine[0] && sine[1] == cosine[1]
            && sine[2] == cosine[2] && sine[3] == cosine[3] && rotary_dimension <= operand[3]
            && rotary_dimension % 2 == 0),
        fits ==> 2 * (rotary_dimension / 2) == rotary_dimension
            && (rotary_dimension / 2) + (rotary_dimension / 2) + (operand[3] - rotary_dimension)
                == operand[3],
))]
pub(crate) fn rotary_operand_fits(
    operand: [usize; 4],
    cosine: [usize; 4],
    sine: [usize; 4],
    rotary_dimension: usize,
) -> bool {
    cosine[0] == operand[0]
        && cosine[1] == 1
        && cosine[2] == operand[2]
        && cosine[3] == rotary_dimension
        && sine[0] == cosine[0]
        && sine[1] == cosine[1]
        && sine[2] == cosine[2]
        && sine[3] == cosine[3]
        && rotary_dimension <= operand[3]
        && rotary_dimension.is_multiple_of(2)
}

/// Whether a channel count splits into complex pairs: non-zero and even.
///
/// Verified: the result is exactly `width > 0 && width % 2 == 0`, and then the
/// `[width / 2, 2]` and `[2, width / 2]` pair views both hold `width` channels.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(fits =>
    ensures
        fits == (width > 0 && width % 2 == 0),
        fits ==> (width / 2) * 2 == width && width / 2 >= 1,
))]
pub(crate) fn split_half_width_fits(width: usize) -> bool {
    width != 0 && width.is_multiple_of(2)
}

/// Validated geometry of a T5 displacement table for one offset query block.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RelativeTablePlan {
    /// Distinct displacements `query_length + key_length - 1`.
    pub length: usize,
    /// Absolute position `query_offset + query_length - 1` of the last (anchor) query.
    pub anchor: usize,
}

/// Why a `[query, key]` relative-bias geometry was rejected, in check order.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RelativeTableFault {
    /// A query or key axis is empty.
    EmptyAxis,
    /// `query_length * key_length` bias entries per head overflow `usize`.
    OutputOverflow,
    /// `query_offset + query_length` overflows `usize`.
    OffsetOverflow,
    /// The table length exceeds the `i64` device index range.
    IndexRange,
}

/// Plans the host displacement table of a `[query, key]` bias whose first query sits at `query_offset`.
///
/// Entry `t` holds the bucket of key position `t` relative to the anchor
/// query `query_offset + query_length - 1`. The table length
/// `query_length + key_length - 1` cannot overflow once `query_length *
/// key_length` fits, because `q + k - 1 <= q * k` for positive lengths, so no
/// separate length check is needed.
///
/// Verified: the result is `Ok` exactly when every check passes and the
/// faults follow the listed order. A plan's anchor and length are exact, and
/// for every query `i < query_length` and key `j < key_length` the entry
/// [`relative_table_index`]`(i, j, query_length)` lies inside the table and
/// its displacement from the anchor is `j - (query_offset + i)`, the pair's
/// true key-minus-query displacement.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(result =>
    ensures
        match result {
            Ok(plan) => {
                &&& query_length >= 1 && key_length >= 1
                &&& query_length * key_length <= usize::MAX
                &&& query_offset + query_length <= usize::MAX
                &&& plan.length == query_length + key_length - 1
                &&& plan.length <= i64::MAX
                &&& plan.anchor == query_offset + query_length - 1
                &&& forall|query: int, key: int|
                    0 <= query < query_length && 0 <= key < key_length ==> {
                        &&& #[trigger] (key + (query_length - 1 - query)) < plan.length
                        &&& key + (query_length - 1 - query) - plan.anchor == key - (query_offset + query)
                    }
            },
            Err(RelativeTableFault::EmptyAxis) => query_length == 0 || key_length == 0,
            Err(RelativeTableFault::OutputOverflow) => {
                &&& query_length >= 1 && key_length >= 1
                &&& query_length * key_length > usize::MAX
            },
            Err(RelativeTableFault::OffsetOverflow) => {
                &&& query_length >= 1 && key_length >= 1
                &&& query_length * key_length <= usize::MAX
                &&& query_offset + query_length > usize::MAX
            },
            Err(RelativeTableFault::IndexRange) => {
                &&& query_length >= 1 && key_length >= 1
                &&& query_length * key_length <= usize::MAX
                &&& query_offset + query_length <= usize::MAX
                &&& query_length + key_length - 1 > i64::MAX
            },
        }
))]
pub(crate) fn plan_relative_table(
    query_length: usize,
    key_length: usize,
    query_offset: usize,
) -> Result<RelativeTablePlan, RelativeTableFault> {
    if query_length == 0 || key_length == 0 {
        return Err(RelativeTableFault::EmptyAxis);
    }
    if query_length.checked_mul(key_length).is_none() {
        return Err(RelativeTableFault::OutputOverflow);
    }
    #[cfg(verus_keep_ghost)]
    proof! {
        // `q + k - 1 <= q * k` because `(q - 1) * (k - 1) >= 0`.
        assert(query_length + key_length - 1 <= query_length * key_length) by (nonlinear_arith)
            requires
                query_length >= 1,
                key_length >= 1,
        ;
    }
    let length = query_length + (key_length - 1);
    let Some(end) = query_offset.checked_add(query_length) else {
        return Err(RelativeTableFault::OffsetOverflow);
    };
    // Compare in `u64`, which holds every `usize` and `i64::MAX` exactly, as
    // `i64::try_from` does on every target width.
    if length as u64 > i64::MAX as u64 {
        return Err(RelativeTableFault::IndexRange);
    }
    Ok(RelativeTablePlan {
        length,
        anchor: end - 1,
    })
}

/// Why a learned `[buckets, heads]` relative-bias table shape was rejected, in check order.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RelativeBiasShapeFault {
    /// The table has no heads.
    ZeroHeads,
    /// Bucket IDs would not fit the signed `i64` device index type.
    BucketIndexRange,
    /// `num_buckets * num_heads` parameters overflow `usize`.
    TableOverflow,
}

/// Checks the learned relative-bias table shape before bucket geometry.
///
/// Verified: `None` exactly when `num_heads > 0`, `num_buckets <= i64::MAX`,
/// and `num_buckets * num_heads` fits in `usize`; otherwise the first failed
/// check in that order.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(fault =>
    ensures
        match fault {
            None => num_heads > 0 && num_buckets <= i64::MAX
                && num_buckets * num_heads <= usize::MAX,
            Some(RelativeBiasShapeFault::ZeroHeads) => num_heads == 0,
            Some(RelativeBiasShapeFault::BucketIndexRange) => num_heads > 0
                && num_buckets > i64::MAX,
            Some(RelativeBiasShapeFault::TableOverflow) => num_heads > 0
                && num_buckets <= i64::MAX && num_buckets * num_heads > usize::MAX,
        }
))]
pub(crate) fn relative_bias_shape_fault(
    num_heads: usize,
    num_buckets: usize,
) -> Option<RelativeBiasShapeFault> {
    if num_heads == 0 {
        return Some(RelativeBiasShapeFault::ZeroHeads);
    }
    // Compared in `u64`, exact for every `usize` width (see `plan_relative_table`).
    if num_buckets as u64 > i64::MAX as u64 {
        return Some(RelativeBiasShapeFault::BucketIndexRange);
    }
    if num_buckets.checked_mul(num_heads).is_none() {
        return Some(RelativeBiasShapeFault::TableOverflow);
    }
    None
}

#[cfg(test)]
#[path = "index_plan_tests.rs"]
mod tests;
