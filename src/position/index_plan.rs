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
//! The proofs establish that no arithmetic overflows, every index lies inside
//! its table, and each lookup preserves the key-minus-query displacement.
//! [`mrope_frequency_axis`] and [`relative_table_length`] run as written at
//! runtime. [`relative_table_index`] is the scalar specification of the T5
//! index that the bias evaluates for all pairs at once as a device tensor
//! expression; the proof applies to this function, and a unit test checks
//! that every device entry equals it.

#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

/// Selects the position axis (0 = temporal, 1 = height, 2 = width) of one frequency.
///
/// Qwen interleaves axes with period three: frequency `i` belongs to height
/// when `i % 3 == 1` and its group `i / 3` is among the first
/// `height_sections` groups, to width when `i % 3 == 2` and `i / 3 <
/// width_sections`, and to temporal otherwise. Comparing groups instead of
/// `i < 3 * sections` avoids overflow for any section count.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(axis =>
    ensures
        axis < 3,
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

#[cfg(test)]
#[path = "index_plan_tests.rs"]
mod tests;
