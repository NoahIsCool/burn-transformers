//! Pure index planning for modular models: clamped relative-position tables.
//!
//! Host-side integer logic that decides which device indices the modular
//! graphs read is dependency-free and checked in place by Verus through
//! `verus/custom.rs` (see `verus/README.md`).
//!
//! # Clamped displacement tables
//!
//! A T5 relative-position bucket depends only on the signed displacement
//! `d = p_key - p_query`. Every bucket with `|d| >= max_distance` is the
//! saturated last bucket of its direction, and positions validated against an
//! evaluated context `C` satisfy `|d| <= C - 1`. Clamping `d` to
//! `[-R, R]` with `R = min(max_distance, C - 1)` therefore changes no bucket,
//! so a lookup table of `2R + 1` host-evaluated buckets indexed by
//! `clamp(d, -R, R) + R` reproduces the pairwise reference exactly for any
//! (gapped, offset, or per-row) positions. [`displacement_table`] sizes that
//! table, `displacement_slot` (compiled for tests and proofs) is the scalar
//! specification of the device index, and [`slot_displacement`] is its inverse
//! used to fill the table.

#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

// ---------------------------------------------------------------------------
// Clamped displacement tables
// ---------------------------------------------------------------------------

/// Extent of a clamped relative-displacement lookup table.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DisplacementTable {
    /// Largest represented displacement magnitude `R`.
    pub(crate) radius: usize,
    /// Number of table entries, `2R + 1`.
    pub(crate) length: usize,
}

/// Sizes the clamped table for buckets saturating at `max_distance` over a context `C`.
///
/// Returns `R = min(max_distance, C - 1)` and `2R + 1`, or `None` for an empty
/// context or a length that does not fit `usize`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(table =>
    ensures
        evaluated_context == 0 ==> table.is_none(),
        table matches Some(t) ==> {
            &&& t.radius <= max_distance
            &&& t.radius < evaluated_context
            &&& (t.radius == max_distance || t.radius + 1 == evaluated_context)
            &&& t.length == 2 * t.radius + 1
        },
))]
pub(crate) fn displacement_table(
    max_distance: usize,
    evaluated_context: usize,
) -> Option<DisplacementTable> {
    if evaluated_context == 0 {
        return None;
    }
    let span = evaluated_context - 1;
    let radius = if max_distance < span {
        max_distance
    } else {
        span
    };
    let doubled = radius.checked_mul(2)?;
    let length = doubled.checked_add(1)?;
    Some(DisplacementTable { radius, length })
}

/// Device index of displacement `d` in a table of radius `R`: `clamp(d, -R, R) + R`.
///
/// This is the scalar specification of the tensor expression the modular
/// models evaluate; a unit test compares every device entry with it, so it is
/// compiled only for tests and proofs.
#[cfg(any(test, verus_keep_ghost))]
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(slot =>
    requires
        radius <= i64::MAX as usize / 2,
    ensures
        slot <= 2 * radius,
        -(radius as int) <= displacement <= radius as int
            ==> slot == displacement + radius as int,
        displacement < -(radius as int) ==> slot == 0,
        displacement > radius as int ==> slot == 2 * radius,
))]
pub(crate) fn displacement_slot(displacement: i64, radius: usize) -> usize {
    let bound = radius as i64;
    if displacement <= -bound {
        0
    } else if displacement >= bound {
        2 * radius
    } else {
        (displacement + bound) as usize
    }
}

/// Displacement `k - R` stored at table slot `k`; the inverse of `displacement_slot`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(displacement =>
    requires
        radius <= i64::MAX as usize / 2,
        slot <= 2 * radius,
    ensures
        displacement == slot as int - radius as int,
        -(radius as int) <= displacement <= radius as int,
))]
pub(crate) fn slot_displacement(slot: usize, radius: usize) -> i64 {
    slot as i64 - radius as i64
}

#[cfg(test)]
#[path = "plan_tests.rs"]
/// Planning examples, edge cases, and properties against naive references.
mod tests;
