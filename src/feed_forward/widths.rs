//! Pure width validation shared by dense and gated feed-forward blocks, checked by Verus.
//!
//! A dense block projects `D -> I -> D` and a gated block adds a second
//! `D -> I` projection, so both need a non-empty residual width `D` and
//! intermediate width `I`. [`width_fault`] reports the first empty width in
//! that order; Verus proves (through `verus/attention.rs`) that it accepts
//! exactly non-empty widths and names the first empty one.
//!
//! Known gap: unlike attention and MLA validation, feed-forward validation
//! does not yet check that the matrix element count `D * I` fits `usize`.
//! Burn allocates parameters lazily, so such a configuration passes
//! validation and fails only when a parameter is materialized. Adding the
//! check would reject configurations that validation accepts today, so it is
//! left to a dedicated follow-up.

#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

/// Which feed-forward width is empty.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WidthFault {
    /// The residual-stream width is zero.
    HiddenSize,
    /// The intermediate width is zero.
    IntermediateSize,
}

/// Returns the first empty width, residual width first, or `None` when both are non-zero.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(fault =>
    ensures
        fault == if hidden_size == 0 {
            Some(WidthFault::HiddenSize)
        } else if intermediate_size == 0 {
            Some(WidthFault::IntermediateSize)
        } else {
            None
        },
))]
pub(crate) fn width_fault(hidden_size: usize, intermediate_size: usize) -> Option<WidthFault> {
    if hidden_size == 0 {
        return Some(WidthFault::HiddenSize);
    }
    if intermediate_size == 0 {
        return Some(WidthFault::IntermediateSize);
    }
    None
}

#[cfg(test)]
#[path = "widths_tests.rs"]
mod tests;
