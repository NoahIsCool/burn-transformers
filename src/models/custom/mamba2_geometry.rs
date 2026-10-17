//! Pure Mamba-2 head-grouping arithmetic for modular specifications.
//!
//! A [`crate::architecture::Mamba2Spec`] states how many heads share one
//! `B/C` group (`group_size`); the shared mixer wants the number of groups.
//! The translation lives here, free of Burn and allocation, so Verus checks
//! this exact body through `verus/integration.rs` (see `verus/README.md`).
//!
//! # Verified properties
//!
//! - [`mamba2_groups`] succeeds exactly when `group_size` is positive and
//!   divides `heads`, and then returns the `g` with `g * group_size == heads`,
//!   so a translated configuration can never silently regroup heads.

#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

/// Number of `B/C` groups when each group is shared by `group_size` contiguous heads.
///
/// Returns `None` unless `group_size` is positive and divides `heads`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(groups =>
    ensures
        groups.is_some() <==> (group_size > 0 && heads % group_size == 0),
        match groups {
            Some(count) => count * group_size == heads && count == heads / group_size,
            None => true,
        },
))]
pub(crate) fn mamba2_groups(heads: usize, group_size: usize) -> Option<usize> {
    if group_size == 0 || !heads.is_multiple_of(group_size) {
        return None;
    }
    let count = heads / group_size;
    #[cfg(verus_keep_ghost)]
    proof! {
        vstd::arithmetic::div_mod::lemma_fundamental_div_mod(heads as int, group_size as int);
        assert(count * group_size == group_size * count) by (nonlinear_arith);
    }
    Some(count)
}

#[cfg(test)]
#[path = "mamba2_geometry_tests.rs"]
mod tests;
