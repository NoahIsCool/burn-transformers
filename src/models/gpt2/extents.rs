//! Dependency-free dimension checks shared by the GPT-2 and GPT-NeoX configurations.
//!
//! Both families index every extent with signed 32-bit tensor indices, so a
//! vocabulary, width, depth, head count, or position count must lie in
//! `1..=i32::MAX`; parameter matrices must have representable element
//! counts; and GPT-2's MLP width defaults to `4 d`.
//!
//! # Verified checks
//!
//! Every function here is checked in place by Verus through
//! `verus/t5_gpt.rs` (see `verus/README.md`): the first invalid extent is
//! found exactly, the MLP width follows GPT-2's default rule exactly, and a
//! product chain is accepted exactly when every running product fits
//! `usize`, which is the checked multiplication it replaces.

#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

#[cfg(verus_keep_ghost)]
verus! {

/// An extent usable as a signed 32-bit tensor index: `1..=i32::MAX`.
pub open spec fn valid_extent(value: usize) -> bool {
    0 < value <= i32::MAX
}

/// Product of the first `count` factors.
pub open spec fn prefix_product(factors: Seq<usize>, count: int) -> int
    decreases count,
{
    if count <= 0 {
        1
    } else {
        prefix_product(factors, count - 1) * factors[count - 1]
    }
}

/// Running products are never negative.
proof fn lemma_prefix_product_nonnegative(factors: Seq<usize>, count: int)
    ensures
        prefix_product(factors, count) >= 0,
    decreases count,
{
    if count > 0 {
        lemma_prefix_product_nonnegative(factors, count - 1);
        let previous = prefix_product(factors, count - 1);
        let factor = factors[count - 1] as int;
        assert(previous * factor >= 0) by (nonlinear_arith)
            requires previous >= 0, factor >= 0;
    }
}

} // verus!

/// Position of the first extent outside `1..=i32::MAX`, if any.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(invalid =>
    ensures
        match invalid {
            Some(position) => {
                &&& position < values.len()
                &&& !valid_extent(values@[position as int])
                &&& forall|q: int| 0 <= q < position ==> valid_extent(#[trigger] values@[q])
            },
            None => forall|q: int| 0 <= q < values.len() ==> valid_extent(#[trigger] values@[q]),
        },
))]
pub(crate) fn first_invalid_extent(values: &[usize]) -> Option<usize> {
    let mut position: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            position <= values.len(),
            forall|q: int| 0 <= q < position ==> valid_extent(#[trigger] values@[q]),
        decreases values.len() - position,
    ))]
    while position < values.len() {
        let value = values[position];
        if value == 0 || value > i32::MAX as usize {
            return Some(position);
        }
        position += 1;
    }
    None
}

/// GPT-2's MLP width: the explicit `n_inner`, or `4 d` when it is absent.
///
/// Returns `None` for an explicit zero width, or for a default `4 d` that is
/// zero or does not fit `usize`; an explicit width is never replaced by the
/// default.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(inner =>
    ensures
        inner == match explicit {
            Some(inner) => if inner != 0 { Some(inner) } else { None },
            None => if 0 < 4 * width <= usize::MAX { Some((4 * width) as usize) } else { None },
        },
))]
pub(crate) fn intermediate_width(explicit: Option<usize>, width: usize) -> Option<usize> {
    let inner = match explicit {
        Some(inner) => inner,
        None => width.checked_mul(4)?,
    };
    if inner == 0 { None } else { Some(inner) }
}

/// Whether multiplying `factors` left to right never overflows `usize`.
///
/// This is exactly the chain `factors[0].checked_mul(factors[1])…` succeeding:
/// every running product must fit, even when a later factor is zero.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(fits =>
    ensures
        fits == forall|count: int| 0 <= count <= factors.len()
            ==> #[trigger] prefix_product(factors@, count) <= usize::MAX,
))]
pub(crate) fn product_fits(factors: &[usize]) -> bool {
    let mut product: usize = 1;
    let mut count: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            count <= factors.len(),
            product == prefix_product(factors@, count as int),
            forall|seen: int| 0 <= seen <= count
                ==> #[trigger] prefix_product(factors@, seen) <= usize::MAX,
        decreases factors.len() - count,
    ))]
    while count < factors.len() {
        #[cfg(verus_keep_ghost)]
        proof! {
            lemma_prefix_product_nonnegative(factors@, count as int);
        }
        match product.checked_mul(factors[count]) {
            Some(next) => product = next,
            None => {
                #[cfg(verus_keep_ghost)]
                proof! {
                    assert(prefix_product(factors@, count as int + 1) > usize::MAX);
                }
                return false;
            }
        }
        count += 1;
    }
    true
}

#[cfg(test)]
#[path = "extents_tests.rs"]
mod tests;
