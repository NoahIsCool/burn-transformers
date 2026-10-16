//! Exact rounded-up integer division shared by the checkpoint loader's planners.
//!
//! `⌈n / d⌉` for `d >= 1` is the least `q` with `n <= q · d`. Quantized group
//! counts, packed Q4 byte counts, and transpose cache blocks, bands, and
//! threads all use it. Computing it as `n / d + [n mod d != 0]` never forms
//! the overflow-prone `n + d - 1`, so it is total on the whole integer range.
//!
//! The module has no crate, Burn, or allocator dependencies so that Verus can
//! check it in place; `verus/checkpoint.rs` and `verus/perf.rs` both include it
//! and mirror its crate path `crate::checkpoint::division`.

#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

#[cfg(verus_keep_ghost)]
verus! {

/// Specification: the exact rounded-up quotient `⌈dividend / divisor⌉` for `divisor > 0`.
pub open spec fn ceiling(dividend: int, divisor: int) -> int {
    if dividend % divisor == 0 {
        dividend / divisor
    } else {
        dividend / divisor + 1
    }
}

/// `⌈n / d⌉` is the least `q` with `n <= q · d`; it never exceeds `n` and is positive for `n > 0`.
pub proof fn lemma_ceiling_bounds(dividend: int, divisor: int)
    requires
        dividend >= 0,
        divisor >= 1,
    ensures
        dividend <= ceiling(dividend, divisor) * divisor,
        ceiling(dividend, divisor) == 0 || (ceiling(dividend, divisor) - 1) * divisor < dividend,
        0 <= ceiling(dividend, divisor) <= dividend,
        dividend > 0 ==> ceiling(dividend, divisor) > 0,
        dividend / divisor <= dividend,
        divisor >= 2 && dividend % divisor != 0 ==> dividend / divisor < dividend,
{
    let q = dividend / divisor;
    let r = dividend % divisor;
    vstd::arithmetic::div_mod::lemma_fundamental_div_mod(dividend, divisor);
    assert(dividend == divisor * q + r);
    assert(0 <= r < divisor);
    assert(q >= 0) by (nonlinear_arith)
        requires
            dividend == divisor * q + r,
            0 <= r < divisor,
            dividend >= 0,
    ;
    assert(q * divisor >= q) by (nonlinear_arith)
        requires
            divisor >= 1,
            q >= 0,
    ;
    assert(q <= dividend) by (nonlinear_arith)
        requires
            dividend == divisor * q + r,
            r >= 0,
            divisor >= 1,
            q >= 0,
    ;
    if r == 0 {
        assert(dividend == q * divisor) by (nonlinear_arith)
            requires
                dividend == divisor * q + r,
                r == 0,
        ;
        if q > 0 {
            assert((q - 1) * divisor < dividend) by (nonlinear_arith)
                requires
                    dividend == q * divisor,
                    divisor >= 1,
            ;
        }
        if dividend > 0 {
            assert(q > 0) by (nonlinear_arith)
                requires
                    dividend == q * divisor,
                    dividend > 0,
                    q >= 0,
            ;
        }
    } else {
        assert(dividend <= (q + 1) * divisor) by (nonlinear_arith)
            requires
                dividend == divisor * q + r,
                r < divisor,
        ;
        assert(q * divisor < dividend) by (nonlinear_arith)
            requires
                dividend == divisor * q + r,
                r > 0,
        ;
        // A nonzero remainder needs a divisor of at least two, so q + 1 <= dividend.
        assert(q + 1 <= dividend) by (nonlinear_arith)
            requires
                dividend == divisor * q + r,
                0 < r < divisor,
                q >= 0,
        ;
    }
    if divisor >= 2 && r != 0 {
        assert(q < dividend) by (nonlinear_arith)
            requires
                dividend == divisor * q + r,
                r > 0,
                divisor >= 2,
                q >= 0,
        ;
    }
}

} // verus!

/// Returns `⌈value / divisor⌉` without forming the overflow-prone `value + divisor - 1`.
///
/// Verified: the exact mathematical ceiling, i.e. the least `q` with
/// `value <= q · divisor`; it never exceeds `value` and is positive whenever
/// `value` is.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(quotient =>
    requires
        divisor > 0,
    ensures
        quotient == ceiling(value as int, divisor as int),
        value <= quotient * divisor,
        quotient == 0 || (quotient - 1) * divisor < value,
        value > 0 ==> quotient > 0,
        quotient <= value,
))]
pub(crate) const fn ceil_div_u64(value: u64, divisor: u64) -> u64 {
    #[cfg(verus_keep_ghost)]
    proof! {
        lemma_ceiling_bounds(value as int, divisor as int);
    }
    // A named remainder keeps this Verus-checkable (`is_multiple_of` has no spec).
    let remainder = value % divisor;
    value / divisor + if remainder == 0 { 0 } else { 1 }
}

/// Returns `⌈value / divisor⌉` over host indices; see [`ceil_div_u64`].
///
/// Verified: the exact mathematical ceiling, i.e. the least `q` with
/// `value <= q · divisor`; it never exceeds `value` and is positive whenever
/// `value` is.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(quotient =>
    requires
        divisor > 0,
    ensures
        quotient == ceiling(value as int, divisor as int),
        value <= quotient * divisor,
        quotient == 0 || (quotient - 1) * divisor < value,
        value > 0 ==> quotient > 0,
        quotient <= value,
))]
pub(crate) const fn ceil_div_usize(value: usize, divisor: usize) -> usize {
    #[cfg(verus_keep_ghost)]
    proof! {
        lemma_ceiling_bounds(value as int, divisor as int);
    }
    // A named remainder keeps this Verus-checkable (`is_multiple_of` has no spec).
    let remainder = value % divisor;
    value / divisor + if remainder == 0 { 0 } else { 1 }
}

#[cfg(test)]
#[path = "division_tests.rs"]
mod tests;
