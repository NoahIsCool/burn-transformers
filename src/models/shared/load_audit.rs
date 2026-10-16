//! Pure counter arithmetic behind [`crate::LoadedModel`] load audits.
//!
//! A successful checkpoint load carries redundant counters that are checked
//! against its strict report before a [`crate::LoadedModel`] is constructed:
//! quantized payload bytes must sum to the decoded total and their maximum
//! must equal the largest decoded payload, and every verified tensor
//! equivalence must have read exactly two source tensors. The integer core
//! of those checks lives here, free of Burn and allocation, so Verus checks
//! these exact bodies through `verus/models.rs` (see `verus/README.md`).
//!
//! # Verified properties
//!
//! - [`payload_totals`] returns the exact sum `Σ b_i` (or `None` when it does
//!   not fit `u64`) together with the exact maximum `max b_i` (zero for no
//!   records).
//! - [`equivalence_reads`] is exactly `2 n` as a `u64`, or `None` when that
//!   does not fit.

#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

#[cfg(verus_keep_ghost)]
verus! {

/// Sum of the first `count` byte counts.
pub open spec fn sum_bytes(bytes: Seq<u64>, count: int) -> int
    decreases count,
{
    if count <= 0 {
        0
    } else {
        sum_bytes(bytes, count - 1) + bytes[count - 1]
    }
}

/// Byte-count prefix sums never decrease.
pub proof fn lemma_sum_bytes_monotone(bytes: Seq<u64>, middle: int, count: int)
    requires
        0 <= middle <= count,
    ensures
        sum_bytes(bytes, middle) <= sum_bytes(bytes, count),
    decreases count - middle,
{
    if count > middle {
        lemma_sum_bytes_monotone(bytes, middle, count - 1);
    }
}

} // verus!

/// Exact total and largest value of per-record payload byte counts.
///
/// Returns `Some((Σ b_i, max b_i))`, with a maximum of zero for no records,
/// or `None` exactly when the total does not fit `u64`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(totals =>
    ensures
        match totals {
            Some((total, largest)) => {
                &&& total == sum_bytes(bytes@, bytes.len() as int)
                &&& forall|i: int| 0 <= i < bytes.len() ==> #[trigger] bytes[i] <= largest
                &&& (bytes.len() == 0 ==> largest == 0)
                &&& (bytes.len() > 0 ==> exists|i: int| 0 <= i < bytes.len() && bytes[i] == largest)
            },
            None => sum_bytes(bytes@, bytes.len() as int) > u64::MAX,
        },
))]
pub(crate) fn payload_totals(bytes: &[u64]) -> Option<(u64, u64)> {
    let mut total: u64 = 0;
    let mut largest: u64 = 0;
    let mut index: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            index <= bytes.len(),
            total == sum_bytes(bytes@, index as int),
            forall|i: int| 0 <= i < index ==> #[trigger] bytes[i] <= largest,
            index == 0 ==> largest == 0,
            index > 0 ==> exists|i: int| 0 <= i < index && bytes[i] == largest,
        decreases bytes.len() - index,
    ))]
    while index < bytes.len() {
        let Some(next) = total.checked_add(bytes[index]) else {
            #[cfg(verus_keep_ghost)]
            proof! {
                lemma_sum_bytes_monotone(bytes@, index as int + 1, bytes.len() as int);
            }
            return None;
        };
        total = next;
        if index == 0 || bytes[index] > largest {
            largest = bytes[index];
        }
        index += 1;
    }
    Some((total, largest))
}

/// Source reads expected for `verified` tensor equivalences: two per pair.
///
/// Returns `Some(2 n)` as a `u64`, or `None` exactly when it does not fit.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(reads =>
    ensures
        match reads {
            Some(reads) => reads == 2 * verified,
            None => 2 * verified > u64::MAX,
        },
))]
pub(crate) fn equivalence_reads(verified: usize) -> Option<u64> {
    let count = verified as u64;
    count.checked_mul(2)
}

#[cfg(test)]
#[path = "load_audit_tests.rs"]
mod tests;
