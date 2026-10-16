//! Admission arithmetic of an incremental Burnpack staging session.
//!
//! A session keeps three running totals, each under a configured limit: the
//! staged tensor count `c <= T`, the staged payload bytes `s <= S`, and the
//! estimated manifest bytes `m <= M`. Every change first asks whether the new
//! total stays within its limit ([`bounded_add`], [`admit_tensor`]) and commits
//! the returned totals only on success, so each limit is an invariant of the
//! session: no admitted transition can leave a total above its limit or wrap
//! it around.
//!
//! The limits themselves are validated first ([`first_limit_violation`]):
//! each must be positive and at most a fixed cap derived from Burn Store's
//! reader bounds.
//!
//! The module has no crate or Burn dependencies so that Verus can check it in
//! place through `verus/checkpoint.rs`; callers map each breach to the
//! contextual quantization error of the total it concerns.

#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

/// Why a bounded running total cannot admit an addition.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BoundBreach {
    /// The new total exceeds `u64::MAX`.
    Overflow,
    /// The new total exceeds the limit.
    Exceeded {
        /// The exact new total that was refused.
        attempted: u64,
    },
}

/// Returns `total + addition` when it stays within `limit`.
///
/// Verified: `Ok` exactly when the mathematical sum is at most `limit`, with
/// the exact sum, so a total kept below its limit stays below it; `Overflow`
/// exactly when the sum exceeds `u64::MAX`, and `Exceeded` carries the exact
/// refused sum otherwise.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(admitted =>
    ensures
        match admitted {
            Ok(sum) => total + addition <= limit && sum == total + addition,
            Err(BoundBreach::Overflow) => total + addition > u64::MAX,
            Err(BoundBreach::Exceeded { attempted }) => total + addition <= u64::MAX
                && attempted == total + addition
                && attempted > limit,
        }
))]
pub(crate) fn bounded_add(total: u64, addition: u64, limit: u64) -> Result<u64, BoundBreach> {
    let Some(sum) = total.checked_add(addition) else {
        return Err(BoundBreach::Overflow);
    };
    if sum > limit {
        return Err(BoundBreach::Exceeded { attempted: sum });
    }
    Ok(sum)
}

/// First limit a prospective tensor would break.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TensorBreach {
    /// The tensor count would exceed `usize::MAX`.
    CountOverflow,
    /// The tensor count `attempted` would exceed the tensor limit.
    TooManyTensors {
        /// The refused tensor count.
        attempted: usize,
    },
    /// The tensor's own payload exceeds the per-tensor limit.
    PayloadTooLarge,
    /// The staged payload total would break its limit.
    Staged(
        /// How the staged total was refused.
        BoundBreach,
    ),
}

/// Counters after admitting one more tensor.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AdmittedTensor {
    /// Staged tensor count including the new tensor.
    pub(crate) count: usize,
    /// Staged payload bytes including the new tensor's payload.
    pub(crate) staged: u64,
}

/// Admits a `payload`-byte tensor after `count` tensors totalling `staged` bytes.
///
/// The checks run in order: tensor count, the payload itself, then the staged
/// total; the first failure is returned.
///
/// Verified: `Ok` exactly when the new count, the payload, and the new staged
/// total are all within their limits, with the exact new count and total;
/// each breach names the first failed check.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(admitted =>
    ensures
        match admitted {
            Ok(next) => {
                &&& count + 1 <= max_tensors
                &&& payload <= max_payload
                &&& staged + payload <= max_staged
                &&& next.count == count + 1
                &&& next.staged == staged + payload
            },
            Err(TensorBreach::CountOverflow) => count == usize::MAX,
            Err(TensorBreach::TooManyTensors { attempted }) => count < usize::MAX
                && attempted == count + 1
                && attempted > max_tensors,
            Err(TensorBreach::PayloadTooLarge) => count + 1 <= max_tensors
                && payload > max_payload,
            Err(TensorBreach::Staged(BoundBreach::Overflow)) => count + 1 <= max_tensors
                && payload <= max_payload
                && staged + payload > u64::MAX,
            Err(TensorBreach::Staged(BoundBreach::Exceeded { attempted })) => {
                &&& count + 1 <= max_tensors
                &&& payload <= max_payload
                &&& attempted == staged + payload
                &&& attempted > max_staged
            },
        }
))]
pub(crate) fn admit_tensor(
    count: usize,
    staged: u64,
    payload: u64,
    max_tensors: usize,
    max_payload: u64,
    max_staged: u64,
) -> Result<AdmittedTensor, TensorBreach> {
    let Some(next_count) = count.checked_add(1) else {
        return Err(TensorBreach::CountOverflow);
    };
    if next_count > max_tensors {
        return Err(TensorBreach::TooManyTensors {
            attempted: next_count,
        });
    }
    if payload > max_payload {
        return Err(TensorBreach::PayloadTooLarge);
    }
    match bounded_add(staged, payload, max_staged) {
        Ok(next_staged) => Ok(AdmittedTensor {
            count: next_count,
            staged: next_staged,
        }),
        Err(breach) => Err(TensorBreach::Staged(breach)),
    }
}

/// Why a configured session limit is refused.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LimitViolation {
    /// The limit is zero, which would admit nothing.
    Zero,
    /// The limit exceeds its fixed cap.
    AboveCap,
}

#[cfg(verus_keep_ghost)]
verus! {

/// Specification: `value` is an admissible limit under `cap`, i.e. `1 <= value <= cap`.
pub open spec fn within_cap(value: u64, cap: u64) -> bool {
    1 <= value <= cap
}

} // verus!

/// Returns the first limit outside `[1, cap]`, checking each limit for zero, then its cap.
///
/// Limits are checked in order, each first for zero and then against its own
/// cap, so the result names the same failure a field-by-field check would.
///
/// Verified: `None` exactly when every limit lies in `[1, cap]`; otherwise the
/// index of the first limit that does not, every earlier limit being
/// admissible, with `Zero` for a zero limit and `AboveCap` for one above its
/// cap. A validated session therefore has positive limits that never exceed
/// the reader bounds the caps encode.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(violation =>
    requires
        values@.len() == caps@.len(),
    ensures
        match violation {
            None => forall|i: int| 0 <= i < values@.len()
                ==> within_cap(#[trigger] values@[i], caps@[i]),
            Some((index, kind)) => {
                &&& index < values@.len()
                &&& forall|i: int| 0 <= i < index ==> within_cap(#[trigger] values@[i], caps@[i])
                &&& match kind {
                    LimitViolation::Zero => values@[index as int] == 0,
                    LimitViolation::AboveCap => values@[index as int] != 0
                        && values@[index as int] > caps@[index as int],
                }
            },
        }
))]
pub(crate) fn first_limit_violation(
    values: &[u64],
    caps: &[u64],
) -> Option<(usize, LimitViolation)> {
    let mut index = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            index <= values@.len(),
            values@.len() == caps@.len(),
            forall|i: int| 0 <= i < index ==> within_cap(#[trigger] values@[i], caps@[i]),
        decreases values@.len() - index,
    ))]
    while index < values.len() {
        if values[index] == 0 {
            return Some((index, LimitViolation::Zero));
        }
        if values[index] > caps[index] {
            return Some((index, LimitViolation::AboveCap));
        }
        index += 1;
    }
    None
}

#[cfg(test)]
#[path = "staging_tests.rs"]
mod tests;
