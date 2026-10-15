//! Input checks that read every device statistic in one host transfer.
//!
//! Validating token IDs, masks, and positions needs a handful of integer
//! reductions: `min`/`max` of each index tensor and the smallest number of
//! valid tokens in any row. Reading each scalar separately forces one blocking
//! device-to-host synchronization per value, which stalls asynchronous GPU
//! queues several times before any model math runs.
//!
//! [`DeferredChecks`] instead records the reductions as `[1]`-shaped device
//! tensors together with host predicates, concatenates the reductions into a
//! single vector, and reads it once. Predicates are evaluated afterwards in
//! the order they were registered, so error precedence matches the order in
//! which a family lists its checks. Reductions run in each tensor's integer
//! dtype; the only conversion is a lossless widening to `i64` when the
//! participating tensors use different integer dtypes.

use burn::tensor::{DType, Int, Tensor, backend::Backend};

use crate::{Error, Result, architecture::TokenMask};

use super::plan;

/// Handle to a `[minimum, maximum]` pair recorded by [`DeferredChecks`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RangeId(usize);

/// Handle to the smallest per-row valid-token count recorded by [`DeferredChecks`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RowCountId(usize);

/// One host predicate evaluated after the fused readback.
#[derive(Debug)]
enum Check<'a> {
    /// Requires `0 <= minimum` and `maximum < limit` for slots `slot, slot + 1`.
    Range {
        /// Logical tensor name reported on failure.
        name: &'a str,
        /// Human-readable quantity, such as `token` or `position`.
        quantity: &'static str,
        /// Exclusive upper bound.
        limit: usize,
        /// Index of the minimum; the maximum immediately follows it.
        slot: usize,
    },
    /// Requires the smallest per-row valid count in `slot` to be positive.
    NonEmptyRows {
        /// Logical mask name reported on failure.
        name: &'a str,
        /// Index of the smallest per-row valid-token count.
        slot: usize,
    },
}

/// Range and occupancy checks resolved by a single device-to-host readback.
///
/// Register checks in the order their errors should take precedence, then
/// call [`DeferredChecks::run`]. Without registered reductions `run` performs
/// no backend work at all.
pub(crate) struct DeferredChecks<'a, B: Backend> {
    /// Scalar reductions shaped `[1]`, read together by [`Self::run`].
    reductions: Vec<Tensor<B, 1, Int>>,
    /// Host predicates over `reductions`, in error-precedence order.
    checks: Vec<Check<'a>>,
}

impl<'a, B: Backend> DeferredChecks<'a, B> {
    /// Creates an empty check list.
    pub(crate) const fn new() -> Self {
        Self {
            reductions: Vec::new(),
            checks: Vec::new(),
        }
    }

    /// Requires every value to lie in `0..limit` and records the observed range.
    ///
    /// On failure the error is
    /// `Shape { tensor: name, expected: "{quantity} values in 0..{limit}",
    /// actual: "observed range {min}..={max}" }`.
    pub(crate) fn values_below<const D: usize>(
        &mut self,
        values: &Tensor<B, D, Int>,
        limit: usize,
        name: &'a str,
        quantity: &'static str,
    ) -> RangeId {
        let slot = self.reductions.len();
        // `Tensor::min`/`max` reduce every axis to a `[1]` tensor without a host read.
        self.reductions.push(values.clone().min());
        self.reductions.push(values.clone().max());
        self.checks.push(Check::Range {
            name,
            quantity,
            limit,
            slot,
        });
        RangeId(slot)
    }

    /// Requires every mask row to contain a valid token and records the smallest count.
    ///
    /// A fully masked row fails with `Shape { tensor: name, expected: "at least
    /// one valid token in every row", actual: "a fully masked row" }`.
    pub(crate) fn row_counts(&mut self, mask: &TokenMask<B>, name: &'a str) -> RowCountId {
        let slot = self.reductions.len();
        // Σ_t m[b, t] is row b's valid-token count; its minimum detects empty rows.
        self.reductions
            .push(mask.as_tensor().clone().int().sum_dim(1).min());
        self.checks.push(Check::NonEmptyRows { name, slot });
        RowCountId(slot)
    }

    /// Reads every recorded reduction once and evaluates the predicates in order.
    ///
    /// # Errors
    ///
    /// Returns the first failing predicate's error, or
    /// [`Error::BackendCapability`] when the backend cannot read the values.
    pub(crate) fn run(self, label: &str) -> Result<Readout> {
        let readout = Readout {
            values: read_int_tensors(self.reductions, label)?,
        };
        for check in &self.checks {
            readout.evaluate(check)?;
        }
        Ok(readout)
    }
}

/// Host copy of the reductions registered on a [`DeferredChecks`].
///
/// Its length equals the number of registered reductions, so every handle
/// issued by the same check list addresses a present value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Readout {
    /// Reduction results in registration order.
    values: Vec<i64>,
}

impl Readout {
    /// Returns the observed `(minimum, maximum)` recorded under `id`.
    pub(crate) fn range(&self, id: RangeId) -> (i64, i64) {
        (self.values[id.0], self.values[id.0 + 1])
    }

    /// Returns the smallest per-row valid-token count recorded under `id`.
    pub(crate) fn min_row_count(&self, id: RowCountId) -> i64 {
        self.values[id.0]
    }

    /// Converts the maximum recorded under `id` into the context extent `max + 1`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Configuration`] for a negative maximum or an extent
    /// that does not fit `usize`.
    pub(crate) fn context(&self, id: RangeId, name: &str) -> Result<usize> {
        let (_, maximum) = self.range(id);
        plan::context_from_maximum(maximum).ok_or_else(|| {
            Error::configuration(
                name,
                format!("maximum position {maximum} has no representable context extent"),
            )
        })
    }

    /// Evaluates one registered predicate against the host values.
    fn evaluate(&self, check: &Check<'_>) -> Result<()> {
        match *check {
            Check::Range {
                name,
                quantity,
                limit,
                slot,
            } => {
                let (minimum, maximum) = self.range(RangeId(slot));
                if plan::range_within(minimum, maximum, limit) {
                    Ok(())
                } else {
                    Err(range_error(name, quantity, limit, minimum, maximum))
                }
            }
            Check::NonEmptyRows { name, slot } => {
                if self.min_row_count(RowCountId(slot)) > 0 {
                    Ok(())
                } else {
                    Err(Error::shape(
                        name,
                        "at least one valid token in every row",
                        "a fully masked row",
                    ))
                }
            }
        }
    }
}

/// Builds the shared out-of-range error for token or position values.
pub(crate) fn range_error(
    name: &str,
    quantity: &str,
    limit: usize,
    minimum: i64,
    maximum: i64,
) -> Error {
    Error::shape(
        name,
        format!("{quantity} values in 0..{limit}"),
        format!("observed range {minimum}..={maximum}"),
    )
}

/// Reads integer tensors to the host in one transfer, in order and flattened.
///
/// Tensors sharing one integer dtype are concatenated as-is; mixed dtypes are
/// first widened to `i64`, which is lossless for every Burn integer dtype the
/// crate produces. An empty list performs no backend work. The result holds
/// exactly one value per input element.
///
/// # Errors
///
/// Returns [`Error::BackendCapability`] when the backend cannot read the data
/// or returns a different number of values.
pub(crate) fn read_int_tensors<B: Backend>(
    tensors: Vec<Tensor<B, 1, Int>>,
    label: &str,
) -> Result<Vec<i64>> {
    let Some(first) = tensors.first() else {
        return Ok(Vec::new());
    };
    let expected = tensors.iter().map(|tensor| tensor.dims()[0]).sum::<usize>();
    let dtype = first.dtype();
    let uniform = tensors.iter().all(|tensor| tensor.dtype() == dtype);
    let tensors = if uniform {
        tensors
    } else {
        tensors
            .into_iter()
            .map(|tensor| tensor.cast(DType::I64))
            .collect()
    };
    let data = Tensor::cat(tensors, 0).try_into_data().map_err(|error| {
        Error::BackendCapability(format!("failed to read {label} from backend: {error}"))
    })?;
    let values = data.iter::<i64>().collect::<Vec<_>>();
    if values.len() != expected {
        return Err(Error::BackendCapability(format!(
            "{label} readback returned {} values, expected {expected}",
            values.len()
        )));
    }
    Ok(values)
}

/// Rejects a tensor that lives on a different device than its owner.
///
/// Called before any reduction or kernel mixes the tensors, so a mixed-device
/// call fails with [`Error::BackendCapability`] instead of a backend panic.
pub(crate) fn ensure_same_device<D: PartialEq>(
    name: &str,
    actual: &D,
    expected: &D,
    owner: &str,
) -> Result<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(Error::BackendCapability(format!(
            "{name} and {owner} are on different devices"
        )))
    }
}

/// Rejects an empty `[batch, sequence]` extent before embedding or reductions.
pub(crate) fn ensure_nonempty(batch: usize, sequence: usize, name: &str) -> Result<()> {
    if batch == 0 || sequence == 0 {
        return Err(Error::shape(
            name,
            "non-empty [batch, sequence]",
            format!("[{batch}, {sequence}]"),
        ));
    }
    Ok(())
}

/// Requires an auxiliary `[batch, sequence]` tensor to match the token grid.
pub(crate) fn ensure_grid(
    actual: [usize; 2],
    batch: usize,
    sequence: usize,
    name: &str,
) -> Result<()> {
    if actual != [batch, sequence] {
        return Err(Error::shape(
            name,
            format!("[{batch}, {sequence}]"),
            format!("{actual:?}"),
        ));
    }
    Ok(())
}

#[cfg(test)]
#[path = "checks_tests.rs"]
mod tests;
