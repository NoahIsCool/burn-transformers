//! Backend-portable validation for tensor-cache allocation geometry.
//!
//! For dimensions `d_i`, allocation size is `product(d_i) * dtype.size()`.
//! Both products are checked before constructing backend shapes, with a
//! conservative `i32::MAX` per-tensor element limit for portable indexing. These
//! checks prevent arithmetic overflow; they cannot guarantee available device
//! memory or account for a backend's temporary allocations.
//!
//! The arithmetic is the Verus-verified `tensor_elements` and
//! `repeated_elements` in `plan.rs`; this module only supplies the dtype's
//! element size and names the rejected tensor in the error.

use burn::tensor::DType;

use super::plan::{repeated_elements, tensor_elements};
use super::{TensorCacheError, TensorCacheResult};

/// Validates a tensor shape and returns its scalar element count.
///
/// The product `d_0 * ... * d_n` is checked before constructing a Burn
/// [`burn::tensor::Shape`], whose ordinary product may otherwise overflow.
pub(crate) fn validate_tensor_shape(
    tensor: &'static str,
    dimensions: &[usize],
    dtype: DType,
) -> TensorCacheResult<usize> {
    tensor_elements(dimensions, dtype.size())
        .map_err(|violation| TensorCacheError::allocation(tensor, violation, dtype.size()))
}

/// Validates aggregate byte size and returns the scalar count across all allocations.
///
/// This is an arithmetic check rather than an admission controller: a backend
/// may still report out-of-memory for a valid but impractically large cache.
pub(crate) fn validate_repeated_allocation(
    tensor: &'static str,
    elements_per_tensor: usize,
    tensor_count: usize,
    dtype: DType,
) -> TensorCacheResult<usize> {
    repeated_elements(elements_per_tensor, tensor_count, dtype.size())
        .map_err(|violation| TensorCacheError::allocation(tensor, violation, dtype.size()))
}

#[cfg(test)]
#[path = "allocation_tests.rs"]
mod tests;
