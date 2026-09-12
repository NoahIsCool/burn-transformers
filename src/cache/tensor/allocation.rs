//! Backend-portable validation for tensor-cache allocation geometry.
//!
//! For dimensions `d_i`, allocation size is `product(d_i) * dtype.size()`.
//! Both products are checked before constructing backend shapes, with a
//! conservative `i32::MAX` per-tensor element limit for portable indexing. These
//! checks prevent arithmetic overflow; they cannot guarantee available device
//! memory or account for a backend's temporary allocations.

use burn::tensor::DType;

use super::{TensorCacheError, TensorCacheResult};

/// Largest per-tensor element count accepted before calling a Burn backend.
///
/// Burn backends and selector tensors commonly use signed 32-bit indices. The
/// conservative bound avoids backend-specific truncation even on 64-bit hosts.
pub(super) const PORTABLE_TENSOR_ELEMENT_LIMIT: usize = i32::MAX as usize;

/// Validates a tensor shape and returns its scalar element count.
///
/// The product `d_0 * ... * d_n` is checked before constructing a Burn
/// [`burn::tensor::Shape`], whose ordinary product may otherwise overflow.
pub(crate) fn validate_tensor_shape(
    tensor: &'static str,
    dimensions: &[usize],
    dtype: DType,
) -> TensorCacheResult<usize> {
    let elements = dimensions.iter().try_fold(1_usize, |product, &dimension| {
        product
            .checked_mul(dimension)
            .ok_or(TensorCacheError::TensorShapeOverflow { tensor })
    })?;
    if elements > PORTABLE_TENSOR_ELEMENT_LIMIT {
        return Err(TensorCacheError::TensorElementLimitExceeded {
            tensor,
            elements,
            limit: PORTABLE_TENSOR_ELEMENT_LIMIT,
        });
    }
    elements
        .checked_mul(dtype.size())
        .ok_or(TensorCacheError::AllocationSizeOverflow {
            tensor,
            elements,
            element_size: dtype.size(),
        })?;
    Ok(elements)
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
    let elements = elements_per_tensor
        .checked_mul(tensor_count)
        .ok_or(TensorCacheError::TensorShapeOverflow { tensor })?;
    elements
        .checked_mul(dtype.size())
        .ok_or(TensorCacheError::AllocationSizeOverflow {
            tensor,
            elements,
            element_size: dtype.size(),
        })?;
    Ok(elements)
}

#[cfg(test)]
#[path = "allocation_tests.rs"]
mod tests;
