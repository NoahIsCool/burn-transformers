//! Evaluation precision for rounding-sensitive transformer operations.
//!
//! Most transformer arithmetic runs in the activation dtype. A few shared
//! operations are much more sensitive to rounding than ordinary elementwise
//! arithmetic and are therefore evaluated in a separately selected dtype:
//!
//! - RMS normalization's mean-square reduction, inverse root, and learned
//!   scaling, `x / sqrt(mean(x^2) + epsilon) * scale`;
//! - explicit eager attention softmax, `exp(s_j - max s) / sum_k exp(s_k - max s)`,
//!   whose exponentials and row sums are taken over a whole key axis, including
//!   multi-head latent attention;
//! - rotary angles `p * omega_i` and their cosine/sine tables built for an
//!   activation dtype by [`crate::position::rotary_angles_for_dtype`] and
//!   [`crate::position::mrope_angles_for_dtype`], whose absolute phase error
//!   grows in proportion to the position `p`. Model families pass their
//!   query/key dtype to these builders.
//!
//! Family-specific copies follow the same rule, including Muse's scale-free
//! embedding and Q/K RMS norms. Vocabulary logits that reference implementations
//! return as FP32 (DeepSeek-V2 and Nemotron-H) are promoted with it rather than
//! cast unconditionally to F32, and sampling's cumulative probability sums use
//! it as well.
//!
//! [`at_least_f32`] is the single rule that selects that evaluation dtype:
//!
//! | activation dtype | evaluation dtype |
//! | --- | --- |
//! | F16, BF16, Flex32 | F32 (promoted) |
//! | F32 | F32 |
//! | F64 | F64 (never narrowed) |
//!
//! Reduced-precision activations are promoted exactly as the historical
//! unconditional FP32 graphs promoted them, so F16, BF16, and F32 results are
//! bit-identical to those graphs. F64 activations are never narrowed: on a
//! backend with native double precision, such as Flex, an F64 model performs
//! these operations in F64, and their rounding contribution is bounded by F64
//! rather than F32 unit roundoff. Each operation still returns its documented
//! output dtype, usually the activation dtype, using its checkpoint-specific
//! rounding order.
//!
//! The rule selects a dtype only; operation order, masks, and epsilon placement
//! are unchanged. Non-floating dtypes map to F32, matching the historical
//! unconditional promotion; callers remain responsible for rejecting
//! non-floating activations where that matters.

use burn::tensor::{DType, Tensor, backend::Backend};

/// Returns the dtype in which a rounding-sensitive operation is evaluated.
///
/// F64 stays F64. Every other dtype, including F32, F16, BF16, and Flex32,
/// evaluates in F32. This preserves the historical FP32 promotion for reduced
/// precision while never narrowing double-precision activations. See the
/// [module documentation](self) for the operations that follow this rule.
#[must_use]
pub const fn at_least_f32(dtype: DType) -> DType {
    match dtype {
        DType::F64 => DType::F64,
        _ => DType::F32,
    }
}

/// Casts a floating tensor to [`at_least_f32`] of its dtype.
///
/// Reduced-precision tensors are promoted to F32; F32 and F64 tensors are
/// returned unchanged. Model families use this where a reference implementation
/// returns full-precision outputs, such as FP32 vocabulary logits, so F64
/// models are not narrowed at that boundary.
pub(crate) fn promote<B: Backend, const D: usize>(tensor: Tensor<B, D>) -> Tensor<B, D> {
    let dtype = at_least_f32(tensor.dtype());
    tensor.cast(dtype)
}

#[cfg(test)]
pub(crate) mod test_support;

#[cfg(test)]
#[path = "precision_tests.rs"]
mod tests;
