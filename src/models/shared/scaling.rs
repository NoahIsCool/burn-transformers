//! Dtype-exact scalar multipliers and tanh soft caps for embeddings and logits.
//!
//! Two readout mechanisms recur across families and the modular stacks, and
//! both must reproduce their references bit for bit in reduced precision:
//!
//! * **Scalar multipliers.** Gemma multiplies token embeddings by
//!   `s = sqrt(d)`. The references first turn `s` into a tensor of the
//!   activation dtype and then multiply, so the product uses `s` *rounded to
//!   that dtype*, not the `f64` value. [`round_to_dtype`] performs that
//!   rounding on the host; a tensor of `dtype` multiplied by its result then
//!   uses a scalar the dtype represents exactly, whatever conversion a backend
//!   applies to scalar operands.
//! * **Soft caps.** [`soft_cap`] maps every element `z` to `c · tanh(z / c)`
//!   (Gemma 2's `final_logit_softcapping`), computed in the tensor's own dtype
//!   with the reference operation order: divide, `tanh`, multiply.
//!
//! # Rounding a scalar like a reference tensor
//!
//! [`round_to_dtype`] rounds the `f64` to F32 and then to the target dtype,
//! which is how the Gemma family and Gemma 3's reference (an F32
//! `embed_scale` buffer cast to the weight dtype) build the multiplier:
//!
//! | dtype | result | PyTorch `torch.tensor(value, dtype=dtype)` (Gemma, Gemma 2) |
//! | --- | --- | --- |
//! | F64 | `value` (exact) | the same |
//! | F32 | `(value)_{f32}` | the same |
//! | F16 | `((value)_{f32})_{f16}` | rounds the `f64` directly |
//! | BF16 | `((value)_{f32})_{bf16}` | the same (`c10::BFloat16` is built from an `f32`) |
//!
//! Every rounding is to nearest, ties to even. Integer and Boolean dtypes use
//! the F32 value (embeddings and logits are always floating point). The F16
//! row can differ from PyTorch's direct rounding only when `(value)_{f32}`
//! lands exactly on an F16 tie that `value` itself misses (a double
//! rounding); no `sqrt(d)` of a shipped Gemma width does, and the tests pin
//! both that agreement and a crafted counterexample. Rounding through F32 is
//! portable: `half`'s direct `f64` conversions are not correctly rounded on
//! every target, while its `f32` conversions are. A family that reproduces an
//! F32 buffer for F64 activations as well (Gemma 3) passes the F32 value,
//! `round_to_dtype(f64::from(value as f32), dtype)`.
//!
//! Both helpers are model-independent and contain no model-name branches; a
//! family passes its own scalar and cap.

use burn::tensor::{DType, Tensor, backend::Backend, bf16, f16};

/// Rounds a scalar multiplier to `dtype` through F32, keeping F64 exact.
///
/// See the module documentation for the per-dtype conversions and how they
/// relate to the references. The result is representable in `dtype`, so
/// multiplying a tensor of `dtype` by it uses exactly that scalar on every
/// backend.
#[must_use]
pub(crate) fn round_to_dtype(value: f64, dtype: DType) -> f64 {
    let single = value as f32;
    match dtype {
        DType::F64 => value,
        DType::F16 => f16::from_f32(single).to_f64(),
        DType::BF16 => bf16::from_f32(single).to_f64(),
        _ => f64::from(single),
    }
}

/// Applies an optional tanh soft cap `c · tanh(z / c)` elementwise in the tensor's dtype.
///
/// `None` returns the tensor unchanged. The output keeps the input's shape and
/// dtype, lies within `[-c, c]` (`tanh` saturates to exactly one in F32, so a
/// large logit maps to `±c`), is odd and monotone, and is close to the identity
/// for `|z| ≪ c`. The derivative `1 - tanh²(z / c)` flows through Burn's
/// autodiff. The caller validates that `c` is finite and positive.
#[must_use]
pub(crate) fn soft_cap<B: Backend, const D: usize>(
    tensor: Tensor<B, D>,
    cap: Option<f64>,
) -> Tensor<B, D> {
    match cap {
        Some(cap) => tensor.div_scalar(cap).tanh().mul_scalar(cap),
        None => tensor,
    }
}

#[cfg(test)]
#[path = "scaling_tests.rs"]
mod tests;
