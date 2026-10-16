//! Fused split-half rotary kernel for the inference Burn Flex backend.
//!
//! # Why a backend kernel
//!
//! [`super::rotate_prefix`] multiplies a `[batch, heads, sequence, width]`
//! operand by `[batch, 1, sequence, rotary]` cosine and sine tables broadcast
//! over heads. Projected queries and keys usually arrive as a heads-swapped
//! view of contiguous `[batch, sequence, heads, width]` storage. Burn Flex
//! 0.21 accelerates only two broadcast shapes (one row shared by every leading
//! position, or one scalar per row); this one matches neither, so each product
//! first copies the strided operand and then walks both operands with a scalar
//! strided iterator. That made RoPE about a fifth of a small Llama's Flex
//! prefill (1.5 s per 512-token Qwen3-4B prefill across 36 layers).
//!
//! # What it computes
//!
//! For every row `(b, h, t)` with `half = rotary / 2` and channel `j`:
//!
//! - `j < half`: `y_j = x_j · cos_j + (−x_{j+half}) · sin_j`;
//! - `half <= j < rotary`: `y_j = x_j · cos_j + x_{j−half} · sin_j`;
//! - `rotary <= j < width`: `y_j = x_j` (the unrotated suffix).
//!
//! This is `x · cos + rotate_half(x) · sin` followed by the suffix
//! concatenation, element by element, with the tables read at `(b, 0, t, j)`.
//! Each product, the negation, and the sum are evaluated with exactly the
//! rounding of Flex's own elementwise kernels: native F32/F64 arithmetic, and
//! for F16/BF16 one F32 operation rounded back to the half type per step. The
//! result is therefore bit-identical to the broadcast formulation, and it is
//! written as contiguous `[batch, heads, sequence, width]` storage, the same
//! layout that formulation produces. The kernel reads every operand through
//! its strides, so it needs no copy of a strided view and no per-head tables:
//! one pass over the operand replaces several.

use burn::{
    backend::flex::FlexTensor,
    tensor::{DType, Element, Tensor, TensorData, backend::Backend, bf16, f16},
};

use crate::operations;

/// Element arithmetic of one Flex dtype, rounded exactly like Flex's elementwise kernels.
trait RotaryScalar: Element + Copy {
    /// Borrows the whole typed host buffer of a Flex tensor of this dtype.
    fn storage(tensor: &FlexTensor) -> &[Self];
    /// `a · b` as Flex's binary multiply computes it.
    fn mul(a: Self, b: Self) -> Self;
    /// `a + b` as Flex's binary add computes it.
    fn add(a: Self, b: Self) -> Self;
    /// `−a` as Flex's unary negation computes it.
    fn neg(a: Self) -> Self;
}

impl RotaryScalar for f32 {
    /// Borrows the buffer as `f32` values.
    fn storage(tensor: &FlexTensor) -> &[Self] {
        tensor.storage::<f32>()
    }
    /// Multiplies with native F32 arithmetic.
    fn mul(a: Self, b: Self) -> Self {
        a * b
    }
    /// Adds with native F32 arithmetic.
    fn add(a: Self, b: Self) -> Self {
        a + b
    }
    /// Negates with native F32 arithmetic (an exact sign flip for non-NaN values).
    fn neg(a: Self) -> Self {
        -a
    }
}

impl RotaryScalar for f64 {
    /// Borrows the buffer as `f64` values.
    fn storage(tensor: &FlexTensor) -> &[Self] {
        tensor.storage::<f64>()
    }
    /// Multiplies with native F64 arithmetic.
    fn mul(a: Self, b: Self) -> Self {
        a * b
    }
    /// Adds with native F64 arithmetic.
    fn add(a: Self, b: Self) -> Self {
        a + b
    }
    /// Negates with native F64 arithmetic (an exact sign flip for non-NaN values).
    fn neg(a: Self) -> Self {
        -a
    }
}

impl RotaryScalar for f16 {
    /// Borrows the buffer as `f16` values.
    fn storage(tensor: &FlexTensor) -> &[Self] {
        tensor.storage::<f16>()
    }
    /// Multiplies with one F32 operation rounded to F16.
    fn mul(a: Self, b: Self) -> Self {
        f16::from_f32(a.to_f32() * b.to_f32())
    }
    /// Adds with one F32 operation rounded to F16.
    fn add(a: Self, b: Self) -> Self {
        f16::from_f32(a.to_f32() + b.to_f32())
    }
    /// Negates with one F32 operation rounded to F16 (an exact sign flip for non-NaN values).
    fn neg(a: Self) -> Self {
        f16::from_f32(-a.to_f32())
    }
}

impl RotaryScalar for bf16 {
    /// Borrows the buffer as `bf16` values.
    fn storage(tensor: &FlexTensor) -> &[Self] {
        tensor.storage::<bf16>()
    }
    /// Multiplies with one F32 operation rounded to BF16.
    fn mul(a: Self, b: Self) -> Self {
        bf16::from_f32(a.to_f32() * b.to_f32())
    }
    /// Adds with one F32 operation rounded to BF16.
    fn add(a: Self, b: Self) -> Self {
        bf16::from_f32(a.to_f32() + b.to_f32())
    }
    /// Negates with one F32 operation rounded to BF16 (an exact sign flip for non-NaN values).
    fn neg(a: Self) -> Self {
        bf16::from_f32(-a.to_f32())
    }
}

/// A rank-four Flex tensor's typed buffer with its non-negative strides and offset.
struct StridedRows<'a, E> {
    /// The whole host buffer, indexed through `offset` and `strides`.
    storage: &'a [E],
    /// Element strides of `[batch, heads, sequence, channel]`.
    strides: [usize; 4],
    /// Element offset of index `[0, 0, 0, 0]`.
    offset: usize,
}

impl<'a, E: RotaryScalar> StridedRows<'a, E> {
    /// Views `tensor`, or `None` for a negative stride (a flipped view) or another rank.
    fn new(tensor: &'a FlexTensor) -> Option<Self> {
        let layout = tensor.layout();
        let signed: [isize; 4] = layout.strides().try_into().ok()?;
        let mut strides = [0_usize; 4];
        for (stride, signed) in strides.iter_mut().zip(signed) {
            *stride = usize::try_from(signed).ok()?;
        }
        Some(Self {
            storage: E::storage(tensor),
            strides,
            offset: layout.start_offset(),
        })
    }

    /// Storage index of channel zero of row `(batch, head, token)`.
    fn row_start(&self, batch: usize, head: usize, token: usize) -> usize {
        self.offset + batch * self.strides[0] + head * self.strides[1] + token * self.strides[2]
    }

    /// Channel `channel` of the row starting at `start`.
    fn at(&self, start: usize, channel: usize) -> E {
        self.storage[start + channel * self.strides[3]]
    }

    /// The row starting at `start` as a slice of `length` channels, when channels are adjacent.
    fn contiguous_row(&self, start: usize, length: usize) -> Option<&'a [E]> {
        (self.strides[3] == 1).then(|| &self.storage[start..start + length])
    }
}

/// Rotates every row of a Flex operand; returns contiguous `[batch, heads, sequence, width]` values.
///
/// `input` is `[batch, heads, sequence, width]` and both tables are
/// `[batch, 1, sequence, rotary]` in the input's dtype, as validated by
/// [`super::rotary_operand_fits`]; `rotary` is even. Returns `None` (so the
/// caller uses the broadcast formulation) for a negative stride.
fn rotate_rows<E: RotaryScalar>(
    input: &FlexTensor,
    cosine: &FlexTensor,
    sine: &FlexTensor,
    shape: [usize; 4],
    rotary: usize,
) -> Option<Vec<E>> {
    let [batch, heads, sequence, width] = shape;
    let half = rotary / 2;
    let input = StridedRows::<E>::new(input)?;
    let cosine = StridedRows::<E>::new(cosine)?;
    let sine = StridedRows::<E>::new(sine)?;
    let mut output = Vec::with_capacity(batch * heads * sequence * width);
    for b in 0..batch {
        for h in 0..heads {
            for t in 0..sequence {
                let row = input.row_start(b, h, t);
                let (cosine_row, sine_row) = (cosine.row_start(b, 0, t), sine.row_start(b, 0, t));
                if let (Some(x), Some(c), Some(s)) = (
                    input.contiguous_row(row, width),
                    cosine.contiguous_row(cosine_row, rotary),
                    sine.contiguous_row(sine_row, rotary),
                ) {
                    // Adjacent channels: zipped slices let the compiler vectorize.
                    let (real, imaginary) = x[..rotary].split_at(half);
                    let (cosine_real, cosine_imaginary) = c.split_at(half);
                    let (sine_real, sine_imaginary) = s.split_at(half);
                    output.extend(
                        real.iter()
                            .zip(imaginary)
                            .zip(cosine_real.iter().zip(sine_real))
                            .map(|((&r, &i), (&cv, &sv))| {
                                E::add(E::mul(r, cv), E::mul(E::neg(i), sv))
                            }),
                    );
                    output.extend(
                        imaginary
                            .iter()
                            .zip(real)
                            .zip(cosine_imaginary.iter().zip(sine_imaginary))
                            .map(|((&i, &r), (&cv, &sv))| E::add(E::mul(i, cv), E::mul(r, sv))),
                    );
                    output.extend_from_slice(&x[rotary..]);
                } else {
                    for j in 0..rotary {
                        let partner = if j < half {
                            E::neg(input.at(row, j + half))
                        } else {
                            input.at(row, j - half)
                        };
                        output.push(E::add(
                            E::mul(input.at(row, j), cosine.at(cosine_row, j)),
                            E::mul(partner, sine.at(sine_row, j)),
                        ));
                    }
                    output.extend((rotary..width).map(|j| input.at(row, j)));
                }
            }
        }
    }
    Some(output)
}

/// Applies split-half RoPE with the fused kernel when `B` is inference Flex.
///
/// Returns `None` on every other backend, for dtypes other than F32, F64,
/// F16, and BF16, for tables in another dtype than the input, for an odd
/// rotary width, and for flipped (negative-stride) views; the caller then uses
/// the broadcast formulation, whose values this kernel reproduces bit for bit.
pub(super) fn rotate_prefix<B: Backend>(
    input: &Tensor<B, 4>,
    cosine: &Tensor<B, 4>,
    sine: &Tensor<B, 4>,
    rotary: usize,
) -> Option<Tensor<B, 4>> {
    if !rotary.is_multiple_of(2) {
        return None;
    }
    let operand = operations::inference_flex_primitive(input)?;
    let cosine = operations::inference_flex_primitive(cosine)?;
    let sine = operations::inference_flex_primitive(sine)?;
    let dtype = operand.dtype();
    if cosine.dtype() != dtype || sine.dtype() != dtype {
        return None;
    }
    let shape = input.dims();
    let data = match dtype {
        DType::F32 => TensorData::new(
            rotate_rows::<f32>(&operand, &cosine, &sine, shape, rotary)?,
            shape,
        ),
        DType::F64 => TensorData::new(
            rotate_rows::<f64>(&operand, &cosine, &sine, shape, rotary)?,
            shape,
        ),
        DType::F16 => TensorData::new(
            rotate_rows::<f16>(&operand, &cosine, &sine, shape, rotary)?,
            shape,
        ),
        DType::BF16 => TensorData::new(
            rotate_rows::<bf16>(&operand, &cosine, &sine, shape, rotary)?,
            shape,
        ),
        _ => return None,
    };
    Some(Tensor::from_data(data, (&input.device(), dtype)))
}
