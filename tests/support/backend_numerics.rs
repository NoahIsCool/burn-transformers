//! Cross-backend numerics shared by the accelerator test suites.
//!
//! An accelerator test asks one question: does a GPU backend evaluate the same
//! graph as deterministic Flex/F32, or as a pinned F32 reference? Answering it
//! needs four pieces, kept here so every suite answers it the same way:
//!
//! * [`copy_parameters`] installs one model's parameters into an independently
//!   constructed model of the same configuration on another backend, so both
//!   sides hold bit-identical weights without a checkpoint on disk;
//! * [`Gap`] measures the largest absolute deviation of a compared set together
//!   with the largest reference magnitude, and [`Bound`] decides it relative to
//!   that scale;
//! * [`MatmulPrecision::probe`] detects whether a device multiplies F32
//!   matrices in F32 or with TF32 inputs, and selects the matching bound;
//! * [`round_to_tf32`] emulates TF32 input rounding on the host, so a CPU test
//!   can measure its effect before any GPU is available.
//!
//! # Why CUDA "F32" is not always F32 for matrix products
//!
//! Burn 0.21 launches matrix products through `cubek-matmul`'s automatic
//! strategy, which first tries the tensor-core (`cmma`) routine. Its
//! `adjust_dtypes` stages F32 operands as TF32 whenever the device registers
//! the TF32 type, and `cubecl-cuda` registers it on every CUDA device, while
//! `cubecl-cpp` offers the TF32 tensor-core shape (`m16n16k8`) only from
//! compute capability 8.0. So on an A100, L40S, or H100 every F32 product
//! rounds both operands to TF32 (10 explicit mantissa bits, round to nearest
//! with ties away from zero through `__float_to_tf32`) and accumulates in
//! F32. On older GPUs such as the V100 the tensor-core launch reports
//! `Unavailable` and the automatic strategy falls back to an F32 kernel.
//! Metal (`cubecl-wgpu`) never registers TF32 and multiplies in F32. Burn
//! offers no switch to turn TF32 off; [`MatmulPrecision::probe`] therefore
//! measures what the device does instead of assuming it.
//!
//! TF32's unit roundoff is `2^-11 ≈ 4.9e-4`, about eight thousand times F32's
//! `2^-24 ≈ 6.0e-8`, so a TF32 device is judged by [`TF32_BOUND`], derived from
//! TF32 rounding, and never by a bound derived from F32 accumulation order.
//!
//! # The TF32 bound
//!
//! Rounding both operands of a product to TF32 perturbs each term by a
//! relative amount up to `2 · 2^-11 ≈ 9.8e-4`; in a dot product of terms with
//! mixed signs those errors partly cancel, so an output moves by about that
//! fraction of its own scale. `tests/backend_parity.rs` measures the weight
//! half of the effect on Flex for every executable family by rounding only the
//! parameters ([`round_to_tf32`]): the logits move by `7.6e-5` (GPT-NeoX) to
//! `2.5e-3` (OLMo 2) of their scale, typically `3e-4`. Rounding
//! the activations at every product's other operand, including both operands
//! of the attention products, is a comparable perturbation, so the full TF32
//! effect is expected near twice the weight effect: at most about `5e-3` of
//! the scale. [`TF32_BOUND`] admits `1e-2` absolute plus `1e-2` of the scale,
//! at least four times the weight effect (asserted by
//! `tf32_weight_rounding_stays_within_a_quarter_of_the_tf32_bound`), which
//! leaves twice the expected full deviation as headroom. A logit-moving kernel
//! fault (a wrong mask, cache slot, ring offset, or routing decision) still
//! fails it, but subtler F32 discrepancies need an F32 device: Flex and Metal
//! qualify F32 numerics, and a TF32 run qualifies execution. The bound is a
//! derivation, not yet a CUDA measurement; the first Ampere-or-newer run
//! records the measured deviations.

#![allow(
    dead_code,
    reason = "each integration-test crate includes this module and uses a subset"
)]

use std::collections::VecDeque;

use burn::{
    module::{Module, ModuleMapper, ModuleVisitor, Param},
    tensor::{Bool, Int, Tensor, TensorData, backend::Backend},
};

/// Elementwise transformation applied to every copied floating parameter value.
pub(crate) type ValueTransform = fn(f32) -> f32;

/// Bound for a device whose F32 matrix products round their inputs to TF32.
///
/// See the module documentation for its derivation.
pub(crate) const TF32_BOUND: Bound = Bound {
    absolute: 1e-2,
    relative: 1e-2,
};

/// Leaves a value unchanged: an exact cross-backend copy.
pub(crate) fn identity(value: f32) -> f32 {
    value
}

/// Rounds an F32 value to TF32 as CUDA's `__float_to_tf32` does.
///
/// TF32 keeps F32's sign and 8-bit exponent and the upper 10 of its 23
/// mantissa bits. Rounding is to nearest with ties away from zero, which in
/// the sign-magnitude bit pattern means adding half of the dropped range
/// (`0x1000`) before clearing the 13 dropped bits; a carry into the exponent
/// is exactly the rounding up to the next binade. Non-finite values are
/// returned unchanged, so infinities stay infinite and NaN stays NaN; the
/// largest finite values round to infinity, as on the device.
pub(crate) fn round_to_tf32(value: f32) -> f32 {
    if !value.is_finite() {
        return value;
    }
    f32::from_bits((value.to_bits() + 0x1000) & !0x1FFF)
}

/// How a device's F32 matrix products treat their inputs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MatmulPrecision {
    /// Inputs keep all 24 significand bits: F32 products, as on Flex and Metal.
    F32,
    /// Inputs are rounded to TF32's 11 significand bits; products accumulate in F32.
    Tf32,
}

/// Rows, inner width, and columns of the probe product: a tensor-core-sized square.
const PROBE_WIDTH: usize = 32;

/// `1 + 2^-12`: exact in F32, a quarter TF32 unit in the last place above one.
const PROBE_FINE: f32 = 1.0 + 1.0 / 4096.0;

/// `1 + 2^-10`: exact in TF32 (its last mantissa bit), lost by BF16 or coarser inputs.
const PROBE_COARSE: f32 = 1.0 + 1.0 / 1024.0;

impl MatmulPrecision {
    /// Measures how `device` multiplies F32 matrices.
    ///
    /// Two `[32, 32]` products of a constant matrix with a matrix of ones
    /// return `32 · c` in every entry when the inputs keep their bits. Every
    /// partial sum `k · c` (`k ≤ 32`) is exact in F32, so accumulation order
    /// cannot disturb the result; only input rounding can. With `c = 1 + 2^-12`
    /// TF32 inputs give exactly `32`; with `c = 1 + 2^-10` they stay exact.
    pub(crate) fn probe<B: Backend>(device: &B::Device) -> Self {
        let ones = Tensor::<B, 2>::ones([PROBE_WIDTH; 2], device);
        let product = |value: f32| {
            host(Tensor::<B, 2>::full([PROBE_WIDTH; 2], value, device).matmul(ones.clone()))
        };
        Self::classify(&product(PROBE_FINE), &product(PROBE_COARSE))
    }

    /// Classifies the two probe products of [`Self::probe`].
    ///
    /// Panics when neither precision explains them (for example BF16 inputs,
    /// which lose `2^-10`), because no bound here would then be justified.
    pub(crate) fn classify(fine: &[f32], coarse: &[f32]) -> Self {
        let width = PROBE_WIDTH as f32;
        let all = |values: &[f32], expected: f32| values.iter().all(|&value| value == expected);
        if all(fine, width * PROBE_FINE) && all(coarse, width * PROBE_COARSE) {
            Self::F32
        } else if all(fine, width) && all(coarse, width * PROBE_COARSE) {
            Self::Tf32
        } else {
            panic!(
                "F32 matrix products match neither F32 nor TF32 inputs: \
                 32 · (1 + 2^-12) gave {:?}, 32 · (1 + 2^-10) gave {:?}",
                fine.first(),
                coarse.first()
            )
        }
    }

    /// Returns `f32` for an F32 device and [`TF32_BOUND`] for a TF32 one.
    pub(crate) fn bound(self, f32: Bound) -> Bound {
        match self {
            Self::F32 => f32,
            Self::Tf32 => TF32_BOUND,
        }
    }
}

/// Host copies of every parameter of a module, in Burn's traversal order.
#[derive(Default)]
struct ParameterCollector {
    /// Floating parameter values.
    floats: Vec<TensorData>,
    /// Integer parameter values.
    ints: Vec<TensorData>,
    /// Boolean parameter values.
    bools: Vec<TensorData>,
}

impl<B: Backend> ModuleVisitor<B> for ParameterCollector {
    /// Materializes and reads one floating parameter to the host.
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        self.floats.push(param.val().into_data());
    }

    /// Materializes and reads one integer parameter to the host.
    fn visit_int<const D: usize>(&mut self, param: &Param<Tensor<B, D, Int>>) {
        self.ints.push(param.val().into_data());
    }

    /// Materializes and reads one Boolean parameter to the host.
    fn visit_bool<const D: usize>(&mut self, param: &Param<Tensor<B, D, Bool>>) {
        self.bools.push(param.val().into_data());
    }
}

/// Replaces every parameter of a module with the collected values, in traversal order.
struct ParameterInstaller {
    /// Floating values still to install.
    floats: VecDeque<TensorData>,
    /// Integer values still to install.
    ints: VecDeque<TensorData>,
    /// Boolean values still to install.
    bools: VecDeque<TensorData>,
    /// Transformation applied to every floating value before upload.
    transform: ValueTransform,
    /// Number of parameters installed so far, for error messages.
    installed: usize,
}

impl ParameterInstaller {
    /// Takes the next collected value and checks it has the target's shape.
    fn next(
        queue: &mut VecDeque<TensorData>,
        kind: &str,
        index: usize,
        shape: &[usize],
    ) -> TensorData {
        let data = queue.pop_front().unwrap_or_else(|| {
            panic!("the source has fewer {kind} parameters than the target (parameter {index})")
        });
        assert_eq!(
            data.shape.to_vec(),
            shape,
            "{kind} parameter {index}: source and target shapes differ"
        );
        data
    }
}

impl<B: Backend> ModuleMapper<B> for ParameterInstaller {
    /// Uploads the next source value, transformed, in the target parameter's dtype.
    ///
    /// The parameter keeps its identity and load mapper; only its value changes.
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let (id, tensor, mapper) = param.consume();
        let index = self.installed;
        self.installed += 1;
        let data = Self::next(&mut self.floats, "float", index, &tensor.dims());
        let values = data
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("F32 host values")
            .into_iter()
            .map(self.transform)
            .collect::<Vec<_>>();
        let data = TensorData::new(values, tensor.dims()).convert_dtype(tensor.dtype());
        Param::from_mapped_value(id, Tensor::from_data(data, &tensor.device()), mapper)
    }

    /// Uploads the next integer source value unchanged.
    fn map_int<const D: usize>(
        &mut self,
        param: Param<Tensor<B, D, Int>>,
    ) -> Param<Tensor<B, D, Int>> {
        let (id, tensor, mapper) = param.consume();
        let index = self.installed;
        self.installed += 1;
        let data = Self::next(&mut self.ints, "int", index, &tensor.dims());
        Param::from_mapped_value(id, Tensor::from_data(data, &tensor.device()), mapper)
    }

    /// Uploads the next Boolean source value unchanged.
    fn map_bool<const D: usize>(
        &mut self,
        param: Param<Tensor<B, D, Bool>>,
    ) -> Param<Tensor<B, D, Bool>> {
        let (id, tensor, mapper) = param.consume();
        let index = self.installed;
        self.installed += 1;
        let data = Self::next(&mut self.bools, "bool", index, &tensor.dims());
        Param::from_mapped_value(id, Tensor::from_data(data, &tensor.device()), mapper)
    }
}

/// Installs `source`'s parameters, transformed by `transform`, into `target`.
///
/// `target` must be constructed from the same configuration as `source`, on
/// any backend and device: Burn visits and maps a module's parameters in the
/// same field order, so the `n`-th parameter of one is the `n`-th of the
/// other, and every pair must agree in kind and shape. Values cross as host
/// [`TensorData`] in logical row-major order, so a backend that stores a
/// projection in another physical layout still receives identical values.
/// Tied parameters (one identity reached through several fields) are copied
/// once per field, with equal values. Panics when the two modules' parameter
/// lists differ in length, kind, or shape.
pub(crate) fn copy_parameters<S, T, MS, MT>(
    source: &MS,
    target: MT,
    transform: ValueTransform,
) -> MT
where
    S: Backend,
    T: Backend,
    MS: Module<S>,
    MT: Module<T>,
{
    let mut collector = ParameterCollector::default();
    source.visit(&mut collector);
    assert!(
        !collector.floats.is_empty(),
        "the source has no floating parameters"
    );
    let mut installer = ParameterInstaller {
        floats: collector.floats.into(),
        ints: collector.ints.into(),
        bools: collector.bools.into(),
        transform,
        installed: 0,
    };
    let target = target.map(&mut installer);
    assert!(
        installer.floats.is_empty() && installer.ints.is_empty() && installer.bools.is_empty(),
        "the source has more parameters than the target"
    );
    target
}

/// Reads any floating tensor to the host as F32 values in row-major order.
pub(crate) fn host<B: Backend, const D: usize>(tensor: Tensor<B, D>) -> Vec<f32> {
    tensor
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .expect("F32 host values")
}

/// The largest deviation of a compared set and the reference scale it is judged against.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Gap {
    /// Largest `|actual - reference|` over the compared values.
    pub(crate) deviation: f32,
    /// Largest `|reference|` over the compared values.
    pub(crate) scale: f32,
}

impl Gap {
    /// Measures `actual` against `reference`, which must have equal lengths.
    ///
    /// A non-finite value on either side yields an infinite deviation, so it
    /// can never pass a [`Bound`].
    pub(crate) fn measure(actual: &[f32], reference: &[f32]) -> Self {
        assert_eq!(actual.len(), reference.len(), "compared lengths differ");
        actual
            .iter()
            .zip(reference)
            .fold(Self::default(), |gap, (&a, &r)| {
                let deviation = if a.is_finite() && r.is_finite() {
                    (a - r).abs()
                } else {
                    f32::INFINITY
                };
                Self {
                    deviation: gap.deviation.max(deviation),
                    scale: gap.scale.max(r.abs()),
                }
            })
    }

    /// Combines two measurements into the gap of their union.
    pub(crate) fn max(self, other: Self) -> Self {
        Self {
            deviation: self.deviation.max(other.deviation),
            scale: self.scale.max(other.scale),
        }
    }

    /// The deviation relative to the reference scale (absolute below a scale of one).
    pub(crate) fn relative(self) -> f32 {
        self.deviation / self.scale.max(1.0)
    }
}

/// An elementwise acceptance bound `|actual - reference| <= absolute + relative · scale`.
///
/// `scale` is the largest reference magnitude of the compared set, so the
/// bound tracks logits of any size without letting near-zero entries demand
/// relative precision that F32 cannot give them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Bound {
    /// Absolute floor, the whole bound for references of magnitude below one.
    pub(crate) absolute: f32,
    /// Fraction of the reference scale added to the floor.
    pub(crate) relative: f32,
}

impl Bound {
    /// Returns the allowed deviation for a reference of largest magnitude `scale`.
    pub(crate) fn allowed(self, scale: f32) -> f32 {
        self.absolute + self.relative * scale
    }

    /// Whether `gap` lies within the bound.
    pub(crate) fn admits(self, gap: Gap) -> bool {
        gap.deviation <= self.allowed(gap.scale)
    }

    /// Panics with `context` when `gap` exceeds the bound.
    #[track_caller]
    pub(crate) fn check(self, gap: Gap, context: &str) {
        assert!(
            self.admits(gap),
            "{context}: max |Δ| = {:e} exceeds {:e} (reference scale {:e})",
            gap.deviation,
            self.allowed(gap.scale),
            gap.scale
        );
    }

    /// Measures `actual` against `reference`, checks the bound, and returns the gap.
    #[track_caller]
    pub(crate) fn assert_close(self, actual: &[f32], reference: &[f32], context: &str) -> Gap {
        let gap = Gap::measure(actual, reference);
        self.check(gap, context);
        gap
    }
}
