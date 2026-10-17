//! Cross-backend numerics shared by the accelerator test suites.
//!
//! An accelerator test asks one question: does a GPU backend evaluate the same
//! graph as deterministic Flex/F32, or as a pinned F32 reference? Answering it
//! needs six pieces, kept here so every suite answers it the same way:
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
//!   can measure its effect before any GPU is available;
//! * [`checkpoint_tolerance`] keeps the real-checkpoint gates' F32 default
//!   tolerances to F32 devices and requires an explicit one on a TF32 device;
//! * [`compare_teacher_forced_greedy`] compares a cached greedy continuation
//!   with the reference's at every step its tolerance decides
//!   ([`greedy_step_is_decisive`]), skipping ties rather than stopping at the
//!   first one (see [Greedy ties](#greedy-ties)). The real-checkpoint gates
//!   decide each step on the bundle's recorded reference margin
//!   ([`check_teacher_forced_greedy`]); the tiny goldens, which record only
//!   tokens, decide it on the run's own margin against
//!   [`Bound::allowed_for_actual_scale`].
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
//!
//! # Greedy ties
//!
//! A real-checkpoint bundle records the reference's own greedy continuation
//! as `(token, margin)` per step, the margin being the gap between its top
//! two logits. If every logit of a run lies within the gate's absolute
//! tolerance `τ` of the reference, the reference's best token `b` keeps
//! `ours[b] ≥ ref[b] − τ` and any other token `j` keeps `ours[j] ≤ ref[j] + τ`,
//! so a margin strictly above `2τ` forces the run to pick `b`. At or below
//! `2τ` the run may pick the runner-up without any error beyond `τ`: the step
//! is a tie at that tolerance and says nothing about correctness.
//!
//! The same bound holds with the roles exchanged, which is what a golden that
//! records only the reference's tokens needs. If the run picks `r` while the
//! reference picks `b ≠ r` (so `ref[b] ≥ ref[r]`), then
//! `ours[r] − ours[b] ≤ (ref[r] + τ) − (ref[b] − τ) ≤ 2τ`, so the run's own
//! top-2 margin is at most `2τ`. A run margin strictly above `2τ` therefore
//! also forces agreement, and a run margin in `(τ, 2τ]` does not: reference
//! `[1.00, 0.99]` and run `[0.98, 1.009]` differ by at most `0.02` everywhere,
//! yet the run's margin `0.029` exceeds `τ = 0.02` and its argmax differs.
//! When `τ` itself depends on the reference's scale, which such a golden does
//! not record either, [`Bound::allowed_for_actual_scale`] bounds it from the
//! run's scale.
//!
//! The gates feed the reference's token back at every step (teacher forcing),
//! so each step continues the reference's own prefix whatever the run chose
//! at a tie. A tie therefore invalidates nothing after it and is skipped.
//! Stopping at the first tie instead discards every later decisive step and
//! makes the minimum-count check depend on where the first close race falls.
//! The full BF16 gpt-oss-20b bundle opens with a margin of `3.0625`, a tie at
//! the half-precision tolerance `2.5`, yet five of its 32 margins exceed `5`.
//! Under TF32 tolerances, any `τ` of `0.1179` or more ends the Llama-3.1-8B
//! comparison at step 0 (margin `0.2356`), and any of `0.0232` or more ends
//! the SmolLM-135M one at step 1 (margin `0.0463`), below the default minimum
//! of two, although 27 of 32 and 21 of 32 of their steps stay decisive at
//! `τ = 0.16`. At the F32 defaults (`1e-3`; `5e-3` across Gemma's window)
//! every step of every pinned F32 or F64 Llama-family, Gemma, and gpt-oss
//! bundle is decisive, so both rules compare all of them and the recorded F32
//! results are unchanged.

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

/// Resolves a real-checkpoint gate's absolute tolerance from the environment variable `variable`.
///
/// See [`resolve_checkpoint_tolerance`].
pub(crate) fn checkpoint_tolerance(
    variable: &str,
    default: f32,
    precision: MatmulPrecision,
) -> f32 {
    resolve_checkpoint_tolerance(
        variable,
        std::env::var(variable).ok().as_deref(),
        default,
        precision,
    )
}

/// Resolves a real-checkpoint gate's absolute tolerance from an optional override.
///
/// The real-checkpoint gates measured their default tolerances with F32
/// arithmetic against F32 (or F64) references. An override (`value`, read
/// from `variable`) applies on any device. Without one, an F32 device gets
/// `default`; a device that rounds F32 matrix inputs to TF32 has no measured
/// default, so the call panics, naming `variable`, rather than holding TF32
/// arithmetic to an F32 tolerance it is not expected to meet or silently
/// loosening the gate. Panics when `value` is not a float.
pub(crate) fn resolve_checkpoint_tolerance(
    variable: &str,
    value: Option<&str>,
    default: f32,
    precision: MatmulPrecision,
) -> f32 {
    match (value, precision) {
        (Some(value), _) => value
            .parse()
            .unwrap_or_else(|_| panic!("{variable} must be a float, not {value:?}")),
        (None, MatmulPrecision::F32) => default,
        (None, MatmulPrecision::Tf32) => panic!(
            "this device rounds F32 matrix inputs to TF32 (a CUDA GPU of compute capability 8.0 \
             or newer), so the default tolerance {default:e}, measured with F32 arithmetic, does \
             not apply; set {variable} to the tolerance this TF32 run is held to (each test \
             prints its measured deviations) and record the result as TF32 evidence, not as F32 \
             qualification"
        ),
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
    /// can never pass a [`Bound`]. The scale counts finite reference values
    /// only: an infinite reference entry would otherwise make a relative
    /// bound's allowance infinite as well.
    pub(crate) fn measure(actual: &[f32], reference: &[f32]) -> Self {
        assert_eq!(actual.len(), reference.len(), "compared lengths differ");
        actual
            .iter()
            .zip(reference)
            .fold(Self::default(), |gap, (&a, &r)| {
                let finite = a.is_finite() && r.is_finite();
                Self {
                    deviation: gap.deviation.max(if finite {
                        (a - r).abs()
                    } else {
                        f32::INFINITY
                    }),
                    scale: if r.is_finite() {
                        gap.scale.max(r.abs())
                    } else {
                        gap.scale
                    },
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

    /// Returns the largest allowance any reference admitting `actual` can have, given `actual`'s scale.
    ///
    /// A check that sees only the run's values (a greedy step of a golden that
    /// records no continuation logits) cannot evaluate [`Self::allowed`] at
    /// the reference's scale `s`. If every entry is within the bound, then
    /// `s ≤ s_actual + absolute + relative · s`, so for `relative < 1`
    /// `s ≤ (s_actual + absolute) / (1 − relative)` and the allowance
    /// `absolute + relative · s` is at most
    /// `(absolute + relative · s_actual) / (1 − relative)`. That bound is
    /// computed in F64 and rounded up to F32, so it never understates the
    /// exact one. With `relative = 0` it equals `absolute`. Panics unless
    /// `0 ≤ relative < 1` and `actual_scale` is finite and non-negative.
    pub(crate) fn allowed_for_actual_scale(self, actual_scale: f32) -> f32 {
        assert!(
            (0.0..1.0).contains(&self.relative),
            "a relative bound of {:e} admits references of any scale",
            self.relative
        );
        assert!(
            actual_scale.is_finite() && actual_scale >= 0.0,
            "the actual scale must be finite and non-negative, not {actual_scale:e}"
        );
        let exact = (f64::from(self.absolute) + f64::from(self.relative) * f64::from(actual_scale))
            / (1.0 - f64::from(self.relative));
        let rounded = exact as f32;
        if f64::from(rounded) < exact {
            rounded.next_up()
        } else {
            rounded
        }
    }

    /// Whether `gap` lies within the bound; a non-finite deviation never does.
    pub(crate) fn admits(self, gap: Gap) -> bool {
        gap.deviation.is_finite() && gap.deviation <= self.allowed(gap.scale)
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

// ---------------------------------------------------------------------------
// Greedy continuations of real-checkpoint gates
// ---------------------------------------------------------------------------

/// Returns the index of the largest logit, preferring the lowest index among equal values.
///
/// Values are ordered by [`f32::total_cmp`], so the choice is deterministic
/// for every input, as in the reference generator's `top_two_margin`.
/// Panics on an empty slice.
pub(crate) fn argmax(logits: &[f32]) -> usize {
    (0..logits.len())
        .max_by(|&a, &b| logits[a].total_cmp(&logits[b]).then(b.cmp(&a)))
        .expect("argmax of an empty slice")
}

/// Whether a reference step with top-2 `margin` decides the argmax at absolute `tolerance`.
///
/// True exactly when `margin > 2 · tolerance`; see [Greedy ties](#greedy-ties)
/// for why that implies argmax agreement whenever every logit is within
/// `tolerance`. Doubling is exact in F32, and the bundle's margin (an exact
/// F64 difference of two F32 logits) rounds to F32 monotonically, so a
/// rounded margin exceeds `2 · tolerance` only when the exact one does.
pub(crate) fn greedy_step_is_decisive(margin: f32, tolerance: f32) -> bool {
    margin > 2.0 * tolerance
}

/// Returns the gap between the largest logit and the next largest.
///
/// The largest is [`argmax`]'s choice and the runner-up the largest of the
/// others under the same [`f32::total_cmp`] order, so equal maxima give a
/// margin of zero. A single logit has no runner-up and an infinite margin.
/// Panics on an empty slice.
pub(crate) fn top_two_margin(logits: &[f32]) -> f32 {
    let best = argmax(logits);
    (0..logits.len())
        .filter(|&index| index != best)
        .map(|index| logits[index])
        .max_by(f32::total_cmp)
        .map_or(f32::INFINITY, |runner_up| logits[best] - runner_up)
}

/// The margin and tolerance on which one greedy step is decided.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct GreedyDecision {
    /// Top-2 margin of the reference (real-checkpoint bundles) or of the run (tiny goldens).
    pub(crate) margin: f32,
    /// Absolute tolerance every logit of the run is held to at this step.
    pub(crate) tolerance: f32,
}

impl GreedyDecision {
    /// Whether the step decides the argmax: [`greedy_step_is_decisive`].
    pub(crate) fn decisive(self) -> bool {
        greedy_step_is_decisive(self.margin, self.tolerance)
    }
}

/// Counts of a teacher-forced greedy comparison.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct GreedyComparison {
    /// Decisive steps, each of which matched the reference token.
    pub(crate) compared: usize,
    /// Steps skipped because their margin is a tie at the tolerance.
    pub(crate) ties: usize,
}

/// Compares a cached greedy continuation with the reference's at every decisive step.
///
/// `tokens` holds the reference's continuation, `logits` the run's
/// last-position logits after its prefill, and `decode(token)` appends one
/// token to the run's cache and returns the next position's logits. After
/// every step but the last, the *reference's* token is decoded, so step `s`
/// always continues the reference's own `s`-token prefix. Every step's
/// logits must be finite. `decide(step, logits)` returns the step's
/// [`GreedyDecision`]; where it is decisive the run's [`argmax`] must equal
/// the reference token, and any other step is a tie and is skipped without
/// ending the comparison (see [Greedy ties](#greedy-ties)).
///
/// At least `minimum` steps, capped at the continuation's length, must be
/// decisive, so a run cannot pass vacuously; that panic ends with `remedy`.
/// Messages are prefixed with `context`. Returns the counts.
#[track_caller]
pub(crate) fn compare_teacher_forced_greedy(
    context: &str,
    tokens: &[i64],
    minimum: usize,
    remedy: &str,
    mut decide: impl FnMut(usize, &[f32]) -> GreedyDecision,
    mut logits: Vec<f32>,
    mut decode: impl FnMut(i64) -> Vec<f32>,
) -> GreedyComparison {
    let mut outcome = GreedyComparison::default();
    let mut smallest = f32::INFINITY;
    for (step, &expected) in tokens.iter().enumerate() {
        assert!(
            logits.iter().all(|value| value.is_finite()),
            "{context}: greedy step {step}: the run's logits are not all finite"
        );
        let decision = decide(step, &logits);
        if decision.decisive() {
            assert_eq!(
                argmax(&logits) as i64,
                expected,
                "{context}: greedy step {step} (margin {:e} above twice the tolerance {:e})",
                decision.margin,
                decision.tolerance
            );
            outcome.compared += 1;
            smallest = smallest.min(decision.margin);
        } else {
            eprintln!(
                "{context}: greedy step {step}: margin {:e} is a tie at tolerance {:e}; skipped",
                decision.margin, decision.tolerance
            );
            outcome.ties += 1;
        }
        if step + 1 < tokens.len() {
            logits = decode(expected);
        }
    }
    let required = minimum.min(tokens.len());
    eprintln!(
        "{context}: {} of {} greedy tokens decisive and matching (smallest decisive margin {:e}), \
         {} ties skipped",
        outcome.compared,
        tokens.len(),
        smallest,
        outcome.ties
    );
    assert!(
        outcome.compared >= required,
        "{context}: only {} decisive greedy tokens were compared (minimum {required}; {} ties); \
         {remedy}",
        outcome.compared,
        outcome.ties
    );
    outcome
}

/// Compares a real-checkpoint gate's cached greedy continuation with its bundle's.
///
/// `reference` holds each step's reference `(token, top-2 margin)`, and a
/// step is decisive when that recorded margin exceeds twice the absolute
/// `tolerance`; otherwise as [`compare_teacher_forced_greedy`]. A shortfall
/// below `minimum` names `minimum_variable`, the environment variable
/// lowering the requirement.
#[track_caller]
pub(crate) fn check_teacher_forced_greedy(
    context: &str,
    reference: &[(i64, f32)],
    tolerance: f32,
    minimum: usize,
    minimum_variable: &str,
    logits: Vec<f32>,
    decode: impl FnMut(i64) -> Vec<f32>,
) -> GreedyComparison {
    let tokens: Vec<i64> = reference.iter().map(|&(token, _)| token).collect();
    compare_teacher_forced_greedy(
        context,
        &tokens,
        minimum,
        &format!(
            "ties are judged at tolerance {tolerance:e}; regenerate the bundle with another \
             first prompt or lower {minimum_variable}"
        ),
        |step, _| GreedyDecision {
            margin: reference[step].1,
            tolerance,
        },
        logits,
        decode,
    )
}
