//! Family-declared numeric-range hazards and the load-time policy that checks them.
//!
//! # Range of a floating-point format
//!
//! A binary format with `e` exponent bits and `m` explicit mantissa bits holds
//! finite magnitudes up to `(2 - 2^-m) · 2^(2^(e-1) - 1)`. Under
//! round-to-nearest-even, an exact result at or above the midpoint between
//! that value and the next power of two rounds to infinity:
//!
//! | dtype | `e` | `m` | largest finite | rounds to infinity from |
//! | --- | --- | --- | --- | --- |
//! | F16 | 5 | 10 | `65,504` | `65,520` |
//! | BF16 | 8 | 7 | `≈ 3.39e38` | `≈ 3.40e38` |
//! | F32 | 8 | 23 | `≈ 3.40e38` | `≈ 3.40e38` |
//!
//! BF16 therefore trades F16's three extra mantissa bits for F32's exponent
//! range. Once one activation is infinite, normalization (`x / rms(x)` with an
//! infinite `rms`), attention softmax, and `inf - inf` turn it into NaN, which
//! then propagates through every later position that reads it.
//!
//! # The hazard
//!
//! This crate's family graphs carry their residual streams in the dtype of
//! the parameters that produce them: the embedding table sets the residual
//! stream's dtype, and normalization, which reduces in F32, rounds back to it
//! (only state that a contract pins to F32, such as recurrent buffers, is
//! exempt). The dtype the loader plans for the parameters is therefore the
//! activation dtype. Some published checkpoints are measured to leave a dtype's range
//! in their *reference* implementation too: the overflow is a property of the
//! weights, not of a kernel, so no faithful evaluation in that dtype is
//! finite. A family declares such a measurement as a [`RangeHazard`]: the
//! checkpoints it covers, the dtype, and the evidence (where the overflow
//! occurs, by how much, which sizes were measured, and the record).
//!
//! # The check
//!
//! A family declares its hazards on its [`super::MappingPlan`]
//! ([`super::MappingPlan::declare_range_hazards`]), the plan the shared
//! planner [`super::build_load_plan`] reads. The planner applies the dtype
//! policy and then runs [`check_range_hazards`] on the finished strict
//! [`LoadPlan`], so the check runs before any payload is read and before the
//! automatic loader allocates the graph, and every caller planning with the
//! family's mapping plan inherits it: the family's contract method, the
//! automatic loader, and a caller passing the contract's expected tensors
//! and mappings to the shared planner. It matches when any planned
//! parameter takes a hazard's dtype, whether [`super::DTypePolicy::Cast`]
//! selected it or the checkpoint stores it and [`super::DTypePolicy::Auto`]
//! kept it. Under [`RangeHazardPolicy::Refuse`], the default of
//! [`super::LoadOptions::range_hazards`], a match is
//! [`CheckpointError::RangeHazard`], whose message names the checkpoints, the
//! dtype, the measured evidence, the wider dtypes the loader can plan instead
//! ([`RangeHazard::alternatives`]), and the opt-in.
//! [`RangeHazardPolicy::Accept`] is that opt-in: the plan is returned
//! unchanged and the faithful graph runs.
//!
//! The check contains no model names; families decide which hazards they
//! declare. It does not cover a mapping plan that does not carry the
//! family's declarations (one built from scratch, from another plan's
//! operations, or by deserialization, which drops them), random
//! initialization, Burn records loaded outside the checkpoint planner, or
//! inputs that overflow a dtype for which no hazard was measured. A caller
//! with such a plan can run [`check_range_hazards`] itself. Refusing is a
//! default policy for experiments that cannot use the result, not a
//! numerical fix: nothing is clamped or widened.

use serde::{Deserialize, Serialize};

use super::{CheckpointError, LoadPlan, Result, TensorDType};

/// A dtype whose finite range a family's published checkpoints are measured to exceed.
///
/// Declarations are compile-time constants of the family that owns the
/// checkpoints, so every text field is `'static`. The type is plain data:
/// [`check_range_hazards`] decides what a match means under the caller's
/// [`RangeHazardPolicy`].
///
/// ```
/// use transformers::checkpoint::{RangeHazard, TensorDType};
///
/// const OVERFLOW: RangeHazard = RangeHazard::new(
///     "Example (model_type example)",
///     TensorDType::F16,
///     "measured: block 3's residual reaches 9.1e4 in the reference's F16 graph",
/// );
/// assert!(OVERFLOW.applies_to(TensorDType::F16));
/// assert!(!OVERFLOW.applies_to(TensorDType::BF16));
/// assert_eq!(OVERFLOW.alternatives(), [TensorDType::F32, TensorDType::BF16]);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RangeHazard {
    /// The checkpoint family the declaration covers, as named in errors.
    checkpoints: &'static str,
    /// The dtype whose range the checkpoints exceed.
    dtype: TensorDType,
    /// The measured overflow: where, by how much, which sizes, and the record.
    evidence: &'static str,
}

impl RangeHazard {
    /// Declares that `checkpoints` are measured to overflow `dtype`, as `evidence` records.
    ///
    /// `checkpoints` names the family and its `model_type` values (for
    /// example `"Gemma 3 (model_type gemma3 or gemma3_text)"`). `evidence`
    /// should separate measured checkpoints from unmeasured ones, because a
    /// family-wide declaration also refuses sizes nobody measured.
    #[must_use]
    pub const fn new(
        checkpoints: &'static str,
        dtype: TensorDType,
        evidence: &'static str,
    ) -> Self {
        Self {
            checkpoints,
            dtype,
            evidence,
        }
    }

    /// Returns the checkpoint family the declaration covers.
    #[must_use]
    pub const fn checkpoints(self) -> &'static str {
        self.checkpoints
    }

    /// Returns the dtype whose range the checkpoints exceed.
    #[must_use]
    pub const fn dtype(self) -> TensorDType {
        self.dtype
    }

    /// Returns the measured evidence quoted by the refusal.
    #[must_use]
    pub const fn evidence(self) -> &'static str {
        self.evidence
    }

    /// Returns whether a parameter planned in `dtype` executes in the hazard's dtype.
    #[must_use]
    pub const fn applies_to(self, dtype: TensorDType) -> bool {
        // Discriminant comparison: derived `==` is not callable in a `const fn`.
        self.dtype as u8 == dtype as u8
    }

    /// Returns the castable dtypes, F32 first, whose finite range exceeds the hazard's.
    ///
    /// The candidates are F32 and BF16, the execution dtypes of half- and
    /// full-precision checkpoint runs, and F64 only when F32 itself is not
    /// wider (an F32 hazard). An F16 hazard therefore offers F32 and BF16
    /// (BF16 keeps F32's exponent range); a BF16 hazard offers F32. A hazard
    /// on a dtype the loader cannot cast to or from (integers, FP8,
    /// microscaling formats) offers none, because no planned activation can
    /// take such a dtype.
    #[must_use]
    pub fn alternatives(self) -> Vec<TensorDType> {
        let Some(limit) = largest_finite(self.dtype) else {
            return Vec::new();
        };
        let wider = |dtype: &TensorDType| largest_finite(*dtype).is_some_and(|value| value > limit);
        let mut alternatives: Vec<TensorDType> = [TensorDType::F32, TensorDType::BF16]
            .into_iter()
            .filter(wider)
            .collect();
        if alternatives.is_empty() && wider(&TensorDType::F64) {
            alternatives.push(TensorDType::F64);
        }
        alternatives
    }
}

/// Whether a load may proceed when a declared [`RangeHazard`] matches the planned dtype.
///
/// Selected by [`super::LoadOptions::range_hazards`]. Families without
/// declared hazards ignore it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum RangeHazardPolicy {
    /// Fail strict planning with [`CheckpointError::RangeHazard`], before any payload is read.
    ///
    /// The error names the checkpoints, the dtype, the measured overflow,
    /// and the alternatives. This is the default, so a load that would
    /// return non-finite logits fails with a contextual error instead.
    #[default]
    Refuse,
    /// Load anyway and evaluate the faithful graph in the hazard's dtype.
    ///
    /// This is the explicit opt-in for experiments that need that graph:
    /// studying where and how the overflow happens, truncated stacks whose
    /// blocks stay in range, or interventions that keep activations finite.
    /// Its consequences are the reference's own: activations beyond the
    /// dtype's largest finite value become infinite, normalization and
    /// attention softmax then turn them into NaN at every position that reads
    /// them, so logits are expected to be non-finite, and generation, whose
    /// processors assume finite or `-inf` logits, has no defined result. Nothing is clamped, widened, or checked at run
    /// time, and the load report does not record the acceptance.
    Accept,
}

/// Refuses a plan that executes a declared hazard's dtype, unless `policy` accepts it.
///
/// Every planned (storage-owning) parameter is compared with every hazard,
/// in declaration order; aliases take their owner's dtype, so they add
/// nothing. The check reads only the plan. [`super::build_load_plan`] runs
/// it for the hazards its [`super::MappingPlan`] declares, under
/// [`super::LoadOptions::range_hazards`]; call it directly only for a plan
/// whose mapping plan does not carry the declarations, after strict planning
/// and before execution.
///
/// # Errors
///
/// [`CheckpointError::RangeHazard`] under [`RangeHazardPolicy::Refuse`] for
/// the first hazard whose dtype some planned parameter takes, naming the first
/// such parameter in canonical target order and how many there are.
pub fn check_range_hazards(
    plan: &LoadPlan,
    hazards: &[RangeHazard],
    policy: RangeHazardPolicy,
) -> Result<()> {
    match policy {
        RangeHazardPolicy::Accept => Ok(()),
        RangeHazardPolicy::Refuse => hazards.iter().try_for_each(|hazard| {
            let mut planned = plan
                .tensors
                .iter()
                .filter(|tensor| hazard.applies_to(tensor.dtype));
            match planned.next() {
                Some(first) => Err(CheckpointError::RangeHazard {
                    hazard: *hazard,
                    target: first.target.clone(),
                    parameters: 1 + planned.count(),
                }),
                None => Ok(()),
            }
        }),
    }
}

/// Returns the largest finite magnitude of a castable float, `(2 - 2^-m) · 2^(2^(e-1) - 1)`.
///
/// `None` for every dtype the loader cannot cast ([`TensorDType::is_castable_float`]).
fn largest_finite(dtype: TensorDType) -> Option<f64> {
    match dtype {
        TensorDType::F16 => Some(65_504.0),
        // (2 - 2^-7) · 2^127, BF16's bit pattern 0x7F7F.
        TensorDType::BF16 => Some(3.389_531_389_251_535_5e38),
        TensorDType::F32 => Some(f64::from(f32::MAX)),
        TensorDType::F64 => Some(f64::MAX),
        _ => None,
    }
}

/// Renders the message of [`CheckpointError::RangeHazard`].
///
/// It states the checkpoints and dtype, how many planned parameters take the
/// dtype (and the first), why that makes the activations overflow, the
/// declared evidence, the alternatives ([`alternatives_text`]), and the opt-in.
pub(super) fn refusal_message(hazard: &RangeHazard, target: &str, parameters: usize) -> String {
    let dtype = hazard.dtype();
    let planned = if parameters == 1 {
        "1 planned parameter".to_owned()
    } else {
        format!("{parameters} planned parameters")
    };
    format!(
        "{checkpoints} checkpoints are known to overflow {dtype}, so loading {planned} in {dtype} \
         (first `{target}`) is refused: activations take their parameters' dtype, and values \
         beyond {dtype}'s largest finite magnitude become infinite and then NaN. Evidence: \
         {evidence}. {alternatives} To evaluate the faithful {dtype} graph anyway, for example \
         to study the overflow, set `LoadOptions::range_hazards` to \
         `RangeHazardPolicy::Accept`; its logits are expected to be non-finite.",
        checkpoints = hazard.checkpoints(),
        evidence = hazard.evidence(),
        alternatives = alternatives_text(hazard),
    )
}

/// Describes the alternatives of `hazard` as one sentence.
///
/// Each wider dtype is named with the [`super::DTypePolicy`] that selects
/// it. BF16 carries the backend caveat of the pinned Burn 0.21, whose Metal
/// backend cannot store it. Whenever F32 is offered, packed Q4/Q8 weights
/// with F32 activations are offered too: they keep the activations in F32,
/// where a family's packed path is certified.
fn alternatives_text(hazard: &RangeHazard) -> String {
    let alternatives = hazard.alternatives();
    let mut options: Vec<String> = alternatives
        .iter()
        .map(|dtype| {
            let policy = format!("`DTypePolicy::Cast(TensorDType::{dtype})`");
            match dtype {
                TensorDType::BF16 => format!(
                    "BF16 ({policy}) on a backend that can store it (Burn 0.21's Metal backend \
                     cannot)"
                ),
                _ => format!("{dtype} ({policy})"),
            }
        })
        .collect();
    if alternatives.contains(&TensorDType::F32) {
        options.push(
            "packed Q4/Q8 weights with F32 activations (a quantized Burnpack sidecar over an \
             F32 plan, where the family's packed path is certified)"
                .to_owned(),
        );
    }
    match options.as_slice() {
        [] => format!(
            "No dtype the loader can cast to is wider than {}.",
            hazard.dtype()
        ),
        [only] => format!("Load the checkpoint in {only} instead."),
        [first @ .., last] => format!(
            "Load the checkpoint in {} instead, or with {last}.",
            first.join(", ")
        ),
    }
}

#[cfg(test)]
#[path = "hazard_tests.rs"]
mod tests;
