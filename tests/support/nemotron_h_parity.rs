//! Pure comparison contracts for Nemotron-H Burn/official parity.
//!
//! The pinned reference intentionally contains two numerical streams. A
//! stateless full prefix uses the official chunked Mamba scan, while cached
//! decoding uses its recurrent update. Those streams can retain the same top-1
//! token while differing substantially below it. These helpers therefore bind
//! every comparison to an execution path before evaluating full-vocabulary
//! cosine or exact argmax agreement.

#![allow(dead_code)]

use super::nemotron_h_base_reference::{
    OfficialExecutionPath, OfficialHiddenPoint, OfficialHiddenTrace, OfficialLogitRow,
};

/// Required full-vector cosine for every corresponding-path comparison.
pub(super) const MINIMUM_CORRESPONDING_PATH_COSINE: f64 = 0.999;

/// Result of one path-checked full-vocabulary comparison.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct RowParityObservation {
    /// F64-accumulated cosine over the complete vocabulary.
    pub(super) cosine: f64,
    /// Stable actual argmax, with lower token ID winning an exact tie.
    pub(super) actual_top1_token_id: usize,
}

/// Ordered aggregate for one complete official execution stream.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct CorrespondingPathSummary {
    /// Official numerical execution stream required for every observed row.
    path: OfficialExecutionPath,
    /// Number of contiguous generation decisions successfully compared.
    rows: usize,
    /// Smallest corresponding-row cosine, initially positive infinity.
    minimum_cosine: f64,
    /// Zero-based decision that first attained the smallest observed cosine.
    minimum_decision: usize,
}

impl CorrespondingPathSummary {
    /// Starts an empty, path-specific row sequence.
    pub(super) const fn new(path: OfficialExecutionPath) -> Self {
        Self {
            path,
            rows: 0,
            minimum_cosine: f64::INFINITY,
            minimum_decision: 0,
        }
    }

    /// Checks and records the next decision in strict contiguous order.
    pub(super) fn observe(
        &mut self,
        label: &str,
        actual: &[f32],
        reference: OfficialLogitRow<'_>,
    ) -> RowParityObservation {
        assert_eq!(
            reference.decision(),
            self.rows,
            "{label} must compare every decision once and in order"
        );
        let observation = assert_corresponding_logit_row(label, self.path, actual, reference);
        if observation.cosine < self.minimum_cosine {
            self.minimum_cosine = observation.cosine;
            self.minimum_decision = self.rows;
        }
        self.rows += 1;
        observation
    }

    /// Completes one stream only after its exact expected row count was seen.
    pub(super) fn finish(self, expected_rows: usize) -> CompletedPathSummary {
        assert!(expected_rows > 0, "a parity stream cannot be empty");
        assert_eq!(
            self.rows, expected_rows,
            "corresponding-path parity did not cover every expected row"
        );
        assert!(
            self.minimum_cosine.is_finite(),
            "completed parity must have a finite minimum"
        );
        CompletedPathSummary {
            path: self.path,
            rows: self.rows,
            minimum_cosine: self.minimum_cosine,
            minimum_decision: self.minimum_decision,
        }
    }
}

/// Immutable result for one fully covered execution stream.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct CompletedPathSummary {
    /// Official/Burn execution stream compared by the completed gate.
    pub(super) path: OfficialExecutionPath,
    /// Number of contiguous generation decisions checked.
    pub(super) rows: usize,
    /// Lowest corresponding-path full-vocabulary cosine.
    pub(super) minimum_cosine: f64,
    /// Decision at which the lowest cosine occurred.
    pub(super) minimum_decision: usize,
}

/// Minimum row-wise result for one complete residual-boundary trace.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct HiddenTraceParitySummary {
    /// Number of embedding/block/final-norm boundaries checked.
    pub(super) rows: usize,
    /// Lowest corresponding-boundary cosine.
    pub(super) minimum_cosine: f64,
    /// Boundary at which the lowest cosine occurred.
    pub(super) minimum_boundary: usize,
}

/// Requires one Burn row to match an official row from the same path.
pub(super) fn assert_corresponding_logit_row(
    label: &str,
    actual_path: OfficialExecutionPath,
    actual: &[f32],
    reference: OfficialLogitRow<'_>,
) -> RowParityObservation {
    assert_eq!(
        actual_path,
        reference.path(),
        "{label} crossed the stateless and cached reference streams"
    );
    assert_eq!(
        actual.len(),
        reference.values().len(),
        "{label} vector length changed"
    );
    assert!(!actual.is_empty(), "{label} vectors must not be empty");
    assert_finite(label, "Burn", actual);
    assert_finite(label, "official", reference.values());

    let expected_top1 = stable_top1(reference.values());
    assert_eq!(
        expected_top1,
        reference.top1_token_id() as usize,
        "{label} authenticated row disagrees with its top-1 contract"
    );
    let actual_top1 = stable_top1(actual);
    assert_eq!(
        actual_top1,
        reference.top1_token_id() as usize,
        "{label} top-1 differs at decision {}",
        reference.decision()
    );

    let cosine = cosine_similarity(actual, reference.values());
    assert!(
        cosine >= MINIMUM_CORRESPONDING_PATH_COSINE,
        "{label} corresponding-path cosine {cosine} is below {} at decision {}",
        MINIMUM_CORRESPONDING_PATH_COSINE,
        reference.decision()
    );
    RowParityObservation {
        cosine,
        actual_top1_token_id: actual_top1,
    }
}

/// Requires a flattened Burn hidden trace to match the same official point.
pub(super) fn assert_corresponding_hidden_trace(
    label: &str,
    actual_point: OfficialHiddenPoint,
    actual: &[f32],
    reference: OfficialHiddenTrace<'_>,
) -> HiddenTraceParitySummary {
    assert_eq!(
        actual_point,
        reference.point(),
        "{label} compared different hidden-trace phases"
    );
    assert_eq!(
        actual_point.path(),
        reference.path(),
        "{label} crossed the stateless and cached hidden streams"
    );
    assert!(reference.rows() > 0, "{label} hidden rows must be nonzero");
    assert!(
        reference.width() > 0,
        "{label} hidden width must be nonzero"
    );
    let expected_elements = reference
        .rows()
        .checked_mul(reference.width())
        .expect("hidden trace shape overflow");
    assert_eq!(
        reference.values().len(),
        expected_elements,
        "{label} official hidden-trace shape changed"
    );
    assert_eq!(
        actual.len(),
        expected_elements,
        "{label} Burn hidden-trace shape changed"
    );
    assert_finite(label, "Burn", actual);
    assert_finite(label, "official", reference.values());
    let mut minimum_cosine = f64::INFINITY;
    let mut minimum_boundary = 0;
    for boundary in 0..reference.rows() {
        let start = boundary * reference.width();
        let end = start + reference.width();
        let cosine = cosine_similarity(&actual[start..end], &reference.values()[start..end]);
        assert!(
            cosine >= MINIMUM_CORRESPONDING_PATH_COSINE,
            "{label} corresponding-boundary cosine {cosine} is below {} at boundary {boundary}",
            MINIMUM_CORRESPONDING_PATH_COSINE
        );
        if cosine < minimum_cosine {
            minimum_cosine = cosine;
            minimum_boundary = boundary;
        }
    }
    HiddenTraceParitySummary {
        rows: reference.rows(),
        minimum_cosine,
        minimum_boundary,
    }
}

/// Computes cosine with F64 products and sums after validating nonzero norms.
pub(super) fn cosine_similarity(left: &[f32], right: &[f32]) -> f64 {
    assert_eq!(left.len(), right.len(), "cosine vector lengths differ");
    assert!(!left.is_empty(), "cosine vectors must not be empty");
    let (dot, left_square, right_square) = left.iter().zip(right).fold(
        (0.0_f64, 0.0_f64, 0.0_f64),
        |(dot, left_square, right_square), (left, right)| {
            let left = f64::from(*left);
            let right = f64::from(*right);
            (
                dot + left * right,
                left_square + left * left,
                right_square + right * right,
            )
        },
    );
    assert!(
        left_square.is_finite() && left_square > 0.0,
        "left cosine norm must be finite and nonzero"
    );
    assert!(
        right_square.is_finite() && right_square > 0.0,
        "right cosine norm must be finite and nonzero"
    );
    let cosine = dot / (left_square.sqrt() * right_square.sqrt());
    assert!(cosine.is_finite(), "cosine must be finite");
    cosine
}

/// Returns a deterministic argmax with lower-index numeric tie breaking.
///
/// Callers validate finiteness first. Numeric comparison intentionally treats
/// positive and negative zero as equal, matching the reference runtime's
/// first-maximum behavior; [`f32::total_cmp`] would order their bit patterns.
fn stable_top1(values: &[f32]) -> usize {
    assert!(!values.is_empty(), "top-1 values must not be empty");
    let mut best = 0;
    for index in 1..values.len() {
        if values[index] > values[best] {
            best = index;
        }
    }
    best
}

/// Rejects NaN and infinities before they can produce misleading diagnostics.
fn assert_finite(label: &str, side: &str, values: &[f32]) {
    assert!(
        values.iter().all(|value| value.is_finite()),
        "{label} {side} values contain a non-finite scalar"
    );
}

#[cfg(test)]
mod tests {
    //! Synthetic tests for path identity, row ordering, and numerical parity gates.

    use std::panic::{AssertUnwindSafe, catch_unwind};

    use super::*;

    /// Constructs one synthetic official row without an external bundle.
    fn row(
        path: OfficialExecutionPath,
        decision: usize,
        values: &[f32],
        top1: u32,
    ) -> OfficialLogitRow<'_> {
        OfficialLogitRow::synthetic(path, decision, values, top1)
    }

    /// Exact and positive-scaled vectors satisfy both mathematical gates.
    #[test]
    fn corresponding_row_accepts_exact_and_scaled_values() {
        let reference_values = [1.0, 4.0, -2.0];
        for actual in [[1.0, 4.0, -2.0], [2.0, 8.0, -4.0]] {
            let observation = assert_corresponding_logit_row(
                "synthetic",
                OfficialExecutionPath::Stateless,
                &actual,
                row(OfficialExecutionPath::Stateless, 0, &reference_values, 1),
            );
            assert!((observation.cosine - 1.0).abs() <= 1.0e-12);
            assert_eq!(observation.actual_top1_token_id, 1);
        }
    }

    /// Cosine equality at the documented threshold is accepted inclusively.
    #[test]
    fn corresponding_row_accepts_exact_threshold() {
        let orthogonal = (1.0 - MINIMUM_CORRESPONDING_PATH_COSINE.powi(2)).sqrt() as f32;
        let actual = [MINIMUM_CORRESPONDING_PATH_COSINE as f32, orthogonal];
        let expected = [1.0, 0.0];
        let observation = assert_corresponding_logit_row(
            "threshold",
            OfficialExecutionPath::Cached,
            &actual,
            row(OfficialExecutionPath::Cached, 0, &expected, 0),
        );
        assert!(observation.cosine >= MINIMUM_CORRESPONDING_PATH_COSINE);
    }

    /// A crossed official stream fails before numerical similarity can pass.
    #[test]
    fn corresponding_row_rejects_crossed_paths() {
        let values = [1.0, 0.0];
        assert!(
            catch_unwind(|| {
                assert_corresponding_logit_row(
                    "crossed",
                    OfficialExecutionPath::Cached,
                    &values,
                    row(OfficialExecutionPath::Stateless, 0, &values, 0),
                );
            })
            .is_err()
        );
    }

    /// Shape, finiteness, nonzero-norm, cosine, and top-1 errors fail closed.
    #[test]
    fn corresponding_row_rejects_every_invalid_numerical_class() {
        let invalid = [
            catch_unwind(|| {
                assert_corresponding_logit_row(
                    "empty",
                    OfficialExecutionPath::Cached,
                    &[],
                    row(OfficialExecutionPath::Cached, 0, &[], 0),
                );
            }),
            catch_unwind(|| {
                assert_corresponding_logit_row(
                    "shape",
                    OfficialExecutionPath::Cached,
                    &[1.0],
                    row(OfficialExecutionPath::Cached, 0, &[1.0, 0.0], 0),
                );
            }),
            catch_unwind(|| {
                assert_corresponding_logit_row(
                    "actual nan",
                    OfficialExecutionPath::Cached,
                    &[f32::NAN, 1.0],
                    row(OfficialExecutionPath::Cached, 0, &[0.0, 1.0], 1),
                );
            }),
            catch_unwind(|| {
                assert_corresponding_logit_row(
                    "reference infinity",
                    OfficialExecutionPath::Cached,
                    &[0.0, 1.0],
                    row(OfficialExecutionPath::Cached, 0, &[0.0, f32::INFINITY], 1),
                );
            }),
            catch_unwind(|| {
                assert_corresponding_logit_row(
                    "zero norm",
                    OfficialExecutionPath::Cached,
                    &[0.0, 0.0],
                    row(OfficialExecutionPath::Cached, 0, &[1.0, 0.0], 0),
                );
            }),
            catch_unwind(|| {
                assert_corresponding_logit_row(
                    "wrong top1 contract",
                    OfficialExecutionPath::Cached,
                    &[1.0, 0.0],
                    row(OfficialExecutionPath::Cached, 0, &[1.0, 0.0], 1),
                );
            }),
            catch_unwind(|| {
                assert_corresponding_logit_row(
                    "wrong Burn top1",
                    OfficialExecutionPath::Cached,
                    &[0.9, 1.0],
                    row(OfficialExecutionPath::Cached, 0, &[1.0, 0.9], 0),
                );
            }),
            catch_unwind(|| {
                assert_corresponding_logit_row(
                    "low cosine",
                    OfficialExecutionPath::Cached,
                    &[1.0, 0.0],
                    row(OfficialExecutionPath::Cached, 0, &[1.0, -1.0], 0),
                );
            }),
        ];
        assert!(invalid.into_iter().all(|result| result.is_err()));
    }

    /// Stable argmax resolves exact ties to the lowest token ID.
    #[test]
    fn top1_ties_select_the_lowest_token_id() {
        assert_eq!(stable_top1(&[2.0, 2.0, 1.0]), 0);
        assert_eq!(stable_top1(&[-2.0, 3.0, 3.0]), 1);
        assert_eq!(stable_top1(&[-0.0, 0.0]), 0);
        assert_eq!(stable_top1(&[0.0, -0.0]), 0);
    }

    /// Ordered accumulation rejects gaps and reports the true minimum row.
    #[test]
    fn summary_enforces_contiguous_complete_coverage() {
        let first_expected = [1.0, 0.0];
        let second_expected = [1.0, 1.0];
        let mut summary = CorrespondingPathSummary::new(OfficialExecutionPath::Stateless);
        summary.observe(
            "row zero",
            &first_expected,
            row(OfficialExecutionPath::Stateless, 0, &first_expected, 0),
        );
        let second_actual = [1.0, 0.95];
        let second = summary.observe(
            "row one",
            &second_actual,
            row(OfficialExecutionPath::Stateless, 1, &second_expected, 0),
        );
        let completed = summary.finish(2);
        assert_eq!(completed.path, OfficialExecutionPath::Stateless);
        assert_eq!(completed.rows, 2);
        assert_eq!(completed.minimum_decision, 1);
        assert_eq!(completed.minimum_cosine, second.cosine);

        let gap = catch_unwind(|| {
            let mut summary = CorrespondingPathSummary::new(OfficialExecutionPath::Cached);
            summary.observe(
                "gap",
                &first_expected,
                row(OfficialExecutionPath::Cached, 1, &first_expected, 0),
            );
        });
        assert!(gap.is_err());

        let incomplete = catch_unwind(|| {
            CorrespondingPathSummary::new(OfficialExecutionPath::Cached).finish(1);
        });
        assert!(incomplete.is_err());

        let empty_contract = catch_unwind(|| {
            CorrespondingPathSummary::new(OfficialExecutionPath::Cached).finish(0);
        });
        assert!(empty_contract.is_err());
    }

    /// Hidden comparisons require the exact semantic point as well as path.
    #[test]
    fn hidden_trace_gate_is_phase_specific() {
        let actual = [1.0, 0.0, 1.0, 0.95];
        let expected = [1.0, 0.0, 1.0, 1.0];
        let summary = assert_corresponding_hidden_trace(
            "hidden",
            OfficialHiddenPoint::CachedPrefill,
            &actual,
            OfficialHiddenTrace::synthetic(OfficialHiddenPoint::CachedPrefill, 2, 2, &expected),
        );
        assert_eq!(summary.rows, 2);
        assert_eq!(summary.minimum_boundary, 1);
        assert!(summary.minimum_cosine < 1.0);
        assert!(
            catch_unwind(AssertUnwindSafe(|| {
                assert_corresponding_hidden_trace(
                    "wrong phase",
                    OfficialHiddenPoint::CachedDecodeOne,
                    &actual,
                    OfficialHiddenTrace::synthetic(
                        OfficialHiddenPoint::CachedPrefill,
                        2,
                        2,
                        &expected,
                    ),
                );
            }))
            .is_err()
        );
        assert!(
            catch_unwind(AssertUnwindSafe(|| {
                assert_corresponding_hidden_trace(
                    "wrong path",
                    OfficialHiddenPoint::StatelessPrompt,
                    &actual,
                    OfficialHiddenTrace::synthetic(
                        OfficialHiddenPoint::CachedPrefill,
                        2,
                        2,
                        &expected,
                    ),
                );
            }))
            .is_err()
        );
    }

    /// Hidden comparisons fail closed for every invalid numerical class.
    #[test]
    fn hidden_trace_gate_rejects_invalid_values() {
        let invalid = [
            catch_unwind(|| {
                assert_corresponding_hidden_trace(
                    "empty",
                    OfficialHiddenPoint::CachedPrefill,
                    &[],
                    OfficialHiddenTrace::synthetic(OfficialHiddenPoint::CachedPrefill, 0, 1, &[]),
                );
            }),
            catch_unwind(|| {
                assert_corresponding_hidden_trace(
                    "shape",
                    OfficialHiddenPoint::CachedPrefill,
                    &[1.0],
                    OfficialHiddenTrace::synthetic(
                        OfficialHiddenPoint::CachedPrefill,
                        1,
                        2,
                        &[1.0, 0.0],
                    ),
                );
            }),
            catch_unwind(|| {
                assert_corresponding_hidden_trace(
                    "actual infinity",
                    OfficialHiddenPoint::CachedPrefill,
                    &[f32::INFINITY, 1.0],
                    OfficialHiddenTrace::synthetic(
                        OfficialHiddenPoint::CachedPrefill,
                        1,
                        2,
                        &[0.0, 1.0],
                    ),
                );
            }),
            catch_unwind(|| {
                assert_corresponding_hidden_trace(
                    "official nan",
                    OfficialHiddenPoint::CachedPrefill,
                    &[0.0, 1.0],
                    OfficialHiddenTrace::synthetic(
                        OfficialHiddenPoint::CachedPrefill,
                        1,
                        2,
                        &[f32::NAN, 1.0],
                    ),
                );
            }),
            catch_unwind(|| {
                assert_corresponding_hidden_trace(
                    "official zero norm",
                    OfficialHiddenPoint::CachedPrefill,
                    &[1.0, 0.0],
                    OfficialHiddenTrace::synthetic(
                        OfficialHiddenPoint::CachedPrefill,
                        1,
                        2,
                        &[0.0, 0.0],
                    ),
                );
            }),
            catch_unwind(|| {
                assert_corresponding_hidden_trace(
                    "low cosine",
                    OfficialHiddenPoint::CachedPrefill,
                    &[1.0, 0.0],
                    OfficialHiddenTrace::synthetic(
                        OfficialHiddenPoint::CachedPrefill,
                        1,
                        2,
                        &[1.0, -1.0],
                    ),
                );
            }),
            catch_unwind(|| {
                assert_corresponding_hidden_trace(
                    "official shape",
                    OfficialHiddenPoint::CachedPrefill,
                    &[1.0, 0.0, 1.0, 0.0],
                    OfficialHiddenTrace::synthetic(
                        OfficialHiddenPoint::CachedPrefill,
                        2,
                        2,
                        &[1.0, 0.0],
                    ),
                );
            }),
            catch_unwind(|| {
                assert_corresponding_hidden_trace(
                    "shape overflow",
                    OfficialHiddenPoint::CachedPrefill,
                    &[],
                    OfficialHiddenTrace::synthetic(
                        OfficialHiddenPoint::CachedPrefill,
                        usize::MAX,
                        2,
                        &[],
                    ),
                );
            }),
        ];
        assert!(invalid.into_iter().all(|result| result.is_err()));
    }
}
