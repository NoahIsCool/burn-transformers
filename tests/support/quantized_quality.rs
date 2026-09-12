//! Fail-closed fixed-sequence quality accounting for quantized model gates.
//!
//! Autoregressive quality must compare corresponding execution paths. Given an
//! authenticated matrix of reference logits, this module consumes quantized
//! rows in the same order and at the same absolute positions. For target token
//! \(y\) and logits \(z\), its per-token negative log-likelihood is
//!
//! \[
//! -\log p(y) = \log \sum_j \exp(z_j) - z_y.
//! \]
//!
//! The implementation subtracts the row maximum before exponentiation and uses
//! `f64` accumulation. It also reports full-vector cosine and exact top-one
//! agreement, but those diagnostics are deliberately separate from the NLL
//! regression policy.

use std::fmt;

/// Identifies which logit stream violated a numerical invariant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LogitStream {
    /// Authenticated full-precision reference rows.
    Official,
    /// Rows produced by the quantized runtime under evaluation.
    Quantized,
}

impl fmt::Display for LogitStream {
    /// Formats a stable human-readable stream name for acceptance diagnostics.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Official => formatter.write_str("official"),
            Self::Quantized => formatter.write_str("quantized"),
        }
    }
}

/// A structural, numerical, ordering, or policy failure in a quality gate.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum QualityError {
    /// A vocabulary cannot contain zero entries.
    ZeroVocabulary,
    /// At least one prediction/target pair is required.
    EmptySequence,
    /// Expected matrix geometry overflowed `usize`.
    ShapeOverflow,
    /// The reference matrix does not contain exactly one row per target.
    ReferenceShape {
        /// Exact element count implied by rows and vocabulary width.
        expected_elements: usize,
        /// Actual number of supplied elements.
        actual_elements: usize,
    },
    /// The absolute-position vector has a different number of rows.
    PositionCount {
        /// Number of target rows.
        expected: usize,
        /// Number of supplied positions.
        actual: usize,
    },
    /// Reference positions are not a contiguous increasing sequence.
    NonContiguousPosition {
        /// Row containing the unexpected position.
        row: usize,
        /// Position required by its predecessor.
        expected: usize,
        /// Position supplied by the contract.
        actual: usize,
    },
    /// Incrementing a reference position would overflow.
    PositionOverflow {
        /// Row whose successor cannot be represented.
        row: usize,
    },
    /// A target token lies outside the declared vocabulary.
    TargetOutOfRange {
        /// Row containing the invalid token.
        row: usize,
        /// Supplied target token.
        token: u32,
        /// Exclusive upper token bound.
        vocabulary_size: usize,
    },
    /// The authenticated reference row does not select its committed target.
    ReferenceTop1Mismatch {
        /// Affected row.
        row: usize,
        /// Committed target token.
        target: u32,
        /// Deterministic low-ID-tie-broken reference top one.
        top1: usize,
    },
    /// A row has the wrong vocabulary width.
    RowWidth {
        /// Stream containing the row.
        stream: LogitStream,
        /// Affected row.
        row: usize,
        /// Expected vocabulary width.
        expected: usize,
        /// Actual row width.
        actual: usize,
    },
    /// A logit is NaN or infinite.
    NonFiniteLogit {
        /// Stream containing the value.
        stream: LogitStream,
        /// Affected row.
        row: usize,
        /// Vocabulary index containing the value.
        column: usize,
    },
    /// Cosine is undefined because a complete row has zero magnitude.
    ZeroNorm {
        /// Stream containing the zero vector.
        stream: LogitStream,
        /// Affected row.
        row: usize,
    },
    /// More quantized rows were supplied than the contract contains.
    TooManyRows {
        /// Number of rows in the immutable contract.
        expected: usize,
    },
    /// A runtime row arrived at the wrong absolute position.
    PositionMismatch {
        /// Affected row.
        row: usize,
        /// Contract position.
        expected: usize,
        /// Runtime position.
        actual: usize,
    },
    /// A runtime row was scored against a different token than the contract.
    TargetMismatch {
        /// Affected row.
        row: usize,
        /// Contract token.
        expected: u32,
        /// Runtime token.
        actual: u32,
    },
    /// Finalization occurred before every contract row was consumed.
    Incomplete {
        /// Number of rows in the immutable contract.
        expected: usize,
        /// Number of rows observed from the runtime.
        actual: usize,
    },
    /// The authenticated reference mean cannot define a relative regression.
    ZeroReferenceBaseline,
    /// A regression limit must be finite and nonnegative.
    InvalidRegressionLimit {
        /// Rejected limit.
        limit: f64,
    },
    /// A caller supplied a non-finite measured regression.
    NonFiniteRegression {
        /// Rejected regression value.
        regression: f64,
    },
    /// Quantization exceeded the release policy's relative NLL budget.
    RegressionExceeded {
        /// Measured relative change `(quantized - reference) / reference`.
        regression: f64,
        /// Inclusive maximum permitted change.
        limit: f64,
    },
}

impl fmt::Display for QualityError {
    /// Formats contextual diagnostics without exposing unstable implementation detail.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for QualityError {}

/// Completed fixed-sequence quality measurements.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct QualitySummary {
    /// Number of full-vocabulary rows compared.
    pub(crate) rows: usize,
    /// Mean NLL of the authenticated official cached rows.
    pub(crate) official_mean_nll: f64,
    /// Mean NLL of the quantized cached rows.
    pub(crate) quantized_mean_nll: f64,
    /// Relative NLL change from official to quantized execution.
    pub(crate) relative_nll_regression: f64,
    /// Rows whose deterministic top one exactly matched the official row.
    pub(crate) exact_top1_matches: usize,
    /// Minimum full-vector cosine across corresponding cached rows.
    pub(crate) minimum_full_vector_cosine: f64,
}

/// Small compensated sum used to prevent row-order-dependent mean drift.
#[derive(Debug, Clone, Copy, Default)]
struct CompensatedSum {
    /// Running F64 total after applying each compensated increment.
    sum: f64,
    /// Rounding error subtracted from the next increment in Kahan summation.
    correction: f64,
}

impl CompensatedSum {
    /// Adds one finite value using Kahan compensation.
    fn add(&mut self, value: f64) {
        let corrected = value - self.correction;
        let next = self.sum + corrected;
        self.correction = (next - self.sum) - corrected;
        self.sum = next;
    }
}

/// Stateful row-order guard and quality accumulator.
pub(crate) struct FixedSequenceQuality<'a> {
    /// Authenticated row-major `[decision, vocabulary]` reference logits.
    official_logits: &'a [f32],
    /// Fixed official top-one token ID scored at each decision.
    target_token_ids: &'a [u32],
    /// Contiguous absolute prediction positions aligned with the reference rows.
    row_position_ids: &'a [usize],
    /// Number of token logits in each reference and observed row.
    vocabulary_size: usize,
    /// Index of the next reference row that `observe` must consume.
    next_row: usize,
    /// Compensated total of official target-token negative log likelihoods.
    official_nll: CompensatedSum,
    /// Compensated total of observed quantized target-token negative log likelihoods.
    quantized_nll: CompensatedSum,
    /// Observed rows whose deterministic argmax matches the official target.
    exact_top1_matches: usize,
    /// Smallest full-vocabulary cosine across successfully observed row pairs.
    minimum_full_vector_cosine: f64,
}

impl<'a> FixedSequenceQuality<'a> {
    /// Validates an immutable reference matrix before any expensive model load.
    ///
    /// The committed target must be the deterministic top one of every official
    /// row. This makes a shifted row, token, or sequence contract fail before it
    /// can produce a plausible but meaningless quality number.
    pub(crate) fn new(
        official_logits: &'a [f32],
        target_token_ids: &'a [u32],
        row_position_ids: &'a [usize],
        vocabulary_size: usize,
    ) -> Result<Self, QualityError> {
        if vocabulary_size == 0 {
            return Err(QualityError::ZeroVocabulary);
        }
        if target_token_ids.is_empty() {
            return Err(QualityError::EmptySequence);
        }
        let expected_elements = target_token_ids
            .len()
            .checked_mul(vocabulary_size)
            .ok_or(QualityError::ShapeOverflow)?;
        if official_logits.len() != expected_elements {
            return Err(QualityError::ReferenceShape {
                expected_elements,
                actual_elements: official_logits.len(),
            });
        }
        if row_position_ids.len() != target_token_ids.len() {
            return Err(QualityError::PositionCount {
                expected: target_token_ids.len(),
                actual: row_position_ids.len(),
            });
        }
        for row in 1..row_position_ids.len() {
            let expected = row_position_ids[row - 1]
                .checked_add(1)
                .ok_or(QualityError::PositionOverflow { row: row - 1 })?;
            let actual = row_position_ids[row];
            if actual != expected {
                return Err(QualityError::NonContiguousPosition {
                    row,
                    expected,
                    actual,
                });
            }
        }

        let mut official_nll = CompensatedSum::default();
        for (row, (&target, logits)) in target_token_ids
            .iter()
            .zip(official_logits.chunks_exact(vocabulary_size))
            .enumerate()
        {
            let target_index =
                usize::try_from(target).map_err(|_| QualityError::TargetOutOfRange {
                    row,
                    token: target,
                    vocabulary_size,
                })?;
            if target_index >= vocabulary_size {
                return Err(QualityError::TargetOutOfRange {
                    row,
                    token: target,
                    vocabulary_size,
                });
            }
            let nll = negative_log_likelihood(logits, target_index, LogitStream::Official, row)?;
            ensure_nonzero_norm(logits, LogitStream::Official, row)?;
            let top1 = top1_token_id(logits, LogitStream::Official, row)?;
            if top1 != target_index {
                return Err(QualityError::ReferenceTop1Mismatch { row, target, top1 });
            }
            official_nll.add(nll);
        }
        if official_nll.sum <= 0.0 {
            return Err(QualityError::ZeroReferenceBaseline);
        }

        Ok(Self {
            official_logits,
            target_token_ids,
            row_position_ids,
            vocabulary_size,
            next_row: 0,
            official_nll,
            quantized_nll: CompensatedSum::default(),
            exact_top1_matches: 0,
            minimum_full_vector_cosine: f64::INFINITY,
        })
    }

    /// Consumes exactly the next corresponding quantized cached-logit row.
    pub(crate) fn observe(
        &mut self,
        position_id: usize,
        target_token_id: u32,
        quantized_logits: &[f32],
    ) -> Result<(), QualityError> {
        let row = self.next_row;
        if row == self.target_token_ids.len() {
            return Err(QualityError::TooManyRows {
                expected: self.target_token_ids.len(),
            });
        }
        let expected_position = self.row_position_ids[row];
        if position_id != expected_position {
            return Err(QualityError::PositionMismatch {
                row,
                expected: expected_position,
                actual: position_id,
            });
        }
        let expected_target = self.target_token_ids[row];
        if target_token_id != expected_target {
            return Err(QualityError::TargetMismatch {
                row,
                expected: expected_target,
                actual: target_token_id,
            });
        }
        if quantized_logits.len() != self.vocabulary_size {
            return Err(QualityError::RowWidth {
                stream: LogitStream::Quantized,
                row,
                expected: self.vocabulary_size,
                actual: quantized_logits.len(),
            });
        }

        let target_index =
            usize::try_from(target_token_id).map_err(|_| QualityError::TargetOutOfRange {
                row,
                token: target_token_id,
                vocabulary_size: self.vocabulary_size,
            })?;
        let quantized_nll =
            negative_log_likelihood(quantized_logits, target_index, LogitStream::Quantized, row)?;
        let official_row =
            &self.official_logits[row * self.vocabulary_size..(row + 1) * self.vocabulary_size];
        let cosine = full_vector_cosine(official_row, quantized_logits, row)?;
        let official_top1 = top1_token_id(official_row, LogitStream::Official, row)?;
        let quantized_top1 = top1_token_id(quantized_logits, LogitStream::Quantized, row)?;
        self.quantized_nll.add(quantized_nll);
        self.exact_top1_matches += usize::from(official_top1 == quantized_top1);
        self.minimum_full_vector_cosine = self.minimum_full_vector_cosine.min(cosine);
        self.next_row += 1;
        Ok(())
    }

    /// Finalizes means only after every immutable row has been observed.
    pub(crate) fn finish(self) -> Result<QualitySummary, QualityError> {
        let rows = self.target_token_ids.len();
        if self.next_row != rows {
            return Err(QualityError::Incomplete {
                expected: rows,
                actual: self.next_row,
            });
        }
        let denominator = rows as f64;
        let official_mean_nll = self.official_nll.sum / denominator;
        if !official_mean_nll.is_finite() || official_mean_nll <= 0.0 {
            return Err(QualityError::ZeroReferenceBaseline);
        }
        let quantized_mean_nll = self.quantized_nll.sum / denominator;
        let relative_nll_regression = (quantized_mean_nll - official_mean_nll) / official_mean_nll;
        if !quantized_mean_nll.is_finite() || !relative_nll_regression.is_finite() {
            return Err(QualityError::NonFiniteLogit {
                stream: LogitStream::Quantized,
                row: rows - 1,
                column: 0,
            });
        }
        Ok(QualitySummary {
            rows,
            official_mean_nll,
            quantized_mean_nll,
            relative_nll_regression,
            exact_top1_matches: self.exact_top1_matches,
            minimum_full_vector_cosine: self.minimum_full_vector_cosine,
        })
    }
}

/// Enforces an inclusive relative-NLL regression budget.
pub(crate) fn enforce_regression_limit(
    summary: &QualitySummary,
    limit: f64,
) -> Result<(), QualityError> {
    if !limit.is_finite() || limit < 0.0 {
        return Err(QualityError::InvalidRegressionLimit { limit });
    }
    if !summary.relative_nll_regression.is_finite() {
        return Err(QualityError::NonFiniteRegression {
            regression: summary.relative_nll_regression,
        });
    }
    if summary.relative_nll_regression > limit {
        return Err(QualityError::RegressionExceeded {
            regression: summary.relative_nll_regression,
            limit,
        });
    }
    Ok(())
}

/// Computes stable full-vocabulary NLL with an F64 log-sum-exp reduction.
fn negative_log_likelihood(
    logits: &[f32],
    target: usize,
    stream: LogitStream,
    row: usize,
) -> Result<f64, QualityError> {
    if target >= logits.len() {
        return Err(QualityError::TargetOutOfRange {
            row,
            token: u32::try_from(target).unwrap_or(u32::MAX),
            vocabulary_size: logits.len(),
        });
    }
    let mut maximum = f64::NEG_INFINITY;
    for (column, value) in logits.iter().copied().enumerate() {
        if !value.is_finite() {
            return Err(QualityError::NonFiniteLogit {
                stream,
                row,
                column,
            });
        }
        maximum = maximum.max(f64::from(value));
    }
    let mut exponential_sum = CompensatedSum::default();
    for value in logits {
        exponential_sum.add((f64::from(*value) - maximum).exp());
    }
    let nll = (maximum - f64::from(logits[target])) + exponential_sum.sum.ln();
    if !nll.is_finite() {
        return Err(QualityError::NonFiniteLogit {
            stream,
            row,
            column: target,
        });
    }
    Ok(nll)
}

/// Selects a deterministic maximum, resolving equal logits to the lowest ID.
fn top1_token_id(logits: &[f32], stream: LogitStream, row: usize) -> Result<usize, QualityError> {
    let mut best = 0;
    for (column, value) in logits.iter().copied().enumerate() {
        if !value.is_finite() {
            return Err(QualityError::NonFiniteLogit {
                stream,
                row,
                column,
            });
        }
        if value > logits[best] {
            best = column;
        }
    }
    Ok(best)
}

/// Rejects zero-magnitude rows before cosine can conceal invalid data.
fn ensure_nonzero_norm(
    logits: &[f32],
    stream: LogitStream,
    row: usize,
) -> Result<f64, QualityError> {
    let mut square_sum = CompensatedSum::default();
    for (column, value) in logits.iter().copied().enumerate() {
        if !value.is_finite() {
            return Err(QualityError::NonFiniteLogit {
                stream,
                row,
                column,
            });
        }
        square_sum.add(f64::from(value).powi(2));
    }
    if square_sum.sum == 0.0 {
        return Err(QualityError::ZeroNorm { stream, row });
    }
    Ok(square_sum.sum)
}

/// Computes cosine over corresponding full-vocabulary rows in F64.
fn full_vector_cosine(
    official: &[f32],
    quantized: &[f32],
    row: usize,
) -> Result<f64, QualityError> {
    if official.len() != quantized.len() {
        return Err(QualityError::RowWidth {
            stream: LogitStream::Quantized,
            row,
            expected: official.len(),
            actual: quantized.len(),
        });
    }
    let official_square = ensure_nonzero_norm(official, LogitStream::Official, row)?;
    let quantized_square = ensure_nonzero_norm(quantized, LogitStream::Quantized, row)?;
    let mut dot = CompensatedSum::default();
    for (left, right) in official.iter().zip(quantized) {
        dot.add(f64::from(*left) * f64::from(*right));
    }
    let cosine = dot.sum / (official_square.sqrt() * quantized_square.sqrt());
    if !cosine.is_finite() {
        return Err(QualityError::NonFiniteLogit {
            stream: LogitStream::Quantized,
            row,
            column: 0,
        });
    }
    Ok(cosine)
}

#[cfg(test)]
mod tests {
    //! Tests for fixed-sequence alignment, numerical diagnostics, and NLL regression limits.

    use super::*;

    /// Builds a two-row contract whose committed tokens are exact top-one IDs.
    fn fixture() -> (Vec<f32>, Vec<u32>, Vec<usize>) {
        (
            vec![2.0, 1.0, -1.0, -2.0, -1.0, 3.0],
            vec![0, 2],
            vec![8, 9],
        )
    }

    /// Observes the fixture unchanged and returns its zero-regression summary.
    fn identical_summary() -> QualitySummary {
        let (logits, targets, positions) = fixture();
        let mut quality = FixedSequenceQuality::new(&logits, &targets, &positions, 3).unwrap();
        quality.observe(8, 0, &logits[..3]).unwrap();
        quality.observe(9, 2, &logits[3..]).unwrap();
        quality.finish().unwrap()
    }

    /// A common large logit offset leaves stable NLL equal to the analytic result.
    #[test]
    fn stable_log_sum_exp_is_invariant_to_large_offsets() {
        let ordinary = negative_log_likelihood(&[2.0, 1.0], 0, LogitStream::Official, 0).unwrap();
        let offset =
            negative_log_likelihood(&[10_002.0, 10_001.0], 0, LogitStream::Official, 0).unwrap();
        assert!((ordinary - offset).abs() < 1.0e-15);
        assert!((ordinary - (1.0_f64 + (-1.0_f64).exp()).ln()).abs() < 1.0e-15);
    }

    /// Identical reference and observed rows produce zero regression and perfect agreement.
    #[test]
    fn unchanged_rows_have_zero_regression_and_complete_diagnostics() {
        let summary = identical_summary();
        assert_eq!(summary.rows, 2);
        assert_eq!(summary.official_mean_nll, summary.quantized_mean_nll);
        assert_eq!(summary.relative_nll_regression, 0.0);
        assert_eq!(summary.exact_top1_matches, 2);
        assert!((summary.minimum_full_vector_cosine - 1.0).abs() < 1.0e-15);
    }

    /// Improved target likelihood yields a negative regression accepted by a zero limit.
    #[test]
    fn lower_quantized_nll_is_not_treated_as_a_regression() {
        let (logits, targets, positions) = fixture();
        let mut quality = FixedSequenceQuality::new(&logits, &targets, &positions, 3).unwrap();
        quality.observe(8, 0, &[4.0, 0.0, -1.0]).unwrap();
        quality.observe(9, 2, &[-2.0, -1.0, 5.0]).unwrap();
        let summary = quality.finish().unwrap();
        assert!(summary.relative_nll_regression < 0.0);
        enforce_regression_limit(&summary, 0.0).unwrap();
    }

    /// A measured regression exactly equal to the configured limit passes.
    #[test]
    fn exact_regression_limit_is_inclusive() {
        let mut summary = identical_summary();
        summary.relative_nll_regression = 0.01;
        enforce_regression_limit(&summary, 0.01).unwrap();
    }

    /// A regression just above the limit returns the measured excess and bound.
    #[test]
    fn regression_above_limit_is_rejected() {
        let mut summary = identical_summary();
        summary.relative_nll_regression = 0.010_000_1;
        assert_eq!(
            enforce_regression_limit(&summary, 0.01),
            Err(QualityError::RegressionExceeded {
                regression: 0.010_000_1,
                limit: 0.01,
            })
        );
    }

    /// Analytic two-token NLL values determine the measured regression gate outcome.
    #[test]
    fn measured_nll_regression_drives_the_threshold_gate() {
        let official = [2.0, 0.0];
        let mut quality = FixedSequenceQuality::new(&official, &[0], &[4], 2).unwrap();
        quality.observe(4, 0, &[1.0, 0.0]).unwrap();
        let summary = quality.finish().unwrap();
        let official_nll = (1.0_f64 + (-2.0_f64).exp()).ln();
        let quantized_nll = (1.0_f64 + (-1.0_f64).exp()).ln();
        assert!((summary.official_mean_nll - official_nll).abs() < 1.0e-15);
        assert!((summary.quantized_mean_nll - quantized_nll).abs() < 1.0e-15);
        let expected_regression = (quantized_nll - official_nll) / official_nll;
        assert!((summary.relative_nll_regression - expected_regression).abs() < 1.0e-15);
        assert!(matches!(
            enforce_regression_limit(&summary, 1.0),
            Err(QualityError::RegressionExceeded { .. })
        ));
    }

    /// Negative, NaN, and infinite regression limits are rejected before comparison.
    #[test]
    fn negative_nan_and_infinite_limits_are_rejected() {
        let summary = identical_summary();
        for limit in [-0.0_f64 - 0.1, f64::NAN, f64::INFINITY] {
            assert!(matches!(
                enforce_regression_limit(&summary, limit),
                Err(QualityError::InvalidRegressionLimit { .. })
            ));
        }
    }

    /// NaN and either infinite measured regression fail even with a valid limit.
    #[test]
    fn nonfinite_measured_regressions_are_rejected() {
        for regression in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut summary = identical_summary();
            summary.relative_nll_regression = regression;
            assert!(matches!(
                enforce_regression_limit(&summary, 0.01),
                Err(QualityError::NonFiniteRegression { .. })
            ));
        }
    }

    /// Empty vocabulary or decision sequences cannot construct a quality accumulator.
    #[test]
    fn zero_vocabulary_and_empty_sequence_are_rejected() {
        assert!(matches!(
            FixedSequenceQuality::new(&[], &[0], &[0], 0),
            Err(QualityError::ZeroVocabulary)
        ));
        assert!(matches!(
            FixedSequenceQuality::new(&[], &[], &[], 3),
            Err(QualityError::EmptySequence)
        ));
    }

    /// Reference element counts and position counts must match the fixed sequence exactly.
    #[test]
    fn reference_shape_and_position_count_must_be_exact() {
        assert_eq!(
            FixedSequenceQuality::new(&[1.0, 0.0], &[0], &[4], 3).err(),
            Some(QualityError::ReferenceShape {
                expected_elements: 3,
                actual_elements: 2,
            })
        );
        assert_eq!(
            FixedSequenceQuality::new(&[1.0, 0.0], &[0], &[], 2).err(),
            Some(QualityError::PositionCount {
                expected: 1,
                actual: 0,
            })
        );
    }

    /// Overflow in decision-count times vocabulary-width calculation returns a shape error.
    #[test]
    fn shape_multiplication_overflow_is_rejected() {
        assert!(matches!(
            FixedSequenceQuality::new(&[], &[0, 0], &[0, 1], usize::MAX),
            Err(QualityError::ShapeOverflow)
        ));
    }

    /// Duplicate, backward, and skipped positions violate the contiguous sequence contract.
    #[test]
    fn duplicate_backward_and_gapped_positions_are_rejected() {
        let logits = [2.0, 0.0, 2.0, 0.0];
        for positions in [[7, 7], [7, 6], [7, 9]] {
            assert!(matches!(
                FixedSequenceQuality::new(&logits, &[0, 0], &positions, 2),
                Err(QualityError::NonContiguousPosition { row: 1, .. })
            ));
        }
    }

    /// A reference position at the integer maximum cannot be advanced to another row.
    #[test]
    fn position_successor_overflow_is_rejected() {
        assert_eq!(
            FixedSequenceQuality::new(&[2.0, 0.0, 2.0, 0.0], &[0, 0], &[usize::MAX, 0], 2,).err(),
            Some(QualityError::PositionOverflow { row: 0 })
        );
    }

    /// A target token equal to the vocabulary width is rejected with its row coordinate.
    #[test]
    fn out_of_range_target_is_rejected() {
        assert_eq!(
            FixedSequenceQuality::new(&[2.0, 0.0], &[2], &[0], 2).err(),
            Some(QualityError::TargetOutOfRange {
                row: 0,
                token: 2,
                vocabulary_size: 2,
            })
        );
    }

    /// Each fixed target must equal its official row argmax before scoring can begin.
    #[test]
    fn official_target_must_be_its_rows_top_one() {
        assert_eq!(
            FixedSequenceQuality::new(&[1.0, 2.0], &[0], &[0], 2).err(),
            Some(QualityError::ReferenceTop1Mismatch {
                row: 0,
                target: 0,
                top1: 1,
            })
        );
    }

    /// Equal maximum logits select the smallest token ID in the reference contract.
    #[test]
    fn top_one_ties_are_resolved_to_the_lowest_token_id() {
        assert!(FixedSequenceQuality::new(&[2.0, 2.0, 0.0], &[0], &[0], 3).is_ok());
        assert!(matches!(
            FixedSequenceQuality::new(&[2.0, 2.0, 0.0], &[1], &[0], 3),
            Err(QualityError::ReferenceTop1Mismatch { top1: 0, .. })
        ));
    }

    /// Positive and negative zero compare as a tie resolved by the smallest token ID.
    #[test]
    fn signed_zero_ties_are_resolved_to_the_lowest_token_id() {
        assert!(FixedSequenceQuality::new(&[-0.0, 0.0, -1.0], &[0], &[0], 3).is_ok());
        assert!(matches!(
            FixedSequenceQuality::new(&[-0.0, 0.0, -1.0], &[1], &[0], 3),
            Err(QualityError::ReferenceTop1Mismatch { top1: 0, .. })
        ));
    }

    /// Nonfinite reference logits report their stream, row, and vocabulary coordinate.
    #[test]
    fn nonfinite_reference_logits_are_rejected_with_their_coordinate() {
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert_eq!(
                FixedSequenceQuality::new(&[2.0, value], &[0], &[0], 2).err(),
                Some(QualityError::NonFiniteLogit {
                    stream: LogitStream::Official,
                    row: 0,
                    column: 1,
                })
            );
        }
    }

    /// A numerically zero reference NLL is rejected as an undefined regression denominator.
    #[test]
    fn zero_reference_baseline_is_rejected() {
        assert!(matches!(
            FixedSequenceQuality::new(&[1_000.0, 0.0], &[0], &[0], 2),
            Err(QualityError::ZeroReferenceBaseline)
        ));
    }

    /// An all-zero reference row is rejected because its cosine norm vanishes.
    #[test]
    fn zero_norm_reference_is_rejected() {
        assert_eq!(
            FixedSequenceQuality::new(&[0.0, 0.0], &[0], &[0], 2).err(),
            Some(QualityError::ZeroNorm {
                stream: LogitStream::Official,
                row: 0,
            })
        );
    }

    /// Mismatched position, target, or width fails without consuming the expected row.
    #[test]
    fn runtime_position_target_and_width_must_match_before_scoring() {
        let (logits, targets, positions) = fixture();
        let mut quality = FixedSequenceQuality::new(&logits, &targets, &positions, 3).unwrap();
        assert!(matches!(
            quality.observe(9, 0, &logits[..3]),
            Err(QualityError::PositionMismatch { row: 0, .. })
        ));
        assert!(matches!(
            quality.observe(8, 1, &logits[..3]),
            Err(QualityError::TargetMismatch { row: 0, .. })
        ));
        assert!(matches!(
            quality.observe(8, 0, &logits[..2]),
            Err(QualityError::RowWidth {
                stream: LogitStream::Quantized,
                row: 0,
                ..
            })
        ));
        quality.observe(8, 0, &logits[..3]).unwrap();
    }

    /// Nonfinite or zero-norm quantized rows fail before contributing quality measurements.
    #[test]
    fn nonfinite_and_zero_norm_quantized_rows_are_rejected() {
        let (logits, targets, positions) = fixture();
        for invalid in [[f32::NAN, 1.0, 0.0], [f32::INFINITY, 1.0, 0.0]] {
            let mut quality = FixedSequenceQuality::new(&logits, &targets, &positions, 3).unwrap();
            assert!(matches!(
                quality.observe(8, 0, &invalid),
                Err(QualityError::NonFiniteLogit {
                    stream: LogitStream::Quantized,
                    row: 0,
                    ..
                })
            ));
        }
        let mut quality = FixedSequenceQuality::new(&logits, &targets, &positions, 3).unwrap();
        assert_eq!(
            quality.observe(8, 0, &[0.0, 0.0, 0.0]),
            Err(QualityError::ZeroNorm {
                stream: LogitStream::Quantized,
                row: 0,
            })
        );
    }

    /// Finishing early or observing beyond the fixed sequence reports the exact row count.
    #[test]
    fn incomplete_and_extra_row_sequences_are_rejected() {
        let (logits, targets, positions) = fixture();
        let mut incomplete = FixedSequenceQuality::new(&logits, &targets, &positions, 3).unwrap();
        incomplete.observe(8, 0, &logits[..3]).unwrap();
        assert_eq!(
            incomplete.finish(),
            Err(QualityError::Incomplete {
                expected: 2,
                actual: 1,
            })
        );

        let mut extra = FixedSequenceQuality::new(&logits, &targets, &positions, 3).unwrap();
        extra.observe(8, 0, &logits[..3]).unwrap();
        extra.observe(9, 2, &logits[3..]).unwrap();
        assert_eq!(
            extra.observe(10, 0, &logits[..3]),
            Err(QualityError::TooManyRows { expected: 2 })
        );
    }

    /// Argmax disagreement reduces its diagnostic count while retaining finite cosine reporting.
    #[test]
    fn top_one_mismatch_is_reported_but_does_not_replace_nll_policy() {
        let (logits, targets, positions) = fixture();
        let mut quality = FixedSequenceQuality::new(&logits, &targets, &positions, 3).unwrap();
        quality.observe(8, 0, &[1.0, 2.0, -1.0]).unwrap();
        quality.observe(9, 2, &logits[3..]).unwrap();
        let summary = quality.finish().unwrap();
        assert_eq!(summary.exact_top1_matches, 1);
        assert!(summary.minimum_full_vector_cosine.is_finite());
    }
}
