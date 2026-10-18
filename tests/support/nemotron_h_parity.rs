//! Pure comparison contracts for Nemotron-H Burn/official parity.
//!
//! The pinned reference intentionally contains two numerical streams. A
//! stateless full prefix uses the official chunked Mamba scan, while cached
//! decoding uses its recurrent update. Those streams can retain the same top-1
//! token while differing substantially below it. These helpers therefore bind
//! every comparison to an execution path before evaluating full-vocabulary
//! cosine or exact argmax agreement.
//!
//! Logit rows are judged against the official class's F32 rows of the same
//! path (the precision bundle), not against the half-precision oracle: an F16
//! row carries its implementation's own rounding, and on this sparse hybrid
//! two official F16 runs of the same code already disagree by cosine `0.9943`
//! on a teacher-forced row.
//!
//! # Gate rule: per-path official F16 floors
//!
//! For a Burn row `a` and its F32 truth row `r` of the same path and decision,
//! the cosine `c = <a, r> / (|a| |r|)` is accumulated in F64
//! ([`cosine_similarity`]): products of F32 values are exact in F64, and the
//! sums' worst-case relative error, `n 2^-53` or about `1.5e-11` over the
//! `n = 131,072` vocabulary entries, is far below the `1e-6`-scale margins the
//! floors separate. For each path `k` (prompt, teacher-forced, cached) the
//! floor is the worst cosine against F32 that either official F16 evaluation
//! the gate is calibrated on reaches on that path ([`half_precision_floor`]):
//!
//! ```text
//! floor[k] = min over g in {MPS/F16 oracle, CPU/F16}, rows i of k, of cos(official[g, i], truth[i])
//! ```
//!
//! Every Burn row of the path must have exact top-1 and
//!
//! ```text
//! cos(burn[i], truth[i]) >= floor[k]      (inclusive, no slack),
//! ```
//!
//! so a Burn F16 row must be at least as faithful to F32 as the worst official
//! F16 row of its path. The floor must lie in
//! `[MINIMUM_HALF_PRECISION_FLOOR, 1]`. The bound is per path, not per
//! decision: half-precision deviations spike at implementation-specific rows.
//! Every row of a path is measured before any row is judged
//! ([`TruthPathSummary`], [`assert_within_floors`]), so a failing run reports
//! every row and every violation rather than stopping at the first.
//!
//! # Four-graph envelope: a printed diagnostic, never a gate
//!
//! The Nemotron-H envelope round registered
//! (`acceptance/measurements/nemotron-h-envelope-20261018/derivation/method.txt`)
//! and derived a rule over four official F16 graphs ([`OfficialF16Graph`]:
//! CPU and MPS, each with eager and SDPA attention). With the *normalized
//! deviation*
//!
//! ```text
//! delta(a, r) = | a/|a| - r/|r| |_2 = sqrt(2 (1 - c))
//! ```
//!
//! ([`normalized_deviation`]), the L2 distance between unit-normalized rows
//! and a monotone function of the cosine, each graph `g` has a worst row
//! `D[g, k] = max delta`, the path's envelope is `E[k] = max_g D[g, k]`, and
//! graph `g`'s leave-one-out ratio is `D[g, k] / max_{h != g} D[h, k]`
//! ([`PathEnvelope`]). The method would have admitted Burn rows with
//! `delta <= s[k] E[k]`, `s[k]` the largest ratio rounded up to two
//! significant digits ([`round_up_two_significant`]).
//!
//! The maintainer did not adopt that rule (decision of 2026-10-18,
//! `acceptance/measurements/nemotron-h-envelope-decision-20261018.json`):
//! every largest ratio is at most `1.0046`, so every `s[k]` is `1.1` only by
//! rounding, and Burn's cached decision 22 lies at `1.023 E`. The envelope
//! therefore judges nothing. When the four graphs are supplied, the gate prints
//! each graph's worst row and leave-one-out ratio, the slack the registered
//! method would round to (marked not adopted), and Burn's worst row of each
//! path as a multiple of `E` ([`BurnEnvelopeRatio`]), beside the floor
//! verdict.

#![allow(dead_code)]

use super::nemotron_h_base_reference::{
    OfficialExecutionPath, OfficialHiddenPoint, OfficialHiddenTrace, OfficialLogitRow,
};
use super::nemotron_h_precision_reference::{RowKind, TruthRow};

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

// ---------------------------------------------------------------------------
// Gate rule: per-path official F16 floors
// ---------------------------------------------------------------------------

/// Lowest per-path floor accepted; a looser official F16 floor is too wide to gate on.
pub(super) const MINIMUM_HALF_PRECISION_FLOOR: f64 = 0.99;

/// Worst cosine any official half-precision row reaches against its F32 truth row.
///
/// `truth` and every matrix in `official` are row-major `[rows, width]`
/// matrices of the same path; rows pair by index. The gate passes the MPS/F16
/// oracle and the precision bundle's CPU/F16 rows, so the floor is the worst
/// row of either official evaluation, with no slack.
///
/// # Panics
///
/// If `width` is zero, `official` is empty, `truth` holds no whole rows, a
/// matrix has another shape, a value is non-finite, a row has a zero norm, or
/// the floor lies outside `[MINIMUM_HALF_PRECISION_FLOOR, 1]`.
pub(super) fn half_precision_floor(
    label: &str,
    truth: &[f32],
    official: &[&[f32]],
    width: usize,
) -> f64 {
    assert!(width > 0, "{label} row width must be nonzero");
    assert!(
        !official.is_empty(),
        "{label} needs at least one official half-precision matrix"
    );
    assert!(
        !truth.is_empty() && truth.len().is_multiple_of(width),
        "{label} truth must hold whole rows"
    );
    assert_finite(label, "F32 truth", truth);
    let mut floor = f64::INFINITY;
    for matrix in official {
        assert_eq!(
            matrix.len(),
            truth.len(),
            "{label} official matrix shape changed"
        );
        assert_finite(label, "official half-precision", matrix);
        for (row, reference) in matrix.chunks_exact(width).zip(truth.chunks_exact(width)) {
            floor = floor.min(cosine_similarity(row, reference));
        }
    }
    assert!(
        (MINIMUM_HALF_PRECISION_FLOOR..=1.0).contains(&floor),
        "{label} official half-precision floor {floor} is outside [{MINIMUM_HALF_PRECISION_FLOOR}, 1]"
    );
    floor
}

/// Whether `value >= bound`; a NaN on either side never is, so it fails every floor.
fn at_least(value: f64, bound: f64) -> bool {
    matches!(
        value.partial_cmp(&bound),
        Some(std::cmp::Ordering::Greater | std::cmp::Ordering::Equal)
    )
}

/// One Burn row measured against its F32 truth row, before any floor is applied.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct TruthRowObservation {
    /// Zero-based decision.
    pub(super) decision: usize,
    /// F64-accumulated cosine against the F32 truth row.
    pub(super) cosine: f64,
    /// Stable Burn argmax, with lower token ID winning an exact tie.
    pub(super) actual_top1_token_id: usize,
    /// Authenticated token the row must predict.
    pub(super) expected_top1_token_id: usize,
}

/// Measures one Burn row against its F32 truth row; judges nothing but the row's identity.
///
/// The row family and stream must match, both rows must be finite and
/// nonempty with equal length, and the truth's argmax must equal the
/// authenticated `expected_top1` (otherwise the truth row is not the one the
/// oracle predicted from). Burn's top-1 and cosine are recorded for
/// [`CompletedTruthSummary::violations`].
pub(super) fn measure_row_against_truth(
    label: &str,
    actual_kind: RowKind,
    actual: &[f32],
    truth: TruthRow<'_>,
    expected_top1: usize,
) -> TruthRowObservation {
    assert_eq!(
        actual_kind,
        truth.kind(),
        "{label} compared a different row family"
    );
    assert_eq!(
        actual_kind.path(),
        truth.path(),
        "{label} crossed the stateless and cached streams"
    );
    assert_eq!(
        actual.len(),
        truth.values().len(),
        "{label} vector length changed"
    );
    assert!(!actual.is_empty(), "{label} vectors must not be empty");
    assert_finite(label, "Burn", actual);
    assert_finite(label, "F32 truth", truth.values());
    assert_eq!(
        stable_top1(truth.values()),
        expected_top1,
        "{label} F32 truth disagrees with the authenticated token at decision {}",
        truth.decision()
    );
    TruthRowObservation {
        decision: truth.decision(),
        cosine: cosine_similarity(actual, truth.values()),
        actual_top1_token_id: stable_top1(actual),
        expected_top1_token_id: expected_top1,
    }
}

/// Ordered rows of one family measured against its F32 truth under one official F16 floor.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct TruthPathSummary {
    /// Row family required for every observed row.
    kind: RowKind,
    /// Official F16 floor the rows must reach.
    floor: f64,
    /// Rows observed so far, in decision order.
    rows: Vec<TruthRowObservation>,
}

impl TruthPathSummary {
    /// Starts an empty sequence for `kind` judged by `floor`.
    pub(super) const fn new(kind: RowKind, floor: f64) -> Self {
        Self {
            kind,
            floor,
            rows: Vec::new(),
        }
    }

    /// Measures and records the next decision in strict contiguous order.
    ///
    /// # Panics
    ///
    /// On a skipped or repeated decision, or as [`measure_row_against_truth`].
    pub(super) fn observe(
        &mut self,
        label: &str,
        actual: &[f32],
        truth: TruthRow<'_>,
        expected_top1: usize,
    ) -> TruthRowObservation {
        assert_eq!(
            truth.decision(),
            self.rows.len(),
            "{label} must compare every decision once and in order"
        );
        let observation = measure_row_against_truth(label, self.kind, actual, truth, expected_top1);
        self.rows.push(observation);
        observation
    }

    /// Completes the sequence only after exactly `expected_rows` decisions.
    pub(super) fn finish(self, expected_rows: usize) -> CompletedTruthSummary {
        assert!(expected_rows > 0, "a truth sequence cannot be empty");
        assert_eq!(
            self.rows.len(),
            expected_rows,
            "truth parity did not cover every expected row"
        );
        CompletedTruthSummary {
            kind: self.kind,
            floor: self.floor,
            rows: self.rows,
        }
    }
}

/// Every row of one fully covered family, with the official F16 floor that judges it.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct CompletedTruthSummary {
    /// Row family compared.
    pub(super) kind: RowKind,
    /// Official F16 floor applied.
    pub(super) floor: f64,
    /// Every row, in decision order.
    pub(super) rows: Vec<TruthRowObservation>,
}

impl CompletedTruthSummary {
    /// The row with the lowest cosine (first on ties).
    pub(super) fn worst(&self) -> TruthRowObservation {
        self.rows
            .iter()
            .copied()
            .reduce(|worst, row| {
                if row.cosine < worst.cosine {
                    row
                } else {
                    worst
                }
            })
            .expect("a completed summary has rows")
    }

    /// Describes every row that misses its top-1 token or the official F16 floor.
    pub(super) fn violations(&self) -> Vec<String> {
        let mut out = Vec::new();
        for row in &self.rows {
            if row.actual_top1_token_id != row.expected_top1_token_id {
                out.push(format!(
                    "{:?} decision {}: top-1 {} instead of {}",
                    self.kind, row.decision, row.actual_top1_token_id, row.expected_top1_token_id
                ));
            }
            if !at_least(row.cosine, self.floor) {
                out.push(format!(
                    "{:?} decision {}: cosine {} against the F32 truth is below the official F16 floor {}",
                    self.kind, row.decision, row.cosine, self.floor
                ));
            }
        }
        out
    }

    /// Prints the lowest cosine, its margin over the floor, and every row's cosine.
    pub(super) fn report(&self, label: &str) {
        let worst = self.worst();
        eprintln!(
            "{label} {:?}: {} rows, minimum cosine {:.12} at decision {} (official F16 floor {:.9}, margin {:+.3e})",
            self.kind,
            self.rows.len(),
            worst.cosine,
            worst.decision,
            self.floor,
            worst.cosine - self.floor,
        );
        let cosines = self
            .rows
            .iter()
            .map(|row| format!("{:.9}", row.cosine))
            .collect::<Vec<_>>();
        eprintln!("{label} {:?} cosines: [{}]", self.kind, cosines.join(", "));
    }
}

/// Asserts that every completed family meets its top-1 and floor contract, listing every violation.
///
/// # Panics
///
/// If any row of any family has another top-1 token or a cosine below its floor.
pub(super) fn assert_within_floors(label: &str, summaries: &[&CompletedTruthSummary]) {
    let violations = summaries
        .iter()
        .flat_map(|summary| summary.violations())
        .collect::<Vec<_>>();
    assert!(
        violations.is_empty(),
        "{label}: {} row(s) outside the official F16 floor contract: {}",
        violations.len(),
        violations.join("; ")
    );
}

// ---------------------------------------------------------------------------
// Four-graph envelope: a printed diagnostic, never a gate
// ---------------------------------------------------------------------------

/// One of the four official F16 graphs of the envelope diagnostic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum OfficialF16Graph {
    /// CPU, eager attention: the precision bundle's `float16` rows (also a floor input).
    CpuEager,
    /// CPU, SDPA attention: the CPU SDPA graph bundle.
    CpuSdpa,
    /// MPS, eager attention: the MPS/F16 oracle (also a floor input).
    MpsEager,
    /// MPS, SDPA attention: the MPS SDPA graph bundle.
    MpsSdpa,
}

impl OfficialF16Graph {
    /// Every graph, in the order [`PathEnvelope`] stores them.
    pub(super) const ALL: [Self; 4] =
        [Self::CpuEager, Self::CpuSdpa, Self::MpsEager, Self::MpsSdpa];

    /// `device-attention`, as the derivation names the graph.
    pub(super) const fn label(self) -> &'static str {
        match self {
            Self::CpuEager => "cpu-eager",
            Self::CpuSdpa => "cpu-sdpa",
            Self::MpsEager => "mps-eager",
            Self::MpsSdpa => "mps-sdpa",
        }
    }
}

/// Returns `sqrt(2 (1 - c))`, the distance between two unit-normalized rows whose cosine is `c`.
///
/// `|a/|a| - r/|r||^2 = 2 - 2 c` for unit rows. The subtraction `1 - c` is
/// exact in F64 for `c` in `[0.5, 1]` (Sterbenz), so the deviation carries the
/// cosine's own relative error. A cosine rounded above 1 reads as 1, so equal
/// rows have deviation 0.
pub(super) fn normalized_deviation(cosine: f64) -> f64 {
    assert!(
        cosine.is_finite(),
        "a normalized deviation needs a finite cosine"
    );
    (2.0 * (1.0 - cosine).max(0.0)).sqrt()
}

/// Rounds a positive `value` up to two significant digits of its shortest round-trip decimal.
///
/// This is `scripts/derive_half_precision_envelope.py`'s `round_up`, which the
/// registered method uses for its (not adopted) slack: `1.4206` becomes `1.5`,
/// `1.2988` becomes `1.3`, `1.0046` becomes `1.1`, and `1.3` stays `1.3`,
/// because the shortest decimal that reads back as the F64 `1.3` is `1.3`
/// itself.
pub(super) fn round_up_two_significant(value: f64) -> f64 {
    assert!(
        value.is_finite() && value > 0.0,
        "a slack must be positive and finite, not {value}"
    );
    // `{:e}` prints the shortest round-trip mantissa and its decimal exponent.
    let printed = format!("{value:e}");
    let (mantissa, exponent) = printed.split_once('e').expect("scientific notation");
    let exponent: i32 = exponent.parse().expect("decimal exponent");
    let digits: Vec<u32> = mantissa
        .chars()
        .filter(char::is_ascii_digit)
        .map(|digit| digit.to_digit(10).expect("decimal digit"))
        .collect();
    let mut leading = digits[0] * 10 + digits.get(1).copied().unwrap_or(0);
    if digits.iter().skip(2).any(|&digit| digit != 0) {
        leading += 1;
    }
    // `leading` holds two significant digits scaled by 10^(exponent - 1).
    format!("{leading}e{}", exponent - 1)
        .parse()
        .expect("rounded slack")
}

/// One official graph's worst row on one path.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct GraphWorstRow {
    /// Largest normalized deviation over the path's rows, `D[g, k]`.
    pub(super) deviation: f64,
    /// Cosine of that row against its F32 truth row.
    pub(super) cosine: f64,
    /// Zero-based decision of that row (first on ties).
    pub(super) decision: usize,
}

/// The four official F16 graphs' worst rows on one path and their envelope; a diagnostic, not a gate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct PathEnvelope {
    /// Row family the envelope describes.
    pub(super) kind: RowKind,
    /// Each graph's worst row, in [`OfficialF16Graph::ALL`] order.
    pub(super) graphs: [GraphWorstRow; 4],
    /// `E[k]`, the largest of the graphs' worst deviations.
    pub(super) envelope: f64,
    /// Graph attaining the envelope (first in [`OfficialF16Graph::ALL`] order on ties).
    pub(super) envelope_graph: OfficialF16Graph,
}

impl PathEnvelope {
    /// Measures the four official graphs of `kind` against its F32 truth.
    ///
    /// `truth` and every `official` matrix are row-major `[rows, width]`
    /// matrices of the same path, in [`OfficialF16Graph::ALL`] order. No bound
    /// or floor is derived: the result judges no Burn row.
    ///
    /// # Panics
    ///
    /// If a matrix holds a different number of values than `kind`'s rows of
    /// `width`, a non-finite value or a zero row, or if the envelope is zero
    /// (four graphs equal to the truth are not half-precision evaluations, and
    /// leave-one-out ratios over them are undefined).
    pub(super) fn measure(
        label: &str,
        kind: RowKind,
        truth: &[f32],
        official: [&[f32]; 4],
        width: usize,
    ) -> Self {
        assert!(width > 0, "{label} row width must be nonzero");
        assert_eq!(
            truth.len(),
            kind.rows() * width,
            "{label} truth must hold every {kind:?} row"
        );
        assert_finite(label, "F32 truth", truth);
        let graphs = official.map(|matrix| {
            assert_eq!(
                matrix.len(),
                truth.len(),
                "{label} official matrix shape changed"
            );
            assert_finite(label, "official half-precision", matrix);
            let mut worst = GraphWorstRow {
                deviation: -1.0,
                cosine: f64::NAN,
                decision: 0,
            };
            for (decision, (row, reference)) in matrix
                .chunks_exact(width)
                .zip(truth.chunks_exact(width))
                .enumerate()
            {
                let cosine = cosine_similarity(row, reference);
                let deviation = normalized_deviation(cosine);
                if deviation > worst.deviation {
                    worst = GraphWorstRow {
                        deviation,
                        cosine,
                        decision,
                    };
                }
            }
            worst
        });
        let (index, largest) =
            graphs
                .iter()
                .enumerate()
                .fold((0, graphs[0]), |(best, largest), (index, graph)| {
                    if graph.deviation > largest.deviation {
                        (index, *graph)
                    } else {
                        (best, largest)
                    }
                });
        assert!(
            largest.deviation > 0.0,
            "{label} official F16 graphs equal the F32 truth, so they cannot form a half-precision envelope"
        );
        Self {
            kind,
            graphs,
            envelope: largest.deviation,
            envelope_graph: OfficialF16Graph::ALL[index],
        }
    }

    /// Each graph's worst deviation over the largest of the other three graphs' worst deviations.
    ///
    /// A zero denominator gives an infinite or NaN ratio.
    pub(super) fn leave_one_out(&self) -> [f64; 4] {
        std::array::from_fn(|graph| {
            let others = (0..4)
                .filter(|&other| other != graph)
                .map(|other| self.graphs[other].deviation)
                .fold(f64::NEG_INFINITY, f64::max);
            self.graphs[graph].deviation / others
        })
    }

    /// The largest leave-one-out ratio and its graph (first on ties).
    pub(super) fn largest_leave_one_out(&self) -> (OfficialF16Graph, f64) {
        let ratios = self.leave_one_out();
        let mut best = 0;
        for index in 1..4 {
            if ratios[index] > ratios[best] {
                best = index;
            }
        }
        (OfficialF16Graph::ALL[best], ratios[best])
    }

    /// The slack the registered method would round the largest leave-one-out ratio to; not adopted.
    pub(super) fn registered_method_slack(&self) -> f64 {
        round_up_two_significant(self.largest_leave_one_out().1)
    }

    /// Prints every graph's worst row and leave-one-out ratio, the envelope, and the not-adopted slack.
    pub(super) fn report(&self, label: &str) {
        for ((graph, worst), ratio) in OfficialF16Graph::ALL
            .iter()
            .zip(self.graphs)
            .zip(self.leave_one_out())
        {
            eprintln!(
                "{label} {:?}: official {} worst delta {:.6e} (cosine {:.9}) at decision {}, leave-one-out {ratio:.4}",
                self.kind,
                graph.label(),
                worst.deviation,
                worst.cosine,
                worst.decision,
            );
        }
        let (graph, largest) = self.largest_leave_one_out();
        let slack = self.registered_method_slack();
        let bound = slack * self.envelope;
        eprintln!(
            "{label} {:?}: envelope {:.6e} ({}); largest leave-one-out {largest:.4} ({}); the registered \
             method's slack {slack} (cosine floor {:.9}) is not adopted and judges no row",
            self.kind,
            self.envelope,
            self.envelope_graph.label(),
            graph.label(),
            1.0 - bound * bound / 2.0,
        );
    }
}

/// Burn's worst row of one path as a multiple of the four-graph envelope; a diagnostic, not a gate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct BurnEnvelopeRatio {
    /// Envelope the rows were compared with.
    pub(super) envelope: PathEnvelope,
    /// Burn's lowest-cosine row of the path (its largest normalized deviation).
    pub(super) worst: TruthRowObservation,
    /// [`normalized_deviation`] of that row's cosine.
    pub(super) deviation: f64,
    /// `deviation / E[k]`.
    pub(super) of_envelope: f64,
    /// Burn rows of the path whose deviation exceeds `E[k]`.
    pub(super) rows_beyond_envelope: usize,
    /// Burn rows of the path.
    pub(super) rows: usize,
}

impl BurnEnvelopeRatio {
    /// Measures a completed Burn path against the envelope of the same path.
    ///
    /// # Panics
    ///
    /// If the summary and the envelope describe different row families.
    pub(super) fn measure(envelope: &PathEnvelope, summary: &CompletedTruthSummary) -> Self {
        assert_eq!(
            envelope.kind, summary.kind,
            "the envelope diagnostic compared different row families"
        );
        let worst = summary.worst();
        let deviation = normalized_deviation(worst.cosine);
        Self {
            envelope: *envelope,
            worst,
            deviation,
            of_envelope: deviation / envelope.envelope,
            rows_beyond_envelope: summary
                .rows
                .iter()
                .filter(|row| normalized_deviation(row.cosine) > envelope.envelope)
                .count(),
            rows: summary.rows.len(),
        }
    }

    /// Whether Burn's worst row lies further outside the envelope than any official graph lies outside the other three.
    pub(super) fn beyond_every_official_graph(&self) -> bool {
        self.of_envelope > self.envelope.largest_leave_one_out().1
    }

    /// Prints Burn's ratio to the envelope beside the official graphs' largest leave-one-out ratio.
    pub(super) fn report(&self, label: &str) {
        let (graph, largest) = self.envelope.largest_leave_one_out();
        eprintln!(
            "{label} {:?}: Burn's worst row, decision {} (cosine {:.12}, delta {:.6e}), is {:.4} of the envelope \
             {:.6e} ({}); {} of {} Burn rows exceed the envelope; the largest official leave-one-out ratio is \
             {largest:.4} ({}){}; diagnostic only, the official F16 floor decides",
            self.envelope.kind,
            self.worst.decision,
            self.worst.cosine,
            self.deviation,
            self.of_envelope,
            self.envelope.envelope,
            self.envelope.envelope_graph.label(),
            self.rows_beyond_envelope,
            self.rows,
            graph.label(),
            if self.beyond_every_official_graph() {
                ", which Burn's row exceeds"
            } else {
                ""
            },
        );
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
    //! Synthetic tests for path identity, row ordering, the official F16 floor gate, and the envelope diagnostic.

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

    /// Rows of `kind` at `angle` radians from `e0` in the `e0, e1` plane; `worst` moves one row to another angle.
    fn plane_rows(kind: RowKind, angle: f64, worst: Option<(usize, f64)>) -> Vec<f32> {
        (0..kind.rows())
            .flat_map(|decision| {
                let angle = match worst {
                    Some((row, other)) if row == decision => other,
                    _ => angle,
                };
                [angle.cos() as f32, angle.sin() as f32]
            })
            .collect()
    }

    /// Distance between two unit rows `angle` apart.
    fn chord(angle: f64) -> f64 {
        2.0 * (angle / 2.0).sin()
    }

    /// The cosine of an F32 unit row at `angle` from `e0`, as the gate measures it against `e0`.
    fn plane_cosine(angle: f64) -> f64 {
        cosine_similarity(&[angle.cos() as f32, angle.sin() as f32], &[1.0, 0.0])
    }

    /// The floor is the worst official half-precision row and must stay meaningful.
    #[test]
    fn half_precision_floor_takes_the_worst_official_row() {
        let truth = [1.0, 0.0, 0.0, 1.0];
        let close = [1.0, 0.01, 0.0, 1.0];
        let closer = [1.0, 0.0, 0.001, 1.0];
        let floor = half_precision_floor("synthetic", &truth, &[&close, &closer], 2);
        assert_eq!(floor, cosine_similarity(&close[..2], &truth[..2]));
        assert!(floor < 1.0 && floor > 0.9999);
        assert_eq!(
            half_precision_floor("synthetic", &truth, &[&closer, &close], 2),
            floor,
            "the floor does not depend on the order of the official evaluations"
        );
        assert_eq!(half_precision_floor("synthetic", &truth, &[&truth], 2), 1.0);
        let far = [1.0, 1.0, 0.0, 1.0];
        assert!(catch_unwind(|| half_precision_floor("synthetic", &truth, &[&far], 2)).is_err());
        assert!(catch_unwind(|| half_precision_floor("synthetic", &truth, &[], 2)).is_err());
        assert!(catch_unwind(|| half_precision_floor("synthetic", &truth, &[&close], 0)).is_err());
        assert!(
            catch_unwind(|| half_precision_floor("synthetic", &truth, &[&close[..2]], 2)).is_err()
        );
        assert!(
            catch_unwind(|| half_precision_floor("synthetic", &truth[..3], &[&close[..3]], 2))
                .is_err()
        );
        let infinite = [1.0, f32::INFINITY, 0.0, 1.0];
        assert!(
            catch_unwind(|| half_precision_floor("synthetic", &truth, &[&infinite], 2)).is_err()
        );
        assert!(
            catch_unwind(|| half_precision_floor("synthetic", &infinite, &[&close], 2)).is_err()
        );
        let zero = [0.0, 0.0, 0.0, 1.0];
        assert!(catch_unwind(|| half_precision_floor("synthetic", &truth, &[&zero], 2)).is_err());
    }

    /// Floors are inclusive, and a NaN never reaches one.
    #[test]
    fn floors_are_inclusive_and_refuse_nan() {
        assert!(at_least(1.0, 1.0));
        assert!(at_least(1.0, 0.5));
        assert!(!at_least(1.0 - f64::EPSILON, 1.0));
        assert!(!at_least(f64::NAN, 0.5));
        assert!(!at_least(0.5, f64::NAN));
    }

    /// Rows are measured without judgment; a completed path lists every top-1 and floor violation.
    #[test]
    fn truth_rows_are_judged_by_the_floor_after_the_whole_path() {
        let floor = plane_cosine(0.015);
        let truth_values = [1.0, 0.0];
        let mut summary = TruthPathSummary::new(RowKind::Cached, floor);
        for decision in 0..RowKind::Cached.rows() {
            let angle: f64 = match decision {
                3 => 0.019,
                5 => 0.015,
                _ => 0.012,
            };
            let actual = if decision == 9 {
                [0.0, 1.0]
            } else {
                [angle.cos() as f32, angle.sin() as f32]
            };
            let observation = summary.observe(
                "synthetic",
                &actual,
                TruthRow::synthetic(RowKind::Cached, decision, &truth_values),
                0,
            );
            assert_eq!(observation.decision, decision);
        }
        let completed = summary.finish(RowKind::Cached.rows());
        assert_eq!((completed.kind, completed.floor), (RowKind::Cached, floor));
        assert_eq!(completed.rows.len(), RowKind::Cached.rows());
        assert_eq!(completed.worst().decision, 9);
        let violations = completed.violations();
        assert_eq!(violations.len(), 3, "{violations:?}");
        assert!(
            violations[0].contains("decision 3") && violations[0].contains("official F16 floor")
        );
        assert!(violations[1].contains("decision 9: top-1 1 instead of 0"));
        assert!(
            violations[2].contains("decision 9") && violations[2].contains("official F16 floor")
        );
        assert!(
            !violations
                .iter()
                .any(|violation| violation.contains("decision 5")),
            "a row exactly at the floor passes"
        );
        completed.report("synthetic");
        let message = catch_unwind(|| assert_within_floors("synthetic", &[&completed]))
            .expect_err("violations must fail")
            .downcast::<String>()
            .expect("formatted panic");
        assert!(message.contains("3 row(s)"));

        let mut within = TruthPathSummary::new(RowKind::Prompt, floor);
        within.observe(
            "synthetic",
            &[0.012_f64.cos() as f32, 0.012_f64.sin() as f32],
            TruthRow::synthetic(RowKind::Prompt, 0, &truth_values),
            0,
        );
        let within = within.finish(1);
        assert!(within.violations().is_empty());
        assert_within_floors("synthetic", &[&within]);
        assert_within_floors("synthetic", &[]);
    }

    /// Order, coverage, row family, the truth's authenticated token, and finiteness fail immediately.
    #[test]
    fn truth_summary_enforces_order_coverage_and_identity() {
        let values = [3.0, 1.0];
        let mut summary = TruthPathSummary::new(RowKind::Cached, 0.99);
        summary.observe(
            "synthetic",
            &values,
            TruthRow::synthetic(RowKind::Cached, 0, &values),
            0,
        );
        let mut skipped = summary.clone();
        assert!(
            catch_unwind(AssertUnwindSafe(|| skipped.observe(
                "synthetic",
                &values,
                TruthRow::synthetic(RowKind::Cached, 2, &values),
                0
            )))
            .is_err()
        );
        assert!(catch_unwind(|| summary.clone().finish(2)).is_err());
        assert!(catch_unwind(|| summary.clone().finish(0)).is_err());
        let done = summary.clone().finish(1);
        assert_eq!(
            (done.rows.len(), done.floor, done.kind),
            (1, 0.99, RowKind::Cached)
        );
        for (kind, actual, top1) in [
            (RowKind::Teacher, &values[..], 0),
            (RowKind::Prompt, &values[..], 0),
            (RowKind::Cached, &values[..], 1),
            (RowKind::Cached, &values[..1], 0),
            (RowKind::Cached, &[][..], 0),
            (RowKind::Cached, &[f32::NAN, 1.0][..], 0),
        ] {
            assert!(
                catch_unwind(|| measure_row_against_truth(
                    "synthetic",
                    RowKind::Cached,
                    actual,
                    TruthRow::synthetic(kind, 0, &values),
                    top1
                ))
                .is_err(),
                "{kind:?} top1={top1} must be refused"
            );
        }
    }

    /// The normalized deviation is the distance between unit-normalized rows.
    #[test]
    fn normalized_deviation_is_the_unit_row_distance() {
        assert_eq!(normalized_deviation(1.0), 0.0);
        assert_eq!(normalized_deviation(1.0 + f64::EPSILON), 0.0);
        assert!((normalized_deviation(0.0) - 2.0_f64.sqrt()).abs() < 1e-15);
        assert!((normalized_deviation(0.1_f64.cos()) - chord(0.1)).abs() < 1e-12);
        let rows = plane_rows(RowKind::Prompt, 0.02, None);
        let truth = plane_rows(RowKind::Prompt, 0.0, None);
        let measured = normalized_deviation(cosine_similarity(&rows, &truth));
        assert!((measured - chord(0.02)).abs() < 1e-7);
        assert!(catch_unwind(|| normalized_deviation(f64::NAN)).is_err());
        assert!(catch_unwind(|| normalized_deviation(f64::INFINITY)).is_err());
    }

    /// Slack rounding is the envelope round's ceiling to two significant digits of the shortest decimal.
    #[test]
    fn slack_rounding_matches_the_envelope_round() {
        for (value, rounded) in [
            (1.4206, 1.5),
            (1.2988, 1.3),
            (1.3, 1.3),
            (1.300_000_01, 1.4),
            (1.0, 1.0),
            (1.0002, 1.1),
            (1.0046, 1.1),
            (0.95, 0.95),
            (12.5, 13.0),
            (99.5, 100.0),
            (1.666, 1.7),
            (1.6981, 1.7),
            (2.0, 2.0),
            (4.045, 4.1),
        ] {
            assert_eq!(round_up_two_significant(value), rounded, "{value}");
        }
        for invalid in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(catch_unwind(|| round_up_two_significant(invalid)).is_err());
        }
    }

    /// The diagnostic's graphs keep the derivation's names and order.
    #[test]
    fn official_graphs_keep_the_derivation_order() {
        assert_eq!(
            OfficialF16Graph::ALL.map(OfficialF16Graph::label),
            ["cpu-eager", "cpu-sdpa", "mps-eager", "mps-sdpa"]
        );
    }

    /// The envelope is the largest of each graph's worst row, with each graph's leave-one-out ratio.
    #[test]
    fn path_envelope_takes_each_graphs_worst_row() {
        let kind = RowKind::Teacher;
        let truth = plane_rows(kind, 0.0, None);
        let graphs = [
            plane_rows(kind, 0.010, None),
            plane_rows(kind, 0.011, Some((7, 0.012))),
            plane_rows(kind, 0.010, Some((18, 0.030))),
            plane_rows(kind, 0.012, None),
        ];
        let envelope = PathEnvelope::measure(
            "synthetic",
            kind,
            &truth,
            [&graphs[0], &graphs[1], &graphs[2], &graphs[3]],
            2,
        );
        assert_eq!(envelope.kind, kind);
        assert_eq!(envelope.envelope_graph, OfficialF16Graph::MpsEager);
        assert_eq!(envelope.graphs[2].decision, 18);
        assert_eq!(envelope.graphs[1].decision, 7);
        assert_eq!(envelope.graphs[0].decision, 0, "the first row wins ties");
        assert!((envelope.envelope - chord(0.030)).abs() < 1e-7);
        let ratios = envelope.leave_one_out();
        assert!((ratios[2] - chord(0.030) / chord(0.012)).abs() < 1e-5);
        assert!((ratios[0] - chord(0.010) / chord(0.030)).abs() < 1e-5);
        assert_eq!(
            envelope.largest_leave_one_out(),
            (OfficialF16Graph::MpsEager, ratios[2])
        );
        assert_eq!(envelope.registered_method_slack(), 2.5);
        envelope.report("synthetic");
    }

    /// The registered method's slack is the largest leave-one-out ratio rounded up, as reported, never applied.
    #[test]
    fn registered_method_slack_rounds_the_largest_ratio() {
        let kind = RowKind::Prompt;
        let truth = plane_rows(kind, 0.0, None);
        let graphs = [
            plane_rows(kind, 0.010, None),
            plane_rows(kind, 0.011, None),
            plane_rows(kind, 0.020, None),
            plane_rows(kind, 0.012, None),
        ];
        let envelope = PathEnvelope::measure(
            "synthetic",
            kind,
            &truth,
            [&graphs[0], &graphs[1], &graphs[2], &graphs[3]],
            2,
        );
        let (graph, largest) = envelope.largest_leave_one_out();
        assert_eq!(graph, OfficialF16Graph::MpsEager);
        assert!((largest - chord(0.020) / chord(0.012)).abs() < 1e-5);
        assert_eq!(envelope.registered_method_slack(), 1.7);
        let tied = PathEnvelope::measure(
            "synthetic",
            kind,
            &truth,
            [&graphs[0], &graphs[0], &graphs[0], &graphs[0]],
            2,
        );
        assert_eq!(tied.leave_one_out(), [1.0; 4]);
        assert_eq!(
            tied.largest_leave_one_out(),
            (OfficialF16Graph::CpuEager, 1.0)
        );
        assert_eq!(tied.envelope_graph, OfficialF16Graph::CpuEager);
        assert_eq!(tied.registered_method_slack(), 1.0);
    }

    /// Equal, malformed, or non-finite official graphs are refused; a wide one is measured, since nothing is gated.
    #[test]
    fn path_envelope_refuses_unusable_graphs() {
        let kind = RowKind::Prompt;
        let truth = plane_rows(kind, 0.0, None);
        let near = plane_rows(kind, 0.01, None);
        let wide = plane_rows(kind, 0.2, None);
        let mut infinite = near.clone();
        infinite[1] = f32::INFINITY;
        let attempt = |graphs: [&[f32]; 4], width: usize| {
            catch_unwind(|| PathEnvelope::measure("synthetic", kind, &truth, graphs, width))
        };
        assert!(attempt([&near, &near, &near, &near], 2).is_ok());
        assert!(attempt([&near, &near, &wide, &near], 2).is_ok());
        assert!(attempt([&truth, &truth, &truth, &truth], 2).is_err());
        assert!(attempt([&near, &near, &near[..1], &near], 2).is_err());
        assert!(attempt([&near, &near, &infinite, &near], 2).is_err());
        assert!(attempt([&near, &near, &near, &near], 0).is_err());
        let short_truth = &truth[..1];
        assert!(
            catch_unwind(|| PathEnvelope::measure(
                "synthetic",
                kind,
                short_truth,
                [&near[..1], &near[..1], &near[..1], &near[..1]],
                2,
            ))
            .is_err(),
            "a truth without a whole row is refused"
        );
        assert!(
            catch_unwind(|| PathEnvelope::measure(
                "synthetic",
                kind,
                &infinite,
                [&near, &near, &near, &near],
                2,
            ))
            .is_err(),
            "a non-finite truth is refused"
        );
    }

    /// Four synthetic graphs of `kind` whose envelope is the chord of `0.012` (MPS eager) and largest ratio `1.2`.
    fn synthetic_envelope(kind: RowKind) -> PathEnvelope {
        let truth = plane_rows(kind, 0.0, None);
        let graphs = [
            plane_rows(kind, 0.010, None),
            plane_rows(kind, 0.009, None),
            plane_rows(kind, 0.012, None),
            plane_rows(kind, 0.008, None),
        ];
        PathEnvelope::measure(
            "synthetic",
            kind,
            &truth,
            [&graphs[0], &graphs[1], &graphs[2], &graphs[3]],
            2,
        )
    }

    /// A completed Burn path whose rows sit at `angles` from the truth, every top-1 exact.
    fn synthetic_burn(kind: RowKind, floor: f64, angles: &[f64]) -> CompletedTruthSummary {
        let truth_values = [1.0, 0.0];
        let mut summary = TruthPathSummary::new(kind, floor);
        for (decision, angle) in angles.iter().enumerate() {
            summary.observe(
                "synthetic",
                &[angle.cos() as f32, angle.sin() as f32],
                TruthRow::synthetic(kind, decision, &truth_values),
                0,
            );
        }
        summary.finish(angles.len())
    }

    /// The diagnostic reports Burn's worst row as a multiple of the envelope and counts rows beyond it.
    #[test]
    fn burn_envelope_ratio_reports_the_worst_row_against_the_envelope() {
        let kind = RowKind::Cached;
        let envelope = synthetic_envelope(kind);
        assert_eq!(envelope.envelope_graph, OfficialF16Graph::MpsEager);
        let (_, largest) = envelope.largest_leave_one_out();
        assert!((largest - chord(0.012) / chord(0.010)).abs() < 1e-5);
        let mut angles = vec![0.006; kind.rows()];
        angles[22] = 0.0124;
        angles[5] = 0.0121;
        let burn = synthetic_burn(kind, plane_cosine(0.0125), &angles);
        let ratio = BurnEnvelopeRatio::measure(&envelope, &burn);
        assert_eq!(ratio.worst.decision, 22);
        assert_eq!(ratio.deviation, normalized_deviation(ratio.worst.cosine));
        assert_eq!(ratio.of_envelope, ratio.deviation / envelope.envelope);
        assert!((ratio.of_envelope - chord(0.0124) / chord(0.012)).abs() < 1e-4);
        assert_eq!((ratio.rows_beyond_envelope, ratio.rows), (2, kind.rows()));
        assert!(
            !ratio.beyond_every_official_graph(),
            "1.033 of the envelope is inside the official graphs' largest ratio 1.2"
        );
        ratio.report("synthetic");

        angles[22] = 0.0150;
        let burn = synthetic_burn(kind, plane_cosine(0.0125), &angles);
        let ratio = BurnEnvelopeRatio::measure(&envelope, &burn);
        assert!(ratio.of_envelope > largest);
        assert!(ratio.beyond_every_official_graph());
        ratio.report("synthetic");

        let within = synthetic_burn(kind, plane_cosine(0.0125), &vec![0.005; kind.rows()]);
        let ratio = BurnEnvelopeRatio::measure(&envelope, &within);
        assert_eq!(ratio.rows_beyond_envelope, 0);
        assert!(ratio.of_envelope < 1.0);

        let other_path = synthetic_burn(RowKind::Prompt, 0.99, &[0.005]);
        assert!(catch_unwind(|| BurnEnvelopeRatio::measure(&envelope, &other_path)).is_err());
    }

    /// The floor alone decides: the envelope never admits a row below the floor or rejects one above it.
    #[test]
    fn the_envelope_diagnostic_never_changes_the_floor_verdict() {
        let kind = RowKind::Cached;
        let envelope = synthetic_envelope(kind);
        let slack_bound = envelope.registered_method_slack() * envelope.envelope;
        // Below the floor yet within the not-adopted `s E` bound, as Burn's cached decision 22 is.
        let mut angles = vec![0.006; kind.rows()];
        angles[22] = 0.0124;
        let failing = synthetic_burn(kind, plane_cosine(0.0122), &angles);
        let ratio = BurnEnvelopeRatio::measure(&envelope, &failing);
        assert!(ratio.of_envelope > 1.0 && ratio.deviation <= slack_bound);
        let violations = failing.violations();
        assert_eq!(violations.len(), 1, "{violations:?}");
        assert!(violations[0].starts_with("Cached decision 22: cosine"));
        assert!(catch_unwind(|| assert_within_floors("synthetic", &[&failing])).is_err());
        // At or above the floor yet outside the envelope: passes.
        let passing = synthetic_burn(kind, plane_cosine(0.0124), &angles);
        assert!(BurnEnvelopeRatio::measure(&envelope, &passing).of_envelope > 1.0);
        assert!(passing.violations().is_empty());
        assert_within_floors("synthetic", &[&passing]);
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
