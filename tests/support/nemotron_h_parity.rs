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
//! # Envelope rule
//!
//! For a row `a` and its F32 truth row `r`, with the cosine `c` accumulated in
//! F64 ([`cosine_similarity`]), the *normalized deviation* is
//!
//! ```text
//! delta(a, r) = | a/|a| - r/|r| |_2 = sqrt(2 (1 - c))
//! ```
//!
//! the L2 distance between the unit-normalized rows ([`normalized_deviation`]):
//! a distance, like the `|a - r|` statistics of the Llama-family and Gemma
//! envelope, and a monotone function of the cosine. For each path `k`
//! (prompt, teacher-forced, cached) the four official F16 graphs
//! ([`OfficialF16Graph`]: CPU and MPS, each with eager and SDPA attention)
//! each have a worst row `D[g, k] = max delta`, and the path's *envelope* is
//! `E[k] = max_g D[g, k]` ([`PathEnvelope`]). Every Burn row of the path must
//! have exact top-1 and
//!
//! ```text
//! delta(burn, truth) <= s[k] E[k]      (inclusive),
//! ```
//!
//! that is, a cosine of at least `1 - (s[k] E[k])^2 / 2`, which must stay at
//! or above [`MINIMUM_HALF_PRECISION_FLOOR`]. The bound is per path, not per
//! decision: half-precision deviations spike at implementation-specific rows.
//!
//! # Slack
//!
//! The slacks [`PROMPT_SLACK`], [`TEACHER_SLACK`], and [`CACHED_SLACK`] come
//! from the official graphs only, never from Burn: graph `g`'s
//! *leave-one-out* ratio `D[g, k] / max_{h != g} D[h, k]` is how far the
//! envelope of the other three must stretch to admit it, and `s[k]` is the
//! largest such ratio of the path rounded up to two significant digits
//! ([`round_up_two_significant`], the envelope round's rounding), as
//! `scripts/derive_nemotron_h_envelope.py` derives them. The method was
//! registered before any new official graph or Burn row
//! (`acceptance/measurements/nemotron-h-envelope-20261018/derivation/method.txt`);
//! the slacks must be re-derived, never tuned, whenever a graph, a version,
//! the prompt, or the checkpoint changes, and
//! [`PathEnvelope::assert_leave_one_out`] fails if the compiled constants are
//! not exactly the derivation's.

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

/// Lowest implied cosine floor accepted; a looser official F16 envelope is too wide to gate on.
pub(super) const MINIMUM_HALF_PRECISION_FLOOR: f64 = 0.99;

/// Slack of the prompt row's envelope: the largest leave-one-out ratio `1.0002` (CPU eager), rounded up.
pub(super) const PROMPT_SLACK: f64 = 1.1;

/// Slack of the teacher-forced rows' envelope: the largest leave-one-out ratio `1.0042` (CPU SDPA), rounded up.
pub(super) const TEACHER_SLACK: f64 = 1.1;

/// Slack of the cached rows' envelope: the largest leave-one-out ratio `1.0046` (MPS SDPA), rounded up.
pub(super) const CACHED_SLACK: f64 = 1.1;

/// Returns the compiled slack of one row family.
pub(super) const fn slack(kind: RowKind) -> f64 {
    match kind {
        RowKind::Prompt => PROMPT_SLACK,
        RowKind::Teacher => TEACHER_SLACK,
        RowKind::Cached => CACHED_SLACK,
    }
}

/// One of the four official F16 graphs whose envelope bounds each path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum OfficialF16Graph {
    /// CPU, eager attention: the precision bundle's `float16` rows.
    CpuEager,
    /// CPU, SDPA attention: the CPU SDPA graph bundle.
    CpuSdpa,
    /// MPS, eager attention: the MPS/F16 oracle.
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

/// Whether `value <= bound`; a NaN on either side never is, so it fails every bound.
fn at_most(value: f64, bound: f64) -> bool {
    matches!(
        value.partial_cmp(&bound),
        Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)
    )
}

/// Returns `sqrt(2 (1 - c))`, the distance between two unit-normalized rows whose cosine is `c`.
///
/// A cosine rounded above 1 reads as 1, so equal rows have deviation 0.
pub(super) fn normalized_deviation(cosine: f64) -> f64 {
    assert!(
        cosine.is_finite(),
        "a normalized deviation needs a finite cosine"
    );
    (2.0 * (1.0 - cosine).max(0.0)).sqrt()
}

/// Rounds a positive `value` up to two significant digits of its shortest round-trip decimal.
///
/// This is `scripts/derive_half_precision_envelope.py`'s `round_up`: `1.4206`
/// becomes `1.5`, `1.2988` becomes `1.3`, and `1.3` stays `1.3`, because the
/// shortest decimal that reads back as the F64 `1.3` is `1.3` itself.
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

/// The four official F16 graphs' worst rows on one path, their envelope, and the bound it sets.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct PathEnvelope {
    /// Row family the envelope bounds.
    pub(super) kind: RowKind,
    /// Each graph's worst row, in [`OfficialF16Graph::ALL`] order.
    pub(super) graphs: [GraphWorstRow; 4],
    /// `E[k]`, the largest of the graphs' worst deviations.
    pub(super) envelope: f64,
    /// Graph attaining the envelope (first in [`OfficialF16Graph::ALL`] order on ties).
    pub(super) envelope_graph: OfficialF16Graph,
    /// `s[k]` applied to the envelope.
    pub(super) slack: f64,
    /// `s[k] E[k]`, the largest normalized deviation a Burn row may have.
    pub(super) bound: f64,
    /// `1 - bound^2 / 2`, the equivalent cosine floor.
    pub(super) floor: f64,
}

impl PathEnvelope {
    /// Measures the official graphs of `kind` against its F32 truth under the compiled slack.
    ///
    /// `truth` and every `official` matrix are row-major `[rows, width]`
    /// matrices of the same path, in [`OfficialF16Graph::ALL`] order.
    ///
    /// # Panics
    ///
    /// As [`Self::with_slack`].
    pub(super) fn measure(
        label: &str,
        kind: RowKind,
        truth: &[f32],
        official: [&[f32]; 4],
        width: usize,
    ) -> Self {
        Self::with_slack(label, kind, truth, official, width, slack(kind))
    }

    /// Measures the official graphs of `kind` and applies `slack` to their envelope.
    ///
    /// # Panics
    ///
    /// If a matrix holds a different number of values than `kind`'s rows of
    /// `width`, a non-finite value or a zero row, if the envelope is zero
    /// (four graphs equal to the truth are not half-precision evaluations),
    /// or if the implied cosine floor is below [`MINIMUM_HALF_PRECISION_FLOOR`].
    pub(super) fn with_slack(
        label: &str,
        kind: RowKind,
        truth: &[f32],
        official: [&[f32]; 4],
        width: usize,
        slack: f64,
    ) -> Self {
        assert!(width > 0, "{label} row width must be nonzero");
        assert_eq!(
            truth.len(),
            kind.rows() * width,
            "{label} truth must hold every {kind:?} row"
        );
        assert_finite(label, "F32 truth", truth);
        assert!(
            slack.is_finite() && slack >= 1.0,
            "{label} slack {slack} must be finite and at least 1"
        );
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
            "{label} official F16 graphs equal the F32 truth, so they cannot set a half-precision envelope"
        );
        let bound = slack * largest.deviation;
        let floor = 1.0 - bound * bound / 2.0;
        assert!(
            (MINIMUM_HALF_PRECISION_FLOOR..=1.0).contains(&floor),
            "{label} official F16 envelope floor {floor} is outside [{MINIMUM_HALF_PRECISION_FLOOR}, 1]"
        );
        Self {
            kind,
            graphs,
            envelope: largest.deviation,
            envelope_graph: OfficialF16Graph::ALL[index],
            slack,
            bound,
            floor,
        }
    }

    /// Each graph's worst deviation over the largest of the other three graphs' worst deviations.
    ///
    /// A zero denominator gives an infinite or NaN ratio, which no slack admits.
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

    /// Prints the envelope and every leave-one-out ratio.
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
        eprintln!(
            "{label} {:?}: envelope {:.6e} ({}), slack {}, bound {:.6e}, cosine floor {:.9}",
            self.kind,
            self.envelope,
            self.envelope_graph.label(),
            self.slack,
            self.bound,
            self.floor,
        );
    }

    /// Asserts that every official graph lies within the envelope of the other three and that the slack is the derivation's.
    ///
    /// # Panics
    ///
    /// If any leave-one-out ratio exceeds the slack, or the slack is not the
    /// largest ratio rounded up to two significant digits: the slacks must then
    /// be re-derived with `scripts/derive_nemotron_h_envelope.py`, never tuned.
    pub(super) fn assert_leave_one_out(&self, label: &str) {
        let ratios = self.leave_one_out();
        let outside = OfficialF16Graph::ALL
            .iter()
            .zip(ratios)
            .filter(|(_, ratio)| !at_most(*ratio, self.slack))
            .map(|(graph, ratio)| format!("{} {ratio:.4}", graph.label()))
            .collect::<Vec<_>>();
        assert!(
            outside.is_empty(),
            "{label} {:?}: official graphs outside the envelope of the other three under slack {}: {}; \
             re-derive the slacks with scripts/derive_nemotron_h_envelope.py",
            self.kind,
            self.slack,
            outside.join(", ")
        );
        let (graph, largest) = self.largest_leave_one_out();
        let derived = round_up_two_significant(largest);
        assert_eq!(
            derived,
            self.slack,
            "{label} {:?}: the largest leave-one-out ratio {largest} ({}) derives slack {derived}, not the compiled {}",
            self.kind,
            graph.label(),
            self.slack
        );
    }
}

/// One Burn row measured against its F32 truth row, before any bound is applied.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct TruthRowObservation {
    /// Zero-based decision.
    pub(super) decision: usize,
    /// F64-accumulated cosine against the F32 truth row.
    pub(super) cosine: f64,
    /// [`normalized_deviation`] of that cosine.
    pub(super) deviation: f64,
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
/// oracle predicted from). Burn's top-1 and distance are recorded for
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
    let cosine = cosine_similarity(actual, truth.values());
    TruthRowObservation {
        decision: truth.decision(),
        cosine,
        deviation: normalized_deviation(cosine),
        actual_top1_token_id: stable_top1(actual),
        expected_top1_token_id: expected_top1,
    }
}

/// Ordered rows of one family measured against its F32 truth under one envelope.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct TruthPathSummary {
    /// Envelope whose bound the rows must meet.
    envelope: PathEnvelope,
    /// Rows observed so far, in decision order.
    rows: Vec<TruthRowObservation>,
}

impl TruthPathSummary {
    /// Starts an empty sequence judged by `envelope`.
    pub(super) const fn new(envelope: PathEnvelope) -> Self {
        Self {
            envelope,
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
        let observation =
            measure_row_against_truth(label, self.envelope.kind, actual, truth, expected_top1);
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
            envelope: self.envelope,
            rows: self.rows,
        }
    }
}

/// Every row of one fully covered family, with the envelope that judges it.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct CompletedTruthSummary {
    /// Envelope and bound applied.
    pub(super) envelope: PathEnvelope,
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

    /// Describes every row that misses its top-1 token or the envelope bound.
    pub(super) fn violations(&self) -> Vec<String> {
        let mut out = Vec::new();
        for row in &self.rows {
            if row.actual_top1_token_id != row.expected_top1_token_id {
                out.push(format!(
                    "{:?} decision {}: top-1 {} instead of {}",
                    self.envelope.kind,
                    row.decision,
                    row.actual_top1_token_id,
                    row.expected_top1_token_id
                ));
            }
            if !at_most(row.deviation, self.envelope.bound) {
                out.push(format!(
                    "{:?} decision {}: cosine {} (delta {:.6e}, {:.4} of the envelope) against the F32 truth is \
                     below the official F16 envelope floor {} (bound {:.6e} = slack {} x envelope {:.6e} from {})",
                    self.envelope.kind,
                    row.decision,
                    row.cosine,
                    row.deviation,
                    row.deviation / self.envelope.envelope,
                    self.envelope.floor,
                    self.envelope.bound,
                    self.envelope.slack,
                    self.envelope.envelope,
                    self.envelope.envelope_graph.label(),
                ));
            }
        }
        out
    }

    /// Prints every row's cosine and its distance relative to the envelope.
    pub(super) fn report(&self, label: &str) {
        let worst = self.worst();
        eprintln!(
            "{label} {:?}: {} rows, minimum cosine {:.12} at decision {} (delta {:.6e}, {:.4} of the envelope, \
             {:.4} of the bound {:.6e}; floor {:.9})",
            self.envelope.kind,
            self.rows.len(),
            worst.cosine,
            worst.decision,
            worst.deviation,
            worst.deviation / self.envelope.envelope,
            worst.deviation / self.envelope.bound,
            self.envelope.bound,
            self.envelope.floor,
        );
        let cosines = self
            .rows
            .iter()
            .map(|row| format!("{:.9}", row.cosine))
            .collect::<Vec<_>>();
        eprintln!(
            "{label} {:?} cosines: [{}]",
            self.envelope.kind,
            cosines.join(", ")
        );
    }
}

/// Asserts that every completed family meets its top-1 and envelope contract, listing every violation.
///
/// # Panics
///
/// If any row of any family has another top-1 token or exceeds its bound.
pub(super) fn assert_within_envelopes(label: &str, summaries: &[&CompletedTruthSummary]) {
    let violations = summaries
        .iter()
        .flat_map(|summary| summary.violations())
        .collect::<Vec<_>>();
    assert!(
        violations.is_empty(),
        "{label}: {} row(s) outside the official F16 envelope contract: {}",
        violations.len(),
        violations.join("; ")
    );
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
    }

    /// Bounds are inclusive, and a NaN never meets one.
    #[test]
    fn bounds_are_inclusive_and_refuse_nan() {
        assert!(at_most(1.0, 1.0));
        assert!(at_most(0.5, 1.0));
        assert!(!at_most(1.0 + f64::EPSILON, 1.0));
        assert!(!at_most(f64::NAN, 1.0));
        assert!(!at_most(0.5, f64::NAN));
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

    /// The compiled slacks are two-significant-digit values of at least one, one per family.
    #[test]
    fn compiled_slacks_are_rounded_and_per_path() {
        for (kind, value) in [
            (RowKind::Prompt, PROMPT_SLACK),
            (RowKind::Teacher, TEACHER_SLACK),
            (RowKind::Cached, CACHED_SLACK),
        ] {
            assert_eq!(slack(kind), value);
            assert!(value >= 1.0);
            assert_eq!(round_up_two_significant(value), value);
        }
        assert_eq!(
            OfficialF16Graph::ALL.map(OfficialF16Graph::label),
            ["cpu-eager", "cpu-sdpa", "mps-eager", "mps-sdpa"]
        );
    }

    /// The envelope is the largest of each graph's worst row, and the bound its slack multiple.
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
        let envelope = PathEnvelope::with_slack(
            "synthetic",
            kind,
            &truth,
            [&graphs[0], &graphs[1], &graphs[2], &graphs[3]],
            2,
            2.5,
        );
        assert_eq!(envelope.envelope_graph, OfficialF16Graph::MpsEager);
        assert_eq!(envelope.graphs[2].decision, 18);
        assert_eq!(envelope.graphs[1].decision, 7);
        assert_eq!(envelope.graphs[0].decision, 0, "the first row wins ties");
        assert!((envelope.envelope - chord(0.030)).abs() < 1e-7);
        assert_eq!(envelope.bound, 2.5 * envelope.envelope);
        assert_eq!(envelope.floor, 1.0 - envelope.bound * envelope.bound / 2.0);
        let ratios = envelope.leave_one_out();
        assert!((ratios[2] - chord(0.030) / chord(0.012)).abs() < 1e-5);
        assert!((ratios[0] - chord(0.010) / chord(0.030)).abs() < 1e-5);
        assert_eq!(
            envelope.largest_leave_one_out(),
            (OfficialF16Graph::MpsEager, ratios[2])
        );
        envelope.assert_leave_one_out("synthetic");
        envelope.report("synthetic");
    }

    /// Only the derived slack passes the leave-one-out check; a wider or narrower one fails.
    #[test]
    fn leave_one_out_check_requires_the_derived_slack() {
        let kind = RowKind::Prompt;
        let truth = plane_rows(kind, 0.0, None);
        let graphs = [
            plane_rows(kind, 0.010, None),
            plane_rows(kind, 0.011, None),
            plane_rows(kind, 0.020, None),
            plane_rows(kind, 0.012, None),
        ];
        let measure = |slack: f64| {
            PathEnvelope::with_slack(
                "synthetic",
                kind,
                &truth,
                [&graphs[0], &graphs[1], &graphs[2], &graphs[3]],
                2,
                slack,
            )
        };
        let (_, largest) = measure(1.0).largest_leave_one_out();
        let derived = round_up_two_significant(largest);
        assert_eq!(derived, 1.7);
        measure(derived).assert_leave_one_out("synthetic");
        for wrong in [1.6, 1.8] {
            let envelope = measure(wrong);
            assert!(catch_unwind(|| envelope.assert_leave_one_out("synthetic")).is_err());
        }
    }

    /// Equal, too-wide, malformed, or non-finite official graphs and an invalid slack are refused.
    #[test]
    fn path_envelope_refuses_unusable_graphs() {
        let kind = RowKind::Prompt;
        let truth = plane_rows(kind, 0.0, None);
        let near = plane_rows(kind, 0.01, None);
        let wide = plane_rows(kind, 0.2, None);
        let mut infinite = near.clone();
        infinite[1] = f32::INFINITY;
        let attempt = |graphs: [&[f32]; 4], width: usize, slack: f64| {
            catch_unwind(|| {
                PathEnvelope::with_slack("synthetic", kind, &truth, graphs, width, slack)
            })
        };
        assert!(attempt([&near, &near, &near, &near], 2, 1.0).is_ok());
        assert!(attempt([&truth, &truth, &truth, &truth], 2, 1.0).is_err());
        assert!(attempt([&near, &near, &wide, &near], 2, 1.0).is_err());
        assert!(attempt([&near, &near, &near[..1], &near], 2, 1.0).is_err());
        assert!(attempt([&near, &near, &infinite, &near], 2, 1.0).is_err());
        assert!(attempt([&near, &near, &near, &near], 0, 1.0).is_err());
        assert!(attempt([&near, &near, &near, &near], 2, 0.9).is_err());
        assert!(attempt([&near, &near, &near, &near], 2, f64::NAN).is_err());
        let short_truth = &truth[..1];
        assert!(
            catch_unwind(|| PathEnvelope::with_slack(
                "synthetic",
                kind,
                short_truth,
                [&near[..1], &near[..1], &near[..1], &near[..1]],
                2,
                1.0
            ))
            .is_err(),
            "a truth without a whole row is refused"
        );
    }

    /// An envelope of the given kind from four graphs at known angles.
    fn synthetic_envelope(kind: RowKind) -> PathEnvelope {
        let truth = plane_rows(kind, 0.0, None);
        let graphs = [
            plane_rows(kind, 0.010, None),
            plane_rows(kind, 0.011, None),
            plane_rows(kind, 0.012, None),
            plane_rows(kind, 0.010, None),
        ];
        PathEnvelope::with_slack(
            "synthetic",
            kind,
            &truth,
            [&graphs[0], &graphs[1], &graphs[2], &graphs[3]],
            2,
            1.5,
        )
    }

    /// Rows are measured without judgment; a completed path lists every top-1 and bound violation.
    #[test]
    fn truth_rows_are_judged_after_the_whole_path() {
        let envelope = synthetic_envelope(RowKind::Cached);
        let truth_values = [1.0, 0.0];
        let mut summary = TruthPathSummary::new(envelope);
        for decision in 0..RowKind::Cached.rows() {
            let angle: f64 = match decision {
                3 => 0.019,
                5 => 0.017,
                9 => 0.0,
                _ => 0.012,
            };
            let mut actual = [angle.cos() as f32, angle.sin() as f32];
            if decision == 9 {
                actual = [0.0, 1.0];
            }
            let observation = summary.observe(
                "synthetic",
                &actual,
                TruthRow::synthetic(RowKind::Cached, decision, &truth_values),
                0,
            );
            assert_eq!(observation.decision, decision);
        }
        let completed = summary.finish(RowKind::Cached.rows());
        assert_eq!(completed.rows.len(), RowKind::Cached.rows());
        assert_eq!(completed.worst().decision, 9);
        let violations = completed.violations();
        assert_eq!(violations.len(), 3, "{violations:?}");
        assert!(violations[0].contains("decision 3") && violations[0].contains("envelope floor"));
        assert!(violations[1].contains("decision 9: top-1 1 instead of 0"));
        assert!(violations[2].contains("decision 9") && violations[2].contains("envelope floor"));
        completed.report("synthetic");
        let message = catch_unwind(|| assert_within_envelopes("synthetic", &[&completed]))
            .expect_err("violations must fail")
            .downcast::<String>()
            .expect("formatted panic");
        assert!(message.contains("3 row(s)"));

        let mut within = TruthPathSummary::new(synthetic_envelope(RowKind::Prompt));
        within.observe(
            "synthetic",
            &[0.012_f64.cos() as f32, 0.012_f64.sin() as f32],
            TruthRow::synthetic(RowKind::Prompt, 0, &truth_values),
            0,
        );
        let within = within.finish(1);
        assert!(within.violations().is_empty());
        assert_within_envelopes("synthetic", &[&within]);
    }

    /// Order, coverage, row family, the truth's authenticated token, and finiteness fail immediately.
    #[test]
    fn truth_summary_enforces_order_coverage_and_identity() {
        let values = [3.0, 1.0];
        let mut summary = TruthPathSummary::new(synthetic_envelope(RowKind::Cached));
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
        assert_eq!(summary.clone().finish(1).rows.len(), 1);
        for (kind, actual, top1) in [
            (RowKind::Teacher, &values[..], 0),
            (RowKind::Cached, &values[..], 1),
            (RowKind::Cached, &values[..1], 0),
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
