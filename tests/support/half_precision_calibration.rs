//! Official-graph envelope of the Llama-family and Gemma half-precision real-checkpoint gates.
//!
//! A half-precision (F16 or BF16) Burn run is judged against the same F32 (or
//! F64) Transformers bundle as an F32 run, called the *truth* here. Its bounds
//! are not chosen by hand: they are read at run time from the four graphs in
//! which Transformers 5.2.0 / PyTorch 2.10.0 themselves evaluate the same
//! checkpoint, prompts, and options in the run's dtype on the host, written by
//! `scripts/generate_causal_lm_reference.py`:
//!
//! | graph | `--device` | `--attn-implementation` |
//! | --- | --- | --- |
//! | CPU eager | `cpu` | `eager` |
//! | CPU SDPA | `cpu` | `sdpa` |
//! | MPS eager | `mps` | `eager` |
//! | MPS SDPA | `mps` | `sdpa` |
//!
//! each with `--decode-tokens-from TRUTH` and, for the short truths,
//! `--prompt-decode`. A tolerance that rejected one of these official graphs
//! would measure which kernel path ran, not correctness; the previous rule
//! (`1.2` times the CPU-eager graph alone) rejected 11 of 57 statistics of the
//! other official BF16 graphs, on 5 of 7 bundles. The F32 gates use only
//! [`DecodeTruth`] from this module.
//!
//! # Rule
//!
//! Write `r[t, v]` for the truth logits of one prompt at position `t` and
//! vocabulary entry `v` (`T` positions, `V` entries) and `o_g[t, v]` for
//! official graph `g`'s logits of the same tokens. Each graph's deviation is
//! summarized by
//!
//! ```text
//! M_g = max_{t,v} |o_g[t,v] - r[t,v]|
//! μ_g = (1 / (T V)) Σ_{t,v} |o_g[t,v] - r[t,v]|
//! ```
//!
//! with each difference of two F32 values taken in F64 (exact unless their
//! binary exponents differ by more than 29) and the sum accumulated in F64
//! ([`Deviation::measure`]). The *envelope* of the prompt is
//! `E_max = max_g M_g` and `E_mean = max_g μ_g` over the four graphs
//! ([`GraphEnvelope`]). A half-precision run with logits `b` passes the prompt
//! when
//!
//! ```text
//! max_{t,v} |b[t,v] - r[t,v]| <= s_max E_max   and   mean_{t,v} |b[t,v] - r[t,v]| <= s_mean E_mean
//! ```
//!
//! both inclusive ([`Bound`]), with the dtype's slacks [`MAX_SLACK`] and
//! [`MEAN_SLACK`]. A check of single rows of the prompt (the last position of
//! a cached prefill, of a batched row, or of a long-prompt chunk) applies
//! `s_max E_max` alone, because the mean is defined over all of the prompt's
//! positions. An envelope with `E_max = 0` or `E_mean = 0` is refused: four
//! graphs equal to the truth cannot be half-precision evaluations, and a zero
//! bound would reject every run.
//!
//! The rule cannot see one extra rounding of one operation: the official
//! graphs themselves spread by up to `1.42` (maximum) and `1.30` (mean)
//! leave-one-out in BF16. Exact operation-level tests (the "rounding points"
//! tests of `src/feed_forward_tests.rs` and its siblings) settle such defects;
//! a logit gate bounds whole-model drift.
//!
//! # Slack
//!
//! The slacks are derived from the official graphs only, never from Burn
//! data. A graph's *leave-one-out* ratio is its statistic over the largest of
//! the other three graphs' statistic: how far the envelope of three official
//! graphs must stretch to admit the fourth. `s_max` and `s_mean` are the
//! largest such ratios of maxima and of means, pooled over every prompt,
//! decode-row ([Decoding](self#decoding)), and prompt-decode statistic of
//! every calibrated truth of the dtype, rounded up to two significant digits
//! (`scripts/derive_half_precision_envelope.py`, which rounds the shortest
//! decimal of each ratio). The truths are SmolLM-135M `1d461723`, Qwen3-4B
//! `531c80e2`, SmolLM3-3B `1fa6fa28`, Qwen2-7B `453ed157` (F64), Llama-3.1-8B
//! `d04e592b`, and gemma-3-4b-it `093f9f38` (short and 1237-token prompts);
//! the derivation and every bundle digest are recorded in
//! `acceptance/measurements/half-precision-envelope-20261018.json`:
//!
//! | dtype | statistic | largest leave-one-out ratio (where) | slack |
//! | --- | --- | --- | --- |
//! | BF16 | maximum | `1.4206` (gemma-3-4b-it prompt 2, MPS SDPA) | `1.5` |
//! | BF16 | mean | `1.2988` (Qwen3-4B prompt 2, CPU eager) | `1.3` |
//! | F16 | maximum | `1.6660` (Qwen3-4B prompt 0, MPS eager) | `1.7` |
//! | F16 | mean | `1.6981` (Qwen3-4B prompt 0, MPS eager) | `1.7` |
//!
//! In BF16 the decode rows reach `1.2694` (maximum) and `1.1141` (mean) and
//! the prompt-decode rows `1.3045` and `1.1318`, so the prompt statistics
//! decide; the mean slack has only `0.09%` headroom over its derivation. In
//! F16 one graph decides both slacks: MPS eager on Qwen3-4B's prompt 0, whose
//! deviation is `1.67` and `1.70` times the other three graphs' largest;
//! without it the largest ratios would be `1.3774` (maximum, SmolLM-135M
//! decode rows) and `1.5060` (mean, Qwen3-4B decode rows). The F16 slacks
//! rest on four calibrated truths (80 ratios), because Qwen2-7B and both
//! gemma-3-4b-it truths have no F16 envelope ([Uncalibrated
//! truths](self#uncalibrated-truths)); BF16's rest on seven (128 ratios).
//! Both pairs are calibration constants, to be re-derived by that procedure,
//! never tuned, whenever a graph, a version, a prompt, or a checkpoint
//! changes; the bundle-only test `official_graphs_admit_each_other_leave_one_out`
//! fails if any official graph falls outside the envelope of the other three
//! under these constants, which means exactly that.
//!
//! A leave-one-out maximum over about a hundred ratios leaves an exchangeable
//! new implementation roughly a one-in-a-hundred chance per statistic of
//! exceeding it. An exceedance is therefore a finding to diagnose with
//! per-block dumps and operation-level comparison, never an automatic defect
//! verdict and never a reason to widen the rule.
//!
//! # F16 runs
//!
//! F16 is calibrated exactly like BF16, on the four official F16 graphs, with
//! its own slacks: no fixed scale relates the two dtypes' deviations (on
//! gpt-oss-20b, Metal/F16 reached `0.007` to `0.874` of the official BF16
//! statistics), so neither the BF16 envelope nor a scaled copy of it bounds an
//! F16 run.
//!
//! # Uncalibrated truths
//!
//! If any official graph of the run's dtype holds a non-finite logit in any
//! row this module reads (prompts, decode rows, prompt-decode rows), the truth
//! has no envelope in that dtype: an official evaluation that overflows marks
//! the checkpoint as range-marginal there, and a finite Burn result would
//! reflect kernel luck rather than fidelity. Building the gate then panics with
//! `uncalibrated: official <graph> <dtype> non-finite at ...` before the
//! checkpoint loads. That is a failure, never a pass. On the recorded bundles
//! this applies to Qwen2-7B and gemma-3-4b-it in F16.
//!
//! # Decoding
//!
//! The truth holds no logits beyond its prompts, so each truth is paired with
//! a *decode truth*: the same truth graph (CPU eager, F32 or F64 like the
//! truth, its prompt logits byte-identical to the truth's) evaluated with
//! `--decode-tokens-from TRUTH`, whose stateless rows at the truth's
//! teacher-forced greedy positions `T, ..., T + steps - 1` are the reference
//! decode rows ([`DecodeTruth`]). The decode envelope `E_dec` is the largest
//! deviation of the four official graphs' cached and stateless rows from those
//! rows. A half-precision run's cached decode rows, and its stateless rows at
//! the same positions, must each meet `s_max E_dec,max` and
//! `s_mean E_dec,mean` ([`HalfPrecisionGate::check_decode`]); their gap from
//! each other is then bounded by `2 s_max E_dec,max` through the truth and is
//! printed as a diagnostic only. This catches decode defects that
//! self-consistency cannot see, such as a decode that is equally wrong in both
//! passes. F32 runs compare their cached decode rows with the same rows at
//! their F32 tolerance ([`DecodeTruth::check_f32`]).
//!
//! The longest prompt (the first of equal ones), prefilled for its first
//! `max(T / 4, 1)` tokens and then decoded token by token, as gpt-oss's cached
//! long-prompt test does, is judged against the envelope of the official
//! graphs' own cached decode under the same protocol (`--prompt-decode`), not
//! against the stateless envelope: cached decoding rounds differently, and on
//! SmolLM-135M the official MPS cached decode reached `1.45` times the
//! stateless `E_max` ([`HalfPrecisionGate::check_prompt_decode`]).
//!
//! Greedy steps of the cached continuation of prompt 0 are compared only when
//! the truth's top-2 margin exceeds twice
//! `max(s_max E_max of prompt 0, s_max E_dec,max)`, the bound of the rows the
//! comparison reads ([`Envelope::greedy_bound`]): the decode envelope exceeds
//! the prompt-0 envelope on gemma-3-4b-it and Llama-3.1-8B. If every logit lies
//! within that bound, the gap between the truth's best and second tokens
//! shrinks by at most twice it, so only such a step has a determined argmax; a
//! margin exactly at the threshold is a tie and is not compared. The
//! continuation is teacher-forced, so a tie is skipped rather than ending the
//! comparison ([`decisive_steps`]). This is the rule the F32 gates share
//! (`tests/support/backend_numerics.rs`, "Greedy ties"), applied at
//! [`HalfPrecisionGate::greedy_tolerance`]; see [`largest_f32_not_above`] for
//! the range in which both decide the same steps, which every recorded bound
//! satisfies.
//!
//! # Self-consistency
//!
//! A cached prefill, each end of a chunked cached prefill (Gemma's long
//! prompt), and every row of a left-padded batch must still agree with the
//! same run's stateless pass within [`SELF_CONSISTENCY_TOLERANCE`]. These
//! checks are provisional and uncalibrated: no official measurement of
//! Transformers' own batched or chunked rows exists yet (Burn measured `0` to
//! `0.0625` there), and the value `0.5` is gpt-oss's batched-row tolerance.
//! Transformers' own BF16 cached decode exceeds it (up to `1.75` on Qwen2-7B),
//! which is why decode steps are no longer judged by it. The checks stay until
//! the generator records the official graphs' own batched and chunked rows.
//!
//! # Pairing
//!
//! [`validate_envelope`] fails closed. Every manifest must be a schema-1
//! bundle ([`Manifest::parse`]). The truth must be a `float32` or `float64`
//! evaluation whose greedy continuation is complete. The official graphs must
//! be exactly the four `(device, attention_implementation)` pairs above, each
//! once, each a plain (not `--bf16-resident`) evaluation in the run's dtype,
//! all written by one generator script (one `script_sha256`). Each must agree
//! with the truth on the model type and class, the `config.json` and
//! weight-file SHA-256s, repository and revision, Transformers and PyTorch
//! versions, vocabulary, layer truncation, every prompt's text and token IDs,
//! and the greedy length (shorter only where its continuation stopped on a
//! non-finite step); its decode rows must feed the truth's greedy tokens at
//! positions `T, ..., T + steps - 1` and name the truth's exact manifest
//! SHA-256 as their token source; its prompt-decode record, if any graph has
//! one, must exist in all four, name the longest prompt, and prefill
//! `max(T / 4, 1)` tokens. The decode truth must be a plain `float32` or
//! `float64` CPU-eager evaluation with the truth's identity, prompt-logit
//! SHA-256s, decode positions, and token source. Every logits file must match
//! its recorded SHA-256 and size.

use std::{
    collections::BTreeMap,
    ffi::OsStr,
    path::{Path, PathBuf},
};

use serde_json::Value;
use sha2::{Digest, Sha256};
use transformers::checkpoint::TensorDType;

/// Slack `s_max` on maximum statistics, per run dtype; see [Slack](self#slack).
pub(crate) const MAX_SLACK: [(TensorDType, f64); 2] =
    [(TensorDType::BF16, 1.5), (TensorDType::F16, 1.7)];

/// Slack `s_mean` on mean statistics, per run dtype; see [Slack](self#slack).
pub(crate) const MEAN_SLACK: [(TensorDType, f64); 2] =
    [(TensorDType::BF16, 1.3), (TensorDType::F16, 1.7)];

/// Provisional absolute agreement of cached prefills, chunk ends, and batched rows with the stateless pass.
///
/// The values are the gpt-oss gate's (`tests/gpt_oss_artifacts.rs`), where a
/// left-padded batch and each prompt alone differed by at most `0.125` (BF16)
/// and `0.41` (F16) on four real blocks on Flex. They are uncalibrated on the
/// Llama-family and Gemma checkpoints; see
/// [Self-consistency](self#self-consistency).
pub(crate) const SELF_CONSISTENCY_TOLERANCE: [(TensorDType, f64); 2] =
    [(TensorDType::F16, 0.5), (TensorDType::BF16, 0.5)];

/// Bundle schema version written by `scripts/generate_causal_lm_reference.py`.
const SCHEMA_VERSION: u64 = 1;

/// The values a run-dtype variable accepts, for messages.
const RUN_DTYPES: &str = "f32 (the default), f16, or bf16";

/// Returns the stored dtype a run-dtype variable names: F32 when `value` is unset.
///
/// # Panics
///
/// If `value` is set to anything but exactly `f32`, `f16`, or `bf16`.
pub(crate) fn parse_run_dtype(variable: &str, value: Option<&OsStr>) -> TensorDType {
    let Some(value) = value else {
        return TensorDType::F32;
    };
    match value.to_str() {
        Some("f32") => TensorDType::F32,
        Some("f16") => TensorDType::F16,
        Some("bf16") => TensorDType::BF16,
        _ => panic!(
            "{variable} must be {RUN_DTYPES}, not {:?}",
            value.to_string_lossy()
        ),
    }
}

/// Reads the stored dtype from the environment variable `variable` (see [`parse_run_dtype`]).
pub(crate) fn run_dtype(variable: &str) -> TensorDType {
    parse_run_dtype(variable, std::env::var_os(variable).as_deref())
}

/// Whether a run in `dtype` is judged by the official half-precision envelope.
pub(crate) fn is_half_precision(dtype: TensorDType) -> bool {
    matches!(dtype, TensorDType::F16 | TensorDType::BF16)
}

/// Returns the names in `names` that `is_set` reports as set, in order.
pub(crate) fn present_overrides<'a>(
    names: &[&'a str],
    is_set: impl Fn(&str) -> bool,
) -> Vec<&'a str> {
    names.iter().copied().filter(|name| is_set(name)).collect()
}

/// Refuses F32 tolerance overrides in a half-precision run.
///
/// # Panics
///
/// If any environment variable in `names` is set: a half-precision run takes
/// its bounds from the official graphs, which no override may replace.
pub(crate) fn reject_overrides(names: &[&str]) {
    let present = present_overrides(names, |name| std::env::var_os(name).is_some());
    assert!(
        present.is_empty(),
        "{present:?} override F32 tolerances; a half-precision run takes its bounds from the official graphs, so unset them"
    );
}

/// Returns the `(retired, replacement)` pairs whose retired variable `is_set` reports as set, in order.
pub(crate) fn present_retired<'a>(
    pairs: &[(&'a str, &'a str)],
    is_set: impl Fn(&str) -> bool,
) -> Vec<(&'a str, &'a str)> {
    pairs
        .iter()
        .copied()
        .filter(|(retired, _)| is_set(retired))
        .collect()
}

/// Refuses retired environment variables, naming each one's replacement.
///
/// # Panics
///
/// If the retired variable of any `(retired, replacement)` pair is set: one
/// official bundle no longer calibrates a run, and silently ignoring the
/// variable would hide that the run is judged by other bounds.
pub(crate) fn refuse_retired(pairs: &[(&str, &str)]) {
    let present = present_retired(pairs, |name| std::env::var_os(name).is_some());
    assert!(
        present.is_empty(),
        "{} retired: a half-precision run is judged by the envelope of the four official graphs; \
         name them in {} (see tests/support/half_precision_calibration.rs)",
        present
            .iter()
            .map(|(retired, _)| *retired)
            .collect::<Vec<_>>()
            .join(", "),
        present
            .iter()
            .map(|(_, replacement)| *replacement)
            .collect::<Vec<_>>()
            .join(", ")
    );
}

/// Returns the directory named by the environment variable `name`.
///
/// # Panics
///
/// If the variable is unset.
pub(crate) fn required_directory(name: &str) -> PathBuf {
    std::env::var_os(name)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("set {name}; see the module documentation"))
}

/// Returns the directories in a path-list value (separated as in `PATH`), in order, skipping empty entries.
pub(crate) fn split_directories(value: &OsStr) -> Vec<PathBuf> {
    std::env::split_paths(value)
        .filter(|path| !path.as_os_str().is_empty())
        .collect()
}

/// Returns the directories the environment variable `name` lists (see [`split_directories`]).
///
/// # Panics
///
/// If the variable is unset or lists no directory.
pub(crate) fn required_directories(name: &str) -> Vec<PathBuf> {
    let value = std::env::var_os(name)
        .unwrap_or_else(|| panic!("set {name}; see the module documentation"));
    let directories = split_directories(&value);
    assert!(!directories.is_empty(), "{name} lists no directory");
    directories
}

/// Lower-case hexadecimal digest.
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// Returns the slack of `dtype` in `table`, naming `what` in the panic.
///
/// # Panics
///
/// For a dtype the table has no entry for: F32 runs keep their own gates.
fn slack_of(table: &[(TensorDType, f64)], dtype: TensorDType, what: &str) -> f64 {
    table
        .iter()
        .find(|(candidate, _)| *candidate == dtype)
        .map(|&(_, slack)| slack)
        .unwrap_or_else(|| panic!("no half-precision {what} slack for {dtype:?}"))
}

/// Returns `s_max` of a half-precision `dtype` ([`MAX_SLACK`]).
///
/// # Panics
///
/// For any other dtype.
pub(crate) fn max_slack(dtype: TensorDType) -> f64 {
    slack_of(&MAX_SLACK, dtype, "maximum")
}

/// Returns `s_mean` of a half-precision `dtype` ([`MEAN_SLACK`]).
///
/// # Panics
///
/// For any other dtype.
pub(crate) fn mean_slack(dtype: TensorDType) -> f64 {
    slack_of(&MEAN_SLACK, dtype, "mean")
}

/// Evaluation dtype recorded in a bundle's provenance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BundleDtype {
    /// `float32`: a truth bundle.
    Float32,
    /// `float64`: a truth bundle without the reference's own F32 rounding.
    Float64,
    /// `bfloat16`: an official BF16 graph.
    BFloat16,
    /// `float16`: an official F16 graph.
    Float16,
}

impl BundleDtype {
    /// Parses a provenance `dtype` label.
    ///
    /// # Panics
    ///
    /// For any label but `float32`, `float64`, `bfloat16`, and `float16`.
    pub(crate) fn parse(label: &str) -> Self {
        match label {
            "float32" => Self::Float32,
            "float64" => Self::Float64,
            "bfloat16" => Self::BFloat16,
            "float16" => Self::Float16,
            other => panic!(
                "unsupported reference dtype label {other:?}; expected float32, float64, bfloat16, or float16"
            ),
        }
    }

    /// Returns the label the generator records for this dtype.
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Float32 => "float32",
            Self::Float64 => "float64",
            Self::BFloat16 => "bfloat16",
            Self::Float16 => "float16",
        }
    }

    /// Returns the official graphs' dtype that calibrates a half-precision run in `dtype`.
    ///
    /// # Panics
    ///
    /// If `dtype` is not F16 or BF16.
    pub(crate) fn of_run(dtype: TensorDType) -> Self {
        match dtype {
            TensorDType::BF16 => Self::BFloat16,
            TensorDType::F16 => Self::Float16,
            other => panic!("the official envelope judges F16 and BF16 runs, not {other:?}"),
        }
    }

    /// Whether a bundle in this dtype can be the truth that runs are judged against.
    pub(crate) fn is_truth(self) -> bool {
        matches!(self, Self::Float32 | Self::Float64)
    }
}

/// Device an official graph ran on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum GraphDevice {
    /// The CPU.
    Cpu,
    /// Apple's Metal Performance Shaders.
    Mps,
}

/// Transformers attention implementation an official graph used.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum GraphAttention {
    /// Transformers' eager attention: scores, softmax, and value product as separate operations.
    Eager,
    /// PyTorch's fused `scaled_dot_product_attention`.
    Sdpa,
}

/// One of the four official graphs that calibrate a half-precision run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct OfficialGraph {
    /// Device of the evaluation.
    pub(crate) device: GraphDevice,
    /// Attention implementation of the evaluation.
    pub(crate) attention: GraphAttention,
}

impl OfficialGraph {
    /// The four official graphs, in the order envelopes list them.
    pub(crate) const ALL: [Self; 4] = [
        Self {
            device: GraphDevice::Cpu,
            attention: GraphAttention::Eager,
        },
        Self {
            device: GraphDevice::Cpu,
            attention: GraphAttention::Sdpa,
        },
        Self {
            device: GraphDevice::Mps,
            attention: GraphAttention::Eager,
        },
        Self {
            device: GraphDevice::Mps,
            attention: GraphAttention::Sdpa,
        },
    ];

    /// Parses a manifest's `provenance.device` and `provenance.attention_implementation`; `None` for any other pair.
    pub(crate) fn parse(device: &str, attention: &str) -> Option<Self> {
        let device = match device {
            "cpu" => GraphDevice::Cpu,
            "mps" => GraphDevice::Mps,
            _ => return None,
        };
        let attention = match attention {
            "eager" => GraphAttention::Eager,
            "sdpa" => GraphAttention::Sdpa,
            _ => return None,
        };
        Some(Self { device, attention })
    }

    /// Returns the graph's label, such as `cpu-eager`.
    pub(crate) fn label(self) -> &'static str {
        match (self.device, self.attention) {
            (GraphDevice::Cpu, GraphAttention::Eager) => "cpu-eager",
            (GraphDevice::Cpu, GraphAttention::Sdpa) => "cpu-sdpa",
            (GraphDevice::Mps, GraphAttention::Eager) => "mps-eager",
            (GraphDevice::Mps, GraphAttention::Sdpa) => "mps-sdpa",
        }
    }
}

/// The first non-finite entry of a set of logits, in row-major order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct NonFinite {
    /// Row of the entry (the prompt position for prompt logits).
    pub(crate) row: usize,
    /// Vocabulary entry.
    pub(crate) token: usize,
}

/// One prompt of a bundle manifest.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct BundlePrompt {
    /// Prompt text as the generator tokenized it.
    pub(crate) text: String,
    /// Token IDs of the unpadded prompt.
    pub(crate) token_ids: Vec<i64>,
    /// Name of the logits file inside the bundle.
    pub(crate) logits_file: String,
    /// SHA-256 of the logits file.
    pub(crate) logits_sha256: String,
    /// First non-finite logit the generator recorded, `None` when every logit is finite.
    pub(crate) non_finite: Option<NonFinite>,
}

/// A manifest's `decode_consistency` record: cached and stateless rows at the truth's greedy positions.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct DecodeRecord {
    /// Position of the first decode row, prompt 0's length.
    pub(crate) first_position: usize,
    /// SHA-256 of the manifest whose greedy tokens were fed.
    pub(crate) tokens_source: String,
    /// Position of each row.
    pub(crate) positions: Vec<usize>,
    /// Token fed at each row's position.
    pub(crate) fed_tokens: Vec<i64>,
    /// File holding the cached rows.
    pub(crate) cached_file: String,
    /// SHA-256 of the cached rows.
    pub(crate) cached_sha256: String,
    /// File holding the stateless rows.
    pub(crate) stateless_file: String,
    /// SHA-256 of the stateless rows.
    pub(crate) stateless_sha256: String,
    /// Whether both passes' rows are finite (an absent flag means finite).
    pub(crate) finite: bool,
}

/// A manifest's `prompt_decode` record: the longest prompt decoded after a quarter-prompt prefill.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PromptDecodeRecord {
    /// Index of the decoded prompt.
    pub(crate) prompt: usize,
    /// Tokens prefilled before decoding one token at a time.
    pub(crate) prefill: usize,
    /// File holding the `[T, V]` rows.
    pub(crate) logits_file: String,
    /// SHA-256 of the rows.
    pub(crate) logits_sha256: String,
    /// Whether every row is finite (an absent flag means finite).
    pub(crate) finite: bool,
}

/// The fields of a bundle manifest (`reference.json`) that pairing and calibration read.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Manifest {
    /// Transformers `model_type`.
    pub(crate) model_type: String,
    /// Transformers model class that produced the logits.
    pub(crate) model_class: String,
    /// Vocabulary size `V`.
    pub(crate) vocabulary: usize,
    /// Evaluation dtype.
    pub(crate) dtype: BundleDtype,
    /// Whether BF16-resident blocks were widened while running (`--bf16-resident`).
    pub(crate) bf16_resident: bool,
    /// Blocks kept by a truncated evaluation, `None` for the whole model.
    pub(crate) layers: Option<usize>,
    /// SHA-256 of the checkpoint's `config.json`.
    pub(crate) config_sha256: String,
    /// SHA-256 of every weight file, by file name.
    pub(crate) weights_sha256: BTreeMap<String, String>,
    /// Upstream repository recorded by the generator, if any.
    pub(crate) repository: Option<String>,
    /// Upstream revision recorded by the generator, if any.
    pub(crate) revision: Option<String>,
    /// Transformers version of the evaluation.
    pub(crate) transformers: String,
    /// PyTorch version of the evaluation.
    pub(crate) torch: String,
    /// `provenance.device` as recorded.
    pub(crate) device: String,
    /// `provenance.attention_implementation` as recorded.
    pub(crate) attention: String,
    /// SHA-256 of the generator script, if recorded.
    pub(crate) script_sha256: Option<String>,
    /// Prompts in bundle order.
    pub(crate) prompts: Vec<BundlePrompt>,
    /// Greedy continuation of prompt 0 as `(token, top-2 margin)`.
    pub(crate) greedy: Vec<(i64, f32)>,
    /// Whether the greedy continuation stopped early on a non-finite step.
    pub(crate) greedy_stopped: bool,
    /// Decode rows at the truth's greedy positions, if recorded.
    pub(crate) decode: Option<DecodeRecord>,
    /// Quarter-prefill decode of the longest prompt, if recorded.
    pub(crate) prompt_decode: Option<PromptDecodeRecord>,
}

/// Returns `value[key]`, which must exist.
///
/// # Panics
///
/// If `value` has no member `key`; `path` names it in the message.
fn member<'a>(value: &'a Value, key: &str, path: &str) -> &'a Value {
    value
        .get(key)
        .unwrap_or_else(|| panic!("reference.json: missing {path}"))
}

/// Returns `value[key]` as a string.
///
/// # Panics
///
/// If the member is missing or not a string.
fn string<'a>(value: &'a Value, key: &str, path: &str) -> &'a str {
    member(value, key, path)
        .as_str()
        .unwrap_or_else(|| panic!("reference.json: {path} must be a string"))
}

/// Returns `value[key]` as an unsigned integer.
///
/// # Panics
///
/// If the member is missing or not an unsigned integer.
fn unsigned(value: &Value, key: &str, path: &str) -> u64 {
    member(value, key, path)
        .as_u64()
        .unwrap_or_else(|| panic!("reference.json: {path} must be an unsigned integer"))
}

/// Returns `value[key]` as a `usize`.
///
/// # Panics
///
/// If the member is missing or not an unsigned integer that fits `usize`.
fn index(value: &Value, key: &str, path: &str) -> usize {
    usize::try_from(unsigned(value, key, path))
        .unwrap_or_else(|_| panic!("reference.json: {path} does not fit usize"))
}

/// Returns `value[key]` as an optional string; a missing member or `null` is `None`.
///
/// # Panics
///
/// If the member is present with another type.
fn optional_text(value: &Value, key: &str, path: &str) -> Option<String> {
    match value.get(key) {
        None | Some(Value::Null) => None,
        Some(Value::String(text)) => Some(text.clone()),
        Some(_) => panic!("reference.json: {path} must be a string or null"),
    }
}

/// Returns `value["finite"]`, which defaults to `true` (the generator's "absent means finite").
///
/// # Panics
///
/// If the member is present and not a boolean.
fn finite_flag(value: &Value, path: &str) -> bool {
    match value.get("finite") {
        None => true,
        Some(Value::Bool(flag)) => *flag,
        Some(_) => panic!("reference.json: {path}.finite must be a boolean"),
    }
}

/// Returns `value[key]` as a plain file name inside the bundle.
///
/// # Panics
///
/// If the member is missing, not a string, or names a path rather than a file.
fn plain_file_name(value: &Value, key: &str, path: &str) -> String {
    let file = string(value, key, path).to_owned();
    assert!(
        !file.is_empty() && file != "." && file != ".." && !file.contains(['/', '\\']),
        "reference.json: {path} must be a plain file name, not {file:?}"
    );
    file
}

/// Requires `value[key]` to be the shape `[rows, vocabulary]`.
///
/// # Panics
///
/// If it is not exactly that shape.
fn require_shape(value: &Value, key: &str, path: &str, rows: usize, vocabulary: usize) {
    let shape: Option<Vec<u64>> = member(value, key, path)
        .as_array()
        .and_then(|shape| shape.iter().map(Value::as_u64).collect());
    assert_eq!(
        shape,
        Some(vec![rows as u64, vocabulary as u64]),
        "reference.json: {path} must be [{rows}, {vocabulary}]"
    );
}

impl Manifest {
    /// Parses a manifest and checks its internal consistency.
    ///
    /// # Panics
    ///
    /// If `text` is not a schema-1 manifest, a field is missing or mistyped, a
    /// logits shape does not match its rows, a logits file name is not a plain
    /// file name, a greedy margin is negative or not finite in F32, or an
    /// optional record (non-finite markers, `greedy.stopped`,
    /// `decode_consistency`, `prompt_decode`) is malformed.
    pub(crate) fn parse(text: &str) -> Self {
        let manifest: Value = serde_json::from_str(text)
            .unwrap_or_else(|error| panic!("reference.json is not JSON: {error}"));
        let schema_version = unsigned(&manifest, "schema_version", "schema_version");
        assert_eq!(
            schema_version, SCHEMA_VERSION,
            "reference.json: unsupported bundle schema {schema_version}"
        );
        let vocabulary = index(&manifest, "vocabulary_size", "vocabulary_size");
        assert!(
            vocabulary >= 2,
            "reference.json: a vocabulary needs at least two entries"
        );
        let provenance = member(&manifest, "provenance", "provenance");
        let bf16_resident = match provenance.get("bf16_resident") {
            None | Some(Value::Null) => false,
            Some(Value::Bool(flag)) => *flag,
            Some(_) => panic!("reference.json: provenance.bf16_resident must be a boolean"),
        };
        let layers = match provenance.get("num_hidden_layers") {
            None | Some(Value::Null) => None,
            Some(layers) => Some(
                layers
                    .as_u64()
                    .and_then(|layers| usize::try_from(layers).ok())
                    .filter(|&layers| layers > 0)
                    .expect("reference.json: provenance.num_hidden_layers must be a positive integer or null"),
            ),
        };
        let weights_sha256: BTreeMap<String, String> =
            member(provenance, "weights_sha256", "provenance.weights_sha256")
                .as_object()
                .expect("reference.json: provenance.weights_sha256 must be an object")
                .iter()
                .map(|(name, digest)| {
                    let digest = digest
                        .as_str()
                        .expect("reference.json: every weight digest must be a string");
                    (name.clone(), digest.to_owned())
                })
                .collect();
        assert!(
            !weights_sha256.is_empty(),
            "reference.json: provenance.weights_sha256 names no weight file"
        );
        let prompts: Vec<BundlePrompt> = member(&manifest, "prompts", "prompts")
            .as_array()
            .expect("reference.json: prompts must be an array")
            .iter()
            .enumerate()
            .map(|(index, prompt)| BundlePrompt::parse(prompt, index, vocabulary))
            .collect();
        assert!(
            !prompts.is_empty(),
            "reference.json: a bundle needs at least one prompt"
        );
        let greedy_record = member(&manifest, "greedy", "greedy");
        assert_eq!(
            unsigned(greedy_record, "prompt_index", "greedy.prompt_index"),
            0,
            "reference.json: the greedy continuation must extend prompt 0"
        );
        let greedy = member(greedy_record, "steps", "greedy.steps")
            .as_array()
            .expect("reference.json: greedy.steps must be an array")
            .iter()
            .map(|step| {
                let token = member(step, "token", "greedy.steps[].token")
                    .as_i64()
                    .expect("reference.json: a greedy token must be an integer");
                let margin = member(step, "margin", "greedy.steps[].margin")
                    .as_f64()
                    .expect("reference.json: a greedy margin must be a number")
                    as f32;
                assert!(
                    margin.is_finite() && margin >= 0.0,
                    "reference.json: greedy margin {margin} is negative or not finite in F32"
                );
                (token, margin)
            })
            .collect();
        let greedy_stopped = match greedy_record.get("stopped") {
            None | Some(Value::Null) => false,
            Some(Value::String(reason)) if reason == "non_finite" => true,
            Some(other) => {
                panic!("reference.json: greedy.stopped must be \"non_finite\", not {other}")
            }
        };
        let decode = manifest
            .get("decode_consistency")
            .map(|record| DecodeRecord::parse(record, vocabulary));
        let prompt_decode = manifest
            .get("prompt_decode")
            .map(|record| PromptDecodeRecord::parse(record, &prompts, vocabulary));
        Self {
            model_type: string(&manifest, "model_type", "model_type").to_owned(),
            model_class: string(&manifest, "model_class", "model_class").to_owned(),
            vocabulary,
            dtype: BundleDtype::parse(string(provenance, "dtype", "provenance.dtype")),
            bf16_resident,
            layers,
            config_sha256: string(provenance, "config_sha256", "provenance.config_sha256")
                .to_owned(),
            weights_sha256,
            repository: optional_text(provenance, "repository", "provenance.repository"),
            revision: optional_text(provenance, "revision", "provenance.revision"),
            transformers: string(provenance, "transformers", "provenance.transformers").to_owned(),
            torch: string(provenance, "torch", "provenance.torch").to_owned(),
            device: string(provenance, "device", "provenance.device").to_owned(),
            attention: string(
                provenance,
                "attention_implementation",
                "provenance.attention_implementation",
            )
            .to_owned(),
            script_sha256: optional_text(provenance, "script_sha256", "provenance.script_sha256"),
            prompts,
            greedy,
            greedy_stopped,
            decode,
            prompt_decode,
        }
    }

    /// Reads `root/reference.json`; returns the manifest and the SHA-256 of its exact bytes.
    ///
    /// # Panics
    ///
    /// If the file cannot be read or does not parse (see [`Self::parse`]).
    pub(crate) fn read(root: &Path) -> (Self, String) {
        let path = root.join("reference.json");
        let bytes =
            std::fs::read(&path).unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
        let text = std::str::from_utf8(&bytes)
            .unwrap_or_else(|error| panic!("{} is not UTF-8: {error}", path.display()));
        (Self::parse(text), hex(&Sha256::digest(&bytes)))
    }

    /// Returns the official graph this bundle was evaluated with, `None` for any other device or attention.
    pub(crate) fn graph(&self) -> Option<OfficialGraph> {
        OfficialGraph::parse(&self.device, &self.attention)
    }

    /// Returns the length of prompt 0, the position where decode rows start.
    pub(crate) fn decode_start(&self) -> usize {
        self.prompts[0].token_ids.len()
    }
}

impl BundlePrompt {
    /// Parses prompt `index` of a manifest whose vocabulary has `vocabulary` entries.
    ///
    /// # Panics
    ///
    /// As described for [`Manifest::parse`].
    fn parse(prompt: &Value, index: usize, vocabulary: usize) -> Self {
        let path = |field: &str| format!("prompts[{index}].{field}");
        let token_ids: Vec<i64> = member(prompt, "token_ids", &path("token_ids"))
            .as_array()
            .unwrap_or_else(|| panic!("reference.json: {} must be an array", path("token_ids")))
            .iter()
            .map(|token| {
                token.as_i64().unwrap_or_else(|| {
                    panic!("reference.json: {} must hold integers", path("token_ids"))
                })
            })
            .collect();
        assert!(
            !token_ids.is_empty(),
            "reference.json: {} is empty",
            path("token_ids")
        );
        require_shape(
            prompt,
            "logits_shape",
            &path("logits_shape"),
            token_ids.len(),
            vocabulary,
        );
        let non_finite = (!finite_flag(prompt, &format!("prompts[{index}]"))).then(|| {
            NonFinite::parse(
                member(prompt, "first_non_finite", &path("first_non_finite")),
                &path("first_non_finite"),
                token_ids.len(),
                vocabulary,
            )
        });
        Self {
            text: string(prompt, "text", &path("text")).to_owned(),
            logits_file: plain_file_name(prompt, "logits_file", &path("logits_file")),
            logits_sha256: string(prompt, "logits_sha256", &path("logits_sha256")).to_owned(),
            token_ids,
            non_finite,
        }
    }
}

impl NonFinite {
    /// Parses a `{position, token}` marker inside `[rows, vocabulary]` logits.
    ///
    /// # Panics
    ///
    /// If either coordinate is missing or out of range.
    fn parse(value: &Value, path: &str, rows: usize, vocabulary: usize) -> Self {
        let row = index(value, "position", &format!("{path}.position"));
        let token = index(value, "token", &format!("{path}.token"));
        assert!(
            row < rows && token < vocabulary,
            "reference.json: {path} ({row}, {token}) lies outside [{rows}, {vocabulary}]"
        );
        Self { row, token }
    }
}

impl DecodeRecord {
    /// Parses a `decode_consistency` record of a manifest whose vocabulary has `vocabulary` entries.
    ///
    /// # Panics
    ///
    /// If a field is missing or mistyped, the record does not extend prompt 0,
    /// its shape is not `[steps, vocabulary]`, or a file name is not plain.
    fn parse(record: &Value, vocabulary: usize) -> Self {
        let path = |field: &str| format!("decode_consistency.{field}");
        assert_eq!(
            unsigned(record, "prompt_index", &path("prompt_index")),
            0,
            "reference.json: decode rows must extend prompt 0"
        );
        let steps = member(record, "steps", &path("steps"))
            .as_array()
            .unwrap_or_else(|| panic!("reference.json: {} must be an array", path("steps")));
        let positions = steps
            .iter()
            .map(|step| index(step, "position", &path("steps[].position")))
            .collect();
        let fed_tokens = steps
            .iter()
            .map(|step| {
                member(step, "fed_token", &path("steps[].fed_token"))
                    .as_i64()
                    .expect("reference.json: a fed token must be an integer")
            })
            .collect();
        require_shape(
            record,
            "logits_shape",
            &path("logits_shape"),
            steps.len(),
            vocabulary,
        );
        let tokens_source = member(record, "tokens_source", &path("tokens_source"));
        Self {
            first_position: index(record, "first_position", &path("first_position")),
            tokens_source: string(
                tokens_source,
                "manifest_sha256",
                &path("tokens_source.manifest_sha256"),
            )
            .to_owned(),
            positions,
            fed_tokens,
            cached_file: plain_file_name(record, "cached_logits_file", &path("cached_logits_file")),
            cached_sha256: string(
                record,
                "cached_logits_sha256",
                &path("cached_logits_sha256"),
            )
            .to_owned(),
            stateless_file: plain_file_name(
                record,
                "stateless_logits_file",
                &path("stateless_logits_file"),
            ),
            stateless_sha256: string(
                record,
                "stateless_logits_sha256",
                &path("stateless_logits_sha256"),
            )
            .to_owned(),
            finite: finite_flag(record, "decode_consistency"),
        }
    }

    /// Returns the number of decode rows.
    pub(crate) fn steps(&self) -> usize {
        self.positions.len()
    }
}

impl PromptDecodeRecord {
    /// Parses a `prompt_decode` record against the manifest's `prompts`.
    ///
    /// # Panics
    ///
    /// If a field is missing or mistyped, the prompt index is out of range, the
    /// prefill is empty or longer than the prompt, the shape does not cover the
    /// prompt, or the file name is not plain.
    fn parse(record: &Value, prompts: &[BundlePrompt], vocabulary: usize) -> Self {
        let path = |field: &str| format!("prompt_decode.{field}");
        let prompt = index(record, "prompt_index", &path("prompt_index"));
        assert!(
            prompt < prompts.len(),
            "reference.json: prompt_decode.prompt_index {prompt} names no prompt"
        );
        let tokens = prompts[prompt].token_ids.len();
        let prefill = index(record, "prefill_tokens", &path("prefill_tokens"));
        assert!(
            (1..=tokens).contains(&prefill),
            "reference.json: prompt_decode.prefill_tokens {prefill} must lie in 1..={tokens}"
        );
        require_shape(
            record,
            "logits_shape",
            &path("logits_shape"),
            tokens,
            vocabulary,
        );
        Self {
            prompt,
            prefill,
            logits_file: plain_file_name(record, "logits_file", &path("logits_file")),
            logits_sha256: string(record, "logits_sha256", &path("logits_sha256")).to_owned(),
            finite: finite_flag(record, "prompt_decode"),
        }
    }
}

/// Returns the index of the longest prompt, the first of equally long ones (the generator's choice).
///
/// # Panics
///
/// If `prompts` is empty.
pub(crate) fn longest_prompt(prompts: &[BundlePrompt]) -> usize {
    let longest = prompts
        .iter()
        .map(|prompt| prompt.token_ids.len())
        .max()
        .expect("a bundle has prompts");
    prompts
        .iter()
        .position(|prompt| prompt.token_ids.len() == longest)
        .expect("the longest prompt exists")
}

/// Returns the quarter-prefill length `max(tokens / 4, 1)` of the decoded-prompt protocol.
pub(crate) fn quarter_prefill(tokens: usize) -> usize {
    (tokens / 4).max(1)
}

/// Requires `other` (named `name` in panics) to evaluate the truth's checkpoint, versions, prompts, and options.
///
/// # Panics
///
/// On the first disagreement listed in [Pairing](self#pairing), except the
/// dtype and greedy length, which depend on the bundle's role.
pub(crate) fn validate_identity(truth: &Manifest, other: &Manifest, name: &str) {
    let same = |what: &str, equal: bool| {
        assert!(equal, "{name} and the reference bundle differ in {what}");
    };
    same("model type", truth.model_type == other.model_type);
    same("model class", truth.model_class == other.model_class);
    same(
        "config.json SHA-256",
        truth.config_sha256 == other.config_sha256,
    );
    same("weight files", truth.weights_sha256 == other.weights_sha256);
    same("repository", truth.repository == other.repository);
    same("revision", truth.revision == other.revision);
    same(
        "Transformers version",
        truth.transformers == other.transformers,
    );
    same("PyTorch version", truth.torch == other.torch);
    same("vocabulary size", truth.vocabulary == other.vocabulary);
    same("layer truncation", truth.layers == other.layers);
    same("prompt count", truth.prompts.len() == other.prompts.len());
    for (index, (expected, actual)) in truth.prompts.iter().zip(&other.prompts).enumerate() {
        same(
            &format!("prompt {index} text"),
            expected.text == actual.text,
        );
        same(
            &format!("prompt {index} token IDs"),
            expected.token_ids == actual.token_ids,
        );
    }
}

/// Requires `bundle`'s decode rows to feed the truth's greedy tokens at its decode positions, sourced from the truth.
///
/// # Panics
///
/// If the bundle has no decode rows, or they start elsewhere, cover other
/// positions, feed other tokens, or name another token source.
pub(crate) fn validate_decode_record(
    truth: &Manifest,
    truth_sha256: &str,
    bundle: &Manifest,
    name: &str,
) {
    let Some(record) = &bundle.decode else {
        panic!(
            "{name} has no decode rows; generate it with --decode-tokens-from the reference bundle"
        );
    };
    let first = truth.decode_start();
    let steps = truth.greedy.len();
    assert_eq!(
        record.first_position, first,
        "{name}: decode rows must start at position {first}, after prompt 0"
    );
    assert!(
        record.positions.iter().copied().eq(first..first + steps),
        "{name}: decode rows must cover positions {first}..{}",
        first + steps
    );
    assert!(
        record
            .fed_tokens
            .iter()
            .copied()
            .eq(truth.greedy.iter().map(|&(token, _)| token)),
        "{name}: decode rows must feed the reference bundle's greedy tokens"
    );
    assert_eq!(
        record.tokens_source, truth_sha256,
        "{name}: decode tokens_source must be the reference bundle's reference.json SHA-256"
    );
}

/// Requires `decode_truth` to be the truth's own graph with decode rows; see [Pairing](self#pairing).
///
/// # Panics
///
/// On the first disagreement.
pub(crate) fn validate_decode_truth(truth: &Manifest, truth_sha256: &str, decode_truth: &Manifest) {
    let name = "the decode-truth bundle";
    assert!(
        truth.dtype.is_truth(),
        "the reference bundle must be a float32 or float64 evaluation, not {}",
        truth.dtype.label()
    );
    assert!(
        decode_truth.dtype.is_truth(),
        "{name} must be a float32 or float64 evaluation, not {}",
        decode_truth.dtype.label()
    );
    assert_eq!(
        decode_truth.graph(),
        Some(OfficialGraph::ALL[0]),
        "{name} must be a CPU eager evaluation"
    );
    assert!(
        !decode_truth.bf16_resident,
        "{name} must be a plain evaluation, not --bf16-resident"
    );
    validate_identity(truth, decode_truth, name);
    for (index, (expected, actual)) in truth.prompts.iter().zip(&decode_truth.prompts).enumerate() {
        assert_eq!(
            expected.logits_sha256, actual.logits_sha256,
            "{name}: prompt {index} logits differ from the reference bundle's, so it is not the reference graph"
        );
    }
    validate_decode_record(truth, truth_sha256, decode_truth, name);
}

/// Requires the official graphs to form the envelope of the truth for a `run` in F16 or BF16.
///
/// `graphs` holds each graph's manifest and the name used in panics.
///
/// # Panics
///
/// On the first disagreement listed in [Pairing](self#pairing).
pub(crate) fn validate_envelope(
    truth: &Manifest,
    truth_sha256: &str,
    decode_truth: &Manifest,
    graphs: &[(Manifest, String)],
    run: TensorDType,
) {
    let dtype = BundleDtype::of_run(run);
    validate_decode_truth(truth, truth_sha256, decode_truth);
    assert!(
        !truth.greedy_stopped,
        "the reference bundle's greedy continuation stopped early"
    );
    assert_eq!(
        graphs.len(),
        OfficialGraph::ALL.len(),
        "an envelope needs exactly {} official graphs, not {}",
        OfficialGraph::ALL.len(),
        graphs.len()
    );
    for graph in OfficialGraph::ALL {
        let count = graphs
            .iter()
            .filter(|(manifest, _)| manifest.graph() == Some(graph))
            .count();
        assert_eq!(
            count,
            1,
            "official graph {} appears {count} times; each of the four must appear once",
            graph.label()
        );
    }
    let script = &graphs[0].0.script_sha256;
    assert!(
        script.is_some()
            && graphs
                .iter()
                .all(|(manifest, _)| &manifest.script_sha256 == script),
        "the official graphs must come from one generator script (one provenance.script_sha256)"
    );
    for (manifest, name) in graphs {
        assert_eq!(
            manifest.dtype,
            dtype,
            "{name}: a {} evaluation cannot calibrate a {run:?} run",
            manifest.dtype.label()
        );
        assert!(
            !manifest.bf16_resident,
            "{name} must be a plain {} evaluation, not --bf16-resident",
            dtype.label()
        );
        validate_identity(truth, manifest, name);
        let (steps, expected) = (manifest.greedy.len(), truth.greedy.len());
        assert!(
            steps == expected || (manifest.greedy_stopped && steps < expected),
            "{name}: greedy length {steps} differs from the reference bundle's {expected}"
        );
        validate_decode_record(truth, truth_sha256, manifest, name);
    }
    if graphs
        .iter()
        .any(|(manifest, _)| manifest.prompt_decode.is_some())
    {
        let longest = longest_prompt(&truth.prompts);
        let prefill = quarter_prefill(truth.prompts[longest].token_ids.len());
        for (manifest, name) in graphs {
            let Some(record) = &manifest.prompt_decode else {
                panic!("{name} has no prompt_decode rows, which the other official graphs hold");
            };
            assert_eq!(
                (record.prompt, record.prefill),
                (longest, prefill),
                "{name}: prompt_decode must decode the longest prompt {longest} after a {prefill}-token prefill"
            );
        }
    }
}

/// Reads `[rows, vocabulary]` row-major F32 logits from `root/file` after checking their digest and size.
///
/// # Panics
///
/// If the file cannot be read, its SHA-256 differs from `sha256`, or it does
/// not hold exactly `rows × vocabulary` little-endian F32 values.
pub(crate) fn read_rows(
    root: &Path,
    file: &str,
    sha256: &str,
    rows: usize,
    vocabulary: usize,
) -> Vec<f32> {
    let path = root.join(file);
    let bytes =
        std::fs::read(&path).unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    assert_eq!(
        hex(&Sha256::digest(&bytes)),
        sha256,
        "{} is corrupted",
        path.display()
    );
    assert_eq!(
        bytes.len(),
        4 * rows * vocabulary,
        "{} must hold [{rows}, {vocabulary}] F32 logits",
        path.display()
    );
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect()
}

/// Reads one prompt's row-major `[tokens, vocabulary]` logits after checking their digest and size (see [`read_rows`]).
pub(crate) fn read_logits(root: &Path, prompt: &BundlePrompt, vocabulary: usize) -> Vec<f32> {
    read_rows(
        root,
        &prompt.logits_file,
        &prompt.logits_sha256,
        prompt.token_ids.len(),
        vocabulary,
    )
}

/// Absolute deviation of a set of logits from the truth.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Deviation {
    /// Largest `|actual - truth|`.
    pub(crate) maximum: f64,
    /// Flat index of the first entry attaining [`Self::maximum`].
    pub(crate) index: usize,
    /// Mean `|actual - truth|` over every entry.
    pub(crate) mean: f64,
}

impl Deviation {
    /// Measures `actual` against `truth` element by element, or returns `actual`'s first non-finite flat index.
    ///
    /// Each difference of two F32 values is taken in F64, where it is exact
    /// unless their binary exponents differ by more than 29 (a 24-bit
    /// significand shifted by `d` bits needs `24 + d` of F64's 53), and the
    /// mean is summed in F64 over `actual.len()` entries. On ties the first
    /// index attaining the maximum is reported.
    ///
    /// # Panics
    ///
    /// If the slices differ in length or are empty, or `truth` holds a
    /// non-finite value; `label` names the comparison.
    pub(crate) fn try_measure(label: &str, actual: &[f32], truth: &[f32]) -> Result<Self, usize> {
        assert_eq!(
            actual.len(),
            truth.len(),
            "{label}: compared logits differ in length"
        );
        assert!(!actual.is_empty(), "{label}: no logits to compare");
        if let Some(first) = actual.iter().position(|value| !value.is_finite()) {
            return Err(first);
        }
        assert!(
            truth.iter().all(|value| value.is_finite()),
            "{label}: the reference logits hold a non-finite value"
        );
        let mut deviation = Self {
            maximum: 0.0,
            index: 0,
            mean: 0.0,
        };
        let mut sum = 0.0_f64;
        for (index, (&actual, &truth)) in actual.iter().zip(truth).enumerate() {
            let gap = (f64::from(actual) - f64::from(truth)).abs();
            sum += gap;
            if gap > deviation.maximum {
                deviation.maximum = gap;
                deviation.index = index;
            }
        }
        deviation.mean = sum / actual.len() as f64;
        Ok(deviation)
    }

    /// Measures `actual` against `truth` (see [`Self::try_measure`]); `label` names the comparison in panics.
    ///
    /// # Panics
    ///
    /// As [`Self::try_measure`], and if `actual` holds a non-finite value.
    pub(crate) fn measure(label: &str, actual: &[f32], truth: &[f32]) -> Self {
        Self::try_measure(label, actual, truth)
            .unwrap_or_else(|_| panic!("{label}: the compared logits hold a non-finite value"))
    }
}

/// The two statistics a deviation or an envelope is judged by.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Statistic {
    /// Largest absolute deviation.
    pub(crate) maximum: f64,
    /// Mean absolute deviation.
    pub(crate) mean: f64,
}

impl Statistic {
    /// Returns a deviation's two statistics.
    pub(crate) fn of(deviation: &Deviation) -> Self {
        Self {
            maximum: deviation.maximum,
            mean: deviation.mean,
        }
    }

    /// Returns the larger maximum and the larger mean of two statistics.
    pub(crate) fn larger(self, other: Self) -> Self {
        Self {
            maximum: self.maximum.max(other.maximum),
            mean: self.mean.max(other.mean),
        }
    }
}

/// Returns the largest F32 value not above `value`: the tolerance at which the shared greedy rule decides the same steps.
///
/// The gates compare greedy continuations through
/// `tests/support/backend_numerics.rs`, whose rule calls an F32 margin `m`
/// decisive when `m > 2 t` in F32, while this module calls it decisive when
/// `m > 2 b` for an F64 bound `b`. With `t` the largest F32 value not above
/// `b`, the two rules agree on every F32 margin when `b` lies in F32's normal
/// range (at least `2^-126`, [`f32::MIN_POSITIVE`]) and below `2^127`, so that
/// `2 t` is finite. There `t` is a normal F32 value: doubling it is exact and
/// maps its binade onto the next one, whose F32 values are spaced twice as far
/// apart, so the F32 successor of `2 t` is twice the successor of `t`. That
/// successor lies above `b`, so `2 t` is the largest F32 value not above
/// `2 b`. Then `m > 2 b` implies `m > 2 t`, and `m > 2 t` means `m` is at
/// least the successor of `2 t`, which lies above `2 b`; a margin exactly at
/// either threshold is a tie under both rules.
///
/// Below `2^-126`, `t` is subnormal or zero, and F32 values are `2^-149`
/// apart both at `t` and at `2 t`, so the successor of `2 t` is
/// `2 t + 2^-149`, which need not lie above `2 b`: for `b` in
/// `[t + 2^-150, t + 2^-149)` the margin `2 t + 2^-149` is decisive under
/// `m > 2 t` but not under `m > 2 b`. From `2^127` on, `2 t` overflows to
/// infinity; neither rule calls a finite margin decisive, but `m > 2 b` still
/// calls an infinite one decisive. Every greedy bound of the recorded
/// envelopes lies far inside the range where the rules agree.
pub(crate) fn largest_f32_not_above(value: f64) -> f32 {
    let nearest = value as f32;
    if f64::from(nearest) > value {
        nearest.next_down()
    } else {
        nearest
    }
}

/// The half-precision bounds of one kind of rows.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Bound {
    /// Bound `s_max E_max` on the largest absolute logit deviation.
    pub(crate) maximum: f64,
    /// Bound `s_mean E_mean` on the mean absolute logit deviation.
    pub(crate) mean: f64,
}

impl Bound {
    /// Applies the rule of a `dtype` run to an official envelope; `what` names the rows in the panic.
    ///
    /// # Panics
    ///
    /// If either envelope statistic is zero or not finite (official graphs
    /// that reproduce their truth exactly are not half-precision
    /// evaluations), or if `dtype` has no slacks.
    pub(crate) fn from_envelope(what: &str, envelope: &Statistic, dtype: TensorDType) -> Self {
        assert!(
            envelope.maximum.is_finite()
                && envelope.maximum > 0.0
                && envelope.mean.is_finite()
                && envelope.mean > 0.0,
            "{what}: the official envelope (max {:e}, mean {:e}) must be positive and finite; \
             official graphs equal to their reference are not half-precision evaluations",
            envelope.maximum,
            envelope.mean
        );
        Self {
            maximum: max_slack(dtype) * envelope.maximum,
            mean: mean_slack(dtype) * envelope.mean,
        }
    }

    /// Whether a deviation over every row meets both bounds (inclusive).
    pub(crate) fn admits(&self, observed: &Deviation) -> bool {
        self.admits_rows(observed) && observed.mean <= self.mean
    }

    /// Whether a deviation over some rows meets the maximum bound (inclusive).
    pub(crate) fn admits_rows(&self, observed: &Deviation) -> bool {
        observed.maximum <= self.maximum
    }

    /// Truth margin a greedy step must exceed under this bound alone: twice the maximum bound.
    pub(crate) fn greedy_threshold(&self) -> f64 {
        2.0 * self.maximum
    }

    /// Whether a truth top-2 `margin` determines the argmax under this bound (strictly above the threshold).
    pub(crate) fn is_decisive(&self, margin: f32) -> bool {
        f64::from(margin) > self.greedy_threshold()
    }

    /// Returns the F32 tolerance at which the shared greedy rule decides the steps [`Self::is_decisive`] decides.
    ///
    /// See [`largest_f32_not_above`] for the range where the two agree.
    pub(crate) fn greedy_tolerance(&self) -> f32 {
        largest_f32_not_above(self.maximum)
    }
}

/// Returns the indices of the greedy steps whose truth margin is decisive under `bound`.
///
/// Every decisive step is listed, also after a tie, because the
/// continuation is teacher-forced; the result is empty when no step is
/// decisive.
pub(crate) fn decisive_steps(greedy: &[(i64, f32)], bound: &Bound) -> Vec<usize> {
    greedy
        .iter()
        .enumerate()
        .filter(|(_, (_, margin))| bound.is_decisive(*margin))
        .map(|(step, _)| step)
        .collect()
}

/// Returns the provisional self-consistency tolerance of a half-precision `dtype`.
///
/// # Panics
///
/// For any other dtype: F32 runs keep their own gates.
pub(crate) fn self_consistency_tolerance(dtype: TensorDType) -> f64 {
    SELF_CONSISTENCY_TOLERANCE
        .iter()
        .find(|(candidate, _)| *candidate == dtype)
        .map(|&(_, tolerance)| tolerance)
        .unwrap_or_else(|| panic!("no half-precision self-consistency tolerance for {dtype:?}"))
}

/// The four official graphs' statistics of one kind of rows, and the envelope and bounds they set.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct GraphEnvelope {
    /// Each official graph's statistics, in [`OfficialGraph::ALL`] order.
    pub(crate) graphs: [Statistic; 4],
    /// The largest maximum and the largest mean over the graphs, `E`.
    pub(crate) envelope: Statistic,
    /// Bounds `s E` the rule derives.
    pub(crate) bound: Bound,
}

impl GraphEnvelope {
    /// Forms the envelope of four graphs' statistics for a `dtype` run; `what` names the rows in panics.
    ///
    /// # Panics
    ///
    /// As [`Bound::from_envelope`].
    pub(crate) fn new(what: &str, graphs: [Statistic; 4], dtype: TensorDType) -> Self {
        let envelope = graphs[1..]
            .iter()
            .fold(graphs[0], |largest, graph| largest.larger(*graph));
        Self {
            graphs,
            envelope,
            bound: Bound::from_envelope(what, &envelope, dtype),
        }
    }

    /// Returns each graph's leave-one-out ratios: its statistic over the largest of the other three graphs'.
    ///
    /// A zero denominator gives an infinite or NaN ratio, which no slack admits.
    pub(crate) fn leave_one_out(&self) -> [Statistic; 4] {
        std::array::from_fn(|graph| {
            let others = (0..4)
                .filter(|&other| other != graph)
                .map(|other| self.graphs[other])
                .reduce(Statistic::larger)
                .expect("three other graphs");
            Statistic {
                maximum: self.graphs[graph].maximum / others.maximum,
                mean: self.graphs[graph].mean / others.mean,
            }
        })
    }
}

/// One prompt's official deviations and envelope.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct PromptEnvelope {
    /// Number of prompt positions `T`.
    pub(crate) tokens: usize,
    /// Each official graph's deviation, in [`OfficialGraph::ALL`] order.
    pub(crate) deviations: [Deviation; 4],
    /// The envelope and bounds they set.
    pub(crate) envelope: GraphEnvelope,
}

/// The official graphs' decode-row deviations from the decode truth, and their envelope.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct DecodeEnvelope {
    /// Each graph's cached rows' deviation, in [`OfficialGraph::ALL`] order.
    pub(crate) cached: [Deviation; 4],
    /// Each graph's stateless rows' deviation.
    pub(crate) stateless: [Deviation; 4],
    /// The envelope over both passes and the bounds it sets.
    pub(crate) envelope: GraphEnvelope,
}

/// The official graphs' quarter-prefill decode of the longest prompt, and its envelope.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct PromptDecodeEnvelope {
    /// Index of the decoded prompt.
    pub(crate) prompt: usize,
    /// Tokens prefilled before decoding one at a time.
    pub(crate) prefill: usize,
    /// Each graph's deviation from the truth's logits of that prompt.
    pub(crate) deviations: [Deviation; 4],
    /// The envelope and bounds they set.
    pub(crate) envelope: GraphEnvelope,
}

/// Panics as uncalibrated: an official graph of the run's dtype is non-finite at `place`.
///
/// # Panics
///
/// Always; see [Uncalibrated truths](self#uncalibrated-truths).
fn uncalibrated(graph: OfficialGraph, dtype: BundleDtype, place: &str) -> ! {
    panic!(
        "uncalibrated: official {} {} non-finite at {place}; a checkpoint whose official evaluation \
         overflows in this dtype has no envelope, so the run fails",
        graph.label(),
        dtype.label()
    )
}

/// The decode truth: the reference graph's stateless rows at the truth's teacher-forced greedy positions.
#[derive(Debug, Clone)]
pub(crate) struct DecodeTruth {
    /// SHA-256 of the decode-truth bundle's `reference.json`.
    pub(crate) sha256: String,
    /// Position of the first row (prompt 0's length).
    pub(crate) first: usize,
    /// Vocabulary size `V`.
    pub(crate) vocabulary: usize,
    /// Row-major `[steps, V]` stateless rows.
    pub(crate) rows: Vec<f32>,
}

impl DecodeTruth {
    /// Reads the decode truth in `decode_truth_dir` after pairing it with the truth in `truth_dir`.
    ///
    /// # Panics
    ///
    /// If either bundle is malformed, the pairing fails
    /// ([`validate_decode_truth`]), or its rows are corrupted.
    pub(crate) fn read(truth_dir: &Path, decode_truth_dir: &Path) -> Self {
        let (truth, truth_sha256) = Manifest::read(truth_dir);
        Self::read_paired(&truth, &truth_sha256, decode_truth_dir)
    }

    /// Reads the decode truth in `decode_truth_dir` after pairing it with an already-read truth.
    ///
    /// # Panics
    ///
    /// As [`Self::read`].
    fn read_paired(truth: &Manifest, truth_sha256: &str, decode_truth_dir: &Path) -> Self {
        let (decode_truth, sha256) = Manifest::read(decode_truth_dir);
        validate_decode_truth(truth, truth_sha256, &decode_truth);
        let record = decode_truth.decode.as_ref().expect("validated decode rows");
        let rows = read_rows(
            decode_truth_dir,
            &record.stateless_file,
            &record.stateless_sha256,
            record.steps(),
            decode_truth.vocabulary,
        );
        Self {
            sha256,
            first: record.first_position,
            vocabulary: decode_truth.vocabulary,
            rows,
        }
    }

    /// Returns the number of decode rows.
    pub(crate) fn steps(&self) -> usize {
        self.rows.len() / self.vocabulary
    }

    /// Checks an F32 run's cached decode rows against these rows at its F32 `tolerance`; prints the worst row.
    ///
    /// `decoded[s]` holds the logits that decode step `s` returned for
    /// position `first + s`.
    ///
    /// # Panics
    ///
    /// If the rows do not cover every decode position, a value is not finite,
    /// or a row deviates by more than `tolerance`.
    pub(crate) fn check_f32(&self, label: &str, decoded: &[Vec<f32>], tolerance: f32) {
        assert_eq!(
            decoded.len(),
            self.steps(),
            "{label}: one decoded row per decode-truth row"
        );
        let mut worst = (0, 0.0_f64);
        for (step, (actual, expected)) in decoded
            .iter()
            .zip(self.rows.chunks_exact(self.vocabulary))
            .enumerate()
        {
            let position = self.first + step;
            let name = format!("{label} step {step} at position {position}");
            let observed = Deviation::measure(&name, actual, expected);
            assert!(
                observed.maximum <= f64::from(tolerance),
                "{name}: max |Δ| = {:e} to the decode truth exceeds the F32 tolerance {tolerance:e}",
                observed.maximum
            );
            if observed.maximum > worst.1 {
                worst = (step, observed.maximum);
            }
        }
        eprintln!(
            "{label}: {} decode rows from position {}: max |Δ| = {:e} to the decode truth, at step {} (tolerance {tolerance:e})",
            decoded.len(),
            self.first,
            worst.1,
            worst.0
        );
    }
}

/// One official graph's bundle as [`Envelope::read`] holds it while measuring.
struct GraphBundle {
    /// Parsed manifest.
    manifest: Manifest,
    /// SHA-256 of its `reference.json`.
    sha256: String,
    /// Bundle directory.
    directory: PathBuf,
}

/// A validated truth, decode truth, and four official graphs, with the envelopes they set for one run dtype.
#[derive(Debug, Clone)]
pub(crate) struct Envelope {
    /// Run dtype the envelope calibrates.
    pub(crate) dtype: TensorDType,
    /// SHA-256 of the truth's `reference.json`.
    pub(crate) truth_sha256: String,
    /// SHA-256 of each official graph's `reference.json`, in [`OfficialGraph::ALL`] order.
    pub(crate) graph_sha256: [String; 4],
    /// Directory of each official graph, in [`OfficialGraph::ALL`] order.
    pub(crate) graph_dirs: [PathBuf; 4],
    /// Truth manifest.
    pub(crate) truth: Manifest,
    /// The decode truth and its rows.
    pub(crate) decode_truth: DecodeTruth,
    /// Envelope of each prompt, in bundle order.
    pub(crate) prompts: Vec<PromptEnvelope>,
    /// Envelope of the decode rows.
    pub(crate) decode: DecodeEnvelope,
    /// Envelope of the quarter-prefill prompt decode, when the graphs recorded it.
    pub(crate) prompt_decode: Option<PromptDecodeEnvelope>,
}

impl Envelope {
    /// Reads, pairs, and measures a truth, its decode truth, and the four official graphs of a `dtype` run.
    ///
    /// At most the truth's and one graph's logits of one prompt are held at a
    /// time (about 2 × 1.3 GB for Gemma's 1237-token prompt).
    ///
    /// # Panics
    ///
    /// If a bundle is malformed or corrupted, the pairing fails
    /// ([`validate_envelope`]), an official graph is non-finite (as
    /// uncalibrated), or an envelope is zero ([`Bound::from_envelope`]).
    pub(crate) fn read(
        dtype: TensorDType,
        truth_dir: &Path,
        decode_truth_dir: &Path,
        official_dirs: &[PathBuf],
    ) -> Self {
        let label = BundleDtype::of_run(dtype);
        let (truth, truth_sha256) = Manifest::read(truth_dir);
        let (decode_manifest, _) = Manifest::read(decode_truth_dir);
        let mut graphs: Vec<GraphBundle> = official_dirs
            .iter()
            .map(|directory| {
                let (manifest, sha256) = Manifest::read(directory);
                GraphBundle {
                    manifest,
                    sha256,
                    directory: directory.clone(),
                }
            })
            .collect();
        let named: Vec<(Manifest, String)> = graphs
            .iter()
            .map(|graph| {
                (
                    graph.manifest.clone(),
                    graph.directory.display().to_string(),
                )
            })
            .collect();
        validate_envelope(&truth, &truth_sha256, &decode_manifest, &named, dtype);
        graphs.sort_by_key(|graph| graph.manifest.graph());
        for bundle in &graphs {
            let graph = bundle.manifest.graph().expect("validated graph");
            for (index, prompt) in bundle.manifest.prompts.iter().enumerate() {
                if let Some(at) = prompt.non_finite {
                    uncalibrated(graph, label, &format!("prompt {index} position {}", at.row));
                }
            }
            if !bundle
                .manifest
                .decode
                .as_ref()
                .expect("validated decode rows")
                .finite
            {
                uncalibrated(graph, label, "its decode rows");
            }
            if bundle
                .manifest
                .prompt_decode
                .as_ref()
                .is_some_and(|record| !record.finite)
            {
                uncalibrated(graph, label, "its prompt-decode rows");
            }
        }
        let decode_truth = DecodeTruth::read_paired(&truth, &truth_sha256, decode_truth_dir);
        let vocabulary = truth.vocabulary;
        let measure = |slot: usize, what: &str, actual: &[f32], reference: &[f32]| {
            Deviation::try_measure(what, actual, reference).unwrap_or_else(|flat| {
                uncalibrated(
                    OfficialGraph::ALL[slot],
                    label,
                    &format!("{what} position {}", flat / vocabulary),
                )
            })
        };
        let prompts = truth
            .prompts
            .iter()
            .enumerate()
            .map(|(index, expected)| {
                let exact = read_logits(truth_dir, expected, vocabulary);
                let what = format!("prompt {index}");
                let deviations: [Deviation; 4] = std::array::from_fn(|slot| {
                    let bundle = &graphs[slot];
                    let rounded = read_logits(
                        &bundle.directory,
                        &bundle.manifest.prompts[index],
                        vocabulary,
                    );
                    measure(slot, &what, &rounded, &exact)
                });
                PromptEnvelope {
                    tokens: expected.token_ids.len(),
                    deviations,
                    envelope: GraphEnvelope::new(
                        &what,
                        deviations.map(|deviation| Statistic::of(&deviation)),
                        dtype,
                    ),
                }
            })
            .collect();
        let passes: [(Deviation, Deviation); 4] = std::array::from_fn(|slot| {
            let bundle = &graphs[slot];
            let record = bundle
                .manifest
                .decode
                .as_ref()
                .expect("validated decode rows");
            let read = |file: &str, sha256: &str| {
                read_rows(&bundle.directory, file, sha256, record.steps(), vocabulary)
            };
            let cached = read(&record.cached_file, &record.cached_sha256);
            let stateless = read(&record.stateless_file, &record.stateless_sha256);
            (
                measure(slot, "cached decode rows,", &cached, &decode_truth.rows),
                measure(
                    slot,
                    "stateless decode rows,",
                    &stateless,
                    &decode_truth.rows,
                ),
            )
        });
        let decode = DecodeEnvelope {
            cached: passes.map(|(cached, _)| cached),
            stateless: passes.map(|(_, stateless)| stateless),
            envelope: GraphEnvelope::new(
                "decode rows",
                passes.map(|(cached, stateless)| {
                    Statistic::of(&cached).larger(Statistic::of(&stateless))
                }),
                dtype,
            ),
        };
        let prompt_decode = graphs[0].manifest.prompt_decode.as_ref().map(|first| {
            let (prompt, prefill) = (first.prompt, first.prefill);
            let exact = read_logits(truth_dir, &truth.prompts[prompt], vocabulary);
            let tokens = truth.prompts[prompt].token_ids.len();
            let deviations: [Deviation; 4] = std::array::from_fn(|slot| {
                let bundle = &graphs[slot];
                let record = bundle
                    .manifest
                    .prompt_decode
                    .as_ref()
                    .expect("validated prompt decode");
                let rows = read_rows(
                    &bundle.directory,
                    &record.logits_file,
                    &record.logits_sha256,
                    tokens,
                    vocabulary,
                );
                measure(slot, "prompt-decode rows,", &rows, &exact)
            });
            PromptDecodeEnvelope {
                prompt,
                prefill,
                deviations,
                envelope: GraphEnvelope::new(
                    "prompt-decode rows",
                    deviations.map(|deviation| Statistic::of(&deviation)),
                    dtype,
                ),
            }
        });
        Self {
            dtype,
            truth_sha256,
            graph_sha256: std::array::from_fn(|slot| graphs[slot].sha256.clone()),
            graph_dirs: std::array::from_fn(|slot| graphs[slot].directory.clone()),
            truth,
            decode_truth,
            prompts,
            decode,
            prompt_decode,
        }
    }

    /// Returns the bound greedy steps are judged at: the larger maximum bound of prompt 0 and of the decode rows.
    pub(crate) fn greedy_bound(&self) -> f64 {
        self.prompts[0]
            .envelope
            .bound
            .maximum
            .max(self.decode.envelope.bound.maximum)
    }

    /// Whether a truth top-2 `margin` is decisive: strictly above twice [`Self::greedy_bound`].
    pub(crate) fn is_decisive(&self, margin: f32) -> bool {
        f64::from(margin) > 2.0 * self.greedy_bound()
    }

    /// Returns the greedy steps of prompt 0's continuation that a half-precision run compares.
    pub(crate) fn decisive_greedy_steps(&self) -> Vec<usize> {
        decisive_steps(
            &self.truth.greedy,
            &Bound {
                maximum: self.greedy_bound(),
                mean: f64::INFINITY,
            },
        )
    }

    /// Lists every envelope as `(rows, envelope)`: each prompt, the decode rows, and the prompt-decode rows.
    pub(crate) fn envelopes(&self) -> Vec<(String, GraphEnvelope)> {
        let mut out: Vec<(String, GraphEnvelope)> = self
            .prompts
            .iter()
            .enumerate()
            .map(|(index, prompt)| (format!("prompt {index}"), prompt.envelope))
            .collect();
        out.push(("decode rows".to_owned(), self.decode.envelope));
        if let Some(prompt_decode) = &self.prompt_decode {
            out.push((
                format!("prompt {} decoded rows", prompt_decode.prompt),
                prompt_decode.envelope,
            ));
        }
        out
    }

    /// Returns every leave-one-out ratio as `(rows, graph, ratios, admitted)` under the compiled slacks.
    pub(crate) fn leave_one_out(&self) -> Vec<(String, OfficialGraph, Statistic, bool)> {
        let (maximum, mean) = (max_slack(self.dtype), mean_slack(self.dtype));
        self.envelopes()
            .into_iter()
            .flat_map(|(rows, envelope)| {
                envelope
                    .leave_one_out()
                    .into_iter()
                    .zip(OfficialGraph::ALL)
                    .map(move |(ratio, graph)| {
                        let admitted = ratio.maximum <= maximum && ratio.mean <= mean;
                        (rows.clone(), graph, ratio, admitted)
                    })
            })
            .collect()
    }

    /// Prints every leave-one-out ratio and asserts that each official graph lies within the envelope of the other three.
    ///
    /// # Panics
    ///
    /// If any ratio exceeds its compiled slack: the slacks must then be
    /// re-derived ([Slack](self#slack)), never tuned.
    pub(crate) fn assert_leave_one_out(&self) {
        let (maximum, mean) = (max_slack(self.dtype), mean_slack(self.dtype));
        let mut refused = Vec::new();
        for (rows, graph, ratio, admitted) in self.leave_one_out() {
            eprintln!(
                "{rows}: official {} leave-one-out max ratio {:.4} (slack {maximum}), mean ratio {:.4} (slack {mean}){}",
                graph.label(),
                ratio.maximum,
                ratio.mean,
                if admitted { "" } else { " OUTSIDE" }
            );
            if !admitted {
                refused.push(format!("{rows} {}", graph.label()));
            }
        }
        assert!(
            refused.is_empty(),
            "official graphs outside the envelope of the other three under the compiled {:?} slacks: {}; \
             re-derive the slacks with scripts/derive_half_precision_envelope.py",
            self.dtype,
            refused.join(", ")
        );
    }

    /// Prints the bundles' identities, every graph's deviations, the envelopes and bounds, and the decisive greedy steps.
    pub(crate) fn report(&self, truth_dir: &Path) {
        let vocabulary = self.truth.vocabulary;
        eprintln!(
            "{} ({:?}): reference.json SHA-256 {}",
            truth_dir.display(),
            self.truth.dtype,
            self.truth_sha256
        );
        eprintln!(
            "decode truth: reference.json SHA-256 {}, {} rows from position {}",
            self.decode_truth.sha256,
            self.decode_truth.steps(),
            self.decode_truth.first
        );
        for (slot, graph) in OfficialGraph::ALL.into_iter().enumerate() {
            eprintln!(
                "official {} {:?}: {} reference.json SHA-256 {}",
                graph.label(),
                self.dtype,
                self.graph_dirs[slot].display(),
                self.graph_sha256[slot]
            );
        }
        let line = |rows: &str, deviations: &[Deviation; 4], envelope: &GraphEnvelope| {
            for (graph, deviation) in OfficialGraph::ALL.into_iter().zip(deviations) {
                eprintln!(
                    "{rows}: official {} max |Δ| = {:e} at row {}, token {}, mean |Δ| = {:e}",
                    graph.label(),
                    deviation.maximum,
                    deviation.index / vocabulary,
                    deviation.index % vocabulary,
                    deviation.mean
                );
            }
            eprintln!(
                "{rows}: envelope max {:e}, mean {:e}; bounds max {:e}, mean {:e}",
                envelope.envelope.maximum,
                envelope.envelope.mean,
                envelope.bound.maximum,
                envelope.bound.mean
            );
        };
        for (index, prompt) in self.prompts.iter().enumerate() {
            line(
                &format!("prompt {index} ({} tokens)", prompt.tokens),
                &prompt.deviations,
                &prompt.envelope,
            );
        }
        line(
            "decode rows (cached)",
            &self.decode.cached,
            &self.decode.envelope,
        );
        line(
            "decode rows (stateless)",
            &self.decode.stateless,
            &self.decode.envelope,
        );
        if let Some(prompt_decode) = &self.prompt_decode {
            line(
                &format!(
                    "prompt {} decoded after a {}-token prefill",
                    prompt_decode.prompt, prompt_decode.prefill
                ),
                &prompt_decode.deviations,
                &prompt_decode.envelope,
            );
        }
        let decisive = self.decisive_greedy_steps();
        eprintln!(
            "greedy: threshold {:e}; {} of {} steps decisive {decisive:?}",
            2.0 * self.greedy_bound(),
            decisive.len(),
            self.truth.greedy.len()
        );
    }
}

/// Which pass produced a set of decode rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DecodeRows {
    /// Rows the cached decode steps returned.
    Cached,
    /// Rows of the same run's stateless pass at the decode positions.
    Stateless,
}

impl DecodeRows {
    /// Returns the pass's label.
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Cached => "cached",
            Self::Stateless => "stateless",
        }
    }
}

/// Judges one half-precision run with the official envelope of the truth it reads.
#[derive(Debug, Clone)]
pub(crate) struct HalfPrecisionGate {
    /// Stored dtype of the run, F16 or BF16.
    dtype: TensorDType,
    /// Envelope of the truth the run is judged against.
    envelope: Envelope,
}

impl HalfPrecisionGate {
    /// Calibrates a `dtype` run against the truth in `truth_dir` from its decode truth and four official graphs.
    ///
    /// Prints every bound.
    ///
    /// # Panics
    ///
    /// If `dtype` is not F16 or BF16, or [`Envelope::read`] refuses the bundles
    /// (including as uncalibrated).
    pub(crate) fn new(
        dtype: TensorDType,
        truth_dir: &Path,
        decode_truth_dir: &Path,
        official_dirs: &[PathBuf],
    ) -> Self {
        assert!(
            is_half_precision(dtype),
            "the official envelope judges F16 and BF16 runs, not {dtype:?}"
        );
        let envelope = Envelope::read(dtype, truth_dir, decode_truth_dir, official_dirs);
        let (maximum, mean) = (max_slack(dtype), mean_slack(dtype));
        for (rows, graphs) in envelope.envelopes() {
            eprintln!(
                "{dtype:?} bounds of {rows}: max {:e}, mean {:e} ({maximum} and {mean} times the official envelope {:e}, {:e})",
                graphs.bound.maximum,
                graphs.bound.mean,
                graphs.envelope.maximum,
                graphs.envelope.mean
            );
        }
        Self { dtype, envelope }
    }

    /// Returns the envelope the gate applies.
    pub(crate) fn envelope(&self) -> &Envelope {
        &self.envelope
    }

    /// Returns the bounds of prompt `prompt`.
    pub(crate) fn bound(&self, prompt: usize) -> Bound {
        self.envelope.prompts[prompt].envelope.bound
    }

    /// Returns the bounds of the decode rows.
    pub(crate) fn decode_bound(&self) -> Bound {
        self.envelope.decode.envelope.bound
    }

    /// Checks every position of prompt `prompt` against both bounds; prints and returns the deviation.
    ///
    /// `label` names the pass that produced `actual` in the printed line and in panics.
    ///
    /// # Panics
    ///
    /// If `actual` and `truth` do not hold the prompt's `[tokens, vocabulary]`
    /// logits, hold a non-finite value, or exceed a bound.
    pub(crate) fn check_prompt(
        &self,
        label: &str,
        prompt: usize,
        actual: &[f32],
        truth: &[f32],
    ) -> Deviation {
        self.check_all_rows(
            label,
            &format!("prompt {prompt}"),
            self.envelope.prompts[prompt].tokens,
            self.bound(prompt),
            actual,
            truth,
        )
    }

    /// Checks `[tokens, V]` rows against both bounds of `rows`; prints and returns the deviation.
    ///
    /// # Panics
    ///
    /// As [`Self::check_prompt`].
    fn check_all_rows(
        &self,
        label: &str,
        rows: &str,
        tokens: usize,
        bound: Bound,
        actual: &[f32],
        truth: &[f32],
    ) -> Deviation {
        let vocabulary = self.envelope.truth.vocabulary;
        assert_eq!(
            truth.len(),
            tokens * vocabulary,
            "{label}: the reference logits must cover every position of {rows}"
        );
        let observed = Deviation::measure(label, actual, truth);
        eprintln!(
            "{label}: {tokens} tokens, max |Δ| = {:e} at position {} (bound {:e}), mean |Δ| = {:e} (bound {:e})",
            observed.maximum,
            observed.index / vocabulary,
            bound.maximum,
            observed.mean,
            bound.mean
        );
        assert!(
            bound.admits_rows(&observed),
            "{label}: max |Δ| = {:e} exceeds the half-precision bound {:e} of {rows}",
            observed.maximum,
            bound.maximum
        );
        assert!(
            bound.admits(&observed),
            "{label}: mean |Δ| = {:e} exceeds the half-precision bound {:e} of {rows}",
            observed.mean,
            bound.mean
        );
        observed
    }

    /// Checks some rows of prompt `prompt` against its maximum bound; prints and returns the deviation.
    ///
    /// # Panics
    ///
    /// If the slices differ in length, hold a non-finite value, or exceed the bound.
    pub(crate) fn check_rows(
        &self,
        label: &str,
        prompt: usize,
        actual: &[f32],
        truth: &[f32],
    ) -> Deviation {
        let observed = Deviation::measure(label, actual, truth);
        let bound = self.bound(prompt);
        eprintln!(
            "{label}: max |Δ| = {:e} to the reference (bound {:e})",
            observed.maximum, bound.maximum
        );
        assert!(
            bound.admits_rows(&observed),
            "{label}: max |Δ| = {:e} exceeds the half-precision bound {:e}",
            observed.maximum,
            bound.maximum
        );
        observed
    }

    /// Checks a cached or batched result against the same run's stateless pass (provisional); prints and returns the deviation.
    ///
    /// # Panics
    ///
    /// If the slices differ in length, hold a non-finite value, or differ by
    /// more than the dtype's [`SELF_CONSISTENCY_TOLERANCE`].
    pub(crate) fn check_consistency(
        &self,
        label: &str,
        actual: &[f32],
        stateless: &[f32],
    ) -> Deviation {
        let observed = Deviation::measure(label, actual, stateless);
        let tolerance = self_consistency_tolerance(self.dtype);
        eprintln!(
            "{label}: max |Δ| = {:e} to the stateless pass (provisional tolerance {tolerance:e})",
            observed.maximum
        );
        assert!(
            observed.maximum <= tolerance,
            "{label}: max |Δ| = {:e} to the stateless pass exceeds the provisional {:?} self-consistency tolerance {tolerance:e}",
            observed.maximum,
            self.dtype
        );
        observed
    }

    /// Checks one pass's `[steps, V]` decode rows against the decode truth's rows by both decode bounds; prints and returns the deviation.
    ///
    /// # Panics
    ///
    /// If `rows` does not match `truth_rows` in length, holds a non-finite
    /// value, or exceeds a bound.
    pub(crate) fn check_decode_rows(
        &self,
        label: &str,
        kind: DecodeRows,
        rows: &[f32],
        truth_rows: &[f32],
    ) -> Deviation {
        let vocabulary = self.envelope.truth.vocabulary;
        let name = format!("{label} ({} rows)", kind.label());
        let observed = Deviation::measure(&name, rows, truth_rows);
        let bound = self.decode_bound();
        let step = observed.index / vocabulary;
        eprintln!(
            "{name}: {} rows from position {}, max |Δ| = {:e} at step {step} (bound {:e}), mean |Δ| = {:e} (bound {:e}) to the decode truth",
            rows.len() / vocabulary,
            self.envelope.decode_truth.first,
            observed.maximum,
            bound.maximum,
            observed.mean,
            bound.mean
        );
        assert!(
            bound.admits_rows(&observed),
            "{name}: max |Δ| = {:e} at step {step} exceeds the half-precision decode bound {:e}",
            observed.maximum,
            bound.maximum
        );
        assert!(
            bound.admits(&observed),
            "{name}: mean |Δ| = {:e} exceeds the half-precision decode bound {:e}",
            observed.mean,
            bound.mean
        );
        observed
    }

    /// Checks a run's cached decode rows, and its stateless rows at the same positions, against the decode truth.
    ///
    /// `decoded[s]` holds the logits that decode step `s` returned for
    /// position `first + s`; `stateless` holds the stateless pass's rows at
    /// those positions in order. Both passes must meet both decode bounds
    /// ([`Self::check_decode_rows`]); their gap from each other is printed as a
    /// diagnostic only (see [Decoding](self#decoding)). Returns the cached and
    /// stateless deviations.
    ///
    /// # Panics
    ///
    /// If `first` is not the decode truth's first position, the rows do not
    /// cover every decode-truth row, a value is not finite, or a pass exceeds
    /// a bound.
    pub(crate) fn check_decode(
        &self,
        label: &str,
        first: usize,
        decoded: &[Vec<f32>],
        stateless: &[f32],
    ) -> (Deviation, Deviation) {
        let truth = &self.envelope.decode_truth;
        assert_eq!(
            first, truth.first,
            "{label}: decode rows must start where the decode truth's do"
        );
        assert_eq!(
            decoded.len(),
            truth.steps(),
            "{label}: one decoded row per decode-truth row"
        );
        let cached: Vec<f32> = decoded.concat();
        let cached_deviation =
            self.check_decode_rows(label, DecodeRows::Cached, &cached, &truth.rows);
        let stateless_deviation =
            self.check_decode_rows(label, DecodeRows::Stateless, stateless, &truth.rows);
        let (step, gap) = cached
            .chunks_exact(truth.vocabulary)
            .zip(stateless.chunks_exact(truth.vocabulary))
            .map(|(left, right)| {
                left.iter()
                    .zip(right)
                    .map(|(&a, &b)| (f64::from(a) - f64::from(b)).abs())
                    .fold(0.0, f64::max)
            })
            .enumerate()
            .fold(
                (0, 0.0),
                |worst, (step, gap)| {
                    if gap > worst.1 { (step, gap) } else { worst }
                },
            );
        eprintln!(
            "{label}: cached-versus-stateless gap max |Δ| = {gap:e} at step {step} (diagnostic only; at most {:e} through the decode truth)",
            2.0 * self.decode_bound().maximum
        );
        (cached_deviation, stateless_deviation)
    }

    /// Returns the prompt the official graphs decoded after a quarter-prompt prefill, and the prefill length.
    ///
    /// # Panics
    ///
    /// If the official graphs recorded no prompt decode (Gemma's long-prompt bundle).
    pub(crate) fn prompt_decode_plan(&self) -> (usize, usize) {
        let prompt_decode = self.envelope.prompt_decode.as_ref().expect(
            "the official graphs recorded no prompt_decode rows; generate them with --prompt-decode",
        );
        (prompt_decode.prompt, prompt_decode.prefill)
    }

    /// Checks a prompt decoded after a quarter-prompt prefill against the official prompt-decode envelope.
    ///
    /// `actual` holds the prefill's logits at its positions followed by one
    /// decode step's per later position; `truth` holds the truth's logits of
    /// that prompt.
    ///
    /// # Panics
    ///
    /// If no prompt-decode envelope exists, the rows do not cover the prompt,
    /// hold a non-finite value, or exceed a bound.
    pub(crate) fn check_prompt_decode(
        &self,
        label: &str,
        actual: &[f32],
        truth: &[f32],
    ) -> Deviation {
        let (prompt, prefill) = self.prompt_decode_plan();
        let bound = self
            .envelope
            .prompt_decode
            .as_ref()
            .expect("checked by prompt_decode_plan")
            .envelope
            .bound;
        self.check_all_rows(
            label,
            &format!("prompt {prompt} decoded after a {prefill}-token prefill"),
            self.envelope.prompts[prompt].tokens,
            bound,
            actual,
            truth,
        )
    }

    /// Whether a greedy step of prompt 0 with truth `margin` is compared ([`Envelope::is_decisive`]).
    pub(crate) fn is_decisive(&self, margin: f32) -> bool {
        self.envelope.is_decisive(margin)
    }

    /// Returns the F32 tolerance at which the shared greedy rule compares prompt 0's continuation.
    ///
    /// It is [`largest_f32_not_above`] the greedy bound
    /// ([`Envelope::greedy_bound`]): at it the rule decides exactly the steps
    /// [`Self::is_decisive`] does whenever that bound lies in `[2^-126, 2^127)`.
    pub(crate) fn greedy_tolerance(&self) -> f32 {
        largest_f32_not_above(self.envelope.greedy_bound())
    }
}

#[cfg(test)]
mod tests {
    //! Synthetic tests of dtype parsing, bundle pairing, the envelope rule, and the gate.
    //!
    //! The synthetic bundles have a two-entry vocabulary. Official graph `g`
    //! deviates from the truth by `a_g` at every prompt logit, its cached decode
    //! rows by `2 a_g` everywhere, its stateless decode rows by `2 a_g` at one
    //! entry of four, and its prompt-decode rows by `3 a_g`, so every envelope
    //! is known in closed form.

    use std::panic::catch_unwind;

    use proptest::prelude::*;
    use serde_json::json;

    use super::*;

    /// An environment variable that no test or harness sets.
    const UNSET: &str = "TRANSFORMERS_HALF_PRECISION_TEST_UNSET_VARIABLE";
    /// Token IDs of the two synthetic prompts; prompt 1 is the longest.
    const PROMPTS: [&[i64]; 2] = [&[7, 8], &[7, 8, 9, 10]];
    /// Truth logits of each prompt, `[tokens, 2]` flattened.
    const TRUTH: [&[f32]; 2] = [
        &[1.0, 2.0, 3.0, 4.0],
        &[0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
    ];
    /// Decode-truth stateless rows `[2 steps, 2]`.
    const DECODE: [f32; 4] = [5.0, 6.0, 7.0, 8.0];
    /// Deviation `a_g` of each official graph, in [`OfficialGraph::ALL`] order.
    const AMOUNTS: [f32; 4] = [0.5, 0.25, 0.125, 0.0625];

    /// A described edit of a synthetic manifest.
    type Mutation = (&'static str, fn(&mut Value));

    /// A described edit of a synthetic manifest and a fragment of the panic message it must cause.
    type Refusal = (&'static str, &'static str, fn(&mut Value));

    /// Returns a valid synthetic manifest value with `prompts` (text, token IDs) and vocabulary 2.
    fn manifest_value(dtype: &str, prompts: &[(&str, &[i64])]) -> Value {
        json!({
            "schema_version": 1,
            "model_type": "llama",
            "model_class": "LlamaForCausalLM",
            "vocabulary_size": 2,
            "provenance": {
                "dtype": dtype,
                "device": "cpu",
                "attention_implementation": "eager",
                "script_sha256": "s0",
                "config_sha256": "c0",
                "weights_sha256": {"model.safetensors": "w0"},
                "repository": "owner/name",
                "revision": "r0",
                "transformers": "5.2.0",
                "torch": "2.10.0",
                "num_hidden_layers": null,
                "bf16_resident": false,
            },
            "prompts": prompts.iter().enumerate().map(|(index, (text, tokens))| json!({
                "text": text,
                "token_ids": tokens,
                "logits_file": format!("logits-{index}.f32le"),
                "logits_shape": [tokens.len(), 2],
                "logits_sha256": "",
            })).collect::<Vec<_>>(),
            "greedy": {"prompt_index": 0, "steps": [
                {"token": 1, "margin": 3.0},
                {"token": 0, "margin": 0.5},
            ]},
        })
    }

    /// Parses a manifest value.
    fn parse(value: &Value) -> Manifest {
        Manifest::parse(&value.to_string())
    }

    /// Encodes F32 values as little-endian bytes.
    fn bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    /// Writes `values` to `directory/file` and returns the file's SHA-256.
    fn write_file(directory: &Path, file: &str, values: &[f32]) -> String {
        let payload = bytes(values);
        std::fs::write(directory.join(file), &payload).unwrap();
        hex(&Sha256::digest(&payload))
    }

    /// Returns `values` with `amount` added to every entry.
    fn shifted(values: &[f32], amount: f32) -> Vec<f32> {
        values.iter().map(|value| value + amount).collect()
    }

    /// The options of one synthetic bundle.
    struct BundleSpec {
        /// Provenance dtype label.
        dtype: &'static str,
        /// Provenance device.
        device: &'static str,
        /// Provenance attention implementation.
        attention: &'static str,
        /// Prompt logits, one slice per prompt.
        prompts: Vec<Vec<f32>>,
        /// Cached and stateless decode rows and their token source, if any.
        decode: Option<(Vec<f32>, Vec<f32>, String)>,
        /// Prompt-decode rows of prompt 1, if any.
        prompt_decode: Option<Vec<f32>>,
    }

    /// Writes a bundle described by `spec` into a new directory `root/name`, then applies `edit` to its manifest.
    fn write_spec(
        root: &Path,
        name: &str,
        spec: &BundleSpec,
        edit: impl FnOnce(&mut Value),
    ) -> PathBuf {
        let directory = root.join(name);
        std::fs::create_dir(&directory).unwrap();
        let prompts: Vec<(String, &[i64])> = PROMPTS
            .iter()
            .enumerate()
            .map(|(index, tokens)| (format!("prompt {index}"), *tokens))
            .collect();
        let borrowed: Vec<(&str, &[i64])> = prompts
            .iter()
            .map(|(text, tokens)| (text.as_str(), *tokens))
            .collect();
        let mut manifest = manifest_value(spec.dtype, &borrowed);
        manifest["provenance"]["device"] = json!(spec.device);
        manifest["provenance"]["attention_implementation"] = json!(spec.attention);
        for (index, values) in spec.prompts.iter().enumerate() {
            let digest = write_file(&directory, &format!("logits-{index}.f32le"), values);
            manifest["prompts"][index]["logits_sha256"] = json!(digest);
        }
        if let Some((cached, stateless, source)) = &spec.decode {
            let cached_sha = write_file(&directory, "decode-cached.f32le", cached);
            let stateless_sha = write_file(&directory, "decode-stateless.f32le", stateless);
            manifest["decode_consistency"] = json!({
                "prompt_index": 0,
                "first_position": 2,
                "tokens_source": {"manifest_sha256": source, "dtype": "float32"},
                "logits_shape": [2, 2],
                "cached_logits_file": "decode-cached.f32le",
                "cached_logits_sha256": cached_sha,
                "stateless_logits_file": "decode-stateless.f32le",
                "stateless_logits_sha256": stateless_sha,
                "steps": [
                    {"position": 2, "fed_token": 1, "max_abs_gap": 0.0},
                    {"position": 3, "fed_token": 0, "max_abs_gap": 0.0},
                ],
                "max_abs_gap": 0.0,
                "finite": true,
            });
        }
        if let Some(rows) = &spec.prompt_decode {
            let digest = write_file(&directory, "decode-prompt.f32le", rows);
            manifest["prompt_decode"] = json!({
                "prompt_index": 1,
                "prefill_tokens": 1,
                "logits_file": "decode-prompt.f32le",
                "logits_sha256": digest,
                "logits_shape": [4, 2],
                "finite": true,
            });
        }
        edit(&mut manifest);
        std::fs::write(directory.join("reference.json"), manifest.to_string()).unwrap();
        directory
    }

    /// Applies `edit` to the manifest of the bundle in `directory` and writes it back.
    fn edit_manifest(directory: &Path, edit: impl FnOnce(&mut Value)) {
        let path = directory.join("reference.json");
        let mut manifest: Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        edit(&mut manifest);
        std::fs::write(&path, manifest.to_string()).unwrap();
    }

    /// A synthetic truth, decode truth, and four official graphs in one temporary directory.
    struct Fixture {
        /// Keeps the bundles alive.
        root: tempfile::TempDir,
        /// The truth's directory.
        truth: PathBuf,
        /// SHA-256 of the truth's `reference.json`.
        truth_sha256: String,
        /// The decode truth's directory.
        decode_truth: PathBuf,
        /// The official graphs, in [`OfficialGraph::ALL`] order.
        graphs: Vec<PathBuf>,
        /// Dtype label of the official graphs.
        dtype: &'static str,
    }

    impl Fixture {
        /// Writes a fixture whose official graphs are `dtype` evaluations deviating by `amounts`.
        fn new(dtype: &'static str, amounts: [f32; 4]) -> Self {
            let root = tempfile::tempdir().unwrap();
            let truth = write_spec(
                root.path(),
                "truth",
                &BundleSpec {
                    dtype: "float32",
                    device: "cpu",
                    attention: "eager",
                    prompts: TRUTH.iter().map(|rows| rows.to_vec()).collect(),
                    decode: None,
                    prompt_decode: None,
                },
                |_| {},
            );
            let truth_sha256 = hex(&Sha256::digest(
                std::fs::read(truth.join("reference.json")).unwrap(),
            ));
            let decode_truth = write_spec(
                root.path(),
                "decode-truth",
                &BundleSpec {
                    dtype: "float32",
                    device: "cpu",
                    attention: "eager",
                    prompts: TRUTH.iter().map(|rows| rows.to_vec()).collect(),
                    decode: Some((DECODE.to_vec(), DECODE.to_vec(), truth_sha256.clone())),
                    prompt_decode: None,
                },
                |_| {},
            );
            let mut fixture = Self {
                root,
                truth,
                truth_sha256,
                decode_truth,
                graphs: Vec::new(),
                dtype,
            };
            fixture.graphs = OfficialGraph::ALL
                .into_iter()
                .zip(amounts)
                .map(|(graph, amount)| fixture.graph(graph, amount, |_| {}))
                .collect();
            fixture
        }

        /// Writes one official graph deviating by `amount`, then applies `edit` to its manifest.
        fn graph(
            &self,
            graph: OfficialGraph,
            amount: f32,
            edit: impl FnOnce(&mut Value),
        ) -> PathBuf {
            let (device, attention) = graph.label().split_once('-').unwrap();
            let device: &'static str = if device == "cpu" { "cpu" } else { "mps" };
            let attention: &'static str = if attention == "eager" {
                "eager"
            } else {
                "sdpa"
            };
            let mut stateless = DECODE.to_vec();
            stateless[0] -= 2.0 * amount;
            let name = format!(
                "{}-{}",
                graph.label(),
                std::fs::read_dir(self.root.path()).unwrap().count()
            );
            write_spec(
                self.root.path(),
                &name,
                &BundleSpec {
                    dtype: self.dtype,
                    device,
                    attention,
                    prompts: TRUTH.iter().map(|rows| shifted(rows, amount)).collect(),
                    decode: Some((
                        shifted(&DECODE, 2.0 * amount),
                        stateless,
                        self.truth_sha256.clone(),
                    )),
                    prompt_decode: Some(shifted(TRUTH[1], 3.0 * amount)),
                },
                edit,
            )
        }

        /// Reads the envelope of `graphs` for a `dtype` run.
        fn envelope(&self, dtype: TensorDType, graphs: &[PathBuf]) -> Envelope {
            Envelope::read(dtype, &self.truth, &self.decode_truth, graphs)
        }

        /// Returns the manifests of the truth, decode truth, and `graphs`, ready for [`validate_envelope`].
        fn manifests(
            &self,
            graphs: &[PathBuf],
        ) -> (Manifest, String, Manifest, Vec<(Manifest, String)>) {
            let (truth, sha) = Manifest::read(&self.truth);
            let (decode_truth, _) = Manifest::read(&self.decode_truth);
            let graphs = graphs
                .iter()
                .map(|directory| (Manifest::read(directory).0, directory.display().to_string()))
                .collect();
            (truth, sha, decode_truth, graphs)
        }
    }

    /// Asserts that `operation` panics.
    fn refuses(description: &str, operation: impl FnOnce() + std::panic::UnwindSafe) {
        assert!(
            catch_unwind(operation).is_err(),
            "{description} must be refused"
        );
    }

    /// Asserts that `operation` panics with a message containing `expected`.
    ///
    /// The message is a `String` for formatted panics and assertions and a
    /// `&str` for `Option::expect`; any other payload fails the test.
    fn refuses_with(
        description: &str,
        expected: &str,
        operation: impl FnOnce() + std::panic::UnwindSafe,
    ) {
        let payload = catch_unwind(operation)
            .err()
            .unwrap_or_else(|| panic!("{description} must be refused"));
        let message = payload
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| payload.downcast_ref::<&str>().copied())
            .unwrap_or_else(|| panic!("{description}: the panic carries no message"));
        assert!(
            message.contains(expected),
            "{description} must be refused with {expected:?}, not {message:?}"
        );
    }

    /// Asserts that pairing `graphs` with the fixture's truth for a `run` panics with `expected`.
    fn pairing_refuses(fixture: &Fixture, graphs: &[PathBuf], run: TensorDType, expected: &str) {
        let (truth, sha, decode_truth, graphs) = fixture.manifests(graphs);
        refuses_with(expected, expected, move || {
            validate_envelope(&truth, &sha, &decode_truth, &graphs, run);
        });
    }

    /// Unset means F32; only the exact spellings f32, f16, and bf16 are accepted.
    #[test]
    fn run_dtype_defaults_to_f32_and_accepts_only_three_spellings() {
        assert_eq!(parse_run_dtype("V", None), TensorDType::F32);
        for (spelling, dtype) in [
            ("f32", TensorDType::F32),
            ("f16", TensorDType::F16),
            ("bf16", TensorDType::BF16),
        ] {
            assert_eq!(parse_run_dtype("V", Some(OsStr::new(spelling))), dtype);
        }
        for spelling in [
            "", "f64", "F16", "BF16", "fp16", "bfloat16", "float32", " f16",
        ] {
            refuses(spelling, || {
                parse_run_dtype("V", Some(OsStr::new(spelling)));
            });
        }
        #[cfg(unix)]
        {
            use std::os::unix::ffi::OsStrExt;
            refuses("non-UTF-8", || {
                parse_run_dtype("V", Some(OsStr::from_bytes(b"f1\xff")));
            });
        }
        assert!(is_half_precision(TensorDType::F16) && is_half_precision(TensorDType::BF16));
        assert!(!is_half_precision(TensorDType::F32) && !is_half_precision(TensorDType::F64));
    }

    /// Set F32 overrides and retired variables are found in order and refused; none set passes.
    #[test]
    fn overrides_and_retired_variables_are_found_and_refused() {
        let names = ["A_TOLERANCE", "B_TOLERANCE", "C_TOLERANCE"];
        assert_eq!(
            present_overrides(&names, |name| name != "B_TOLERANCE"),
            ["A_TOLERANCE", "C_TOLERANCE"]
        );
        assert!(present_overrides(&names, |_| false).is_empty());
        reject_overrides(&[UNSET]);
        refuses("a set override", || reject_overrides(&["PATH"]));
        let pairs = [("OLD_A", "NEW_A"), ("OLD_B", "NEW_B")];
        assert_eq!(
            present_retired(&pairs, |name| name == "OLD_B"),
            [("OLD_B", "NEW_B")]
        );
        refuse_retired(&[(UNSET, "NEW")]);
        refuses_with(
            "a set retired variable",
            "PATH retired: a half-precision run is judged by the envelope of the four official graphs; name them in NEW",
            || refuse_retired(&[("PATH", "NEW")]),
        );
    }

    /// The environment readers default an unset dtype to F32, require directories, and split path lists.
    #[test]
    fn environment_readers_default_the_dtype_and_require_directories() {
        assert_eq!(run_dtype(UNSET), TensorDType::F32);
        assert_eq!(
            required_directory("PATH"),
            PathBuf::from(std::env::var_os("PATH").unwrap())
        );
        refuses("an unset directory", || {
            required_directory(UNSET);
        });
        refuses("an unset directory list", || {
            required_directories(UNSET);
        });
        assert!(!required_directories("PATH").is_empty());
        let joined = std::env::join_paths(["/a b/c", "/d"]).unwrap();
        assert_eq!(
            split_directories(&joined),
            [PathBuf::from("/a b/c"), PathBuf::from("/d")]
        );
        assert!(split_directories(OsStr::new("")).is_empty());
    }

    /// Truth bundles are float32 or float64, official ones bfloat16 or float16; other labels are refused.
    #[test]
    fn every_bundle_dtype_label_has_one_role() {
        for (label, dtype) in [
            ("float32", BundleDtype::Float32),
            ("float64", BundleDtype::Float64),
            ("bfloat16", BundleDtype::BFloat16),
            ("float16", BundleDtype::Float16),
        ] {
            assert_eq!(BundleDtype::parse(label), dtype);
            assert_eq!(dtype.label(), label);
        }
        assert!(BundleDtype::Float32.is_truth() && BundleDtype::Float64.is_truth());
        assert!(!BundleDtype::BFloat16.is_truth() && !BundleDtype::Float16.is_truth());
        assert_eq!(
            BundleDtype::of_run(TensorDType::BF16),
            BundleDtype::BFloat16
        );
        assert_eq!(BundleDtype::of_run(TensorDType::F16), BundleDtype::Float16);
        for run in [TensorDType::F32, TensorDType::F64] {
            refuses("a full-precision run", move || {
                BundleDtype::of_run(run);
            });
        }
        for label in ["Float32", "bf16", "f32", "", "int8", "half"] {
            refuses(label, || {
                BundleDtype::parse(label);
            });
        }
    }

    /// The four official graphs parse from their provenance labels, in a fixed order; others do not.
    #[test]
    fn official_graphs_are_the_four_device_and_attention_pairs() {
        let labels: Vec<&str> = OfficialGraph::ALL
            .iter()
            .map(|graph| graph.label())
            .collect();
        assert_eq!(labels, ["cpu-eager", "cpu-sdpa", "mps-eager", "mps-sdpa"]);
        for graph in OfficialGraph::ALL {
            let (device, attention) = graph.label().split_once('-').unwrap();
            assert_eq!(OfficialGraph::parse(device, attention), Some(graph));
        }
        let mut sorted = OfficialGraph::ALL;
        sorted.reverse();
        sorted.sort();
        assert_eq!(sorted, OfficialGraph::ALL);
        for (device, attention) in [
            ("cuda", "eager"),
            ("cpu", "flash_attention_2"),
            ("", ""),
            ("CPU", "eager"),
        ] {
            assert_eq!(OfficialGraph::parse(device, attention), None);
        }
    }

    /// Any difference between a bundle's and the truth's identities, prompts, or options fails closed.
    #[test]
    fn identity_refuses_every_mismatch() {
        let prompts: &[(&str, &[i64])] = &[("a", &[1, 2]), ("b", &[3])];
        let truth = parse(&manifest_value("float32", prompts));
        let other = manifest_value("bfloat16", prompts);
        validate_identity(&truth, &parse(&other), "other");
        let mutations: [Mutation; 13] = [
            ("model type", |m| m["model_type"] = json!("qwen2")),
            ("model class", |m| {
                m["model_class"] = json!("Qwen2ForCausalLM")
            }),
            ("config", |m| m["provenance"]["config_sha256"] = json!("c1")),
            ("weights", |m| {
                m["provenance"]["weights_sha256"]["model.safetensors"] = json!("w1")
            }),
            ("extra weight file", |m| {
                m["provenance"]["weights_sha256"]["extra.safetensors"] = json!("w0")
            }),
            ("repository", |m| {
                m["provenance"]["repository"] = json!("other/name")
            }),
            ("revision", |m| m["provenance"]["revision"] = Value::Null),
            ("transformers", |m| {
                m["provenance"]["transformers"] = json!("5.1.0")
            }),
            ("torch", |m| m["provenance"]["torch"] = json!("2.9.0")),
            ("truncation", |m| {
                m["provenance"]["num_hidden_layers"] = json!(4)
            }),
            ("prompt text", |m| m["prompts"][1]["text"] = json!("c")),
            ("token IDs", |m| m["prompts"][1]["token_ids"] = json!([4])),
            ("prompt count", |m| {
                m["prompts"].as_array_mut().unwrap().pop();
            }),
        ];
        for (what, mutate) in mutations {
            let mut changed = other.clone();
            mutate(&mut changed);
            let changed = parse(&changed);
            let truth = truth.clone();
            refuses(what, move || validate_identity(&truth, &changed, "other"));
        }
        let mut wider = other.clone();
        wider["vocabulary_size"] = json!(3);
        wider["prompts"][0]["logits_shape"] = json!([2, 3]);
        wider["prompts"][1]["logits_shape"] = json!([1, 3]);
        let wider = parse(&wider);
        refuses_with(
            "a vocabulary mismatch",
            "other and the reference bundle differ in vocabulary size",
            move || validate_identity(&truth, &wider, "other"),
        );
    }

    /// Malformed, mistyped, or incomplete manifests are refused, each for its own reason, before any logit is read.
    #[test]
    fn malformed_manifests_are_refused() {
        let prompts: &[(&str, &[i64])] = &[("a", &[1, 2])];
        let valid = manifest_value("float32", prompts);
        parse(&valid);
        let refusals: [Refusal; 39] = [
            ("old schema", "unsupported bundle schema 0", |m| {
                m["schema_version"] = json!(0)
            }),
            ("new schema", "unsupported bundle schema 2", |m| {
                m["schema_version"] = json!(2)
            }),
            (
                "textual schema",
                "schema_version must be an unsigned integer",
                |m| m["schema_version"] = json!("1"),
            ),
            ("vocabulary", "at least two entries", |m| {
                m["vocabulary_size"] = json!(1)
            }),
            (
                "textual vocabulary",
                "vocabulary_size must be an unsigned integer",
                |m| m["vocabulary_size"] = json!("2"),
            ),
            ("mistyped model type", "model_type must be a string", |m| {
                m["model_type"] = json!(7)
            }),
            ("missing model class", "missing model_class", |m| {
                m.as_object_mut().unwrap().remove("model_class");
            }),
            ("missing provenance", "missing provenance", |m| {
                m.as_object_mut().unwrap().remove("provenance");
            }),
            ("missing dtype", "missing provenance.dtype", |m| {
                m["provenance"].as_object_mut().unwrap().remove("dtype");
            }),
            ("missing device", "missing provenance.device", |m| {
                m["provenance"].as_object_mut().unwrap().remove("device");
            }),
            (
                "missing attention",
                "missing provenance.attention_implementation",
                |m| {
                    m["provenance"]
                        .as_object_mut()
                        .unwrap()
                        .remove("attention_implementation");
                },
            ),
            (
                "mistyped script digest",
                "provenance.script_sha256 must be a string or null",
                |m| m["provenance"]["script_sha256"] = json!(1),
            ),
            (
                "mistyped revision",
                "provenance.revision must be a string or null",
                |m| m["provenance"]["revision"] = json!(5),
            ),
            (
                "weights not an object",
                "weights_sha256 must be an object",
                |m| m["provenance"]["weights_sha256"] = json!(["w0"]),
            ),
            (
                "mistyped weight digest",
                "every weight digest must be a string",
                |m| m["provenance"]["weights_sha256"]["model.safetensors"] = json!(0),
            ),
            ("no weights", "names no weight file", |m| {
                m["provenance"]["weights_sha256"] = json!({})
            }),
            (
                "layers",
                "num_hidden_layers must be a positive integer",
                |m| m["provenance"]["num_hidden_layers"] = json!(0),
            ),
            ("flag", "bf16_resident must be a boolean", |m| {
                m["provenance"]["bf16_resident"] = json!("no")
            }),
            ("prompts not an array", "prompts must be an array", |m| {
                m["prompts"] = json!({})
            }),
            ("no prompts", "at least one prompt", |m| {
                m["prompts"] = json!([])
            }),
            (
                "non-integer token ID",
                "prompts[0].token_ids must hold integers",
                |m| m["prompts"][0]["token_ids"] = json!(["x", 2]),
            ),
            ("empty tokens", "prompts[0].token_ids is empty", |m| {
                m["prompts"][0]["token_ids"] = json!([]);
                m["prompts"][0]["logits_shape"] = json!([0, 2]);
            }),
            ("shape", "prompts[0].logits_shape must be [2, 2]", |m| {
                m["prompts"][0]["logits_shape"] = json!([1, 2])
            }),
            ("path", "logits_file must be a plain file name", |m| {
                m["prompts"][0]["logits_file"] = json!("../logits-0.f32le")
            }),
            (
                "Windows path",
                "logits_file must be a plain file name",
                |m| m["prompts"][0]["logits_file"] = json!("sub\\logits-0.f32le"),
            ),
            ("missing digest", "missing prompts[0].logits_sha256", |m| {
                m["prompts"][0]
                    .as_object_mut()
                    .unwrap()
                    .remove("logits_sha256");
            }),
            (
                "textual finite flag",
                "prompts[0].finite must be a boolean",
                |m| m["prompts"][0]["finite"] = json!("no"),
            ),
            (
                "non-finite without a marker",
                "missing prompts[0].first_non_finite",
                |m| m["prompts"][0]["finite"] = json!(false),
            ),
            (
                "marker beyond the logits",
                "prompts[0].first_non_finite (2, 0) lies outside [2, 2]",
                |m| {
                    m["prompts"][0]["finite"] = json!(false);
                    m["prompts"][0]["first_non_finite"] = json!({"position": 2, "token": 0});
                },
            ),
            ("greedy prompt", "must extend prompt 0", |m| {
                m["greedy"]["prompt_index"] = json!(1)
            }),
            (
                "textual greedy token",
                "a greedy token must be an integer",
                |m| m["greedy"]["steps"][0]["token"] = json!("1"),
            ),
            ("margin overflow", "is negative or not finite in F32", |m| {
                m["greedy"]["steps"][0]["margin"] = json!(1e39)
            }),
            (
                "other stop reason",
                "greedy.stopped must be \"non_finite\"",
                |m| m["greedy"]["stopped"] = json!("length"),
            ),
            (
                "decode rows of prompt 1",
                "decode rows must extend prompt 0",
                |m| m["decode_consistency"] = json!({"prompt_index": 1}),
            ),
            (
                "decode shape",
                "decode_consistency.logits_shape must be [1, 2]",
                |m| {
                    m["decode_consistency"] = json!({
                        "prompt_index": 0, "steps": [{"position": 2, "fed_token": 1}],
                        "logits_shape": [2, 2],
                    })
                },
            ),
            (
                "decode file path",
                "decode_consistency.cached_logits_file must be a plain file name",
                |m| {
                    m["decode_consistency"] = json!({
                        "prompt_index": 0, "first_position": 2, "steps": [{"position": 2, "fed_token": 1}],
                        "logits_shape": [1, 2], "tokens_source": {"manifest_sha256": "x"},
                        "cached_logits_file": "../c", "cached_logits_sha256": "",
                        "stateless_logits_file": "s", "stateless_logits_sha256": "",
                    })
                },
            ),
            (
                "prompt-decode index",
                "prompt_decode.prompt_index 1 names no prompt",
                |m| m["prompt_decode"] = json!({"prompt_index": 1}),
            ),
            (
                "prompt-decode prefill",
                "prompt_decode.prefill_tokens 3 must lie in 1..=2",
                |m| m["prompt_decode"] = json!({"prompt_index": 0, "prefill_tokens": 3}),
            ),
            (
                "prompt-decode shape",
                "prompt_decode.logits_shape must be [2, 2]",
                |m| {
                    m["prompt_decode"] = json!({
                        "prompt_index": 0, "prefill_tokens": 1, "logits_shape": [1, 2],
                    })
                },
            ),
        ];
        for (what, expected, mutate) in refusals {
            let mut changed = valid.clone();
            mutate(&mut changed);
            refuses_with(what, expected, move || {
                parse(&changed);
            });
        }
        let mut negative = valid.clone();
        negative["greedy"]["steps"][0]["margin"] = json!(-1.0);
        refuses_with("negative margin", "greedy margin -1", move || {
            parse(&negative);
        });
        refuses_with("not JSON", "reference.json is not JSON", || {
            Manifest::parse("{");
        });
        // Optional markers parse when well formed; absent means finite.
        let mut marked = valid.clone();
        marked["prompts"][0]["finite"] = json!(false);
        marked["prompts"][0]["first_non_finite"] = json!({"position": 1, "token": 0});
        marked["greedy"]["stopped"] = json!("non_finite");
        let marked = parse(&marked);
        assert_eq!(
            marked.prompts[0].non_finite,
            Some(NonFinite { row: 1, token: 0 })
        );
        assert!(marked.greedy_stopped);
        let plain = parse(&valid);
        assert_eq!(plain.prompts[0].non_finite, None);
        assert!(!plain.greedy_stopped && plain.decode.is_none() && plain.prompt_decode.is_none());
        assert_eq!(plain.graph(), Some(OfficialGraph::ALL[0]));
    }

    /// The deviation reports the first largest gap and an F64 mean; `try_measure` returns the first non-finite index.
    #[test]
    fn deviation_locates_the_first_largest_gap_and_the_first_non_finite_logit() {
        let truth = [0.0, 1.0, 2.0, 3.0];
        let deviation = Deviation::measure("t", &[0.5, 1.0, 0.0, 5.0], &truth);
        assert_eq!((deviation.maximum, deviation.index), (2.0, 2));
        assert_eq!(deviation.mean, 4.5 / 4.0);
        let exact = Deviation::measure("t", &truth, &truth);
        assert_eq!((exact.maximum, exact.index, exact.mean), (0.0, 0, 0.0));
        // The difference is taken in F64: F32 subtraction would round this one.
        let small = Deviation::measure("t", &[1.0 + f32::EPSILON], &[-f32::EPSILON / 4.0]);
        assert_eq!(small.maximum, 1.0 + 1.25 * f64::from(f32::EPSILON));
        assert_eq!(
            Deviation::try_measure("t", &[0.0, f32::NAN, f32::INFINITY, 0.0], &truth),
            Err(1)
        );
        assert_eq!(Deviation::try_measure("t", &truth, &truth), Ok(exact));
        refuses_with(
            "a NaN logit",
            "t: the compared logits hold a non-finite value",
            || {
                Deviation::measure("t", &[f32::NAN, 0.0], &[0.0, 0.0]);
            },
        );
        for (actual, expected) in [
            (vec![0.0, 0.0], vec![f32::NEG_INFINITY, 0.0]),
            (vec![0.0], vec![0.0, 0.0]),
            (vec![], vec![]),
        ] {
            refuses("invalid logits", move || {
                Deviation::measure("t", &actual, &expected);
            });
        }
    }

    /// Bounds are the dtype's two slacks times the envelope, inclusive at the bound, and zero envelopes are refused.
    #[test]
    fn rule_applies_separate_slacks_and_admits_ties() {
        const {
            assert!(MAX_SLACK.len() == 2 && MEAN_SLACK.len() == 2);
        };
        assert_eq!(
            (max_slack(TensorDType::BF16), mean_slack(TensorDType::BF16)),
            (1.5, 1.3)
        );
        assert_eq!(
            (max_slack(TensorDType::F16), mean_slack(TensorDType::F16)),
            (1.7, 1.7)
        );
        for dtype in [TensorDType::F32, TensorDType::F64] {
            refuses("a full-precision slack", move || {
                max_slack(dtype);
            });
            refuses("a full-precision slack", move || {
                mean_slack(dtype);
            });
        }
        let envelope = Statistic {
            maximum: 1.25,
            mean: 0.125,
        };
        for dtype in [TensorDType::BF16, TensorDType::F16] {
            let bound = Bound::from_envelope("p", &envelope, dtype);
            assert_eq!(bound.maximum, max_slack(dtype) * 1.25);
            assert_eq!(bound.mean, mean_slack(dtype) * 0.125);
            let at = |maximum: f64, mean: f64| Deviation {
                maximum,
                index: 0,
                mean,
            };
            assert!(bound.admits(&at(bound.maximum, bound.mean)), "ties pass");
            assert!(!bound.admits(&at(bound.maximum.next_up(), 0.0)));
            assert!(!bound.admits(&at(0.0, bound.mean.next_up())));
            assert!(
                bound.admits_rows(&at(bound.maximum, 1e9)),
                "rows ignore the mean"
            );
            assert!(!bound.admits_rows(&at(bound.maximum.next_up(), 0.0)));
        }
        for (maximum, mean) in [
            (0.0, 0.1),
            (0.1, 0.0),
            (f64::NAN, 0.1),
            (f64::INFINITY, 0.1),
            (0.1, f64::NAN),
        ] {
            refuses_with(
                "a degenerate envelope",
                "p: the official envelope",
                move || {
                    Bound::from_envelope("p", &Statistic { maximum, mean }, TensorDType::BF16);
                },
            );
        }
    }

    /// Greedy steps are compared exactly when the margin exceeds twice the maximum bound; ties are skipped.
    #[test]
    fn greedy_steps_are_compared_only_beyond_twice_the_bound() {
        let bound = Bound {
            maximum: 0.75,
            mean: 0.1,
        };
        assert_eq!(bound.greedy_threshold(), 1.5);
        assert!(
            !bound.is_decisive(1.5),
            "a margin at the threshold is a tie"
        );
        assert!(bound.is_decisive(1.5_f32.next_up()));
        let greedy = [(1, 3.0), (2, 1.5), (3, 0.2), (4, 2.0), (5, 1.0)];
        assert_eq!(decisive_steps(&greedy, &bound), [0, 3]);
        assert!(decisive_steps(&greedy[1..3], &bound).is_empty());
        assert!(decisive_steps(&[], &bound).is_empty());
    }

    /// Returns `2^exponent` exactly, for an exponent in F64's normal range.
    fn two_to(exponent: i32) -> f64 {
        f64::from_bits(u64::try_from(1023 + exponent).unwrap() << 52)
    }

    /// The F32 greedy tolerance makes the shared rule `m > 2 t` decide exactly the decisive steps.
    ///
    /// The bounds span the documented range `[2^-126, 2^127)`: exact F32
    /// values, values between two F32 values, F32's smallest normal value
    /// itself, values between F32 values of its binade, and values near the top
    /// of the range. Every F32 margin within eight units in the last place of
    /// each threshold is decided both ways.
    #[test]
    fn greedy_tolerance_decides_exactly_the_decisive_steps() {
        for maximum in [
            0.75,
            1.0 / 3.0,
            1.5 * 1.463_715_076_446_533_2,
            1.5 * 10.183_731_079_101_562,
            1e-30,
            f64::from(f32::MIN_POSITIVE),
            f64::from(f32::MIN_POSITIVE) + two_to(-150),
            two_to(-125) - two_to(-151),
            f64::from(f32::MAX) / 4.0,
            f64::from(f32::MAX) / 2.0,
            f64::from(f32::MAX) / 2.0 + two_to(100),
        ] {
            let bound = Bound { maximum, mean: 0.1 };
            let tolerance = bound.greedy_tolerance();
            assert_eq!(tolerance, largest_f32_not_above(maximum));
            assert!(
                f64::from(tolerance) <= maximum && f64::from(tolerance.next_up()) > maximum,
                "{tolerance:e} is the largest F32 value not above {maximum:e}"
            );
            let mut margin = bound.greedy_threshold() as f32;
            for _ in 0..8 {
                margin = margin.next_down();
            }
            for _ in 0..17 {
                assert_eq!(
                    margin > 2.0 * tolerance,
                    bound.is_decisive(margin),
                    "margin {margin:e} against the bound {maximum:e}"
                );
                margin = margin.next_up();
            }
        }
        assert_eq!(largest_f32_not_above(0.75), 0.75, "an F32 bound is kept");
        assert!(f64::from(largest_f32_not_above(1.0 / 3.0)) < 1.0 / 3.0);
    }

    /// Outside `[2^-126, 2^127)` the two greedy rules can disagree, as [`largest_f32_not_above`] documents.
    #[test]
    fn greedy_tolerance_can_disagree_outside_the_normal_range() {
        for maximum in [
            two_to(-150),
            two_to(-127) + two_to(-150),
            two_to(-126) - two_to(-151),
        ] {
            let bound = Bound { maximum, mean: 0.1 };
            let tolerance = bound.greedy_tolerance();
            assert!(!tolerance.is_normal(), "{tolerance:e} is subnormal or zero");
            let margin = (2.0 * tolerance).next_up();
            assert_eq!(f64::from(margin), 2.0 * f64::from(tolerance) + two_to(-149));
            assert!(
                margin > 2.0 * tolerance && !bound.is_decisive(margin),
                "margin {margin:e} is decisive only under the F32 rule at the bound {maximum:e}"
            );
        }
        let overflowing = Bound {
            maximum: two_to(127),
            mean: 0.1,
        };
        let tolerance = overflowing.greedy_tolerance();
        assert_eq!(2.0 * tolerance, f32::INFINITY);
        for margin in [f32::MAX.next_down(), f32::MAX] {
            assert!(margin <= 2.0 * tolerance && !overflowing.is_decisive(margin));
        }
        assert!(f32::INFINITY <= 2.0 * tolerance && overflowing.is_decisive(f32::INFINITY));
    }

    /// The provisional self-consistency tolerances exist exactly for the half-precision dtypes.
    #[test]
    fn self_consistency_tolerances_cover_only_half_precision() {
        assert_eq!(self_consistency_tolerance(TensorDType::F16), 0.5);
        assert_eq!(self_consistency_tolerance(TensorDType::BF16), 0.5);
        for dtype in [TensorDType::F32, TensorDType::F64] {
            refuses("a full-precision dtype", move || {
                self_consistency_tolerance(dtype);
            });
        }
    }

    /// The longest prompt is the first of equal ones and the prefill is a quarter of it, at least one token.
    #[test]
    fn decoded_prompt_follows_the_generator_protocol() {
        let prompts: Vec<BundlePrompt> = [3, 13, 13, 8]
            .into_iter()
            .map(|tokens| BundlePrompt {
                text: String::new(),
                token_ids: vec![0; tokens],
                logits_file: "f".into(),
                logits_sha256: String::new(),
                non_finite: None,
            })
            .collect();
        assert_eq!(longest_prompt(&prompts), 1);
        assert_eq!(longest_prompt(&prompts[3..]), 0);
        assert_eq!(
            [1, 3, 4, 7, 8, 13, 1237].map(quarter_prefill),
            [1, 1, 1, 1, 2, 3, 309]
        );
        refuses("no prompt", || {
            longest_prompt(&[]);
        });
    }

    /// The envelope is the largest official statistic of each kind; ratios are leave-one-out; the report runs.
    #[test]
    fn envelope_is_the_largest_official_deviation_of_each_kind() {
        let fixture = Fixture::new("bfloat16", AMOUNTS);
        // Hand the graphs over in reverse: the envelope orders them itself.
        let mut graphs = fixture.graphs.clone();
        graphs.reverse();
        let envelope = fixture.envelope(TensorDType::BF16, &graphs);
        assert_eq!(envelope.graph_dirs.as_slice(), fixture.graphs.as_slice());
        assert_eq!(envelope.prompts.len(), 2);
        let prompt = &envelope.prompts[0];
        assert_eq!(prompt.tokens, 2);
        assert_eq!(
            prompt.envelope.envelope,
            Statistic {
                maximum: 0.5,
                mean: 0.5
            }
        );
        assert_eq!(
            prompt.envelope.bound,
            Bound {
                maximum: 0.75,
                mean: 0.65
            }
        );
        assert_eq!(prompt.deviations[3].maximum, 0.0625);
        let ratios = prompt.envelope.leave_one_out();
        assert_eq!(
            ratios[0],
            Statistic {
                maximum: 2.0,
                mean: 2.0
            }
        );
        assert_eq!(
            ratios[1],
            Statistic {
                maximum: 0.5,
                mean: 0.5
            }
        );
        assert_eq!(
            envelope.decode.envelope.envelope,
            Statistic {
                maximum: 1.0,
                mean: 1.0
            },
            "cached rows deviate by 2a everywhere"
        );
        assert_eq!(envelope.decode.stateless[0].mean, 0.25);
        assert_eq!(
            envelope.decode.envelope.bound,
            Bound {
                maximum: 1.5,
                mean: 1.3
            }
        );
        let prompt_decode = envelope.prompt_decode.unwrap();
        assert_eq!((prompt_decode.prompt, prompt_decode.prefill), (1, 1));
        assert_eq!(prompt_decode.envelope.envelope.maximum, 1.5);
        assert_eq!(envelope.envelopes().len(), 4);
        // Every graph but cpu-eager lies within the others' envelope; cpu-eager needs 2x.
        let outside: Vec<String> = envelope
            .leave_one_out()
            .into_iter()
            .filter(|(_, _, _, admitted)| !admitted)
            .map(|(rows, graph, _, _)| format!("{rows} {}", graph.label()))
            .collect();
        assert_eq!(
            outside,
            [
                "prompt 0 cpu-eager",
                "prompt 1 cpu-eager",
                "decode rows cpu-eager",
                "prompt 1 decoded rows cpu-eager"
            ]
        );
        refuses_with(
            "a graph outside the others' envelope",
            "official graphs outside the envelope of the other three under the compiled BF16 slacks: prompt 0 cpu-eager",
            {
                let envelope = envelope.clone();
                move || envelope.assert_leave_one_out()
            },
        );
        // Greedy steps are judged at the larger of prompt 0's (0.75) and the decode rows' (1.5) maximum bound.
        assert_eq!(envelope.greedy_bound(), 1.5);
        assert!(envelope.is_decisive(3.0_f32.next_up()) && !envelope.is_decisive(3.0));
        assert!(envelope.decisive_greedy_steps().is_empty());
        envelope.report(&fixture.truth);
        // Graphs within a factor 1.3 of each other admit each other leave-one-out.
        let close = Fixture::new("bfloat16", [0.5, 0.5, 0.4375, 0.4375]);
        close
            .envelope(TensorDType::BF16, &close.graphs)
            .assert_leave_one_out();
    }

    /// Every pairing refusal of the official graphs: count, labels, dtype, script, identity, greedy, decode rows, prompt decode.
    #[test]
    fn pairing_refuses_every_mismatched_official_graph() {
        let fixture = Fixture::new("bfloat16", AMOUNTS);
        let graphs = &fixture.graphs;
        let (truth, sha, decode_truth, named) = fixture.manifests(graphs);
        validate_envelope(&truth, &sha, &decode_truth, &named, TensorDType::BF16);
        let with = |slot: usize, path: PathBuf| {
            let mut changed = graphs.clone();
            changed[slot] = path;
            changed
        };
        pairing_refuses(
            &fixture,
            &graphs[..3],
            TensorDType::BF16,
            "exactly 4 official graphs, not 3",
        );
        pairing_refuses(
            &fixture,
            &with(1, graphs[2].clone()),
            TensorDType::BF16,
            "official graph cpu-sdpa appears 0 times",
        );
        pairing_refuses(
            &fixture,
            &[
                graphs[0].clone(),
                graphs[0].clone(),
                graphs[2].clone(),
                graphs[3].clone(),
            ],
            TensorDType::BF16,
            "official graph cpu-eager appears 2 times",
        );
        let cuda = fixture.graph(OfficialGraph::ALL[0], 0.5, |m| {
            m["provenance"]["device"] = json!("cuda")
        });
        pairing_refuses(
            &fixture,
            &with(0, cuda),
            TensorDType::BF16,
            "cpu-eager appears 0 times",
        );
        let flash = fixture.graph(OfficialGraph::ALL[3], 0.5, |m| {
            m["provenance"]["attention_implementation"] = json!("flash_attention_2")
        });
        pairing_refuses(
            &fixture,
            &with(3, flash),
            TensorDType::BF16,
            "mps-sdpa appears 0 times",
        );
        pairing_refuses(
            &fixture,
            graphs,
            TensorDType::F16,
            "a bfloat16 evaluation cannot calibrate a F16 run",
        );
        let half = fixture.graph(OfficialGraph::ALL[0], 0.5, |m| {
            m["provenance"]["dtype"] = json!("float16")
        });
        pairing_refuses(
            &fixture,
            &with(0, half),
            TensorDType::BF16,
            "a float16 evaluation cannot calibrate a BF16 run",
        );
        let resident = fixture.graph(OfficialGraph::ALL[0], 0.5, |m| {
            m["provenance"]["bf16_resident"] = json!(true)
        });
        pairing_refuses(
            &fixture,
            &with(0, resident),
            TensorDType::BF16,
            "not --bf16-resident",
        );
        let script = fixture.graph(OfficialGraph::ALL[2], 0.5, |m| {
            m["provenance"]["script_sha256"] = json!("s1")
        });
        pairing_refuses(
            &fixture,
            &with(2, script),
            TensorDType::BF16,
            "one generator script",
        );
        let unscripted = fixture.graph(OfficialGraph::ALL[0], 0.5, |m| {
            m["provenance"]
                .as_object_mut()
                .unwrap()
                .remove("script_sha256");
        });
        pairing_refuses(
            &fixture,
            &with(0, unscripted),
            TensorDType::BF16,
            "one generator script",
        );
        let edits: [Refusal; 9] = [
            ("weights", "differ in weight files", |m| {
                m["provenance"]["weights_sha256"]["model.safetensors"] = json!("w1")
            }),
            ("tokens", "differ in prompt 1 token IDs", |m| {
                m["prompts"][1]["token_ids"] = json!([7, 8, 9, 11])
            }),
            (
                "short greedy",
                "greedy length 1 differs from the reference bundle's 2",
                |m| {
                    m["greedy"]["steps"].as_array_mut().unwrap().pop();
                },
            ),
            ("no decode rows", "has no decode rows", |m| {
                m.as_object_mut().unwrap().remove("decode_consistency");
            }),
            (
                "decode start",
                "decode rows must start at position 2",
                |m| m["decode_consistency"]["first_position"] = json!(3),
            ),
            (
                "decode positions",
                "decode rows must cover positions 2..4",
                |m| m["decode_consistency"]["steps"][1]["position"] = json!(5),
            ),
            (
                "fed tokens",
                "decode rows must feed the reference bundle's greedy tokens",
                |m| m["decode_consistency"]["steps"][0]["fed_token"] = json!(0),
            ),
            (
                "token source",
                "decode tokens_source must be the reference bundle's",
                |m| m["decode_consistency"]["tokens_source"]["manifest_sha256"] = json!("0"),
            ),
            (
                "prompt decode",
                "prompt_decode must decode the longest prompt 1 after a 1-token prefill",
                |m| {
                    m["prompt_decode"]["prompt_index"] = json!(0);
                    m["prompt_decode"]["logits_shape"] = json!([2, 2]);
                },
            ),
        ];
        for (what, expected, edit) in edits {
            let edited = fixture.graph(OfficialGraph::ALL[1], 0.25, edit);
            let (truth, sha, decode_truth, named) = fixture.manifests(&with(1, edited));
            refuses_with(what, expected, move || {
                validate_envelope(&truth, &sha, &decode_truth, &named, TensorDType::BF16);
            });
        }
        let stopped = fixture.graph(OfficialGraph::ALL[1], 0.25, |m| {
            m["greedy"]["steps"].as_array_mut().unwrap().pop();
            m["greedy"]["stopped"] = json!("non_finite");
        });
        let (truth, sha, decode_truth, named) = fixture.manifests(&with(1, stopped));
        validate_envelope(&truth, &sha, &decode_truth, &named, TensorDType::BF16);
        let missing = fixture.graph(OfficialGraph::ALL[1], 0.25, |m| {
            m.as_object_mut().unwrap().remove("prompt_decode");
        });
        pairing_refuses(
            &fixture,
            &with(1, missing),
            TensorDType::BF16,
            "has no prompt_decode rows, which the other official graphs hold",
        );
        refuses("an F32 run", || {
            BundleDtype::of_run(TensorDType::F32);
        });
    }

    /// The truth and the decode truth are paired fail closed.
    #[test]
    fn pairing_refuses_a_mismatched_truth_or_decode_truth() {
        let fixture = Fixture::new("float16", AMOUNTS);
        let (truth, sha, decode_truth, named) = fixture.manifests(&fixture.graphs);
        validate_envelope(&truth, &sha, &decode_truth, &named, TensorDType::F16);
        let decode_edits: [Refusal; 6] = [
            (
                "half-precision decode truth",
                "must be a float32 or float64 evaluation, not bfloat16",
                |m| m["provenance"]["dtype"] = json!("bfloat16"),
            ),
            ("SDPA decode truth", "must be a CPU eager evaluation", |m| {
                m["provenance"]["attention_implementation"] = json!("sdpa")
            }),
            ("resident decode truth", "must be a plain evaluation", |m| {
                m["provenance"]["bf16_resident"] = json!(true)
            }),
            (
                "other prompt logits",
                "prompt 1 logits differ from the reference bundle's",
                |m| m["prompts"][1]["logits_sha256"] = json!("0"),
            ),
            (
                "other revision",
                "the decode-truth bundle and the reference bundle differ in revision",
                |m| m["provenance"]["revision"] = json!("r1"),
            ),
            (
                "other source",
                "decode tokens_source must be the reference bundle's",
                |m| m["decode_consistency"]["tokens_source"]["manifest_sha256"] = json!("1"),
            ),
        ];
        for (what, expected, edit) in decode_edits {
            let mut changed: Value = serde_json::from_str(
                &std::fs::read_to_string(fixture.decode_truth.join("reference.json")).unwrap(),
            )
            .unwrap();
            edit(&mut changed);
            let changed = parse(&changed);
            let (truth, sha, named) = (truth.clone(), sha.clone(), named.clone());
            refuses_with(what, expected, move || {
                validate_envelope(&truth, &sha, &changed, &named, TensorDType::F16);
            });
        }
        let mut half_truth = truth.clone();
        half_truth.dtype = BundleDtype::BFloat16;
        let (decode_truth_, named_) = (decode_truth.clone(), named.clone());
        refuses_with(
            "a half-precision truth",
            "the reference bundle must be a float32 or float64 evaluation, not bfloat16",
            move || validate_envelope(&half_truth, &sha, &decode_truth_, &named_, TensorDType::F16),
        );
        let mut stopped = truth.clone();
        stopped.greedy_stopped = true;
        let sha = fixture.truth_sha256.clone();
        refuses_with(
            "a stopped truth",
            "the reference bundle's greedy continuation stopped early",
            move || validate_envelope(&stopped, &sha, &decode_truth, &named, TensorDType::F16),
        );
    }

    /// A non-finite official graph makes the truth uncalibrated, whether the manifest flags it or only the file shows it.
    #[test]
    fn a_non_finite_official_graph_is_uncalibrated() {
        let fixture = Fixture::new("float16", AMOUNTS);
        let with = |slot: usize, path: PathBuf| {
            let mut changed = fixture.graphs.clone();
            changed[slot] = path;
            changed
        };
        let flagged = fixture.graph(OfficialGraph::ALL[2], 0.125, |m| {
            m["prompts"][1]["finite"] = json!(false);
            m["prompts"][1]["first_non_finite"] = json!({"position": 2, "token": 0});
        });
        let graphs = with(2, flagged);
        let truth = fixture.truth.clone();
        let decode_truth = fixture.decode_truth.clone();
        refuses_with(
            "a flagged non-finite prompt",
            "uncalibrated: official mps-eager float16 non-finite at prompt 1 position 2",
            {
                let (truth, decode_truth) = (truth.clone(), decode_truth.clone());
                move || {
                    Envelope::read(TensorDType::F16, &truth, &decode_truth, &graphs);
                }
            },
        );
        for (what, expected, edit) in [
            (
                "decode",
                "uncalibrated: official cpu-sdpa float16 non-finite at its decode rows",
                (|m: &mut Value| m["decode_consistency"]["finite"] = json!(false))
                    as fn(&mut Value),
            ),
            (
                "prompt decode",
                "uncalibrated: official cpu-sdpa float16 non-finite at its prompt-decode rows",
                |m| m["prompt_decode"]["finite"] = json!(false),
            ),
        ] {
            let graphs = with(1, fixture.graph(OfficialGraph::ALL[1], 0.25, edit));
            let (truth, decode_truth) = (truth.clone(), decode_truth.clone());
            refuses_with(what, expected, move || {
                Envelope::read(TensorDType::F16, &truth, &decode_truth, &graphs);
            });
        }
        // A NaN that the manifest does not flag is still found while measuring.
        let unflagged = fixture.graph(OfficialGraph::ALL[3], 0.0625, |_| {});
        let digest = write_file(
            &unflagged,
            "logits-1.f32le",
            &[0.0, 1.0, 0.0, f32::NAN, 0.0, 1.0, 0.0, 1.0],
        );
        edit_manifest(&unflagged, |m| {
            m["prompts"][1]["logits_sha256"] = json!(digest)
        });
        let graphs = with(3, unflagged);
        refuses_with(
            "an unflagged NaN",
            "uncalibrated: official mps-sdpa float16 non-finite at prompt 1 position 1",
            move || {
                Envelope::read(TensorDType::F16, &truth, &decode_truth, &graphs);
            },
        );
    }

    /// Official graphs equal to the truth, and corrupted or missing logits, are refused.
    #[test]
    fn degenerate_or_corrupted_official_graphs_are_refused() {
        let fixture = Fixture::new("bfloat16", [0.0; 4]);
        let (truth, decode_truth, graphs) = (
            fixture.truth.clone(),
            fixture.decode_truth.clone(),
            fixture.graphs.clone(),
        );
        refuses_with(
            "graphs equal to the truth",
            "prompt 0: the official envelope (max 0e0, mean 0e0) must be positive",
            move || {
                Envelope::read(TensorDType::BF16, &truth, &decode_truth, &graphs);
            },
        );
        let fixture = Fixture::new("bfloat16", AMOUNTS);
        std::fs::write(
            fixture.graphs[1].join("decode-cached.f32le"),
            bytes(&[0.0; 4]),
        )
        .unwrap();
        let (truth, decode_truth, graphs) = (
            fixture.truth.clone(),
            fixture.decode_truth.clone(),
            fixture.graphs.clone(),
        );
        refuses_with(
            "corrupted decode rows",
            "decode-cached.f32le is corrupted",
            move || {
                Envelope::read(TensorDType::BF16, &truth, &decode_truth, &graphs);
            },
        );
        std::fs::remove_file(fixture.graphs[2].join("logits-0.f32le")).unwrap();
        let (truth, decode_truth, graphs) = (
            fixture.truth.clone(),
            fixture.decode_truth.clone(),
            fixture.graphs.clone(),
        );
        refuses_with("a missing logits file", "logits-0.f32le: ", move || {
            Envelope::read(TensorDType::BF16, &truth, &decode_truth, &graphs);
        });
        let short = fixture.root.path().join("short");
        std::fs::create_dir(&short).unwrap();
        let digest = write_file(&short, "rows.f32le", &[1.0, 2.0, 3.0]);
        refuses_with(
            "a file of the wrong size",
            "must hold [2, 2] F32 logits",
            move || {
                read_rows(&short, "rows.f32le", &digest, 2, 2);
            },
        );
    }

    /// The gate applies the prompt rule, the row bound, and the provisional self-consistency tolerance.
    #[test]
    fn gate_applies_the_rule_to_prompts_rows_and_self_consistency() {
        let fixture = Fixture::new("bfloat16", AMOUNTS);
        refuses("an F32 run", || {
            HalfPrecisionGate::new(
                TensorDType::F32,
                &fixture.truth,
                &fixture.decode_truth,
                &fixture.graphs,
            );
        });
        let gate = HalfPrecisionGate::new(
            TensorDType::BF16,
            &fixture.truth,
            &fixture.decode_truth,
            &fixture.graphs,
        );
        assert_eq!(gate.envelope().dtype, TensorDType::BF16);
        // Prompt 0: E = (0.5, 0.5), so the bounds are 0.75 and 0.65.
        let reference = TRUTH[0];
        let observed = gate.check_prompt("prompt 0", 0, &[1.0, 2.0, 3.0, 4.75], reference);
        assert_eq!((observed.maximum, observed.index), (0.75, 3));
        let gate_ = gate.clone();
        refuses_with(
            "a maximum beyond the bound",
            "decoded: max |Δ| = 8.75e-1 exceeds the half-precision bound 7.5e-1 of prompt 0",
            move || {
                gate_.check_prompt("decoded", 0, &[1.0, 2.0, 3.0, 4.875], reference);
            },
        );
        let gate_ = gate.clone();
        refuses_with(
            "a mean beyond the bound",
            "decoded: mean |Δ| = 6.875e-1 exceeds the half-precision bound",
            move || {
                gate_.check_prompt("decoded", 0, &[1.75, 2.75, 3.625, 4.625], reference);
            },
        );
        let gate_ = gate.clone();
        refuses_with(
            "a partial prompt",
            "decoded: the reference logits must cover every position of prompt 0",
            move || {
                gate_.check_prompt("decoded", 0, &[1.0, 2.0], &reference[..2]);
            },
        );
        gate.check_rows("row", 0, &[4.5, 1.0], &[3.75, 0.5]);
        let gate_ = gate.clone();
        refuses("a row beyond the maximum bound", move || {
            gate_.check_rows("row", 0, &[4.75], &[3.75]);
        });
        gate.check_consistency("batched", &[1.5, 2.0], &[1.0, 2.0]);
        let gate_ = gate.clone();
        refuses_with(
            "an inconsistent batched row",
            "exceeds the provisional BF16 self-consistency tolerance 5e-1",
            move || {
                gate_.check_consistency("batched", &[1.5, 2.0], &[0.75, 2.0]);
            },
        );
        // Greedy steps are judged at the decode rows' maximum bound 1.5, the larger one.
        assert_eq!(gate.bound(0).maximum, 0.75);
        assert_eq!(
            gate.decode_bound(),
            Bound {
                maximum: 1.5,
                mean: 1.3
            }
        );
        assert!(gate.is_decisive(3.0_f32.next_up()) && !gate.is_decisive(3.0));
        assert_eq!(gate.greedy_tolerance(), 1.5);
    }

    /// Both decode passes are held to the decode envelope; the gap is printed only, and any pass beyond it fails.
    #[test]
    fn decode_rows_are_each_checked_against_the_decode_truth() {
        let fixture = Fixture::new("bfloat16", AMOUNTS);
        let gate = HalfPrecisionGate::new(
            TensorDType::BF16,
            &fixture.truth,
            &fixture.decode_truth,
            &fixture.graphs,
        );
        // E_dec = (1, 1): bounds 1.5 and 1.3 under the BF16 slacks.
        let bound = gate.decode_bound();
        assert_eq!(
            bound,
            Bound {
                maximum: 1.5,
                mean: 1.3
            }
        );
        assert_eq!(gate.envelope().decode_truth.rows, DECODE);
        let decoded = vec![vec![5.5, 6.5], vec![7.5, 8.5]];
        let stateless = [4.0, 6.0, 7.0, 8.0];
        let (cached, alone) = gate.check_decode("decode", 2, &decoded, &stateless);
        assert_eq!((cached.maximum, cached.mean), (0.5, 0.5));
        assert_eq!((alone.maximum, alone.index, alone.mean), (1.0, 0, 0.25));
        let far = 1.0 + bound.maximum as f32;
        for (what, expected, decoded, stateless) in [
            (
                "a cached row beyond the bound",
                "decode (cached rows): max |Δ| = ",
                vec![vec![5.0, 6.0], vec![7.0, 8.0 + far]],
                stateless.to_vec(),
            ),
            (
                "a stateless row beyond the mean bound",
                "decode (stateless rows): mean |Δ| = ",
                decoded.clone(),
                vec![6.4, 7.4, 8.4, 9.4],
            ),
            (
                "a non-finite cached row",
                "decode (cached rows): the compared logits hold a non-finite value",
                vec![vec![f32::NAN, 6.0], vec![7.0, 8.0]],
                stateless.to_vec(),
            ),
            (
                "a missing row",
                "decode: one decoded row per decode-truth row",
                vec![vec![5.0, 6.0]],
                stateless.to_vec(),
            ),
        ] {
            let gate = gate.clone();
            refuses_with(what, expected, move || {
                gate.check_decode("decode", 2, &decoded, &stateless);
            });
        }
        let gate_ = gate.clone();
        refuses_with(
            "another first position",
            "decode rows must start where the decode truth's do",
            move || {
                gate_.check_decode("decode", 3, &[vec![5.0, 6.0], vec![7.0, 8.0]], &DECODE);
            },
        );
        // A decode that is equally wrong in both passes has no gap, yet fails.
        let wrong = [5.0 + far, 6.0, 7.0, 8.0];
        let gate_ = gate.clone();
        refuses_with(
            "a consistent but wrong decode",
            "decode (cached rows): max |Δ|",
            move || {
                gate_.check_decode(
                    "decode",
                    2,
                    &[wrong[..2].to_vec(), wrong[2..].to_vec()],
                    &wrong,
                );
            },
        );
        let observed = gate.check_decode_rows("rows", DecodeRows::Stateless, &DECODE, &DECODE);
        assert_eq!(observed.maximum, 0.0);
        assert_eq!(
            (DecodeRows::Cached.label(), DecodeRows::Stateless.label()),
            ("cached", "stateless")
        );
    }

    /// The decoded prompt follows the official plan and is judged by the prompt-decode envelope.
    #[test]
    fn decoded_prompt_is_judged_by_the_prompt_decode_envelope() {
        let fixture = Fixture::new("bfloat16", AMOUNTS);
        let gate = HalfPrecisionGate::new(
            TensorDType::BF16,
            &fixture.truth,
            &fixture.decode_truth,
            &fixture.graphs,
        );
        assert_eq!(gate.prompt_decode_plan(), (1, 1));
        // E_pd = (1.5, 1.5): bounds 2.25 and 1.95, wider than prompt 1's 0.75 and 0.65.
        let decoded = shifted(TRUTH[1], 1.875);
        let observed = gate.check_prompt_decode("prompt 1 decoded", &decoded, TRUTH[1]);
        assert_eq!((observed.maximum, observed.mean), (1.875, 1.875));
        let gate_ = gate.clone();
        let beyond = shifted(TRUTH[1], 2.375);
        refuses_with(
            "a decoded prompt beyond the bound",
            "exceeds the half-precision bound 2.25e0 of prompt 1 decoded after a 1-token prefill",
            move || {
                gate_.check_prompt_decode("prompt 1 decoded", &beyond, TRUTH[1]);
            },
        );
        let (gate_, stateless) = (gate.clone(), decoded.clone());
        refuses_with(
            "a stateless prompt with these rows",
            "exceeds the half-precision bound 7.5e-1 of prompt 1",
            move || {
                gate_.check_prompt("prompt 1", 1, &stateless, TRUTH[1]);
            },
        );
        // Without prompt-decode rows (Gemma's long bundle) there is no plan.
        let fixture = Fixture::new("bfloat16", AMOUNTS);
        let plain: Vec<PathBuf> = OfficialGraph::ALL
            .into_iter()
            .zip(AMOUNTS)
            .map(|(graph, amount)| {
                fixture.graph(graph, amount, |m| {
                    m.as_object_mut().unwrap().remove("prompt_decode");
                })
            })
            .collect();
        let gate = HalfPrecisionGate::new(
            TensorDType::BF16,
            &fixture.truth,
            &fixture.decode_truth,
            &plain,
        );
        assert!(gate.envelope().prompt_decode.is_none());
        let gate_ = gate.clone();
        refuses_with(
            "no prompt-decode rows",
            "recorded no prompt_decode rows",
            move || {
                gate_.prompt_decode_plan();
            },
        );
        refuses_with(
            "no prompt-decode rows",
            "recorded no prompt_decode rows",
            move || {
                gate.check_prompt_decode("p", &decoded, TRUTH[1]);
            },
        );
    }

    /// F32 runs compare cached decode rows with the decode truth at their own tolerance.
    #[test]
    fn f32_decode_rows_are_held_to_the_f32_tolerance() {
        let fixture = Fixture::new("bfloat16", AMOUNTS);
        let decode_truth = DecodeTruth::read(&fixture.truth, &fixture.decode_truth);
        assert_eq!(
            (
                decode_truth.first,
                decode_truth.steps(),
                decode_truth.vocabulary
            ),
            (2, 2, 2)
        );
        assert_eq!(decode_truth.rows, DECODE);
        assert_eq!(decode_truth.sha256.len(), 64);
        decode_truth.check_f32("decode", &[vec![5.0, 6.0005], vec![7.0, 8.0]], 1e-3);
        for (what, expected, rows) in [
            (
                "a row beyond",
                "decode step 1 at position 3: max |Δ| = ",
                vec![vec![5.0, 6.0], vec![7.0, 8.01]],
            ),
            (
                "a NaN row",
                "decode step 0 at position 2: the compared logits hold a non-finite value",
                vec![vec![f32::NAN, 6.0], vec![7.0, 8.0]],
            ),
            (
                "too few rows",
                "decode: one decoded row per decode-truth row",
                vec![vec![5.0, 6.0]],
            ),
        ] {
            let decode_truth = decode_truth.clone();
            refuses_with(what, expected, move || {
                decode_truth.check_f32("decode", &rows, 1e-3)
            });
        }
        let (truth, decode_truth) = (fixture.truth.clone(), fixture.graphs[0].clone());
        refuses_with(
            "an official graph as decode truth",
            "must be a float32 or float64 evaluation, not bfloat16",
            move || {
                DecodeTruth::read(&truth, &decode_truth);
            },
        );
    }

    /// Returns four positive official statistics from proptest-generated maxima and means.
    fn statistics(maxima: [f64; 4], means: [f64; 4]) -> [Statistic; 4] {
        std::array::from_fn(|graph| Statistic {
            maximum: maxima[graph],
            mean: means[graph],
        })
    }

    proptest! {
        /// Scaling every official statistic by a power of two scales both bounds exactly and leaves the ratios unchanged.
        #[test]
        fn bounds_scale_linearly_with_the_official_deviations(
            maxima in proptest::array::uniform4(1e-3_f64..1e3),
            means in proptest::array::uniform4(1e-4_f64..1e2),
            exponent in -20_i32..=20,
            bf16 in proptest::bool::ANY,
        ) {
            let dtype = if bf16 { TensorDType::BF16 } else { TensorDType::F16 };
            let scale = 2.0_f64.powi(exponent);
            let graphs = statistics(maxima, means);
            let base = GraphEnvelope::new("p", graphs, dtype);
            let scaled = GraphEnvelope::new(
                "p",
                graphs.map(|graph| Statistic { maximum: graph.maximum * scale, mean: graph.mean * scale }),
                dtype,
            );
            prop_assert_eq!(scaled.bound.maximum, base.bound.maximum * scale);
            prop_assert_eq!(scaled.bound.mean, base.bound.mean * scale);
            prop_assert_eq!(scaled.leave_one_out(), base.leave_one_out());
        }

        /// The envelope of four graphs is at least each graph's and each leave-one-out envelope's statistic.
        #[test]
        fn adding_a_graph_never_shrinks_the_envelope(
            maxima in proptest::array::uniform4(1e-3_f64..1e3),
            means in proptest::array::uniform4(1e-4_f64..1e2),
        ) {
            let envelope = GraphEnvelope::new("p", statistics(maxima, means), TensorDType::BF16);
            for (graph, ratio) in envelope.graphs.iter().zip(envelope.leave_one_out()) {
                prop_assert!(envelope.envelope.maximum >= graph.maximum && envelope.envelope.mean >= graph.mean);
                // The others' envelope is the graph's statistic over its ratio; four graphs never have less.
                prop_assert!(envelope.envelope.maximum >= graph.maximum / ratio.maximum * (1.0 - 1e-15));
                prop_assert!(envelope.envelope.mean >= graph.mean / ratio.mean * (1.0 - 1e-15));
            }
        }

        /// A run that deviates exactly as any official graph does passes, since every slack is at least one.
        #[test]
        fn a_run_equal_to_any_official_graph_passes(
            maxima in proptest::array::uniform4(1e-3_f64..1e3),
            means in proptest::array::uniform4(1e-4_f64..1e2),
            bf16 in proptest::bool::ANY,
        ) {
            let dtype = if bf16 { TensorDType::BF16 } else { TensorDType::F16 };
            prop_assert!(max_slack(dtype) >= 1.0 && mean_slack(dtype) >= 1.0);
            let envelope = GraphEnvelope::new("p", statistics(maxima, means), dtype);
            for graph in envelope.graphs {
                let run = Deviation { maximum: graph.maximum, index: 0, mean: graph.mean };
                prop_assert!(envelope.bound.admits(&run));
            }
        }
    }
}
