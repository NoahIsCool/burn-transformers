//! Official half-precision calibration of the Llama-family and Gemma real-checkpoint gates.
//!
//! A half-precision (F16 or BF16) Burn run is judged against the same F32 (or
//! F64) Transformers bundle as an F32 run, called the *truth* here. Its bounds
//! are not chosen by hand: they are the official implementation's own
//! half-precision distance from that truth, read at run time from an official
//! BF16 bundle that `scripts/generate_causal_lm_reference.py --dtype bfloat16`
//! wrote from the same checkpoint, prompts, and options. The F32 gates do not
//! use this module.
//!
//! # Rule
//!
//! Write `r[t, v]` for the truth logits of one prompt at position `t` and
//! vocabulary entry `v` (`T` positions, `V` entries), `o[t, v]` for the
//! official BF16 logits of the same tokens, and `b[t, v]` for a Burn run's.
//! The official deviation of the prompt is summarized by
//!
//! ```text
//! M = max_{t,v} |o[t,v] - r[t,v]|
//! μ = (1 / (T V)) Σ_{t,v} |o[t,v] - r[t,v]|
//! ```
//!
//! with each difference of two F32 values taken exactly in F64 and the sum
//! accumulated in F64 ([`Deviation::measure`]). One rule applies to every
//! checkpoint and every prompt: a half-precision run passes the prompt when
//!
//! ```text
//! max_{t,v} |b[t,v] - r[t,v]| <= s M   and   mean_{t,v} |b[t,v] - r[t,v]| <= s μ,   s = SLACK = 1.2
//! ```
//!
//! both inclusive, so a deviation exactly at a bound passes ([`Bound`]). A
//! check of single rows of the prompt (the last position of a cached prefill
//! or of a batched row) applies the maximum bound `s M` alone, because the mean
//! is defined over all of the prompt's positions. A bundle whose official BF16
//! logits equal the truth on a prompt (`M = 0`) is refused: it cannot be a
//! half-precision evaluation, and a zero bound would reject every run.
//!
//! Greedy steps of the cached continuation of prompt 0 are compared only when
//! the truth's top-2 margin exceeds `2 s M` of prompt 0. If every logit lies
//! within `s M` of the truth, the gap between the truth's best and second
//! tokens shrinks by at most `2 s M`, so only such a step has a determined
//! argmax; a margin exactly `2 s M` is a tie and is not compared. The
//! continuation is teacher-forced (each step feeds the truth's token), so a
//! tie is skipped rather than ending the comparison ([`decisive_steps`]), and
//! the set of compared steps may be empty. This assumes that decoding steps
//! deviate no more than the prompt does, as the F32 and gpt-oss gates assume
//! for their own tolerances.
//!
//! # Slack
//!
//! The factor `s` covers the difference between two half-precision
//! implementations of one model: Burn normalizes softmaxes in F32 and orders
//! its reductions differently, so its rounding is not Transformers'. It is
//! fixed from the only measurement of Burn half-precision runs next to the
//! official BF16 evaluation of the same truth: gpt-oss-20b, all 24 blocks, on
//! the four prompts of `gpt-oss-20b-6cee5e81-f32-reference.bundle`, recorded
//! in `acceptance/measurements/reference-precision-followups-20261016.json`
//! (the Burn runs at `c15f24e`, the official figures as its run
//! `gpt-oss-20b-transformers-bf16-vs-f32-bundles`). Each Burn statistic over
//! the official statistic of the same prompt:
//!
//! | run | maximum, prompts 0-3 | mean, prompts 0-3 |
//! | --- | --- | --- |
//! | Flex/BF16 | `0.822`, `1.013`, `0.814`, `1.009` | `1.034`, `0.997`, `1.173`, `0.844` |
//! | Metal/F16 | `0.007`, `0.146`, `0.759`, `0.694` | `0.026`, `0.158`, `0.874`, `0.353` |
//!
//! With `s = 1` the rule would reject four of the eight Flex/BF16 statistics.
//! The largest ratio, `1.1725` (Flex/BF16 prompt 2: mean `0.033998` against
//! the official `0.028996`), rounded up to two significant digits, as the
//! gpt-oss gate rounded `2.665` to `2.7`, gives `s = 1.2`. The factor was
//! fixed before any Llama-family or Gemma half-precision run and is not to be
//! changed to make one pass; it is a measured property of Burn's
//! half-precision kernels on one other family, so a failure under it is a
//! finding to review.
//!
//! # Self-consistency
//!
//! Independently of the truth, a cached prefill and every row of a
//! left-padded batch must agree with the same run's stateless pass within
//! [`SELF_CONSISTENCY_TOLERANCE`] for the run's dtype.
//!
//! # Pairing
//!
//! [`validate_pairing`] accepts the official bundle only as the BF16 twin of
//! the truth and fails closed otherwise. Both manifests must be schema-1
//! bundles ([`Manifest::parse`]); the truth must be a `float32` or `float64`
//! evaluation and the official bundle a plain `bfloat16` one (not
//! `--bf16-resident`); and both must agree on the model type and class, the
//! `config.json` and weight-file SHA-256s, repository and revision,
//! Transformers and PyTorch versions, vocabulary, layer truncation, every
//! prompt's text and token IDs, and the greedy length. Every logits file must
//! match its recorded SHA-256 and shape, and every logit must be finite.

use std::{
    collections::BTreeMap,
    ffi::OsStr,
    path::{Path, PathBuf},
};

use serde_json::Value;
use sha2::{Digest, Sha256};
use transformers::checkpoint::TensorDType;

/// Factor `s` applied to both official statistics; see [Slack](self#slack).
pub(crate) const SLACK: f64 = 1.2;

/// Absolute agreement of cached and batched passes with the same run's stateless pass, per dtype.
///
/// The values are the gpt-oss gate's (`tests/gpt_oss_artifacts.rs`), where a
/// left-padded batch and each prompt alone differed by at most `0.125` (BF16)
/// and `0.41` (F16) on four real blocks on Flex, and by `0.3125` on Flex/BF16
/// and `0` on Metal/F16 on the full model
/// (`acceptance/measurements/reference-precision-followups-20261016.json`).
/// They are not calibrated on the Llama-family or Gemma checkpoints.
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

/// Whether a run in `dtype` is judged by the official half-precision calibration.
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
/// its bounds from the official bundle, which no override may replace.
pub(crate) fn reject_overrides(names: &[&str]) {
    let present = present_overrides(names, |name| std::env::var_os(name).is_some());
    assert!(
        present.is_empty(),
        "{present:?} override F32 tolerances; a half-precision run takes its bounds from the official BF16 bundle, so unset them"
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

/// Lower-case hexadecimal digest.
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// Evaluation dtype recorded in a bundle's provenance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BundleDtype {
    /// `float32`: a truth bundle.
    Float32,
    /// `float64`: a truth bundle without the reference's own F32 rounding.
    Float64,
    /// `bfloat16`: an official half-precision evaluation.
    BFloat16,
}

impl BundleDtype {
    /// Parses a provenance `dtype` label.
    ///
    /// # Panics
    ///
    /// For any label but `float32`, `float64`, and `bfloat16`.
    pub(crate) fn parse(label: &str) -> Self {
        match label {
            "float32" => Self::Float32,
            "float64" => Self::Float64,
            "bfloat16" => Self::BFloat16,
            other => panic!(
                "unsupported reference dtype label {other:?}; expected float32, float64, or bfloat16"
            ),
        }
    }

    /// Whether a bundle in this dtype can be the truth that runs are judged against.
    pub(crate) fn is_truth(self) -> bool {
        matches!(self, Self::Float32 | Self::Float64)
    }
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
    /// Prompts in bundle order.
    pub(crate) prompts: Vec<BundlePrompt>,
    /// Greedy continuation of prompt 0 as `(token, top-2 margin)`.
    pub(crate) greedy: Vec<(i64, f32)>,
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

impl Manifest {
    /// Parses a manifest and checks its internal consistency.
    ///
    /// # Panics
    ///
    /// If `text` is not a schema-1 manifest, a field is missing or mistyped, a
    /// prompt's `logits_shape` is not `[tokens, vocabulary]`, a logits file
    /// name is not a plain file name, or a greedy margin is negative or not
    /// finite in F32.
    pub(crate) fn parse(text: &str) -> Self {
        let manifest: Value = serde_json::from_str(text)
            .unwrap_or_else(|error| panic!("reference.json is not JSON: {error}"));
        let schema_version = unsigned(&manifest, "schema_version", "schema_version");
        assert_eq!(
            schema_version, SCHEMA_VERSION,
            "reference.json: unsupported bundle schema {schema_version}"
        );
        let vocabulary = usize::try_from(unsigned(&manifest, "vocabulary_size", "vocabulary_size"))
            .expect("vocabulary size fits usize");
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
            prompts,
            greedy,
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
        let shape: Option<Vec<u64>> = member(prompt, "logits_shape", &path("logits_shape"))
            .as_array()
            .and_then(|shape| shape.iter().map(Value::as_u64).collect());
        assert_eq!(
            shape,
            Some(vec![token_ids.len() as u64, vocabulary as u64]),
            "reference.json: {} must be [tokens, vocabulary]",
            path("logits_shape")
        );
        let logits_file = string(prompt, "logits_file", &path("logits_file")).to_owned();
        assert!(
            !logits_file.is_empty()
                && logits_file != "."
                && logits_file != ".."
                && !logits_file.contains(['/', '\\']),
            "reference.json: {} must be a plain file name, not {logits_file:?}",
            path("logits_file")
        );
        Self {
            text: string(prompt, "text", &path("text")).to_owned(),
            token_ids,
            logits_file,
            logits_sha256: string(prompt, "logits_sha256", &path("logits_sha256")).to_owned(),
        }
    }
}

/// Requires `official` to be the official BF16 twin of the `truth` bundle.
///
/// # Panics
///
/// On the first disagreement listed in [Pairing](self#pairing).
pub(crate) fn validate_pairing(truth: &Manifest, official: &Manifest) {
    assert!(
        truth.dtype.is_truth(),
        "the reference bundle must be a float32 or float64 evaluation, not {:?}",
        truth.dtype
    );
    assert_eq!(
        official.dtype,
        BundleDtype::BFloat16,
        "the official half-precision bundle must be a bfloat16 evaluation"
    );
    assert!(
        !official.bf16_resident,
        "the official half-precision bundle must be a plain bfloat16 evaluation, not --bf16-resident"
    );
    let same = |what: &str, equal: bool| {
        assert!(
            equal,
            "the official BF16 bundle and the reference bundle differ in {what}"
        );
    };
    same("model type", truth.model_type == official.model_type);
    same("model class", truth.model_class == official.model_class);
    same(
        "config.json SHA-256",
        truth.config_sha256 == official.config_sha256,
    );
    same(
        "weight files",
        truth.weights_sha256 == official.weights_sha256,
    );
    same("repository", truth.repository == official.repository);
    same("revision", truth.revision == official.revision);
    same(
        "Transformers version",
        truth.transformers == official.transformers,
    );
    same("PyTorch version", truth.torch == official.torch);
    same("vocabulary size", truth.vocabulary == official.vocabulary);
    same("layer truncation", truth.layers == official.layers);
    same(
        "prompt count",
        truth.prompts.len() == official.prompts.len(),
    );
    for (index, (expected, actual)) in truth.prompts.iter().zip(&official.prompts).enumerate() {
        same(
            &format!("prompt {index} text"),
            expected.text == actual.text,
        );
        same(
            &format!("prompt {index} token IDs"),
            expected.token_ids == actual.token_ids,
        );
    }
    same("greedy length", truth.greedy.len() == official.greedy.len());
}

/// Reads one prompt's row-major `[tokens, vocabulary]` logits after checking their digest and size.
///
/// # Panics
///
/// If the file cannot be read, its SHA-256 differs from the manifest's, or it
/// does not hold exactly `tokens × vocabulary` little-endian F32 values.
pub(crate) fn read_logits(root: &Path, prompt: &BundlePrompt, vocabulary: usize) -> Vec<f32> {
    let file = root.join(&prompt.logits_file);
    let bytes =
        std::fs::read(&file).unwrap_or_else(|error| panic!("read {}: {error}", file.display()));
    assert_eq!(
        hex(&Sha256::digest(&bytes)),
        prompt.logits_sha256,
        "{} is corrupted",
        file.display()
    );
    let values = prompt.token_ids.len() * vocabulary;
    assert_eq!(
        bytes.len(),
        4 * values,
        "{} must hold [{}, {vocabulary}] F32 logits",
        file.display(),
        prompt.token_ids.len()
    );
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect()
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
    /// Measures `actual` against `truth`, element by element.
    ///
    /// Each difference of two F32 values is exact in F64, and the mean is
    /// summed in F64 over `actual.len()` entries. On ties the first index
    /// attaining the maximum is reported.
    ///
    /// # Panics
    ///
    /// If the slices differ in length, are empty, or hold a non-finite value.
    pub(crate) fn measure(actual: &[f32], truth: &[f32]) -> Self {
        assert_eq!(
            actual.len(),
            truth.len(),
            "compared logits differ in length"
        );
        assert!(!actual.is_empty(), "no logits to compare");
        assert!(
            actual.iter().all(|value| value.is_finite()),
            "the compared logits hold a non-finite value"
        );
        assert!(
            truth.iter().all(|value| value.is_finite()),
            "the reference logits hold a non-finite value"
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
        deviation
    }
}

/// The half-precision bounds of one prompt.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Bound {
    /// Bound `s M` on the largest absolute logit deviation.
    pub(crate) maximum: f64,
    /// Bound `s μ` on the mean absolute logit deviation over every position.
    pub(crate) mean: f64,
}

impl Bound {
    /// Applies the rule to the official BF16 deviation of one prompt.
    ///
    /// # Panics
    ///
    /// If either official statistic is zero or not finite: an official bundle
    /// that reproduces its truth exactly is not a half-precision evaluation.
    pub(crate) fn from_official(official: &Deviation) -> Self {
        assert!(
            official.maximum.is_finite()
                && official.maximum > 0.0
                && official.mean.is_finite()
                && official.mean > 0.0,
            "the official BF16 deviation (max {:e}, mean {:e}) must be positive and finite; \
             an official bundle equal to its reference is not a half-precision evaluation",
            official.maximum,
            official.mean
        );
        Self {
            maximum: SLACK * official.maximum,
            mean: SLACK * official.mean,
        }
    }

    /// Whether a deviation over every position of the prompt meets both bounds (inclusive).
    pub(crate) fn admits(&self, observed: &Deviation) -> bool {
        self.admits_rows(observed) && observed.mean <= self.mean
    }

    /// Whether a deviation over some rows of the prompt meets the maximum bound (inclusive).
    pub(crate) fn admits_rows(&self, observed: &Deviation) -> bool {
        observed.maximum <= self.maximum
    }

    /// Truth margin a greedy step must exceed to be compared: `2 s M`.
    pub(crate) fn greedy_threshold(&self) -> f64 {
        2.0 * self.maximum
    }

    /// Whether a truth top-2 `margin` determines the argmax under this bound (strictly above the threshold).
    pub(crate) fn is_decisive(&self, margin: f32) -> bool {
        f64::from(margin) > self.greedy_threshold()
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

/// Returns the self-consistency tolerance of a half-precision `dtype`.
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

/// Official deviation and bounds of one prompt.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct PromptCalibration {
    /// Number of prompt positions `T`.
    pub(crate) tokens: usize,
    /// Official BF16 deviation from the truth over every position.
    pub(crate) official: Deviation,
    /// Bounds the rule derives from it.
    pub(crate) bound: Bound,
}

/// A validated truth bundle with its official BF16 twin, and the bounds they set.
#[derive(Debug, Clone)]
pub(crate) struct Calibration {
    /// SHA-256 of the truth bundle's `reference.json`.
    pub(crate) truth_sha256: String,
    /// SHA-256 of the official bundle's `reference.json`.
    pub(crate) official_sha256: String,
    /// Truth manifest.
    pub(crate) truth: Manifest,
    /// Official BF16 manifest.
    pub(crate) official: Manifest,
    /// Calibration of each prompt, in bundle order.
    pub(crate) prompts: Vec<PromptCalibration>,
}

impl Calibration {
    /// Reads and pairs the truth bundle in `truth_dir` with the official BF16 bundle in `official_dir`, then measures every prompt.
    ///
    /// The logits of one prompt pair are dropped once measured, so at most
    /// two prompts' logits are held at a time.
    ///
    /// # Panics
    ///
    /// If either bundle is malformed or corrupted, the pairing fails
    /// ([`validate_pairing`]), a logit is not finite, or an official deviation
    /// is zero ([`Bound::from_official`]).
    pub(crate) fn read(truth_dir: &Path, official_dir: &Path) -> Self {
        let (truth, truth_sha256) = Manifest::read(truth_dir);
        let (official, official_sha256) = Manifest::read(official_dir);
        validate_pairing(&truth, &official);
        let prompts = truth
            .prompts
            .iter()
            .zip(&official.prompts)
            .map(|(expected, rounded)| {
                let exact = read_logits(truth_dir, expected, truth.vocabulary);
                let half = read_logits(official_dir, rounded, official.vocabulary);
                let official = Deviation::measure(&half, &exact);
                PromptCalibration {
                    tokens: expected.token_ids.len(),
                    official,
                    bound: Bound::from_official(&official),
                }
            })
            .collect();
        Self {
            truth_sha256,
            official_sha256,
            truth,
            official,
            prompts,
        }
    }

    /// Returns the greedy steps of prompt 0 that a half-precision run compares.
    pub(crate) fn decisive_greedy_steps(&self) -> Vec<usize> {
        decisive_steps(&self.truth.greedy, &self.prompts[0].bound)
    }

    /// Prints both bundles' identities, each prompt's official deviation and bounds, and the decisive greedy steps.
    pub(crate) fn report(&self, truth_dir: &Path, official_dir: &Path) {
        eprintln!(
            "{} ({:?}): reference.json SHA-256 {}",
            truth_dir.display(),
            self.truth.dtype,
            self.truth_sha256
        );
        eprintln!(
            "{} ({:?}): reference.json SHA-256 {}",
            official_dir.display(),
            self.official.dtype,
            self.official_sha256
        );
        let vocabulary = self.truth.vocabulary;
        for (index, prompt) in self.prompts.iter().enumerate() {
            eprintln!(
                "prompt {index}: {} tokens, official BF16 max |Δ| = {:e} at position {}, token {}, mean |Δ| = {:e}; \
                 bounds max {:e}, mean {:e}",
                prompt.tokens,
                prompt.official.maximum,
                prompt.official.index / vocabulary,
                prompt.official.index % vocabulary,
                prompt.official.mean,
                prompt.bound.maximum,
                prompt.bound.mean
            );
        }
        let decisive = self.decisive_greedy_steps();
        eprintln!(
            "greedy: threshold {:e}; {} of {} steps decisive {decisive:?}",
            self.prompts[0].bound.greedy_threshold(),
            decisive.len(),
            self.truth.greedy.len()
        );
        if let (Some(exact), Some(rounded)) =
            (self.truth.greedy.first(), self.official.greedy.first())
        {
            eprintln!(
                "greedy step 0: token {} margin {:e} (reference), token {} margin {:e} (official BF16)",
                exact.0, exact.1, rounded.0, rounded.1
            );
        }
    }
}

/// Judges one half-precision run with the official calibration of the bundle it reads.
#[derive(Debug, Clone)]
pub(crate) struct HalfPrecisionGate {
    /// Stored dtype of the run, F16 or BF16.
    dtype: TensorDType,
    /// Calibration of the bundle the run is judged against.
    calibration: Calibration,
}

impl HalfPrecisionGate {
    /// Calibrates a `dtype` run against the truth in `truth_dir` from the official BF16 bundle in `official_dir`.
    ///
    /// Prints each prompt's bounds.
    ///
    /// # Panics
    ///
    /// If `dtype` is not F16 or BF16, or [`Calibration::read`] refuses the bundles.
    pub(crate) fn new(dtype: TensorDType, truth_dir: &Path, official_dir: &Path) -> Self {
        assert!(
            is_half_precision(dtype),
            "the official half-precision calibration judges F16 and BF16 runs, not {dtype:?}"
        );
        let calibration = Calibration::read(truth_dir, official_dir);
        for (index, prompt) in calibration.prompts.iter().enumerate() {
            eprintln!(
                "{dtype:?} bounds of prompt {index}: max {:e}, mean {:e} ({SLACK} × official BF16 {:e}, {:e}; {})",
                prompt.bound.maximum,
                prompt.bound.mean,
                prompt.official.maximum,
                prompt.official.mean,
                official_dir.display()
            );
        }
        Self { dtype, calibration }
    }

    /// Returns the bounds of prompt `prompt`.
    pub(crate) fn bound(&self, prompt: usize) -> Bound {
        self.calibration.prompts[prompt].bound
    }

    /// Checks every position of prompt `prompt` against both bounds; prints and returns the deviation.
    ///
    /// # Panics
    ///
    /// If `actual` and `truth` do not hold the prompt's `[tokens, vocabulary]`
    /// logits, hold a non-finite value, or exceed a bound.
    pub(crate) fn check_prompt(&self, prompt: usize, actual: &[f32], truth: &[f32]) -> Deviation {
        let calibration = &self.calibration.prompts[prompt];
        let vocabulary = self.calibration.truth.vocabulary;
        assert_eq!(
            truth.len(),
            calibration.tokens * vocabulary,
            "prompt {prompt}: the reference logits must cover every position"
        );
        let observed = Deviation::measure(actual, truth);
        let bound = calibration.bound;
        eprintln!(
            "prompt {prompt}: {} tokens, max |Δ| = {:e} at position {} (bound {:e}), mean |Δ| = {:e} (bound {:e})",
            calibration.tokens,
            observed.maximum,
            observed.index / vocabulary,
            bound.maximum,
            observed.mean,
            bound.mean
        );
        assert!(
            bound.admits_rows(&observed),
            "prompt {prompt}: max |Δ| = {:e} exceeds the half-precision bound {:e}",
            observed.maximum,
            bound.maximum
        );
        assert!(
            bound.admits(&observed),
            "prompt {prompt}: mean |Δ| = {:e} exceeds the half-precision bound {:e}",
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
        let observed = Deviation::measure(actual, truth);
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

    /// Checks a cached or batched result against the same run's stateless pass; prints and returns the deviation.
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
        let observed = Deviation::measure(actual, stateless);
        let tolerance = self_consistency_tolerance(self.dtype);
        eprintln!(
            "{label}: max |Δ| = {:e} to the stateless pass (tolerance {tolerance:e})",
            observed.maximum
        );
        assert!(
            observed.maximum <= tolerance,
            "{label}: max |Δ| = {:e} to the stateless pass exceeds the {:?} self-consistency tolerance {tolerance:e}",
            observed.maximum,
            self.dtype
        );
        observed
    }

    /// Whether a greedy step of prompt 0 with truth `margin` is compared.
    pub(crate) fn is_decisive(&self, margin: f32) -> bool {
        self.bound(0).is_decisive(margin)
    }
}

#[cfg(test)]
mod tests {
    //! Synthetic tests of dtype parsing, bundle pairing, the rule, and the gate.

    use std::panic::catch_unwind;

    use serde_json::json;

    use super::*;

    /// An environment variable that no test or harness sets.
    const UNSET: &str = "TRANSFORMERS_HALF_PRECISION_TEST_UNSET_VARIABLE";

    /// A described edit of a synthetic manifest.
    type Mutation = (&'static str, fn(&mut Value));

    /// Returns a valid synthetic manifest value with `prompts` (text, token IDs) and vocabulary 2.
    fn manifest_value(dtype: &str, prompts: &[(&str, &[i64])]) -> Value {
        json!({
            "schema_version": 1,
            "model_type": "llama",
            "model_class": "LlamaForCausalLM",
            "vocabulary_size": 2,
            "provenance": {
                "dtype": dtype,
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

    /// Writes a bundle with the given per-prompt logits into a new directory under `root`.
    fn write_bundle(root: &Path, name: &str, dtype: &str, logits: &[&[f32]]) -> PathBuf {
        let directory = root.join(name);
        std::fs::create_dir(&directory).unwrap();
        let prompts: Vec<(String, Vec<i64>)> = logits
            .iter()
            .enumerate()
            .map(|(index, values)| (format!("prompt {index}"), vec![7; values.len() / 2]))
            .collect();
        let borrowed: Vec<(&str, &[i64])> = prompts
            .iter()
            .map(|(text, tokens)| (text.as_str(), tokens.as_slice()))
            .collect();
        let mut manifest = manifest_value(dtype, &borrowed);
        for (index, values) in logits.iter().enumerate() {
            let bytes: Vec<u8> = values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect();
            std::fs::write(directory.join(format!("logits-{index}.f32le")), &bytes).unwrap();
            manifest["prompts"][index]["logits_sha256"] = json!(hex(&Sha256::digest(&bytes)));
        }
        std::fs::write(directory.join("reference.json"), manifest.to_string()).unwrap();
        directory
    }

    /// Asserts that `operation` panics.
    fn refuses(description: &str, operation: impl FnOnce() + std::panic::UnwindSafe) {
        assert!(
            catch_unwind(operation).is_err(),
            "{description} must be refused"
        );
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

    /// Set F32 overrides are listed in order; none set passes.
    #[test]
    fn f32_overrides_are_found_and_refused() {
        let names = ["A_TOLERANCE", "B_TOLERANCE", "C_TOLERANCE"];
        assert_eq!(
            present_overrides(&names, |name| name != "B_TOLERANCE"),
            ["A_TOLERANCE", "C_TOLERANCE"]
        );
        assert!(present_overrides(&names, |_| false).is_empty());
        reject_overrides(&[UNSET]);
        refuses("a set override", || reject_overrides(&["PATH"]));
    }

    /// The environment readers default an unset dtype to F32 and require directories to be named.
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
    }

    /// Truth bundles are float32 or float64, official ones bfloat16; other labels are refused.
    #[test]
    fn every_bundle_dtype_label_has_one_role() {
        assert_eq!(BundleDtype::parse("float32"), BundleDtype::Float32);
        assert_eq!(BundleDtype::parse("float64"), BundleDtype::Float64);
        assert_eq!(BundleDtype::parse("bfloat16"), BundleDtype::BFloat16);
        assert!(BundleDtype::Float32.is_truth() && BundleDtype::Float64.is_truth());
        assert!(!BundleDtype::BFloat16.is_truth());
        for label in ["float16", "Float32", "bf16", "f32", "", "int8"] {
            refuses(label, || {
                BundleDtype::parse(label);
            });
        }
        let prompts: &[(&str, &[i64])] = &[("a", &[1])];
        for truth in ["float32", "float64"] {
            validate_pairing(
                &parse(&manifest_value(truth, prompts)),
                &parse(&manifest_value("bfloat16", prompts)),
            );
        }
        for (truth, official) in [
            ("bfloat16", "bfloat16"),
            ("float32", "float32"),
            ("float64", "float32"),
            ("float32", "float64"),
        ] {
            let truth = parse(&manifest_value(truth, prompts));
            let official = parse(&manifest_value(official, prompts));
            refuses("a misassigned dtype", || {
                validate_pairing(&truth, &official)
            });
        }
    }

    /// Any difference between the bundles' identities, prompts, or options fails closed.
    #[test]
    fn pairing_refuses_every_mismatch() {
        let prompts: &[(&str, &[i64])] = &[("a", &[1, 2]), ("b", &[3])];
        let truth = parse(&manifest_value("float32", prompts));
        let official = manifest_value("bfloat16", prompts);
        validate_pairing(&truth, &parse(&official));
        let mutations: [Mutation; 16] = [
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
            ("bf16-resident", |m| {
                m["provenance"]["bf16_resident"] = json!(true)
            }),
            ("truncation", |m| {
                m["provenance"]["num_hidden_layers"] = json!(4)
            }),
            ("prompt text", |m| m["prompts"][1]["text"] = json!("c")),
            ("token IDs", |m| m["prompts"][1]["token_ids"] = json!([4])),
            ("prompt count", |m| {
                m["prompts"].as_array_mut().unwrap().pop();
            }),
            ("greedy length", |m| {
                m["greedy"]["steps"].as_array_mut().unwrap().pop();
            }),
            ("vocabulary", |m| {
                m["vocabulary_size"] = json!(3);
                m["prompts"][0]["logits_shape"] = json!([2, 3]);
                m["prompts"][1]["logits_shape"] = json!([1, 3]);
            }),
        ];
        for (what, mutate) in mutations {
            let mut changed = official.clone();
            mutate(&mut changed);
            let changed = parse(&changed);
            let truth = truth.clone();
            refuses(what, move || validate_pairing(&truth, &changed));
        }
        // A missing optional field is the same as null.
        let mut without = official.clone();
        let provenance = without["provenance"].as_object_mut().unwrap();
        provenance.remove("num_hidden_layers");
        provenance.remove("bf16_resident");
        validate_pairing(&truth, &parse(&without));
    }

    /// Malformed manifests are refused before any logit is read.
    #[test]
    fn malformed_manifests_are_refused() {
        let prompts: &[(&str, &[i64])] = &[("a", &[1, 2])];
        let valid = manifest_value("float32", prompts);
        parse(&valid);
        let mutations: [Mutation; 15] = [
            ("old schema", |m| m["schema_version"] = json!(0)),
            ("new schema", |m| m["schema_version"] = json!(2)),
            ("vocabulary", |m| m["vocabulary_size"] = json!(1)),
            ("missing dtype", |m| {
                m["provenance"].as_object_mut().unwrap().remove("dtype");
            }),
            ("shape", |m| m["prompts"][0]["logits_shape"] = json!([1, 2])),
            ("empty tokens", |m| {
                m["prompts"][0]["token_ids"] = json!([]);
                m["prompts"][0]["logits_shape"] = json!([0, 2]);
            }),
            ("no prompts", |m| m["prompts"] = json!([])),
            ("path", |m| {
                m["prompts"][0]["logits_file"] = json!("../logits-0.f32le")
            }),
            ("parent", |m| m["prompts"][0]["logits_file"] = json!("..")),
            ("no weights", |m| {
                m["provenance"]["weights_sha256"] = json!({})
            }),
            ("layers", |m| {
                m["provenance"]["num_hidden_layers"] = json!(0)
            }),
            ("flag", |m| m["provenance"]["bf16_resident"] = json!("no")),
            ("greedy prompt", |m| m["greedy"]["prompt_index"] = json!(1)),
            ("margin overflow", |m| {
                m["greedy"]["steps"][0]["margin"] = json!(1e39)
            }),
            ("negative margin", |m| {
                m["greedy"]["steps"][0]["margin"] = json!(-1.0)
            }),
        ];
        for (what, mutate) in mutations {
            let mut changed = valid.clone();
            mutate(&mut changed);
            refuses(what, move || {
                parse(&changed);
            });
        }
        refuses("not JSON", || {
            Manifest::parse("{");
        });
    }

    /// The deviation reports the first largest gap, an F64 mean, and refuses non-finite or mismatched input.
    #[test]
    fn deviation_locates_the_first_largest_gap_and_rejects_non_finite_logits() {
        let truth = [0.0, 1.0, 2.0, 3.0];
        let deviation = Deviation::measure(&[0.5, 1.0, 0.0, 5.0], &truth);
        assert_eq!((deviation.maximum, deviation.index), (2.0, 2));
        assert_eq!(deviation.mean, 4.5 / 4.0);
        let exact = Deviation::measure(&truth, &truth);
        assert_eq!((exact.maximum, exact.index, exact.mean), (0.0, 0, 0.0));
        // The difference is taken in F64: F32 subtraction would round this one.
        let small = Deviation::measure(&[1.0 + f32::EPSILON], &[-f32::EPSILON / 4.0]);
        assert_eq!(small.maximum, 1.0 + 1.25 * f64::from(f32::EPSILON));
        for (actual, expected) in [
            (vec![f32::NAN, 0.0], vec![0.0, 0.0]),
            (vec![0.0, f32::INFINITY], vec![0.0, 0.0]),
            (vec![0.0, 0.0], vec![f32::NEG_INFINITY, 0.0]),
            (vec![0.0], vec![0.0, 0.0]),
            (vec![], vec![]),
        ] {
            refuses("invalid logits", move || {
                Deviation::measure(&actual, &expected);
            });
        }
    }

    /// Both bounds are SLACK times the official statistic, inclusive at the bound.
    #[test]
    fn rule_scales_both_official_statistics_and_admits_ties() {
        let official = Deviation {
            maximum: 1.25,
            index: 3,
            mean: 0.125,
        };
        let bound = Bound::from_official(&official);
        assert_eq!(bound.maximum, SLACK * 1.25);
        assert_eq!(bound.mean, SLACK * 0.125);
        const { assert!(SLACK > 1.1725 && SLACK <= 1.2) };
        let at = |maximum: f64, mean: f64| Deviation {
            maximum,
            index: 0,
            mean,
        };
        assert!(bound.admits(&at(bound.maximum, bound.mean)), "ties pass");
        assert!(
            bound.admits(&official),
            "the official run passes its own rule"
        );
        assert!(!bound.admits(&at(bound.maximum.next_up(), 0.0)));
        assert!(!bound.admits(&at(0.0, bound.mean.next_up())));
        assert!(
            bound.admits_rows(&at(bound.maximum, 1e9)),
            "rows ignore the mean"
        );
        assert!(!bound.admits_rows(&at(bound.maximum.next_up(), 0.0)));
        for (maximum, mean) in [
            (0.0, 0.1),
            (0.1, 0.0),
            (f64::NAN, 0.1),
            (f64::INFINITY, 0.1),
            (0.1, f64::NAN),
        ] {
            refuses("a degenerate official deviation", move || {
                Bound::from_official(&at(maximum, mean));
            });
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
        assert!(
            decisive_steps(&greedy[1..3], &bound).is_empty(),
            "an empty greedy set"
        );
        assert!(decisive_steps(&[], &bound).is_empty());
    }

    /// Self-consistency tolerances exist exactly for the half-precision dtypes.
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

    /// Calibration reads both bundles, measures each prompt, and refuses corrupted or degenerate bundles.
    #[test]
    fn calibration_measures_paired_bundles_and_refuses_corruption() {
        let root = tempfile::tempdir().unwrap();
        let exact: [&[f32]; 2] = [&[1.0, 2.0, 3.0, 4.0], &[0.5, -0.5]];
        let rounded: [&[f32]; 2] = [&[1.0, 2.5, 3.0, 3.0], &[0.75, -0.5]];
        let truth = write_bundle(root.path(), "truth", "float32", &exact);
        let official = write_bundle(root.path(), "official", "bfloat16", &rounded);
        let calibration = Calibration::read(&truth, &official);
        assert_eq!(calibration.prompts.len(), 2);
        assert_eq!(calibration.prompts[0].tokens, 2);
        assert_eq!(calibration.prompts[0].official.maximum, 1.0);
        assert_eq!(calibration.prompts[0].official.index, 3);
        assert_eq!(calibration.prompts[0].official.mean, 1.5 / 4.0);
        assert_eq!(calibration.prompts[1].official.maximum, 0.25);
        assert_eq!(calibration.prompts[1].bound.maximum, SLACK * 0.25);
        assert_eq!(calibration.truth_sha256.len(), 64);
        assert_ne!(calibration.truth_sha256, calibration.official_sha256);
        // Prompt 0's threshold is 2.4: only the first step (margin 3) is decisive.
        assert_eq!(calibration.decisive_greedy_steps(), [0]);
        calibration.report(&truth, &official);

        let identical = write_bundle(root.path(), "identical", "bfloat16", &exact);
        refuses("an official bundle equal to the truth", || {
            Calibration::read(&truth, &identical);
        });
        let infinite = write_bundle(
            root.path(),
            "infinite",
            "bfloat16",
            &[&[1.0, 2.0, 3.0, f32::INFINITY], &[0.75, -0.5]],
        );
        refuses("a non-finite official logit", || {
            Calibration::read(&truth, &infinite);
        });
        let relabeled = write_bundle(root.path(), "relabeled", "float32", &rounded);
        refuses("a float32 official bundle", || {
            Calibration::read(&truth, &relabeled);
        });
        std::fs::write(official.join("logits-1.f32le"), [0_u8; 8]).unwrap();
        refuses("a corrupted logits file", || {
            Calibration::read(&truth, &official);
        });
        let short = write_bundle(
            root.path(),
            "short",
            "bfloat16",
            &[&[1.0, 2.5, 3.0, 3.0], &[0.75, -0.5]],
        );
        let mut manifest: Value =
            serde_json::from_str(&std::fs::read_to_string(short.join("reference.json")).unwrap())
                .unwrap();
        let bytes = [
            0.75_f32.to_le_bytes(),
            (-0.5_f32).to_le_bytes(),
            0.0_f32.to_le_bytes(),
        ]
        .concat();
        std::fs::write(short.join("logits-1.f32le"), &bytes).unwrap();
        manifest["prompts"][1]["logits_sha256"] = json!(hex(&Sha256::digest(&bytes)));
        std::fs::write(short.join("reference.json"), manifest.to_string()).unwrap();
        refuses("a logits file of the wrong size", || {
            Calibration::read(&truth, &short);
        });
        refuses("a missing bundle", || {
            Calibration::read(&truth, &root.path().join("absent"));
        });
    }

    /// The gate applies both bounds to whole prompts, the maximum to rows, and the dtype tolerance to self-consistency.
    #[test]
    fn gate_applies_the_rule_to_prompts_rows_and_self_consistency() {
        let root = tempfile::tempdir().unwrap();
        let truth = write_bundle(root.path(), "truth", "float64", &[&[1.0, 2.0, 3.0, 4.0]]);
        let official = write_bundle(
            root.path(),
            "official",
            "bfloat16",
            &[&[1.0, 2.5, 3.0, 3.0]],
        );
        refuses("an F32 run", || {
            HalfPrecisionGate::new(TensorDType::F32, &truth, &official);
        });
        let gate = HalfPrecisionGate::new(TensorDType::BF16, &truth, &official);
        // M = 1 and μ = 0.375, so the bounds are 1.2 and 0.45.
        let reference = [1.0, 2.0, 3.0, 4.0];
        let observed = gate.check_prompt(0, &[1.0, 2.0, 3.0, 5.0], &reference);
        assert_eq!((observed.maximum, observed.index), (1.0, 3));
        let gate_ = gate.clone();
        refuses("a maximum beyond the bound", move || {
            gate_.check_prompt(0, &[1.0, 2.0, 3.0, 5.25], &reference);
        });
        let gate_ = gate.clone();
        refuses("a mean beyond the bound", move || {
            gate_.check_prompt(0, &[1.5, 2.5, 3.5, 5.0], &reference);
        });
        let gate_ = gate.clone();
        refuses("a partial prompt", move || {
            gate_.check_prompt(0, &[1.0, 2.0], &reference[..2]);
        });
        gate.check_rows("row", 0, &[4.5, 1.0], &[3.5, 0.0]);
        let gate_ = gate.clone();
        refuses("a row beyond the maximum bound", move || {
            gate_.check_rows("row", 0, &[4.75], &[3.5]);
        });
        gate.check_consistency("batched", &[1.5, 2.0], &[1.0, 2.0]);
        let gate_ = gate.clone();
        refuses("an inconsistent batched row", move || {
            gate_.check_consistency("batched", &[1.5, 2.0], &[0.75, 2.0]);
        });
        assert!(gate.is_decisive(3.0) && !gate.is_decisive(2.375));
        assert_eq!(
            gate.bound(0),
            Bound {
                maximum: SLACK,
                mean: SLACK * 0.375
            }
        );
    }
}
