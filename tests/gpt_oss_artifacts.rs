//! Real gpt-oss checkpoints against locally generated Transformers references.
//!
//! Reference bundles come from `scripts/generate_causal_lm_reference.py`
//! (Transformers `AutoModelForCausalLM`, CPU, eager attention). Without CUDA
//! kernels Transformers decodes the MXFP4 experts to BF16 (its
//! `convert_moe_packed_tensors`, which this crate reproduces bit for bit). Two
//! kinds of bundle are useful:
//!
//! * a **full** F32 reference of gpt-oss-20b, the release gate. The F32 model
//!   does not fit a 96 GiB host, so `--bf16-resident` keeps the decoded BF16
//!   weights resident and widens each block to F32 only while it runs, which
//!   computes exactly what the F32 model computes, because BF16 widens to F32
//!   exactly (`scripts/tests/test_generate_causal_lm_reference.py` checks
//!   bitwise equality with a permanently widened model on the tiny gpt-oss
//!   fixture); it needs about 47 GB:
//!
//!   ```text
//!   python3 scripts/generate_causal_lm_reference.py --dtype float32 --bf16-resident \
//!     --model-dir /Volumes/T9/models/openai/gpt-oss-20b \
//!     --output /path/to/gpt-oss-20b-6cee5e81-f32-reference.bundle \
//!     --repository openai/gpt-oss-20b --revision 6cee5e81ee83917806bbde320786a8fb61efebee
//!   ```
//!
//!   A full `--dtype bfloat16` bundle measures Transformers' own BF16
//!   rounding against it.
//!
//! * a **truncated** F32 reference (`--num-hidden-layers 1`): the real
//!   embedding, first block (sliding attention with real sinks, the real
//!   router, 32 MXFP4-decoded experts), final norm, and head, in about 9 GiB.
//!   It checks real weights end to end at F32 precision on Flex. This test
//!   reads the truncation from the bundle's provenance and loads only those
//!   blocks, allowlisting the checkpoint's other blocks as unread.
//!
//! The checks cover every prompt position's logits and decisive argmaxes,
//! the cached greedy continuation, a ragged left-padded batched prefill
//! (also against the crate's own stateless pass), and the longest prompt
//! decoded token by token through the caches; include a prompt longer than
//! the 128-token window (`--prompt`, repeatable) so sliding layers wrap
//! their ring.
//!
//! Each bundle records token IDs of a few fixed prompts, every prompt
//! position's full-vocabulary logits (as F32), and a 32-token cached greedy
//! continuation with each step's top-2 margin. The tests are ignored by
//! default and need two local directories:
//!
//! ```text
//! TRANSFORMERS_GPT_OSS_DIR=/Volumes/T9/models/openai/gpt-oss-20b \
//! TRANSFORMERS_GPT_OSS_REFERENCE_DIR=/path/to/reference.bundle \
//! cargo test --release [--features metal|cuda] --test gpt_oss_artifacts -- --ignored --nocapture --test-threads 1
//! ```
//!
//! A full bundle loads through the strict automatic loader; a truncated one
//! through the family loader. Either way no tensor may be missing or
//! unexpectedly present, and every expert projection of the loaded blocks
//! must be decoded from MXFP4. The pairing of checkpoint and bundle is
//! verified through the bundle's SHA-256 of `config.json`.
//!
//! # Memory
//!
//! gpt-oss-20b has 20.9 B parameters, 19.1 B of them in routed experts.
//! Loaded in F16 or BF16 the model needs about 39 GiB, plus at most about
//! 1.4 GiB of load staging (the largest expert tensor's 265 MB of blocks,
//! 17 MB of scales, and its 1.06 GB decoded F16 output). F32 needs about
//! 78 GiB and does not fit a 96 GiB host beside the operating system. The
//! reference generator holds the BF16 model (about 39 GiB) plus the
//! checkpoint's MXFP4 tensors while converting, so it needs about 55 GiB. A
//! one-block truncation needs about 9 GiB in F32 on either side.
//!
//! # Dtype and tolerance
//!
//! `TRANSFORMERS_GPT_OSS_DTYPE` selects the stored dtype (`f16`, `bf16`,
//! `f32`, or `f64`); it defaults to the bundle's own dtype, so an F32 bundle
//! runs in F32 and a BF16 bundle in BF16 (set `f16` for the Metal F16
//! profile). An F32 run against an F32 bundle differs only by reduction
//! order: every logit must lie within [`F32_TOLERANCE`] and their mean
//! deviation within [`F32_MEAN_TOLERANCE`]. Whenever the run or the bundle is
//! half precision, rounding dominates and [`HALF_TOLERANCE`] and
//! [`HALF_MEAN_TOLERANCE`] apply instead. `TRANSFORMERS_GPT_OSS_TOLERANCE`
//! and `TRANSFORMERS_GPT_OSS_MEAN_TOLERANCE` override them.
//!
//! Matching storage dtypes does not make a half-precision comparison like for
//! like: Transformers' BF16 eager path also rounds its attention and router
//! softmaxes to BF16, while this crate normalizes in F32. On four real blocks
//! (Flex, the four prompts below), the crate's BF16 run is no closer to the
//! BF16 reference than to the F32 one, and the BF16 reference itself is about
//! as far from the F32 reference:
//!
//! | run | reference | max \|Δ\| | mean \|Δ\| per prompt |
//! | --- | --- | --- | --- |
//! | crate F32 | F32 | `1.3e-4` | below `5e-6` |
//! | crate BF16 | F32 | `1.37` | at most `0.030` |
//! | crate F16 | F32 | `1.82` | at most `0.062` |
//! | crate BF16 | BF16 | `1.71` | at most `0.054` |
//! | crate F16 | BF16 | `1.85` | at most `0.060` |
//! | Transformers BF16 | F32 | `1.73` | at most `0.055` |
//!
//! So a half-precision run (Metal F16 included) is qualified against an
//! **F32** bundle, which measures its own rounding. Against a half-precision
//! bundle the two implementations' roundings add, which a single
//! half-precision bound does not describe. On the full 24-block model against
//! the full F32 bundle (the four prompts below):
//!
//! | run | max \|Δ\| per prompt | mean \|Δ\| per prompt |
//! | --- | --- | --- |
//! | Transformers BF16 | `2.665`, `1.24`, `0.70`, `1.66` | at most `0.108` |
//! | crate Flex BF16 | `2.19`, `1.26`, `0.57`, `1.67` | at most `0.112` |
//! | crate Metal F16 | `0.018`, `0.18`, `0.53`, `1.15` | at most `0.025` |
//!
//! The maximum bound, [`HALF_TOLERANCE`], is therefore calibrated on the
//! official implementation itself: Transformers' own BF16 evaluation of the
//! full model deviates from the F32 reference by at most `2.665` (prompt 0,
//! position 3), rounded up to `2.7`. The mean bound, [`HALF_MEAN_TOLERANCE`],
//! is not calibrated that way. Its `0.15` is kept from the earlier four-block
//! calibration as a separator from architectural errors. It lies above both
//! Transformers' largest BF16 mean (`0.108`) and the crate's Flex/BF16 one
//! (`0.112`), so on the mean the crate's Flex/BF16 run is slightly less close
//! to F32 than the official BF16 evaluation.
//!
//! The Transformers BF16 row needs no checkpoint to reproduce:
//! `transformers_bf16_reference_stays_within_the_half_precision_bounds`
//! compares the two full bundles and asserts that row against both bounds:
//!
//! ```text
//! TRANSFORMERS_GPT_OSS_BF16_REFERENCE_DIR=/path/to/gpt-oss-20b-6cee5e81-bf16-reference.bundle \
//! TRANSFORMERS_GPT_OSS_F32_REFERENCE_DIR=/path/to/gpt-oss-20b-6cee5e81-f32-reference.bundle \
//! cargo test --test gpt_oss_artifacts -- --ignored --exact \
//!   transformers_bf16_reference_stays_within_the_half_precision_bounds --nocapture
//! ```
//!
//! Its output is recorded as run `gpt-oss-20b-transformers-bf16-vs-f32-bundles`
//! of `acceptance/measurements/reference-precision-followups-20261016.json`.
//! That run measures the official reference against itself and evaluates no
//! Burn model, so it is not a pass of any gate.
//!
//! Independently of the reference, cached and batched passes must agree with
//! this crate's own stateless pass within [`SELF_CONSISTENCY_TOLERANCE`] for
//! the dtype. The greedy check feeds the reference's tokens back, so every
//! step continues the reference's prefix, and compares each step whose
//! reference top-2 margin exceeds twice the tolerance; a closer race is a tie
//! at that tolerance and is skipped rather than ending the comparison
//! (`tests/support/backend_numerics.rs`, "Greedy ties"). At least
//! [`DEFAULT_MIN_GREEDY`] tokens must be compared
//! (`TRANSFORMERS_GPT_OSS_MIN_GREEDY` overrides it). At the half-precision
//! tolerance only margins above `5.4` are decisive: the full F32 bundle has
//! seven of 32 and the full BF16 bundle five (both first margins, `3.09` and
//! `3.06`, are ties), so the defaults apply to them, but a truncated model's
//! greedy margins are mostly below one logit (the pinned one-block F32 bundle
//! has one above `5.4`, the four-block bundles none), so a half-precision run
//! against a truncated bundle needs `TRANSFORMERS_GPT_OSS_MIN_GREEDY=0`.
//!
//! # Measured
//!
//! On Flex/F32 against truncated F32 bundles of gpt-oss-20b at `6cee5e8`
//! (Transformers 5.2.0, PyTorch 2.10.0, generated on this host on
//! 2026-10-15): one block with the default prompts deviates by at most
//! `7.6e-5` (mean `3.4e-6`); four blocks (two sliding, two full), with a
//! 183-token fourth prompt, by at most `1.3e-4` (mean below `5e-6`), the
//! token-by-token cached decode of that prompt (wrapping the 128-slot ring)
//! by `7.6e-5`, and all 32 greedy tokens match. The half-precision rows of
//! the table above come from the same four-block F32 bundle and a four-block
//! BF16 bundle generated from the same prompts
//! (`gpt-oss-20b-6cee5e81-4layer-{f32,bf16}-reference.bundle`).
//!
//! Against the full 24-block bundles of the same four prompts
//! (`acceptance/measurements/large-checkpoint-replay-20261016.json` and
//! `acceptance/measurements/reference-precision-followups-20261016.json`):
//! against the F32 bundle, Metal/F16 and Flex/BF16 deviate as tabulated
//! above, their batched rows by at most `0.53` and `0.68`, and the cached
//! long-prompt decode by `0.94` and `1.59`. Against the BF16 bundle Metal/F16
//! reached `2.66` at prompt 0, about the BF16 bundle's own distance from F32
//! there. That comparison of two half-precision roundings is informational,
//! not a gate pass. Those runs predate the tie-skipping greedy check: under the
//! earlier stop-at-the-first-tie rule the F32 reference's first margin,
//! `3.09`, ended the comparison, so they ran with
//! `TRANSFORMERS_GPT_OSS_MIN_GREEDY=0`. Against the full-model BF16 bundle at
//! default settings, the tie-skipping cached greedy check on Flex in BF16 and
//! on Metal in F16 (`TRANSFORMERS_GPT_OSS_DTYPE=f16`) each matched all five
//! decisive steps and skipped 27 ties (2026-10-17, at the earlier `2.5`
//! bound, whose decisive steps are the same five); no tie-skipping
//! half-precision run against the full F32 bundle has been recorded yet.
//!
//! # CUDA
//!
//! The `cuda` module repeats the four checks on Burn's `Cuda` backend
//! (device 0; `CUDA_VISIBLE_DEVICES` selects the GPU), built and run on the
//! GPU host with the CUDA toolkit available (`nvcc` at build time,
//! `$CUDA_PATH/include` for run-time kernel compilation):
//!
//! ```text
//! TRANSFORMERS_GPT_OSS_DIR=/path/to/gpt-oss-20b \
//! TRANSFORMERS_GPT_OSS_REFERENCE_DIR=/path/to/reference.bundle \
//! cargo test --release --features cuda --test gpt_oss_artifacts cuda_ -- --ignored --nocapture --test-threads 1
//! ```
//!
//! The memory figures above become device-memory requirements there.
//! Half-precision runs multiply through half-precision tensor cores and keep
//! their rounding-derived defaults. For an F32 run, each load first probes
//! how the device multiplies F32 matrices (`tests/support/backend_numerics.rs`):
//! a GPU below compute capability 8.0 multiplies in F32 and the F32 defaults
//! apply, while from compute capability 8.0 Burn 0.21 rounds F32 matrix
//! inputs to TF32 (10-bit mantissas), which no measured default covers. Such
//! an F32 run against an F32 bundle requires `TRANSFORMERS_GPT_OSS_TOLERANCE`
//! and `TRANSFORMERS_GPT_OSS_MEAN_TOLERANCE`, holds its self-consistency to
//! the former, and qualifies TF32 execution, not F32 numerics. No CUDA run
//! has been recorded yet.

#![cfg(feature = "flex")]

#[path = "support/backend_numerics.rs"]
mod backend_numerics;

use std::path::{Path, PathBuf};

use backend_numerics::{
    Gap, MatmulPrecision, argmax, check_teacher_forced_greedy, checkpoint_tolerance,
    greedy_step_is_decisive, top_two_margin,
};
use burn::{
    backend::Flex,
    tensor::{DType, Int, Tensor, TensorData, backend::Backend},
};
use serde_json::Value;
use sha2::{Digest, Sha256};
use transformers::{
    AutoCausalLm, AutoLoadOptions, CausalLmInput, DecodeInput, ForwardOptions, LogitsSelection,
    TokenMask,
    cache::CacheConfig,
    checkpoint::{DTypePolicy, KeyMatcher, LoadOptions, LoadReport, TensorDType},
    models::{
        config::ModelConfig,
        gpt_oss::{GptOssConfig, GptOssForCausalLm},
    },
};

/// Default absolute logit tolerance of an F32 (or F64) run against an F32 bundle.
const F32_TOLERANCE: f32 = 1e-3;

/// Default bound on one prompt's mean absolute logit deviation, F32 run against an F32 bundle.
const F32_MEAN_TOLERANCE: f32 = 1e-4;

/// Default absolute logit tolerance when the run or the bundle is half precision.
///
/// Transformers' own BF16 evaluation of the full model deviates from the full
/// F32 reference by at most `2.665` (see the module documentation). This
/// bound is that figure rounded up, so a half-precision run may deviate about
/// as far as the official implementation's own BF16 evaluation. The figure is
/// reproduced by
/// `transformers_bf16_reference_stays_within_the_half_precision_bounds` and
/// recorded as run `gpt-oss-20b-transformers-bf16-vs-f32-bundles` of
/// `acceptance/measurements/reference-precision-followups-20261016.json`.
const HALF_TOLERANCE: f32 = 2.7;

/// Default bound on one prompt's mean absolute logit deviation in half precision.
///
/// Not calibrated to the official BF16 mean: `0.15` is kept from the earlier
/// four-block calibration. Against the full F32 reference, Transformers' own
/// BF16 evaluation reaches `0.108` per prompt, the crate's Flex/BF16 run
/// `0.112` (slightly above the official figure), and Metal/F16 `0.025`. An
/// architectural error moves most logits of most positions, so the mean
/// separates it from rounding noise far better than the maximum.
const HALF_MEAN_TOLERANCE: f32 = 0.15;

/// Absolute agreement of cached/batched passes with this crate's stateless pass, per dtype.
///
/// The batched check compares a left-padded batch with each prompt run
/// alone, so it also absorbs any shape dependence of half-precision kernels.
/// On four real blocks on Flex the largest gaps were `3.6e-5` (F32), `0.125`
/// (BF16), and `0.41` (F16): Flex's F16 matmul accumulates in F16, so F16
/// runs there are noisier than BF16 ones.
const SELF_CONSISTENCY_TOLERANCE: [(TensorDType, f32); 3] = [
    (TensorDType::F32, 1e-3),
    (TensorDType::F16, 0.5),
    (TensorDType::BF16, 0.5),
];

/// Default minimum number of decisive greedy tokens that must be compared.
const DEFAULT_MIN_GREEDY: usize = 2;

/// One prompt of the reference bundle.
struct Prompt {
    /// Token IDs of the unpadded prompt.
    tokens: Vec<i64>,
    /// Row-major `[tokens, vocabulary]` logits.
    logits: Vec<f32>,
}

/// A parsed reference bundle.
struct Reference {
    /// Vocabulary size `V`.
    vocabulary: usize,
    /// Prompts in bundle order.
    prompts: Vec<Prompt>,
    /// Greedy continuation of prompt 0 as `(token, margin)`.
    greedy: Vec<(i64, f32)>,
    /// SHA-256 of the configuration the bundle was generated from.
    config_sha256: String,
    /// Evaluation dtype of the reference (`float32` or `bfloat16`).
    dtype: String,
    /// Blocks kept by a truncated reference, or `None` for the whole model.
    layers: Option<usize>,
}

impl Reference {
    /// Reads and checks a bundle directory.
    fn read(root: &Path) -> Self {
        let manifest: Value =
            serde_json::from_str(&std::fs::read_to_string(root.join("reference.json")).unwrap())
                .unwrap();
        assert_eq!(manifest["model_type"], "gpt_oss", "not a gpt-oss bundle");
        let vocabulary = manifest["vocabulary_size"].as_u64().unwrap() as usize;
        let prompts = manifest["prompts"]
            .as_array()
            .unwrap()
            .iter()
            .map(|prompt| {
                let tokens: Vec<i64> = serde_json::from_value(prompt["token_ids"].clone()).unwrap();
                let file = root.join(prompt["logits_file"].as_str().unwrap());
                let bytes = std::fs::read(&file).unwrap();
                assert_eq!(
                    hex(&Sha256::digest(&bytes)),
                    prompt["logits_sha256"].as_str().unwrap(),
                    "{} is corrupted",
                    file.display()
                );
                let logits: Vec<f32> = bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|chunk| f32::from_le_bytes(*chunk))
                    .collect();
                assert_eq!(logits.len(), tokens.len() * vocabulary);
                Prompt { tokens, logits }
            })
            .collect();
        let greedy = manifest["greedy"]["steps"]
            .as_array()
            .unwrap()
            .iter()
            .map(|step| {
                (
                    step["token"].as_i64().unwrap(),
                    step["margin"].as_f64().unwrap() as f32,
                )
            })
            .collect();
        let provenance = &manifest["provenance"];
        Self {
            vocabulary,
            prompts,
            greedy,
            config_sha256: provenance["config_sha256"].as_str().unwrap().into(),
            dtype: provenance["dtype"].as_str().unwrap_or("float32").into(),
            layers: provenance["num_hidden_layers"]
                .as_u64()
                .map(|layers| layers as usize),
        }
    }
}

/// Lower-case hexadecimal digest.
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// Run settings read from the environment and the bundle.
struct Settings {
    /// Checkpoint directory.
    model_dir: PathBuf,
    /// Stored dtype of every floating parameter.
    dtype: TensorDType,
    /// Absolute logit tolerance against the reference.
    tolerance: f32,
    /// Bound on each prompt's mean absolute logit deviation from the reference.
    mean_tolerance: f32,
    /// Minimum number of decisive greedy tokens.
    minimum_greedy: usize,
    /// Whether this is a full-precision run whose matrix products round their inputs to TF32.
    tf32: bool,
}

/// Returns the dtype a bundle was evaluated in, from its provenance name.
fn bundle_dtype(name: &str) -> TensorDType {
    match name {
        "float32" => TensorDType::F32,
        "bfloat16" => TensorDType::BF16,
        other => panic!("unsupported reference dtype {other}"),
    }
}

/// Whether a `run` against a `bundle` reference is full precision on both sides.
fn full_precision(bundle: TensorDType, run: TensorDType) -> bool {
    bundle == TensorDType::F32 && matches!(run, TensorDType::F32 | TensorDType::F64)
}

/// Returns the default `(maximum, mean)` logit tolerances of running `run` against a `bundle` reference.
///
/// Only a full-precision run against an F32 bundle is held to reduction-order
/// accuracy; any half-precision side is held to rounding accuracy.
fn default_tolerances(bundle: TensorDType, run: TensorDType) -> (f32, f32) {
    if full_precision(bundle, run) {
        (F32_TOLERANCE, F32_MEAN_TOLERANCE)
    } else {
        (HALF_TOLERANCE, HALF_MEAN_TOLERANCE)
    }
}

impl Settings {
    /// Reads the settings; dtype and tolerance defaults follow the bundle's dtype.
    ///
    /// `precision` is how the device multiplies F32 matrices. A full-precision
    /// run on a TF32 device has no default tolerances: both
    /// `TRANSFORMERS_GPT_OSS_TOLERANCE` and `TRANSFORMERS_GPT_OSS_MEAN_TOLERANCE`
    /// must be set. Half-precision runs keep their rounding-derived defaults.
    fn new(model_dir: PathBuf, reference: &Reference, precision: MatmulPrecision) -> Self {
        let bundle = bundle_dtype(&reference.dtype);
        let dtype = match std::env::var("TRANSFORMERS_GPT_OSS_DTYPE").as_deref() {
            Ok("f64") => TensorDType::F64,
            Ok("f32") => TensorDType::F32,
            Ok("bf16") => TensorDType::BF16,
            Ok("f16") => TensorDType::F16,
            Err(_) => bundle,
            Ok(other) => {
                panic!("TRANSFORMERS_GPT_OSS_DTYPE must be f16, bf16, f32, or f64, not {other}")
            }
        };
        let (default_tolerance, default_mean) = default_tolerances(bundle, dtype);
        // Only the full-precision defaults assume F32 matrix products.
        let tf32 = full_precision(bundle, dtype) && precision == MatmulPrecision::Tf32;
        let defaults = if tf32 {
            MatmulPrecision::Tf32
        } else {
            MatmulPrecision::F32
        };
        let parse = |name: &str| {
            std::env::var(name)
                .ok()
                .map(|value| value.parse::<f32>().expect(name))
        };
        Self {
            model_dir,
            dtype,
            tolerance: checkpoint_tolerance(
                "TRANSFORMERS_GPT_OSS_TOLERANCE",
                default_tolerance,
                defaults,
            ),
            mean_tolerance: checkpoint_tolerance(
                "TRANSFORMERS_GPT_OSS_MEAN_TOLERANCE",
                default_mean,
                defaults,
            ),
            minimum_greedy: parse("TRANSFORMERS_GPT_OSS_MIN_GREEDY")
                .map_or(DEFAULT_MIN_GREEDY, |value| value as usize),
            tf32,
        }
    }

    /// Returns the self-consistency tolerance of the selected dtype.
    ///
    /// A full-precision run on a TF32 device is held to its explicit
    /// tolerance instead: TF32 input rounding can turn F32-level differences
    /// between the cached, batched, and stateless graphs into TF32-level ones.
    fn consistency(&self) -> f32 {
        if self.tf32 {
            return self.tolerance;
        }
        SELF_CONSISTENCY_TOLERANCE
            .iter()
            .find(|(dtype, _)| *dtype == self.dtype)
            .map_or(1.0, |&(_, tolerance)| tolerance)
    }
}

/// Returns a required directory from the environment.
fn directory(name: &str) -> PathBuf {
    std::env::var_os(name)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("set {name}; see the module documentation"))
}

/// Checks the strict report: nothing missing or unexpected, and `layers` blocks of MXFP4 experts.
fn check_report(report: &LoadReport, layers: usize) {
    assert!(
        report.missing.is_empty() && report.unexpected.is_empty(),
        "missing {:?}, unexpected {:?}",
        report.missing,
        report.unexpected
    );
    let decoded = report
        .transformed
        .iter()
        .filter(|record| record.operation == "dequantize_mxfp4")
        .count();
    assert_eq!(
        decoded,
        2 * layers,
        "every expert projection decodes from MXFP4"
    );
}

/// Loads the checkpoint (truncated as the bundle says) after checking it matches the bundle.
fn load<B: Backend>(device: &B::Device) -> (AutoCausalLm<B>, Reference, Settings) {
    let model_dir = directory("TRANSFORMERS_GPT_OSS_DIR");
    let reference = Reference::read(&directory("TRANSFORMERS_GPT_OSS_REFERENCE_DIR"));
    let precision = MatmulPrecision::probe::<B>(device);
    eprintln!("F32 matrix products use {precision:?} inputs");
    let settings = Settings::new(model_dir, &reference, precision);
    let text = std::fs::read_to_string(settings.model_dir.join("config.json")).unwrap();
    assert_eq!(
        hex(&Sha256::digest(text.as_bytes())),
        reference.config_sha256,
        "the bundle was generated from another config.json"
    );
    let checkpoint = LoadOptions {
        dtype_policy: DTypePolicy::Cast(settings.dtype),
        ..LoadOptions::default()
    };
    let started = std::time::Instant::now();
    let model = match reference.layers {
        None => {
            let options = AutoLoadOptions {
                checkpoint,
                load_tokenizer: false,
                ..AutoLoadOptions::default()
            };
            let loaded = AutoCausalLm::<B>::load_local(&settings.model_dir, &options, device)
                .expect("strict load");
            let layers = GptOssConfig::from_json_str(&text)
                .unwrap()
                .num_hidden_layers;
            check_report(loaded.load_report(), layers);
            loaded.into_model()
        }
        Some(layers) => {
            let mut config = GptOssConfig::from_json_str(&text).unwrap();
            let total = config.num_hidden_layers;
            config.num_hidden_layers = layers;
            config.layer_types.truncate(layers);
            let options = LoadOptions {
                allow_unexpected: (layers..total)
                    .map(|layer| KeyMatcher::prefix(format!("model.layers.{layer}.")))
                    .collect(),
                ..checkpoint
            };
            let mut model = GptOssForCausalLm::<B>::init(config, device).unwrap();
            let outcome = model
                .load_checkpoint(&settings.model_dir, &options)
                .expect("strict truncated load");
            check_report(&outcome.report, layers);
            AutoCausalLm::from(model)
        }
    };
    eprintln!(
        "loaded {} blocks as {:?} in {:?}",
        reference
            .layers
            .map_or("all".into(), |layers| layers.to_string()),
        settings.dtype,
        started.elapsed()
    );
    assert_eq!(model.config().vocabulary_size(), reference.vocabulary);
    (model, reference, settings)
}

/// Uploads one row of token IDs.
fn row<B: Backend>(tokens: &[i64], device: &B::Device) -> Tensor<B, 2, Int> {
    Tensor::from_data(TensorData::new(tokens.to_vec(), [1, tokens.len()]), device)
}

/// Reads logits to the host as F32.
fn host<B: Backend, const D: usize>(tensor: Tensor<B, D>) -> Vec<f32> {
    tensor.cast(DType::F32).into_data().to_vec::<f32>().unwrap()
}

/// Returns the largest absolute difference of two equally long sets of logits.
///
/// A non-finite value on either side counts as an infinite difference, so a
/// NaN or infinite logit fails every tolerance (see [`Gap::measure`], which
/// also rejects a length mismatch).
fn gap(actual: &[f32], expected: &[f32]) -> f32 {
    Gap::measure(actual, expected).deviation
}

/// Returns the flat index and value of the largest absolute difference, the first on ties.
///
/// Callers check finiteness first; a NaN difference never replaces the running maximum.
fn largest_gap(actual: &[f32], expected: &[f32]) -> (usize, f32) {
    actual
        .iter()
        .zip(expected)
        .map(|(a, e)| (a - e).abs())
        .enumerate()
        .fold(
            (0, 0.0),
            |best, (index, value)| {
                if value > best.1 { (index, value) } else { best }
            },
        )
}

/// Returns the mean absolute difference, accumulated in F64 over `actual`'s length.
fn mean_gap(actual: &[f32], expected: &[f32]) -> f64 {
    actual
        .iter()
        .zip(expected)
        .map(|(a, e)| f64::from((a - e).abs()))
        .sum::<f64>()
        / actual.len() as f64
}

/// Stateless logits at every prompt position match the reference; decisive argmaxes agree.
fn check_prompt_logits<B: Backend>(device: &B::Device) {
    let (model, reference, settings) = load::<B>(device);
    let options = ForwardOptions {
        logits: LogitsSelection::All,
        ..ForwardOptions::default()
    };
    let vocabulary = reference.vocabulary;
    for (index, prompt) in reference.prompts.iter().enumerate() {
        let input = CausalLmInput {
            token_ids: row::<B>(&prompt.tokens, device),
            token_mask: None,
            position_ids: None,
        };
        let logits = host(model.forward(input, &options).unwrap().logits);
        let largest = gap(&logits, &prompt.logits);
        let mean = mean_gap(&logits, &prompt.logits);
        eprintln!(
            "prompt {index}: {} tokens, max |Δ| = {largest:e}, mean |Δ| = {mean:e}",
            prompt.tokens.len()
        );
        for position in 0..prompt.tokens.len() {
            let expected = &prompt.logits[position * vocabulary..(position + 1) * vocabulary];
            if greedy_step_is_decisive(top_two_margin(expected), settings.tolerance) {
                let actual = &logits[position * vocabulary..(position + 1) * vocabulary];
                assert_eq!(
                    argmax(actual),
                    argmax(expected),
                    "prompt {index}, position {position}: decisive argmax"
                );
            }
        }
        assert!(
            largest <= settings.tolerance,
            "prompt {index}: max |Δ| = {largest:e}"
        );
        assert!(
            mean <= f64::from(settings.mean_tolerance),
            "prompt {index}: mean |Δ| = {mean:e}"
        );
    }
}

/// The cached greedy continuation of prompt 0 matches at every decisive step.
///
/// The reference's tokens are fed back, so ties are skipped rather than
/// ending the comparison; see [`check_teacher_forced_greedy`].
fn check_cached_greedy<B: Backend>(device: &B::Device) {
    let (model, reference, settings) = load::<B>(device);
    let prompt = &reference.prompts[0];
    let capacity = prompt.tokens.len() + reference.greedy.len();
    let mut state = model
        .new_decode_state(CacheConfig::new(1, capacity).unwrap(), 1, device)
        .unwrap();
    let input = CausalLmInput {
        token_ids: row::<B>(&prompt.tokens, device),
        token_mask: None,
        position_ids: None,
    };
    let logits = host(
        model
            .prefill(input, &mut state, &ForwardOptions::default())
            .unwrap()
            .logits,
    );
    let last = &prompt.logits[(prompt.tokens.len() - 1) * reference.vocabulary..];
    assert!(
        gap(&logits, last) <= settings.tolerance,
        "cached prefill logits"
    );
    check_teacher_forced_greedy(
        "greedy",
        &reference.greedy,
        settings.tolerance,
        settings.minimum_greedy,
        "TRANSFORMERS_GPT_OSS_MIN_GREEDY",
        logits,
        |token| {
            let next = DecodeInput {
                token_ids: row::<B>(&[token], device),
                position_ids: None,
                token_mask: None,
            };
            host(
                model
                    .decode(next, &mut state, &ForwardOptions::default())
                    .unwrap()
                    .logits,
            )
        },
    );
}

/// All prompts in one left-padded batch reproduce the stateless pass and the reference.
fn check_batched_prefill<B: Backend>(device: &B::Device) {
    let (model, reference, settings) = load::<B>(device);
    let width = reference
        .prompts
        .iter()
        .map(|prompt| prompt.tokens.len())
        .max()
        .unwrap();
    let rows = reference.prompts.len();
    let mut tokens = Vec::with_capacity(rows * width);
    let mut valid = Vec::with_capacity(rows * width);
    for prompt in &reference.prompts {
        let padding = width - prompt.tokens.len();
        tokens.extend(std::iter::repeat_n(0, padding).chain(prompt.tokens.iter().copied()));
        valid.extend(
            std::iter::repeat_n(0_i64, padding).chain(std::iter::repeat_n(1, prompt.tokens.len())),
        );
    }
    let token_mask =
        Tensor::<B, 2, Int>::from_data(TensorData::new(valid, [rows, width]), device).equal_elem(1);
    let mut state = model
        .new_decode_state(CacheConfig::new(rows, width).unwrap(), rows, device)
        .unwrap();
    let input = CausalLmInput {
        token_ids: Tensor::from_data(TensorData::new(tokens, [rows, width]), device),
        token_mask: Some(TokenMask::new(token_mask)),
        position_ids: None,
    };
    let logits = host(
        model
            .prefill(input, &mut state, &ForwardOptions::default())
            .unwrap()
            .logits,
    );
    for (index, prompt) in reference.prompts.iter().enumerate() {
        let actual = &logits[index * reference.vocabulary..(index + 1) * reference.vocabulary];
        let alone = host(
            model
                .forward(
                    CausalLmInput {
                        token_ids: row::<B>(&prompt.tokens, device),
                        token_mask: None,
                        position_ids: None,
                    },
                    &ForwardOptions::default(),
                )
                .unwrap()
                .logits,
        );
        let consistency = gap(actual, &alone);
        let expected = &prompt.logits[(prompt.tokens.len() - 1) * reference.vocabulary..];
        let largest = gap(actual, expected);
        eprintln!(
            "batched row {index}: max |Δ| = {largest:e} to the reference, {consistency:e} to the stateless pass"
        );
        assert!(
            consistency <= settings.consistency(),
            "batched row {index} vs stateless"
        );
        assert!(largest <= settings.tolerance, "batched row {index}");
    }
}

/// The longest prompt decodes token by token through the caches, past the sliding window.
///
/// A quarter of the prompt is prefilled into a state whose capacity exceeds
/// gpt-oss's 128-token window (so sliding layers use their ring), and every
/// remaining token is decoded alone and compared with the reference logits
/// at its position. A prompt longer than the window wraps the ring.
fn check_cached_long_prompt<B: Backend>(device: &B::Device) {
    let (model, reference, settings) = load::<B>(device);
    let prompt = reference
        .prompts
        .iter()
        .max_by_key(|prompt| prompt.tokens.len())
        .unwrap();
    let length = prompt.tokens.len();
    let prefill = (length / 4).max(1);
    let capacity = length.max(256);
    let mut state = model
        .new_decode_state(CacheConfig::new(1, capacity).unwrap(), 1, device)
        .unwrap();
    let vocabulary = reference.vocabulary;
    let expected =
        |position: usize| &prompt.logits[position * vocabulary..(position + 1) * vocabulary];
    let logits = host(
        model
            .prefill(
                CausalLmInput {
                    token_ids: row::<B>(&prompt.tokens[..prefill], device),
                    token_mask: None,
                    position_ids: None,
                },
                &mut state,
                &ForwardOptions::default(),
            )
            .unwrap()
            .logits,
    );
    let mut largest = gap(&logits, expected(prefill - 1));
    for position in prefill..length {
        let logits = host(
            model
                .decode(
                    DecodeInput {
                        token_ids: row::<B>(&prompt.tokens[position..=position], device),
                        position_ids: None,
                        token_mask: None,
                    },
                    &mut state,
                    &ForwardOptions::default(),
                )
                .unwrap()
                .logits,
        );
        largest = largest.max(gap(&logits, expected(position)));
    }
    eprintln!(
        "cached decode of a {length}-token prompt (prefill {prefill}): max |Δ| = {largest:e}"
    );
    assert!(largest <= settings.tolerance, "cached long prompt");
}

/// Runs default to the bundle's dtype; only full precision on both sides gets the tight tolerances.
#[test]
fn tolerance_defaults_follow_the_bundle_and_run_dtypes() {
    use TensorDType::{BF16, F16, F32, F64};
    assert_eq!(bundle_dtype("float32"), F32);
    assert_eq!(bundle_dtype("bfloat16"), BF16);
    assert!(std::panic::catch_unwind(|| bundle_dtype("float16")).is_err());
    let tight = (F32_TOLERANCE, F32_MEAN_TOLERANCE);
    let loose = (HALF_TOLERANCE, HALF_MEAN_TOLERANCE);
    assert_eq!(default_tolerances(F32, F32), tight);
    assert_eq!(default_tolerances(F32, F64), tight);
    for (bundle, run) in [
        (F32, F16),
        (F32, BF16),
        (BF16, BF16),
        (BF16, F16),
        (BF16, F32),
    ] {
        assert_eq!(
            default_tolerances(bundle, run),
            loose,
            "{bundle:?} bundle, {run:?} run"
        );
    }
    // The mean bound is the sharper half-precision check, and both admit
    // Transformers' own full-model BF16 deviation from F32.
    const { assert!(HALF_MEAN_TOLERANCE < HALF_TOLERANCE / 10.0) };
    const { assert!(HALF_TOLERANCE > 2.665 && HALF_MEAN_TOLERANCE > 0.108) };
}

/// The deviation helpers report the first largest gap and the F64 mean.
#[test]
fn deviation_helpers_locate_the_largest_gap_and_average_over_every_logit() {
    let expected = [0.0, 1.0, 2.0, 3.0];
    let actual = [0.5, 1.0, 0.0, 5.0];
    assert_eq!(largest_gap(&actual, &expected), (2, 2.0));
    assert_eq!(gap(&actual, &expected), 2.0);
    assert_eq!(mean_gap(&actual, &expected), 4.5 / 4.0);
    assert_eq!(largest_gap(&expected, &expected), (0, 0.0));
    assert_eq!(mean_gap(&expected, &expected), 0.0);
}

/// Transformers' own full-model BF16 evaluation stays within the half-precision bounds of its F32 one.
///
/// This reproduces the calibration of [`HALF_TOLERANCE`] without a
/// checkpoint. It reads the official BF16 and F32 bundles of the full model,
/// checks that both come from the same configuration and prompts, and prints
/// each manifest's SHA-256, each prompt's largest deviation (with its
/// position and vocabulary entry) and mean deviation, and both bundles' first
/// greedy margin. No Burn model runs, so a pass qualifies no Burn run.
#[test]
#[ignore = "needs TRANSFORMERS_GPT_OSS_BF16_REFERENCE_DIR and _F32_REFERENCE_DIR (full bundles)"]
fn transformers_bf16_reference_stays_within_the_half_precision_bounds() {
    let bundles = [
        directory("TRANSFORMERS_GPT_OSS_BF16_REFERENCE_DIR"),
        directory("TRANSFORMERS_GPT_OSS_F32_REFERENCE_DIR"),
    ];
    for bundle in &bundles {
        let manifest = std::fs::read(bundle.join("reference.json")).unwrap();
        eprintln!(
            "{}: reference.json SHA-256 {}",
            bundle.display(),
            hex(&Sha256::digest(&manifest))
        );
    }
    let [half, full] = bundles.map(|bundle| Reference::read(&bundle));
    assert_eq!(bundle_dtype(&half.dtype), TensorDType::BF16, "first bundle");
    assert_eq!(bundle_dtype(&full.dtype), TensorDType::F32, "second bundle");
    assert_eq!(
        (half.layers, full.layers),
        (None, None),
        "both bundles must evaluate the full model"
    );
    assert_eq!(
        half.config_sha256, full.config_sha256,
        "the bundles were generated from different config.json files"
    );
    assert_eq!(half.vocabulary, full.vocabulary);
    assert_eq!(half.prompts.len(), full.prompts.len());
    for (index, (rounded, exact)) in half.prompts.iter().zip(&full.prompts).enumerate() {
        assert_eq!(rounded.tokens, exact.tokens, "prompt {index}: token IDs");
        assert!(
            rounded
                .logits
                .iter()
                .chain(&exact.logits)
                .all(|logit| logit.is_finite()),
            "prompt {index}: non-finite logit"
        );
        let (at, largest) = largest_gap(&rounded.logits, &exact.logits);
        let mean = mean_gap(&rounded.logits, &exact.logits);
        eprintln!(
            "prompt {index}: {} tokens, max |Δ| = {largest:e} at position {}, token {}, mean |Δ| = {mean:e}",
            exact.tokens.len(),
            at / full.vocabulary,
            at % full.vocabulary
        );
        assert!(
            largest <= HALF_TOLERANCE,
            "prompt {index}: max |Δ| = {largest:e}"
        );
        assert!(
            mean <= f64::from(HALF_MEAN_TOLERANCE),
            "prompt {index}: mean |Δ| = {mean:e}"
        );
    }
    if let (Some(rounded), Some(exact)) = (half.greedy.first(), full.greedy.first()) {
        eprintln!(
            "greedy step 0: token {} margin {:e} (BF16), token {} margin {:e} (F32)",
            rounded.0, rounded.1, exact.0, exact.1
        );
    }
}

/// Flex: every prompt position's logits.
#[test]
#[ignore = "needs TRANSFORMERS_GPT_OSS_DIR, _REFERENCE_DIR, and 9-80 GiB of memory"]
fn flex_prompt_logits_match_the_reference() {
    check_prompt_logits::<Flex<f32, i32>>(&Default::default());
}

/// Flex: the cached greedy continuation.
#[test]
#[ignore = "needs TRANSFORMERS_GPT_OSS_DIR, _REFERENCE_DIR, and 9-80 GiB of memory"]
fn flex_cached_greedy_matches_the_reference() {
    check_cached_greedy::<Flex<f32, i32>>(&Default::default());
}

/// Flex: a ragged left-padded batch of every prompt.
#[test]
#[ignore = "needs TRANSFORMERS_GPT_OSS_DIR, _REFERENCE_DIR, and 9-80 GiB of memory"]
fn flex_batched_prefill_matches_the_reference() {
    check_batched_prefill::<Flex<f32, i32>>(&Default::default());
}

/// Flex: the longest prompt decoded token by token, past the window.
#[test]
#[ignore = "needs TRANSFORMERS_GPT_OSS_DIR, _REFERENCE_DIR, and 9-80 GiB of memory"]
fn flex_cached_long_prompt_matches_the_reference() {
    check_cached_long_prompt::<Flex<f32, i32>>(&Default::default());
}

#[cfg(all(feature = "metal", target_os = "macos"))]
mod metal {
    //! Metal counterparts of the Flex checks, run serially by the integrator.
    //!
    //! The stored dtype follows the bundle unless `TRANSFORMERS_GPT_OSS_DTYPE`
    //! is set; the F16 profile sets `f16` and is best qualified against a
    //! truncated F32 bundle (see this file's module documentation).

    use burn::backend::Metal;

    /// Metal: every prompt position's logits.
    #[test]
    #[ignore = "needs Metal, TRANSFORMERS_GPT_OSS_DIR, and _REFERENCE_DIR"]
    fn metal_prompt_logits_match_the_reference() {
        super::check_prompt_logits::<Metal>(&Default::default());
    }

    /// Metal: the cached greedy continuation.
    #[test]
    #[ignore = "needs Metal, TRANSFORMERS_GPT_OSS_DIR, and _REFERENCE_DIR"]
    fn metal_cached_greedy_matches_the_reference() {
        super::check_cached_greedy::<Metal>(&Default::default());
    }

    /// Metal: the longest prompt decoded token by token, past the window.
    #[test]
    #[ignore = "needs Metal, TRANSFORMERS_GPT_OSS_DIR, and _REFERENCE_DIR"]
    fn metal_cached_long_prompt_matches_the_reference() {
        super::check_cached_long_prompt::<Metal>(&Default::default());
    }

    /// Metal: a ragged left-padded batch of every prompt.
    #[test]
    #[ignore = "needs Metal, TRANSFORMERS_GPT_OSS_DIR, and _REFERENCE_DIR"]
    fn metal_batched_prefill_matches_the_reference() {
        super::check_batched_prefill::<Metal>(&Default::default());
    }
}

#[cfg(feature = "cuda")]
mod cuda {
    //! CUDA counterparts of the Flex checks, run serially on the GPU host.
    //!
    //! The stored dtype follows the bundle unless `TRANSFORMERS_GPT_OSS_DTYPE`
    //! is set. An F32 run on a GPU that rounds F32 matrix inputs to TF32 needs
    //! explicit tolerances; see the [module documentation](super#cuda).

    use burn::backend::Cuda;

    /// CUDA: every prompt position's logits.
    #[test]
    #[ignore = "needs CUDA, TRANSFORMERS_GPT_OSS_DIR, and _REFERENCE_DIR"]
    fn cuda_prompt_logits_match_the_reference() {
        super::check_prompt_logits::<Cuda>(&Default::default());
    }

    /// CUDA: the cached greedy continuation.
    #[test]
    #[ignore = "needs CUDA, TRANSFORMERS_GPT_OSS_DIR, and _REFERENCE_DIR"]
    fn cuda_cached_greedy_matches_the_reference() {
        super::check_cached_greedy::<Cuda>(&Default::default());
    }

    /// CUDA: the longest prompt decoded token by token, past the window.
    #[test]
    #[ignore = "needs CUDA, TRANSFORMERS_GPT_OSS_DIR, and _REFERENCE_DIR"]
    fn cuda_cached_long_prompt_matches_the_reference() {
        super::check_cached_long_prompt::<Cuda>(&Default::default());
    }

    /// CUDA: a ragged left-padded batch of every prompt.
    #[test]
    #[ignore = "needs CUDA, TRANSFORMERS_GPT_OSS_DIR, and _REFERENCE_DIR"]
    fn cuda_batched_prefill_matches_the_reference() {
        super::check_batched_prefill::<Cuda>(&Default::default());
    }
}
