//! Accelerator-versus-Flex parity of every executable family with identical weights.
//!
//! Each case builds a tiny model on deterministic Flex/F32 from a seeded
//! random initialization, constructs the same configuration on an accelerator,
//! and installs Flex's parameters into it with [`copy_parameters`], so both
//! sides evaluate one function with bit-identical weights. The suite then
//! compares, at every valid position,
//!
//! 1. **stateless** full-vocabulary logits of a ragged two-row batch: row 0
//!    fills all [`COLUMNS`] columns and row 1 is left-padded by [`PADDING`]
//!    masked columns;
//! 2. **unpadded** stateless logits of two full rows without a mask;
//! 3. **cached** logits: a batched prefill of the ragged batch's first
//!    [`PREFILL`] columns (rows of 5 and 2 valid tokens), then one-token
//!    decode steps through the remaining columns, against Flex's own cached
//!    run;
//! 4. **self-consistency** on the accelerator: each cached step against the
//!    accelerator's own stateless logits at the same column, after which the
//!    cached lengths must count valid tokens only.
//!
//! Encoder-decoder models (T5 and the modular seq2seq model) encode a
//! left-padded source and compare encoder states at valid positions,
//! teacher-forced decoder logits, and a cached decoder prefill plus decode
//! steps. GPT-2 and GPT-NeoX have no decode cache and run the stateless checks
//! only. Families with an attention-precision switch run under both their
//! default (backend-fused) attention and the eager F32-softmax graph.
//!
//! The cases cover every family the registries execute (asserted by
//! [`every_registered_family_has_a_parity_case`]): the seven Llama-family
//! tiny-golden configurations (Llama 3, biased and tied Llama, Qwen2, Qwen3,
//! sliding-window Mistral, NoPE-interleaved SmolLM3, OLMo 2), Gemma 2 and
//! Gemma 3 text, gpt-oss, Qwen3.5 (gated DeltaNet plus full attention),
//! Qwen3-MoE, DeepSeek-V2 (multi-head latent attention with shared and routed
//! experts), Muse Glimmer (sliding and full attention), Nemotron-H (Mamba-2,
//! attention, dense, and routed blocks), GPT-2, GPT-NeoX, T5, a modular
//! encoder-decoder, and two modular causal decoders: one with a layer of each
//! mixer kind (RoPE attention, MLA, differential attention, gated DeltaNet
//! with routed experts, Mamba-2) and one whose every mixer and router has a
//! single head or a single selected expert. Single-head axes are the unit axes
//! that Burn's CubeCL reshape has mishandled (see `trailing_unit_axis` in
//! `src/delta_net.rs`). Sliding windows (3 or 4) lie below the cache capacity
//! [`CAPACITY`], so their ring caches wrap during decoding. A kernel-level
//! case compares the gated delta rule's recurrent, snapshot, and chunked
//! kernels with one and two heads, with masked and with strided inputs.
//!
//! # Running
//!
//! The Flex-against-Flex run [`every_family_matches_itself_on_flex`] is an
//! ordinary test: it proves the copy is exact (stateless and cached logits
//! agree bit for bit) and exercises every comparison on CPU. The accelerator
//! tests are ignored, gated by backend feature, and print each case's
//! measured deviations:
//!
//! ```text
//! cargo test --locked --features metal --test backend_parity -- --ignored --nocapture --test-threads 1
//! cargo test --locked --features cuda --test backend_parity -- --ignored --nocapture --test-threads 1
//! ```
//!
//! With the `cpu` feature, the `cpu` module runs the same cases on Burn's
//! CubeCL CPU backend, which compiles the GPU backends' kernels to host code.
//! Its delta-rule case, `cpu::cpu_delta_rule_kernels_match_flex`, is an
//! ordinary test: the backend shares the GPU backends' reshape, so it checks
//! the DeltaNet reshape workaround without a GPU (it fails when
//! `trailing_unit_axis` is reverted to `unsqueeze_dim(3)`). The family cases
//! are ignored for their run time, and two of them for a CubeCL CPU copy
//! defect of Burn 0.21; the module documents both and the command that runs
//! every passing case:
//!
//! ```text
//! cargo test --locked --features cpu --test backend_parity cpu::
//! ```
//!
//! Build and run the CUDA tests on the GPU host with the CUDA toolkit
//! available: `cudarc` reads `nvcc --version` at build time, and CubeCL
//! compiles every kernel at run time with NVRTC against
//! `$CUDA_PATH/include` (default `/usr/local/cuda`; after loading a toolkit
//! module that sets only `CUDA_HOME`, export `CUDA_PATH=$CUDA_HOME`). Burn's
//! default `Cuda` device is ordinal 0, so `CUDA_VISIBLE_DEVICES` selects the
//! GPU. Each CUDA test first probes how the device multiplies F32 matrices
//! ([`MatmulPrecision::probe`]) and prints the result with the bound it
//! selects.
//!
//! # Tolerance
//!
//! Every comparison passes when `|accelerator - flex| <= absolute + relative ·
//! scale`, where `scale` is the largest Flex magnitude of the compared set
//! ([`Bound`]). The tiny models' logits have magnitudes between about 0.2
//! and 38.
//!
//! **F32 accumulation order** ([`F32_BOUND`]: Metal, the CubeCL CPU backend,
//! and CUDA GPUs below compute capability 8.0). Both sides compute every operation in F32 but
//! reduce in different orders (Flex's blocked GEMM versus a GPU's tiled one,
//! another softmax and norm reduction tree, other `exp` and `tanh`
//! implementations), so each output carries a few units in the last place of
//! its accumulated terms: a relative error near `n · 2^-24` for a reduction of
//! `n` terms (`2^-24 ≈ 6.0e-8`). Through at most five blocks of width 8 to 32
//! that stays near `1e-6` of the logit scale; "Measured" below records Metal.
//! The bound is `2e-5` absolute plus `2e-5` of the scale: its relative part
//! is about seventeen times the largest relative Metal deviation (`1.2e-6`),
//! and its absolute floor alone twice the largest absolute one (`9.5e-6` at
//! scale `25.6`), leaving room for other GPUs' reduction orders while staying
//! three orders of magnitude below any real kernel fault (a wrong
//! mask, cache slot, ring offset, rotary layout, or routing decision moves
//! logits by `1e-2` of their scale or more; the DeltaNet reshape defect moved
//! them by `0.5`).
//!
//! **TF32 tensor cores** (`TF32_BOUND`: CUDA GPUs of compute capability 8.0 or
//! newer). Burn 0.21 multiplies F32 matrices there with TF32 inputs; the
//! support module documents why, and derives the bound (`1e-2` absolute plus
//! `1e-2` of the scale) from TF32 rounding. The derivation's measured half is
//! [`tf32_weight_rounding_stays_within_a_quarter_of_the_tf32_bound`]: rounding
//! only the weights to TF32 on Flex moves each family's logits by the printed
//! fraction of their scale, which the test bounds by a quarter of the TF32
//! bound. A CUDA run on such a GPU qualifies execution (masks, caches, rings,
//! routing, and kernel dispatch), not F32 bit-level numerics.
//!
//! Mixture-of-experts layers route by comparing router scores, so a
//! deviation could flip an expert choice only where two scores tie within
//! it; a flip would move the affected logits by far more than either bound
//! and fail the test rather than pass silently.
//!
//! # Measured
//!
//! On Metal (Apple M3 Max, Burn 0.21.0, macOS 27), every case passes
//! [`F32_BOUND`]. The largest absolute deviations over the four comparisons
//! are `9.5e-6` for Gemma 2 (logit scale `25.6`), `7.6e-6` for Gemma 3 text,
//! SmolLM3, and the tied Llama (scales `35` to `38`), `1.4e-6` for T5, `1.2e-6`
//! for OLMo 2 and Muse Glimmer, and at most `4.2e-7` for every other family,
//! modular decoder, and attention mode; the largest relative deviation is
//! `1.2e-6` (Muse Glimmer, scale `0.19`). The gated delta rule's kernels
//! deviate by at most `6e-8`. CUDA has not been measured yet: record the first
//! run's per-case output, and the probed matrix precision, in
//! `RELEASE_GATES.md`.
//!
//! On the CubeCL CPU backend (the same machine, debug build), every case but
//! the two the `cpu` module ignores for an upstream defect passes
//! [`F32_BOUND`]: the largest absolute deviations are `6.7e-6` for SmolLM3
//! and Gemma 3 text (scales `37.5` and `37.9`), `5.7e-6` for Gemma 2 and the
//! tied Llama, `1.0e-6` for OLMo 2, `9.5e-7` for T5, and at most `4.8e-7` for
//! every other family and attention mode; the largest relative deviation is
//! `9.2e-7` (OLMo 2, scale `1.12`), and the gated delta rule's kernels deviate
//! by at most `6e-8`.

#![cfg(feature = "flex")]

#[path = "support/backend_numerics.rs"]
mod backend_numerics;

use std::sync::{Mutex, PoisonError};

use backend_numerics::{
    Bound, Gap, GreedyComparison, GreedyDecision, MatmulPrecision, TF32_BOUND, argmax,
    check_teacher_forced_greedy, checkpoint_tolerance, compare_teacher_forced_greedy,
    copy_parameters, greedy_step_is_decisive, host, identity, resolve_checkpoint_tolerance,
    round_to_tf32, top_two_margin,
};
use burn::{
    backend::Flex,
    nn::LinearConfig,
    tensor::{Bool, Tensor, TensorData, backend::Backend},
};
use transformers::{
    ArchitectureConfig, AutoCausalConfig, AutoCausalLm, AutoSeq2SeqConfig, CausalLanguageModel,
    CausalLmInput, CausalLmOutput, DecodeInput, DecodeState, ForwardOptions, LogitsSelection,
    Result, Seq2SeqInput, Seq2SeqLanguageModel, TokenMask,
    architecture::{
        Activation, AttentionOptions, AttentionSpec, ClampedSwiGluSpec, DecoderLayerSpec,
        DecoderSpec, DenseFeedForwardSpec, DifferentialAttentionSpec, FeedForwardSpec, Mamba2Spec,
        MixerSpec, MixtureOfExpertsSpec, MlaSpec, NormPlacement, NormalizationSpec,
        PositionEncodingSpec, ResidualTopology, RopeScaling, ScoreScale, Seq2SeqSpec,
    },
    attention::{AttentionConnectivity, AttentionPrecisionMode, depth_lambda_init},
    cache::CacheConfig,
    delta_net::{
        DeltaRuleConfig, DeltaRuleInput, chunked_gated_delta_rule, recurrent_gated_delta_rule,
        recurrent_gated_delta_rule_with_snapshots,
    },
    models::{
        config::TokenIds,
        custom::{ModularCausalLm, ModularSeq2SeqLm},
        deepseek_v2::{
            DeepseekScoringFunction, DeepseekTopKMethod, DeepseekV2Config, DeepseekV2ForCausalLm,
        },
        gpt_neox::{GptNeoXConfig, GptNeoXForCausalLm},
        gpt2::{Gpt2Config, Gpt2ForCausalLm},
        muse_glimmer::{MuseGlimmerForCausalLm, MuseGlimmerLayerType, MuseGlimmerTextConfig},
        nemotron_h::{NemotronHBlockType, NemotronHConfig, NemotronHForCausalLm},
        qwen3_5::{Qwen3_5ForCausalLm, Qwen3_5LayerType, Qwen3_5TextConfig},
        qwen3_moe::{Qwen3MoeConfig, Qwen3MoeForCausalLm},
        t5::{T5Config, T5FeedForwardProjection, T5ForConditionalGeneration},
    },
    moe::RoutingPolicy,
};

/// Deterministic F32 CPU reference backend.
type Reference = Flex<f32, i32>;

/// Columns of every causal batch.
const COLUMNS: usize = 10;

/// Masked leading columns of the ragged batch's second row.
const PADDING: usize = 3;

/// Columns prefilled before cached one-token decoding.
const PREFILL: usize = 5;

/// Decode-cache capacity: above every sliding window, so ring caches wrap.
const CAPACITY: usize = 16;

/// Source columns of every encoder-decoder batch.
const SOURCE_COLUMNS: usize = 8;

/// Target columns of every encoder-decoder batch.
const TARGET_COLUMNS: usize = 6;

/// Target columns prefilled before cached decoder steps.
const TARGET_PREFILL: usize = 2;

/// Seed of every Flex reference initialization.
const SEED: u64 = 20_261_016;

/// Serializes every draw from Flex's random number generator in this test binary.
///
/// Burn Flex keeps one process-wide RNG: `Backend::seed` stores it, and each
/// random initialization takes it, draws, and puts it back. Tests run on
/// parallel threads, so two tests seeding and initializing models at once
/// interleave their draws, and each model's weights, and every deviation
/// measured on them, then depend on thread scheduling. Under load that
/// moved Muse Glimmer's TF32 weight-rounding deviation from `1.30e-3` to
/// `4.23e-3`, past the quarter bound `2.89e-3` of
/// [`tf32_weight_rounding_stays_within_a_quarter_of_the_tf32_bound`]. Every
/// Flex parameter initialization here, seeded or not, therefore runs inside
/// [`seeded`], which holds this lock. Burn initializes a parameter lazily, on
/// its first access, so the closure must also make that access:
/// [`copy_parameters`] visits every parameter of both models, and a model
/// whose parameters are never read draws nothing.
static FLEX_RNG: Mutex<()> = Mutex::new(());

/// Seeds Flex's RNG with [`SEED`] and runs `build` while no other test of this binary can draw from it.
///
/// A poisoned lock (a test panicked while holding it, as the refused
/// parameter copies do) is recovered, because the guarded state is the RNG
/// itself, which every call reseeds.
fn seeded<T>(build: impl FnOnce() -> T) -> T {
    let _exclusive = FLEX_RNG.lock().unwrap_or_else(PoisonError::into_inner);
    Reference::seed(&Default::default(), SEED);
    build()
}

/// Bound for two F32 backends that differ only in reduction order; see the module documentation.
const F32_BOUND: Bound = Bound {
    absolute: 2e-5,
    relative: 2e-5,
};

/// A two-row token batch with its validity mask.
struct Batch {
    /// `[2, columns]` token IDs; masked positions hold token 0.
    tokens: Vec<Vec<i64>>,
    /// `[2, columns]` validity, true for a real token.
    valid: Vec<Vec<bool>>,
}

impl Batch {
    /// Two rows of `columns` tokens, the second left-padded by `padding`.
    ///
    /// Tokens `1 + (7 c + 13 r + 5) mod (V - 1)` avoid ID 0 and differ
    /// between rows and columns.
    fn new(columns: usize, padding: usize, vocabulary: usize) -> Self {
        let token =
            |row: usize, column: usize| 1 + ((7 * column + 13 * row + 5) % (vocabulary - 1)) as i64;
        let tokens = (0..2)
            .map(|row| {
                (0..columns)
                    .map(|column| {
                        if row == 1 && column < padding {
                            0
                        } else {
                            token(row, column)
                        }
                    })
                    .collect()
            })
            .collect();
        let valid = (0..2)
            .map(|row| {
                (0..columns)
                    .map(|column| row == 0 || column >= padding)
                    .collect()
            })
            .collect();
        Self { tokens, valid }
    }

    /// Number of columns.
    fn columns(&self) -> usize {
        self.tokens[0].len()
    }

    /// Whether any position is masked.
    fn padded(&self) -> bool {
        self.valid.iter().flatten().any(|&valid| !valid)
    }

    /// Columns `range` of both rows as a model input, masked only when padded.
    fn input<B: Backend>(
        &self,
        range: std::ops::Range<usize>,
        device: &B::Device,
    ) -> CausalLmInput<B> {
        let tokens: Vec<i64> = self
            .tokens
            .iter()
            .flat_map(|row| row[range.clone()].iter().copied())
            .collect();
        let valid: Vec<bool> = self
            .valid
            .iter()
            .flat_map(|row| row[range.clone()].iter().copied())
            .collect();
        let shape = [2, range.len()];
        let token_mask = self.padded().then(|| {
            TokenMask::new(Tensor::<B, 2, Bool>::from_data(
                TensorData::new(valid, shape),
                device,
            ))
        });
        CausalLmInput {
            token_ids: Tensor::from_data(TensorData::new(tokens, shape), device),
            token_mask,
            position_ids: None,
        }
    }

    /// Column `column` of both rows as one unmasked decode step.
    fn step<B: Backend>(&self, column: usize, device: &B::Device) -> DecodeInput<B> {
        assert!(
            self.valid.iter().all(|row| row[column]),
            "decode steps are valid"
        );
        let tokens = vec![self.tokens[0][column], self.tokens[1][column]];
        DecodeInput {
            token_ids: Tensor::from_data(TensorData::new(tokens, [2, 1]), device),
            position_ids: None,
            token_mask: None,
        }
    }

    /// Number of valid tokens of each row.
    fn lengths(&self) -> Vec<usize> {
        self.valid
            .iter()
            .map(|row| row.iter().filter(|&&valid| valid).count())
            .collect()
    }

    /// Compares `[2, columns, width]` values at every valid position.
    fn compare_valid(&self, actual: &[f32], reference: &[f32], width: usize) -> Gap {
        let columns = self.columns();
        assert_eq!(actual.len(), 2 * columns * width, "compared shape");
        let mut gap = Gap::default();
        for (row, valid) in self.valid.iter().enumerate() {
            for column in (0..columns).filter(|&column| valid[column]) {
                let at = (row * columns + column) * width..(row * columns + column + 1) * width;
                gap = gap.max(Gap::measure(&actual[at.clone()], &reference[at]));
            }
        }
        gap
    }

    /// Returns `[2, columns, width]` values of `column` as `[2, width]`.
    fn column(&self, values: &[f32], column: usize, width: usize) -> Vec<f32> {
        let columns = self.columns();
        (0..2)
            .flat_map(|row| {
                let start = (row * columns + column) * width;
                values[start..start + width].iter().copied()
            })
            .collect()
    }
}

/// Logit-selection options requesting every position.
fn all_positions() -> ForwardOptions {
    ForwardOptions {
        logits: LogitsSelection::All,
        ..ForwardOptions::default()
    }
}

/// Measured deviations of one model on one accelerator.
#[derive(Clone, Copy, Debug, Default)]
struct Report {
    /// Ragged stateless logits against Flex (encoder states for seq2seq models).
    stateless: Gap,
    /// Unpadded stateless logits against Flex (teacher-forced logits for seq2seq models).
    unpadded: Gap,
    /// Cached prefill and decode logits against Flex's cached run.
    cached: Gap,
    /// Cached logits against the accelerator's own stateless logits.
    self_consistency: Gap,
}

impl Report {
    /// Asserts every measured gap lies within `bound` and prints the measurement.
    #[track_caller]
    fn check(self, case: &str, bound: Bound) {
        eprintln!(
            "{case}: stateless {:.2e}, unpadded {:.2e}, cached {:.2e}, cached vs stateless {:.2e} \
             (largest relative {:.2e}, logit scale {:.2})",
            self.stateless.deviation,
            self.unpadded.deviation,
            self.cached.deviation,
            self.self_consistency.deviation,
            self.largest_relative(),
            self.scale(),
        );
        bound.check(self.stateless, &format!("{case}: stateless"));
        bound.check(self.unpadded, &format!("{case}: unpadded"));
        bound.check(self.cached, &format!("{case}: cached"));
        bound.check(
            self.self_consistency,
            &format!("{case}: cached vs stateless"),
        );
    }

    /// The four comparisons in declaration order.
    fn gaps(self) -> [Gap; 4] {
        [
            self.stateless,
            self.unpadded,
            self.cached,
            self.self_consistency,
        ]
    }

    /// The largest deviation relative to its reference scale over every comparison.
    fn largest_relative(self) -> f32 {
        self.gaps()
            .map(Gap::relative)
            .into_iter()
            .fold(0.0, f32::max)
    }

    /// The largest reference magnitude over every comparison.
    fn scale(self) -> f32 {
        self.gaps()
            .map(|gap| gap.scale)
            .into_iter()
            .fold(0.0, f32::max)
    }

    /// Combines two reports into the largest measurement of each comparison.
    fn merge(self, other: Self) -> Self {
        Self {
            stateless: self.stateless.max(other.stateless),
            unpadded: self.unpadded.max(other.unpadded),
            cached: self.cached.max(other.cached),
            self_consistency: self.self_consistency.max(other.self_consistency),
        }
    }
}

/// Runs the ragged batch's cached prefill and decode steps; returns each step's `[2, V]` logits.
fn cached_steps<B: Backend, M: CausalLanguageModel<B>>(
    model: &M,
    batch: &Batch,
    device: &B::Device,
) -> Vec<Vec<f32>> {
    let mut state = model
        .new_decode_state_for(CacheConfig::new(2, CAPACITY).unwrap(), device)
        .unwrap();
    let options = ForwardOptions::default();
    let prefill = model
        .prefill(batch.input(0..PREFILL, device), &mut state, &options)
        .unwrap();
    let mut steps = vec![host(prefill.logits)];
    for column in PREFILL..batch.columns() {
        let output = model
            .decode(batch.step(column, device), &mut state, &options)
            .unwrap();
        steps.push(host(output.logits));
    }
    assert_eq!(
        DecodeState::lengths(&state),
        batch.lengths(),
        "cached lengths count valid tokens only"
    );
    steps
}

/// Compares a causal model with decode state on the accelerator against its Flex original.
fn compare_causal<A, M, N>(
    reference: &M,
    accelerated: &N,
    device: &A::Device,
    vocabulary: usize,
) -> Report
where
    A: Backend,
    M: CausalLanguageModel<Reference>,
    N: CausalLanguageModel<A>,
{
    let mut report = compare_stateless(
        |input, options| reference.forward(input, options),
        |input, options| accelerated.forward(input, options),
        device,
        vocabulary,
    );
    let ragged = Batch::new(COLUMNS, PADDING, vocabulary);
    let stateless = host(
        accelerated
            .forward(ragged.input(0..COLUMNS, device), &all_positions())
            .unwrap()
            .logits,
    );
    let flex = cached_steps(reference, &ragged, &Default::default());
    let steps = cached_steps(accelerated, &ragged, device);
    for (step, (actual, expected)) in steps.iter().zip(&flex).enumerate() {
        report.cached = report.cached.max(Gap::measure(actual, expected));
        let column = PREFILL - 1 + step;
        let own = ragged.column(&stateless, column, vocabulary);
        report.self_consistency = report.self_consistency.max(Gap::measure(actual, &own));
    }
    report
}

/// Compares ragged and unpadded stateless logits of two forward functions.
fn compare_stateless<A: Backend>(
    reference: impl Fn(CausalLmInput<Reference>, &ForwardOptions) -> Result<CausalLmOutput<Reference>>,
    accelerated: impl Fn(CausalLmInput<A>, &ForwardOptions) -> Result<CausalLmOutput<A>>,
    device: &A::Device,
    vocabulary: usize,
) -> Report {
    let flex_device = Default::default();
    let mut report = Report::default();
    for (padding, gap) in [(PADDING, &mut report.stateless), (0, &mut report.unpadded)] {
        let batch = Batch::new(COLUMNS, padding, vocabulary);
        let expected = host(
            reference(batch.input(0..COLUMNS, &flex_device), &all_positions())
                .unwrap()
                .logits,
        );
        let actual = host(
            accelerated(batch.input(0..COLUMNS, device), &all_positions())
                .unwrap()
                .logits,
        );
        *gap = batch.compare_valid(&actual, &expected, vocabulary);
    }
    report
}

/// Encoder states, teacher-forced logits, and cached step logits of one seq2seq run.
struct Seq2SeqRun {
    /// `[2, SOURCE_COLUMNS, d]` final encoder states.
    encoder: Vec<f32>,
    /// Encoder width `d`.
    width: usize,
    /// `[2, TARGET_COLUMNS, V]` stateless decoder logits.
    logits: Vec<f32>,
    /// `[2, V]` logits of the cached prefill and each decode step.
    steps: Vec<Vec<f32>>,
}

impl Seq2SeqRun {
    /// Evaluates every compared quantity of `model` on `device`.
    ///
    /// The cached run encodes the source once, prefills [`TARGET_PREFILL`]
    /// target tokens, and decodes the rest one token at a time.
    fn new<B: Backend, M: Seq2SeqLanguageModel<B>>(
        model: &M,
        source: &Batch,
        target: &Batch,
        device: &B::Device,
    ) -> Self {
        let output = model
            .forward(
                Seq2SeqInput {
                    encoder: source.input(0..SOURCE_COLUMNS, device),
                    decoder: target.input(0..TARGET_COLUMNS, device),
                },
                &all_positions(),
            )
            .unwrap();
        let width = output.encoder.hidden_state.dims()[2];
        let encoder = host(output.encoder.hidden_state);
        let logits = host(output.decoder.logits);
        let options = ForwardOptions::default();
        let encoded = model
            .encode(source.input(0..SOURCE_COLUMNS, device), &options)
            .unwrap();
        let mut state = model
            .new_decode_state_for(CacheConfig::new(2, CAPACITY).unwrap(), device)
            .unwrap();
        let prefill = model
            .prefill_decoder(
                &encoded,
                target.input(0..TARGET_PREFILL, device),
                &mut state,
                &options,
            )
            .unwrap();
        let mut steps = vec![host(prefill.decoder.logits)];
        for column in TARGET_PREFILL..TARGET_COLUMNS {
            let output = model
                .decode(&encoded, target.step(column, device), &mut state, &options)
                .unwrap();
            steps.push(host(output.decoder.logits));
        }
        assert_eq!(
            DecodeState::lengths(&state),
            [TARGET_COLUMNS, TARGET_COLUMNS]
        );
        Self {
            encoder,
            width,
            logits,
            steps,
        }
    }
}

/// Compares an encoder-decoder model on the accelerator against its Flex original.
///
/// The source is left-padded; the teacher-forced target is unpadded. The
/// report's stateless gap covers encoder states at valid source positions,
/// its unpadded gap every teacher-forced decoder logit, and its cached gaps
/// the cached decoder run of [`Seq2SeqRun::new`].
fn compare_seq2seq<A, M, N>(
    reference: &M,
    accelerated: &N,
    device: &A::Device,
    vocabulary: usize,
) -> Report
where
    A: Backend,
    M: Seq2SeqLanguageModel<Reference>,
    N: Seq2SeqLanguageModel<A>,
{
    let source = Batch::new(SOURCE_COLUMNS, PADDING, vocabulary);
    let target = Batch::new(TARGET_COLUMNS, 0, vocabulary);
    let flex = Seq2SeqRun::new(reference, &source, &target, &Default::default());
    let actual = Seq2SeqRun::new(accelerated, &source, &target, device);
    let mut report = Report {
        stateless: source.compare_valid(&actual.encoder, &flex.encoder, flex.width),
        unpadded: target.compare_valid(&actual.logits, &flex.logits, vocabulary),
        ..Report::default()
    };
    for (step, (steps, expected)) in actual.steps.iter().zip(&flex.steps).enumerate() {
        report.cached = report.cached.max(Gap::measure(steps, expected));
        let own = target.column(&actual.logits, TARGET_PREFILL - 1 + step, vocabulary);
        report.self_consistency = report.self_consistency.max(Gap::measure(steps, &own));
    }
    report
}

/// Returns `model` with eager F32-softmax attention when its family has the switch.
fn eager<B: Backend>(model: AutoCausalLm<B>) -> Option<AutoCausalLm<B>> {
    let mode = AttentionPrecisionMode::EagerFp32Softmax;
    match model {
        AutoCausalLm::Llama(model) => {
            Some(AutoCausalLm::Llama(model.with_attention_precision(mode)))
        }
        AutoCausalLm::Gemma(model) => {
            Some(AutoCausalLm::Gemma(model.with_attention_precision(mode)))
        }
        AutoCausalLm::GptOss(model) => {
            Some(AutoCausalLm::GptOss(model.with_attention_precision(mode)))
        }
        _ => None,
    }
}

/// The tiny configuration of one registry case.
enum RegistryConfig {
    /// A committed tiny-golden configuration, loaded through the registry.
    Json(
        /// The fixture's `config.json` text.
        &'static str,
    ),
    /// Qwen3.5 gated-DeltaNet and full-attention hybrid.
    Qwen3_5(
        /// The text configuration.
        Qwen3_5TextConfig,
    ),
    /// Qwen3 sparse-expert decoder.
    Qwen3Moe(
        /// The model configuration.
        Qwen3MoeConfig,
    ),
    /// DeepSeek-V2 latent attention with shared and routed experts.
    DeepseekV2(
        /// The model configuration.
        DeepseekV2Config,
    ),
    /// Muse Glimmer sliding/full hybrid.
    MuseGlimmer(
        /// The text configuration.
        MuseGlimmerTextConfig,
    ),
    /// Nemotron-H Mamba-2, attention, dense, and routed hybrid.
    NemotronH(
        /// The model configuration.
        NemotronHConfig,
    ),
}

impl RegistryConfig {
    /// Builds a randomly initialized registry model on `device`.
    fn build<B: Backend>(&self, device: &B::Device) -> AutoCausalLm<B> {
        match self {
            Self::Json(json) => AutoCausalLm::from_config_json(json, device).unwrap(),
            Self::Qwen3_5(config) => {
                AutoCausalLm::from(Qwen3_5ForCausalLm::init(config.clone(), device).unwrap())
            }
            Self::Qwen3Moe(config) => {
                AutoCausalLm::from(Qwen3MoeForCausalLm::init(config.clone(), device).unwrap())
            }
            Self::DeepseekV2(config) => {
                AutoCausalLm::from(DeepseekV2ForCausalLm::init(config.clone(), device).unwrap())
            }
            Self::MuseGlimmer(config) => {
                AutoCausalLm::from(MuseGlimmerForCausalLm::init(config.clone(), device).unwrap())
            }
            Self::NemotronH(config) => {
                AutoCausalLm::from(NemotronHForCausalLm::init(config.clone(), device).unwrap())
            }
        }
    }
}

/// Every Llama-family tiny-golden configuration.
const LLAMA_FAMILY: [(&str, &str); 7] = [
    (
        "Llama 3",
        include_str!("fixtures/tiny_goldens/llama3/config.json"),
    ),
    (
        "Llama with bias and tied embeddings",
        include_str!("fixtures/tiny_goldens/llama-bias-tied/config.json"),
    ),
    (
        "Qwen2",
        include_str!("fixtures/tiny_goldens/qwen2/config.json"),
    ),
    (
        "Qwen3",
        include_str!("fixtures/tiny_goldens/qwen3/config.json"),
    ),
    (
        "Mistral sliding window",
        include_str!("fixtures/tiny_goldens/mistral-sliding/config.json"),
    ),
    (
        "SmolLM3 NoPE layers",
        include_str!("fixtures/tiny_goldens/smollm3/config.json"),
    ),
    (
        "OLMo 2",
        include_str!("fixtures/tiny_goldens/olmo2/config.json"),
    ),
];

/// Gemma 2 and Gemma 3 text tiny-golden configurations.
const GEMMA: [(&str, &str); 2] = [
    (
        "Gemma 2",
        include_str!("fixtures/tiny_goldens/gemma2/config.json"),
    ),
    (
        "Gemma 3 text",
        include_str!("fixtures/tiny_goldens/gemma3-text/config.json"),
    ),
];

/// The gpt-oss tiny-golden configuration.
const GPT_OSS: (&str, &str) = (
    "gpt-oss",
    include_str!("fixtures/tiny_goldens/gpt-oss/config.json"),
);

/// Qwen3.5 with three gated-DeltaNet layers and one full-attention layer.
fn qwen3_5() -> Qwen3_5TextConfig {
    Qwen3_5TextConfig {
        vocab_size: 32,
        hidden_size: 8,
        intermediate_size: 16,
        num_hidden_layers: 4,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: 4,
        max_position_embeddings: 64,
        partial_rotary_factor: Some(0.5),
        layer_types: vec![
            Qwen3_5LayerType::LinearAttention,
            Qwen3_5LayerType::LinearAttention,
            Qwen3_5LayerType::LinearAttention,
            Qwen3_5LayerType::FullAttention,
        ],
        linear_conv_kernel_dim: 2,
        linear_key_head_dim: 2,
        linear_value_head_dim: 2,
        linear_num_key_heads: 2,
        linear_num_value_heads: 2,
        mtp_num_hidden_layers: 0,
        ..Qwen3_5TextConfig::default()
    }
}

/// Qwen3-MoE with two sparse layers of four experts, two selected.
fn qwen3_moe() -> Qwen3MoeConfig {
    Qwen3MoeConfig {
        vocab_size: 32,
        hidden_size: 8,
        intermediate_size: 16,
        num_hidden_layers: 2,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: 4,
        max_position_embeddings: 64,
        max_window_layers: 2,
        decoder_sparse_step: 1,
        moe_intermediate_size: 12,
        num_experts: 4,
        num_experts_per_tok: 2,
        ..Qwen3MoeConfig::default()
    }
}

/// DeepSeek-V2 with one dense and one sparse latent-attention layer.
fn deepseek_v2() -> DeepseekV2Config {
    DeepseekV2Config {
        vocab_size: 32,
        hidden_size: 8,
        intermediate_size: 12,
        moe_intermediate_size: 6,
        num_hidden_layers: 2,
        num_attention_heads: 2,
        num_key_value_heads: 2,
        n_shared_experts: 2,
        n_routed_experts: 4,
        ep_size: 1,
        routed_scaling_factor: 1.0,
        kv_lora_rank: 5,
        q_lora_rank: None,
        qk_nope_head_dim: 3,
        qk_rope_head_dim: 2,
        v_head_dim: 4,
        topk_method: DeepseekTopKMethod::Greedy,
        n_group: 1,
        topk_group: 1,
        num_experts_per_tok: 2,
        moe_layer_freq: 1,
        first_k_dense_replace: 1,
        norm_topk_prob: false,
        scoring_func: DeepseekScoringFunction::Softmax,
        max_position_embeddings: 32,
        tie_word_embeddings: false,
        rope_scaling: None,
        pad_token_id: None,
        bos_token_id: None,
        eos_token_id: None,
        ..DeepseekV2Config::default()
    }
}

/// Muse Glimmer with three window-3 sliding layers and one full layer.
fn muse_glimmer() -> MuseGlimmerTextConfig {
    MuseGlimmerTextConfig {
        vocab_size: 32,
        hidden_size: 8,
        intermediate_size: 12,
        num_hidden_layers: 4,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: 4,
        max_position_embeddings: 32,
        sliding_window: Some(3),
        bos_token_id: Some(TokenIds::One(1)),
        eos_token_id: Some(TokenIds::One(2)),
        layer_types: vec![
            MuseGlimmerLayerType::SlidingAttention,
            MuseGlimmerLayerType::SlidingAttention,
            MuseGlimmerLayerType::SlidingAttention,
            MuseGlimmerLayerType::FullAttention,
        ],
        ..MuseGlimmerTextConfig::default()
    }
}

/// Nemotron-H with one block of each kind: Mamba-2, attention, dense, routed.
fn nemotron_h() -> NemotronHConfig {
    NemotronHConfig {
        vocab_size: 32,
        hidden_size: 8,
        layers_block_type: vec![
            NemotronHBlockType::LinearAttention,
            NemotronHBlockType::FullAttention,
            NemotronHBlockType::Mlp,
            NemotronHBlockType::Moe,
        ],
        num_hidden_layers: Some(4),
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: 4,
        max_position_embeddings: 32,
        intermediate_size: 12,
        ssm_state_size: 3,
        mamba_num_heads: 4,
        mamba_head_dim: 2,
        n_groups: 2,
        conv_kernel: 3,
        chunk_size: 2,
        n_routed_experts: 4,
        n_shared_experts: 1,
        moe_shared_expert_intermediate_size: 8,
        moe_intermediate_size: 6,
        num_experts_per_tok: 2,
        n_group: 2,
        topk_group: 1,
        num_nextn_predict_layers: 0,
        ..NemotronHConfig::default()
    }
}

/// Every registry case other than the tiny-golden JSON configurations.
fn hybrid_registry_cases() -> [(&'static str, RegistryConfig); 5] {
    [
        ("Qwen3.5", RegistryConfig::Qwen3_5(qwen3_5())),
        ("Qwen3-MoE", RegistryConfig::Qwen3Moe(qwen3_moe())),
        ("DeepSeek-V2", RegistryConfig::DeepseekV2(deepseek_v2())),
        ("Muse Glimmer", RegistryConfig::MuseGlimmer(muse_glimmer())),
        ("Nemotron-H", RegistryConfig::NemotronH(nemotron_h())),
    ]
}

/// Wraps JSON configurations as registry cases.
fn json_cases<const N: usize>(
    cases: [(&'static str, &'static str); N],
) -> impl Iterator<Item = (&'static str, RegistryConfig)> {
    cases
        .into_iter()
        .map(|(name, json)| (name, RegistryConfig::Json(json)))
}

/// Every registry case: the tiny-golden configurations and the hybrid families.
fn registry_cases() -> impl Iterator<Item = (&'static str, RegistryConfig)> {
    json_cases(LLAMA_FAMILY)
        .chain(json_cases(GEMMA))
        .chain(json_cases([GPT_OSS]))
        .chain(hybrid_registry_cases())
}

/// Builds a seeded Flex model and its parameter-identical copy on the accelerator.
fn registry_pair<A: Backend>(
    config: &RegistryConfig,
    device: &A::Device,
) -> (AutoCausalLm<Reference>, AutoCausalLm<A>) {
    let flex_device = Default::default();
    let (reference, accelerated) = seeded(|| {
        let reference = config.build::<Reference>(&flex_device);
        let accelerated = copy_parameters(&reference, config.build::<A>(device), identity);
        (reference, accelerated)
    });
    (reference, accelerated)
}

/// Compares registry cases under default and, where available, eager attention.
fn check_registry<A: Backend>(
    cases: impl IntoIterator<Item = (&'static str, RegistryConfig)>,
    device: &A::Device,
    bound: Bound,
) -> Report {
    let mut largest = Report::default();
    for (name, config) in cases {
        let (reference, accelerated) = registry_pair::<A>(&config, device);
        let vocabulary = reference.config().vocabulary_size();
        let report = compare_causal(&reference, &accelerated, device, vocabulary);
        report.check(name, bound);
        largest = largest.merge(report);
        if let (Some(reference), Some(accelerated)) = (eager(reference), eager(accelerated)) {
            let report = compare_causal(&reference, &accelerated, device, vocabulary);
            report.check(&format!("{name} (eager attention)"), bound);
            largest = largest.merge(report);
        }
    }
    largest
}

/// Llama family, every tiny-golden variant, on the accelerator.
fn llama_family_matches_flex<A: Backend>(device: &A::Device, bound: Bound) -> Report {
    check_registry::<A>(json_cases(LLAMA_FAMILY), device, bound)
}

/// Gemma 2 and Gemma 3 text on the accelerator.
fn gemma_matches_flex<A: Backend>(device: &A::Device, bound: Bound) -> Report {
    check_registry::<A>(json_cases(GEMMA), device, bound)
}

/// gpt-oss on the accelerator.
fn gpt_oss_matches_flex<A: Backend>(device: &A::Device, bound: Bound) -> Report {
    check_registry::<A>(json_cases([GPT_OSS]), device, bound)
}

/// Qwen3.5, Qwen3-MoE, DeepSeek-V2, Muse Glimmer, and Nemotron-H on the accelerator.
fn hybrid_registry_families_match_flex<A: Backend>(device: &A::Device, bound: Bound) -> Report {
    check_registry::<A>(hybrid_registry_cases(), device, bound)
}

/// Standard scale-only normalization of the modular cases.
fn rms() -> NormalizationSpec {
    NormalizationSpec::Rms {
        epsilon: 1e-6,
        zero_centered: false,
    }
}

/// One pre-normalized additive modular layer with a gated SiLU MLP.
fn modular_layer(mixer: MixerSpec) -> DecoderLayerSpec {
    DecoderLayerSpec {
        mixer,
        feed_forward: FeedForwardSpec::Dense(DenseFeedForwardSpec {
            intermediate_size: 12,
            activation: Activation::Silu,
            bias: false,
        }),
        mixer_norm: rms(),
        feed_forward_norm: rms(),
        norm_placement: NormPlacement::Pre,
        residual: ResidualTopology::Add,
    }
}

/// Grouped-query attention with per-head Q/K norms.
fn modular_attention(
    query_heads: usize,
    head_dim: usize,
    position: PositionEncodingSpec,
) -> MixerSpec {
    MixerSpec::Attention(AttentionSpec {
        query_heads,
        key_value_heads: 1,
        head_dim,
        query_key_norm: true,
        output_gate: false,
        position,
    })
}

/// RoPE over the leading `rotary_dim` channels of every head.
fn rope(rotary_dim: usize) -> PositionEncodingSpec {
    PositionEncodingSpec::Rope {
        theta: 10_000.0,
        rotary_dim,
        scaling: RopeScaling::None,
    }
}

/// A width-8 modular stack of `layers` over a 32-token vocabulary.
fn modular_stack(layers: Vec<DecoderLayerSpec>) -> DecoderSpec {
    DecoderSpec {
        vocabulary_size: 32,
        hidden_size: 8,
        layers,
        final_norm: rms(),
        tie_word_embeddings: false,
        max_position_embeddings: 32,
        embedding_scale: None,
        final_logit_softcap: None,
    }
}

/// Latent attention with a low-rank query of `query_heads` heads.
fn modular_latent(query_heads: usize) -> MixerSpec {
    MixerSpec::Mla(MlaSpec {
        query_heads,
        content_head_dim: 4,
        rotary_head_dim: 4,
        value_head_dim: 4,
        key_value_lora_rank: 4,
        query_lora_rank: Some(4),
        position: rope(4),
        score_scale: ScoreScale::Auto,
        connectivity: AttentionConnectivity::Global,
    })
}

/// Routed experts with one shared expert, `experts_per_token` of `routed_experts` selected.
fn modular_experts(routed_experts: usize, experts_per_token: usize) -> FeedForwardSpec {
    FeedForwardSpec::MixtureOfExperts(MixtureOfExpertsSpec {
        routed_experts,
        experts_per_token,
        shared_experts: 1,
        expert_intermediate_size: 4,
        activation: Activation::Silu,
        renormalize_selected: true,
        routed_scale: 1.0,
        routing: RoutingPolicy::default(),
        shared_expert_gate: false,
        expert_bias: false,
    })
}

/// A modular decoder with one layer of every mixer kind and every feed-forward kind.
///
/// RoPE attention with a dense MLP; latent attention (low-rank query) with a
/// clamped SwiGLU; differential attention with an ungated GELU MLP; gated
/// DeltaNet with shared and routed experts; Mamba-2 with a dense MLP.
fn modular_decoder() -> DecoderSpec {
    let mut latent = modular_layer(modular_latent(2));
    latent.feed_forward = FeedForwardSpec::ClampedSwiGlu(ClampedSwiGluSpec {
        intermediate_size: 6,
        bias: true,
        limit: 0.5,
        alpha: 1.702,
        up_offset: 1.0,
    });
    let mut differential = modular_layer(MixerSpec::DifferentialAttention(
        DifferentialAttentionSpec {
            query_heads: 4,
            key_value_heads: 2,
            head_dim: 2,
            lambda_init: depth_lambda_init(2),
            position: rope(2),
            connectivity: AttentionConnectivity::Global,
            bias: false,
        },
    ));
    differential.feed_forward = FeedForwardSpec::Ungated(DenseFeedForwardSpec {
        intermediate_size: 6,
        activation: Activation::Gelu,
        bias: true,
    });
    let mut delta = modular_layer(MixerSpec::DeltaNet {
        key_heads: 1,
        value_heads: 2,
        key_head_dim: 4,
        value_head_dim: 4,
        convolution_kernel: 2,
    });
    delta.feed_forward = modular_experts(3, 2);
    let mamba = modular_layer(MixerSpec::Mamba2(Mamba2Spec {
        heads: 2,
        head_dim: 4,
        state_dim: 8,
        convolution_kernel: 2,
        group_size: 1,
    }));
    modular_stack(vec![
        modular_layer(modular_attention(2, 4, rope(4))),
        latent,
        differential,
        delta,
        mamba,
    ])
}

/// A modular decoder whose every mixer has one head and whose router selects one expert.
///
/// One-head attention of width 8, one-head latent attention, one-head gated
/// DeltaNet with top-1 routing over two experts, and one-head Mamba-2: every
/// per-head axis is a unit axis.
fn modular_single_head_decoder() -> DecoderSpec {
    let mut delta = modular_layer(MixerSpec::DeltaNet {
        key_heads: 1,
        value_heads: 1,
        key_head_dim: 4,
        value_head_dim: 8,
        convolution_kernel: 2,
    });
    delta.feed_forward = modular_experts(2, 1);
    let mamba = modular_layer(MixerSpec::Mamba2(Mamba2Spec {
        heads: 1,
        head_dim: 8,
        state_dim: 4,
        convolution_kernel: 3,
        group_size: 1,
    }));
    modular_stack(vec![
        modular_layer(modular_attention(1, 8, rope(8))),
        modular_layer(modular_latent(1)),
        delta,
        mamba,
    ])
}

/// The modular causal decoders under both attention modes on the accelerator.
fn modular_causal_matches_flex<A: Backend>(device: &A::Device, bound: Bound) -> Report {
    let flex_device = Default::default();
    let mut largest = Report::default();
    for (name, spec) in [
        ("modular causal", modular_decoder()),
        ("single-head modular causal", modular_single_head_decoder()),
    ] {
        let (reference, accelerated) = seeded(|| {
            let reference = ModularCausalLm::<Reference>::init(spec.clone(), &flex_device).unwrap();
            let accelerated = copy_parameters(
                &reference,
                ModularCausalLm::<A>::init(spec, device).unwrap(),
                identity,
            );
            (reference, accelerated)
        });
        for mode in [
            AttentionPrecisionMode::BackendDefault,
            AttentionPrecisionMode::EagerFp32Softmax,
        ] {
            let reference = reference.clone().with_attention_precision(mode);
            let accelerated = accelerated.clone().with_attention_precision(mode);
            let report = compare_causal(&reference, &accelerated, device, 32);
            report.check(&format!("{name} ({mode:?})"), bound);
            largest = largest.merge(report);
        }
    }
    largest
}

/// The modular encoder-decoder: two RoPE attention layers on each side.
fn modular_seq2seq() -> Seq2SeqSpec {
    let stack = || {
        modular_stack(vec![
            modular_layer(modular_attention(2, 4, rope(4))),
            modular_layer(modular_attention(2, 4, rope(2))),
        ])
    };
    Seq2SeqSpec {
        encoder: stack(),
        decoder: stack(),
        cross_attention_heads: 2,
        cross_attention_head_dim: 4,
        cross_attention_options: AttentionOptions::default(),
    }
}

/// The modular encoder-decoder under both attention modes on the accelerator.
fn modular_seq2seq_matches_flex<A: Backend>(device: &A::Device, bound: Bound) -> Report {
    let flex_device = Default::default();
    let (reference, accelerated) = seeded(|| {
        let reference =
            ModularSeq2SeqLm::<Reference>::init(modular_seq2seq(), &flex_device).unwrap();
        let accelerated = copy_parameters(
            &reference,
            ModularSeq2SeqLm::<A>::init(modular_seq2seq(), device).unwrap(),
            identity,
        );
        (reference, accelerated)
    });
    let mut largest = Report::default();
    for mode in [
        AttentionPrecisionMode::BackendDefault,
        AttentionPrecisionMode::EagerFp32Softmax,
    ] {
        let reference = reference.clone().with_attention_precision(mode);
        let accelerated = accelerated.clone().with_attention_precision(mode);
        let report = compare_seq2seq(&reference, &accelerated, device, 32);
        report.check(&format!("modular seq2seq ({mode:?})"), bound);
        largest = largest.merge(report);
    }
    largest
}

/// A two-layer gated-GELU T5 with relative-position buckets below the batch width.
fn t5() -> T5Config {
    T5Config {
        vocab_size: 32,
        d_model: 8,
        d_kv: 4,
        d_ff: 16,
        num_layers: 2,
        num_decoder_layers: Some(2),
        num_heads: 2,
        relative_attention_num_buckets: 8,
        relative_attention_max_distance: 16,
        dropout_rate: 0.0,
        feed_forward_proj: T5FeedForwardProjection::GatedGelu,
        ..T5Config::default()
    }
}

/// T5 on the accelerator.
fn t5_matches_flex<A: Backend>(device: &A::Device, bound: Bound) -> Report {
    let flex_device = Default::default();
    let (reference, accelerated) = seeded(|| {
        let reference = T5ForConditionalGeneration::<Reference>::init(t5(), &flex_device).unwrap();
        let accelerated = copy_parameters(
            &reference,
            T5ForConditionalGeneration::<A>::init(t5(), device).unwrap(),
            identity,
        );
        (reference, accelerated)
    });
    let report = compare_seq2seq(&reference, &accelerated, device, 32);
    report.check("T5", bound);
    report
}

/// A two-block GPT-2 of width 16.
fn gpt2() -> Gpt2Config {
    Gpt2Config {
        vocab_size: 32,
        n_embd: 16,
        n_head: 4,
        n_layer: 2,
        n_positions: 16,
        n_inner: Some(24),
        ..Gpt2Config::default()
    }
}

/// GPT-2 under eager and fused attention on the accelerator (stateless only).
fn gpt2_matches_flex<A: Backend>(device: &A::Device, bound: Bound) -> Report {
    let flex_device = Default::default();
    let (reference, accelerated) = seeded(|| {
        let reference = Gpt2ForCausalLm::<Reference>::init(gpt2(), &flex_device).unwrap();
        let accelerated = copy_parameters(
            &reference,
            Gpt2ForCausalLm::<A>::init(gpt2(), device).unwrap(),
            identity,
        );
        (reference, accelerated)
    });
    let mut largest = Report::default();
    for mode in [
        AttentionPrecisionMode::EagerFp32Softmax,
        AttentionPrecisionMode::BackendDefault,
    ] {
        let reference = reference.clone().with_attention_precision(mode);
        let accelerated = accelerated.clone().with_attention_precision(mode);
        let report = compare_stateless(
            |input, options| reference.forward(input, options),
            |input, options| accelerated.forward(input, options),
            device,
            32,
        );
        report.check(&format!("GPT-2 ({mode:?})"), bound);
        largest = largest.merge(report);
    }
    largest
}

/// A two-layer GPT-NeoX with half-width rotary embeddings and parallel residuals.
fn gpt_neox() -> GptNeoXConfig {
    GptNeoXConfig {
        hidden_size: 16,
        intermediate_size: 24,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        vocab_size: 32,
        max_position_embeddings: 16,
        rotary_pct: 0.5,
        ..GptNeoXConfig::default()
    }
}

/// GPT-NeoX under eager and fused attention on the accelerator (stateless only).
fn gpt_neox_matches_flex<A: Backend>(device: &A::Device, bound: Bound) -> Report {
    let flex_device = Default::default();
    let (reference, accelerated) = seeded(|| {
        let reference = GptNeoXForCausalLm::<Reference>::init(gpt_neox(), &flex_device).unwrap();
        let accelerated = copy_parameters(
            &reference,
            GptNeoXForCausalLm::<A>::init(gpt_neox(), device).unwrap(),
            identity,
        );
        (reference, accelerated)
    });
    let mut largest = Report::default();
    for mode in [
        AttentionPrecisionMode::EagerFp32Softmax,
        AttentionPrecisionMode::BackendDefault,
    ] {
        let reference = reference.clone().with_attention_precision(mode);
        let accelerated = accelerated.clone().with_attention_precision(mode);
        let report = compare_stateless(
            |input, options| reference.forward(input, options),
            |input, options| accelerated.forward(input, options),
            device,
            32,
        );
        report.check(&format!("GPT-NeoX ({mode:?})"), bound);
        largest = largest.merge(report);
    }
    largest
}

/// Uploads row-major `values` as a floating tensor of `shape`.
fn upload<B: Backend, const D: usize>(
    values: Vec<f32>,
    shape: [usize; D],
    device: &B::Device,
) -> Tensor<B, D> {
    Tensor::from_data(TensorData::new(values, shape), device)
}

/// Deterministic values `sin(0.731 (i + phase))` for `count` elements.
fn wave(count: usize, phase: f32) -> Vec<f32> {
    (0..count)
        .map(|index| ((index as f32 + phase) * 0.731).sin())
        .collect()
}

/// How a gated delta-rule case presents its per-head decay and update strength.
#[derive(Clone, Copy, Debug)]
enum DeltaRuleLayout {
    /// Contiguous `[batch, sequence, heads]` factors and a left-padded second row.
    Masked,
    /// Factors viewed from `[batch, heads, sequence]` storage, without a mask.
    Strided,
}

/// A two-row gated delta-rule input with an incoming state.
///
/// `heads` heads of key and value width 4 over six tokens; decays are
/// negative (`-0.1` to `-0.6`) and update strengths lie in `(0.1, 0.9)`. A
/// masked input hides row 1's first two tokens; masking rewrites the factors
/// into fresh tensors, so the strided layout instead passes them unmasked as
/// transposed views, which the kernels must read through their strides.
fn delta_rule_input<B: Backend>(
    heads: usize,
    layout: DeltaRuleLayout,
    device: &B::Device,
) -> DeltaRuleInput<B> {
    let [batch, sequence, width] = [2, 6, 4];
    let tokens = batch * sequence * heads;
    let factor = |values: Vec<f32>| match layout {
        DeltaRuleLayout::Masked => upload::<B, 3>(values, [batch, sequence, heads], device),
        DeltaRuleLayout::Strided => {
            upload::<B, 3>(values, [batch, heads, sequence], device).swap_dims(1, 2)
        }
    };
    let valid = (0..batch * sequence)
        .map(|index| index < sequence || index % sequence >= 2)
        .collect::<Vec<_>>();
    DeltaRuleInput {
        query: upload(
            wave(tokens * width, 1.0),
            [batch, sequence, heads, width],
            device,
        ),
        key: upload(
            wave(tokens * width, 2.0),
            [batch, sequence, heads, width],
            device,
        ),
        value: upload(
            wave(tokens * width, 3.0),
            [batch, sequence, heads, width],
            device,
        ),
        log_decay: factor(
            wave(tokens, 4.0)
                .into_iter()
                .map(|value| -0.1 - 0.5 * value.abs())
                .collect(),
        ),
        beta: factor(
            wave(tokens, 5.0)
                .into_iter()
                .map(|value| 0.5 + 0.4 * value)
                .collect(),
        ),
        initial_state: Some(upload(
            wave(batch * heads * width * width, 6.0),
            [batch, heads, width, width],
            device,
        )),
        token_mask: matches!(layout, DeltaRuleLayout::Masked)
            .then(|| Tensor::from_data(TensorData::new(valid, [batch, sequence]), device)),
    }
}

/// Values and final state of the recurrent, snapshot, and chunked delta-rule kernels.
struct DeltaRuleRun {
    /// Recurrent values followed by the recurrent final state.
    recurrent: Vec<f32>,
    /// Snapshot values followed by every per-token state.
    snapshots: Vec<f32>,
    /// Chunked values followed by the chunked final state.
    chunked: Vec<f32>,
}

impl DeltaRuleRun {
    /// Evaluates the three kernels on `device`; the chunked kernel uses chunks of 4.
    fn new<B: Backend>(heads: usize, layout: DeltaRuleLayout, device: &B::Device) -> Self {
        let config = DeltaRuleConfig {
            chunk_size: 4,
            ..DeltaRuleConfig::default()
        };
        let input = || delta_rule_input::<B>(heads, layout, device);
        let recurrent = recurrent_gated_delta_rule(input(), config).unwrap();
        let snapshots = recurrent_gated_delta_rule_with_snapshots(input(), config).unwrap();
        let chunked = chunked_gated_delta_rule(input(), config).unwrap();
        let join = |values: Vec<f32>, state: Vec<f32>| values.into_iter().chain(state).collect();
        Self {
            recurrent: join(host(recurrent.values), host(recurrent.final_state)),
            snapshots: join(host(snapshots.values), host(snapshots.states)),
            chunked: join(host(chunked.values), host(chunked.final_state)),
        }
    }
}

/// The gated delta rule's recurrent, snapshot, and chunked kernels on the accelerator.
///
/// The batch of two rows exercises every per-row slice of the token-by-token
/// recurrence, where Burn 0.21's CubeCL reshape of a transposed token slice
/// once read the wrong rows for every batch above one (see
/// `trailing_unit_axis` in `src/delta_net.rs`); one head makes that axis a
/// unit axis. Each kernel's values and states are compared with Flex, and the
/// accelerator's recurrent and chunked results with each other.
fn delta_rule_kernels_match_flex<A: Backend>(device: &A::Device, bound: Bound) -> Gap {
    let mut largest = Gap::default();
    for heads in [1, 2] {
        for layout in [DeltaRuleLayout::Masked, DeltaRuleLayout::Strided] {
            let flex = DeltaRuleRun::new::<Reference>(heads, layout, &Default::default());
            let actual = DeltaRuleRun::new::<A>(heads, layout, device);
            let gaps = [
                (
                    "recurrent",
                    Gap::measure(&actual.recurrent, &flex.recurrent),
                ),
                (
                    "snapshots",
                    Gap::measure(&actual.snapshots, &flex.snapshots),
                ),
                ("chunked", Gap::measure(&actual.chunked, &flex.chunked)),
                (
                    "recurrent vs chunked",
                    Gap::measure(&actual.recurrent, &actual.chunked),
                ),
            ];
            for (label, gap) in gaps {
                let case = format!("gated delta rule, {heads} head(s), {layout:?}, {label}");
                eprintln!(
                    "{case}: max |Δ| {:.2e} (scale {:.2})",
                    gap.deviation, gap.scale
                );
                bound.check(gap, &case);
                largest = largest.max(gap);
            }
        }
    }
    largest
}

/// Runs every family on the accelerator and returns the largest measurement.
fn every_family_matches_flex<A: Backend>(device: &A::Device, bound: Bound) -> Report {
    [
        llama_family_matches_flex::<A>(device, bound),
        gemma_matches_flex::<A>(device, bound),
        gpt_oss_matches_flex::<A>(device, bound),
        hybrid_registry_families_match_flex::<A>(device, bound),
        modular_causal_matches_flex::<A>(device, bound),
        modular_seq2seq_matches_flex::<A>(device, bound),
        t5_matches_flex::<A>(device, bound),
        gpt2_matches_flex::<A>(device, bound),
        gpt_neox_matches_flex::<A>(device, bound),
    ]
    .into_iter()
    .fold(Report::default(), Report::merge)
}

/// Flex against a parameter copy of itself: the copy is exact and every comparison runs on CPU.
///
/// Stateless and cached logits must agree bit for bit, which a missed,
/// reordered, or mis-shaped parameter would break; cached-versus-stateless
/// self-consistency is Flex's own F32 drift and must meet [`F32_BOUND`].
#[test]
fn every_family_matches_itself_on_flex() {
    let device = Default::default();
    let largest = every_family_matches_flex::<Reference>(&device, F32_BOUND);
    delta_rule_kernels_match_flex::<Reference>(&device, F32_BOUND);
    assert_eq!(largest.stateless.deviation, 0.0, "stateless copy is exact");
    assert_eq!(largest.unpadded.deviation, 0.0, "unpadded copy is exact");
    assert_eq!(largest.cached.deviation, 0.0, "cached copy is exact");
}

/// The cases cover every family of both automatic registries.
#[test]
fn every_registered_family_has_a_parity_case() {
    let device = Default::default();
    let covered: Vec<AutoCausalConfig> = registry_cases()
        .map(|(_, config)| seeded(|| config.build::<Reference>(&device)).family())
        .collect();
    for family in AutoCausalConfig::ALL {
        assert!(covered.contains(family), "{family:?} lacks a parity case");
    }
    // T5 is the only encoder-decoder family; `t5_matches_flex` covers it.
    assert_eq!(AutoSeq2SeqConfig::MODEL_TYPES, ["t5"]);
}

/// TF32 rounding of the weights alone stays within a quarter of the TF32 bound for every family.
///
/// Each family's Flex model is copied with every parameter rounded to TF32
/// ([`round_to_tf32`]) and compared with the exact model on Flex. That is
/// the weight half of a TF32 device's input rounding; activations contribute
/// a comparable half (support module documentation), so a quarter of the
/// bound leaves twice the expected full deviation as headroom. The printed
/// fractions are the measurements the support module quotes.
///
/// The quarter holds for the pinned draw ([`SEED`]), which [`seeded`] keeps
/// independent of test scheduling and which every accelerator case shares,
/// not for random weights in general. Re-seeding 400 draws (`SEED` to
/// `SEED + 399`, each in a sequential process) failed 100, mostly Muse
/// Glimmer (42), Qwen3.5 (35), and gpt-oss (13); gpt-oss's deviation reached
/// 36.9 times the quarter bound, about nine times the full TF32 bound. On the
/// pinned draw Muse Glimmer uses about half of it (`1.30e-3` against
/// `2.89e-3`). A change to `SEED`, to a family's parameter order, or to
/// Burn's generator therefore calls for re-measuring this test and the bound
/// it supports, not for re-seeding until it passes.
#[test]
fn tf32_weight_rounding_stays_within_a_quarter_of_the_tf32_bound() {
    let device = Default::default();
    let tf32 = MatmulPrecision::Tf32.bound(F32_BOUND);
    let quarter = Bound {
        absolute: tf32.absolute / 4.0,
        relative: tf32.relative / 4.0,
    };
    let mut largest = 0.0_f32;
    let mut check = |name: &str, report: Report| {
        eprintln!(
            "{name}: TF32 weights move logits by {:.2e} of the scale",
            report.largest_relative()
        );
        largest = largest.max(report.largest_relative());
        for gap in [report.stateless, report.unpadded, report.cached] {
            quarter.check(gap, name);
        }
    };
    for (name, config) in registry_cases() {
        let (exact, rounded) = seeded(|| {
            let exact = config.build::<Reference>(&device);
            let rounded =
                copy_parameters(&exact, config.build::<Reference>(&device), round_to_tf32);
            (exact, rounded)
        });
        let vocabulary = exact.config().vocabulary_size();
        check(name, compare_causal(&exact, &rounded, &device, vocabulary));
    }
    for (name, spec) in [
        ("modular causal", modular_decoder()),
        ("single-head modular causal", modular_single_head_decoder()),
    ] {
        let (exact, rounded) = seeded(|| {
            let exact = ModularCausalLm::<Reference>::init(spec.clone(), &device).unwrap();
            let rounded = copy_parameters(
                &exact,
                ModularCausalLm::<Reference>::init(spec, &device).unwrap(),
                round_to_tf32,
            );
            (exact, rounded)
        });
        check(name, compare_causal(&exact, &rounded, &device, 32));
    }
    let (exact, rounded) = seeded(|| {
        let exact = ModularSeq2SeqLm::<Reference>::init(modular_seq2seq(), &device).unwrap();
        let rounded = copy_parameters(
            &exact,
            ModularSeq2SeqLm::<Reference>::init(modular_seq2seq(), &device).unwrap(),
            round_to_tf32,
        );
        (exact, rounded)
    });
    check(
        "modular seq2seq",
        compare_seq2seq(&exact, &rounded, &device, 32),
    );
    let (exact, rounded) = seeded(|| {
        let exact = T5ForConditionalGeneration::<Reference>::init(t5(), &device).unwrap();
        let rounded = copy_parameters(
            &exact,
            T5ForConditionalGeneration::<Reference>::init(t5(), &device).unwrap(),
            round_to_tf32,
        );
        (exact, rounded)
    });
    check("T5", compare_seq2seq(&exact, &rounded, &device, 32));
    let (exact, rounded) = seeded(|| {
        let exact = Gpt2ForCausalLm::<Reference>::init(gpt2(), &device).unwrap();
        let rounded = copy_parameters(
            &exact,
            Gpt2ForCausalLm::<Reference>::init(gpt2(), &device).unwrap(),
            round_to_tf32,
        );
        (exact, rounded)
    });
    check(
        "GPT-2",
        compare_stateless(
            |input, options| exact.forward(input, options),
            |input, options| rounded.forward(input, options),
            &device,
            32,
        ),
    );
    let (exact, rounded) = seeded(|| {
        let exact = GptNeoXForCausalLm::<Reference>::init(gpt_neox(), &device).unwrap();
        let rounded = copy_parameters(
            &exact,
            GptNeoXForCausalLm::<Reference>::init(gpt_neox(), &device).unwrap(),
            round_to_tf32,
        );
        (exact, rounded)
    });
    check(
        "GPT-NeoX",
        compare_stateless(
            |input, options| exact.forward(input, options),
            |input, options| rounded.forward(input, options),
            &device,
            32,
        ),
    );
    eprintln!("largest relative TF32 weight-rounding deviation: {largest:.2e}");
}

/// TF32 rounding keeps ten mantissa bits, rounds ties away from zero, and carries into the exponent.
#[test]
fn tf32_rounding_matches_the_device_conversion() {
    let ulp = 1.0 / 1024.0;
    // Exact TF32 values are fixed points, of either sign.
    for value in [
        0.0,
        -0.0,
        1.0,
        -1.0,
        1.0 + ulp,
        3.5,
        -1024.0,
        f32::MIN_POSITIVE,
    ] {
        assert_eq!(round_to_tf32(value).to_bits(), value.to_bits(), "{value}");
    }
    // Below half a unit rounds down, half a unit and above rounds away from zero.
    assert_eq!(round_to_tf32(1.0 + ulp / 4.0), 1.0);
    assert_eq!(round_to_tf32(1.0 + ulp / 2.0), 1.0 + ulp);
    assert_eq!(round_to_tf32(-(1.0 + ulp / 2.0)), -(1.0 + ulp));
    assert_eq!(round_to_tf32(1.0 + 0.75 * ulp), 1.0 + ulp);
    // The largest value below two rounds up into the next binade.
    assert_eq!(round_to_tf32(2.0 - f32::EPSILON), 2.0);
    // Non-finite values pass through; the largest finite value overflows.
    assert_eq!(round_to_tf32(f32::INFINITY), f32::INFINITY);
    assert_eq!(round_to_tf32(f32::NEG_INFINITY), f32::NEG_INFINITY);
    assert!(round_to_tf32(f32::NAN).is_nan());
    assert_eq!(round_to_tf32(f32::MAX), f32::INFINITY);
    // The relative error never exceeds half a TF32 unit, 2^-11.
    for index in 1..10_000 {
        let value = index as f32 * 0.123_456_7;
        let rounded = round_to_tf32(value);
        assert!(((rounded - value) / value).abs() <= 1.0 / 2048.0, "{value}");
    }
}

/// The probe recognizes F32 and TF32 products and refuses anything coarser.
#[test]
fn matmul_precision_probe_classifies_input_rounding() {
    assert_eq!(
        MatmulPrecision::probe::<Reference>(&Default::default()),
        MatmulPrecision::F32
    );
    // Emulated TF32 products: each of the 32 terms is `round_to_tf32(c) · 1`.
    let emulate = |value: f32| vec![32.0 * round_to_tf32(value); 4];
    let fine = 1.0 + 1.0 / 4096.0;
    let coarse = 1.0 + 1.0 / 1024.0;
    assert_eq!(
        MatmulPrecision::classify(&emulate(fine), &emulate(coarse)),
        MatmulPrecision::Tf32
    );
    assert_eq!(
        MatmulPrecision::classify(&[32.0 * fine; 4], &[32.0 * coarse; 4]),
        MatmulPrecision::F32
    );
    assert_eq!(MatmulPrecision::F32.bound(F32_BOUND), F32_BOUND);
    assert_eq!(
        MatmulPrecision::Tf32.bound(F32_BOUND),
        Bound {
            absolute: 1e-2,
            relative: 1e-2
        }
    );
}

/// Inputs coarser than TF32 (BF16 loses `2^-10` too) justify no bound and are refused.
#[test]
#[should_panic(expected = "match neither F32 nor TF32 inputs")]
fn matmul_precision_probe_refuses_inputs_coarser_than_tf32() {
    MatmulPrecision::classify(&[32.0; 4], &[32.0; 4]);
}

/// Real-checkpoint tolerances: an override applies anywhere, the F32 default only to an F32 device.
#[test]
fn checkpoint_tolerances_take_overrides_and_keep_f32_defaults_to_f32_devices() {
    for precision in [MatmulPrecision::F32, MatmulPrecision::Tf32] {
        assert_eq!(
            resolve_checkpoint_tolerance("TOLERANCE", Some("2.5e-2"), 1e-3, precision),
            2.5e-2
        );
    }
    assert_eq!(
        resolve_checkpoint_tolerance("TOLERANCE", None, 1e-3, MatmulPrecision::F32),
        1e-3
    );
    assert_eq!(
        checkpoint_tolerance(
            "TRANSFORMERS_BACKEND_PARITY_UNSET_TOLERANCE",
            1e-3,
            MatmulPrecision::F32
        ),
        1e-3
    );
}

/// A TF32 device without an override is refused, naming the variable to set.
#[test]
#[should_panic(expected = "set TRANSFORMERS_EXAMPLE_TOLERANCE to the tolerance this TF32 run")]
fn checkpoint_tolerances_require_an_override_on_tf32_devices() {
    resolve_checkpoint_tolerance(
        "TRANSFORMERS_EXAMPLE_TOLERANCE",
        None,
        1e-3,
        MatmulPrecision::Tf32,
    );
}

/// A malformed override is refused rather than ignored.
#[test]
#[should_panic(expected = "TRANSFORMERS_EXAMPLE_TOLERANCE must be a float, not \"loose\"")]
fn checkpoint_tolerances_refuse_a_malformed_override() {
    resolve_checkpoint_tolerance(
        "TRANSFORMERS_EXAMPLE_TOLERANCE",
        Some("loose"),
        1e-3,
        MatmulPrecision::F32,
    );
}

// ---------------------------------------------------------------------------
// Greedy continuations of the real-checkpoint gates
// ---------------------------------------------------------------------------

/// Top-2 margins of the full BF16 gpt-oss-20b bundle's 32-step greedy continuation.
///
/// Copied from `gpt-oss-20b-6cee5e81-bf16-reference.bundle/reference.json`
/// (SHA-256 `9a7891117900c5b90e049ce51616ce68ade3132d95ff4a1a5fa460da14660eac`;
/// Transformers 5.2.0, PyTorch 2.10.0, revision `6cee5e81`). BF16 logits make
/// every margin a multiple of `1/16` here, so each is exact in F32.
const GPT_OSS_FULL_BF16_GREEDY_MARGINS: [f32; 32] = [
    3.0625, 0.0625, 0.6875, 0.0625, 0.4375, 0.25, 1.9375, 0.375, 0.375, 0.0625, 2.125, 10.25, 1.0,
    0.125, 0.375, 3.375, 0.25, 0.375, 1.875, 2.25, 1.5, 10.25, 4.375, 7.5, 3.875, 5.625, 1.375,
    2.5, 1.375, 1.125, 2.25, 5.625,
];

/// Eight logits whose argmax is `token`, by a margin of one.
fn logits_choosing(token: usize) -> Vec<f32> {
    let mut logits = vec![0.0; 8];
    logits[token] = 1.0;
    logits
}

/// Argmax prefers the lowest index among equal maxima, as the reference generator does.
#[test]
fn greedy_argmax_prefers_the_lowest_index_among_equal_maxima() {
    assert_eq!(argmax(&[0.5, 2.0, -1.0, 2.0]), 1);
    assert_eq!(argmax(&[-3.0]), 0);
    assert_eq!(argmax(&[f32::NEG_INFINITY, -1e30]), 1);
    assert_eq!(argmax(&[1.0, f32::INFINITY, f32::INFINITY]), 1);
}

/// The argmax of no logits is refused.
#[test]
#[should_panic(expected = "argmax of an empty slice")]
fn greedy_argmax_refuses_an_empty_slice() {
    argmax(&[]);
}

/// A margin decides the argmax only strictly above twice the tolerance, and that bound is tight.
#[test]
fn greedy_steps_are_decisive_only_strictly_above_twice_the_tolerance() {
    let above_two = f32::from_bits(2.0_f32.to_bits() + 1);
    assert!(!greedy_step_is_decisive(2.0, 1.0));
    assert!(greedy_step_is_decisive(above_two, 1.0));
    // An exact reference tie never decides; with no tolerance any positive margin does.
    assert!(!greedy_step_is_decisive(0.0, 0.0));
    assert!(greedy_step_is_decisive(f32::MIN_POSITIVE, 0.0));
    // A NaN margin is never decisive; the minimum count still guards the gate.
    assert!(!greedy_step_is_decisive(f32::NAN, 1e-3));
    // The full BF16 gpt-oss-20b bundle at the half-precision tolerance.
    assert!(!greedy_step_is_decisive(3.0625, 2.5));
    assert!(greedy_step_is_decisive(5.625, 2.5));
    // Tightness at τ = 1: at a margin of exactly 2τ, moving the best logit
    // down by τ and the runner-up up by τ ties them, and the lowest index wins.
    let tight = [1.0, 3.0];
    assert_eq!(argmax(&tight), 1);
    assert!(!greedy_step_is_decisive(tight[1] - tight[0], 1.0));
    assert_eq!(argmax(&[tight[0] + 1.0, tight[1] - 1.0]), 0);
    // One F32 unit above 2τ (every value here is exact), the same shift cannot.
    let decisive = [1.0, 1.0 + above_two];
    assert!(greedy_step_is_decisive(decisive[1] - decisive[0], 1.0));
    assert_eq!(argmax(&[decisive[0] + 1.0, decisive[1] - 1.0]), 1);
}

/// A tie is skipped, later decisive steps are compared, and every fed token is the reference's.
#[test]
fn teacher_forced_greedy_skips_ties_and_feeds_the_reference_tokens() {
    // At τ = 1 step 1 is a tie (margin 0.5 ≤ 2) and the run picks token 6 there
    // instead of 4. That is no error, and step 2 continues the prefix [3, 4].
    let reference = [(3, 4.0), (4, 0.5), (5, 2.5)];
    let mut fed = Vec::new();
    let mut later = [logits_choosing(6), logits_choosing(5)].into_iter();
    let outcome = check_teacher_forced_greedy(
        "synthetic",
        &reference,
        1.0,
        2,
        "TRANSFORMERS_EXAMPLE_MIN_GREEDY",
        logits_choosing(3),
        |token| {
            fed.push(token);
            later.next().expect("one decode per later step")
        },
    );
    assert_eq!(
        outcome,
        GreedyComparison {
            compared: 2,
            ties: 1
        }
    );
    assert_eq!(
        fed,
        [3, 4],
        "reference tokens only, none after the last step"
    );
}

/// Regression: the full BF16 gpt-oss-20b bundle passes its default minimum at the half tolerance.
///
/// Its first margin, `3.0625`, is a tie at `2τ = 5`, so stopping at the first
/// tie compared nothing and failed `DEFAULT_MIN_GREEDY = 2` on every backend.
/// Skipping ties compares the five steps whose margins exceed `5`.
#[test]
fn teacher_forced_greedy_compares_every_decisive_step_of_the_full_bf16_gpt_oss_bundle() {
    let tolerance = 2.5;
    let reference: Vec<(i64, f32)> = GPT_OSS_FULL_BF16_GREEDY_MARGINS
        .iter()
        .enumerate()
        .map(|(step, &margin)| ((step % 8) as i64, margin))
        .collect();
    let stopping_at_the_first_tie = reference
        .iter()
        .take_while(|&&(_, margin)| greedy_step_is_decisive(margin, tolerance))
        .count();
    assert_eq!(stopping_at_the_first_tie, 0);
    let mut fed = Vec::new();
    let outcome = check_teacher_forced_greedy(
        "gpt-oss-20b BF16",
        &reference,
        tolerance,
        2,
        "TRANSFORMERS_GPT_OSS_MIN_GREEDY",
        logits_choosing(0),
        |token| {
            fed.push(token);
            logits_choosing(((token + 1) % 8) as usize)
        },
    );
    assert_eq!(
        outcome,
        GreedyComparison {
            compared: 5,
            ties: 27
        }
    );
    let tokens: Vec<i64> = reference.iter().map(|&(token, _)| token).collect();
    assert_eq!(fed, tokens[..31]);
}

/// A decisive step whose argmax differs from the reference fails, naming the step.
#[test]
#[should_panic(expected = "synthetic: greedy step 1 (margin 3e0 above twice the tolerance 1e0)")]
fn teacher_forced_greedy_rejects_a_decisive_mismatch() {
    check_teacher_forced_greedy(
        "synthetic",
        &[(3, 4.0), (4, 3.0)],
        1.0,
        0,
        "TRANSFORMERS_EXAMPLE_MIN_GREEDY",
        logits_choosing(3),
        |_| logits_choosing(6),
    );
}

/// Too few decisive steps fail, naming the variable that lowers the minimum.
#[test]
#[should_panic(
    expected = "synthetic: only 1 decisive greedy tokens were compared (minimum 2; 2 ties); \
                ties are judged at tolerance 1e0; regenerate the bundle with another first \
                prompt or lower TRANSFORMERS_EXAMPLE_MIN_GREEDY"
)]
fn teacher_forced_greedy_requires_the_minimum_decisive_count() {
    check_teacher_forced_greedy(
        "synthetic",
        &[(3, 0.5), (4, 9.0), (5, 2.0)],
        1.0,
        2,
        "TRANSFORMERS_EXAMPLE_MIN_GREEDY",
        logits_choosing(7),
        |token| logits_choosing(token as usize + 1),
    );
}

/// The minimum is capped at the continuation's length; zero admits a continuation of ties.
#[test]
fn teacher_forced_greedy_caps_the_minimum_at_the_continuation_length() {
    let single = check_teacher_forced_greedy(
        "synthetic",
        &[(2, 9.0)],
        1.0,
        2,
        "TRANSFORMERS_EXAMPLE_MIN_GREEDY",
        logits_choosing(2),
        |_| unreachable!("a one-step continuation decodes nothing"),
    );
    assert_eq!(
        single,
        GreedyComparison {
            compared: 1,
            ties: 0
        }
    );
    let ties = check_teacher_forced_greedy(
        "synthetic",
        &[(2, 0.1), (3, 0.2)],
        1.0,
        0,
        "TRANSFORMERS_EXAMPLE_MIN_GREEDY",
        logits_choosing(7),
        |_| logits_choosing(7),
    );
    assert_eq!(
        ties,
        GreedyComparison {
            compared: 0,
            ties: 2
        }
    );
    let empty = check_teacher_forced_greedy(
        "synthetic",
        &[],
        1.0,
        2,
        "TRANSFORMERS_EXAMPLE_MIN_GREEDY",
        logits_choosing(0),
        |_| unreachable!("an empty continuation decodes nothing"),
    );
    assert_eq!(empty, GreedyComparison::default());
}

/// The top-2 margin is the gap to the runner-up under `argmax`'s order.
#[test]
fn top_two_margins_measure_the_gap_to_the_runner_up() {
    assert_eq!(top_two_margin(&[1.0, 3.0]), 2.0);
    assert_eq!(top_two_margin(&[0.5, 2.0, -1.0, 1.5]), 0.5);
    assert_eq!(top_two_margin(&[0.5, 2.0, -1.0, 2.0]), 0.0, "equal maxima");
    assert_eq!(top_two_margin(&[-3.0]), f32::INFINITY, "no runner-up");
    assert_eq!(top_two_margin(&[f32::NEG_INFINITY, 4.0]), f32::INFINITY);
}

/// Without logits there is no margin.
#[test]
#[should_panic(expected = "argmax of an empty slice")]
fn top_two_margins_refuse_an_empty_slice() {
    top_two_margin(&[]);
}

/// The allowance bounded from the run's scale covers the reference's and reduces to an absolute bound.
#[test]
fn run_scale_allowances_cover_every_admissible_reference() {
    let absolute = Bound {
        absolute: 1e-5,
        relative: 0.0,
    };
    assert_eq!(absolute.allowed_for_actual_scale(0.0), 1e-5);
    assert_eq!(absolute.allowed_for_actual_scale(1e6), 1e-5);
    // TF32 at a run scale of 1.8: (1e-2 + 1e-2 · 1.8) / 0.99, rounded up.
    let tf32 = TF32_BOUND.allowed_for_actual_scale(1.8);
    let exact = (0.01_f64 + 0.01 * 1.8_f64) / 0.99;
    assert!(f64::from(tf32) >= exact && f64::from(tf32.next_down()) < exact);
    // Tight case: the run sits a full allowance below a reference of scale 2.
    let bound = Bound {
        absolute: 0.25,
        relative: 0.5,
    };
    let reference_scale = 2.0;
    let run_scale = reference_scale - bound.allowed(reference_scale);
    assert_eq!(run_scale, 0.75);
    assert_eq!(
        bound.allowed_for_actual_scale(run_scale),
        bound.allowed(reference_scale)
    );
}

/// A relative bound of one or more admits references of unbounded scale.
#[test]
#[should_panic(expected = "a relative bound of 1e0 admits references of any scale")]
fn run_scale_allowances_refuse_a_relative_bound_of_one() {
    Bound {
        absolute: 0.0,
        relative: 1.0,
    }
    .allowed_for_actual_scale(1.0);
}

/// A non-finite run scale has no allowance.
#[test]
#[should_panic(expected = "the actual scale must be finite and non-negative, not NaN")]
fn run_scale_allowances_refuse_a_non_finite_scale() {
    TF32_BOUND.allowed_for_actual_scale(f32::NAN);
}

/// Regression: a run margin between `τ` and `2τ` is a tie, not a mismatch.
///
/// Reference `[1.00, 0.99]` and run `[0.98, 1.009]` differ by at most `0.02`,
/// so the run is within `τ = 0.02`, yet its argmax differs. Its margin `0.029`
/// exceeds `τ`, which the tiny goldens once treated as decisive, but not
/// `2τ`, so the step is skipped.
#[test]
fn run_margins_within_twice_the_tolerance_are_ties() {
    let reference = [1.00_f32, 0.99];
    let run = vec![0.98_f32, 1.009];
    let bound = Bound {
        absolute: 0.02,
        relative: 0.0,
    };
    bound.assert_close(&run, &reference, "within the bound");
    assert_ne!(argmax(&run), argmax(&reference));
    let margin = top_two_margin(&run);
    let tolerance = bound.allowed_for_actual_scale(1.009);
    assert!(margin > tolerance && !greedy_step_is_decisive(margin, tolerance));
    let outcome = compare_teacher_forced_greedy(
        "tiny",
        &[argmax(&reference) as i64],
        0,
        "unused",
        |_, logits| GreedyDecision {
            margin: top_two_margin(logits),
            tolerance: bound.allowed_for_actual_scale(1.009),
        },
        run,
        |_| unreachable!("a one-step continuation decodes nothing"),
    );
    assert_eq!(
        outcome,
        GreedyComparison {
            compared: 0,
            ties: 1
        }
    );
}

/// A run-margin comparison also enforces its minimum, ending with the caller's remedy.
#[test]
#[should_panic(
    expected = "tiny: only 0 decisive greedy tokens were compared (minimum 2; 2 ties); the \
                golden's continuation is too close to call at this bound"
)]
fn run_margin_comparisons_require_the_minimum_decisive_count() {
    compare_teacher_forced_greedy(
        "tiny",
        &[0, 1],
        2,
        "the golden's continuation is too close to call at this bound",
        |_, logits| GreedyDecision {
            margin: top_two_margin(logits),
            tolerance: 0.5,
        },
        vec![1.0, 0.5],
        |_| vec![0.5, 1.0],
    );
}

/// Non-finite run logits fail a greedy step even when it is a tie.
#[test]
#[should_panic(expected = "synthetic: greedy step 0: the run's logits are not all finite")]
fn teacher_forced_greedy_rejects_non_finite_logits() {
    check_teacher_forced_greedy(
        "synthetic",
        &[(3, 0.5)],
        1.0,
        0,
        "TRANSFORMERS_EXAMPLE_MIN_GREEDY",
        vec![0.0, f32::NAN, 0.0, 1.0],
        |_| unreachable!("a one-step continuation decodes nothing"),
    );
}

proptest::proptest! {
    #![proptest_config(proptest::test_runner::Config {
        cases: 512,
        rng_seed: proptest::test_runner::RngSeed::Fixed(20_261_017),
        ..proptest::test_runner::Config::default()
    })]

    /// Within `τ` of every reference logit, a decisive reference margin forces the reference's argmax.
    ///
    /// Each run logit is the reference plus `u · τ` with `|u| ≤ 1`. F32
    /// rounding of that sum can land up to half a unit beyond `τ`, which the
    /// guarantee does not cover, so such draws are discarded.
    #[test]
    fn decisive_reference_margins_force_the_argmax_within_the_tolerance(
        reference in proptest::collection::vec(-30.0_f32..30.0, 2..12),
        units in proptest::collection::vec(-1.0_f32..=1.0, 12),
        tolerance in 1e-3_f32..2.0,
    ) {
        let run: Vec<f32> = reference
            .iter()
            .zip(&units)
            .map(|(&value, &unit)| value + unit * tolerance)
            .collect();
        proptest::prop_assume!(reference.iter().zip(&run).all(|(&expected, &actual)| {
            (f64::from(actual) - f64::from(expected)).abs() <= f64::from(tolerance)
        }));
        let best = argmax(&reference);
        let runner_up = reference
            .iter()
            .enumerate()
            .filter(|&(index, _)| index != best)
            .fold(f32::NEG_INFINITY, |largest, (_, &value)| largest.max(value));
        if greedy_step_is_decisive(reference[best] - runner_up, tolerance) {
            proptest::prop_assert_eq!(argmax(&run), best);
        }
    }

    /// Within a relative bound of every reference logit, a decisive run margin forces the reference's argmax.
    ///
    /// Each run logit is the reference plus `u · allowed(s)`, `s` being the
    /// reference's largest magnitude and `|u| ≤ 1`; draws whose F32 rounding
    /// leaves the exact allowance `absolute + relative · s` are discarded.
    /// The tolerance comes from the run's scale alone, as in the tiny goldens,
    /// and must cover that exact allowance.
    #[test]
    fn decisive_run_margins_force_the_reference_argmax_within_the_bound(
        reference in proptest::collection::vec(-30.0_f32..30.0, 2..12),
        units in proptest::collection::vec(-1.0_f32..=1.0, 12),
        absolute in 0.0_f32..0.5,
        relative in 0.0_f32..0.5,
    ) {
        let bound = Bound { absolute, relative };
        let scale = reference.iter().fold(0.0_f32, |scale, &value| scale.max(value.abs()));
        let exact = f64::from(absolute) + f64::from(relative) * f64::from(scale);
        let run: Vec<f32> = reference
            .iter()
            .zip(&units)
            .map(|(&value, &unit)| value + unit * bound.allowed(scale))
            .collect();
        proptest::prop_assume!(reference.iter().zip(&run).all(|(&expected, &actual)| {
            (f64::from(actual) - f64::from(expected)).abs() <= exact
        }));
        let run_scale = run.iter().fold(0.0_f32, |scale, &value| scale.max(value.abs()));
        let tolerance = bound.allowed_for_actual_scale(run_scale);
        proptest::prop_assert!(f64::from(tolerance) >= exact);
        if greedy_step_is_decisive(top_two_margin(&run), tolerance) {
            proptest::prop_assert_eq!(argmax(&run), argmax(&reference));
        }
    }
}

/// Gaps track the largest deviation and scale, and bounds scale with the reference.
#[test]
fn gaps_and_bounds_measure_relative_to_the_reference_scale() {
    let gap = Gap::measure(&[1.0, -2.5, 3.0], &[1.0, -2.0, 4.0]);
    assert_eq!(
        gap,
        Gap {
            deviation: 1.0,
            scale: 4.0
        }
    );
    assert_eq!(gap.relative(), 0.25);
    assert_eq!(
        Gap::measure(&[0.5], &[0.25]).relative(),
        0.25,
        "absolute below one"
    );
    assert_eq!(
        Gap::measure(&[f32::NAN], &[0.0]).deviation,
        f32::INFINITY,
        "non-finite values never pass"
    );
    assert_eq!(
        gap.max(Gap {
            deviation: 0.5,
            scale: 9.0
        }),
        Gap {
            deviation: 1.0,
            scale: 9.0
        }
    );
    let bound = Bound {
        absolute: 0.5,
        relative: 0.125,
    };
    assert_eq!(bound.allowed(4.0), 1.0);
    assert!(bound.admits(gap));
    assert!(!bound.admits(Gap {
        deviation: 1.01,
        scale: 4.0
    }));
    assert_eq!(
        bound.assert_close(&[1.0], &[1.5], "within"),
        Gap {
            deviation: 0.5,
            scale: 1.5
        }
    );
}

/// Non-finite values never pass a bound, even one that scales with the reference.
#[test]
fn gaps_with_non_finite_values_never_pass_a_bound() {
    let relative = Bound {
        absolute: 1e-5,
        relative: 1e-5,
    };
    // Regression: an infinite reference once made the scale, and with it a
    // relative allowance, infinite, so an infinite run passed.
    let infinite = Gap::measure(&[f32::INFINITY, 1.0], &[f32::INFINITY, 2.0]);
    assert_eq!(
        infinite,
        Gap {
            deviation: f32::INFINITY,
            scale: 2.0
        }
    );
    assert!(!relative.admits(infinite));
    for (actual, reference) in [
        (f32::NAN, 0.0),
        (0.0, f32::NAN),
        (f32::NEG_INFINITY, 0.0),
        (0.0, f32::INFINITY),
    ] {
        let gap = Gap::measure(&[actual], &[reference]);
        assert_eq!(gap.deviation, f32::INFINITY, "{actual} against {reference}");
        assert!(!relative.admits(gap));
    }
    assert!(!relative.admits(Gap {
        deviation: f32::INFINITY,
        scale: f32::INFINITY
    }));
    assert!(!relative.admits(Gap {
        deviation: f32::NAN,
        scale: 1.0
    }));
}

/// Sets of different lengths are refused rather than silently truncated.
#[test]
#[should_panic(expected = "compared lengths differ")]
fn gaps_refuse_sets_of_different_lengths() {
    Gap::measure(&[1.0, 2.0], &[1.0]);
}

/// A deviation beyond the bound fails with its context, deviation, and allowance.
#[test]
#[should_panic(expected = "far: max |Δ| = 2e0 exceeds 6.25e-1 (reference scale 1e0)")]
fn bounds_reject_deviations_beyond_the_allowance() {
    Bound {
        absolute: 0.5,
        relative: 0.125,
    }
    .check(Gap::measure(&[3.0], &[1.0]), "far");
}

/// Parameter copies are exact and apply the value transform.
#[test]
fn parameter_copies_are_exact_and_apply_the_transform() {
    let device = Default::default();
    let fresh = || LinearConfig::new(3, 2).init::<Reference>(&device);
    let (source, copied, doubled) = seeded(|| {
        let source = fresh();
        let copied = copy_parameters(&source, fresh(), identity);
        let doubled = copy_parameters(&source, fresh(), |value| 2.0 * value);
        (source, copied, doubled)
    });
    let input = Tensor::<Reference, 2>::from_data([[0.5, -1.0, 2.0]], &device);
    assert_eq!(
        host(copied.forward(input.clone())),
        host(source.forward(input))
    );
    assert_eq!(
        host(doubled.weight.val()),
        host(source.weight.val())
            .into_iter()
            .map(|value| 2.0 * value)
            .collect::<Vec<_>>()
    );
}

/// A target whose parameter has another shape is refused.
#[test]
#[should_panic(expected = "float parameter 0: source and target shapes differ")]
fn parameter_copies_refuse_a_shape_mismatch() {
    let device = Default::default();
    seeded(|| {
        copy_parameters(
            &LinearConfig::new(3, 2).init::<Reference>(&device),
            LinearConfig::new(4, 2).init::<Reference>(&device),
            identity,
        )
    });
}

/// A target with fewer parameters than the source is refused.
#[test]
#[should_panic(expected = "the source has more parameters than the target")]
fn parameter_copies_refuse_a_parameter_count_mismatch() {
    let device = Default::default();
    seeded(|| {
        copy_parameters(
            &LinearConfig::new(3, 2).init::<Reference>(&device),
            LinearConfig::new(3, 2)
                .with_bias(false)
                .init::<Reference>(&device),
            identity,
        )
    });
}

/// The Metal bound: Metal must multiply F32 matrices in F32, so [`F32_BOUND`] applies.
#[cfg(all(feature = "metal", target_os = "macos"))]
fn metal_bound() -> Bound {
    let precision = MatmulPrecision::probe::<burn::backend::Metal>(&Default::default());
    assert_eq!(
        precision,
        MatmulPrecision::F32,
        "Metal is expected to multiply F32 matrices in F32"
    );
    F32_BOUND
}

/// The CUDA bound: [`F32_BOUND`] when the device multiplies in F32, `TF32_BOUND` when it rounds to TF32.
#[cfg(feature = "cuda")]
fn cuda_bound() -> Bound {
    let precision = MatmulPrecision::probe::<burn::backend::Cuda>(&Default::default());
    let bound = precision.bound(F32_BOUND);
    eprintln!("CUDA F32 matrix products use {precision:?} inputs; bound {bound:?}");
    bound
}

#[cfg(all(feature = "metal", target_os = "macos"))]
mod metal {
    //! Metal/F32 against Flex/F32 with identical weights; Metal multiplies in F32.

    use burn::backend::Metal;

    use super::metal_bound;

    /// Metal: every Llama-family variant.
    #[test]
    #[ignore = "requires a Metal device"]
    fn metal_llama_family_matches_flex() {
        super::llama_family_matches_flex::<Metal>(&Default::default(), metal_bound());
    }

    /// Metal: Gemma 2 and Gemma 3 text.
    #[test]
    #[ignore = "requires a Metal device"]
    fn metal_gemma_matches_flex() {
        super::gemma_matches_flex::<Metal>(&Default::default(), metal_bound());
    }

    /// Metal: gpt-oss.
    #[test]
    #[ignore = "requires a Metal device"]
    fn metal_gpt_oss_matches_flex() {
        super::gpt_oss_matches_flex::<Metal>(&Default::default(), metal_bound());
    }

    /// Metal: Qwen3.5, Qwen3-MoE, DeepSeek-V2, Muse Glimmer, and Nemotron-H.
    #[test]
    #[ignore = "requires a Metal device"]
    fn metal_hybrid_registry_families_match_flex() {
        super::hybrid_registry_families_match_flex::<Metal>(&Default::default(), metal_bound());
    }

    /// Metal: both modular causal decoders.
    #[test]
    #[ignore = "requires a Metal device"]
    fn metal_modular_causal_matches_flex() {
        super::modular_causal_matches_flex::<Metal>(&Default::default(), metal_bound());
    }

    /// Metal: the modular encoder-decoder.
    #[test]
    #[ignore = "requires a Metal device"]
    fn metal_modular_seq2seq_matches_flex() {
        super::modular_seq2seq_matches_flex::<Metal>(&Default::default(), metal_bound());
    }

    /// Metal: T5.
    #[test]
    #[ignore = "requires a Metal device"]
    fn metal_t5_matches_flex() {
        super::t5_matches_flex::<Metal>(&Default::default(), metal_bound());
    }

    /// Metal: GPT-2.
    #[test]
    #[ignore = "requires a Metal device"]
    fn metal_gpt2_matches_flex() {
        super::gpt2_matches_flex::<Metal>(&Default::default(), metal_bound());
    }

    /// Metal: GPT-NeoX.
    #[test]
    #[ignore = "requires a Metal device"]
    fn metal_gpt_neox_matches_flex() {
        super::gpt_neox_matches_flex::<Metal>(&Default::default(), metal_bound());
    }

    /// Metal: the gated delta rule's recurrent, snapshot, and chunked kernels.
    #[test]
    #[ignore = "requires a Metal device"]
    fn metal_delta_rule_kernels_match_flex() {
        super::delta_rule_kernels_match_flex::<Metal>(&Default::default(), metal_bound());
    }
}

#[cfg(feature = "cuda")]
mod cuda {
    //! CUDA against Flex/F32 with identical weights.
    //!
    //! Each test probes the device first: a GPU that multiplies F32 matrices
    //! in F32 (compute capability below 8.0) is held to [`super::F32_BOUND`],
    //! one that rounds their inputs to TF32 (8.0 or newer) to the TF32 bound
    //! of the support module.

    use burn::backend::Cuda;

    use super::cuda_bound;

    /// CUDA: every Llama-family variant.
    #[test]
    #[ignore = "requires a CUDA device"]
    fn cuda_llama_family_matches_flex() {
        super::llama_family_matches_flex::<Cuda>(&Default::default(), cuda_bound());
    }

    /// CUDA: Gemma 2 and Gemma 3 text.
    #[test]
    #[ignore = "requires a CUDA device"]
    fn cuda_gemma_matches_flex() {
        super::gemma_matches_flex::<Cuda>(&Default::default(), cuda_bound());
    }

    /// CUDA: gpt-oss.
    #[test]
    #[ignore = "requires a CUDA device"]
    fn cuda_gpt_oss_matches_flex() {
        super::gpt_oss_matches_flex::<Cuda>(&Default::default(), cuda_bound());
    }

    /// CUDA: Qwen3.5, Qwen3-MoE, DeepSeek-V2, Muse Glimmer, and Nemotron-H.
    #[test]
    #[ignore = "requires a CUDA device"]
    fn cuda_hybrid_registry_families_match_flex() {
        super::hybrid_registry_families_match_flex::<Cuda>(&Default::default(), cuda_bound());
    }

    /// CUDA: both modular causal decoders.
    #[test]
    #[ignore = "requires a CUDA device"]
    fn cuda_modular_causal_matches_flex() {
        super::modular_causal_matches_flex::<Cuda>(&Default::default(), cuda_bound());
    }

    /// CUDA: the modular encoder-decoder.
    #[test]
    #[ignore = "requires a CUDA device"]
    fn cuda_modular_seq2seq_matches_flex() {
        super::modular_seq2seq_matches_flex::<Cuda>(&Default::default(), cuda_bound());
    }

    /// CUDA: T5.
    #[test]
    #[ignore = "requires a CUDA device"]
    fn cuda_t5_matches_flex() {
        super::t5_matches_flex::<Cuda>(&Default::default(), cuda_bound());
    }

    /// CUDA: GPT-2.
    #[test]
    #[ignore = "requires a CUDA device"]
    fn cuda_gpt2_matches_flex() {
        super::gpt2_matches_flex::<Cuda>(&Default::default(), cuda_bound());
    }

    /// CUDA: GPT-NeoX.
    #[test]
    #[ignore = "requires a CUDA device"]
    fn cuda_gpt_neox_matches_flex() {
        super::gpt_neox_matches_flex::<Cuda>(&Default::default(), cuda_bound());
    }

    /// CUDA: the gated delta rule's recurrent, snapshot, and chunked kernels.
    #[test]
    #[ignore = "requires a CUDA device"]
    fn cuda_delta_rule_kernels_match_flex() {
        super::delta_rule_kernels_match_flex::<Cuda>(&Default::default(), cuda_bound());
    }
}

/// The CubeCL CPU bound: the backend must multiply F32 matrices in F32, so [`F32_BOUND`] applies.
#[cfg(feature = "cpu")]
fn cpu_bound() -> Bound {
    let precision = MatmulPrecision::probe::<burn::backend::Cpu>(&Default::default());
    assert_eq!(
        precision,
        MatmulPrecision::F32,
        "the CubeCL CPU backend is expected to multiply F32 matrices in F32"
    );
    F32_BOUND
}

#[cfg(feature = "cpu")]
mod cpu {
    //! Burn's CubeCL CPU backend against Flex/F32 with identical weights.
    //!
    //! The CubeCL CPU backend lowers the same `burn-cubecl` kernels as Metal,
    //! CUDA, and ROCm through MLIR and LLVM to host code, so it exercises the
    //! accelerators' tensor layouts, reshapes, fused attention, cache
    //! scatters, and routing kernels without a GPU. It multiplies F32 matrices
    //! in F32 (each test probes that first), so every family is held to
    //! [`super::F32_BOUND`], as on Metal.
    //!
    //! # The DeltaNet reshape guard
    //!
    //! The backend shares the GPU backends' reshape decisions, so it also
    //! reproduces on the CPU the reshape defect that `trailing_unit_axis` in
    //! `src/delta_net.rs` works around (unsqueezing a strided token slice
    //! behind its unit axis reads the wrong rows):
    //! [`cpu_delta_rule_kernels_match_flex`] fails if the workaround is
    //! reverted or a kernel reintroduces such a reshape.
    //!
    //! # A CubeCL CPU copy defect
    //!
    //! Burn 0.21's CubeCL CPU backend (cubecl-std 0.10) copies a reshape of a
    //! view whose innermost stride is not one (a transposed or expanded view)
    //! wrongly: its CPU-only copy kernel reads the input at the output's
    //! coordinates, which is right only when the shapes agree. The GPU
    //! backends take the general copy and are unaffected; cubecl-std 0.11
    //! (Burn 0.22) fixes it. `docs/upstream/15-cubecl-cpu-reshape-copy.md`
    //! records it; [`cpu_reshaped_strided_views_match_flex`] is its ignored
    //! reproducer. It is not worked around in the crate: reshaping a permuted
    //! or broadcast view is the ordinary way to materialize it, used
    //! throughout the crate, and the defect is confined to an unsupported
    //! backend. Two cases meet it and are ignored until the pin moves:
    //! Nemotron-H ([`cpu_nemotron_h_matches_flex`]: its group-limited
    //! router flattens an expanded group mask) and the modular causal
    //! decoders ([`cpu_modular_causal_matches_flex`]: multi-head latent
    //! attention de-interleaves rotary pairs, and the masked gated DeltaNet
    //! with grouped key heads copies a strided view). With cubecl-std's fix
    //! applied both pass the F32 bound.
    //!
    //! # Running
    //!
    //! Only [`cpu_delta_rule_kernels_match_flex`] is an ordinary test: wherever
    //! the `cpu` feature builds, `cargo test --features cpu` runs it (60 to
    //! 80 s in a debug build on the machine below). The family cases are
    //! ignored for their run time: the backend compiles every kernel shape
    //! with LLVM when it is first launched, and on an Apple M3 Max (debug
    //! build, one test thread, other jobs running) the nine passing cases took
    //! 405 to 420 s over two runs, 277 to 290 s of it the four hybrid
    //! families, and the CubeCL CPU modules of the three tiny-golden suites
    //! another 73 s, far beyond a two-minute budget for ordinary tests.
    //! Parallel test threads gave no speed-up (441 s for the parity cases),
    //! since every test's CubeCL CPU runtime occupies all cores, and a release
    //! build saved little (313 s on parallel threads). Run every passing case,
    //! without the known failures:
    //!
    //! ```text
    //! cargo test --locked --features cpu --test backend_parity cpu:: -- --include-ignored \
    //!   --skip cpu_nemotron_h_matches_flex --skip cpu_modular_causal_matches_flex \
    //!   --skip cpu_reshaped_strided_views_match_flex --nocapture --test-threads 1
    //! ```
    //!
    //! CI runs the same selection on `ubuntu-latest` (`.github/workflows/ci.yml`).

    use burn::{
        backend::{Cpu, Flex},
        tensor::{Int, Tensor, TensorData, backend::Backend},
    };
    use transformers::position::interleaved_to_split_half;

    use super::{RegistryConfig, cpu_bound, host};

    /// The hybrid registry cases whose names satisfy `keep`; panics when none does.
    fn hybrid_cases_where(keep: impl Fn(&str) -> bool) -> Vec<(&'static str, RegistryConfig)> {
        let cases: Vec<_> = super::hybrid_registry_cases()
            .into_iter()
            .filter(|(name, _)| keep(name))
            .collect();
        assert!(!cases.is_empty(), "no hybrid registry case was selected");
        cases
    }

    /// CubeCL CPU: every Llama-family variant.
    #[test]
    #[ignore = "slow: the CubeCL CPU backend compiles every kernel at run time; run with --include-ignored (module documentation)"]
    fn cpu_llama_family_matches_flex() {
        super::llama_family_matches_flex::<Cpu>(&Default::default(), cpu_bound());
    }

    /// CubeCL CPU: Gemma 2 and Gemma 3 text.
    #[test]
    #[ignore = "slow: the CubeCL CPU backend compiles every kernel at run time; run with --include-ignored (module documentation)"]
    fn cpu_gemma_matches_flex() {
        super::gemma_matches_flex::<Cpu>(&Default::default(), cpu_bound());
    }

    /// CubeCL CPU: gpt-oss.
    #[test]
    #[ignore = "slow: the CubeCL CPU backend compiles every kernel at run time; run with --include-ignored (module documentation)"]
    fn cpu_gpt_oss_matches_flex() {
        super::gpt_oss_matches_flex::<Cpu>(&Default::default(), cpu_bound());
    }

    /// CubeCL CPU: Qwen3.5, Qwen3-MoE, DeepSeek-V2, and Muse Glimmer.
    ///
    /// The hybrid families of `hybrid_registry_families_match_flex` other
    /// than Nemotron-H, which [`cpu_nemotron_h_matches_flex`] covers.
    #[test]
    #[ignore = "slow: the CubeCL CPU backend compiles every kernel at run time; run with --include-ignored (module documentation)"]
    fn cpu_hybrid_registry_families_match_flex() {
        super::check_registry::<Cpu>(
            hybrid_cases_where(|name| name != "Nemotron-H"),
            &Default::default(),
            cpu_bound(),
        );
    }

    /// CubeCL CPU: Nemotron-H; fails on Burn 0.21 through the CubeCL CPU copy defect.
    ///
    /// Group-limited routing flattens an expanded `[tokens, groups, 1]`
    /// group mask to `[tokens, experts]`; the defective copy marks the wrong
    /// experts eligible, so routing differs from Flex.
    #[test]
    #[ignore = "Burn 0.21's CubeCL CPU copies reshaped strided views wrongly; see docs/upstream/15-cubecl-cpu-reshape-copy.md"]
    fn cpu_nemotron_h_matches_flex() {
        super::check_registry::<Cpu>(
            hybrid_cases_where(|name| name == "Nemotron-H"),
            &Default::default(),
            cpu_bound(),
        );
    }

    /// CubeCL CPU: both modular causal decoders; fails on Burn 0.21 through the CubeCL CPU copy defect.
    #[test]
    #[ignore = "Burn 0.21's CubeCL CPU copies reshaped strided views wrongly; see docs/upstream/15-cubecl-cpu-reshape-copy.md"]
    fn cpu_modular_causal_matches_flex() {
        super::modular_causal_matches_flex::<Cpu>(&Default::default(), cpu_bound());
    }

    /// CubeCL CPU: the modular encoder-decoder.
    #[test]
    #[ignore = "slow: the CubeCL CPU backend compiles every kernel at run time; run with --include-ignored (module documentation)"]
    fn cpu_modular_seq2seq_matches_flex() {
        super::modular_seq2seq_matches_flex::<Cpu>(&Default::default(), cpu_bound());
    }

    /// CubeCL CPU: T5.
    #[test]
    #[ignore = "slow: the CubeCL CPU backend compiles every kernel at run time; run with --include-ignored (module documentation)"]
    fn cpu_t5_matches_flex() {
        super::t5_matches_flex::<Cpu>(&Default::default(), cpu_bound());
    }

    /// CubeCL CPU: GPT-2.
    #[test]
    #[ignore = "slow: the CubeCL CPU backend compiles every kernel at run time; run with --include-ignored (module documentation)"]
    fn cpu_gpt2_matches_flex() {
        super::gpt2_matches_flex::<Cpu>(&Default::default(), cpu_bound());
    }

    /// CubeCL CPU: GPT-NeoX.
    #[test]
    #[ignore = "slow: the CubeCL CPU backend compiles every kernel at run time; run with --include-ignored (module documentation)"]
    fn cpu_gpt_neox_matches_flex() {
        super::gpt_neox_matches_flex::<Cpu>(&Default::default(), cpu_bound());
    }

    /// CubeCL CPU: the gated delta rule's recurrent, snapshot, and chunked kernels.
    #[test]
    fn cpu_delta_rule_kernels_match_flex() {
        super::delta_rule_kernels_match_flex::<Cpu>(&Default::default(), cpu_bound());
    }

    /// Three reshapes of strided views, each evaluated on `B`, row-major.
    ///
    /// A transposed `[2, 3]` matrix flattened to `[6]`; the crate's
    /// [`interleaved_to_split_half`] of eight rotary channels (a reshape of a
    /// swapped pair axis); and a `[2, 2, 1]` column expanded to `[2, 2, 3]`
    /// and flattened to `[2, 6]`. Every input's innermost stride differs from
    /// one, and every reshape changes the shape, so each needs a copy.
    fn reshaped_strided_views<B: Backend>(device: &B::Device) -> [Vec<f32>; 3] {
        let ramp = |count: usize| (0..count).map(|value| value as f32).collect::<Vec<_>>();
        let matrix = Tensor::<B, 2>::from_data(TensorData::new(ramp(6), [2, 3]), device);
        let pairs = Tensor::<B, 4>::from_data(TensorData::new(ramp(8), [1, 1, 1, 8]), device);
        let column = Tensor::<B, 1, Int>::from_data(TensorData::new(vec![0, 1, 1, 0], [4]), device)
            .float()
            .reshape([2, 2, 1]);
        [
            host(matrix.swap_dims(0, 1).reshape([6])),
            host(interleaved_to_split_half(pairs).expect("an even rotary width")),
            host(column.expand([2, 2, 3]).reshape([2, 6])),
        ]
    }

    /// Reshaped strided views match Flex; the reproducer of the CubeCL CPU copy defect.
    ///
    /// Fails on Burn 0.21 (cubecl-std 0.10) and passes with cubecl-std
    /// 0.11's fix; `docs/upstream/15-cubecl-cpu-reshape-copy.md` records the
    /// measured outputs.
    #[test]
    #[ignore = "reproduces Burn 0.21's CubeCL CPU reshape-copy defect; see docs/upstream/15-cubecl-cpu-reshape-copy.md"]
    fn cpu_reshaped_strided_views_match_flex() {
        let expected = reshaped_strided_views::<Flex<f32, i32>>(&Default::default());
        let actual = reshaped_strided_views::<Cpu>(&Default::default());
        assert_eq!(actual, expected);
    }
}
