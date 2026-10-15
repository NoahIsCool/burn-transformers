//! Modular graphs configured like family blocks reproduce the families' outputs (TEST-1).
//!
//! Each test builds a tiny family model with deterministic weights, an
//! equivalent [`DecoderSpec`] or [`Seq2SeqSpec`], and copies every parameter
//! with [`TransferPlan`] path mappings (no tensor is transformed). The two
//! graphs must then agree on logits within F32 reassociation tolerance for
//! unpadded, right-padded, and explicit-position inputs, and on cached
//! decoding. This verifies that the modular path recombines the same shared
//! mechanisms the families ship: per-head Q/K norms and RoPE (Qwen3-MoE),
//! routed and shared experts, multi-head latent attention with its decoupled
//! rotary sub-head (DeepSeek-V2), zero-centered norms, sigmoid-gated
//! attention, and Gated DeltaNet (Qwen3.5), and T5's unscaled attention,
//! learned relative buckets, ungated ReLU MLP, and unscaled cross-attention.
//!
//! The comparison tolerance is `1e-5` absolute on logits of magnitude below
//! about two: both graphs evaluate the same F32 operations, differing only in
//! summation order (for example folded versus per-head attention matmuls).
//!
//! Fixtures are sized so the comparison can fail: every rotary sub-head has at
//! least two pairs with base-dependent frequencies that turn measurably within
//! the inputs' positions. Negative controls build deliberately wrong modular
//! twins (wrong rotary base, unscaled or near-miss RoPE scaling, a missing
//! score correction) and require them to differ from the family by at least
//! one hundred times the tolerance.

use burn::{
    module::Module,
    tensor::{Int, Tensor, TensorData, Tolerance, backend::BackendTypes},
};

use super::*;
use crate::{
    CausalLmOutput,
    architecture::{
        AttentionOptions, CrossAttentionSpec, DecoderLayerSpec, DenseFeedForwardSpec, Seq2SeqSpec,
    },
    models::{
        config::{RopeParameters, RopeType},
        deepseek_v2::{
            DeepseekScoringFunction, DeepseekTopKMethod, DeepseekV2Config, DeepseekV2ForCausalLm,
        },
        qwen3_5::{Qwen3_5ForCausalLm, Qwen3_5LayerType, Qwen3_5TextConfig},
        qwen3_moe::{Qwen3MoeConfig, Qwen3MoeForCausalLm},
        t5::{T5Config, T5FeedForwardProjection, T5ForConditionalGeneration},
    },
    transfer::TransferPlan,
};

/// Device used by every parity fixture.
type Device = <TestBackend as BackendTypes>::Device;

/// Absolute logit tolerance for F32 graphs that differ only in summation order.
const LOGIT_TOLERANCE: f32 = 1e-5;

/// Copies every parameter of `source` into `target` by path prefix, requiring full coverage.
fn transferred<S: Module<TestBackend>, T: Module<TestBackend>>(
    source: &S,
    target: &T,
    prefixes: &[(String, String)],
) -> T {
    let plan = prefixes
        .iter()
        .fold(TransferPlan::new(), |plan, (from, to)| {
            plan.copy_prefix(from.as_str(), to.as_str())
        });
    let (model, report) = plan.apply(source, target).unwrap();
    assert!(
        report.newly_initialized.is_empty(),
        "every modular parameter must come from the family: {:?}",
        report.newly_initialized
    );
    assert!(
        report.omitted.is_empty(),
        "every family parameter must be used: {:?}",
        report.omitted
    );
    model
}

/// Owned `(family, modular)` path-prefix pairs.
fn prefixes(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
    pairs
        .iter()
        .map(|(from, to)| ((*from).to_owned(), (*to).to_owned()))
        .collect()
}

/// A `[2, 5]` integer tensor from host rows.
fn grid(rows: [[i32; 5]; 2], device: &Device) -> Tensor<TestBackend, 2, Int> {
    Tensor::from_data(TensorData::new(rows.concat(), [2, 5]), device)
}

/// Unpadded, right-padded, and explicit gapped-position inputs over a 32-token vocabulary.
fn causal_inputs(device: &Device) -> Vec<CausalLmInput<TestBackend>> {
    let tokens = grid([[3, 17, 5, 9, 30], [8, 1, 22, 4, 11]], device);
    vec![
        CausalLmInput {
            token_ids: tokens.clone(),
            token_mask: None,
            position_ids: None,
        },
        CausalLmInput {
            token_ids: tokens.clone(),
            token_mask: Some(mask(
                [
                    [true, true, true, false, false],
                    [true, true, true, true, true],
                ],
                device,
            )),
            position_ids: None,
        },
        CausalLmInput {
            token_ids: tokens,
            token_mask: None,
            position_ids: Some(grid([[0, 2, 5, 6, 19], [1, 2, 3, 7, 12]], device)),
        },
    ]
}

/// Every-position logits, zeroed at padded tokens so padding outputs never compare.
fn valid_logits(
    output: CausalLmOutput<TestBackend>,
    input: &CausalLmInput<TestBackend>,
) -> TensorData {
    zero_padding(output.logits, input)
}

/// Zeroes `[batch, token, channel]` values at padded tokens of `input`.
fn zero_padding(values: Tensor<TestBackend, 3>, input: &CausalLmInput<TestBackend>) -> TensorData {
    match &input.token_mask {
        Some(mask) => (values * mask.as_tensor().clone().float().unsqueeze_dim::<3>(2)).into_data(),
        None => values.into_data(),
    }
}

/// Options requesting logits at every position.
fn all_logits() -> ForwardOptions {
    ForwardOptions {
        logits: LogitsSelection::All,
        ..Default::default()
    }
}

/// Requires two causal models to agree on every input and on one cached decode step.
fn assert_causal_parity<F: CausalLanguageModel<TestBackend>>(
    family: &F,
    custom: &ModularCausalLm<TestBackend>,
    device: &Device,
) {
    for input in causal_inputs(device) {
        let expected = valid_logits(
            family.forward(input.clone(), &all_logits()).unwrap(),
            &input,
        );
        let actual = valid_logits(
            custom.forward(input.clone(), &all_logits()).unwrap(),
            &input,
        );
        actual.assert_approx_eq::<f32>(&expected, Tolerance::absolute(LOGIT_TOLERANCE));
    }
    let config = CacheConfig::new(2, 8).unwrap();
    let mut family_state = family.new_decode_state(config, 2, device).unwrap();
    let mut custom_state = custom.new_decode_state(config, 2, device).unwrap();
    let prompt = CausalLmInput {
        token_ids: Tensor::from_ints([[3, 17, 5], [8, 1, 22]], device),
        token_mask: None,
        position_ids: None,
    };
    let step = || DecodeInput {
        token_ids: Tensor::from_ints([[9], [4]], device),
        token_mask: None,
        position_ids: None,
    };
    let options = ForwardOptions::default();
    let expected = family
        .prefill(prompt.clone(), &mut family_state, &options)
        .unwrap();
    let actual = custom.prefill(prompt, &mut custom_state, &options).unwrap();
    actual.logits.into_data().assert_approx_eq::<f32>(
        &expected.logits.into_data(),
        Tolerance::absolute(LOGIT_TOLERANCE),
    );
    let expected = family.decode(step(), &mut family_state, &options).unwrap();
    let actual = custom.decode(step(), &mut custom_state, &options).unwrap();
    actual.logits.into_data().assert_approx_eq::<f32>(
        &expected.logits.into_data(),
        Tolerance::absolute(LOGIT_TOLERANCE),
    );
}

/// Largest absolute difference between two causal models' valid logits over [`causal_inputs`].
///
/// Negative controls use it to show that a deliberately wrong modular twin is
/// separated from its family by far more than [`LOGIT_TOLERANCE`], so a parity
/// test can actually detect that class of translation error.
fn max_logit_gap<F: CausalLanguageModel<TestBackend>>(
    family: &F,
    custom: &ModularCausalLm<TestBackend>,
    device: &Device,
) -> f32 {
    causal_inputs(device)
        .into_iter()
        .map(|input| {
            let expected = valid_logits(
                family.forward(input.clone(), &all_logits()).unwrap(),
                &input,
            )
            .to_vec::<f32>()
            .unwrap();
            let actual = valid_logits(
                custom.forward(input.clone(), &all_logits()).unwrap(),
                &input,
            )
            .to_vec::<f32>()
            .unwrap();
            expected
                .iter()
                .zip(&actual)
                .map(|(left, right)| (left - right).abs())
                .fold(0.0, f32::max)
        })
        .fold(0.0, f32::max)
}

/// Smallest logit gap a negative control must produce: 100 times [`LOGIT_TOLERANCE`].
///
/// A mutation that moved logits by less would sit within two orders of
/// magnitude of the pass/fail boundary, where a slightly different fixture
/// could hide it behind summation-order noise.
const DETECTABLE_GAP: f32 = 100.0 * LOGIT_TOLERANCE;

/// Requires a deliberately wrong modular twin to differ from `family` by at least [`DETECTABLE_GAP`].
fn assert_detectable<F: CausalLanguageModel<TestBackend>>(
    family: &F,
    custom: &ModularCausalLm<TestBackend>,
    mutation: &str,
) {
    let gap = max_logit_gap(family, custom, &Default::default());
    assert!(
        gap >= DETECTABLE_GAP,
        "{mutation}: logit gap {gap:e} is below {DETECTABLE_GAP:e}, so parity could not detect it"
    );
}

/// RMS normalization with the given stabilizer and plain (not zero-centered) scales.
fn rms_with(epsilon: f64) -> NormalizationSpec {
    NormalizationSpec::Rms {
        epsilon,
        zero_centered: false,
    }
}

/// A pre-normalized block over `mixer` and `feed_forward` with one RMS convention.
fn pre_norm_block(
    mixer: MixerSpec,
    feed_forward: FeedForwardSpec,
    norm: NormalizationSpec,
) -> DecoderLayerSpec {
    DecoderLayerSpec {
        mixer,
        feed_forward,
        mixer_norm: norm,
        feed_forward_norm: norm,
        norm_placement: NormPlacement::Pre,
        residual: ResidualTopology::Add,
    }
}

/// Rotary base of the Qwen3-MoE fixtures.
///
/// Smaller than the checkpoint's `1e6` so that every rotary pair turns
/// measurably within the fixtures' ten positions: the four inverse
/// frequencies are `1, 0.1, 0.01, 0.001`, and the scaled low-frequency pairs
/// move logits by much more than [`DETECTABLE_GAP`] when mis-scaled.
const QWEN3_MOE_THETA: f64 = 10_000.0;

/// Query/key head width of the Qwen3-MoE fixtures: four rotary pairs with distinct frequencies.
const QWEN3_MOE_HEAD_DIM: usize = 8;

/// Context limit of the Qwen3-MoE fixtures.
///
/// Twice the YaRN case's original context, so a translator that derives the
/// YaRN factor as `max_position_embeddings / original` (as recent Hugging Face
/// releases do when the original context is serialized) agrees with the
/// explicit factor of two.
const QWEN3_MOE_MAX_POSITIONS: usize = 128;

/// Builds a one-block Qwen3-MoE model and its modular twin with transferred weights.
///
/// `rope` is the family's serialized RoPE metadata and `scaling` the modular
/// spec policy; the pair agrees exactly when `scaling` translates to the same
/// runtime frequencies and amplitude.
fn qwen3_moe_pair(
    rope: Option<RopeParameters>,
    scaling: RopeScaling,
) -> (
    Qwen3MoeForCausalLm<TestBackend>,
    ModularCausalLm<TestBackend>,
) {
    let device = Default::default();
    let config = Qwen3MoeConfig {
        vocab_size: 32,
        hidden_size: 8,
        intermediate_size: 12,
        num_hidden_layers: 1,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: QWEN3_MOE_HEAD_DIM,
        max_position_embeddings: QWEN3_MOE_MAX_POSITIONS,
        rope_theta: Some(QWEN3_MOE_THETA),
        max_window_layers: 1,
        decoder_sparse_step: 1,
        moe_intermediate_size: 6,
        num_experts: 4,
        num_experts_per_tok: 2,
        norm_topk_prob: true,
        rope_parameters: rope,
        ..Default::default()
    };
    let norm = rms_with(config.rms_norm_eps);
    let family = Qwen3MoeForCausalLm::<TestBackend>::init(config, &device)
        .unwrap()
        .map(&mut DeterministicParameters::default());
    let spec = DecoderSpec {
        vocabulary_size: 32,
        hidden_size: 8,
        layers: vec![pre_norm_block(
            MixerSpec::Attention(AttentionSpec {
                query_heads: 2,
                key_value_heads: 1,
                head_dim: QWEN3_MOE_HEAD_DIM,
                query_key_norm: true,
                output_gate: false,
                position: PositionEncodingSpec::Rope {
                    theta: QWEN3_MOE_THETA,
                    rotary_dim: QWEN3_MOE_HEAD_DIM,
                    scaling,
                },
            }),
            FeedForwardSpec::MixtureOfExperts(MixtureOfExpertsSpec {
                routed_experts: 4,
                experts_per_token: 2,
                shared_experts: 0,
                expert_intermediate_size: 6,
                activation: Activation::Silu,
                renormalize_selected: true,
                routed_scale: 1.0,
                routing: RoutingPolicy::default(),
                shared_expert_gate: false,
                expert_bias: false,
            }),
            norm,
        )],
        final_norm: norm,
        tie_word_embeddings: false,
        max_position_embeddings: QWEN3_MOE_MAX_POSITIONS,
    };
    let custom = transferred(
        &family,
        &ModularCausalLm::<TestBackend>::init(spec, &device).unwrap(),
        &prefixes(&[
            ("model.embed_tokens", "embed_tokens"),
            ("model.norm", "final_norm.Rms"),
            ("lm_head", "lm_head"),
            (
                "model.layers.0.input_layernorm",
                "layers.0.mixer_norm.before.Rms",
            ),
            (
                "model.layers.0.post_attention_layernorm",
                "layers.0.feed_forward_norm.before.Rms",
            ),
            (
                "model.layers.0.self_attn",
                "layers.0.mixer.Attention.attention",
            ),
            (
                "model.layers.0.sparse_mlp",
                "layers.0.feed_forward.MixtureOfExperts",
            ),
        ]),
    );
    (family, custom)
}

/// Builds the Qwen3-MoE pair for `rope`/`scaling` and requires full parity.
fn assert_qwen3_moe_parity(rope: Option<RopeParameters>, scaling: RopeScaling) {
    let (family, custom) = qwen3_moe_pair(rope, scaling);
    assert_causal_parity(&family, &custom, &Default::default());
}

/// Serialized Hugging Face RoPE metadata of one scaled Qwen3-MoE fixture.
///
/// Every case doubles the context. Following Transformers 5.2, linear RoPE
/// reads no original context and dynamic NTK scales against the family's
/// `max_position_embeddings`. Qwen3-MoE rejects positions at or beyond that
/// bound, so its dynamic-NTK base never grows: the case pins that the modular
/// translation agrees, and the negative controls skip it. YaRN is static
/// and uses an original context of 64: its correction range is then the
/// fractional pair interval `[0.41, 1.01]`, truncated to `[0, 2]`, so the ramp
/// blends pair 1 halfway between extrapolated and interpolated frequencies.
fn qwen3_moe_rope_parameters(rope_type: RopeType) -> RopeParameters {
    let yarn = rope_type == RopeType::Yarn;
    RopeParameters {
        rope_type,
        factor: Some(2.0),
        original_max_position_embeddings: yarn.then_some(64),
        attention_factor: yarn.then_some(1.25),
        beta_fast: yarn.then_some(4.0),
        beta_slow: yarn.then_some(1.0),
        ..Default::default()
    }
}

/// A modular YaRN policy over the YaRN fixture's factor, original context, and betas.
///
/// `qwen3_moe_yarn(Some(1.25), true)` matches
/// [`qwen3_moe_rope_parameters`]`(RopeType::Yarn)`; other arguments are near-misses.
fn qwen3_moe_yarn(attention_factor: Option<f64>, truncate: bool) -> RopeScaling {
    RopeScaling::Yarn {
        factor: 2.0,
        original_max_positions: 64,
        attention_factor,
        beta_slow: 1.0,
        beta_fast: 4.0,
        truncate,
    }
}

/// Every scaled family case paired with the modular policy that reproduces it.
fn qwen3_moe_scaled_cases() -> [(RopeType, RopeScaling); 3] {
    [
        (RopeType::Linear, RopeScaling::Linear { factor: 2.0 }),
        (
            RopeType::Dynamic,
            RopeScaling::DynamicNtk {
                factor: 2.0,
                original_max_positions: QWEN3_MOE_MAX_POSITIONS,
            },
        ),
        (RopeType::Yarn, qwen3_moe_yarn(Some(1.25), true)),
    ]
}

/// Qwen3-MoE: per-head Q/K RMS norms, full-head RoPE, and renormalized top-k experts.
#[test]
fn qwen3_moe_block_matches_the_family() {
    assert_qwen3_moe_parity(None, RopeScaling::None);
}

/// Linear, dynamic-NTK, and YaRN scaling translate to the family's RoPE frequencies.
#[test]
fn qwen3_moe_rope_scaling_matches_the_family() {
    for (rope_type, scaling) in qwen3_moe_scaled_cases() {
        assert_qwen3_moe_parity(Some(qwen3_moe_rope_parameters(rope_type)), scaling);
    }
}

/// Negative controls: unscaled or near-miss modular RoPE is far outside the parity tolerance.
///
/// Each statically scaled family is compared with an unscaled twin and with
/// the near-misses a translation could plausibly make: a wrong linear factor,
/// untruncated YaRN correction bounds, and YaRN's default amplitude instead of
/// the serialized one. Each must exceed [`DETECTABLE_GAP`], which proves
/// [`qwen3_moe_rope_scaling_matches_the_family`] can detect it. Dynamic NTK is
/// excluded: within Qwen3-MoE's position bound it equals the unscaled model.
#[test]
fn qwen3_moe_rope_parity_detects_wrong_scaling() {
    for (rope_type, _) in qwen3_moe_scaled_cases() {
        if rope_type == RopeType::Dynamic {
            continue;
        }
        let (family, custom) = qwen3_moe_pair(
            Some(qwen3_moe_rope_parameters(rope_type)),
            RopeScaling::None,
        );
        assert_detectable(
            &family,
            &custom,
            &format!("{rope_type:?} family against an unscaled twin"),
        );
    }
    let near_misses = [
        (
            RopeType::Linear,
            RopeScaling::Linear { factor: 1.5 },
            "linear factor 1.5 instead of 2",
        ),
        (
            RopeType::Yarn,
            qwen3_moe_yarn(Some(1.25), false),
            "untruncated YaRN correction bounds",
        ),
        (
            RopeType::Yarn,
            qwen3_moe_yarn(None, true),
            "default YaRN amplitude instead of 1.25",
        ),
    ];
    for (rope_type, scaling, mutation) in near_misses {
        let (family, custom) = qwen3_moe_pair(Some(qwen3_moe_rope_parameters(rope_type)), scaling);
        assert_detectable(&family, &custom, mutation);
    }
}

/// Rotary width of the DeepSeek-V2 fixtures' decoupled sub-head: two pairs.
///
/// A single pair would rotate at frequency `theta^0 = 1` for every base and
/// width, so the second pair (frequency `theta^(-1/2)`) is what makes the
/// fixtures sensitive to the rotary base and to the rotary-width plumbing.
const DEEPSEEK_ROPE_HEAD_DIM: usize = 4;

/// Content (non-rotary) query/key width of the DeepSeek-V2 fixtures.
const DEEPSEEK_CONTENT_HEAD_DIM: usize = 3;

/// The rotary policy of one DeepSeek-V2 fixture and its modular translation.
struct LatentRope {
    /// Family metadata (`config.rope_scaling`); `None` is unscaled RoPE.
    family: Option<RopeParameters>,
    /// Modular rotary base; the family always uses `10_000`.
    theta: f64,
    /// Modular frequency scaling.
    scaling: RopeScaling,
    /// Modular score multiplier for the summed content and rotary logits.
    score_scale: ScoreScale,
}

impl LatentRope {
    /// Unscaled RoPE with base `10_000` and the conventional `1/sqrt(d_qk)` scale.
    fn unscaled() -> Self {
        Self {
            family: None,
            theta: 10_000.0,
            scaling: RopeScaling::None,
            score_scale: ScoreScale::Auto,
        }
    }
}

/// DeepSeek's YaRN magnitude `1 + 0.1 m ln(factor)` (one for factors up to one).
fn deepseek_yarn_magnitude(factor: f64, multiplier: f64) -> f64 {
    if factor <= 1.0 {
        1.0
    } else {
        1.0 + 0.1 * multiplier * factor.ln()
    }
}

/// Builds a two-block DeepSeek-V2 model and its modular twin with transferred weights.
///
/// Block 0 is latent attention with a dense SwiGLU MLP; block 1 is latent
/// attention with one shared and four routed experts. Both use the rotary
/// policy `rope`.
fn deepseek_v2_pair(
    rope: LatentRope,
) -> (
    DeepseekV2ForCausalLm<TestBackend>,
    ModularCausalLm<TestBackend>,
) {
    let device = Default::default();
    let config = DeepseekV2Config {
        vocab_size: 32,
        hidden_size: 8,
        intermediate_size: 12,
        moe_intermediate_size: 6,
        num_hidden_layers: 2,
        num_attention_heads: 2,
        num_key_value_heads: 2,
        n_shared_experts: 1,
        n_routed_experts: 4,
        routed_scaling_factor: 1.0,
        kv_lora_rank: 5,
        q_lora_rank: None,
        qk_nope_head_dim: DEEPSEEK_CONTENT_HEAD_DIM,
        qk_rope_head_dim: DEEPSEEK_ROPE_HEAD_DIM,
        v_head_dim: 4,
        topk_method: DeepseekTopKMethod::Greedy,
        n_group: 1,
        topk_group: 1,
        num_experts_per_tok: 2,
        moe_layer_freq: 1,
        first_k_dense_replace: 1,
        norm_topk_prob: false,
        scoring_func: DeepseekScoringFunction::Softmax,
        hidden_act: "silu".into(),
        max_position_embeddings: 32,
        rms_norm_eps: 1e-6,
        tie_word_embeddings: false,
        rope_theta: 10_000.0,
        rope_scaling: rope.family,
        pad_token_id: None,
        bos_token_id: None,
        eos_token_id: None,
        ..Default::default()
    };
    let family = DeepseekV2ForCausalLm::<TestBackend>::init(config, &device)
        .unwrap()
        .map(&mut DeterministicParameters::default());
    let mla = MixerSpec::Mla(MlaSpec {
        query_heads: 2,
        content_head_dim: DEEPSEEK_CONTENT_HEAD_DIM,
        rotary_head_dim: DEEPSEEK_ROPE_HEAD_DIM,
        value_head_dim: 4,
        key_value_lora_rank: 5,
        query_lora_rank: None,
        position: PositionEncodingSpec::Rope {
            theta: rope.theta,
            rotary_dim: DEEPSEEK_ROPE_HEAD_DIM,
            scaling: rope.scaling,
        },
        score_scale: rope.score_scale,
        connectivity: AttentionConnectivity::Global,
    });
    let norm = rms_with(1e-6);
    let spec = DecoderSpec {
        vocabulary_size: 32,
        hidden_size: 8,
        layers: vec![
            pre_norm_block(
                mla.clone(),
                FeedForwardSpec::Dense(DenseFeedForwardSpec {
                    intermediate_size: 12,
                    activation: Activation::Silu,
                    bias: false,
                }),
                norm,
            ),
            pre_norm_block(
                mla,
                FeedForwardSpec::MixtureOfExperts(MixtureOfExpertsSpec {
                    routed_experts: 4,
                    experts_per_token: 2,
                    shared_experts: 1,
                    expert_intermediate_size: 6,
                    activation: Activation::Silu,
                    renormalize_selected: false,
                    routed_scale: 1.0,
                    routing: RoutingPolicy::default(),
                    shared_expert_gate: false,
                    expert_bias: false,
                }),
                norm,
            ),
        ],
        final_norm: norm,
        tie_word_embeddings: false,
        max_position_embeddings: 32,
    };
    let custom = transferred(
        &family,
        &ModularCausalLm::<TestBackend>::init(spec, &device).unwrap(),
        &prefixes(&[
            ("model.embed_tokens", "embed_tokens"),
            ("model.norm", "final_norm.Rms"),
            ("lm_head", "lm_head"),
            (
                "model.layers.0.input_layernorm",
                "layers.0.mixer_norm.before.Rms",
            ),
            (
                "model.layers.0.post_attention_layernorm",
                "layers.0.feed_forward_norm.before.Rms",
            ),
            ("model.layers.0.self_attn", "layers.0.mixer.Mla.attention"),
            ("model.layers.0.dense_mlp", "layers.0.feed_forward.Dense"),
            (
                "model.layers.1.input_layernorm",
                "layers.1.mixer_norm.before.Rms",
            ),
            (
                "model.layers.1.post_attention_layernorm",
                "layers.1.feed_forward_norm.before.Rms",
            ),
            ("model.layers.1.self_attn", "layers.1.mixer.Mla.attention"),
            (
                "model.layers.1.sparse_mlp.routed.routed",
                "layers.1.feed_forward.MixtureOfExperts.routed",
            ),
            (
                "model.layers.1.sparse_mlp.routed.router",
                "layers.1.feed_forward.MixtureOfExperts.router",
            ),
            (
                "model.layers.1.sparse_mlp.shared_experts",
                "layers.1.feed_forward.MixtureOfExperts.shared.0",
            ),
        ]),
    );
    (family, custom)
}

/// DeepSeek-V2: latent attention with a decoupled two-pair rotary sub-head, then dense and expert MLPs.
#[test]
fn deepseek_v2_latent_attention_blocks_match_the_family() {
    let (family, custom) = deepseek_v2_pair(LatentRope::unscaled());
    assert_causal_parity(&family, &custom, &Default::default());
}

/// DeepSeek-V2 YaRN: the serialized `mscale`/`mscale_all_dim` pair maps to explicit spec fields.
///
/// DeepSeek multiplies the rotary cosines and sines by
/// `magnitude(factor, mscale) / magnitude(factor, mscale_all_dim)` and the
/// summed content-plus-rotary logits by `magnitude(factor, mscale_all_dim)^2`
/// on top of `1/sqrt(d_qk)`, where `magnitude(s, m) = 1 + 0.1 m ln s`. The
/// modular spec states the first as the YaRN `attention_factor` and the
/// second as a [`ScoreScale::Fixed`] multiplier. Distinct `mscale` and
/// `mscale_all_dim` values keep both corrections away from one.
#[test]
fn deepseek_v2_yarn_latent_attention_matches_the_family() {
    let (factor, mscale, mscale_all_dim) = (2.0, 1.0, 0.5);
    let amplitude =
        deepseek_yarn_magnitude(factor, mscale) / deepseek_yarn_magnitude(factor, mscale_all_dim);
    let score = ((DEEPSEEK_CONTENT_HEAD_DIM + DEEPSEEK_ROPE_HEAD_DIM) as f64)
        .sqrt()
        .recip()
        * deepseek_yarn_magnitude(factor, mscale_all_dim).powi(2);
    let (family, custom) = deepseek_v2_pair(LatentRope {
        family: Some(RopeParameters {
            rope_type: RopeType::Yarn,
            factor: Some(factor),
            original_max_position_embeddings: Some(16),
            beta_fast: Some(4.0),
            beta_slow: Some(1.0),
            mscale: Some(mscale),
            mscale_all_dim: Some(mscale_all_dim),
            ..Default::default()
        }),
        theta: 10_000.0,
        scaling: RopeScaling::Yarn {
            factor,
            original_max_positions: 16,
            attention_factor: Some(amplitude),
            beta_slow: 1.0,
            beta_fast: 4.0,
            truncate: true,
        },
        score_scale: ScoreScale::Fixed { value: score },
    });
    assert_causal_parity(&family, &custom, &Default::default());
}

/// Negative controls: a wrong latent rotary base or a missing YaRN score correction is detectable.
///
/// With one rotary pair (the earlier fixture) a modular base of 10 instead of
/// 10 000 was indistinguishable from the family; the two-pair sub-head must
/// now separate it by at least [`DETECTABLE_GAP`].
#[test]
fn deepseek_v2_latent_parity_detects_wrong_rotary_policy() {
    let (family, custom) = deepseek_v2_pair(LatentRope {
        theta: 10.0,
        ..LatentRope::unscaled()
    });
    assert_detectable(&family, &custom, "latent rotary base 10 instead of 10 000");
    let factor = 2.0;
    let yarn = |attention_factor| RopeScaling::Yarn {
        factor,
        original_max_positions: 16,
        attention_factor,
        beta_slow: 1.0,
        beta_fast: 4.0,
        truncate: true,
    };
    let family_yarn = RopeParameters {
        rope_type: RopeType::Yarn,
        factor: Some(factor),
        original_max_position_embeddings: Some(16),
        beta_fast: Some(4.0),
        beta_slow: Some(1.0),
        mscale: Some(1.0),
        mscale_all_dim: Some(0.5),
        ..Default::default()
    };
    let amplitude = deepseek_yarn_magnitude(factor, 1.0) / deepseek_yarn_magnitude(factor, 0.5);
    let (family, custom) = deepseek_v2_pair(LatentRope {
        family: Some(family_yarn),
        theta: 10_000.0,
        scaling: yarn(Some(amplitude)),
        score_scale: ScoreScale::Auto,
    });
    assert_detectable(
        &family,
        &custom,
        "YaRN without the mscale_all_dim score correction",
    );
}

/// T5: unscaled attention with learned relative buckets, ReLU MLP, and unscaled cross-attention.
///
/// T5 compensates for its unscaled logits through initialization, so the
/// modular spec states the score scale explicitly: the legacy `Auto` keeps
/// T5-biased self-attention unscaled, and the cross-attention branch selects
/// [`ScoreScale::Unscaled`].
#[test]
fn t5_encoder_decoder_matches_the_family() {
    let device = Default::default();
    let config = T5Config {
        vocab_size: 32,
        d_model: 8,
        d_kv: 4,
        d_ff: 12,
        num_layers: 1,
        num_heads: 2,
        relative_attention_num_buckets: 8,
        relative_attention_max_distance: 16,
        feed_forward_proj: T5FeedForwardProjection::Relu,
        tie_word_embeddings: false,
        ..Default::default()
    };
    let norm = rms_with(config.layer_norm_epsilon);
    let family = T5ForConditionalGeneration::<TestBackend>::init(config, &device)
        .unwrap()
        .map(&mut DeterministicParameters::default());
    let stack = |bidirectional| DecoderSpec {
        vocabulary_size: 32,
        hidden_size: 8,
        layers: vec![pre_norm_block(
            MixerSpec::Attention(AttentionSpec {
                query_heads: 2,
                key_value_heads: 2,
                head_dim: 4,
                query_key_norm: false,
                output_gate: false,
                position: PositionEncodingSpec::T5RelativeBias {
                    buckets: 8,
                    max_distance: 16,
                    bidirectional,
                },
            }),
            FeedForwardSpec::Ungated(DenseFeedForwardSpec {
                intermediate_size: 12,
                activation: Activation::Relu,
                bias: false,
            }),
            norm,
        )],
        final_norm: norm,
        tie_word_embeddings: false,
        max_position_embeddings: 32,
    };
    let cross = CrossAttentionSpec {
        options: AttentionOptions {
            score_scale: ScoreScale::Unscaled,
            ..AttentionOptions::default()
        },
        ..CrossAttentionSpec::new(2, 4)
    };
    let spec = Seq2SeqSpec {
        encoder: stack(true),
        decoder: stack(false),
        cross_attention_heads: cross.heads,
        cross_attention_head_dim: cross.head_dim,
        cross_attention_options: cross.options,
    };
    let mut mapping = prefixes(&[
        ("shared", "encoder_embed_tokens"),
        ("shared", "decoder_embed_tokens"),
        ("lm_head", "lm_head"),
        ("encoder.final_layer_norm", "encoder_final_norm.Rms"),
        ("decoder.final_layer_norm", "decoder_final_norm.Rms"),
        (
            "decoder.block.0.cross_attention.layer_norm",
            "decoder_layers.0.cross_attention_norm.before.Rms",
        ),
        (
            "decoder.block.0.cross_attention.enc_dec_attention.attention",
            "decoder_layers.0.cross_attention.attention",
        ),
    ]);
    for (family_stack, modular_stack) in
        [("encoder", "encoder_layers"), ("decoder", "decoder_layers")]
    {
        let block = format!("{family_stack}.block.0");
        let layer = format!("{modular_stack}.0");
        mapping.extend([
            (
                format!("{block}.self_attention.layer_norm"),
                format!("{layer}.mixer_norm.before.Rms"),
            ),
            (
                format!("{block}.self_attention.self_attention.attention"),
                format!("{layer}.mixer.Attention.attention"),
            ),
            (
                format!("{block}.self_attention.self_attention.relative_attention_bias"),
                format!("{layer}.mixer.Attention.relative_bias"),
            ),
            (
                format!("{block}.feed_forward.layer_norm"),
                format!("{layer}.feed_forward_norm.before.Rms"),
            ),
            (
                format!("{block}.feed_forward.dense"),
                format!("{layer}.feed_forward.Ungated"),
            ),
        ]);
    }
    let custom = transferred(
        &family,
        &ModularSeq2SeqLm::<TestBackend>::init(spec, &device).unwrap(),
        &mapping,
    );
    // T5 numbers positions by column, so only unpadded and right-padded rows compare.
    let inputs = causal_inputs(&device);
    for (encoder, decoder) in [(&inputs[0], &inputs[1]), (&inputs[1], &inputs[0])] {
        let input = Seq2SeqInput {
            encoder: encoder.clone(),
            decoder: decoder.clone(),
        };
        let expected = family.forward(input.clone(), &all_logits()).unwrap();
        let actual = custom.forward(input.clone(), &all_logits()).unwrap();
        zero_padding(actual.encoder.hidden_state, &input.encoder).assert_approx_eq::<f32>(
            &zero_padding(expected.encoder.hidden_state, &input.encoder),
            Tolerance::absolute(LOGIT_TOLERANCE),
        );
        valid_logits(actual.decoder, &input.decoder).assert_approx_eq::<f32>(
            &valid_logits(expected.decoder, &input.decoder),
            Tolerance::absolute(LOGIT_TOLERANCE),
        );
    }
}

/// Qwen3.5: Gated DeltaNet then sigmoid-gated attention, with zero-centered RMS norms.
#[test]
fn qwen3_5_hybrid_blocks_match_the_family() {
    let device = Default::default();
    let config = Qwen3_5TextConfig {
        vocab_size: 32,
        hidden_size: 8,
        intermediate_size: 16,
        num_hidden_layers: 2,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: 4,
        max_position_embeddings: 64,
        layer_types: vec![
            Qwen3_5LayerType::LinearAttention,
            Qwen3_5LayerType::FullAttention,
        ],
        linear_key_head_dim: 4,
        linear_value_head_dim: 4,
        linear_num_key_heads: 1,
        linear_num_value_heads: 2,
        linear_conv_kernel_dim: 3,
        partial_rotary_factor: Some(1.0),
        rope_theta: Some(10_000.0),
        mtp_num_hidden_layers: 0,
        tie_word_embeddings: false,
        attn_output_gate: true,
        ..Default::default()
    };
    let norm = NormalizationSpec::Rms {
        epsilon: config.rms_norm_eps,
        zero_centered: true,
    };
    let family = Qwen3_5ForCausalLm::<TestBackend>::init(config, &device)
        .unwrap()
        .map(&mut DeterministicParameters::default());
    let mlp = FeedForwardSpec::Dense(DenseFeedForwardSpec {
        intermediate_size: 16,
        activation: Activation::Silu,
        bias: false,
    });
    let spec = DecoderSpec {
        vocabulary_size: 32,
        hidden_size: 8,
        layers: vec![
            pre_norm_block(
                MixerSpec::DeltaNet {
                    key_heads: 1,
                    value_heads: 2,
                    key_head_dim: 4,
                    value_head_dim: 4,
                    convolution_kernel: 3,
                },
                mlp.clone(),
                norm,
            ),
            pre_norm_block(
                MixerSpec::Attention(AttentionSpec {
                    query_heads: 2,
                    key_value_heads: 1,
                    head_dim: 4,
                    query_key_norm: true,
                    output_gate: true,
                    position: PositionEncodingSpec::Rope {
                        theta: 10_000.0,
                        rotary_dim: 4,
                        scaling: RopeScaling::None,
                    },
                }),
                mlp,
                norm,
            ),
        ],
        final_norm: norm,
        tie_word_embeddings: false,
        max_position_embeddings: 64,
    };
    let custom = transferred(
        &family,
        &ModularCausalLm::<TestBackend>::init(spec, &device).unwrap(),
        &prefixes(&[
            ("model.embed_tokens", "embed_tokens"),
            ("model.norm", "final_norm.Rms"),
            ("lm_head", "lm_head"),
            (
                "model.layers.0.input_layernorm",
                "layers.0.mixer_norm.before.Rms",
            ),
            (
                "model.layers.0.post_attention_layernorm",
                "layers.0.feed_forward_norm.before.Rms",
            ),
            ("model.layers.0.linear_attn", "layers.0.mixer.DeltaNet"),
            ("model.layers.0.mlp", "layers.0.feed_forward.Dense"),
            (
                "model.layers.1.input_layernorm",
                "layers.1.mixer_norm.before.Rms",
            ),
            (
                "model.layers.1.post_attention_layernorm",
                "layers.1.feed_forward_norm.before.Rms",
            ),
            (
                "model.layers.1.self_attn",
                "layers.1.mixer.Attention.attention",
            ),
            ("model.layers.1.mlp", "layers.1.feed_forward.Dense"),
        ]),
    );
    assert_causal_parity(&family, &custom, &device);
}
