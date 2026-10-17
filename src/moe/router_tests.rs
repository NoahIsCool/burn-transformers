//! Router parity with Hugging Face routing functions, total top-k, and policy validation.
//!
//! Golden constants were generated with the local Hugging Face Transformers
//! 5.2.0 (torch 2.10.0). Logits come from `torch.randn(tokens, experts,
//! generator=torch.Generator().manual_seed(seed)) * 1.5`; routers that own a
//! projection use an identity weight so their hidden states are the logits:
//!
//! ```text
//! python3 -I - <<'PY'
//! import torch
//! from transformers import MixtralConfig
//! from transformers.models.mixtral.modeling_mixtral import MixtralTopKRouter
//! x = torch.randn(4, 8, generator=torch.Generator().manual_seed(1)) * 1.5
//! r = MixtralTopKRouter(MixtralConfig(hidden_size=8, num_local_experts=8, num_experts_per_tok=2))
//! with torch.no_grad(): r.weight.copy_(torch.eye(8))
//! print(r(x))  # (probabilities, renormalized top-2 weights, top-2 indices)
//! PY
//! ```
//!
//! The other cases follow the same pattern with the parameters named in each
//! test: `Qwen3MoeTopKRouter` (seed 2), `DeepseekV3MoE.route_tokens_to_experts`
//! (seed 3), `DeepseekV2Moe.route_tokens_to_experts` with `group_limited_greedy`
//! (seed 4; 5.2 reads an unset `num_experts` attribute, so the generator sets
//! it), and `GptOssTopKRouter` with a `linspace(-0.3, 0.4, 8)` bias (seed 5).
//! DeepSeek routers use `topk(sorted=False)`, so their IDs and weights are
//! compared after sorting by expert ID. Weights match within `2e-6`.

use burn::{
    backend::Flex,
    tensor::{DType, Int, Tensor, TensorData},
};

use super::*;

/// Deterministic F32 CPU backend with I32 indices.
type TestBackend = Flex<f32, i32>;

/// Absolute tolerance between this crate's and PyTorch's F32 routing weights.
const TOLERANCE: f32 = 2e-6;
/// Mixtral router logits `[4 tokens, 8 experts]`, row-major (seed 1, scaled by 1.5).
const MIXTRAL_LOGITS: [f32; 32] = [
    -2.288394,
    -1.1253477,
    -0.98097134,
    -2.4142275,
    -0.15025076,
    -0.9137834,
    -1.4696584,
    -2.4136443,
    -1.0682169,
    0.45558298,
    -1.1659715,
    -0.3771823,
    -0.33340573,
    2.5306702,
    0.34263778,
    0.70145327,
    -1.0454583,
    -1.7411423,
    1.0493135,
    0.29862243,
    1.2985386,
    0.3666059,
    -0.9943675,
    1.2109624,
    1.6525211,
    -0.26390406,
    -3.3683367,
    -2.169687,
    0.09173292,
    -0.9266168,
    -1.1971042,
    -0.19743481,
];
/// Reference top-2 expert IDs `[4, 2]` in Hugging Face's descending-score order.
const MIXTRAL_IDS: [i64; 8] = [4, 5, 5, 7, 4, 7, 0, 4];
/// Reference renormalized top-2 softmax weights `[4, 2]`; each token's pair sums to 1.
const MIXTRAL_WEIGHTS: [f32; 8] = [
    0.6821202, 0.31787977, 0.8616684, 0.13833158, 0.52188003, 0.4781199, 0.82646644, 0.17353359,
];
/// Qwen3-MoE router logits `[4 tokens, 8 experts]`, row-major (seed 2, scaled by 1.5).
const QWEN_LOGITS: [f32; 32] = [
    -1.5612018,
    1.3749063,
    -1.956286,
    -1.6645176,
    -1.828167,
    1.7513885,
    -1.5860882,
    -0.17825906,
    -1.3617599,
    0.51783335,
    -0.85700035,
    -0.35266298,
    1.5114083,
    -1.1293222,
    -0.3374988,
    -0.649029,
    -2.2607157,
    -0.6878412,
    -1.272001,
    0.7899065,
    0.044874273,
    -0.07475705,
    1.5976169,
    1.3290551,
    0.6960276,
    -0.7479487,
    0.19329554,
    4.1446,
    0.2107147,
    1.6786523,
    0.47284734,
    2.6291647,
];
/// Reference top-3 expert IDs `[4, 3]` with `norm_topk_prob = false`.
const QWEN_NORM_FALSE_IDS: [i64; 12] = [5, 1, 7, 4, 1, 6, 6, 7, 3, 3, 7, 5];
/// Reference top-3 full-softmax probabilities `[4, 3]`, not renormalized.
const QWEN_NORM_FALSE_WEIGHTS: [f32; 12] = [
    0.50284404,
    0.34508744,
    0.07301263,
    0.4951885,
    0.18334389,
    0.07794718,
    0.3585033,
    0.27406794,
    0.15984866,
    0.710145,
    0.15602776,
    0.060311425,
];
/// Reference top-3 expert IDs `[4, 3]` with `norm_topk_prob = true`; renormalization
/// cannot change the selection, so these equal [`QWEN_NORM_FALSE_IDS`].
const QWEN_NORM_TRUE_IDS: [i64; 12] = [5, 1, 7, 4, 1, 6, 6, 7, 3, 3, 7, 5];
/// Reference top-3 weights `[4, 3]` renormalized to sum to 1 per token.
const QWEN_NORM_TRUE_WEIGHTS: [f32; 12] = [
    0.54600924,
    0.3747105,
    0.07928019,
    0.65459603,
    0.24236463,
    0.10303937,
    0.45241582,
    0.345862,
    0.20172216,
    0.76649445,
    0.16840844,
    0.065097086,
];
/// DeepSeek-V3 router logits `[5 tokens, 16 experts]`, row-major (seed 3, scaled by 1.5).
const DSV3_LOGITS: [f32; 80] = [
    -0.11496647,
    0.5398223,
    -1.1730251,
    0.1072916,
    0.99725914,
    -0.43017042,
    2.4308262,
    -2.395062,
    -0.07752141,
    -0.45896357,
    0.3727529,
    -0.33384126,
    1.3697278,
    0.30651176,
    0.86097705,
    0.62449694,
    0.39225185,
    1.3966867,
    -0.771693,
    -2.4774985,
    1.5690343,
    0.7832805,
    -0.25021374,
    0.0795501,
    0.84564453,
    3.3848357,
    2.8038902,
    -1.792851,
    1.4968836,
    0.68877065,
    3.6545956,
    -0.22021227,
    -0.71404654,
    -0.43936872,
    -0.52213395,
    0.52311957,
    0.055600315,
    -0.10152839,
    0.6435497,
    -1.3021789,
    -0.40682173,
    0.2124428,
    0.19427675,
    1.0221167,
    -1.4374506,
    0.095886126,
    0.9883709,
    1.2291888,
    -0.68313605,
    3.3186274,
    -0.5654799,
    -0.21552573,
    0.9720305,
    -3.488411,
    1.9025204,
    -0.3724904,
    1.4366708,
    -1.9335151,
    -2.4724736,
    1.2435546,
    -1.2559097,
    -0.79441047,
    2.031643,
    2.0666375,
    -0.11280376,
    -0.6349489,
    0.63253057,
    -0.38646796,
    -2.3752415,
    2.0940156,
    -1.547832,
    1.7086319,
    0.7686966,
    -0.02976796,
    -1.6823927,
    -0.73360056,
    -0.95044076,
    -1.1840076,
    -1.3465996,
    -2.8313587,
];
/// Selection-only correction bias `((e mod 5) - 2) * 0.07` for experts `e = 0..16`.
const DSV3_BIAS: [f32; 16] = [
    -0.14, -0.07, 0.0, 0.07, 0.14, -0.14, -0.07, 0.0, 0.07, 0.14, -0.14, -0.07, 0.0, 0.07, 0.14,
    -0.14,
];
/// Reference `noaux_tc` top-4 expert IDs `[5, 4]`, sorted by ID within each token
/// because Hugging Face selects with `topk(sorted=False)`.
const DSV3_IDS_SORTED: [i64; 20] = [
    4, 6, 12, 14, 9, 10, 12, 14, 9, 11, 14, 15, 4, 6, 14, 15, 5, 7, 8, 9,
];
/// Reference `noaux_tc` weights `[5, 4]` (unbiased sigmoid scores renormalized with
/// `+1e-20` and scaled by 2.5), permuted with [`DSV3_IDS_SORTED`].
const DSV3_WEIGHTS_SORTED: [f32; 20] = [
    0.57980156, 0.72951305, 0.632833, 0.5578525, 0.6531784, 0.6367403, 0.55180216, 0.6582792,
    0.4953095, 0.65877235, 0.6528432, 0.69307494, 0.5386385, 0.6460303, 0.65635204, 0.6589791,
    0.76415086, 0.7266768, 0.58641547, 0.42275694,
];
/// DeepSeek-V2 router logits `[5 tokens, 16 experts]`, row-major (seed 4, scaled by 1.5).
const DSV2_LOGITS: [f32; 80] = [
    -1.4121246,
    1.89487,
    -0.27565408,
    0.22575818,
    0.1612533,
    -0.4170485,
    -3.9032028,
    0.9366863,
    -1.3025445,
    -0.30758384,
    0.59634113,
    1.004889,
    -0.08056054,
    0.07004286,
    -2.6506705,
    -3.1806893,
    2.2786455,
    -1.002297,
    0.004616564,
    -0.2303217,
    1.7094231,
    -0.3452812,
    1.7815988,
    1.1516032,
    -1.1382488,
    -0.2780148,
    -1.2836864,
    -0.3519594,
    -0.63229483,
    1.2732548,
    -1.0163639,
    -1.4168118,
    -0.72230875,
    1.8651717,
    3.5539985,
    0.42441612,
    -0.35180932,
    2.5338652,
    0.40745717,
    -0.2048176,
    -1.0421795,
    -1.9778783,
    -1.454079,
    0.96038055,
    1.2301852,
    -1.3726661,
    -3.2155995,
    2.1107674,
    -0.039475303,
    4.0806303,
    -0.8933097,
    1.4807215,
    1.6290779,
    0.09147127,
    0.062491793,
    1.0173972,
    -1.3427309,
    -1.5215135,
    -0.36429542,
    -2.3590875,
    2.090928,
    -0.29108793,
    0.007227528,
    -1.9747074,
    0.030602023,
    -0.24778828,
    0.3163919,
    0.7750667,
    0.36448008,
    -0.5283622,
    -1.1495333,
    0.44598636,
    1.9064693,
    2.7174966,
    2.2979412,
    -0.39031386,
    -1.7416348,
    -1.8570249,
    1.5799209,
    0.3003106,
];
/// Reference `group_limited_greedy` top-3 expert IDs `[5, 3]`, sorted by ID per token.
const DSV2_IDS_SORTED: [i64; 15] = [1, 10, 11, 0, 4, 6, 1, 2, 5, 1, 3, 12, 8, 9, 10];
/// Reference softmax weights `[5, 3]` scaled by 1.5 without renormalization,
/// permuted with [`DSV2_IDS_SORTED`].
const DSV2_WEIGHTS_SORTED: [f32; 15] = [
    0.475692, 0.12983204, 0.19534937, 0.43033528, 0.24355486, 0.26178354, 0.13026142, 0.70512146,
    0.25422934, 1.0295098, 0.07647235, 0.1407712, 0.20897958, 0.4702497, 0.30911344,
];
/// gpt-oss router inputs `[4 tokens, 8 experts]` (seed 5, scaled by 1.5); with the
/// identity projection they are the unbiased router logits.
const GPTOSS_HIDDEN: [f32; 32] = [
    2.7634947,
    0.77833074,
    -2.5678825,
    -2.5520692,
    3.029129,
    -0.40292168,
    -0.19607624,
    -2.1561208,
    0.5861523,
    -0.02845164,
    -2.0290308,
    -1.096226,
    1.4818834,
    -0.62911195,
    -0.8773438,
    -1.1735021,
    4.169794,
    1.8330379,
    -0.5046741,
    -1.4475917,
    -0.19449401,
    -0.90264857,
    0.21749786,
    -0.22474074,
    -0.6560996,
    1.1688505,
    -0.087509066,
    -3.045742,
    2.2243986,
    0.7410571,
    0.37382817,
    2.6205473,
];
/// gpt-oss router bias `linspace(-0.3, 0.4, 8)` as rounded by F32 arithmetic.
const GPTOSS_ROUTER_BIAS: [f32; 8] = [
    -0.3,
    -0.2,
    -0.099999994,
    1.4901161e-08,
    0.09999998,
    0.19999999,
    0.3,
    0.4,
];
/// Reference top-2 expert IDs `[4, 2]` chosen on the biased logits.
const GPTOSS_IDS: [i64; 8] = [4, 0, 4, 0, 0, 1, 7, 4];
/// Reference weights `[4, 2]`: a softmax over each token's two selected biased logits.
const GPTOSS_WEIGHTS: [f32; 8] = [
    0.6605249, 0.33947513, 0.78511566, 0.21488436, 0.90350205, 0.09649798, 0.66733336, 0.3326666,
];

/// Uploads row-major F32 logits as `[tokens, experts]`.
fn logits(values: &[f32], experts: usize) -> Tensor<TestBackend, 2> {
    Tensor::from_data(
        TensorData::new(values.to_vec(), [values.len() / experts, experts]),
        &Default::default(),
    )
}

/// Reads a decision's IDs and weights to the host.
fn host(decision: RouteDecision<TestBackend>) -> (Vec<i64>, Vec<f32>, Vec<bool>) {
    let ids = decision
        .experts
        .into_data()
        .to_vec::<i32>()
        .expect("I32 IDs")
        .into_iter()
        .map(i64::from)
        .collect();
    let weights = decision
        .weights
        .into_data()
        .to_vec::<f32>()
        .expect("F32 weights");
    let flags = decision
        .non_finite_tokens
        .into_data()
        .to_vec::<bool>()
        .expect("Bool flags");
    (ids, weights, flags)
}

/// Sorts each token's `(id, weight)` pairs by expert ID.
fn sorted_by_id(ids: &[i64], weights: &[f32], k: usize) -> (Vec<i64>, Vec<f32>) {
    let mut sorted_ids = Vec::new();
    let mut sorted_weights = Vec::new();
    for (row_ids, row_weights) in ids.chunks(k).zip(weights.chunks(k)) {
        let mut pairs = row_ids.iter().zip(row_weights).collect::<Vec<_>>();
        pairs.sort_by_key(|(id, _)| **id);
        sorted_ids.extend(pairs.iter().map(|(id, _)| **id));
        sorted_weights.extend(pairs.iter().map(|(_, weight)| **weight));
    }
    (sorted_ids, sorted_weights)
}

/// Asserts element-wise agreement within [`TOLERANCE`].
fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= TOLERANCE,
            "entry {index}: {actual} vs {expected}"
        );
    }
}

/// Mixtral: softmax over 8 experts, top-2, renormalized (seed 1, 4 tokens).
#[test]
fn mixtral_routing_matches_reference() {
    let router = RouterConfig::new(8, 2);
    let (ids, weights, flags) = host(router.route(logits(&MIXTRAL_LOGITS, 8), None).unwrap());
    assert_eq!(ids, MIXTRAL_IDS);
    assert_close(&weights, &MIXTRAL_WEIGHTS);
    assert!(flags.iter().all(|flag| !flag));
}

/// Qwen3-MoE: softmax, top-3, with and without renormalization (seed 2, 4 tokens).
#[test]
fn qwen3_moe_routing_matches_reference() {
    for (renormalize, expected) in [
        (false, QWEN_NORM_FALSE_WEIGHTS),
        (true, QWEN_NORM_TRUE_WEIGHTS),
    ] {
        let mut router = RouterConfig::new(8, 3);
        router.renormalize_selected = renormalize;
        let (ids, weights, _) = host(router.route(logits(&QWEN_LOGITS, 8), None).unwrap());
        assert_eq!(ids, QWEN_NORM_TRUE_IDS);
        assert_eq!(ids, QWEN_NORM_FALSE_IDS);
        assert_close(&weights, &expected);
    }
}

/// DeepSeek-V3 `noaux_tc` as written by Hugging Face 5.2.
///
/// Sigmoid scores plus a selection-only bias `((e mod 5) - 2) * 0.07`; 16 experts
/// in 4 groups ranked by top-two sum, 2 groups kept with pruned experts set to
/// zero, top-4, weights renormalized with `+1e-20` and scaled by 2.5 (seed 3).
#[test]
fn deepseek_v3_routing_matches_reference() {
    let router = RouterConfig {
        experts: 16,
        experts_per_token: 4,
        policy: RoutingPolicy {
            scoring: RouterScoring::Sigmoid,
            selection: ExpertSelection::GroupLimited(GroupLimitedSelection {
                groups: 4,
                top_groups: 2,
                group_score: GroupScore::TopTwoSum,
                pruned_score: PrunedExpertScore::Zero,
            }),
            correction_bias: true,
            denominator: RenormalizationDenominator::PlusEpsilon(1e-20),
            ..RoutingPolicy::default()
        },
        renormalize_selected: true,
        routed_scale: 2.5,
    };
    let bias = Tensor::from_data(
        TensorData::new(DSV3_BIAS.to_vec(), [16]),
        &Default::default(),
    );
    let (ids, weights, _) = host(router.route(logits(&DSV3_LOGITS, 16), Some(bias)).unwrap());
    let (ids, weights) = sorted_by_id(&ids, &weights, 4);
    assert_eq!(ids, DSV3_IDS_SORTED);
    assert_close(&weights, &DSV3_WEIGHTS_SORTED);
}

/// DeepSeek-V2 `group_limited_greedy`: softmax, group max, 2 of 4 groups, top-3, scale 1.5 (seed 4).
#[test]
fn deepseek_v2_group_limited_routing_matches_reference() {
    let router = RouterConfig {
        experts: 16,
        experts_per_token: 3,
        policy: RoutingPolicy {
            selection: ExpertSelection::GroupLimited(GroupLimitedSelection {
                groups: 4,
                top_groups: 2,
                group_score: GroupScore::Max,
                pruned_score: PrunedExpertScore::Zero,
            }),
            ..RoutingPolicy::default()
        },
        renormalize_selected: false,
        routed_scale: 1.5,
    };
    let (ids, weights, _) = host(router.route(logits(&DSV2_LOGITS, 16), None).unwrap());
    let (ids, weights) = sorted_by_id(&ids, &weights, 3);
    assert_eq!(ids, DSV2_IDS_SORTED);
    assert_close(&weights, &DSV2_WEIGHTS_SORTED);
}

/// gpt-oss: biased router logits, top-2 on logits, softmax over the selected logits (seed 5).
#[test]
fn gpt_oss_top_k_then_softmax_matches_reference() {
    let mut router = RouterConfig::new(8, 2);
    router.policy.order = ScoringOrder::TopKThenScore;
    router.renormalize_selected = false;
    let device = Default::default();
    let bias = Tensor::<TestBackend, 1>::from_data(
        TensorData::new(GPTOSS_ROUTER_BIAS.to_vec(), [8]),
        &device,
    );
    let biased = logits(&GPTOSS_HIDDEN, 8) + bias.reshape([1, 8]);
    let (ids, weights, _) = host(router.route(biased, None).unwrap());
    assert_eq!(ids, GPTOSS_IDS);
    assert_close(&weights, &GPTOSS_WEIGHTS);
}

/// Llama 4 order: top-k on logits, then an independent sigmoid of each selected logit.
#[test]
fn top_k_then_sigmoid_scores_only_selected_logits() {
    let mut router = RouterConfig::new(4, 2);
    router.policy.order = ScoringOrder::TopKThenScore;
    router.policy.scoring = RouterScoring::Sigmoid;
    router.renormalize_selected = false;
    let values = [0.5_f32, -1.0, 2.0, 0.0];
    let (ids, weights, _) = host(router.route(logits(&values, 4), None).unwrap());
    assert_eq!(ids, [2, 0]);
    let sigmoid = |value: f32| 1.0 / (1.0 + (-value).exp());
    assert_close(&weights, &[sigmoid(2.0), sigmoid(0.5)]);
}

/// Pruned experts set to zero can outrank negative eligible scores; `-inf` cannot.
#[test]
fn pruned_score_choice_matters_for_negative_selection_scores() {
    // Group 0 = experts {0,1}, group 1 = {2,3}. Selection scores are sigmoid
    // plus a bias that makes every corrected score negative.
    let values = [4.0_f32, 3.0, -4.0, -5.0];
    let bias = Tensor::<TestBackend, 1>::from_data(
        TensorData::new(vec![-2.0_f32; 4], [4]),
        &Default::default(),
    );
    let route = |pruned_score| {
        let router = RouterConfig {
            experts: 4,
            experts_per_token: 2,
            policy: RoutingPolicy {
                scoring: RouterScoring::Sigmoid,
                selection: ExpertSelection::GroupLimited(GroupLimitedSelection {
                    groups: 2,
                    top_groups: 1,
                    group_score: GroupScore::TopTwoSum,
                    pruned_score,
                }),
                correction_bias: true,
                ..RoutingPolicy::default()
            },
            renormalize_selected: false,
            routed_scale: 1.0,
        };
        host(
            router
                .route(logits(&values, 4), Some(bias.clone()))
                .unwrap(),
        )
        .0
    };
    assert_eq!(route(PrunedExpertScore::NegativeInfinity), [0, 1]);
    // With zero-filled pruned experts, the pruned group outranks the kept one.
    assert_eq!(route(PrunedExpertScore::Zero), [2, 3]);
}

/// The default router reproduces the historical softmax/top-k/renormalize path bit for bit.
#[test]
fn default_router_is_bit_identical_to_the_legacy_formulation() {
    let device = Default::default();
    let raw = Tensor::<TestBackend, 1, Int>::arange(0..96, &device)
        .float()
        .reshape([3, 4, 8])
        .mul_scalar(0.37)
        .sin()
        * 3.0;
    let legacy_probabilities = burn::tensor::activation::softmax(raw.clone(), 2);
    let (legacy_weights, legacy_ids) = legacy_top_k(legacy_probabilities, 3);
    let legacy_weights = legacy_weights.reshape([12, 3]);
    let legacy_weights =
        legacy_weights.clone() / legacy_weights.sum_dim(1).clamp_min(f32::MIN_POSITIVE) * 1.0;
    let decision = RouterConfig::new(8, 3)
        .route(raw.reshape([12, 8]), None)
        .unwrap();
    assert_eq!(
        decision.experts.into_data(),
        legacy_ids.reshape([12, 3]).into_data()
    );
    assert_eq!(decision.weights.into_data(), legacy_weights.into_data());
}

/// The pre-refactor deterministic top-k, kept verbatim as the parity oracle.
fn legacy_top_k(
    probabilities: Tensor<TestBackend, 3>,
    k: usize,
) -> (Tensor<TestBackend, 3>, Tensor<TestBackend, 3, Int>) {
    let [batch, sequence, experts] = probabilities.dims();
    let device = probabilities.device();
    let expert_ids = Tensor::<TestBackend, 1, Int>::arange(0..experts as i64, &device)
        .reshape([1, 1, experts])
        .expand([batch, sequence, experts]);
    let mut remaining = probabilities.clone();
    let mut weights = Vec::new();
    let mut ids = Vec::new();
    for _ in 0..k {
        let maximum = remaining.clone().max_dim(2);
        let tied = remaining.clone().equal(maximum);
        let selected = expert_ids
            .clone()
            .mask_fill(tied.bool_not(), experts as i64)
            .min_dim(2);
        weights.push(probabilities.clone().gather(2, selected.clone()));
        ids.push(selected.clone());
        remaining = remaining.mask_fill(
            expert_ids
                .clone()
                .equal(selected.expand([batch, sequence, experts])),
            f32::NEG_INFINITY,
        );
    }
    (Tensor::cat(weights, 2), Tensor::cat(ids, 2))
}

/// NaN, infinite, and fully masked rows still yield distinct in-range IDs and are flagged.
#[test]
fn top_k_is_total_for_non_finite_rows() {
    check_top_k_is_total_for_non_finite_rows::<TestBackend>(&Default::default());
}

/// Rows of NaN, `-inf`, and `+inf` keys for [`check_top_k_is_total_for_non_finite_rows`], `[4, 4]`.
const NON_FINITE_ROWS: [f32; 16] = [
    f32::NAN,
    f32::NAN,
    f32::NAN,
    f32::NAN,
    // A single NaN among finite scores ranks last.
    0.2,
    f32::NAN,
    0.5,
    0.1,
    f32::NEG_INFINITY,
    f32::NEG_INFINITY,
    f32::NEG_INFINITY,
    f32::NEG_INFINITY,
    f32::INFINITY,
    1.0,
    f32::INFINITY,
    2.0,
];

/// Uploads row-major F32 values as `[rows, 4]` on any backend.
fn four_columns<B: Backend>(values: &[f32], device: &B::Device) -> Tensor<B, 2> {
    Tensor::from_data(
        TensorData::new(values.to_vec(), [values.len() / 4, 4]),
        device,
    )
}

/// Reads integer IDs or Boolean flags (as 0/1) of any backend to the host.
fn host_ints<B: Backend, const D: usize>(tensor: Tensor<B, D, Int>) -> Vec<i64> {
    tensor
        .into_data()
        .convert::<i64>()
        .to_vec::<i64>()
        .expect("I64 host values")
}

/// Top-k over [`NON_FINITE_ROWS`] picks every index once per row, NaN and `-inf` last.
///
/// Every key of the all-NaN and all-`-inf` rows is in the lowest class from
/// the first selection: every index ties and the smallest available one wins.
/// The router flags the NaN rows. Generic so the CubeCL CPU backend runs the
/// same assertions as Flex (see "Finite ranking keys" in
/// [`deterministic_top_k`]).
fn check_top_k_is_total_for_non_finite_rows<B: Backend>(device: &B::Device) {
    let (selected, ids) = deterministic_top_k(
        four_columns::<B>(&NON_FINITE_ROWS, device).unsqueeze_dim::<3>(0),
        3,
    );
    assert_eq!(host_ints(ids), [0, 1, 2, 2, 0, 3, 0, 1, 2, 0, 2, 3]);
    let selected = selected
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .expect("F32 host values");
    assert!(selected[..3].iter().all(|value| value.is_nan()));

    let mut router = RouterConfig::new(4, 2);
    router.policy.scoring = RouterScoring::Sigmoid;
    router.renormalize_selected = false;
    let decision = router
        .route(four_columns::<B>(&NON_FINITE_ROWS, device), None)
        .unwrap();
    assert_eq!(host_ints(decision.non_finite_tokens.int()), [1, 1, 0, 0]);
    assert_eq!(host_ints(decision.experts), [0, 1, 2, 0, 0, 1, 0, 2]);
}

/// Rows mixing `-inf`, NaN, and the F32 minimum with ordinary keys, `[2, 4]`.
const LOWEST_CLASS_ROWS: [f32; 8] = [
    f32::NEG_INFINITY,
    f32::MIN,
    0.0,
    f32::NAN,
    f32::MIN,
    -1.0,
    f32::NEG_INFINITY,
    f32::MIN,
];

/// `-inf`, NaN, and the dtype's lowest finite key form one class ordered by index.
///
/// Ordinary keys keep their exact order ahead of that class. Generic so the
/// CubeCL CPU backend runs the same assertions as Flex (see "Finite ranking
/// keys" in [`deterministic_top_k`]).
fn check_top_k_ranks_the_lowest_class_by_index<B: Backend>(device: &B::Device) {
    let (_, ids) = deterministic_top_k(
        four_columns::<B>(&LOWEST_CLASS_ROWS, device).unsqueeze_dim::<3>(0),
        4,
    );
    assert_eq!(host_ints(ids), [2, 0, 1, 3, 1, 0, 2, 3]);
}

/// Flex: the lowest key class is ordered by index.
#[test]
fn top_k_ranks_the_lowest_class_by_index() {
    check_top_k_ranks_the_lowest_class_by_index::<TestBackend>(&Default::default());
}

/// Half-precision keys stay total: an F16 row of `-inf` selects every index once.
///
/// `f32::MIN` rounds to `-inf` in F16, so the clamp must use the dtype's own
/// minimum ([`lowest_finite`]).
#[test]
fn half_precision_top_k_is_total_for_negative_infinity_rows() {
    let device = Default::default();
    for dtype in [DType::F16, DType::BF16] {
        let values = four_columns::<TestBackend>(&[f32::NEG_INFINITY; 4], &device)
            .cast(dtype)
            .unsqueeze_dim::<3>(0);
        let (_, ids) = deterministic_top_k(values, 4);
        assert_eq!(host_ints(ids), [0, 1, 2, 3], "{dtype:?}");
    }
}

/// Each float dtype's lowest finite value is exact, finite, and the dtype's minimum.
#[test]
fn lowest_finite_is_each_dtype_minimum() {
    assert_eq!(lowest_finite(DType::F64), f64::MIN);
    assert_eq!(lowest_finite(DType::F32), f64::from(f32::MIN));
    assert_eq!(lowest_finite(DType::F16), -65504.0);
    assert_eq!(
        burn::tensor::f16::from_f64(lowest_finite(DType::F16)),
        burn::tensor::f16::MIN
    );
    assert_eq!(
        burn::tensor::bf16::from_f64(lowest_finite(DType::BF16)),
        burn::tensor::bf16::MIN
    );
    // A non-float dtype never reaches the router and falls back to F32's minimum.
    assert_eq!(lowest_finite(DType::I32), f64::from(f32::MIN));
}

/// Tokens whose final weights are not finite are flagged even when no score is NaN.
///
/// Top-k-then-softmax ranks raw logits, so a `+inf` logit (an overflowed F16
/// projection) or an all-`-inf` selection only turns into NaN inside the
/// softmax over the selected logits. A zero renormalization guard divides an
/// all-zero sigmoid sum by zero. Each case must reach the caller's error path.
#[test]
fn non_finite_weights_are_flagged_in_every_order() {
    let mut gpt_oss = RouterConfig::new(4, 2);
    gpt_oss.policy.order = ScoringOrder::TopKThenScore;
    gpt_oss.renormalize_selected = false;
    let values = [
        // softmax([+inf, 1]) = [NaN, NaN].
        f32::INFINITY,
        1.0,
        0.0,
        -1.0,
        // softmax([-inf, -inf]) = [NaN, NaN].
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
        // Finite logits route normally.
        0.5,
        2.0,
        -0.5,
        1.0,
    ];
    let (ids, weights, flags) = host(gpt_oss.route(logits(&values, 4), None).unwrap());
    assert_eq!(flags, [true, true, false]);
    assert!(ids.iter().all(|&id| (0..4).contains(&id)));
    assert!(weights[4..].iter().all(|weight| weight.is_finite()));

    let mut zero_guard = RouterConfig::new(4, 2);
    zero_guard.policy.scoring = RouterScoring::Sigmoid;
    zero_guard.policy.denominator = RenormalizationDenominator::PlusEpsilon(0.0);
    // sigmoid(-inf) = 0 for every expert, so the selected sum is exactly zero.
    let (_, _, flags) = host(zero_guard.route(logits(&values[4..], 4), None).unwrap());
    assert_eq!(flags, [true, false]);
}

/// Configuration validation names each violated routing constraint.
#[test]
fn router_validation_rejects_each_invalid_policy() {
    let group = |groups, top_groups| {
        let mut config = RouterConfig::new(8, 2);
        config.policy.selection = ExpertSelection::GroupLimited(GroupLimitedSelection {
            groups,
            top_groups,
            group_score: GroupScore::Max,
            pruned_score: PrunedExpertScore::NegativeInfinity,
        });
        config
    };
    let mut cases = vec![
        (RouterConfig::new(0, 1), "moe.router.experts"),
        (RouterConfig::new(4, 0), "moe.router.experts_per_token"),
        (RouterConfig::new(4, 5), "moe.router.experts_per_token"),
        (group(3, 1), "moe.router.groups"),
        (group(0, 1), "moe.router.groups"),
        (group(4, 0), "moe.router.top_groups"),
        (group(4, 5), "moe.router.top_groups"),
    ];
    // One kept group of two experts cannot supply three selections.
    let mut too_few = group(4, 1);
    too_few.experts_per_token = 3;
    cases.push((too_few, "moe.router.top_groups"));
    for scale in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let mut config = RouterConfig::new(4, 1);
        config.routed_scale = scale;
        cases.push((config, "moe.router.routed_scale"));
    }
    for denominator in [
        RenormalizationDenominator::ClampedBelow(-1.0),
        RenormalizationDenominator::PlusEpsilon(f32::NAN),
    ] {
        let mut config = RouterConfig::new(4, 1);
        config.policy.denominator = denominator;
        cases.push((config, "moe.router.denominator"));
    }
    let mut biased_top_k_first = RouterConfig::new(4, 1);
    biased_top_k_first.policy.order = ScoringOrder::TopKThenScore;
    biased_top_k_first.policy.correction_bias = true;
    cases.push((biased_top_k_first, "moe.router.order"));
    let mut grouped_top_k_first = group(4, 2);
    grouped_top_k_first.policy.order = ScoringOrder::TopKThenScore;
    cases.push((grouped_top_k_first, "moe.router.order"));
    for (config, expected) in cases {
        let error = config.validate().expect_err(expected);
        assert!(
            matches!(&error, crate::Error::Configuration { field, .. } if field == expected),
            "{expected}: {error}"
        );
    }
    assert!(group(4, 4).validate().is_ok());
}

/// Correction-bias presence and shape must agree with the policy, and logits with the expert count.
#[test]
fn route_checks_bias_presence_and_logit_shape() {
    let device = Default::default();
    let mut biased = RouterConfig::new(4, 1);
    biased.policy.correction_bias = true;
    let bias = Tensor::<TestBackend, 1>::zeros([4], &device);
    let cases = [
        (
            RouterConfig::new(4, 1).route(logits(&[0.0; 4], 4), Some(bias.clone())),
            "moe.router.correction_bias",
        ),
        (
            biased.route(logits(&[0.0; 4], 4), None),
            "moe.router.correction_bias",
        ),
        (
            biased.route(
                logits(&[0.0; 4], 4),
                Some(Tensor::<TestBackend, 1>::zeros([3], &device)),
            ),
            "moe.router.correction_bias",
        ),
        (
            RouterConfig::new(4, 1).route(logits(&[0.0; 6], 3), None),
            "moe.router.logits",
        ),
        (
            RouterConfig::new(4, 1).route(
                Tensor::<TestBackend, 2>::from_data(
                    TensorData::new(Vec::<f32>::new(), [0, 4]),
                    &device,
                ),
                None,
            ),
            "moe.router.logits",
        ),
    ];
    for (result, field) in cases {
        let error = result.expect_err(field);
        assert!(
            matches!(&error, crate::Error::Configuration { field: name, .. }
                | crate::Error::Shape { tensor: name, .. } if name == field),
            "{error}"
        );
    }
    assert!(biased.route(logits(&[0.0; 4], 4), Some(bias)).is_ok());
}

/// F32 projection casts all operands, while model-dtype projection keeps the activation dtype.
#[test]
fn router_projection_precision_controls_logit_dtype() {
    use burn::nn::{LinearConfig, LinearLayout};
    let device = Default::default();
    let mut linear = LinearConfig::new(2, 3)
        .with_bias(true)
        .with_layout(LinearLayout::Col)
        .init::<TestBackend>(&device);
    linear.weight = burn::module::Param::from_tensor(
        Tensor::from_floats([[1.0, 0.0, 2.0], [0.0, 1.0, -1.0]], &device).cast(DType::F16),
    );
    linear.bias = Some(burn::module::Param::from_tensor(
        Tensor::from_floats([0.5, 0.0, 0.0], &device).cast(DType::F16),
    ));
    let input = Tensor::<TestBackend, 2>::from_floats([[1.0, 2.0]], &device).cast(DType::F16);
    let model = router_logits(
        input.clone(),
        &linear,
        RouterProjectionPrecision::ModelDtype,
    )
    .expect("model dtype");
    assert_eq!(model.dtype(), DType::F16);
    let fp32 = router_logits(input, &linear, RouterProjectionPrecision::Fp32).expect("fp32");
    assert_eq!(fp32.dtype(), DType::F32);
    assert_eq!(fp32.into_data().to_vec::<f32>().unwrap(), [1.5, 2.0, 0.0]);
    let error = router_logits(
        Tensor::<TestBackend, 2>::zeros([1, 3], &device),
        &linear,
        RouterProjectionPrecision::Fp32,
    )
    .expect_err("width mismatch");
    assert!(matches!(error, crate::Error::Shape { tensor, .. } if tensor == "moe.router.weight"));
}

/// The rank-2 F32 router product reproduces Burn's `module::linear` bit for bit.
///
/// The FP32 projection moved from `burn::tensor::module::linear` (a batched
/// product against an unsqueezed weight) to one flattened `[rows, I] · W`
/// GEMM to keep router gradients on every backward pass. On Flex both reduce
/// each logit over the same input axis in the same order, so DeepSeek-V2 and
/// Nemotron-H router logits, and therefore expert selection, are unchanged.
/// Covers F32, F16, and BF16 activations, rank-two and rank-three inputs, and
/// biased and unbiased routers.
#[test]
fn fp32_router_projection_matches_module_linear_bitwise() {
    use burn::nn::{LinearConfig, LinearLayout};
    let device = Default::default();
    let hidden = 24;
    let experts = 10;
    let values = |count: usize, salt: f32| {
        (0..count)
            .map(|index| (index as f32 * 0.731 + salt).sin() * 1.7)
            .collect::<Vec<_>>()
    };
    for bias in [false, true] {
        let mut linear = LinearConfig::new(hidden, experts)
            .with_bias(bias)
            .with_layout(LinearLayout::Col)
            .init::<TestBackend>(&device);
        linear.weight = burn::module::Param::from_tensor(Tensor::from_data(
            TensorData::new(values(hidden * experts, 0.3), [hidden, experts]),
            &device,
        ));
        if bias {
            linear.bias = Some(burn::module::Param::from_tensor(Tensor::from_data(
                TensorData::new(values(experts, 1.1), [experts]),
                &device,
            )));
        }
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            let input = Tensor::<TestBackend, 3>::from_data(
                TensorData::new(values(2 * 5 * hidden, 2.9), [2, 5, hidden]),
                &device,
            )
            .cast(dtype);
            let reference = burn::tensor::module::linear(
                input.clone().cast(DType::F32),
                linear.weight.val().cast(DType::F32),
                linear.bias.as_ref().map(|bias| bias.val().cast(DType::F32)),
            );
            let actual = router_logits(input.clone(), &linear, RouterProjectionPrecision::Fp32)
                .expect("rank-three projection");
            assert_eq!(actual.dims(), [2, 5, experts]);
            let bits = |tensor: Tensor<TestBackend, 3>| {
                tensor
                    .into_data()
                    .to_vec::<f32>()
                    .expect("F32 logits")
                    .into_iter()
                    .map(f32::to_bits)
                    .collect::<Vec<_>>()
            };
            assert_eq!(
                bits(actual),
                bits(reference.clone()),
                "{dtype:?}, bias {bias}"
            );
            let flat = router_logits(
                input.reshape([10, hidden]),
                &linear,
                RouterProjectionPrecision::Fp32,
            )
            .expect("rank-two projection");
            assert_eq!(
                bits(flat.reshape([2, 5, experts])),
                bits(reference),
                "rank two, {dtype:?}, bias {bias}"
            );
        }
    }
}

/// Keeping every group is the identity, so group-limited routing then equals greedy routing.
///
/// Nemotron-H configures `n_group = topk_group = 1`; any `top_groups ==
/// groups` keeps every expert, so the pruning step must return the scores
/// unchanged (Burn cannot one-hot a single class) and selection must match the
/// greedy policy exactly, including tie order.
#[test]
fn keeping_every_group_matches_greedy_routing() {
    let values = [0.5_f32, -1.0, 2.0, 2.0, 0.25, 3.0, -0.5, 1.0];
    let mut greedy = RouterConfig::new(4, 2);
    greedy.policy.scoring = RouterScoring::Sigmoid;
    let expected = host(greedy.route(logits(&values, 4), None).unwrap());
    for (groups, group_score) in [(1, GroupScore::Max), (2, GroupScore::TopTwoSum)] {
        let mut grouped = greedy;
        grouped.policy.selection = ExpertSelection::GroupLimited(GroupLimitedSelection {
            groups,
            top_groups: groups,
            group_score,
            pruned_score: PrunedExpertScore::Zero,
        });
        assert_eq!(
            host(grouped.route(logits(&values, 4), None).unwrap()),
            expected,
            "{groups} groups"
        );
    }
    // The tie between experts 2 and 3 resolves to the smaller index first.
    assert_eq!(expected.0[..2], [2, 3]);
}

/// Policies round-trip through serde and default when omitted.
#[test]
fn routing_policy_serializes_with_defaults() {
    let policy = RoutingPolicy {
        scoring: RouterScoring::Sigmoid,
        selection: ExpertSelection::GroupLimited(GroupLimitedSelection {
            groups: 8,
            top_groups: 4,
            group_score: GroupScore::TopTwoSum,
            pruned_score: PrunedExpertScore::Zero,
        }),
        correction_bias: true,
        denominator: RenormalizationDenominator::PlusEpsilon(1e-20),
        ..RoutingPolicy::default()
    };
    let json = serde_json::to_value(policy).unwrap();
    assert_eq!(
        serde_json::from_value::<RoutingPolicy>(json).unwrap(),
        policy
    );
    assert_eq!(
        serde_json::from_str::<RoutingPolicy>("{}").unwrap(),
        RoutingPolicy::default()
    );
}

proptest::proptest! {
    #![proptest_config(proptest::test_runner::Config::with_cases(48))]

    /// Top-k agrees with a host sort that breaks ties by the smallest index.
    #[test]
    fn top_k_matches_host_stable_sort(
        rows in 1usize..5,
        experts in 1usize..9,
        k_seed in 0usize..9,
        // Small integers create frequent ties.
        values in proptest::collection::vec(-3i32..4, 40),
    ) {
        let k = 1 + k_seed % experts;
        let values = values[..rows * experts].iter().map(|&value| value as f32).collect::<Vec<_>>();
        let (selected, ids) = deterministic_top_k(logits(&values, experts).unsqueeze_dim::<3>(0), k);
        let ids = ids.into_data().to_vec::<i32>().unwrap();
        let selected = selected.into_data().to_vec::<f32>().unwrap();
        for row in 0..rows {
            let mut order = (0..experts).collect::<Vec<_>>();
            order.sort_by(|&a, &b| values[row * experts + b]
                .partial_cmp(&values[row * experts + a]).unwrap().then(a.cmp(&b)));
            for slot in 0..k {
                proptest::prop_assert_eq!(ids[row * k + slot] as usize, order[slot]);
                proptest::prop_assert_eq!(selected[row * k + slot], values[row * experts + order[slot]]);
            }
        }
    }

    /// Renormalized, scaled weights sum to the routed scale and IDs are distinct.
    #[test]
    fn renormalized_weights_sum_to_scale(
        tokens in 1usize..6,
        k in 1usize..5,
        scale in 0.1f64..4.0,
        sigmoid in proptest::bool::ANY,
        values in proptest::collection::vec(-4.0f32..4.0, 48),
    ) {
        let mut router = RouterConfig::new(8, k);
        router.routed_scale = scale;
        if sigmoid {
            router.policy.scoring = RouterScoring::Sigmoid;
        }
        let (ids, weights, flags) =
            host(router.route(logits(&values[..tokens * 8], 8), None).unwrap());
        proptest::prop_assert!(flags.iter().all(|flag| !flag));
        for (row_ids, row_weights) in ids.chunks(k).zip(weights.chunks(k)) {
            let sum: f32 = row_weights.iter().sum();
            proptest::prop_assert!((f64::from(sum) - scale).abs() < 1e-5 * scale.max(1.0));
            let mut unique = row_ids.to_vec();
            unique.sort_unstable();
            unique.dedup();
            proptest::prop_assert_eq!(unique.len(), k);
            proptest::prop_assert!(row_ids.iter().all(|&id| (0..8).contains(&id)));
        }
    }
}

#[cfg(feature = "cpu")]
mod cubecl_cpu {
    //! Routing on Burn's CubeCL CPU backend, which shares the GPU backends' reductions.
    //!
    //! These tests run wherever the `cpu` feature builds (`cargo test
    //! --features cpu --lib cubecl_cpu`). The CubeCL CPU backend compiles the
    //! same `cubek-reduce` kernels as Metal, CUDA, and ROCm, so it reproduces
    //! their maximum reduction without a GPU.

    use burn::{backend::Cpu, tensor::Tensor};

    /// [`super::deterministic_top_k`] stays total on CubeCL, where a row of `-inf` has maximum `f32::MIN`.
    #[test]
    fn cubecl_cpu_top_k_is_total_for_non_finite_rows() {
        super::check_top_k_is_total_for_non_finite_rows::<Cpu>(&Default::default());
    }

    /// CubeCL orders the lowest key class by index, exactly as Flex does.
    #[test]
    fn cubecl_cpu_top_k_ranks_the_lowest_class_by_index() {
        super::check_top_k_ranks_the_lowest_class_by_index::<Cpu>(&Default::default());
    }

    /// Burn 0.21's CubeCL maximum of a row of `-inf` is `f32::MIN`, not `-inf`.
    ///
    /// `cubek-reduce` 0.2 seeds `Max` with the lowest finite value
    /// (`docs/upstream/16-cubecl-max-reduction-negative-infinity.md`), which is
    /// why [`super::deterministic_top_k`] clamps its ranking keys to finite
    /// values. Flex returns `-inf`. When a Burn upgrade fixes the reduction,
    /// this test fails and the clamp can be revisited.
    #[test]
    fn cubecl_max_reduction_of_negative_infinities_justifies_the_finite_keys() {
        let row = || [[f32::NEG_INFINITY; 4]];
        let cubecl = Tensor::<Cpu, 2>::from_floats(row(), &Default::default())
            .max_dim(1)
            .into_data()
            .to_vec::<f32>()
            .expect("F32 host values");
        assert_eq!(cubecl, [f32::MIN]);
        let flex = Tensor::<super::TestBackend, 2>::from_floats(row(), &Default::default())
            .max_dim(1)
            .into_data()
            .to_vec::<f32>()
            .expect("F32 host values");
        assert_eq!(flex, [f32::NEG_INFINITY]);
    }
}
