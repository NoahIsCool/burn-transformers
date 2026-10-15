//! Routing-policy mapping, NaN safety, and auxiliary-loss oracles for DeepSeek MoE.
//!
//! Model-level routing parity with Hugging Face lives in `reference_tests.rs`;
//! these tests pin the configuration-to-policy map, the official auxiliary
//! loss values (`goldens.rs`), padding and inactive-row semantics, and a
//! property test against an F64 host transcription of the loss equations.

use burn::{
    backend::Flex,
    tensor::{Bool, Tensor, TensorData},
};
use proptest::prelude::*;

use super::super::{goldens, tests_support};
use super::*;
use crate::models::deepseek_v2::{DeepseekScoringFunction, DeepseekTopKMethod};

/// Deterministic F32 CPU backend with I32 indices.
type TestBackend = Flex<f32, i32>;

/// Relative tolerance of F32 loss reductions against the F64/PyTorch oracles.
const LOSS_TOLERANCE: f32 = 2e-6;

/// Builds `[batch, sequence, experts]` logits from row-major values.
fn logits(values: &[f32], shape: [usize; 3]) -> Tensor<TestBackend, 3> {
    Tensor::from_data(TensorData::new(values.to_vec(), shape), &Default::default())
}

/// Builds `[batch, sequence, K]` expert IDs from row-major values.
fn selections(values: &[i64], shape: [usize; 3]) -> Tensor<TestBackend, 3, Int> {
    Tensor::from_data(TensorData::new(values.to_vec(), shape), &Default::default())
}

/// Builds a `true == valid` token mask.
fn token_mask(values: Vec<bool>, shape: [usize; 2]) -> TokenMask<TestBackend> {
    TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_data(
        TensorData::new(values, shape),
        &Default::default(),
    ))
}

/// Reads a one-element loss.
fn scalar(loss: Tensor<TestBackend, 1>) -> f32 {
    loss.into_data().to_vec::<f32>().unwrap()[0]
}

/// Asserts `actual` is within the relative loss tolerance of `expected`.
fn assert_loss(actual: f32, expected: f32) {
    assert!(
        (actual - expected).abs() <= LOSS_TOLERANCE * expected.abs().max(1.0),
        "loss {actual} differs from {expected}"
    );
}

/// The three referenced scoring/selection pairs map onto exact shared policies.
#[test]
fn routing_policies_follow_the_pinned_references() {
    let mut config = tests_support::tiny_config();
    let lite = router_config(&config).unwrap();
    assert_eq!(lite.policy.scoring, RouterScoring::Softmax);
    assert_eq!(lite.policy.selection, ExpertSelection::Greedy);
    assert!(!lite.policy.correction_bias);
    assert_eq!(lite.policy.projection, RouterProjectionPrecision::Fp32);
    assert!(!lite.renormalize_selected);
    assert_eq!(lite.routed_scale, config.routed_scaling_factor);

    // V2 renormalizes a multi-expert mixture and then drops the routed scale.
    config.norm_topk_prob = true;
    config.routed_scaling_factor = 3.0;
    let normalized = router_config(&config).unwrap();
    assert!(normalized.renormalize_selected);
    assert_eq!(normalized.routed_scale, 1.0);
    assert_eq!(
        normalized.policy.denominator,
        RenormalizationDenominator::PlusEpsilon(1e-20)
    );

    let grouped = router_config(&tests_support::reference_v2_config()).unwrap();
    assert_eq!(
        grouped.policy.selection,
        ExpertSelection::GroupLimited(GroupLimitedSelection {
            groups: 4,
            top_groups: 2,
            group_score: GroupScore::Max,
            pruned_score: PrunedExpertScore::Zero,
        })
    );
    assert_eq!(grouped.routed_scale, 1.5);

    // V3 always applies the routed scale, also after renormalization.
    let v3 = router_config(&tests_support::reference_v3_config()).unwrap();
    assert_eq!(v3.policy.scoring, RouterScoring::Sigmoid);
    assert!(v3.policy.correction_bias);
    assert!(v3.renormalize_selected);
    assert_eq!(v3.routed_scale, 2.5);
    assert!(matches!(
        v3.policy.selection,
        ExpertSelection::GroupLimited(GroupLimitedSelection {
            group_score: GroupScore::TopTwoSum,
            pruned_score: PrunedExpertScore::Zero,
            ..
        })
    ));
}

/// Scoring/selection pairs without a reference implementation fail at initialization.
#[test]
fn unreferenced_routing_combinations_are_rejected() {
    for (scoring, method) in [
        (
            DeepseekScoringFunction::Softmax,
            DeepseekTopKMethod::NoAuxTc,
        ),
        (DeepseekScoringFunction::Sigmoid, DeepseekTopKMethod::Greedy),
        (
            DeepseekScoringFunction::Sigmoid,
            DeepseekTopKMethod::GroupLimitedGreedy,
        ),
    ] {
        let mut config = tests_support::tiny_config();
        config.scoring_func = scoring;
        config.topk_method = method;
        let error = DeepseekV2Moe::<TestBackend>::init(&config, &Default::default()).unwrap_err();
        assert!(
            matches!(&error, Error::BackendCapability(message) if message.contains("no reference implementation")),
            "{scoring:?}/{method:?}: {error}"
        );
    }
}

/// `noaux_tc` allocates the frozen F32 selection bias; other policies have none.
#[test]
fn noaux_routing_allocates_a_frozen_f32_correction_bias() {
    let device = Default::default();
    let v3 =
        DeepseekV2Moe::<TestBackend>::init(&tests_support::reference_v3_config(), &device).unwrap();
    let bias = v3.routed.correction_bias.as_ref().expect("noaux_tc bias");
    assert_eq!(bias.val().dims(), [8]);
    assert_eq!(bias.val().dtype(), DType::F32);
    assert!(!bias.is_require_grad());
    let lite = DeepseekV2Moe::<TestBackend>::init(&tests_support::tiny_config(), &device).unwrap();
    assert!(lite.routed.correction_bias.is_none());
}

/// Equal scores pick the lowest-indexed groups and experts on every backend.
#[test]
fn group_limited_ties_resolve_to_the_smallest_ids() {
    let router = router_config(&tests_support::reference_v2_config()).unwrap();
    let decision = router
        .route(
            Tensor::<TestBackend, 2>::zeros([1, 8], &Default::default()),
            None,
        )
        .unwrap();
    let ids = decision
        .experts
        .into_data()
        .convert::<i64>()
        .to_vec::<i64>()
        .unwrap();
    assert_eq!(ids, [0, 1, 2]);
}

/// NaN router scores are contextual errors, never out-of-range expert indices (DSV-4).
#[test]
fn nan_router_scores_are_contextual_errors() {
    let device = Default::default();
    for config in [
        tests_support::tiny_config(),
        tests_support::reference_v2_config(),
        tests_support::reference_v3_config(),
    ] {
        let moe = DeepseekV2Moe::<TestBackend>::init(&config, &device).unwrap();
        let mut values = vec![0.25_f32; 2 * 8];
        values[8] = f32::NAN;
        let input =
            Tensor::<TestBackend, 3>::from_data(TensorData::new(values, [1, 2, 8]), &device);
        let error = moe.forward(input).unwrap_err();
        assert!(
            matches!(&error, Error::Shape { tensor, .. } if tensor == "moe.router_scores"),
            "{error}"
        );
        assert!(error.to_string().contains("flattened token 1"), "{error}");
    }
}

/// Both loss forms reproduce the official `MoEGate` values on the HF golden routes.
#[test]
fn auxiliary_losses_match_the_pinned_reference() {
    let router = logits(&goldens::V2_ROUTER_LOGITS, [2, 6, 8]);
    let ids = selections(&goldens::V2_IDS_SORTED, [2, 6, 3]);
    for sequence_wise in [true, false] {
        let row = scalar(
            auxiliary_loss(
                router.clone().narrow(0, 0, 1),
                ids.clone().narrow(0, 0, 1),
                None,
                RouterScoring::Softmax,
                sequence_wise,
            )
            .unwrap(),
        );
        assert_loss(row, goldens::AUX_ROW0);
    }
    let mask = token_mask(
        [[true; 6], [false, false, true, true, true, true]].concat(),
        [2, 6],
    );
    let padded = |sequence_wise| {
        scalar(
            auxiliary_loss(
                router.clone(),
                ids.clone(),
                Some(&mask),
                RouterScoring::Softmax,
                sequence_wise,
            )
            .unwrap(),
        )
    };
    assert_loss(padded(true), goldens::AUX_PADDED_SEQUENCE);
    assert_loss(padded(false), goldens::AUX_PADDED_GLOBAL);
}

/// The global loss is the reference value, not `K` times it (DSV-2 regression).
///
/// The shared Qwen/Mixtral loss sums assignment means over the `K` slots,
/// which is exactly `K` times DeepSeek's `one_hot(topk_idx).mean(0)` form.
#[test]
fn global_auxiliary_loss_is_not_scaled_by_top_k() {
    let router = logits(&goldens::V2_ROUTER_LOGITS, [2, 6, 8]);
    let ids = selections(&goldens::V2_IDS_SORTED, [2, 6, 3]);
    let deepseek = scalar(
        auxiliary_loss(
            router.clone(),
            ids.clone(),
            None,
            RouterScoring::Softmax,
            false,
        )
        .unwrap(),
    );
    let shared = scalar(crate::moe::load_balancing_loss(router, ids, None).unwrap());
    assert_loss(shared, 3.0 * deepseek);
    assert!(
        (shared - deepseek).abs() > 1.0,
        "the regression must be observable"
    );
}

/// Diagnostics scale the loss by `aux_loss_alpha` and keep logits unchanged.
#[test]
fn router_diagnostics_apply_alpha_and_requested_outputs() {
    let config = tests_support::reference_v2_config();
    let router = logits(&goldens::V2_ROUTER_LOGITS, [2, 6, 8]).narrow(0, 0, 1);
    let ids = selections(&goldens::V2_IDS_SORTED, [2, 6, 3]).narrow(0, 0, 1);
    let options = ForwardOptions {
        router_logits: true,
        auxiliary_losses: true,
        ..ForwardOptions::default()
    };
    let diagnostics =
        router_diagnostics(router.clone(), ids.clone(), None, &options, &config).unwrap();
    assert_loss(
        scalar(diagnostics.load_balancing_loss.unwrap()),
        goldens::AUX_ROW0 * config.aux_loss_alpha as f32,
    );
    assert_eq!(diagnostics.logits.unwrap().dims(), [1, 6, 8]);
    let quiet = router_diagnostics(router, ids, None, &ForwardOptions::default(), &config).unwrap();
    assert!(quiet.logits.is_none() && quiet.load_balancing_loss.is_none());
}

/// Fully masked (finished) rows neither contribute nor dilute the per-row mean.
#[test]
fn inactive_rows_do_not_dilute_the_sequence_loss() {
    let router = logits(&goldens::V2_ROUTER_LOGITS, [2, 6, 8]);
    let ids = selections(&goldens::V2_IDS_SORTED, [2, 6, 3]);
    let mask = token_mask([[true; 6], [false; 6]].concat(), [2, 6]);
    let loss =
        scalar(auxiliary_loss(router, ids, Some(&mask), RouterScoring::Softmax, true).unwrap());
    assert_loss(loss, goldens::AUX_ROW0);
}

/// Non-finite scores at padded tokens cannot poison the loss.
#[test]
fn padded_non_finite_scores_are_ignored() {
    let mut values = goldens::V2_ROUTER_LOGITS.to_vec();
    values[6 * 8] = f32::NAN;
    values[7 * 8 + 3] = f32::INFINITY;
    let mask = token_mask(
        [[true; 6], [false, false, true, true, true, true]].concat(),
        [2, 6],
    );
    let loss = scalar(
        auxiliary_loss(
            logits(&values, [2, 6, 8]),
            selections(&goldens::V2_IDS_SORTED, [2, 6, 3]),
            Some(&mask),
            RouterScoring::Softmax,
            true,
        )
        .unwrap(),
    );
    assert_loss(loss, goldens::AUX_PADDED_SEQUENCE);
}

/// Mismatched or empty inputs are shape errors.
#[test]
fn auxiliary_loss_rejects_inconsistent_shapes() {
    let router = logits(&[0.0; 8], [1, 2, 4]);
    let error = auxiliary_loss(
        router.clone(),
        selections(&[0, 1, 2], [1, 3, 1]),
        None,
        RouterScoring::Softmax,
        true,
    )
    .unwrap_err();
    assert!(matches!(error, Error::Shape { tensor, .. } if tensor == "deepseek_v2.router_aux"));
    let error = auxiliary_loss(
        router,
        selections(&[0, 1], [1, 2, 1]),
        Some(&token_mask(vec![true; 3], [1, 3])),
        RouterScoring::Softmax,
        true,
    )
    .unwrap_err();
    assert!(
        matches!(error, Error::Shape { tensor, .. } if tensor == "deepseek_v2.router_aux.mask")
    );
}

/// F64 host transcription of the module equations.
fn host_loss(
    router: &[f32],
    ids: &[i64],
    valid: &[bool],
    [batch, sequence, experts, top_k]: [usize; 4],
    scoring: RouterScoring,
    sequence_wise: bool,
) -> f64 {
    let probability = |token: usize| -> Vec<f64> {
        let row = &router[token * experts..(token + 1) * experts];
        match scoring {
            RouterScoring::Softmax => {
                let peak = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
                let exps = row
                    .iter()
                    .map(|&z| (z as f64 - peak).exp())
                    .collect::<Vec<_>>();
                let sum: f64 = exps.iter().sum();
                exps.iter().map(|e| e / sum).collect()
            }
            RouterScoring::Sigmoid => {
                let scores = row
                    .iter()
                    .map(|&z| 1.0 / (1.0 + (-(z as f64)).exp()))
                    .collect::<Vec<_>>();
                let sum: f64 = scores.iter().sum();
                scores.iter().map(|s| s / sum).collect()
            }
        }
    };
    let mut counts = vec![vec![0.0_f64; experts]; batch];
    let mut mass = vec![vec![0.0_f64; experts]; batch];
    let mut tokens = vec![0.0_f64; batch];
    for b in 0..batch {
        for t in 0..sequence {
            let token = b * sequence + t;
            if !valid[token] {
                continue;
            }
            tokens[b] += 1.0;
            for slot in 0..top_k {
                counts[b][ids[token * top_k + slot] as usize] += 1.0;
            }
            for (e, p) in probability(token).into_iter().enumerate() {
                mass[b][e] += p;
            }
        }
    }
    let (e, k) = (experts as f64, top_k as f64);
    if sequence_wise {
        let active: Vec<usize> = (0..batch).filter(|&b| tokens[b] > 0.0).collect();
        let total: f64 = active
            .iter()
            .map(|&b| {
                (0..experts)
                    .map(|x| counts[b][x] * e / (tokens[b] * k) * mass[b][x] / tokens[b])
                    .sum::<f64>()
            })
            .sum();
        total / active.len().max(1) as f64
    } else {
        let all: f64 = tokens.iter().sum::<f64>().max(1.0);
        (0..experts)
            .map(|x| {
                let count: f64 = (0..batch).map(|b| counts[b][x]).sum();
                let p: f64 = (0..batch).map(|b| mass[b][x]).sum();
                count * e / (all * k) * p / all
            })
            .sum()
    }
}

/// Strategy for a loss case: geometry, logits, distinct routes, and validity.
fn loss_case() -> impl Strategy<Value = ([usize; 4], Vec<f32>, Vec<i64>, Vec<bool>)> {
    (1usize..=3, 1usize..=4, 2usize..=6)
        .prop_flat_map(|(batch, sequence, experts)| {
            (1usize..=experts).prop_flat_map(move |top_k| {
                let tokens = batch * sequence;
                (
                    Just([batch, sequence, experts, top_k]),
                    proptest::collection::vec(-4.0_f32..4.0, tokens * experts),
                    proptest::collection::vec(
                        Just((0..experts as i64).collect::<Vec<_>>()).prop_shuffle(),
                        tokens,
                    ),
                    proptest::collection::vec(any::<bool>(), tokens),
                )
            })
        })
        .prop_map(|(shape, router, permutations, valid)| {
            let ids = permutations
                .into_iter()
                .flat_map(|permutation| permutation.into_iter().take(shape[3]))
                .collect();
            (shape, router, ids, valid)
        })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Device losses equal the F64 equations for both forms, scorings, and masks.
    #[test]
    fn auxiliary_loss_matches_the_host_equations(
        (shape, router, ids, valid) in loss_case(),
        sequence_wise in any::<bool>(),
        sigmoid in any::<bool>(),
    ) {
        let [batch, sequence, experts, top_k] = shape;
        let scoring = if sigmoid { RouterScoring::Sigmoid } else { RouterScoring::Softmax };
        let mask = token_mask(valid.clone(), [batch, sequence]);
        let actual = scalar(auxiliary_loss(
            logits(&router, [batch, sequence, experts]),
            selections(&ids, [batch, sequence, top_k]),
            Some(&mask),
            scoring,
            sequence_wise,
        ).unwrap());
        let expected = host_loss(&router, &ids, &valid, shape, scoring, sequence_wise);
        prop_assert!(
            (f64::from(actual) - expected).abs() <= 1e-5 * expected.abs().max(1.0),
            "device {actual} vs host {expected}"
        );
    }
}

/// The balance loss back-propagates into the router projection (training path).
///
/// Training adds `alpha * aux` to the language-model loss and runs one
/// backward pass, so the test differentiates `sum(hidden) + 100 aux` and
/// checks that the router gradient differs from that of `sum(hidden)` alone.
/// (A backward pass over the auxiliary loss *alone* after routed dispatch
/// currently yields no router gradient on Burn 0.21 autodiff; see the
/// package report's follow-ups.)
#[cfg(feature = "autodiff")]
#[test]
fn auxiliary_loss_gradient_reaches_the_router() {
    use burn::backend::Autodiff;

    /// Reverse-mode backend over the deterministic CPU backend.
    type AdBackend = Autodiff<TestBackend>;
    let device = Default::default();
    // Installed (not lazily initialized) parameters, as after checkpoint loading.
    let model =
        tests_support::reference_model::<AdBackend>(tests_support::reference_v2_config(), &device);
    let moe = model.model.layers[1].sparse_mlp.as_ref().unwrap();
    let input = || {
        Tensor::<AdBackend, 3>::from_data(
            TensorData::new(
                (0..2 * 3 * 8)
                    .map(|value| (value as f32 * 0.37).sin())
                    .collect(),
                [2, 3, 8],
            ),
            &device,
        )
    };
    let router_gradient = |auxiliary_weight: Option<(f32, bool)>| {
        let output = moe.forward(input()).unwrap();
        let mut loss = output.hidden_state.sum();
        if let Some((weight, sequence_wise)) = auxiliary_weight {
            loss = loss
                + auxiliary_loss(
                    output.routed.router_logits,
                    output.routed.selected_experts,
                    None,
                    RouterScoring::Softmax,
                    sequence_wise,
                )
                .unwrap()
                .sum()
                    * weight;
        }
        let gradients = loss.backward();
        moe.routed
            .router
            .weight
            .grad(&gradients)
            .expect("router gradient")
            .into_data()
            .to_vec::<f32>()
            .unwrap()
    };
    let base = router_gradient(None);
    for sequence_wise in [true, false] {
        let with_auxiliary = router_gradient(Some((100.0, sequence_wise)));
        assert!(with_auxiliary.iter().all(|value| value.is_finite()));
        let change: f32 = base
            .iter()
            .zip(&with_auxiliary)
            .map(|(left, right)| (left - right).abs())
            .sum();
        assert!(
            change > 1e-3,
            "auxiliary loss changed the router gradient by {change}"
        );
    }
}
