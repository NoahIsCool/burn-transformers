//! Nemotron-H hybrid forward, sparse routing, precision, and autodiff tests.

use burn::{
    backend::Flex,
    tensor::{TensorData, Tolerance, f16},
};

use super::*;
use crate::LogitsSelection;

/// CPU Flex backend with F32 activations and I32 token indices for tiny models.
type TestBackend = Flex<f32, i32>;

/// Converts one test projection to F16 without changing its logical shape.
fn cast_linear_to_f16(linear: &mut Linear<TestBackend>) {
    linear.weight = Param::from_tensor(linear.weight.val().cast(DType::F16));
    if let Some(bias) = linear.bias.take() {
        linear.bias = Some(Param::from_tensor(bias.val().cast(DType::F16)));
    }
}

/// Converts both projections of one test expert to F16.
fn cast_dense_to_f16(dense: &mut NemotronHDense<TestBackend>) {
    cast_linear_to_f16(&mut dense.up_proj);
    cast_linear_to_f16(&mut dense.down_proj);
}

/// Reduced schedule retaining every released block kind.
pub(crate) fn tiny_config(mtp: bool) -> NemotronHConfig {
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
        moe_intermediate_size: 6,
        moe_shared_expert_intermediate_size: 8,
        num_experts_per_tok: 2,
        n_group: 2,
        topk_group: 1,
        num_nextn_predict_layers: usize::from(mtp),
        ..Default::default()
    }
}

/// The schedule instantiates exactly one matching component in every block.
#[test]
fn schedule_builds_each_mixer_kind() {
    let device = Default::default();
    let model = NemotronHForCausalLm::<TestBackend>::init(tiny_config(false), &device).unwrap();
    assert!(model.model.layers[0].mamba.is_some());
    assert!(model.model.layers[1].attention.is_some());
    assert!(model.model.layers[2].mlp.is_some());
    assert!(model.model.layers[3].moe.is_some());
}

/// Nemotron's upstream `rsqrt` plus early-rounding graph is F16 observable.
///
/// The expected bits come from the pinned Transformers Nemotron-H fallback
/// expression `weight * (x.float() * torch.rsqrt(ms)).to(x.dtype)`.
#[test]
fn rms_norm_matches_pinned_f16_operation_order() {
    let device = Default::default();
    let mut norm = nemotron_rms_norm::<TestBackend>(2, 1e-6, &device).unwrap();
    norm.weight =
        Param::from_tensor(Tensor::from_floats([0.288_818_36, -3.437_5], &device).cast(DType::F16));
    let input = Tensor::<TestBackend, 2>::from_floats([[7.457_031_25, -0.452_148_44]], &device)
        .cast(DType::F16);

    assert!(norm.round_before_scale);
    assert_eq!(norm.root_mode, RmsNormRootMode::MultiplyByReciprocalSqrt);
    let bits = norm
        .forward(input)
        .into_data()
        .to_vec::<f16>()
        .unwrap()
        .into_iter()
        .map(f16::to_bits)
        .collect::<Vec<_>>();
    assert_eq!(bits, [0x3685, 0x34b5]);
}

/// Stateless inference honors output and router diagnostic selections.
#[test]
fn stateless_forward_returns_requested_shapes() {
    let device = Default::default();
    let model = NemotronHForCausalLm::<TestBackend>::init(tiny_config(false), &device).unwrap();
    let output = model
        .forward(
            CausalLmInput {
                token_ids: Tensor::from_ints([[1, 2, 3]], &device),
                token_mask: None,
                position_ids: None,
            },
            &ForwardOptions {
                logits: LogitsSelection::All,
                hidden_states: true,
                router_logits: true,
                auxiliary_losses: true,
                ..ForwardOptions::default()
            },
        )
        .unwrap();
    assert_eq!(output.logits.dims(), [1, 3, 32]);
    assert_eq!(output.logits.dtype(), DType::F32);
    assert_eq!(output.hidden_states.len(), 6);
    assert_eq!(output.routers.len(), 1);
    assert_eq!(output.routers[0].logits.as_ref().unwrap().dims(), [1, 3, 4]);
    assert!(output.routers[0].load_balancing_loss.is_some());
}

/// Sparse routing never evaluates experts without assignments.
#[test]
fn moe_dispatches_only_selected_experts() {
    let device = Default::default();
    let config = tiny_config(false);
    let moe = NemotronHMoe::<TestBackend>::init(&config, &device).unwrap();
    let output = moe.forward(Tensor::zeros([1, 1, 8], &device)).unwrap();
    assert_eq!(output.expert_token_counts.iter().sum::<usize>(), 2);
    assert_eq!(output.evaluated_experts, 2);
    assert_eq!(output.expert_token_counts, vec![1, 1, 0, 0]);
}

/// Lightning's single router group is the identity eligibility transform.
#[test]
fn moe_single_group_matches_all_groups_retained() {
    let device = Default::default();
    let config = tiny_config(false);
    let mut moe = NemotronHMoe::<TestBackend>::init(&config, &device).unwrap();
    let input = Tensor::from_floats([[[0.2, -0.5, 0.8, 0.1, -0.3, 0.7, 0.4, -0.9]]], &device);

    moe.groups = 1;
    moe.top_groups = 1;
    let single_group = moe.forward(input.clone()).unwrap();

    moe.groups = 2;
    moe.top_groups = 2;
    let all_groups = moe.forward(input).unwrap();

    assert_eq!(
        single_group.selected_experts.into_data(),
        all_groups.selected_experts.into_data()
    );
    single_group
        .selected_weights
        .into_data()
        .assert_approx_eq::<f32>(
            &all_groups.selected_weights.into_data(),
            Tolerance::absolute(0.0),
        );
    single_group
        .hidden_state
        .into_data()
        .assert_approx_eq::<f32>(
            &all_groups.hidden_state.into_data(),
            Tolerance::absolute(1e-6),
        );
}

/// Router buffers and sparse accumulation retain their reference precisions.
#[test]
fn moe_mixed_precision_keeps_router_and_accumulator_fp32() {
    let device = Default::default();
    let config = tiny_config(false);
    let mut moe = NemotronHMoe::<TestBackend>::init(&config, &device).unwrap();
    cast_linear_to_f16(&mut moe.router);
    for expert in &mut moe.routed {
        cast_dense_to_f16(expert);
    }
    cast_dense_to_f16(&mut moe.shared);
    if let Some(projection) = &mut moe.fc1_latent_proj {
        cast_linear_to_f16(projection);
    }
    if let Some(projection) = &mut moe.fc2_latent_proj {
        cast_linear_to_f16(projection);
    }

    assert_eq!(moe.correction_bias.val().dtype(), DType::F32);
    assert!(!moe.correction_bias.val().is_require_grad());
    let output = moe
        .forward(Tensor::ones([1, 3, 8], (&device, DType::F16)))
        .unwrap();
    assert_eq!(output.hidden_state.dtype(), DType::F16);
    assert_eq!(output.router_logits.dtype(), DType::F32);
    assert_eq!(output.selected_weights.dtype(), DType::F32);
    assert!(
        output
            .hidden_state
            .cast(DType::F32)
            .to_data()
            .as_slice::<f32>()
            .unwrap()
            .iter()
            .all(|value| value.is_finite())
    );
}

/// Invalid MTP widths are rejected before any parameter allocation occurs.
#[test]
fn mtp_width_overflow_returns_configuration_error() {
    let device = Default::default();
    let mut config = tiny_config(true);
    config.hidden_size = usize::MAX;
    let error = NemotronHForCausalLm::<TestBackend>::init(config, &device).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("MTP concatenation width overflows")
    );
}

/// The optional MTP body consumes aligned base hidden states and shared head.
#[test]
fn checkpoint_declared_next_token_head_runs() {
    let device = Default::default();
    let model = NemotronHForCausalLm::<TestBackend>::init(tiny_config(true), &device).unwrap();
    let logits = model
        .predict_next_token_logits(
            Tensor::from_ints([[2, 3]], &device),
            Tensor::zeros([1, 2, 8], &device),
        )
        .unwrap();
    assert_eq!(logits.dims(), [1, 2, 32]);
}

/// Invalid block diagnostics and token indices fail without panicking.
#[test]
fn invalid_inputs_are_contextual_errors() {
    let device = Default::default();
    let model = NemotronHForCausalLm::<TestBackend>::init(tiny_config(false), &device).unwrap();
    let error = model
        .forward(
            CausalLmInput {
                token_ids: Tensor::from_ints([[99]], &device),
                token_mask: None,
                position_ids: None,
            },
            &ForwardOptions::default(),
        )
        .unwrap_err();
    assert!(error.to_string().contains("0..32"));
    assert!(
        model
            .forward(
                CausalLmInput {
                    token_ids: Tensor::from_ints([[1]], &device),
                    token_mask: None,
                    position_ids: None,
                },
                &ForwardOptions {
                    attentions: true,
                    ..ForwardOptions::default()
                },
            )
            .is_err()
    );
}

/// All-logit and last-logit projections agree at the final position.
#[test]
fn last_logits_match_full_forward_tail() {
    let device = Default::default();
    let model = NemotronHForCausalLm::<TestBackend>::init(tiny_config(false), &device).unwrap();
    let input = || CausalLmInput {
        token_ids: Tensor::from_ints([[1, 2, 3]], &device),
        token_mask: None,
        position_ids: None,
    };
    let all = model
        .forward(
            input(),
            &ForwardOptions {
                logits: LogitsSelection::All,
                ..ForwardOptions::default()
            },
        )
        .unwrap()
        .logits
        .narrow(1, 2, 1);
    let last = model
        .forward(input(), &ForwardOptions::default())
        .unwrap()
        .logits;
    last.into_data()
        .assert_approx_eq::<f32>(&all.into_data(), Tolerance::absolute(1e-5));
}

/// Full-precision stateless execution retains parameter gradients.
#[cfg(feature = "autodiff")]
#[test]
fn autodiff_smoke_reaches_embedding() {
    use burn::backend::Autodiff;

    /// Autodiff wrapper used to check gradient flow back to the embedding table.
    type AdBackend = Autodiff<TestBackend>;
    let device = Default::default();
    let model = NemotronHForCausalLm::<AdBackend>::init(tiny_config(false), &device).unwrap();
    let loss = model
        .forward(
            CausalLmInput {
                token_ids: Tensor::<AdBackend, 2, Int>::from_ints([[1, 2]], &device),
                token_mask: None,
                position_ids: None,
            },
            &ForwardOptions::default(),
        )
        .unwrap()
        .logits
        .sum();
    let gradients = loss.backward();
    assert!(model.model.embed_tokens.weight.grad(&gradients).is_some());
}

/// Replaces a projection's weight and bias with deterministic, non-trivial values.
///
/// `salt` decorrelates projections so routing is non-degenerate: distinct
/// experts produce distinct outputs and router scores rarely tie.
fn fill_linear(linear: &mut Linear<TestBackend>, salt: usize) {
    let device = linear.weight.val().device();
    let dtype = linear.weight.val().dtype();
    let dims = linear.weight.val().dims();
    let values = (0..dims[0] * dims[1])
        .map(|index| (((index * 7 + salt * 131) as f32 + 0.25) * 0.618).sin() * 0.4)
        .collect::<Vec<_>>();
    linear.weight = Param::from_tensor(
        Tensor::<TestBackend, 2>::from_data(TensorData::new(values, dims), &device).cast(dtype),
    );
    if let Some(bias) = linear.bias.take() {
        let width = bias.val().dims()[0];
        let values = (0..width)
            .map(|index| ((index + salt) as f32 * 0.37).cos() * 0.1)
            .collect::<Vec<_>>();
        linear.bias = Some(Param::from_tensor(
            Tensor::<TestBackend, 1>::from_data(TensorData::new(values, [width]), &device)
                .cast(dtype),
        ));
    }
}

/// Builds an MoE whose router, experts, and correction bias are all non-trivial.
fn deterministic_moe(config: &NemotronHConfig) -> NemotronHMoe<TestBackend> {
    let device = Default::default();
    let mut moe = NemotronHMoe::<TestBackend>::init(config, &device).unwrap();
    fill_linear(&mut moe.router, 1);
    for (index, expert) in moe.routed.iter_mut().enumerate() {
        fill_linear(&mut expert.up_proj, 10 + 2 * index);
        fill_linear(&mut expert.down_proj, 11 + 2 * index);
    }
    fill_linear(&mut moe.shared.up_proj, 90);
    fill_linear(&mut moe.shared.down_proj, 91);
    if let Some(projection) = &mut moe.fc1_latent_proj {
        fill_linear(projection, 95);
    }
    if let Some(projection) = &mut moe.fc2_latent_proj {
        fill_linear(projection, 96);
    }
    let experts = moe.routed.len();
    moe.correction_bias = Param::from_tensor(Tensor::from_data(
        TensorData::new(
            (0..experts)
                .map(|expert| (expert as f32 * 1.3).sin() * 0.05)
                .collect::<Vec<_>>(),
            [experts],
        ),
        &device,
    ))
    .set_require_grad(false);
    moe
}

/// Deterministic `[batch, sequence, width]` activations with varied signs.
fn moe_input(batch: usize, sequence: usize, width: usize) -> Tensor<TestBackend, 3> {
    let values = (0..batch * sequence * width)
        .map(|index| ((index as f32 + 0.5) * 0.731).sin() * 1.7)
        .collect::<Vec<_>>();
    Tensor::from_data(
        TensorData::new(values, [batch, sequence, width]),
        &Default::default(),
    )
}

/// The private top-k Nemotron-H used before adopting `moe::router` (test oracle only).
fn legacy_top_k(
    values: Tensor<TestBackend, 3>,
    k: usize,
) -> (Tensor<TestBackend, 3>, Tensor<TestBackend, 3, Int>) {
    let [batch, rows, columns] = values.dims();
    let ids = Tensor::<TestBackend, 1, Int>::arange(0..columns as i64, &values.device())
        .reshape([1, 1, columns])
        .expand([batch, rows, columns]);
    let mut remaining = values.clone();
    let mut selected_values = Vec::with_capacity(k);
    let mut selected_indices = Vec::with_capacity(k);
    for _ in 0..k {
        let maximum = remaining.clone().max_dim(2);
        let ties = remaining.clone().equal(maximum);
        let index = ids
            .clone()
            .mask_fill(ties.bool_not(), columns as i64)
            .min_dim(2);
        selected_values.push(values.clone().gather(2, index.clone()));
        selected_indices.push(index.clone());
        remaining = remaining.mask_fill(
            ids.clone().equal(index.expand([batch, rows, columns])),
            f32::NEG_INFINITY,
        );
    }
    (
        Tensor::cat(selected_values, 2),
        Tensor::cat(selected_indices, 2),
    )
}

/// The complete pre-migration Nemotron-H MoE forward pass (test oracle only).
///
/// Returns `(hidden, router_logits, selected_experts, selected_weights, counts)`.
#[allow(clippy::type_complexity)]
fn legacy_moe_forward(
    moe: &NemotronHMoe<TestBackend>,
    input: Tensor<TestBackend, 3>,
) -> (
    Tensor<TestBackend, 3>,
    Tensor<TestBackend, 3>,
    Tensor<TestBackend, 3, Int>,
    Tensor<TestBackend, 3>,
    Vec<usize>,
) {
    use burn::tensor::{IndexingUpdateOp, TensorData};
    let [batch, sequence, hidden] = input.dims();
    let experts = moe.routed.len();
    let tokens = batch * sequence;
    let flat = input.reshape([tokens, hidden]);
    let router_logits = flat
        .clone()
        .cast(DType::F32)
        .matmul(moe.router.weight.val().cast(DType::F32))
        .reshape([batch, sequence, experts]);
    let scores = activation::sigmoid(router_logits.clone()).reshape([tokens, experts]);
    let corrected = scores.clone()
        + moe
            .correction_bias
            .val()
            .cast(DType::F32)
            .reshape([1, experts]);
    let eligible = if moe.groups == 1 {
        corrected
    } else {
        let per_group = experts / moe.groups;
        let candidates = legacy_top_k(
            corrected.clone().reshape([tokens, moe.groups, per_group]),
            2.min(per_group),
        )
        .0
        .sum_dim(2)
        .reshape([1, tokens, moe.groups]);
        let kept = legacy_top_k(candidates, moe.top_groups)
            .1
            .reshape([tokens, moe.top_groups]);
        let mask = kept
            .one_hot::<3>(moe.groups)
            .sum_dim(1)
            .greater_elem(0)
            .reshape([tokens, moe.groups, 1])
            .expand([tokens, moe.groups, per_group])
            .reshape([tokens, experts]);
        corrected.mask_fill(mask.bool_not(), f32::NEG_INFINITY)
    };
    let selected = legacy_top_k(
        eligible.reshape([1, tokens, experts]),
        moe.experts_per_token,
    )
    .1
    .reshape([tokens, moe.experts_per_token]);
    let mut weights = scores.gather(1, selected.clone());
    if moe.renormalize_selected {
        let denominator = weights.clone().sum_dim(1).clamp_min(1e-20);
        weights = weights / denominator;
    }
    weights = weights * moe.routed_scale;
    let routed_input = match &moe.fc1_latent_proj {
        Some(projection) => operations::linear(flat.clone(), projection).unwrap(),
        None => flat.clone(),
    };
    let dtype = routed_input.dtype();
    let groups = crate::moe::group_selected_experts(selected.clone(), experts).unwrap();
    let flat_weights = weights.clone().reshape([tokens * moe.experts_per_token]);
    let mut routed = Tensor::<TestBackend, 2>::zeros(
        [tokens, routed_input.dims()[1]],
        (&flat.device(), DType::F32),
    );
    let mut counts = Vec::new();
    for (expert, group) in moe.routed.iter().zip(groups) {
        let count = group.token_indices.len();
        counts.push(count);
        if count == 0 {
            continue;
        }
        let token_indices = Tensor::<TestBackend, 1, Int>::from_data(
            TensorData::new(group.token_indices, [count]),
            &flat.device(),
        );
        let weight_indices = Tensor::<TestBackend, 1, Int>::from_data(
            TensorData::new(group.weight_indices, [count]),
            &flat.device(),
        );
        let weights = flat_weights
            .clone()
            .select(0, weight_indices)
            .unsqueeze_dim::<2>(1);
        let contribution = expert
            .forward(routed_input.clone().select(0, token_indices.clone()))
            .unwrap()
            .cast(DType::F32)
            * weights;
        routed = routed.select_assign(0, token_indices, contribution, IndexingUpdateOp::Add);
    }
    let routed = routed.cast(dtype);
    let routed = match &moe.fc2_latent_proj {
        Some(projection) => operations::linear(routed, projection).unwrap(),
        None => routed,
    };
    let hidden_state =
        (routed + moe.shared.forward(flat).unwrap()).reshape([batch, sequence, hidden]);
    (
        hidden_state,
        router_logits,
        selected.reshape([batch, sequence, moe.experts_per_token]),
        weights.reshape([batch, sequence, moe.experts_per_token]),
        counts,
    )
}

/// Returns the raw bit patterns of a float tensor in its own dtype.
fn float_bits<const D: usize>(tensor: Tensor<TestBackend, D>) -> Vec<u32> {
    match tensor.dtype() {
        DType::F16 => tensor
            .into_data()
            .to_vec::<f16>()
            .unwrap()
            .into_iter()
            .map(|value| u32::from(value.to_bits()))
            .collect(),
        _ => tensor
            .cast(DType::F32)
            .into_data()
            .to_vec::<f32>()
            .unwrap()
            .into_iter()
            .map(f32::to_bits)
            .collect(),
    }
}

/// The shared router and dispatcher reproduce the former private MoE bit for bit.
///
/// Covers group pruning (2 groups, 1 kept), a single group, every group kept,
/// renormalization on and off, a non-unit routed scale, latent projections,
/// and reduced-precision experts with an F32 router and accumulator.
#[test]
fn moe_matches_pre_migration_implementation_bit_for_bit() {
    for case in 0..6 {
        let mut config = tiny_config(false);
        config.n_routed_experts = 8;
        config.num_experts_per_tok = 3;
        config.n_group = 4;
        config.topk_group = 2;
        config.routed_scaling_factor = 2.5;
        match case {
            1 => {
                config.n_group = 1;
                config.topk_group = 1;
            }
            2 => config.topk_group = 4,
            3 => config.norm_topk_prob = false,
            4 => config.moe_latent_size = Some(6),
            _ => {}
        }
        let mut moe = deterministic_moe(&config);
        let mut input = moe_input(2, 5, config.hidden_size);
        if case == 5 {
            cast_linear_to_f16(&mut moe.router);
            for expert in &mut moe.routed {
                cast_dense_to_f16(expert);
            }
            cast_dense_to_f16(&mut moe.shared);
            input = input.cast(DType::F16);
        }
        let (hidden, logits, selected, weights, counts) = legacy_moe_forward(&moe, input.clone());
        let actual = moe.forward(input).unwrap();
        assert_eq!(
            float_bits(actual.hidden_state),
            float_bits(hidden),
            "case {case}"
        );
        assert_eq!(
            float_bits(actual.router_logits),
            float_bits(logits),
            "case {case}"
        );
        assert_eq!(
            actual.selected_experts.into_data(),
            selected.into_data(),
            "case {case}"
        );
        assert_eq!(
            float_bits(actual.selected_weights),
            float_bits(weights),
            "case {case}"
        );
        assert_eq!(actual.expert_token_counts, counts, "case {case}");
        assert_eq!(
            actual.evaluated_experts,
            counts.iter().filter(|&&count| count > 0).count()
        );
    }
}

/// A NaN router row is a contextual error naming the token, not an out-of-bounds gather.
#[test]
fn moe_rejects_non_finite_router_scores_contextually() {
    let config = tiny_config(false);
    let moe = deterministic_moe(&config);
    let mut values = moe_input(1, 3, config.hidden_size)
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    values[config.hidden_size..2 * config.hidden_size].fill(f32::NAN);
    let input = Tensor::from_data(
        TensorData::new(values, [1, 3, config.hidden_size]),
        &Default::default(),
    );
    let error = moe.forward(input).unwrap_err();
    assert!(
        matches!(&error, Error::Shape { tensor, actual, .. }
            if tensor == "moe.router_scores" && actual.contains("flattened token 1")),
        "{error}"
    );
}

/// Mutated cardinalities that group pruning cannot satisfy fail before routing.
#[test]
fn moe_rejects_more_experts_than_retained_groups_hold() {
    let config = tiny_config(false);
    let mut moe = deterministic_moe(&config);
    // Two of four experts survive top-1-of-2 group pruning, fewer than three.
    moe.experts_per_token = 3;
    let error = moe
        .forward(moe_input(1, 2, config.hidden_size))
        .unwrap_err();
    assert!(
        matches!(&error, Error::Configuration { field, .. } if field == "moe.router.top_groups"),
        "{error}"
    );
    moe.experts_per_token = 2;
    moe.top_groups = 3;
    assert!(moe.forward(moe_input(1, 2, config.hidden_size)).is_err());
}

/// The router specification is DeepSeek-V3 `noaux_tc` with an F32 projection.
#[test]
fn moe_router_config_matches_the_reference_policy() {
    let config = tiny_config(false);
    let moe = deterministic_moe(&config);
    let router = moe.router_config();
    router.validate().unwrap();
    assert_eq!(router.experts, 4);
    assert_eq!(router.experts_per_token, 2);
    assert_eq!(router.policy.scoring, RouterScoring::Sigmoid);
    assert_eq!(router.policy.order, ScoringOrder::ScoreThenTopK);
    assert!(router.policy.correction_bias);
    assert!(!router.policy.router_bias);
    assert_eq!(router.policy.projection, RouterProjectionPrecision::Fp32);
    assert_eq!(
        router.policy.selection,
        ExpertSelection::GroupLimited(GroupLimitedSelection {
            groups: 2,
            top_groups: 1,
            group_score: GroupScore::TopTwoSum,
            pruned_score: PrunedExpertScore::NegativeInfinity,
        })
    );
    assert_eq!(
        router.policy.denominator,
        RenormalizationDenominator::ClampedBelow(1e-20)
    );
}

/// Megatron-LM's normalized sigmoid switch loss is exactly one under perfect balance.
///
/// Token 0 prefers experts {0, 1} and token 1 prefers {2, 3}, so every expert
/// receives `K T / E = 1` assignment and normalized score mass `T / E = 0.5`.
/// The former raw-sigmoid formula reported `2.0` here (scores summed to
/// `sigmoid(1) + sigmoid(-1) = 1` per expert), inflating the loss by `E/2`.
#[test]
fn load_balancing_loss_is_one_under_perfect_balance() {
    let device = Default::default();
    let logits = Tensor::<TestBackend, 3>::from_floats(
        [[[1.0, 1.0, -1.0, -1.0], [-1.0, -1.0, 1.0, 1.0]]],
        &device,
    );
    let loss = nemotron_load_balancing_loss(logits, 2, None).unwrap();
    assert_eq!(loss.dims(), [1]);
    let value = f64::from(loss.into_scalar());
    assert!((value - 1.0).abs() < 1e-6, "{value}");
}

/// A host oracle of Megatron-LM's formula pins masked, unbalanced routing.
///
/// The oracle normalizes `sigmoid` per token with `+1e-20`, recomputes top-k
/// on the normalized (uncorrected) scores, and evaluates
/// `E / (K T^2) * sum_e (sum_t s) (sum_t m)` over valid tokens only.
#[test]
fn load_balancing_loss_matches_megatron_oracle_with_padding() {
    let device = Default::default();
    let rows = [
        [2.0_f32, -0.5, 0.25, -3.0, 1.5],
        [-1.0, 0.75, 0.75, 2.5, -0.25],
        [9.0, 9.0, 9.0, 9.0, 9.0],
        [0.1, -0.2, 0.3, -0.4, 0.5],
    ];
    let valid = [true, true, false, true];
    let (experts, top_k) = (5usize, 2usize);
    let mut scores = vec![0.0_f64; experts];
    let mut assignments = vec![0.0_f64; experts];
    let mut tokens = 0.0;
    for (row, &valid) in rows.iter().zip(&valid) {
        if !valid {
            continue;
        }
        tokens += 1.0;
        let sigmoid = row
            .iter()
            .map(|&logit| 1.0 / (1.0 + (-f64::from(logit)).exp()))
            .collect::<Vec<_>>();
        let total = sigmoid.iter().sum::<f64>() + 1e-20;
        let mut order = (0..experts).collect::<Vec<_>>();
        order.sort_by(|&left, &right| sigmoid[right].total_cmp(&sigmoid[left]));
        for &expert in &order[..top_k] {
            assignments[expert] += 1.0;
        }
        for expert in 0..experts {
            scores[expert] += sigmoid[expert] / total;
        }
    }
    let expected = scores
        .iter()
        .zip(&assignments)
        .map(|(score, assignment)| score * assignment)
        .sum::<f64>()
        * experts as f64
        / (top_k as f64 * tokens * tokens);
    let logits = Tensor::<TestBackend, 3>::from_floats([rows], &device);
    let mask = TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
        TensorData::new(valid.to_vec(), [1, 4]),
        &device,
    ));
    let actual = f64::from(
        nemotron_load_balancing_loss(logits.clone(), top_k, Some(&mask))
            .unwrap()
            .into_scalar(),
    );
    assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
    // The padded token's extreme logits must not influence the result.
    let mut changed = rows;
    changed[2] = [-50.0, 50.0, -50.0, 50.0, -50.0];
    let unchanged = f64::from(
        nemotron_load_balancing_loss(
            Tensor::<TestBackend, 3>::from_floats([changed], &device),
            top_k,
            Some(&mask),
        )
        .unwrap()
        .into_scalar(),
    );
    assert!((unchanged - actual).abs() < 1e-7);
    // Mismatched masks and impossible top-k values are contextual errors.
    assert!(nemotron_load_balancing_loss(logits.clone(), 6, None).is_err());
    assert!(nemotron_load_balancing_loss(logits.clone(), 0, None).is_err());
    let short = TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
        TensorData::new(vec![true; 3], [1, 3]),
        &device,
    ));
    assert!(nemotron_load_balancing_loss(logits, top_k, Some(&short)).is_err());
}

/// A single routed expert receives every assignment without a one-hot kernel.
#[test]
fn load_balancing_loss_supports_one_expert() {
    let logits = Tensor::<TestBackend, 3>::from_floats([[[0.3], [-2.0]]], &Default::default());
    let value = f64::from(
        nemotron_load_balancing_loss(logits, 1, None)
            .unwrap()
            .into_scalar(),
    );
    assert!((value - 1.0).abs() < 1e-6, "{value}");
}

/// Gradients of the auxiliary loss reach the router projection.
#[cfg(feature = "autodiff")]
#[test]
fn load_balancing_loss_is_differentiable() {
    use burn::backend::Autodiff;

    /// Autodiff wrapper used to check gradient flow into router logits.
    type AdBackend = Autodiff<TestBackend>;
    let device = Default::default();
    let logits = Tensor::<AdBackend, 3>::from_floats(
        // Both tokens select expert 0, so assignments are unbalanced; under
        // perfectly balanced routing normalized scores make the loss constant.
        [[[1.0, 0.2, -1.0, 0.4], [0.9, 0.8, 0.1, -0.6]]],
        &device,
    )
    .require_grad();
    let loss = nemotron_load_balancing_loss(logits.clone(), 2, None).unwrap();
    let gradients = loss.backward();
    let gradient = logits.grad(&gradients).unwrap();
    assert!(
        gradient
            .into_data()
            .to_vec::<f32>()
            .unwrap()
            .iter()
            .any(|value| value.abs() > 1e-6)
    );
}
