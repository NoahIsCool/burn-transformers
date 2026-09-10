use burn::{
    backend::Flex,
    tensor::{Tolerance, f16},
};

use super::*;

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
