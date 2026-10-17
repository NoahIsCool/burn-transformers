//! Feed-forward variants against the shared reference implementations and their hook points.

use burn::{
    module::{Module, ModuleMapper, Param},
    tensor::{DType, TensorData, Tolerance, backend::BackendTypes},
};

use super::*;
use crate::moe::RoutedWeightingPrecision;
use crate::{
    architecture::{Activation, DenseFeedForwardSpec, MixtureOfExpertsSpec},
    models::custom::test_support::{DeterministicParameters, TestBackend},
    moe::RoutingPolicy,
};

/// Device used by every feed-forward fixture.
type Device = <TestBackend as BackendTypes>::Device;

/// Deterministic `[2, 3, 8]` inputs with values on both sides of the clamp limit.
fn input(device: &Device) -> Tensor<TestBackend, 3> {
    let values = (0..48)
        .map(|index| ((index * 5 + 1) as f32 * 0.37).sin() * 6.0)
        .collect::<Vec<_>>();
    Tensor::from_data(TensorData::new(values, [2, 3, 8]), device)
}

/// Builds a deterministic branch of width eight.
fn branch(spec: &FeedForwardSpec, device: &Device) -> ModularFeedForward<TestBackend> {
    ModularFeedForward::<TestBackend>::init(spec, 8, device)
        .unwrap()
        .map(&mut DeterministicParameters::default())
}

/// Runs a branch inside a scope that captures both dense MLP points.
fn run_captured(
    branch: &ModularFeedForward<TestBackend>,
    input: Tensor<TestBackend, 3>,
) -> (
    Tensor<TestBackend, 3>,
    Tensor<TestBackend, 3>,
    Tensor<TestBackend, 3>,
) {
    let layer = Layer::decoder(0);
    let mut hooks = Hooks::new();
    let pre = hooks.capture(layer.mlp_pre_activation());
    let activation = hooks.capture(layer.mlp_activation());
    let output = hooks
        .run(branch.activation_points(layer), |hooks| {
            branch.forward(input, layer, hooks)
        })
        .unwrap()
        .0;
    (
        output,
        hooks.get(&pre).unwrap(),
        hooks.get(&activation).unwrap(),
    )
}

/// The clamped branch equals `forward_clamped` and exposes the gate before and after its unit.
#[test]
fn clamped_swiglu_matches_the_shared_unit_and_emits_gate_points() {
    let device = Default::default();
    let spec = FeedForwardSpec::ClampedSwiGlu(ClampedSwiGluSpec {
        intermediate_size: 6,
        bias: true,
        limit: 0.5,
        alpha: 1.702,
        up_offset: 1.0,
    });
    let branch = branch(&spec, &device);
    let ModularFeedForward::ClampedSwiGlu(clamped) = &branch else {
        panic!("expected a clamped branch");
    };
    let expected = clamped
        .projections
        .forward_clamped(input(&device), &clamped.glu)
        .unwrap();
    let (output, pre, activation) = run_captured(&branch, input(&device));
    output
        .into_data()
        .assert_approx_eq::<f32>(&expected.into_data(), Tolerance::absolute(1e-6));
    let gate = operations::linear(input(&device), &clamped.projections.gate_proj).unwrap();
    pre.into_data()
        .assert_approx_eq::<f32>(&gate.clone().into_data(), Tolerance::absolute(0.0));
    let clamped_gate = gate.clamp_max(0.5);
    activation.into_data().assert_approx_eq::<f32>(
        &(clamped_gate.clone() * sigmoid(clamped_gate * 1.702)).into_data(),
        Tolerance::absolute(1e-6),
    );
}

/// The ungated branch equals the shared dense MLP and exposes `up x` and its activation.
#[test]
fn ungated_mlp_matches_the_shared_dense_block() {
    let device = Default::default();
    let spec = FeedForwardSpec::Ungated(DenseFeedForwardSpec {
        intermediate_size: 5,
        activation: Activation::GeluTanh,
        bias: true,
    });
    let branch = branch(&spec, &device);
    let ModularFeedForward::Ungated(dense) = &branch else {
        panic!("expected an ungated branch");
    };
    let expected = dense.forward(input(&device)).unwrap();
    let (output, pre, _) = run_captured(&branch, input(&device));
    output
        .into_data()
        .assert_approx_eq::<f32>(&expected.into_data(), Tolerance::absolute(0.0));
    assert_eq!(pre.dims(), [2, 3, 5]);
}

/// Expert banks honor the routing policy and expose no dense MLP point; `None` builds nothing.
#[test]
fn expert_and_absent_branches_have_the_expected_shape() {
    let device: Device = Default::default();
    let moe = FeedForwardSpec::MixtureOfExperts(MixtureOfExpertsSpec {
        routed_experts: 4,
        experts_per_token: 2,
        shared_experts: 1,
        expert_intermediate_size: 4,
        activation: Activation::Silu,
        renormalize_selected: true,
        routed_scale: 1.0,
        routing: RoutingPolicy {
            router_bias: true,
            ..RoutingPolicy::default()
        },
        shared_expert_gate: true,
        expert_bias: true,
    });
    let branch = ModularFeedForward::<TestBackend>::init(&moe, 8, &device).unwrap();
    assert!(branch.activation_points(Layer::decoder(0)).is_empty());
    let ModularFeedForward::MixtureOfExperts(experts) = &branch else {
        panic!("expected experts");
    };
    assert!(experts.router.bias.is_some());
    assert!(experts.shared_expert_gate.is_some());
    assert!(experts.routed[0].down_proj.bias.is_some());
    let absent =
        ModularFeedForward::<TestBackend>::init(&FeedForwardSpec::None, 8, &device).unwrap();
    assert!(absent.is_absent());
    assert_eq!(absent.num_params(), 0);
    assert!(absent.activation_points(Layer::decoder(0)).is_empty());
}

/// An absent branch contributes exact zeros, has no points, and routes nothing.
#[test]
fn absent_branch_contributes_exact_zeros() {
    let device = Default::default();
    let absent = branch(&FeedForwardSpec::None, &device);
    assert!(absent.is_absent());
    assert!(absent.activation_points(Layer::decoder(0)).is_empty());
    let mut hooks = Hooks::new();
    let (output, routing) = absent
        .forward(input(&device), Layer::decoder(0), &mut hooks)
        .unwrap();
    assert!(routing.is_none());
    output
        .into_data()
        .assert_eq(&TensorData::zeros::<f32, _>([2, 3, 8]), true);
}

/// Four gpt-oss-style experts with a biased router, one gated shared expert, and expert biases.
fn clamped_experts_spec() -> FeedForwardSpec {
    FeedForwardSpec::ClampedMixtureOfExperts {
        experts: MixtureOfExpertsSpec {
            routed_experts: 4,
            experts_per_token: 2,
            shared_experts: 1,
            expert_intermediate_size: 5,
            activation: Activation::Silu,
            renormalize_selected: false,
            routed_scale: 1.0,
            routing: RoutingPolicy {
                router_bias: true,
                ..RoutingPolicy::default()
            },
            shared_expert_gate: true,
            expert_bias: true,
        },
        limit: 0.5,
        alpha: 1.702,
        up_offset: 1.0,
    }
}

/// Clamped experts mix `forward_clamped` outputs with the routed weights and gated shared sum.
///
/// The oracle evaluates every expert densely on every token and weights the
/// selected ones with the reported routes, so it checks the sparse dispatch
/// (gathers, scatter-adds, and weighting) and the clamped unit independently
/// of the dispatcher. The fixture's inputs reach the clamp limit, so a
/// branch that ignored the unit would differ.
#[test]
fn clamped_experts_mix_the_clamped_unit_with_routed_and_shared_weights() {
    let device = Default::default();
    let spec = clamped_experts_spec();
    let branch = branch(&spec, &device);
    assert!(branch.activation_points(Layer::decoder(0)).is_empty());
    let ModularFeedForward::ClampedMixtureOfExperts(clamped) = &branch else {
        panic!("expected clamped experts");
    };
    assert_eq!(Some(clamped.glu), spec.expert_glu());
    let (output, routing) = branch
        .forward(input(&device), Layer::decoder(0), &mut Hooks::new())
        .unwrap();
    let routing = routing.expect("experts report their routing");
    assert_eq!(routing.selected_experts.dims(), [2, 3, 2]);
    assert_eq!(routing.expert_token_counts.iter().sum::<usize>(), 12);

    let moe = &clamped.experts;
    let flat = input(&device).reshape([6, 8]);
    let selected = routing
        .selected_experts
        .reshape([6, 2])
        .into_data()
        .to_vec::<i32>()
        .unwrap();
    let weights = routing
        .selected_weights
        .reshape([6, 2])
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    let experts = moe
        .routed
        .iter()
        .map(|expert| {
            expert
                .forward_clamped(flat.clone(), &clamped.glu)
                .unwrap()
                .into_data()
                .to_vec::<f32>()
                .unwrap()
        })
        .collect::<Vec<_>>();
    let shared = moe.shared[0]
        .forward_clamped(flat.clone(), &clamped.glu)
        .unwrap()
        * sigmoid(operations::linear(flat, moe.shared_expert_gate.as_ref().unwrap()).unwrap());
    let shared = shared.into_data().to_vec::<f32>().unwrap();
    let expected = (0..6 * 8)
        .map(|index| {
            let (token, channel) = (index / 8, index % 8);
            shared[index]
                + (0..2)
                    .map(|slot| {
                        let expert = selected[token * 2 + slot] as usize;
                        weights[token * 2 + slot] * experts[expert][token * 8 + channel]
                    })
                    .sum::<f32>()
        })
        .collect::<Vec<_>>();
    output.into_data().assert_approx_eq::<f32>(
        &TensorData::new(expected, [2, 3, 8]),
        Tolerance::absolute(1e-5),
    );

    // The same parameters evaluated as plain SiLU experts give a different branch.
    let plain = moe
        .forward(input(&device))
        .unwrap()
        .hidden_state
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    let clamped_output = branch
        .forward(input(&device), Layer::decoder(0), &mut Hooks::new())
        .unwrap()
        .0
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    let gap = plain
        .iter()
        .zip(&clamped_output)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0, f32::max);
    assert!(
        gap > 1e-2,
        "the clamped unit must change the branch: {gap:e}"
    );
}

/// Clamped experts without shared experts add nothing beyond the routed mixture.
#[test]
fn clamped_experts_without_shared_experts_return_the_routed_mixture() {
    let device = Default::default();
    let FeedForwardSpec::ClampedMixtureOfExperts {
        mut experts,
        limit,
        alpha,
        up_offset,
    } = clamped_experts_spec()
    else {
        unreachable!("the fixture is a clamped expert branch");
    };
    experts.shared_experts = 0;
    experts.shared_expert_gate = false;
    experts.routing = RoutingPolicy::default();
    let spec = FeedForwardSpec::ClampedMixtureOfExperts {
        experts,
        limit,
        alpha,
        up_offset,
    };
    let branch = branch(&spec, &device);
    let ModularFeedForward::ClampedMixtureOfExperts(clamped) = &branch else {
        panic!("expected clamped experts");
    };
    assert!(clamped.experts.shared.is_empty());
    let (output, routing) = branch
        .forward(input(&device), Layer::decoder(0), &mut Hooks::new())
        .unwrap();
    assert_eq!(output.dims(), [2, 3, 8]);
    assert!(routing.is_some());
}

/// Rounds every floating parameter to F16, emulating a half-precision checkpoint.
struct HalfPrecision;

impl ModuleMapper<TestBackend> for HalfPrecision {
    /// Casts one floating parameter to F16 while keeping its shape and device.
    fn map_float<const D: usize>(
        &mut self,
        param: Param<Tensor<TestBackend, D>>,
    ) -> Param<Tensor<TestBackend, D>> {
        param.map(|tensor| tensor.cast(DType::F16))
    }
}

/// Half-precision clamped experts weight each routed contribution at the layer's precision.
///
/// With F16 activations the two [`RoutedWeightingPrecision`] policies are
/// observable: `Fp32ThenResidual` multiplies each F16 expert output by its
/// F32 routing weight and rounds the product, `(E(x) · w)_{f16}`, while
/// `Residual` rounds the weight first and multiplies in F16,
/// `E(x) ·_{f16} (w)_{f16}`. Evaluating one token per call keeps every
/// expert on a single row, so the oracle reproduces the dispatcher's
/// arithmetic bit for bit: the two routed contributions summed in F16 (two
/// terms, so the scatter order cannot matter), then the gated shared expert
/// added. Swapping the policy arms, or always multiplying in the residual
/// dtype, makes one of the exact comparisons fail; the final assertion proves
/// the fixture separates the two policies.
#[test]
fn clamped_experts_honor_the_layer_weighting_precision() {
    let device = Default::default();
    let mut branch = branch(&clamped_experts_spec(), &device).map(&mut HalfPrecision);
    let rows = input(&device).cast(DType::F16).reshape([6, 8]);
    let mut policies_differ = false;
    for token in 0..6 {
        let row = rows.clone().narrow(0, token, 1);
        let mut run = |precision| {
            let ModularFeedForward::ClampedMixtureOfExperts(clamped) = &mut branch else {
                panic!("expected clamped experts");
            };
            clamped.experts.weighting_precision = precision;
            let (output, routing) = branch
                .forward(
                    row.clone().reshape([1, 1, 8]),
                    Layer::decoder(0),
                    &mut Hooks::new(),
                )
                .unwrap();
            assert_eq!(output.dtype(), DType::F16);
            (
                output.reshape([1, 8]),
                routing.expect("experts report their routing"),
            )
        };
        let (residual, residual_routing) = run(RoutedWeightingPrecision::Residual);
        let (fp32, fp32_routing) = run(RoutedWeightingPrecision::Fp32ThenResidual);
        let ModularFeedForward::ClampedMixtureOfExperts(clamped) = &branch else {
            panic!("expected clamped experts");
        };

        // Both runs share one F32 routing decision; `Residual` reports its F16 rounding.
        let weights = fp32_routing.selected_weights.reshape([1, 2]);
        assert_eq!(weights.dtype(), DType::F32);
        assert_eq!(residual_routing.selected_weights.dtype(), DType::F16);
        residual_routing
            .selected_weights
            .reshape([1, 2])
            .into_data()
            .assert_eq(&weights.clone().cast(DType::F16).into_data(), true);
        let selected = fp32_routing.selected_experts.into_data();
        residual_routing
            .selected_experts
            .into_data()
            .assert_eq(&selected, true);
        let selected = selected.to_vec::<i32>().unwrap();

        let shared = clamped
            .experts
            .shared_output(row.clone(), Some(&clamped.glu))
            .unwrap()
            .unwrap();
        let mut expected_fp32 = Tensor::<TestBackend, 2>::zeros([1, 8], (&device, DType::F16));
        let mut expected_residual = expected_fp32.clone();
        for (slot, &expert) in selected.iter().enumerate() {
            let unit = clamped.experts.routed[expert as usize]
                .forward_clamped(row.clone(), &clamped.glu)
                .unwrap();
            let weight = weights.clone().narrow(1, slot, 1);
            expected_fp32 =
                expected_fp32 + (unit.clone().cast(DType::F32) * weight.clone()).cast(DType::F16);
            expected_residual = expected_residual + unit * weight.cast(DType::F16);
        }
        let expected_fp32 = expected_fp32 + shared.clone();
        let expected_residual = expected_residual + shared;
        fp32.into_data()
            .assert_eq(&expected_fp32.clone().into_data(), true);
        residual
            .into_data()
            .assert_eq(&expected_residual.clone().into_data(), true);
        policies_differ |= expected_fp32.into_data() != expected_residual.into_data();
    }
    assert!(
        policies_differ,
        "the fixture must separate the two weighting precisions"
    );
}

/// Gradients reach the router, a routed expert's projections, and the shared expert gate.
#[cfg(feature = "autodiff")]
#[test]
fn clamped_experts_are_differentiable() {
    use burn::backend::Autodiff;

    /// Reverse-mode CPU backend.
    type AdBackend = Autodiff<TestBackend>;
    let device = Default::default();
    let branch =
        ModularFeedForward::<AdBackend>::init(&clamped_experts_spec(), 8, &device).unwrap();
    let input = Tensor::<AdBackend, 3>::from_data(input(&device).into_data(), &device);
    let (output, _) = branch
        .forward(input, Layer::decoder(0), &mut Hooks::new())
        .unwrap();
    let gradients = output.sum().backward();
    let ModularFeedForward::ClampedMixtureOfExperts(clamped) = &branch else {
        panic!("expected clamped experts");
    };
    let moe = &clamped.experts;
    assert!(moe.router.weight.grad(&gradients).is_some());
    assert!(moe.shared[0].down_proj.weight.grad(&gradients).is_some());
    assert!(
        moe.shared_expert_gate
            .as_ref()
            .unwrap()
            .weight
            .grad(&gradients)
            .is_some()
    );
    assert!(
        moe.routed
            .iter()
            .any(|expert| expert.gate_proj.weight.grad(&gradients).is_some())
    );
}

/// Several ungated shared experts are summed after their clamped units.
///
/// The routed mixture is recovered from the same branch with its shared
/// experts removed, so the difference must be exactly the plain sum of the
/// two clamped shared experts (no sigmoid gate).
#[test]
fn clamped_experts_sum_several_ungated_shared_experts() {
    let device = Default::default();
    let FeedForwardSpec::ClampedMixtureOfExperts {
        mut experts,
        limit,
        alpha,
        up_offset,
    } = clamped_experts_spec()
    else {
        unreachable!("the fixture is a clamped expert branch");
    };
    experts.shared_experts = 2;
    experts.shared_expert_gate = false;
    let spec = FeedForwardSpec::ClampedMixtureOfExperts {
        experts,
        limit,
        alpha,
        up_offset,
    };
    let branch = branch(&spec, &device);
    let ModularFeedForward::ClampedMixtureOfExperts(clamped) = &branch else {
        panic!("expected clamped experts");
    };
    let flat = input(&device).reshape([6, 8]);
    let expected = clamped
        .experts
        .shared
        .iter()
        .map(|expert| expert.forward_clamped(flat.clone(), &clamped.glu).unwrap())
        .reduce(|left, right| left + right)
        .unwrap();
    clamped
        .experts
        .shared_output(flat, Some(&clamped.glu))
        .unwrap()
        .expect("two shared experts")
        .into_data()
        .assert_approx_eq::<f32>(&expected.into_data(), Tolerance::absolute(0.0));
}

/// A non-finite router score is reported as an error naming the token, not silently routed.
#[test]
fn clamped_experts_reject_non_finite_router_scores() {
    let device = Default::default();
    let branch = branch(&clamped_experts_spec(), &device);
    let mut values = input(&device).into_data().to_vec::<f32>().unwrap();
    values[8] = f32::NAN;
    let poisoned = Tensor::<TestBackend, 3>::from_data(TensorData::new(values, [2, 3, 8]), &device);
    let error = branch
        .forward(poisoned, Layer::decoder(0), &mut Hooks::new())
        .unwrap_err();
    assert!(error.to_string().contains("token"), "{error}");
}
