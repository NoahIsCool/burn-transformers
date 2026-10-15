//! Feed-forward variants against the shared reference implementations and their hook points.

use burn::{
    module::Module,
    tensor::{TensorData, Tolerance, backend::BackendTypes},
};

use super::*;
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
    let branch = ModularFeedForward::<TestBackend>::init(&moe, 8, &device)
        .unwrap()
        .unwrap();
    assert!(branch.activation_points(Layer::decoder(0)).is_empty());
    let ModularFeedForward::MixtureOfExperts(experts) = &branch else {
        panic!("expected experts");
    };
    assert!(experts.router.bias.is_some());
    assert!(experts.shared_expert_gate.is_some());
    assert!(experts.routed[0].down_proj.bias.is_some());
    assert!(
        ModularFeedForward::<TestBackend>::init(&FeedForwardSpec::None, 8, &device)
            .unwrap()
            .is_none()
    );
}
