//! Feed-forward branch of a modular block: gated, ungated, clamped, or routed experts.
//!
//! Every variant maps `[batch, token, hidden]` to the same shape and is built
//! from the shared configuration its spec translates to. Dense variants emit
//! the canonical MLP points through the shared hook-aware paths:
//!
//! | variant | `mlp_pre_activation` | `mlp_activation` |
//! | --- | --- | --- |
//! | gated `down(act(gate x) * up x)` | `gate x` | `act(gate x)` |
//! | ungated `down(act(up x))` | `up x` | `act(up x)` |
//! | clamped SwiGLU | `gate x` | `g * sigmoid(alpha g)`, `g = min(gate x, limit)` |
//!
//! Sparse experts expose only the surrounding MLP input and output points, and
//! attention-only blocks have no feed-forward branch at all.
//!
//! # Clamped experts
//!
//! [`FeedForwardSpec::ClampedMixtureOfExperts`] keeps every parameter of the
//! shared [`SparseMoe`] (router, routed and shared experts, optional
//! correction bias and shared-expert gate) and its routing, weighting
//! precision, and memory-bounded dispatch; only the expert evaluation
//! changes, from `down(act(gate x) * up x)` to
//! [`GatedFeedForward::forward_clamped`]. For routed experts `S(t)` selected
//! for token `t` with weights `w_{t,e}` and shared experts `H`,
//!
//! ```text
//! y_t = Σ_{e ∈ S(t)} w_{t,e} · E_e(x_t) + γ(x_t) · Σ_{h ∈ H} E_h(x_t)
//! E(x) = down((clamp(up x, ±limit) + up_offset) · g · σ(alpha g)),  g = min(gate x, limit)
//! ```
//!
//! where `γ` is the optional sigmoid shared-expert gate (one otherwise).
//!
//! # Record compatibility
//!
//! Burn records store an enum module by variant (by index in binary
//! recorders), so variants are append-only: `Dense` and `MixtureOfExperts`
//! keep their original positions and later mechanisms follow them.

use core::marker::PhantomData;

use burn::{
    module::{Module, Param},
    tensor::{Tensor, activation::sigmoid, backend::Backend},
};

use crate::{
    Result,
    architecture::{ClampedSwiGluSpec, FeedForwardSpec, MixtureOfExpertsSpec},
    feed_forward::{
        ClampedSwiGlu, DenseFeedForward, DenseFeedForwardConfig, GatedFeedForward,
        GatedFeedForwardConfig, mlp_hook_points,
    },
    intervention::{Hooks, Layer, PointInfo},
    moe::{
        RoutedExpertPrecision, RoutedWeightingPrecision, SparseMoe, SparseMoeOutput,
        dispatch_routed_experts, router_logits,
    },
    operations,
};

/// Dense gate/up/down projections evaluated with a clamped SwiGLU unit.
#[derive(Module, Debug)]
pub(super) struct ClampedFeedForward<B: Backend> {
    /// Gate, up, and down projections in checkpoint-native layout.
    pub(super) projections: GatedFeedForward<B>,
    /// Clamp limit, sigmoid sharpness, and up offset of the gated unit.
    #[module(skip)]
    glu: ClampedSwiGlu,
}

impl<B: Backend> ClampedFeedForward<B> {
    /// Evaluates `down(glu(gate x, up x))` while emitting the canonical MLP points.
    ///
    /// The pre-activation point is the raw gate projection and the activation
    /// point is the clamped gate nonlinearity `g * sigmoid(alpha g)` before
    /// its product with `clamp(up, -limit, limit) + up_offset`, matching
    /// [`ClampedSwiGlu::combine`] exactly when no hook edits the values.
    fn forward_with_hooks(
        &self,
        input: Tensor<B, 3>,
        layer: Layer,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<Tensor<B, 3>> {
        let gate = hooks.apply(
            layer.mlp_pre_activation(),
            operations::linear(input.clone(), &self.projections.gate_proj)?,
        )?;
        let gate = gate.clamp_max(self.glu.limit);
        let gate = hooks.apply(
            layer.mlp_activation(),
            gate.clone() * sigmoid(gate * self.glu.alpha),
        )?;
        let content = operations::linear(input, &self.projections.up_proj)?
            .clamp(-self.glu.limit, self.glu.limit)
            + self.glu.up_offset;
        operations::linear(gate * content, &self.projections.down_proj)
    }
}

/// Routed and optional shared experts that all evaluate one clamped SwiGLU unit.
///
/// The parameters are exactly those of the shared [`SparseMoe`], so they are
/// addressed as `…ClampedMixtureOfExperts.experts.{router, routed.N, shared.N}`;
/// see the module documentation for the computation.
#[derive(Module, Debug)]
pub(super) struct ClampedExperts<B: Backend> {
    /// Router, expert projections, and routing policy of the shared sparse layer.
    pub(super) experts: SparseMoe<B>,
    /// Clamp limit, sigmoid sharpness, and up offset of every expert.
    #[module(skip)]
    glu: ClampedSwiGlu,
}

impl<B: Backend> ClampedExperts<B> {
    /// Routes `[batch, token, hidden]` states and mixes the selected clamped experts.
    ///
    /// Routing and weighting follow [`SparseMoe::forward`] step for step:
    /// router logits in the policy's projection precision, FP32 ranking and
    /// normalization, weights rounded per the layer's
    /// [`RoutedWeightingPrecision`], and expert-major accumulation through the
    /// shared memory-bounded dispatcher. Returns the same [`SparseMoeOutput`]
    /// diagnostics. The caller has validated the block width, so the input is
    /// a nonempty `[batch, token, hidden]` grid.
    ///
    /// # Errors
    ///
    /// Propagates routing errors (including a non-finite router score, named
    /// with its token) and expert projection errors.
    fn forward(&self, input: Tensor<B, 3>) -> Result<SparseMoeOutput<B>> {
        let moe = &self.experts;
        let [batch, sequence, hidden] = input.dims();
        // The tensor holds `batch * sequence * hidden` elements, so the product fits.
        let tokens = batch * sequence;
        let residual_dtype = input.dtype();
        let flat_input = input.clone().reshape([tokens, hidden]);
        let logits = router_logits(input, &moe.router, moe.routing.projection)?;
        let decision = moe.router_config().route(
            logits.clone().reshape([tokens, moe.routed.len()]),
            moe.correction_bias.as_ref().map(Param::val),
        )?;
        let (weights, precision) = match moe.weighting_precision {
            RoutedWeightingPrecision::Residual => (
                decision.weights.cast(residual_dtype),
                RoutedExpertPrecision::uniform(residual_dtype),
            ),
            RoutedWeightingPrecision::Fp32ThenResidual => (
                decision.weights,
                RoutedExpertPrecision::fp32_weights_round_each(residual_dtype),
            ),
        };
        let dispatch = dispatch_routed_experts(
            flat_input.clone(),
            weights.clone(),
            decision.experts.clone(),
            Some(decision.non_finite_tokens),
            moe.routed.len(),
            precision,
            |expert, rows| moe.routed[expert].forward_clamped(rows, &self.glu),
        )?;
        let hidden_state = match self.shared_output(flat_input)? {
            Some(shared) => dispatch.hidden_state + shared,
            None => dispatch.hidden_state,
        };
        let slots = moe.experts_per_token;
        Ok(SparseMoeOutput {
            hidden_state: hidden_state.reshape([batch, sequence, hidden]),
            router_logits: logits,
            selected_experts: decision.experts.reshape([batch, sequence, slots]),
            selected_weights: weights.reshape([batch, sequence, slots]),
            expert_token_counts: dispatch.expert_token_counts,
            evaluated_experts: dispatch.evaluated_experts,
        })
    }

    /// Sums the clamped shared experts over `[tokens, hidden]` rows and applies the optional gate.
    ///
    /// Returns `None` without computing anything when there are no shared experts.
    fn shared_output(&self, flat_input: Tensor<B, 2>) -> Result<Option<Tensor<B, 2>>> {
        let moe = &self.experts;
        let mut sum: Option<Tensor<B, 2>> = None;
        for expert in &moe.shared {
            let output = expert.forward_clamped(flat_input.clone(), &self.glu)?;
            sum = Some(match sum {
                Some(total) => total + output,
                None => output,
            });
        }
        let Some(sum) = sum else {
            return Ok(None);
        };
        Ok(Some(match &moe.shared_expert_gate {
            Some(gate) => sum * sigmoid(operations::linear(flat_input, gate)?),
            None => sum,
        }))
    }
}

/// One executable feed-forward implementation, or its explicit absence.
#[derive(Module, Debug)]
// See `ModularMixer`: indirection would make the Burn parameter record non-visitable.
#[allow(clippy::large_enum_variant)]
pub(super) enum ModularFeedForward<B: Backend> {
    /// Gated MLP `down(act(gate x) * up x)`.
    Dense(
        /// Dense gate, content, and output projection parameters.
        GatedFeedForward<B>,
    ),
    /// Top-k routed and optional shared experts.
    MixtureOfExperts(
        /// Router and routed/shared expert parameters.
        SparseMoe<B>,
    ),
    /// Classic MLP `down(act(up x))`.
    Ungated(
        /// Up and down projection parameters.
        DenseFeedForward<B>,
    ),
    /// Gated MLP with the clamped SwiGLU unit of gpt-oss.
    ClampedSwiGlu(
        /// Projections and clamp parameters.
        ClampedFeedForward<B>,
    ),
    /// No feed-forward branch (attention-only blocks); it owns no parameters.
    Absent(
        /// Parameter-free marker of the backend.
        PhantomData<B>,
    ),
    /// Routed and shared experts evaluated with the clamped SwiGLU unit of gpt-oss.
    ClampedMixtureOfExperts(
        /// Shared sparse-expert parameters and the experts' clamp parameters.
        ClampedExperts<B>,
    ),
}

impl<B: Backend> ModularFeedForward<B> {
    /// Builds the branch of a validated spec, [`Self::Absent`] for [`FeedForwardSpec::None`].
    pub(super) fn init(
        spec: &FeedForwardSpec,
        hidden_size: usize,
        device: &B::Device,
    ) -> Result<Self> {
        Ok(match spec {
            FeedForwardSpec::Dense(dense) => Self::Dense(
                GatedFeedForwardConfig::new(hidden_size, dense.intermediate_size)
                    .with_activation(dense.activation)
                    .with_bias(dense.bias)
                    .init(device)?,
            ),
            FeedForwardSpec::Ungated(dense) => Self::Ungated(
                DenseFeedForwardConfig::new(hidden_size, dense.intermediate_size)
                    .with_activation(dense.activation)
                    .with_bias(dense.bias)
                    .init(device)?,
            ),
            FeedForwardSpec::ClampedSwiGlu(clamped) => {
                Self::ClampedSwiGlu(clamped_feed_forward(clamped, hidden_size, device)?)
            }
            FeedForwardSpec::MixtureOfExperts(moe) => {
                Self::MixtureOfExperts(moe.sparse_config(hidden_size).init(device)?)
            }
            FeedForwardSpec::ClampedMixtureOfExperts {
                experts,
                limit,
                alpha,
                up_offset,
            } => {
                // The same translation as `FeedForwardSpec::expert_glu`, which validation uses.
                let glu = ClampedSwiGlu::new(*limit)
                    .with_alpha(*alpha)
                    .with_up_offset(*up_offset);
                Self::ClampedMixtureOfExperts(clamped_experts(experts, glu, hidden_size, device)?)
            }
            FeedForwardSpec::None => Self::Absent(PhantomData),
        })
    }

    /// Whether the block has no feed-forward branch.
    pub(super) const fn is_absent(&self) -> bool {
        matches!(self, Self::Absent(_))
    }

    /// Lists the dense pre-activation and activation points; experts and absent branches expose none.
    pub(super) fn activation_points(&self, layer: Layer) -> Vec<PointInfo> {
        match self {
            Self::Dense(_) | Self::Ungated(_) | Self::ClampedSwiGlu(_) => mlp_hook_points(layer),
            Self::MixtureOfExperts(_) | Self::ClampedMixtureOfExperts(_) | Self::Absent(_) => {
                Vec::new()
            }
        }
    }

    /// Evaluates the branch and returns sparse-routing state only for an expert layer.
    ///
    /// An absent branch contributes zeros; blocks skip it (and its norms) entirely.
    pub(super) fn forward(
        &self,
        hidden: Tensor<B, 3>,
        layer: Layer,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<(Tensor<B, 3>, Option<SparseMoeOutput<B>>)> {
        match self {
            Self::Dense(dense) => Ok((dense.forward_with_hooks(hidden, layer, hooks)?, None)),
            Self::Ungated(dense) => Ok((dense.forward_with_hooks(hidden, layer, hooks)?, None)),
            Self::ClampedSwiGlu(clamped) => {
                Ok((clamped.forward_with_hooks(hidden, layer, hooks)?, None))
            }
            Self::MixtureOfExperts(moe) => {
                let output = moe.forward(hidden)?;
                Ok((output.hidden_state.clone(), Some(output)))
            }
            Self::ClampedMixtureOfExperts(experts) => {
                let output = experts.forward(hidden)?;
                Ok((output.hidden_state.clone(), Some(output)))
            }
            Self::Absent(_) => Ok((hidden.zeros_like(), None)),
        }
    }
}

/// Builds gate/up/down projections and the clamped unit of a validated spec.
fn clamped_feed_forward<B: Backend>(
    spec: &ClampedSwiGluSpec,
    hidden_size: usize,
    device: &B::Device,
) -> Result<ClampedFeedForward<B>> {
    let glu = spec.glu();
    glu.validate()?;
    Ok(ClampedFeedForward {
        projections: GatedFeedForwardConfig::new(hidden_size, spec.intermediate_size)
            .with_bias(spec.bias)
            .init(device)?,
        glu,
    })
}

/// Builds the shared sparse layer of a validated spec with every expert clamped by `glu`.
fn clamped_experts<B: Backend>(
    experts: &MixtureOfExpertsSpec,
    glu: ClampedSwiGlu,
    hidden_size: usize,
    device: &B::Device,
) -> Result<ClampedExperts<B>> {
    glu.validate()?;
    Ok(ClampedExperts {
        experts: experts.sparse_config(hidden_size).init(device)?,
        glu,
    })
}

#[cfg(test)]
#[path = "feed_forward_tests.rs"]
/// Feed-forward variants against the shared reference blocks and their hook points.
mod tests;
