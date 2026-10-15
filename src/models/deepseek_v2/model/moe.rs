//! DeepSeek's routed-plus-shared expert block, router policy, and balance losses.
//!
//! # Block
//!
//! A sparse DeepSeek block evaluates `y = sum_{j in S(x)} w_j E_j(x) + Shared(x)`:
//! `K` of `E` routed SwiGLU experts selected per token plus one always-active
//! SwiGLU whose intermediate width is `n_shared_experts` expert widths (the
//! checkpoint stores the shared experts as one widened network, so it is not
//! split into independent experts here).
//!
//! # Routing
//!
//! The router projects `z = x W` in F32 (activations and weight are cast
//! first, so an F16 rounding event cannot flip a discrete choice) and then
//! follows the shared [`RouterConfig`](crate::moe::RouterConfig). The configuration fields map onto it
//! exactly as the pinned references compute them:
//!
//! | `scoring_func` | `topk_method` | scores `s` | eligible experts | weights | reference |
//! | --- | --- | --- | --- | --- | --- |
//! | `softmax` | `greedy` | `softmax(z)` | all | `s_S`, renormalized iff `K > 1` and `norm_topk_prob`, else `* routed_scaling_factor` | DeepSeek-V2-Lite `MoEGate` |
//! | `softmax` | `group_limited_greedy` | `softmax(z)` | best `topk_group` of `n_group` groups by group max; others scored `0` | as above | DeepSeek-V2 `MoEGate` |
//! | `sigmoid` | `noaux_tc` | `sigmoid(z)` | best groups by the sum of their two largest `s + b`; others scored `0`; top-`K` of `s + b` | `s_S`, renormalized iff `norm_topk_prob`, then `* routed_scaling_factor` | Transformers 5.2 `DeepseekV3MoE` |
//!
//! `b` is the frozen F32 `e_score_correction_bias`, used for selection only.
//! Renormalization divides by `sum + 1e-20`, as both references do. Ties
//! resolve to the smallest expert ID on every backend, and a token whose
//! scores contain NaN (for example after an F16 overflow upstream) is
//! reported as a contextual error instead of indexing out of range. Other
//! combinations have no reference implementation and are rejected at
//! initialization.
//!
//! # Auxiliary balance loss
//!
//! With per-token expert probabilities `p[t,e]` (softmax, or for sigmoid
//! scoring the per-token-normalized `s[t,e] / sum_e s[t,e]` of the DeepSeek-V3
//! report, eqs. 17-20), selection indicators `1[e in S_t]`, `E` experts and
//! `K` selections per token:
//!
//! * `seq_aux = true`: per row `b` with `T_b` valid tokens,
//!   `L_b = sum_e (E / (K T_b) * count_b[e]) * (1 / T_b) sum_t p[b,t,e]`, and
//!   the loss is the mean of `L_b` over rows with `T_b > 0`;
//! * `seq_aux = false`: over all `T` valid tokens,
//!   `L = sum_e (E / (K T) * count[e]) * (1 / T) sum_t p[t,e]`, where
//!   `count[e] / (K T)` is the reference's `one_hot(topk_idx).mean(0)`.
//!
//! The global form differs from the shared Qwen/Mixtral
//! [`crate::moe::load_balancing_loss`] by the factor `1 / K`; reusing the
//! shared loss here made the DeepSeek value `K` times too large. Padding
//! tokens (`token_mask == false`) are excluded from every count and mean, and
//! rows without a valid token (finished rows in a decode step) are excluded
//! from the per-row mean; both are extensions of the references, which have no
//! padding. Both forms are multiplied by `aux_loss_alpha` by the caller.

use burn::{
    module::{Module, Param},
    tensor::{DType, Int, Tensor, activation, backend::Backend},
};

use crate::{
    Error, ForwardOptions, Result,
    architecture::{RouterDiagnostics, TokenMask},
    feed_forward::{GatedFeedForward, GatedFeedForwardConfig},
    intervention::{Axis, Hooks, Layer, Point, PointInfo},
    moe::{
        ExpertSelection, GroupLimitedSelection, GroupScore, PrunedExpertScore,
        RenormalizationDenominator, RoutedExpertPrecision, RouterConfig, RouterProjectionPrecision,
        RouterScoring, RoutingPolicy, ScoringOrder, SparseMoe, SparseMoeConfig, SparseMoeOutput,
        dispatch_routed_experts, router_logits,
    },
};

use super::super::{DeepseekScoringFunction, DeepseekTopKMethod, DeepseekV2Config};

/// Dtype of every router projection, probability, and diagnostic logit.
///
/// The reference router casts both operands to FP32 before the projection, so
/// router logits are FP32 even in F16/BF16 checkpoints. Placeholders for
/// inactive decode rows must use the same dtype to be concatenated with them.
pub(super) const ROUTER_DTYPE: DType = DType::F32;

/// Additive renormalization stabilizer shared by both DeepSeek references.
const RENORMALIZATION_EPSILON: f32 = 1e-20;

/// Routed expert output together with the always-active shared contribution.
#[derive(Clone, Debug)]
pub(super) struct DeepseekMoeOutput<B: Backend> {
    /// Sum of selected routed contributions and the widened shared network.
    pub(super) hidden_state: Tensor<B, 3>,
    /// Routing choices and accounting retained for optional diagnostics.
    pub(super) routed: SparseMoeOutput<B>,
}

/// DeepSeek's top-k routed bank plus one widened shared SwiGLU network.
#[derive(Module, Debug)]
pub struct DeepseekV2Moe<B: Backend> {
    /// Selected-only routed expert bank, router projection, and routing policy.
    pub routed: SparseMoe<B>,
    /// One always-active checkpoint module of width `n_shared * expert_width`.
    pub shared_experts: GatedFeedForward<B>,
}

/// Resolves the routing semantics of a configuration (see the module table).
///
/// # Errors
///
/// Returns [`Error::BackendCapability`] for scoring/selection combinations
/// without a reference implementation and the validation errors of
/// [`RouterConfig::validate`].
pub(super) fn router_config(config: &DeepseekV2Config) -> Result<RouterConfig> {
    let experts_per_token = config.num_experts_per_tok;
    let grouped = |group_score| {
        ExpertSelection::GroupLimited(GroupLimitedSelection {
            groups: config.n_group,
            top_groups: config.topk_group,
            group_score,
            pruned_score: PrunedExpertScore::Zero,
        })
    };
    let (scoring, selection, correction_bias, renormalize_selected, routed_scale) =
        match (config.scoring_func, config.topk_method) {
            (
                DeepseekScoringFunction::Softmax,
                method @ (DeepseekTopKMethod::Greedy | DeepseekTopKMethod::GroupLimitedGreedy),
            ) => {
                // The V2 reference renormalizes only a multi-expert mixture and
                // applies routed_scaling_factor only on its unnormalized branch;
                // normalizing a top-1 route would erase the router's confidence.
                let renormalize = experts_per_token > 1 && config.norm_topk_prob;
                let selection = if method == DeepseekTopKMethod::Greedy {
                    ExpertSelection::Greedy
                } else {
                    grouped(GroupScore::Max)
                };
                (
                    RouterScoring::Softmax,
                    selection,
                    false,
                    renormalize,
                    if renormalize {
                        1.0
                    } else {
                        config.routed_scaling_factor
                    },
                )
            }
            (DeepseekScoringFunction::Sigmoid, DeepseekTopKMethod::NoAuxTc) => (
                RouterScoring::Sigmoid,
                grouped(GroupScore::TopTwoSum),
                true,
                config.norm_topk_prob,
                config.routed_scaling_factor,
            ),
            (scoring, method) => {
                return Err(Error::BackendCapability(format!(
                    "DeepSeek routing {scoring:?}/{method:?} has no reference implementation: \
                     softmax scoring supports greedy and group_limited_greedy (DeepSeek-V2), \
                     sigmoid scoring supports noaux_tc (DeepSeek-V3)"
                )));
            }
        };
    let router = RouterConfig {
        experts: config.n_routed_experts,
        experts_per_token,
        policy: RoutingPolicy {
            scoring,
            order: ScoringOrder::ScoreThenTopK,
            selection,
            correction_bias,
            router_bias: false,
            projection: RouterProjectionPrecision::Fp32,
            denominator: RenormalizationDenominator::PlusEpsilon(RENORMALIZATION_EPSILON),
        },
        renormalize_selected,
        routed_scale,
    };
    router.validate()?;
    Ok(router)
}

impl<B: Backend> DeepseekV2Moe<B> {
    /// Allocates the single-process expert topology and its routing policy.
    pub(super) fn init(config: &DeepseekV2Config, device: &B::Device) -> Result<Self> {
        if config.ep_size != 1 {
            return Err(Error::BackendCapability(
                "DeepSeek expert-parallel checkpoints require distributed execution, which is deferred from v1"
                    .into(),
            ));
        }
        let router = router_config(config)?;
        let mut routed = SparseMoeConfig::new(
            config.hidden_size,
            config.moe_intermediate_size,
            config.n_routed_experts,
            config.num_experts_per_tok,
        );
        routed.routing = router.policy;
        routed.renormalize_selected = router.renormalize_selected;
        routed.routed_scale = router.routed_scale;
        let shared_width = config
            .moe_intermediate_size
            .checked_mul(config.n_shared_experts)
            .ok_or_else(|| {
                Error::configuration(
                    "deepseek_v2.shared_experts",
                    "intermediate width product overflows usize",
                )
            })?;
        Ok(Self {
            routed: routed.init(device)?,
            shared_experts: GatedFeedForwardConfig::new(config.hidden_size, shared_width)
                .init(device)?,
        })
    }

    /// Evaluates only selected routed experts and always evaluates the shared MLP.
    ///
    /// `input` is `[batch, token, hidden]` in the activation dtype. Routing
    /// runs in F32 (module documentation); the routed mixture is accumulated
    /// in F32 and rounded to the activation dtype once, then added to the
    /// shared branch.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Shape`] for empty or mis-sized input and for tokens
    /// whose router scores are NaN, plus the router's configuration errors.
    pub(super) fn forward(&self, input: Tensor<B, 3>) -> Result<DeepseekMoeOutput<B>> {
        self.forward_inner(input, None)
    }

    /// Evaluates the block while emitting its router-logit and branch-output points.
    ///
    /// Editing [`router_logits_point`] changes which experts run and their
    /// weights; [`routed_output_point`] and [`shared_output_point`] are the two
    /// branch outputs before they are added. Call inside the caller's
    /// [`Hooks::run`] scope.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::forward`] and of the registered actions.
    pub(super) fn forward_with_hooks(
        &self,
        input: Tensor<B, 3>,
        layer: Layer,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<DeepseekMoeOutput<B>> {
        self.forward_inner(input, Some((layer, hooks)))
    }

    /// Shared routed/shared evaluation, emitting hook points when requested.
    fn forward_inner(
        &self,
        input: Tensor<B, 3>,
        mut hooks: Option<(Layer, &mut Hooks<'_, B>)>,
    ) -> Result<DeepseekMoeOutput<B>> {
        let [batch, sequence, hidden] = input.dims();
        let experts = self.routed.routed.len();
        let top_k = self.routed.experts_per_token;
        if batch == 0 || sequence == 0 || hidden != self.routed.hidden_size {
            return Err(Error::shape(
                "deepseek_v2.moe.input",
                format!("[batch>0, sequence>0, {}]", self.routed.hidden_size),
                format!("[{batch}, {sequence}, {hidden}]"),
            ));
        }
        let tokens = batch.checked_mul(sequence).ok_or_else(|| {
            Error::shape(
                "deepseek_v2.moe.input",
                "a representable batch * sequence product",
                format!("{batch} * {sequence}"),
            )
        })?;
        let input_dtype = input.dtype();
        let flat_input = input.clone().reshape([tokens, hidden]);
        let mut logits = router_logits(
            input.clone(),
            &self.routed.router,
            self.routed.routing.projection,
        )?;
        if let Some((layer, hooks)) = hooks.as_mut() {
            logits = hooks.apply(router_logits_point(*layer), logits)?;
        }
        let decision = self.routed.router_config().route(
            logits.clone().reshape([tokens, experts]),
            self.routed.correction_bias.as_ref().map(Param::val),
        )?;
        // The reference keeps weighted expert contributions and their sum in
        // FP32; only the complete routed mixture is rounded back to the
        // residual dtype, avoiding one rounding event per expert in F16/BF16.
        let dispatch = dispatch_routed_experts(
            flat_input,
            decision.weights.clone(),
            decision.experts.clone(),
            Some(decision.non_finite_tokens),
            experts,
            RoutedExpertPrecision::uniform(ROUTER_DTYPE),
            |expert, rows| self.routed.routed[expert].forward(rows),
        )?;
        let mut routed_hidden = dispatch
            .hidden_state
            .cast(input_dtype)
            .reshape([batch, sequence, hidden]);
        let mut shared_hidden = self.shared_experts.forward(input)?;
        if let Some((layer, hooks)) = hooks.as_mut() {
            routed_hidden = hooks.apply(routed_output_point(*layer), routed_hidden)?;
            shared_hidden = hooks.apply(shared_output_point(*layer), shared_hidden)?;
        }
        Ok(DeepseekMoeOutput {
            hidden_state: routed_hidden.clone() + shared_hidden,
            routed: SparseMoeOutput {
                hidden_state: routed_hidden,
                router_logits: logits,
                selected_experts: decision.experts.reshape([batch, sequence, top_k]),
                selected_weights: decision.weights.reshape([batch, sequence, top_k]),
                expert_token_counts: dispatch.expert_token_counts,
                evaluated_experts: dispatch.evaluated_experts,
            },
        })
    }
}

/// F32 router logits `[batch, token, expert]` of a sparse layer; edits change routing.
pub(super) fn router_logits_point(layer: Layer) -> Point<3> {
    Point::custom(
        format!("decoder.layers.{}.mlp.router_logits", layer.index()),
        [Axis::Batch, Axis::Token, Axis::Other("expert")],
    )
}

/// Weighted routed-expert sum `[batch, token, channel]`, before the shared branch is added.
pub(super) fn routed_output_point(layer: Layer) -> Point<3> {
    Point::custom(
        format!("decoder.layers.{}.mlp.routed_output", layer.index()),
        [Axis::Batch, Axis::Token, Axis::Channel],
    )
}

/// Always-active shared-expert output `[batch, token, channel]`.
pub(super) fn shared_output_point(layer: Layer) -> Point<3> {
    Point::custom(
        format!("decoder.layers.{}.mlp.shared_output", layer.index()),
        [Axis::Batch, Axis::Token, Axis::Channel],
    )
}

/// Sparse-block points in emission order: router logits, routed sum, shared output.
pub(super) fn activation_points(layer: Layer) -> [PointInfo; 3] {
    [
        router_logits_point(layer).info(),
        routed_output_point(layer).info(),
        shared_output_point(layer).info(),
    ]
}

/// Produces router diagnostics with DeepSeek's configured auxiliary-loss semantics.
///
/// `router_logits` is `[batch, token, expert]` (F32) and `selected` the
/// routed expert IDs `[batch, token, K]`; `mask` (`true` = valid) excludes
/// padding from the loss while logits are reported unchanged. The loss is
/// [`auxiliary_loss`] multiplied by `aux_loss_alpha`.
///
/// # Errors
///
/// Returns the shape errors of [`auxiliary_loss`].
pub(super) fn router_diagnostics<B: Backend>(
    router_logits: Tensor<B, 3>,
    selected: Tensor<B, 3, Int>,
    mask: Option<&TokenMask<B>>,
    options: &ForwardOptions,
    config: &DeepseekV2Config,
) -> Result<RouterDiagnostics<B>> {
    let auxiliary = if options.auxiliary_losses {
        Some(
            auxiliary_loss(
                router_logits.clone(),
                selected,
                mask,
                scoring_function(config.scoring_func),
                config.seq_aux,
            )? * config.aux_loss_alpha,
        )
    } else {
        None
    };
    Ok(RouterDiagnostics {
        logits: options.router_logits.then_some(router_logits),
        load_balancing_loss: auxiliary,
        z_loss: None,
    })
}

/// Maps the configuration's scoring function onto the shared router vocabulary.
const fn scoring_function(scoring: DeepseekScoringFunction) -> RouterScoring {
    match scoring {
        DeepseekScoringFunction::Softmax => RouterScoring::Softmax,
        DeepseekScoringFunction::Sigmoid => RouterScoring::Sigmoid,
    }
}

/// Computes DeepSeek's expert-balance loss before `aux_loss_alpha` (module docs).
///
/// `router_logits` is `[batch, token, expert]`, `selected` holds the `K`
/// routed expert IDs `[batch, token, K]` produced by the router (always in
/// `0..E`), and `mask` excludes padding. All arithmetic is F32, and
/// gradients flow through the probabilities only.
///
/// # Errors
///
/// Returns [`Error::Shape`] for empty or mismatched tensors.
pub(super) fn auxiliary_loss<B: Backend>(
    router_logits: Tensor<B, 3>,
    selected: Tensor<B, 3, Int>,
    mask: Option<&TokenMask<B>>,
    scoring: RouterScoring,
    sequence_wise: bool,
) -> Result<Tensor<B, 1>> {
    let [batch, sequence, experts] = router_logits.dims();
    let [selected_batch, selected_sequence, top_k] = selected.dims();
    if batch == 0
        || sequence == 0
        || experts == 0
        || top_k == 0
        || [selected_batch, selected_sequence] != [batch, sequence]
    {
        return Err(Error::shape(
            "deepseek_v2.router_aux",
            "matching non-empty [batch, sequence, experts/top_k] tensors",
            format!(
                "logits={:?}, selected={:?}",
                router_logits.dims(),
                selected.dims()
            ),
        ));
    }
    let device = router_logits.device();
    let logits = router_logits.cast(DType::F32);
    let probabilities = match scoring {
        RouterScoring::Softmax => activation::softmax(logits, 2),
        RouterScoring::Sigmoid => {
            let scores = activation::sigmoid(logits);
            scores.clone() / scores.sum_dim(2)
        }
    };
    // Padding is removed with `mask_fill`, not multiplication, so a
    // non-finite score at a padded token cannot poison the sums (`0 * NaN`).
    let (valid, invalid) = match mask {
        Some(mask) => {
            if mask.as_tensor().dims() != [batch, sequence] {
                return Err(Error::shape(
                    "deepseek_v2.router_aux.mask",
                    format!("[{batch}, {sequence}]"),
                    format!("{:?}", mask.as_tensor().dims()),
                ));
            }
            (
                mask.as_tensor().clone().float().cast(DType::F32),
                Some(mask.as_tensor().clone().bool_not()),
            )
        }
        None => (Tensor::ones([batch, sequence], (&device, DType::F32)), None),
    };
    let assignments = if experts == 1 {
        // Burn's one-hot needs two classes; with one expert every slot selects it.
        Tensor::<B, 4>::ones([batch, sequence, top_k, 1], (&device, DType::F32))
    } else {
        selected.one_hot::<4>(experts).float().cast(DType::F32)
    };
    let (assignments, probabilities) = match invalid {
        Some(invalid) => (
            assignments.mask_fill(
                invalid
                    .clone()
                    .reshape([batch, sequence, 1, 1])
                    .expand([batch, sequence, top_k, experts]),
                0.0,
            ),
            probabilities.mask_fill(
                invalid
                    .reshape([batch, sequence, 1])
                    .expand([batch, sequence, experts]),
                0.0,
            ),
        ),
        None => (assignments, probabilities),
    };
    // count[b, e]: selections of expert e by row b's valid tokens.
    let counts = assignments.sum_dim(1).sum_dim(2).reshape([batch, experts]);
    // mass[b, e]: summed probability of expert e over row b's valid tokens.
    let mass = probabilities.sum_dim(1).reshape([batch, experts]);
    let experts = experts as f64;
    let top_k = top_k as f64;
    let tokens = valid.sum_dim(1).reshape([batch, 1]);
    Ok(if sequence_wise {
        // Rows without valid tokens (finished decode rows) contribute no term
        // and are excluded from the mean instead of diluting it.
        let active = tokens
            .clone()
            .greater_elem(0.0)
            .float()
            .cast(DType::F32)
            .sum()
            .clamp_min(1.0);
        let per_row_tokens = tokens.clamp_min(1.0);
        let load = counts * experts / (per_row_tokens.clone() * top_k);
        let importance = mass / per_row_tokens;
        ((load * importance).sum_dim(1).sum() / active).reshape([1])
    } else {
        let total = tokens.sum().clamp_min(1.0).reshape([1, 1]);
        let load = counts.sum_dim(0) * experts / (total.clone() * top_k);
        let importance = mass.sum_dim(0) / total;
        (load * importance).sum().reshape([1])
    })
}

#[cfg(test)]
#[path = "moe_tests.rs"]
/// Reference routing, group selection, NaN safety, and auxiliary-loss oracles.
mod tests;
