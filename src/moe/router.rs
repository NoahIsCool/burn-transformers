//! Shared top-k expert routing policies for mixture-of-experts blocks.
//!
//! A router maps token logits `z in R^E` to `K` expert IDs and mixture weights.
//! Published families differ along a few independent axes, each represented
//! here as data rather than a model-name branch:
//!
//! | Axis | Choices | Used by |
//! | --- | --- | --- |
//! | [`RouterScoring`] | `softmax(z)`, `sigmoid(z)` | Mixtral/Qwen; DeepSeek-V3, Nemotron-H, GLM-4.5 |
//! | [`ScoringOrder`] | score then top-k; top-k on `z` then score | most; gpt-oss (softmax), Llama 4 (sigmoid) |
//! | [`ExpertSelection`] | greedy; group-limited | most; DeepSeek-V2/V3, Nemotron-H |
//! | correction bias | `s + b` used for selection only | DeepSeek-V3 `noaux_tc`, Nemotron-H |
//! | renormalization | `w / max(sum w, floor)`, `w / (sum w + eps)` | Qwen/Mixtral; DeepSeek-V3/Nemotron |
//! | routed scale | `w * scale` | DeepSeek, Nemotron |
//!
//! With scores `s = score(z)`, selection scores `c = s + b` (or `s` without a
//! correction bias), the selected set `S` is the top `K` of `c` after optional
//! group pruning, and the mixture weights are `w_j = s_j` for `j in S`,
//! optionally renormalized and scaled. In top-k-then-score order `S` is the
//! top `K` of `z` and `w = score(z_S)` over the selected logits only.
//!
//! All routing arithmetic runs in F32 because nearby scores collapse in F16 or
//! BF16 and a flipped selection changes an entire expert computation.
//! [`deterministic_top_k`] breaks ties by the smallest expert ID on every
//! backend and never emits an out-of-range or repeated ID, even for NaN scores;
//! [`RouteDecision::non_finite_tokens`] lets callers report such tokens.

use burn::{
    nn::Linear,
    tensor::{Bool, DType, Int, Tensor, activation, backend::Backend},
};
use serde::{Deserialize, Serialize};

use crate::{Error, Result};

/// Function that converts router logits into nonnegative expert scores.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RouterScoring {
    /// `s = softmax(z)` over all experts (or over the selected logits).
    #[default]
    Softmax,
    /// `s = sigmoid(z)` independently per expert.
    Sigmoid,
}

/// Whether scoring precedes or follows top-k selection.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScoringOrder {
    /// Score every expert, then select the top `K` scores.
    #[default]
    ScoreThenTopK,
    /// Select the top `K` raw logits, then score only those logits.
    ///
    /// With softmax this is gpt-oss's renormalized-by-construction router;
    /// with sigmoid it is Llama 4's. Correction biases and group pruning are
    /// undefined for this order and are rejected.
    TopKThenScore,
}

/// How a group's eligibility score is formed from its experts' selection scores.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GroupScore {
    /// The group's largest score (DeepSeek-V2 `group_limited_greedy`).
    #[default]
    Max,
    /// The sum of the group's two largest scores, or its only score for
    /// single-expert groups (DeepSeek-V3 `noaux_tc`, Nemotron-H).
    TopTwoSum,
}

/// Value written to experts of pruned groups before the final top-k.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PrunedExpertScore {
    /// `-inf`: pruned experts can never be selected (official DeepSeek-V3, Nemotron-H).
    #[default]
    NegativeInfinity,
    /// `0.0`, as Hugging Face Transformers 5.2 writes for DeepSeek-V2/V3. A
    /// pruned expert can then outrank eligible experts whose selection score
    /// is negative.
    Zero,
}

/// Group-limited routing: keep the best `top_groups` of `groups` equal expert groups.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GroupLimitedSelection {
    /// Number of equal, contiguous expert groups; must divide the expert count.
    pub groups: usize,
    /// Number of groups whose experts stay eligible, in `1..=groups`.
    pub top_groups: usize,
    /// How each group is scored.
    #[serde(default)]
    pub group_score: GroupScore,
    /// Value given to experts in pruned groups.
    #[serde(default)]
    pub pruned_score: PrunedExpertScore,
}

/// Which experts are eligible for the final top-k.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExpertSelection {
    /// Every expert is eligible.
    #[default]
    Greedy,
    /// Only experts in the best groups are eligible.
    GroupLimited(
        /// Group geometry and scoring.
        GroupLimitedSelection,
    ),
}

/// Denominator guard used when selected weights are renormalized to sum to one.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RenormalizationDenominator {
    /// `w / max(sum w, floor)`; with `f32::MIN_POSITIVE` this is plain division
    /// for every normal sum (Qwen, Mixtral, this crate's historical default).
    ClampedBelow(
        /// Nonnegative finite lower bound of the denominator.
        f32,
    ),
    /// `w / (sum w + epsilon)`, the Hugging Face DeepSeek-V3/Nemotron-H form.
    PlusEpsilon(
        /// Nonnegative finite additive stabilizer.
        f32,
    ),
}

impl Default for RenormalizationDenominator {
    /// Plain division guarded only against a zero or subnormal sum.
    fn default() -> Self {
        Self::ClampedBelow(f32::MIN_POSITIVE)
    }
}

/// Dtype in which the router projection `z = x W (+ b)` is evaluated.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RouterProjectionPrecision {
    /// Project in the activation dtype, then cast logits to F32 (Qwen, Mixtral).
    #[default]
    ModelDtype,
    /// Cast activations and router parameters to F32 before projecting
    /// (DeepSeek, Nemotron-H), so F16 rounding cannot flip a selection.
    Fp32,
}

/// Routing policy shared by every sparse expert block, independent of sizes.
///
/// The default reproduces softmax/greedy routing with model-dtype logits and
/// no biases, which is this crate's historical [`super::SparseMoe`] behavior.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct RoutingPolicy {
    /// Score function.
    #[serde(default)]
    pub scoring: RouterScoring,
    /// Whether scoring precedes top-k selection.
    #[serde(default)]
    pub order: ScoringOrder,
    /// Greedy or group-limited eligibility.
    #[serde(default)]
    pub selection: ExpertSelection,
    /// Whether a frozen F32 per-expert bias is added to scores for selection only.
    #[serde(default)]
    pub correction_bias: bool,
    /// Whether the router projection has an additive bias (gpt-oss).
    #[serde(default)]
    pub router_bias: bool,
    /// Precision of the router projection.
    #[serde(default)]
    pub projection: RouterProjectionPrecision,
    /// Denominator guard used when renormalization is enabled.
    #[serde(default)]
    pub denominator: RenormalizationDenominator,
}

/// Complete routing specification for one expert bank.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct RouterConfig {
    /// Number of routed experts `E`.
    pub experts: usize,
    /// Number of experts selected per token `K`, in `1..=E`.
    pub experts_per_token: usize,
    /// Scoring, order, eligibility, biases, and precision.
    pub policy: RoutingPolicy,
    /// Whether selected weights are renormalized to sum to one.
    pub renormalize_selected: bool,
    /// Positive finite multiplier applied to the final mixture weights.
    pub routed_scale: f64,
}

/// Discrete experts and F32 mixture weights chosen for every token.
#[derive(Clone, Debug)]
pub struct RouteDecision<B: Backend> {
    /// Selected expert IDs `[tokens, K]`, distinct within a token and in `0..E`,
    /// ordered by decreasing selection score with smallest-ID tie-breaking.
    pub experts: Tensor<B, 2, Int>,
    /// F32 mixture weights `[tokens, K]` after renormalization and scaling.
    pub weights: Tensor<B, 2>,
    /// `[tokens]` flags marking tokens whose scores contained NaN. Their IDs
    /// are still valid but meaningless; callers should report these tokens.
    pub non_finite_tokens: Tensor<B, 1, Bool>,
}

impl RouterConfig {
    /// Creates softmax/greedy routing with renormalization and unit scale.
    #[must_use]
    pub fn new(experts: usize, experts_per_token: usize) -> Self {
        Self {
            experts,
            experts_per_token,
            policy: RoutingPolicy::default(),
            renormalize_selected: true,
            routed_scale: 1.0,
        }
    }

    /// Validates cardinalities, group geometry, and policy combinations.
    ///
    /// # Errors
    ///
    /// Returns a contextual configuration error for zero experts, `K` outside
    /// `1..=E`, a non-positive or non-finite scale, an invalid denominator
    /// guard, groups that do not divide `E` or leave fewer than `K` eligible
    /// experts, and a correction bias or group pruning combined with
    /// top-k-then-score order.
    pub fn validate(&self) -> Result<()> {
        if self.experts == 0 {
            return Err(Error::configuration(
                "moe.router.experts",
                "must be non-zero",
            ));
        }
        if self.experts_per_token == 0 || self.experts_per_token > self.experts {
            return Err(Error::configuration(
                "moe.router.experts_per_token",
                format!(
                    "must be in 1..={}, got {}",
                    self.experts, self.experts_per_token
                ),
            ));
        }
        if !self.routed_scale.is_finite() || self.routed_scale <= 0.0 {
            return Err(Error::configuration(
                "moe.router.routed_scale",
                "must be finite and greater than zero",
            ));
        }
        let guard = match self.policy.denominator {
            RenormalizationDenominator::ClampedBelow(value)
            | RenormalizationDenominator::PlusEpsilon(value) => value,
        };
        if !guard.is_finite() || guard < 0.0 {
            return Err(Error::configuration(
                "moe.router.denominator",
                "renormalization guard must be finite and nonnegative",
            ));
        }
        if self.policy.order == ScoringOrder::TopKThenScore
            && (self.policy.correction_bias || self.policy.selection != ExpertSelection::Greedy)
        {
            return Err(Error::configuration(
                "moe.router.order",
                "top-k-then-score routing ranks raw logits and supports neither a correction bias nor group pruning",
            ));
        }
        if let ExpertSelection::GroupLimited(groups) = self.policy.selection {
            self.validate_groups(groups)?;
        }
        Ok(())
    }

    /// Checks that groups partition the experts and leave at least `K` eligible.
    fn validate_groups(&self, selection: GroupLimitedSelection) -> Result<()> {
        if selection.groups == 0 || !self.experts.is_multiple_of(selection.groups) {
            return Err(Error::configuration(
                "moe.router.groups",
                format!(
                    "must be a non-zero divisor of {} experts, got {}",
                    self.experts, selection.groups
                ),
            ));
        }
        if selection.top_groups == 0 || selection.top_groups > selection.groups {
            return Err(Error::configuration(
                "moe.router.top_groups",
                format!(
                    "must be in 1..={}, got {}",
                    selection.groups, selection.top_groups
                ),
            ));
        }
        let eligible = selection.top_groups * (self.experts / selection.groups);
        if eligible < self.experts_per_token {
            return Err(Error::configuration(
                "moe.router.top_groups",
                format!(
                    "retained groups hold {eligible} experts, fewer than the {} selected per token",
                    self.experts_per_token
                ),
            ));
        }
        Ok(())
    }

    /// Routes `[tokens, E]` logits to `K` experts per token.
    ///
    /// Logits may use any float dtype and are cast to F32 first.
    /// `correction_bias` is the `[E]` selection-only bias and must be present
    /// exactly when the policy enables it.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::validate`], a shape error for empty or
    /// mismatched logits or bias, and a configuration error when the bias
    /// presence disagrees with the policy.
    pub fn route<B: Backend>(
        &self,
        logits: Tensor<B, 2>,
        correction_bias: Option<Tensor<B, 1>>,
    ) -> Result<RouteDecision<B>> {
        self.validate()?;
        let [tokens, experts] = logits.dims();
        if tokens == 0 || experts != self.experts {
            return Err(Error::shape(
                "moe.router.logits",
                format!("[tokens>0, {}]", self.experts),
                format!("[{tokens}, {experts}]"),
            ));
        }
        let correction_bias = self.checked_correction_bias(correction_bias)?;
        let logits = logits.cast(DType::F32);
        let (experts, weights, ranked) = match self.policy.order {
            ScoringOrder::ScoreThenTopK => self.score_then_select(logits, correction_bias),
            ScoringOrder::TopKThenScore => self.select_then_score(logits),
        };
        Ok(RouteDecision {
            experts,
            weights: self.finish_weights(weights),
            non_finite_tokens: ranked.is_nan().any_dim(1).reshape([tokens]),
        })
    }

    /// Requires a `[E]` bias exactly when the policy uses one and casts it to F32.
    fn checked_correction_bias<B: Backend>(
        &self,
        correction_bias: Option<Tensor<B, 1>>,
    ) -> Result<Option<Tensor<B, 1>>> {
        match (self.policy.correction_bias, correction_bias) {
            (false, None) => Ok(None),
            (true, Some(bias)) if bias.dims() == [self.experts] => Ok(Some(bias.cast(DType::F32))),
            (true, Some(bias)) => Err(Error::shape(
                "moe.router.correction_bias",
                format!("[{}]", self.experts),
                format!("{:?}", bias.dims()),
            )),
            (true, None) => Err(Error::configuration(
                "moe.router.correction_bias",
                "the routing policy requires a selection bias, but none was supplied",
            )),
            (false, Some(_)) => Err(Error::configuration(
                "moe.router.correction_bias",
                "a selection bias was supplied, but the routing policy does not use one",
            )),
        }
    }

    /// Scores all experts, prunes groups, selects top-k, and gathers uncorrected scores.
    ///
    /// Returns `(ids, weights, ranked)`, where `ranked` holds the selection
    /// scores used for NaN reporting.
    fn score_then_select<B: Backend>(
        &self,
        logits: Tensor<B, 2>,
        correction_bias: Option<Tensor<B, 1>>,
    ) -> (Tensor<B, 2, Int>, Tensor<B, 2>, Tensor<B, 2>) {
        let [tokens, experts] = logits.dims();
        let scores = score(self.policy.scoring, logits);
        let selection_scores = match correction_bias {
            Some(bias) => scores.clone() + bias.reshape([1, experts]),
            None => scores.clone(),
        };
        let eligible = match self.policy.selection {
            ExpertSelection::Greedy => selection_scores.clone(),
            ExpertSelection::GroupLimited(groups) => prune_groups(selection_scores.clone(), groups),
        };
        let k = self.experts_per_token;
        let (selected_values, ids) = deterministic_top_k(eligible.unsqueeze_dim::<3>(0), k);
        let ids = ids.reshape([tokens, k]);
        // Without a correction bias the selected values already are the scores;
        // otherwise mix with the uncorrected scores.
        let weights = if self.policy.correction_bias {
            scores.gather(1, ids.clone())
        } else {
            selected_values.reshape([tokens, k])
        };
        (ids, weights, selection_scores)
    }

    /// Selects the top-k raw logits and scores only those logits.
    fn select_then_score<B: Backend>(
        &self,
        logits: Tensor<B, 2>,
    ) -> (Tensor<B, 2, Int>, Tensor<B, 2>, Tensor<B, 2>) {
        let [tokens, _] = logits.dims();
        let k = self.experts_per_token;
        let (top_logits, ids) = deterministic_top_k(logits.clone().unsqueeze_dim::<3>(0), k);
        let weights = score(self.policy.scoring, top_logits.reshape([tokens, k]));
        (ids.reshape([tokens, k]), weights, logits)
    }

    /// Applies optional renormalization and the routed scale to `[tokens, K]` weights.
    fn finish_weights<B: Backend>(&self, weights: Tensor<B, 2>) -> Tensor<B, 2> {
        let weights = if self.renormalize_selected {
            let sum = weights.clone().sum_dim(1);
            let denominator = match self.policy.denominator {
                RenormalizationDenominator::ClampedBelow(floor) => sum.clamp_min(floor),
                RenormalizationDenominator::PlusEpsilon(epsilon) => sum.add_scalar(epsilon),
            };
            weights / denominator
        } else {
            weights
        };
        // Multiplying by one is exact, so skipping it changes no value.
        if self.routed_scale == 1.0 {
            weights
        } else {
            weights * self.routed_scale
        }
    }
}

/// Applies the routing score function along the expert axis of `[tokens, E]` values.
fn score<B: Backend>(scoring: RouterScoring, logits: Tensor<B, 2>) -> Tensor<B, 2> {
    match scoring {
        RouterScoring::Softmax => activation::softmax(logits, 1),
        RouterScoring::Sigmoid => activation::sigmoid(logits),
    }
}

/// Keeps the experts of the best `top_groups` groups and overwrites the rest.
///
/// Experts form `groups` contiguous groups of `E / groups`. A group's score is
/// its maximum or its top-two sum; groups are ranked with the same
/// deterministic top-k as experts.
fn prune_groups<B: Backend>(
    scores: Tensor<B, 2>,
    selection: GroupLimitedSelection,
) -> Tensor<B, 2> {
    if selection.top_groups == selection.groups {
        // Every group survives (including Nemotron-H's `n_group = 1`), so
        // pruning is the identity. Burn also rejects one-hot with one class.
        return scores;
    }
    let [tokens, experts] = scores.dims();
    let per_group = experts / selection.groups;
    let grouped = scores
        .clone()
        .reshape([tokens, selection.groups, per_group]);
    let group_scores = match selection.group_score {
        GroupScore::Max => grouped.max_dim(2),
        GroupScore::TopTwoSum => deterministic_top_k(grouped, 2.min(per_group)).0.sum_dim(2),
    }
    .reshape([1, tokens, selection.groups]);
    let kept_groups = deterministic_top_k(group_scores, selection.top_groups)
        .1
        .reshape([tokens, selection.top_groups]);
    let kept_experts = kept_groups
        .one_hot::<3>(selection.groups)
        .sum_dim(1)
        .greater_elem(0)
        .reshape([tokens, selection.groups, 1])
        .expand([tokens, selection.groups, per_group])
        .reshape([tokens, experts]);
    let fill = match selection.pruned_score {
        PrunedExpertScore::NegativeInfinity => f32::NEG_INFINITY,
        PrunedExpertScore::Zero => 0.0,
    };
    scores.mask_fill(kept_experts.bool_not(), fill)
}

/// Projects `[.., hidden]` activations to `[.., E]` router logits at a chosen precision.
///
/// `ModelDtype` evaluates the module in the activation dtype; `Fp32` casts the
/// activations, weight, and optional bias to F32 first. The result keeps the
/// projection dtype; [`RouterConfig::route`] casts to F32 itself.
///
/// # Errors
///
/// Returns the shape and dtype errors of [`crate::operations::linear`] for
/// model-dtype projection, and a shape error when the F32 weight does not
/// match the activation width.
pub fn router_logits<B: Backend, const D: usize>(
    input: Tensor<B, D>,
    router: &Linear<B>,
    precision: RouterProjectionPrecision,
) -> Result<Tensor<B, D>> {
    match precision {
        RouterProjectionPrecision::ModelDtype => crate::operations::linear(input, router),
        RouterProjectionPrecision::Fp32 => {
            let weight = router.weight.val();
            let [weight_input, _] = weight.dims();
            if input.dims()[D - 1] != weight_input {
                return Err(Error::shape(
                    "moe.router.weight",
                    format!("[{}, experts]", input.dims()[D - 1]),
                    format!("{:?}", weight.dims()),
                ));
            }
            Ok(burn::tensor::module::linear(
                input.cast(DType::F32),
                weight.cast(DType::F32),
                router.bias.as_ref().map(|bias| bias.val().cast(DType::F32)),
            ))
        }
    }
}

/// Selects the largest `k` values along the last axis with a backend-independent tie rule.
///
/// Burn's native top-k kernels leave equal-value ordering to the backend.
/// Routing must instead be reproducible across Flex, WGPU, and CUDA, because
/// changing one tied expert changes an entire matrix computation. Each
/// iteration finds the maximum remaining ranking key, resolves its equality
/// set to the smallest still-available index, gathers that entry from
/// `values`, and retires exactly that index. The complexity is `O(kE)` and no
/// `O(E^2)` pairwise-ranking tensor is allocated.
///
/// The kernel is total: NaN ranks like `-inf`, and retired indices are tracked
/// separately from scores, so for `k <= E` every returned index is in range and
/// distinct within a row even when all remaining keys are `-inf` or NaN. The
/// gathered values come from the original tensor and may therefore be NaN.
/// For finite inputs the result equals the historical implementation.
pub(crate) fn deterministic_top_k<B: Backend>(
    values: Tensor<B, 3>,
    k: usize,
) -> (Tensor<B, 3>, Tensor<B, 3, Int>) {
    let [batch, rows, columns] = values.dims();
    debug_assert!(k <= columns, "top-k needs k <= {columns}, got {k}");
    let device = values.device();
    let ids = Tensor::<B, 1, Int>::arange(0..columns as i64, &device)
        .reshape([1, 1, columns])
        .expand([batch, rows, columns]);
    let mut keys = values
        .clone()
        .mask_fill(values.clone().is_nan(), f32::NEG_INFINITY);
    let mut available = ids.clone().lower_elem(columns as i64);
    let mut selected_values = Vec::with_capacity(k);
    let mut selected_indices = Vec::with_capacity(k);
    for _ in 0..k {
        let maximum = keys.clone().max_dim(2);
        let tied = keys.clone().equal(maximum).bool_and(available.clone());
        let index = ids
            .clone()
            .mask_fill(tied.bool_not(), columns as i64)
            .min_dim(2);
        let chosen = ids
            .clone()
            .equal(index.clone().expand([batch, rows, columns]));
        selected_values.push(values.clone().gather(2, index.clone()));
        selected_indices.push(index);
        keys = keys.mask_fill(chosen.clone(), f32::NEG_INFINITY);
        available = available.bool_and(chosen.bool_not());
    }
    (
        Tensor::cat(selected_values, 2),
        Tensor::cat(selected_indices, 2),
    )
}

#[cfg(test)]
#[path = "router_tests.rs"]
mod tests;
