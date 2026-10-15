//! Sparse mixture-of-experts routing and dispatch.
//!
//! A router assigns scores to `E` experts and evaluates only `K` of them for
//! each token. The output is `sum_{j in S} w_j Expert_j(x)`, optionally plus
//! always-active shared experts `g(x) * sum_s Shared_s(x)` with an optional
//! sigmoid gate `g`. The routing policy (softmax or sigmoid scores, ordering,
//! group pruning, selection-only correction bias, renormalization, scale, and
//! projection precision) lives in [`router`] and is shared by every family;
//! the defaults reproduce Qwen/Mixtral softmax top-k routing.
//!
//! Dispatch reads the compact `[tokens, K]` expert IDs to the host once (one
//! synchronization, which also carries a per-token non-finite flag), groups
//! the flattened `(token, slot)` assignments by expert with a verified stable
//! counting sort, and uploads that permutation once. Experts are then taken
//! in blocks of whole experts holding at most `max(tokens, 16384)` routed rows:
//! each block gathers its input rows and weights with one `select`, evaluates
//! each active expert on a contiguous slice, and accumulates its weighted
//! contributions with one `select_assign`. Decode and short prompts fit one
//! block; a long prefill takes about `K` blocks, so the gathered buffers stay
//! near the residual input's size instead of `K` times larger. The
//! permutation is expert-major and token-major within an expert, so each
//! token's contributions are summed in the same order as a per-expert loop
//! and results are bit-identical to it for any block size. Unselected experts
//! are never evaluated. Expert inputs and mixture weights remain
//! differentiable tensors; only the discrete IDs leave the device.

use burn::{
    module::{Module, Param},
    nn::{Linear, LinearConfig, LinearLayout},
    tensor::{
        Bool, DType, ElementConversion, FloatDType, IndexingUpdateOp, Int, Tensor, TensorData,
        activation, backend::Backend,
    },
};

use crate::{
    Error, Result,
    architecture::{Activation, TokenMask},
    feed_forward::{GatedFeedForward, GatedFeedForwardConfig},
    operations,
};

mod assignment;
pub mod router;

use assignment::{ExpertAssignmentPlan, InvalidExpertId, plan_expert_assignments};
pub(crate) use router::deterministic_top_k;
pub use router::{
    ExpertSelection, GroupLimitedSelection, GroupScore, PrunedExpertScore,
    RenormalizationDenominator, RouteDecision, RouterConfig, RouterProjectionPrecision,
    RouterScoring, RoutingPolicy, ScoringOrder, router_logits,
};

/// Rounding boundary for selected router weights in a sparse expert block.
///
/// This is independent of router-softmax precision. Both modes may rank and
/// normalize in FP32; they differ in whether a selected probability is rounded
/// before or after multiplication by a reduced-precision expert output.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RoutedWeightingPrecision {
    /// Round selected weights to the residual dtype before multiplication.
    #[default]
    Residual,
    /// Multiply expert outputs and selected weights in FP32, then round each contribution.
    Fp32ThenResidual,
}

/// Configuration for routed and optionally shared gated experts.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SparseMoeConfig {
    /// Residual-stream width accepted and returned by every expert.
    pub hidden_size: usize,
    /// Intermediate width of each routed and shared expert.
    pub intermediate_size: usize,
    /// Number of independently parameterized routed experts.
    pub routed_experts: usize,
    /// Number of routed experts selected for each token.
    pub experts_per_token: usize,
    /// Number of experts evaluated for every token.
    pub shared_experts: usize,
    /// Expert gate activation.
    pub activation: Activation,
    /// Whether selected router weights are normalized to sum to one.
    pub renormalize_selected: bool,
    /// Scale applied to the complete routed-expert contribution.
    pub routed_scale: f64,
    /// Rounding boundary for selected probabilities and expert outputs.
    #[serde(default)]
    pub weighting_precision: RoutedWeightingPrecision,
    /// Whether expert projections contain additive biases.
    pub expert_bias: bool,
    /// Scoring, ordering, eligibility, bias, and precision policy of the router.
    #[serde(default)]
    pub routing: RoutingPolicy,
    /// Whether a sigmoid gate `sigmoid(x w)` scales the summed shared experts
    /// (Qwen3-Next / Qwen3.5-MoE `shared_expert_gate`).
    #[serde(default)]
    pub shared_expert_gate: bool,
}

impl SparseMoeConfig {
    /// Creates a bias-free top-k SwiGLU router without shared experts.
    pub fn new(
        hidden_size: usize,
        intermediate_size: usize,
        routed_experts: usize,
        experts_per_token: usize,
    ) -> Self {
        Self {
            hidden_size,
            intermediate_size,
            routed_experts,
            experts_per_token,
            shared_experts: 0,
            activation: Activation::Silu,
            renormalize_selected: true,
            routed_scale: 1.0,
            weighting_precision: RoutedWeightingPrecision::Residual,
            expert_bias: false,
            routing: RoutingPolicy::default(),
            shared_expert_gate: false,
        }
    }

    /// Returns the complete routing specification implied by this configuration.
    #[must_use]
    pub fn router_config(&self) -> RouterConfig {
        RouterConfig {
            experts: self.routed_experts,
            experts_per_token: self.experts_per_token,
            policy: self.routing,
            renormalize_selected: self.renormalize_selected,
            routed_scale: self.routed_scale,
        }
    }

    /// Validates sparse cardinalities and expert numerical parameters.
    pub fn validate(&self) -> Result<()> {
        if self.hidden_size == 0 || self.intermediate_size == 0 || self.routed_experts == 0 {
            return Err(Error::configuration(
                "moe",
                "hidden_size, intermediate_size, and routed_experts must be non-zero",
            ));
        }
        // Burn parameters are lazy, so initialization alone cannot reject a
        // matrix whose logical element count will overflow at first use.
        for (width, projection) in [
            (self.intermediate_size, "expert"),
            (self.routed_experts, "router"),
        ] {
            if self.hidden_size.checked_mul(width).is_none() {
                return Err(Error::configuration(
                    "moe.projection_shape",
                    format!("{projection} matrix element count overflows usize"),
                ));
            }
        }
        if self.experts_per_token == 0 || self.experts_per_token > self.routed_experts {
            return Err(Error::configuration(
                "moe.experts_per_token",
                "must be in 1..=routed_experts",
            ));
        }
        if !self.routed_scale.is_finite() || self.routed_scale <= 0.0 {
            return Err(Error::configuration(
                "moe.routed_scale",
                "must be finite and greater than zero",
            ));
        }
        if self.shared_expert_gate && self.shared_experts == 0 {
            return Err(Error::configuration(
                "moe.shared_expert_gate",
                "requires at least one shared expert to gate",
            ));
        }
        self.router_config().validate()
    }

    /// Initializes the router and all expert parameter sets.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Result<SparseMoe<B>> {
        self.validate()?;
        let expert_config = GatedFeedForwardConfig::new(self.hidden_size, self.intermediate_size)
            .with_activation(self.activation)
            .with_bias(self.expert_bias);
        let routed = (0..self.routed_experts)
            .map(|_| expert_config.init(device))
            .collect::<Result<Vec<_>>>()?;
        let shared = (0..self.shared_experts)
            .map(|_| expert_config.init(device))
            .collect::<Result<Vec<_>>>()?;
        let router = operations::init_linear(
            &LinearConfig::new(self.hidden_size, self.routed_experts)
                .with_bias(self.routing.router_bias)
                .with_layout(LinearLayout::Col),
            device,
        );
        // Like DeepSeek-V3 and Nemotron-H, the selection bias is an FP32
        // buffer: it changes discrete choices but is not trained by gradients.
        let correction_bias = self.routing.correction_bias.then(|| {
            Param::from_tensor(Tensor::zeros([self.routed_experts], (device, DType::F32)))
                .set_require_grad(false)
        });
        let shared_expert_gate = self.shared_expert_gate.then(|| {
            operations::init_linear(
                &LinearConfig::new(self.hidden_size, 1)
                    .with_bias(false)
                    .with_layout(LinearLayout::Col),
                device,
            )
        });
        Ok(SparseMoe {
            router,
            routed,
            shared,
            experts_per_token: self.experts_per_token,
            renormalize_selected: self.renormalize_selected,
            routed_scale: self.routed_scale,
            weighting_precision: self.weighting_precision,
            hidden_size: self.hidden_size,
            routing: self.routing,
            correction_bias,
            shared_expert_gate,
        })
    }
}

/// Sparse router output and optional inspection data.
#[derive(Clone, Debug)]
pub struct SparseMoeOutput<B: Backend> {
    /// Weighted routed and shared expert output.
    pub hidden_state: Tensor<B, 3>,
    /// Pre-softmax router scores shaped `[batch, sequence, experts]`.
    pub router_logits: Tensor<B, 3>,
    /// Selected expert IDs shaped `[batch, sequence, experts_per_token]`.
    pub selected_experts: Tensor<B, 3, Int>,
    /// Selected probabilities after optional renormalization and routed scaling.
    pub selected_weights: Tensor<B, 3>,
    /// Number of token assignments dispatched to each routed expert.
    ///
    /// The sum is `batch * sequence * experts_per_token`. A zero entry proves
    /// that the portable dispatcher skipped that expert's feed-forward graph.
    pub expert_token_counts: Vec<usize>,
    /// Number of routed expert graphs that were actually evaluated.
    pub evaluated_experts: usize,
}

/// Device output and host-side accounting from one selected-expert dispatch.
///
/// This internal result deliberately excludes routing tensors: callers retain
/// those tensors for diagnostics while this value owns only the differentiable
/// weighted expert sum and inexpensive assignment counts.
#[derive(Clone, Debug)]
pub(crate) struct RoutedExpertDispatch<B: Backend> {
    /// Weighted routed-expert sum shaped `[tokens, hidden]`.
    pub(crate) hidden_state: Tensor<B, 2>,
    /// Number of `(token, slot)` assignments owned by every expert.
    pub(crate) expert_token_counts: Vec<usize>,
    /// Number of expert graphs evaluated by this invocation.
    pub(crate) evaluated_experts: usize,
}

/// Numerical policy for weighting and accumulating routed expert outputs.
///
/// The distinction is observable in reduced precision. Qwen multiplies each
/// reduced-precision expert output by its FP32 routing weight, rounds that
/// weighted contribution to the residual dtype, and only then performs
/// `index_add`. DeepSeek instead retains the weighted contributions and their
/// complete sum in FP32 before one final cast at the routed-block boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RoutedExpertPrecision {
    /// Dtype used for the elementwise expert-output/router-weight product.
    pub(crate) multiplication_dtype: DType,
    /// Dtype of the destination and every contribution entering index-add.
    pub(crate) accumulation_dtype: DType,
}

impl RoutedExpertPrecision {
    /// Uses one dtype for multiplication and accumulation.
    pub(crate) const fn uniform(dtype: DType) -> Self {
        Self {
            multiplication_dtype: dtype,
            accumulation_dtype: dtype,
        }
    }

    /// Keeps routing weights FP32 but rounds each contribution before summing.
    pub(crate) const fn fp32_weights_round_each(accumulation_dtype: DType) -> Self {
        Self {
            multiplication_dtype: DType::F32,
            accumulation_dtype,
        }
    }

    /// Rejects integer or Boolean policies before backend kernels execute.
    fn validate(self) -> Result<()> {
        if !self.multiplication_dtype.is_float() || !self.accumulation_dtype.is_float() {
            return Err(Error::configuration(
                "moe.dispatch.precision",
                format!(
                    "multiplication and accumulation dtypes must be floating point, found {:?} and {:?}",
                    self.multiplication_dtype, self.accumulation_dtype
                ),
            ));
        }
        Ok(())
    }
}

/// Multiplies an expert output by its routing weights at the configured precision.
///
/// Explicit casts make the rounding point backend-independent. In particular,
/// this does not rely on implicit mixed-dtype promotion when an F16 expert is
/// multiplied by an FP32 routing probability.
fn weighted_expert_contribution<B: Backend>(
    expert_output: Tensor<B, 2>,
    weights: Tensor<B, 2>,
    precision: RoutedExpertPrecision,
) -> Tensor<B, 2> {
    (expert_output.cast(precision.multiplication_dtype)
        * weights.cast(precision.multiplication_dtype))
    .cast(precision.accumulation_dtype)
}

/// Host metadata for one expert, ordered by flattened `(token, slot)` index.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct ExpertAssignmentGroup {
    /// Token rows selected from the residual stream.
    pub(crate) token_indices: Vec<i64>,
    /// Flattened `(token, top-k slot)` indices selected from router weights.
    pub(crate) weight_indices: Vec<i64>,
}

/// Reads discrete routing IDs once and partitions assignments by expert.
///
/// Expert IDs are control-flow metadata rather than differentiable values, so
/// copying this compact `tokens * top_k` tensor to the host does not sever the
/// gradient graph for either residual states or selected router weights.
/// Within each expert, assignments retain token-major, slot-minor order,
/// making grouping deterministic across backends even when router scores tie.
/// [`dispatch_routed_experts`] uses the same plan without materializing one
/// vector pair per expert.
#[cfg_attr(not(test), allow(dead_code))] // Production callers moved to dispatch_routed_experts.
pub(crate) fn group_selected_experts<B: Backend>(
    selected_experts: Tensor<B, 2, Int>,
    routed_experts: usize,
) -> Result<Vec<ExpertAssignmentGroup>> {
    let slots = selected_experts.dims()[1];
    let plan = read_assignment_plan(selected_experts, None, routed_experts)?;
    let mut groups = Vec::with_capacity(routed_experts);
    for expert in 0..routed_experts {
        let assignments = &plan.order[plan.offsets[expert]..plan.offsets[expert + 1]];
        let mut group = ExpertAssignmentGroup::default();
        for &assignment in assignments {
            // `read_assignment_plan` checked that every index fits in i64.
            group.token_indices.push((assignment / slots) as i64);
            group.weight_indices.push(assignment as i64);
        }
        groups.push(group);
    }
    Ok(groups)
}

/// Reads `[tokens, K]` expert IDs (and optional NaN flags) in one transfer and plans dispatch.
///
/// The NaN flags, when present, are appended as an extra integer column so a
/// single device-to-host copy carries both. A flagged token is reported as a
/// contextual error rather than being dispatched with meaningless weights.
fn read_assignment_plan<B: Backend>(
    selected_experts: Tensor<B, 2, Int>,
    non_finite_tokens: Option<Tensor<B, 1, Bool>>,
    routed_experts: usize,
) -> Result<ExpertAssignmentPlan> {
    let [tokens, slots] = selected_experts.dims();
    if routed_experts == 0 || tokens == 0 || slots == 0 {
        return Err(Error::shape(
            "moe.selected_experts",
            "[tokens>0, top_k>0] with at least one routed expert",
            format!("shape=[{tokens}, {slots}], routed_experts={routed_experts}"),
        ));
    }
    let assignments = tokens
        .checked_mul(slots)
        .filter(|&count| i64::try_from(count).is_ok())
        .ok_or_else(|| {
            Error::shape(
                "moe.selected_experts",
                "a tokens * top_k product representable by the i64 index type",
                format!("{tokens} * {slots}"),
            )
        })?;
    let flagged = non_finite_tokens.is_some();
    let columns = slots + usize::from(flagged);
    let host = match non_finite_tokens {
        Some(flags) => Tensor::cat(vec![selected_experts, flags.int().reshape([tokens, 1])], 1),
        None => selected_experts,
    };
    let values = host
        .try_into_data()
        .map_err(|error| {
            Error::BackendCapability(format!(
                "failed to read selected expert IDs from backend: {error}"
            ))
        })?
        .to_vec::<B::IntElem>()
        .map_err(|error| {
            Error::BackendCapability(format!(
                "failed to decode selected expert IDs from backend: {error}"
            ))
        })?;
    if values.len() != tokens * columns {
        return Err(Error::shape(
            "moe.selected_experts",
            format!("{assignments} integer routing IDs"),
            format!("{} values", values.len()),
        ));
    }
    let mut ids = Vec::with_capacity(assignments);
    for (token, row) in values.chunks_exact(columns).enumerate() {
        if flagged && row[slots].elem::<i64>() != 0 {
            return Err(Error::shape(
                "moe.router_scores",
                "finite router scores and weights for every token",
                format!("non-finite router score or weight for flattened token {token}"),
            ));
        }
        ids.extend(row[..slots].iter().map(|value| value.elem::<i64>()));
    }
    plan_expert_assignments(&ids, routed_experts).map_err(
        |InvalidExpertId { assignment, expert }| {
            Error::shape(
                "moe.selected_experts",
                format!("expert IDs in 0..{routed_experts}"),
                format!("expert ID {expert} at flattened assignment {assignment}"),
            )
        },
    )
}

/// Fewest routed rows that sparse dispatch gathers at once.
///
/// Dispatch materializes the gathered inputs and weighted outputs of a block
/// of experts together. Blocks hold up to `max(tokens, this)` assignments, so
/// those buffers stay about as large as the residual input itself (or a few
/// MiB for short inputs) instead of `top_k` times larger, while still taking
/// only about `top_k` gather/scatter rounds for a long prefill and a single
/// round for decode-sized inputs.
pub(crate) const MIN_DISPATCH_BLOCK_ASSIGNMENTS: usize = 16_384;

/// Evaluates selected experts with one ID transfer and a few gathers and scatter-adds.
///
/// `flat_input` is `[tokens, width]`, `flat_weights` and `selected_experts`
/// are `[tokens, K]`, and `run_expert(e, rows)` evaluates expert `e` on its
/// gathered `[assignments_e, width]` rows, returning `[assignments_e, out]`.
/// Contributions `run_expert(e, x) * w` are formed with `precision` and
/// accumulated in expert-major, token-major order, which matches a per-expert
/// loop of `select_assign` calls bit for bit. Experts are processed in blocks
/// of at most `max(tokens, MIN_DISPATCH_BLOCK_ASSIGNMENTS)` routed rows (an
/// expert is never split), which bounds peak memory without changing that
/// order. A `[tokens]` non-finite flag tensor can ride along with the ID
/// transfer; any flagged token is a contextual error.
pub(crate) fn dispatch_routed_experts<B, F>(
    flat_input: Tensor<B, 2>,
    flat_weights: Tensor<B, 2>,
    selected_experts: Tensor<B, 2, Int>,
    non_finite_tokens: Option<Tensor<B, 1, Bool>>,
    routed_experts: usize,
    precision: RoutedExpertPrecision,
    run_expert: F,
) -> Result<RoutedExpertDispatch<B>>
where
    B: Backend,
    F: FnMut(usize, Tensor<B, 2>) -> Result<Tensor<B, 2>>,
{
    let budget = flat_input.dims()[0].max(MIN_DISPATCH_BLOCK_ASSIGNMENTS);
    dispatch_routed_experts_in_blocks(
        flat_input,
        flat_weights,
        selected_experts,
        non_finite_tokens,
        routed_experts,
        precision,
        budget,
        run_expert,
    )
}

/// [`dispatch_routed_experts`] with an explicit routed-row budget per block.
///
/// The plan is uploaded once as an expert-major permutation of flattened
/// assignments; token rows follow on the device as `assignment / K`. Each
/// block of whole experts (see [`assignment::next_dispatch_block`]) narrows
/// that permutation, gathers its input rows and weights, evaluates its
/// experts on contiguous slices, and scatter-adds the concatenated
/// contributions. Blocks run in expert order, so every token still receives
/// its contributions in the per-expert loop's order for any budget.
#[allow(clippy::too_many_arguments)]
fn dispatch_routed_experts_in_blocks<B, F>(
    flat_input: Tensor<B, 2>,
    flat_weights: Tensor<B, 2>,
    selected_experts: Tensor<B, 2, Int>,
    non_finite_tokens: Option<Tensor<B, 1, Bool>>,
    routed_experts: usize,
    precision: RoutedExpertPrecision,
    budget: usize,
    mut run_expert: F,
) -> Result<RoutedExpertDispatch<B>>
where
    B: Backend,
    F: FnMut(usize, Tensor<B, 2>) -> Result<Tensor<B, 2>>,
{
    precision.validate()?;
    let tokens = flat_input.dims()[0];
    let slots = selected_experts.dims()[1];
    let device = flat_input.device();
    let plan = read_assignment_plan(selected_experts, non_finite_tokens, routed_experts)?;
    let assignments = plan.order.len();
    let order = Tensor::<B, 1, Int>::from_data(
        TensorData::new(
            plan.order
                .iter()
                .map(|&index| index as i64)
                .collect::<Vec<_>>(),
            [assignments],
        ),
        &device,
    );
    let flat_weights = flat_weights.reshape([assignments]);
    let expert_token_counts = plan
        .offsets
        .windows(2)
        .map(|range| range[1] - range[0])
        .collect::<Vec<_>>();

    let mut hidden_state: Option<Tensor<B, 2>> = None;
    let mut evaluated_experts = 0;
    let mut first = 0;
    while first < routed_experts {
        let end = assignment::next_dispatch_block(&plan.offsets, first, budget);
        let (block_start, block_rows) =
            (plan.offsets[first], plan.offsets[end] - plan.offsets[first]);
        let experts = first..end;
        first = end;
        if block_rows == 0 {
            continue;
        }
        let block_order = order.clone().narrow(0, block_start, block_rows);
        let token_rows = block_order.clone().div_scalar(slots as i64);
        let inputs = flat_input.clone().select(0, token_rows.clone());
        let weights = flat_weights
            .clone()
            .select(0, block_order)
            .unsqueeze_dim::<2>(1);
        let mut contributions = Vec::with_capacity(experts.len());
        for expert in experts {
            let count = expert_token_counts[expert];
            if count == 0 {
                continue;
            }
            let local = plan.offsets[expert] - block_start;
            let output = run_expert(expert, inputs.clone().narrow(0, local, count))?;
            let weights = weights.clone().narrow(0, local, count);
            contributions.push(weighted_expert_contribution(output, weights, precision));
        }
        evaluated_experts += contributions.len();
        let contributions = Tensor::cat(contributions, 0);
        let accumulator = hidden_state.unwrap_or_else(|| {
            Tensor::zeros(
                [tokens, contributions.dims()[1]],
                (&device, precision.accumulation_dtype),
            )
        });
        hidden_state =
            Some(accumulator.select_assign(0, token_rows, contributions, IndexingUpdateOp::Add));
    }
    // `read_assignment_plan` rejects empty routing, so some block had rows.
    let hidden_state = hidden_state.ok_or_else(|| {
        Error::shape(
            "moe.selected_experts",
            "at least one routed assignment",
            "no assignments",
        )
    })?;
    Ok(RoutedExpertDispatch {
        hidden_state,
        expert_token_counts,
        evaluated_experts,
    })
}

impl<B: Backend> SparseMoeOutput<B> {
    /// Computes Qwen-style differentiable switch-transformer load balancing.
    ///
    /// For routing probabilities `p[t,e]` and top-k assignment indicators
    /// `a[t,j,e]`, the loss is
    /// `E * sum(j,e, mean_t(a[t,j,e]) * mean_t(p[t,e]))`.
    /// An optional semantic mask excludes padding from both means. The selected
    /// expert IDs are treated as discrete decisions while gradients continue
    /// through the full FP32 router probabilities.
    pub fn load_balancing_loss(&self, token_mask: Option<&TokenMask<B>>) -> Result<Tensor<B, 1>> {
        load_balancing_loss(
            self.router_logits.clone(),
            self.selected_experts.clone(),
            token_mask,
        )
    }
}

/// Computes load balancing from router logits and discrete top-k assignments.
///
/// This free form is useful to architecture runtimes that compact ragged rows
/// independently, then reconstruct rectangular diagnostic tensors afterward.
pub fn load_balancing_loss<B: Backend>(
    router_logits: Tensor<B, 3>,
    selected_experts: Tensor<B, 3, Int>,
    token_mask: Option<&TokenMask<B>>,
) -> Result<Tensor<B, 1>> {
    let [batch, sequence, experts] = router_logits.dims();
    let selected = selected_experts.dims()[2];
    if batch == 0 || sequence == 0 || experts == 0 || selected == 0 {
        return Err(Error::shape(
            "moe.router_diagnostics",
            "non-empty [batch, sequence, experts] logits and top-k assignments",
            format!(
                "logits={:?}, selected={:?}",
                router_logits.dims(),
                selected_experts.dims()
            ),
        ));
    }
    if selected_experts.dims()[..2] != [batch, sequence] {
        return Err(Error::shape(
            "moe.selected_experts",
            format!("[{batch}, {sequence}, top_k]"),
            format!("{:?}", selected_experts.dims()),
        ));
    }
    if let Some(mask) = token_mask
        && mask.as_tensor().dims() != [batch, sequence]
    {
        return Err(Error::shape(
            "moe.token_mask",
            format!("[{batch}, {sequence}]"),
            format!("{:?}", mask.as_tensor().dims()),
        ));
    }
    let (minimum, maximum) = expert_id_range(selected_experts.clone())?;
    if minimum < 0 || maximum as usize >= experts {
        return Err(Error::shape(
            "moe.selected_experts",
            format!("expert IDs in 0..{experts}"),
            format!("observed range {minimum}..={maximum}"),
        ));
    }
    let tokens = batch.checked_mul(sequence).ok_or_else(|| {
        Error::shape(
            "moe.router_diagnostics",
            "a batch * sequence product representable by usize",
            format!("{batch} * {sequence}"),
        )
    })?;
    // Indicators and masks are built directly in F32 so a reduced-precision
    // device default cannot introduce a mixed-dtype product.
    let assignments = if experts == 1 {
        // Burn's one-hot needs two classes; with one expert every slot selects it.
        Tensor::<B, 4>::ones(
            [batch, sequence, selected, 1],
            (&router_logits.device(), DType::F32),
        )
    } else {
        selected_experts.one_hot::<4>(experts).cast(FloatDType::F32)
    };
    let probabilities = activation::softmax(router_logits.cast(DType::F32), 2);
    let (assignment_mean, probability_mean) = match token_mask {
        Some(mask) => {
            let valid = mask.as_tensor().clone().cast(FloatDType::F32);
            let denominator = valid.clone().sum().clamp_min(1.0);
            let assignment_mean = (assignments
                * valid
                    .clone()
                    .reshape([batch, sequence, 1, 1])
                    .expand([batch, sequence, selected, experts]))
            .reshape([tokens, selected, experts])
            .sum_dim(0)
                / denominator.clone().reshape([1, 1, 1]);
            let probability_mean = (probabilities
                * valid
                    .reshape([batch, sequence, 1])
                    .expand([batch, sequence, experts]))
            .reshape([tokens, experts])
            .sum_dim(0)
                / denominator.reshape([1, 1]);
            (assignment_mean, probability_mean)
        }
        None => (
            assignments.reshape([tokens, selected, experts]).mean_dim(0),
            probabilities.reshape([tokens, experts]).mean_dim(0),
        ),
    };
    Ok((assignment_mean * probability_mean.unsqueeze_dim::<3>(1)).sum() * experts as f64)
}

/// Reads the smallest and largest expert ID with one device-to-host transfer.
///
/// Concatenating both reductions before the read halves the synchronizations
/// of two separate scalar reads.
fn expert_id_range<B: Backend>(selected_experts: Tensor<B, 3, Int>) -> Result<(i64, i64)> {
    let failure = |error: &dyn core::fmt::Display| {
        Error::BackendCapability(format!("failed to validate routed expert IDs: {error}"))
    };
    let range = Tensor::cat(
        vec![selected_experts.clone().min(), selected_experts.max()],
        0,
    )
    .try_into_data()
    .map_err(|error| failure(&error))?
    .to_vec::<B::IntElem>()
    .map_err(|error| failure(&format!("{error:?}")))?;
    match range.as_slice() {
        [minimum, maximum] => Ok((minimum.elem::<i64>(), maximum.elem::<i64>())),
        other => Err(Error::BackendCapability(format!(
            "expected two expert-ID bounds, read {} values",
            other.len()
        ))),
    }
}

/// Trainable sparse MoE with independent routed and shared expert weights.
#[derive(Module, Debug)]
pub struct SparseMoe<B: Backend> {
    /// Token-to-expert router projection `[hidden, experts]`, biased only when
    /// the routing policy enables `router_bias`.
    pub router: Linear<B>,
    /// Experts invoked only when selected by top-k routing.
    pub routed: Vec<GatedFeedForward<B>>,
    /// Experts evaluated and added for every token.
    pub shared: Vec<GatedFeedForward<B>>,
    /// Number of selected routed experts per token.
    pub experts_per_token: usize,
    /// Whether top-k weights are renormalized.
    pub renormalize_selected: bool,
    /// Multiplicative routed-output scale.
    pub routed_scale: f64,
    /// Explicit route-weight/expert-output rounding policy.
    #[module(skip)]
    pub weighting_precision: RoutedWeightingPrecision,
    /// Residual-stream width used when flattening tokens.
    pub hidden_size: usize,
    /// Scoring, ordering, eligibility, bias, and precision policy of the router.
    #[module(skip)]
    pub routing: RoutingPolicy,
    /// Frozen FP32 `[experts]` bias added to scores for selection only.
    pub correction_bias: Option<Param<Tensor<B, 1>>>,
    /// Optional `[hidden, 1]` projection whose sigmoid scales the shared-expert sum.
    pub shared_expert_gate: Option<Linear<B>>,
}

impl<B: Backend> SparseMoe<B> {
    /// Returns the routing specification of this instantiated module.
    #[must_use]
    pub fn router_config(&self) -> RouterConfig {
        RouterConfig {
            experts: self.routed.len(),
            experts_per_token: self.experts_per_token,
            policy: self.routing,
            renormalize_selected: self.renormalize_selected,
            routed_scale: self.routed_scale,
        }
    }

    /// Evaluates and accumulates only the experts named by top-k assignments.
    ///
    /// Discrete IDs are materialized once and grouped by expert. Input rows and
    /// mixture weights remain device tensors gathered with the uploaded
    /// permutation, so gradients still flow through expert parameters, selected
    /// router probabilities, and the residual input. The caller chooses an
    /// accumulation dtype independently of expert execution, allowing families
    /// such as DeepSeek to sum reduced-precision experts in FP32 while Qwen
    /// retains its residual-stream convention.
    pub(crate) fn dispatch_selected_experts(
        &self,
        flat_input: Tensor<B, 2>,
        flat_weights: Tensor<B, 2>,
        selected_experts: Tensor<B, 2, Int>,
        accumulation_dtype: DType,
    ) -> Result<RoutedExpertDispatch<B>> {
        self.dispatch_selected_experts_with_precision(
            flat_input,
            flat_weights,
            selected_experts,
            None,
            RoutedExpertPrecision::uniform(accumulation_dtype),
        )
    }

    /// Validates dispatch shapes, then dispatches with an explicit multiply/accumulate policy.
    fn dispatch_selected_experts_with_precision(
        &self,
        flat_input: Tensor<B, 2>,
        flat_weights: Tensor<B, 2>,
        selected_experts: Tensor<B, 2, Int>,
        non_finite_tokens: Option<Tensor<B, 1, Bool>>,
        precision: RoutedExpertPrecision,
    ) -> Result<RoutedExpertDispatch<B>> {
        let [tokens, hidden] = flat_input.dims();
        let [weight_tokens, weight_slots] = flat_weights.dims();
        let [selected_tokens, selected_slots] = selected_experts.dims();
        if tokens == 0 || hidden != self.hidden_size {
            return Err(Error::shape(
                "moe.dispatch.input",
                format!("[tokens>0, {}]", self.hidden_size),
                format!("[{tokens}, {hidden}]"),
            ));
        }
        if weight_tokens != tokens
            || selected_tokens != tokens
            || weight_slots != self.experts_per_token
            || selected_slots != self.experts_per_token
        {
            return Err(Error::shape(
                "moe.dispatch.routes",
                format!(
                    "weights and expert IDs [{tokens}, {}]",
                    self.experts_per_token
                ),
                format!(
                    "weights=[{weight_tokens}, {weight_slots}], expert_ids=[{selected_tokens}, {selected_slots}]"
                ),
            ));
        }
        if self.routed.is_empty()
            || self.experts_per_token == 0
            || self.experts_per_token > self.routed.len()
        {
            return Err(Error::configuration(
                "moe.dispatch.experts_per_token",
                format!(
                    "must be in 1..={} for the instantiated expert set",
                    self.routed.len()
                ),
            ));
        }
        precision.validate()?;
        let device = flat_input.device();
        if flat_weights.device() != device || selected_experts.device() != device {
            return Err(Error::BackendCapability(
                "MoE input, selected weights, and selected expert IDs must reside on the same device"
                    .into(),
            ));
        }
        dispatch_routed_experts(
            flat_input,
            flat_weights,
            selected_experts,
            non_finite_tokens,
            self.routed.len(),
            precision,
            |expert, rows| self.routed[expert].forward(rows),
        )
    }

    /// Routes and dispatches `[batch, sequence, hidden]` states sparsely.
    ///
    /// # Errors
    ///
    /// Returns a contextual configuration or shape error before top-k/indexing
    /// kernels run when a manually assembled module violates its routing
    /// invariants or the input is empty or has the wrong hidden width, and a
    /// shape error naming the token when router scores contain NaN.
    pub fn forward(&self, input: Tensor<B, 3>) -> Result<SparseMoeOutput<B>> {
        let [batch, sequence, hidden] = input.dims();
        if batch == 0 || sequence == 0 || hidden != self.hidden_size {
            return Err(Error::shape(
                "moe.input",
                format!("[batch>0, sequence>0, {}]", self.hidden_size),
                format!("[{batch}, {sequence}, {hidden}]"),
            ));
        }
        if self.experts_per_token == 0 || self.experts_per_token > self.routed.len() {
            return Err(Error::configuration(
                "moe.experts_per_token",
                format!(
                    "must be in 1..={} for the instantiated expert set",
                    self.routed.len()
                ),
            ));
        }
        let tokens = batch.checked_mul(sequence).ok_or_else(|| {
            Error::shape(
                "moe.input",
                "a batch * sequence product representable by usize",
                format!("{batch} * {sequence}"),
            )
        })?;
        let residual_dtype = input.dtype();
        let flat_input = input.clone().reshape([tokens, hidden]);
        let router_logits = router_logits(input, &self.router, self.routing.projection)?;
        // Routing ranks and normalizes in FP32: casting before top-k could
        // collapse nearby scores in F16/BF16 and silently select a different
        // expert. The weighting policy below controls the later rounding point.
        let decision = self.router_config().route(
            router_logits.clone().reshape([tokens, self.routed.len()]),
            self.correction_bias.as_ref().map(Param::val),
        )?;
        let (flat_weights, precision) = match self.weighting_precision {
            RoutedWeightingPrecision::Residual => (
                decision.weights.cast(residual_dtype),
                RoutedExpertPrecision::uniform(residual_dtype),
            ),
            RoutedWeightingPrecision::Fp32ThenResidual => (
                decision.weights,
                RoutedExpertPrecision::fp32_weights_round_each(residual_dtype),
            ),
        };
        let dispatch = self.dispatch_selected_experts_with_precision(
            flat_input.clone(),
            flat_weights.clone(),
            decision.experts.clone(),
            Some(decision.non_finite_tokens),
            precision,
        )?;
        let hidden_state = match self.shared_output(flat_input)? {
            Some(shared) => dispatch.hidden_state + shared,
            None => dispatch.hidden_state,
        };
        let k = self.experts_per_token;
        Ok(SparseMoeOutput {
            hidden_state: hidden_state.reshape([batch, sequence, hidden]),
            router_logits,
            selected_experts: decision.experts.reshape([batch, sequence, k]),
            selected_weights: flat_weights.reshape([batch, sequence, k]),
            expert_token_counts: dispatch.expert_token_counts,
            evaluated_experts: dispatch.evaluated_experts,
        })
    }

    /// Sums the always-active shared experts and applies the optional sigmoid gate.
    ///
    /// Returns `None` without allocating when there are no shared experts.
    fn shared_output(&self, flat_input: Tensor<B, 2>) -> Result<Option<Tensor<B, 2>>> {
        let mut experts = self.shared.iter();
        let Some(first) = experts.next() else {
            return Ok(None);
        };
        let mut sum = first.forward(flat_input.clone())?;
        for expert in experts {
            sum = sum + expert.forward(flat_input.clone())?;
        }
        Ok(Some(match &self.shared_expert_gate {
            Some(gate) => sum * activation::sigmoid(operations::linear(flat_input, gate)?),
            None => sum,
        }))
    }
}

#[cfg(test)]
#[path = "moe_tests.rs"]
mod tests;
