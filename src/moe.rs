//! Sparse mixture-of-experts routing and dispatch.
//!
//! A router assigns probabilities to `E` experts and evaluates only the largest
//! `K` assignments for each token. The output is
//! `sum_{j in TopK(p)} w_j Expert_j(x)`, optionally followed by always-active
//! shared experts. Router softmax is evaluated in FP32 because small ranking
//! changes can switch an entire expert computation in lower precision.
//!
//! The portable dispatcher copies only the compact discrete top-k IDs once,
//! groups their `(token, slot)` locations on the host, then uses Burn indexing
//! to preserve differentiable expert inputs and mixture weights. It never
//! evaluates an unselected expert. This is a correctness path; accelerator-
//! specific grouped-GEMM dispatch can replace its grouping mechanics without
//! altering routing or checkpoint parameters.

use burn::{
    module::Module,
    nn::{Linear, LinearConfig, LinearLayout},
    tensor::{
        DType, ElementConversion, IndexingUpdateOp, Int, Tensor, TensorData, activation::softmax,
        backend::Backend,
    },
};

use crate::{
    Error, Result,
    architecture::{Activation, TokenMask},
    feed_forward::{GatedFeedForward, GatedFeedForwardConfig},
    operations,
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
        Ok(())
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
                .with_bias(false)
                .with_layout(LinearLayout::Col),
            device,
        );
        Ok(SparseMoe {
            router,
            routed,
            shared,
            experts_per_token: self.experts_per_token,
            renormalize_selected: self.renormalize_selected,
            routed_scale: self.routed_scale,
            weighting_precision: self.weighting_precision,
            hidden_size: self.hidden_size,
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
struct RoutedExpertPrecision {
    /// Dtype used for the elementwise expert-output/router-weight product.
    multiplication_dtype: DType,
    /// Dtype of the destination and every contribution entering index-add.
    accumulation_dtype: DType,
}

impl RoutedExpertPrecision {
    /// Uses one dtype for multiplication and accumulation.
    const fn uniform(dtype: DType) -> Self {
        Self {
            multiplication_dtype: dtype,
            accumulation_dtype: dtype,
        }
    }

    /// Keeps routing weights FP32 but rounds each contribution before summing.
    const fn fp32_weights_round_each(accumulation_dtype: DType) -> Self {
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
/// gradient graph for either residual states or selected router weights. The
/// single transfer replaces one `argwhere` synchronization per expert. Within
/// each expert, assignments retain token-major, slot-minor order, making
/// grouping deterministic across backends even when router scores tie.
pub(crate) fn group_selected_experts<B: Backend>(
    selected_experts: Tensor<B, 2, Int>,
    routed_experts: usize,
) -> Result<Vec<ExpertAssignmentGroup>> {
    let [tokens, slots] = selected_experts.dims();
    if routed_experts == 0 || tokens == 0 || slots == 0 {
        return Err(Error::shape(
            "moe.selected_experts",
            "[tokens>0, top_k>0] with at least one routed expert",
            format!("shape=[{tokens}, {slots}], routed_experts={routed_experts}"),
        ));
    }
    let assignments = tokens.checked_mul(slots).ok_or_else(|| {
        Error::shape(
            "moe.selected_experts",
            "a tokens * top_k product representable by usize",
            format!("{tokens} * {slots}"),
        )
    })?;
    let values = selected_experts
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
    if values.len() != assignments {
        return Err(Error::shape(
            "moe.selected_experts",
            format!("{assignments} integer routing IDs"),
            format!("{} values", values.len()),
        ));
    }

    let mut groups = (0..routed_experts)
        .map(|_| ExpertAssignmentGroup::default())
        .collect::<Vec<_>>();
    for (weight_index, value) in values.into_iter().enumerate() {
        let expert = value.elem::<i64>();
        if expert < 0 || expert as usize >= routed_experts {
            return Err(Error::shape(
                "moe.selected_experts",
                format!("expert IDs in 0..{routed_experts}"),
                format!("expert ID {expert} at flattened assignment {weight_index}"),
            ));
        }
        let token_index = weight_index / slots;
        let token_index = i64::try_from(token_index).map_err(|_| {
            Error::shape(
                "moe.selected_experts",
                "token indices representable by the backend i64 index type",
                format!("token index {token_index}"),
            )
        })?;
        let weight_index = i64::try_from(weight_index).map_err(|_| {
            Error::shape(
                "moe.selected_experts",
                "assignment indices representable by the backend i64 index type",
                format!("assignment index {weight_index}"),
            )
        })?;
        let group = &mut groups[expert as usize];
        group.token_indices.push(token_index);
        group.weight_indices.push(weight_index);
    }
    Ok(groups)
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
    let minimum = selected_experts
        .clone()
        .min()
        .try_into_scalar()
        .map_err(|error| {
            Error::BackendCapability(format!("failed to validate routed expert IDs: {error}"))
        })?
        .elem::<i64>();
    let maximum = selected_experts
        .clone()
        .max()
        .try_into_scalar()
        .map_err(|error| {
            Error::BackendCapability(format!("failed to validate routed expert IDs: {error}"))
        })?
        .elem::<i64>();
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
    let assignments = selected_experts.one_hot::<4>(experts).float();
    let probabilities = softmax(router_logits.cast(DType::F32), 2);
    let (assignment_mean, probability_mean) = match token_mask {
        Some(mask) => {
            let valid = mask.as_tensor().clone().float();
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

/// Trainable sparse MoE with independent routed and shared expert weights.
#[derive(Module, Debug)]
pub struct SparseMoe<B: Backend> {
    /// Bias-free token-to-expert router.
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
}

impl<B: Backend> SparseMoe<B> {
    /// Evaluates and accumulates only the experts named by top-k assignments.
    ///
    /// Discrete IDs are materialized once by [`group_selected_experts`]. Input
    /// rows and mixture weights remain device tensors selected with the host-
    /// constructed index lists, so gradients still flow through expert
    /// parameters, selected router probabilities, and the residual input. The
    /// caller chooses an accumulation dtype independently of expert execution,
    /// allowing families such as DeepSeek to sum reduced-precision experts in
    /// FP32 while Qwen retains its residual-stream convention.
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
            RoutedExpertPrecision::uniform(accumulation_dtype),
        )
    }

    /// Dispatches selected experts with an explicit multiply/accumulate policy.
    fn dispatch_selected_experts_with_precision(
        &self,
        flat_input: Tensor<B, 2>,
        flat_weights: Tensor<B, 2>,
        selected_experts: Tensor<B, 2, Int>,
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
        let assignments = tokens.checked_mul(self.experts_per_token).ok_or_else(|| {
            Error::shape(
                "moe.dispatch.routes",
                "a tokens * top_k product representable by usize",
                format!("{tokens} * {}", self.experts_per_token),
            )
        })?;
        let groups = group_selected_experts(selected_experts, self.routed.len())?;
        let flat_weights = flat_weights.reshape([assignments]);
        let mut hidden_state =
            Tensor::<B, 2>::zeros([tokens, hidden], (&device, precision.accumulation_dtype));
        let mut expert_token_counts = Vec::with_capacity(self.routed.len());
        let mut evaluated_experts = 0;

        for (expert, group) in self.routed.iter().zip(groups) {
            let assignments = group.token_indices.len();
            expert_token_counts.push(assignments);
            if assignments == 0 {
                continue;
            }
            evaluated_experts += 1;
            let token_indices = Tensor::<B, 1, Int>::from_data(
                TensorData::new(group.token_indices, [assignments]),
                &device,
            );
            let weight_indices = Tensor::<B, 1, Int>::from_data(
                TensorData::new(group.weight_indices, [assignments]),
                &device,
            );
            let weights = flat_weights
                .clone()
                .select(0, weight_indices)
                .unsqueeze_dim::<2>(1);
            let expert_input = flat_input.clone().select(0, token_indices.clone());
            let contribution =
                weighted_expert_contribution(expert.forward(expert_input)?, weights, precision);
            hidden_state =
                hidden_state.select_assign(0, token_indices, contribution, IndexingUpdateOp::Add);
        }

        Ok(RoutedExpertDispatch {
            hidden_state,
            expert_token_counts,
            evaluated_experts,
        })
    }

    /// Routes and dispatches `[batch, sequence, hidden]` states sparsely.
    ///
    /// # Errors
    ///
    /// Returns a contextual configuration or shape error before top-k/indexing
    /// kernels run when a manually assembled module violates its routing
    /// invariants or the input is empty or has the wrong hidden width.
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
        let flat_input = input.clone().reshape([tokens, hidden]);
        let router_logits = operations::linear(input, &self.router)?;
        // Routing ranks the softmax probabilities in FP32. Casting before top-k
        // can collapse nearby scores in F16/BF16 and silently select a different
        // expert. The explicit weighting policy below controls the later rounding
        // boundary without changing this discrete selection invariant.
        let probabilities = softmax(router_logits.clone().cast(DType::F32), 2);
        let (selected_weights, selected_experts) =
            deterministic_top_k(probabilities.clone(), self.experts_per_token);
        let mut flat_weights = selected_weights
            .clone()
            .reshape([tokens, self.experts_per_token]);
        if self.renormalize_selected {
            let denominator = flat_weights.clone().sum_dim(1).clamp_min(f32::MIN_POSITIVE);
            flat_weights = flat_weights / denominator;
        }
        flat_weights = flat_weights * self.routed_scale;
        let precision = match self.weighting_precision {
            RoutedWeightingPrecision::Residual => {
                flat_weights = flat_weights.cast(flat_input.dtype());
                RoutedExpertPrecision::uniform(flat_input.dtype())
            }
            RoutedWeightingPrecision::Fp32ThenResidual => {
                RoutedExpertPrecision::fp32_weights_round_each(flat_input.dtype())
            }
        };
        let flat_experts = selected_experts
            .clone()
            .reshape([tokens, self.experts_per_token]);
        let dispatch = self.dispatch_selected_experts_with_precision(
            flat_input.clone(),
            flat_weights.clone(),
            flat_experts,
            precision,
        )?;

        let mut shared_output =
            Tensor::<B, 2>::zeros([tokens, hidden], (&flat_input.device(), flat_input.dtype()));
        for expert in &self.shared {
            shared_output = shared_output + expert.forward(flat_input.clone())?;
        }
        let selected_weights = flat_weights.reshape([batch, sequence, self.experts_per_token]);
        Ok(SparseMoeOutput {
            hidden_state: (dispatch.hidden_state + shared_output)
                .reshape([batch, sequence, hidden]),
            router_logits,
            selected_experts,
            selected_weights,
            expert_token_counts: dispatch.expert_token_counts,
            evaluated_experts: dispatch.evaluated_experts,
        })
    }
}

/// Selects the largest `k` values with a backend-independent tie rule.
///
/// Burn's native top-k kernels intentionally leave equal-value ordering to the
/// backend. Routing must instead be reproducible across Flex, WGPU, and CUDA,
/// because changing one tied expert changes an entire matrix computation. Each
/// iteration therefore finds the maximum value, resolves its equality set to
/// the smallest expert index, gathers that probability, and masks precisely the
/// chosen index. The complexity is `O(kE)` and does not allocate the `O(E^2)`
/// pairwise-ranking tensor that a lexicographic sort would require.
pub(crate) fn deterministic_top_k<B: Backend>(
    probabilities: Tensor<B, 3>,
    k: usize,
) -> (Tensor<B, 3>, Tensor<B, 3, Int>) {
    let [batch, sequence, experts] = probabilities.dims();
    let device = probabilities.device();
    let expert_ids = Tensor::<B, 1, Int>::arange(0..experts as i64, &device)
        .reshape([1, 1, experts])
        .expand([batch, sequence, experts]);
    let mut remaining = probabilities.clone();
    let mut selected_weights = Vec::with_capacity(k);
    let mut selected_experts = Vec::with_capacity(k);

    for _ in 0..k {
        let maximum = remaining.clone().max_dim(2);
        let tied = remaining.clone().equal(maximum);
        let selected = expert_ids
            .clone()
            .mask_fill(tied.bool_not(), experts as i64)
            .min_dim(2);
        selected_weights.push(probabilities.clone().gather(2, selected.clone()));
        selected_experts.push(selected.clone());
        remaining = remaining.mask_fill(
            expert_ids
                .clone()
                .equal(selected.expand([batch, sequence, experts])),
            f32::NEG_INFINITY,
        );
    }

    (
        Tensor::cat(selected_weights, 2),
        Tensor::cat(selected_experts, 2),
    )
}

#[cfg(test)]
mod tests {
    use burn::{
        backend::Flex,
        module::Param,
        tensor::{DType, Int, Tensor, backend::BackendTypes},
    };

    use super::{
        RoutedExpertPrecision, RoutedWeightingPrecision, SparseMoe, SparseMoeConfig,
        group_selected_experts, load_balancing_loss, weighted_expert_contribution,
    };

    /// Installs zero parameters without advancing Burn's process-wide test RNG.
    fn deterministic_module(
        config: SparseMoeConfig,
        device: &<Flex as BackendTypes>::Device,
    ) -> SparseMoe<Flex> {
        let mut module = config.init::<Flex>(device).expect("valid MoE");
        module.router.weight = Param::from_tensor(Tensor::zeros(
            [config.hidden_size, config.routed_experts],
            device,
        ));
        for expert in module.routed.iter_mut().chain(module.shared.iter_mut()) {
            expert.gate_proj.weight = Param::from_tensor(Tensor::zeros(
                [config.hidden_size, config.intermediate_size],
                device,
            ));
            expert.up_proj.weight = Param::from_tensor(Tensor::zeros(
                [config.hidden_size, config.intermediate_size],
                device,
            ));
            expert.down_proj.weight = Param::from_tensor(Tensor::zeros(
                [config.intermediate_size, config.hidden_size],
                device,
            ));
        }
        module
    }

    /// Invalid top-k routing is rejected before allocating hundreds of experts.
    #[test]
    fn invalid_top_k_is_rejected() {
        let error = SparseMoeConfig::new(4, 8, 2, 3)
            .validate()
            .expect_err("top-k greater than expert count must fail");
        assert!(error.to_string().contains("experts_per_token"));
    }

    /// Serialized configs default legacy payloads to residual-dtype route weighting.
    #[test]
    fn weighting_precision_is_explicit_and_backward_compatible() {
        let mut config = SparseMoeConfig::new(4, 8, 3, 2);
        let mut legacy = serde_json::to_value(&config).unwrap();
        legacy
            .as_object_mut()
            .expect("configuration serializes as an object")
            .remove("weighting_precision");
        let decoded: SparseMoeConfig = serde_json::from_value(legacy).unwrap();
        assert_eq!(
            decoded.weighting_precision,
            RoutedWeightingPrecision::Residual
        );

        config.weighting_precision = RoutedWeightingPrecision::Fp32ThenResidual;
        assert_eq!(
            serde_json::to_value(config).unwrap()["weighting_precision"],
            "fp32_then_residual"
        );
    }

    /// Sparse routing returns stable public shapes for diagnostics and hidden states.
    #[test]
    fn sparse_moe_preserves_shapes() {
        let device = Default::default();
        let module = deterministic_module(SparseMoeConfig::new(4, 8, 3, 2), &device);
        let output = module
            .forward(Tensor::<Flex, 3>::zeros([2, 3, 4], &device))
            .expect("valid input must dispatch");
        assert_eq!(output.hidden_state.dims(), [2, 3, 4]);
        assert_eq!(output.router_logits.dims(), [2, 3, 3]);
        assert_eq!(output.selected_experts.dims(), [2, 3, 2]);
        assert_eq!(output.selected_weights.dims(), [2, 3, 2]);
    }

    /// Renormalized selected probabilities sum to the routed scale for every token.
    #[test]
    fn selected_weights_are_renormalized() {
        let device = Default::default();
        let mut config = SparseMoeConfig::new(4, 8, 4, 2);
        config.routed_scale = 2.5;
        let module = deterministic_module(config, &device);
        let output = module
            .forward(Tensor::<Flex, 3>::ones([1, 2, 4], &device))
            .expect("valid input must dispatch");
        let sums = output.selected_weights.sum_dim(2).to_data();
        let values = sums
            .as_slice::<f32>()
            .expect("Flex f32 data should expose an f32 slice");
        assert!(values.iter().all(|value| (*value - 2.5).abs() < 1e-5));
    }

    /// Disabling top-k renormalization preserves each selected global softmax probability.
    #[test]
    fn non_normalized_weights_preserve_global_softmax_mass() {
        let device = Default::default();
        let mut config = SparseMoeConfig::new(4, 8, 4, 2);
        config.renormalize_selected = false;
        let module = deterministic_module(config, &device);
        let output = module
            .forward(Tensor::<Flex, 3>::zeros([1, 2, 4], &device))
            .expect("tied logits should route successfully");
        let sums = output.selected_weights.sum_dim(2).to_data();
        let values = sums
            .as_slice::<f32>()
            .expect("Flex f32 data should expose an f32 slice");
        assert!(values.iter().all(|value| (*value - 0.5).abs() < 1e-5));
    }

    /// Invalid shapes fail through the crate error taxonomy instead of tensor panics.
    #[test]
    fn invalid_runtime_shape_is_rejected() {
        let device = Default::default();
        let module = SparseMoeConfig::new(4, 8, 3, 2)
            .init::<Flex>(&device)
            .expect("valid MoE should initialize");
        let error = module
            .forward(Tensor::<Flex, 3>::zeros([2, 3, 5], &device))
            .expect_err("wrong hidden width must fail");
        assert!(error.to_string().contains("moe.input"));
    }

    /// Dispatcher metadata is validated before casts or unchecked index kernels.
    #[test]
    fn invalid_dispatch_metadata_is_rejected() {
        let device = Default::default();
        let module = deterministic_module(SparseMoeConfig::new(4, 8, 3, 2), &device);
        let error = module
            .dispatch_selected_experts(
                Tensor::zeros([2, 4], &device),
                Tensor::ones([2, 1], &device),
                Tensor::<Flex, 2, Int>::from_ints([[0, 1], [1, 2]], &device),
                DType::F32,
            )
            .expect_err("route weights with the wrong top-k width must fail");
        assert!(error.to_string().contains("moe.dispatch.routes"));

        let error = module
            .dispatch_selected_experts(
                Tensor::zeros([1, 4], &device),
                Tensor::ones([1, 2], &device),
                Tensor::<Flex, 2, Int>::from_ints([[0, 1]], &device),
                DType::I32,
            )
            .expect_err("integer accumulation must fail before a tensor cast");
        assert!(error.to_string().contains("moe.dispatch.precision"));
    }

    /// Qwen retains an FP32 route weight until after multiplication by an F16 expert output.
    #[test]
    fn qwen_f16_route_weight_rounds_after_multiplication() {
        let device = Default::default();
        let expert_output = Tensor::<Flex, 2>::from_floats([[1.5]], &device).cast(DType::F16);
        let fp32_weight = Tensor::<Flex, 2>::from_floats([[0.1]], &device);

        let actual = weighted_expert_contribution(
            expert_output.clone(),
            fp32_weight.clone(),
            RoutedExpertPrecision::fp32_weights_round_each(DType::F16),
        );
        let upstream =
            (expert_output.clone().cast(DType::F32) * fp32_weight.clone()).cast(DType::F16);
        let weight_rounded_too_early = expert_output * fp32_weight.cast(DType::F16);

        assert_eq!(actual.dtype(), DType::F16);
        assert_eq!(actual.to_data(), upstream.to_data());
        assert_ne!(actual.to_data(), weight_rounded_too_early.to_data());
        assert_eq!(
            actual.cast(DType::F32).into_data().to_vec::<f32>().unwrap(),
            [0.150_024_41]
        );
    }

    /// Equal router scores select lower expert IDs first on every backend.
    #[test]
    fn routing_ties_are_lexicographically_deterministic() {
        let device = Default::default();
        let module = deterministic_module(SparseMoeConfig::new(4, 8, 4, 2), &device);
        let output = module
            .forward(Tensor::<Flex, 3>::zeros([1, 3, 4], &device))
            .expect("zero inputs produce tied router logits");
        assert_eq!(
            output.selected_experts.into_data(),
            Tensor::<Flex, 3, Int>::from_ints([[[0, 1], [0, 1], [0, 1]]], &device).into_data()
        );
    }

    /// Host grouping retains token-major order and represents empty experts.
    #[test]
    fn selected_expert_grouping_is_stable_and_complete() {
        let device = Default::default();
        let selected = Tensor::<Flex, 2, Int>::from_ints([[0, 2], [2, 0], [0, 2]], &device);
        let groups = group_selected_experts(selected, 4).expect("valid routing IDs");

        assert_eq!(groups[0].token_indices, [0, 1, 2]);
        assert_eq!(groups[0].weight_indices, [0, 3, 4]);
        assert!(groups[1].token_indices.is_empty());
        assert_eq!(groups[2].token_indices, [0, 1, 2]);
        assert_eq!(groups[2].weight_indices, [1, 2, 5]);
        assert!(groups[3].token_indices.is_empty());
    }

    /// Invalid host-materialized IDs fail before unchecked backend indexing.
    #[test]
    fn selected_expert_grouping_rejects_out_of_range_ids() {
        let device = Default::default();
        let selected = Tensor::<Flex, 2, Int>::from_ints([[0, -1], [1, 3]], &device);
        let error = group_selected_experts(selected, 3)
            .expect_err("negative and upper-bound expert IDs must fail");
        assert!(error.to_string().contains("expert IDs in 0..3"));
        assert!(error.to_string().contains("flattened assignment 1"));
    }

    /// The dispatcher invokes only experts that own at least one assignment.
    #[test]
    fn dispatch_counters_prove_unselected_experts_are_skipped() {
        let device = Default::default();
        let module = deterministic_module(SparseMoeConfig::new(4, 8, 4, 2), &device);
        let output = module
            .forward(Tensor::<Flex, 3>::zeros([2, 3, 4], &device))
            .expect("tied routing");
        assert_eq!(output.expert_token_counts, vec![6, 6, 0, 0]);
        assert_eq!(output.evaluated_experts, 2);
        assert_eq!(output.expert_token_counts.iter().sum::<usize>(), 12);
    }

    /// Host grouping preserves gradients while omitting unselected expert graphs.
    #[cfg(feature = "autodiff")]
    #[test]
    fn selected_dispatch_retains_input_and_expert_gradients() {
        use burn::backend::Autodiff;

        type AdBackend = Autodiff<Flex>;
        let device = Default::default();
        let mut config = SparseMoeConfig::new(4, 8, 2, 1);
        config.renormalize_selected = false;
        let mut module = config.init::<AdBackend>(&device).expect("valid MoE");
        module.router.weight = Param::from_tensor(Tensor::zeros([4, 2], &device));
        for expert in &mut module.routed {
            expert.gate_proj.weight = Param::from_tensor(Tensor::ones([4, 8], &device));
            expert.up_proj.weight = Param::from_tensor(Tensor::ones([4, 8], &device));
            expert.down_proj.weight = Param::from_tensor(Tensor::ones([8, 4], &device));
        }
        let input = Tensor::<AdBackend, 3>::ones([1, 2, 4], &device).require_grad();
        let tracked_input = input.clone();

        let output = module
            .forward(input)
            .expect("tied routing should select expert zero");
        assert_eq!(output.expert_token_counts, [2, 0]);
        let gradients = output.hidden_state.sum().backward();

        assert!(tracked_input.grad(&gradients).is_some());
        assert!(module.routed[0].gate_proj.weight.grad(&gradients).is_some());
        assert!(module.routed[0].up_proj.weight.grad(&gradients).is_some());
        assert!(module.routed[0].down_proj.weight.grad(&gradients).is_some());
        assert!(module.routed[1].gate_proj.weight.grad(&gradients).is_none());
    }

    /// Load-balancing diagnostics reject mismatched masks before broadcasting.
    #[test]
    fn auxiliary_loss_validates_mask_shape() {
        use crate::architecture::TokenMask;
        use burn::tensor::Bool;

        let device = Default::default();
        let module = deterministic_module(SparseMoeConfig::new(4, 8, 4, 2), &device);
        let output = module
            .forward(Tensor::<Flex, 3>::zeros([1, 2, 4], &device))
            .unwrap();
        let mask = TokenMask::new(Tensor::<Flex, 2, Bool>::from_bool(
            [[true, true, true]].into(),
            &device,
        ));
        let error = output
            .load_balancing_loss(Some(&mask))
            .expect_err("wrong mask width must fail");
        assert!(error.to_string().contains("moe.token_mask"));
    }

    /// Public auxiliary-loss inputs reject invalid expert IDs before one-hot encoding.
    #[test]
    fn auxiliary_loss_validates_selected_expert_range() {
        let device = Default::default();
        let logits = Tensor::<Flex, 3>::zeros([1, 1, 4], &device);
        let selected = Tensor::<Flex, 3, Int>::from_ints([[[0, 4]]], &device);
        let error = load_balancing_loss(logits, selected, None)
            .expect_err("an out-of-range expert ID must fail cleanly");
        assert!(error.to_string().contains("expert IDs in 0..4"));
    }

    /// Rejects every zero cardinality and invalid routed-output scaling value.
    #[test]
    fn config_validates_dimensions_top_k_and_finite_positive_scale() {
        for field in ["hidden", "intermediate", "experts", "top_k"] {
            let mut config = SparseMoeConfig::new(2, 2, 2, 1);
            match field {
                "hidden" => config.hidden_size = 0,
                "intermediate" => config.intermediate_size = 0,
                "experts" => config.routed_experts = 0,
                "top_k" => config.experts_per_token = 0,
                _ => unreachable!(),
            }
            assert!(matches!(
                config.validate(),
                Err(crate::Error::Configuration { .. })
            ));
        }
        for scale in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut config = SparseMoeConfig::new(2, 2, 2, 1);
            config.routed_scale = scale;
            assert!(
                matches!(config.validate(), Err(crate::Error::Configuration { field, .. }) if field == "moe.routed_scale")
            );
        }
        let device = Default::default();
        assert!(
            SparseMoeConfig::new(usize::MAX, 2, 1, 1)
                .init::<Flex>(&device)
                .is_err()
        );
        assert!(
            matches!(SparseMoeConfig::new(2, 1, usize::MAX, 1).validate(),
            Err(crate::Error::Configuration { field, reason })
            if field == "moe.projection_shape" && reason.contains("router"))
        );
    }

    /// Empty routing axes and upper-bound IDs fail before host dispatch indexing.
    #[test]
    fn grouping_rejects_empty_metadata_and_positive_out_of_range_ids() {
        let device = Default::default();
        for (tokens, slots, experts) in [(0, 1, 2), (1, 0, 2), (1, 1, 0)] {
            let ids = Tensor::<Flex, 2, Int>::zeros([tokens, slots], &device);
            assert!(
                matches!(group_selected_experts(ids, experts), Err(crate::Error::Shape { tensor, .. }) if tensor == "moe.selected_experts")
            );
        }
        let ids = Tensor::<Flex, 2, Int>::from_ints([[0, 2]], &device);
        let error = group_selected_experts(ids, 2).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("expert ID 2 at flattened assignment 1")
        );
    }

    /// Dispatcher input, route axes, cardinalities, and precision fail separately.
    #[test]
    fn dispatcher_validates_each_independent_shape_and_expert_boundary() {
        let device = Default::default();
        let module = deterministic_module(SparseMoeConfig::new(2, 2, 2, 1), &device);
        for (input, weights, ids, field) in [
            ([0, 2], [0, 1], [0, 1], "moe.dispatch.input"),
            ([2, 3], [2, 1], [2, 1], "moe.dispatch.input"),
            ([2, 2], [1, 1], [2, 1], "moe.dispatch.routes"),
            ([2, 2], [2, 1], [1, 1], "moe.dispatch.routes"),
            ([2, 2], [2, 1], [2, 2], "moe.dispatch.routes"),
        ] {
            let error = module
                .dispatch_selected_experts(
                    Tensor::zeros(input, &device),
                    Tensor::zeros(weights, &device),
                    Tensor::zeros(ids, &device),
                    DType::F32,
                )
                .unwrap_err();
            assert!(matches!(error, crate::Error::Shape { tensor, .. } if tensor == field));
        }
        for top_k in [0, 3] {
            let mut changed = module.clone();
            changed.experts_per_token = top_k;
            let error = changed
                .dispatch_selected_experts(
                    Tensor::zeros([1, 2], &device),
                    Tensor::zeros([1, top_k], &device),
                    Tensor::zeros([1, top_k], &device),
                    DType::F32,
                )
                .unwrap_err();
            assert!(
                matches!(error, crate::Error::Configuration { field, .. } if field == "moe.dispatch.experts_per_token")
            );
        }
        let mut empty = module.clone();
        empty.routed.clear();
        assert!(
            matches!(empty.dispatch_selected_experts(Tensor::zeros([1, 2], &device),
            Tensor::zeros([1, 1], &device), Tensor::zeros([1, 1], &device), DType::F32),
            Err(crate::Error::Configuration { field, .. }) if field == "moe.dispatch.experts_per_token")
        );
        for (multiplication_dtype, accumulation_dtype) in
            [(DType::I32, DType::F32), (DType::F32, DType::I32)]
        {
            let precision = RoutedExpertPrecision {
                multiplication_dtype,
                accumulation_dtype,
            };
            assert!(
                matches!(precision.validate(), Err(crate::Error::Configuration { field, .. }) if field == "moe.dispatch.precision")
            );
        }
    }

    /// Manually edited modules still reject invalid routing before top-k executes.
    #[test]
    fn forward_rejects_empty_inputs_and_mutated_expert_cardinalities() {
        let device = Default::default();
        let module = deterministic_module(SparseMoeConfig::new(2, 2, 2, 1), &device);
        for shape in [[0, 1, 2], [1, 0, 2]] {
            assert!(matches!(module.forward(Tensor::zeros(shape, &device)),
                Err(crate::Error::Shape { tensor, .. }) if tensor == "moe.input"));
        }
        for top_k in [0, 3] {
            let mut changed = module.clone();
            changed.experts_per_token = top_k;
            assert!(matches!(changed.forward(Tensor::zeros([1, 1, 2], &device)),
                Err(crate::Error::Configuration { field, .. }) if field == "moe.experts_per_token"));
        }
    }

    /// Biased shared experts add independently of routed scaling and sparse choice.
    #[test]
    fn shared_experts_add_exact_values_while_unselected_nan_experts_are_skipped() {
        let device = Default::default();
        let mut config = SparseMoeConfig::new(1, 1, 3, 1);
        config.shared_experts = 2;
        config.expert_bias = true;
        config.activation = crate::architecture::Activation::Relu;
        config.routed_scale = 2.0;
        config.weighting_precision = RoutedWeightingPrecision::Fp32ThenResidual;
        let mut module = deterministic_module(config, &device);
        for (expert, coefficient) in module
            .routed
            .iter_mut()
            .chain(module.shared.iter_mut())
            .zip([1.0, 1.0, 1.0, 3.0, 5.0])
        {
            expert.gate_proj.weight = Param::from_tensor(Tensor::ones([1, 1], &device));
            expert.up_proj.weight = Param::from_tensor(Tensor::ones([1, 1], &device));
            expert.down_proj.weight =
                Param::from_tensor(Tensor::from_floats([[coefficient]], &device));
            expert.gate_proj.bias = Some(Param::from_tensor(Tensor::ones([1], &device)));
            expert.up_proj.bias = Some(Param::from_tensor(Tensor::zeros([1], &device)));
            expert.down_proj.bias = Some(Param::from_tensor(Tensor::zeros([1], &device)));
        }
        module.routed[1].up_proj.weight =
            Param::from_tensor(Tensor::from_floats([[f32::NAN]], &device));
        module.routed[2].up_proj.weight =
            Param::from_tensor(Tensor::from_floats([[f32::NAN]], &device));
        let output = module
            .forward(Tensor::from_floats([[[1.0], [2.0]]], &device))
            .unwrap();
        // Each live expert is c * ReLU(x+1) * x: (2*1 + 3 + 5) * x*(x+1).
        assert_eq!(
            output.hidden_state.into_data().to_vec::<f32>().unwrap(),
            [20.0, 60.0]
        );
        assert_eq!(
            output.selected_weights.into_data().to_vec::<f32>().unwrap(),
            [2.0, 2.0]
        );
        assert_eq!(output.expert_token_counts, [2, 0, 0]);
        assert_eq!(output.evaluated_experts, 1);
    }

    /// Selected/shared expert parameter errors remain contextual instead of panicking.
    #[test]
    fn malformed_selected_and_shared_experts_propagate_linear_shape_errors() {
        let device = Default::default();
        for shared in [false, true] {
            let mut config = SparseMoeConfig::new(2, 2, 2, 1);
            config.shared_experts = usize::from(shared);
            let mut module = deterministic_module(config, &device);
            let expert = if shared {
                &mut module.shared[0]
            } else {
                &mut module.routed[0]
            };
            expert.up_proj.weight = Param::from_tensor(Tensor::zeros([3, 2], &device));
            assert!(matches!(
                module.forward(Tensor::ones([1, 1, 2], &device)),
                Err(crate::Error::Shape { .. })
            ));
        }
    }

    /// Balancing loss obeys the masked means, including the all-padding zero case.
    #[test]
    fn load_balancing_loss_matches_two_expert_probability_oracle() {
        use crate::architecture::TokenMask;
        use burn::tensor::Bool;

        let device = Default::default();
        let logits =
            Tensor::<Flex, 3>::from_floats([[[3.0_f32.ln(), 0.0]], [[0.0, 3.0_f32.ln()]]], &device);
        let selected = Tensor::<Flex, 3, Int>::from_ints([[[0]], [[1]]], &device);
        let unmasked = load_balancing_loss(logits.clone(), selected.clone(), None)
            .unwrap()
            .into_scalar();
        assert!((unmasked - 1.0).abs() < 1e-6);
        for (mask, expected) in [([[true], [false]], 1.5), ([[false], [false]], 0.0)] {
            let mask = TokenMask::new(Tensor::<Flex, 2, Bool>::from_bool(mask.into(), &device));
            let loss = load_balancing_loss(logits.clone(), selected.clone(), Some(&mask))
                .unwrap()
                .into_scalar();
            assert!((loss - expected).abs() < 1e-6);
        }
        let all_experts = Tensor::<Flex, 3, Int>::from_ints([[[0, 1]], [[1, 0]]], &device);
        let loss = load_balancing_loss(logits, all_experts, None)
            .unwrap()
            .into_scalar();
        assert!((loss - 2.0).abs() < 1e-6);
    }

    /// Diagnostics reject empty axes, unequal token axes, and negative IDs.
    #[test]
    fn load_balancing_loss_validates_all_external_diagnostic_axes() {
        let device = Default::default();
        for (logits, ids, field) in [
            ([0, 1, 2], [0, 1, 1], "moe.router_diagnostics"),
            ([1, 0, 2], [1, 0, 1], "moe.router_diagnostics"),
            ([1, 1, 0], [1, 1, 1], "moe.router_diagnostics"),
            ([1, 1, 2], [1, 1, 0], "moe.router_diagnostics"),
            ([1, 2, 2], [2, 2, 1], "moe.selected_experts"),
            ([1, 2, 2], [1, 1, 1], "moe.selected_experts"),
        ] {
            assert!(
                matches!(load_balancing_loss(Tensor::<Flex, 3>::zeros(logits, &device),
                Tensor::<Flex, 3, Int>::zeros(ids, &device), None),
                Err(crate::Error::Shape { tensor, .. }) if tensor == field)
            );
        }
        assert!(
            matches!(load_balancing_loss(Tensor::<Flex, 3>::zeros([1, 1, 2], &device),
            Tensor::<Flex, 3, Int>::from_ints([[[-1]]], &device), None),
            Err(crate::Error::Shape { tensor, actual, .. }) if tensor == "moe.selected_experts" && actual.contains("-1"))
        );
    }
}
