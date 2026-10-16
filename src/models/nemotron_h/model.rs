//! Executable NVIDIA Nemotron-H hybrid causal decoder.
//!
//! A Nemotron-H schedule is heterogeneous at residual-block granularity. Every
//! block computes `x + Mixer(RMSNorm(x))`, but `Mixer` may be Mamba2, causal
//! grouped-query attention, a dense ReLU² MLP, or a sparse ReLU² MoE. Unlike a
//! conventional transformer layer there is no mandatory attention/FFN pair.
//!
//! The released 3.5 Lightning checkpoint uses 52 blocks, alternating Mamba2 and
//! top-6/128 MoE blocks with periodic GQA. Its MoE router scores experts with a
//! sigmoid, applies a learned correction only to discrete selection, and uses
//! the uncorrected score as the mixture weight. A shared expert always executes;
//! routed experts execute only for tokens that selected them.
//!
//! Parameters never own generation state. [`NemotronHDecodeState`] combines a
//! native KV bank for attention blocks and recurrent matrix/convolution banks for
//! Mamba2 blocks. Stateless paths stay differentiable for training v2.
//!
//! # Submodules
//!
//! - `decode`: batched stateful prefill/decode, one model pass per call over
//!   a right-padded compact batch (`batch` realizes, and `plan` plans and
//!   verifies, its index maps);
//! - `mtp` and `speculative`: the shifted MTP stage and lossless speculation
//!   that verifies a whole draft in one target pass (`cycle` verifies their
//!   extent and alignment checks);
//! - `hooks`: activation points, [`HookedModel`](crate::architecture::HookedModel),
//!   and [`StagedModel`](crate::architecture::StagedModel) for experiments;
//! - `mamba2`: the family's names for the shared [`crate::mamba2`] mixer.

use burn::{
    module::{Module, Param},
    nn::{Embedding, EmbeddingConfig, Linear, LinearConfig, LinearLayout},
    tensor::{Bool, DType, Int, Tensor, activation, backend::Backend},
};

use crate::{
    ArchitectureConfig, CausalLmInput, CausalLmOutput, Error, ForwardOptions, Result,
    architecture::{Activation, RouterDiagnostics, TokenMask},
    attention::{GroupedQueryAttention, GroupedQueryAttentionConfig, padding_attention_mask},
    experiment::AttentionPolicy,
    feed_forward::{DenseFeedForward, DenseFeedForwardConfig},
    intervention::{Hooks, Layer, Point, Stack},
    moe::{
        ExpertSelection, GroupLimitedSelection, GroupScore, PrunedExpertScore,
        RenormalizationDenominator, RoutedExpertPrecision, RouterConfig, RouterProjectionPrecision,
        RouterScoring, RoutingPolicy, ScoringOrder, deterministic_top_k, dispatch_routed_experts,
        router_logits,
    },
    normalization::{RmsNorm, RmsNormConfig, RmsNormRootMode},
    operations,
};

use super::{NemotronHBlockType, NemotronHConfig};
use crate::models::shared::{
    ImplicitPositions, InputContract, InputNames, select_hidden, validate_input,
    validate_token_values,
};

mod batch;
mod cycle;
mod decode;
mod hooks;
pub mod mamba2;
mod mtp;
mod plan;
mod speculative;

pub use decode::NemotronHDecodeState;
use mamba2::{
    NemotronMamba2, NemotronMamba2Config, NemotronMamba2Mode, TimeStepClamp, TimeStepLimits,
};
pub use mtp::{NemotronHMtpPrediction, NemotronHSpeculativePrefill};
pub use speculative::{
    NemotronHSpeculativeConfig, NemotronHSpeculativeDecoder, NemotronHSpeculativeOutput,
};

/// Tensor names reported by Nemotron-H input errors.
const INPUT_NAMES: InputNames<'static> = InputNames {
    token_ids: "nemotron_h.input_ids",
    token_mask: "nemotron_h.token_mask",
    position_ids: "nemotron_h.position_ids",
};

/// Owner named when a stateless tensor is on another device than the weights.
const PARAMETERS: &str = "Nemotron-H model parameters";

/// Ordinary two-projection MLP used by dense, routed, and shared blocks.
///
/// Nemotron-H's MLPs are ungated `down(phi(up(x)))` with `phi = ReLU²` for the
/// released checkpoints (SiLU remains accepted for legacy configs), exactly the
/// crate's shared [`DenseFeedForward`]. The alias keeps the family name
/// stable for loaders and downstream code.
pub type NemotronHDense<B> = DenseFeedForward<B>;

/// Initializes one checkpoint-native `[out, in]` Nemotron-H MLP.
///
/// Widths are validated by [`DenseFeedForwardConfig::init`], which reports a
/// contextual error for zero extents before allocating.
fn dense_mlp<B: Backend>(
    hidden_size: usize,
    intermediate_size: usize,
    activation: Activation,
    bias: bool,
    device: &B::Device,
) -> Result<NemotronHDense<B>> {
    DenseFeedForwardConfig::new(hidden_size, intermediate_size)
        .with_activation(activation)
        .with_bias(bias)
        .init(device)
}

/// Sparse routing output retained for requested diagnostics and tests.
#[derive(Clone, Debug)]
pub struct NemotronHMoeOutput<B: Backend> {
    /// Sum of routed and always-active shared expert outputs.
    pub hidden_state: Tensor<B, 3>,
    /// FP32 pre-sigmoid router logits `[batch, sequence, experts]`.
    pub router_logits: Tensor<B, 3>,
    /// Chosen expert IDs `[batch, sequence, top_k]`.
    pub selected_experts: Tensor<B, 3, Int>,
    /// Uncorrected sigmoid mixture weights after normalization/scaling.
    pub selected_weights: Tensor<B, 3>,
    /// Assignment count for every routed expert.
    pub expert_token_counts: Vec<usize>,
    /// Number of routed expert graphs actually evaluated.
    pub evaluated_experts: usize,
}

impl<B: Backend> NemotronHMoeOutput<B> {
    /// Computes NVIDIA's differentiable sigmoid-router load-balancing loss.
    ///
    /// This is Megatron-LM's switch loss for sigmoid routers, without its
    /// coefficient: with per-token normalized scores
    /// `s = sigmoid(z) / (sum sigmoid(z) + 1e-20)` and assignments `m` given by
    /// the top-k of `s` (uncorrected, ignoring groups), the loss is
    /// `E / (K T^2) * sum_e (sum_t s[t,e]) (sum_t m[t,e])`, exactly one under
    /// perfect balance. `token_mask` (`true == valid`) excludes padding from
    /// every sum and from `T`.
    ///
    /// # Errors
    ///
    /// Returns a shape error when the mask does not match the routed grid.
    pub fn load_balancing_loss(&self, token_mask: Option<&TokenMask<B>>) -> Result<Tensor<B, 1>> {
        nemotron_load_balancing_loss(
            self.router_logits.clone(),
            self.selected_experts.dims()[2],
            token_mask,
        )
    }
}

/// Additive guard of Megatron's per-token sigmoid-score normalization.
const AUXILIARY_SCORE_EPSILON: f64 = 1e-20;

/// Computes Megatron-LM's switch load-balancing loss for a sigmoid router.
///
/// NVIDIA trains Nemotron-H with Megatron-LM, whose sigmoid routers feed the
/// auxiliary loss *normalized* scores (`compute_routing_scores_for_aux_loss`
/// in `megatron/core/transformer/moe/moe_utils.py`):
///
/// ```text
/// s[t,e] = sigmoid(z[t,e]) / (sum_e' sigmoid(z[t,e']) + 1e-20)
/// m[t,e] = 1 if e is among the top-k of s[t,:] else 0
/// loss   = E / (K * T^2) * sum_e (sum_t s[t,e]) * (sum_t m[t,e])
/// ```
///
/// which is `switch_load_balancing_loss_func` without its coefficient (the
/// trainer multiplies by `moe_aux_loss_coeff`). Perfectly balanced routing
/// gives exactly one, so NVIDIA's published coefficient keeps its meaning.
/// Like Megatron, the assignment map `m` is recomputed from the uncorrected,
/// group-unrestricted scores rather than taken from the correction-biased
/// routing decision; ties select the smallest expert ID. Sums run over the
/// `T` valid tokens of the whole batch (Megatron's global `aux_loss`, not its
/// per-sequence variant), all in F32. Gradients flow through `s` only; `m` is
/// a discrete decision.
///
/// # Errors
///
/// Returns a shape error for empty logits, an invalid `experts_per_token`, or a
/// token mask that does not match `[batch, sequence]`.
pub(super) fn nemotron_load_balancing_loss<B: Backend>(
    router_logits: Tensor<B, 3>,
    experts_per_token: usize,
    token_mask: Option<&TokenMask<B>>,
) -> Result<Tensor<B, 1>> {
    let [batch, sequence, experts] = router_logits.dims();
    if batch == 0
        || sequence == 0
        || experts == 0
        || experts_per_token == 0
        || experts_per_token > experts
    {
        return Err(Error::shape(
            "nemotron_h.moe.router_logits",
            format!("non-empty [batch, sequence, experts >= {experts_per_token} >= 1]"),
            format!("{:?} with top-k {experts_per_token}", router_logits.dims()),
        ));
    }
    let tokens = batch.checked_mul(sequence).ok_or_else(|| {
        Error::configuration("nemotron_h.moe", "batch * sequence overflows usize")
    })?;
    let device = router_logits.device();
    let valid = match token_mask {
        Some(mask) if mask.as_tensor().dims() != [batch, sequence] => {
            return Err(Error::shape(
                "nemotron_h.moe.token_mask",
                format!("[{batch}, {sequence}]"),
                format!("{:?}", mask.as_tensor().dims()),
            ));
        }
        Some(mask) => mask.as_tensor().clone().float().cast(DType::F32),
        None => Tensor::<B, 2>::ones([batch, sequence], (&device, DType::F32)),
    }
    .reshape([tokens, 1]);
    let scores = activation::sigmoid(router_logits.cast(DType::F32).reshape([tokens, experts]));
    let scores = scores.clone() / (scores.sum_dim(1) + AUXILIARY_SCORE_EPSILON);
    let assignments = if experts == 1 {
        // Burn's one-hot needs two classes; one expert receives every token.
        Tensor::<B, 2>::ones([tokens, 1], (&device, DType::F32))
    } else {
        deterministic_top_k(
            scores.clone().detach().unsqueeze_dim::<3>(0),
            experts_per_token,
        )
        .1
        .reshape([tokens, experts_per_token])
        .one_hot::<3>(experts)
        .sum_dim(1)
        .reshape([tokens, experts])
        .float()
        .cast(DType::F32)
    };
    let valid_tokens = valid.clone().sum().clamp_min(1.0);
    let tokens_per_expert = (assignments * valid.clone()).sum_dim(0);
    let score_per_expert = (scores * valid).sum_dim(0);
    let scale = experts as f64 / experts_per_token as f64;
    Ok((score_per_expert * tokens_per_expert).sum() * scale / valid_tokens.square())
}

/// Nemotron's correction-biased sigmoid router plus shared ReLU² expert.
#[derive(Module, Debug)]
pub struct NemotronHMoe<B: Backend> {
    /// Bias-free token-to-expert projection.
    pub router: Linear<B>,
    /// FP32 score correction used only for top-k selection.
    pub correction_bias: Param<Tensor<B, 1>>,
    /// Independently dispatched routed experts.
    pub routed: Vec<NemotronHDense<B>>,
    /// Always-evaluated shared expert.
    pub shared: NemotronHDense<B>,
    /// Optional projection into a lower-dimensional routed-expert space.
    pub fc1_latent_proj: Option<Linear<B>>,
    /// Optional projection from routed latent space to the residual width.
    pub fc2_latent_proj: Option<Linear<B>>,
    /// Experts selected per token.
    pub experts_per_token: usize,
    /// Number of equal expert groups.
    pub groups: usize,
    /// Groups retained before expert top-k.
    pub top_groups: usize,
    /// Whether selected sigmoid scores sum to one before scaling.
    pub renormalize_selected: bool,
    /// Multiplicative routed contribution scale.
    pub routed_scale: f64,
    /// Residual-stream width.
    pub hidden_size: usize,
}

impl<B: Backend> NemotronHMoe<B> {
    /// Initializes exact routed/shared expert shapes from family configuration.
    fn init(config: &NemotronHConfig, device: &B::Device) -> Result<Self> {
        let activation = activation_from_name(&config.mlp_hidden_act)?;
        let latent = config.moe_latent_size.unwrap_or(config.hidden_size);
        let routed = (0..config.n_routed_experts)
            .map(|_| {
                dense_mlp(
                    latent,
                    config.moe_intermediate_size,
                    activation,
                    config.mlp_bias,
                    device,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let projection = |input, output| {
            operations::init_linear(
                &LinearConfig::new(input, output)
                    .with_bias(config.mlp_bias)
                    .with_layout(LinearLayout::Col),
                device,
            )
        };
        Ok(Self {
            router: operations::init_linear(
                &LinearConfig::new(config.hidden_size, config.n_routed_experts)
                    .with_bias(false)
                    .with_layout(LinearLayout::Col),
                device,
            ),
            // NVIDIA declares this as an FP32 buffer: it changes the discrete
            // top-k decision but is not optimized by back-propagation.
            correction_bias: Param::from_tensor(Tensor::zeros(
                [config.n_routed_experts],
                (device, DType::F32),
            ))
            .set_require_grad(false),
            routed,
            shared: dense_mlp(
                config.hidden_size,
                config.moe_shared_expert_intermediate_size,
                activation,
                config.mlp_bias,
                device,
            )?,
            fc1_latent_proj: config
                .moe_latent_size
                .map(|width| projection(config.hidden_size, width)),
            fc2_latent_proj: config
                .moe_latent_size
                .map(|width| projection(width, config.hidden_size)),
            experts_per_token: config.num_experts_per_tok,
            groups: config.n_group,
            top_groups: config.topk_group,
            renormalize_selected: config.norm_topk_prob,
            routed_scale: config.routed_scaling_factor,
            hidden_size: config.hidden_size,
        })
    }

    /// Returns the shared routing specification this module executes.
    ///
    /// NVIDIA's router is DeepSeek-V3's `noaux_tc` policy: experts are scored
    /// by `s = sigmoid(z)` from an F32 projection, a frozen F32 correction bias
    /// `b` changes only the discrete choice (`top-k(s + b)` after keeping the
    /// best `topk_group` groups ranked by their top-two corrected scores, with
    /// pruned experts at `-inf`), and the mixture weights are the uncorrected
    /// `s` of the chosen experts, renormalized as `s / max(sum s, 1e-20)` when
    /// `norm_topk_prob` holds and multiplied by `routed_scaling_factor`.
    ///
    /// The denominator guard is a clamp rather than the Hugging Face
    /// `sum + 1e-20`: both are the identity for every sum of positive sigmoid
    /// scores representable in F32 (the smallest such sum is far above the
    /// guard), and the clamp keeps this module bit-identical to its earlier
    /// releases.
    #[must_use]
    pub fn router_config(&self) -> RouterConfig {
        RouterConfig {
            experts: self.routed.len(),
            experts_per_token: self.experts_per_token,
            policy: RoutingPolicy {
                scoring: RouterScoring::Sigmoid,
                order: ScoringOrder::ScoreThenTopK,
                selection: ExpertSelection::GroupLimited(GroupLimitedSelection {
                    groups: self.groups,
                    top_groups: self.top_groups,
                    group_score: GroupScore::TopTwoSum,
                    pruned_score: PrunedExpertScore::NegativeInfinity,
                }),
                correction_bias: true,
                router_bias: false,
                projection: RouterProjectionPrecision::Fp32,
                denominator: RenormalizationDenominator::ClampedBelow(ROUTER_DENOMINATOR_FLOOR),
            },
            renormalize_selected: self.renormalize_selected,
            routed_scale: self.routed_scale,
        }
    }

    /// Routes `[batch, sequence, hidden]` tokens and evaluates only selected experts.
    ///
    /// Routing ranks, normalizes, and weights in F32 even when the expert
    /// projections run in F16/BF16, matching the reference; every routed
    /// contribution `expert(x) * w` is formed and summed in F32 and rounded to
    /// the expert dtype once, before the optional latent up-projection. The
    /// always-active shared expert is then added in the residual dtype.
    ///
    /// # Errors
    ///
    /// Returns a shape error for an empty or mis-sized input, a configuration
    /// error when the instantiated cardinalities cannot be routed (for example
    /// `experts_per_token` above the experts left after group pruning), and a
    /// contextual `moe.router_scores` shape error naming the first token whose
    /// router scores or weights are not finite.
    pub fn forward(&self, input: Tensor<B, 3>) -> Result<NemotronHMoeOutput<B>> {
        let [batch, sequence, hidden] = input.dims();
        if batch == 0 || sequence == 0 || hidden != self.hidden_size {
            return Err(Error::shape(
                "nemotron_h.moe.input",
                format!("[batch>0, sequence>0, {}]", self.hidden_size),
                format!("[{batch}, {sequence}, {hidden}]"),
            ));
        }
        let tokens = batch.checked_mul(sequence).ok_or_else(|| {
            Error::configuration("nemotron_h.moe", "batch * sequence overflows usize")
        })?;
        let output = self.forward_tokens(input.reshape([tokens, hidden]))?;
        let k = self.experts_per_token;
        Ok(NemotronHMoeOutput {
            hidden_state: output.hidden_state.reshape([batch, sequence, hidden]),
            router_logits: output
                .router_logits
                .reshape([batch, sequence, self.routed.len()]),
            selected_experts: output.selected_experts.reshape([batch, sequence, k]),
            selected_weights: output.selected_weights.reshape([batch, sequence, k]),
            expert_token_counts: output.expert_token_counts,
            evaluated_experts: output.evaluated_experts,
        })
    }

    /// Routes flattened `[tokens, hidden]` states; the core of [`Self::forward`].
    ///
    /// Stateful execution uses this directly on packed valid tokens so padded
    /// batch slots never reach the router or an expert.
    pub(crate) fn forward_tokens(&self, flat: Tensor<B, 2>) -> Result<NemotronHFlatMoeOutput<B>> {
        let [tokens, hidden] = flat.dims();
        if tokens == 0 || hidden != self.hidden_size {
            return Err(Error::shape(
                "nemotron_h.moe.input",
                format!("[tokens>0, {}]", self.hidden_size),
                format!("[{tokens}, {hidden}]"),
            ));
        }
        let config = self.router_config();
        let router_logits = router_logits(flat.clone(), &self.router, config.policy.projection)?;
        let decision = config.route(router_logits.clone(), Some(self.correction_bias.val()))?;
        let routed_input = match &self.fc1_latent_proj {
            Some(projection) => operations::linear(flat.clone(), projection)?,
            None => flat.clone(),
        };
        let routed_dtype = routed_input.dtype();
        let dispatch = dispatch_routed_experts(
            routed_input,
            decision.weights.clone(),
            decision.experts.clone(),
            Some(decision.non_finite_tokens),
            self.routed.len(),
            RoutedExpertPrecision::uniform(DType::F32),
            |expert, rows| self.routed[expert].forward(rows),
        )?;
        let routed_output = dispatch.hidden_state.cast(routed_dtype);
        let routed_output = match &self.fc2_latent_proj {
            Some(projection) => operations::linear(routed_output, projection)?,
            None => routed_output,
        };
        let shared = self.shared.forward(flat)?;
        Ok(NemotronHFlatMoeOutput {
            hidden_state: routed_output + shared,
            router_logits,
            selected_experts: decision.experts,
            selected_weights: decision.weights,
            expert_token_counts: dispatch.expert_token_counts,
            evaluated_experts: dispatch.evaluated_experts,
        })
    }
}

/// Smallest renormalization denominator of selected sigmoid router weights.
const ROUTER_DENOMINATOR_FLOOR: f32 = 1e-20;

/// Routing result over flattened tokens, before reshaping to `[batch, sequence, ..]`.
pub(crate) struct NemotronHFlatMoeOutput<B: Backend> {
    /// Sum of routed and shared expert outputs `[tokens, hidden]`.
    pub(crate) hidden_state: Tensor<B, 2>,
    /// F32 router logits `[tokens, experts]`.
    pub(crate) router_logits: Tensor<B, 2>,
    /// Selected expert IDs `[tokens, top_k]`.
    pub(crate) selected_experts: Tensor<B, 2, Int>,
    /// F32 mixture weights `[tokens, top_k]` after renormalization and scaling.
    pub(crate) selected_weights: Tensor<B, 2>,
    /// Assignment count of every routed expert.
    pub(crate) expert_token_counts: Vec<usize>,
    /// Number of routed expert graphs evaluated.
    pub(crate) evaluated_experts: usize,
}

/// One pre-normalized heterogeneous residual block.
#[derive(Module, Debug)]
pub struct NemotronHBlock<B: Backend> {
    /// RMS normalization applied before the selected mixer.
    pub norm: RmsNorm<B>,
    /// Mamba2 mixer for `linear_attention` blocks.
    pub mamba: Option<NemotronMamba2<B>>,
    /// Causal GQA mixer for `full_attention` blocks.
    pub attention: Option<GroupedQueryAttention<B>>,
    /// Dense two-projection mixer for `mlp` blocks.
    pub mlp: Option<NemotronHDense<B>>,
    /// Sparse shared-plus-routed mixer for `moe` blocks.
    pub moe: Option<NemotronHMoe<B>>,
    /// Serialized block discriminator.
    pub block_type: NemotronHBlockType,
}

/// Borrowed view of a block's single mixer, checked against its declared type.
///
/// [`NemotronHBlock`] stores its mixer in four optional fields so that
/// checkpoint record paths (`layers.{i}.mamba`, `.attention`, `.mlp`, `.moe`)
/// stay stable. Every execution path matches on this view instead, so the
/// topology is validated in exactly one place.
#[derive(Debug, Clone, Copy)]
pub(super) enum BlockMixer<'a, B: Backend> {
    /// Mamba-2 selective state-space mixer.
    Mamba2(
        /// The block's `mamba` field, mixing `[batch, sequence, hidden]` through
        /// its convolution and selective scan.
        &'a NemotronMamba2<B>,
    ),
    /// Causal grouped-query attention.
    Attention(
        /// The block's `attention` field; its native key/value heads are what
        /// the attention cache stores.
        &'a GroupedQueryAttention<B>,
    ),
    /// Dense two-projection MLP.
    Mlp(
        /// The block's `mlp` field, the ungated `down(phi(up(x)))` map.
        &'a NemotronHDense<B>,
    ),
    /// Sparse routed-plus-shared MoE.
    Moe(
        /// The block's `moe` field: router, routed experts, and shared expert.
        &'a NemotronHMoe<B>,
    ),
}

impl<B: Backend> NemotronHBlock<B> {
    /// Returns the block's mixer after checking that exactly the declared one is present.
    ///
    /// # Errors
    ///
    /// Returns a checkpoint error when the optional fields disagree with
    /// `block_type` (for example after manual module surgery).
    pub(super) fn mixer(&self) -> Result<BlockMixer<'_, B>> {
        match (
            self.block_type,
            &self.mamba,
            &self.attention,
            &self.mlp,
            &self.moe,
        ) {
            (NemotronHBlockType::LinearAttention, Some(mamba), None, None, None) => {
                Ok(BlockMixer::Mamba2(mamba))
            }
            (NemotronHBlockType::FullAttention, None, Some(attention), None, None) => {
                Ok(BlockMixer::Attention(attention))
            }
            (NemotronHBlockType::Mlp, None, None, Some(mlp), None) => Ok(BlockMixer::Mlp(mlp)),
            (NemotronHBlockType::Moe, None, None, None, Some(moe)) => Ok(BlockMixer::Moe(moe)),
            _ => Err(Error::Checkpoint(
                "Nemotron-H block contains an invalid mixer topology".into(),
            )),
        }
    }

    /// Initializes exactly one mixer selected by the authoritative schedule.
    fn init(
        config: &NemotronHConfig,
        block_type: NemotronHBlockType,
        device: &B::Device,
    ) -> Result<Self> {
        let norm = nemotron_rms_norm(config.hidden_size, config.layer_norm_epsilon, device)?;
        let activation = activation_from_name(&config.mlp_hidden_act)?;
        let (mamba, attention, mlp, moe) = match block_type {
            NemotronHBlockType::LinearAttention => {
                (Some(mamba_config(config).init(device)?), None, None, None)
            }
            NemotronHBlockType::FullAttention => {
                let mut attention = GroupedQueryAttentionConfig::new(
                    config.hidden_size,
                    config.num_attention_heads,
                    config.num_key_value_heads,
                    config.head_dim,
                );
                attention.bias = config.attention_bias;
                (None, Some(attention.init(device)?), None, None)
            }
            NemotronHBlockType::Mlp => (
                None,
                None,
                Some(dense_mlp(
                    config.hidden_size,
                    config.intermediate_size,
                    activation,
                    config.mlp_bias,
                    device,
                )?),
                None,
            ),
            NemotronHBlockType::Moe => {
                (None, None, None, Some(NemotronHMoe::init(config, device)?))
            }
        };
        Ok(Self {
            norm,
            mamba,
            attention,
            mlp,
            moe,
            block_type,
        })
    }

    /// Evaluates one stateless residual block and its optional router output.
    ///
    /// With `hooks`, the block emits its activation points (see
    /// [`NemotronHForCausalLm::activation_points`]) exactly where they occur
    /// in the graph, and an edit replaces the activation for every later
    /// operation. `context.policy` changes attention connectivity for staged
    /// experiments; the plain forward pass uses causal attention.
    fn forward(
        &self,
        hidden: Tensor<B, 3>,
        context: &BlockContext<'_, B>,
        mut hooks: Option<&mut Hooks<'_, B>>,
    ) -> Result<(Tensor<B, 3>, Option<NemotronHMoeOutput<B>>)> {
        let layer = Layer::decoder(context.index);
        let hidden = emit(hooks.as_deref_mut(), layer.residual_pre(), hidden)?;
        let residual = hidden.clone();
        let normalized = self.norm.forward(hidden);
        let (mixed, routed) = match self.mixer()? {
            BlockMixer::Mamba2(mamba) => {
                let normalized = emit(hooks.as_deref_mut(), layer.mixer_input(), normalized)?;
                let mixed = mamba.forward(
                    normalized,
                    context.token_mask,
                    NemotronMamba2Mode::ChunkedPrefill,
                )?;
                (
                    emit(hooks.as_deref_mut(), layer.mixer_output(), mixed)?,
                    None,
                )
            }
            BlockMixer::Attention(attention) => {
                let normalized = emit(hooks.as_deref_mut(), layer.mixer_input(), normalized)?;
                let projection = attention.project_self(normalized)?;
                let [batch, _, sequence, _] = projection.query.dims();
                let device = projection.query.device();
                let mut mask = stateless_attention_mask(
                    context.token_mask,
                    batch,
                    attention.query_heads,
                    sequence,
                    context.attention_window,
                    &device,
                )?;
                let (mut bias, mut causal) = (None, true);
                if let Some(policy) = context.policy {
                    let shape = [batch, attention.query_heads, sequence, sequence];
                    mask = policy.exclusion_mask(mask, shape, &device)?;
                    bias = policy.expanded_bias(shape, &device)?;
                    causal = policy.causal;
                }
                let mixed = match hooks.as_deref_mut() {
                    Some(hooks) => attention.attend_with_hooks(
                        projection,
                        mask,
                        bias,
                        causal,
                        layer.self_attention(),
                        hooks,
                    )?,
                    None => attention.attend(projection, mask, bias, causal)?,
                };
                (
                    emit(hooks.as_deref_mut(), layer.mixer_output(), mixed)?,
                    None,
                )
            }
            BlockMixer::Mlp(mlp) => {
                let normalized = emit(hooks.as_deref_mut(), layer.mlp_input(), normalized)?;
                let mixed = match hooks.as_deref_mut() {
                    Some(hooks) => mlp.forward_with_hooks(normalized, layer, hooks)?,
                    None => mlp.forward(normalized)?,
                };
                (emit(hooks.as_deref_mut(), layer.mlp_output(), mixed)?, None)
            }
            BlockMixer::Moe(moe) => {
                let normalized = emit(hooks.as_deref_mut(), layer.mlp_input(), normalized)?;
                let output = moe.forward(normalized)?;
                let mixed = emit(
                    hooks.as_deref_mut(),
                    layer.mlp_output(),
                    output.hidden_state.clone(),
                )?;
                (mixed, Some(output))
            }
        };
        let hidden = emit(hooks, layer.residual_post(), residual + mixed)?;
        Ok((hidden, routed))
    }
}

/// Per-block inputs of one stateless pass.
pub(super) struct BlockContext<'a, B: Backend> {
    /// Zero-based block index, which addresses the block's activation points.
    pub(super) index: usize,
    /// Optional `[batch, sequence]` token validity (`true == valid`).
    pub(super) token_mask: Option<&'a TokenMask<B>>,
    /// Optional experiment-only causal attention window ([`NemotronHConfig::attention_window`]).
    pub(super) attention_window: Option<usize>,
    /// Attention connectivity override for staged experiments.
    pub(super) policy: Option<&'a AttentionPolicy<B>>,
}

/// Emits `tensor` at `point` when hooks are active; otherwise returns it unchanged.
pub(super) fn emit<B: Backend, const D: usize>(
    hooks: Option<&mut Hooks<'_, B>>,
    point: Point<D>,
    tensor: Tensor<B, D>,
) -> Result<Tensor<B, D>> {
    match hooks {
        Some(hooks) => hooks.apply(point, tensor),
        None => Ok(tensor),
    }
}

/// Token embedding, heterogeneous residual stack, and final RMSNorm.
#[derive(Module, Debug)]
pub struct NemotronHModel<B: Backend> {
    /// Token embedding table.
    pub embed_tokens: Embedding<B>,
    /// Authoritative ordered heterogeneous blocks.
    pub layers: Vec<NemotronHBlock<B>>,
    /// Final RMS normalization.
    pub norm: RmsNorm<B>,
    /// Validated family configuration excluded from parameter records.
    #[module(skip)]
    pub config: NemotronHConfig,
}

impl<B: Backend> NemotronHModel<B> {
    /// Allocates the exact schedule declared by a validated config.
    pub fn init(config: NemotronHConfig, device: &B::Device) -> Result<Self> {
        ArchitectureConfig::validate(&config)?;
        if config.attention_dropout != 0.0 || config.hidden_dropout != 0.0 {
            return Err(Error::BackendCapability(
                "Nemotron-H inference currently requires released zero-dropout configs".into(),
            ));
        }
        if config.residual_in_fp32 {
            return Err(Error::BackendCapability(
                "Nemotron-H FP32 residual accumulation is not implemented".into(),
            ));
        }
        let layers = config
            .layers_block_type
            .iter()
            .copied()
            .map(|kind| NemotronHBlock::init(&config, kind, device))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            embed_tokens: EmbeddingConfig::new(config.vocab_size, config.hidden_size).init(device),
            layers,
            norm: nemotron_rms_norm(config.hidden_size, config.layer_norm_epsilon, device)?,
            config,
        })
    }

    /// Lists which compact cache every block uses, in schedule order.
    pub(super) fn cache_kinds(&self) -> Vec<plan::CacheKind> {
        self.layers
            .iter()
            .map(|layer| match layer.block_type {
                NemotronHBlockType::FullAttention => plan::CacheKind::Attention,
                NemotronHBlockType::LinearAttention => plan::CacheKind::Recurrent,
                NemotronHBlockType::Mlp | NemotronHBlockType::Moe => plan::CacheKind::Stateless,
            })
            .collect()
    }

    /// Counts blocks of one kind; cache groups index attention and Mamba blocks separately.
    pub(super) fn count_blocks(&self, kind: NemotronHBlockType) -> usize {
        self.layers
            .iter()
            .filter(|layer| layer.block_type == kind)
            .count()
    }

    /// Evaluates the stateless backbone and retains requested diagnostics.
    ///
    /// With `hooks`, the embedding, every block's points, and the final norm
    /// are emitted in graph order; the caller owns the hook scope.
    fn forward(
        &self,
        input: &CausalLmInput<B>,
        options: &ForwardOptions,
        mut hooks: Option<&mut Hooks<'_, B>>,
    ) -> Result<NemotronHBackboneOutput<B>> {
        // Mamba and attention mixers are position-free; explicit positions
        // are still range-checked so callers get the shared input contract.
        validate_input(
            input,
            &self.embed_tokens.weight.lazy_device(),
            &InputContract::new(INPUT_NAMES, PARAMETERS, self.config.vocab_size)
                .with_position_limit(self.config.max_position_embeddings)
                .with_implicit_positions(ImplicitPositions::Unused),
        )?;
        let mut hidden = emit(
            hooks.as_deref_mut(),
            Stack::Decoder.embedding(),
            self.embed_tokens.forward(input.token_ids.clone()),
        )?;
        let mut hidden_states = Vec::with_capacity(if options.hidden_states {
            self.layers.len() + 2
        } else {
            0
        });
        let mut routers = Vec::new();
        if options.hidden_states {
            hidden_states.push(hidden.clone());
        }
        for (index, layer) in self.layers.iter().enumerate() {
            let context = BlockContext {
                index,
                token_mask: input.token_mask.as_ref(),
                attention_window: self.config.attention_window,
                policy: None,
            };
            let (next, routed) = layer.forward(hidden, &context, hooks.as_deref_mut())?;
            hidden = next;
            if options.hidden_states {
                hidden_states.push(hidden.clone());
            }
            if let Some(routed) = routed
                && (options.router_logits || options.auxiliary_losses)
            {
                routers.push(RouterDiagnostics {
                    logits: options.router_logits.then(|| routed.router_logits.clone()),
                    load_balancing_loss: options
                        .auxiliary_losses
                        .then(|| routed.load_balancing_loss(input.token_mask.as_ref()))
                        .transpose()?,
                    z_loss: None,
                });
            }
        }
        hidden = emit(
            hooks,
            Stack::Decoder.final_norm(),
            self.norm.forward(hidden),
        )?;
        if options.hidden_states {
            hidden_states.push(hidden.clone());
        }
        Ok(NemotronHBackboneOutput {
            hidden,
            hidden_states,
            routers,
        })
    }
}

/// Internal backbone output before vocabulary projection.
struct NemotronHBackboneOutput<B: Backend> {
    /// Final normalized activations `[batch, sequence, hidden]`.
    hidden: Tensor<B, 3>,
    /// Requested embedding, post-block, and final-normalized activations in order.
    hidden_states: Vec<Tensor<B, 3>>,
    /// Requested routing diagnostics in the order of the backbone's MoE blocks.
    routers: Vec<RouterDiagnostics<B>>,
}

/// Optional checkpoint-declared one-step MTP body.
///
/// Lightning contains one next-token stage. It normalizes a shifted token
/// embedding and the base final hidden state independently, concatenates them,
/// projects back to the residual width, then applies the checkpoint-declared
/// attention/MoE schedule and a final norm. Vocabulary projection is shared with
/// the base language-model head.
#[derive(Module, Debug)]
pub struct NemotronHNextTokenHead<B: Backend> {
    /// Normalization for the shifted token embedding.
    pub embedding_norm: RmsNorm<B>,
    /// Normalization for the base final hidden state.
    pub hidden_norm: RmsNorm<B>,
    /// Concatenation projection from `2H` back to `H`.
    pub eh_proj: Linear<B>,
    /// Pre-normalization before MTP attention.
    pub attention_norm: RmsNorm<B>,
    /// Causal GQA MTP mixer.
    pub attention: GroupedQueryAttention<B>,
    /// Pre-normalization before MTP MoE.
    pub moe_norm: RmsNorm<B>,
    /// MTP shared-plus-routed expert mixer.
    pub moe: NemotronHMoe<B>,
    /// Final MTP normalization before the shared LM head.
    pub final_layer_norm: RmsNorm<B>,
    /// Optional experiment-only causal window applied by the MTP attention stage.
    ///
    /// [`Self::init`] copies [`NemotronHConfig::attention_window`], never the
    /// serialized `sliding_window`, which the reference ignores.
    pub sliding_window: Option<usize>,
}

impl<B: Backend> NemotronHNextTokenHead<B> {
    /// Allocates the single Lightning MTP stage.
    fn init(config: &NemotronHConfig, device: &B::Device) -> Result<Self> {
        if config.num_nextn_predict_layers != 1
            || config.mtp_layers_block_type
                != [NemotronHBlockType::FullAttention, NemotronHBlockType::Moe]
        {
            return Err(Error::BackendCapability(
                "Nemotron-H MTP currently supports Lightning's single [attention, moe] stage"
                    .into(),
            ));
        }
        let fusion_width = config.hidden_size.checked_mul(2).ok_or_else(|| {
            Error::configuration(
                "nemotron_h.next_token_head.eh_proj",
                "2 * hidden_size overflows usize",
            )
        })?;
        let norm = || nemotron_rms_norm(config.hidden_size, config.layer_norm_epsilon, device);
        let mut attention = GroupedQueryAttentionConfig::new(
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
        );
        attention.bias = config.attention_bias;
        Ok(Self {
            embedding_norm: norm()?,
            hidden_norm: norm()?,
            eh_proj: operations::init_linear(
                &LinearConfig::new(fusion_width, config.hidden_size)
                    .with_bias(false)
                    .with_layout(LinearLayout::Col),
                device,
            ),
            attention_norm: norm()?,
            attention: attention.init(device)?,
            moe_norm: norm()?,
            moe: NemotronHMoe::init(config, device)?,
            final_layer_norm: norm()?,
            sliding_window: config.attention_window,
        })
    }

    /// Evaluates the MTP body and returns its normalized hidden state.
    pub fn forward(
        &self,
        shifted_embedding: Tensor<B, 3>,
        base_hidden: Tensor<B, 3>,
    ) -> Result<(Tensor<B, 3>, NemotronHMoeOutput<B>)> {
        let mut hidden = self.fuse(shifted_embedding, base_hidden)?;
        let residual = hidden.clone();
        let projection = self
            .attention
            .project_self(self.attention_norm.forward(hidden))?;
        let [batch, _, sequence, _] = projection.query.dims();
        let mask = stateless_attention_mask(
            None,
            batch,
            self.attention.query_heads,
            sequence,
            self.sliding_window,
            &projection.query.device(),
        )?;
        hidden = residual + self.attention.attend(projection, mask, None, true)?;
        self.finish(hidden)
    }

    /// Applies the independently normalized embedding/target fusion equation.
    ///
    /// The concatenation order is checkpoint-significant: vLLM 0.27.1 forms
    /// `[enorm(embedding), hnorm(target_hidden)]` before the `2H -> H` map.
    pub(super) fn fuse(
        &self,
        shifted_embedding: Tensor<B, 3>,
        base_hidden: Tensor<B, 3>,
    ) -> Result<Tensor<B, 3>> {
        if shifted_embedding.dims() != base_hidden.dims() {
            return Err(Error::shape(
                "nemotron_h.next_token_head",
                format!(
                    "matching shifted/base tensors, base={:?}",
                    base_hidden.dims()
                ),
                format!("shifted={:?}", shifted_embedding.dims()),
            ));
        }
        if shifted_embedding.device() != base_hidden.device()
            || self.eh_proj.weight.val().device() != base_hidden.device()
        {
            return Err(Error::BackendCapability(
                "Nemotron-H MTP inputs and parameters must share a device".into(),
            ));
        }
        let fused = Tensor::cat(
            vec![
                self.embedding_norm.forward(shifted_embedding),
                self.hidden_norm.forward(base_hidden),
            ],
            2,
        );
        operations::linear(fused, &self.eh_proj)
    }

    /// Applies the MoE residual and terminal normalization after attention.
    ///
    /// This is equivalent to vLLM's fused RMSNorm residual bookkeeping: the
    /// attention residual is materialized before MoE normalization, then the
    /// MoE result is added and normalized exactly once at the stage boundary.
    pub(super) fn finish(
        &self,
        hidden: Tensor<B, 3>,
    ) -> Result<(Tensor<B, 3>, NemotronHMoeOutput<B>)> {
        let residual = hidden.clone();
        let routed = self.moe.forward(self.moe_norm.forward(hidden))?;
        let hidden = self
            .final_layer_norm
            .forward(residual + routed.hidden_state.clone());
        Ok((hidden, routed))
    }
}

/// Decoder-only Nemotron-H language model and optional MTP body.
#[derive(Module, Debug)]
pub struct NemotronHForCausalLm<B: Backend> {
    /// Hybrid backbone.
    pub model: NemotronHModel<B>,
    /// Independent vocabulary projection, absent for tied embeddings.
    pub lm_head: Option<Linear<B>>,
    /// Checkpoint-declared MTP body when `num_nextn_predict_layers == 1`.
    pub next_token_head: Option<NemotronHNextTokenHead<B>>,
}

impl<B: Backend> NemotronHForCausalLm<B> {
    /// Initializes a validated full-precision inference graph.
    pub fn init(config: NemotronHConfig, device: &B::Device) -> Result<Self> {
        // Validate before allocating the optional MTP body: malformed widths
        // must return a contextual error instead of overflowing in LinearConfig.
        ArchitectureConfig::validate(&config)?;
        let hidden = config.hidden_size;
        let vocabulary = config.vocab_size;
        let tied = config.tie_word_embeddings;
        let next_token_head = (config.num_nextn_predict_layers > 0)
            .then(|| NemotronHNextTokenHead::init(&config, device))
            .transpose()?;
        Ok(Self {
            model: NemotronHModel::init(config, device)?,
            lm_head: (!tied).then(|| {
                operations::init_linear(
                    &LinearConfig::new(hidden, vocabulary)
                        .with_bias(false)
                        .with_layout(LinearLayout::Col),
                    device,
                )
            }),
            next_token_head,
        })
    }

    /// Evaluates differentiable stateless inference or training math.
    pub fn forward(
        &self,
        input: CausalLmInput<B>,
        options: &ForwardOptions,
    ) -> Result<CausalLmOutput<B>> {
        if options.attentions {
            return Err(Error::BackendCapability(
                "Nemotron-H attention probability capture is not yet exposed".into(),
            ));
        }
        let backbone = self.model.forward(&input, options, None)?;
        self.finish_forward(&input, options, backbone)
    }

    /// Projects the requested positions of a completed backbone pass.
    fn finish_forward(
        &self,
        input: &CausalLmInput<B>,
        options: &ForwardOptions,
        backbone: NemotronHBackboneOutput<B>,
    ) -> Result<CausalLmOutput<B>> {
        let hidden = backbone.hidden;
        let selected = select_hidden(hidden.clone(), input.token_mask.as_ref(), &options.logits)?;
        let logits = self.project_logits(selected)?.cast(DType::F32);
        Ok(CausalLmOutput {
            logits,
            final_hidden_state: options.hidden_states.then_some(hidden),
            hidden_states: backbone.hidden_states,
            attentions: Vec::new(),
            routers: backbone.routers,
            load_balancing_loss: None,
        })
    }

    /// Returns the validated architecture configuration.
    #[must_use]
    pub const fn config(&self) -> &NemotronHConfig {
        &self.model.config
    }

    /// Runs the optional Lightning MTP body against aligned base hidden states.
    pub fn predict_next_token_logits(
        &self,
        shifted_token_ids: Tensor<B, 2, Int>,
        base_hidden: Tensor<B, 3>,
    ) -> Result<Tensor<B, 3>> {
        let head = self.next_token_head.as_ref().ok_or_else(|| {
            Error::BackendCapability("this Nemotron-H config has no next-token head".into())
        })?;
        let [batch, sequence] = shifted_token_ids.dims();
        if base_hidden.dims() != [batch, sequence, self.config().hidden_size] {
            return Err(Error::shape(
                "nemotron_h.next_token_head.base_hidden",
                format!("[{batch}, {sequence}, {}]", self.config().hidden_size),
                format!("{:?}", base_hidden.dims()),
            ));
        }
        validate_token_values(
            &shifted_token_ids,
            self.config().vocab_size,
            INPUT_NAMES.token_ids,
        )?;
        let embedding = self.model.embed_tokens.forward(shifted_token_ids);
        let (hidden, _) = head.forward(embedding, base_hidden)?;
        Ok(self.project_logits(hidden)?.cast(DType::F32))
    }

    /// Applies the independent or tied vocabulary projection.
    ///
    /// A tied head multiplies by the transposed view of the embedding table
    /// without copying it ([`operations::matmul_transposed`]).
    fn project_logits(&self, hidden: Tensor<B, 3>) -> Result<Tensor<B, 3>> {
        #[cfg(test)]
        LOGIT_PROJECTIONS.with(|projections| projections.set(projections.get() + 1));
        let weight_device = self.lm_head.as_ref().map_or_else(
            || self.model.embed_tokens.weight.val().device(),
            |head| head.weight.val().device(),
        );
        if weight_device != hidden.device() {
            return Err(Error::BackendCapability(
                "Nemotron-H LM head and hidden states must share a device".into(),
            ));
        }
        match &self.lm_head {
            Some(head) => operations::linear(hidden, head),
            None => operations::matmul_transposed(hidden, self.model.embed_tokens.weight.val()),
        }
    }
}

#[cfg(test)]
thread_local! {
    /// Number of vocabulary projections run on this thread (test instrumentation).
    ///
    /// MTP tests read it to prove that prompt alignment never projects logits.
    pub(super) static LOGIT_PROJECTIONS: core::cell::Cell<usize> =
        const { core::cell::Cell::new(0) };
}

/// Converts family activation metadata into the shared mathematical enum.
fn activation_from_name(name: &str) -> Result<Activation> {
    match name {
        "relu2" => Ok(Activation::ReluSquared),
        "silu" => Ok(Activation::Silu),
        _ => Err(Error::configuration(
            "nemotron_h.mlp_hidden_act",
            format!("unsupported activation {name:?}"),
        )),
    }
}

/// Initializes the exact Nemotron-H RMSNorm floating-point operation graph.
///
/// NVIDIA's reference first computes `x * rsqrt(mean(x^2) + epsilon)` in
/// FP32, rounds that unit-RMS activation back to the residual dtype, and only
/// then multiplies by the learned scale. This differs by representable F16
/// values from both division by `sqrt` and late rounding after the scale.
fn nemotron_rms_norm<B: Backend>(
    size: usize,
    epsilon: f64,
    device: &B::Device,
) -> Result<RmsNorm<B>> {
    RmsNormConfig::new(size)
        .with_epsilon(epsilon)
        .with_round_before_scale(true)
        .with_root_mode(RmsNormRootMode::MultiplyByReciprocalSqrt)
        .init(device)
}

/// Builds one exact Mamba2 component config from the family schema.
///
/// The discretization clamp follows the pinned Transformers revision
/// (`cfa309cfa34b5497a32fd50af49aa97aac94c054`), whose chunk scan clamps to
/// `(time_step_min, inf)` and whose single-step update is unclamped. The
/// declared `time_step_limit` is compatibility metadata there and is ignored.
fn mamba_config(config: &NemotronHConfig) -> NemotronMamba2Config {
    NemotronMamba2Config {
        hidden_size: config.hidden_size,
        num_heads: config.mamba_num_heads,
        head_dim: config.mamba_head_dim,
        state_size: config.ssm_state_size,
        groups: config.n_groups,
        convolution_kernel: config.conv_kernel,
        chunk_size: config.chunk_size,
        time_step_clamp: TimeStepClamp::chunked_only(TimeStepLimits::at_least(
            config.time_step_min,
        )),
        normalization_epsilon: config.layer_norm_epsilon,
        projection_bias: config.use_bias,
        convolution_bias: config.use_conv_bias,
    }
}

/// Combines true-valid padding semantics with an optional experiment-only causal left window.
fn stateless_attention_mask<B: Backend>(
    token_mask: Option<&TokenMask<B>>,
    batch: usize,
    heads: usize,
    sequence: usize,
    window: Option<usize>,
    device: &B::Device,
) -> Result<Option<Tensor<B, 4, Bool>>> {
    let padding = padding_attention_mask(token_mask, token_mask, batch, heads, sequence, sequence)?;
    let Some(window) = window else {
        return Ok(padding);
    };
    let end = i64::try_from(sequence).map_err(|_| {
        Error::configuration(
            "nemotron_h.attention_mask",
            "sequence length exceeds portable signed index range",
        )
    })?;
    let query = Tensor::<B, 1, Int>::arange(0..end, device)
        .reshape([1, 1, sequence, 1])
        .expand([batch, heads, sequence, sequence]);
    let key = Tensor::<B, 1, Int>::arange(0..end, device)
        .reshape([1, 1, 1, sequence])
        .expand([batch, heads, sequence, sequence]);
    let window = i64::try_from(window).map_err(|_| {
        Error::configuration(
            "nemotron_h.attention_window",
            "window exceeds portable signed index range",
        )
    })?;
    let too_old = (key + window).lower_equal(query);
    Ok(Some(match padding {
        Some(padding) => padding.bool_or(too_old),
        None => too_old,
    }))
}

#[cfg(test)]
#[path = "model_tests.rs"]
mod tests;
