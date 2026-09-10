//! Gated DeltaNet recurrence and blockwise prefill algorithms.
//!
//! DeltaNet replaces a quadratic attention history with a matrix-valued state
//! `S_t` for every value head. With L2-normalized keys and queries, one step is
//!
//! ```text
//! S'_t = exp(g_t) S_{t-1}
//! e_t  = v_t - k_t^T S'_t
//! S_t  = S'_t + k_t (beta_t e_t)^T
//! y_t  = q_t^T S_t / sqrt(d_k)
//! ```
//!
//! The delta term first reads the value currently associated with `k_t`, then
//! writes only the prediction error. This corrects existing associations rather
//! than continually accumulating duplicate values. Negative `g_t` supplies a
//! learned exponential forgetting factor.
//!
//! [`recurrent_gated_delta_rule`] is the transparent token-by-token reference
//! used for decode and parity tests. [`chunked_gated_delta_rule`] algebraically
//! resolves all within-chunk delta updates as a triangular system, then performs
//! dense batched matrix products. It preserves the recurrent result while
//! exposing substantially more parallelism during prompt prefill.

use burn::{
    module::{Initializer, Module, Param},
    nn::{
        Linear, LinearConfig, LinearLayout, PaddingConfig1d,
        conv::{Conv1d, Conv1dConfig},
    },
    tensor::{
        Bool, DType, Distribution, Tensor, activation, backend::Backend, module::conv1d,
        ops::ConvOptions,
    },
};
use serde::{Deserialize, Serialize};

use crate::{Error, Result, architecture::TokenMask, normalization::RmsNormConfig, operations};

/// Numerically sensitive controls for both DeltaNet execution paths.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DeltaRuleConfig {
    /// Stabilizer in query/key L2 normalization.
    pub normalization_epsilon: f64,
    /// Maximum time-axis block processed by the parallel prefill path.
    pub chunk_size: usize,
}

impl Default for DeltaRuleConfig {
    /// Uses the official Qwen reference constants.
    fn default() -> Self {
        Self {
            normalization_epsilon: 1e-6,
            chunk_size: 64,
        }
    }
}

impl DeltaRuleConfig {
    /// Validates constants before a backend operation is dispatched.
    pub fn validate(&self) -> Result<()> {
        if !self.normalization_epsilon.is_finite() || self.normalization_epsilon <= 0.0 {
            return Err(Error::configuration(
                "delta_net.normalization_epsilon",
                "must be finite and greater than zero",
            ));
        }
        if self.chunk_size == 0 {
            return Err(Error::configuration(
                "delta_net.chunk_size",
                "must be non-zero",
            ));
        }
        Ok(())
    }
}

/// Inputs shared by recurrent decode and chunked prompt prefill.
#[derive(Clone, Debug)]
pub struct DeltaRuleInput<B: Backend> {
    /// Queries shaped `[batch, sequence, heads, key_width]`.
    pub query: Tensor<B, 4>,
    /// Keys shaped `[batch, sequence, heads, key_width]`.
    pub key: Tensor<B, 4>,
    /// Values shaped `[batch, sequence, heads, value_width]`.
    pub value: Tensor<B, 4>,
    /// Logarithmic state decays shaped `[batch, sequence, heads]`.
    pub log_decay: Tensor<B, 3>,
    /// Delta update strengths shaped `[batch, sequence, heads]`.
    pub beta: Tensor<B, 3>,
    /// Optional FP32 recurrent state `[batch, heads, key_width, value_width]`.
    pub initial_state: Option<Tensor<B, 4>>,
    /// Optional semantic mask `[batch, sequence]`, where `true` means valid.
    pub token_mask: Option<Tensor<B, 2, Bool>>,
}

/// Delta-rule outputs and the state required by a subsequent invocation.
#[derive(Clone, Debug)]
pub struct DeltaRuleOutput<B: Backend> {
    /// Mixed values shaped `[batch, sequence, heads, value_width]`.
    pub values: Tensor<B, 4>,
    /// Final FP32 state shaped `[batch, heads, key_width, value_width]`.
    pub final_state: Tensor<B, 4>,
}

/// Checkpoint-shaped Qwen Gated DeltaNet configuration.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GatedDeltaNetConfig {
    /// Width of the residual stream accepted and returned by the mixer.
    pub hidden_size: usize,
    /// Number of independently projected query/key heads.
    pub key_heads: usize,
    /// Number of recurrent value heads; an integer multiple of `key_heads`.
    pub value_heads: usize,
    /// Width of a normalized query/key head.
    pub key_head_dim: usize,
    /// Width of a recurrent value head.
    pub value_head_dim: usize,
    /// Width of the depthwise causal convolution.
    pub convolution_kernel: usize,
    /// Stabilizer used by the output head-wise RMS normalization.
    pub normalization_epsilon: f64,
    /// Chunk size used by the parallel prompt path.
    pub prefill_chunk_size: usize,
}

impl GatedDeltaNetConfig {
    /// Creates a Qwen-compatible mixer topology.
    pub fn new(
        hidden_size: usize,
        key_heads: usize,
        value_heads: usize,
        key_head_dim: usize,
        value_head_dim: usize,
    ) -> Self {
        Self {
            hidden_size,
            key_heads,
            value_heads,
            key_head_dim,
            value_head_dim,
            convolution_kernel: 4,
            normalization_epsilon: 1e-6,
            prefill_chunk_size: 64,
        }
    }

    /// Returns the concatenated Q/K/V convolution channel count.
    pub fn convolution_channels(&self) -> Result<usize> {
        self.key_heads
            .checked_mul(self.key_head_dim)
            .and_then(|key_width| key_width.checked_mul(2))
            .and_then(|query_key_width| {
                self.value_heads
                    .checked_mul(self.value_head_dim)
                    .and_then(|value_width| query_key_width.checked_add(value_width))
            })
            .ok_or_else(|| {
                Error::configuration("delta_net", "projection dimensions overflow usize")
            })
    }

    /// Validates every projection, grouping, convolution, and numerical invariant.
    pub fn validate(&self) -> Result<()> {
        if self.hidden_size == 0
            || self.key_heads == 0
            || self.value_heads == 0
            || self.key_head_dim == 0
            || self.value_head_dim == 0
            || self.convolution_kernel == 0
        {
            return Err(Error::configuration(
                "delta_net",
                "all model and convolution dimensions must be non-zero",
            ));
        }
        if !self.value_heads.is_multiple_of(self.key_heads) {
            return Err(Error::configuration(
                "delta_net.value_heads",
                "must be an integer multiple of key_heads",
            ));
        }
        DeltaRuleConfig {
            normalization_epsilon: self.normalization_epsilon,
            chunk_size: self.prefill_chunk_size,
        }
        .validate()?;
        let _ = self.convolution_channels()?;
        self.value_heads
            .checked_mul(self.value_head_dim)
            .ok_or_else(|| {
                Error::configuration("delta_net.value_width", "dimension overflows usize")
            })?;
        Ok(())
    }

    /// Initializes lazy Burn projections and checkpoint-compatible scalar parameters.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Result<GatedDeltaNet<B>> {
        self.validate()?;
        let key_width = self.key_heads * self.key_head_dim;
        let value_width = self.value_heads * self.value_head_dim;
        let convolution_channels = self.convolution_channels()?;
        let linear = |output| {
            operations::init_linear(
                &LinearConfig::new(self.hidden_size, output)
                    .with_bias(false)
                    .with_layout(LinearLayout::Col),
                device,
            )
        };
        let convolution = Conv1dConfig::new(
            convolution_channels,
            convolution_channels,
            self.convolution_kernel,
        )
        .with_groups(convolution_channels)
        .with_bias(false)
        .with_padding(PaddingConfig1d::Explicit(self.convolution_kernel - 1, 0))
        .init(device);
        let a_log = Tensor::random(
            [self.value_heads],
            Distribution::Uniform(0.01, 16.0),
            device,
        )
        .log();
        Ok(GatedDeltaNet {
            in_proj_qkv: linear(key_width * 2 + value_width),
            in_proj_z: linear(value_width),
            in_proj_b: linear(self.value_heads),
            in_proj_a: linear(self.value_heads),
            conv1d: convolution,
            dt_bias: Initializer::Ones.init([self.value_heads], device),
            a_log: Param::from_tensor(a_log),
            norm: RmsNormConfig::new(self.value_head_dim)
                .with_epsilon(self.normalization_epsilon)
                .init(device)?,
            out_proj: operations::init_linear(
                &LinearConfig::new(value_width, self.hidden_size)
                    .with_bias(false)
                    .with_layout(LinearLayout::Col),
                device,
            ),
            key_heads: self.key_heads,
            value_heads: self.value_heads,
            key_head_dim: self.key_head_dim,
            value_head_dim: self.value_head_dim,
            convolution_kernel: self.convolution_kernel,
            hidden_size: self.hidden_size,
            convolution_channels,
            rule_config: DeltaRuleConfig {
                normalization_epsilon: 1e-6,
                chunk_size: self.prefill_chunk_size,
            },
        })
    }
}

/// Chooses the transparent recurrent kernel or blockwise prompt kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeltaNetMode {
    /// Token-by-token recurrence, intended for decode and golden checks.
    Recurrent,
    /// Blockwise-parallel prompt processing.
    ChunkedPrefill,
}

/// External convolution and matrix state for one Gated DeltaNet layer.
#[derive(Clone, Debug)]
pub struct GatedDeltaNetState<B: Backend> {
    /// Most recent `kernel - 1` projected Q/K/V channels.
    pub convolution: Tensor<B, 3>,
    /// FP32 key/value association matrix for every value head.
    pub recurrent: Tensor<B, 4>,
}

impl<B: Backend> GatedDeltaNetState<B> {
    /// Allocates zero state for one layer and batch.
    pub fn zeros(
        config: &GatedDeltaNetConfig,
        batch_size: usize,
        device: &B::Device,
    ) -> Result<Self> {
        config.validate()?;
        if batch_size == 0 {
            return Err(Error::configuration(
                "delta_net.batch_size",
                "must be non-zero",
            ));
        }
        let convolution_channels = config.convolution_channels()?;
        Ok(Self {
            convolution: Tensor::zeros(
                [
                    batch_size,
                    convolution_channels,
                    config.convolution_kernel - 1,
                ],
                device,
            ),
            recurrent: Tensor::<B, 4>::zeros(
                [
                    batch_size,
                    config.value_heads,
                    config.key_head_dim,
                    config.value_head_dim,
                ],
                device,
            )
            .cast(DType::F32),
        })
    }
}

/// Qwen checkpoint-compatible projected Gated DeltaNet mixer.
#[derive(Module, Debug)]
pub struct GatedDeltaNet<B: Backend> {
    /// Fused query/key/value projection.
    pub in_proj_qkv: Linear<B>,
    /// Value-width output gate projection.
    pub in_proj_z: Linear<B>,
    /// Per-value-head delta strength projection.
    pub in_proj_b: Linear<B>,
    /// Per-value-head discretized decay projection.
    pub in_proj_a: Linear<B>,
    /// Depthwise causal convolution over projected Q/K/V channels.
    pub conv1d: Conv1d<B>,
    /// Learned positive-time-step offset used inside softplus.
    pub dt_bias: Param<Tensor<B, 1>>,
    /// Logarithm of the positive continuous-time decay rate.
    pub a_log: Param<Tensor<B, 1>>,
    /// Per-value-head RMS normalization applied before the output gate.
    pub norm: crate::normalization::RmsNorm<B>,
    /// Projection from concatenated value heads to the residual stream.
    pub out_proj: Linear<B>,
    /// Number of independently projected query/key heads.
    pub key_heads: usize,
    /// Number of recurrent value heads.
    pub value_heads: usize,
    /// Query/key width per head.
    pub key_head_dim: usize,
    /// Value width per head.
    pub value_head_dim: usize,
    /// Causal convolution width.
    pub convolution_kernel: usize,
    /// Residual-stream width.
    pub hidden_size: usize,
    /// Fused projected Q/K/V channel count.
    pub convolution_channels: usize,
    /// Numerical controls shared by recurrence and prefill.
    pub rule_config: DeltaRuleConfig,
}

impl<B: Backend> GatedDeltaNet<B> {
    /// Evaluates a stateless differentiable mixer forward.
    ///
    /// A temporary zero state is used but never serialized with model weights.
    pub fn forward(
        &self,
        hidden: Tensor<B, 3>,
        token_mask: Option<&TokenMask<B>>,
    ) -> Result<Tensor<B, 3>> {
        let batch = hidden.dims()[0];
        let config = self.config();
        let mut state = GatedDeltaNetState::zeros(&config, batch, &hidden.device())?;
        self.forward_with_state(hidden, token_mask, &mut state, DeltaNetMode::ChunkedPrefill)
    }

    /// Evaluates a prompt or decode segment and replaces external recurrent state.
    pub fn forward_with_state(
        &self,
        hidden: Tensor<B, 3>,
        token_mask: Option<&TokenMask<B>>,
        state: &mut GatedDeltaNetState<B>,
        mode: DeltaNetMode,
    ) -> Result<Tensor<B, 3>> {
        let [batch, sequence, hidden_size] = hidden.dims();
        let expected_hidden = self.hidden_size;
        if batch == 0 || sequence == 0 || hidden_size != expected_hidden {
            return Err(Error::shape(
                "delta_net.hidden",
                format!("[batch>0, sequence>0, {expected_hidden}]"),
                format!("[{batch}, {sequence}, {hidden_size}]"),
            ));
        }
        if let Some(mask) = token_mask
            && mask.as_tensor().dims() != [batch, sequence]
        {
            return Err(Error::shape(
                "delta_net.token_mask",
                format!("[{batch}, {sequence}]"),
                format!("{:?}", mask.as_tensor().dims()),
            ));
        }
        self.validate_state(state, batch)?;

        let masked_hidden = mask_hidden(hidden, token_mask);
        let projected =
            operations::linear(masked_hidden.clone(), &self.in_proj_qkv)?.swap_dims(1, 2);
        let (convolved, convolution_state) =
            self.causal_convolution(projected, state.convolution.clone());
        let mixed = activation::silu(convolved).swap_dims(1, 2);
        let key_width = self.key_heads * self.key_head_dim;
        let value_width = self.value_heads * self.value_head_dim;
        let query = mixed.clone().narrow(2, 0, key_width).reshape([
            batch,
            sequence,
            self.key_heads,
            self.key_head_dim,
        ]);
        let key = mixed.clone().narrow(2, key_width, key_width).reshape([
            batch,
            sequence,
            self.key_heads,
            self.key_head_dim,
        ]);
        let value = mixed.narrow(2, key_width * 2, value_width).reshape([
            batch,
            sequence,
            self.value_heads,
            self.value_head_dim,
        ]);
        let query = repeat_delta_heads(query, self.value_heads);
        let key = repeat_delta_heads(key, self.value_heads);
        let z = operations::linear(masked_hidden.clone(), &self.in_proj_z)?.reshape([
            batch,
            sequence,
            self.value_heads,
            self.value_head_dim,
        ]);
        let beta = activation::sigmoid(operations::linear(masked_hidden.clone(), &self.in_proj_b)?);
        let a = operations::linear(masked_hidden, &self.in_proj_a)?.cast(DType::F32);
        let decay_rate = self
            .a_log
            .val()
            .cast(DType::F32)
            .exp()
            .reshape([1, 1, self.value_heads]);
        let time_bias = self
            .dt_bias
            .val()
            .cast(DType::F32)
            .reshape([1, 1, self.value_heads]);
        let log_decay = -(decay_rate * activation::softplus(a + time_bias, 1.0));
        let rule_input = DeltaRuleInput {
            query,
            key,
            value,
            log_decay,
            beta,
            initial_state: Some(state.recurrent.clone()),
            token_mask: token_mask.map(|mask| mask.as_tensor().clone()),
        };
        let rule_output = match mode {
            DeltaNetMode::Recurrent => recurrent_gated_delta_rule(rule_input, self.rule_config)?,
            DeltaNetMode::ChunkedPrefill => chunked_gated_delta_rule(rule_input, self.rule_config)?,
        };
        let gated = gated_delta_norm(rule_output.values, z, &self.norm);
        state.convolution = convolution_state;
        state.recurrent = rule_output.final_state;
        operations::linear(
            gated.reshape([batch, sequence, value_width]),
            &self.out_proj,
        )
    }

    /// Reconstructs the allocation topology without inspecting parameter values.
    pub fn config(&self) -> GatedDeltaNetConfig {
        GatedDeltaNetConfig {
            hidden_size: self.hidden_size,
            key_heads: self.key_heads,
            value_heads: self.value_heads,
            key_head_dim: self.key_head_dim,
            value_head_dim: self.value_head_dim,
            convolution_kernel: self.convolution_kernel,
            normalization_epsilon: self.norm.epsilon,
            prefill_chunk_size: self.rule_config.chunk_size,
        }
    }

    /// Applies valid convolution to history plus new channels and retains its tail.
    fn causal_convolution(
        &self,
        projected: Tensor<B, 3>,
        history: Tensor<B, 3>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let sequence = projected.dims()[2];
        let history_width = self.convolution_kernel - 1;
        // Recurrent matrices intentionally accumulate in FP32, but convolution
        // history follows the projection dtype. Casting here also makes a state
        // created before checkpoint dtype selection safe on its first call.
        let history = history.cast(projected.dtype());
        let joined = Tensor::cat(vec![history, projected], 2);
        let convolved = conv1d(
            joined.clone(),
            self.conv1d.weight.val(),
            None,
            ConvOptions::new([1], [0], [1], self.conv1d.groups),
        );
        let next_history = joined.narrow(2, sequence, history_width);
        (convolved, next_history)
    }

    /// Ensures external state axes match this layer before backend dispatch.
    fn validate_state(&self, state: &GatedDeltaNetState<B>, batch: usize) -> Result<()> {
        let convolution_channels = self.convolution_channels;
        let expected_convolution = [batch, convolution_channels, self.convolution_kernel - 1];
        if state.convolution.dims() != expected_convolution {
            return Err(Error::shape(
                "delta_net.state.convolution",
                format!("{expected_convolution:?}"),
                format!("{:?}", state.convolution.dims()),
            ));
        }
        let expected_recurrent = [
            batch,
            self.value_heads,
            self.key_head_dim,
            self.value_head_dim,
        ];
        if state.recurrent.dims() != expected_recurrent {
            return Err(Error::shape(
                "delta_net.state.recurrent",
                format!("{expected_recurrent:?}"),
                format!("{:?}", state.recurrent.dims()),
            ));
        }
        Ok(())
    }
}

/// Zeros padded residual-stream vectors without changing activation precision.
///
/// The semantic mask is embedded as `m[b,t] in {0,1}` and applies
/// `x'[b,t,d] = x[b,t,d] * m[b,t]`. Integer-to-float conversion uses a backend's
/// default float type, so the mask is explicitly restored to `x`'s dtype before
/// multiplication; otherwise an F16 inference graph can be silently promoted to
/// F32 merely because an all-valid mask was supplied.
fn mask_hidden<B: Backend>(
    hidden: Tensor<B, 3>,
    token_mask: Option<&TokenMask<B>>,
) -> Tensor<B, 3> {
    let Some(mask) = token_mask else {
        return hidden;
    };
    let dtype = hidden.dtype();
    hidden
        * mask
            .as_tensor()
            .clone()
            .int()
            .float()
            .cast(dtype)
            .unsqueeze_dim::<3>(2)
}

/// Applies Qwen's output RMSNorm before its SiLU gate with exact cast points.
///
/// The reference implementation computes variance in FP32, casts the normalized
/// value back to the activation dtype before multiplying the learned scale, then
/// promotes the product for the FP32 SiLU gate and finally restores the activation
/// dtype. These intermediate roundings are observable in F16 acceptance logits.
fn gated_delta_norm<B: Backend>(
    values: Tensor<B, 4>,
    gate: Tensor<B, 4>,
    norm: &crate::normalization::RmsNorm<B>,
) -> Tensor<B, 4> {
    let dtype = values.dtype();
    let values = values.cast(DType::F32);
    let denominator = (values.clone().square().mean_dim(3) + norm.epsilon).sqrt();
    let normalized = (values / denominator).cast(dtype);
    let weighted = normalized * norm.weight.val().cast(dtype).unsqueeze();
    (weighted.cast(DType::F32) * activation::silu(gate.cast(DType::F32))).cast(dtype)
}

/// Repeats normalized key heads into contiguous value-head groups.
fn repeat_delta_heads<B: Backend>(tensor: Tensor<B, 4>, value_heads: usize) -> Tensor<B, 4> {
    let [batch, sequence, key_heads, head_dim] = tensor.dims();
    let repetitions = value_heads / key_heads;
    if repetitions == 1 {
        return tensor;
    }
    tensor
        .unsqueeze_dim::<5>(3)
        .expand([batch, sequence, key_heads, repetitions, head_dim])
        .reshape([batch, sequence, value_heads, head_dim])
}

/// Evaluates the exact sequential gated delta recurrence.
///
/// Reductions and recurrent state updates execute in FP32 even when projections
/// use F16/BF16. The returned values are cast back to the query dtype; the state
/// intentionally remains FP32 for stable long-context decode.
pub fn recurrent_gated_delta_rule<B: Backend>(
    input: DeltaRuleInput<B>,
    config: DeltaRuleConfig,
) -> Result<DeltaRuleOutput<B>> {
    config.validate()?;
    let prepared = PreparedDeltaRule::new(input, config.normalization_epsilon)?;
    let [batch, heads, sequence, key_width] = prepared.query.dims();
    let value_width = prepared.value.dims()[3];
    let mut state = prepared.initial_state.unwrap_or_else(|| {
        Tensor::<B, 4>::zeros([batch, heads, key_width, value_width], &prepared.device)
            .cast(DType::F32)
    });
    let mut outputs = Vec::with_capacity(sequence);

    for index in 0..sequence {
        let query = prepared.query.clone().narrow(2, index, 1);
        let key = prepared.key.clone().narrow(2, index, 1);
        let value = prepared.value.clone().narrow(2, index, 1);
        let beta = prepared
            .beta
            .clone()
            .narrow(2, index, 1)
            .unsqueeze_dim::<4>(3);
        let decay = prepared
            .log_decay
            .clone()
            .narrow(2, index, 1)
            .exp()
            .unsqueeze_dim::<4>(3);

        state = state * decay;
        let retrieved = key.clone().matmul(state.clone());
        let delta = (value - retrieved) * beta;
        state = state + key.swap_dims(2, 3).matmul(delta);
        outputs.push(query.matmul(state.clone()));
    }

    let values = Tensor::cat(outputs, 2)
        .swap_dims(1, 2)
        .cast(prepared.output_dtype);
    Ok(DeltaRuleOutput {
        values,
        final_state: state,
    })
}

/// Evaluates a blockwise-parallel formulation for prompt prefill.
///
/// Within a chunk, lower-triangular delta interactions are solved once. Each
/// chunk then reads the incoming recurrent state with one matrix product and
/// emits its outgoing state with another. Only the number of chunks remains
/// sequential, reducing a length-`L` dependency chain to roughly
/// `ceil(L / chunk_size)` steps.
pub fn chunked_gated_delta_rule<B: Backend>(
    input: DeltaRuleInput<B>,
    config: DeltaRuleConfig,
) -> Result<DeltaRuleOutput<B>> {
    config.validate()?;
    let prepared = PreparedDeltaRule::new(input, config.normalization_epsilon)?;
    let [batch, heads, sequence, key_width] = prepared.query.dims();
    let value_width = prepared.value.dims()[3];
    let mut state = prepared.initial_state.unwrap_or_else(|| {
        Tensor::<B, 4>::zeros([batch, heads, key_width, value_width], &prepared.device)
            .cast(DType::F32)
    });
    let mut outputs = Vec::with_capacity(sequence.div_ceil(config.chunk_size));

    for start in (0..sequence).step_by(config.chunk_size) {
        let width = config.chunk_size.min(sequence - start);
        let query = prepared.query.clone().narrow(2, start, width);
        let key = prepared.key.clone().narrow(2, start, width);
        let value = prepared.value.clone().narrow(2, start, width);
        let beta = prepared.beta.clone().narrow(2, start, width);
        let cumulative_decay = prepared.log_decay.clone().narrow(2, start, width).cumsum(2);

        let beta_expanded = beta.unsqueeze_dim::<4>(3);
        let value_beta = value.clone() * beta_expanded.clone();
        let key_beta = key.clone() * beta_expanded;
        let decay_difference = cumulative_decay.clone().unsqueeze_dim::<4>(3)
            - cumulative_decay.clone().unsqueeze_dim::<4>(2);
        let decay_matrix = decay_difference.tril(0).exp().tril(0);

        // Strictly lower coefficients encode how an earlier delta write changes
        // the value retrieved by each later key in the same block.
        let mut inverse_delta = (-(key_beta.clone().matmul(key.clone().swap_dims(2, 3)))
            * decay_matrix.clone())
        .tril(-1);
        for row in 1..width {
            let row_values =
                inverse_delta
                    .clone()
                    .slice([0..batch, 0..heads, row..row + 1, 0..row]);
            let solved_prefix = inverse_delta
                .clone()
                .slice([0..batch, 0..heads, 0..row, 0..row]);
            let solved_row = row_values.clone() + row_values.matmul(solved_prefix);
            inverse_delta =
                inverse_delta.slice_assign([0..batch, 0..heads, row..row + 1, 0..row], solved_row);
        }
        let identity = Tensor::<B, 2>::eye(width, &prepared.device)
            .cast(DType::F32)
            .reshape([1, 1, width, width])
            .expand([batch, heads, width, width]);
        let delta_transform = inverse_delta + identity;
        let transformed_value = delta_transform.clone().matmul(value_beta);
        let decayed_key_beta = key_beta * cumulative_decay.clone().exp().unsqueeze_dim::<4>(3);
        let cumulative_key = delta_transform.matmul(decayed_key_beta);

        let corrected_value = transformed_value - cumulative_key.matmul(state.clone());
        let state_output = (query.clone() * cumulative_decay.clone().exp().unsqueeze_dim::<4>(3))
            .matmul(state.clone());
        let causal_attention = (query.matmul(key.clone().swap_dims(2, 3)) * decay_matrix).tril(0);
        outputs.push(state_output + causal_attention.matmul(corrected_value.clone()));

        let final_decay = cumulative_decay.clone().narrow(2, width - 1, 1);
        let decayed_keys = key
            * (final_decay.clone() - cumulative_decay)
                .exp()
                .unsqueeze_dim::<4>(3);
        state = state * final_decay.exp().unsqueeze_dim::<4>(3)
            + decayed_keys.swap_dims(2, 3).matmul(corrected_value);
    }

    let values = Tensor::cat(outputs, 2)
        .swap_dims(1, 2)
        .cast(prepared.output_dtype);
    Ok(DeltaRuleOutput {
        values,
        final_state: state,
    })
}

/// Validated FP32 tensors in head-major layout used by both algorithms.
struct PreparedDeltaRule<B: Backend> {
    query: Tensor<B, 4>,
    key: Tensor<B, 4>,
    value: Tensor<B, 4>,
    log_decay: Tensor<B, 3>,
    beta: Tensor<B, 3>,
    initial_state: Option<Tensor<B, 4>>,
    output_dtype: DType,
    device: B::Device,
}

impl<B: Backend> PreparedDeltaRule<B> {
    /// Validates axes, masks padding updates, normalizes Q/K, and casts to FP32.
    fn new(input: DeltaRuleInput<B>, epsilon: f64) -> Result<Self> {
        let query_shape = input.query.dims();
        let key_shape = input.key.dims();
        let value_shape = input.value.dims();
        let [batch, sequence, heads, key_width] = query_shape;
        if batch == 0 || sequence == 0 || heads == 0 || key_width == 0 {
            return Err(Error::shape(
                "delta_net.query",
                "non-empty [batch, sequence, heads, key_width]",
                format!("{query_shape:?}"),
            ));
        }
        if key_shape != query_shape {
            return Err(Error::shape(
                "delta_net.key",
                format!("{query_shape:?}"),
                format!("{key_shape:?}"),
            ));
        }
        if value_shape[0..3] != query_shape[0..3] || value_shape[3] == 0 {
            return Err(Error::shape(
                "delta_net.value",
                format!("[{batch}, {sequence}, {heads}, value_width>0]"),
                format!("{value_shape:?}"),
            ));
        }
        validate_rank3_shape(
            "delta_net.log_decay",
            input.log_decay.dims(),
            [batch, sequence, heads],
        )?;
        validate_rank3_shape(
            "delta_net.beta",
            input.beta.dims(),
            [batch, sequence, heads],
        )?;
        if let Some(state) = &input.initial_state {
            let expected = [batch, heads, key_width, value_shape[3]];
            if state.dims() != expected {
                return Err(Error::shape(
                    "delta_net.initial_state",
                    format!("{expected:?}"),
                    format!("{:?}", state.dims()),
                ));
            }
        }
        if let Some(mask) = &input.token_mask
            && mask.dims() != [batch, sequence]
        {
            return Err(Error::shape(
                "delta_net.token_mask",
                format!("[{batch}, {sequence}]"),
                format!("{:?}", mask.dims()),
            ));
        }

        let output_dtype = input.query.dtype();
        let device = input.query.device();
        let mut query =
            l2_normalize(input.query.cast(DType::F32), epsilon) / (key_width as f64).sqrt();
        let mut key = l2_normalize(input.key.cast(DType::F32), epsilon);
        let mut value = input.value.cast(DType::F32);
        let mut log_decay = input.log_decay.cast(DType::F32);
        let mut beta = input.beta.cast(DType::F32);

        if let Some(mask) = input.token_mask {
            let head_mask = mask.clone().unsqueeze_dim::<3>(2).swap_dims(1, 2);
            let value_mask = head_mask.clone().int().float().unsqueeze_dim::<4>(3);
            query = query.swap_dims(1, 2) * value_mask.clone();
            key = key.swap_dims(1, 2) * value_mask.clone();
            value = value.swap_dims(1, 2) * value_mask;
            beta = beta.swap_dims(1, 2) * head_mask.clone().int().float();
            log_decay = log_decay
                .swap_dims(1, 2)
                .mask_fill(head_mask.bool_not(), 0.0);
        } else {
            query = query.swap_dims(1, 2);
            key = key.swap_dims(1, 2);
            value = value.swap_dims(1, 2);
            beta = beta.swap_dims(1, 2);
            log_decay = log_decay.swap_dims(1, 2);
        }

        Ok(Self {
            query,
            key,
            value,
            log_decay,
            beta,
            initial_state: input.initial_state.map(|state| state.cast(DType::F32)),
            output_dtype,
            device,
        })
    }
}

/// Normalizes the last axis using the FLA-compatible sum-of-squares convention.
fn l2_normalize<B: Backend>(input: Tensor<B, 4>, epsilon: f64) -> Tensor<B, 4> {
    let inverse_norm = (input.clone().square().sum_dim(3) + epsilon).sqrt().recip();
    input * inverse_norm
}

/// Compares one rank-three shape while preserving a tensor-specific error path.
fn validate_rank3_shape(tensor: &str, actual: [usize; 3], expected: [usize; 3]) -> Result<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(Error::shape(
            tensor,
            format!("{expected:?}"),
            format!("{actual:?}"),
        ))
    }
}

#[cfg(test)]
mod tests {
    use burn::{
        backend::Flex,
        tensor::{Bool, DType, Int, Tensor, TensorData, Tolerance, backend::Backend},
    };

    use crate::architecture::TokenMask;

    use super::{
        DeltaNetMode, DeltaRuleConfig, DeltaRuleInput, GatedDeltaNetConfig, GatedDeltaNetState,
        chunked_gated_delta_rule, mask_hidden, recurrent_gated_delta_rule,
    };

    type TestBackend = Flex<f32, i32>;

    /// Supplying an all-valid mask cannot promote an F16 residual stream to F32.
    #[test]
    fn all_valid_mask_preserves_activation_dtype_and_values() {
        let device = Default::default();
        let hidden = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(vec![0.5_f32, -1.0, 2.0, -3.0], [1, 2, 2]),
            &device,
        )
        .cast(DType::F16);
        let mask = TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
            [[true, true]].into(),
            &device,
        ));
        let masked = mask_hidden(hidden.clone(), Some(&mask));

        assert_eq!(masked.dtype(), DType::F16);
        masked.cast(DType::F32).into_data().assert_approx_eq::<f32>(
            &hidden.cast(DType::F32).into_data(),
            Tolerance::absolute(0.0),
        );
    }

    /// Builds deterministic nontrivial inputs spanning two heads and unequal widths.
    fn fixture(sequence: usize) -> DeltaRuleInput<TestBackend> {
        let device = Default::default();
        let query = Tensor::arange(0..(2 * sequence * 2 * 3) as i64, &device)
            .float()
            .reshape([2, sequence, 2, 3])
            / 19.0
            - 0.7;
        let key = Tensor::arange(1..(2 * sequence * 2 * 3 + 1) as i64, &device)
            .float()
            .reshape([2, sequence, 2, 3])
            / 23.0
            - 0.4;
        let value = Tensor::arange(0..(2 * sequence * 2 * 2) as i64, &device)
            .float()
            .reshape([2, sequence, 2, 2])
            / 11.0
            - 0.2;
        let log_decay = Tensor::full([2, sequence, 2], -0.17, &device);
        let beta = Tensor::full([2, sequence, 2], 0.63, &device);
        DeltaRuleInput {
            query,
            key,
            value,
            log_decay,
            beta,
            initial_state: None,
            token_mask: None,
        }
    }

    /// Every chunk geometry matches the transparent recurrent definition.
    #[test]
    fn chunked_prefill_matches_recurrence() {
        for sequence in 1..=9 {
            for chunk_size in 1..=6 {
                let config = DeltaRuleConfig {
                    chunk_size,
                    ..Default::default()
                };
                let recurrent = recurrent_gated_delta_rule(fixture(sequence), config)
                    .expect("fixture must be valid");
                let chunked = chunked_gated_delta_rule(fixture(sequence), config)
                    .expect("fixture must be valid");
                chunked.values.into_data().assert_approx_eq::<f32>(
                    &recurrent.values.into_data(),
                    Tolerance::rel_abs(1e-4, 1e-5),
                );
                chunked.final_state.into_data().assert_approx_eq::<f32>(
                    &recurrent.final_state.into_data(),
                    Tolerance::rel_abs(1e-4, 1e-5),
                );
            }
        }
    }

    /// Passing the first segment's state reproduces a single uninterrupted scan.
    #[test]
    fn recurrent_state_composes_across_decode_calls() {
        let complete = recurrent_gated_delta_rule(fixture(4), Default::default())
            .expect("fixture must be valid");
        let base = fixture(4);
        let first = recurrent_gated_delta_rule(
            DeltaRuleInput {
                query: base.query.clone().narrow(1, 0, 2),
                key: base.key.clone().narrow(1, 0, 2),
                value: base.value.clone().narrow(1, 0, 2),
                log_decay: base.log_decay.clone().narrow(1, 0, 2),
                beta: base.beta.clone().narrow(1, 0, 2),
                initial_state: None,
                token_mask: None,
            },
            Default::default(),
        )
        .expect("first segment must be valid");
        let second = recurrent_gated_delta_rule(
            DeltaRuleInput {
                query: base.query.narrow(1, 2, 2),
                key: base.key.narrow(1, 2, 2),
                value: base.value.narrow(1, 2, 2),
                log_decay: base.log_decay.narrow(1, 2, 2),
                beta: base.beta.narrow(1, 2, 2),
                initial_state: Some(first.final_state),
                token_mask: None,
            },
            Default::default(),
        )
        .expect("second segment must be valid");
        let expected_suffix = complete.values.narrow(1, 2, 2);
        second
            .values
            .into_data()
            .assert_approx_eq::<f32>(&expected_suffix.into_data(), Tolerance::rel_abs(1e-4, 1e-5));
        second.final_state.into_data().assert_approx_eq::<f32>(
            &complete.final_state.into_data(),
            Tolerance::rel_abs(1e-4, 1e-5),
        );
    }

    /// Invalid tokens emit zero and leave recurrent memory unchanged.
    #[test]
    fn token_mask_blocks_state_updates() {
        let device = Default::default();
        let mut masked = fixture(3);
        masked.token_mask = Some(Tensor::<TestBackend, 2, Bool>::from_bool(
            [[true, false, true], [true, false, true]].into(),
            &device,
        ));
        let output = recurrent_gated_delta_rule(masked, Default::default())
            .expect("masked fixture must be valid");
        let middle = output.values.narrow(1, 1, 1).into_data();
        middle.assert_approx_eq::<f32>(
            &TensorData::zeros::<f32, _>([2, 1, 2, 2]),
            Tolerance::absolute(1e-7),
        );
    }

    /// Every public tensor extent is checked before reshape or matmul dispatch.
    #[test]
    fn shape_validation_is_contextual() {
        let device = Default::default();
        let mut input = fixture(2);
        input.key = Tensor::zeros([2, 2, 2, 4], &device);
        assert!(
            recurrent_gated_delta_rule(input, Default::default())
                .expect_err("wrong key width must fail")
                .to_string()
                .contains("delta_net.key")
        );

        let mut input = fixture(2);
        input.beta = Tensor::zeros([2, 3, 2], &device);
        assert!(
            chunked_gated_delta_rule(input, Default::default())
                .expect_err("wrong beta sequence must fail")
                .to_string()
                .contains("delta_net.beta")
        );

        let mut input = fixture(2);
        input.initial_state = Some(Tensor::zeros([2, 2, 4, 2], &device));
        assert!(
            recurrent_gated_delta_rule(input, Default::default())
                .expect_err("wrong state key width must fail")
                .to_string()
                .contains("initial_state")
        );

        let mut input = fixture(2);
        input.token_mask = Some(Tensor::<TestBackend, 2, Int>::ones([2, 3], &device).bool());
        assert!(
            recurrent_gated_delta_rule(input, Default::default())
                .expect_err("wrong mask sequence must fail")
                .to_string()
                .contains("token_mask")
        );
    }

    /// Numerical and block controls reject non-finite or degenerate values.
    #[test]
    fn config_validation_rejects_invalid_controls() {
        assert!(
            DeltaRuleConfig {
                normalization_epsilon: f64::NAN,
                ..Default::default()
            }
            .validate()
            .is_err()
        );
        assert!(
            DeltaRuleConfig {
                chunk_size: 0,
                ..Default::default()
            }
            .validate()
            .is_err()
        );
    }

    /// The projected module's blockwise prefill equals repeated cached decode.
    #[test]
    fn projected_mixer_prefill_matches_decode() {
        let device = Default::default();
        TestBackend::seed(&device, 7);
        let mut config = GatedDeltaNetConfig::new(4, 1, 2, 2, 2);
        config.convolution_kernel = 3;
        config.prefill_chunk_size = 2;
        let mixer = config
            .init::<TestBackend>(&device)
            .expect("valid mixer must initialize");
        let hidden = Tensor::from_floats(
            [
                [
                    [0.1, -0.2, 0.3, 0.4],
                    [0.5, 0.2, -0.1, 0.7],
                    [-0.4, 0.6, 0.2, -0.3],
                    [0.9, -0.8, 0.1, 0.2],
                    [0.3, 0.5, -0.7, 0.4],
                ],
                [
                    [-0.3, 0.8, 0.4, -0.1],
                    [0.2, -0.5, 0.9, 0.6],
                    [0.1, 0.7, -0.6, 0.3],
                    [-0.8, 0.4, 0.5, 0.2],
                    [0.6, -0.2, 0.1, -0.9],
                ],
            ],
            &device,
        );

        let mut prefill_state =
            GatedDeltaNetState::zeros(&config, 2, &device).expect("valid state");
        let prefill = mixer
            .forward_with_state(
                hidden.clone(),
                None,
                &mut prefill_state,
                DeltaNetMode::ChunkedPrefill,
            )
            .expect("prefill must succeed");

        let mut decode_state = GatedDeltaNetState::zeros(&config, 2, &device).expect("valid state");
        let mut decode_tokens = Vec::new();
        for index in 0..hidden.dims()[1] {
            decode_tokens.push(
                mixer
                    .forward_with_state(
                        hidden.clone().narrow(1, index, 1),
                        None,
                        &mut decode_state,
                        DeltaNetMode::Recurrent,
                    )
                    .expect("decode step must succeed"),
            );
        }
        let decode = Tensor::cat(decode_tokens, 1);
        prefill
            .into_data()
            .assert_approx_eq::<f32>(&decode.into_data(), Tolerance::rel_abs(2e-4, 2e-5));
        prefill_state.recurrent.into_data().assert_approx_eq::<f32>(
            &decode_state.recurrent.into_data(),
            Tolerance::rel_abs(2e-4, 2e-5),
        );
        prefill_state
            .convolution
            .into_data()
            .assert_approx_eq::<f32>(
                &decode_state.convolution.into_data(),
                Tolerance::rel_abs(2e-4, 2e-5),
            );
    }

    /// Model topology validation rejects grouping and allocation overflows.
    #[test]
    fn projected_config_rejects_invalid_topology() {
        assert!(GatedDeltaNetConfig::new(4, 3, 4, 2, 2).validate().is_err());
        assert!(GatedDeltaNetConfig::new(0, 1, 1, 1, 1).validate().is_err());
        assert!(
            GatedDeltaNetConfig::new(1, usize::MAX, usize::MAX, usize::MAX, usize::MAX)
                .validate()
                .is_err()
        );
    }

    /// External state dimensions are checked before convolution or recurrence.
    #[test]
    fn projected_mixer_rejects_wrong_state_shape() {
        let device = Default::default();
        let config = GatedDeltaNetConfig::new(4, 1, 2, 2, 2);
        let mixer = config
            .init::<TestBackend>(&device)
            .expect("valid mixer must initialize");
        let mut state = GatedDeltaNetState::zeros(&config, 1, &device).expect("valid state");
        state.recurrent = Tensor::zeros([1, 2, 3, 2], &device);
        let error = mixer
            .forward_with_state(
                Tensor::zeros([1, 1, 4], &device),
                None,
                &mut state,
                DeltaNetMode::Recurrent,
            )
            .expect_err("wrong recurrent width must fail");
        assert!(error.to_string().contains("state.recurrent"));
    }

    /// Malformed input axes fail before either nonzero state buffer is replaced.
    #[test]
    fn projected_mixer_shape_rejections_preserve_both_state_buffers() {
        let device = Default::default();
        let config = GatedDeltaNetConfig::new(4, 1, 2, 2, 2);
        let mixer = config
            .init::<TestBackend>(&device)
            .expect("valid mixer must initialize");
        let valid_convolution = [1, config.convolution_channels().unwrap(), 3];
        let cases = [
            ("delta_net.hidden", [0, 2, 4], None, valid_convolution),
            ("delta_net.hidden", [1, 0, 4], None, valid_convolution),
            ("delta_net.hidden", [1, 2, 3], None, valid_convolution),
            (
                "delta_net.token_mask",
                [1, 2, 4],
                Some([2, 2]),
                valid_convolution,
            ),
            (
                "delta_net.token_mask",
                [1, 2, 4],
                Some([1, 3]),
                valid_convolution,
            ),
            ("delta_net.state.convolution", [1, 2, 4], None, [2, 8, 3]),
            ("delta_net.state.convolution", [1, 2, 4], None, [1, 9, 3]),
            ("delta_net.state.convolution", [1, 2, 4], None, [1, 8, 2]),
        ];
        for mode in [DeltaNetMode::Recurrent, DeltaNetMode::ChunkedPrefill] {
            for (context, hidden_shape, mask_shape, convolution_shape) in cases {
                let mut state =
                    GatedDeltaNetState::zeros(&config, 1, &device).expect("valid state");
                state.convolution = Tensor::full(convolution_shape, 0.25, &device);
                state.recurrent = Tensor::full([1, 2, 2, 2], -0.5, &device);
                // Own independent host bytes: a shared tensor clone alone could
                // conceal an accidental in-place mutation of the same storage.
                let convolution_before = TensorData::new(
                    vec![0.25_f32; convolution_shape.iter().product()],
                    convolution_shape,
                );
                let recurrent_before = TensorData::new(vec![-0.5_f32; 8], [1, 2, 2, 2]);
                let mask = mask_shape.map(|shape| {
                    TokenMask::new(Tensor::<TestBackend, 2, Int>::ones(shape, &device).bool())
                });
                let error = mixer
                    .forward_with_state(
                        Tensor::zeros(hidden_shape, &device),
                        mask.as_ref(),
                        &mut state,
                        mode,
                    )
                    .expect_err("invalid shape must fail before computation");
                let crate::Error::Shape {
                    tensor,
                    expected,
                    actual,
                } = error
                else {
                    panic!("expected contextual shape error for {context}: {error}");
                };
                assert_eq!(tensor, context);
                let (expected_shape, actual_shape) = match context {
                    "delta_net.hidden" => (
                        "[batch>0, sequence>0, 4]".to_owned(),
                        format!("{hidden_shape:?}"),
                    ),
                    "delta_net.token_mask" => {
                        ("[1, 2]".to_owned(), format!("{:?}", mask_shape.unwrap()))
                    }
                    _ => (
                        format!("{valid_convolution:?}"),
                        format!("{convolution_shape:?}"),
                    ),
                };
                assert_eq!(expected, expected_shape);
                assert_eq!(actual, actual_shape);
                state
                    .convolution
                    .into_data()
                    .assert_eq(&convolution_before, true);
                state
                    .recurrent
                    .into_data()
                    .assert_eq(&recurrent_before, true);
            }
        }
    }
}
