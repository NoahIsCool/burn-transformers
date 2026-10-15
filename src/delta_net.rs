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
//! exposing substantially more parallelism during prompt prefill: every
//! chunk's triangular solve and intra-chunk products run batched across up to
//! [`CHUNKS_PER_BLOCK`] chunks, and only the three-product state carry between
//! chunks remains sequential.
//!
//! The projected [`GatedDeltaNet`] mixer adds a depthwise causal convolution
//! before the rule and a gated RMS normalization after it. Padding (a `false`
//! token mask) is removed from both: padded tokens neither update the matrix
//! state nor occupy convolution taps or history, so a padded row produces the
//! same valid outputs and state as its compacted, unpadded form.

use burn::{
    module::{Initializer, Module, Param},
    nn::{
        Linear, LinearConfig, LinearLayout, PaddingConfig1d,
        conv::{Conv1d, Conv1dConfig},
    },
    tensor::{
        Bool, DType, Distribution, FloatDType, IndexingUpdateOp, Int, Tensor, activation,
        backend::Backend, module::conv1d, ops::ConvOptions,
    },
};
use serde::{Deserialize, Serialize};

use crate::{Error, Result, architecture::TokenMask, normalization::RmsNormConfig, operations};

/// Number of prefill chunks whose intra-chunk algebra is batched together.
///
/// Larger blocks issue fewer kernels but hold `[batch, heads, block, C, C]`
/// F32 matrices; 16 chunks of 64 tokens keep that at 256 KiB per head and
/// batch row while cutting the triangular-solve launches sixteenfold. The
/// launch savings target GPU backends; on the scalar Flex CPU backend the
/// arithmetic is unchanged and the ignored `prefill_timing` benchmark measures
/// batching as roughly neutral for 128-wide heads and faster for narrow ones.
pub const CHUNKS_PER_BLOCK: usize = 16;

/// Default stabilizer of the query/key L2 normalization (Qwen and FLA reference).
const DEFAULT_QK_L2_EPSILON: f64 = 1e-6;

/// Returns [`DEFAULT_QK_L2_EPSILON`] for configurations serialized before the field existed.
const fn default_qk_l2_epsilon() -> f64 {
    DEFAULT_QK_L2_EPSILON
}

/// Numerically sensitive controls for both DeltaNet execution paths.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DeltaRuleConfig {
    /// Stabilizer in query/key L2 normalization: `x / sqrt(sum(x^2) + epsilon)`.
    pub normalization_epsilon: f64,
    /// Maximum time-axis block processed by the parallel prefill path.
    pub chunk_size: usize,
}

impl Default for DeltaRuleConfig {
    /// Uses the official Qwen reference constants.
    fn default() -> Self {
        Self {
            normalization_epsilon: DEFAULT_QK_L2_EPSILON,
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
    /// Width of the depthwise causal convolution; one disables temporal mixing.
    pub convolution_kernel: usize,
    /// Stabilizer of the output head-wise RMS normalization only.
    ///
    /// The query/key L2 normalization uses [`Self::qk_l2_epsilon`].
    pub normalization_epsilon: f64,
    /// Chunk size used by the parallel prompt path.
    pub prefill_chunk_size: usize,
    /// Stabilizer of the query/key L2 normalization inside the delta rule.
    #[serde(default = "default_qk_l2_epsilon")]
    pub qk_l2_epsilon: f64,
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
            qk_l2_epsilon: DEFAULT_QK_L2_EPSILON,
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
    ///
    /// # Errors
    ///
    /// Returns a contextual configuration error for zero dimensions or kernel,
    /// value heads that are not a multiple of key heads, a non-finite or
    /// non-positive output-norm or query/key epsilon, a zero chunk size, or
    /// projection widths that overflow `usize`.
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
        for (field, epsilon) in [
            (
                "delta_net.normalization_epsilon",
                self.normalization_epsilon,
            ),
            ("delta_net.qk_l2_epsilon", self.qk_l2_epsilon),
        ] {
            if !epsilon.is_finite() || epsilon <= 0.0 {
                return Err(Error::configuration(
                    field,
                    "must be finite and greater than zero",
                ));
            }
        }
        self.rule_config().validate()?;
        let _ = self.convolution_channels()?;
        self.value_heads
            .checked_mul(self.value_head_dim)
            .ok_or_else(|| {
                Error::configuration("delta_net.value_width", "dimension overflows usize")
            })?;
        Ok(())
    }

    /// Returns the delta-rule controls: query/key epsilon and prefill chunk size.
    #[must_use]
    pub const fn rule_config(&self) -> DeltaRuleConfig {
        DeltaRuleConfig {
            normalization_epsilon: self.qk_l2_epsilon,
            chunk_size: self.prefill_chunk_size,
        }
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
            rule_config: self.rule_config(),
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
                (device, DType::F32),
            ),
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
    ///
    /// A token mask removes padded tokens from the computation: they are
    /// zeroed before projection, excluded from the causal convolution's taps
    /// and stored history (valid tokens of each row are compacted before the
    /// convolution and scattered back after it), and they neither decay nor
    /// update the recurrent state. Padded positions return zero. Left, right,
    /// and interior padding therefore all reproduce the unpadded row.
    ///
    /// # Errors
    ///
    /// Returns a shape error for an empty or wrongly sized hidden tensor, a
    /// mismatched mask, or state buffers that do not match this layer.
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
        let (convolved, convolution_state) = self.causal_convolution(
            projected,
            state.convolution.clone(),
            token_mask.map(|mask| mask.as_tensor().clone()),
        );
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
        // g = -exp(A_log) * softplus(a + dt_bias), with PyTorch's overflow-safe softplus.
        let log_decay = -(decay_rate * softplus(a + time_bias));
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
            qk_l2_epsilon: self.rule_config.normalization_epsilon,
        }
    }

    /// Applies the depthwise causal convolution and returns its new history.
    ///
    /// `projected` is `[batch, channels, sequence]` and `history` holds the
    /// previous `kernel - 1` columns. The convolution runs over
    /// `[history | tokens]` without padding, so output `t` mixes the `kernel`
    /// most recent columns. With a token mask, each row's valid tokens are
    /// first moved to the front of the timeline (order preserved); outputs are
    /// scattered back to their original positions and the new history is the
    /// last `kernel - 1` columns of `[history | valid tokens]`. A width-one
    /// kernel has no history and reduces to a per-channel scale.
    fn causal_convolution(
        &self,
        projected: Tensor<B, 3>,
        history: Tensor<B, 3>,
        token_mask: Option<Tensor<B, 2, Bool>>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let [batch, channels, sequence] = projected.dims();
        let history_width = self.convolution_kernel - 1;
        let convolve = |input: Tensor<B, 3>| {
            conv1d(
                input,
                self.conv1d.weight.val(),
                None,
                ConvOptions::new([1], [0], [1], self.conv1d.groups),
            )
        };
        if history_width == 0 {
            // No temporal taps: the empty history is preserved as-is and
            // padding cannot leak into neighbouring tokens.
            let empty = Tensor::zeros(
                [batch, channels, 0],
                (&projected.device(), projected.dtype()),
            );
            return (convolve(projected), empty);
        }
        // Recurrent matrices intentionally accumulate in FP32, but convolution
        // history follows the projection dtype. Casting here also makes a state
        // created before checkpoint dtype selection safe on its first call.
        let history = history.cast(projected.dtype());
        let Some(mask) = token_mask else {
            let joined = Tensor::cat(vec![history, projected], 2);
            let next_history = joined.clone().narrow(2, sequence, history_width);
            return (convolve(joined), next_history);
        };
        let compaction = TokenCompaction::new(mask);
        let joined = Tensor::cat(vec![history, compaction.compact(projected, channels)], 2);
        let next_history = compaction.history(joined.clone(), channels, history_width);
        let convolved = compaction.restore(convolve(joined), channels);
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

/// Input above which PyTorch's `softplus` returns its argument unchanged.
const SOFTPLUS_THRESHOLD: f64 = 20.0;

/// PyTorch's default `softplus`: `x` when `x > 20`, otherwise `log1p(exp(x))`.
///
/// Burn 0.21's `activation::softplus` evaluates `log(exp(x) + 1)`, which
/// overflows to infinity for `x > 88.7` in F32. A Gated DeltaNet decay
/// `g = -A * softplus(a + dt_bias)` would then be `-inf`, and the chunked
/// prefill's decay differences `g_i - g_j` would become NaN and poison the
/// state. Clamping before `exp` keeps the unselected branch finite, so both
/// the value and its gradient (`sigmoid(x)`, or one above the threshold) stay
/// finite for every finite input. `log1p` is also accurate for very negative
/// inputs, where `log(exp(x) + 1)` rounds to zero.
pub(crate) fn softplus<B: Backend, const D: usize>(input: Tensor<B, D>) -> Tensor<B, D> {
    let linear = input.clone().greater_elem(SOFTPLUS_THRESHOLD);
    input
        .clone()
        .clamp_max(SOFTPLUS_THRESHOLD)
        .exp()
        .log1p()
        .mask_where(linear, input)
}

/// Device-side, order-preserving permutation that moves each row's valid tokens first.
///
/// For row `b` with `n_b` valid tokens, valid token `t` moves to position
/// `valid_before[b, t]` and padded token `t` to `n_b + padded_before[b, t]`,
/// where both counts are exclusive prefix sums. Building the permutation needs
/// one integer cumulative sum and one scatter; no host synchronization.
struct TokenCompaction<B: Backend> {
    /// `[batch, sequence]` compacted position of every original token.
    destination: Tensor<B, 2, Int>,
    /// `[batch, sequence]` original token placed at every compacted position.
    source: Tensor<B, 2, Int>,
    /// `[batch, 1]` number of valid tokens in each row.
    valid_counts: Tensor<B, 2, Int>,
}

impl<B: Backend> TokenCompaction<B> {
    /// Builds the permutation from a `[batch, sequence]` mask where `true` means valid.
    fn new(mask: Tensor<B, 2, Bool>) -> Self {
        let [batch, sequence] = mask.dims();
        let device = mask.device();
        let valid = mask.clone().int();
        let valid_before = valid.clone().cumsum(1) - valid.clone();
        let valid_counts = valid.sum_dim(1);
        let positions = Tensor::<B, 1, Int>::arange(0..sequence as i64, &device)
            .reshape([1, sequence])
            .expand([batch, sequence]);
        // Padded token t has t - valid_before[t] padded tokens before it.
        let padded_destination = valid_counts.clone().expand([batch, sequence]) + positions.clone()
            - valid_before.clone();
        let destination = padded_destination.mask_where(mask, valid_before);
        // `destination` is a permutation, so scatter-adding into zeros inverts it.
        let source = Tensor::<B, 2, Int>::zeros([batch, sequence], &device).scatter(
            1,
            destination.clone(),
            positions,
            IndexingUpdateOp::Add,
        );
        Self {
            destination,
            source,
            valid_counts,
        }
    }

    /// Reorders `[batch, channels, sequence]` columns into compacted order.
    fn compact(&self, values: Tensor<B, 3>, channels: usize) -> Tensor<B, 3> {
        let [batch, sequence] = self.source.dims();
        let index = self
            .source
            .clone()
            .unsqueeze_dim::<3>(1)
            .expand([batch, channels, sequence]);
        values.gather(2, index)
    }

    /// Returns compacted `[batch, channels, sequence]` columns to their original positions.
    fn restore(&self, values: Tensor<B, 3>, channels: usize) -> Tensor<B, 3> {
        let [batch, sequence] = self.destination.dims();
        let index = self
            .destination
            .clone()
            .unsqueeze_dim::<3>(1)
            .expand([batch, channels, sequence]);
        values.gather(2, index)
    }

    /// Reads the last `width` columns of `[history | valid tokens]` from a compacted timeline.
    ///
    /// In `joined = [history (width) | compacted tokens]`, the valid tokens of
    /// row `b` end at column `width + n_b`, so the retained columns start at `n_b`.
    fn history(&self, joined: Tensor<B, 3>, channels: usize, width: usize) -> Tensor<B, 3> {
        let [batch, _] = self.valid_counts.dims();
        let offsets = Tensor::<B, 1, Int>::arange(0..width as i64, &joined.device())
            .reshape([1, width])
            .expand([batch, width])
            + self.valid_counts.clone().expand([batch, width]);
        joined.gather(
            2,
            offsets
                .unsqueeze_dim::<3>(1)
                .expand([batch, channels, width]),
        )
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
        Tensor::<B, 4>::zeros(
            [batch, heads, key_width, value_width],
            (&prepared.device, DType::F32),
        )
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
/// Tokens are split into chunks of `C = chunk_size`. Inside a chunk with
/// cumulative log decays `G_i`, decay matrix `D_ij = exp(G_i - G_j)` (`i >= j`),
/// and `L = strict_lower((beta k) k^T * D)`, all delta writes are resolved at
/// once by `T = (I + L)^-1`, computed by forward substitution. With the
/// incoming state `S`, a chunk then emits
///
/// ```text
/// U   = T (beta v) - T (beta k * exp(G)) S        corrected values
/// out = (q * exp(G)) S + lower(q k^T * D) U
/// S'  = exp(G_last) S + (k * exp(G_last - G))^T U
/// ```
///
/// Everything except the last three lines is independent of `S`, so it runs
/// batched over blocks of up to [`CHUNKS_PER_BLOCK`] chunks: one
/// `C - 1`-step substitution per block instead of per chunk. Inside a
/// multi-chunk block a trailing partial chunk is zero-padded, which is exact
/// because padded tokens have `k = v = beta = 0` and `log_decay = 0`, so they
/// neither change `S` nor the valid rows of `T`. A block that fits in one
/// chunk is never padded: it runs at its own width, so a prompt of `L < C`
/// tokens costs an `L - 1`-step substitution on `L x L` tiles. The sequential
/// dependency is `ceil(L / C)` short state updates.
///
/// # Errors
///
/// Returns configuration or shape errors for invalid controls or inputs.
pub fn chunked_gated_delta_rule<B: Backend>(
    input: DeltaRuleInput<B>,
    config: DeltaRuleConfig,
) -> Result<DeltaRuleOutput<B>> {
    config.validate()?;
    let prepared = PreparedDeltaRule::new(input, config.normalization_epsilon)?;
    let [batch, heads, sequence, key_width] = prepared.query.dims();
    let value_width = prepared.value.dims()[3];
    let mut state = prepared.initial_state.clone().unwrap_or_else(|| {
        Tensor::<B, 4>::zeros(
            [batch, heads, key_width, value_width],
            (&prepared.device, DType::F32),
        )
    });
    let chunk = config.chunk_size;
    let block_tokens = chunk.saturating_mul(CHUNKS_PER_BLOCK);
    // One `[batch, heads, <= C, value]` output per chunk, concatenated once.
    let mut outputs = Vec::with_capacity(sequence.div_ceil(chunk));
    let mut start = 0;
    while start < sequence {
        let width = block_tokens.min(sequence - start);
        let block = ChunkBlock::new(&prepared, start, width, chunk);
        state = block.scan(state, &mut outputs);
        start += width;
    }

    let values = Tensor::cat(outputs, 2)
        .swap_dims(1, 2)
        .cast(prepared.output_dtype);
    Ok(DeltaRuleOutput {
        values,
        final_state: state,
    })
}

/// State-independent terms of up to [`CHUNKS_PER_BLOCK`] consecutive chunks.
///
/// Every tensor is laid out `[batch, heads, chunks, C, ·]`. In a multi-chunk
/// block the trailing chunk is zero-padded to `C` tokens; a single-chunk block
/// uses `C = width` and has no padding.
struct ChunkBlock<B: Backend> {
    /// `T (beta v)`: values with all intra-chunk delta writes resolved.
    transformed_value: Tensor<B, 5>,
    /// `T (beta k * exp(G))`: how each corrected value reads the incoming state.
    cumulative_key: Tensor<B, 5>,
    /// `q * exp(G)`: queries decayed to the chunk start for the state read-out.
    decayed_query: Tensor<B, 5>,
    /// `lower(q k^T * D)`: causal intra-chunk attention weights.
    attention: Tensor<B, 5>,
    /// `k * exp(G_last - G)`: keys decayed to the chunk end for the state write.
    decayed_key: Tensor<B, 5>,
    /// `exp(G_last)` shaped `[batch, heads, chunks, 1]`: state decay across a chunk.
    chunk_decay: Tensor<B, 4>,
    /// Number of real (unpadded) tokens in the block.
    width: usize,
}

impl<B: Backend> ChunkBlock<B> {
    /// Pads, reshapes, and computes every state-independent term for one block.
    ///
    /// A block of at most one chunk (a short prompt, a decode-sized segment, or
    /// the tail of a long prompt) uses its own width as `C`. Padding it to the
    /// configured chunk would run a `C - 1`-step substitution on `C x C` tiles
    /// for a handful of tokens; only multi-chunk blocks share one padded `C`.
    fn new(prepared: &PreparedDeltaRule<B>, start: usize, width: usize, chunk: usize) -> Self {
        let chunk = chunk.min(width);
        let chunks = width.div_ceil(chunk);
        let padded = chunks * chunk;
        let split = |tensor: &Tensor<B, 4>| split_chunks(tensor, start, width, padded, chunk);
        let query = split(&prepared.query);
        let key = split(&prepared.key);
        let value = split(&prepared.value);
        let beta = split(&prepared.beta.clone().unsqueeze_dim::<4>(3));
        let log_decay = split(&prepared.log_decay.clone().unsqueeze_dim::<4>(3));
        // Cumulative log decay G within each chunk, [b, h, n, C, 1].
        let cumulative = log_decay.cumsum(3);
        let decay_to_start = cumulative.clone().exp();
        let final_decay = cumulative.clone().narrow(3, chunk - 1, 1);

        let row_decay = cumulative.clone();
        let column_decay = cumulative.clone().swap_dims(3, 4);
        let decay_matrix = (row_decay - column_decay).tril(0).exp().tril(0);
        let key_beta = key.clone() * beta.clone();
        let strict =
            (key_beta.clone().matmul(key.clone().swap_dims(3, 4)) * decay_matrix.clone()).tril(-1);
        let transform = unit_lower_inverse(strict);
        let attention = (query.clone().matmul(key.clone().swap_dims(3, 4)) * decay_matrix).tril(0);
        let decayed_key = key * (final_decay.clone() - cumulative).exp();
        let [batch, heads, _, _, _] = final_decay.dims();
        Self {
            transformed_value: transform.clone().matmul(value * beta),
            cumulative_key: transform.matmul(key_beta * decay_to_start.clone()),
            decayed_query: query * decay_to_start,
            attention,
            decayed_key,
            chunk_decay: final_decay.exp().reshape([batch, heads, chunks, 1]),
            width,
        }
    }

    /// Carries the recurrent state across the block's chunks in order.
    ///
    /// Appends one `[batch, heads, tokens, value_width]` output per chunk to
    /// `outputs` (the padded tail of a trailing chunk is dropped) and returns
    /// the outgoing FP32 state. Collecting chunks across blocks lets the
    /// caller concatenate the whole sequence with a single copy.
    fn scan(self, mut state: Tensor<B, 4>, outputs: &mut Vec<Tensor<B, 4>>) -> Tensor<B, 4> {
        let [_, _, chunks, chunk, _] = self.attention.dims();
        let take = |tensor: &Tensor<B, 5>, index: usize| {
            tensor.clone().narrow(2, index, 1).squeeze_dim::<4>(2)
        };
        for index in 0..chunks {
            let corrected = take(&self.transformed_value, index)
                - take(&self.cumulative_key, index).matmul(state.clone());
            let output = take(&self.decayed_query, index).matmul(state.clone())
                + take(&self.attention, index).matmul(corrected.clone());
            let real_tokens = chunk.min(self.width - index * chunk);
            outputs.push(if real_tokens == chunk {
                output
            } else {
                output.narrow(2, 0, real_tokens)
            });
            // `[batch, heads, 1, 1]` broadcasts over the `[key, value]` state.
            let decay = self.chunk_decay.clone().narrow(2, index, 1);
            state = state * decay
                + take(&self.decayed_key, index)
                    .swap_dims(2, 3)
                    .matmul(corrected);
        }
        state
    }
}

/// Narrows `[batch, heads, sequence, d]` to a block, zero-pads it, and splits it into chunks.
///
/// Returns `[batch, heads, padded / chunk, chunk, d]`. The prepared tensors
/// are head-major views of sequence-major `[batch, sequence, heads, d]`
/// storage, so the split happens in that storage order and the result is a
/// permuted view: for an unmasked block starting at token zero no element is
/// copied. Other blocks (an offset narrow, padding, or masked inputs) cost
/// the one contiguous copy that any reshape of a strided tensor needs.
fn split_chunks<B: Backend>(
    tensor: &Tensor<B, 4>,
    start: usize,
    width: usize,
    padded: usize,
    chunk: usize,
) -> Tensor<B, 5> {
    let [batch, heads, _, depth] = tensor.dims();
    let block = tensor.clone().swap_dims(1, 2).narrow(1, start, width);
    let block = if padded == width {
        block
    } else {
        let padding = Tensor::zeros(
            [batch, padded - width, heads, depth],
            (&block.device(), block.dtype()),
        );
        Tensor::cat(vec![block, padding], 1)
    };
    block
        .reshape([batch, padded / chunk, chunk, heads, depth])
        .permute([0, 3, 1, 2, 4])
}

/// Returns `(I + L)^-1` for strictly lower-triangular `L` over the last two axes.
///
/// Forward substitution builds `M = (I + L)^-1 - I` row by row:
/// `M[i, :i] = -L[i, :i] - L[i, :i] M[:i, :i]`, written in place as
/// `R[i, :i] += R[i, :i] R[:i, :i]` on `R = -L`. Each step is batched over
/// every leading axis, so all chunks of a block share one `C - 1`-step loop.
/// Repeated squaring of the Neumann series is avoided on purpose: its
/// intermediate powers of `L` grow binomially for repeated keys and cancel
/// catastrophically in F32, while substitution never forms them.
fn unit_lower_inverse<B: Backend>(strict_lower: Tensor<B, 5>) -> Tensor<B, 5> {
    let [batch, heads, chunks, width, _] = strict_lower.dims();
    let mut inverse = -strict_lower;
    for row in 1..width {
        let row_values =
            inverse
                .clone()
                .slice([0..batch, 0..heads, 0..chunks, row..row + 1, 0..row]);
        let solved_prefix = inverse
            .clone()
            .slice([0..batch, 0..heads, 0..chunks, 0..row, 0..row]);
        let solved_row = row_values.clone() + row_values.matmul(solved_prefix);
        inverse = inverse.slice_assign(
            [0..batch, 0..heads, 0..chunks, row..row + 1, 0..row],
            solved_row,
        );
    }
    let identity = Tensor::<B, 2>::eye(width, &inverse.device())
        .cast(DType::F32)
        .reshape([1, 1, 1, width, width]);
    inverse + identity
}

/// Validated FP32 tensors in head-major layout used by both algorithms.
struct PreparedDeltaRule<B: Backend> {
    /// L2-normalized queries scaled by `key_width^-1/2`, in `[batch, heads, sequence, key_width]` order.
    query: Tensor<B, 4>,
    /// Normalized keys with the same head-major layout as queries.
    key: Tensor<B, 4>,
    /// Values in `[batch, heads, sequence, value_width]` order.
    value: Tensor<B, 4>,
    /// Per-head logarithmic state decay in `[batch, heads, sequence]` order.
    log_decay: Tensor<B, 3>,
    /// Per-head update strengths multiplied by the token validity indicator.
    beta: Tensor<B, 3>,
    /// Optional incoming FP32 state `[batch, heads, key_width, value_width]`.
    initial_state: Option<Tensor<B, 4>>,
    /// Original query dtype restored at the output boundary.
    output_dtype: DType,
    /// Common input device used for temporary tensors.
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
            // Explicit F32 indicators: a reduced-precision device default must
            // not create a mixed-dtype product with the F32 operands.
            let head_indicator = head_mask.clone().cast(FloatDType::F32);
            let value_mask = head_indicator.clone().unsqueeze_dim::<4>(3);
            query = query.swap_dims(1, 2) * value_mask.clone();
            key = key.swap_dims(1, 2) * value_mask.clone();
            value = value.swap_dims(1, 2) * value_mask;
            beta = beta.swap_dims(1, 2) * head_indicator;
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
#[path = "delta_net_tests.rs"]
mod tests;
