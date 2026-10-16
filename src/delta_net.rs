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
//! used for decode and parity tests; [`recurrent_gated_delta_rule_with_snapshots`]
//! additionally keeps the state after every token, which speculative
//! verification needs to commit an accepted prefix. [`chunked_gated_delta_rule`]
//! algebraically resolves all within-chunk delta updates as a triangular
//! system, then performs dense batched matrix products. It preserves the
//! recurrent result while exposing substantially more parallelism during
//! prompt prefill: every chunk's triangular inverse (by recursive block
//! doubling) and intra-chunk products run batched across up to
//! [`CHUNKS_PER_BLOCK`] chunks, and only the three-product state carry between
//! chunks remains sequential. The projected mixer's prefill mode picks the
//! recurrence instead for a single token or a very short segment with a tiny
//! state, where it is cheaper.
//!
//! The projected [`GatedDeltaNet`] mixer adds a depthwise causal convolution
//! before the rule and a gated RMS normalization after it. Padding (a `false`
//! token mask) is removed from both: padded tokens neither update the matrix
//! state nor occupy convolution taps or history, so a padded row produces the
//! same valid outputs and state as its compacted, unpadded form.
//!
//! The integer planning behind both paths (chunk width and count, padding,
//! balanced chunk blocks, the doubling-inverse levels, projection widths and
//! the query/key/value channel layout, geometry checks, and the
//! recurrence-versus-chunked choice) lives in the dependency-free `plan`
//! submodule, whose functions Verus proves exact through `verus/mixers.rs`.

use core::ops::Range;

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

pub(crate) mod plan;

use plan::{
    ChunkPlan, DeltaGeometryViolation, InverseLevel, MixerLayout, ProjectionWidths,
    block_chunk_ranges, check_delta_geometry, chunk_plan, head_repetitions, inverse_levels,
    mixer_layout, prefers_recurrence, projection_widths,
};

/// Number of prefill chunks whose intra-chunk algebra is batched together.
///
/// Larger blocks issue fewer kernels but hold `[block, batch, heads, C, C]`
/// F32 matrices; 16 chunks of 64 tokens keep that at 256 KiB per head and
/// batch row while cutting the triangular-inverse launches sixteenfold. The
/// ignored `prefill_timing` benchmark measures the batched algorithm against
/// the per-chunk one on Flex (SIMD and rayon): 2.2x faster for 1024 tokens
/// with 16 heads of width 128, and 2.7x for narrow heads.
pub const CHUNKS_PER_BLOCK: usize = 16;

/// Default stabilizer of the query/key L2 normalization (Qwen and FLA reference).
const DEFAULT_QK_L2_EPSILON: f64 = 1e-6;

/// Returns [`DEFAULT_QK_L2_EPSILON`] for configurations serialized before the field existed.
const fn default_qk_l2_epsilon() -> f64 {
    DEFAULT_QK_L2_EPSILON
}

/// Numerically sensitive controls for both DeltaNet execution paths.
///
/// The rule owns only the query/key L2-normalization stabilizer; the mixer's
/// output RMS normalization has its own epsilon,
/// [`GatedDeltaNetConfig::normalization_epsilon`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DeltaRuleConfig {
    /// Stabilizer in query/key L2 normalization: `x / sqrt(sum(x^2) + epsilon)`.
    ///
    /// The same quantity as [`GatedDeltaNetConfig::qk_l2_epsilon`], which
    /// [`GatedDeltaNetConfig::rule_config`] copies here. Configurations
    /// serialized before the rename spelled it `normalization_epsilon`; that
    /// key is still accepted on deserialization.
    #[serde(alias = "normalization_epsilon")]
    pub qk_l2_epsilon: f64,
    /// Maximum time-axis block processed by the parallel prefill path.
    pub chunk_size: usize,
}

impl Default for DeltaRuleConfig {
    /// Uses the official Qwen reference constants.
    fn default() -> Self {
        Self {
            qk_l2_epsilon: DEFAULT_QK_L2_EPSILON,
            chunk_size: 64,
        }
    }
}

impl DeltaRuleConfig {
    /// Returns the query/key L2-normalization stabilizer under its former name.
    ///
    /// The field was renamed to [`Self::qk_l2_epsilon`] because
    /// `normalization_epsilon` named the *output* RMS-norm stabilizer on
    /// [`GatedDeltaNetConfig`], so one name meant two different quantities.
    #[deprecated(
        note = "renamed to the `qk_l2_epsilon` field; `GatedDeltaNetConfig::normalization_epsilon` is the output-norm epsilon"
    )]
    #[must_use]
    pub const fn normalization_epsilon(&self) -> f64 {
        self.qk_l2_epsilon
    }

    /// Validates constants before a backend operation is dispatched.
    ///
    /// # Errors
    ///
    /// Returns a configuration error naming `delta_net.rule.qk_l2_epsilon`
    /// for a non-finite or non-positive query/key epsilon and
    /// `delta_net.chunk_size` for a zero chunk size.
    pub fn validate(&self) -> Result<()> {
        if !self.qk_l2_epsilon.is_finite() || self.qk_l2_epsilon <= 0.0 {
            return Err(Error::configuration(
                "delta_net.rule.qk_l2_epsilon",
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
        Ok(self.projection_widths()?.channels)
    }

    /// Returns the verified key, value, and fused channel widths of the Q/K/V projection.
    ///
    /// [`projection_widths`] is exact and fails exactly when the channel count
    /// overflows `usize`, in which case every partial width fits too.
    fn projection_widths(&self) -> Result<ProjectionWidths> {
        projection_widths(
            self.key_heads,
            self.key_head_dim,
            self.value_heads,
            self.value_head_dim,
        )
        .ok_or_else(|| Error::configuration("delta_net", "projection dimensions overflow usize"))
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
        match check_delta_geometry(
            self.hidden_size,
            self.key_heads,
            self.value_heads,
            self.key_head_dim,
            self.value_head_dim,
            self.convolution_kernel,
        ) {
            Ok(()) => {}
            Err(DeltaGeometryViolation::ZeroWidth) => {
                return Err(Error::configuration(
                    "delta_net",
                    "all model and convolution dimensions must be non-zero",
                ));
            }
            Err(DeltaGeometryViolation::ValueHeads) => {
                return Err(Error::configuration(
                    "delta_net.value_heads",
                    "must be an integer multiple of key_heads",
                ));
            }
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
        // The verified channel count fits only if the key and value widths fit,
        // so this one check covers every projection width.
        let _ = self.projection_widths()?;
        Ok(())
    }

    /// Returns the delta-rule controls: query/key epsilon and prefill chunk size.
    #[must_use]
    pub const fn rule_config(&self) -> DeltaRuleConfig {
        DeltaRuleConfig {
            qk_l2_epsilon: self.qk_l2_epsilon,
            chunk_size: self.prefill_chunk_size,
        }
    }

    /// Initializes lazy Burn projections and checkpoint-compatible scalar parameters.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Result<GatedDeltaNet<B>> {
        self.validate()?;
        let ProjectionWidths {
            value_width,
            channels: convolution_channels,
            ..
        } = self.projection_widths()?;
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
            in_proj_qkv: linear(convolution_channels),
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
    /// Prompt processing with the faster of the two exact algorithms.
    ///
    /// This is the blockwise-parallel [`chunked_gated_delta_rule`] except for
    /// a single token, and for segments so short and states so small that the
    /// recurrence costs less (for example a dozen tokens of a `2 x 4 x 3`
    /// state; never for Qwen3.5-sized states beyond one token). Both compute
    /// the same function up to floating-point reassociation.
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

/// Per-token states of one [`GatedDeltaNet::forward_with_snapshots`] call.
#[derive(Clone, Debug)]
pub struct GatedDeltaNetSnapshots<B: Backend> {
    /// FP32 association matrices after every column,
    /// `[batch, sequence, value_heads, key_head_dim, value_head_dim]`.
    pub recurrent: Tensor<B, 5>,
    /// Convolution histories after every column, `[batch, sequence, channels, kernel - 1]`.
    pub convolution: Tensor<B, 4>,
}

/// Delta-rule algorithm selected for one mixer call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RuleKernel {
    /// Prompt processing: chunked, or the recurrence when it is cheaper.
    Prefill,
    /// The token-by-token recurrence.
    Recurrent,
    /// The recurrence, keeping the state after every token.
    Snapshots,
}

/// Results of the mixer's depthwise causal convolution.
struct CausalConvolution<B: Backend> {
    /// Convolved projections `[batch, channels, sequence]`, zero at padding.
    output: Tensor<B, 3>,
    /// Next history: the last `kernel - 1` columns of `[history | valid tokens]`.
    history: Tensor<B, 3>,
    /// Timeline `[history | valid tokens | padding]` the convolution read,
    /// `[batch, channels, kernel - 1 + sequence]` (just the tokens for kernel one).
    joined: Tensor<B, 3>,
}

/// Returns the convolution history after every input column.
///
/// In `joined = [history | valid tokens | padding]`, the history after column
/// `t` is the `width` columns ending just after the row's `v_t`-th valid
/// token, where `v_t` counts the valid columns among `0..=t`; it therefore
/// starts at `joined` column `v_t`. The result is
/// `[batch, sequence, channels, width]`, gathered with one index tensor.
fn convolution_snapshots<B: Backend>(
    joined: Tensor<B, 3>,
    token_mask: Option<Tensor<B, 2, Bool>>,
    width: usize,
) -> Tensor<B, 4> {
    let [batch, channels, timeline] = joined.dims();
    let sequence = timeline - width;
    let device = joined.device();
    if width == 0 {
        return Tensor::zeros([batch, sequence, channels, 0], (&device, joined.dtype()));
    }
    // v_t: valid columns among 0..=t, `[batch, sequence]`.
    let valid_through = match token_mask {
        Some(mask) => mask.int().cumsum(1),
        None => Tensor::<B, 1, Int>::arange(1..sequence as i64 + 1, &device)
            .reshape([1, sequence])
            .expand([batch, sequence]),
    };
    let offsets = Tensor::<B, 1, Int>::arange(0..width as i64, &device).reshape([1, 1, width]);
    let index = (valid_through.reshape([batch, sequence, 1]) + offsets)
        .reshape([batch, sequence, 1, width])
        .expand([batch, sequence, channels, width]);
    joined
        .reshape([batch, 1, channels, timeline])
        .expand([batch, sequence, channels, timeline])
        .gather(3, index)
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
    ///
    /// # Errors
    ///
    /// Returns a configuration error when the zero state cannot be allocated:
    /// an empty batch, or a topology reconstructed by [`Self::config`] that
    /// fails validation (for example a manually set zero chunk size).
    /// Otherwise returns the errors of [`Self::forward_with_state`].
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
    /// mismatched mask, or state buffers that do not match this layer. A
    /// manually assembled module can also return projection shape errors and,
    /// from the delta-rule kernels, a configuration error for an invalid
    /// `rule_config` (a non-finite or non-positive `qk_l2_epsilon`, or
    /// a zero `chunk_size`). On any error `state` is left exactly as supplied.
    pub fn forward_with_state(
        &self,
        hidden: Tensor<B, 3>,
        token_mask: Option<&TokenMask<B>>,
        state: &mut GatedDeltaNetState<B>,
        mode: DeltaNetMode,
    ) -> Result<Tensor<B, 3>> {
        let kernel = match mode {
            DeltaNetMode::Recurrent => RuleKernel::Recurrent,
            DeltaNetMode::ChunkedPrefill => RuleKernel::Prefill,
        };
        let (output, _) = self.mix(hidden, token_mask, state, kernel)?;
        Ok(output)
    }

    /// Evaluates a segment like [`Self::forward_with_state`] and also returns the state after every token.
    ///
    /// Snapshot `t` is the state after input column `t`: the FP32 association
    /// matrix and the convolution history `[history | valid tokens]` truncated
    /// to its last `kernel - 1` columns. A padded column leaves both unchanged,
    /// so its snapshot repeats the previous one. Speculative verification feeds
    /// several proposed tokens in one call and later keeps only an accepted
    /// prefix; because the recurrence cannot be rewound, the cache retains
    /// these snapshots and selects the one at the accepted boundary. The rule
    /// runs as the exact token-by-token recurrence. `state` receives the final
    /// state, which equals the last snapshot.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::forward_with_state`].
    pub fn forward_with_snapshots(
        &self,
        hidden: Tensor<B, 3>,
        token_mask: Option<&TokenMask<B>>,
        state: &mut GatedDeltaNetState<B>,
    ) -> Result<(Tensor<B, 3>, GatedDeltaNetSnapshots<B>)> {
        let (output, snapshots) = self.mix(hidden, token_mask, state, RuleKernel::Snapshots)?;
        let snapshots = snapshots.ok_or_else(|| {
            Error::configuration("delta_net.snapshots", "snapshot kernel returned no states")
        })?;
        Ok((output, snapshots))
    }

    /// Shared mixer body: projections, causal convolution, delta rule, gated norm, output.
    fn mix(
        &self,
        hidden: Tensor<B, 3>,
        token_mask: Option<&TokenMask<B>>,
        state: &mut GatedDeltaNetState<B>,
        kernel: RuleKernel,
    ) -> Result<(Tensor<B, 3>, Option<GatedDeltaNetSnapshots<B>>)> {
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
        let mask = token_mask.map(|mask| mask.as_tensor().clone());
        let convolution =
            self.causal_convolution(projected, state.convolution.clone(), mask.clone());
        let convolution_snapshots = (kernel == RuleKernel::Snapshots).then(|| {
            convolution_snapshots(
                convolution.joined.clone(),
                mask.clone(),
                self.convolution_kernel - 1,
            )
        });
        let mixed = activation::silu(convolution.output).swap_dims(1, 2);
        // Verified tiling of the convolved channels: queries `0..K`, keys
        // `K..2K`, values `2K..2K + V`; a validated configuration's channel
        // count fits, which is the precondition.
        let MixerLayout {
            key_width,
            value_start,
            value_width,
        } = mixer_layout(
            self.key_heads,
            self.key_head_dim,
            self.value_heads,
            self.value_head_dim,
        );
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
        let value = mixed.narrow(2, value_start, value_width).reshape([
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
            token_mask: mask,
        };
        // The recurrent state passed `validate_state`, so its element count fits.
        let state_elements =
            plan::state_elements(self.value_heads, self.key_head_dim, self.value_head_dim);
        let (values, final_state, snapshots) = match kernel {
            RuleKernel::Prefill if !prefers_recurrence(sequence, state_elements) => {
                let output = chunked_gated_delta_rule(rule_input, self.rule_config)?;
                (output.values, output.final_state, None)
            }
            RuleKernel::Prefill | RuleKernel::Recurrent => {
                let output = recurrent_gated_delta_rule(rule_input, self.rule_config)?;
                (output.values, output.final_state, None)
            }
            RuleKernel::Snapshots => {
                let output =
                    recurrent_gated_delta_rule_with_snapshots(rule_input, self.rule_config)?;
                let [rows, tokens, heads, key_dim, value_dim] = output.states.dims();
                let final_state = output
                    .states
                    .clone()
                    .narrow(1, tokens - 1, 1)
                    .reshape([rows, heads, key_dim, value_dim]);
                let snapshots = convolution_snapshots.map(|convolution| GatedDeltaNetSnapshots {
                    recurrent: output.states,
                    convolution,
                });
                (output.values, final_state, snapshots)
            }
        };
        let gated = gated_delta_norm(values, z, &self.norm);
        let output = operations::linear(
            gated.reshape([batch, sequence, value_width]),
            &self.out_proj,
        )?;
        // Publish the advanced state only after the last fallible step, so a
        // failed call leaves the caller's buffers exactly as supplied.
        state.convolution = convolution.history;
        state.recurrent = final_state;
        Ok((output, snapshots))
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
            qk_l2_epsilon: self.rule_config.qk_l2_epsilon,
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
    ) -> CausalConvolution<B> {
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
            return CausalConvolution {
                output: convolve(projected.clone()),
                history: empty,
                joined: projected,
            };
        }
        // Recurrent matrices intentionally accumulate in FP32, but convolution
        // history follows the projection dtype. Casting here also makes a state
        // created before checkpoint dtype selection safe on its first call.
        let history = history.cast(projected.dtype());
        let Some(mask) = token_mask else {
            let joined = Tensor::cat(vec![history, projected], 2);
            return CausalConvolution {
                output: convolve(joined.clone()),
                history: joined.clone().narrow(2, sequence, history_width),
                joined,
            };
        };
        let compaction = TokenCompaction::new(mask);
        let joined = Tensor::cat(vec![history, compaction.compact(projected, channels)], 2);
        CausalConvolution {
            output: compaction.restore(convolve(joined.clone()), channels),
            history: compaction.history(joined.clone(), channels, history_width),
            joined,
        }
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
///
/// The validated geometry makes `key_heads` divide `value_heads`, so the
/// verified repetition count tiles the value heads exactly.
fn repeat_delta_heads<B: Backend>(tensor: Tensor<B, 4>, value_heads: usize) -> Tensor<B, 4> {
    let [batch, sequence, key_heads, head_dim] = tensor.dims();
    let repetitions = head_repetitions(value_heads, key_heads);
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
    let prepared = PreparedDeltaRule::new(input, config.qk_l2_epsilon)?;
    let (values, final_state, _) = prepared.recurrence(false);
    Ok(DeltaRuleOutput {
        values,
        final_state,
    })
}

/// Per-token recurrent states of one [`recurrent_gated_delta_rule_with_snapshots`] call.
#[derive(Clone, Debug)]
pub struct DeltaRuleSnapshots<B: Backend> {
    /// Mixed values shaped `[batch, sequence, heads, value_width]`.
    pub values: Tensor<B, 4>,
    /// FP32 state after every token, `[batch, sequence, heads, key_width, value_width]`.
    ///
    /// Entry `t` is `S_t`, the state after token `t`; the last entry equals the
    /// final state. A masked token leaves the state unchanged, so its entry
    /// repeats the previous one (or the initial state).
    pub states: Tensor<B, 5>,
}

/// Evaluates the sequential recurrence and also returns the state after every token.
///
/// Speculative decoding verifies `n` proposed tokens in one call but may keep
/// only an accepted prefix of `a <= n` tokens. A recurrence cannot be rewound,
/// so the cache must retain `S_1, ..., S_n` and later select `S_a`. The values
/// and states are exactly those of [`recurrent_gated_delta_rule`]; the extra
/// cost is one `[batch, heads, key_width, value_width]` FP32 tensor per token.
///
/// # Errors
///
/// Returns the errors of [`recurrent_gated_delta_rule`].
pub fn recurrent_gated_delta_rule_with_snapshots<B: Backend>(
    input: DeltaRuleInput<B>,
    config: DeltaRuleConfig,
) -> Result<DeltaRuleSnapshots<B>> {
    config.validate()?;
    let prepared = PreparedDeltaRule::new(input, config.qk_l2_epsilon)?;
    let (values, _, states) = prepared.recurrence(true);
    // `recurrence(true)` returns one state per token in token order.
    let states = Tensor::stack::<5>(states, 1);
    Ok(DeltaRuleSnapshots { values, states })
}

/// Evaluates a blockwise-parallel formulation for prompt prefill.
///
/// Tokens are split into chunks of `C = min(chunk_size, L)`. Inside a chunk
/// with cumulative log decays `G_i`, decay matrix `D_ij = exp(G_i - G_j)`
/// (`i >= j`), and `L = strict_lower((beta k) k^T * D)`, all delta writes are
/// resolved at once by `T = (I + L)^-1`, computed by recursive block
/// doubling. With the incoming state `S`, a chunk then emits
///
/// ```text
/// U   = T (beta v) - T (beta k * exp(G)) S        corrected values
/// out = (q * exp(G)) S + lower(q k^T * D) U
/// S'  = exp(G_last) S + (k * exp(G_last - G))^T U
/// ```
///
/// Everything except the last three lines is independent of `S`, so it runs
/// batched over blocks of up to [`CHUNKS_PER_BLOCK`] chunks: one triangular
/// inverse per block instead of per chunk. The sequential dependency is
/// `ceil(L / C)` short state updates.
///
/// # Layout
///
/// The operands are rearranged once into dense chunk-major tensors
/// `[chunks, batch, C, heads, ·]`. Every block is then
/// a zero-copy narrow of the outermost axis, every elementwise product runs
/// on dense memory (vectorized CPU kernels need dense operands; strided views
/// fall back to scalar index walks), and chunk `i` of the scan is again a
/// dense slice. The intra-chunk inverse uses recursive block doubling
/// (`log2 C` batched levels instead of `C - 1` row
/// updates). A trailing partial chunk is zero-padded once, which is
/// exact because padded tokens have `k = v = beta = 0` and `log_decay = 0`,
/// so they neither change `S` nor the valid rows of `T`. A prompt of
/// `L < chunk_size` tokens is one unpadded chunk of width `L`, whose
/// inverse runs on `L x L` tiles (padded to the next power of two). Blocks
/// are balanced (sizes differing by at most one), so a lone partial tail
/// never needs a block, and thus a full set of inverse levels, of its own.
///
/// # Errors
///
/// Returns configuration or shape errors for invalid controls or inputs.
pub fn chunked_gated_delta_rule<B: Backend>(
    input: DeltaRuleInput<B>,
    config: DeltaRuleConfig,
) -> Result<DeltaRuleOutput<B>> {
    config.validate()?;
    let prepared = PreparedDeltaRule::new(input, config.qk_l2_epsilon)?;
    let [batch, sequence, heads, _] = prepared.query.dims();
    let value_width = prepared.value.dims()[3];
    let operands = ChunkOperands::new(&prepared, config.chunk_size);
    // The scan keeps a leading unit chunk axis so every product stays rank 5
    // and no per-chunk reshape (a copy for offset views) is needed.
    let mut state = prepared.initial_state_or_zeros().unsqueeze_dim::<5>(0);
    let ranges = block_chunk_ranges(operands.chunks, CHUNKS_PER_BLOCK);
    let mut outputs = Vec::with_capacity(operands.chunks);
    for chunks in ranges {
        state = ChunkBlock::new(&operands, chunks).scan(state, &mut outputs);
    }
    let [_, _, _, key_width, _] = state.dims();
    let padded = operands.padded;
    // [chunks, batch, heads, C, value] -> [batch, chunks * C, heads, value].
    let values = Tensor::cat(outputs, 0)
        .permute([1, 0, 3, 2, 4])
        .reshape([batch, padded, heads, value_width])
        .narrow(1, 0, sequence)
        .cast(prepared.output_dtype);
    Ok(DeltaRuleOutput {
        values,
        final_state: state.reshape([batch, heads, key_width, value_width]),
    })
}

/// Chunk-major FP32 operands of one prefill call.
///
/// Each tensor is a dense `[chunks, batch, C, heads, ·]` arrangement of the
/// token-major `[batch, sequence, heads, ·]` input, zero-padded to
/// `chunks * C` tokens. Within a chunk the token-major order is kept, so for
/// one batch row the arrangement is a free reshape and for several rows a
/// plain 2-D transpose of `[batch, chunks]`. Blocks and single chunks are then
/// zero-copy narrows of the leading axis, elementwise products run on dense
/// memory (vectorized CPU kernels need dense operands; strided views fall back
/// to scalar index walks), and per-head matrix products read the
/// `[.., heads, C, ·]` view of [`heads_major`], which GEMM accepts as strided
/// input without a copy.
struct ChunkOperands<B: Backend> {
    /// L2-normalized, scaled queries `[chunks, batch, C, heads, key_width]`.
    query: Tensor<B, 5>,
    /// L2-normalized keys `[chunks, batch, C, heads, key_width]`.
    key: Tensor<B, 5>,
    /// Values `[chunks, batch, C, heads, value_width]`.
    value: Tensor<B, 5>,
    /// Masked update strengths `[chunks, batch, C, heads, 1]`.
    beta: Tensor<B, 5>,
    /// Masked log decays `[chunks, batch, C, heads, 1]`.
    log_decay: Tensor<B, 5>,
    /// Chunk width `C = min(chunk_size, sequence)`.
    chunk: usize,
    /// Number of chunks `ceil(sequence / C)`.
    chunks: usize,
    /// Zero-padded token count `chunks * C`, less than one chunk above the sequence.
    padded: usize,
}

impl<B: Backend> ChunkOperands<B> {
    /// Rearranges prepared token-major operands into chunk-major blocks of width `chunk_size`.
    ///
    /// The verified [`chunk_plan`] fixes `C`, the chunk count, and the padded
    /// length; prepared operands are non-empty and `chunk_size` is validated
    /// non-zero, which are its preconditions.
    fn new(prepared: &PreparedDeltaRule<B>, chunk_size: usize) -> Self {
        let sequence = prepared.query.dims()[1];
        let plan = chunk_plan(sequence, chunk_size);
        let arrange = |tensor: Tensor<B, 4>| chunk_major(tensor, plan);
        Self {
            query: arrange(prepared.query.clone()),
            key: arrange(prepared.key.clone()),
            value: arrange(prepared.value.clone()),
            beta: arrange(prepared.beta.clone().unsqueeze_dim(3)),
            log_decay: arrange(prepared.log_decay.clone().unsqueeze_dim(3)),
            chunk: plan.chunk,
            chunks: plan.chunks,
            padded: plan.padded,
        }
    }
}

/// Converts token-major `[batch, sequence, heads, d]` into dense `[chunks, batch, C, heads, d]`.
///
/// The token axis is zero-padded to `chunks * C` and split into
/// `[chunks, C]`, a view of dense storage. Moving `chunks` in front of
/// `batch` is the identity for one row; otherwise the final reshape, which
/// merges the swapped axes and therefore cannot be a view, materializes it as
/// one 2-D transpose of `[batch, chunks]` blocks.
fn chunk_major<B: Backend>(tokens: Tensor<B, 4>, plan: ChunkPlan) -> Tensor<B, 5> {
    let [batch, sequence, heads, depth] = tokens.dims();
    let ChunkPlan {
        chunk,
        chunks,
        padded,
    } = plan;
    let tokens = if padded == sequence {
        tokens
    } else {
        let padding = Tensor::zeros(
            [batch, padded - sequence, heads, depth],
            (&tokens.device(), tokens.dtype()),
        );
        Tensor::cat(vec![tokens, padding], 1)
    };
    let width = chunk * heads * depth;
    tokens
        .reshape([batch, chunks, width])
        .swap_dims(0, 1)
        .reshape([chunks * batch, width])
        .reshape([chunks, batch, chunk, heads, depth])
}

/// Views `[chunks, batch, C, heads, d]` as `[chunks, batch, heads, C, d]` for per-head products.
fn heads_major<B: Backend>(tensor: Tensor<B, 5>) -> Tensor<B, 5> {
    tensor.swap_dims(2, 3)
}

/// State-independent terms of one block of consecutive chunks.
///
/// Per-head matrices are `[chunks, batch, heads, C, ·]`; chunk `i` of the scan
/// is the zero-copy slice `narrow(0, i, 1)`.
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
    /// `exp(G_last)` shaped `[chunks, batch, heads, 1, 1]`: state decay across a chunk.
    chunk_decay: Tensor<B, 5>,
}

impl<B: Backend> ChunkBlock<B> {
    /// Computes every state-independent term for chunks `chunks` of `operands`.
    ///
    /// Elementwise products run in the dense token-minor layout; only matrix
    /// products read the strided per-head views.
    fn new(operands: &ChunkOperands<B>, chunks: Range<usize>) -> Self {
        let block = |tensor: &Tensor<B, 5>| tensor.clone().narrow(0, chunks.start, chunks.len());
        let query = block(&operands.query);
        let key = block(&operands.key);
        let value = block(&operands.value);
        let beta = block(&operands.beta);
        // Cumulative log decay G within each chunk, [n, b, C, h, 1].
        let cumulative = block(&operands.log_decay).cumsum(2);
        let decay_to_start = cumulative.clone().exp();
        let final_decay = cumulative.clone().narrow(2, operands.chunk - 1, 1);
        let decay_to_end = (final_decay.clone() - cumulative.clone()).exp();
        let decay_matrix = causal_decay_matrix(heads_major(cumulative));

        let key_beta = key.clone() * beta.clone();
        let key_columns = heads_major(key.clone()).swap_dims(3, 4);
        let strict = (heads_major(key_beta.clone()).matmul(key_columns.clone())
            * decay_matrix.clone())
        .tril(-1);
        let transform = unit_lower_inverse(strict);
        let attention = (heads_major(query.clone()).matmul(key_columns) * decay_matrix).tril(0);
        Self {
            transformed_value: transform.clone().matmul(heads_major(value * beta)),
            cumulative_key: transform.matmul(heads_major(key_beta * decay_to_start.clone())),
            decayed_query: heads_major(query * decay_to_start),
            attention,
            decayed_key: heads_major(key * decay_to_end),
            chunk_decay: heads_major(final_decay).exp(),
        }
    }

    /// Carries the rank-5 `[1, batch, heads, key, value]` state across the block's chunks.
    ///
    /// Appends one `[1, batch, heads, C, value_width]` output per chunk to
    /// `outputs` (padded rows included; the caller trims them once) and
    /// returns the outgoing FP32 state.
    fn scan(self, mut state: Tensor<B, 5>, outputs: &mut Vec<Tensor<B, 5>>) -> Tensor<B, 5> {
        let chunks = self.attention.dims()[0];
        let take = |tensor: &Tensor<B, 5>, index: usize| tensor.clone().narrow(0, index, 1);
        for index in 0..chunks {
            let corrected = take(&self.transformed_value, index)
                - take(&self.cumulative_key, index).matmul(state.clone());
            outputs.push(
                take(&self.decayed_query, index).matmul(state.clone())
                    + take(&self.attention, index).matmul(corrected.clone()),
            );
            // `[1, batch, heads, 1, 1]` broadcasts over the `[key, value]` state.
            state = state * take(&self.chunk_decay, index)
                + take(&self.decayed_key, index)
                    .swap_dims(3, 4)
                    .matmul(corrected);
        }
        state
    }
}

/// Builds `D[i, j] = exp(G_i - G_j)` for `i >= j` and zero above the diagonal.
///
/// `cumulative` holds `G` as `[chunks, batch, heads, C, 1]` with any strides;
/// the result is dense `[chunks, batch, heads, C, C]`.
///
/// # Exactness
///
/// Every `G_i - G_j` is one correctly rounded F32 subtraction on every
/// backend. The difference is deliberately never formed by a matrix product:
/// accelerator GEMM kernels may round F32 operands to TF32 (a 10-bit
/// mantissa), and `exp` turns that absolute error in `G` (about `|G| / 2^11`,
/// with `|G|` reaching ~100 inside a 64-token chunk) into a relative error of
/// several percent in `D`.
///
/// # Layout
///
/// The `M = chunks * batch * heads` matrices are flattened to `[M * C, C]`
/// rows: row `(m, i)` subtracts the dense row `(G_(m, j))_j` from the
/// broadcast scalar `G_(m, i)`. The dense rows come from a leading-axis repeat
/// (block copies) and the causal mask is materialized at full shape the same
/// way, so the subtraction, mask, and exponential all run on dense operands;
/// a doubly broadcast 5-D subtraction or a broadcast mask would fall back to
/// scalar index walks on Flex. Filling the future with `-inf` before `exp`
/// zeroes it, keeps its positive differences from overflowing, and keeps the
/// gradient finite there.
fn causal_decay_matrix<B: Backend>(cumulative: Tensor<B, 5>) -> Tensor<B, 5> {
    let [chunks, batch, heads, width, _] = cumulative.dims();
    let matrices = chunks * batch * heads;
    let device = cumulative.device();
    // `[M * C, 1]`: entry `(m, i)` is `G_i` of matrix `m`.
    let rows = cumulative.reshape([matrices * width, 1]);
    // `[M * C, C]`: entry `((m, i), j)` is `G_j` of matrix `m`, for every `i`.
    let columns = rows
        .clone()
        .reshape([matrices, 1, width])
        .repeat_dim(1, width)
        .reshape([matrices * width, width]);
    let future = Tensor::<B, 2, Bool>::tril_mask([width, width], 0, &device)
        .reshape([1, width * width])
        .repeat_dim(0, matrices)
        .reshape([matrices * width, width]);
    (rows - columns)
        .mask_fill(future, f32::NEG_INFINITY)
        .exp()
        .reshape([chunks, batch, heads, width, width])
}

/// Returns `(I + L)^-1` for strictly lower-triangular `L` over the last two axes.
///
/// # Algorithm
///
/// Recursive doubling over diagonal blocks. A unit lower-triangular matrix
/// split into halves inverts blockwise as
///
/// ```text
/// [ A  0 ]^-1   [  A^-1           0    ]
/// [ Q  D ]    = [ -D^-1 Q A^-1    D^-1 ]
/// ```
///
/// Starting from the unit `1 x 1` diagonal, level `s` combines every pair of
/// adjacent `s x s` inverses into one `2s x 2s` inverse with two batched
/// matrix products, so `C` (padded to a power of two `P`) needs `log2 P`
/// levels. Every operand of a level is one batched tensor; no step writes
/// into a slice of a larger tensor.
///
/// Row-by-row forward substitution needs `C - 1` sequential steps, each of
/// which assigns one row into the whole matrix; on Flex every such assignment
/// copies the complete destination, which made substitution cost `O(C^3)`
/// memory traffic per matrix. The Neumann series is avoided on purpose: its
/// intermediate powers of `L` grow binomially for repeated keys and cancel
/// catastrophically in F32. The blocked products only ever form entries of
/// the inverse itself, which is the stability property that makes blocked
/// triangular inversion as accurate as substitution (Du Croz and Higham,
/// 1992).
///
/// Padding `L` with zero rows and columns to `P` is exact: `[[L, 0], [0, 0]]`
/// inverts to `[[T, 0], [0, I]]`, and the padding is narrowed away.
fn unit_lower_inverse<B: Backend>(strict_lower: Tensor<B, 5>) -> Tensor<B, 5> {
    let [chunks, batch, heads, width, _] = strict_lower.dims();
    let device = strict_lower.device();
    if width == 1 {
        // A 1 x 1 strictly lower matrix is zero, so `I + L` is the identity.
        return Tensor::ones([chunks, batch, heads, 1, 1], (&device, DType::F32));
    }
    let items = chunks * batch * heads;
    // Verified levels: one `P x P` block, then `2^k` blocks of side `P / 2^k`
    // down to side two, with `P` the least power of two `>= width`.
    let levels = inverse_levels(width);
    let size = levels[0].block;
    let mut lower = strict_lower.reshape([items, 1, width, width]);
    if size != width {
        let pad = size - width;
        let zeros = |shape: [usize; 4]| Tensor::<B, 4>::zeros(shape, (&device, DType::F32));
        lower = Tensor::cat(vec![lower, zeros([items, 1, width, pad])], 3);
        lower = Tensor::cat(vec![lower, zeros([items, 1, pad, size])], 2);
    }
    // Diagonal blocks of L by size: `diagonal[k]` holds the blocks of size
    // `size >> k` as `[items, 2^k, size >> k, size >> k]`, finest last.
    let mut diagonal = vec![lower];
    for pair in levels.windows(2) {
        // Each parent block splits into two children of side `half`, and the
        // next level holds twice as many blocks.
        let (InverseLevel { half, .. }, InverseLevel { count, .. }) = (pair[0], pair[1]);
        let parent = diagonal.last().expect("starts non-empty").clone();
        // Rows 0..half of the concatenation are the upper-left quadrant and
        // rows half..block the lower-right one: reshaping interleaves them as
        // children 2j and 2j + 1 of parent j.
        let children = Tensor::cat(
            vec![
                parent.clone().narrow(2, 0, half).narrow(3, 0, half),
                parent.narrow(2, half, half).narrow(3, half, half),
            ],
            2,
        );
        diagonal.push(children.reshape([items, count, half, half]));
    }
    // `inverse` holds the inverted diagonal blocks of size `half`.
    let mut inverse: Option<Tensor<B, 4>> = None;
    // `diagonal[k]` is `[items, count, block, block]` for `levels[k]`.
    for (parent, level) in diagonal.into_iter().zip(&levels).rev() {
        let InverseLevel { block, half, count } = *level;
        // Q: the lower-left quadrant of every parent block.
        let quadrant = parent.narrow(2, half, half).narrow(3, 0, half);
        let (upper, lower_right, corner) = match inverse {
            // Unit 1 x 1 blocks invert to one: the corner is -Q.
            None => {
                let ones = Tensor::<B, 4>::ones([items, count, 1, 1], (&device, DType::F32));
                (ones.clone(), ones, quadrant.neg())
            }
            Some(children) => {
                // Children 2j (rows 0..half) and 2j + 1 (rows half..block).
                let pairs = children.reshape([items, count, block, half]);
                let upper = pairs.clone().narrow(2, 0, half);
                let lower_right = pairs.narrow(2, half, half);
                let corner = lower_right
                    .clone()
                    .matmul(quadrant)
                    .matmul(upper.clone())
                    .neg();
                (upper, lower_right, corner)
            }
        };
        let zeros = Tensor::<B, 4>::zeros([items, count, half, half], (&device, DType::F32));
        inverse = Some(Tensor::cat(
            vec![
                Tensor::cat(vec![upper, zeros], 3),
                Tensor::cat(vec![corner, lower_right], 3),
            ],
            2,
        ));
    }
    let inverse = inverse.expect("a power-of-two size of at least two has one level");
    let inverse = if size == width {
        inverse
    } else {
        inverse.narrow(2, 0, width).narrow(3, 0, width)
    };
    inverse.reshape([chunks, batch, heads, width, width])
}

/// Validated FP32 operands in token-major layout used by both algorithms.
///
/// Token-major storage is what the projections produce, so preparation never
/// transposes: the recurrence slices one token at a time, and the chunked
/// path rearranges everything once into chunk-major blocks.
struct PreparedDeltaRule<B: Backend> {
    /// L2-normalized queries scaled by `key_width^-1/2`, `[batch, sequence, heads, key_width]`.
    query: Tensor<B, 4>,
    /// Normalized keys with the same token-major layout as queries.
    key: Tensor<B, 4>,
    /// Values `[batch, sequence, heads, value_width]`.
    value: Tensor<B, 4>,
    /// Per-head logarithmic state decay `[batch, sequence, heads]`, zero at padding.
    log_decay: Tensor<B, 3>,
    /// Per-head update strengths `[batch, sequence, heads]`, zero at padding.
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
            // `[batch, sequence, 1]` validity broadcasts over heads.
            let head_mask = mask.unsqueeze_dim::<3>(2);
            // Explicit F32 indicators: a reduced-precision device default must
            // not create a mixed-dtype product with the F32 operands.
            let head_indicator = head_mask.clone().cast(FloatDType::F32);
            let feature_indicator = head_indicator.clone().unsqueeze_dim::<4>(3);
            query = query * feature_indicator.clone();
            key = key * feature_indicator.clone();
            value = value * feature_indicator;
            beta = beta * head_indicator;
            log_decay = log_decay.mask_fill(head_mask.bool_not(), 0.0);
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

    /// Returns the incoming FP32 state, or zeros `[batch, heads, key_width, value_width]`.
    fn initial_state_or_zeros(&self) -> Tensor<B, 4> {
        self.initial_state.clone().unwrap_or_else(|| {
            let [batch, _, heads, key_width] = self.query.dims();
            let value_width = self.value.dims()[3];
            Tensor::<B, 4>::zeros(
                [batch, heads, key_width, value_width],
                (&self.device, DType::F32),
            )
        })
    }

    /// Runs the exact token-by-token recurrence of the module documentation.
    ///
    /// Returns token-major values `[batch, sequence, heads, value_width]` in
    /// the input dtype, the final FP32 state, and, when `keep_states` is set,
    /// the `[batch, heads, key_width, value_width]` state after every token.
    fn recurrence(&self, keep_states: bool) -> (Tensor<B, 4>, Tensor<B, 4>, Vec<Tensor<B, 4>>) {
        let sequence = self.query.dims()[1];
        let mut state = self.initial_state_or_zeros();
        let mut outputs = Vec::with_capacity(sequence);
        let mut states = Vec::with_capacity(if keep_states { sequence } else { 0 });
        // One token as `[batch, heads, 1, width]`: a view of token-major storage.
        let token = |tensor: &Tensor<B, 4>, index: usize| {
            tensor.clone().narrow(1, index, 1).swap_dims(1, 2)
        };
        // One per-head scalar as `[batch, heads, 1, 1]`.
        let scalar = |tensor: &Tensor<B, 3>, index: usize| {
            tensor
                .clone()
                .narrow(1, index, 1)
                .swap_dims(1, 2)
                .unsqueeze_dim::<4>(3)
        };
        for index in 0..sequence {
            let query = token(&self.query, index);
            let key = token(&self.key, index);
            let value = token(&self.value, index);
            let beta = scalar(&self.beta, index);
            let decay = scalar(&self.log_decay, index).exp();

            state = state * decay;
            let retrieved = key.clone().matmul(state.clone());
            let delta = (value - retrieved) * beta;
            state = state + key.swap_dims(2, 3).matmul(delta);
            outputs.push(query.matmul(state.clone()));
            if keep_states {
                states.push(state.clone());
            }
        }
        let values = Tensor::cat(outputs, 2)
            .swap_dims(1, 2)
            .cast(self.output_dtype);
        (values, state, states)
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
