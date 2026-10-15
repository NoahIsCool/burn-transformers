//! Mamba-2 selective state-space mixer: chunked SSD scan and recurrent decode.
//!
//! # Recurrence
//!
//! A Mamba-2 layer keeps one matrix state `S[h] ∈ R^{P×N}` per head `h`
//! (`P = head_dim` value channels, `N = state_size` state coordinates). For
//! token `t`, with `A[h] = -exp(A_log[h]) < 0` and an input-selected step
//! `dt_t[h] > 0`,
//!
//! ```text
//! S_t[h,p,n] = exp(dt_t[h] A[h]) S_{t-1}[h,p,n] + dt_t[h] x_t[h,p] B_t[g(h),n]
//! y_t[h,p]   = sum_n S_t[h,p,n] C_t[g(h),n] + D[h] x_t[h,p]
//! ```
//!
//! `B` and `C` are projected once per *group* `g` and shared by the
//! `num_heads / groups` contiguous heads `h` with `g(h) = h / (num_heads /
//! groups)`. The decay `exp(dt A)` lies in `(0, 1)`, so the state is a
//! learned exponential moving sum of outer products `x_t B_tᵀ`.
//!
//! # Layer graph
//!
//! The fused input projection emits `[z, x, B, C, dt]`. A depthwise causal
//! convolution of width `K` (with SiLU) mixes the `x/B/C` channels over the
//! previous `K - 1` tokens; `dt = softplus(dt + dt_bias)` is optionally
//! clamped; the scan produces `y`; and the output is
//! `out_proj(RMSNorm_group(y * SiLU(z)))`. Decoding therefore carries two
//! states per layer: the matrix `S` and the last `K - 1` convolution inputs.
//!
//! # Execution modes
//!
//! [`Mamba2Mode::Recurrent`] is the transparent token loop of the reference
//! `selective_state_update`, used for single-token decode and as a golden
//! oracle. [`Mamba2Mode::ChunkedPrefill`] evaluates the same recurrence in
//! the state-space-duality (SSD) form: inside a chunk of `T` tokens,
//!
//! ```text
//! y_t = sum_{s<=t} (C_t · B_s) L[t,s] dt_s x_s + C_t S_0 exp(cum_t) + D x_t
//! L[t,s] = exp(sum_{u=s+1..t} dt_u A)   (zero for s > t)
//! ```
//!
//! a masked semiseparable matrix product. Like the reference SSD algorithm,
//! the scan evaluates many chunks at once: intra-chunk outputs and per-chunk
//! states for a whole block of chunks, then the chunk-entry states through one
//! segmented decay product. `C · Bᵀ` is formed once per *group* and broadcast
//! over the group's heads, so `B` and `C` are never copied per head. A block
//! holds at most `2^24` F32 elements per `[batch, heads, chunks, T, T]` score
//! plane (64 MiB) and only the terminal state crosses block boundaries, so
//! prefill memory is independent of the prompt length.
//!
//! # Batched rows
//!
//! [`Mamba2::forward_rows`] advances a right-padded batch whose rows differ in
//! length and mode in one pass per mode: padding is an identity transition
//! (`dt = 0`, zero inputs) and each row's convolution history is taken at its
//! own end. [`Mamba2::forward_rows_with_snapshots`] additionally returns the
//! state after every token, which lets a speculative verifier evaluate all
//! drafted tokens in one pass and commit any accepted prefix.
//!
//! # Precision
//!
//! The chunked scan promotes `x`, `B`, `C`, `dt`, and all state arithmetic
//! to F32 and returns F32 outputs, like the reference `chunk_scan` fallback.
//! The recurrent path follows the reference single-step update: decay and
//! state accumulation are F32, but `(dt B) x`, the state-to-`C` contraction,
//! and the `D` skip keep the activation dtype. External state may be stored
//! as F32, F16, or BF16; a reduced-precision state is rounded after every
//! recurrent token, exactly as an in-place reference cache would be.
//!
//! # Masks and caches
//!
//! A [`TokenMask`] is semantic: `false` means no recurrent time passes, so
//! `dt = 0` and the state transition is an exact identity. (Transformers'
//! padding helper zeroes `x/B/C` but leaves `dt` active, which still decays
//! the state; stateful callers here compact padding away instead.) A masked
//! call keeps the reference convolution's physical-column history, so callers
//! that persist convolution state must not leave interior holes. Every path
//! returns a new [`Mamba2State`] instead of mutating its input, which lets an
//! architecture publish all cache layers atomically.

use burn::{
    module::{Module, Param},
    nn::{
        Initializer, Linear, LinearConfig, LinearLayout, PaddingConfig1d,
        conv::{Conv1d, Conv1dConfig},
    },
    tensor::{
        Bool, DType, Int, Tensor, TensorData, activation, backend::Backend, module::conv1d,
        ops::ConvOptions,
    },
};
use serde::{Deserialize, Serialize};

use crate::{Error, Result, architecture::TokenMask, delta_net::softplus, operations};

/// Closed interval `[lower, upper]` for discretization steps; `upper = None` is `+inf`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct TimeStepLimits {
    /// Finite, non-negative lower bound.
    pub lower: f64,
    /// Optional finite upper bound, at least `lower`.
    pub upper: Option<f64>,
}

impl TimeStepLimits {
    /// Creates `[lower, +inf)`.
    #[must_use]
    pub const fn at_least(lower: f64) -> Self {
        Self { lower, upper: None }
    }

    /// Rejects non-finite, negative, or inverted bounds.
    fn validate(self, field: &str) -> Result<()> {
        if !self.lower.is_finite() || self.lower < 0.0 {
            return Err(Error::configuration(
                field,
                "lower discretization bound must be finite and non-negative",
            ));
        }
        if self
            .upper
            .is_some_and(|upper| !upper.is_finite() || upper < self.lower)
        {
            return Err(Error::configuration(
                field,
                "upper discretization bound must be finite and no smaller than the lower bound",
            ));
        }
        Ok(())
    }
}

/// Which execution paths clamp the discretization step `dt`, and to what interval.
///
/// Reference implementations disagree, so the policy is explicit data rather
/// than two loosely named scalars. Hugging Face Transformers' Mamba-2 family
/// fallbacks (including Nemotron-H at the revision this crate pins) clamp only
/// the chunked prefill scan, to `[time_step_min, +inf)`, and leave the
/// single-token update unclamped; NVIDIA's Nemotron Nano remote code clamps
/// both paths to the configured `time_step_limit`. These bounds are runtime
/// limits and are unrelated to the `time_step_min/max` *initialization*
/// range of `dt_bias`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct TimeStepClamp {
    /// Interval applied by [`Mamba2Mode::ChunkedPrefill`]; `None` leaves `dt` unclamped.
    pub chunked: Option<TimeStepLimits>,
    /// Interval applied by [`Mamba2Mode::Recurrent`]; `None` leaves `dt` unclamped.
    pub recurrent: Option<TimeStepLimits>,
}

impl TimeStepClamp {
    /// Clamps neither path.
    #[must_use]
    pub const fn unclamped() -> Self {
        Self {
            chunked: None,
            recurrent: None,
        }
    }

    /// Clamps only chunked prefill, as the Transformers Mamba-2 fallbacks do.
    #[must_use]
    pub const fn chunked_only(limits: TimeStepLimits) -> Self {
        Self {
            chunked: Some(limits),
            recurrent: None,
        }
    }

    /// Clamps both paths to one interval, as NVIDIA's Nemotron Nano remote code does.
    #[must_use]
    pub const fn both(limits: TimeStepLimits) -> Self {
        Self {
            chunked: Some(limits),
            recurrent: Some(limits),
        }
    }

    /// Returns the interval applied by `mode`, if any.
    #[must_use]
    pub const fn for_mode(self, mode: Mamba2Mode) -> Option<TimeStepLimits> {
        match mode {
            Mamba2Mode::Recurrent => self.recurrent,
            Mamba2Mode::ChunkedPrefill => self.chunked,
        }
    }

    /// Validates every configured interval.
    fn validate(self) -> Result<()> {
        if let Some(limits) = self.chunked {
            limits.validate("mamba2.time_step_clamp.chunked")?;
        }
        if let Some(limits) = self.recurrent {
            limits.validate("mamba2.time_step_clamp.recurrent")?;
        }
        Ok(())
    }
}

/// Numerical and tensor geometry of one Mamba-2 mixer.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Mamba2Config {
    /// Residual-stream width `H` consumed and produced by the mixer.
    pub hidden_size: usize,
    /// Number of independent state-space heads.
    pub num_heads: usize,
    /// Value channels `P` per head.
    pub head_dim: usize,
    /// State coordinates `N` retained for every head channel.
    pub state_size: usize,
    /// Number of groups sharing token-selected `B/C` vectors; divides `num_heads`.
    pub groups: usize,
    /// Width `K >= 2` of the depthwise causal convolution.
    pub convolution_kernel: usize,
    /// Maximum chunk width of the SSD prefill scan.
    pub chunk_size: usize,
    /// Runtime clamp of discretization steps per execution path.
    pub time_step_clamp: TimeStepClamp,
    /// Epsilon of the gated grouped RMS normalization.
    pub normalization_epsilon: f64,
    /// Whether the fused input and output projections carry a bias.
    pub projection_bias: bool,
    /// Whether the depthwise convolution carries a bias.
    pub convolution_bias: bool,
}

impl Mamba2Config {
    /// Returns the expanded value width `num_heads * head_dim`.
    ///
    /// # Errors
    ///
    /// Returns a configuration error when the product overflows `usize`.
    pub fn intermediate_size(&self) -> Result<usize> {
        self.num_heads
            .checked_mul(self.head_dim)
            .ok_or_else(|| Error::configuration("mamba2", "intermediate width overflows usize"))
    }

    /// Returns convolution channels `intermediate + 2 * groups * state_size` (`x`, `B`, `C`).
    ///
    /// # Errors
    ///
    /// Returns a configuration error when a product or sum overflows `usize`.
    pub fn convolution_channels(&self) -> Result<usize> {
        let state_channels = self
            .groups
            .checked_mul(self.state_size)
            .and_then(|value| value.checked_mul(2))
            .ok_or_else(|| Error::configuration("mamba2", "B/C channel width overflows usize"))?;
        self.intermediate_size()?
            .checked_add(state_channels)
            .ok_or_else(|| Error::configuration("mamba2", "convolution width overflows usize"))
    }

    /// Returns the fused projection width of `[z, x, B, C, dt]`.
    ///
    /// # Errors
    ///
    /// Returns a configuration error when a sum overflows `usize`.
    pub fn projection_size(&self) -> Result<usize> {
        self.intermediate_size()?
            .checked_add(self.convolution_channels()?)
            .and_then(|value| value.checked_add(self.num_heads))
            .ok_or_else(|| Error::configuration("mamba2", "projection width overflows usize"))
    }

    /// Validates every divisibility, extent, and numerical invariant.
    ///
    /// # Errors
    ///
    /// Returns a contextual configuration error for zero widths, a kernel
    /// below two, groups that do not divide the heads, an invalid clamp, a
    /// non-positive epsilon, or overflowing projection widths.
    pub fn validate(&self) -> Result<()> {
        if self.hidden_size == 0
            || self.num_heads == 0
            || self.head_dim == 0
            || self.state_size == 0
            || self.groups == 0
            || self.convolution_kernel < 2
            || self.chunk_size == 0
        {
            return Err(Error::configuration(
                "mamba2",
                "all widths must be non-zero and convolution_kernel must be at least two",
            ));
        }
        if !self.num_heads.is_multiple_of(self.groups) {
            return Err(Error::configuration(
                "mamba2.groups",
                "must divide num_heads exactly",
            ));
        }
        self.time_step_clamp.validate()?;
        if !self.normalization_epsilon.is_finite() || self.normalization_epsilon <= 0.0 {
            return Err(Error::configuration(
                "mamba2.normalization_epsilon",
                "must be finite and greater than zero",
            ));
        }
        let _ = self.projection_size()?;
        Ok(())
    }

    /// Initializes one checkpoint-compatible mixer.
    ///
    /// Projections use the checkpoint-native `[out, in]` layout. `dt_bias`,
    /// `D`, and the norm scale start at one and `A_log = log(1..=num_heads)`,
    /// the Mamba-2 reference initialization of the decay magnitudes.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::validate`].
    pub fn init<B: Backend>(&self, device: &B::Device) -> Result<Mamba2<B>> {
        self.validate()?;
        let intermediate = self.intermediate_size()?;
        let convolution_channels = self.convolution_channels()?;
        let linear = |input, output, bias| {
            operations::init_linear(
                &LinearConfig::new(input, output)
                    .with_bias(bias)
                    .with_layout(LinearLayout::Col),
                device,
            )
        };
        let a = (1..=self.num_heads)
            .map(|value| value as f32)
            .collect::<Vec<_>>();
        Ok(Mamba2 {
            in_proj: linear(
                self.hidden_size,
                self.projection_size()?,
                self.projection_bias,
            ),
            conv1d: Conv1dConfig::new(
                convolution_channels,
                convolution_channels,
                self.convolution_kernel,
            )
            .with_groups(convolution_channels)
            .with_bias(self.convolution_bias)
            .with_padding(PaddingConfig1d::Valid)
            .init(device),
            dt_bias: Initializer::Ones.init([self.num_heads], device),
            a_log: Param::from_tensor(
                Tensor::from_data(TensorData::new(a, [self.num_heads]), device).log(),
            ),
            d: Initializer::Ones.init([self.num_heads], device),
            norm: GatedGroupRmsNorm {
                weight: Initializer::Ones.init([intermediate], device),
                groups: self.groups,
                epsilon: self.normalization_epsilon,
            },
            out_proj: linear(intermediate, self.hidden_size, self.projection_bias),
            hidden_size: self.hidden_size,
            num_heads: self.num_heads,
            head_dim: self.head_dim,
            state_size: self.state_size,
            groups: self.groups,
            convolution_kernel: self.convolution_kernel,
            chunk_size: self.chunk_size,
            time_step_clamp: self.time_step_clamp,
            convolution_channels,
        })
    }
}

/// Selects the transparent recurrent loop or the chunked SSD scan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mamba2Mode {
    /// One explicit state update per token (the reference decode step).
    Recurrent,
    /// Masked semiseparable scan inside bounded chunks (the reference prefill).
    ChunkedPrefill,
}

/// External terminal state of one Mamba-2 layer.
#[derive(Clone, Debug)]
pub struct Mamba2State<B: Backend> {
    /// Selective state `[batch, heads, head_dim, state_size]` in F32, F16, or BF16.
    pub recurrent: Tensor<B, 4>,
    /// The last `K - 1` projected convolution inputs `[batch, channels, K - 1]`.
    pub convolution: Tensor<B, 3>,
}

impl<B: Backend> Mamba2State<B> {
    /// Allocates F32 zero matrix state with an explicit convolution dtype.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::zeros_with_dtypes`].
    pub fn zeros(
        config: &Mamba2Config,
        batch_size: usize,
        convolution_dtype: DType,
        device: &B::Device,
    ) -> Result<Self> {
        Self::zeros_with_dtypes(config, batch_size, DType::F32, convolution_dtype, device)
    }

    /// Allocates zero state with explicit matrix and convolution dtypes.
    ///
    /// # Errors
    ///
    /// Returns a configuration error for an invalid config, a zero batch, a
    /// matrix dtype other than F32/F16/BF16, or a non-float convolution dtype.
    pub fn zeros_with_dtypes(
        config: &Mamba2Config,
        batch_size: usize,
        recurrent_dtype: DType,
        convolution_dtype: DType,
        device: &B::Device,
    ) -> Result<Self> {
        config.validate()?;
        if batch_size == 0 {
            return Err(Error::configuration(
                "mamba2.batch_size",
                "must be non-zero",
            ));
        }
        if !matches!(recurrent_dtype, DType::F32 | DType::F16 | DType::BF16)
            || !convolution_dtype.is_float()
        {
            return Err(Error::configuration(
                "mamba2.state_dtype",
                "recurrent storage must be F32/F16/BF16 and convolution storage must be floating-point",
            ));
        }
        Ok(Self {
            recurrent: Tensor::zeros(
                [
                    batch_size,
                    config.num_heads,
                    config.head_dim,
                    config.state_size,
                ],
                (device, recurrent_dtype),
            ),
            convolution: Tensor::zeros(
                [
                    batch_size,
                    config.convolution_channels()?,
                    config.convolution_kernel - 1,
                ],
                (device, convolution_dtype),
            ),
        })
    }
}

/// Groupwise RMS normalization of the SiLU-gated scan output.
#[derive(Module, Debug)]
pub struct GatedGroupRmsNorm<B: Backend> {
    /// Per-channel learned scale `[intermediate]`.
    pub weight: Param<Tensor<B, 1>>,
    /// Number of independent normalization groups.
    pub groups: usize,
    /// Positive denominator epsilon.
    pub epsilon: f64,
}

impl<B: Backend> GatedGroupRmsNorm<B> {
    /// Computes `RMSNorm_group(y * SiLU(gate))` with the Zamba2/Mamba-2 operation graph.
    ///
    /// Both the gate and the group variance are evaluated in F32. The unit-RMS
    /// activation is rounded back to the input dtype *before* the learned
    /// scale multiplies it, and the reciprocal square root is multiplied
    /// rather than divided by: the two are algebraically equivalent but differ
    /// by one F16 ULP, and the pinned reference multiplies.
    ///
    /// # Errors
    ///
    /// Returns a configuration error for a non-positive epsilon and a shape
    /// error when the gate, grouping, or scale do not match `values`.
    pub fn forward(&self, values: Tensor<B, 3>, gate: Tensor<B, 3>) -> Result<Tensor<B, 3>> {
        let [batch, sequence, width] = values.dims();
        if !self.epsilon.is_finite() || self.epsilon <= 0.0 {
            return Err(Error::configuration(
                "mamba2.gated_norm.epsilon",
                "must be finite and greater than zero",
            ));
        }
        if gate.dims() != [batch, sequence, width]
            || self.groups == 0
            || !width.is_multiple_of(self.groups)
            || self.weight.shape().dims() != [width]
        {
            return Err(Error::shape(
                "mamba2.gated_norm",
                format!("matching [batch, sequence, {width}] tensors and {width} scale values"),
                format!("values={:?}, gate={:?}", values.dims(), gate.dims()),
            ));
        }
        let dtype = values.dtype();
        let group_width = width / self.groups;
        let gated = values.cast(DType::F32) * activation::silu(gate.cast(DType::F32));
        let grouped = gated.reshape([batch, sequence, self.groups, group_width]);
        let inverse_rms = (grouped.clone().square().mean_dim(3) + self.epsilon)
            .sqrt()
            .recip();
        let normalized = (grouped * inverse_rms)
            .reshape([batch, sequence, width])
            .cast(dtype);
        Ok(normalized * self.weight.val().cast(dtype).reshape([1, 1, width]))
    }
}

/// Checkpoint-compatible Mamba-2 mixer with external state.
#[derive(Module, Debug)]
pub struct Mamba2<B: Backend> {
    /// Fused `[z, x, B, C, dt]` input projection.
    pub in_proj: Linear<B>,
    /// Depthwise causal convolution over the `x/B/C` channels.
    pub conv1d: Conv1d<B>,
    /// Learned inverse-softplus offset of the discretization step, `[num_heads]`.
    pub dt_bias: Param<Tensor<B, 1>>,
    /// Logarithm of the positive continuous-time decay magnitudes, `[num_heads]`.
    pub a_log: Param<Tensor<B, 1>>,
    /// Learned per-head direct skip coefficient `D`, `[num_heads]`.
    pub d: Param<Tensor<B, 1>>,
    /// Gated grouped RMS normalization before the output projection.
    pub norm: GatedGroupRmsNorm<B>,
    /// Projection from expanded head channels back to the residual stream.
    pub out_proj: Linear<B>,
    /// Residual-stream width.
    pub hidden_size: usize,
    /// State-space head count.
    pub num_heads: usize,
    /// Value channels per head.
    pub head_dim: usize,
    /// State coordinates per value channel.
    pub state_size: usize,
    /// Number of `B/C` sharing groups.
    pub groups: usize,
    /// Causal convolution kernel width.
    pub convolution_kernel: usize,
    /// Maximum SSD chunk width.
    pub chunk_size: usize,
    /// Runtime discretization clamp per execution path.
    #[module(skip)]
    pub time_step_clamp: TimeStepClamp,
    /// Number of `x/B/C` convolution channels.
    pub convolution_channels: usize,
}

impl<B: Backend> Mamba2<B> {
    /// Reconstructs the validated allocation configuration.
    pub fn config(&self) -> Mamba2Config {
        Mamba2Config {
            hidden_size: self.hidden_size,
            num_heads: self.num_heads,
            head_dim: self.head_dim,
            state_size: self.state_size,
            groups: self.groups,
            convolution_kernel: self.convolution_kernel,
            chunk_size: self.chunk_size,
            time_step_clamp: self.time_step_clamp,
            normalization_epsilon: self.norm.epsilon,
            projection_bias: self.in_proj.bias.is_some(),
            convolution_bias: self.conv1d.bias.is_some(),
        }
    }

    /// Runs a stateless, differentiable segment from zero initial state.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::forward_with_state`].
    pub fn forward(
        &self,
        hidden: Tensor<B, 3>,
        token_mask: Option<&TokenMask<B>>,
        mode: Mamba2Mode,
    ) -> Result<Tensor<B, 3>> {
        let state = Mamba2State::zeros(
            &self.config(),
            hidden.dims()[0],
            hidden.dtype(),
            &hidden.device(),
        )?;
        self.forward_with_state(hidden, token_mask, &state, mode)
            .map(|(output, _)| output)
    }

    /// Runs a prompt or decode segment from `state` and returns the unpublished terminal state.
    ///
    /// `hidden` is `[batch, sequence, hidden_size]`; the optional mask is
    /// `[batch, sequence]` with `true == valid`. The returned convolution state
    /// holds the last `K - 1` physical columns (see the module documentation).
    ///
    /// # Errors
    ///
    /// Returns shape errors for mismatched inputs or state, and a backend
    /// capability error for tensors on different devices.
    pub fn forward_with_state(
        &self,
        hidden: Tensor<B, 3>,
        token_mask: Option<&TokenMask<B>>,
        state: &Mamba2State<B>,
        mode: Mamba2Mode,
    ) -> Result<(Tensor<B, 3>, Mamba2State<B>)> {
        let [batch, sequence, _] = self.validate_hidden(&hidden)?;
        let valid = match token_mask {
            Some(mask) => {
                if mask.as_tensor().dims() != [batch, sequence] {
                    return Err(Error::shape(
                        "mamba2.token_mask",
                        format!("[{batch}, {sequence}]"),
                        format!("{:?}", mask.as_tensor().dims()),
                    ));
                }
                if mask.as_tensor().device() != hidden.device() {
                    return Err(Error::BackendCapability(
                        "Mamba-2 input and token mask must share a device".into(),
                    ));
                }
                Some(mask.as_tensor().clone())
            }
            None => None,
        };
        let run = self.run(
            hidden,
            valid,
            state,
            mode,
            &ConvolutionTail::Physical,
            false,
        )?;
        Ok((run.output, run.state))
    }

    /// Advances right-padded rows that may differ in length and execution mode.
    ///
    /// Row `b` of `hidden` holds `rows[b].tokens` valid tokens in its first
    /// columns; later columns are padding that neither advances the state nor
    /// reaches the convolution history, whose last `K - 1` inputs are taken at
    /// each row's own end. A row with zero tokens keeps its state exactly and
    /// its outputs are unspecified. Rows are grouped by mode and each group
    /// runs as one batch, so an ordinary decode step (every row recurrent) or
    /// prefill (every row chunked) is a single mixer pass.
    ///
    /// # Errors
    ///
    /// Returns a shape error when `rows` does not describe every batch row or
    /// claims more tokens than `hidden` has columns, and the errors of
    /// [`Self::forward_with_state`].
    pub fn forward_rows(
        &self,
        hidden: Tensor<B, 3>,
        rows: &[Mamba2Row],
        state: &Mamba2State<B>,
    ) -> Result<(Tensor<B, 3>, Mamba2State<B>)> {
        let [batch, sequence, _] = self.validate_hidden(&hidden)?;
        let counts = rows.iter().map(|row| row.tokens).collect::<Vec<_>>();
        validate_row_counts(&counts, batch, sequence)?;
        let groups = mode_groups(rows);
        let device = hidden.device();
        if let [(mode, _)] = groups.as_slice() {
            let valid = right_padded_validity::<B>(&counts, sequence, &device);
            let run = self.run(
                hidden,
                valid,
                state,
                *mode,
                &ConvolutionTail::RowEnds(counts),
                false,
            )?;
            return Ok((run.output, run.state));
        }
        let mut outputs = Vec::with_capacity(groups.len());
        let mut matrices = Vec::with_capacity(groups.len());
        let mut convolutions = Vec::with_capacity(groups.len());
        let mut order = Vec::with_capacity(batch);
        for (mode, members) in groups {
            let selector = row_selector::<B>(&members, &device)?;
            let group_counts = members.iter().map(|&row| counts[row]).collect::<Vec<_>>();
            let group_state = Mamba2State {
                recurrent: state.recurrent.clone().select(0, selector.clone()),
                convolution: state.convolution.clone().select(0, selector.clone()),
            };
            let run = self.run(
                hidden.clone().select(0, selector),
                right_padded_validity::<B>(&group_counts, sequence, &device),
                &group_state,
                mode,
                &ConvolutionTail::RowEnds(group_counts),
                false,
            )?;
            outputs.push(run.output);
            matrices.push(run.state.recurrent);
            convolutions.push(run.state.convolution);
            order.extend(members);
        }
        // `order[i]` is the batch row at concatenated position `i`; invert it.
        let mut inverse = vec![0; batch];
        for (position, &row) in order.iter().enumerate() {
            inverse[row] = position;
        }
        let restore = row_selector::<B>(&inverse, &device)?;
        Ok((
            Tensor::cat(outputs, 0).select(0, restore.clone()),
            Mamba2State {
                recurrent: Tensor::cat(matrices, 0).select(0, restore.clone()),
                convolution: Tensor::cat(convolutions, 0).select(0, restore),
            },
        ))
    }

    /// Advances right-padded rows token by token and returns the state after every token.
    ///
    /// This is the recurrent update applied to `token_counts[b]` tokens of row
    /// `b`, exactly as that many single-token decode calls would apply it
    /// (including rounding through a reduced-precision state dtype). Snapshot
    /// `t` of a row is its state after token `t`, which lets a speculative
    /// verifier commit any accepted prefix of one batched pass. Snapshots at
    /// or beyond a row's token count are unspecified.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::forward_rows`].
    pub fn forward_rows_with_snapshots(
        &self,
        hidden: Tensor<B, 3>,
        token_counts: &[usize],
        state: &Mamba2State<B>,
    ) -> Result<(Tensor<B, 3>, Mamba2Snapshots<B>)> {
        let [batch, sequence, _] = self.validate_hidden(&hidden)?;
        validate_row_counts(token_counts, batch, sequence)?;
        let valid = right_padded_validity::<B>(token_counts, sequence, &hidden.device());
        let run = self.run(
            hidden,
            valid,
            state,
            Mamba2Mode::Recurrent,
            &ConvolutionTail::RowEnds(token_counts.to_vec()),
            true,
        )?;
        let snapshots = run.snapshots.ok_or_else(|| {
            Error::shape(
                "mamba2.snapshots",
                "per-token recurrent states",
                "no recorded snapshots",
            )
        })?;
        Ok((run.output, snapshots))
    }

    /// Checks the `[batch>0, sequence>0, hidden_size]` input extent.
    fn validate_hidden(&self, hidden: &Tensor<B, 3>) -> Result<[usize; 3]> {
        let [batch, sequence, width] = hidden.dims();
        if batch == 0 || sequence == 0 || width != self.hidden_size {
            return Err(Error::shape(
                "mamba2.hidden",
                format!("[batch>0, sequence>0, {}]", self.hidden_size),
                format!("[{batch}, {sequence}, {width}]"),
            ));
        }
        Ok([batch, sequence, width])
    }

    /// Shared core of every entry point: projection, convolution, scan, norm, and output.
    ///
    /// `valid` is the `[batch, sequence]` semantic mask (`None` = all valid).
    /// `tail` chooses which `K - 1` convolution inputs become the next history,
    /// and `snapshots` (recurrent mode only) records the state after every token.
    fn run(
        &self,
        hidden: Tensor<B, 3>,
        valid: Option<Tensor<B, 2, Bool>>,
        state: &Mamba2State<B>,
        mode: Mamba2Mode,
        tail: &ConvolutionTail,
        snapshots: bool,
    ) -> Result<ScanRun<B>> {
        let [batch, sequence, _] = hidden.dims();
        let hidden_dtype = hidden.dtype();
        let device = hidden.device();
        let recurrent_dtype = state.recurrent.dtype();
        self.validate_state(state, batch, hidden_dtype, &device)?;
        // Invalid tokens are *replaced* by zeros rather than multiplied by a
        // zero mask, so a non-finite value in a padded slot (for example the
        // output of a fully masked attention row upstream) can never become
        // `0 * inf = NaN` and leak into the state.
        let invalid = valid.as_ref().map(|valid| valid.clone().bool_not());
        let masked = zero_invalid(hidden, invalid.as_ref());
        let projected = operations::linear(masked, &self.in_proj)?;
        let intermediate = self.num_heads * self.head_dim;
        let gate = projected.clone().narrow(2, 0, intermediate);
        let xbc = projected
            .clone()
            .narrow(2, intermediate, self.convolution_channels)
            .swap_dims(1, 2);
        let dt = projected.narrow(2, intermediate + self.convolution_channels, self.num_heads);

        // `joined` is `[batch, channels, (K - 1) + sequence]`: history, then inputs.
        let history = state.convolution.clone().cast(xbc.dtype());
        let joined = Tensor::cat(vec![history, xbc], 2);
        let convolved = conv1d(
            joined.clone(),
            self.conv1d.weight.val(),
            self.conv1d.bias.as_ref().map(|bias| bias.val()),
            ConvOptions::new([1], [0], [1], self.conv1d.groups),
        );
        let convolved = zero_invalid(
            activation::silu(convolved).swap_dims(1, 2),
            invalid.as_ref(),
        );
        let group_state_width = self.groups * self.state_size;
        let x = convolved.clone().narrow(2, 0, intermediate).reshape([
            batch,
            sequence,
            self.num_heads,
            self.head_dim,
        ]);
        let b = convolved
            .clone()
            .narrow(2, intermediate, group_state_width)
            .reshape([batch, sequence, self.groups, self.state_size]);
        let c = convolved
            .narrow(2, intermediate + group_state_width, group_state_width)
            .reshape([batch, sequence, self.groups, self.state_size]);

        let dt = zero_invalid(
            discretization_steps(dt, self.dt_bias.val(), self.time_step_clamp.for_mode(mode)),
            invalid.as_ref(),
        );
        let a = -self.a_log.val().cast(DType::F32).exp();
        let (scan, recurrent, matrices) = match mode {
            Mamba2Mode::Recurrent => recurrent_scan(
                x,
                b,
                c,
                dt,
                a,
                self.d.val(),
                state.recurrent.clone(),
                snapshots,
            )?,
            Mamba2Mode::ChunkedPrefill => {
                let (scan, recurrent) = chunked_scan(
                    x,
                    b,
                    c,
                    dt,
                    a,
                    self.d.val().cast(DType::F32),
                    state.recurrent.clone().cast(DType::F32),
                    self.chunk_size,
                )?;
                (scan, recurrent, None)
            }
        };
        let scan = scan.reshape([batch, sequence, intermediate]);
        let normalized = self.norm.forward(scan, gate)?;
        let output = operations::linear(normalized.cast(hidden_dtype), &self.out_proj)?;
        let width = self.convolution_kernel - 1;
        let snapshots = match matrices {
            Some(recurrent) => Some(Mamba2Snapshots {
                recurrent,
                convolution: convolution_windows(joined.clone(), sequence, width)?,
            }),
            None => None,
        };
        Ok(ScanRun {
            output,
            state: Mamba2State {
                recurrent: recurrent.cast(recurrent_dtype),
                convolution: tail.select(joined, sequence, width)?,
            },
            snapshots,
        })
    }

    /// Checks external matrix and convolution geometry, dtypes, and devices before tensor math.
    fn validate_state(
        &self,
        state: &Mamba2State<B>,
        batch: usize,
        convolution_dtype: DType,
        device: &B::Device,
    ) -> Result<()> {
        let recurrent = [batch, self.num_heads, self.head_dim, self.state_size];
        let convolution = [
            batch,
            self.convolution_channels,
            self.convolution_kernel - 1,
        ];
        if state.recurrent.dims() != recurrent
            || !matches!(
                state.recurrent.dtype(),
                DType::F32 | DType::F16 | DType::BF16
            )
        {
            return Err(Error::shape(
                "mamba2.state.recurrent",
                format!("{recurrent:?} F32/F16/BF16"),
                format!("{:?} {:?}", state.recurrent.dims(), state.recurrent.dtype()),
            ));
        }
        if state.convolution.dims() != convolution || state.convolution.dtype() != convolution_dtype
        {
            return Err(Error::shape(
                "mamba2.state.convolution",
                format!("{convolution:?} {convolution_dtype:?}"),
                format!(
                    "{:?} {:?}",
                    state.convolution.dims(),
                    state.convolution.dtype()
                ),
            ));
        }
        if &state.recurrent.device() != device || &state.convolution.device() != device {
            return Err(Error::BackendCapability(
                "Mamba-2 input and recurrent state must share a device".into(),
            ));
        }
        Ok(())
    }
}

/// One right-padded batch row of [`Mamba2::forward_rows`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Mamba2Row {
    /// Number of valid tokens at the start of the row; zero leaves its state unchanged.
    pub tokens: usize,
    /// Execution mode of the row's segment.
    pub mode: Mamba2Mode,
}

/// The state of every row after each token of a recurrent segment.
#[derive(Clone, Debug)]
pub struct Mamba2Snapshots<B: Backend> {
    /// Matrices `[batch, tokens, heads, head_dim, state_size]` in the state dtype.
    pub recurrent: Tensor<B, 5>,
    /// Convolution histories `[batch, tokens, channels, K - 1]`.
    pub convolution: Tensor<B, 4>,
}

/// Output, terminal state, and optional per-token snapshots of one mixer pass.
struct ScanRun<B: Backend> {
    /// Mixer output `[batch, sequence, hidden_size]`.
    output: Tensor<B, 3>,
    /// Terminal state after every row's segment.
    state: Mamba2State<B>,
    /// States after every token, when requested in recurrent mode.
    snapshots: Option<Mamba2Snapshots<B>>,
}

/// Which `K - 1` convolution inputs become the next history.
#[derive(Debug, Clone, PartialEq, Eq)]
enum ConvolutionTail {
    /// The last `K - 1` physical columns, whatever the mask says.
    Physical,
    /// Columns `count_b .. count_b + K - 1` of `[history, inputs]`: each row's own end.
    RowEnds(
        /// Valid tokens per row, all at the start of the row.
        Vec<usize>,
    ),
}

impl ConvolutionTail {
    /// Selects `[batch, channels, width]` history from `joined = [history, inputs]`.
    fn select<B: Backend>(
        &self,
        joined: Tensor<B, 3>,
        sequence: usize,
        width: usize,
    ) -> Result<Tensor<B, 3>> {
        let counts = match self {
            Self::RowEnds(counts) if counts.iter().any(|&count| count != sequence) => counts,
            // Every row ends at the last column: the physical tail is the row tail.
            Self::Physical | Self::RowEnds(_) => {
                return Ok(joined.narrow(2, sequence, width));
            }
        };
        let [batch, channels, _] = joined.dims();
        let mut starts = Vec::with_capacity(batch * width);
        for &count in counts {
            for offset in 0..width {
                starts.push(signed_index(count + offset)?);
            }
        }
        let index = Tensor::<B, 3, Int>::from_data(
            TensorData::new(starts, [batch, 1, width]),
            &joined.device(),
        )
        .expand([batch, channels, width]);
        Ok(joined.gather(2, index))
    }
}

/// Returns `[batch, tokens, channels, width]` convolution histories after every token.
///
/// The history after token `t` is columns `t + 1 .. t + 1 + width` of
/// `joined = [history, inputs]`; one `select` gathers all of them.
fn convolution_windows<B: Backend>(
    joined: Tensor<B, 3>,
    sequence: usize,
    width: usize,
) -> Result<Tensor<B, 4>> {
    let [batch, channels, _] = joined.dims();
    let mut columns = Vec::with_capacity(sequence * width);
    for token in 0..sequence {
        for offset in 0..width {
            columns.push(signed_index(token + 1 + offset)?);
        }
    }
    let index = Tensor::<B, 1, Int>::from_data(
        TensorData::new(columns, [sequence * width]),
        &joined.device(),
    );
    Ok(joined
        .select(2, index)
        .reshape([batch, channels, sequence, width])
        .swap_dims(1, 2))
}

/// Converts a host index to Burn's signed index representation.
fn signed_index(index: usize) -> Result<i64> {
    i64::try_from(index)
        .map_err(|_| Error::configuration("mamba2.index", "index exceeds the signed index range"))
}

/// Uploads batch-row indices as a `[rows]` selector.
fn row_selector<B: Backend>(rows: &[usize], device: &B::Device) -> Result<Tensor<B, 1, Int>> {
    let values = rows
        .iter()
        .map(|&row| signed_index(row))
        .collect::<Result<Vec<_>>>()?;
    Ok(Tensor::from_data(
        TensorData::new(values, [rows.len()]),
        device,
    ))
}

/// Checks one token count per row, none above the padded width.
fn validate_row_counts(counts: &[usize], batch: usize, sequence: usize) -> Result<()> {
    if counts.len() != batch || counts.iter().any(|&count| count > sequence) {
        return Err(Error::shape(
            "mamba2.rows",
            format!("{batch} token counts of at most {sequence}"),
            format!("{counts:?}"),
        ));
    }
    Ok(())
}

/// Builds the `[batch, sequence]` validity of right-padded rows, or `None` when all are full.
fn right_padded_validity<B: Backend>(
    counts: &[usize],
    sequence: usize,
    device: &B::Device,
) -> Option<Tensor<B, 2, Bool>> {
    if counts.iter().all(|&count| count == sequence) {
        return None;
    }
    let flags = counts
        .iter()
        .flat_map(|&count| (0..sequence).map(move |column| column < count))
        .collect::<Vec<_>>();
    Some(Tensor::from_data(
        TensorData::new(flags, [counts.len(), sequence]),
        device,
    ))
}

/// Partitions rows by execution mode, in first-appearance order of the modes.
///
/// Inactive rows (zero tokens) are identity transitions under either mode, so
/// they join the first active row's group instead of forcing a second pass.
fn mode_groups(rows: &[Mamba2Row]) -> Vec<(Mamba2Mode, Vec<usize>)> {
    let fallback = rows
        .iter()
        .find(|row| row.tokens > 0)
        .map_or(Mamba2Mode::ChunkedPrefill, |row| row.mode);
    let mut groups: Vec<(Mamba2Mode, Vec<usize>)> = Vec::new();
    for (index, row) in rows.iter().enumerate() {
        let mode = if row.tokens > 0 { row.mode } else { fallback };
        match groups.iter_mut().find(|(group, _)| *group == mode) {
            Some((_, members)) => members.push(index),
            None => groups.push((mode, vec![index])),
        }
    }
    groups
}

/// Applies the learned discretization bias, the positive-step transform, and the clamp.
///
/// The reference adds `dt_bias` and evaluates `softplus` while `dt` still has
/// the fused projection's dtype; PyTorch's reduced-precision softplus kernel
/// computes `x > 20 ? x : log1p(exp(x))` in F32 and rounds once. This
/// function reproduces that graph exactly: the bias addition rounds in the
/// projection dtype, the thresholded softplus (shared with Gated DeltaNet)
/// runs in F32, and the result returns to the projection dtype immediately.
/// Burn's own `softplus`, `log(exp(x) + 1)`, would overflow above `x = 88.7`
/// and round small F16 results to zero. Only the scan promotes the stored
/// result to F32. Callers give invalid semantic tokens `dt = 0`, an identity
/// transition, with [`zero_invalid`].
fn discretization_steps<B: Backend>(
    dt: Tensor<B, 3>,
    bias: Tensor<B, 1>,
    limits: Option<TimeStepLimits>,
) -> Tensor<B, 3> {
    let dtype = dt.dtype();
    let heads = dt.dims()[2];
    let biased = dt + bias.cast(dtype).reshape([1, 1, heads]);
    let mut dt = softplus(biased.cast(DType::F32)).cast(dtype);
    if let Some(limits) = limits {
        dt = dt.clamp_min(limits.lower);
        if let Some(upper) = limits.upper {
            dt = dt.clamp_max(upper);
        }
    }
    dt
}

/// Replaces every channel of invalid tokens (`invalid` is `[batch, sequence]`) with zero.
///
/// Valid entries are returned bit for bit, so this equals multiplying by a
/// `{0, 1}` mask for finite inputs while also clearing non-finite padding.
fn zero_invalid<B: Backend>(
    values: Tensor<B, 3>,
    invalid: Option<&Tensor<B, 2, Bool>>,
) -> Tensor<B, 3> {
    match invalid {
        Some(invalid) => {
            let shape = values.dims();
            values.mask_fill(invalid.clone().unsqueeze_dim::<3>(2).expand(shape), 0.0)
        }
        None => values,
    }
}

/// Expands one token's group-shared `B` or `C` `[batch, groups, state]` to every head.
///
/// Heads of a group are contiguous, so this is a broadcast followed by a
/// reshape to `[batch, heads, state]`; it materializes only one token.
fn heads_from_groups<B: Backend>(grouped: Tensor<B, 3>, heads: usize) -> Tensor<B, 3> {
    let [batch, groups, state] = grouped.dims();
    if groups == heads {
        return grouped;
    }
    grouped
        .unsqueeze_dim::<4>(2)
        .expand([batch, groups, heads / groups, state])
        .reshape([batch, heads, state])
}

/// Executes the reference single-token selective-state update for every token.
///
/// Inputs are `x [b, t, h, p]`, group-shared `b/c [b, t, g, n]`, `dt [b, t, h]`,
/// F32 `a [h]`, `d [h]`, and state `[b, h, p, n]`. The decay and the
/// accumulated state are F32. The input-dependent update is formed as
/// `(dt * B) * x` in the activation dtype, matching the reference's two
/// rounding points, before promotion into the state addition. For the output,
/// the new F32 state is rounded to `C`'s dtype before contraction and the
/// `D * x` skip is also evaluated there. When the state dtype is F16/BF16,
/// every iteration rounds through it, so a multi-token call equals repeated
/// single-token calls. With `collect`, the rounded state after every token is
/// returned as `[b, t, h, p, n]` snapshots.
///
/// Decode is one token, so the group-to-head expansion of `B` and `C` costs
/// one `[b, h, n]` copy per token and keeps the reference contraction layout.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(crate) fn recurrent_scan<B: Backend>(
    x: Tensor<B, 4>,
    b: Tensor<B, 4>,
    c: Tensor<B, 4>,
    dt: Tensor<B, 3>,
    a: Tensor<B, 1>,
    d: Tensor<B, 1>,
    state: Tensor<B, 4>,
    collect: bool,
) -> Result<(Tensor<B, 4>, Tensor<B, 4>, Option<Tensor<B, 5>>)> {
    let [batch, sequence, heads, head_dim] = x.dims();
    let state_size = b.dims()[3];
    let activation_dtype = x.dtype();
    let state_dtype = state.dtype();
    let mut state = state.cast(DType::F32);
    let mut outputs = Vec::with_capacity(sequence);
    let mut snapshots = Vec::with_capacity(if collect { sequence } else { 0 });
    let a = a.reshape([1, heads]);
    let d = d.cast(activation_dtype).reshape([1, heads, 1]);
    for token in 0..sequence {
        let x_t = x.clone().narrow(1, token, 1).squeeze_dim::<3>(1);
        let b_t = heads_from_groups(b.clone().narrow(1, token, 1).squeeze_dim::<3>(1), heads);
        let c_t = heads_from_groups(c.clone().narrow(1, token, 1).squeeze_dim::<3>(1), heads);
        let dt_t = dt.clone().narrow(1, token, 1).squeeze_dim::<2>(1);
        let decay = (dt_t.clone().cast(DType::F32) * a.clone())
            .exp()
            .reshape([batch, heads, 1, 1]);
        let discretized_b = dt_t.reshape([batch, heads, 1, 1]) * b_t.unsqueeze_dim::<4>(2);
        let update = discretized_b * x_t.clone().unsqueeze_dim::<4>(3);
        let next_state = state * decay + update.cast(DType::F32);

        let state_for_output =
            next_state
                .clone()
                .cast(c_t.dtype())
                .reshape([batch * heads, head_dim, state_size]);
        let c_for_output = c_t.reshape([batch * heads, state_size, 1]);
        let output = state_for_output
            .matmul(c_for_output)
            .reshape([batch, heads, head_dim])
            + x_t * d.clone();
        outputs.push(output.unsqueeze_dim::<4>(1));

        // `copy_(ssm_states.to(state.dtype))` in the reference makes reduced
        // cache precision observable on the next recurrent token.
        let stored = next_state.cast(state_dtype);
        if collect {
            snapshots.push(stored.clone().unsqueeze_dim::<5>(1));
        }
        state = stored.cast(DType::F32);
    }
    if state.dims() != [batch, heads, head_dim, state_size] {
        return Err(Error::shape(
            "mamba2.scan_state",
            format!("[{batch}, {heads}, {head_dim}, {state_size}]"),
            format!("{:?}", state.dims()),
        ));
    }
    Ok((
        Tensor::cat(outputs, 1),
        state.cast(state_dtype),
        collect.then(|| Tensor::cat(snapshots, 1)),
    ))
}

/// Largest number of score-plane elements one vectorized block of chunks may hold.
///
/// A block of `c` chunks of width `l` materializes `[batch, heads, c, l, l]`
/// F32 transition and weighted-kernel planes. `2^24` elements (64 MiB per
/// plane) admits 16 chunks of 128 tokens for 64 heads at batch one, which
/// replaces 16 sequential chunk iterations of about 40 kernels each by one
/// batched pass, while keeping prefill memory independent of prompt length.
pub(crate) const SCAN_BLOCK_ELEMENTS: usize = 1 << 24;

/// Executes the SSD scan over blocks of chunks, carrying only terminal state between blocks.
///
/// Inputs are `x [b, t, h, p]`, group-shared `b/c [b, t, g, n]`, `dt [b, t, h]`,
/// F32 `a [h]`, F32 `d [h]`, and the F32 initial state `[b, h, p, n]`.
/// Returns F32 outputs `[b, t, h, p]` and the F32 terminal state. Chunk
/// boundaries are at multiples of `chunk_size` from the segment start, as in
/// the reference; see [`scan_block`] for the per-block algebra.
#[allow(clippy::too_many_arguments)]
pub(crate) fn chunked_scan<B: Backend>(
    x: Tensor<B, 4>,
    b: Tensor<B, 4>,
    c: Tensor<B, 4>,
    dt: Tensor<B, 3>,
    a: Tensor<B, 1>,
    d: Tensor<B, 1>,
    state: Tensor<B, 4>,
    chunk_size: usize,
) -> Result<(Tensor<B, 4>, Tensor<B, 4>)> {
    chunked_scan_with_budget(x, b, c, dt, a, d, state, chunk_size, SCAN_BLOCK_ELEMENTS)
}

/// [`chunked_scan`] with an explicit per-block score-plane budget.
#[allow(clippy::too_many_arguments)]
pub(crate) fn chunked_scan_with_budget<B: Backend>(
    x: Tensor<B, 4>,
    b: Tensor<B, 4>,
    c: Tensor<B, 4>,
    dt: Tensor<B, 3>,
    a: Tensor<B, 1>,
    d: Tensor<B, 1>,
    mut state: Tensor<B, 4>,
    chunk_size: usize,
    budget: usize,
) -> Result<(Tensor<B, 4>, Tensor<B, 4>)> {
    let [batch, sequence, heads, head_dim] = x.dims();
    let [_, _, groups, state_size] = b.dims();
    if sequence == 0 || chunk_size == 0 || groups == 0 || !heads.is_multiple_of(groups) {
        return Err(Error::configuration(
            "mamba2.chunk",
            "the scan needs tokens, a non-zero chunk width, and groups dividing the heads",
        ));
    }
    let device = x.device();
    // Head-major F32 operands, permuted once for the whole segment.
    let x = x.cast(DType::F32).swap_dims(1, 2);
    let dt = dt.cast(DType::F32).swap_dims(1, 2);
    let b = b.cast(DType::F32).swap_dims(1, 2);
    let c = c.cast(DType::F32).swap_dims(1, 2);
    let a = a.reshape([1, heads, 1, 1]);
    let d = d.reshape([1, heads, 1, 1]);
    let chunks = sequence.div_ceil(chunk_size);
    let plane = batch
        .saturating_mul(heads)
        .saturating_mul(chunk_size)
        .saturating_mul(chunk_size)
        .max(1);
    let block_chunks = (budget / plane).clamp(1, chunks);
    let masks = SegmentMasks::new(chunk_size, &device);
    let mut outputs = Vec::with_capacity(chunks.div_ceil(block_chunks));
    let mut first_chunk = 0;
    while first_chunk < chunks {
        let count = block_chunks.min(chunks - first_chunk);
        let start = first_chunk * chunk_size;
        let width = (count * chunk_size).min(sequence - start);
        let padding = count * chunk_size - width;
        let operands = BlockOperands {
            x: pad_time(x.clone().narrow(2, start, width), padding)
                .reshape([batch, heads, count, chunk_size, head_dim]),
            dt: pad_time(
                dt.clone().narrow(2, start, width).unsqueeze_dim::<4>(3),
                padding,
            )
            .reshape([batch, heads, count, chunk_size]),
            b: pad_time(b.clone().narrow(2, start, width), padding)
                .reshape([batch, groups, count, chunk_size, state_size]),
            c: pad_time(c.clone().narrow(2, start, width), padding)
                .reshape([batch, groups, count, chunk_size, state_size]),
        };
        let x_block = operands.x.clone();
        let (output, next) = scan_block(operands, a.clone(), state, &masks)?;
        let output = (output + x_block * d.clone().unsqueeze_dim::<5>(4))
            .reshape([batch, heads, count * chunk_size, head_dim])
            .narrow(2, 0, width);
        outputs.push(output);
        state = next;
        first_chunk += count;
    }
    Ok((Tensor::cat(outputs, 2).swap_dims(1, 2), state))
}

/// Appends `padding` zero steps along the time axis (dimension 2) of a head- or group-major tensor.
///
/// Zero `x`, `B`, `C`, and `dt` are exact identity transitions: they leave the
/// state unchanged and their outputs are discarded.
fn pad_time<B: Backend>(tensor: Tensor<B, 4>, padding: usize) -> Tensor<B, 4> {
    if padding == 0 {
        return tensor;
    }
    let [batch, heads, _, width] = tensor.dims();
    let zeros = Tensor::zeros(
        [batch, heads, padding, width],
        (&tensor.device(), tensor.dtype()),
    );
    Tensor::cat(vec![tensor, zeros], 2)
}

/// One block of `c` whole chunks of width `l`, head- or group-major, in F32.
struct BlockOperands<B: Backend> {
    /// Values `[b, h, c, l, p]`.
    x: Tensor<B, 5>,
    /// Discretization steps `[b, h, c, l]`.
    dt: Tensor<B, 4>,
    /// Group-shared input projections `[b, g, c, l, n]`.
    b: Tensor<B, 5>,
    /// Group-shared output projections `[b, g, c, l, n]`.
    c: Tensor<B, 5>,
}

/// Evaluates every chunk of a block at once and returns `(outputs, terminal_state)`.
///
/// With `ld = dt A`, within-chunk cumulative decays `cum`, and chunk totals
/// `D_k = cum_k[l - 1]`, the algorithm (Mamba-2's SSD "minimal" form) is:
///
/// 1. intra-chunk outputs `Y_diag = ((C Bᵀ) ⊙ L) (dt x)` with the stable
///    segmented transitions `L[t,s] = exp(sum_{s<u<=t} ld[u])`; `C Bᵀ` is
///    computed once per *group* and broadcast over its heads;
/// 2. each chunk's state from zero, `S_k = sum_t exp(D_k - cum_t) dt_t x_t B_tᵀ`;
/// 3. the chunk-entry states `S_in(k) = exp(D_{k-1}) S_in(k-1) + S_{k-1}`
///    for all `k` at once, as one `[c+1, c+1]` segmented-decay matrix
///    (the same stable segment sum over `[0, D_0, …, D_{c-1}]`) applied to
///    `[S_0, S_1, …]` with the block's initial state prepended;
/// 4. the carried-state outputs `Y_off[t] = exp(cum_t) C_t · S_in(k)`.
///
/// `B` and `C` stay group-shared throughout: the state contractions broadcast
/// them across the heads of their group instead of copying them per head.
fn scan_block<B: Backend>(
    operands: BlockOperands<B>,
    a: Tensor<B, 4>,
    initial: Tensor<B, 4>,
    masks: &SegmentMasks<B>,
) -> Result<(Tensor<B, 5>, Tensor<B, 4>)> {
    let BlockOperands { x, dt, b, c } = operands;
    let [batch, heads, chunks, width, head_dim] = x.dims();
    let [_, groups, _, _, state_size] = b.dims();
    let per_group = heads / groups;
    let log_decay = dt.clone() * a;
    let cumulative = log_decay.clone().cumsum(3);
    let transitions = masks.transitions(log_decay);
    // C_t · B_s for every target/source pair, once per group: [b, g, c, l, l].
    let content = c.clone().matmul(b.clone().swap_dims(3, 4));
    let weighted = (transitions.reshape([batch, groups, per_group, chunks, width, width])
        * content.unsqueeze_dim::<6>(2))
    .reshape([batch, heads, chunks, width, width]);
    let input = x * dt.unsqueeze_dim::<5>(4);
    let diagonal = weighted.matmul(input.clone());

    // Chunk-local states from zero: [b, h, c, p, n].
    let totals = cumulative.clone().narrow(3, width - 1, 1);
    let to_end = (totals.clone() - cumulative.clone()).exp();
    let local = (input * to_end.unsqueeze_dim::<5>(4))
        .reshape([batch, groups, per_group, chunks, width, head_dim])
        .swap_dims(4, 5)
        .matmul(b.unsqueeze_dim::<6>(2))
        .reshape([batch, heads, chunks, head_dim * state_size]);

    // Chunk-entry states from the segmented chunk decays: [b, h, c + 1, p * n].
    let chunk_decay = Tensor::cat(
        vec![
            Tensor::zeros([batch, heads, 1], (&totals.device(), DType::F32)),
            totals.reshape([batch, heads, chunks]),
        ],
        2,
    )
    .reshape([batch, heads, 1, chunks + 1]);
    let carry = SegmentMasks::new(chunks + 1, &chunk_decay.device())
        .transitions(chunk_decay)
        .reshape([batch, heads, chunks + 1, chunks + 1]);
    let states = carry.matmul(Tensor::cat(
        vec![
            initial.reshape([batch, heads, 1, head_dim * state_size]),
            local,
        ],
        2,
    ));
    let terminal = states
        .clone()
        .narrow(2, chunks, 1)
        .reshape([batch, heads, head_dim, state_size]);
    let entry = states
        .narrow(2, 0, chunks)
        .reshape([batch, groups, per_group, chunks, head_dim, state_size]);

    // Outputs carried from each chunk's entry state: [b, h, c, l, p].
    let off_diagonal = entry
        .matmul(c.swap_dims(3, 4).unsqueeze_dim::<6>(2))
        .reshape([batch, heads, chunks, head_dim, width])
        .swap_dims(3, 4)
        * cumulative.exp().unsqueeze_dim::<5>(4);
    Ok((diagonal + off_diagonal, terminal))
}

/// Triangular masks for stable segmented sums over one chunk width, built once per scan.
pub(crate) struct SegmentMasks<B: Backend> {
    /// `[1, 1, 1, w, w]`, true where source `s >= u` term row: terms outside `s < u`.
    outside_segment: Tensor<B, 5, Bool>,
    /// `[1, 1, 1, w, w]`, true where source `s > t` target: future sources.
    future: Tensor<B, 5, Bool>,
    /// Width `w`.
    width: usize,
}

impl<B: Backend> SegmentMasks<B> {
    /// Builds both masks for width `width` on `device`.
    pub(crate) fn new(width: usize, device: &B::Device) -> Self {
        let rows =
            Tensor::<B, 1, Int>::arange(0..width as i64, device).reshape([1, 1, 1, width, 1]);
        let columns =
            Tensor::<B, 1, Int>::arange(0..width as i64, device).reshape([1, 1, 1, 1, width]);
        let shape = [1, 1, 1, width, width];
        Self {
            outside_segment: columns
                .clone()
                .expand(shape)
                .greater_equal(rows.clone().expand(shape)),
            future: columns.expand(shape).greater(rows.expand(shape)),
            width,
        }
    }

    /// Returns `[b, h, c, w, w]` transitions `exp(sum_{s<u<=t} ld[u])`, zero for `s > t`.
    ///
    /// A difference of two cumulative sums would lose every local decay after
    /// a very large earlier one; the masked cumulative sum of the per-step
    /// terms is the stable formulation of the reference SSD algorithm.
    pub(crate) fn transitions(&self, log_decay: Tensor<B, 4>) -> Tensor<B, 5> {
        let [batch, heads, chunks, width] = log_decay.dims();
        debug_assert_eq!(width, self.width, "segment masks built for another width");
        let shape = [batch, heads, chunks, width, width];
        log_decay
            .unsqueeze_dim::<5>(4)
            .expand(shape)
            .mask_fill(self.outside_segment.clone().expand(shape), 0.0)
            .cumsum(3)
            .mask_fill(self.future.clone().expand(shape), f32::NEG_INFINITY)
            .exp()
    }
}

#[cfg(test)]
#[path = "mamba2_tests.rs"]
mod tests;
