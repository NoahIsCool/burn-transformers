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
//! a masked semiseparable matrix product, and only the terminal state crosses
//! chunk boundaries. The quadratic temporaries are bounded by `chunk_size²`.
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
        let [batch, sequence, width] = hidden.dims();
        if batch == 0 || sequence == 0 || width != self.hidden_size {
            return Err(Error::shape(
                "mamba2.hidden",
                format!("[batch>0, sequence>0, {}]", self.hidden_size),
                format!("[{batch}, {sequence}, {width}]"),
            ));
        }
        if let Some(mask) = token_mask
            && mask.as_tensor().dims() != [batch, sequence]
        {
            return Err(Error::shape(
                "mamba2.token_mask",
                format!("[{batch}, {sequence}]"),
                format!("{:?}", mask.as_tensor().dims()),
            ));
        }
        let hidden_device = hidden.device();
        if token_mask.is_some_and(|mask| mask.as_tensor().device() != hidden_device) {
            return Err(Error::BackendCapability(
                "Mamba-2 input and token mask must share a device".into(),
            ));
        }
        let hidden_dtype = hidden.dtype();
        let recurrent_dtype = state.recurrent.dtype();
        self.validate_state(state, batch, hidden_dtype, &hidden_device)?;

        let valid = token_mask.map_or_else(
            || Tensor::<B, 2, Bool>::ones([batch, sequence], &hidden.device()),
            |mask| mask.as_tensor().clone(),
        );
        let masked = hidden
            * valid
                .clone()
                .float()
                .unsqueeze_dim::<3>(2)
                .cast(hidden_dtype);
        let projected = operations::linear(masked, &self.in_proj)?;
        let intermediate = self.num_heads * self.head_dim;
        let gate = projected.clone().narrow(2, 0, intermediate);
        let xbc = projected
            .clone()
            .narrow(2, intermediate, self.convolution_channels)
            .swap_dims(1, 2);
        let dt = projected.narrow(2, intermediate + self.convolution_channels, self.num_heads);

        let history = state.convolution.clone().cast(xbc.dtype());
        let joined = Tensor::cat(vec![history, xbc], 2);
        let convolved = conv1d(
            joined.clone(),
            self.conv1d.weight.val(),
            self.conv1d.bias.as_ref().map(|bias| bias.val()),
            ConvOptions::new([1], [0], [1], self.conv1d.groups),
        );
        let next_convolution = joined.narrow(2, sequence, self.convolution_kernel - 1);
        let convolved = activation::silu(convolved).swap_dims(1, 2)
            * valid
                .clone()
                .float()
                .unsqueeze_dim::<3>(2)
                .cast(hidden_dtype);
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
        let b = repeat_groups(b, self.num_heads);
        let c = repeat_groups(c, self.num_heads);

        let dt = discretization_steps(
            dt,
            self.dt_bias.val(),
            self.time_step_clamp.for_mode(mode),
            valid,
        );
        let a = -self.a_log.val().cast(DType::F32).exp();
        let (scan, recurrent) = match mode {
            Mamba2Mode::Recurrent => recurrent_scan(
                x.clone(),
                b,
                c,
                dt,
                a,
                self.d.val(),
                state.recurrent.clone(),
            ),
            Mamba2Mode::ChunkedPrefill => chunked_scan(
                x.clone(),
                b,
                c,
                dt,
                a,
                self.d.val().cast(DType::F32),
                state.recurrent.clone().cast(DType::F32),
                self.chunk_size,
            ),
        }?;
        let scan = scan.reshape([batch, sequence, intermediate]);
        let normalized = self.norm.forward(scan, gate)?;
        let output = operations::linear(normalized.cast(hidden_dtype), &self.out_proj)?;
        Ok((
            output,
            Mamba2State {
                recurrent: recurrent.cast(recurrent_dtype),
                convolution: next_convolution,
            },
        ))
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
/// result to F32. Multiplying by the validity mask in the same dtype gives
/// invalid semantic tokens `dt = 0`, an identity transition.
fn discretization_steps<B: Backend>(
    dt: Tensor<B, 3>,
    bias: Tensor<B, 1>,
    limits: Option<TimeStepLimits>,
    valid: Tensor<B, 2, Bool>,
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
    dt * valid.float().unsqueeze_dim::<3>(2).cast(dtype)
}

/// Repeats each `B/C` group across its contiguous state-space heads.
fn repeat_groups<B: Backend>(tensor: Tensor<B, 4>, heads: usize) -> Tensor<B, 4> {
    let [batch, sequence, groups, state] = tensor.dims();
    let repeats = heads / groups;
    if repeats == 1 {
        return tensor;
    }
    tensor
        .unsqueeze_dim::<5>(3)
        .expand([batch, sequence, groups, repeats, state])
        .reshape([batch, sequence, heads, state])
}

/// Executes the reference single-token selective-state update for every token.
///
/// Inputs are `x [b, t, h, p]`, `b/c [b, t, h, n]`, `dt [b, t, h]`, F32
/// `a [h]`, `d [h]`, and state `[b, h, p, n]`. The decay and the accumulated
/// state are F32. The input-dependent update is formed as `(dt * B) * x` in the
/// activation dtype, matching the reference's two rounding points, before
/// promotion into the state addition. For the output, the new F32 state is
/// rounded to `C`'s dtype before contraction and the `D * x` skip is also
/// evaluated there. When the state dtype is F16/BF16, every iteration rounds
/// through it, so a multi-token call equals repeated single-token calls.
pub(crate) fn recurrent_scan<B: Backend>(
    x: Tensor<B, 4>,
    b: Tensor<B, 4>,
    c: Tensor<B, 4>,
    dt: Tensor<B, 3>,
    a: Tensor<B, 1>,
    d: Tensor<B, 1>,
    state: Tensor<B, 4>,
) -> Result<(Tensor<B, 4>, Tensor<B, 4>)> {
    let [batch, sequence, heads, head_dim] = x.dims();
    let state_size = b.dims()[3];
    let activation_dtype = x.dtype();
    let state_dtype = state.dtype();
    let mut state = state.cast(DType::F32);
    let mut outputs = Vec::with_capacity(sequence);
    let a = a.reshape([1, heads]);
    let d = d.cast(activation_dtype).reshape([1, heads, 1]);
    for token in 0..sequence {
        let x_t = x.clone().narrow(1, token, 1).squeeze_dim::<3>(1);
        let b_t = b.clone().narrow(1, token, 1).squeeze_dim::<3>(1);
        let c_t = c.clone().narrow(1, token, 1).squeeze_dim::<3>(1);
        let dt_t = dt.clone().narrow(1, token, 1).squeeze_dim::<2>(1);
        let decay = (dt_t.clone().cast(DType::F32) * a.clone())
            .exp()
            .reshape([batch, heads, 1, 1]);
        let discretized_b = dt_t.reshape([batch, heads, 1, 1]) * b_t.clone().unsqueeze_dim::<4>(2);
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
        state = next_state.cast(state_dtype).cast(DType::F32);
    }
    if state.dims() != [batch, heads, head_dim, state_size] {
        return Err(Error::shape(
            "mamba2.scan_state",
            format!("[{batch}, {heads}, {head_dim}, {state_size}]"),
            format!("{:?}", state.dims()),
        ));
    }
    Ok((Tensor::cat(outputs, 1), state.cast(state_dtype)))
}

/// Executes bounded semiseparable chunks and carries only the terminal state.
#[allow(clippy::too_many_arguments)]
pub(crate) fn chunked_scan<B: Backend>(
    x: Tensor<B, 4>,
    b: Tensor<B, 4>,
    c: Tensor<B, 4>,
    dt: Tensor<B, 3>,
    a: Tensor<B, 1>,
    d: Tensor<B, 1>,
    mut state: Tensor<B, 4>,
    chunk_size: usize,
) -> Result<(Tensor<B, 4>, Tensor<B, 4>)> {
    let sequence = x.dims()[1];
    let mut chunks = Vec::with_capacity(sequence.div_ceil(chunk_size));
    for start in (0..sequence).step_by(chunk_size) {
        let width = (sequence - start).min(chunk_size);
        let (output, next) = scan_chunk(
            x.clone().narrow(1, start, width),
            b.clone().narrow(1, start, width),
            c.clone().narrow(1, start, width),
            dt.clone().narrow(1, start, width),
            a.clone(),
            d.clone(),
            state,
        )?;
        chunks.push(output);
        state = next;
    }
    Ok((Tensor::cat(chunks, 1), state))
}

/// Evaluates one dense lower-triangular semiseparable scan block.
#[allow(clippy::too_many_arguments)]
fn scan_chunk<B: Backend>(
    x: Tensor<B, 4>,
    b: Tensor<B, 4>,
    c: Tensor<B, 4>,
    dt: Tensor<B, 3>,
    a: Tensor<B, 1>,
    d: Tensor<B, 1>,
    initial: Tensor<B, 4>,
) -> Result<(Tensor<B, 4>, Tensor<B, 4>)> {
    let [batch, width, heads, head_dim] = x.dims();
    let state_size = b.dims()[3];
    if width == 0 {
        return Err(Error::configuration(
            "mamba2.chunk",
            "chunk width must be non-zero",
        ));
    }
    let x = x.cast(DType::F32);
    let b = b.cast(DType::F32);
    let c = c.cast(DType::F32);
    let dt = dt.cast(DType::F32);
    let log_decay = dt.clone() * a.reshape([1, 1, heads]);
    let cumulative = log_decay.clone().cumsum(1);

    // L[t,s] = exp(sum_{u=s+1..t} log_decay[u]) for s <= t.
    let transitions = stable_decay_transitions(log_decay.clone());
    // Batched C @ B^T forms the semiseparable content kernel without
    // materializing `[batch, target, source, heads, state_size]`.
    let content_kernel = c
        .clone()
        .permute([0, 2, 1, 3])
        .matmul(b.clone().permute([0, 2, 3, 1]))
        .permute([0, 2, 3, 1]);
    let input = x.clone() * dt.clone().unsqueeze_dim::<4>(3);
    // Kernel @ input likewise avoids a `[target, source, head_dim]` product.
    let within = (content_kernel * transitions)
        .permute([0, 3, 1, 2])
        .matmul(input.clone().permute([0, 2, 1, 3]))
        .permute([0, 2, 1, 3]);
    // Initial-state contribution is one state-matrix/C batched contraction.
    let from_initial = initial
        .clone()
        .matmul(c.clone().permute([0, 2, 3, 1]))
        .permute([0, 3, 1, 2])
        * cumulative.clone().exp().unsqueeze_dim::<4>(3);
    let output = within + from_initial + x * d.reshape([1, 1, heads, 1]);

    let terminal = cumulative
        .clone()
        .narrow(1, width - 1, 1)
        .reshape([batch, heads]);
    let terminal_initial = initial * terminal.clone().exp().reshape([batch, heads, 1, 1]);
    let contribution_decay = (terminal.reshape([batch, 1, heads]) - cumulative).exp();
    // Sum weighted x_t outer B_t as `[D,T] @ [T,N]`, avoiding `[T,D,N]`.
    let terminal_updates = (input * contribution_decay.unsqueeze_dim::<4>(3))
        .permute([0, 2, 3, 1])
        .matmul(b.permute([0, 2, 1, 3]));
    if terminal_updates.dims() != [batch, heads, head_dim, state_size] {
        return Err(Error::shape(
            "mamba2.chunk_terminal",
            format!("[{batch}, {heads}, {head_dim}, {state_size}]"),
            format!("{:?}", terminal_updates.dims()),
        ));
    }
    Ok((output, terminal_initial + terminal_updates))
}

/// Builds lower-triangular chunk transitions without subtracting large prefixes.
///
/// For target `t` and source `s`, the value is `exp(sum(log_decay[s+1..=t]))`
/// when `s <= t` and zero otherwise. A difference of two cumulative sums loses
/// every local decay after a very large earlier one; the segmented sum used
/// here is the stable formulation of the reference SSD algorithm.
pub(crate) fn stable_decay_transitions<B: Backend>(log_decay: Tensor<B, 3>) -> Tensor<B, 4> {
    let [batch, width, heads] = log_decay.dims();
    let device = log_decay.device();
    let target = Tensor::<B, 1, Int>::arange(0..width as i64, &device)
        .reshape([1, width, 1, 1])
        .expand([batch, width, width, heads]);
    let source = Tensor::<B, 1, Int>::arange(0..width as i64, &device)
        .reshape([1, 1, width, 1])
        .expand([batch, width, width, heads]);
    let strict_lower = source.clone().lower(target.clone());
    let segment_terms = log_decay
        .unsqueeze_dim::<4>(2)
        .expand([batch, width, width, heads])
        .mask_fill(strict_lower.bool_not(), 0.0);
    segment_terms
        .cumsum(1)
        .mask_fill(source.greater(target), f32::NEG_INFINITY)
        .exp()
}

#[cfg(test)]
#[path = "mamba2_tests.rs"]
mod tests;
