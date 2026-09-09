//! Nemotron-H's grouped Mamba2 selective state-space mixer.
//!
//! For head `h`, channel `p`, and state coordinate `n`, one recurrent step is
//!
//! `S_t[h,p,n] = exp(dt_t[h] A[h]) S_{t-1}[h,p,n]
//!                  + dt_t[h] x_t[h,p] B_t[h,n]`,
//!
//! followed by `y_t[h,p] = sum_n S_t[h,p,n] C_t[h,n] + D[h] x_t[h,p]`.
//! `A = -exp(A_log)` is stable and input independent, whereas `dt`, `B`, and
//! `C` are selected by each token. Several heads share one projected B/C group.
//! The chunked scan promotes `x`, `B`, `C`, `dt`, and state arithmetic to FP32.
//! The recurrent decode path deliberately follows the pinned Transformers
//! fallback more closely: decay and state accumulation are FP32, but `dt * B *
//! x`, the state-to-`C` contraction, and the `D` skip retain the activation
//! dtype. An external cache may store terminal state as F32, F16, or BF16; a
//! reduced-precision state is rounded after every recurrent token, just as an
//! in-place reference cache would be.
//!
//! [`NemotronMamba2Mode::Recurrent`] is the transparent token loop used as a
//! golden reference and for single-token decode. The chunked path evaluates all
//! within-chunk state transitions as a masked semiseparable matrix, limiting the
//! quadratic temporary to `chunk_size^2` while carrying only the terminal state
//! between chunks. Both paths return new state instead of mutating their input,
//! which lets an architecture publish every cache layer atomically.
//!
//! A [`TokenMask`] is semantic in this crate: `false` means that no recurrent
//! time passes, so the corresponding `dt` is zero and the matrix state is an
//! exact identity transition. This intentionally differs from Transformers
//! revision `cfa309cfa34b5497a32fd50af49aa97aac94c054`, whose padding helper
//! zeroes `x/B/C` but leaves `dt` active. The architecture-level stateful API
//! compacts valid tokens before calling this mixer, making both recurrent and
//! convolution caches true semantic-token caches. A direct masked call keeps
//! the reference convolution's physical-column history; callers that persist
//! that history must therefore compact interior holes first.

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

use crate::{Error, Result, architecture::TokenMask, operations};

/// Numerical and tensor geometry for one Nemotron-H Mamba2 block.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct NemotronMamba2Config {
    /// Residual-stream width.
    pub hidden_size: usize,
    /// Number of independent state-space heads.
    pub num_heads: usize,
    /// Per-head value channels.
    pub head_dim: usize,
    /// State coordinates retained for every head channel.
    pub state_size: usize,
    /// Number of groups sharing token-selected B/C vectors.
    pub groups: usize,
    /// Width of the depthwise causal convolution.
    pub convolution_kernel: usize,
    /// Maximum vectorized scan chunk width.
    pub chunk_size: usize,
    /// Lower clamp for positive discretization steps.
    pub time_step_min: f64,
    /// Optional upper clamp for discretization steps.
    pub time_step_max: Option<f64>,
    /// Epsilon for the gated grouped RMS normalization.
    pub normalization_epsilon: f64,
    /// Whether the fused input and output projections contain bias.
    pub projection_bias: bool,
    /// Whether the depthwise convolution contains bias.
    pub convolution_bias: bool,
}

impl NemotronMamba2Config {
    /// Returns the expanded head width `num_heads * head_dim`.
    pub fn intermediate_size(&self) -> Result<usize> {
        self.num_heads.checked_mul(self.head_dim).ok_or_else(|| {
            Error::configuration("nemotron_h.mamba2", "intermediate width overflows usize")
        })
    }

    /// Returns convolution channels `intermediate + 2 * groups * state_size`.
    pub fn convolution_channels(&self) -> Result<usize> {
        let state_channels = self
            .groups
            .checked_mul(self.state_size)
            .and_then(|value| value.checked_mul(2))
            .ok_or_else(|| {
                Error::configuration("nemotron_h.mamba2", "B/C channel width overflows usize")
            })?;
        self.intermediate_size()?
            .checked_add(state_channels)
            .ok_or_else(|| {
                Error::configuration("nemotron_h.mamba2", "convolution width overflows usize")
            })
    }

    /// Returns fused projection width `[gate, x/B/C, dt]`.
    pub fn projection_size(&self) -> Result<usize> {
        self.intermediate_size()?
            .checked_add(self.convolution_channels()?)
            .and_then(|value| value.checked_add(self.num_heads))
            .ok_or_else(|| {
                Error::configuration("nemotron_h.mamba2", "projection width overflows usize")
            })
    }

    /// Validates all divisibility, extent, and numerical invariants.
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
                "nemotron_h.mamba2",
                "all widths must be non-zero and convolution_kernel must be at least two",
            ));
        }
        if !self.num_heads.is_multiple_of(self.groups) {
            return Err(Error::configuration(
                "nemotron_h.mamba2.groups",
                "must divide num_heads exactly",
            ));
        }
        if !self.time_step_min.is_finite() || self.time_step_min < 0.0 {
            return Err(Error::configuration(
                "nemotron_h.mamba2.time_step_min",
                "must be finite and non-negative",
            ));
        }
        if self
            .time_step_max
            .is_some_and(|upper| !upper.is_finite() || upper < self.time_step_min)
        {
            return Err(Error::configuration(
                "nemotron_h.mamba2.time_step_max",
                "must be finite and no smaller than time_step_min",
            ));
        }
        if !self.normalization_epsilon.is_finite() || self.normalization_epsilon <= 0.0 {
            return Err(Error::configuration(
                "nemotron_h.mamba2.normalization_epsilon",
                "must be finite and greater than zero",
            ));
        }
        let _ = self.projection_size()?;
        Ok(())
    }

    /// Initializes one checkpoint-compatible Mamba2 module.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Result<NemotronMamba2<B>> {
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
        Ok(NemotronMamba2 {
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
            norm: NemotronGroupedGatedRmsNorm {
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
            time_step_min: self.time_step_min,
            time_step_max: self.time_step_max,
            convolution_channels,
        })
    }
}

/// Selects the transparent or blockwise selective scan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NemotronMamba2Mode {
    /// One explicit state update per token.
    Recurrent,
    /// Vectorized semiseparable scan inside bounded chunks.
    ChunkedPrefill,
}

/// External terminal state for one Mamba2 layer.
#[derive(Clone, Debug)]
pub struct NemotronMamba2State<B: Backend> {
    /// Selective state `[batch, heads, head_dim, state_size]` in F32/F16/BF16.
    pub recurrent: Tensor<B, 4>,
    /// Recent projected convolution channels `[batch, channels, kernel - 1]`.
    pub convolution: Tensor<B, 3>,
}

impl<B: Backend> NemotronMamba2State<B> {
    /// Allocates FP32 zero recurrent state with an explicit convolution dtype.
    pub fn zeros(
        config: &NemotronMamba2Config,
        batch_size: usize,
        convolution_dtype: DType,
        device: &B::Device,
    ) -> Result<Self> {
        Self::zeros_with_dtypes(config, batch_size, DType::F32, convolution_dtype, device)
    }

    /// Allocates zero terminal state using explicit matrix and convolution dtypes.
    pub fn zeros_with_dtypes(
        config: &NemotronMamba2Config,
        batch_size: usize,
        recurrent_dtype: DType,
        convolution_dtype: DType,
        device: &B::Device,
    ) -> Result<Self> {
        config.validate()?;
        if batch_size == 0 {
            return Err(Error::configuration(
                "nemotron_h.mamba2.batch_size",
                "must be non-zero",
            ));
        }
        if !matches!(recurrent_dtype, DType::F32 | DType::F16 | DType::BF16)
            || !convolution_dtype.is_float()
        {
            return Err(Error::configuration(
                "nemotron_h.mamba2.state_dtype",
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

/// Groupwise RMS normalization applied after the SiLU output gate.
#[derive(Module, Debug)]
pub struct NemotronGroupedGatedRmsNorm<B: Backend> {
    /// Per-channel learned scale.
    pub weight: Param<Tensor<B, 1>>,
    /// Number of independent normalization groups.
    pub groups: usize,
    /// Positive denominator epsilon.
    pub epsilon: f64,
}

impl<B: Backend> NemotronGroupedGatedRmsNorm<B> {
    /// Computes `RMSNorm_group(y * SiLU(gate))` using the Zamba2 operation graph.
    ///
    /// Both the gate and group variance are evaluated in FP32. The normalized
    /// activation is then rounded back to the input dtype *before* multiplying
    /// by the learned scale. Multiplication by `reciprocal(sqrt(variance + eps))`
    /// is kept explicit because replacing it with division is algebraically
    /// equivalent but can differ by one ULP in F16.
    pub fn forward(&self, values: Tensor<B, 3>, gate: Tensor<B, 3>) -> Result<Tensor<B, 3>> {
        let [batch, sequence, width] = values.dims();
        if !self.epsilon.is_finite() || self.epsilon <= 0.0 {
            return Err(Error::configuration(
                "nemotron_h.mamba2.gated_norm.epsilon",
                "must be finite and greater than zero",
            ));
        }
        if gate.dims() != [batch, sequence, width]
            || self.groups == 0
            || !width.is_multiple_of(self.groups)
            || self.weight.shape().dims() != [width]
        {
            return Err(Error::shape(
                "nemotron_h.mamba2.gated_norm",
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

/// Checkpoint-compatible Nemotron-H Mamba2 mixer.
#[derive(Module, Debug)]
pub struct NemotronMamba2<B: Backend> {
    /// Fused `[gate, x, B, C, dt]` projection.
    pub in_proj: Linear<B>,
    /// Depthwise causal convolution over x/B/C channels.
    pub conv1d: Conv1d<B>,
    /// Learned inverse-softplus discretization offset.
    pub dt_bias: Param<Tensor<B, 1>>,
    /// Logarithm of positive continuous-time decay magnitudes.
    pub a_log: Param<Tensor<B, 1>>,
    /// Learned per-head direct skip coefficient.
    pub d: Param<Tensor<B, 1>>,
    /// Gated, grouped RMS normalization before the output projection.
    pub norm: NemotronGroupedGatedRmsNorm<B>,
    /// Projection from expanded head channels to the residual stream.
    pub out_proj: Linear<B>,
    /// Residual-stream width.
    pub hidden_size: usize,
    /// State-space head count.
    pub num_heads: usize,
    /// Channels in each head.
    pub head_dim: usize,
    /// State coordinates per channel.
    pub state_size: usize,
    /// B/C sharing groups.
    pub groups: usize,
    /// Causal convolution kernel width.
    pub convolution_kernel: usize,
    /// Vectorized scan chunk width.
    pub chunk_size: usize,
    /// Lower discretization clamp.
    pub time_step_min: f64,
    /// Optional upper discretization clamp.
    pub time_step_max: Option<f64>,
    /// x/B/C convolution channel count.
    pub convolution_channels: usize,
}

impl<B: Backend> NemotronMamba2<B> {
    /// Reconstructs the validated allocation configuration.
    pub fn config(&self) -> NemotronMamba2Config {
        NemotronMamba2Config {
            hidden_size: self.hidden_size,
            num_heads: self.num_heads,
            head_dim: self.head_dim,
            state_size: self.state_size,
            groups: self.groups,
            convolution_kernel: self.convolution_kernel,
            chunk_size: self.chunk_size,
            time_step_min: self.time_step_min,
            time_step_max: self.time_step_max,
            normalization_epsilon: self.norm.epsilon,
            projection_bias: self.in_proj.bias.is_some(),
            convolution_bias: self.conv1d.bias.is_some(),
        }
    }

    /// Runs a stateless differentiable prompt using zero initial state.
    pub fn forward(
        &self,
        hidden: Tensor<B, 3>,
        token_mask: Option<&TokenMask<B>>,
        mode: NemotronMamba2Mode,
    ) -> Result<Tensor<B, 3>> {
        let state = NemotronMamba2State::zeros(
            &self.config(),
            hidden.dims()[0],
            hidden.dtype(),
            &hidden.device(),
        )?;
        self.forward_with_state(hidden, token_mask, &state, mode)
            .map(|(output, _)| output)
    }

    /// Runs a prompt/decode segment and returns an unpublished terminal state.
    pub fn forward_with_state(
        &self,
        hidden: Tensor<B, 3>,
        token_mask: Option<&TokenMask<B>>,
        state: &NemotronMamba2State<B>,
        mode: NemotronMamba2Mode,
    ) -> Result<(Tensor<B, 3>, NemotronMamba2State<B>)> {
        let [batch, sequence, width] = hidden.dims();
        if batch == 0 || sequence == 0 || width != self.hidden_size {
            return Err(Error::shape(
                "nemotron_h.mamba2.hidden",
                format!("[batch>0, sequence>0, {}]", self.hidden_size),
                format!("[{batch}, {sequence}, {width}]"),
            ));
        }
        if let Some(mask) = token_mask
            && mask.as_tensor().dims() != [batch, sequence]
        {
            return Err(Error::shape(
                "nemotron_h.mamba2.token_mask",
                format!("[{batch}, {sequence}]"),
                format!("{:?}", mask.as_tensor().dims()),
            ));
        }
        let hidden_device = hidden.device();
        if token_mask.is_some_and(|mask| mask.as_tensor().device() != hidden_device) {
            return Err(Error::BackendCapability(
                "Nemotron-H Mamba input and token mask must share a device".into(),
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
            self.time_step_min,
            self.time_step_max,
            mode,
            valid,
        );
        let a = -self.a_log.val().cast(DType::F32).exp();
        let (scan, recurrent) = match mode {
            NemotronMamba2Mode::Recurrent => recurrent_scan(
                x.clone(),
                b,
                c,
                dt,
                a,
                self.d.val(),
                state.recurrent.clone(),
            ),
            NemotronMamba2Mode::ChunkedPrefill => chunked_scan(
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
            NemotronMamba2State {
                recurrent: recurrent.cast(recurrent_dtype),
                convolution: next_convolution,
            },
        ))
    }

    /// Checks external recurrent and convolution geometry before tensor math.
    fn validate_state(
        &self,
        state: &NemotronMamba2State<B>,
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
                "nemotron_h.mamba2.state.recurrent",
                format!("{recurrent:?} F32/F16/BF16"),
                format!("{:?} {:?}", state.recurrent.dims(), state.recurrent.dtype()),
            ));
        }
        if state.convolution.dims() != convolution || state.convolution.dtype() != convolution_dtype
        {
            return Err(Error::shape(
                "nemotron_h.mamba2.state.convolution",
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
                "Nemotron-H Mamba input and recurrent state must share a device".into(),
            ));
        }
        Ok(())
    }
}

/// Applies the learned Mamba discretization bias and positive-step transform.
///
/// The pinned Nemotron-H fallback adds `dt_bias`, evaluates softplus, and clamps
/// while `dt` still has the fused projection's dtype. Burn's generic softplus is
/// a literal `log(1 + exp(x))` graph and can round small F16 results to zero, so
/// this implementation evaluates the stable identity `max(x, 0) +
/// log1p(exp(-abs(x)))` in FP32 after the bias addition has already rounded. It
/// immediately restores the projection dtype, reproducing PyTorch's
/// reduced-precision softplus kernel without moving the bias addition across the
/// dtype boundary. Only the scan promotes the stored result to FP32. This crate
/// then multiplies by the true-valid mask in the same dtype so an invalid semantic
/// token has `dt = 0` and cannot decay the recurrent state.
///
/// Only chunked prefill applies the configured bounds. The pinned recurrent
/// `mamba2_selective_state_update` fallback accepts `dt_limit` through keyword
/// arguments but does not use it; matching that asymmetry is necessary when a
/// learned decode step falls below the chunk path's minimum.
fn discretization_steps<B: Backend>(
    dt: Tensor<B, 3>,
    bias: Tensor<B, 1>,
    minimum: f64,
    maximum: Option<f64>,
    mode: NemotronMamba2Mode,
    valid: Tensor<B, 2, Bool>,
) -> Tensor<B, 3> {
    let dtype = dt.dtype();
    let heads = dt.dims()[2];
    let biased = dt + bias.cast(dtype).reshape([1, 1, heads]);
    let biased = biased.cast(DType::F32);
    let mut dt = (biased.clone().clamp_min(0.0) + (-biased.abs()).exp().log1p()).cast(dtype);
    if mode == NemotronMamba2Mode::ChunkedPrefill {
        dt = dt.clamp_min(minimum);
        if let Some(upper) = maximum {
            dt = dt.clamp_max(upper);
        }
    }
    dt * valid.float().unsqueeze_dim::<3>(2).cast(dtype)
}

/// Repeats each B/C group across its contiguous state-space heads.
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

/// Executes the reference single-token selective-state update operation graph.
///
/// The stable decay and accumulated matrix state are FP32. In contrast, the
/// input-dependent update is formed as `(dt * B) * x` in the activation dtype,
/// matching the fallback's two rounding points, before promotion into the state
/// addition. For the output, the new FP32 state is rounded to `C`'s dtype before
/// contraction and the direct `D * x` skip is also evaluated there. When state
/// storage is F16/BF16, every loop iteration rounds through that storage dtype;
/// therefore a multi-token call is equivalent to repeated cached decode calls.
fn recurrent_scan<B: Backend>(
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
            "nemotron_h.mamba2.scan_state",
            format!("[{batch}, {heads}, {head_dim}, {state_size}]"),
            format!("{:?}", state.dims()),
        ));
    }
    Ok((Tensor::cat(outputs, 1), state.cast(state_dtype)))
}

/// Executes bounded semiseparable chunks and carries only terminal state.
#[allow(clippy::too_many_arguments)]
fn chunked_scan<B: Backend>(
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
            "nemotron_h.mamba2.chunk",
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
            "nemotron_h.mamba2.chunk_terminal",
            format!("[{batch}, {heads}, {head_dim}, {state_size}]"),
            format!("{:?}", terminal_updates.dims()),
        ));
    }
    Ok((output, terminal_initial + terminal_updates))
}

/// Builds lower-triangular chunk transitions without subtracting large prefixes.
///
/// For target `t` and source `s`, the returned value is
/// `exp(sum(log_decay[s + 1..=t]))` when `s <= t`, and zero otherwise. A direct
/// difference of two cumulative sums loses all local decay information after a
/// very large earlier decay. The lower-triangular segmented sum used here is the
/// stable formulation from the reference Mamba2 SSD algorithm.
fn stable_decay_transitions<B: Backend>(log_decay: Tensor<B, 3>) -> Tensor<B, 4> {
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
mod tests {
    use burn::{
        backend::Flex,
        module::Param,
        tensor::{Tolerance, bf16, f16},
    };

    use super::*;

    type TestBackend = Flex<f32, i32>;

    /// Small grouped topology that exercises two chunks and head repetition.
    fn config() -> NemotronMamba2Config {
        NemotronMamba2Config {
            hidden_size: 8,
            num_heads: 4,
            head_dim: 2,
            state_size: 3,
            groups: 2,
            convolution_kernel: 3,
            chunk_size: 2,
            time_step_min: 0.001,
            time_step_max: None,
            normalization_epsilon: 1e-5,
            projection_bias: false,
            convolution_bias: true,
        }
    }

    /// Chunk algebra and the defining token recurrence agree from non-zero state.
    #[test]
    fn chunked_and_recurrent_paths_match() {
        let device = Default::default();
        let mixer = config().init::<TestBackend>(&device).unwrap();
        let input = Tensor::from_floats(
            [[
                [0.1, 0.2, -0.3, 0.4, 0.5, -0.6, 0.7, 0.8],
                [0.3, -0.1, 0.2, 0.6, -0.4, 0.5, 0.9, -0.2],
                [-0.5, 0.7, 0.1, -0.8, 0.2, 0.4, -0.6, 0.3],
                [0.8, 0.1, -0.2, 0.5, 0.6, -0.7, 0.4, -0.3],
                [0.2, -0.4, 0.6, 0.1, -0.5, 0.3, 0.7, 0.9],
            ]],
            &device,
        );
        let mut state = NemotronMamba2State::zeros(&config(), 1, DType::F32, &device).unwrap();
        state.recurrent = Tensor::ones([1, 4, 2, 3], &device) * 0.07;
        state.convolution = Tensor::ones([1, 20, 2], &device) * -0.03;
        let (reference, reference_state) = mixer
            .forward_with_state(input.clone(), None, &state, NemotronMamba2Mode::Recurrent)
            .unwrap();
        let (chunked, chunked_state) = mixer
            .forward_with_state(input, None, &state, NemotronMamba2Mode::ChunkedPrefill)
            .unwrap();
        chunked
            .into_data()
            .assert_approx_eq::<f32>(&reference.into_data(), Tolerance::absolute(2e-5));
        chunked_state.recurrent.into_data().assert_approx_eq::<f32>(
            &reference_state.recurrent.into_data(),
            Tolerance::absolute(2e-5),
        );
        chunked_state
            .convolution
            .into_data()
            .assert_approx_eq::<f32>(
                &reference_state.convolution.into_data(),
                Tolerance::absolute(1e-6),
            );
    }

    /// Segmenting prompt prefill preserves convolution and recurrent cache state.
    ///
    /// The split is intentionally not aligned to the configured scan chunk width,
    /// so this covers both the `kernel - 1` convolution history convention and
    /// the algebra that carries an FP32 terminal matrix between arbitrary calls.
    #[test]
    fn segmented_prefill_matches_one_shot_cache_and_outputs() {
        let device = Default::default();
        let mixer = config().init::<TestBackend>(&device).unwrap();
        let input = Tensor::from_floats(
            [[
                [0.1, 0.2, -0.3, 0.4, 0.5, -0.6, 0.7, 0.8],
                [0.3, -0.1, 0.2, 0.6, -0.4, 0.5, 0.9, -0.2],
                [-0.5, 0.7, 0.1, -0.8, 0.2, 0.4, -0.6, 0.3],
                [0.8, 0.1, -0.2, 0.5, 0.6, -0.7, 0.4, -0.3],
                [0.2, -0.4, 0.6, 0.1, -0.5, 0.3, 0.7, 0.9],
            ]],
            &device,
        );
        let mut initial = NemotronMamba2State::zeros(&config(), 1, DType::F32, &device).unwrap();
        initial.recurrent = Tensor::ones([1, 4, 2, 3], &device) * 0.037;
        initial.convolution = Tensor::ones([1, 20, 2], &device) * -0.019;

        let (one_shot, one_shot_state) = mixer
            .forward_with_state(
                input.clone(),
                None,
                &initial,
                NemotronMamba2Mode::ChunkedPrefill,
            )
            .unwrap();
        let (prefix, prefix_state) = mixer
            .forward_with_state(
                input.clone().narrow(1, 0, 3),
                None,
                &initial,
                NemotronMamba2Mode::ChunkedPrefill,
            )
            .unwrap();
        let (suffix, segmented_state) = mixer
            .forward_with_state(
                input.narrow(1, 3, 2),
                None,
                &prefix_state,
                NemotronMamba2Mode::ChunkedPrefill,
            )
            .unwrap();
        let segmented = Tensor::cat(vec![prefix, suffix], 1);

        segmented
            .into_data()
            .assert_approx_eq::<f32>(&one_shot.into_data(), Tolerance::absolute(3e-5));
        segmented_state
            .recurrent
            .into_data()
            .assert_approx_eq::<f32>(
                &one_shot_state.recurrent.into_data(),
                Tolerance::absolute(3e-5),
            );
        assert_eq!(
            segmented_state.convolution.to_data(),
            one_shot_state.convolution.to_data()
        );
    }

    /// A one-head scalar recurrence matches a hand-computed selective-SSM oracle.
    #[test]
    fn recurrent_scan_matches_scalar_golden_values() {
        let device = Default::default();
        let x = Tensor::<TestBackend, 4>::from_floats([[[[3.0]], [[-2.0]]]], &device);
        let b = Tensor::<TestBackend, 4>::from_floats([[[[4.0]], [[0.5]]]], &device);
        let c = Tensor::<TestBackend, 4>::from_floats([[[[5.0]], [[3.0]]]], &device);
        let dt = Tensor::<TestBackend, 3>::from_floats([[[0.1], [0.2]]], &device);
        let a = Tensor::<TestBackend, 1>::from_floats([-1.0], &device);
        let d = Tensor::<TestBackend, 1>::from_floats([0.5], &device);
        let initial = Tensor::<TestBackend, 4>::from_floats([[[[2.0]]]], &device);
        let (output, terminal) = recurrent_scan(x, b, c, dt, a, d, initial).unwrap();
        output.into_data().assert_approx_eq::<f32>(
            &TensorData::from([[[[16.548_374]], [[5.792_34]]]]),
            Tolerance::absolute(1e-5),
        );
        terminal.into_data().assert_approx_eq::<f32>(
            &TensorData::from([[[[2.264_113_4]]]]),
            Tolerance::absolute(1e-6),
        );
    }

    /// Segmented decay retains local increments after an enormous earlier prefix.
    #[test]
    fn chunk_transition_sum_avoids_prefix_cancellation() {
        let device = Default::default();
        let transitions = stable_decay_transitions(Tensor::<TestBackend, 3>::from_floats(
            [[[-100_000_000.0], [-1.0], [-2.0]]],
            &device,
        ));
        transitions.into_data().assert_approx_eq::<f32>(
            &TensorData::from([[
                [[1.0], [0.0], [0.0]],
                [[(-1.0_f32).exp()], [1.0], [0.0]],
                [[(-3.0_f32).exp()], [(-2.0_f32).exp()], [1.0]],
            ]]),
            Tolerance::absolute(1e-6),
        );
    }

    /// Grouped gated normalization follows gate-then-normalize ordering.
    #[test]
    fn grouped_gate_is_applied_before_rms_normalization() {
        let device = Default::default();
        let norm = NemotronGroupedGatedRmsNorm::<TestBackend> {
            weight: Param::from_tensor(Tensor::ones([4], &device)),
            groups: 2,
            epsilon: 1e-6,
        };
        let values = Tensor::from_floats([[[1.0, 2.0, 3.0, 4.0]]], &device);
        let gate = Tensor::ones([1, 1, 4], &device);
        let output = norm.forward(values, gate).unwrap();
        let data = output.into_data();
        let values = data.as_slice::<f32>().unwrap();
        assert!(values.iter().all(|value| value.is_finite()));
        assert!((values[0] / values[1] - 0.5).abs() < 1e-5);
        assert!((values[2] / values[3] - 0.75).abs() < 1e-5);
    }

    /// Grouped normalization retains Zamba2's multiply-by-inverse-root graph.
    ///
    /// These literals were generated with `Zamba2RMSNormGated.forward` from
    /// Transformers revision `cfa309cfa34b5497a32fd50af49aa97aac94c054`.
    /// A saturated gate isolates RMS application from backend-specific fused
    /// SiLU approximations. The first component is a one-ULP discriminator:
    /// replacing multiplication by reciprocal-sqrt with division produces
    /// `0xbfaa`, not `0xbfa9`.
    #[test]
    fn grouped_norm_matches_pinned_reduced_precision_oracles() {
        let device = Default::default();
        let norm = NemotronGroupedGatedRmsNorm::<TestBackend> {
            weight: Param::from_tensor(
                Tensor::from_floats(
                    [-1.934_570_3, 0.954_101_56, -0.644_531_25, -2.945_312_5],
                    &device,
                )
                .cast(DType::F16),
            ),
            groups: 2,
            epsilon: 1e-6,
        };
        let values = Tensor::from_floats(
            [[[6.406_25, -6.535_156_2, 7.539_062_5, -5.781_25]]],
            &device,
        )
        .cast(DType::F16);
        let gate = Tensor::full([1, 1, 4], 80.0, (&device, DType::F16));

        let actual = norm.forward(values.clone(), gate.clone()).unwrap();
        let actual_bits = actual
            .into_data()
            .to_vec::<f16>()
            .unwrap()
            .into_iter()
            .map(f16::to_bits)
            .collect::<Vec<_>>();
        assert_eq!(actual_bits, [0xbfa9, 0xbbb5, 0xb9c9, 0x4111]);

        let gated = values.cast(DType::F32) * activation::silu(gate.cast(DType::F32));
        let grouped = gated.reshape([1, 1, 2, 2]);
        let divided = (grouped.clone() / (grouped.square().mean_dim(3) + norm.epsilon).sqrt())
            .reshape([1, 1, 4])
            .cast(DType::F16)
            * norm.weight.val().reshape([1, 1, 4]);
        let divided_bits = divided
            .into_data()
            .to_vec::<f16>()
            .unwrap()
            .into_iter()
            .map(f16::to_bits)
            .collect::<Vec<_>>();
        assert_eq!(divided_bits[0], 0xbfaa);

        let bf16_norm = NemotronGroupedGatedRmsNorm::<TestBackend> {
            weight: Param::from_tensor(
                Tensor::from_floats(
                    [-1.934_570_3, 0.954_101_56, -0.644_531_25, -2.945_312_5],
                    &device,
                )
                .cast(DType::BF16),
            ),
            groups: 2,
            epsilon: 1e-6,
        };
        let bf16_output = bf16_norm
            .forward(
                Tensor::from_floats(
                    [[[6.406_25, -6.535_156_2, 7.539_062_5, -5.781_25]]],
                    &device,
                )
                .cast(DType::BF16),
                Tensor::full([1, 1, 4], 80.0, (&device, DType::BF16)),
            )
            .unwrap();
        let bf16_bits = bf16_output
            .into_data()
            .to_vec::<bf16>()
            .unwrap()
            .into_iter()
            .map(bf16::to_bits)
            .collect::<Vec<_>>();
        assert_eq!(bf16_bits, [0xbff6, 0xbf76, 0xbf3a, 0x4022]);
    }

    /// Softplus and bias addition occur in projection dtype before scan promotion.
    ///
    /// The expected F16 value is from the pinned fallback. Promoting before
    /// softplus yields adjacent value `0x0675`, so the literal guards the exact
    /// mixed-precision boundary as well as the result dtype.
    #[test]
    fn discretization_rounds_before_fp32_scan_math() {
        let device = Default::default();
        let raw =
            Tensor::<TestBackend, 3>::from_floats([[[-9.726_562_5]]], &device).cast(DType::F16);
        let bias = Tensor::<TestBackend, 1>::from_floats([0.501_464_84], &device).cast(DType::F16);
        let valid = Tensor::<TestBackend, 2, Bool>::from_bool([[true]].into(), &device);
        let actual = discretization_steps(
            raw.clone(),
            bias.clone(),
            0.0,
            None,
            NemotronMamba2Mode::Recurrent,
            valid,
        );
        assert_eq!(actual.dtype(), DType::F16);
        let actual_bits = actual
            .into_data()
            .to_vec::<f16>()
            .unwrap()
            .into_iter()
            .map(f16::to_bits)
            .collect::<Vec<_>>();
        assert_eq!(actual_bits, [0x0673]);

        let promoted_biased =
            raw.clone().cast(DType::F32) + bias.clone().cast(DType::F32).reshape([1, 1, 1]);
        let promoted = (promoted_biased.clone().clamp_min(0.0)
            + (-promoted_biased.abs()).exp().log1p())
        .cast(DType::F16);
        let promoted_bits = promoted
            .into_data()
            .to_vec::<f16>()
            .unwrap()
            .into_iter()
            .map(f16::to_bits)
            .collect::<Vec<_>>();
        assert_eq!(promoted_bits, [0x0675]);

        let all_valid = || Tensor::<TestBackend, 2, Bool>::from_bool([[true]].into(), &device);
        let recurrent = discretization_steps(
            raw.clone(),
            bias.clone(),
            0.001,
            Some(0.01),
            NemotronMamba2Mode::Recurrent,
            all_valid(),
        );
        let chunked = discretization_steps(
            raw,
            bias,
            0.001,
            Some(0.01),
            NemotronMamba2Mode::ChunkedPrefill,
            all_valid(),
        );
        assert_eq!(
            recurrent
                .into_data()
                .to_vec::<f16>()
                .unwrap()
                .into_iter()
                .map(f16::to_bits)
                .collect::<Vec<_>>(),
            [0x0673]
        );
        assert_eq!(
            chunked.into_data().to_vec::<f16>().unwrap(),
            [f16::from_f32(0.001)]
        );

        let bf16_dt = discretization_steps(
            Tensor::<TestBackend, 3>::from_floats([[[-9.726_562_5]]], &device).cast(DType::BF16),
            Tensor::<TestBackend, 1>::from_floats([0.501_464_84], &device).cast(DType::BF16),
            0.0,
            None,
            NemotronMamba2Mode::Recurrent,
            all_valid(),
        );
        let bf16_bits = bf16_dt
            .into_data()
            .to_vec::<bf16>()
            .unwrap()
            .into_iter()
            .map(bf16::to_bits)
            .collect::<Vec<_>>();
        assert_eq!(bf16_bits, [0x38ca]);
    }

    /// Recurrent decode preserves the fallback's low-precision update and output graph.
    ///
    /// The oracle comes from `mamba2_selective_state_update` at pinned revision
    /// `cfa309cfa34b5497a32fd50af49aa97aac94c054`. Computing `dt * x * B` and
    /// the state-to-`C` contraction entirely in FP32 changes the second output
    /// from `0x480e` to `0x480d`.
    #[test]
    fn recurrent_scan_matches_pinned_f16_operation_order() {
        let device = Default::default();
        let x = Tensor::<TestBackend, 4>::from_floats([[[[-2.974_609_4, -2.005_859_4]]]], &device)
            .cast(DType::F16);
        let b = Tensor::<TestBackend, 4>::from_floats([[[[-0.223_266_6, -0.369_873_05]]]], &device)
            .cast(DType::F16);
        let c = Tensor::<TestBackend, 4>::from_floats([[[[2.910_156_2, 3.505_859_4]]]], &device)
            .cast(DType::F16);
        let dt =
            Tensor::<TestBackend, 3>::from_floats([[[0.836_914_06]]], &device).cast(DType::F16);
        let a = Tensor::<TestBackend, 1>::from_floats([-1.803_729_5], &device);
        let d = Tensor::<TestBackend, 1>::from_floats([-1.289_062_5], &device).cast(DType::F16);
        let initial = Tensor::<TestBackend, 4>::from_floats(
            [[[[2.960_324_3, -0.659_675_4], [2.499_563_9, 0.827_825_8]]]],
            &device,
        );

        let (output, terminal) = recurrent_scan(x, b, c, dt, a, d, initial).unwrap();
        assert_eq!(output.dtype(), DType::F16);
        let bits = output
            .into_data()
            .to_vec::<f16>()
            .unwrap()
            .into_iter()
            .map(f16::to_bits)
            .collect::<Vec<_>>();
        assert_eq!(bits, [0x4909, 0x480e]);
        terminal.into_data().assert_approx_eq::<f32>(
            &TensorData::from([[[[1.210_400_9_f32, 0.775_106_43], [0.927_173_8, 0.804_048]]]]),
            Tolerance::absolute(2e-6),
        );
    }

    /// Chunked prefill keeps FP32 outputs while recurrent decode restores model dtype.
    ///
    /// This asymmetry is present in the pinned fallback: `mamba2_chunk_scan`
    /// promotes its scan tensors, whereas `mamba2_selective_state_update` casts
    /// its result back to the input activation dtype before grouped normalization.
    #[test]
    fn recurrent_and_chunked_scans_preserve_reference_output_dtypes() {
        let device = Default::default();
        let x = Tensor::<TestBackend, 4>::from_floats([[[[0.7, -0.2]], [[-0.4, 1.1]]]], &device)
            .cast(DType::F16);
        let b =
            Tensor::<TestBackend, 4>::from_floats([[[[0.3]], [[-0.8]]]], &device).cast(DType::F16);
        let c =
            Tensor::<TestBackend, 4>::from_floats([[[[1.2]], [[0.6]]]], &device).cast(DType::F16);
        let dt = Tensor::<TestBackend, 3>::from_floats([[[0.2], [0.9]]], &device).cast(DType::F16);
        let a = Tensor::<TestBackend, 1>::from_floats([-0.75], &device);
        let d = Tensor::<TestBackend, 1>::from_floats([0.4], &device).cast(DType::F16);
        let initial = Tensor::<TestBackend, 4>::from_floats([[[[0.25], [-0.5]]]], &device);

        let (recurrent, recurrent_state) = recurrent_scan(
            x.clone(),
            b.clone(),
            c.clone(),
            dt.clone(),
            a.clone(),
            d.clone(),
            initial.clone(),
        )
        .unwrap();
        let (chunked, chunked_state) =
            chunked_scan(x, b, c, dt, a, d.cast(DType::F32), initial, 2).unwrap();
        assert_eq!(recurrent.dtype(), DType::F16);
        assert_eq!(chunked.dtype(), DType::F32);
        assert_eq!(recurrent_state.dtype(), DType::F32);
        assert_eq!(chunked_state.dtype(), DType::F32);
        recurrent
            .cast(DType::F32)
            .into_data()
            .assert_approx_eq::<f32>(&chunked.into_data(), Tolerance::absolute(3e-3));

        let bf16_x = Tensor::<TestBackend, 4>::ones([1, 1, 1, 1], (&device, DType::BF16));
        let bf16_b = Tensor::<TestBackend, 4>::ones([1, 1, 1, 1], (&device, DType::BF16));
        let bf16_c = Tensor::<TestBackend, 4>::ones([1, 1, 1, 1], (&device, DType::BF16));
        let bf16_dt = Tensor::<TestBackend, 3>::ones([1, 1, 1], (&device, DType::BF16));
        let bf16_d = Tensor::<TestBackend, 1>::ones([1], (&device, DType::BF16));
        let bf16_initial = Tensor::<TestBackend, 4>::zeros([1, 1, 1, 1], &device);
        let (bf16_recurrent, _) = recurrent_scan(
            bf16_x.clone(),
            bf16_b.clone(),
            bf16_c.clone(),
            bf16_dt.clone(),
            Tensor::from_floats([-1.0], &device),
            bf16_d.clone(),
            bf16_initial.clone(),
        )
        .unwrap();
        let (bf16_chunked, _) = chunked_scan(
            bf16_x,
            bf16_b,
            bf16_c,
            bf16_dt,
            Tensor::from_floats([-1.0], &device),
            bf16_d.cast(DType::F32),
            bf16_initial,
            1,
        )
        .unwrap();
        assert_eq!(bf16_recurrent.dtype(), DType::BF16);
        assert_eq!(bf16_chunked.dtype(), DType::F32);
    }

    /// Reduced-precision recurrent caches round between consecutive tokens.
    ///
    /// A multi-token recurrent call is defined as repeated decode updates. This
    /// exact equality prevents an optimization from accidentally retaining an
    /// unpublished FP32 state between tokens when the configured cache is F16.
    #[test]
    fn f16_recurrent_cache_matches_repeated_single_token_updates() {
        let device = Default::default();
        let x = Tensor::<TestBackend, 4>::from_floats([[[[1.3, -0.7]], [[-0.2, 0.9]]]], &device)
            .cast(DType::F16);
        let b =
            Tensor::<TestBackend, 4>::from_floats([[[[0.6]], [[1.1]]]], &device).cast(DType::F16);
        let c =
            Tensor::<TestBackend, 4>::from_floats([[[[-0.4]], [[0.8]]]], &device).cast(DType::F16);
        let dt =
            Tensor::<TestBackend, 3>::from_floats([[[0.35], [0.55]]], &device).cast(DType::F16);
        let a = Tensor::<TestBackend, 1>::from_floats([-1.25], &device);
        let d = Tensor::<TestBackend, 1>::from_floats([0.75], &device).cast(DType::F16);
        let initial =
            Tensor::<TestBackend, 4>::from_floats([[[[0.33], [-0.27]]]], &device).cast(DType::F16);

        let (combined_output, combined_state) = recurrent_scan(
            x.clone(),
            b.clone(),
            c.clone(),
            dt.clone(),
            a.clone(),
            d.clone(),
            initial.clone(),
        )
        .unwrap();
        let (first_output, first_state) = recurrent_scan(
            x.clone().narrow(1, 0, 1),
            b.clone().narrow(1, 0, 1),
            c.clone().narrow(1, 0, 1),
            dt.clone().narrow(1, 0, 1),
            a.clone(),
            d.clone(),
            initial,
        )
        .unwrap();
        let (second_output, repeated_state) = recurrent_scan(
            x.narrow(1, 1, 1),
            b.narrow(1, 1, 1),
            c.narrow(1, 1, 1),
            dt.narrow(1, 1, 1),
            a,
            d,
            first_state,
        )
        .unwrap();
        assert_eq!(
            combined_output.to_data(),
            Tensor::cat(vec![first_output, second_output], 1).to_data()
        );
        assert_eq!(combined_state.to_data(), repeated_state.to_data());
    }

    /// Semantic invalid tokens produce identity recurrent transitions.
    ///
    /// Transformers' padding helper leaves `dt` active and would decay the
    /// state here. This crate intentionally zeroes it because stateful model
    /// execution compacts true-valid tokens before invoking Mamba.
    #[test]
    fn semantic_mask_prevents_padding_decay() {
        let device = Default::default();
        let valid = Tensor::<TestBackend, 2, Bool>::from_bool([[false]].into(), &device);
        let dt = discretization_steps(
            Tensor::zeros([1, 1, 1], &device),
            Tensor::ones([1], &device),
            0.001,
            None,
            NemotronMamba2Mode::Recurrent,
            valid,
        );
        assert_eq!(dt.to_data(), TensorData::zeros::<f32, _>([1, 1, 1]));

        let initial = Tensor::<TestBackend, 4>::from_floats([[[[3.25]]]], &device);
        let (_, terminal) = recurrent_scan(
            Tensor::zeros([1, 1, 1, 1], &device),
            Tensor::zeros([1, 1, 1, 1], &device),
            Tensor::zeros([1, 1, 1, 1], &device),
            dt,
            Tensor::from_floats([-7.0], &device),
            Tensor::ones([1], &device),
            initial.clone(),
        )
        .unwrap();
        assert_eq!(terminal.to_data(), initial.to_data());
    }

    /// Gated normalization rejects every invalid grouping or tensor geometry.
    #[test]
    fn grouped_norm_rejects_invalid_geometry() {
        let device = Default::default();
        let values = Tensor::<TestBackend, 3>::ones([1, 2, 4], &device);
        let gate = Tensor::<TestBackend, 3>::ones([1, 2, 4], &device);
        let norm = NemotronGroupedGatedRmsNorm::<TestBackend> {
            weight: Param::from_tensor(Tensor::ones([4], &device)),
            groups: 3,
            epsilon: 1e-6,
        };
        assert!(norm.forward(values.clone(), gate.clone()).is_err());

        let norm = NemotronGroupedGatedRmsNorm::<TestBackend> {
            weight: Param::from_tensor(Tensor::ones([3], &device)),
            groups: 2,
            epsilon: 1e-6,
        };
        assert!(norm.forward(values.clone(), gate).is_err());

        let norm = NemotronGroupedGatedRmsNorm::<TestBackend> {
            weight: Param::from_tensor(Tensor::ones([4], &device)),
            groups: 2,
            epsilon: 1e-6,
        };
        assert!(
            norm.forward(values, Tensor::ones([1, 1, 4], &device))
                .is_err()
        );

        let norm = NemotronGroupedGatedRmsNorm::<TestBackend> {
            weight: Param::from_tensor(Tensor::ones([4], &device)),
            groups: 2,
            epsilon: f64::NAN,
        };
        assert!(
            norm.forward(
                Tensor::ones([1, 2, 4], &device),
                Tensor::ones([1, 2, 4], &device),
            )
            .is_err()
        );
    }

    /// Invalid groups and state dtypes return contextual errors before kernels.
    #[test]
    fn invalid_configuration_and_state_are_rejected() {
        let device = Default::default();
        let mut invalid = config();
        invalid.groups = 3;
        assert!(invalid.validate().is_err());

        let mixer = config().init::<TestBackend>(&device).unwrap();
        let mut state = NemotronMamba2State::zeros(&config(), 1, DType::F32, &device).unwrap();
        state.recurrent = state.recurrent.cast(DType::F64);
        let error = mixer
            .forward_with_state(
                Tensor::zeros([1, 1, 8], &device),
                None,
                &state,
                NemotronMamba2Mode::Recurrent,
            )
            .unwrap_err();
        assert!(error.to_string().contains("recurrent"));
    }

    /// Low-precision terminal storage is promoted for scan math and restored afterward.
    #[test]
    fn recurrent_storage_dtype_round_trips_across_a_segment() {
        let device = Default::default();
        let mixer = config().init::<TestBackend>(&device).unwrap();
        let state =
            NemotronMamba2State::zeros_with_dtypes(&config(), 1, DType::F16, DType::F32, &device)
                .unwrap();
        let (_, next) = mixer
            .forward_with_state(
                Tensor::zeros([1, 2, 8], &device),
                None,
                &state,
                NemotronMamba2Mode::ChunkedPrefill,
            )
            .unwrap();
        assert_eq!(next.recurrent.dtype(), DType::F16);
        assert_eq!(next.convolution.dtype(), DType::F32);
        assert!(
            NemotronMamba2State::<TestBackend>::zeros_with_dtypes(
                &config(),
                1,
                DType::I32,
                DType::F32,
                &device,
            )
            .is_err()
        );
    }
}
