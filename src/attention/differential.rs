//! Differential attention (DIFF Transformer, DiffLlama).
//!
//! Differential attention subtracts a second softmax map to cancel common-mode
//! attention noise. With `H` query heads split into halves, pair `i < H / 2`
//! combines heads `i` and `i + H / 2`:
//!
//! `out_i = (1 - lambda_init) * RMS((A_i - lambda * A_(i + H/2)) [V_a ; V_b])`
//!
//! where `A_h = softmax(Q_h K_kv(h)^T / sqrt(d) + mask)`, `[V_a ; V_b]` is the
//! channel concatenation (`2d` wide) of the values of query heads `i` and
//! `i + H / 2`, `RMS` is a weightless RMS norm over those `2d` channels, and
//! `lambda = exp(lq1 . lk1) - exp(lq2 . lk2) + lambda_init` is reparameterized
//! from four learned `d`-vectors. The pairs flatten to `H / 2 * 2d = H * d`
//! channels for the output projection.
//!
//! # Native KV heads
//!
//! The reference repeats K/V to query heads and builds per-head paired values.
//! With an even number `kv` of KV heads, query head `h` of KV head `j` always
//! pairs `V_j` with its partner `V_(j +- kv / 2)`, so the paired value is a
//! property of the KV head: `W_j = [V_(j mod kv/2) ; V_(j mod kv/2 + kv/2)]`.
//! Evaluation therefore runs one grouped attention call over native KV heads
//! with `2d`-wide values (no repetition), giving `A_h W_kv(h)` for every head at
//! once. A single shared KV head pairs with itself; an odd count above one has
//! no per-KV-head pairing and falls back to repeating K/V to query heads.

use burn::{
    module::{Initializer, Module, Param},
    nn::{Linear, LinearConfig, LinearLayout},
    tensor::{Bool, DType, Tensor, backend::Backend, ops::AttentionModuleOptions},
};

use super::{
    AttentionPrecisionMode, AttentionProjection,
    dispatch::{Operands, evaluate, repeat_key_value},
    grouping::{ValuePairing, value_pairing},
    widths::{PairedWidthError, paired_projection_widths},
};
use crate::{
    Error, Result,
    normalization::{RmsNormRootMode, scale_free_rms_norm},
    operations,
};

/// Returns the DIFF Transformer depth schedule `0.8 - 0.6 * exp(-0.3 * layer)`.
///
/// `layer_index` is zero-based, matching DiffLlama's `lambda_init_fn`.
#[must_use]
pub fn depth_lambda_init(layer_index: usize) -> f64 {
    0.8 - 0.6 * (-0.3 * layer_index as f64).exp()
}

/// Configuration of one differential-attention layer.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct DifferentialAttentionConfig {
    /// Input and output residual-stream width.
    pub hidden_size: usize,
    /// Query heads; must be even because heads are used in pairs.
    pub query_heads: usize,
    /// Native key/value heads; must divide `query_heads`.
    pub key_value_heads: usize,
    /// Query/key/value width per head.
    pub head_dim: usize,
    /// Constant `lambda_init` added to the learned lambda and used as `1 - lambda_init` scale.
    pub lambda_init: f64,
    /// Stabilizer of the weightless RMS norm over each `2 * head_dim` pair.
    ///
    /// DiffLlama uses the model's `rms_norm_eps` (default `1e-5`); set it from
    /// the checkpoint configuration.
    pub norm_epsilon: f64,
    /// Whether all four projections carry additive biases.
    pub bias: bool,
    /// Numerical graph of the two attention maps.
    #[serde(default)]
    pub precision_mode: AttentionPrecisionMode,
}

impl DifferentialAttentionConfig {
    /// Creates a bias-free layer with the depth schedule's `lambda_init` and epsilon `1e-5`.
    ///
    /// The epsilon matches DiffLlama's default `rms_norm_eps`; checkpoints that
    /// configure another value must override [`Self::norm_epsilon`].
    #[must_use]
    pub fn new(
        hidden_size: usize,
        query_heads: usize,
        key_value_heads: usize,
        head_dim: usize,
        layer_index: usize,
    ) -> Self {
        Self {
            hidden_size,
            query_heads,
            key_value_heads,
            head_dim,
            lambda_init: depth_lambda_init(layer_index),
            norm_epsilon: 1e-5,
            bias: false,
            precision_mode: AttentionPrecisionMode::BackendDefault,
        }
    }

    /// Validates widths, head pairing, and numerical constants.
    ///
    /// The integer checks are verified with Verus (`paired_projection_widths`):
    /// success implies representable projections, grouped heads, and an even
    /// query-head count, which the pairing plan of [`Self::init`]'s layer needs.
    pub fn validate(&self) -> Result<()> {
        paired_projection_widths(
            self.hidden_size,
            self.query_heads,
            self.key_value_heads,
            self.head_dim,
        )
        .map_err(|error| match error {
            PairedWidthError::Widths(_) => Error::configuration(
                "differential_attention",
                "widths must be non-zero, key_value_heads must divide query_heads, and projections must fit usize",
            ),
            PairedWidthError::OddQueryHeads => Error::configuration(
                "differential_attention.query_heads",
                "must be even because heads are used in pairs",
            ),
        })?;
        if !self.lambda_init.is_finite() {
            return Err(Error::configuration(
                "differential_attention.lambda_init",
                "must be finite",
            ));
        }
        if !self.norm_epsilon.is_finite() || self.norm_epsilon <= 0.0 {
            return Err(Error::configuration(
                "differential_attention.norm_epsilon",
                "must be finite and greater than zero",
            ));
        }
        Ok(())
    }

    /// Initializes `[out, in]` checkpoint-layout projections and zero lambda vectors.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Result<DifferentialAttention<B>> {
        self.validate()?;
        let query_width = self.query_heads * self.head_dim;
        let key_value_width = self.key_value_heads * self.head_dim;
        let linear = |input, output| {
            operations::init_linear(
                &LinearConfig::new(input, output)
                    .with_bias(self.bias)
                    .with_layout(LinearLayout::Col),
                device,
            )
        };
        let lambda = || Initializer::Zeros.init([self.head_dim], device);
        Ok(DifferentialAttention {
            q_proj: linear(self.hidden_size, query_width),
            k_proj: linear(self.hidden_size, key_value_width),
            v_proj: linear(self.hidden_size, key_value_width),
            o_proj: linear(query_width, self.hidden_size),
            lambda_q1: lambda(),
            lambda_k1: lambda(),
            lambda_q2: lambda(),
            lambda_k2: lambda(),
            hidden_size: self.hidden_size,
            query_heads: self.query_heads,
            key_value_heads: self.key_value_heads,
            head_dim: self.head_dim,
            lambda_init: self.lambda_init,
            norm_epsilon: self.norm_epsilon,
            precision_mode: self.precision_mode,
        })
    }
}

/// Trainable differential-attention layer with DiffLlama parameter names.
#[derive(Module, Debug)]
pub struct DifferentialAttention<B: Backend> {
    /// Query projection to `query_heads * head_dim`.
    pub q_proj: Linear<B>,
    /// Native key-head projection.
    pub k_proj: Linear<B>,
    /// Native value-head projection.
    pub v_proj: Linear<B>,
    /// Output projection from `query_heads * head_dim`.
    pub o_proj: Linear<B>,
    /// First lambda query vector, `[head_dim]`.
    pub lambda_q1: Param<Tensor<B, 1>>,
    /// First lambda key vector, `[head_dim]`.
    pub lambda_k1: Param<Tensor<B, 1>>,
    /// Second lambda query vector, `[head_dim]`.
    pub lambda_q2: Param<Tensor<B, 1>>,
    /// Second lambda key vector, `[head_dim]`.
    pub lambda_k2: Param<Tensor<B, 1>>,
    /// Residual-stream width.
    pub hidden_size: usize,
    /// Query heads (even).
    pub query_heads: usize,
    /// Native key/value heads.
    pub key_value_heads: usize,
    /// Per-head channel width.
    pub head_dim: usize,
    /// Constant added to the learned lambda; the output is scaled by `1 - lambda_init`.
    pub lambda_init: f64,
    /// Stabilizer of the weightless pair RMS norm.
    pub norm_epsilon: f64,
    /// Numerical graph of the attention maps.
    pub precision_mode: AttentionPrecisionMode,
}

impl<B: Backend> DifferentialAttention<B> {
    /// Projects self-attention Q/K/V without a position transform.
    pub fn project_self(&self, hidden: Tensor<B, 3>) -> Result<AttentionProjection<B>> {
        self.project(hidden.clone(), hidden)
    }

    /// Projects queries and independent key/value sources to `[batch, head, token, head_dim]`.
    pub fn project(
        &self,
        query_hidden: Tensor<B, 3>,
        key_value_hidden: Tensor<B, 3>,
    ) -> Result<AttentionProjection<B>> {
        let [batch, _, query_width] = query_hidden.dims();
        let [key_batch, _, key_width] = key_value_hidden.dims();
        if query_width != self.hidden_size || key_width != self.hidden_size || key_batch != batch {
            return Err(Error::shape(
                "differential_attention.input",
                format!("query and source [batch, token, {}]", self.hidden_size),
                format!(
                    "query={:?}, source={:?}",
                    query_hidden.dims(),
                    key_value_hidden.dims()
                ),
            ));
        }
        let heads_first = |projected: Tensor<B, 3>, heads| {
            let [batch, sequence, _] = projected.dims();
            projected
                .reshape([batch, sequence, heads, self.head_dim])
                .swap_dims(1, 2)
        };
        Ok(AttentionProjection {
            query: heads_first(
                operations::linear(query_hidden, &self.q_proj)?,
                self.query_heads,
            ),
            key: heads_first(
                operations::linear(key_value_hidden.clone(), &self.k_proj)?,
                self.key_value_heads,
            ),
            value: heads_first(
                operations::linear(key_value_hidden, &self.v_proj)?,
                self.key_value_heads,
            ),
            gate: None,
        })
    }

    /// Returns `lambda = exp(lq1 . lk1) - exp(lq2 . lk2) + lambda_init` as `[1]` in `dtype`.
    ///
    /// Follows DiffLlama's operation order exactly: elementwise products in the
    /// parameter dtype, then an F32 sum and exponential, cast to `dtype` before
    /// the two terms combine. (For reduced-precision parameters the products
    /// round once in that dtype, as in the reference.)
    pub fn lambda(&self, dtype: DType) -> Tensor<B, 1> {
        let term = |query: &Param<Tensor<B, 1>>, key: &Param<Tensor<B, 1>>| {
            (query.val() * key.val())
                .cast(DType::F32)
                .sum()
                .exp()
                .cast(dtype)
        };
        term(&self.lambda_q1, &self.lambda_k1) - term(&self.lambda_q2, &self.lambda_k2)
            + self.lambda_init
    }

    /// Evaluates differential attention after the caller applies positional transforms.
    ///
    /// `mask` uses `true == excluded` with shape `[batch, query_head, query, key]`;
    /// causality is bottom-right aligned. Keys and values keep native KV heads
    /// and `head_dim` channels. Differential attention has no output gate, so
    /// a projection carrying one is rejected rather than silently ignored.
    pub fn attend(
        &self,
        projection: AttentionProjection<B>,
        mask: Option<Tensor<B, 4, Bool>>,
        causal: bool,
    ) -> Result<Tensor<B, 3>> {
        let AttentionProjection {
            query,
            key,
            value,
            gate,
        } = projection;
        if gate.is_some() {
            return Err(Error::configuration(
                "differential_attention.projection.gate",
                "differential attention has no output gate",
            ));
        }
        let [batch, query_heads, query_sequence, head_dim] = query.dims();
        if query_heads != self.query_heads
            || head_dim != self.head_dim
            || key.dims()[1] != self.key_value_heads
            || value.dims()[1] != self.key_value_heads
            || value.dims()[3] != self.head_dim
        {
            return Err(Error::shape(
                "differential_attention.projection",
                format!(
                    "query [batch, {}, query, {}] and K/V [batch, {}, key, {}]",
                    self.query_heads, self.head_dim, self.key_value_heads, self.head_dim
                ),
                format!(
                    "query={:?}, key={:?}, value={:?}",
                    query.dims(),
                    key.dims(),
                    value.dims()
                ),
            ));
        }
        let dtype = query.dtype();
        let (key, value) = paired_values(key, value, self.query_heads)?;
        // Every head's map times its pair's 2d-wide value: [b, H, q, 2d].
        let context = evaluate(
            Operands {
                query,
                key,
                value,
                mask,
                bias: None,
                sinks: None,
            },
            AttentionModuleOptions {
                scale: None,
                softcap: None,
                is_causal: causal,
            },
            self.precision_mode,
        )?;
        let pairs = self.query_heads / 2;
        let first = context.clone().narrow(1, 0, pairs);
        let second = context.narrow(1, pairs, pairs);
        let lambda = self.lambda(dtype).reshape([1, 1, 1, 1]);
        let difference = first - second * lambda;
        let normalized = scale_free_rms_norm(
            difference,
            self.norm_epsilon,
            RmsNormRootMode::MultiplyByReciprocalSqrt,
        )? * (1.0 - self.lambda_init);
        let flattened = normalized.swap_dims(1, 2).reshape([
            batch,
            query_sequence,
            self.query_heads * self.head_dim,
        ]);
        operations::linear(flattened, &self.o_proj)
    }
}

/// Builds `2d`-wide paired values over native KV heads (see the module docs).
///
/// Returns the keys and paired values to attend with: native heads when the
/// pairing is a property of each KV head, otherwise K/V repeated to query heads.
/// The verified plan ([`value_pairing`]) proves that every query head then
/// reads its reference pair `[V_kv(i) ; V_kv(i + H/2)]`.
fn paired_values<B: Backend>(
    key: Tensor<B, 4>,
    value: Tensor<B, 4>,
    query_heads: usize,
) -> Result<(Tensor<B, 4>, Tensor<B, 4>)> {
    let key_value_heads = value.dims()[1];
    let (key, value, half) = match value_pairing(query_heads, key_value_heads) {
        ValuePairing::Shared => return Ok((key, Tensor::cat(vec![value.clone(), value], 3))),
        ValuePairing::Native { half } => (key, value, half),
        // An odd head count above one pairs different partners within one
        // KV group, so pairing must happen per query head.
        ValuePairing::Repeated { half } => (
            repeat_key_value(key, query_heads)?,
            repeat_key_value(value, query_heads)?,
            half,
        ),
    };
    let low = Tensor::cat(
        vec![
            value.clone().narrow(1, 0, half),
            value.narrow(1, half, half),
        ],
        3,
    );
    // Heads j and j + half share the same pair [V_j ; V_(j + half)].
    Ok((key, Tensor::cat(vec![low.clone(), low], 1)))
}

#[cfg(test)]
#[path = "differential_tests.rs"]
mod tests;
