//! Projected full, grouped-query, and multi-query attention.
//!
//! Attention maps a query to a convex combination of values using
//! `softmax(Q K^T / sqrt(d)) V`. Grouped-query attention stores fewer key/value
//! heads than query heads and repeats each KV head across a contiguous query
//! group only for the attention operation. The cache therefore retains native
//! KV heads instead of paying the repeated representation's memory cost.
//!
//! Rotary and relative-position transformations are deliberately separate from
//! projection and attention dispatch. This lets one architecture rotate only a
//! prefix, another use NoPE layers, and T5 provide an additive relative bias
//! without duplicating projection code. Backend-native attention is the default.
//! Architectures whose pinned eager graphs materialize reduced-precision scores
//! before an explicit F32 softmax can opt into that exact operation order; the
//! unfused path can be paired with [`AttentionExecutionPolicy`] to cap every
//! score-like plane by evaluating complete attention rows in synchronized query
//! chunks. This is a storage bound, not a linear-time attention approximation:
//! full-attention chunks still visit every key.

use burn::{
    module::Module,
    nn::{Linear, LinearConfig, LinearLayout},
    tensor::{
        Bool, DType, Int, Tensor,
        activation::{sigmoid, silu, softmax},
        backend::Backend,
        module::attention,
        ops::AttentionModuleOptions,
    },
};

use crate::normalization::RmsNorm;
use crate::{Error, Result, architecture::TokenMask, normalization::RmsNormConfig, operations};

/// Multiplicative gate emitted alongside Qwen/Muse query projections.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttentionOutputGate {
    /// Do not emit or apply an output gate.
    #[default]
    None,
    /// Multiply head outputs by the logistic sigmoid of the gate projection.
    Sigmoid,
    /// Multiply head outputs by `gate * sigmoid(gate)`.
    Silu,
}

/// Numerical operation graph used to evaluate attention probabilities.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttentionPrecisionMode {
    /// Delegate arithmetic and kernel selection to the active Burn backend.
    #[default]
    BackendDefault,
    /// Materialize scores in the query dtype, run softmax in F32, then cast back.
    ///
    /// This matches eager reference graphs that explicitly call an FP32 softmax
    /// on reduced-precision scores. It deliberately uses an `O(query * key)`
    /// score tensor because flash kernels with strict-F32 QK accumulation omit
    /// the observable reduced-precision boundary.
    EagerFp32Softmax,
}

/// Controls the peak score-plane size of an attention operation.
///
/// A score plane contains one element for every
/// `[batch, head, query, key]` pair. Query chunking preserves the mathematical
/// attention rows because every chunk still sees its complete allowed key
/// range, while changing peak temporary storage from `O(query * key)` to
/// `O(query_chunk * key)`. Backends are synchronized between chunks so dropped
/// score, mask, and probability buffers can be reclaimed before the next chunk.
///
/// This policy bounds score-like intermediates, not parameters, Q/K/V storage,
/// the linear-size output, or an autodiff tape. Multi-chunk execution is
/// therefore rejected on autodiff backends; callers doing training can select
/// [`Self::Unbounded`] explicitly after provisioning the required memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttentionExecutionPolicy {
    /// Evaluate all query rows together without a score-plane bound.
    ///
    /// This is useful for small golden-oracle calls and explicitly provisioned
    /// training runs. Its peak attention workspace is quadratic for self
    /// attention.
    Unbounded,
    /// Split query rows while retaining the complete key support of every row.
    QueryChunked {
        /// Hard upper bound on query rows submitted in one attention call.
        max_query_tokens: usize,
        /// Hard upper bound on elements in any score-like chunk plane.
        max_score_elements: usize,
    },
}

impl Default for AttentionExecutionPolicy {
    /// Uses a conservative production bound suitable for 32K Muse inference.
    fn default() -> Self {
        Self::QueryChunked {
            max_query_tokens: 512,
            max_score_elements: 1 << 26,
        }
    }
}

impl AttentionExecutionPolicy {
    /// Constructs a synchronized query-chunked policy.
    #[must_use]
    pub const fn query_chunked(max_query_tokens: usize, max_score_elements: usize) -> Self {
        Self::QueryChunked {
            max_query_tokens,
            max_score_elements,
        }
    }

    /// Validates non-zero chunk and workspace limits.
    pub fn validate(self) -> Result<()> {
        if let Self::QueryChunked {
            max_query_tokens,
            max_score_elements,
        } = self
        {
            if max_query_tokens == 0 {
                return Err(Error::configuration(
                    "attention.execution.max_query_tokens",
                    "must be greater than zero",
                ));
            }
            if max_score_elements == 0 {
                return Err(Error::configuration(
                    "attention.execution.max_score_elements",
                    "must be greater than zero",
                ));
            }
        }
        Ok(())
    }

    /// Plans query chunks for a conservative upper bound on keys per chunk.
    ///
    /// `key_sequence_bound` may be smaller than the physical key tensor when a
    /// caller can prove that each chunk slices a local window first. The caller
    /// remains responsible for checking every concrete key slice against the
    /// resulting score-element limit.
    pub fn plan(
        self,
        batch: usize,
        heads: usize,
        query_sequence: usize,
        key_sequence_bound: usize,
    ) -> Result<AttentionExecutionPlan> {
        self.validate()?;
        if batch == 0 || heads == 0 || query_sequence == 0 || key_sequence_bound == 0 {
            return Err(Error::configuration(
                "attention.execution_shape",
                "batch, heads, query sequence, and key bound must be non-zero",
            ));
        }
        let elements_per_query = batch
            .checked_mul(heads)
            .and_then(|elements| elements.checked_mul(key_sequence_bound))
            .ok_or_else(|| {
                Error::configuration(
                    "attention.execution_shape",
                    "score-plane dimensions overflow usize",
                )
            })?;
        let (query_chunk_size, score_element_limit) = match self {
            Self::Unbounded => (query_sequence, None),
            Self::QueryChunked {
                max_query_tokens,
                max_score_elements,
            } => {
                if elements_per_query > max_score_elements {
                    return Err(Error::BackendCapability(format!(
                        "one attention query row requires {elements_per_query} score elements, exceeding the configured bound of {max_score_elements}"
                    )));
                }
                (
                    query_sequence
                        .min(max_query_tokens)
                        .min(max_score_elements / elements_per_query),
                    Some(max_score_elements),
                )
            }
        };
        let chunk_count = query_sequence.div_ceil(query_chunk_size);
        let peak_score_elements = elements_per_query
            .checked_mul(query_chunk_size.min(query_sequence))
            .ok_or_else(|| {
                Error::configuration(
                    "attention.execution_shape",
                    "planned score-plane size overflows usize",
                )
            })?;
        Ok(AttentionExecutionPlan {
            query_sequence,
            key_sequence_bound,
            query_chunk_size,
            chunk_count,
            peak_score_elements,
            score_element_limit,
        })
    }

    /// Returns the configured maximum query rows before score-bound reduction.
    #[must_use]
    pub const fn maximum_query_tokens(self, query_sequence: usize) -> usize {
        match self {
            Self::Unbounded => query_sequence,
            Self::QueryChunked {
                max_query_tokens, ..
            } => {
                if max_query_tokens < query_sequence {
                    max_query_tokens
                } else {
                    query_sequence
                }
            }
        }
    }

    /// Checks that one concrete chunk stays within this policy's score bound.
    pub fn validate_chunk(
        self,
        batch: usize,
        heads: usize,
        query_sequence: usize,
        key_sequence: usize,
    ) -> Result<()> {
        let plan = self.plan(batch, heads, query_sequence, key_sequence)?;
        if plan.chunk_count != 1 {
            return Err(Error::BackendCapability(format!(
                "attention chunk [{batch}, {heads}, {query_sequence}, {key_sequence}] exceeds the configured single-chunk score bound"
            )));
        }
        Ok(())
    }
}

/// Immutable allocation plan produced by [`AttentionExecutionPolicy`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttentionExecutionPlan {
    /// Complete query length covered by this plan.
    query_sequence: usize,
    /// Conservative maximum key length permitted for each planned chunk.
    key_sequence_bound: usize,
    /// Query rows evaluated in each chunk except possibly the final remainder.
    query_chunk_size: usize,
    /// Number of sequential attention calls.
    chunk_count: usize,
    /// Maximum score elements produced by the planned shape.
    peak_score_elements: usize,
    /// Configured bound, or `None` for unbounded execution.
    score_element_limit: Option<usize>,
}

impl AttentionExecutionPlan {
    /// Returns the complete query length covered by this plan.
    #[must_use]
    pub const fn query_sequence(self) -> usize {
        self.query_sequence
    }

    /// Returns the conservative maximum key length permitted for each chunk.
    #[must_use]
    pub const fn key_sequence_bound(self) -> usize {
        self.key_sequence_bound
    }

    /// Returns query rows evaluated per chunk.
    #[must_use]
    pub const fn query_chunk_size(self) -> usize {
        self.query_chunk_size
    }

    /// Returns the number of sequential chunks.
    #[must_use]
    pub const fn chunk_count(self) -> usize {
        self.chunk_count
    }

    /// Returns the maximum number of elements in one score-like plane.
    #[must_use]
    pub const fn peak_score_elements(self) -> usize {
        self.peak_score_elements
    }

    /// Returns the configured score-element limit when execution is bounded.
    #[must_use]
    pub const fn score_element_limit(self) -> Option<usize> {
        self.score_element_limit
    }
}

/// Configuration for full MHA, GQA, or MQA projections and attention dispatch.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct GroupedQueryAttentionConfig {
    /// Input and output residual-stream width.
    pub hidden_size: usize,
    /// Number of independently projected query heads.
    pub query_heads: usize,
    /// Number of cached key/value heads.
    pub key_value_heads: usize,
    /// Query/key/value width per head.
    pub head_dim: usize,
    /// Whether all projection layers include additive biases.
    pub bias: bool,
    /// Optional Q/K per-head RMS normalization epsilon.
    pub query_key_norm_epsilon: Option<f64>,
    /// Whether Q/K norm checkpoints store `scale - 1` instead of `scale`.
    pub query_key_norm_zero_centered: bool,
    /// Optional multiplicative output-gate convention.
    pub output_gate: AttentionOutputGate,
    /// Optional non-default score scale. `None` preserves fused-kernel eligibility.
    pub scale: Option<f64>,
    /// Optional tanh softcap applied to raw attention logits.
    pub softcap: Option<f64>,
    /// Numerical operation graph used for score and softmax evaluation.
    #[serde(default)]
    pub precision_mode: AttentionPrecisionMode,
}

impl GroupedQueryAttentionConfig {
    /// Creates bias-free attention with conventional `1/sqrt(head_dim)` scaling.
    pub fn new(
        hidden_size: usize,
        query_heads: usize,
        key_value_heads: usize,
        head_dim: usize,
    ) -> Self {
        Self {
            hidden_size,
            query_heads,
            key_value_heads,
            head_dim,
            bias: false,
            query_key_norm_epsilon: None,
            query_key_norm_zero_centered: false,
            output_gate: AttentionOutputGate::None,
            scale: None,
            softcap: None,
            precision_mode: AttentionPrecisionMode::BackendDefault,
        }
    }

    /// Selects an explicit score/softmax operation graph.
    pub fn with_precision_mode(mut self, precision_mode: AttentionPrecisionMode) -> Self {
        self.precision_mode = precision_mode;
        self
    }

    /// Validates projection sizes, GQA grouping, and numerical options.
    pub fn validate(&self) -> Result<()> {
        if self.hidden_size == 0
            || self.query_heads == 0
            || self.key_value_heads == 0
            || self.head_dim == 0
        {
            return Err(Error::configuration(
                "attention",
                "hidden_size, query_heads, key_value_heads, and head_dim must be non-zero",
            ));
        }
        if !self.query_heads.is_multiple_of(self.key_value_heads) {
            return Err(Error::configuration(
                "attention.key_value_heads",
                "must divide query_heads exactly",
            ));
        }
        let query_width = self.query_heads.checked_mul(self.head_dim);
        let key_value_width = self.key_value_heads.checked_mul(self.head_dim);
        let query_output_width = query_width.and_then(|width| {
            if self.output_gate == AttentionOutputGate::None {
                Some(width)
            } else {
                width.checked_mul(2)
            }
        });
        // Validate the complete matrix products, not just per-head widths:
        // lazy Burn parameters otherwise defer overflowing shapes until use.
        if query_output_width
            .and_then(|width| self.hidden_size.checked_mul(width))
            .is_none()
            || key_value_width
                .and_then(|width| self.hidden_size.checked_mul(width))
                .is_none()
        {
            return Err(Error::configuration(
                "attention",
                "projected head or matrix dimensions overflow usize",
            ));
        }
        if self
            .query_key_norm_epsilon
            .is_some_and(|epsilon| !epsilon.is_finite() || epsilon <= 0.0)
        {
            return Err(Error::configuration(
                "attention.query_key_norm_epsilon",
                "must be finite and greater than zero when present",
            ));
        }
        if self
            .scale
            .is_some_and(|scale| !scale.is_finite() || scale <= 0.0)
        {
            return Err(Error::configuration(
                "attention.scale",
                "must be finite and greater than zero when present",
            ));
        }
        if self
            .softcap
            .is_some_and(|softcap| !softcap.is_finite() || softcap <= 0.0)
        {
            return Err(Error::configuration(
                "attention.softcap",
                "must be finite and greater than zero when present",
            ));
        }
        Ok(())
    }

    /// Initializes checkpoint-native `[out, in]` projections and optional Q/K norms.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Result<GroupedQueryAttention<B>> {
        self.validate()?;
        let query_width = self
            .query_heads
            .checked_mul(self.head_dim)
            .ok_or_else(|| Error::configuration("attention", "query width overflows usize"))?;
        let key_value_width = self
            .key_value_heads
            .checked_mul(self.head_dim)
            .ok_or_else(|| Error::configuration("attention", "key/value width overflows usize"))?;
        let query_output_width = if self.output_gate == AttentionOutputGate::None {
            query_width
        } else {
            query_width.checked_mul(2).ok_or_else(|| {
                Error::configuration("attention", "gated query width overflows usize")
            })?
        };
        let linear = |input, output| {
            operations::init_linear(
                &LinearConfig::new(input, output)
                    .with_bias(self.bias)
                    .with_layout(LinearLayout::Col),
                device,
            )
        };
        let query_key_norm = |epsilon| {
            RmsNormConfig::new(self.head_dim)
                .with_epsilon(epsilon)
                .with_zero_centered(self.query_key_norm_zero_centered)
                .init(device)
        };
        let (query_norm, key_norm) = match self.query_key_norm_epsilon {
            Some(epsilon) => (
                Some(query_key_norm(epsilon)?),
                Some(query_key_norm(epsilon)?),
            ),
            None => (None, None),
        };
        Ok(GroupedQueryAttention {
            q_proj: linear(self.hidden_size, query_output_width),
            k_proj: linear(self.hidden_size, key_value_width),
            v_proj: linear(self.hidden_size, key_value_width),
            o_proj: linear(query_width, self.hidden_size),
            query_norm,
            key_norm,
            hidden_size: self.hidden_size,
            query_heads: self.query_heads,
            key_value_heads: self.key_value_heads,
            head_dim: self.head_dim,
            output_gate: self.output_gate,
            scale: self.scale,
            softcap: self.softcap,
            precision_mode: self.precision_mode,
        })
    }
}

/// Query, key, value, and optional gate tensors before position transformation.
#[derive(Clone, Debug)]
pub struct AttentionProjection<B: Backend> {
    /// Queries shaped `[batch, query_heads, query_sequence, head_dim]`.
    pub query: Tensor<B, 4>,
    /// Native keys shaped `[batch, key_value_heads, key_sequence, head_dim]`.
    pub key: Tensor<B, 4>,
    /// Native values shaped `[batch, key_value_heads, key_sequence, head_dim]`.
    pub value: Tensor<B, 4>,
    /// Optional gate shaped like the flattened query-head output.
    pub gate: Option<Tensor<B, 3>>,
}

/// Checkpoint-compatible grouped-query attention projections.
#[derive(Module, Debug)]
pub struct GroupedQueryAttention<B: Backend> {
    /// Query projection, optionally concatenated with an output gate.
    pub q_proj: Linear<B>,
    /// Native key-head projection.
    pub k_proj: Linear<B>,
    /// Native value-head projection.
    pub v_proj: Linear<B>,
    /// Projection from concatenated query-head outputs to the residual stream.
    pub o_proj: Linear<B>,
    /// Optional per-query-head normalization.
    pub query_norm: Option<RmsNorm<B>>,
    /// Optional per-key-head normalization.
    pub key_norm: Option<RmsNorm<B>>,
    /// Input/output width of the residual stream.
    pub hidden_size: usize,
    /// Number of query heads.
    pub query_heads: usize,
    /// Number of native key/value heads.
    pub key_value_heads: usize,
    /// Per-head channel width.
    pub head_dim: usize,
    /// Output-gating convention.
    pub output_gate: AttentionOutputGate,
    /// Optional explicit attention score scale.
    pub scale: Option<f64>,
    /// Optional attention-logit softcap.
    pub softcap: Option<f64>,
    /// Numerical operation graph used for score and softmax evaluation.
    pub precision_mode: AttentionPrecisionMode,
}

impl<B: Backend> GroupedQueryAttention<B> {
    /// Projects self-attention Q/K/V tensors without applying a position transform.
    pub fn project_self(&self, hidden: Tensor<B, 3>) -> Result<AttentionProjection<B>> {
        self.project(hidden.clone(), hidden)
    }

    /// Projects query states and independent key/value source states.
    pub fn project(
        &self,
        query_hidden: Tensor<B, 3>,
        key_value_hidden: Tensor<B, 3>,
    ) -> Result<AttentionProjection<B>> {
        let [batch, query_sequence, query_hidden_size] = query_hidden.dims();
        let [key_batch, key_sequence, key_value_hidden_size] = key_value_hidden.dims();
        if batch == 0 || query_sequence == 0 || query_hidden_size != self.hidden_size {
            return Err(Error::shape(
                "attention.query_input",
                format!("[batch>0, sequence>0, {}]", self.hidden_size),
                format!("[{batch}, {query_sequence}, {query_hidden_size}]"),
            ));
        }
        if key_batch != batch || key_sequence == 0 || key_value_hidden_size != self.hidden_size {
            return Err(Error::shape(
                "attention.key_value_input",
                format!("[{batch}, sequence>0, {}]", self.hidden_size),
                format!("[{key_batch}, {key_sequence}, {key_value_hidden_size}]"),
            ));
        }
        let query_width = self.query_heads * self.head_dim;
        let projected_query = operations::linear(query_hidden, &self.q_proj)?;
        let (query, gate) = if self.output_gate == AttentionOutputGate::None {
            (
                projected_query
                    .reshape([batch, query_sequence, self.query_heads, self.head_dim])
                    .swap_dims(1, 2),
                None,
            )
        } else {
            // Qwen-family gated projections are interleaved by head:
            // `[head_0.query, head_0.gate, head_1.query, ...]`.
            let projected_query = projected_query.reshape([
                batch,
                query_sequence,
                self.query_heads,
                self.head_dim * 2,
            ]);
            (
                projected_query
                    .clone()
                    .narrow(3, 0, self.head_dim)
                    .swap_dims(1, 2),
                Some(
                    projected_query
                        .narrow(3, self.head_dim, self.head_dim)
                        .reshape([batch, query_sequence, query_width]),
                ),
            )
        };
        let key = operations::linear(key_value_hidden.clone(), &self.k_proj)?
            .reshape([batch, key_sequence, self.key_value_heads, self.head_dim])
            .swap_dims(1, 2);
        let value = operations::linear(key_value_hidden, &self.v_proj)?
            .reshape([batch, key_sequence, self.key_value_heads, self.head_dim])
            .swap_dims(1, 2);
        let query = self
            .query_norm
            .as_ref()
            .map_or(query.clone(), |norm| norm.forward(query));
        let key = self
            .key_norm
            .as_ref()
            .map_or(key.clone(), |norm| norm.forward(key));
        Ok(AttentionProjection {
            query,
            key,
            value,
            gate,
        })
    }

    /// Dispatches projected attention after the caller applies positional transforms.
    ///
    /// `mask` follows Burn's low-level convention: `true` means excluded. Public
    /// token-valid masks should first pass through [`padding_attention_mask`].
    pub fn attend(
        &self,
        projection: AttentionProjection<B>,
        mask: Option<Tensor<B, 4, Bool>>,
        bias: Option<Tensor<B, 4>>,
        causal: bool,
    ) -> Result<Tensor<B, 3>> {
        validate_projection(self, &projection, mask.as_ref(), bias.as_ref())?;
        let [batch, _, query_sequence, _] = projection.query.dims();
        let key = repeat_key_value(projection.key, self.query_heads)?;
        let value = repeat_key_value(projection.value, self.query_heads)?;
        let context = dispatch_attention(
            projection.query,
            key,
            value,
            mask,
            bias,
            AttentionModuleOptions {
                scale: self.scale,
                softcap: self.softcap,
                is_causal: causal,
            },
            self.precision_mode,
        )?
        .swap_dims(1, 2)
        .reshape([batch, query_sequence, self.query_heads * self.head_dim]);
        let context = match (self.output_gate, projection.gate) {
            (AttentionOutputGate::None, None) => context,
            (AttentionOutputGate::Sigmoid, Some(gate)) => context * sigmoid(gate),
            (AttentionOutputGate::Silu, Some(gate)) => context * silu(gate),
            _ => {
                return Err(Error::shape(
                    "attention.gate",
                    format!("a gate matching {:?}", self.output_gate),
                    "a mismatched gate payload",
                ));
            }
        };
        operations::linear(context, &self.o_proj)
    }

    /// Evaluates attention without RoPE or relative-position bias.
    pub fn forward_without_position_encoding(
        &self,
        hidden: Tensor<B, 3>,
        token_mask: Option<&TokenMask<B>>,
        causal: bool,
    ) -> Result<Tensor<B, 3>> {
        let projection = self.project_self(hidden)?;
        let [batch, _, query_sequence, _] = projection.query.dims();
        let key_sequence = projection.key.dims()[2];
        let mask = padding_attention_mask(
            token_mask,
            token_mask,
            batch,
            self.query_heads,
            query_sequence,
            key_sequence,
        )?;
        self.attend(projection, mask, None, causal)
    }
}

/// Dispatches either backend-native or explicitly ordered eager attention.
///
/// The eager mode computes score scaling and optional softcapping in the input
/// dtype, applies exclusions and additive bias, promotes only the softmax to
/// F32, casts probabilities back, and multiplies by `V` in the original dtype.
/// Callers must provide shape-compatible Q/K/V, mask, and bias tensors.
pub(crate) fn dispatch_attention<B: Backend>(
    query: Tensor<B, 4>,
    key: Tensor<B, 4>,
    value: Tensor<B, 4>,
    mask: Option<Tensor<B, 4, Bool>>,
    bias: Option<Tensor<B, 4>>,
    options: AttentionModuleOptions,
    precision_mode: AttentionPrecisionMode,
) -> Result<Tensor<B, 4>> {
    if precision_mode == AttentionPrecisionMode::BackendDefault {
        return Ok(attention(query, key, value, mask, bias, options));
    }
    let [batch, heads, query_sequence, head_dim] = query.dims();
    let key_sequence = key.dims()[2];
    let scale = options
        .scale
        .unwrap_or_else(|| (head_dim as f64).sqrt().recip());
    let scores = query.matmul(key.swap_dims(2, 3)) * scale;
    let scores = if let Some(softcap) = options.softcap {
        (scores / softcap).tanh() * softcap
    } else {
        scores
    };
    let causal_mask = if options.is_causal {
        Some(bottom_right_causal_mask(
            batch,
            heads,
            query_sequence,
            key_sequence,
            &scores.device(),
        )?)
    } else {
        None
    };
    let exclusion = match (mask, causal_mask) {
        (Some(mask), Some(causal_mask)) => Some(mask.bool_or(causal_mask)),
        (Some(mask), None) => Some(mask),
        (None, Some(causal_mask)) => Some(causal_mask),
        (None, None) => None,
    };
    let scores = if let Some(exclusion) = &exclusion {
        scores.mask_fill(exclusion.clone(), f32::NEG_INFINITY)
    } else {
        scores
    };
    let scores = if let Some(bias) = bias {
        scores + bias
    } else {
        scores
    };
    let probabilities = fp32_softmax(scores, exclusion);
    Ok(probabilities.matmul(value))
}

/// Applies FP32 softmax and defines fully excluded rows as exact zeros.
fn fp32_softmax<B: Backend>(
    scores: Tensor<B, 4>,
    exclusion: Option<Tensor<B, 4, Bool>>,
) -> Tensor<B, 4> {
    let dtype = scores.dtype();
    let Some(exclusion) = exclusion else {
        return softmax(scores.cast(DType::F32), 3).cast(dtype);
    };
    let shape = scores.dims();
    let row_has_key = exclusion
        .clone()
        .bool_not()
        .int()
        .sum_dim(3)
        .greater_elem(0)
        .expand(shape);
    softmax(
        scores
            .cast(DType::F32)
            .mask_fill(row_has_key.bool_not(), 0.0),
        3,
    )
    .mask_fill(exclusion, 0.0)
    .cast(dtype)
}

/// Builds the bottom-right causal exclusion `key > query + (K - Q)`.
///
/// Bottom-right alignment is essential for cached decoding, where a query
/// chunk is a suffix of a longer key sequence. `true` excludes a pair.
fn bottom_right_causal_mask<B: Backend>(
    batch: usize,
    heads: usize,
    query_sequence: usize,
    key_sequence: usize,
    device: &B::Device,
) -> Result<Tensor<B, 4, Bool>> {
    let query_end = i64::try_from(query_sequence)
        .map_err(|_| Error::configuration("attention.causal_mask", "query length exceeds i64"))?;
    let key_end = i64::try_from(key_sequence)
        .map_err(|_| Error::configuration("attention.causal_mask", "key length exceeds i64"))?;
    let offset = key_end - query_end;
    let query = (Tensor::<B, 1, Int>::arange(0..query_end, device) + offset)
        .reshape([1, 1, query_sequence, 1])
        .expand([batch, heads, query_sequence, key_sequence]);
    let key = Tensor::<B, 1, Int>::arange(0..key_end, device)
        .reshape([1, 1, 1, key_sequence])
        .expand([batch, heads, query_sequence, key_sequence]);
    Ok(key.greater(query))
}

/// Repeats native KV heads into contiguous query-head groups for one attention call.
pub fn repeat_key_value<B: Backend>(
    tensor: Tensor<B, 4>,
    query_heads: usize,
) -> Result<Tensor<B, 4>> {
    let [batch, key_value_heads, sequence, head_dim] = tensor.dims();
    if key_value_heads == 0 || query_heads == 0 || !query_heads.is_multiple_of(key_value_heads) {
        return Err(Error::configuration(
            "attention.head_grouping",
            format!(
                "query_heads ({query_heads}) must be non-zero and divisible by key_value_heads ({key_value_heads})"
            ),
        ));
    }
    let groups = query_heads / key_value_heads;
    if groups == 1 {
        return Ok(tensor);
    }
    Ok(tensor
        .unsqueeze_dim::<5>(2)
        .expand([batch, key_value_heads, groups, sequence, head_dim])
        .reshape([batch, query_heads, sequence, head_dim]))
}

/// Validates projected tensor topology before dispatching a backend attention kernel.
fn validate_projection<B: Backend>(
    attention: &GroupedQueryAttention<B>,
    projection: &AttentionProjection<B>,
    mask: Option<&Tensor<B, 4, Bool>>,
    bias: Option<&Tensor<B, 4>>,
) -> Result<()> {
    let [batch, query_heads, query_sequence, query_dim] = projection.query.dims();
    let [key_batch, key_value_heads, key_sequence, key_dim] = projection.key.dims();
    let [value_batch, value_heads, value_sequence, value_dim] = projection.value.dims();
    // External callers can assemble projections without going through project().
    // Preserve its non-empty contract here; in particular, an empty key axis
    // cannot define the probability distribution required by softmax attention.
    if batch == 0 || query_sequence == 0 || key_sequence == 0 {
        return Err(Error::shape(
            "attention.projection",
            "non-empty batch, query sequence, and key sequence",
            format!(
                "query={:?}, key={:?}",
                projection.query.dims(),
                projection.key.dims()
            ),
        ));
    }
    let expected_query = [
        batch,
        attention.query_heads,
        query_sequence,
        attention.head_dim,
    ];
    if [batch, query_heads, query_sequence, query_dim] != expected_query {
        return Err(Error::shape(
            "attention.query",
            format!("{:?}", expected_query),
            format!("{:?}", projection.query.dims()),
        ));
    }
    let expected_key = [
        batch,
        attention.key_value_heads,
        key_sequence,
        attention.head_dim,
    ];
    if [key_batch, key_value_heads, key_sequence, key_dim] != expected_key {
        return Err(Error::shape(
            "attention.key",
            format!("{:?}", expected_key),
            format!("{:?}", projection.key.dims()),
        ));
    }
    let expected_value = [
        batch,
        attention.key_value_heads,
        key_sequence,
        attention.head_dim,
    ];
    if [value_batch, value_heads, value_sequence, value_dim] != expected_value {
        return Err(Error::shape(
            "attention.value",
            format!("{:?}", expected_value),
            format!("{:?}", projection.value.dims()),
        ));
    }
    let expected_gate = [
        batch,
        query_sequence,
        attention.query_heads * attention.head_dim,
    ];
    match (attention.output_gate, projection.gate.as_ref()) {
        (AttentionOutputGate::None, None) => {}
        (AttentionOutputGate::None, Some(_)) => {
            return Err(Error::shape(
                "attention.gate",
                "no gate for ungated attention",
                "a gate tensor",
            ));
        }
        (_, Some(gate)) if gate.dims() == expected_gate => {}
        (_, Some(gate)) => {
            return Err(Error::shape(
                "attention.gate",
                format!("{:?}", expected_gate),
                format!("{:?}", gate.dims()),
            ));
        }
        (_, None) => {
            return Err(Error::shape(
                "attention.gate",
                format!("{:?}", expected_gate),
                "no gate tensor",
            ));
        }
    }
    let expected_additive = [batch, attention.query_heads, query_sequence, key_sequence];
    if let Some(mask) = mask
        && mask.dims() != expected_additive
    {
        return Err(Error::shape(
            "attention.mask",
            format!("{:?}", expected_additive),
            format!("{:?}", mask.dims()),
        ));
    }
    if let Some(bias) = bias {
        let [bias_batch, bias_heads, bias_query, bias_key] = bias.dims();
        if ![1, batch].contains(&bias_batch)
            || bias_heads != attention.query_heads
            || bias_query != query_sequence
            || bias_key != key_sequence
        {
            return Err(Error::shape(
                "attention.bias",
                format!(
                    "[batch in {{1, {batch}}}, {}, {query_sequence}, {key_sequence}]",
                    attention.query_heads
                ),
                format!("{:?}", bias.dims()),
            ));
        }
    }
    Ok(())
}

/// Converts semantic valid-token masks into Burn's `true == excluded` mask.
pub fn padding_attention_mask<B: Backend>(
    query_mask: Option<&TokenMask<B>>,
    key_mask: Option<&TokenMask<B>>,
    batch: usize,
    heads: usize,
    query_sequence: usize,
    key_sequence: usize,
) -> Result<Option<Tensor<B, 4, Bool>>> {
    if batch == 0 || heads == 0 || query_sequence == 0 || key_sequence == 0 {
        return Err(Error::configuration(
            "attention.mask_shape",
            "batch, heads, query sequence, and key sequence must be non-zero",
        ));
    }
    let query_invalid = if let Some(mask) = query_mask {
        let [mask_batch, mask_sequence] = mask.as_tensor().dims();
        if [mask_batch, mask_sequence] != [batch, query_sequence] {
            return Err(Error::shape(
                "query_mask",
                format!("[{batch}, {query_sequence}]"),
                format!("[{mask_batch}, {mask_sequence}]"),
            ));
        }
        Some(
            mask.as_tensor()
                .clone()
                .bool_not()
                .reshape([batch, 1, query_sequence, 1])
                .expand([batch, heads, query_sequence, key_sequence]),
        )
    } else {
        None
    };
    let key_invalid = if let Some(mask) = key_mask {
        let [mask_batch, mask_sequence] = mask.as_tensor().dims();
        if [mask_batch, mask_sequence] != [batch, key_sequence] {
            return Err(Error::shape(
                "key_mask",
                format!("[{batch}, {key_sequence}]"),
                format!("[{mask_batch}, {mask_sequence}]"),
            ));
        }
        Some(
            mask.as_tensor()
                .clone()
                .bool_not()
                .reshape([batch, 1, 1, key_sequence])
                .expand([batch, heads, query_sequence, key_sequence]),
        )
    } else {
        None
    };
    Ok(match (query_invalid, key_invalid) {
        (Some(query), Some(key)) => Some(query.bool_or(key)),
        (Some(query), None) => Some(query),
        (None, Some(key)) => Some(key),
        (None, None) => None,
    })
}

#[cfg(test)]
#[path = "attention_tests.rs"]
mod tests;
