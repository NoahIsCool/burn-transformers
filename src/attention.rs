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
mod tests {
    use burn::{
        backend::Flex,
        tensor::{Bool, Tensor, ops::AttentionModuleOptions},
    };

    use crate::architecture::TokenMask;

    use super::{
        AttentionExecutionPolicy, AttentionOutputGate, AttentionPrecisionMode, AttentionProjection,
        GroupedQueryAttentionConfig, bottom_right_causal_mask, dispatch_attention,
        padding_attention_mask, repeat_key_value,
    };

    /// The production default bounds official-width 32K score planes to 64 queries.
    #[test]
    fn execution_policy_plans_official_width_32k_attention() {
        let policy = AttentionExecutionPolicy::default();
        let global = policy
            .plan(1, 32, 32_768, 32_768)
            .expect("one 32K score row fits the production bound");
        assert_eq!(global.query_chunk_size(), 64);
        assert_eq!(global.chunk_count(), 512);
        assert_eq!(global.peak_score_elements(), 1 << 26);
        assert_eq!(global.score_element_limit(), Some(1 << 26));
        assert_eq!(global.query_sequence(), 32_768);
        assert_eq!(global.key_sequence_bound(), 32_768);
        policy
            .validate_chunk(1, 32, 64, 32_768)
            .expect("the exact boundary is one valid chunk");
        assert!(policy.validate_chunk(1, 32, 65, 32_768).is_err());

        let local = policy
            .plan(1, 32, 32_768, 2048 + 511)
            .expect("Muse's local key-union bound must fit");
        assert_eq!(local.query_chunk_size(), 512);
        assert_eq!(local.chunk_count(), 64);
        assert_eq!(local.peak_score_elements(), 41_926_656);
        assert_eq!(local.key_sequence_bound(), 2559);

        let oracle = policy
            .plan(1, 32, 67, 67)
            .expect("the pinned Muse oracle prompt is intentionally unchunked");
        assert_eq!(oracle.query_chunk_size(), 67);
        assert_eq!(oracle.chunk_count(), 1);
    }

    /// Invalid or impossible bounds fail before allocating device tensors.
    #[test]
    fn execution_policy_rejects_zero_and_oversized_rows() {
        let zero_queries = AttentionExecutionPolicy::query_chunked(0, 1);
        assert!(
            zero_queries
                .validate()
                .unwrap_err()
                .to_string()
                .contains("max_query_tokens")
        );
        let zero_scores = AttentionExecutionPolicy::query_chunked(1, 0);
        assert!(
            zero_scores
                .validate()
                .unwrap_err()
                .to_string()
                .contains("max_score_elements")
        );
        let too_small = AttentionExecutionPolicy::query_chunked(8, 63);
        let error = too_small
            .plan(1, 2, 8, 32)
            .expect_err("one row contains 64 score elements");
        assert!(error.to_string().contains("one attention query row"));
        assert!(
            AttentionExecutionPolicy::default()
                .plan(0, 1, 1, 1)
                .is_err()
        );
    }

    /// Explicit unbounded plans retain a single full-query operation.
    #[test]
    fn unbounded_execution_plan_is_transparent() {
        let plan = AttentionExecutionPolicy::Unbounded
            .plan(2, 3, 5, 7)
            .expect("small shape");
        assert_eq!(plan.query_chunk_size(), 5);
        assert_eq!(plan.chunk_count(), 1);
        assert_eq!(plan.peak_score_elements(), 210);
        assert_eq!(plan.score_element_limit(), None);
    }

    /// Backend-native attention remains the serialized and construction default.
    #[test]
    fn attention_precision_mode_is_explicitly_opt_in() {
        let device = Default::default();
        let default = GroupedQueryAttentionConfig::new(8, 2, 1, 4)
            .init::<Flex>(&device)
            .expect("default attention should initialize");
        assert_eq!(
            default.precision_mode,
            AttentionPrecisionMode::BackendDefault
        );
        let eager = GroupedQueryAttentionConfig::new(8, 2, 1, 4)
            .with_precision_mode(AttentionPrecisionMode::EagerFp32Softmax)
            .init::<Flex>(&device)
            .expect("eager attention should initialize");
        assert_eq!(
            eager.precision_mode,
            AttentionPrecisionMode::EagerFp32Softmax
        );
    }

    /// Explicit eager softmax returns zero rather than NaN for an excluded row.
    #[test]
    fn eager_fp32_softmax_zeroes_fully_excluded_rows() {
        let device = Default::default();
        let query = Tensor::<Flex, 4>::zeros([1, 1, 2, 2], &device);
        let key = Tensor::<Flex, 4>::zeros([1, 1, 2, 2], &device);
        let value = Tensor::<Flex, 4>::from_floats([[[[2.0, 3.0], [5.0, 7.0]]]], &device);
        let exclusion =
            Tensor::<Flex, 4, Bool>::from_bool([[[[true, true], [false, true]]]].into(), &device);
        let output = dispatch_attention(
            query,
            key,
            value,
            Some(exclusion),
            None,
            AttentionModuleOptions::default(),
            AttentionPrecisionMode::EagerFp32Softmax,
        )
        .expect("valid eager attention");
        assert_eq!(
            output
                .into_data()
                .as_slice::<f32>()
                .expect("Flex output should expose f32"),
            &[0.0, 0.0, 2.0, 3.0]
        );
    }

    /// Cached causal alignment anchors each query row at the key suffix.
    #[test]
    fn eager_causal_mask_is_bottom_right_aligned() {
        let device = Default::default();
        let suffix = bottom_right_causal_mask::<Flex>(1, 1, 2, 4, &device)
            .expect("representable suffix mask")
            .into_data();
        assert_eq!(
            suffix
                .as_slice::<bool>()
                .expect("boolean mask should expose bool"),
            &[false, false, false, true, false, false, false, false]
        );
        let longer_query = bottom_right_causal_mask::<Flex>(1, 1, 3, 2, &device)
            .expect("representable negative offset")
            .into_data();
        assert_eq!(
            longer_query
                .as_slice::<bool>()
                .expect("boolean mask should expose bool"),
            &[true, true, false, true, false, false]
        );
    }

    /// Qwen's projected query width may exceed its residual width.
    #[test]
    fn attention_accepts_non_square_query_projection() {
        GroupedQueryAttentionConfig::new(10, 3, 1, 4)
            .validate()
            .expect("query width need not equal residual width");
    }

    /// Projected-width arithmetic is checked before an initializer sees a shape.
    #[test]
    fn attention_rejects_projection_dimension_overflow() {
        let error = GroupedQueryAttentionConfig::new(8, usize::MAX, 1, 2)
            .validate()
            .expect_err("query head product must not wrap");
        assert!(error.to_string().contains("overflow"));
    }

    /// KV repetition duplicates each native head into one contiguous GQA group.
    #[test]
    fn key_value_repetition_preserves_group_order() {
        let device = Default::default();
        let tensor = Tensor::<Flex, 4>::from_floats([[[[1.0]], [[2.0]]]], &device);
        let repeated = repeat_key_value(tensor, 4)
            .expect("four query heads group over two key/value heads")
            .to_data();
        assert_eq!(
            repeated
                .as_slice::<f32>()
                .expect("Flex f32 data should expose an f32 slice"),
            &[1.0, 1.0, 2.0, 2.0]
        );
    }

    /// Invalid head ratios are reported instead of reaching a division or reshape panic.
    #[test]
    fn key_value_repetition_rejects_invalid_grouping() {
        let device = Default::default();
        let tensor = Tensor::<Flex, 4>::zeros([1, 2, 1, 3], &device);
        let error = repeat_key_value(tensor, 3).expect_err("three heads cannot group over two");
        assert!(error.to_string().contains("head_grouping"));
    }

    /// External projected-attention callers receive shape diagnostics before dispatch.
    #[test]
    fn attention_rejects_malformed_projection() {
        let device = Default::default();
        let module = GroupedQueryAttentionConfig::new(8, 4, 2, 3)
            .init::<Flex>(&device)
            .expect("valid attention should initialize");
        let projection = AttentionProjection {
            query: Tensor::zeros([1, 3, 2, 3], &device),
            key: Tensor::zeros([1, 2, 2, 3], &device),
            value: Tensor::zeros([1, 2, 2, 3], &device),
            gate: None,
        };
        let error = module
            .attend(projection, None, None, true)
            .expect_err("wrong query-head count must fail");
        assert!(error.to_string().contains("attention.query"));
    }

    /// Projection validates residual width before dispatching a linear backend kernel.
    #[test]
    fn attention_rejects_malformed_projection_input() {
        let device = Default::default();
        let module = GroupedQueryAttentionConfig::new(8, 4, 2, 3)
            .init::<Flex>(&device)
            .expect("valid attention should initialize");
        let error = module
            .project_self(Tensor::zeros([1, 2, 7], &device))
            .expect_err("wrong residual width must fail");
        assert!(error.to_string().contains("attention.query_input"));
    }

    /// Valid-token masks invert exactly once at the low-level attention boundary.
    #[test]
    fn padding_mask_marks_invalid_queries_and_keys() {
        let device = Default::default();
        let valid = TokenMask::new(Tensor::<Flex, 2, Bool>::from_bool(
            [[true, false]].into(),
            &device,
        ));
        let mask = padding_attention_mask(Some(&valid), Some(&valid), 1, 1, 2, 2)
            .expect("valid mask shape should pass")
            .expect("provided masks should produce a tensor")
            .to_data();
        assert_eq!(
            mask.as_slice::<bool>()
                .expect("boolean data should expose a bool slice"),
            &[false, true, true, true]
        );
    }

    /// Invalid mask extents return a contextual shape error rather than panicking.
    #[test]
    fn padding_mask_rejects_wrong_shape() {
        let device = Default::default();
        let valid = TokenMask::new(Tensor::<Flex, 2, Bool>::from_bool(
            [[true, true, true]].into(),
            &device,
        ));
        let error = padding_attention_mask(Some(&valid), None, 1, 1, 2, 2)
            .expect_err("mismatched query mask must fail");
        assert!(error.to_string().contains("query_mask"));
    }

    /// Projection, GQA dispatch, output gating, and output projection preserve shape.
    #[test]
    fn gated_attention_preserves_residual_shape() {
        let device = Default::default();
        let mut config = GroupedQueryAttentionConfig::new(8, 4, 2, 3);
        config.output_gate = AttentionOutputGate::Silu;
        config.query_key_norm_epsilon = Some(1e-6);
        let module = config
            .init::<Flex>(&device)
            .expect("valid attention should initialize");
        let output = module
            .forward_without_position_encoding(
                Tensor::<Flex, 3>::zeros([2, 5, 8], &device),
                None,
                true,
            )
            .expect("attention should execute");
        assert_eq!(output.dims(), [2, 5, 8]);
    }

    /// Rejects numerical and cardinality options before parameter initialization.
    #[test]
    fn config_validates_zero_axes_head_ratios_and_each_numerical_option() {
        for dimensions in [
            (0, 2, 1, 2),
            (4, 0, 1, 2),
            (4, 2, 0, 2),
            (4, 2, 1, 0),
            (4, 3, 2, 2),
        ] {
            let (hidden, queries, keys, width) = dimensions;
            assert!(matches!(
                GroupedQueryAttentionConfig::new(hidden, queries, keys, width).validate(),
                Err(crate::Error::Configuration { .. })
            ));
        }
        for option in ["query_key_norm_epsilon", "scale", "softcap"] {
            for value in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                let mut config = GroupedQueryAttentionConfig::new(4, 2, 1, 2);
                match option {
                    "query_key_norm_epsilon" => config.query_key_norm_epsilon = Some(value),
                    "scale" => config.scale = Some(value),
                    "softcap" => config.softcap = Some(value),
                    _ => unreachable!(),
                }
                assert!(
                    matches!(config.validate(), Err(crate::Error::Configuration { field, .. })
                    if field == format!("attention.{option}"))
                );
            }
        }
        let mut gated = GroupedQueryAttentionConfig::new(1, usize::MAX, 1, 1);
        gated.output_gate = AttentionOutputGate::Sigmoid;
        assert!(
            gated
                .validate()
                .unwrap_err()
                .to_string()
                .contains("overflow")
        );
        assert!(
            GroupedQueryAttentionConfig::new(usize::MAX, 2, 1, 1)
                .validate()
                .unwrap_err()
                .to_string()
                .contains("matrix dimensions overflow")
        );
    }

    /// Query chunk planning checks products without ever allocating score planes.
    #[test]
    fn execution_policy_covers_overflow_remaining_zero_axes_and_remainders() {
        for shape in [(1, 0, 1, 1), (1, 1, 0, 1), (1, 1, 1, 0)] {
            assert!(
                AttentionExecutionPolicy::Unbounded
                    .plan(shape.0, shape.1, shape.2, shape.3)
                    .is_err()
            );
        }
        for shape in [
            (usize::MAX, 2, 1, 1),
            (1, usize::MAX, 1, 2),
            (usize::MAX / 2 + 1, 1, 2, 1),
        ] {
            assert!(
                matches!(AttentionExecutionPolicy::Unbounded.plan(shape.0, shape.1, shape.2, shape.3),
                Err(crate::Error::Configuration { field, reason }) if field == "attention.execution_shape" && reason.contains("overflow"))
            );
        }
        let policy = AttentionExecutionPolicy::query_chunked(4, 30);
        let plan = policy.plan(1, 2, 7, 5).unwrap();
        assert_eq!(plan.query_chunk_size(), 3);
        assert_eq!(plan.chunk_count(), 3);
        assert_eq!(plan.peak_score_elements(), 30);
        assert_eq!(policy.maximum_query_tokens(2), 2);
        assert_eq!(policy.maximum_query_tokens(7), 4);
        assert_eq!(
            AttentionExecutionPolicy::Unbounded.maximum_query_tokens(7),
            7
        );
        policy.validate_chunk(1, 2, 1, 5).unwrap();
    }

    /// Constructs a rectangular projection without initializing random parameters.
    fn projection(batch: usize, query: usize, key: usize) -> AttentionProjection<Flex> {
        let device = Default::default();
        AttentionProjection {
            query: Tensor::zeros([batch, 2, query, 1], &device),
            key: Tensor::zeros([batch, 1, key, 1], &device),
            value: Tensor::zeros([batch, 1, key, 1], &device),
            gate: None,
        }
    }

    /// Empty externally assembled projections must fail before backend kernels.
    #[test]
    fn projected_attention_rejects_empty_batch_query_and_key_axes() {
        let module = GroupedQueryAttentionConfig::new(2, 2, 1, 1)
            .init::<Flex>(&Default::default())
            .unwrap();
        for (batch, query, key) in [(0, 1, 1), (1, 0, 1), (1, 1, 0)] {
            assert!(
                matches!(module.attend(projection(batch, query, key), None, None, false),
                    Err(crate::Error::Shape { tensor, .. }) if tensor == "attention.projection"),
                "accepted empty projected axes ({batch}, {query}, {key})"
            );
        }
    }

    /// Projected keys, values, gates, masks, and every bias axis are checked.
    #[test]
    fn projected_attention_checks_each_external_topology_boundary() {
        let device = Default::default();
        let module = GroupedQueryAttentionConfig::new(2, 2, 1, 1)
            .init::<Flex>(&device)
            .unwrap();
        for (field, shape) in [
            ("key", [1, 1, 3, 1]),
            ("key", [2, 2, 3, 1]),
            ("key", [2, 1, 3, 2]),
            ("value", [1, 1, 3, 1]),
            ("value", [2, 2, 3, 1]),
            ("value", [2, 1, 2, 1]),
            ("value", [2, 1, 3, 2]),
        ] {
            let mut projected = projection(2, 2, 3);
            if field == "key" {
                projected.key = Tensor::zeros(shape, &device);
            } else {
                projected.value = Tensor::zeros(shape, &device);
            }
            assert!(
                matches!(super::validate_projection(&module, &projected, None, None),
                Err(crate::Error::Shape { tensor, .. }) if tensor == format!("attention.{field}"))
            );
        }
        let projected = projection(2, 2, 3);
        let mask = Tensor::<Flex, 4, Bool>::zeros([1, 2, 2, 3], &device);
        assert!(
            matches!(super::validate_projection(&module, &projected, Some(&mask), None),
            Err(crate::Error::Shape { tensor, .. }) if tensor == "attention.mask")
        );
        for shape in [[3, 2, 2, 3], [1, 1, 2, 3], [1, 2, 1, 3], [1, 2, 2, 2]] {
            let bias = Tensor::zeros(shape, &device);
            assert!(
                matches!(super::validate_projection(&module, &projected, None, Some(&bias)),
                Err(crate::Error::Shape { tensor, .. }) if tensor == "attention.bias")
            );
        }
        for (gate_mode, gate_shape) in [
            (AttentionOutputGate::None, Some([2, 2, 2])),
            (AttentionOutputGate::Sigmoid, None),
            (AttentionOutputGate::Silu, Some([2, 2, 1])),
        ] {
            let mut module = module.clone();
            module.output_gate = gate_mode;
            let mut projected = projection(2, 2, 3);
            projected.gate = gate_shape.map(|shape| Tensor::zeros(shape, &device));
            assert!(
                matches!(super::validate_projection(&module, &projected, None, None),
                Err(crate::Error::Shape { tensor, .. }) if tensor == "attention.gate")
            );
        }
    }

    /// Cross-attention validates both independent inputs before any linear call.
    #[test]
    fn projection_checks_empty_query_and_independent_key_value_extents() {
        let device = Default::default();
        let module = GroupedQueryAttentionConfig::new(2, 2, 1, 1)
            .init::<Flex>(&device)
            .unwrap();
        for shape in [[0, 1, 2], [1, 0, 2]] {
            assert!(matches!(module.project_self(Tensor::zeros(shape, &device)),
                Err(crate::Error::Shape { tensor, .. }) if tensor == "attention.query_input"));
        }
        for shape in [[2, 1, 2], [1, 0, 2], [1, 1, 3]] {
            assert!(
                matches!(module.project(Tensor::zeros([1, 1, 2], &device), Tensor::zeros(shape, &device)),
                Err(crate::Error::Shape { tensor, .. }) if tensor == "attention.key_value_input")
            );
        }
    }

    /// Query-only and key-only masks expand on different axes and invert once.
    #[test]
    fn padding_masks_cover_one_sided_masks_and_invalid_key_extents() {
        let device = Default::default();
        let query = TokenMask::new(Tensor::<Flex, 2, Bool>::from_bool(
            [[false, true]].into(),
            &device,
        ));
        let key = TokenMask::new(Tensor::<Flex, 2, Bool>::from_bool(
            [[true, false, true]].into(),
            &device,
        ));
        let query_only = padding_attention_mask(Some(&query), None, 1, 2, 2, 3)
            .unwrap()
            .unwrap();
        assert_eq!(
            query_only.into_data().to_vec::<bool>().unwrap(),
            [true, true, true, false, false, false].repeat(2)
        );
        let key_only = padding_attention_mask(None, Some(&key), 1, 2, 2, 3)
            .unwrap()
            .unwrap();
        assert_eq!(
            key_only.into_data().to_vec::<bool>().unwrap(),
            [false, true, false].repeat(4)
        );
        assert!(
            padding_attention_mask::<Flex>(None, None, 1, 2, 2, 3)
                .unwrap()
                .is_none()
        );
        assert!(
            matches!(padding_attention_mask(None, Some(&query), 1, 2, 2, 3),
            Err(crate::Error::Shape { tensor, .. }) if tensor == "key_mask")
        );
        for (batch, heads, queries, keys) in
            [(0, 1, 1, 1), (1, 0, 1, 1), (1, 1, 0, 1), (1, 1, 1, 0)]
        {
            assert!(
                padding_attention_mask::<Flex>(None, None, batch, heads, queries, keys).is_err()
            );
        }
    }

    /// Eager logits apply scale/softcap before bias and combine exclusion sources.
    #[test]
    fn eager_softcap_bias_and_causal_mask_match_scalar_softmax_oracle() {
        let device = Default::default();
        for causal in [false, true] {
            for excluded_middle in [false, true] {
                let query = Tensor::<Flex, 4>::ones([1, 1, 2, 1], &device);
                let key = Tensor::from_floats([[[[1.0], [2.0], [3.0]]]], &device);
                let value = Tensor::from_floats([[[[2.0], [4.0], [8.0]]]], &device);
                let mask = excluded_middle.then(|| {
                    Tensor::<Flex, 4, Bool>::from_bool(
                        [[[[false, true, false], [false, true, false]]]].into(),
                        &device,
                    )
                });
                let bias = Tensor::from_floats(
                    [[[[0.0, 2.0_f32.ln(), 0.0], [0.0, 2.0_f32.ln(), 0.0]]]],
                    &device,
                );
                let output = dispatch_attention(
                    query,
                    key,
                    value,
                    mask,
                    Some(bias),
                    AttentionModuleOptions {
                        scale: Some(2.0),
                        softcap: Some(1.0),
                        is_causal: causal,
                    },
                    AttentionPrecisionMode::EagerFp32Softmax,
                )
                .unwrap()
                .into_data()
                .to_vec::<f32>()
                .unwrap();
                for (query_index, actual) in output.into_iter().enumerate() {
                    let mut numerator = 0.0_f64;
                    let mut denominator = 0.0_f64;
                    for (key_index, value) in [2.0, 4.0, 8.0].into_iter().enumerate() {
                        if (causal && key_index > query_index + 1)
                            || (excluded_middle && key_index == 1)
                        {
                            continue;
                        }
                        let logit = (2.0 * (key_index + 1) as f64).tanh()
                            + if key_index == 1 { 2.0_f64.ln() } else { 0.0 };
                        numerator += value * logit.exp();
                        denominator += logit.exp();
                    }
                    assert!((f64::from(actual) - numerator / denominator).abs() < 1e-6);
                }
            }
        }
    }

    /// Full/MQA projections share the same gate convention and broadcast bias.
    #[test]
    fn output_gates_and_broadcast_bias_produce_exact_projected_contexts() {
        use burn::module::Param;

        let device = Default::default();
        for gate_mode in [
            AttentionOutputGate::None,
            AttentionOutputGate::Sigmoid,
            AttentionOutputGate::Silu,
        ] {
            let mut config = GroupedQueryAttentionConfig::new(2, 2, 1, 1);
            config.output_gate = gate_mode;
            config.precision_mode = AttentionPrecisionMode::EagerFp32Softmax;
            let mut module = config.init::<Flex>(&device).unwrap();
            module.o_proj.weight =
                Param::from_tensor(Tensor::from_floats([[1.0, 0.0], [0.0, 1.0]], &device));
            let mut projected = projection(2, 2, 2);
            projected.value = Tensor::from_floats([[[[2.0], [4.0]]], [[[2.0], [4.0]]]], &device);
            projected.gate = (gate_mode != AttentionOutputGate::None)
                .then(|| Tensor::full([2, 2, 2], 2.0, &device));
            let bias = Tensor::from_floats(
                [[
                    [[0.0, 3.0_f32.ln()], [0.0, 3.0_f32.ln()]],
                    [[0.0, 3.0_f32.ln()], [0.0, 3.0_f32.ln()]],
                ]],
                &device,
            );
            let output = module.attend(projected, None, Some(bias), false).unwrap();
            let expected = match gate_mode {
                AttentionOutputGate::None => 3.5,
                AttentionOutputGate::Sigmoid => 3.5 / (1.0 + (-2.0_f32).exp()),
                AttentionOutputGate::Silu => 7.0 / (1.0 + (-2.0_f32).exp()),
            };
            for value in output.into_data().to_vec::<f32>().unwrap() {
                assert!((value - expected).abs() < 1e-6);
            }
        }
    }

    /// Zero-centered Q/K norms affect heads while V retains its projection bias.
    #[test]
    fn biased_cross_projection_applies_zero_centered_qk_norms_only_to_queries_and_keys() {
        use burn::module::Param;

        let device = Default::default();
        let mut config = GroupedQueryAttentionConfig::new(2, 1, 1, 2);
        config.bias = true;
        config.query_key_norm_epsilon = Some(1e-6);
        config.query_key_norm_zero_centered = true;
        let mut module = config.init::<Flex>(&device).unwrap();
        for linear in [&mut module.q_proj, &mut module.k_proj, &mut module.v_proj] {
            linear.weight =
                Param::from_tensor(Tensor::from_floats([[1.0, 0.0], [0.0, 1.0]], &device));
            linear.bias = Some(Param::from_tensor(Tensor::ones([2], &device)));
        }
        let output = module
            .project(
                Tensor::from_floats([[[3.0, 4.0]]], &device),
                Tensor::from_floats([[[0.0, 0.0], [1.0, 2.0]]], &device),
            )
            .unwrap();
        let expected_query = [4.0_f32, 5.0].map(|value| value / (20.5_f32 + 1e-6).sqrt());
        for (actual, expected) in output
            .query
            .into_data()
            .to_vec::<f32>()
            .unwrap()
            .into_iter()
            .zip(expected_query)
        {
            assert!((actual - expected).abs() < 1e-6);
        }
        let expected_key = [
            1.0 / (1.0_f32 + 1e-6).sqrt(),
            1.0 / (1.0_f32 + 1e-6).sqrt(),
            2.0 / (6.5_f32 + 1e-6).sqrt(),
            3.0 / (6.5_f32 + 1e-6).sqrt(),
        ];
        for (actual, expected) in output
            .key
            .into_data()
            .to_vec::<f32>()
            .unwrap()
            .into_iter()
            .zip(expected_key)
        {
            assert!((actual - expected).abs() < 1e-6);
        }
        assert_eq!(
            output.value.into_data().to_vec::<f32>().unwrap(),
            [1.0, 1.0, 2.0, 3.0]
        );
    }

    /// Unrepresentable causal positions fail before an arange can allocate storage.
    #[cfg(target_pointer_width = "64")]
    #[test]
    fn causal_mask_rejects_lengths_beyond_i64() {
        let device = Default::default();
        let oversized = i64::MAX as usize + 1;
        for (query, key, message) in [(oversized, 1, "query length"), (1, oversized, "key length")]
        {
            let error = bottom_right_causal_mask::<Flex>(1, 1, query, key, &device).unwrap_err();
            assert!(error.to_string().contains(message));
        }
    }
}
