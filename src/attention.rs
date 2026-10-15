//! Projected full, grouped-query, and multi-query attention.
//!
//! Attention maps a query to a convex combination of values using
//! `softmax(Q K^T / sqrt(d)) V`. Grouped-query attention stores fewer key/value
//! heads than query heads; each KV head serves a contiguous group of query
//! heads. Caches and every K/V intervention point keep the native KV heads.
//! Attention evaluation also keeps them: a group's queries are folded into
//! extra query rows of one call instead of repeating the KV history per query
//! head (see [`grouped_query_attention`]).
//!
//! Rotary and relative-position transformations are deliberately separate from
//! projection and attention dispatch. This lets one architecture rotate only a
//! prefix, another use NoPE layers, and T5 provide an additive relative bias
//! without duplicating projection code. Backend-native attention is the default.
//! Architectures whose pinned eager graphs materialize reduced-precision scores
//! before an explicit F32 softmax can opt into that exact operation order.
//!
//! [`AttentionExecutionPolicy`] plans query chunks that bound every score-like
//! plane; it is a storage bound, not a linear-time approximation, because each
//! chunk still visits its complete key support. [`GroupedQueryAttention`]
//! itself always evaluates complete planes; the Muse-Glimmer family currently
//! executes such plans with its synchronized chunk loop.
//!
//! [`AttentionConnectivity`] builds padding-aware causal, sliding-window, and
//! chunked-local exclusion masks for per-layer local/global schedules, and
//! [`sliding_window_key_range`] gives the matching key slice of a query chunk.
//!
//! Queries use `[batch, query_head, query_token, head_dim]`; keys and values
//! retain native KV heads until dispatch. Score masks use `true == excluded`,
//! whereas incoming token-validity masks use `true == valid`. Causal alignment is
//! bottom-right for cached prefixes on every backend: query row `i` of `Q` sees
//! keys `j <= i + K - Q`. Exclusions are restored after interventions, and eager
//! softmax defines fully excluded rows as zero probabilities.

mod connectivity;
mod dispatch;
mod mask;
mod plan;

use core::ops::Range;

use burn::{
    module::Module,
    nn::{Linear, LinearConfig, LinearLayout},
    tensor::{
        Bool, DType, Tensor,
        activation::{sigmoid, silu},
        backend::Backend,
        ops::AttentionModuleOptions,
    },
};

pub use connectivity::{AttentionConnectivity, MaskGeometry, valid_token_ordinals};
use dispatch::{Operands, causal_exclusion, default_scale, grouped_mix, scaled_scores};
pub(crate) use dispatch::{dispatch_attention, eager_probabilities, zero_excluded};
pub use dispatch::{grouped_query_attention, repeat_key_value};
use mask::padding_plane;
pub(crate) use mask::{bottom_right_causal_mask, union_exclusions};

use plan::{ChunkLimits, ChunkPlanError, plan_query_chunks, query_chunk_range};

use crate::intervention::{AttentionPoints, Hooks};
use crate::normalization::RmsNorm;
use crate::{Error, Result, architecture::TokenMask, normalization::RmsNormConfig, operations};

/// Borrowed structural edit applied to materialized attention probabilities.
type PatternTransform<'a, B> = dyn FnMut(Tensor<B, 4>) -> Result<Tensor<B, 4>> + 'a;

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
    /// caller can prove that each chunk slices a local window first (see
    /// [`sliding_window_key_range`]). The caller remains responsible for
    /// checking every concrete key slice against the resulting score-element
    /// limit. The integer planning is verified with Verus (`verus/attention.rs`).
    pub fn plan(
        self,
        batch: usize,
        heads: usize,
        query_sequence: usize,
        key_sequence_bound: usize,
    ) -> Result<AttentionExecutionPlan> {
        let limits = match self {
            Self::Unbounded => None,
            Self::QueryChunked {
                max_query_tokens,
                max_score_elements,
            } => Some(ChunkLimits {
                max_query_tokens,
                max_score_elements,
            }),
        };
        let geometry = plan_query_chunks(batch, heads, query_sequence, key_sequence_bound, limits)
            .map_err(chunk_plan_error)?;
        Ok(AttentionExecutionPlan {
            query_sequence,
            key_sequence_bound,
            query_chunk_size: geometry.query_chunk_size,
            chunk_count: geometry.chunk_count,
            peak_score_elements: geometry.peak_score_elements,
            score_element_limit: limits.map(|limits| limits.max_score_elements),
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

    /// Returns the query rows of chunk `index`, or `None` past the last chunk.
    ///
    /// Chunk `i` covers `[i * c, min((i + 1) * c, Q))`; consecutive chunks
    /// share their boundary and together partition `0..Q` (verified with Verus).
    #[must_use]
    pub fn query_range(self, index: usize) -> Option<Range<usize>> {
        query_chunk_range(
            self.query_chunk_size,
            self.chunk_count,
            self.query_sequence,
            index,
        )
        .map(|(start, end)| start..end)
    }

    /// Iterates every chunk's query rows in order.
    pub fn query_ranges(self) -> impl Iterator<Item = Range<usize>> {
        (0..self.chunk_count).filter_map(move |index| self.query_range(index))
    }
}

/// Returns the keys visible to aligned query rows under a causal sliding window.
///
/// Queries are the bottom-right suffix of the keys (`query_sequence <=
/// key_sequence`), so row `t` sits at key position `t + K - Q` and sees keys
/// `(t + K - Q - window, t + K - Q]`. The result is the exact union over
/// `query_rows`, `[max(0, start + K - Q + 1 - window), end + K - Q)`, whose
/// width is at most `query_rows.len() + window - 1`. Returns `None` for an
/// empty or out-of-bounds row range, `query_sequence > key_sequence`, or a
/// zero window. Containment of every visible key is verified with Verus.
#[must_use]
pub fn sliding_window_key_range(
    query_sequence: usize,
    key_sequence: usize,
    query_rows: Range<usize>,
    window: usize,
) -> Option<Range<usize>> {
    plan::sliding_window_key_range(
        query_sequence,
        key_sequence,
        query_rows.start,
        query_rows.end,
        window,
    )
    .map(|(start, end)| start..end)
}

/// Maps a verified planning failure to the crate's contextual error.
fn chunk_plan_error(error: ChunkPlanError) -> Error {
    match error {
        ChunkPlanError::ZeroMaxQueryTokens => Error::configuration(
            "attention.execution.max_query_tokens",
            "must be greater than zero",
        ),
        ChunkPlanError::ZeroMaxScoreElements => Error::configuration(
            "attention.execution.max_score_elements",
            "must be greater than zero",
        ),
        ChunkPlanError::EmptyAxis => Error::configuration(
            "attention.execution_shape",
            "batch, heads, query sequence, and key bound must be non-zero",
        ),
        ChunkPlanError::RowOverflow => Error::configuration(
            "attention.execution_shape",
            "score-plane dimensions overflow usize",
        ),
        ChunkPlanError::RowExceedsLimit {
            row_elements,
            max_score_elements,
        } => Error::BackendCapability(format!(
            "one attention query row requires {row_elements} score elements, exceeding the configured bound of {max_score_elements}"
        )),
        ChunkPlanError::PeakOverflow => Error::configuration(
            "attention.execution_shape",
            "planned score-plane size overflows usize",
        ),
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
    /// Projected tensors and additive bias must share a dense floating dtype and
    /// device; the mask and output projection must also use that device. No
    /// implicit dtype conversions are performed. Keys and values keep their
    /// native KV heads; see [`grouped_query_attention`] for the evaluation.
    pub fn attend(
        &self,
        projection: AttentionProjection<B>,
        mask: Option<Tensor<B, 4, Bool>>,
        bias: Option<Tensor<B, 4>>,
        causal: bool,
    ) -> Result<Tensor<B, 3>> {
        validate_projection(self, &projection, mask.as_ref(), bias.as_ref())?;
        let AttentionProjection {
            query,
            key,
            value,
            gate,
        } = projection;
        let context = self.head_context(
            Operands {
                query,
                key,
                value,
                mask,
                bias,
                sinks: None,
            },
            causal,
        )?;
        self.project_heads(context, gate)
    }

    /// Evaluates attention with selected Q/K/V, score, pattern, or head edits.
    ///
    /// Q/K/V points observe the supplied projection: after any normalization
    /// and positional transforms already performed by the caller, with native
    /// KV heads (`[batch, kv_head, token, head_dim]` for K/V). Head context is
    /// `[batch, query_head, query_token, head_dim]` before the output gate and
    /// projection. These points preserve the configured attention dispatch.
    ///
    /// Scores and patterns require [`AttentionPrecisionMode::EagerFp32Softmax`]
    /// because backend-native kernels do not expose a portable intermediate
    /// arithmetic graph. Scores are scaled and softcapped, before exclusion
    /// masks and additive bias, as `[batch, query_head, query_token, key_token]`.
    /// Patterns follow softmax and the cast back to the query dtype. Selecting
    /// either point materializes the full quadratic score plane; callers must
    /// provision memory for it.
    ///
    /// Pattern edits need not remain normalized or positive, allowing direct
    /// ablation and other experiments. Excluded links are zeroed again after
    /// each edit, so later captures observe the effective pattern and edits
    /// cannot undo causal or padding exclusions. Change
    /// `causal` or `mask` explicitly to change visibility. Fully excluded rows
    /// are zero before a head-context callback, which can intentionally inject
    /// a new context. Without requested points, this calls [`Self::attend`].
    /// Model wrappers supply the surrounding [`Hooks::run`] scope; direct
    /// callers should do the same using their supported point catalog.
    pub fn attend_with_hooks(
        &self,
        projection: AttentionProjection<B>,
        mask: Option<Tensor<B, 4, Bool>>,
        bias: Option<Tensor<B, 4>>,
        causal: bool,
        points: AttentionPoints,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<Tensor<B, 3>> {
        self.attend_with_transform(projection, mask, bias, causal, points, hooks, None)
    }

    /// Builds a modified attention operation with a structural pattern transform.
    ///
    /// Unlike an experimental observer, `transform` runs on every call, even
    /// with empty hooks. It receives the post-softmax weights and must preserve
    /// shape, dtype, and device. Its result need not be normalized or positive.
    /// Exclusions are restored before user pattern captures and edits, so those
    /// actions observe the transformed weights actually used by this model.
    ///
    /// This operation requires [`AttentionPrecisionMode::EagerFp32Softmax`] and
    /// materializes a full score plane. Use it to implement altered attention
    /// mathematics in a module, while retaining independent experimental hooks.
    #[allow(clippy::too_many_arguments)]
    pub fn attend_with_pattern(
        &self,
        projection: AttentionProjection<B>,
        mask: Option<Tensor<B, 4, Bool>>,
        bias: Option<Tensor<B, 4>>,
        causal: bool,
        points: AttentionPoints,
        hooks: &mut Hooks<'_, B>,
        mut transform: impl FnMut(Tensor<B, 4>) -> Result<Tensor<B, 4>>,
    ) -> Result<Tensor<B, 3>> {
        self.attend_with_transform(
            projection,
            mask,
            bias,
            causal,
            points,
            hooks,
            Some(&mut transform),
        )
    }

    /// Dispatches projected attention while preserving hook order and exclusion masks.
    ///
    /// Native kernels retain Q/K/V and head-context access without allocating scores.
    /// Score hooks or a structural pattern transform require eager F32 softmax and
    /// materialize `[batch, query_head, query_token, key_token]` planes. Edits cannot
    /// restore excluded links; excluded pattern entries remain zero.
    #[allow(clippy::too_many_arguments)]
    fn attend_with_transform(
        &self,
        mut projection: AttentionProjection<B>,
        mask: Option<Tensor<B, 4, Bool>>,
        bias: Option<Tensor<B, 4>>,
        causal: bool,
        points: AttentionPoints,
        hooks: &mut Hooks<'_, B>,
        transform: Option<&mut PatternTransform<'_, B>>,
    ) -> Result<Tensor<B, 3>> {
        if !hooks.has_work() && transform.is_none() {
            return self.attend(projection, mask, bias, causal);
        }
        let inspect_scores = hooks.needs_point(points.scores().info());
        let inspect_pattern = hooks.needs_point(points.pattern().info());
        let materialize = inspect_scores || inspect_pattern || transform.is_some();
        let inspect_projection = [points.query(), points.key(), points.value()]
            .into_iter()
            .any(|point| hooks.needs_point(point.info()));
        let inspect_context = hooks.needs_point(points.head_context().info());
        if !inspect_projection && !materialize && !inspect_context {
            return self.attend(projection, mask, bias, causal);
        }
        if materialize && self.precision_mode != AttentionPrecisionMode::EagerFp32Softmax {
            return Err(Error::BackendCapability(
                "attention score/pattern access and structural pattern transforms require AttentionPrecisionMode::EagerFp32Softmax; backend-native kernels have no portable intermediate score graph"
                    .into(),
            ));
        }
        validate_projection(self, &projection, mask.as_ref(), bias.as_ref())?;
        projection.query = hooks.apply(points.query(), projection.query)?;
        projection.key = hooks.apply(points.key(), projection.key)?;
        projection.value = hooks.apply(points.value(), projection.value)?;
        if !materialize && !inspect_context {
            return self.attend(projection, mask, bias, causal);
        }
        let AttentionProjection {
            query,
            key,
            value,
            gate,
        } = projection;
        let operands = Operands {
            query,
            key,
            value,
            mask,
            bias,
            sinks: None,
        };
        let context = if materialize {
            self.materialized_context(operands, causal, points, hooks, transform)?
        } else {
            self.head_context(operands, causal)?
        };
        let context = hooks.apply(points.head_context(), context)?;
        self.project_heads(context, gate)
    }

    /// Evaluates per-head context `[batch, query_head, query, head_dim]` with the configured graph.
    fn head_context(&self, operands: Operands<B>, causal: bool) -> Result<Tensor<B, 4>> {
        dispatch::evaluate(operands, self.module_options(causal), self.precision_mode)
    }

    /// Evaluates the eager graph while exposing scores and patterns to hooks.
    ///
    /// Scores and patterns are per query head even though the matmuls run on
    /// folded native KV heads. A structural transform runs before pattern
    /// hooks; exclusions are re-applied after the transform and every edit.
    fn materialized_context(
        &self,
        operands: Operands<B>,
        causal: bool,
        points: AttentionPoints,
        hooks: &mut Hooks<'_, B>,
        transform: Option<&mut PatternTransform<'_, B>>,
    ) -> Result<Tensor<B, 4>> {
        let geometry = operands.geometry()?;
        let Operands {
            query,
            key,
            value,
            mask,
            bias,
            sinks,
        } = operands;
        let device = query.device();
        let scale = self.scale.unwrap_or_else(|| default_scale(self.head_dim));
        let scores = hooks.apply(
            points.scores(),
            scaled_scores(query, key, scale, self.softcap),
        )?;
        let exclusion = union_exclusions(mask, causal_exclusion(causal, geometry, &device)?);
        let pattern = eager_probabilities(scores, exclusion.as_ref(), bias, sinks);
        let pattern = match transform {
            Some(transform) => apply_pattern_transform(transform, pattern, exclusion.as_ref())?,
            None => pattern,
        };
        let pattern = hooks.apply_constrained(points.pattern(), pattern, |pattern| {
            Ok(zero_excluded(pattern, exclusion.as_ref()))
        })?;
        Ok(grouped_mix(pattern, value))
    }

    /// Applies the output gate and projection to per-head context `[batch, query_head, query, head_dim]`.
    fn project_heads(
        &self,
        context: Tensor<B, 4>,
        gate: Option<Tensor<B, 3>>,
    ) -> Result<Tensor<B, 3>> {
        let [batch, _, query_sequence, _] = context.dims();
        let context = context.swap_dims(1, 2).reshape([
            batch,
            query_sequence,
            self.query_heads * self.head_dim,
        ]);
        operations::linear(
            apply_output_gate(context, gate, self.output_gate)?,
            &self.o_proj,
        )
    }

    /// Returns Burn attention options for this module and causal flag.
    fn module_options(&self, causal: bool) -> AttentionModuleOptions {
        AttentionModuleOptions {
            scale: self.scale,
            softcap: self.softcap,
            is_causal: causal,
        }
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

/// Multiplies flattened head outputs by the configured output gate.
///
/// `Sigmoid` computes `context * sigmoid(gate)` and `Silu` computes
/// `context * gate * sigmoid(gate)`; the gate must be present exactly when a
/// gating convention is configured.
fn apply_output_gate<B: Backend>(
    context: Tensor<B, 3>,
    gate: Option<Tensor<B, 3>>,
    kind: AttentionOutputGate,
) -> Result<Tensor<B, 3>> {
    match (kind, gate) {
        (AttentionOutputGate::None, None) => Ok(context),
        (AttentionOutputGate::Sigmoid, Some(gate)) => Ok(context * sigmoid(gate)),
        (AttentionOutputGate::Silu, Some(gate)) => Ok(context * silu(gate)),
        _ => Err(Error::shape(
            "attention.gate",
            format!("a gate matching {kind:?}"),
            "a mismatched gate payload",
        )),
    }
}

/// Runs a structural pattern transform and checks that it preserves the pattern contract.
///
/// The transform's result is re-masked so it cannot reactivate excluded links.
fn apply_pattern_transform<B: Backend>(
    transform: &mut PatternTransform<'_, B>,
    pattern: Tensor<B, 4>,
    exclusion: Option<&Tensor<B, 4, Bool>>,
) -> Result<Tensor<B, 4>> {
    let shape = pattern.dims();
    let dtype = pattern.dtype();
    let device = pattern.device();
    let pattern = transform(pattern).map_err(|error| {
        Error::configuration(
            "attention.pattern_transform",
            format!("structural transform failed: {error}"),
        )
    })?;
    if pattern.dims() != shape || pattern.dtype() != dtype || pattern.device() != device {
        return Err(Error::configuration(
            "attention.pattern_transform",
            "structural transform must preserve pattern shape, dtype, and device",
        ));
    }
    Ok(zero_excluded(pattern, exclusion))
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

    let dtype = projection.query.dtype();
    let device = projection.query.device();
    if matches!(dtype, DType::QFloat(_)) {
        return Err(Error::BackendCapability(
            "attention projections must be dense floating tensors; quantized projection weights remain supported".into(),
        ));
    }
    let validate_type = |field: &str, actual_dtype: DType, actual_device: &B::Device| {
        if actual_dtype != dtype || actual_device != &device {
            return Err(Error::configuration(
                field,
                format!(
                    "must share query dtype {dtype:?} and device {device:?}; found {actual_dtype:?} on {actual_device:?}"
                ),
            ));
        }
        Ok(())
    };
    validate_type(
        "attention.key",
        projection.key.dtype(),
        &projection.key.device(),
    )?;
    validate_type(
        "attention.value",
        projection.value.dtype(),
        &projection.value.device(),
    )?;
    if let Some(gate) = &projection.gate {
        validate_type("attention.gate", gate.dtype(), &gate.device())?;
    }
    if let Some(mask) = mask
        && mask.device() != device
    {
        return Err(Error::configuration(
            "attention.mask",
            "mask must share the query device",
        ));
    }
    if let Some(bias) = bias {
        validate_type("attention.bias", bias.dtype(), &bias.device())?;
    }
    let output_weight = attention.o_proj.weight.val();
    if output_weight.device() != device
        || (!matches!(output_weight.dtype(), DType::QFloat(_)) && output_weight.dtype() != dtype)
    {
        return Err(Error::configuration(
            "attention.output_projection",
            "output projection must share the query device and, for dense weights, its dtype",
        ));
    }
    if let Some(bias) = &attention.o_proj.bias {
        let bias = bias.val();
        validate_type(
            "attention.output_projection.bias",
            bias.dtype(),
            &bias.device(),
        )?;
    }
    Ok(())
}

/// Converts semantic valid-token masks into Burn's `true == excluded` mask.
///
/// Query masks are `[batch, query]` and key masks `[batch, key]`, both with
/// `true == valid`. The result is `[batch, heads, query, key]`: one Boolean
/// `[batch, 1, query, key]` plane expanded (not copied) across heads, or
/// `None` when neither mask is supplied.
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
    Ok(
        padding_plane(query_mask, key_mask, batch, query_sequence, key_sequence)?
            .map(|plane| plane.expand([batch, heads, query_sequence, key_sequence])),
    )
}

#[cfg(test)]
#[path = "attention_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "attention_intervention_tests.rs"]
mod intervention_tests;
