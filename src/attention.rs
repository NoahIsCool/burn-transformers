//! Projected full, grouped-query, and multi-query attention.
//!
//! Attention maps a query to a convex combination of values using
//! `softmax(Q K^T / sqrt(d)) V`. Grouped-query attention stores fewer key/value
//! heads than query heads; each KV head serves a contiguous group of query
//! heads. Caches and every K/V intervention point keep the native KV heads.
//! Attention evaluation also keeps them: a group's queries are folded into
//! extra query rows of one call instead of repeating the KV history per query
//! head (see [`grouped_query_attention`]). One exception remains: square
//! causal prefill under [`AttentionPrecisionMode::BackendDefault`] (without
//! sinks) repeats K/V so the backend's fused `is_causal` kernel can skip
//! masked tiles. Every other call folds, including decode, cached suffixes,
//! explicit masks, sinks, and the eager graph.
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
//! chunk still visits its complete key support. [`GroupedQueryAttention::attend`]
//! evaluates complete planes, while [`GroupedQueryAttention::attend_bounded`]
//! executes a plan in synchronized query chunks (Muse-Glimmer still runs its
//! own equivalent loop).
//!
//! [`DifferentialAttention`] subtracts two softmax maps (DIFF Transformer)
//! while reusing the same grouped dispatch over native KV heads.
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
mod differential;
mod dispatch;
mod execution;
mod geometry;
mod grouping;
mod mask;

/// Largest position an integer tensor of a given dtype can hold (shared with cache masks).
pub(crate) use mask::position_limit;
mod plan;
mod rules;
mod widths;

use burn::{
    module::{Initializer, Module, Param},
    nn::{Linear, LinearConfig, LinearLayout},
    tensor::{
        Bool, DType, Tensor,
        activation::{sigmoid, silu},
        backend::Backend,
        ops::AttentionModuleOptions,
    },
};

use connectivity::ExclusionPlan;
pub use connectivity::{AttentionConnectivity, MaskGeometry, valid_token_ordinals};
pub use differential::{DifferentialAttention, DifferentialAttentionConfig, depth_lambda_init};
use dispatch::{Operands, causal_exclusion, default_scale, grouped_mix, scaled_scores};
pub(crate) use dispatch::{dispatch_attention, eager_probabilities, zero_excluded};
pub use dispatch::{grouped_query_attention, repeat_key_value};
use mask::padding_plane;
pub(crate) use mask::{bottom_right_causal_mask, union_exclusions};

pub use execution::{AttentionExecutionPlan, AttentionExecutionPolicy, sliding_window_key_range};
pub(crate) use geometry::causal_mask_needed;
use geometry::{HeadLayout, ProjectionFault, projection_fault};
use widths::{ProjectionWidths, WidthError, projection_widths};

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

/// Which of the four attention projections carry an additive bias.
///
/// Most checkpoints bias all four or none; Qwen2/2.5 and GLM-4.5 bias only the
/// query, key, and value projections.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ProjectionBias {
    /// Whether the query (and fused gate) projection has a bias.
    pub query: bool,
    /// Whether the key projection has a bias.
    pub key: bool,
    /// Whether the value projection has a bias.
    pub value: bool,
    /// Whether the output projection has a bias.
    pub output: bool,
}

impl ProjectionBias {
    /// Biases all four projections, or none of them.
    #[must_use]
    pub const fn uniform(enabled: bool) -> Self {
        Self {
            query: enabled,
            key: enabled,
            value: enabled,
            output: enabled,
        }
    }

    /// Biases the query, key, and value projections but not the output projection.
    #[must_use]
    pub const fn query_key_value_only() -> Self {
        Self {
            query: true,
            key: true,
            value: true,
            output: false,
        }
    }
}

/// Channels over which optional Q/K RMS normalization computes its statistic.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QueryKeyNormScope {
    /// One `head_dim`-wide norm applied to every head separately (Qwen3, Gemma 3).
    #[default]
    PerHead,
    /// One norm over all projected query (or key) channels before the head
    /// split, with `heads * head_dim` learned scales (OLMo 2/3).
    FullWidth,
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
    ///
    /// [`Self::projection_bias`], when present, overrides this per projection.
    pub bias: bool,
    /// Optional per-projection bias flags overriding [`Self::bias`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub projection_bias: Option<ProjectionBias>,
    /// Optional Q/K RMS normalization epsilon.
    pub query_key_norm_epsilon: Option<f64>,
    /// Channels normalized by the optional Q/K RMS norm.
    #[serde(default)]
    pub query_key_norm_scope: QueryKeyNormScope,
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
    /// Whether each query head learns an attention-sink logit (gpt-oss, StreamingLLM).
    ///
    /// The sink is an extra softmax column per head that is dropped after
    /// normalization, so attention rows sum to `1 - p_sink` and a row whose
    /// keys are all excluded yields zero context. Backend-default dispatch
    /// rejects sinks combined with a softcap, which would also cap the sink.
    #[serde(default)]
    pub attention_sinks: bool,
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
            projection_bias: None,
            query_key_norm_epsilon: None,
            query_key_norm_scope: QueryKeyNormScope::PerHead,
            query_key_norm_zero_centered: false,
            output_gate: AttentionOutputGate::None,
            scale: None,
            softcap: None,
            precision_mode: AttentionPrecisionMode::BackendDefault,
            attention_sinks: false,
        }
    }

    /// Selects an explicit score/softmax operation graph.
    pub fn with_precision_mode(mut self, precision_mode: AttentionPrecisionMode) -> Self {
        self.precision_mode = precision_mode;
        self
    }

    /// Selects the bias flag of each projection independently.
    pub fn with_projection_bias(mut self, projection_bias: ProjectionBias) -> Self {
        self.projection_bias = Some(projection_bias);
        self
    }

    /// Returns the bias flags actually used: [`Self::projection_bias`] or uniform [`Self::bias`].
    #[must_use]
    pub fn effective_projection_bias(&self) -> ProjectionBias {
        self.projection_bias
            .unwrap_or(ProjectionBias::uniform(self.bias))
    }

    /// Enables Q/K RMS normalization with `epsilon` over the chosen channels.
    pub fn with_query_key_norm(mut self, epsilon: f64, scope: QueryKeyNormScope) -> Self {
        self.query_key_norm_epsilon = Some(epsilon);
        self.query_key_norm_scope = scope;
        self
    }

    /// Enables or disables learned per-head attention-sink logits.
    pub fn with_attention_sinks(mut self, attention_sinks: bool) -> Self {
        self.attention_sinks = attention_sinks;
        self
    }

    /// Validates projection sizes, GQA grouping, and numerical options.
    pub fn validate(&self) -> Result<()> {
        self.projection_widths()?;
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
        if self.attention_sinks
            && self.softcap.is_some()
            && self.precision_mode == AttentionPrecisionMode::BackendDefault
        {
            return Err(Error::configuration(
                "attention.attention_sinks",
                "backend-default dispatch cannot combine sinks with a softcap; select EagerFp32Softmax",
            ));
        }
        Ok(())
    }

    /// Returns the verified projection widths, or why they are not representable.
    ///
    /// Success implies `key_value_heads` divides `query_heads` and that every
    /// projection matrix's element count fits in `usize` (checked with Verus).
    fn projection_widths(&self) -> Result<ProjectionWidths> {
        projection_widths(
            self.hidden_size,
            self.query_heads,
            self.key_value_heads,
            self.head_dim,
            self.output_gate != AttentionOutputGate::None,
        )
        .map_err(|error| match error {
            WidthError::EmptyAxis => Error::configuration(
                "attention",
                "hidden_size, query_heads, key_value_heads, and head_dim must be non-zero",
            ),
            WidthError::UngroupedHeads => Error::configuration(
                "attention.key_value_heads",
                "must divide query_heads exactly",
            ),
            WidthError::Overflow => Error::configuration(
                "attention",
                "projected head or matrix dimensions overflow usize",
            ),
        })
    }

    /// Initializes checkpoint-native `[out, in]` projections, optional Q/K norms, and sinks.
    ///
    /// Sink logits start at zero, i.e. one extra unit of softmax mass per row.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Result<GroupedQueryAttention<B>> {
        self.validate()?;
        let ProjectionWidths {
            query_width,
            key_value_width,
            query_output_width,
        } = self.projection_widths()?;
        let bias = self.effective_projection_bias();
        let linear = |input, output, bias| {
            operations::init_linear(
                &LinearConfig::new(input, output)
                    .with_bias(bias)
                    .with_layout(LinearLayout::Col),
                device,
            )
        };
        let query_key_norm = |epsilon, width| {
            RmsNormConfig::new(width)
                .with_epsilon(epsilon)
                .with_zero_centered(self.query_key_norm_zero_centered)
                .init(device)
        };
        let (query_norm_width, key_norm_width) = match self.query_key_norm_scope {
            QueryKeyNormScope::PerHead => (self.head_dim, self.head_dim),
            QueryKeyNormScope::FullWidth => (query_width, key_value_width),
        };
        let (query_norm, key_norm) = match self.query_key_norm_epsilon {
            Some(epsilon) => (
                Some(query_key_norm(epsilon, query_norm_width)?),
                Some(query_key_norm(epsilon, key_norm_width)?),
            ),
            None => (None, None),
        };
        let sinks = self
            .attention_sinks
            .then(|| Initializer::Zeros.init([self.query_heads], device));
        Ok(GroupedQueryAttention {
            q_proj: linear(self.hidden_size, query_output_width, bias.query),
            k_proj: linear(self.hidden_size, key_value_width, bias.key),
            v_proj: linear(self.hidden_size, key_value_width, bias.value),
            o_proj: linear(query_width, self.hidden_size, bias.output),
            query_norm,
            key_norm,
            query_key_norm_scope: self.query_key_norm_scope,
            hidden_size: self.hidden_size,
            query_heads: self.query_heads,
            key_value_heads: self.key_value_heads,
            head_dim: self.head_dim,
            output_gate: self.output_gate,
            scale: self.scale,
            softcap: self.softcap,
            precision_mode: self.precision_mode,
            sinks,
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
    /// Optional query normalization (per head or full width, see `query_key_norm_scope`).
    pub query_norm: Option<RmsNorm<B>>,
    /// Optional key normalization (per head or full width, see `query_key_norm_scope`).
    pub key_norm: Option<RmsNorm<B>>,
    /// Channels over which the optional Q/K norms compute their statistic.
    pub query_key_norm_scope: QueryKeyNormScope,
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
    /// Optional learned sink logit per query head, `[query_heads]`, in the activation dtype.
    pub sinks: Option<Param<Tensor<B, 1>>>,
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
                projected_query.reshape([batch, query_sequence, self.query_heads, self.head_dim]),
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
                projected_query.clone().narrow(3, 0, self.head_dim),
                Some(
                    projected_query
                        .narrow(3, self.head_dim, self.head_dim)
                        .reshape([batch, query_sequence, query_width]),
                ),
            )
        };
        let key = operations::linear(key_value_hidden.clone(), &self.k_proj)?.reshape([
            batch,
            key_sequence,
            self.key_value_heads,
            self.head_dim,
        ]);
        let value = operations::linear(key_value_hidden, &self.v_proj)?
            .reshape([batch, key_sequence, self.key_value_heads, self.head_dim])
            .swap_dims(1, 2);
        let query = self.normalize_heads(query, self.query_norm.as_ref());
        let key = self.normalize_heads(key, self.key_norm.as_ref());
        Ok(AttentionProjection {
            query,
            key,
            value,
            gate,
        })
    }

    /// Applies an optional Q/K norm and moves heads first: `[b, s, h, d] -> [b, h, s, d]`.
    ///
    /// Per-head norms reduce over `d`; full-width norms reduce over all `h * d`
    /// projected channels of a token before the head split.
    fn normalize_heads(&self, tensor: Tensor<B, 4>, norm: Option<&RmsNorm<B>>) -> Tensor<B, 4> {
        let [batch, sequence, heads, width] = tensor.dims();
        match (norm, self.query_key_norm_scope) {
            (None, _) => tensor.swap_dims(1, 2),
            (Some(norm), QueryKeyNormScope::PerHead) => norm.forward(tensor.swap_dims(1, 2)),
            (Some(norm), QueryKeyNormScope::FullWidth) => norm
                .forward(tensor.reshape([batch, sequence, heads * width]))
                .reshape([batch, sequence, heads, width])
                .swap_dims(1, 2),
        }
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
                sinks: self.sink_logits(),
            },
            causal,
        )?;
        self.project_heads(context, gate)
    }

    /// Evaluates attention in synchronized query chunks that bound every score plane.
    ///
    /// Each chunk evaluates complete attention rows for at most
    /// `policy`-limited query rows, so peak score, mask, and probability planes
    /// follow [`AttentionExecutionPolicy::plan`] instead of `O(query * key)`.
    /// This is a storage bound, not an approximation: results equal
    /// [`Self::attend`] with the corresponding full exclusion mask.
    ///
    /// Exclusions come from token masks (`true == valid`), `causal`, and
    /// `connectivity`, as in [`AttentionConnectivity::exclusion_mask`]; queries
    /// are the bottom-right suffix of the keys, and token masks are validated
    /// without building any full plane. Keys start at absolute position zero
    /// ([`MaskGeometry::new`]); a chunked-local layer over a cache that evicted
    /// earlier keys needs [`MaskGeometry::with_key_start`], so build its mask
    /// with [`AttentionConnectivity::exclusion_mask`] and call [`Self::attend`].
    /// A causal sliding window over
    /// unpadded keys also narrows every chunk's keys to
    /// [`sliding_window_key_range`], so local layers need only
    /// `O(chunk * (chunk + window))` score elements. Additive bias and hooks
    /// need the full plane and are not supported here; use [`Self::attend`] or
    /// [`Self::attend_with_hooks`]. The backend is synchronized between chunks
    /// so dropped workspaces can be reclaimed. Autodiff backends, whose tape
    /// would retain every chunk anyway, evaluate one unbounded call with the
    /// same values ([`AttentionExecutionPolicy::effective_on`]), exactly as the
    /// Muse and MLA attention paths do; size training memory accordingly.
    ///
    /// # Errors
    ///
    /// Returns a configuration error for an invalid `policy` (checked on every
    /// backend), and the projection, mask, and chunk-limit errors of the
    /// unbounded path.
    pub fn attend_bounded(
        &self,
        projection: AttentionProjection<B>,
        query_mask: Option<&TokenMask<B>>,
        key_mask: Option<&TokenMask<B>>,
        causal: bool,
        connectivity: AttentionConnectivity,
        policy: AttentionExecutionPolicy,
    ) -> Result<Tensor<B, 3>> {
        validate_projection(self, &projection, None, None)?;
        let [batch, _, query_sequence, _] = projection.query.dims();
        let key_sequence = projection.key.dims()[2];
        let device = projection.query.device();
        policy.validate()?;
        let policy = policy.effective_on::<B>(&device);
        let exclusions = ExclusionPlan::new(
            connectivity,
            causal,
            MaskGeometry::new(batch, self.query_heads, query_sequence, key_sequence),
            query_mask,
            key_mask,
            &device,
        )?;
        // Column positions make a causal window's key union narrow and verified.
        let window = match connectivity {
            AttentionConnectivity::SlidingWindow { window } if causal && key_mask.is_none() => {
                Some(window)
            }
            _ => None,
        };
        // `ExclusionPlan::new` validated the window as non-zero and the keys as
        // at least as long as the queries. Every planned chunk therefore has a
        // window key range within the verified bound, and `validate_chunk`
        // below accepts it (`plan::lemma_window_chunk_admitted`, which builds
        // on `plan::lemma_planned_chunk_fits_one_call`).
        let key_bound = window.map_or(key_sequence, |window| {
            plan::windowed_key_bound(
                policy.maximum_query_tokens(query_sequence),
                window,
                key_sequence,
            )
        });
        let plan = policy.plan(batch, self.query_heads, query_sequence, key_bound)?;
        // `effective_on` made the policy unbounded on autodiff backends, so a
        // multi-chunk plan here is always an inference plan.
        let chunked = plan.chunk_count() > 1;
        let AttentionProjection {
            query,
            key,
            value,
            gate,
        } = projection;
        let mut contexts = Vec::with_capacity(plan.chunk_count());
        for rows in plan.query_ranges() {
            let keys = match window {
                Some(window) => {
                    sliding_window_key_range(query_sequence, key_sequence, rows.clone(), window)
                        .ok_or_else(|| {
                            Error::configuration(
                                "attention.bounded",
                                "sliding-window chunk has no key range",
                            )
                        })?
                }
                None => 0..key_sequence,
            };
            policy.validate_chunk(batch, self.query_heads, rows.len(), keys.len())?;
            let context = dispatch::evaluate(
                Operands {
                    query: query.clone().narrow(2, rows.start, rows.len()),
                    key: key.clone().narrow(2, keys.start, keys.len()),
                    value: value.clone().narrow(2, keys.start, keys.len()),
                    mask: exclusions.chunk(rows, keys, self.query_heads)?,
                    bias: None,
                    sinks: self.sink_logits(),
                },
                self.module_options(false),
                self.precision_mode,
            )?;
            if chunked {
                B::sync(&device).map_err(|error| {
                    Error::BackendCapability(format!(
                        "failed to synchronize a bounded attention chunk: {error}"
                    ))
                })?;
                B::memory_cleanup(&device);
            }
            contexts.push(context);
        }
        self.project_heads(Tensor::cat(contexts, 2), gate)
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
    /// Patterns follow softmax and the cast back to the query dtype. With
    /// attention sinks the pattern excludes the dropped sink column, so its rows
    /// sum to `1 - p_sink`. Selecting either point materializes the full
    /// quadratic score plane; callers must provision memory for it.
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
            sinks: self.sink_logits(),
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

    /// Returns the learned sink logits, if this module has them.
    fn sink_logits(&self) -> Option<Tensor<B, 1>> {
        self.sinks.as_ref().map(Param::val)
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
    // External callers can assemble projections without going through project().
    // Preserve its non-empty contract here; in particular, an empty key axis
    // cannot define the probability distribution required by softmax attention.
    // The shape checks are the verified `geometry::projection_fault`.
    let layout = HeadLayout {
        query_heads: attention.query_heads,
        key_value_heads: attention.key_value_heads,
        head_dim: attention.head_dim,
        gated: attention.output_gate != AttentionOutputGate::None,
    };
    let fault = projection_fault(
        layout,
        projection.query.dims(),
        projection.key.dims(),
        projection.value.dims(),
        projection.gate.as_ref().map(Tensor::dims),
        mask.map(Tensor::dims),
        bias.map(Tensor::dims),
    );
    if let Some(fault) = fault {
        return Err(projection_shape_error(
            attention, projection, mask, bias, fault,
        ));
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
    if let Some(sinks) = &attention.sinks {
        let sinks = sinks.val();
        if sinks.dims() != [attention.query_heads] {
            return Err(Error::shape(
                "attention.sinks",
                format!("[{}]", attention.query_heads),
                format!("{:?}", sinks.dims()),
            ));
        }
        validate_type("attention.sinks", sinks.dtype(), &sinks.device())?;
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

/// Describes one projection shape fault with the expected and actual extents.
fn projection_shape_error<B: Backend>(
    attention: &GroupedQueryAttention<B>,
    projection: &AttentionProjection<B>,
    mask: Option<&Tensor<B, 4, Bool>>,
    bias: Option<&Tensor<B, 4>>,
    fault: ProjectionFault,
) -> Error {
    let [batch, _, query_sequence, _] = projection.query.dims();
    let key_sequence = projection.key.dims()[2];
    let expected_query = [
        batch,
        attention.query_heads,
        query_sequence,
        attention.head_dim,
    ];
    let expected_key_value = [
        batch,
        attention.key_value_heads,
        key_sequence,
        attention.head_dim,
    ];
    let expected_gate = [
        batch,
        query_sequence,
        attention.query_heads * attention.head_dim,
    ];
    let expected_additive = [batch, attention.query_heads, query_sequence, key_sequence];
    match fault {
        ProjectionFault::EmptyAxis => Error::shape(
            "attention.projection",
            "non-empty batch, query sequence, and key sequence",
            format!(
                "query={:?}, key={:?}",
                projection.query.dims(),
                projection.key.dims()
            ),
        ),
        ProjectionFault::QueryShape => Error::shape(
            "attention.query",
            format!("{:?}", expected_query),
            format!("{:?}", projection.query.dims()),
        ),
        ProjectionFault::KeyShape => Error::shape(
            "attention.key",
            format!("{:?}", expected_key_value),
            format!("{:?}", projection.key.dims()),
        ),
        ProjectionFault::ValueShape => Error::shape(
            "attention.value",
            format!("{:?}", expected_key_value),
            format!("{:?}", projection.value.dims()),
        ),
        ProjectionFault::UnexpectedGate => Error::shape(
            "attention.gate",
            "no gate for ungated attention",
            "a gate tensor",
        ),
        ProjectionFault::GateShape => Error::shape(
            "attention.gate",
            format!("{:?}", expected_gate),
            format!(
                "{:?}",
                projection
                    .gate
                    .as_ref()
                    .map(Tensor::dims)
                    .unwrap_or_default()
            ),
        ),
        ProjectionFault::MissingGate => Error::shape(
            "attention.gate",
            format!("{:?}", expected_gate),
            "no gate tensor",
        ),
        ProjectionFault::MaskShape => Error::shape(
            "attention.mask",
            format!("{:?}", expected_additive),
            format!("{:?}", mask.map(Tensor::dims).unwrap_or_default()),
        ),
        ProjectionFault::BiasShape => Error::shape(
            "attention.bias",
            format!(
                "[batch in {{1, {batch}}}, {}, {query_sequence}, {key_sequence}]",
                attention.query_heads
            ),
            format!("{:?}", bias.map(Tensor::dims).unwrap_or_default()),
        ),
    }
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
