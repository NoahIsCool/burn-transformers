//! Executable Muse-Glimmer dense hybrid decoder.
//!
//! A decoder layer applies a four-norm "sandwich" residual layout:
//! `x' = x + PostAttnNorm(Attention(InputNorm(x)))` and
//! `y = x' + PostFfnNorm(SwiGLU(PreFfnNorm(x')))`. All four layer norms use
//! zero-centered checkpoint scales, `scale = 1 + weight`; the two post norms
//! intentionally use the smaller `post_norm_eps` stabilizer. Token embeddings
//! are RMS-normalized without a learned scale, while the final norm stores an
//! ordinary (non-centered) learned scale.
//!
//! # Attention
//!
//! Attention is grouped-query attention with a distinct gate projection. After
//! scale-free per-head Q/K normalization, queries are multiplied by
//! `qk_scale_factor`; dot products still receive the standard `1/sqrt(d)`
//! factor. Query heads attend to the *native* key/value heads: groups fold into
//! query rows ([`grouped_query_attention`]), so the two cached KV heads are
//! never repeated sixteen times. Muse's eager reference materializes
//! reduced-precision scores, evaluates softmax in FP32, then restores the query
//! dtype; Muse uses that explicit eager graph for F16/BF16 and backend dispatch
//! for F32. The attention result is multiplied by `sigmoid(gate_proj(h))`
//! before the output projection. Three local RoPE layers normally precede each
//! global NoPE layer. Finally, logits follow the exact bounded map
//! `T * tanh(output_multiplier * raw_logits / T)`.
//!
//! # Bounded execution
//!
//! Every call evaluates synchronized query chunks planned by the model's
//! [`AttentionExecutionPolicy`]. Each chunk slices the keys its rows may see:
//! the causal prefix for global layers (keys after the chunk's last row are
//! never scored), the union of row windows for unpadded local layers, and,
//! for cached local layers over `[ring history | fresh block]`, the window
//! union once no chunk row can see history (the whole ring before that; see
//! `schedule::ring_chunk_keys`). Every score, mask, and probability plane
//! therefore stays within the policy's score bound, including single-chunk
//! calls. Masks are built from
//! token positions (valid-token ordinals with padding, so holes consume no
//! window capacity), one `[batch, 1, rows, keys]` plane per chunk. Autodiff
//! evaluates one call instead, because its tape retains every chunk anyway.
//!
//! # Positions
//!
//! RoPE metadata goes through the shared translator
//! ([`RopeParameters::to_rope_config`](crate::models::config::RopeParameters::to_rope_config)).
//! The reference rotates every RoPE layer with the global
//! `rope_parameters.rope_theta` and reads `layer_rope_theta` only as a
//! RoPE/NoPE switch; the crate additionally honors explicit per-layer thetas
//! as an experiment extension, which equals the reference whenever every
//! nonzero entry equals the global theta (as in every published config). Whole
//! heads always rotate, so a `partial_rotary_factor` other than one is
//! rejected. Cosine/sine tables are built once per forward for each distinct
//! rotary configuration and shared by its layers.

use core::ops::Range;

use burn::{
    module::Module,
    nn::{Embedding, EmbeddingConfig, Linear, LinearConfig, LinearLayout},
    tensor::{
        Bool, DType, Int, Tensor, activation::sigmoid, backend::Backend,
        module::linear as tensor_linear, ops::AttentionModuleOptions,
    },
};

use crate::{
    ArchitectureConfig, CausalLmInput, CausalLmOutput, Error, ForwardOptions, Result,
    architecture::TokenMask,
    attention::{
        AttentionExecutionPlan, AttentionExecutionPolicy, AttentionPrecisionMode,
        dispatch_attention, grouped_query_attention, sliding_window_key_range,
    },
    feed_forward::{GatedFeedForward, GatedFeedForwardConfig},
    intervention::{Axis, Hooks, Layer, Point, PointInfo, Stack},
    mla::PositionalVisibility,
    normalization::{RmsNorm, RmsNormConfig, RmsNormRootMode, scale_free_rms_norm},
    operations::linear,
    position::{
        RopeConfig, RotaryAngles, RotaryFrequencies, apply_rotary, rotary_angles_from_frequencies,
    },
};

use super::{
    MuseGlimmerLayerType, MuseGlimmerTextConfig,
    schedule::{KeyRule, plane_elements, planning_bound, prefix_chunk_limit, ring_chunk_keys},
};
use crate::models::{
    config::{PartialRotary, RopeContext},
    shared::{
        InputContract, InputNames, ValidatedInput, ensure_same_device, select_hidden,
        validate_input,
    },
};

mod decode;
/// Activation hooks and staged execution (`HookedModel`, `StagedModel`).
mod hooks;

pub use decode::{MuseGlimmerDecodeState, MuseGlimmerLayerCache};

/// Tensor names reported by Muse-Glimmer input errors.
const INPUT_NAMES: InputNames<'static> = InputNames {
    token_ids: "muse_glimmer.input_ids",
    token_mask: "muse_glimmer.token_mask",
    position_ids: "muse_glimmer.position_ids",
};

/// Owner named when a stateless tensor is on another device than the weights.
const PARAMETERS: &str = "Muse-Glimmer model parameters";

/// Projected Q/K/V and the independent Muse output gate.
#[derive(Clone, Debug)]
struct MuseAttentionProjection<B: Backend> {
    /// Queries shaped `[batch, query_heads, query_sequence, head_dim]`.
    query: Tensor<B, 4>,
    /// Native keys shaped `[batch, key_value_heads, key_sequence, head_dim]`.
    key: Tensor<B, 4>,
    /// Native values shaped `[batch, key_value_heads, key_sequence, head_dim]`.
    value: Tensor<B, 4>,
    /// Unactivated gate shaped `[batch, query_sequence, query_heads * head_dim]`.
    gate: Tensor<B, 3>,
}

/// Muse-Glimmer grouped-query attention with scale-free Q/K norm and a separate gate.
#[derive(Module, Debug)]
pub struct MuseGlimmerAttention<B: Backend> {
    /// Query projection without a fused gate payload.
    pub q_proj: Linear<B>,
    /// Native key-head projection.
    pub k_proj: Linear<B>,
    /// Native value-head projection.
    pub v_proj: Linear<B>,
    /// Independent attention-output gate projection.
    pub gate_proj: Linear<B>,
    /// Projection from concatenated query heads to the residual width.
    pub o_proj: Linear<B>,
    /// Residual-stream width accepted by every input projection.
    pub hidden_size: usize,
    /// Number of independently projected query heads.
    pub query_heads: usize,
    /// Number of native key/value heads retained in the cache.
    pub key_value_heads: usize,
    /// Scalar width of each attention head.
    pub head_dim: usize,
    /// Stabilizer for weightless per-head RMS normalization.
    pub qk_norm_epsilon: f64,
    /// Multiplier applied to normalized queries before dot products.
    pub qk_scale_factor: f64,
    /// Peak-workspace policy for score, mask, and probability planes.
    #[module(skip)]
    execution_policy: AttentionExecutionPolicy,
}

impl<B: Backend> MuseGlimmerAttention<B> {
    /// Initializes exact checkpoint-shaped attention projections.
    fn init(config: &MuseGlimmerTextConfig, device: &B::Device) -> Result<Self> {
        let query_width = checked_product(
            "muse_glimmer.attention.query_width",
            config.num_attention_heads,
            config.head_dim,
        )?;
        let key_value_width = checked_product(
            "muse_glimmer.attention.key_value_width",
            config.num_key_value_heads,
            config.head_dim,
        )?;
        let projected = |input, output, bias| {
            crate::operations::init_linear(
                &LinearConfig::new(input, output)
                    .with_bias(bias)
                    .with_layout(LinearLayout::Col),
                device,
            )
        };
        Ok(Self {
            q_proj: projected(config.hidden_size, query_width, config.attention_bias),
            k_proj: projected(config.hidden_size, key_value_width, config.attention_bias),
            v_proj: projected(config.hidden_size, key_value_width, config.attention_bias),
            gate_proj: projected(config.hidden_size, query_width, false),
            o_proj: projected(query_width, config.hidden_size, config.attention_bias),
            hidden_size: config.hidden_size,
            query_heads: config.num_attention_heads,
            key_value_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            qk_norm_epsilon: config.rms_norm_eps,
            qk_scale_factor: config.qk_scale_factor,
            execution_policy: AttentionExecutionPolicy::default(),
        })
    }

    /// Projects one hidden sequence and applies Muse's scale-free Q/K normalization.
    ///
    /// Queries are `[batch, query_heads, sequence, head_dim]`, keys and values
    /// keep their native `[batch, kv_heads, sequence, head_dim]` heads, and the
    /// gate stays unactivated `[batch, sequence, query_heads * head_dim]`.
    fn project_self(&self, hidden: Tensor<B, 3>) -> Result<MuseAttentionProjection<B>> {
        let [batch, sequence, width] = hidden.dims();
        if batch == 0 || sequence == 0 || width != self.hidden_size {
            return Err(Error::shape(
                "muse_glimmer.attention_input",
                format!("[batch>0, sequence>0, {}]", self.hidden_size),
                format!("[{batch}, {sequence}, {width}]"),
            ));
        }
        let query = linear(hidden.clone(), &self.q_proj)?
            .reshape([batch, sequence, self.query_heads, self.head_dim])
            .swap_dims(1, 2);
        let key = linear(hidden.clone(), &self.k_proj)?
            .reshape([batch, sequence, self.key_value_heads, self.head_dim])
            .swap_dims(1, 2);
        let value = linear(hidden.clone(), &self.v_proj)?
            .reshape([batch, sequence, self.key_value_heads, self.head_dim])
            .swap_dims(1, 2);
        let gate = linear(hidden, &self.gate_proj)?;
        let norm = |tensor| {
            scale_free_rms_norm(
                tensor,
                self.qk_norm_epsilon,
                RmsNormRootMode::MultiplyByPower,
            )
        };
        Ok(MuseAttentionProjection {
            query: norm(query)? * self.qk_scale_factor,
            key: norm(key)?,
            value,
            gate,
        })
    }

    /// Returns the policy actually executed on `device`.
    ///
    /// An autodiff tape keeps every chunk's planes alive until backward, so
    /// chunking would cost synchronizations without bounding memory: autodiff
    /// backends evaluate one unbounded call (same values) instead of failing.
    fn effective_policy(&self, device: &B::Device) -> AttentionExecutionPolicy {
        if B::ad_enabled(device) {
            AttentionExecutionPolicy::Unbounded
        } else {
            self.execution_policy
        }
    }

    /// Evaluates bottom-right-aligned causal attention and returns the head context.
    ///
    /// Keys `K >= Q` end where the queries end (stateless self-attention has
    /// `K = Q`; a cached suffix has its history in front). A `token_mask`
    /// (`true` = valid, requires `K = Q`) excludes padded queries and keys and
    /// measures windows in valid-token ordinals. The result is
    /// `[batch, query_heads, query, head_dim]`, before gating.
    fn attend_aligned(
        &self,
        projection: &MuseAttentionProjection<B>,
        token_mask: Option<&TokenMask<B>>,
        window: Option<usize>,
    ) -> Result<Tensor<B, 4>> {
        let [batch, query_sequence, key_sequence] = self.validate_projection(projection)?;
        let device = projection.query.device();
        let visibility = match token_mask {
            Some(mask) => {
                if query_sequence != key_sequence {
                    return Err(Error::shape(
                        "muse_glimmer.stateless_attention",
                        "equal query and key sequence lengths with a token mask",
                        format!("query={query_sequence}, key={key_sequence}"),
                    ));
                }
                PositionalVisibility::causal_ordinals(query_sequence, Some(mask), &device)?
            }
            None => PositionalVisibility::suffix(query_sequence, key_sequence, &device)?,
        }
        .with_window(window);
        let policy = self.effective_policy(&device);
        // Column-indexed window slicing is exact only without padding holes.
        let slicing = MuseKeySlicing::Aligned {
            window: if token_mask.is_none() { window } else { None },
        };
        let chunks = plan_attention_chunks(
            policy,
            batch,
            self.query_heads,
            query_sequence,
            key_sequence,
            slicing,
        )?;
        self.chunked_context(projection, &chunks, &visibility)
    }

    /// Evaluates every planned chunk against its key slice and joins the contexts.
    ///
    /// Each chunk materializes exactly one `[batch, query_heads, rows, keys]`
    /// mask, score, and probability plane, with keys and values sliced from
    /// their native heads. Multi-chunk execution synchronizes the backend
    /// between chunks so dropped workspaces are reclaimed.
    fn chunked_context(
        &self,
        projection: &MuseAttentionProjection<B>,
        chunks: &[MuseAttentionChunk],
        visibility: &PositionalVisibility<B>,
    ) -> Result<Tensor<B, 4>> {
        let [batch, _, _] = self.validate_projection(projection)?;
        let device = projection.query.device();
        let precision = muse_attention_precision(projection.query.dtype());
        let chunked = chunks.len() > 1;
        let mut contexts = Vec::with_capacity(chunks.len());
        for chunk in chunks {
            let rows = chunk.query_range.clone();
            let keys = chunk.key_range.clone();
            let mask = visibility.exclusion(batch, self.query_heads, rows.clone(), keys.clone());
            // `dispatch_attention` is the crate-internal entry of
            // `grouped_query_attention`: native KV heads fold into query rows.
            contexts.push(dispatch_attention(
                projection.query.clone().narrow(2, rows.start, rows.len()),
                projection.key.clone().narrow(2, keys.start, keys.len()),
                projection.value.clone().narrow(2, keys.start, keys.len()),
                Some(mask),
                None,
                muse_attention_options(false),
                precision,
            )?);
            if chunked {
                B::sync(&device).map_err(|error| {
                    Error::BackendCapability(format!(
                        "failed to synchronize a bounded Muse attention chunk: {error}"
                    ))
                })?;
                B::memory_cleanup(&device);
            }
        }
        Ok(Tensor::cat(contexts, 2))
    }

    /// Evaluates a cached suffix (history in front of the queries) and gates the result.
    ///
    /// Equivalent to [`Self::attend_aligned`] without padding, followed by the
    /// output gate and projection.
    #[cfg_attr(not(test), allow(dead_code, reason = "Metal stress fixtures call it"))]
    fn attend_cached(
        &self,
        projection: MuseAttentionProjection<B>,
        window: Option<usize>,
    ) -> Result<Tensor<B, 3>> {
        let context = self.attend_aligned(&projection, None, window)?;
        self.gated_output(context, projection.gate, None)
    }

    /// Runs one unchunked attention call with an explicit exclusion mask.
    ///
    /// `mask` is `[batch, query_heads, query, key]` with `true` = excluded;
    /// `causal` adds bottom-right causality. Returns the head context.
    #[cfg_attr(not(test), allow(dead_code, reason = "oracle fixtures call it"))]
    fn attend_unbounded(
        &self,
        projection: &MuseAttentionProjection<B>,
        mask: Option<Tensor<B, 4, Bool>>,
        causal: bool,
    ) -> Result<Tensor<B, 4>> {
        let [batch, query_sequence, key_sequence] = self.validate_projection(projection)?;
        validate_materialized_mask(
            mask.as_ref(),
            batch,
            self.query_heads,
            query_sequence,
            key_sequence,
        )?;
        grouped_query_attention(
            projection.query.clone(),
            projection.key.clone(),
            projection.value.clone(),
            mask,
            None,
            muse_attention_options(causal),
            muse_attention_precision(projection.query.dtype()),
        )
    }

    /// Applies `sigmoid(gate)` to the head context and projects to the residual width.
    ///
    /// `context` is `[batch, query_heads, query, head_dim]` and `gate` the
    /// unactivated `[batch, query, query_heads * head_dim]` projection. With
    /// hooks, the head context, gate logits, and activated gate are emitted.
    fn gated_output(
        &self,
        context: Tensor<B, 4>,
        gate: Tensor<B, 3>,
        hooks: Option<(Layer, &mut Hooks<'_, B>)>,
    ) -> Result<Tensor<B, 3>> {
        let [batch, _, query_sequence, _] = context.dims();
        let width = self.query_heads * self.head_dim;
        let (context, gate) = match hooks {
            Some((layer, hooks)) => {
                let context = hooks.apply(layer.self_attention().head_context(), context)?;
                let logits = hooks.apply(gate_logits_point(layer), gate)?;
                (context, hooks.apply(gate_point(layer), sigmoid(logits))?)
            }
            None => (context, sigmoid(gate)),
        };
        let context = context
            .swap_dims(1, 2)
            .reshape([batch, query_sequence, width]);
        linear(context * gate, &self.o_proj)
    }

    /// Validates projected tensor topology and returns batch/query/key lengths.
    fn validate_projection(&self, projection: &MuseAttentionProjection<B>) -> Result<[usize; 3]> {
        let [batch, query_heads, query_sequence, query_dim] = projection.query.dims();
        let [key_batch, key_value_heads, key_sequence, key_dim] = projection.key.dims();
        let [value_batch, value_heads, value_sequence, value_dim] = projection.value.dims();
        let query_width = self.query_heads * self.head_dim;
        if query_heads != self.query_heads
            || query_dim != self.head_dim
            || key_batch != batch
            || value_batch != batch
            || key_value_heads != self.key_value_heads
            || value_heads != self.key_value_heads
            || key_sequence == 0
            || value_sequence != key_sequence
            || key_dim != self.head_dim
            || value_dim != self.head_dim
            || projection.gate.dims() != [batch, query_sequence, query_width]
        {
            return Err(Error::shape(
                "muse_glimmer.attention_projection",
                "compatible Q/K/V heads and a [batch, query, query_width] gate",
                format!(
                    "q={:?}, k={:?}, v={:?}, gate={:?}",
                    projection.query.dims(),
                    projection.key.dims(),
                    projection.value.dims(),
                    projection.gate.dims()
                ),
            ));
        }
        Ok([batch, query_sequence, key_sequence])
    }
}

/// Unactivated gate `[batch, token, query_heads * head_dim]` of a layer's attention.
fn gate_logits_point(layer: Layer) -> Point<3> {
    Point::custom(
        format!("decoder.layers.{}.attention.gate_logits", layer.index()),
        [Axis::Batch, Axis::Token, Axis::Channel],
    )
}

/// Activated gate `sigmoid(gate_logits)` that multiplies the head context.
fn gate_point(layer: Layer) -> Point<3> {
    Point::custom(
        format!("decoder.layers.{}.attention.gate", layer.index()),
        [Axis::Batch, Axis::Token, Axis::Channel],
    )
}

/// Output of the post-attention sandwich norm, before its residual addition.
fn post_attention_norm_point(layer: Layer) -> Point<3> {
    Point::custom(
        format!("decoder.layers.{}.post_attention_norm", layer.index()),
        [Axis::Batch, Axis::Token, Axis::Channel],
    )
}

/// Output of the post-feed-forward sandwich norm, before its residual addition.
fn post_feedforward_norm_point(layer: Layer) -> Point<3> {
    Point::custom(
        format!("decoder.layers.{}.post_feedforward_norm", layer.index()),
        [Axis::Batch, Axis::Token, Axis::Channel],
    )
}

/// One fully audited Muse query chunk before any score-like tensor is allocated.
///
/// Scores, causal or padding masks, the F32 softmax workspace, and the rounded
/// probability tensor all share `plane_shape`. Retaining that geometry beside
/// the exact query/key ranges lets tests audit the same schedule consumed by
/// production execution rather than reconstructing a parallel approximation.
#[derive(Clone, Debug, PartialEq, Eq)]
struct MuseAttentionChunk {
    /// Absolute query rows sliced from the projected query tensor.
    query_range: Range<usize>,
    /// Absolute key rows sliced from the projected key and value tensors.
    key_range: Range<usize>,
    /// Physical `[batch, query_heads, query_rows, key_rows]` plane shape.
    plane_shape: [usize; 4],
    /// Product of `plane_shape`, checked before any backend tensor allocation.
    plane_elements: usize,
}

/// Key geometry of one Muse attention call, which fixes every chunk's key slice.
///
/// Every variant has queries `0..Q` and keys `0..K` with `K >= Q`; the chunk
/// planner validates each slice against the plan's bound and the policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MuseKeySlicing<'a> {
    /// Keys end where the queries end: query row `i` sits at key column
    /// `K - Q + i` (stateless self-attention, or an unpadded cached suffix).
    /// With a column-indexed `window`, a chunk reads the union of its rows'
    /// windows ([`aligned_key_range`]); without one, its causal prefix
    /// `0..K - Q + end`.
    Aligned {
        /// Local window in key columns, or `None` for causal-prefix slicing.
        window: Option<usize>,
    },
    /// A cached sliding layer: `K - Q` ring slots in *physical* (not
    /// chronological) order followed by the `Q` fresh block entries, so fresh
    /// entry `k` of every row is key column `K - Q + k`. A chunk reads the
    /// whole ring only while one of its rows can still see history, and
    /// otherwise exactly its rows' window union (the Verus-verified
    /// [`ring_chunk_keys`]).
    RingHistory {
        /// Local window `W > 0` in positions.
        window: usize,
    },
    /// Position-ordered cached keys with verified causal prefix bounds
    /// (full-storage cache reads).
    PrefixLimits {
        /// Nondecreasing bounds: query rows `0..=i` need only keys `0..limits[i]`.
        limits: &'a [usize],
    },
}

impl MuseKeySlicing<'_> {
    /// Returns the integer rule that fixes this slicing's key bound.
    const fn rule(self) -> KeyRule {
        match self {
            Self::Aligned {
                window: Some(window),
            } => KeyRule::AlignedWindow { window },
            Self::Aligned { window: None } => KeyRule::AlignedPrefix,
            Self::RingHistory { window } => KeyRule::Ring { window },
            Self::PrefixLimits { .. } => KeyRule::Limits,
        }
    }

    /// Returns the widest key slice of any chunk of at most `maximum_query` rows.
    ///
    /// The policy sizes its chunks with this bound, and the planner rejects
    /// any realized slice that exceeds it, so the bound is never trusted
    /// blindly. Aligned windows need `W + q - 1` keys, ring chunks at most
    /// `(K - Q) + W - 1 + q`, and causal prefixes up to all `K` keys. The
    /// formula is [`planning_bound`], whose proof also shows that no ring or
    /// aligned slice of an admitted chunk can exceed it.
    fn planning_bound(self, queries: usize, keys: usize, maximum_query: usize) -> usize {
        planning_bound(self.rule(), queries, keys, maximum_query)
    }

    /// Returns the key slice that query rows `rows` (non-empty, in `0..Q`) may see.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Configuration`] for a zero window, an empty or
    /// out-of-range query range, or prefix limits that do not cover `Q` rows
    /// within `1..=K`, and [`Error::Shape`] when `K < Q`.
    fn key_range(self, queries: usize, keys: usize, rows: Range<usize>) -> Result<Range<usize>> {
        let offset = keys.checked_sub(queries).ok_or_else(|| {
            Error::shape(
                "muse_glimmer.attention_alignment",
                "key sequence at least as long as query sequence",
                format!("query={queries}, key={keys}"),
            )
        })?;
        if rows.is_empty() || rows.end > queries {
            return Err(Error::configuration(
                "muse_glimmer.attention_chunk",
                "query range must be non-empty and in bounds",
            ));
        }
        match self {
            Self::Aligned {
                window: Some(window),
            } => aligned_key_range(queries, keys, rows, window),
            // Keys at or after `K - Q + end` follow every query of the chunk.
            Self::Aligned { window: None } => Ok(0..offset + rows.end),
            Self::RingHistory { window } => (window > 0)
                .then(|| ring_chunk_keys(offset, rows.start, rows.end, window))
                .flatten()
                .map(|(start, end)| start..end)
                .ok_or_else(|| {
                    Error::configuration(
                        "muse_glimmer.attention_chunk",
                        "a ring window must be non-zero",
                    )
                }),
            // Verified: the chunk's last row's limit, when the limits cover
            // every query row within `1..=K`.
            Self::PrefixLimits { limits } => prefix_chunk_limit(limits, queries, keys, rows.end)
                .map(|limit| 0..limit)
                .ok_or_else(|| {
                    Error::configuration(
                        "muse_glimmer.attention_chunk",
                        format!(
                            "{} key limits must cover {queries} query rows within 1..={keys}",
                            limits.len()
                        ),
                    )
                }),
        }
    }
}

/// Plans the policy's query chunks for one call and expands them with `slicing`.
///
/// The plan is sized with [`MuseKeySlicing::planning_bound`] for the
/// policy's largest chunk, then [`plan_muse_attention_chunks`] realizes and
/// audits every chunk against it.
fn plan_attention_chunks(
    policy: AttentionExecutionPolicy,
    batch: usize,
    heads: usize,
    queries: usize,
    keys: usize,
    slicing: MuseKeySlicing<'_>,
) -> Result<Vec<MuseAttentionChunk>> {
    let bound = slicing.planning_bound(queries, keys, policy.maximum_query_tokens(queries));
    let plan = policy.plan(batch, heads, queries, bound)?;
    plan_muse_attention_chunks(policy, plan, batch, heads, queries, keys, slicing)
}

/// Expands one policy plan into every concrete, trimmed Muse chunk.
///
/// Muse attention is always causal with queries bottom-right aligned to the
/// keys (`K >= Q`). Query chunks come from [`AttentionExecutionPlan::query_ranges`],
/// and each chunk's keys from `slicing` ([`MuseKeySlicing::key_range`]):
/// global attention scores only the causal prefix of the chunk's last row,
/// and local attention only the union of its rows' windows. The function
/// rejects any concrete key slice wider than the bound used to choose the
/// chunk size and validates every realized plane against the policy, so a
/// successful return proves that each score, mask, and probability plane is
/// bounded.
#[allow(clippy::too_many_arguments)]
fn plan_muse_attention_chunks(
    policy: AttentionExecutionPolicy,
    plan: AttentionExecutionPlan,
    batch: usize,
    heads: usize,
    query_sequence: usize,
    key_sequence: usize,
    slicing: MuseKeySlicing<'_>,
) -> Result<Vec<MuseAttentionChunk>> {
    if query_sequence != plan.query_sequence() {
        return Err(Error::configuration(
            "muse_glimmer.attention_plan",
            format!(
                "plan covers {} query rows but execution requested {query_sequence}",
                plan.query_sequence()
            ),
        ));
    }
    if key_sequence == 0 {
        return Err(Error::configuration(
            "muse_glimmer.attention_plan",
            "key sequence must be non-zero",
        ));
    }
    if key_sequence < query_sequence {
        return Err(Error::shape(
            "muse_glimmer.attention_alignment",
            "key sequence at least as long as query sequence",
            format!("query={query_sequence}, key={key_sequence}"),
        ));
    }
    let mut chunks = Vec::with_capacity(plan.chunk_count());
    for query_range in plan.query_ranges() {
        let key_range = slicing.key_range(query_sequence, key_sequence, query_range.clone())?;
        if key_range.len() > plan.key_sequence_bound() {
            return Err(Error::BackendCapability(format!(
                "Muse attention key union {} exceeds planned bound {} for query range {query_range:?}",
                key_range.len(),
                plan.key_sequence_bound()
            )));
        }
        policy.validate_chunk(batch, heads, query_range.len(), key_range.len())?;
        let plane_shape = [batch, heads, query_range.len(), key_range.len()];
        // Verified: exactly the product of the four extents, or `None` on overflow.
        let plane_elements = plane_elements(batch, heads, query_range.len(), key_range.len())
            .ok_or_else(|| {
                Error::configuration(
                    "muse_glimmer.attention_plan",
                    "concrete score-plane dimensions overflow usize",
                )
            })?;
        if plane_elements > plan.peak_score_elements()
            || plan
                .score_element_limit()
                .is_some_and(|limit| plane_elements > limit)
        {
            return Err(Error::BackendCapability(format!(
                "Muse attention plane {plane_shape:?} has {plane_elements} elements, exceeding its immutable execution plan"
            )));
        }
        chunks.push(MuseAttentionChunk {
            query_range,
            key_range,
            plane_shape,
            plane_elements,
        });
    }
    if chunks.len() != plan.chunk_count() {
        return Err(Error::configuration(
            "muse_glimmer.attention_plan",
            format!(
                "plan declares {} chunks but concrete expansion produced {}",
                plan.chunk_count(),
                chunks.len()
            ),
        ));
    }
    Ok(chunks)
}

/// Builds Muse's backend attention options without disguising the default scale.
///
/// Burn defines `scale: None` as `1 / sqrt(head_dim)`, which is precisely Muse's
/// checkpoint equation. On Burn 0.21, setting the same value explicitly is not a
/// harmless representation detail: CubeCL classifies any custom scale as a
/// fallback-only feature. Keeping the scale implicit therefore preserves both
/// the reference mathematics and fused-attention eligibility.
fn muse_attention_options(causal: bool) -> AttentionModuleOptions {
    AttentionModuleOptions {
        scale: None,
        softcap: None,
        is_causal: causal,
    }
}

/// Selects the pinned eager rounding graph only when reduced precision needs it.
fn muse_attention_precision(dtype: DType) -> AttentionPrecisionMode {
    if matches!(dtype, DType::F16 | DType::BF16) {
        AttentionPrecisionMode::EagerFp32Softmax
    } else {
        AttentionPrecisionMode::BackendDefault
    }
}

/// Validates one materialized exclusion tensor against an attention slice.
fn validate_materialized_mask<B: Backend>(
    mask: Option<&Tensor<B, 4, Bool>>,
    batch: usize,
    heads: usize,
    query_sequence: usize,
    key_sequence: usize,
) -> Result<()> {
    if let Some(mask) = mask
        && mask.dims() != [batch, heads, query_sequence, key_sequence]
    {
        return Err(Error::shape(
            "muse_glimmer.attention_mask",
            format!("[{batch}, {heads}, {query_sequence}, {key_sequence}]"),
            format!("{:?}", mask.dims()),
        ));
    }
    Ok(())
}

/// Computes the union of keys visible to one aligned local-query chunk.
///
/// Query row `i` aligns with key row `i + K - Q`, as in bottom-right causal
/// attention. For a window `W`, the union over `start..end` is therefore
/// `[max(0, start + K - Q - W + 1), end + K - Q)`, computed by the shared
/// Verus-verified [`sliding_window_key_range`].
fn aligned_key_range(
    query_sequence: usize,
    key_sequence: usize,
    query_range: Range<usize>,
    window: usize,
) -> Result<Range<usize>> {
    if window == 0 || query_range.is_empty() || query_range.end > query_sequence {
        return Err(Error::configuration(
            "muse_glimmer.attention_chunk",
            "window and query range must be non-empty and in bounds",
        ));
    }
    sliding_window_key_range(query_sequence, key_sequence, query_range, window).ok_or_else(|| {
        Error::shape(
            "muse_glimmer.attention_alignment",
            "key sequence at least as long as query sequence",
            format!("query={query_sequence}, key={key_sequence}"),
        )
    })
}

/// Per-forward inputs shared by every stateless layer.
struct LayerInputs<'a, B: Backend> {
    /// Cosine/sine tables of each distinct rotary configuration.
    tables: &'a RotaryTables<B>,
    /// Padding mask (`true` = valid), absent when every token is valid.
    token_mask: Option<&'a TokenMask<B>>,
    /// Local receptive field of sliding layers.
    sliding_window: Option<usize>,
}

/// Cosine/sine tables keyed by rotary configuration, built once per forward.
struct RotaryTables<B: Backend> {
    /// `(configuration, [batch, 1, sequence, head_dim] tables)` pairs.
    entries: Vec<(RopeConfig, RotaryAngles<B>)>,
}

impl<B: Backend> RotaryTables<B> {
    /// Returns the tables of `rope`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Configuration`] when no table was built for `rope`.
    fn get(&self, rope: &RopeConfig) -> Result<&RotaryAngles<B>> {
        self.entries
            .iter()
            .find(|(candidate, _)| candidate == rope)
            .map(|(_, angles)| angles)
            .ok_or_else(|| {
                Error::configuration(
                    "muse_glimmer.rope",
                    "no rotary table was built for a layer's configuration",
                )
            })
    }
}

/// One dense Muse decoder block with sandwich normalization.
#[derive(Module, Debug)]
pub struct MuseGlimmerDecoderLayer<B: Backend> {
    /// Gated grouped-query attention.
    pub self_attn: MuseGlimmerAttention<B>,
    /// Dense SwiGLU feed-forward network.
    pub mlp: GatedFeedForward<B>,
    /// Zero-centered pre-attention norm using `rms_norm_eps`.
    pub input_layernorm: RmsNorm<B>,
    /// Zero-centered post-attention norm using `post_norm_eps`.
    pub post_attention_layernorm: RmsNorm<B>,
    /// Zero-centered pre-feed-forward norm using `rms_norm_eps`.
    pub pre_feedforward_layernorm: RmsNorm<B>,
    /// Zero-centered post-feed-forward norm using `post_norm_eps`.
    pub post_feedforward_layernorm: RmsNorm<B>,
    /// Whether the attention kernel uses a local or global causal mask.
    #[module(skip)]
    pub layer_type: MuseGlimmerLayerType,
    /// Layer-specific rotary configuration; `None` means exact NoPE.
    #[module(skip)]
    rope: Option<RopeConfig>,
}

impl<B: Backend> MuseGlimmerDecoderLayer<B> {
    /// Initializes one layer from its explicit schedule and per-layer theta.
    fn init(
        config: &MuseGlimmerTextConfig,
        layer_type: MuseGlimmerLayerType,
        theta: f64,
        device: &B::Device,
    ) -> Result<Self> {
        let pre_norm = || {
            RmsNormConfig::new(config.hidden_size)
                .with_epsilon(config.rms_norm_eps)
                .with_zero_centered(true)
                .with_root_mode(RmsNormRootMode::MultiplyByReciprocalSqrt)
                .init(device)
        };
        let post_norm = || {
            RmsNormConfig::new(config.hidden_size)
                .with_epsilon(config.post_norm_eps)
                .with_zero_centered(true)
                .with_root_mode(RmsNormRootMode::MultiplyByReciprocalSqrt)
                .init(device)
        };
        Ok(Self {
            self_attn: MuseGlimmerAttention::init(config, device)?,
            mlp: GatedFeedForwardConfig::new(config.hidden_size, config.intermediate_size)
                .init(device)?,
            input_layernorm: pre_norm()?,
            post_attention_layernorm: post_norm()?,
            pre_feedforward_layernorm: pre_norm()?,
            post_feedforward_layernorm: post_norm()?,
            layer_type,
            rope: (theta > 0.0)
                .then(|| muse_rope_config(config, theta))
                .transpose()?,
        })
    }

    /// Returns this layer's local window, or `None` for global attention.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Configuration`] for a sliding layer without a window.
    fn window(&self, sliding_window: Option<usize>) -> Result<Option<usize>> {
        match self.layer_type {
            MuseGlimmerLayerType::SlidingAttention => sliding_window.map(Some).ok_or_else(|| {
                Error::configuration(
                    "muse_glimmer.sliding_window",
                    "is required by a sliding-attention layer",
                )
            }),
            MuseGlimmerLayerType::FullAttention => Ok(None),
        }
    }

    /// Normalizes, projects, and rotates the attention input, emitting its points.
    ///
    /// `angles` are this layer's tables (ignored by NoPE layers). With hooks,
    /// the residual input, attention input, and the post-rotation query and
    /// native-head key/value are emitted in that order.
    fn attention_projection(
        &self,
        hidden: Tensor<B, 3>,
        angles: Option<&RotaryAngles<B>>,
        hooks: &mut Option<(Layer, &mut Hooks<'_, B>)>,
    ) -> Result<(Tensor<B, 3>, MuseAttentionProjection<B>)> {
        let mut residual = hidden;
        if let Some((layer, hooks)) = hooks.as_mut() {
            residual = hooks.apply(layer.residual_pre(), residual)?;
        }
        let mut normalized = self.input_layernorm.forward(residual.clone());
        if let Some((layer, hooks)) = hooks.as_mut() {
            normalized = hooks.apply(layer.attention_input(), normalized)?;
        }
        let mut projection = self.self_attn.project_self(normalized)?;
        if self.rope.is_some() {
            let angles = angles.ok_or_else(|| {
                Error::configuration("muse_glimmer.rope", "a RoPE layer needs its rotary tables")
            })?;
            (projection.query, projection.key) =
                apply_rotary(projection.query, projection.key, angles)?;
        }
        if let Some((layer, hooks)) = hooks.as_mut() {
            let points = layer.self_attention();
            projection.query = hooks.apply(points.query(), projection.query)?;
            projection.key = hooks.apply(points.key(), projection.key)?;
            projection.value = hooks.apply(points.value(), projection.value)?;
        }
        Ok((residual, projection))
    }

    /// Evaluates one stateless sandwich-normalized residual layer.
    fn forward(
        &self,
        hidden: Tensor<B, 3>,
        inputs: &LayerInputs<'_, B>,
        mut hooks: Option<(Layer, &mut Hooks<'_, B>)>,
    ) -> Result<Tensor<B, 3>> {
        let angles = self
            .rope
            .as_ref()
            .map(|rope| inputs.tables.get(rope))
            .transpose()?;
        let (residual, projection) = self.attention_projection(hidden, angles, &mut hooks)?;
        let window = self.window(inputs.sliding_window)?;
        let context = self
            .self_attn
            .attend_aligned(&projection, inputs.token_mask, window)?;
        let attended = self.self_attn.gated_output(
            context,
            projection.gate,
            hooks.as_mut().map(|(layer, hooks)| (*layer, &mut **hooks)),
        )?;
        self.finish_block(residual, attended, hooks)
    }

    /// Adds the normalized attention branch, then the normalized feed-forward branch.
    ///
    /// `attended` is the gated output projection before the post-attention
    /// norm. With hooks, emits `attention.output`, `post_attention_norm`,
    /// `mlp.input`, the gate branch of the SwiGLU, `mlp.output`,
    /// `post_feedforward_norm`, and `residual.post`.
    fn finish_block(
        &self,
        residual: Tensor<B, 3>,
        attended: Tensor<B, 3>,
        mut hooks: Option<(Layer, &mut Hooks<'_, B>)>,
    ) -> Result<Tensor<B, 3>> {
        let mut attended = attended;
        if let Some((layer, hooks)) = hooks.as_mut() {
            attended = hooks.apply(layer.attention_output(), attended)?;
        }
        let mut normalized = self.post_attention_layernorm.forward(attended);
        if let Some((layer, hooks)) = hooks.as_mut() {
            normalized = hooks.apply(post_attention_norm_point(*layer), normalized)?;
        }
        let hidden = residual + normalized;
        let mut fed_input = self.pre_feedforward_layernorm.forward(hidden.clone());
        if let Some((layer, hooks)) = hooks.as_mut() {
            fed_input = hooks.apply(layer.mlp_input(), fed_input)?;
        }
        let mut fed = match hooks.as_mut() {
            Some((layer, hooks)) => self.mlp.forward_with_hooks(fed_input, *layer, hooks)?,
            None => self.mlp.forward(fed_input)?,
        };
        if let Some((layer, hooks)) = hooks.as_mut() {
            fed = hooks.apply(layer.mlp_output(), fed)?;
        }
        let mut normalized = self.post_feedforward_layernorm.forward(fed);
        if let Some((layer, hooks)) = hooks.as_mut() {
            normalized = hooks.apply(post_feedforward_norm_point(*layer), normalized)?;
        }
        let mut output = hidden + normalized;
        if let Some((layer, hooks)) = hooks {
            output = hooks.apply(layer.residual_post(), output)?;
        }
        Ok(output)
    }

    /// Lists this layer's points in emission order (see the `hooks` module).
    fn activation_points(&self, layer: Layer) -> Vec<PointInfo> {
        let attention = layer.self_attention();
        vec![
            layer.residual_pre().info(),
            layer.attention_input().info(),
            attention.query().info(),
            attention.key().info(),
            attention.value().info(),
            attention.head_context().info(),
            gate_logits_point(layer).info(),
            gate_point(layer).info(),
            layer.attention_output().info(),
            post_attention_norm_point(layer).info(),
            layer.mlp_input().info(),
            layer.mlp_pre_activation().info(),
            layer.mlp_activation().info(),
            layer.mlp_output().info(),
            post_feedforward_norm_point(layer).info(),
            layer.residual_post().info(),
        ]
    }

    /// Returns whether this layer applies rotary positions.
    #[must_use]
    pub const fn uses_rope(&self) -> bool {
        self.rope.is_some()
    }
}

/// Muse token embedding, dense decoder stack, and final learned normalization.
#[derive(Module, Debug)]
pub struct MuseGlimmerModel<B: Backend> {
    /// Raw token embedding table; a scale-free RMS norm is applied after lookup.
    pub embed_tokens: Embedding<B>,
    /// Ordered hybrid decoder layers.
    pub layers: Vec<MuseGlimmerDecoderLayer<B>>,
    /// Conventional learned RMSNorm after the final decoder layer.
    pub norm: RmsNorm<B>,
    /// Validated family configuration excluded from parameter records.
    #[module(skip)]
    pub config: MuseGlimmerTextConfig,
}

impl<B: Backend> MuseGlimmerModel<B> {
    /// Validates a public input and resolves Muse's valid-token-ordinal positions.
    ///
    /// Token IDs, the mask, and positions are checked against the vocabulary
    /// and the declared context with a single device-to-host readback, which
    /// also reports whether the batch contains any padding.
    fn validate_public_input(&self, input: &CausalLmInput<B>) -> Result<ValidatedInput<B>> {
        validate_input(
            input,
            &self.embed_tokens.weight.lazy_device(),
            &InputContract::new(INPUT_NAMES, PARAMETERS, self.config.vocab_size)
                .with_position_limit(self.config.max_position_embeddings),
        )
    }

    /// Initializes the exact dense text topology without allocating cache state.
    pub fn init(config: MuseGlimmerTextConfig, device: &B::Device) -> Result<Self> {
        ArchitectureConfig::validate(&config)?;
        if config.attention_dropout != 0.0 {
            return Err(Error::BackendCapability(
                "Muse-Glimmer inference supports the released zero attention dropout only".into(),
            ));
        }
        let schedule = config.resolved_layer_types();
        let theta = config.resolved_layer_rope_theta();
        let layers = schedule
            .into_iter()
            .zip(theta)
            .map(|(layer_type, theta)| {
                MuseGlimmerDecoderLayer::init(&config, layer_type, theta, device)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            embed_tokens: EmbeddingConfig::new(config.vocab_size, config.hidden_size).init(device),
            layers,
            norm: RmsNormConfig::new(config.hidden_size)
                .with_epsilon(config.rms_norm_eps)
                .with_root_mode(RmsNormRootMode::MultiplyByPower)
                .init(device)?,
            config,
        })
    }

    /// Embeds tokens and applies the scale-free embedding norm.
    fn embed(&self, token_ids: Tensor<B, 2, Int>) -> Result<Tensor<B, 3>> {
        scale_free_rms_norm(
            self.embed_tokens.forward(token_ids),
            self.config.rms_norm_eps,
            RmsNormRootMode::MultiplyByPower,
        )
    }

    /// Returns each distinct rotary configuration of the stack, in layer order.
    fn distinct_ropes(&self) -> Vec<RopeConfig> {
        let mut ropes: Vec<RopeConfig> = Vec::new();
        for rope in self.layers.iter().filter_map(|layer| layer.rope) {
            if !ropes.contains(&rope) {
                ropes.push(rope);
            }
        }
        ropes
    }

    /// Builds every distinct configuration's tables once, in the activation dtype.
    ///
    /// Frequencies depend on `evaluated_context` only for context-dependent
    /// scalings (dynamic NTK, LongRoPE).
    fn rotary_tables(
        &self,
        positions: Tensor<B, 2, Int>,
        evaluated_context: usize,
    ) -> Result<RotaryTables<B>> {
        let dtype = self.embed_tokens.weight.val().dtype();
        let entries = self
            .distinct_ropes()
            .into_iter()
            .map(|rope| {
                let frequencies = RotaryFrequencies::new(rope, evaluated_context)?;
                Ok((
                    rope,
                    rotary_angles_from_frequencies(&frequencies, positions.clone())?
                        .to_dtype(dtype),
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(RotaryTables { entries })
    }

    /// Evaluates the stateless backbone and retains only requested hidden states.
    ///
    /// With hooks, the embedding, every layer point, and the final norm are
    /// emitted; the caller owns the surrounding [`Hooks::run`] scope.
    fn forward(
        &self,
        input: &CausalLmInput<B>,
        retain_hidden_states: bool,
        mut hooks: Option<&mut Hooks<'_, B>>,
    ) -> Result<(Tensor<B, 3>, Vec<Tensor<B, 3>>)> {
        let validated = self.validate_public_input(input)?;
        let positions = validated.require_positions(INPUT_NAMES.position_ids)?;
        let tables = self.rotary_tables(positions, validated.evaluated_context)?;
        // An all-valid mask excludes nothing; eliding it enables window key slicing.
        let inputs = LayerInputs {
            tables: &tables,
            token_mask: if validated.fully_valid {
                None
            } else {
                input.token_mask.as_ref()
            },
            sliding_window: self.config.resolved_sliding_window(),
        };
        let mut hidden = self.embed(input.token_ids.clone())?;
        if let Some(hooks) = hooks.as_deref_mut() {
            hidden = hooks.apply(Stack::Decoder.embedding(), hidden)?;
        }
        let mut hidden_states = Vec::with_capacity(if retain_hidden_states {
            self.layers.len() + 2
        } else {
            0
        });
        if retain_hidden_states {
            hidden_states.push(hidden.clone());
        }
        for (index, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward(
                hidden,
                &inputs,
                hooks
                    .as_deref_mut()
                    .map(|hooks| (Layer::decoder(index), hooks)),
            )?;
            if retain_hidden_states {
                hidden_states.push(hidden.clone());
            }
        }
        hidden = self.norm.forward(hidden);
        if let Some(hooks) = hooks {
            hidden = hooks.apply(Stack::Decoder.final_norm(), hidden)?;
        }
        if retain_hidden_states {
            hidden_states.push(hidden.clone());
        }
        Ok((hidden, hidden_states))
    }
}

/// Decoder-only Muse-Glimmer language model.
#[derive(Module, Debug)]
pub struct MuseGlimmerForCausalLm<B: Backend> {
    /// Dense hybrid text backbone.
    pub model: MuseGlimmerModel<B>,
    /// Independent vocabulary projection, absent only for experimental tied configs.
    pub lm_head: Option<Linear<B>>,
}

impl<B: Backend> MuseGlimmerForCausalLm<B> {
    /// Initializes a differentiable full-precision graph.
    pub fn init(config: MuseGlimmerTextConfig, device: &B::Device) -> Result<Self> {
        ArchitectureConfig::validate(&config)?;
        let hidden = config.hidden_size;
        let vocabulary = config.vocab_size;
        let tied = config.tie_word_embeddings;
        Ok(Self {
            model: MuseGlimmerModel::init(config, device)?,
            lm_head: (!tied).then(|| {
                crate::operations::init_linear(
                    &LinearConfig::new(hidden, vocabulary)
                        .with_bias(false)
                        .with_layout(LinearLayout::Col),
                    device,
                )
            }),
        })
    }

    /// Applies one validated attention-workspace policy to every decoder layer.
    ///
    /// The default is synchronized query chunking, which bounds every score
    /// plane for long prompts. Selecting [`AttentionExecutionPolicy::Unbounded`]
    /// evaluates each layer in one call. Autodiff backends always evaluate one
    /// call, because their tape retains every chunk anyway, so training works
    /// with the default policy at any sequence length.
    pub fn with_attention_execution_policy(
        mut self,
        policy: AttentionExecutionPolicy,
    ) -> Result<Self> {
        policy.validate()?;
        for layer in &mut self.model.layers {
            layer.self_attn.execution_policy = policy;
        }
        Ok(self)
    }

    /// Returns the common attention-workspace policy used by decoder layers.
    #[must_use]
    pub fn attention_execution_policy(&self) -> AttentionExecutionPolicy {
        self.model
            .layers
            .first()
            .map_or_else(AttentionExecutionPolicy::default, |layer| {
                layer.self_attn.execution_policy
            })
    }

    /// Evaluates stateless causal inference and applies Muse's final logit softcap.
    pub fn forward(
        &self,
        input: CausalLmInput<B>,
        options: &ForwardOptions,
    ) -> Result<CausalLmOutput<B>> {
        if options.attentions {
            return Err(Error::BackendCapability(
                "Muse-Glimmer attention-probability capture is not exposed by its bounded execution path"
                    .into(),
            ));
        }
        let (hidden, hidden_states) = self.model.forward(&input, options.hidden_states, None)?;
        self.finish_forward(&input, options, hidden, hidden_states)
    }

    /// Selects the requested positions and projects them to soft-capped logits.
    fn finish_forward(
        &self,
        input: &CausalLmInput<B>,
        options: &ForwardOptions,
        hidden: Tensor<B, 3>,
        hidden_states: Vec<Tensor<B, 3>>,
    ) -> Result<CausalLmOutput<B>> {
        let selected = select_hidden(hidden.clone(), input.token_mask.as_ref(), &options.logits)?;
        let logits = self.project_logits(selected)?;
        Ok(CausalLmOutput {
            logits,
            final_hidden_state: options.hidden_states.then_some(hidden),
            hidden_states,
            attentions: Vec::new(),
            routers: Vec::new(),
            load_balancing_loss: None,
        })
    }

    /// Returns the validated text configuration.
    #[must_use]
    pub const fn config(&self) -> &MuseGlimmerTextConfig {
        &self.model.config
    }

    /// Projects selected hidden states and applies `T * tanh(m * logits / T)`.
    fn project_logits(&self, hidden: Tensor<B, 3>) -> Result<Tensor<B, 3>> {
        let hidden_device = hidden.device();
        let raw = match &self.lm_head {
            Some(head) => {
                ensure_same_device(
                    "muse_glimmer.lm_head",
                    &head.weight.lazy_device(),
                    &hidden_device,
                    PARAMETERS,
                )?;
                linear(hidden, head)?
            }
            None => {
                ensure_same_device(
                    "muse_glimmer.model.embed_tokens",
                    &self.model.embed_tokens.weight.lazy_device(),
                    &hidden_device,
                    PARAMETERS,
                )?;
                tied_projection(hidden, self.model.embed_tokens.weight.val())?
            }
        };
        let cap = self.config().final_logit_softcapping;
        Ok(((raw * self.config().output_multiplier) / cap).tanh() * cap)
    }
}

/// Applies a tied embedding matrix as a vocabulary projection.
fn tied_projection<B: Backend>(
    hidden: Tensor<B, 3>,
    embedding: Tensor<B, 2>,
) -> Result<Tensor<B, 3>> {
    let [batch, sequence, hidden_width] = hidden.dims();
    let [vocabulary, embedding_width] = embedding.dims();
    if hidden_width != embedding_width || vocabulary == 0 {
        return Err(Error::shape(
            "muse_glimmer.tied_lm_head",
            format!("embedding [vocab>0, {hidden_width}]"),
            format!("[{vocabulary}, {embedding_width}]"),
        ));
    }
    Ok(tensor_linear(hidden, embedding.transpose(), None).reshape([batch, sequence, vocabulary]))
}

/// Translates one RoPE layer's configuration with the shared translator.
///
/// `theta` is the layer's nonzero `layer_rope_theta` entry, which supersedes
/// the global value (see the module documentation on positions). Whole heads
/// rotate, so a partial rotary factor other than one is an error.
fn muse_rope_config(config: &MuseGlimmerTextConfig, theta: f64) -> Result<RopeConfig> {
    config
        .rope_parameters
        .clone()
        .unwrap_or_default()
        .to_rope_config(
            &RopeContext::new(
                "muse_glimmer",
                config.head_dim,
                config.max_position_embeddings,
            )
            .with_layer_rope_theta(theta)
            .with_partial_rotary(PartialRotary::FullHeadOnly),
        )
}

/// Multiplies two shape terms without integer wraparound.
fn checked_product(field: &'static str, left: usize, right: usize) -> Result<usize> {
    left.checked_mul(right)
        .ok_or_else(|| Error::configuration(field, "dimension product overflows usize"))
}

#[cfg(test)]
#[path = "model_tests.rs"]
mod tests;
