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
//! Attention is grouped-query attention with a distinct gate projection. After
//! scale-free per-head Q/K normalization, queries are multiplied by
//! `qk_scale_factor`; dot products still receive the standard `1/sqrt(d)`
//! factor. Muse's eager reference materializes reduced-precision scores, evaluates
//! softmax in FP32, then restores the query dtype. Burn 0.21 flash attention uses
//! strict-F32 QK accumulators and therefore omits the first rounding boundary;
//! Muse uses the shared explicit eager mode for F16/BF16 and retains backend
//! dispatch for F32. Short calls remain one unchanged eager operation for oracle
//! parity. Longer calls split only the query axis, synchronize and reclaim each
//! workspace, and cap every score-like plane; sliding layers additionally slice
//! the union of locally visible keys when no padding holes are present. This is
//! the supported memory-bounded Metal path. It deliberately makes no flash claim:
//! Burn 0.21's ordinary `metal` feature selects fallback, and its opt-in flash
//! graph changes the pinned F16 score-rounding boundary. The attention result is
//! multiplied by `sigmoid(gate_proj(h))` before the output projection. Three local
//! RoPE layers normally precede each global NoPE layer. Finally, logits follow the
//! exact bounded map `T * tanh(output_multiplier * raw_logits / T)`.

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
        dispatch_attention, padding_attention_mask, repeat_key_value,
    },
    feed_forward::{GatedFeedForward, GatedFeedForwardConfig, apply_activation},
    normalization::{RmsNorm, RmsNormConfig, RmsNormRootMode},
    operations::linear,
    position::{
        DynamicNtkScaling, LinearRopeScaling, RopeConfig, RopeScaling, YarnConfig, apply_rotary,
        rotary_angles,
    },
};

use super::{MuseGlimmerLayerType, MuseGlimmerTextConfig};
use crate::models::{
    config::RopeType,
    shared::{
        InputContract, InputNames, ValidatedInput, ensure_same_device, select_hidden,
        validate_input,
    },
};

mod decode;

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
        Ok(MuseAttentionProjection {
            query: scale_free_rms_norm(query, self.qk_norm_epsilon) * self.qk_scale_factor,
            key: scale_free_rms_norm(key, self.qk_norm_epsilon),
            value,
            gate,
        })
    }

    /// Evaluates stateless self-attention without materializing a full mask.
    fn attend_stateless(
        &self,
        projection: MuseAttentionProjection<B>,
        token_mask: Option<&TokenMask<B>>,
        window: Option<usize>,
    ) -> Result<Tensor<B, 3>> {
        let [batch, query_sequence, key_sequence] = self.validate_projection(&projection)?;
        if query_sequence != key_sequence {
            return Err(Error::shape(
                "muse_glimmer.stateless_attention",
                "equal query and key sequence lengths",
                format!("query={query_sequence}, key={key_sequence}"),
            ));
        }
        let maximum_query = self.execution_policy.maximum_query_tokens(query_sequence);
        let bounded_keys = if token_mask.is_none() {
            window
                .and_then(|window| window.checked_add(maximum_query.saturating_sub(1)))
                .map_or(key_sequence, |bound| bound.min(key_sequence))
        } else {
            key_sequence
        };
        let plan =
            self.execution_policy
                .plan(batch, self.query_heads, query_sequence, bounded_keys)?;
        if plan.chunk_count() == 1 {
            let mask = stateless_attention_mask(
                token_mask,
                batch,
                self.query_heads,
                query_sequence,
                window,
                &projection.query.device(),
            )?;
            return self.attend_unbounded(projection, mask, true);
        }
        let valid = token_mask.map(|mask| mask.as_tensor().clone());
        let ordinals = match (&valid, window) {
            (Some(valid), Some(_)) => {
                Some((valid.clone().int().cumsum(1) - 1).mask_fill(valid.clone().bool_not(), 0))
            }
            _ => None,
        };
        let indexed_window = token_mask.is_none().then_some(window).flatten();
        self.attend_chunked(
            projection,
            plan,
            true,
            indexed_window,
            |query_range, key_range| {
                stateless_attention_mask_chunk(
                    valid.as_ref(),
                    ordinals.as_ref(),
                    batch,
                    self.query_heads,
                    query_range,
                    key_range,
                    window,
                )
            },
        )
    }

    /// Evaluates a compact cached suffix without a quadratic absolute-time mask.
    fn attend_cached(
        &self,
        projection: MuseAttentionProjection<B>,
        window: Option<usize>,
    ) -> Result<Tensor<B, 3>> {
        let [batch, query_sequence, key_sequence] = self.validate_projection(&projection)?;
        if batch != 1 || key_sequence < query_sequence {
            return Err(Error::shape(
                "muse_glimmer.cached_attention",
                "batch one with key sequence at least as long as query sequence",
                format!("batch={batch}, query={query_sequence}, key={key_sequence}"),
            ));
        }
        let maximum_query = self.execution_policy.maximum_query_tokens(query_sequence);
        let bounded_keys = window
            .and_then(|window| window.checked_add(maximum_query.saturating_sub(1)))
            .map_or(key_sequence, |bound| bound.min(key_sequence));
        let plan =
            self.execution_policy
                .plan(batch, self.query_heads, query_sequence, bounded_keys)?;
        if plan.chunk_count() == 1 {
            let mask = aligned_attention_mask(
                batch,
                self.query_heads,
                query_sequence,
                key_sequence,
                0..query_sequence,
                0..key_sequence,
                true,
                window,
                &projection.query.device(),
            )?;
            return self.attend_unbounded(projection, mask, false);
        }
        self.attend_chunked(projection, plan, true, window, |_, _| Ok(None))
    }

    /// Runs one unchanged full-query call for short oracle and explicit unbounded paths.
    fn attend_unbounded(
        &self,
        projection: MuseAttentionProjection<B>,
        mask: Option<Tensor<B, 4, Bool>>,
        causal: bool,
    ) -> Result<Tensor<B, 3>> {
        let [batch, _, query_sequence, _] = projection.query.dims();
        let query_width = self.query_heads * self.head_dim;
        let MuseAttentionProjection {
            query,
            key,
            value,
            gate,
        } = projection;
        let key = repeat_key_value(key, self.query_heads)?;
        let value = repeat_key_value(value, self.query_heads)?;
        let precision = muse_attention_precision(query.dtype());
        let context = dispatch_attention(
            query,
            key,
            value,
            mask,
            None,
            muse_attention_options(causal),
            precision,
        )?
        .swap_dims(1, 2)
        .reshape([batch, query_sequence, query_width]);
        linear(context * sigmoid(gate), &self.o_proj)
    }

    /// Sequentially evaluates score-bounded query chunks and reclaims each workspace.
    fn attend_chunked<Mask>(
        &self,
        projection: MuseAttentionProjection<B>,
        plan: AttentionExecutionPlan,
        causal: bool,
        indexed_window: Option<usize>,
        mut additional_mask: Mask,
    ) -> Result<Tensor<B, 3>>
    where
        Mask: FnMut(Range<usize>, Range<usize>) -> Result<Option<Tensor<B, 4, Bool>>>,
    {
        let [batch, _, query_sequence, _] = projection.query.dims();
        let key_sequence = projection.key.dims()[2];
        if B::ad_enabled(&projection.query.device()) {
            return Err(Error::BackendCapability(
                "bounded multi-chunk Muse attention is inference-only because an autodiff tape retains every chunk workspace; use an explicit unbounded policy for provisioned training"
                    .into(),
            ));
        }
        let chunks = plan_muse_attention_chunks(
            self.execution_policy,
            plan,
            batch,
            self.query_heads,
            query_sequence,
            key_sequence,
            indexed_window,
        )?;
        let MuseAttentionProjection {
            query,
            key,
            value,
            gate,
        } = projection;
        let key = repeat_key_value(key, self.query_heads)?;
        let value = repeat_key_value(value, self.query_heads)?;
        let device = query.device();
        let precision = muse_attention_precision(query.dtype());
        let mut contexts = Vec::with_capacity(plan.chunk_count());
        for chunk in chunks {
            let MuseAttentionChunk {
                query_range,
                key_range,
                plane_shape,
                plane_elements,
            } = chunk;
            debug_assert_eq!(
                plane_shape,
                [batch, self.query_heads, query_range.len(), key_range.len()]
            );
            debug_assert_eq!(plane_elements, plane_shape.into_iter().product::<usize>());
            let external = additional_mask(query_range.clone(), key_range.clone())?;
            validate_materialized_mask(
                external.as_ref(),
                batch,
                self.query_heads,
                query_range.len(),
                key_range.len(),
            )?;
            let positional = aligned_attention_mask(
                batch,
                self.query_heads,
                query_sequence,
                key_sequence,
                query_range.clone(),
                key_range.clone(),
                causal,
                indexed_window,
                &device,
            )?;
            let exclusion = match (external, positional) {
                (Some(external), Some(positional)) => Some(external.bool_or(positional)),
                (Some(external), None) => Some(external),
                (None, Some(positional)) => Some(positional),
                (None, None) => None,
            };
            let context = dispatch_attention(
                query
                    .clone()
                    .slice([0..batch, 0..self.query_heads, query_range, 0..self.head_dim]),
                key.clone().slice([
                    0..batch,
                    0..self.query_heads,
                    key_range.clone(),
                    0..self.head_dim,
                ]),
                value
                    .clone()
                    .slice([0..batch, 0..self.query_heads, key_range, 0..self.head_dim]),
                exclusion,
                None,
                muse_attention_options(false),
                precision,
            )?;
            B::sync(&device).map_err(|error| {
                Error::BackendCapability(format!(
                    "failed to synchronize a bounded Muse attention chunk: {error}"
                ))
            })?;
            B::memory_cleanup(&device);
            contexts.push(context);
        }
        let context = Tensor::cat(contexts, 2).swap_dims(1, 2).reshape([
            batch,
            query_sequence,
            self.query_heads * self.head_dim,
        ]);
        linear(context * sigmoid(gate), &self.o_proj)
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

/// Expands one conservative policy plan into every concrete Muse tensor slice.
///
/// Full attention retains all keys for every query chunk. Unpadded local
/// attention narrows each chunk to the union of its row-wise windows. The
/// function rejects any concrete key union wider than the bound used to choose
/// the query chunk size, then independently validates every realized plane
/// against the execution policy. Consequently, successful return proves that
/// each score, mask, and probability plane passed to the backend is bounded.
#[allow(clippy::too_many_arguments)]
fn plan_muse_attention_chunks(
    policy: AttentionExecutionPolicy,
    plan: AttentionExecutionPlan,
    batch: usize,
    heads: usize,
    query_sequence: usize,
    key_sequence: usize,
    indexed_window: Option<usize>,
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
    let mut chunks = Vec::with_capacity(plan.chunk_count());
    for query_start in (0..query_sequence).step_by(plan.query_chunk_size()) {
        let query_end = query_start
            .checked_add(plan.query_chunk_size())
            .unwrap_or(query_sequence)
            .min(query_sequence);
        let query_range = query_start..query_end;
        let key_range = match indexed_window {
            Some(window) => {
                aligned_key_range(query_sequence, key_sequence, query_range.clone(), window)?
            }
            None => 0..key_sequence,
        };
        if key_range.len() > plan.key_sequence_bound() {
            return Err(Error::BackendCapability(format!(
                "Muse attention key union {} exceeds planned bound {} for query range {query_range:?}",
                key_range.len(),
                plan.key_sequence_bound()
            )));
        }
        policy.validate_chunk(batch, heads, query_range.len(), key_range.len())?;
        let plane_shape = [batch, heads, query_range.len(), key_range.len()];
        let plane_elements = plane_shape
            .into_iter()
            .try_fold(1usize, usize::checked_mul)
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
/// `[max(0, start + K - Q - W + 1), end + K - Q)`. Muse self-attention and
/// cached suffix attention both satisfy `K >= Q`.
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
    let offset = key_sequence.checked_sub(query_sequence).ok_or_else(|| {
        Error::shape(
            "muse_glimmer.attention_alignment",
            "key sequence at least as long as query sequence",
            format!("query={query_sequence}, key={key_sequence}"),
        )
    })?;
    let first_aligned = offset.checked_add(query_range.start).ok_or_else(|| {
        Error::configuration(
            "muse_glimmer.attention_chunk",
            "aligned query offset overflows usize",
        )
    })?;
    let key_start = first_aligned.saturating_sub(window.saturating_sub(1));
    let key_end = offset
        .checked_add(query_range.end)
        .ok_or_else(|| {
            Error::configuration(
                "muse_glimmer.attention_chunk",
                "aligned query end overflows usize",
            )
        })?
        .min(key_sequence);
    if key_start >= key_end {
        return Err(Error::shape(
            "muse_glimmer.attention_chunk",
            "a non-empty local key interval",
            format!("{key_start}..{key_end}"),
        ));
    }
    Ok(key_start..key_end)
}

/// Builds causal and optional local exclusions for aligned tensor slices.
#[allow(clippy::too_many_arguments)]
fn aligned_attention_mask<B: Backend>(
    batch: usize,
    heads: usize,
    total_queries: usize,
    total_keys: usize,
    query_range: Range<usize>,
    key_range: Range<usize>,
    causal: bool,
    window: Option<usize>,
    device: &B::Device,
) -> Result<Option<Tensor<B, 4, Bool>>> {
    if !causal && window.is_none() {
        return Ok(None);
    }
    if query_range.is_empty()
        || key_range.is_empty()
        || query_range.end > total_queries
        || key_range.end > total_keys
    {
        return Err(Error::configuration(
            "muse_glimmer.attention_chunk",
            "query and key ranges must be non-empty and in bounds",
        ));
    }
    if window == Some(0) {
        return Err(Error::configuration(
            "muse_glimmer.attention_window",
            "must be greater than zero",
        ));
    }
    let query_offset = i64::try_from(total_keys)
        .and_then(|keys| i64::try_from(total_queries).map(|queries| keys - queries))
        .map_err(|_| {
            Error::configuration(
                "muse_glimmer.attention_alignment",
                "query/key lengths exceed i64",
            )
        })?;
    let query_start = i64::try_from(query_range.start)
        .map_err(|_| Error::configuration("muse_glimmer.attention_chunk", "offset exceeds i64"))?;
    let query_end = i64::try_from(query_range.end)
        .map_err(|_| Error::configuration("muse_glimmer.attention_chunk", "end exceeds i64"))?;
    let key_start = i64::try_from(key_range.start)
        .map_err(|_| Error::configuration("muse_glimmer.attention_chunk", "offset exceeds i64"))?;
    let key_end = i64::try_from(key_range.end)
        .map_err(|_| Error::configuration("muse_glimmer.attention_chunk", "end exceeds i64"))?;
    let query_length = query_range.len();
    let key_length = key_range.len();
    let query = (Tensor::<B, 1, Int>::arange(query_start..query_end, device) + query_offset)
        .reshape([1, 1, query_length, 1])
        .expand([batch, heads, query_length, key_length]);
    let key = Tensor::<B, 1, Int>::arange(key_start..key_end, device)
        .reshape([1, 1, 1, key_length])
        .expand([batch, heads, query_length, key_length]);
    let mut exclusion = causal.then(|| key.clone().greater(query.clone()));
    if let Some(window) = window {
        let window = i64::try_from(window).map_err(|_| {
            Error::configuration(
                "muse_glimmer.attention_window",
                "must fit in signed position arithmetic",
            )
        })?;
        let too_old = (key + window).lower_equal(query);
        exclusion = Some(match exclusion {
            Some(exclusion) => exclusion.bool_or(too_old),
            None => too_old,
        });
    }
    Ok(exclusion)
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

    /// Evaluates one stateless sandwich-normalized residual layer.
    fn forward(
        &self,
        hidden: Tensor<B, 3>,
        positions: Tensor<B, 2, Int>,
        token_mask: Option<&TokenMask<B>>,
        evaluated_context: usize,
        sliding_window: Option<usize>,
    ) -> Result<Tensor<B, 3>> {
        let residual = hidden.clone();
        let normalized = self.input_layernorm.forward(hidden);
        let mut projection = self.self_attn.project_self(normalized)?;
        if let Some(rope) = self.rope {
            let angles = rotary_angles(rope, positions, evaluated_context)?;
            (projection.query, projection.key) =
                apply_rotary(projection.query, projection.key, &angles)?;
        }
        let local_window = match self.layer_type {
            MuseGlimmerLayerType::SlidingAttention => Some(sliding_window.ok_or_else(|| {
                Error::configuration(
                    "muse_glimmer.sliding_window",
                    "is required by a sliding-attention layer",
                )
            })?),
            MuseGlimmerLayerType::FullAttention => None,
        };
        let attended = self
            .self_attn
            .attend_stateless(projection, token_mask, local_window)?;
        let hidden = residual + self.post_attention_layernorm.forward(attended);
        let residual = hidden.clone();
        let fed =
            muse_gated_feed_forward(&self.mlp, self.pre_feedforward_layernorm.forward(hidden))?;
        Ok(residual + self.post_feedforward_layernorm.forward(fed))
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

    /// Evaluates the stateless backbone and retains only requested hidden states.
    fn forward(
        &self,
        input: &CausalLmInput<B>,
        retain_hidden_states: bool,
    ) -> Result<(Tensor<B, 3>, Vec<Tensor<B, 3>>)> {
        let validated = self.validate_public_input(input)?;
        let [batch, sequence] = input.token_ids.dims();
        let mask_is_all_valid = validated.fully_valid;
        let positions = validated.require_positions(INPUT_NAMES.position_ids)?;
        let evaluated_context = validated.evaluated_context;
        let mut hidden = scale_free_rms_norm(
            self.embed_tokens.forward(input.token_ids.clone()),
            self.config.rms_norm_eps,
        );
        let mut hidden_states = Vec::with_capacity(if retain_hidden_states {
            self.layers.len() + 2
        } else {
            0
        });
        if retain_hidden_states {
            hidden_states.push(hidden.clone());
        }
        let window = self.config.resolved_sliding_window();
        let attention = self.layers.first().ok_or_else(|| {
            Error::configuration(
                "muse_glimmer.layers",
                "validated models must retain at least one decoder layer",
            )
        })?;
        let full_plan = attention.self_attn.execution_policy.plan(
            batch,
            self.config.num_attention_heads,
            sequence,
            sequence,
        )?;
        // Elide a redundant all-valid mask only when bounded execution actually
        // chunks the prompt. Single-chunk oracle calls retain the exact prior
        // mask/softmax operation graph; long local layers gain the safe
        // `window + chunk - 1` key slice.
        let attention_token_mask = if mask_is_all_valid && full_plan.chunk_count() > 1 {
            None
        } else {
            input.token_mask.as_ref()
        };
        for layer in &self.layers {
            hidden = layer.forward(
                hidden,
                positions.clone(),
                attention_token_mask,
                evaluated_context,
                window,
            )?;
            if retain_hidden_states {
                hidden_states.push(hidden.clone());
            }
        }
        hidden = self.norm.forward(hidden);
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
    /// The default is synchronized query chunking. Selecting
    /// [`AttentionExecutionPolicy::Unbounded`] is an explicit opt-out intended
    /// for small parity fixtures or externally provisioned training runs.
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
        let (hidden, hidden_states) = self.model.forward(&input, options.hidden_states)?;
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

/// Applies a learned-scale-free RMS norm in FP32 and restores the input dtype.
fn scale_free_rms_norm<B: Backend, const D: usize>(
    input: Tensor<B, D>,
    epsilon: f64,
) -> Tensor<B, D> {
    let dtype = input.dtype();
    let input = input.cast(DType::F32);
    let inverse_rms = (input.clone().square().mean_dim(D - 1) + epsilon).powf_scalar(-0.5);
    (input * inverse_rms).cast(dtype)
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

/// Evaluates Muse's SwiGLU with packed-weight-preserving linear dispatch.
///
/// The equation is `down(SiLU(gate(x)) * up(x))`. Keeping all three maps at
/// the shared [`linear`] boundary lets CubeCL consume Q4/Q8 weights directly
/// instead of materializing a complete dequantized projection matrix.
fn muse_gated_feed_forward<B: Backend, const D: usize>(
    mlp: &GatedFeedForward<B>,
    input: Tensor<B, D>,
) -> Result<Tensor<B, D>> {
    let gate = apply_activation(linear(input.clone(), &mlp.gate_proj)?, mlp.activation);
    let content = linear(input, &mlp.up_proj)?;
    linear(gate * content, &mlp.down_proj)
}

/// Combines padding exclusions with an optional left-bounded causal window.
///
/// A token mask defines a semantic sequence by removing invalid source columns.
/// Consequently, the local distance between two valid tokens is the difference
/// between their cumulative valid-token ordinals, not their rectangular column
/// indices. This agrees with cached execution, which compacts every row before
/// appending it to the KV cache. Without that distinction an interior padding
/// hole would consume one position of the stateless window but no position of
/// the cached window.
fn stateless_attention_mask<B: Backend>(
    token_mask: Option<&TokenMask<B>>,
    batch: usize,
    heads: usize,
    sequence: usize,
    window: Option<usize>,
    device: &B::Device,
) -> Result<Option<Tensor<B, 4, Bool>>> {
    let padding = padding_attention_mask(token_mask, token_mask, batch, heads, sequence, sequence)?;
    let Some(window) = window else {
        return Ok(padding);
    };
    let ordinals = match token_mask {
        Some(mask) => {
            let valid = mask.as_tensor().clone();
            (valid.clone().int().cumsum(1) - 1).mask_fill(valid.bool_not(), 0)
        }
        None => Tensor::<B, 1, Int>::arange(0..sequence as i64, device)
            .reshape([1, sequence])
            .expand([batch, sequence]),
    };
    let query = ordinals
        .clone()
        .reshape([batch, 1, sequence, 1])
        .expand([batch, heads, sequence, sequence]);
    let key = ordinals
        .reshape([batch, 1, 1, sequence])
        .expand([batch, heads, sequence, sequence]);
    let too_old = (key + window as i64).lower_equal(query);
    Ok(Some(match padding {
        Some(padding) => padding.bool_or(too_old),
        None => too_old,
    }))
}

/// Builds only one stateless chunk's padding and semantic-window exclusions.
///
/// Causality is added separately from physical tensor indices. When padding is
/// present, `ordinals` carries cumulative valid-token indices so holes do not
/// consume local-window capacity.
#[allow(clippy::too_many_arguments)]
fn stateless_attention_mask_chunk<B: Backend>(
    valid: Option<&Tensor<B, 2, Bool>>,
    ordinals: Option<&Tensor<B, 2, Int>>,
    batch: usize,
    heads: usize,
    query_range: Range<usize>,
    key_range: Range<usize>,
    window: Option<usize>,
) -> Result<Option<Tensor<B, 4, Bool>>> {
    let query_sequence = query_range.len();
    let key_sequence = key_range.len();
    let Some(valid) = valid else {
        if ordinals.is_some() {
            return Err(Error::configuration(
                "muse_glimmer.attention_ordinals",
                "cannot exist without a token-valid mask",
            ));
        }
        return Ok(None);
    };
    if valid.dims()[0] != batch
        || query_range.end > valid.dims()[1]
        || key_range.end > valid.dims()[1]
    {
        return Err(Error::shape(
            "muse_glimmer.token_mask",
            format!("[{batch}, sequence covering both chunk ranges]"),
            format!("{:?}", valid.dims()),
        ));
    }
    let query_invalid = valid
        .clone()
        .slice([0..batch, query_range.clone()])
        .bool_not()
        .reshape([batch, 1, query_sequence, 1])
        .expand([batch, heads, query_sequence, key_sequence]);
    let key_invalid = valid
        .clone()
        .slice([0..batch, key_range.clone()])
        .bool_not()
        .reshape([batch, 1, 1, key_sequence])
        .expand([batch, heads, query_sequence, key_sequence]);
    let mut exclusion = query_invalid.bool_or(key_invalid);
    if let Some(window) = window {
        let ordinals = ordinals.ok_or_else(|| {
            Error::configuration(
                "muse_glimmer.attention_ordinals",
                "are required for a padded sliding-window chunk",
            )
        })?;
        if ordinals.dims() != valid.dims() {
            return Err(Error::shape(
                "muse_glimmer.attention_ordinals",
                format!("{:?}", valid.dims()),
                format!("{:?}", ordinals.dims()),
            ));
        }
        let window = i64::try_from(window).map_err(|_| {
            Error::configuration(
                "muse_glimmer.attention_window",
                "must fit in signed position arithmetic",
            )
        })?;
        let query = ordinals
            .clone()
            .slice([0..batch, query_range])
            .reshape([batch, 1, query_sequence, 1])
            .expand([batch, heads, query_sequence, key_sequence]);
        let key = ordinals
            .clone()
            .slice([0..batch, key_range])
            .reshape([batch, 1, 1, key_sequence])
            .expand([batch, heads, query_sequence, key_sequence]);
        exclusion = exclusion.bool_or((key + window).lower_equal(query));
    }
    Ok(Some(exclusion))
}

/// Converts per-layer serialized RoPE metadata to a shared validated config.
fn muse_rope_config(config: &MuseGlimmerTextConfig, theta: f64) -> Result<RopeConfig> {
    let parameters = config.rope_parameters.clone().unwrap_or_default();
    let scaling = match parameters.rope_type {
        RopeType::Default => RopeScaling::None,
        RopeType::Linear => LinearRopeScaling::new(parameters.factor.unwrap_or(1.0))
            .map(RopeScaling::from)
            .map_err(|error| Error::configuration("muse_glimmer.rope", error.to_string()))?,
        RopeType::Dynamic => DynamicNtkScaling::new(
            parameters.factor.unwrap_or(1.0),
            parameters
                .original_max_position_embeddings
                .unwrap_or(config.max_position_embeddings),
        )
        .map(RopeScaling::from)
        .map_err(|error| Error::configuration("muse_glimmer.rope", error.to_string()))?,
        RopeType::Yarn => {
            let mut yarn = YarnConfig::new(
                parameters.factor.unwrap_or(1.0),
                parameters
                    .original_max_position_embeddings
                    .unwrap_or(config.max_position_embeddings),
            )
            .map_err(|error| Error::configuration("muse_glimmer.rope", error.to_string()))?;
            if let Some(factor) = parameters.attention_factor {
                yarn = yarn.with_attention_factor(factor).map_err(|error| {
                    Error::configuration("muse_glimmer.rope", error.to_string())
                })?;
            }
            yarn = yarn
                .with_beta_overrides(parameters.beta_fast, parameters.beta_slow)
                .map_err(|error| Error::configuration("muse_glimmer.rope", error.to_string()))?;
            RopeScaling::Yarn(yarn)
        }
        unsupported => {
            return Err(Error::configuration(
                "muse_glimmer.rope_type",
                format!("{unsupported:?} is not implemented for Muse-Glimmer"),
            ));
        }
    };
    RopeConfig::new(config.head_dim, config.head_dim, theta, scaling)
        .map_err(|error| Error::configuration("muse_glimmer.rope", error.to_string()))
}

/// Multiplies two shape terms without integer wraparound.
fn checked_product(field: &'static str, left: usize, right: usize) -> Result<usize> {
    left.checked_mul(right)
        .ok_or_else(|| Error::configuration(field, "dimension product overflows usize"))
}

#[cfg(test)]
#[path = "model_tests.rs"]
mod tests;
