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
        Bool, DType, ElementConversion, Int, Tensor, TensorData, activation::sigmoid,
        backend::Backend, module::linear as tensor_linear, ops::AttentionModuleOptions,
    },
};

use crate::{
    ArchitectureConfig, CausalLmInput, CausalLmOutput, Error, ForwardOptions, LogitsSelection,
    Result,
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
use crate::models::config::RopeType;

mod decode;

pub use decode::{MuseGlimmerDecodeState, MuseGlimmerLayerCache};

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
            LinearConfig::new(input, output)
                .with_bias(bias)
                .with_layout(LinearLayout::Col)
                .init(device)
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

    /// Evaluates already positioned attention and gates its result before `o_proj`.
    ///
    /// `mask` follows Burn's kernel convention: `true` excludes a query/key pair.
    /// This entry point can slice score computation but cannot undo an already
    /// materialized quadratic mask. Stateless and cached model paths use the
    /// lazy-mask entry points below for long contexts.
    #[cfg(test)]
    fn attend(
        &self,
        projection: MuseAttentionProjection<B>,
        mask: Option<Tensor<B, 4, Bool>>,
        causal: bool,
    ) -> Result<Tensor<B, 3>> {
        let [batch, query_sequence, key_sequence] = self.validate_projection(&projection)?;
        validate_materialized_mask(
            mask.as_ref(),
            batch,
            self.query_heads,
            query_sequence,
            key_sequence,
        )?;
        let plan =
            self.execution_policy
                .plan(batch, self.query_heads, query_sequence, key_sequence)?;
        if plan.chunk_count() == 1 {
            return self.attend_unbounded(projection, mask, causal);
        }
        self.attend_chunked(projection, plan, causal, None, |query_range, key_range| {
            Ok(mask.as_ref().map(|mask| {
                mask.clone()
                    .slice([0..batch, 0..self.query_heads, query_range, key_range])
            }))
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
        for query_start in (0..query_sequence).step_by(plan.query_chunk_size()) {
            let query_end = query_start
                .saturating_add(plan.query_chunk_size())
                .min(query_sequence);
            let query_range = query_start..query_end;
            let key_range = match indexed_window {
                Some(window) => {
                    aligned_key_range(query_sequence, key_sequence, query_range.clone(), window)?
                }
                None => 0..key_sequence,
            };
            self.execution_policy.validate_chunk(
                batch,
                self.query_heads,
                query_range.len(),
                key_range.len(),
            )?;
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
        let [batch, sequence] = input.token_ids.dims();
        validate_nonempty_input(batch, sequence)?;
        validate_input_devices(input, &self.embed_tokens.weight.lazy_device())?;
        validate_token_ids(input.token_ids.clone(), self.config.vocab_size)?;
        let mask_is_all_valid = validate_mask(input.token_mask.as_ref(), batch, sequence, false)?;
        let positions = resolve_positions(input, batch, sequence)?;
        validate_position_range(positions.clone(), self.config.max_position_embeddings)?;
        let evaluated_context = evaluated_context(positions.clone())?;
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
                LinearConfig::new(hidden, vocabulary)
                    .with_bias(false)
                    .with_layout(LinearLayout::Col)
                    .init(device)
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
                ensure_model_device(
                    "muse_glimmer.lm_head",
                    &head.weight.lazy_device(),
                    &hidden_device,
                )?;
                linear(hidden, head)?
            }
            None => {
                ensure_model_device(
                    "muse_glimmer.model.embed_tokens",
                    &self.model.embed_tokens.weight.lazy_device(),
                    &hidden_device,
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

/// Rejects empty tensor extents before embedding or reduction kernels execute.
fn validate_nonempty_input(batch: usize, sequence: usize) -> Result<()> {
    if batch == 0 || sequence == 0 {
        return Err(Error::shape(
            "muse_glimmer.input_ids",
            "non-empty [batch, sequence]",
            format!("[{batch}, {sequence}]"),
        ));
    }
    Ok(())
}

/// Rejects stateless input tensors that do not share the model parameter device.
///
/// Device identity is checked before any synchronizing reduction, positional
/// arithmetic, or embedding lookup so callers receive a contextual error
/// rather than a backend-specific mixed-device failure.
fn validate_input_devices<B: Backend>(
    input: &CausalLmInput<B>,
    model_device: &B::Device,
) -> Result<()> {
    ensure_model_device(
        "muse_glimmer.input_ids",
        &input.token_ids.device(),
        model_device,
    )?;
    if let Some(mask) = &input.token_mask {
        ensure_model_device(
            "muse_glimmer.token_mask",
            &mask.as_tensor().device(),
            model_device,
        )?;
    }
    if let Some(positions) = &input.position_ids {
        ensure_model_device(
            "muse_glimmer.position_ids",
            &positions.device(),
            model_device,
        )?;
    }
    Ok(())
}

/// Reports one stateless tensor/parameter device mismatch contextually.
fn ensure_model_device<D: PartialEq>(tensor: &str, actual: &D, expected: &D) -> Result<()> {
    if actual != expected {
        return Err(Error::BackendCapability(format!(
            "{tensor} and Muse-Glimmer model parameters are on different devices"
        )));
    }
    Ok(())
}

/// Validates token values before the embedding backend receives indices.
fn validate_token_ids<B: Backend>(token_ids: Tensor<B, 2, Int>, vocabulary: usize) -> Result<()> {
    let minimum = token_ids
        .clone()
        .min()
        .try_into_scalar()
        .map_err(|error| Error::BackendCapability(format!("failed to inspect token IDs: {error}")))?
        .elem::<i64>();
    let maximum = token_ids
        .max()
        .try_into_scalar()
        .map_err(|error| Error::BackendCapability(format!("failed to inspect token IDs: {error}")))?
        .elem::<i64>();
    if minimum < 0 || maximum as usize >= vocabulary {
        return Err(Error::shape(
            "muse_glimmer.input_ids",
            format!("values in 0..{vocabulary}"),
            format!("observed range {minimum}..={maximum}"),
        ));
    }
    Ok(())
}

/// Validates a semantic token mask and reports whether every entry is valid.
fn validate_mask<B: Backend>(
    mask: Option<&TokenMask<B>>,
    batch: usize,
    sequence: usize,
    permit_empty_rows: bool,
) -> Result<bool> {
    let Some(mask) = mask else {
        return Ok(true);
    };
    if mask.as_tensor().dims() != [batch, sequence] {
        return Err(Error::shape(
            "muse_glimmer.token_mask",
            format!("[{batch}, {sequence}]"),
            format!("{:?}", mask.as_tensor().dims()),
        ));
    }
    let minimum = mask
        .as_tensor()
        .clone()
        .int()
        .sum_dim(1)
        .min()
        .try_into_scalar()
        .map_err(|error| Error::BackendCapability(format!("failed to inspect mask: {error}")))?
        .elem::<i64>();
    if !permit_empty_rows && minimum == 0 {
        return Err(Error::shape(
            "muse_glimmer.token_mask",
            "at least one valid token in every prompt row",
            "a fully masked row",
        ));
    }
    let sequence = i64::try_from(sequence).map_err(|_| {
        Error::configuration(
            "muse_glimmer.token_mask",
            "sequence length exceeds signed mask arithmetic",
        )
    })?;
    Ok(minimum == sequence)
}

/// Resolves explicit positions or padding-independent cumulative token ordinals.
fn resolve_positions<B: Backend>(
    input: &CausalLmInput<B>,
    batch: usize,
    sequence: usize,
) -> Result<Tensor<B, 2, Int>> {
    if let Some(positions) = &input.position_ids {
        if positions.dims() != [batch, sequence] {
            return Err(Error::shape(
                "muse_glimmer.position_ids",
                format!("[{batch}, {sequence}]"),
                format!("{:?}", positions.dims()),
            ));
        }
        return Ok(positions.clone());
    }
    let device = input.token_ids.device();
    match &input.token_mask {
        Some(mask) => {
            let valid = mask.as_tensor().clone();
            Ok((valid.clone().int().cumsum(1) - 1).mask_fill(valid.bool_not(), 0))
        }
        None => Ok(Tensor::<B, 1, Int>::arange(0..sequence as i64, &device)
            .reshape([1, sequence])
            .expand([batch, sequence])),
    }
}

/// Validates absolute position IDs against the configured context bound.
fn validate_position_range<B: Backend>(
    positions: Tensor<B, 2, Int>,
    capacity: usize,
) -> Result<()> {
    let minimum = positions
        .clone()
        .min()
        .try_into_scalar()
        .map_err(|error| Error::BackendCapability(format!("failed to inspect positions: {error}")))?
        .elem::<i64>();
    let maximum = positions
        .max()
        .try_into_scalar()
        .map_err(|error| Error::BackendCapability(format!("failed to inspect positions: {error}")))?
        .elem::<i64>();
    if minimum < 0 || maximum as usize >= capacity {
        return Err(Error::shape(
            "muse_glimmer.position_ids",
            format!("values in 0..{capacity}"),
            format!("observed range {minimum}..={maximum}"),
        ));
    }
    Ok(())
}

/// Returns one plus the largest position in a nonempty tensor.
fn evaluated_context<B: Backend>(positions: Tensor<B, 2, Int>) -> Result<usize> {
    let maximum = positions
        .max()
        .try_into_scalar()
        .map_err(|error| Error::BackendCapability(format!("failed to inspect positions: {error}")))?
        .elem::<i64>();
    usize::try_from(maximum)
        .map_err(|_| Error::configuration("muse_glimmer.position_ids", "must be non-negative"))?
        .checked_add(1)
        .ok_or_else(|| Error::configuration("muse_glimmer.position_ids", "context overflows usize"))
}

/// Selects hidden positions before the expensive vocabulary projection.
fn select_hidden<B: Backend>(
    hidden: Tensor<B, 3>,
    mask: Option<&TokenMask<B>>,
    selection: &LogitsSelection,
) -> Result<Tensor<B, 3>> {
    let [batch, sequence, width] = hidden.dims();
    validate_logits_selection(selection, sequence)?;
    match selection {
        LogitsSelection::All => Ok(hidden),
        LogitsSelection::Last => match mask {
            None => Ok(hidden.narrow(1, sequence - 1, 1)),
            Some(mask) => {
                let device = hidden.device();
                let positions = Tensor::<B, 1, Int>::arange(0..sequence as i64, &device)
                    .reshape([1, sequence])
                    .expand([batch, sequence]);
                let last = positions
                    .mask_fill(mask.as_tensor().clone().bool_not(), -1)
                    .argmax(1)
                    .unsqueeze_dim::<3>(2)
                    .expand([batch, 1, width]);
                Ok(hidden.gather(1, last))
            }
        },
        LogitsSelection::Positions(positions) => {
            let indices = positions
                .iter()
                .map(|&value| value as i64)
                .collect::<Vec<_>>();
            let count = indices.len();
            let device = hidden.device();
            Ok(hidden.select(
                1,
                Tensor::<B, 1, Int>::from_data(TensorData::new(indices, [count]), &device),
            ))
        }
    }
}

/// Validates explicit selective-logit indices.
fn validate_logits_selection(selection: &LogitsSelection, sequence: usize) -> Result<()> {
    if let LogitsSelection::Positions(positions) = selection
        && (positions.is_empty() || positions.iter().any(|&position| position >= sequence))
    {
        return Err(Error::shape(
            "forward_options.logits.positions",
            format!("a non-empty list of indices below {sequence}"),
            format!("{positions:?}"),
        ));
    }
    Ok(())
}

/// Multiplies two shape terms without integer wraparound.
fn checked_product(field: &'static str, left: usize, right: usize) -> Result<usize> {
    left.checked_mul(right)
        .ok_or_else(|| Error::configuration(field, "dimension product overflows usize"))
}

#[cfg(test)]
mod tests {
    #[cfg(all(feature = "attention-kernel-tracing", target_os = "macos"))]
    use std::sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    };

    #[cfg(all(feature = "attention-kernel-tracing", target_os = "macos"))]
    use burn::backend::Metal;
    #[cfg(all(feature = "attention-kernel-tracing", target_os = "macos"))]
    use burn::tensor::module::attention;
    use burn::{
        backend::Flex,
        module::Param,
        tensor::{Int, Tensor, TensorData, Tolerance, backend::BackendTypes, f16},
    };
    #[cfg(all(feature = "attention-kernel-tracing", target_os = "macos"))]
    use tracing::{
        Event, Metadata, Subscriber,
        field::{Field, Visit},
        span::{Attributes, Id, Record},
    };

    use super::*;
    use crate::{
        CausalLmInput, ForwardOptions, checkpoint::TensorSink,
        models::muse_glimmer::MuseGlimmerCheckpointOptions,
    };

    type TestBackend = Flex<f32, i32>;

    /// Serializes opt-in GPU work and process-local attention autotuning.
    #[cfg(all(feature = "attention-kernel-tracing", target_os = "macos"))]
    static METAL_ATTENTION_TEST_LOCK: Mutex<()> = Mutex::new(());

    /// Sequence length that makes Burn 0.21 prioritize flash candidates over fallback.
    #[cfg(all(feature = "attention-kernel-tracing", target_os = "macos"))]
    const METAL_ATTENTION_SEQUENCE: usize = 4097;

    /// Pinned Muse-Glimmer attention head width.
    #[cfg(all(feature = "attention-kernel-tracing", target_os = "macos"))]
    const METAL_ATTENTION_HEAD_DIM: usize = 128;

    /// One CubeCL kernel submitted by the captured Metal attention call.
    #[cfg(all(feature = "attention-kernel-tracing", target_os = "macos"))]
    #[derive(Clone, Debug, Eq, PartialEq)]
    struct MetalKernelLaunch {
        /// Generated kernel name.
        name: String,
        /// Configuration-bearing kernel identifier.
        id: String,
    }

    /// Extracts the two display fields attached to CubeCL's launch span.
    #[cfg(all(feature = "attention-kernel-tracing", target_os = "macos"))]
    #[derive(Default)]
    struct MetalKernelFieldVisitor {
        /// Generated kernel name when present.
        name: Option<String>,
        /// Configuration-bearing kernel identifier when present.
        id: Option<String>,
    }

    #[cfg(all(feature = "attention-kernel-tracing", target_os = "macos"))]
    impl MetalKernelFieldVisitor {
        /// Stores one tracing field after removing debug-string quotes.
        fn record_text(&mut self, field: &Field, value: String) {
            let value = value
                .strip_prefix('"')
                .and_then(|value| value.strip_suffix('"'))
                .unwrap_or(&value)
                .to_owned();
            match field.name() {
                "kernel.name" => self.name = Some(value),
                "kernel.id" => self.id = Some(value),
                _ => {}
            }
        }
    }

    #[cfg(all(feature = "attention-kernel-tracing", target_os = "macos"))]
    impl Visit for MetalKernelFieldVisitor {
        fn record_debug(&mut self, field: &Field, value: &dyn core::fmt::Debug) {
            self.record_text(field, format!("{value:?}"));
        }

        fn record_str(&mut self, field: &Field, value: &str) {
            self.record_text(field, value.to_owned());
        }
    }

    /// Thread-local tracing subscriber retaining launch names and identifiers.
    #[cfg(all(feature = "attention-kernel-tracing", target_os = "macos"))]
    #[derive(Clone, Default)]
    struct MetalKernelSubscriber {
        /// Launches shared with the capture caller.
        launches: Arc<Mutex<Vec<MetalKernelLaunch>>>,
        /// Non-zero span identifier source.
        next_span: Arc<AtomicU64>,
    }

    #[cfg(all(feature = "attention-kernel-tracing", target_os = "macos"))]
    impl Subscriber for MetalKernelSubscriber {
        fn enabled(&self, metadata: &Metadata<'_>) -> bool {
            metadata.name() == "launch_inner"
        }

        fn new_span(&self, attributes: &Attributes<'_>) -> Id {
            let mut visitor = MetalKernelFieldVisitor::default();
            attributes.record(&mut visitor);
            if let (Some(name), Some(id)) = (visitor.name, visitor.id) {
                self.launches
                    .lock()
                    .expect("Metal kernel trace mutex must not be poisoned")
                    .push(MetalKernelLaunch { name, id });
            }
            Id::from_u64(self.next_span.fetch_add(1, Ordering::Relaxed) + 1)
        }

        fn record(&self, _span: &Id, _values: &Record<'_>) {}

        fn record_follows_from(&self, _span: &Id, _follows: &Id) {}

        fn event(&self, _event: &Event<'_>) {}

        fn enter(&self, _span: &Id) {}

        fn exit(&self, _span: &Id) {}
    }

    /// Captures every CubeCL launch submitted while one closure runs.
    #[cfg(all(feature = "attention-kernel-tracing", target_os = "macos"))]
    fn capture_metal_kernel_launches<T>(
        operation: impl FnOnce() -> T,
    ) -> (T, Vec<MetalKernelLaunch>) {
        let subscriber = MetalKernelSubscriber::default();
        let launches = subscriber.launches.clone();
        let output = tracing::subscriber::with_default(subscriber, operation);
        let launches = Arc::try_unwrap(launches)
            .expect("capture must release its only launch-state owner")
            .into_inner()
            .expect("Metal kernel trace mutex must not be poisoned");
        (output, launches)
    }

    /// Identifies Burn 0.21's CubeCL flash-attention kernel by launch identity.
    #[cfg(all(feature = "attention-kernel-tracing", target_os = "macos"))]
    fn is_flash_attention_launch(launch: &MetalKernelLaunch) -> bool {
        [&launch.name, &launch.id].into_iter().any(|identity| {
            let identity = identity.to_ascii_lowercase();
            identity.contains("cubek_attention") || identity.contains("flash_attention")
        })
    }

    /// Builds a four-layer miniature with the released local/local/local/global schedule.
    fn tiny_config() -> MuseGlimmerTextConfig {
        MuseGlimmerTextConfig {
            vocab_size: 32,
            hidden_size: 8,
            intermediate_size: 12,
            num_hidden_layers: 4,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 4,
            max_position_embeddings: 32,
            sliding_window: Some(3),
            bos_token_id: Some(crate::models::config::TokenIds::One(1)),
            eos_token_id: Some(crate::models::config::TokenIds::One(2)),
            ..Default::default()
        }
    }

    /// Installs deterministic small parameters without relying on process-global RNG state.
    fn initialized_model<B: Backend>(
        config: MuseGlimmerTextConfig,
        device: &B::Device,
    ) -> MuseGlimmerForCausalLm<B> {
        let contract = config
            .checkpoint_contract(MuseGlimmerCheckpointOptions::default())
            .expect("tiny contract");
        let mut model = MuseGlimmerForCausalLm::init(config, device).expect("tiny model");
        for (tensor_index, expected) in contract.expected_tensors().iter().enumerate() {
            let shape = expected
                .shape
                .iter()
                .map(|&dimension| dimension as usize)
                .collect::<Vec<_>>();
            let elements = shape.iter().product();
            let values = (0..elements)
                .map(|element| {
                    let code = (element + tensor_index * 5) % 29;
                    (code as f32 - 14.0) * 0.003
                })
                .collect::<Vec<_>>();
            TensorSink::write_tensor(&mut model, &expected.name, TensorData::new(values, shape))
                .unwrap_or_else(|error| panic!("{}: {error}", expected.name));
        }
        model
    }

    /// Returns an initialized miniature and its device.
    fn tiny_model() -> (
        MuseGlimmerForCausalLm<TestBackend>,
        <TestBackend as BackendTypes>::Device,
    ) {
        let device = Default::default();
        (initialized_model(tiny_config(), &device), device)
    }

    /// Builds an isolated attention fixture whose values expose its softmax weights.
    ///
    /// The value rows are basis vectors and the output projection is the identity,
    /// so the first three result channels equal the three attention probabilities.
    /// A saturated gate retains them unchanged. This makes the fixture sensitive
    /// to a one-ULP half-softmax regression without involving checkpoint loading.
    fn softmax_oracle_attention(
        dtype: DType,
    ) -> (
        MuseGlimmerAttention<TestBackend>,
        MuseAttentionProjection<TestBackend>,
    ) {
        let device = Default::default();
        let config = MuseGlimmerTextConfig {
            hidden_size: 4,
            intermediate_size: 8,
            num_hidden_layers: 1,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            head_dim: 4,
            max_position_embeddings: 8,
            sliding_window: Some(2),
            layer_types: vec![MuseGlimmerLayerType::FullAttention],
            layer_rope_theta: vec![0.0],
            vocab_size: 8,
            bos_token_id: None,
            eos_token_id: None,
            ..Default::default()
        };
        let mut attention = MuseGlimmerAttention::<TestBackend>::init(&config, &device).unwrap();
        attention.o_proj.weight = Param::from_tensor(
            Tensor::from_floats(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                &device,
            )
            .cast(dtype),
        );
        let projection = MuseAttentionProjection {
            // With head_dim=4 the implicit scale is 1/2, so these dot
            // products produce exact scores [-0.125, 5.875, 0.4375].
            query: Tensor::from_floats([[[[2.0, 0.0, 0.0, 0.0]]]], &device).cast(dtype),
            key: Tensor::<TestBackend, 3>::from_floats(
                [[
                    [-0.125, 0.0, 0.0, 0.0],
                    [5.875, 0.0, 0.0, 0.0],
                    [0.4375, 0.0, 0.0, 0.0],
                ]],
                &device,
            )
            .unsqueeze_dim::<4>(0)
            .cast(dtype),
            value: Tensor::<TestBackend, 3>::from_floats(
                [[
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                ]],
                &device,
            )
            .unsqueeze_dim::<4>(0)
            .cast(dtype),
            gate: Tensor::<TestBackend, 3>::full([1, 1, 4], 80.0, &device).cast(dtype),
        };
        (attention, projection)
    }

    /// F32 attention matches the literal upstream eager-softmax oracle.
    ///
    /// The expected probabilities were generated from
    /// `eager_attention_forward` in Hugging Face Transformers revision
    /// `cfa309cfa34b5497a32fd50af49aa97aac94c054`. Keeping literals here makes
    /// the test independent of Burn's own softmax implementation.
    #[test]
    fn attention_matches_pinned_fp32_softmax_oracle() {
        let (attention, projection) = softmax_oracle_attention(DType::F32);
        let actual = attention
            .attend(projection, None, false)
            .expect("valid isolated attention");
        let expected = TensorData::new(
            vec![0.002_461_939_6_f32, 0.993_217_3, 0.004_320_838_5, 0.0],
            [1, 1, 4],
        );
        actual
            .into_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::absolute(1e-7));
    }

    /// F16 attention rounds the pinned FP32-softmax result, not a half-softmax.
    ///
    /// The third expected half has bits `0x1c6d`; evaluating exponentiation and
    /// normalization in F16 instead produces adjacent value `0x1c6c`. This
    /// one-ULP discriminator directly covers Muse's explicit upstream
    /// `softmax(..., dtype=torch.float32).to(query.dtype)` requirement at
    /// Transformers revision `cfa309cfa34b5497a32fd50af49aa97aac94c054`.
    #[test]
    fn f16_attention_uses_fp32_softmax_before_rounding() {
        let (attention, projection) = softmax_oracle_attention(DType::F16);
        let actual = attention
            .attend(projection, None, false)
            .expect("valid isolated attention");
        assert_eq!(actual.dtype(), DType::F16);
        let bits = actual
            .into_data()
            .to_vec::<f16>()
            .expect("F16 result")
            .into_iter()
            .map(f16::to_bits)
            .collect::<Vec<_>>();
        assert_eq!(bits, [0x190b, 0x3bf2, 0x1c6d, 0x0000]);
        assert_ne!(bits[2], 0x1c6c, "half-precision softmax sentinel");
    }

    /// Builds an official-width Metal fixture whose score rounding changes F16 softmax bits.
    #[cfg(all(feature = "attention-kernel-tracing", target_os = "macos"))]
    fn metal_score_rounding_fixture(
        device: &<Metal as BackendTypes>::Device,
    ) -> (MuseGlimmerAttention<Metal>, MuseAttentionProjection<Metal>) {
        const HEADS: usize = 1;
        const HIDDEN_SIZE: usize = HEADS * METAL_ATTENTION_HEAD_DIM;
        const VISIBLE_KEYS: usize = 4;

        let config = MuseGlimmerTextConfig {
            hidden_size: HIDDEN_SIZE,
            intermediate_size: HIDDEN_SIZE * 2,
            num_hidden_layers: 1,
            num_attention_heads: HEADS,
            num_key_value_heads: HEADS,
            head_dim: METAL_ATTENTION_HEAD_DIM,
            max_position_embeddings: METAL_ATTENTION_SEQUENCE,
            sliding_window: None,
            layer_types: vec![MuseGlimmerLayerType::FullAttention],
            layer_rope_theta: vec![0.0],
            vocab_size: 8,
            bos_token_id: None,
            eos_token_id: None,
            ..Default::default()
        };
        let mut muse_attention = MuseGlimmerAttention::<Metal>::init(&config, device).unwrap();
        let mut identity = vec![0.0_f32; HIDDEN_SIZE * HIDDEN_SIZE];
        for diagonal in 0..HIDDEN_SIZE {
            identity[diagonal * HIDDEN_SIZE + diagonal] = 1.0;
        }
        muse_attention.o_proj.weight = Param::from_tensor(
            Tensor::<Metal, 2>::from_data(
                TensorData::new(identity, [HIDDEN_SIZE, HIDDEN_SIZE]),
                device,
            )
            .cast(DType::F16),
        );

        let mut query = vec![0.0_f32; HEADS * METAL_ATTENTION_SEQUENCE * METAL_ATTENTION_HEAD_DIM];
        let mut key = vec![0.0_f32; HEADS * METAL_ATTENTION_SEQUENCE * METAL_ATTENTION_HEAD_DIM];
        let mut value = vec![0.0_f32; HEADS * METAL_ATTENTION_SEQUENCE * METAL_ATTENTION_HEAD_DIM];
        for head in 0..HEADS {
            for row in 0..METAL_ATTENTION_SEQUENCE {
                query[(head * METAL_ATTENTION_SEQUENCE + row) * METAL_ATTENTION_HEAD_DIM] =
                    1.730_468_8;
                key[(head * METAL_ATTENTION_SEQUENCE + row) * METAL_ATTENTION_HEAD_DIM] = -100.0;
            }
            for (row, candidate) in [-19.593_75, 7.781_25, -24.937_5, 16.812_5]
                .into_iter()
                .enumerate()
            {
                key[(head * METAL_ATTENTION_SEQUENCE + row) * METAL_ATTENTION_HEAD_DIM] = candidate;
            }
            for row in 0..VISIBLE_KEYS {
                value[(head * METAL_ATTENTION_SEQUENCE + row) * METAL_ATTENTION_HEAD_DIM + row] =
                    1.0;
            }
        }

        let projection = MuseAttentionProjection {
            query: Tensor::<Metal, 4>::from_data(
                TensorData::new(
                    query,
                    [1, HEADS, METAL_ATTENTION_SEQUENCE, METAL_ATTENTION_HEAD_DIM],
                ),
                device,
            )
            .cast(DType::F16),
            key: Tensor::<Metal, 4>::from_data(
                TensorData::new(
                    key,
                    [1, HEADS, METAL_ATTENTION_SEQUENCE, METAL_ATTENTION_HEAD_DIM],
                ),
                device,
            )
            .cast(DType::F16),
            value: Tensor::<Metal, 4>::from_data(
                TensorData::new(
                    value,
                    [1, HEADS, METAL_ATTENTION_SEQUENCE, METAL_ATTENTION_HEAD_DIM],
                ),
                device,
            )
            .cast(DType::F16),
            gate: Tensor::<Metal, 3>::full(
                [1, METAL_ATTENTION_SEQUENCE, HIDDEN_SIZE],
                80.0,
                device,
            )
            .cast(DType::F16),
        };
        (muse_attention, projection)
    }

    /// Evaluates the pinned eager score-rounding and FP32-softmax operation graph.
    #[cfg(all(feature = "attention-kernel-tracing", target_os = "macos"))]
    fn pinned_eager_last_query(projection: &MuseAttentionProjection<Metal>) -> Tensor<Metal, 4> {
        let query = projection.query.clone().slice([
            0..1,
            0..1,
            METAL_ATTENTION_SEQUENCE - 1..METAL_ATTENTION_SEQUENCE,
            0..METAL_ATTENTION_HEAD_DIM,
        ]);
        let key = projection.key.clone().slice([
            0..1,
            0..1,
            0..METAL_ATTENTION_SEQUENCE,
            0..METAL_ATTENTION_HEAD_DIM,
        ]);
        let value = projection.value.clone().slice([
            0..1,
            0..1,
            0..METAL_ATTENTION_SEQUENCE,
            0..METAL_ATTENTION_HEAD_DIM,
        ]);
        let scores =
            query.matmul(key.swap_dims(2, 3)) * (METAL_ATTENTION_HEAD_DIM as f64).sqrt().recip();
        let probabilities =
            burn::tensor::activation::softmax(scores.cast(DType::F32), 3).cast(DType::F16);
        probabilities.matmul(value)
    }

    /// Evaluates flash attention's strict-F32 QK and softmax arithmetic for the sentinel.
    #[cfg(all(feature = "attention-kernel-tracing", target_os = "macos"))]
    fn strict_f32_last_query(projection: &MuseAttentionProjection<Metal>) -> Tensor<Metal, 4> {
        let query = projection
            .query
            .clone()
            .slice([
                0..1,
                0..1,
                METAL_ATTENTION_SEQUENCE - 1..METAL_ATTENTION_SEQUENCE,
                0..METAL_ATTENTION_HEAD_DIM,
            ])
            .cast(DType::F32);
        let key = projection
            .key
            .clone()
            .slice([
                0..1,
                0..1,
                0..METAL_ATTENTION_SEQUENCE,
                0..METAL_ATTENTION_HEAD_DIM,
            ])
            .cast(DType::F32);
        let value = projection
            .value
            .clone()
            .slice([
                0..1,
                0..1,
                0..METAL_ATTENTION_SEQUENCE,
                0..METAL_ATTENTION_HEAD_DIM,
            ])
            .cast(DType::F32);
        let scores =
            query.matmul(key.swap_dims(2, 3)) * (METAL_ATTENTION_HEAD_DIM as f64).sqrt().recip();
        burn::tensor::activation::softmax(scores, 3)
            .matmul(value)
            .cast(DType::F16)
    }

    /// Reads the first four channels of one `[1, 1, 1, head_dim]` F16 result.
    #[cfg(all(feature = "attention-kernel-tracing", target_os = "macos"))]
    fn first_four_f16_bits(tensor: Tensor<Metal, 4>) -> Vec<u16> {
        tensor
            .slice([0..1, 0..1, 0..1, 0..4])
            .into_data()
            .to_vec::<f16>()
            .expect("Metal result must retain F16 storage")
            .into_iter()
            .map(f16::to_bits)
            .collect()
    }

    /// Muse F16 bypasses flash when its strict-F32 score graph changes results.
    ///
    /// Run this opt-in gate alone because it enables process-local autotuning:
    /// `cargo test --manifest-path crates/transformers/Cargo.toml
    /// --no-default-features --features std,flex,attention-kernel-tracing
    /// --lib
    /// models::muse_glimmer::model::tests::metal_f16_attention_matches_pinned_eager_score_rounding
    /// -- --ignored --exact --nocapture --test-threads=1`.
    #[cfg(all(feature = "attention-kernel-tracing", target_os = "macos"))]
    #[test]
    #[ignore = "requires a Metal-capable macOS host and serialized attention autotuning"]
    fn metal_f16_attention_matches_pinned_eager_score_rounding() {
        let _serial = METAL_ATTENTION_TEST_LOCK
            .lock()
            .expect("Metal attention test lock must not be poisoned");
        let device = Default::default();
        let (muse_attention, projection) = metal_score_rounding_fixture(&device);

        let eager_bits = first_four_f16_bits(pinned_eager_last_query(&projection));
        let strict_bits = first_four_f16_bits(strict_f32_last_query(&projection));
        assert_ne!(
            eager_bits, strict_bits,
            "Metal fixture must distinguish the pinned F16 score boundary from strict F32"
        );

        let warmup = attention(
            projection.query.clone(),
            projection.key.clone(),
            projection.value.clone(),
            None,
            None,
            muse_attention_options(true),
        );
        <Metal<f32, i32> as Backend>::sync(&device).expect("synchronize attention autotuning");
        let _ = warmup.into_data();

        let (fused_bits, launches) = capture_metal_kernel_launches(|| {
            let output = attention(
                projection.query.clone(),
                projection.key.clone(),
                projection.value.clone(),
                None,
                None,
                muse_attention_options(true),
            );
            <Metal<f32, i32> as Backend>::sync(&device).expect("synchronize fused attention");
            first_four_f16_bits(output.slice([
                0..1,
                0..1,
                METAL_ATTENTION_SEQUENCE - 1..METAL_ATTENTION_SEQUENCE,
                0..METAL_ATTENTION_HEAD_DIM,
            ]))
        });
        let used_flash = launches.iter().any(is_flash_attention_launch);
        assert!(
            used_flash,
            "autotuned call must execute an attention kernel: {launches:#?}"
        );
        assert_eq!(
            fused_bits, strict_bits,
            "Burn 0.21 flash must exercise the strict-F32 discriminator"
        );
        assert_ne!(
            fused_bits, eager_bits,
            "fixture must expose the missing F16 score-rounding boundary"
        );

        let plan = muse_attention
            .execution_policy
            .plan(1, 1, METAL_ATTENTION_SEQUENCE, METAL_ATTENTION_SEQUENCE)
            .expect("default Muse policy must cover the Metal sentinel");
        assert_eq!(plan.query_chunk_size(), 512);
        assert_eq!(plan.chunk_count(), 9);
        let (muse_bits, muse_launches) = capture_metal_kernel_launches(|| {
            muse_attention
                .attend(projection, None, true)
                .expect("valid Muse Metal attention")
                .slice([
                    0..1,
                    METAL_ATTENTION_SEQUENCE - 1..METAL_ATTENTION_SEQUENCE,
                    0..4,
                ])
                .into_data()
                .to_vec::<f16>()
                .expect("Muse output must retain F16 storage")
                .into_iter()
                .map(f16::to_bits)
                .collect::<Vec<_>>()
        });
        let muse_used_flash = muse_launches.iter().any(is_flash_attention_launch);
        assert!(
            !muse_used_flash,
            "the pinned eager Muse path must not silently dispatch flash: {muse_launches:#?}"
        );
        eprintln!(
            "Muse Metal score-rounding gate: eager={eager_bits:04x?}, strict={strict_bits:04x?}, fused={fused_bits:04x?}, used_flash={used_flash}, muse_used_flash={muse_used_flash}, muse_chunks={}x{}",
            plan.chunk_count(),
            plan.query_chunk_size(),
        );
        assert_eq!(muse_bits, eager_bits, "Muse must match pinned eager F16");
    }

    /// Non-F16 Muse options retain Burn's flash-attention eligibility contract.
    ///
    /// Burn 0.21 CubeCL falls back whenever `scale.is_some()`, even when that
    /// value redundantly equals `1/sqrt(head_dim)`. Its default Metal build also
    /// chooses fallback because `autotune` is disabled; eligibility and runtime
    /// selection are distinct. F16 Muse intentionally uses the explicit shared
    /// eager mode before these options reach a backend kernel.
    #[test]
    fn attention_options_preserve_non_f16_flash_eligibility() {
        assert_eq!(
            muse_attention_options(true),
            AttentionModuleOptions {
                scale: None,
                softcap: None,
                is_causal: true,
            }
        );
        assert_eq!(muse_attention_options(false).scale, None);
    }

    /// Sliding query chunks retain precisely the union of row-local key ranges.
    #[test]
    fn aligned_local_key_ranges_respect_inclusive_window_boundary() {
        assert_eq!(
            aligned_key_range(32_768, 32_768, 0..512, 2048).unwrap(),
            0..512
        );
        assert_eq!(
            aligned_key_range(32_768, 32_768, 2048..2560, 2048).unwrap(),
            1..2560
        );
        assert_eq!(aligned_key_range(2, 5, 0..2, 3).unwrap(), 1..5);
        assert!(aligned_key_range(5, 4, 0..1, 3).is_err());
        assert!(aligned_key_range(5, 5, 0..1, 0).is_err());
    }

    /// Muse defaults to a bounded policy and rejects invalid runtime overrides.
    #[test]
    fn model_exposes_validated_attention_execution_policy() {
        let (model, _) = tiny_model();
        assert_eq!(
            model.attention_execution_policy(),
            AttentionExecutionPolicy::default()
        );
        let error = model
            .clone()
            .with_attention_execution_policy(AttentionExecutionPolicy::query_chunked(0, 1))
            .expect_err("zero query chunks are invalid");
        assert!(error.to_string().contains("max_query_tokens"));
        let unbounded = model
            .with_attention_execution_policy(AttentionExecutionPolicy::Unbounded)
            .expect("explicit unbounded policy");
        assert_eq!(
            unbounded.attention_execution_policy(),
            AttentionExecutionPolicy::Unbounded
        );
    }

    /// Query chunking retains full/sliding causal and padded-window semantics.
    #[test]
    fn bounded_stateless_attention_matches_unbounded_with_padding_holes() {
        let (model, device) = tiny_model();
        let unbounded = model
            .clone()
            .with_attention_execution_policy(AttentionExecutionPolicy::Unbounded)
            .expect("unbounded fixture");
        let bounded = model
            .with_attention_execution_policy(AttentionExecutionPolicy::query_chunked(2, 20))
            .expect("two-row chunks");
        let input = || CausalLmInput {
            token_ids: Tensor::from_ints([[1, 2, 3, 4, 5]], &device),
            token_mask: Some(TokenMask::new(Tensor::from_bool(
                TensorData::from([[true, false, true, true, true]]),
                &device,
            ))),
            position_ids: None,
        };
        let options = ForwardOptions {
            logits: LogitsSelection::All,
            ..Default::default()
        };
        let expected = unbounded
            .forward(input(), &options)
            .expect("unbounded forward")
            .logits;
        let actual = bounded
            .forward(input(), &options)
            .expect("bounded forward")
            .logits;
        actual
            .into_data()
            .assert_approx_eq::<f32>(&expected.into_data(), Tolerance::absolute(2e-6));
    }

    /// A redundant all-valid mask takes the optimized unpadded sliding path.
    #[test]
    fn bounded_stateless_all_valid_mask_matches_absent_mask() {
        let (model, device) = tiny_model();
        let model = model
            .with_attention_execution_policy(AttentionExecutionPolicy::query_chunked(2, 20))
            .expect("two-row chunks");
        let options = ForwardOptions {
            logits: LogitsSelection::All,
            ..Default::default()
        };
        let token_ids = || Tensor::from_ints([[1, 2, 3, 4, 5]], &device);
        let expected = model
            .forward(
                CausalLmInput {
                    token_ids: token_ids(),
                    token_mask: None,
                    position_ids: None,
                },
                &options,
            )
            .expect("unmasked forward")
            .logits;
        let actual = model
            .forward(
                CausalLmInput {
                    token_ids: token_ids(),
                    token_mask: Some(TokenMask::new(Tensor::from_bool(
                        TensorData::from([[true, true, true, true, true]]),
                        &device,
                    ))),
                    position_ids: None,
                },
                &options,
            )
            .expect("all-valid forward")
            .logits;
        actual
            .into_data()
            .assert_approx_eq::<f32>(&expected.into_data(), Tolerance::absolute(2e-6));
    }

    /// Muse initialization selects the two inverse-root graphs used upstream.
    #[test]
    fn model_selects_pinned_inverse_root_modes() {
        let (model, _) = tiny_model();
        assert_eq!(model.model.norm.root_mode, RmsNormRootMode::MultiplyByPower);
        for layer in &model.model.layers {
            for mode in [
                layer.input_layernorm.root_mode,
                layer.post_attention_layernorm.root_mode,
                layer.pre_feedforward_layernorm.root_mode,
                layer.post_feedforward_layernorm.root_mode,
            ] {
                assert_eq!(mode, RmsNormRootMode::MultiplyByReciprocalSqrt);
            }
        }
    }

    /// Scale-free Muse norms multiply by `pow(mean_square, -0.5)` before F16 rounding.
    #[test]
    fn scale_free_norm_matches_pinned_f16_power_oracle() {
        let device = Default::default();
        let input = Tensor::<TestBackend, 2>::from_floats([[0.096_557_62, 2.664_062_5]], &device)
            .cast(DType::F16);
        let actual = scale_free_rms_norm(input.clone(), 1e-5)
            .into_data()
            .to_vec::<f16>()
            .unwrap()
            .into_iter()
            .map(f16::to_bits)
            .collect::<Vec<_>>();
        let fp32 = input.cast(DType::F32);
        let rms = (fp32.clone().square().mean_dim(1) + 1e-5).sqrt();
        let division = (fp32 / rms)
            .cast(DType::F16)
            .into_data()
            .to_vec::<f16>()
            .unwrap()
            .into_iter()
            .map(f16::to_bits)
            .collect::<Vec<_>>();

        assert_eq!(actual, [0x2a8f, 0x3da7]);
        assert_eq!(division, [0x2a8e, 0x3da7]);
    }

    /// Sliding masks measure an interior hole in semantic rather than physical time.
    #[test]
    fn sliding_mask_uses_valid_token_ordinals() {
        let device = Default::default();
        let valid = TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
            [[true, true, false, true, true]].into(),
            &device,
        ));
        let excluded = stateless_attention_mask(Some(&valid), 1, 1, 5, Some(3), &device)
            .unwrap()
            .unwrap()
            .reshape([5, 5])
            .into_data()
            .to_vec::<bool>()
            .unwrap();

        assert_eq!(
            excluded,
            [
                false, false, true, false, false, // semantic query 0
                false, false, true, false, false, // semantic query 1
                true, true, true, true, true, // invalid query column
                false, false, true, false, false, // semantic query 2
                true, false, true, false, false, // semantic query 3
            ]
        );
    }

    /// Stateless inference emits exact requested shapes and dense diagnostics.
    #[test]
    fn stateless_forward_shapes_and_options_are_exact() {
        let (model, device) = tiny_model();
        let output = model
            .forward(
                CausalLmInput {
                    token_ids: Tensor::<TestBackend, 2, Int>::from_ints(
                        [[1, 2, 3], [4, 5, 6]],
                        &device,
                    ),
                    token_mask: None,
                    position_ids: None,
                },
                &ForwardOptions {
                    logits: LogitsSelection::All,
                    hidden_states: true,
                    ..Default::default()
                },
            )
            .expect("tiny forward");
        assert_eq!(output.logits.dims(), [2, 3, 32]);
        assert_eq!(output.hidden_states.len(), 6);
        assert_eq!(output.final_hidden_state.unwrap().dims(), [2, 3, 8]);
        assert!(output.attentions.is_empty());
        assert!(output.routers.is_empty());
    }

    /// NoPE makes a global layer invariant to otherwise different position IDs.
    #[test]
    fn nope_layer_does_not_consume_positions() {
        let (model, device) = tiny_model();
        let layer = &model.model.layers[3];
        assert!(!layer.uses_rope());
        let hidden = Tensor::<TestBackend, 3>::from_floats(
            [[[0.1, -0.2, 0.3, 0.4, -0.5, 0.6, 0.7, -0.8]]],
            &device,
        );
        let first = layer
            .forward(
                hidden.clone(),
                Tensor::from_ints([[0]], &device),
                None,
                1,
                None,
            )
            .unwrap();
        let later = layer
            .forward(hidden, Tensor::from_ints([[17]], &device), None, 18, None)
            .unwrap();
        first
            .into_data()
            .assert_approx_eq::<f32>(&later.into_data(), Tolerance::absolute(1e-6));
    }

    /// The separate gate is sigmoid-activated before the output projection.
    #[test]
    fn attention_gate_precedes_output_projection() {
        let device = Default::default();
        let config = MuseGlimmerTextConfig {
            hidden_size: 2,
            intermediate_size: 4,
            num_hidden_layers: 1,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            head_dim: 2,
            max_position_embeddings: 8,
            sliding_window: Some(2),
            layer_types: vec![MuseGlimmerLayerType::FullAttention],
            layer_rope_theta: vec![0.0],
            vocab_size: 8,
            bos_token_id: None,
            eos_token_id: None,
            ..Default::default()
        };
        let mut attention = MuseGlimmerAttention::<TestBackend>::init(&config, &device).unwrap();
        let identity = Tensor::from_floats([[1.0, 0.0], [0.0, 1.0]], &device);
        attention.q_proj.weight = Param::from_tensor(identity.clone());
        attention.k_proj.weight = Param::from_tensor(identity.clone());
        attention.v_proj.weight = Param::from_tensor(identity.clone());
        attention.o_proj.weight = Param::from_tensor(identity);
        attention.gate_proj.weight = Param::from_tensor(Tensor::zeros([2, 2], &device));
        let hidden = Tensor::from_floats([[[3.0, 4.0]]], &device);
        let projection = attention.project_self(hidden).unwrap();
        let value = projection.value.clone().swap_dims(1, 2).reshape([1, 1, 2]);
        let actual = attention.attend(projection, None, true).unwrap();
        (actual * 2.0)
            .into_data()
            .assert_approx_eq::<f32>(&value.into_data(), Tolerance::absolute(1e-6));
    }

    /// Query heads are scale-free normalized and then multiplied by the configured factor.
    #[test]
    fn query_normalization_applies_extra_scale() {
        let (model, device) = tiny_model();
        let attention = &model.model.layers[0].self_attn;
        let hidden = Tensor::<TestBackend, 3>::ones([1, 1, 8], &device);
        let projection = attention.project_self(hidden).unwrap();
        let query_rms = projection
            .query
            .square()
            .mean_dim(3)
            .sqrt()
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        for value in query_rms {
            assert!((value - attention.qk_scale_factor as f32).abs() < 2e-3);
        }
    }

    /// The two residual branches follow the documented four-norm sandwich order.
    #[test]
    fn decoder_layer_matches_explicit_sandwich_equation() {
        let (model, device) = tiny_model();
        let layer = &model.model.layers[3];
        let hidden = Tensor::<TestBackend, 3>::from_floats(
            [[[0.2, -0.1, 0.3, -0.4, 0.5, -0.6, 0.7, 0.8]]],
            &device,
        );
        let position = Tensor::from_ints([[0]], &device);
        let normalized = layer.input_layernorm.forward(hidden.clone());
        let projection = layer.self_attn.project_self(normalized).unwrap();
        let attended = layer.self_attn.attend(projection, None, true).unwrap();
        let after_attention = hidden.clone() + layer.post_attention_layernorm.forward(attended);
        let fed = layer
            .mlp
            .forward(
                layer
                    .pre_feedforward_layernorm
                    .forward(after_attention.clone()),
            )
            .expect("validated feed-forward projection");
        let expected = after_attention + layer.post_feedforward_layernorm.forward(fed);
        let actual = layer.forward(hidden, position, None, 1, Some(3)).unwrap();
        actual
            .into_data()
            .assert_approx_eq::<f32>(&expected.into_data(), Tolerance::absolute(1e-6));
    }

    /// Final logits match the exact output multiplier and tanh softcap equation.
    #[test]
    fn final_logit_softcap_matches_closed_form_and_bounds() {
        let (model, device) = tiny_model();
        let hidden = Tensor::<TestBackend, 3>::from_floats(
            [[[0.5, -1.0, 1.5, 0.25, -0.75, 2.0, 0.0, -0.5]]],
            &device,
        );
        let raw = linear(hidden.clone(), model.lm_head.as_ref().unwrap()).unwrap();
        let expected = ((raw * model.config().output_multiplier)
            / model.config().final_logit_softcapping)
            .tanh()
            * model.config().final_logit_softcapping;
        let actual = model.project_logits(hidden).unwrap();
        actual
            .clone()
            .into_data()
            .assert_approx_eq::<f32>(&expected.into_data(), Tolerance::absolute(1e-6));
        assert!(
            actual.abs().max().into_scalar().elem::<f32>()
                <= model.config().final_logit_softcapping as f32
        );
    }

    /// Invalid input values and unsupported diagnostics return contextual errors.
    #[test]
    fn public_forward_rejects_invalid_inputs_without_panicking() {
        let (model, device) = tiny_model();
        let invalid_token = model.forward(
            CausalLmInput {
                token_ids: Tensor::from_ints([[32]], &device),
                token_mask: None,
                position_ids: None,
            },
            &ForwardOptions::default(),
        );
        assert!(invalid_token.is_err());
        let attention_request = model.forward(
            CausalLmInput {
                token_ids: Tensor::from_ints([[1]], &device),
                token_mask: None,
                position_ids: None,
            },
            &ForwardOptions {
                attentions: true,
                ..Default::default()
            },
        );
        assert!(matches!(
            attention_request,
            Err(Error::BackendCapability(_))
        ));
    }

    /// Stateless device preflight identifies every independently supplied tensor.
    #[test]
    fn stateless_device_affinity_errors_are_contextual() {
        assert!(ensure_model_device("muse_glimmer.input_ids", &0_u8, &0_u8).is_ok());
        for tensor in [
            "muse_glimmer.input_ids",
            "muse_glimmer.token_mask",
            "muse_glimmer.position_ids",
            "muse_glimmer.lm_head",
        ] {
            let error = ensure_model_device(tensor, &1_u8, &0_u8)
                .expect_err("different device identities must fail preflight");
            assert!(error.to_string().contains(tensor));
            assert!(error.to_string().contains("model parameters"));
        }
    }

    /// Stateless graph construction remains differentiable through embeddings and logits.
    #[cfg(feature = "autodiff")]
    #[test]
    fn autodiff_smoke_reaches_embedding_weights() {
        use burn::backend::Autodiff;

        type AdBackend = Autodiff<TestBackend>;
        let device = Default::default();
        let model = MuseGlimmerForCausalLm::<AdBackend>::init(tiny_config(), &device).unwrap();
        let loss = model
            .forward(
                CausalLmInput {
                    token_ids: Tensor::<AdBackend, 2, Int>::from_ints([[1, 2]], &device),
                    token_mask: None,
                    position_ids: None,
                },
                &ForwardOptions::default(),
            )
            .unwrap()
            .logits
            .sum();
        let gradients = loss.backward();
        assert!(model.model.embed_tokens.weight.grad(&gradients).is_some());
    }

    /// Multi-chunk autodiff fails closed because its tape is not memory-bounded.
    #[cfg(feature = "autodiff")]
    #[test]
    fn bounded_multichunk_autodiff_is_not_mislabeled_as_memory_bounded() {
        use burn::backend::Autodiff;

        type AdBackend = Autodiff<TestBackend>;
        let device = Default::default();
        let model = MuseGlimmerForCausalLm::<AdBackend>::init(tiny_config(), &device)
            .unwrap()
            .with_attention_execution_policy(AttentionExecutionPolicy::query_chunked(1, 64))
            .unwrap();
        let error = model
            .forward(
                CausalLmInput {
                    token_ids: Tensor::<AdBackend, 2, Int>::from_ints([[1, 2]], &device),
                    token_mask: None,
                    position_ids: None,
                },
                &ForwardOptions::default(),
            )
            .expect_err("autodiff retains all chunk intermediates");
        assert!(error.to_string().contains("inference-only"));
        assert!(error.to_string().contains("autodiff tape"));
    }
}
