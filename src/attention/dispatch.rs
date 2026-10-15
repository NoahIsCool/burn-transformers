//! Attention evaluation over native key/value heads.
//!
//! # Mathematics
//!
//! For query head `h`, attention computes
//! `context_h = softmax(scale * Q_h K_kv(h)^T + bias_h, masked) V_kv(h)`,
//! where grouped-query attention maps query head `h` to key/value head
//! `kv(h) = floor(h / g)` with `g = query_heads / key_value_heads`. Query heads
//! of one group are contiguous: `h = kv * g + j` for `j in 0..g`.
//!
//! # Folding groups instead of repeating K/V
//!
//! Every query head in a group reads the same `K_kv` and `V_kv`, so the group's
//! queries can be stacked as extra *rows* of one attention call:
//! `[batch, query_head, query, d] -> [batch, kv_head, g * query, d]`, where
//! folded row `j * query + t` is token `t` of group member `j`. The reshape is
//! a view of a contiguous tensor, masks and biases fold the same way, and the
//! context unfolds back with the inverse reshape. Softmax is row-wise, so the
//! result is mathematically identical to repeating each KV head `g` times, but
//! it never allocates `g` copies of the key/value history. This matters most
//! for cached decoding, where repetition costs `O(g * key * d)` per layer and
//! token while the folded call reads the cache once.
//!
//! Folding needs an explicit row mask whenever causality applies, because a
//! folded row index is not a token position. Square causal planes therefore
//! keep the backend's native `is_causal` flag (fused kernels then skip fully
//! masked tiles) and repeat K/V; every other call folds.
//!
//! # Causal alignment
//!
//! Causality is bottom-right aligned: query row `i` of `Q` rows sits at key
//! position `i + K - Q`. Burn's fallback and Flex kernels agree, but CubeCL's
//! flash kernels (selected by autotuning) align top-left. The native path
//! therefore passes `is_causal` only for square planes, where both alignments
//! coincide, and materializes the bottom-right mask otherwise. One query row
//! (`Q == 1`) sees every key under bottom-right alignment, so causality is a
//! no-op there and costs nothing.
//!
//! # Precision and exclusions
//!
//! [`AttentionPrecisionMode::EagerFp32Softmax`] materializes input-dtype
//! scores, fills exclusions with `-inf`, adds the bias, runs softmax in F32,
//! and casts probabilities back before the value product. Fully excluded rows
//! are defined as zero probability (and zero context). Additive bias must be
//! finite or `-inf`; `+inf`/NaN bias is unsupported. Backend-native kernels
//! define fully excluded rows themselves (Burn's fallback and Flex return zero).
//!
//! # Attention sinks
//!
//! A learned per-head sink logit `s_h` adds one extra softmax column that is
//! dropped after normalization (gpt-oss, StreamingLLM):
//! `p_j = exp(z_j) / (exp(s_h) + sum_k exp(z_k))`. Rows therefore sum to
//! `1 - p_sink`, and a row whose keys are all excluded attends only to the sink
//! and yields zero context. The eager path appends the column explicitly. The
//! native path stays a single backend attention call by appending one zero-value
//! *sink key* per KV head and `g` selector channels: folded row `j * Q + t`
//! carries a one-hot `e_j`, real keys carry zeros, and the sink key carries
//! `s_(kv*g+j) / scale` in channel `j`, so its score is exactly `s_h`. This
//! copies K/V once per call (`O(key * (d + g))`) instead of materializing the
//! score plane; it requires an explicit scale and rejects softcapping, which
//! would otherwise also cap the sink logit.

use burn::tensor::{
    BasicOps, Bool, DType, Int, Tensor, activation::softmax, backend::Backend, module::attention,
    ops::AttentionModuleOptions,
};

use super::{
    AttentionPrecisionMode,
    mask::{bottom_right_causal_mask, union_exclusions},
};
use crate::{Error, Result};

/// Tensors consumed by one attention evaluation over native KV heads.
pub(crate) struct Operands<B: Backend> {
    /// Queries `[batch, query_head, query, head_dim]`.
    pub(crate) query: Tensor<B, 4>,
    /// Native keys `[batch, kv_head, key, head_dim]`.
    pub(crate) key: Tensor<B, 4>,
    /// Native values `[batch, kv_head, key, value_dim]`.
    pub(crate) value: Tensor<B, 4>,
    /// Optional exclusions `[batch, query_head, query, key]`, `true == excluded`.
    pub(crate) mask: Option<Tensor<B, 4, Bool>>,
    /// Optional additive logits `[1 or batch, query_head, query, key]`.
    pub(crate) bias: Option<Tensor<B, 4>>,
    /// Optional learned sink logits `[query_head]` in the query dtype.
    pub(crate) sinks: Option<Tensor<B, 1>>,
}

/// Validated extents of one grouped attention call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct HeadGeometry {
    /// Batch rows.
    pub(crate) batch: usize,
    /// Query heads.
    pub(crate) query_heads: usize,
    /// Native key/value heads; divides `query_heads`.
    pub(crate) key_value_heads: usize,
    /// Query tokens.
    pub(crate) query_sequence: usize,
    /// Key/value tokens.
    pub(crate) key_sequence: usize,
    /// Query/key channels per head.
    pub(crate) head_dim: usize,
}

impl HeadGeometry {
    /// Query heads that share one key/value head.
    pub(crate) const fn groups(self) -> usize {
        self.query_heads / self.key_value_heads
    }

    /// Score-plane extents `[batch, query_head, query, key]`.
    pub(crate) const fn score_shape(self) -> [usize; 4] {
        [
            self.batch,
            self.query_heads,
            self.query_sequence,
            self.key_sequence,
        ]
    }
}

impl<B: Backend> Operands<B> {
    /// Checks every operand against one grouped geometry before backend dispatch.
    pub(crate) fn geometry(&self) -> Result<HeadGeometry> {
        let [batch, query_heads, query_sequence, head_dim] = self.query.dims();
        let [key_batch, key_value_heads, key_sequence, key_dim] = self.key.dims();
        let [value_batch, value_heads, value_sequence, value_dim] = self.value.dims();
        if batch == 0
            || query_heads == 0
            || query_sequence == 0
            || head_dim == 0
            || key_value_heads == 0
            || key_sequence == 0
            || value_dim == 0
        {
            return Err(Error::shape(
                "attention.operands",
                "non-empty query, key, and value axes",
                format!(
                    "query={:?}, key={:?}, value={:?}",
                    self.query.dims(),
                    self.key.dims(),
                    self.value.dims()
                ),
            ));
        }
        if !query_heads.is_multiple_of(key_value_heads) {
            return Err(Error::configuration(
                "attention.head_grouping",
                format!(
                    "query_heads ({query_heads}) must be non-zero and divisible by key_value_heads ({key_value_heads})"
                ),
            ));
        }
        if [key_batch, key_dim] != [batch, head_dim]
            || [value_batch, value_heads, value_sequence] != [batch, key_value_heads, key_sequence]
        {
            return Err(Error::shape(
                "attention.operands",
                format!(
                    "key [{batch}, kv_heads, key, {head_dim}] and value [{batch}, kv_heads, key, value_dim]"
                ),
                format!("key={:?}, value={:?}", self.key.dims(), self.value.dims()),
            ));
        }
        let geometry = HeadGeometry {
            batch,
            query_heads,
            key_value_heads,
            query_sequence,
            key_sequence,
            head_dim,
        };
        let score_shape = geometry.score_shape();
        if let Some(mask) = &self.mask
            && mask.dims() != score_shape
        {
            return Err(Error::shape(
                "attention.mask",
                format!("{score_shape:?}"),
                format!("{:?}", mask.dims()),
            ));
        }
        if let Some(bias) = &self.bias {
            let [bias_batch, bias_heads, bias_query, bias_key] = bias.dims();
            if ![1, batch].contains(&bias_batch)
                || [bias_heads, bias_query, bias_key] != [query_heads, query_sequence, key_sequence]
            {
                return Err(Error::shape(
                    "attention.bias",
                    format!(
                        "[batch in {{1, {batch}}}, {query_heads}, {query_sequence}, {key_sequence}]"
                    ),
                    format!("{:?}", bias.dims()),
                ));
            }
        }
        if let Some(sinks) = &self.sinks
            && sinks.dims() != [query_heads]
        {
            return Err(Error::shape(
                "attention.sinks",
                format!("[{query_heads}]"),
                format!("{:?}", sinks.dims()),
            ));
        }
        Ok(geometry)
    }
}

/// Evaluates attention for query heads grouped over native key/value heads.
///
/// `query` is `[batch, query_head, query, d]`; `key` and `value` keep their
/// native `[batch, kv_head, key, d]` heads, where `kv_head` divides
/// `query_head`. `mask` (`true == excluded`) is `[batch, query_head, query, key]`
/// and `bias` is `[1 or batch, query_head, query, key]`. The result is the
/// per-head context `[batch, query_head, query, value_dim]`.
///
/// Groups are folded into query rows rather than repeating K/V (see the
/// module documentation), and causal attention is bottom-right aligned on
/// every backend. Operands must share one dense floating dtype and device; no
/// implicit dtype conversion is performed.
pub fn grouped_query_attention<B: Backend>(
    query: Tensor<B, 4>,
    key: Tensor<B, 4>,
    value: Tensor<B, 4>,
    mask: Option<Tensor<B, 4, Bool>>,
    bias: Option<Tensor<B, 4>>,
    options: AttentionModuleOptions,
    precision_mode: AttentionPrecisionMode,
) -> Result<Tensor<B, 4>> {
    evaluate(
        Operands {
            query,
            key,
            value,
            mask,
            bias,
            sinks: None,
        },
        options,
        precision_mode,
    )
}

/// Dispatches attention whose K/V already have one head per query head.
///
/// Kept for callers that repeat K/V themselves; prefer
/// [`grouped_query_attention`], which accepts native KV heads directly.
pub(crate) fn dispatch_attention<B: Backend>(
    query: Tensor<B, 4>,
    key: Tensor<B, 4>,
    value: Tensor<B, 4>,
    mask: Option<Tensor<B, 4, Bool>>,
    bias: Option<Tensor<B, 4>>,
    options: AttentionModuleOptions,
    precision_mode: AttentionPrecisionMode,
) -> Result<Tensor<B, 4>> {
    grouped_query_attention(query, key, value, mask, bias, options, precision_mode)
}

/// Validates operands and evaluates them with the selected operation graph.
pub(crate) fn evaluate<B: Backend>(
    operands: Operands<B>,
    options: AttentionModuleOptions,
    precision_mode: AttentionPrecisionMode,
) -> Result<Tensor<B, 4>> {
    let geometry = operands.geometry()?;
    match precision_mode {
        AttentionPrecisionMode::BackendDefault => native(operands, options, geometry),
        AttentionPrecisionMode::EagerFp32Softmax => eager(operands, options, geometry),
    }
}

/// Repeats native KV heads into contiguous query-head groups.
///
/// This materializes `[batch, query_heads, sequence, head_dim]`. Attention in
/// this crate folds groups instead; repetition remains for callers that need
/// an explicit per-query-head K/V tensor and for square native causal calls.
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

// ---------------------------------------------------------------------------
// Eager reference graph
// ---------------------------------------------------------------------------

/// Evaluates the explicitly ordered eager graph over native KV heads.
fn eager<B: Backend>(
    operands: Operands<B>,
    options: AttentionModuleOptions,
    geometry: HeadGeometry,
) -> Result<Tensor<B, 4>> {
    let Operands {
        query,
        key,
        value,
        mask,
        bias,
        sinks,
    } = operands;
    let device = query.device();
    let scale = options
        .scale
        .unwrap_or_else(|| default_scale(geometry.head_dim));
    let scores = scaled_scores(query, key, scale, options.softcap);
    let exclusion = union_exclusions(
        mask,
        causal_exclusion(options.is_causal, geometry, &device)?,
    );
    let probabilities = eager_probabilities(scores, exclusion.as_ref(), bias, sinks);
    Ok(grouped_mix(probabilities, value))
}

/// Returns Burn's default score scale `1 / sqrt(head_dim)`.
pub(crate) fn default_scale(head_dim: usize) -> f64 {
    (head_dim as f64).sqrt().recip()
}

/// Computes `scale * Q K^T`, optionally soft-capped, as `[batch, query_head, query, key]`.
///
/// Keys keep native heads: grouped queries are folded into rows for the
/// matmul and the result is reshaped (a free view) back to per-head scores.
/// Softcapping applies `cap * tanh(scores / cap)` before any exclusion, so
/// `-inf` masking is never mapped to a finite value.
pub(crate) fn scaled_scores<B: Backend>(
    query: Tensor<B, 4>,
    key: Tensor<B, 4>,
    scale: f64,
    softcap: Option<f64>,
) -> Tensor<B, 4> {
    let [batch, query_heads, query_sequence, _] = query.dims();
    let [_, key_value_heads, key_sequence, _] = key.dims();
    let scores = (fold_groups(query, key_value_heads).matmul(key.swap_dims(2, 3)) * scale)
        .reshape([batch, query_heads, query_sequence, key_sequence]);
    match softcap {
        Some(softcap) => (scores / softcap).tanh() * softcap,
        None => scores,
    }
}

/// Returns the bottom-right causal exclusion for a call, or `None` when it is a no-op.
///
/// One query row aligned with the final key sees every key, so `Q == 1`
/// needs no mask at all.
pub(crate) fn causal_exclusion<B: Backend>(
    causal: bool,
    geometry: HeadGeometry,
    device: &B::Device,
) -> Result<Option<Tensor<B, 4, Bool>>> {
    if !causal || geometry.query_sequence == 1 {
        return Ok(None);
    }
    bottom_right_causal_mask(
        geometry.batch,
        geometry.query_heads,
        geometry.query_sequence,
        geometry.key_sequence,
        device,
    )
    .map(Some)
}

/// Applies exclusions, additive bias, optional sinks, and the FP32 softmax.
///
/// Order matches eager reference graphs: excluded scores become `-inf`, bias
/// is added, an optional sink column joins the row, the row is normalized in
/// F32, the sink column is dropped, and probabilities return to the score
/// dtype. `bias` may broadcast over the score plane.
pub(crate) fn eager_probabilities<B: Backend>(
    scores: Tensor<B, 4>,
    exclusion: Option<&Tensor<B, 4, Bool>>,
    bias: Option<Tensor<B, 4>>,
    sinks: Option<Tensor<B, 1>>,
) -> Tensor<B, 4> {
    let scores = match exclusion {
        Some(exclusion) => scores.mask_fill(exclusion.clone(), f32::NEG_INFINITY),
        None => scores,
    };
    let scores = match bias {
        Some(bias) => scores + bias,
        None => scores,
    };
    let Some(sinks) = sinks else {
        return fp32_softmax(scores);
    };
    let [batch, heads, queries, keys] = scores.dims();
    let sink_column = sinks
        .reshape([1, heads, 1, 1])
        .expand([batch, heads, queries, 1]);
    fp32_softmax(Tensor::cat(vec![scores, sink_column], 3)).narrow(3, 0, keys)
}

/// Normalizes the last axis in F32 and defines all-`-inf` rows as zero.
///
/// Callers must already have filled excluded links with `-inf`: `exp(-inf)`
/// then gives those links an exact zero, so no separate exclusion pass is
/// needed. A row with no finite score is unsupported by softmax (it would be
/// NaN); it is zeroed before and after normalization instead. The result is
/// cast back to the input dtype.
pub(crate) fn fp32_softmax<B: Backend>(scores: Tensor<B, 4>) -> Tensor<B, 4> {
    let dtype = scores.dtype();
    let shape = scores.dims();
    let unsupported_row = scores
        .clone()
        .equal_elem(f32::NEG_INFINITY)
        .all_dim(3)
        .expand(shape);
    softmax(
        scores
            .mask_fill(unsupported_row.clone(), 0.0)
            .cast(DType::F32),
        3,
    )
    .mask_fill(unsupported_row, 0.0)
    .cast(dtype)
}

/// Restores exact zeros on excluded links after a pattern edit or transform.
pub(crate) fn zero_excluded<B: Backend>(
    pattern: Tensor<B, 4>,
    exclusion: Option<&Tensor<B, 4, Bool>>,
) -> Tensor<B, 4> {
    match exclusion {
        Some(exclusion) => pattern.mask_fill(exclusion.clone(), 0.0),
        None => pattern,
    }
}

/// Multiplies per-head probabilities by native-head values: `[b, qh, q, k] x [b, kvh, k, dv]`.
pub(crate) fn grouped_mix<B: Backend>(
    probabilities: Tensor<B, 4>,
    value: Tensor<B, 4>,
) -> Tensor<B, 4> {
    let query_heads = probabilities.dims()[1];
    let key_value_heads = value.dims()[1];
    unfold_groups(
        fold_groups(probabilities, key_value_heads).matmul(value),
        query_heads,
    )
}

// ---------------------------------------------------------------------------
// Backend-native dispatch
// ---------------------------------------------------------------------------

/// Evaluates attention with the backend's own kernels over native KV heads.
fn native<B: Backend>(
    operands: Operands<B>,
    options: AttentionModuleOptions,
    geometry: HeadGeometry,
) -> Result<Tensor<B, 4>> {
    let causal = options.is_causal && geometry.query_sequence > 1;
    let explicit_options = AttentionModuleOptions {
        is_causal: false,
        ..options
    };
    if operands.sinks.is_some() {
        return native_with_sinks(operands, explicit_options, causal, geometry);
    }
    let Operands {
        query,
        key,
        value,
        mask,
        bias,
        ..
    } = operands;
    if causal && geometry.query_sequence == geometry.key_sequence {
        // Square planes are the one case where every backend's `is_causal`
        // alignment agrees; fused kernels then skip masked tiles entirely.
        let key = repeat_key_value(key, geometry.query_heads)?;
        let value = repeat_key_value(value, geometry.query_heads)?;
        return Ok(attention(query, key, value, mask, bias, options));
    }
    let exclusion = union_exclusions(mask, causal_exclusion(causal, geometry, &query.device())?);
    let kv_heads = geometry.key_value_heads;
    let context = attention(
        fold_groups(query, kv_heads),
        key,
        value,
        exclusion.map(|mask| fold_groups(mask, kv_heads)),
        bias.map(|bias| fold_groups(bias, kv_heads)),
        explicit_options,
    );
    Ok(unfold_groups(context, geometry.query_heads))
}

/// Evaluates native attention with one appended zero-value sink key per KV head.
///
/// See the module documentation for the selector-channel construction that
/// makes the sink score exactly `s_h` for every folded query row.
fn native_with_sinks<B: Backend>(
    operands: Operands<B>,
    options: AttentionModuleOptions,
    causal: bool,
    geometry: HeadGeometry,
) -> Result<Tensor<B, 4>> {
    if options.softcap.is_some() {
        return Err(Error::BackendCapability(
            "attention sinks cannot be combined with logit softcapping: a fused kernel would also cap the sink logit".into(),
        ));
    }
    let Operands {
        query,
        key,
        value,
        mask,
        bias,
        sinks,
    } = operands;
    let sinks =
        sinks.ok_or_else(|| Error::configuration("attention.sinks", "missing sink logits"))?;
    let HeadGeometry {
        batch,
        query_heads,
        key_value_heads,
        query_sequence,
        key_sequence,
        head_dim,
    } = geometry;
    let groups = geometry.groups();
    let folded_rows = groups * query_sequence;
    let device = query.device();
    let dtype = query.dtype();
    let scale = options.scale.unwrap_or_else(|| default_scale(head_dim));
    let exclusion = union_exclusions(mask, causal_exclusion(causal, geometry, &device)?);

    // Queries: [b, kvh, g*q, d] plus a one-hot group selector [.., .., g*q, g].
    let selector = group_selector::<B>(groups, query_sequence, dtype, &device).expand([
        batch,
        key_value_heads,
        folded_rows,
        groups,
    ]);
    let query = Tensor::cat(vec![fold_groups(query, key_value_heads), selector], 3);

    // Keys: real keys gain g zero channels; each KV head gains one sink key
    // whose channel j holds s_(kv*g + j) / scale.
    let sink_channels = (sinks / scale).reshape([1, key_value_heads, 1, groups]);
    let sink_key = Tensor::cat(
        vec![
            Tensor::zeros([1, key_value_heads, 1, head_dim], (&device, dtype)),
            sink_channels,
        ],
        3,
    )
    .expand([batch, key_value_heads, 1, head_dim + groups]);
    let key = Tensor::cat(
        vec![
            Tensor::cat(
                vec![
                    key,
                    Tensor::zeros(
                        [batch, key_value_heads, key_sequence, groups],
                        (&device, dtype),
                    ),
                ],
                3,
            ),
            sink_key,
        ],
        2,
    );
    let value_dim = value.dims()[3];
    let value = Tensor::cat(
        vec![
            value,
            Tensor::zeros([batch, key_value_heads, 1, value_dim], (&device, dtype)),
        ],
        2,
    );
    // The sink column is never excluded and carries no bias.
    let mask = exclusion.map(|mask| {
        Tensor::cat(
            vec![
                fold_groups(mask, key_value_heads),
                Tensor::<B, 4, Bool>::zeros([batch, key_value_heads, folded_rows, 1], &device),
            ],
            3,
        )
    });
    let bias = bias.map(|bias| {
        let bias_batch = bias.dims()[0];
        Tensor::cat(
            vec![
                fold_groups(bias, key_value_heads),
                Tensor::zeros(
                    [bias_batch, key_value_heads, folded_rows, 1],
                    (&device, dtype),
                ),
            ],
            3,
        )
    });
    let context = attention(
        query,
        key,
        value,
        mask,
        bias,
        AttentionModuleOptions {
            scale: Some(scale),
            softcap: None,
            is_causal: false,
        },
    );
    Ok(unfold_groups(context, query_heads))
}

/// One-hot `[1, 1, g * q, g]` selector: folded row `j * q + t` selects channel `j`.
fn group_selector<B: Backend>(
    groups: usize,
    query_sequence: usize,
    dtype: DType,
    device: &B::Device,
) -> Tensor<B, 4> {
    let rows = (groups * query_sequence) as i64;
    let member = Tensor::<B, 1, Int>::arange(0..rows, device)
        .div_scalar(query_sequence as i64)
        .reshape([groups * query_sequence, 1]);
    let channel = Tensor::<B, 1, Int>::arange(0..groups as i64, device).reshape([1, groups]);
    Tensor::<B, 2>::zeros([groups * query_sequence, groups], (device, dtype))
        .mask_fill(member.equal(channel), 1.0)
        .reshape([1, 1, groups * query_sequence, groups])
}

// ---------------------------------------------------------------------------
// Group folding
// ---------------------------------------------------------------------------

/// Folds `[b, kvh * g, rows, x]` into `[b, kvh, g * rows, x]` (row `j * rows + t`).
///
/// Query heads of one group are contiguous, so this is a view of a contiguous
/// tensor. It is the identity when every query head has its own KV head.
pub(crate) fn fold_groups<B: Backend, K: BasicOps<B>>(
    tensor: Tensor<B, 4, K>,
    key_value_heads: usize,
) -> Tensor<B, 4, K> {
    let [batch, query_heads, rows, width] = tensor.dims();
    if query_heads == key_value_heads {
        return tensor;
    }
    let groups = query_heads / key_value_heads;
    tensor.reshape([batch, key_value_heads, groups * rows, width])
}

/// Inverts [`fold_groups`]: `[b, kvh, g * rows, x]` to `[b, query_heads, rows, x]`.
pub(crate) fn unfold_groups<B: Backend, K: BasicOps<B>>(
    tensor: Tensor<B, 4, K>,
    query_heads: usize,
) -> Tensor<B, 4, K> {
    let [batch, key_value_heads, folded_rows, width] = tensor.dims();
    if query_heads == key_value_heads {
        return tensor;
    }
    let groups = query_heads / key_value_heads;
    tensor.reshape([batch, query_heads, folded_rows / groups, width])
}

#[cfg(test)]
#[path = "dispatch_tests.rs"]
mod tests;
