//! Multi-head latent attention (MLA) projection and mixing kernels.
//!
//! Ordinary multi-head attention caches one key and one value vector per head
//! and token. MLA instead learns a low-rank latent `c_t in R^r` and a single
//! rotary key `k^R_t in R^d_r`. The up-projection can be partitioned per head
//! as `W^K_h in R^(d_n x r)` and `W^V_h in R^(d_v x r)`. Reference prefill
//! expands `k^C_{t,h} = W^K_h c_t` and `v_{t,h} = W^V_h c_t` before ordinary
//! attention. Absorbed decode uses associativity instead:
//!
//! `q^C_h (W^K_h c_t) = (q^C_h W^K_h) c_t`
//!
//! `sum_t p_t (W^V_h c_t) = W^V_h (sum_t p_t c_t)`.
//!
//! Consequently decode never materializes historical per-head keys or values.
//! After absorption every head attends to the *same* cached latents and shared
//! rotary key (multi-query form), so heads fold into the rows of one GEMM per
//! step: the content logits and the latent aggregation each read the `r`
//! cached latent scalars per token once, and the rotary logits read the `d_r`
//! rotary scalars once, independent of the head count. Only the per-head
//! factors `W^K_h` and `W^V_h` remain head-specific. Both paths
//! share the same parameters and are numerically equivalent up to
//! floating-point reassociation.
//!
//! Configuration fields keep the checkpoint vocabulary: `num_heads` is the
//! query-head count, `qk_nope_head_dim`/`qk_rope_head_dim` are the content and
//! rotary widths, `kv_lora_rank`/`q_lora_rank` the latent ranks `r`, and
//! `softmax_scale` the explicit score scale.
//!
//! Latents use `[batch, token, rank]`; content queries use
//! `[batch, head, query_token, content_channel]`, and the shared rotary key has one
//! head. Expanded score/pattern hooks require the reference path; absorbed decode
//! prioritizes compact cached storage. Projection components share dtype/device,
//! with F32 softmax normalization and explicit exclusion masks for padded rows.
//!
//! # Bounded reference attention
//!
//! The reference path materializes one `[batch, head, query, key]` score plane,
//! so a single long prefill grows quadratically. `PositionalVisibility` (crate-private) describes
//! causal and padding visibility from token positions instead of a materialized
//! mask, which lets the reference path evaluate synchronized query chunks under
//! an [`AttentionExecutionPolicy`]: every chunk still sees its complete causal
//! key prefix (an exact storage bound, not an approximation), and keys past the
//! chunk's last visible position are never scored.

mod geometry;

use core::ops::Range;

use burn::{
    module::Module,
    nn::{Linear, LinearConfig, LinearLayout},
    tensor::{Bool, Int, Tensor, backend::Backend},
};

use crate::{
    Error, Result,
    attention::{
        AttentionExecutionPolicy, bottom_right_causal_mask, causal_mask_needed,
        eager_probabilities, union_exclusions, zero_excluded,
    },
    intervention::{AttentionPoints, Hooks, PointInfo},
    normalization::{RmsNorm, RmsNormConfig},
    operations,
    position::{
        RopeConfig, RotaryAngles, apply_rotary, apply_rotary_to, interleaved_to_split_half,
        rotary_angles,
    },
};
use geometry::{
    MlaAxis, MlaAxisFault, MlaMatrix, MlaWidthFault, chunk_key_end, key_limits_valid,
    mla_axis_fault, mla_widths, saturating_window, suffix_key_limits,
};

/// Shape-bearing configuration for one multi-head latent-attention layer.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct MultiLatentAttentionConfig {
    /// Residual-stream width accepted and returned by the layer.
    pub hidden_size: usize,
    /// Number of expanded attention heads.
    pub num_heads: usize,
    /// Per-head content query/key width not transformed by RoPE.
    pub qk_nope_head_dim: usize,
    /// Per-head rotary query/key width.
    pub qk_rope_head_dim: usize,
    /// Per-head value width.
    pub value_head_dim: usize,
    /// Width of the cached key/value latent.
    pub kv_lora_rank: usize,
    /// Optional query low-rank bottleneck; absence selects one direct projection.
    pub q_lora_rank: Option<usize>,
    /// RMS epsilon for normalized query and key/value latents.
    pub rms_norm_epsilon: f64,
    /// Whether the first-stage latent projections and output projection use bias.
    pub bias: bool,
    /// Scalar applied to content-plus-rotary attention logits.
    pub softmax_scale: f64,
}

impl MultiLatentAttentionConfig {
    /// Creates an MLA configuration with the standard inverse-root Q/K scale.
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        qk_nope_head_dim: usize,
        qk_rope_head_dim: usize,
        value_head_dim: usize,
        kv_lora_rank: usize,
        q_lora_rank: Option<usize>,
    ) -> Self {
        let qk_head_dim = qk_nope_head_dim.saturating_add(qk_rope_head_dim);
        Self {
            hidden_size,
            num_heads,
            qk_nope_head_dim,
            qk_rope_head_dim,
            value_head_dim,
            kv_lora_rank,
            q_lora_rank,
            rms_norm_epsilon: 1e-6,
            bias: false,
            softmax_scale: (qk_head_dim as f64).sqrt().recip(),
        }
    }

    /// Returns the full per-head query/key width.
    pub fn qk_head_dim(&self) -> Result<usize> {
        self.qk_nope_head_dim
            .checked_add(self.qk_rope_head_dim)
            .ok_or_else(|| Error::configuration("mla.qk_head_dim", "dimension sum overflows usize"))
    }

    /// Checks the flattened query-head width before constructing a projection shape.
    fn query_projection_width(&self) -> Result<usize> {
        self.num_heads
            .checked_mul(self.qk_head_dim()?)
            .ok_or_else(|| Error::configuration("mla.query_width", "head product overflows usize"))
    }

    /// Returns the width emitted by the joint KV-A projection.
    pub fn compressed_projection_width(&self) -> Result<usize> {
        self.kv_lora_rank
            .checked_add(self.qk_rope_head_dim)
            .ok_or_else(|| {
                Error::configuration(
                    "mla.compressed_projection_width",
                    "dimension sum overflows usize",
                )
            })
    }

    /// Returns the width emitted by the KV-B expansion projection.
    pub fn expanded_kv_width(&self) -> Result<usize> {
        let per_head = self
            .qk_nope_head_dim
            .checked_add(self.value_head_dim)
            .ok_or_else(|| {
                Error::configuration("mla.expanded_kv_width", "per-head sum overflows usize")
            })?;
        self.num_heads.checked_mul(per_head).ok_or_else(|| {
            Error::configuration("mla.expanded_kv_width", "head product overflows usize")
        })
    }

    /// Validates projection extents, matrix element counts, and numerical invariants.
    ///
    /// Checks matrix products before Burn's lazy parameters can defer an overflow
    /// to their first materialization. Valid geometry does not guarantee sufficient
    /// backend memory for the requested tensors. The integer checks are the
    /// Verus-verified `mla_axis_fault` and `mla_widths` (`verus/mla.rs`): success
    /// implies exact widths and representable element counts for every projection.
    pub fn validate(&self) -> Result<()> {
        if let Some(fault) = mla_axis_fault(
            self.hidden_size,
            self.num_heads,
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            self.value_head_dim,
            self.kv_lora_rank,
            self.q_lora_rank,
        ) {
            return Err(match fault {
                MlaAxisFault::Zero(axis) => {
                    let field = match axis {
                        MlaAxis::HiddenSize => "hidden_size",
                        MlaAxis::NumHeads => "num_heads",
                        MlaAxis::QkNopeHeadDim => "qk_nope_head_dim",
                        MlaAxis::QkRopeHeadDim => "qk_rope_head_dim",
                        MlaAxis::ValueHeadDim => "value_head_dim",
                        MlaAxis::KvLoraRank => "kv_lora_rank",
                    };
                    Error::configuration(format!("mla.{field}"), "must be non-zero")
                }
                MlaAxisFault::OddRotary => Error::configuration(
                    "mla.qk_rope_head_dim",
                    "must be even because RoPE rotates channel pairs",
                ),
                MlaAxisFault::ZeroQueryRank => {
                    Error::configuration("mla.q_lora_rank", "must be non-zero when present")
                }
            });
        }
        if !self.rms_norm_epsilon.is_finite() || self.rms_norm_epsilon <= 0.0 {
            return Err(Error::configuration(
                "mla.rms_norm_epsilon",
                "must be finite and positive",
            ));
        }
        if !self.softmax_scale.is_finite() || self.softmax_scale <= 0.0 {
            return Err(Error::configuration(
                "mla.softmax_scale",
                "must be finite and positive",
            ));
        }
        // Burn parameters initialize lazily. Reject impossible widths and matrix
        // element counts here, rather than overflowing when tensors are first read.
        mla_widths(
            self.hidden_size,
            self.num_heads,
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            self.value_head_dim,
            self.kv_lora_rank,
            self.q_lora_rank,
        )
        .map(|_| ())
        .map_err(mla_width_error)
    }

    /// Allocates all MLA projections in checkpoint-native column layout.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Result<MultiLatentAttention<B>> {
        self.validate()?;
        let query_width = self.query_projection_width()?;
        let projection = |input, output, bias| {
            operations::init_linear(
                &LinearConfig::new(input, output)
                    .with_bias(bias)
                    .with_layout(LinearLayout::Col),
                device,
            )
        };
        let (q_proj, q_a_proj, q_a_layernorm, q_b_proj) = match self.q_lora_rank {
            None => (
                Some(projection(self.hidden_size, query_width, false)),
                None,
                None,
                None,
            ),
            Some(rank) => (
                None,
                Some(projection(self.hidden_size, rank, self.bias)),
                Some(
                    RmsNormConfig::new(rank)
                        .with_epsilon(self.rms_norm_epsilon)
                        .with_round_before_scale(true)
                        .init(device)?,
                ),
                Some(projection(rank, query_width, false)),
            ),
        };
        Ok(MultiLatentAttention {
            q_proj,
            q_a_proj,
            q_a_layernorm,
            q_b_proj,
            kv_a_proj_with_mqa: projection(
                self.hidden_size,
                self.compressed_projection_width()?,
                self.bias,
            ),
            kv_a_layernorm: RmsNormConfig::new(self.kv_lora_rank)
                .with_epsilon(self.rms_norm_epsilon)
                .with_round_before_scale(true)
                .init(device)?,
            kv_b_proj: projection(self.kv_lora_rank, self.expanded_kv_width()?, false),
            o_proj: projection(
                self.num_heads * self.value_head_dim,
                self.hidden_size,
                self.bias,
            ),
            config: self.clone(),
        })
    }
}

/// Projected current-token components retained before attention mixing.
#[derive(Clone, Debug)]
pub struct MlaProjection<B: Backend> {
    /// Per-head query channels whose dot products use compressed content keys.
    pub query_content: Tensor<B, 4>,
    /// Per-head positional query channels `[batch, head, query, rotary]` in
    /// split-half layout. Unrotated after [`MultiLatentAttention::project`];
    /// rotated once [`MlaProjection::with_rotary`] has been applied.
    pub query_rotary: Tensor<B, 4>,
    /// Normalized token latent `[batch, key, rank]` persisted in the decode cache.
    pub normalized_latent: Tensor<B, 3>,
    /// Single shared positional key `[batch, 1, key, rotary]` in split-half layout.
    ///
    /// Despite its historical name, this key is *unrotated* after
    /// [`MultiLatentAttention::project`] and rotated only once
    /// [`MlaProjection::with_rotary`] has been applied.
    pub rotated_key: Tensor<B, 4>,
}

impl<B: Backend> MlaProjection<B> {
    /// Rotates the positional channels using independent query and source positions.
    ///
    /// This supports cross-attention with different query and source lengths.
    /// Omitting this operation gives ordinary unrotated attention; an additive
    /// relative-position bias can instead be supplied to attention. Content
    /// channels and the compressed KV latent are unchanged.
    pub fn with_rotary(
        mut self,
        query_positions: Tensor<B, 2, Int>,
        key_positions: Tensor<B, 2, Int>,
        rope: RopeConfig,
        evaluated_context: usize,
    ) -> Result<Self> {
        let query_angles = rotary_angles(rope, query_positions, evaluated_context)?;
        let key_angles = rotary_angles(rope, key_positions, evaluated_context)?;
        self.query_rotary = apply_rotary_to(self.query_rotary, &query_angles)?;
        self.rotated_key = apply_rotary_to(self.rotated_key, &key_angles)?;
        Ok(self)
    }
}

/// Attention output and optional probability diagnostics.
#[derive(Clone, Debug)]
pub(crate) struct MlaMixOutput<B: Backend> {
    /// Output-projected attention mixture in residual-stream layout.
    pub(crate) hidden_state: Tensor<B, 3>,
    /// Softmax probabilities retained only when diagnostics request them.
    pub(crate) probabilities: Option<Tensor<B, 4>>,
}

/// Trainable multi-head latent-attention module.
#[derive(Module, Debug)]
pub struct MultiLatentAttention<B: Backend> {
    /// Direct query projection used when `q_lora_rank` is absent.
    pub q_proj: Option<Linear<B>>,
    /// Query down-projection used by low-rank-query checkpoints.
    pub q_a_proj: Option<Linear<B>>,
    /// Normalization applied between the two query projections.
    pub q_a_layernorm: Option<RmsNorm<B>>,
    /// Query up-projection used by low-rank-query checkpoints.
    pub q_b_proj: Option<Linear<B>>,
    /// Joint KV-latent and shared rotary-key projection.
    pub kv_a_proj_with_mqa: Linear<B>,
    /// Normalization applied to the cacheable KV latent.
    pub kv_a_layernorm: RmsNorm<B>,
    /// Expansion into per-head content keys and values.
    pub kv_b_proj: Linear<B>,
    /// Projection from concatenated per-head values to the residual stream.
    pub o_proj: Linear<B>,
    /// Validated shape metadata excluded from parameter records.
    #[module(skip)]
    pub config: MultiLatentAttentionConfig,
}

impl<B: Backend> MultiLatentAttention<B> {
    /// Projects self-attention without applying a positional transformation.
    pub fn project_self(&self, hidden: Tensor<B, 3>) -> Result<MlaProjection<B>> {
        self.project(hidden.clone(), hidden)
    }

    /// Projects independent query and key/value source states.
    ///
    /// Both inputs have the configured residual width; source and query lengths
    /// may differ. The returned compressed latent is normalized before K/V
    /// expansion. Positional channels use split-half layout but remain unrotated;
    /// apply [`MlaProjection::with_rotary`] or edit the projection explicitly.
    pub fn project(
        &self,
        query_hidden: Tensor<B, 3>,
        key_value_hidden: Tensor<B, 3>,
    ) -> Result<MlaProjection<B>> {
        self.project_with_hooks(
            query_hidden,
            key_value_hidden,
            crate::intervention::Layer::decoder(0).self_attention(),
            &mut Hooks::new(),
        )
    }

    /// Projects states while exposing an optional normalized query bottleneck.
    ///
    /// Run this inside the same [`Hooks::run`] scope as [`Self::attend_with_hooks`].
    /// The query-latent callback runs before the query up-projection; the KV
    /// latent callback belongs to attention, immediately before K/V expansion.
    pub fn project_with_hooks(
        &self,
        query_hidden: Tensor<B, 3>,
        key_value_hidden: Tensor<B, 3>,
        points: AttentionPoints,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<MlaProjection<B>> {
        let [batch, query_sequence, query_width] = query_hidden.dims();
        let [key_batch, key_sequence, key_width] = key_value_hidden.dims();
        if batch == 0 || query_sequence == 0 || query_width != self.config.hidden_size {
            return Err(Error::shape(
                "mla.query_input",
                format!("[batch>0, query>0, {}]", self.config.hidden_size),
                format!("{:?}", query_hidden.dims()),
            ));
        }
        if key_batch != batch || key_sequence == 0 || key_width != self.config.hidden_size {
            return Err(Error::shape(
                "mla.key_value_input",
                format!("[{batch}, key>0, {}]", self.config.hidden_size),
                format!("{:?}", key_value_hidden.dims()),
            ));
        }
        if query_hidden.dtype() != key_value_hidden.dtype()
            || query_hidden.device() != key_value_hidden.device()
        {
            return Err(Error::configuration(
                "mla.inputs",
                "query and source must share dtype and device",
            ));
        }
        let query = match (
            &self.q_proj,
            &self.q_a_proj,
            &self.q_a_layernorm,
            &self.q_b_proj,
        ) {
            (Some(direct), None, None, None) => operations::linear(query_hidden, direct)?,
            (None, Some(down), Some(norm), Some(up)) => {
                let latent = norm.forward(operations::linear(query_hidden, down)?);
                let latent = hooks.apply(points.query_latent(), latent)?;
                operations::linear(latent, up)?
            }
            _ => {
                return Err(Error::Checkpoint(
                    "MLA query projection topology disagrees with q_lora_rank".into(),
                ));
            }
        };
        let query = query
            .reshape([
                batch,
                query_sequence,
                self.config.num_heads,
                self.config.qk_head_dim()?,
            ])
            .swap_dims(1, 2);
        let query_content = query.clone().narrow(3, 0, self.config.qk_nope_head_dim);
        // Checkpoints interleave each rotary pair's real and imaginary parts;
        // the shared split-half rotation needs them in two contiguous halves.
        let query_rotary = interleaved_to_split_half(query.narrow(
            3,
            self.config.qk_nope_head_dim,
            self.config.qk_rope_head_dim,
        ))?;
        let compressed = operations::linear(key_value_hidden, &self.kv_a_proj_with_mqa)?;
        let normalized_latent =
            self.kv_a_layernorm
                .forward(compressed.clone().narrow(2, 0, self.config.kv_lora_rank));
        let rotated_key = interleaved_to_split_half(
            compressed
                .narrow(2, self.config.kv_lora_rank, self.config.qk_rope_head_dim)
                .reshape([batch, key_sequence, 1, self.config.qk_rope_head_dim])
                .swap_dims(1, 2),
        )?;
        Ok(MlaProjection {
            query_content,
            query_rotary,
            normalized_latent,
            rotated_key,
        })
    }

    /// Lists points emitted by projection followed by attention.
    ///
    /// The latent has axes `[batch, source_token, latent_channel]`; it is emitted
    /// before expansion, so modifying it changes both expanded keys and values.
    /// MLA uses an explicit F32-softmax reference graph and supports all six
    /// standard attention points without a precision-mode switch.
    pub fn activation_points(&self, points: AttentionPoints) -> Vec<PointInfo> {
        let mut catalog = self.projection_points(points);
        catalog.extend(self.attention_points(points));
        catalog
    }

    /// Lists points visited by [`Self::project_with_hooks`] alone.
    ///
    /// Direct-query configurations have no low-rank query bottleneck and return
    /// an empty catalog. Its absence is explicit rather than a synthetic tensor.
    pub fn projection_points(&self, points: AttentionPoints) -> Vec<PointInfo> {
        if self.config.q_lora_rank.is_some() {
            vec![points.query_latent().info()]
        } else {
            Vec::new()
        }
    }

    /// Lists points visited by [`Self::attend_with_hooks`] alone.
    pub fn attention_points(&self, points: AttentionPoints) -> Vec<PointInfo> {
        let mut catalog = vec![points.kv_latent().info()];
        catalog.extend(points.all());
        catalog
    }

    /// Evaluates stateless MLA, optionally excluding future query-to-key links.
    ///
    /// Query and source lengths may differ. `mask` uses `true == excluded` and
    /// has shape `[batch, head, query, key]`; bias may broadcast along any axis.
    /// Causal alignment is bottom-right, useful when queries form a key suffix.
    /// The graph expands K/V, forms input-dtype scores, softmaxes in F32, casts
    /// weights back, and multiplies by V. Its score-plane memory is quadratic.
    pub fn attend(
        &self,
        projection: MlaProjection<B>,
        mask: Option<Tensor<B, 4, Bool>>,
        bias: Option<Tensor<B, 4>>,
        causal: bool,
    ) -> Result<Tensor<B, 3>> {
        self.attend_with_hooks(
            projection,
            mask,
            bias,
            causal,
            crate::intervention::Layer::decoder(0).self_attention(),
            &mut Hooks::new(),
        )
    }

    /// Evaluates MLA with edits to its compressed latent and attention tensors.
    ///
    /// Emit this operation inside the caller's [`Hooks::run`] scope. The latent
    /// callback runs before its K/V expansion. Q/K/V callbacks receive expanded
    /// heads (Q/K combine content and positional channels). Scores follow scaling
    /// and precede additive bias and exclusions. Pattern edits need not remain
    /// normalized; exclusions are restored before every subsequent capture.
    /// Head context precedes the output projection and has `value_head_dim`
    /// channels. Captures alone preserve this explicit reference graph.
    pub fn attend_with_hooks(
        &self,
        projection: MlaProjection<B>,
        mask: Option<Tensor<B, 4, Bool>>,
        bias: Option<Tensor<B, 4>>,
        causal: bool,
        points: AttentionPoints,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<Tensor<B, 3>> {
        self.validate_projection(&projection, mask.as_ref(), bias.as_ref())?;
        let [batch, heads, query_sequence, _] = projection.query_content.dims();
        let key_sequence = projection.normalized_latent.dims()[1];
        let latent = hooks.apply(points.kv_latent(), projection.normalized_latent)?;
        let (content_keys, values) = self.expand_latents(latent)?;
        let rotary_keys = projection.rotated_key.expand([
            batch,
            heads,
            key_sequence,
            self.config.qk_rope_head_dim,
        ]);
        let query = hooks.apply(
            points.query(),
            Tensor::cat(vec![projection.query_content, projection.query_rotary], 3),
        )?;
        let keys = hooks.apply(
            points.key(),
            Tensor::cat(vec![content_keys, rotary_keys], 3),
        )?;
        let values = hooks.apply(points.value(), values)?;
        let scores = query.matmul(keys.swap_dims(2, 3)) * self.config.softmax_scale;
        let scores = hooks.apply(points.scores(), scores)?;
        // One query row aligned with the final key sees every key, so causality
        // is a no-op there and needs no mask.
        // Verified (`causal_mask_needed`): `query_sequence >= 1` here, so this is `Q > 1`.
        let causal_mask = if causal_mask_needed(causal, query_sequence) {
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
        let exclusion = union_exclusions(mask, causal_mask);
        let pattern = eager_probabilities(scores, exclusion.as_ref(), bias, None);
        let pattern = hooks.apply_constrained(points.pattern(), pattern, |pattern| {
            Ok(zero_excluded(pattern, exclusion.as_ref()))
        })?;
        let context = hooks.apply(points.head_context(), pattern.matmul(values))?;
        let context = context.swap_dims(1, 2).reshape([
            batch,
            query_sequence,
            heads * self.config.value_head_dim,
        ]);
        operations::linear(context, &self.o_proj)
    }

    /// Checks latent/query geometry and shared device/dtype before MLA tensor operations.
    ///
    /// The score plane is `[batch, head, query_token, key_token]`; masks and biases must
    /// match it under the supported broadcast rules. Content and rotary widths remain
    /// separate so absorbed and expanded paths share the same validated contract.
    fn validate_projection(
        &self,
        projection: &MlaProjection<B>,
        mask: Option<&Tensor<B, 4, Bool>>,
        bias: Option<&Tensor<B, 4>>,
    ) -> Result<()> {
        let [batch, heads, queries, content] = projection.query_content.dims();
        let [key_batch, keys, rank] = projection.normalized_latent.dims();
        if batch == 0
            || queries == 0
            || keys == 0
            || heads != self.config.num_heads
            || content != self.config.qk_nope_head_dim
            || key_batch != batch
            || rank != self.config.kv_lora_rank
            || projection.query_rotary.dims()
                != [batch, heads, queries, self.config.qk_rope_head_dim]
            || projection.rotated_key.dims() != [batch, 1, keys, self.config.qk_rope_head_dim]
        {
            return Err(Error::shape(
                "mla.projection",
                "non-empty projections matching the configured head and latent geometry",
                format!(
                    "query={:?}, rotary_query={:?}, latent={:?}, rotary_key={:?}",
                    projection.query_content.dims(),
                    projection.query_rotary.dims(),
                    projection.normalized_latent.dims(),
                    projection.rotated_key.dims()
                ),
            ));
        }
        let shape = [batch, heads, queries, keys];
        let device = projection.query_content.device();
        let dtype = projection.query_content.dtype();
        if projection.query_rotary.device() != device
            || projection.normalized_latent.device() != device
            || projection.rotated_key.device() != device
            || projection.query_rotary.dtype() != dtype
            || projection.normalized_latent.dtype() != dtype
            || projection.rotated_key.dtype() != dtype
        {
            return Err(Error::configuration(
                "mla.projection",
                "all components must share dtype and device",
            ));
        }
        if let Some(mask) = mask {
            if mask.dims() != shape {
                return Err(Error::shape(
                    "mla.attention_mask",
                    format!("{shape:?}"),
                    format!("{:?}", mask.dims()),
                ));
            }
            if mask.device() != device {
                return Err(Error::configuration(
                    "mla.attention_mask",
                    "mask must share the query device",
                ));
            }
        }
        if let Some(bias) = bias {
            if bias
                .dims()
                .iter()
                .zip(shape)
                .any(|(&actual, expected)| actual != 1 && actual != expected)
            {
                return Err(Error::shape(
                    "mla.attention_bias",
                    format!("broadcast-compatible with {shape:?}"),
                    format!("{:?}", bias.dims()),
                ));
            }
            if bias.device() != device || bias.dtype() != dtype {
                return Err(Error::configuration(
                    "mla.attention_bias",
                    "bias must share query dtype and device",
                ));
            }
        }
        Ok(())
    }

    /// Projects one self-attention segment and rotates its positional channels.
    ///
    /// Equivalent to [`Self::project`] followed by [`MlaProjection::with_rotary`]
    /// with identical query and key positions, but rotates both in one pass.
    /// Models build their tables once per forward and call
    /// [`Self::project_with_angles`]; this test convenience builds them per call.
    #[cfg(test)]
    pub(crate) fn project_rotary(
        &self,
        hidden: Tensor<B, 3>,
        positions: Tensor<B, 2, Int>,
        rope: RopeConfig,
        evaluated_context: usize,
    ) -> Result<MlaProjection<B>> {
        self.validate_self_input(&hidden)?;
        let [batch, sequence, _] = hidden.dims();
        if positions.dims() != [batch, sequence] {
            return Err(Error::shape(
                "mla.position_ids",
                format!("[{batch}, {sequence}]"),
                format!("{:?}", positions.dims()),
            ));
        }
        let angles = rotary_angles(rope, positions, evaluated_context)?;
        self.project_with_angles(hidden, &angles)
    }

    /// Projects one self-attention segment and rotates it with precomputed tables.
    ///
    /// `angles` are `[batch, 1, sequence, rotary]` tables for the segment's
    /// positions. Every layer of a model shares one table per forward pass, so
    /// computing it once (and casting it once with [`RotaryAngles::to_dtype`])
    /// removes a frequency evaluation, an upload, and an outer product from
    /// each layer. The query rotary channels and the shared key rotate in one
    /// joint pass.
    pub(crate) fn project_with_angles(
        &self,
        hidden: Tensor<B, 3>,
        angles: &RotaryAngles<B>,
    ) -> Result<MlaProjection<B>> {
        self.project_with_angles_and_hooks(
            hidden,
            angles,
            crate::intervention::Layer::decoder(0).self_attention(),
            &mut Hooks::new(),
        )
    }

    /// [`Self::project_with_angles`] that also emits the optional query-latent point.
    ///
    /// Call inside the caller's [`Hooks::run`] scope, as for
    /// [`Self::project_with_hooks`]; the query latent is emitted before the
    /// query up-projection and the rotation.
    pub(crate) fn project_with_angles_and_hooks(
        &self,
        hidden: Tensor<B, 3>,
        angles: &RotaryAngles<B>,
        points: AttentionPoints,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<MlaProjection<B>> {
        self.validate_self_input(&hidden)?;
        // Self-attention projection shares `project`'s code; one joint rotation
        // then applies the same positions to the queries and the shared key.
        let projection = self.project_with_hooks(hidden.clone(), hidden, points, hooks)?;
        let (query_rotary, rotated_key) =
            apply_rotary(projection.query_rotary, projection.rotated_key, angles)?;
        Ok(MlaProjection {
            query_rotary,
            rotated_key,
            ..projection
        })
    }

    /// Requires a non-empty `[batch, sequence, hidden_size]` self-attention input.
    fn validate_self_input(&self, hidden: &Tensor<B, 3>) -> Result<()> {
        let [batch, sequence, hidden_width] = hidden.dims();
        if batch == 0 || sequence == 0 || hidden_width != self.config.hidden_size {
            return Err(Error::shape(
                "mla.input",
                format!("[batch>0, sequence>0, {}]", self.config.hidden_size),
                format!("[{batch}, {sequence}, {hidden_width}]"),
            ));
        }
        Ok(())
    }

    /// Runs the reference unabsorbed path, expanding all historical K/V heads.
    pub(crate) fn mix_unabsorbed(
        &self,
        projection: &MlaProjection<B>,
        all_latents: Tensor<B, 3>,
        all_rotary_keys: Tensor<B, 4>,
        mask: Option<Tensor<B, 4, Bool>>,
        retain_probabilities: bool,
    ) -> Result<MlaMixOutput<B>> {
        let [batch, heads, query_sequence, _] = projection.query_content.dims();
        let [latent_batch, key_sequence, latent_width] = all_latents.dims();
        if latent_batch != batch
            || latent_width != self.config.kv_lora_rank
            || all_rotary_keys.dims() != [batch, 1, key_sequence, self.config.qk_rope_head_dim]
        {
            return Err(Error::shape(
                "mla.unabsorbed_history",
                format!(
                    "latents [{batch}, key_sequence, {}], rotary [{batch}, 1, key_sequence, {}]",
                    self.config.kv_lora_rank, self.config.qk_rope_head_dim
                ),
                format!(
                    "latents={:?}, rotary={:?}",
                    all_latents.dims(),
                    all_rotary_keys.dims()
                ),
            ));
        }
        let (content_keys, values) = self.expand_latents(all_latents)?;
        let rotary_keys =
            all_rotary_keys.expand([batch, heads, key_sequence, self.config.qk_rope_head_dim]);
        let query = Tensor::cat(
            vec![
                projection.query_content.clone(),
                projection.query_rotary.clone(),
            ],
            3,
        );
        let keys = Tensor::cat(vec![content_keys, rotary_keys], 3);
        let logits = query.matmul(keys.swap_dims(2, 3)) * self.config.softmax_scale;
        let probabilities = attention_probabilities(logits, mask)?;
        let mixed = probabilities
            .clone()
            .matmul(values)
            .swap_dims(1, 2)
            .reshape([batch, query_sequence, heads * self.config.value_head_dim]);
        Ok(MlaMixOutput {
            hidden_state: operations::linear(mixed, &self.o_proj)?,
            probabilities: retain_probabilities.then_some(probabilities),
        })
    }

    /// Runs the reference path in synchronized query chunks bounded by `policy`.
    ///
    /// Mathematically identical to [`Self::mix_unabsorbed`] with the full mask
    /// of [`PositionalVisibility::exclusion`]: the expanded keys `[batch, head, key,
    /// qk]` and values are formed once, then query rows `rows` attend only to
    /// the key prefix `0..visibility.key_end(rows)` (keys beyond it are
    /// causally invisible to every row of the chunk), so each score, mask, and
    /// F32 softmax plane holds at most `batch * heads * rows * key_end`
    /// elements. Multi-chunk execution synchronizes the backend between chunks
    /// so dropped planes are reclaimed. On autodiff backends the tape retains
    /// every chunk anyway, so one unbounded call is evaluated instead (same
    /// values, no false bound); this keeps training on long sequences working
    /// with the default policy.
    ///
    /// `all_latents` is `[batch, key, rank]` and `all_rotary_keys`
    /// `[batch, 1, key, rotary]` (already rotated).
    ///
    /// # Errors
    ///
    /// Returns [`Error::Shape`] for inconsistent history or visibility
    /// extents, the policy's planning errors, and a backend error when a chunk
    /// synchronization fails.
    pub(crate) fn mix_unabsorbed_bounded(
        &self,
        projection: &MlaProjection<B>,
        all_latents: Tensor<B, 3>,
        all_rotary_keys: Tensor<B, 4>,
        visibility: &PositionalVisibility<B>,
        policy: AttentionExecutionPolicy,
    ) -> Result<Tensor<B, 3>> {
        let [batch, heads, query_sequence, _] = projection.query_content.dims();
        let [latent_batch, key_sequence, latent_width] = all_latents.dims();
        if latent_batch != batch
            || latent_width != self.config.kv_lora_rank
            || all_rotary_keys.dims() != [batch, 1, key_sequence, self.config.qk_rope_head_dim]
        {
            return Err(Error::shape(
                "mla.unabsorbed_history",
                format!(
                    "latents [{batch}, key_sequence, {}], rotary [{batch}, 1, key_sequence, {}]",
                    self.config.kv_lora_rank, self.config.qk_rope_head_dim
                ),
                format!(
                    "latents={:?}, rotary={:?}",
                    all_latents.dims(),
                    all_rotary_keys.dims()
                ),
            ));
        }
        visibility.validate(batch, query_sequence, key_sequence)?;
        let (content_keys, values) = self.expand_latents(all_latents)?;
        let rotary_keys =
            all_rotary_keys.expand([batch, heads, key_sequence, self.config.qk_rope_head_dim]);
        let query = Tensor::cat(
            vec![
                projection.query_content.clone(),
                projection.query_rotary.clone(),
            ],
            3,
        );
        let keys = Tensor::cat(vec![content_keys, rotary_keys], 3);
        let device = query.device();
        // An autodiff tape keeps every chunk's planes alive until backward, so
        // chunking would cost synchronizations without bounding memory.
        let policy = if B::ad_enabled(&device) {
            AttentionExecutionPolicy::Unbounded
        } else {
            policy
        };
        let plan = policy.plan(batch, heads, query_sequence, key_sequence)?;
        let chunked = plan.chunk_count() > 1;
        let mut contexts = Vec::with_capacity(plan.chunk_count());
        for rows in plan.query_ranges() {
            let key_end = visibility.key_end(&rows);
            let exclusion = visibility.exclusion(batch, heads, rows.clone(), 0..key_end);
            let scores = query
                .clone()
                .narrow(2, rows.start, rows.len())
                .matmul(keys.clone().narrow(2, 0, key_end).swap_dims(2, 3))
                * self.config.softmax_scale;
            let probabilities = eager_probabilities(scores, Some(&exclusion), None, None);
            contexts.push(probabilities.matmul(values.clone().narrow(2, 0, key_end)));
            if chunked {
                B::sync(&device).map_err(|error| {
                    Error::BackendCapability(format!(
                        "failed to synchronize a bounded MLA attention chunk: {error}"
                    ))
                })?;
                B::memory_cleanup(&device);
            }
        }
        let mixed = Tensor::cat(contexts, 2).swap_dims(1, 2).reshape([
            batch,
            query_sequence,
            heads * self.config.value_head_dim,
        ]);
        operations::linear(mixed, &self.o_proj)
    }

    /// Runs the absorbed path over compressed history without expanded K/V tensors.
    ///
    /// `all_latents` is `[batch, key, rank]` and `all_rotary_keys` is
    /// `[batch, 1, key, rotary]`. Heads fold into GEMM rows, so each cached
    /// scalar is read once per call regardless of the head count.
    pub(crate) fn mix_absorbed(
        &self,
        projection: &MlaProjection<B>,
        all_latents: Tensor<B, 3>,
        all_rotary_keys: Tensor<B, 4>,
        mask: Option<Tensor<B, 4, Bool>>,
        retain_probabilities: bool,
    ) -> Result<MlaMixOutput<B>> {
        let [batch, heads, query_sequence, content_width] = projection.query_content.dims();
        let [latent_batch, key_sequence, latent_width] = all_latents.dims();
        if latent_batch != batch
            || heads != self.config.num_heads
            || content_width != self.config.qk_nope_head_dim
            || latent_width != self.config.kv_lora_rank
            || all_rotary_keys.dims() != [batch, 1, key_sequence, self.config.qk_rope_head_dim]
        {
            return Err(Error::shape(
                "mla.absorbed_history",
                "projection and compressed history matching the configured MLA geometry",
                format!(
                    "query={:?}, latents={:?}, rotary={:?}",
                    projection.query_content.dims(),
                    all_latents.dims(),
                    all_rotary_keys.dims()
                ),
            ));
        }
        let per_head = self.config.qk_nope_head_dim + self.config.value_head_dim;
        let weight = self.kv_b_proj.weight.val();
        if weight.dims() != [self.config.kv_lora_rank, heads * per_head] {
            return Err(Error::shape(
                "mla.kv_b_proj.weight",
                format!("[{}, {}]", self.config.kv_lora_rank, heads * per_head),
                format!("{:?}", weight.dims()),
            ));
        }
        let rank = self.config.kv_lora_rank;
        let content = self.config.qk_nope_head_dim;
        let rotary = self.config.qk_rope_head_dim;
        let value_width = self.config.value_head_dim;
        // Per-head factors of the KV-B expansion, as zero-copy views of its
        // weight: W^K_h is [content, rank] and W^V_h is [rank, value].
        let per_head_weight = weight.reshape([rank, heads, per_head]).swap_dims(0, 1);
        let key_weight = per_head_weight
            .clone()
            .narrow(2, 0, content)
            .swap_dims(1, 2)
            .unsqueeze_dim::<4>(0)
            .expand([batch, heads, content, rank]);
        let value_weight = per_head_weight
            .narrow(2, content, value_width)
            .unsqueeze_dim::<4>(0)
            .expand([batch, heads, rank, value_width]);
        // Absorb W^K into the content queries: q^C_h W^K_h is [b, h, q, rank].
        let absorbed_query = projection.query_content.clone().matmul(key_weight);
        // All heads now read the same cached latents and shared rotary key, so
        // heads fold into the rows of one GEMM: [b, h*q, rank] x [b, rank, key].
        let rows = heads * query_sequence;
        let content_logits = absorbed_query
            .reshape([batch, rows, rank])
            .matmul(all_latents.clone().swap_dims(1, 2));
        let rotary_logits = projection
            .query_rotary
            .clone()
            .reshape([batch, rows, rotary])
            .matmul(
                all_rotary_keys
                    .reshape([batch, key_sequence, rotary])
                    .swap_dims(1, 2),
            );
        let logits = ((content_logits + rotary_logits) * self.config.softmax_scale).reshape([
            batch,
            heads,
            query_sequence,
            key_sequence,
        ]);
        let probabilities = attention_probabilities(logits, mask)?;
        // sum_t p_t c_t for every head, again as one folded GEMM over the cache.
        let aggregated_latent = probabilities
            .clone()
            .reshape([batch, rows, key_sequence])
            .matmul(all_latents)
            .reshape([batch, heads, query_sequence, rank]);
        let mixed = aggregated_latent
            .matmul(value_weight)
            .swap_dims(1, 2)
            .reshape([batch, query_sequence, heads * value_width]);
        Ok(MlaMixOutput {
            hidden_state: operations::linear(mixed, &self.o_proj)?,
            probabilities: retain_probabilities.then_some(probabilities),
        })
    }

    /// Expands normalized latent vectors into ordinary per-head content K/V.
    pub fn expand_latents(&self, latents: Tensor<B, 3>) -> Result<(Tensor<B, 4>, Tensor<B, 4>)> {
        let [batch, sequence, rank] = latents.dims();
        if rank != self.config.kv_lora_rank {
            return Err(Error::shape(
                "mla.latents",
                format!("[batch, sequence, {}]", self.config.kv_lora_rank),
                format!("{:?}", latents.dims()),
            ));
        }
        let per_head = self.config.qk_nope_head_dim + self.config.value_head_dim;
        let expanded = operations::linear(latents, &self.kv_b_proj)?
            .reshape([batch, sequence, self.config.num_heads, per_head])
            .swap_dims(1, 2);
        Ok((
            expanded.clone().narrow(3, 0, self.config.qk_nope_head_dim),
            expanded.narrow(3, self.config.qk_nope_head_dim, self.config.value_head_dim),
        ))
    }
}

/// Causal, sliding-window, and padding visibility of keys, described by token positions.
///
/// Query `i` of row `b` may attend to key `j` exactly when
///
/// * `key_positions[b, j] <= query_positions[b, i]` (causality),
/// * with a window `W`, `query_positions[b, i] - key_positions[b, j] < W`
///   (the `W` newest positions, Hugging Face's `sliding_window` convention),
/// * key `j` is valid (`key_valid[b, j]`, or always when absent), and
/// * query `i` is valid (`query_valid[b, i]`, or always when absent);
///
/// an invalid query row sees nothing and yields a zero context. Positions are
/// indices on one shared token timeline: input columns or valid-token
/// ordinals for stateless self-attention, cache slots for cached attention.
/// Position tensors may have a leading extent of one, which broadcasts over
/// the batch. Building masks from `[batch, tokens]` integer vectors means a
/// chunk materializes only its own `[batch, 1, rows, keys]` Boolean plane,
/// expanded (not copied) over heads. The type is attention-generic: MLA and
/// grouped-query families share it.
///
/// `key_limits[i]` is the number of leading keys that queries `0..=i` may
/// need: every key at index `>= key_limits[i]` is causally invisible to all of
/// them. It holds whenever keys are ordered by position (column order,
/// ordinal order, or slot order of full-history storage), is nondecreasing,
/// and lets bounded attention skip keys a query chunk cannot see. Storage
/// that is not position-ordered (a ring) uses the full key count.
#[derive(Clone, Debug)]
pub(crate) struct PositionalVisibility<B: Backend> {
    /// Query positions `[batch or 1, query]`.
    query_positions: Tensor<B, 2, Int>,
    /// Key positions `[batch or 1, key]`.
    key_positions: Tensor<B, 2, Int>,
    /// Optional query validity `[batch, query]` (`true` = valid).
    query_valid: Option<Tensor<B, 2, Bool>>,
    /// Optional key validity `[batch, key]` (`true` = valid).
    key_valid: Option<Tensor<B, 2, Bool>>,
    /// Host prefix bounds: queries `0..=i` see only keys `0..key_limits[i]`.
    key_limits: Vec<usize>,
    /// Optional sliding window `W > 0` in positions.
    window: Option<usize>,
}

impl<B: Backend> PositionalVisibility<B> {
    /// Describes causal self-attention over `sequence` columns with optional padding.
    ///
    /// Query and key `t` both sit at column `t`, and `mask` (`true` = valid)
    /// excludes padded queries and keys, exactly as a causal mask combined with
    /// a two-sided padding mask.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Configuration`] when `sequence` exceeds `i64`.
    pub(crate) fn causal_columns(
        sequence: usize,
        mask: Option<Tensor<B, 2, Bool>>,
        device: &B::Device,
    ) -> Result<Self> {
        let end = i64::try_from(sequence)
            .map_err(|_| Error::configuration("mla.visibility", "sequence exceeds i64"))?;
        let positions = Tensor::<B, 1, Int>::arange(0..end, device).reshape([1, sequence]);
        Ok(Self {
            query_positions: positions.clone(),
            key_positions: positions,
            query_valid: mask.clone(),
            key_valid: mask,
            key_limits: suffix_key_limits(sequence, sequence),
            window: None,
        })
    }

    /// Describes causal self-attention whose positions are valid-token ordinals.
    ///
    /// With a `mask` (`true` = valid), query and key positions are each row's
    /// valid-token ordinals `cumsum(valid) - 1`, so padding holes consume no
    /// sliding-window capacity (exactly as a cache that compacts each row);
    /// without one they are columns, as in [`Self::causal_columns`]. Column
    /// order still bounds the causal key prefix, because a later valid column
    /// has a larger ordinal.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Configuration`] when `sequence` exceeds `i64`.
    pub(crate) fn causal_ordinals(
        sequence: usize,
        mask: Option<&crate::architecture::TokenMask<B>>,
        device: &B::Device,
    ) -> Result<Self> {
        let mut visibility =
            Self::causal_columns(sequence, mask.map(|mask| mask.as_tensor().clone()), device)?;
        if let Some(mask) = mask {
            let ordinals = crate::attention::valid_token_ordinals(mask);
            visibility.query_positions = ordinals.clone();
            visibility.key_positions = ordinals;
        }
        Ok(visibility)
    }

    /// Describes `queries` bottom-right-aligned to the end of `keys` unpadded columns.
    ///
    /// Query `i` sits at position `keys - queries + i`, so a cached suffix sees
    /// its complete history; `queries <= keys` is required.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Shape`] for `queries > keys` and
    /// [`Error::Configuration`] for lengths beyond `i64`.
    pub(crate) fn suffix(queries: usize, keys: usize, device: &B::Device) -> Result<Self> {
        let offset = keys.checked_sub(queries).ok_or_else(|| {
            Error::shape(
                "attention.visibility",
                "key sequence at least as long as query sequence",
                format!("query={queries}, key={keys}"),
            )
        })?;
        let end = i64::try_from(keys)
            .map_err(|_| Error::configuration("attention.visibility", "keys exceed i64"))?;
        let start = i64::try_from(offset)
            .map_err(|_| Error::configuration("attention.visibility", "keys exceed i64"))?;
        Ok(Self {
            query_positions: Tensor::<B, 1, Int>::arange(start..end, device).reshape([1, queries]),
            key_positions: Tensor::<B, 1, Int>::arange(0..end, device).reshape([1, keys]),
            query_valid: None,
            key_valid: None,
            key_limits: suffix_key_limits(queries, keys),
            window: None,
        })
    }

    /// Restricts every query to the `window` newest positions up to itself.
    ///
    /// `None` (or no call) keeps unrestricted causal attention; `Some(0)` is
    /// rejected by [`Self::validate`].
    #[must_use]
    pub(crate) const fn with_window(mut self, window: Option<usize>) -> Self {
        self.window = window;
        self
    }

    /// Describes queries at explicit cache slots attending to slot-ordered keys.
    ///
    /// `query_positions` is `[batch, query]` with optional `query_valid`; keys
    /// are `[batch or 1, key]` slots whose validity is `key_valid`, as returned
    /// by a batched cache read. `key_limits` must satisfy the type's prefix
    /// contract; [`PositionalVisibility::validate`] checks its shape and monotonicity.
    pub(crate) const fn cached(
        query_positions: Tensor<B, 2, Int>,
        query_valid: Option<Tensor<B, 2, Bool>>,
        key_positions: Tensor<B, 2, Int>,
        key_valid: Option<Tensor<B, 2, Bool>>,
        key_limits: Vec<usize>,
    ) -> Self {
        Self {
            query_positions,
            key_positions,
            query_valid,
            key_valid,
            key_limits,
            window: None,
        }
    }

    /// Checks every tensor and host bound against one `[batch, query, key]` geometry.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Shape`] naming `mla.visibility` on any mismatch.
    pub(crate) fn validate(&self, batch: usize, queries: usize, keys: usize) -> Result<()> {
        let rows_match = |rows: usize| rows == batch || rows == 1;
        let [query_rows, query_columns] = self.query_positions.dims();
        let [key_rows, key_columns] = self.key_positions.dims();
        let valid_ok = |valid: &Option<Tensor<B, 2, Bool>>, columns: usize| {
            valid
                .as_ref()
                .is_none_or(|valid| valid.dims() == [batch, columns])
        };
        // Verified: accepts exactly one nondecreasing bound in `1..=keys` per query.
        let limits_ok = key_limits_valid(&self.key_limits, queries, keys);
        if self.window == Some(0) {
            return Err(Error::configuration(
                "attention.visibility.window",
                "must be greater than zero",
            ));
        }
        if !rows_match(query_rows)
            || !rows_match(key_rows)
            || query_columns != queries
            || key_columns != keys
            || !valid_ok(&self.query_valid, queries)
            || !valid_ok(&self.key_valid, keys)
            || !limits_ok
        {
            return Err(Error::shape(
                "mla.visibility",
                format!(
                    "positions [{batch} or 1, {queries}] and [{batch} or 1, {keys}], \
                     validity [{batch}, tokens], and {queries} nondecreasing key limits in 1..={keys}"
                ),
                format!(
                    "query positions {:?}, key positions {:?}, {} key limits",
                    self.query_positions.dims(),
                    self.key_positions.dims(),
                    self.key_limits.len()
                ),
            ));
        }
        Ok(())
    }

    /// Returns the key prefix length that query rows `rows` (non-empty) may need.
    ///
    /// For validated bounds this covers every row of the chunk (verified).
    pub(crate) fn key_end(&self, rows: &Range<usize>) -> usize {
        chunk_key_end(&self.key_limits, rows.end)
    }

    /// Returns the host prefix bounds: query rows `0..=i` see only keys `0..limits[i]`.
    pub(crate) fn key_limits(&self) -> &[usize] {
        &self.key_limits
    }

    /// Builds `[batch, heads, rows, keys]` exclusions (`true` = excluded) for one chunk.
    ///
    /// Ranges index the full query and key axes and must lie inside them; the
    /// result is an expanded view of one `[batch, 1, rows, keys]` plane.
    pub(crate) fn exclusion(
        &self,
        batch: usize,
        heads: usize,
        rows: Range<usize>,
        keys: Range<usize>,
    ) -> Tensor<B, 4, Bool> {
        let plane = [batch, 1, rows.len(), keys.len()];
        let query = self
            .query_positions
            .clone()
            .narrow(1, rows.start, rows.len());
        let query_rows = query.dims()[0];
        let query = query.reshape([query_rows, 1, rows.len(), 1]).expand(plane);
        let key = self.key_positions.clone().narrow(1, keys.start, keys.len());
        let key_rows = key.dims()[0];
        let key = key.reshape([key_rows, 1, 1, keys.len()]).expand(plane);
        let mut excluded = key.clone().greater(query.clone());
        if let Some(window) = self.window {
            // `key + W <= query`: only the `W` newest positions remain visible.
            let width = saturating_window(window);
            excluded = excluded.bool_or(key.add_scalar(width).lower_equal(query));
        }
        if let Some(valid) = &self.key_valid {
            excluded = excluded.bool_or(
                valid
                    .clone()
                    .narrow(1, keys.start, keys.len())
                    .bool_not()
                    .reshape([batch, 1, 1, keys.len()])
                    .expand(plane),
            );
        }
        if let Some(valid) = &self.query_valid {
            excluded = excluded.bool_or(
                valid
                    .clone()
                    .narrow(1, rows.start, rows.len())
                    .bool_not()
                    .reshape([batch, 1, rows.len(), 1])
                    .expand(plane),
            );
        }
        excluded.expand([batch, heads, rows.len(), keys.len()])
    }
}

/// Maps a verified width fault to the configuration error naming the failed width.
fn mla_width_error(fault: MlaWidthFault) -> Error {
    match fault {
        MlaWidthFault::QkHeadDim => {
            Error::configuration("mla.qk_head_dim", "dimension sum overflows usize")
        }
        MlaWidthFault::CompressedWidth => Error::configuration(
            "mla.compressed_projection_width",
            "dimension sum overflows usize",
        ),
        MlaWidthFault::ExpandedPerHead => {
            Error::configuration("mla.expanded_kv_width", "per-head sum overflows usize")
        }
        MlaWidthFault::ExpandedHeads => {
            Error::configuration("mla.expanded_kv_width", "head product overflows usize")
        }
        MlaWidthFault::QueryWidth => {
            Error::configuration("mla.query_width", "head product overflows usize")
        }
        MlaWidthFault::Matrix {
            matrix,
            input,
            output,
        } => {
            let projection = match matrix {
                MlaMatrix::Query => "q_proj",
                MlaMatrix::QueryDown => "q_a_proj",
                MlaMatrix::QueryUp => "q_b_proj",
                MlaMatrix::JointKeyValue => "kv_a_proj_with_mqa",
                MlaMatrix::KeyValueExpansion => "kv_b_proj",
                MlaMatrix::Output => "o_proj",
            };
            Error::configuration(
                "mla.projection_shape",
                format!("{projection} matrix element count overflows usize: {input} * {output}"),
            )
        }
    }
}

/// Applies a `true == excluded` mask and the shared FP32 softmax.
///
/// Excluded logits become `-inf` in the input dtype, which the shared softmax
/// maps to exact zeros; rows without any visible key become zero rows.
fn attention_probabilities<B: Backend>(
    logits: Tensor<B, 4>,
    mask: Option<Tensor<B, 4, Bool>>,
) -> Result<Tensor<B, 4>> {
    if let Some(mask) = &mask
        && mask.dims() != logits.dims()
    {
        return Err(Error::shape(
            "mla.attention_mask",
            format!("{:?}", logits.dims()),
            format!("{:?}", mask.dims()),
        ));
    }
    Ok(eager_probabilities(logits, mask.as_ref(), None, None))
}

#[cfg(test)]
#[path = "mla_tests.rs"]
mod tests;
