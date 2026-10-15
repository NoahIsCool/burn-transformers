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
//! step: content logits, rotary logits, and latent aggregation each stream the
//! `r + d_r` cached scalars per token once per layer, not once per head. Only
//! the per-head factors `W^K_h` and `W^V_h` remain head-specific. Both paths
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

use burn::{
    module::Module,
    nn::{Linear, LinearConfig, LinearLayout},
    tensor::{Bool, Int, Tensor, backend::Backend},
};

use crate::{
    Error, Result,
    attention::{bottom_right_causal_mask, eager_probabilities, union_exclusions, zero_excluded},
    intervention::{AttentionPoints, Hooks, PointInfo},
    normalization::{RmsNorm, RmsNormConfig},
    operations,
    position::{RopeConfig, RotaryAngles, apply_rotary, rotary_angles},
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
    /// backend memory for the requested tensors.
    pub fn validate(&self) -> Result<()> {
        for (field, value) in [
            ("hidden_size", self.hidden_size),
            ("num_heads", self.num_heads),
            ("qk_nope_head_dim", self.qk_nope_head_dim),
            ("qk_rope_head_dim", self.qk_rope_head_dim),
            ("value_head_dim", self.value_head_dim),
            ("kv_lora_rank", self.kv_lora_rank),
        ] {
            if value == 0 {
                return Err(Error::configuration(
                    format!("mla.{field}"),
                    "must be non-zero",
                ));
            }
        }
        if !self.qk_rope_head_dim.is_multiple_of(2) {
            return Err(Error::configuration(
                "mla.qk_rope_head_dim",
                "must be even because RoPE rotates channel pairs",
            ));
        }
        if self.q_lora_rank == Some(0) {
            return Err(Error::configuration(
                "mla.q_lora_rank",
                "must be non-zero when present",
            ));
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
        self.qk_head_dim()?;
        let compressed_width = self.compressed_projection_width()?;
        let expanded_width = self.expanded_kv_width()?;
        let query_width = self.query_projection_width()?;
        // Burn parameters initialize lazily. Reject impossible matrix element
        // counts here, rather than overflowing when their tensors are first read.
        // The checked expanded width also bounds num_heads * value_head_dim.
        for (projection, input, output) in [
            self.q_lora_rank
                .is_none()
                .then_some(("q_proj", self.hidden_size, query_width)),
            self.q_lora_rank
                .map(|rank| ("q_a_proj", self.hidden_size, rank)),
            self.q_lora_rank.map(|rank| ("q_b_proj", rank, query_width)),
            Some(("kv_a_proj_with_mqa", self.hidden_size, compressed_width)),
            Some(("kv_b_proj", self.kv_lora_rank, expanded_width)),
            Some((
                "o_proj",
                self.num_heads * self.value_head_dim,
                self.hidden_size,
            )),
        ]
        .into_iter()
        .flatten()
        {
            if input.checked_mul(output).is_none() {
                return Err(Error::configuration(
                    "mla.projection_shape",
                    format!(
                        "{projection} matrix element count overflows usize: {input} * {output}"
                    ),
                ));
            }
        }
        Ok(())
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
        self.query_rotary = rotate(self.query_rotary, &query_angles)?;
        self.rotated_key = rotate(self.rotated_key, &key_angles)?;
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
        let query_rotary = interleaved_to_split(
            query.narrow(
                3,
                self.config.qk_nope_head_dim,
                self.config.qk_rope_head_dim,
            ),
            "mla.query_rotary",
        )?;
        let compressed = operations::linear(key_value_hidden, &self.kv_a_proj_with_mqa)?;
        let normalized_latent =
            self.kv_a_layernorm
                .forward(compressed.clone().narrow(2, 0, self.config.kv_lora_rank));
        let rotated_key = interleaved_to_split(
            compressed
                .narrow(2, self.config.kv_lora_rank, self.config.qk_rope_head_dim)
                .reshape([batch, key_sequence, 1, self.config.qk_rope_head_dim])
                .swap_dims(1, 2),
            "mla.rotary_key",
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
        let causal_mask = if causal && query_sequence > 1 {
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
    pub(crate) fn project_rotary(
        &self,
        hidden: Tensor<B, 3>,
        positions: burn::tensor::Tensor<B, 2, burn::tensor::Int>,
        rope: RopeConfig,
        evaluated_context: usize,
    ) -> Result<MlaProjection<B>> {
        let [batch, sequence, hidden_width] = hidden.dims();
        if batch == 0 || sequence == 0 || hidden_width != self.config.hidden_size {
            return Err(Error::shape(
                "mla.input",
                format!("[batch>0, sequence>0, {}]", self.config.hidden_size),
                format!("[{batch}, {sequence}, {hidden_width}]"),
            ));
        }
        if positions.dims() != [batch, sequence] {
            return Err(Error::shape(
                "mla.position_ids",
                format!("[{batch}, {sequence}]"),
                format!("{:?}", positions.dims()),
            ));
        }
        // Self-attention projection shares `project`'s code; one joint rotation
        // then applies the same positions to the queries and the shared key.
        let projection = self.project(hidden.clone(), hidden)?;
        let angles = rotary_angles(rope, positions, evaluated_context)?;
        let (query_rotary, rotated_key) =
            apply_rotary(projection.query_rotary, projection.rotated_key, &angles)?;
        Ok(MlaProjection {
            query_rotary,
            rotated_key,
            ..projection
        })
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

/// Rotates one split-half tensor `[batch, heads, sequence, rotary]` by `angles`.
///
/// The shared [`apply_rotary`] rotates an equal-length query/key pair. Its
/// companion here is a one-head slice of the same tensor whose rotation is
/// discarded, so the redundant work is at most one head instead of a second
/// full rotation.
fn rotate<B: Backend>(tensor: Tensor<B, 4>, angles: &RotaryAngles<B>) -> Result<Tensor<B, 4>> {
    let companion = tensor.clone().narrow(1, 0, 1);
    Ok(apply_rotary(tensor, companion, angles)?.0)
}

/// Converts `[real_0, imag_0, real_1, imag_1, ...]` to split-half layout.
fn interleaved_to_split<B: Backend>(
    input: Tensor<B, 4>,
    name: &'static str,
) -> Result<Tensor<B, 4>> {
    let [batch, heads, sequence, width] = input.dims();
    if width == 0 || !width.is_multiple_of(2) {
        return Err(Error::shape(
            name,
            "a non-zero even final dimension",
            format!("{:?}", input.dims()),
        ));
    }
    Ok(input
        .reshape([batch, heads, sequence, width / 2, 2])
        .swap_dims(3, 4)
        .reshape([batch, heads, sequence, width]))
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
