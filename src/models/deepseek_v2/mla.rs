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
//! Consequently decode reads only `r + d_r` cached scalars per layer/token and
//! never materializes historical per-head keys or values. Both paths share the
//! same parameters and are numerically equivalent up to floating-point
//! reassociation.

use burn::{
    module::Module,
    nn::{Linear, LinearConfig, LinearLayout},
    tensor::{Bool, DType, Tensor, activation::softmax, backend::Backend},
};

use crate::{
    Error, Result,
    normalization::{RmsNorm, RmsNormConfig},
    operations,
    position::{RopeConfig, apply_rotary, rotary_angles},
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

    /// Validates every projection extent and numerical invariant.
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
        self.compressed_projection_width()?;
        self.expanded_kv_width()?;
        Ok(())
    }

    /// Allocates all MLA projections in checkpoint-native column layout.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Result<MultiLatentAttention<B>> {
        self.validate()?;
        let qk_width = self.qk_head_dim()?;
        let query_width = self.num_heads.checked_mul(qk_width).ok_or_else(|| {
            Error::configuration("mla.query_width", "head product overflows usize")
        })?;
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
pub(crate) struct MlaProjection<B: Backend> {
    /// Per-head query channels whose dot products use compressed content keys.
    pub(crate) query_content: Tensor<B, 4>,
    /// Per-head query channels after interleaved conversion and RoPE.
    pub(crate) query_rotary: Tensor<B, 4>,
    /// Normalized token latent that is persisted in the decode cache.
    pub(crate) normalized_latent: Tensor<B, 3>,
    /// Single shared positional key after interleaved conversion and RoPE.
    pub(crate) rotated_key: Tensor<B, 4>,
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
    /// Projects one segment and applies DeepSeek's interleaved rotary convention.
    pub(crate) fn project(
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
        let query = match (
            &self.q_proj,
            &self.q_a_proj,
            &self.q_a_layernorm,
            &self.q_b_proj,
        ) {
            (Some(direct), None, None, None) => operations::linear(hidden.clone(), direct)?,
            (None, Some(down), Some(norm), Some(up)) => {
                operations::linear(norm.forward(operations::linear(hidden.clone(), down)?), up)?
            }
            _ => {
                return Err(Error::Checkpoint(
                    "MLA query projection topology disagrees with q_lora_rank".into(),
                ));
            }
        };
        let qk_width = self.config.qk_head_dim()?;
        let query = query
            .reshape([batch, sequence, self.config.num_heads, qk_width])
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

        let compressed = operations::linear(hidden, &self.kv_a_proj_with_mqa)?;
        let latent = compressed.clone().narrow(2, 0, self.config.kv_lora_rank);
        let normalized_latent = self.kv_a_layernorm.forward(latent);
        let raw_key = compressed
            .narrow(2, self.config.kv_lora_rank, self.config.qk_rope_head_dim)
            .reshape([batch, sequence, 1, self.config.qk_rope_head_dim])
            .swap_dims(1, 2);
        let raw_key = interleaved_to_split(raw_key, "mla.rotary_key")?;
        let angles = rotary_angles(rope, positions, evaluated_context)?;
        let (query_rotary, rotated_key) = apply_rotary(query_rotary, raw_key, &angles)?;
        Ok(MlaProjection {
            query_content,
            query_rotary,
            normalized_latent,
            rotated_key,
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
        let per_head_weight = weight
            .reshape([self.config.kv_lora_rank, heads, per_head])
            .swap_dims(0, 1);
        let key_weight = per_head_weight
            .clone()
            .narrow(2, 0, self.config.qk_nope_head_dim)
            .swap_dims(1, 2)
            .unsqueeze_dim::<4>(0)
            .expand([
                batch,
                heads,
                self.config.qk_nope_head_dim,
                self.config.kv_lora_rank,
            ]);
        let value_weight = per_head_weight
            .narrow(2, self.config.qk_nope_head_dim, self.config.value_head_dim)
            .unsqueeze_dim::<4>(0)
            .expand([
                batch,
                heads,
                self.config.kv_lora_rank,
                self.config.value_head_dim,
            ]);
        let latent_heads = all_latents.unsqueeze_dim::<4>(1).expand([
            batch,
            heads,
            key_sequence,
            self.config.kv_lora_rank,
        ]);
        let absorbed_query = projection.query_content.clone().matmul(key_weight);
        let content_logits = absorbed_query.matmul(latent_heads.clone().swap_dims(2, 3));
        let rotary_keys =
            all_rotary_keys.expand([batch, heads, key_sequence, self.config.qk_rope_head_dim]);
        let rotary_logits = projection
            .query_rotary
            .clone()
            .matmul(rotary_keys.swap_dims(2, 3));
        let probabilities = attention_probabilities(
            (content_logits + rotary_logits) * self.config.softmax_scale,
            mask,
        )?;
        let aggregated_latent = probabilities.clone().matmul(latent_heads);
        let mixed = aggregated_latent
            .matmul(value_weight)
            .swap_dims(1, 2)
            .reshape([batch, query_sequence, heads * self.config.value_head_dim]);
        Ok(MlaMixOutput {
            hidden_state: operations::linear(mixed, &self.o_proj)?,
            probabilities: retain_probabilities.then_some(probabilities),
        })
    }

    /// Expands normalized latent vectors into ordinary per-head content K/V.
    fn expand_latents(&self, latents: Tensor<B, 3>) -> Result<(Tensor<B, 4>, Tensor<B, 4>)> {
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

/// Applies a true-means-masked attention mask and FP32 softmax reduction.
fn attention_probabilities<B: Backend>(
    logits: Tensor<B, 4>,
    mask: Option<Tensor<B, 4, Bool>>,
) -> Result<Tensor<B, 4>> {
    if let Some(mask) = mask {
        if mask.dims() != logits.dims() {
            return Err(Error::shape(
                "mla.attention_mask",
                format!("{:?}", logits.dims()),
                format!("{:?}", mask.dims()),
            ));
        }
        let dtype = logits.dtype();
        let shape = logits.dims();
        let row_has_key = mask
            .clone()
            .bool_not()
            .int()
            .sum_dim(3)
            .greater_elem(0)
            .expand(shape);
        let probabilities = softmax(
            logits
                .cast(DType::F32)
                .mask_fill(mask.clone(), f32::NEG_INFINITY)
                .mask_fill(row_has_key.bool_not(), 0.0),
            3,
        )
        .mask_fill(mask, 0.0);
        Ok(probabilities.cast(dtype))
    } else {
        let dtype = logits.dtype();
        Ok(softmax(logits.cast(DType::F32), 3).cast(dtype))
    }
}

#[cfg(test)]
#[path = "mla_tests.rs"]
mod tests;
