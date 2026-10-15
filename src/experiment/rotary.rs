//! Rotary (RoPE) self-attention for open residual compositions.
//!
//! [`GroupedQueryAttention`]'s own [`HiddenTransform`] implementation is
//! position-free (NoPE): it reads padding and the attention policy from the
//! [`ExecutionContext`] but never its positions. [`RotaryAttention`] wraps the
//! same projections and rotates queries and keys by their logical positions
//! before scores are formed.
//!
//! # Mathematics
//!
//! RoPE treats the leading `r = rotary_dimension` channels of each head as
//! `r / 2` complex coordinates in the Hugging Face "rotate half" layout:
//! channel `k` is the real part and channel `k + r / 2` the imaginary part.
//! At position `p`, coordinate `k` is multiplied by `exp(i * p * omega_k)` with
//! `omega_k = theta^(-2k / r)` (adjusted by the configured YaRN scaling):
//!
//! ```text
//! x'_k       = x_k cos(p omega_k) - x_{k+r/2} sin(p omega_k)
//! x'_{k+r/2} = x_{k+r/2} cos(p omega_k) + x_k sin(p omega_k)
//! ```
//!
//! Rotations are orthogonal, so `<R(m) q, R(n) k> = <q, R(n - m) k>`: scores
//! depend only on the relative offset `n - m`, and shifting every position by a
//! constant leaves the output unchanged. Channels past `r` and the values are
//! not rotated.
//!
//! # Contracts
//!
//! Positions are the context's `[batch, token]` tensor. The evaluated context
//! (largest position plus one) selects dynamic frequency tables; plain RoPE and
//! static YaRN ignore it. Angle tables are computed in F32 and cast to the
//! activation dtype before the rotation. Query/key hook points observe the
//! rotated tensors, exactly what the score computation consumes.

use burn::{
    module::Module,
    tensor::{Tensor, backend::Backend},
};

use crate::{
    Error, Result,
    attention::{GroupedQueryAttention, GroupedQueryAttentionConfig},
    intervention::{Hooks, Layer, PointInfo},
    position::{RopeConfig, apply_rotary, rotary_angles},
};

use super::{
    ExecutionContext, HiddenTransform,
    block::{SelfAttentionPlan, self_attention_points},
};

/// Grouped-query self-attention whose queries and keys are rotated by position.
///
/// Use it as the mixer of a [`super::ResidualBlock`] to compose a Llama-style
/// block from ordinary modules:
///
/// ```
/// use burn::backend::Flex;
/// use transformers::{
///     attention::GroupedQueryAttentionConfig,
///     experiment::{ExecutionContext, HiddenTransform, RotaryAttention},
///     intervention::{Hooks, Layer},
///     position::{RopeConfig, RopeScaling},
/// };
///
/// let device = Default::default();
/// let config = GroupedQueryAttentionConfig::new(8, 2, 1, 4);
/// let rope = RopeConfig::new(4, 4, 10_000.0, RopeScaling::None).unwrap();
/// let attention = RotaryAttention::<Flex>::init(&config, rope, &device)?;
/// let context = ExecutionContext::contiguous(1, 3, &device)?;
/// let hidden = burn::tensor::Tensor::zeros([1, 3, 8], &device);
/// let layer = Layer::decoder(0);
/// let output = Hooks::new().run(attention.activation_points(layer), |hooks| {
///     attention.forward(hidden, &context, layer, hooks)
/// })?;
/// assert_eq!(output.dims(), [1, 3, 8]);
/// # Ok::<(), transformers::Error>(())
/// ```
#[derive(Module, Debug)]
pub struct RotaryAttention<B: Backend> {
    /// Query/key/value/output projections and the attention dispatch policy.
    pub attention: GroupedQueryAttention<B>,
    /// Rotary frequency policy; it owns no trainable parameters.
    #[module(skip)]
    pub rope: RopeConfig,
}

impl<B: Backend> RotaryAttention<B> {
    /// Wraps existing projections with a rotary policy for the same head width.
    ///
    /// `rope.head_dimension()` must equal `attention.head_dim`; a smaller
    /// `rope.rotary_dimension()` rotates only that prefix of every head.
    pub fn new(attention: GroupedQueryAttention<B>, rope: RopeConfig) -> Result<Self> {
        if rope.head_dimension() != attention.head_dim {
            return Err(Error::configuration(
                "rotary_attention.rope",
                format!(
                    "RoPE head dimension {} must equal the attention head_dim {}",
                    rope.head_dimension(),
                    attention.head_dim
                ),
            ));
        }
        Ok(Self { attention, rope })
    }

    /// Initializes fresh projections from `config` and wraps them with `rope`.
    pub fn init(
        config: &GroupedQueryAttentionConfig,
        rope: RopeConfig,
        device: &B::Device,
    ) -> Result<Self> {
        Self::new(config.init(device)?, rope)
    }
}

impl<B: Backend> HiddenTransform<B> for RotaryAttention<B> {
    /// Advertises the same Q/K/V/head-context (and eager score/pattern) points as plain GQA.
    fn activation_points(&self, layer: Layer) -> Vec<PointInfo> {
        self_attention_points(&self.attention, layer)
    }

    /// Rotates projected queries and keys by the context's positions, then attends.
    ///
    /// The attention policy, padding exclusions, and dtype checks are identical
    /// to the position-free implementation; only the Q/K rotation differs.
    fn forward(
        &self,
        hidden: Tensor<B, 3>,
        context: &ExecutionContext<B>,
        layer: Layer,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<Tensor<B, 3>> {
        let plan = SelfAttentionPlan::resolve(&self.attention, &hidden, context, layer)?;
        let mut projection = self.attention.project_self(hidden)?;
        let angles = rotary_angles(
            self.rope,
            context.positions.clone(),
            context.evaluated_context,
        )?;
        (projection.query, projection.key) =
            apply_rotary(projection.query, projection.key, &angles)?;
        plan.attend(&self.attention, projection, layer, hooks)
    }
}

#[cfg(all(test, feature = "flex"))]
#[path = "rotary_tests.rs"]
mod tests;
