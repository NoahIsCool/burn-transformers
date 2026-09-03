//! Qwen3.5/Qwen3.8 multi-token-prediction head.
//!
//! The released checkpoint adds one full-attention decoder block beside the
//! target transformer.  At shifted position `t` it combines the normalized
//! embedding of the already selected next token `x_(t+1)` with the normalized
//! target hidden state `h_t`,
//!
//! ```text
//! u_t = W_c [ RMSNorm(E[x_(t+1)]) ; RMSNorm(h_t) ],
//! m_t = RMSNorm(DecoderMTP(u_t)).
//! ```
//!
//! The ordinary (shared) language-model head maps `m_t` to the distribution of
//! `x_(t+2)`.  Thus this module intentionally owns neither a token embedding nor
//! an LM head: sharing those two parameter stores is part of the architecture,
//! not merely a checkpoint-loading optimization.

use burn::{
    module::Module,
    nn::{Linear, LinearConfig, LinearLayout},
    tensor::{Int, Tensor, backend::Backend},
};

use crate::{
    Error, Result,
    normalization::{RmsNorm, RmsNormConfig},
};

use super::{Qwen3_5DecoderLayer, Qwen3_5LayerType, Qwen3_5TextConfig};

/// Output of one MTP layer before the shared vocabulary projection.
#[derive(Debug)]
pub struct Qwen3_5MtpOutput<B: Backend> {
    /// Final normalized MTP hidden state `[batch, sequence, hidden]`.
    pub hidden_state: Tensor<B, 3>,
}

/// One MTP prediction after projection through the base model's shared LM head.
#[derive(Debug)]
pub struct Qwen3_5MtpPrediction<B: Backend> {
    /// Processed vocabulary scores `[batch, sequence, vocabulary]`.
    pub logits: Tensor<B, 3>,
    /// Final normalized auxiliary hidden state used by the next recurrent step.
    pub hidden_state: Tensor<B, 3>,
}

/// Exact auxiliary prediction graph serialized below the `mtp.*` namespace.
///
/// Released Qwen3.8 uses one layer.  The vector form preserves the official
/// `step mod mtp_num_hidden_layers` contract for compatible future checkpoints.
#[derive(Module, Debug)]
pub struct Qwen3_5Mtp<B: Backend> {
    /// Concatenation projection `W_c : R^(2H) -> R^H`.
    pub fc: Linear<B>,
    /// Full-attention prediction blocks.
    pub layers: Vec<Qwen3_5DecoderLayer<B>>,
    /// Final zero-centered RMS normalization.
    pub norm: RmsNorm<B>,
    /// Normalization applied to the shared next-token embedding.
    pub pre_fc_norm_embedding: RmsNorm<B>,
    /// Normalization applied to the target transformer's final hidden state.
    pub pre_fc_norm_hidden: RmsNorm<B>,
}

impl<B: Backend> Qwen3_5Mtp<B> {
    /// Initializes the auxiliary graph using the target transformer's topology.
    pub(super) fn init(config: &Qwen3_5TextConfig, device: &B::Device) -> Result<Self> {
        if config.mtp_num_hidden_layers == 0 {
            return Err(Error::configuration(
                "qwen3_5.mtp_num_hidden_layers",
                "must be non-zero when initializing the MTP graph",
            ));
        }
        if config.mtp_use_dedicated_embeddings {
            return Err(Error::configuration(
                "qwen3_5.mtp_use_dedicated_embeddings",
                "dedicated MTP embeddings have no released Qwen3.5/Qwen3.8 checkpoint contract",
            ));
        }
        let norm = || {
            RmsNormConfig::new(config.hidden_size)
                .with_epsilon(config.rms_norm_eps)
                .with_zero_centered(true)
                .init(device)
        };
        let concatenated_width = config.hidden_size.checked_mul(2).ok_or_else(|| {
            Error::configuration(
                "qwen3_5.hidden_size",
                "MTP concatenated width overflows usize",
            )
        })?;
        let layers = (0..config.mtp_num_hidden_layers)
            .map(|_| Qwen3_5DecoderLayer::init(config, Qwen3_5LayerType::FullAttention, device))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            fc: LinearConfig::new(concatenated_width, config.hidden_size)
                .with_bias(false)
                .with_layout(LinearLayout::Col)
                .init(device),
            layers,
            norm: norm()?,
            pre_fc_norm_embedding: norm()?,
            pre_fc_norm_hidden: norm()?,
        })
    }

    /// Evaluates one stateless shifted MTP step.
    ///
    /// `next_token_embeddings` and `target_hidden_state` must both be
    /// `[batch, sequence, H]`; `positions` is `[batch, sequence]`.  `step`
    /// selects `layers[step mod layers.len()]`, matching the reference runtime.
    pub fn forward(
        &self,
        next_token_embeddings: Tensor<B, 3>,
        target_hidden_state: Tensor<B, 3>,
        positions: Tensor<B, 2, Int>,
        step: usize,
        rope: crate::position::RopeConfig,
        evaluated_context: usize,
    ) -> Result<Qwen3_5MtpOutput<B>> {
        let [batch, sequence, _] = next_token_embeddings.dims();
        if positions.device() != next_token_embeddings.device() {
            return Err(Error::BackendCapability(
                "Qwen MTP position IDs and embeddings must use the same device".into(),
            ));
        }
        if positions.dims() != [batch, sequence] {
            return Err(Error::shape(
                "qwen.mtp.position_ids",
                format!("[{batch}, {sequence}]"),
                format!("{:?}", positions.dims()),
            ));
        }
        let layer_count = self.layers.len();
        if layer_count == 0 {
            return Err(Error::configuration(
                "qwen3_5.mtp",
                "MTP graph contains no prediction layer",
            ));
        }
        let layer = self.layers.get(step % layer_count).ok_or_else(|| {
            Error::configuration("qwen3_5.mtp", "MTP graph contains no prediction layer")
        })?;
        let hidden = self.fuse(next_token_embeddings, target_hidden_state)?;
        let hidden = layer.forward(hidden, positions, None, rope, evaluated_context)?;
        Ok(Qwen3_5MtpOutput {
            hidden_state: self.norm.forward(hidden),
        })
    }

    /// Applies the two pre-normalizations and `2H -> H` concatenation map.
    pub(super) fn fuse(
        &self,
        next_token_embeddings: Tensor<B, 3>,
        target_hidden_state: Tensor<B, 3>,
    ) -> Result<Tensor<B, 3>> {
        if next_token_embeddings.device() != target_hidden_state.device() {
            return Err(Error::BackendCapability(
                "Qwen MTP embedding and hidden-state tensors must use the same device".into(),
            ));
        }
        let embedding_shape = next_token_embeddings.dims();
        let hidden_shape = target_hidden_state.dims();
        if embedding_shape != hidden_shape {
            return Err(Error::shape(
                "qwen.mtp.target_hidden_state",
                format!("{:?}", embedding_shape),
                format!("{:?}", hidden_shape),
            ));
        }
        let [batch, sequence, hidden] = embedding_shape;
        if batch == 0
            || sequence == 0
            || hidden.checked_mul(2) != Some(self.fc.weight.val().dims()[0])
        {
            return Err(Error::shape(
                "qwen.mtp.inputs",
                "matching non-empty [batch, sequence, hidden_size] tensors",
                format!("embedding={embedding_shape:?}, hidden={hidden_shape:?}"),
            ));
        }
        let embedding = self.pre_fc_norm_embedding.forward(next_token_embeddings);
        let target = self.pre_fc_norm_hidden.forward(target_hidden_state);
        Ok(self.fc.forward(Tensor::cat(vec![embedding, target], 2)))
    }
}
