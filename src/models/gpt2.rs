//! GPT-2's learned-position, pre-normalized transformer for local experiments.
//!
//! The backbone, blocks, final normalization, and tied vocabulary readout are
//! public independently. Use [`Gpt2ForCausalLm::forward_with_hooks`] to inspect or
//! change actual intermediate tensors. [`Gpt2ForCausalLm::readout`] accepts the
//! residual before final normalization, making direct component effects easy to
//! examine without running subsequent blocks again.
//!
//! This is a stateless evaluation graph: dropout is disabled and no decode cache
//! is implied. Gradients remain connected, but training policy belongs to the
//! experiment. Exact GELU and the original GPT-2 tanh GELU are distinct explicit
//! configurations; importing weights never silently changes this choice.
//!
//! ```
//! use burn::{backend::Flex, tensor::{Int, Tensor}};
//! use transformers::{CausalLmInput, ForwardOptions};
//! use transformers::intervention::{Hooks, Layer};
//! use transformers::models::gpt2::{Gpt2Config, Gpt2ForCausalLm};
//!
//! let device = Default::default();
//! let config = Gpt2Config {
//!     vocab_size: 20, n_embd: 12, n_head: 3, n_layer: 2,
//!     n_positions: 16, n_inner: Some(24), ..Default::default()
//! };
//! let model = Gpt2ForCausalLm::<Flex>::init(config, &device)?;
//! let input = CausalLmInput {
//!     token_ids: Tensor::<Flex, 2, Int>::from_ints([[1, 2, 3]], &device),
//!     token_mask: None, position_ids: None,
//! };
//! let mut hooks = Hooks::new();
//! let point = Layer::decoder(0).mlp_output().token(2);
//! let original = hooks.capture(&point);
//! hooks.edit(point, |activation| activation * 0.5);
//! let output = model.forward_with_hooks(input, &ForwardOptions::default(), &mut hooks)?;
//! assert_eq!(hooks.get(&original)?.dims(), [1, 1, 12]);
//! assert_eq!(output.logits.dims(), [1, 1, 20]);
//! # Ok::<(), transformers::Error>(())
//! ```
//!
//! For residual width `d` and MLP width `m`, each block applies affine LayerNorm
//! over the last axis, scaled dot-product attention, then
//! `GELU(x W_up + b_up) W_down + b_down`.
//! The exact nonlinearity is `x * (1 + erf(x / sqrt(2))) / 2`; `gelu_new` uses
//! the source tanh approximation. Head count divides `d`, while `m` is independently
//! configurable. See `src/models/gpt2/README.md` for the pinned checkpoint,
//! reference implementation versions, and measured F32 tolerances.

use serde::{Deserialize, Serialize};

use crate::{Error, Result, architecture::Activation};

mod checkpoint;
mod model;

pub use checkpoint::{Gpt2CheckpointContract, Gpt2CheckpointLayout, Gpt2LoadError};
pub use model::{Gpt2Block, Gpt2ForCausalLm, Gpt2Model, Gpt2ModelOutput};

// Explicit graph choices; checkpoint data never selects a different nonlinearity.

/// The exact nonlinearity used by the source model's computation graph.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Gpt2Activation {
    /// Exact Gaussian error linear unit, used by some converted research graphs.
    #[serde(rename = "gelu")]
    Gelu,
    /// Original GPT-2's tanh approximation, called `gelu_new` in its config.
    #[default]
    #[serde(rename = "gelu_new")]
    GeluNew,
}

impl Gpt2Activation {
    /// Maps the explicit checkpoint nonlinearity to shared exact-erf or tanh-approximate GELU.
    fn activation(self) -> Activation {
        match self {
            Self::Gelu => Activation::Gelu,
            Self::GeluNew => Activation::GeluTanh,
        }
    }
}

// Source configuration translation and allocation preflight.

/// GPT-2 configuration using the public checkpoint's field names.
///
/// Defaults describe GPT-2 small. `n_inner=None` means `4 * n_embd`. Ordinary
/// Hugging Face metadata not affecting this evaluation graph is ignored by serde.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Gpt2Config {
    /// Must be `gpt2` for this architecture.
    pub model_type: String,
    /// Number of token embeddings and output logits.
    pub vocab_size: usize,
    /// Residual width.
    #[serde(alias = "hidden_size")]
    pub n_embd: usize,
    /// Number of sequential attention/MLP blocks.
    #[serde(alias = "num_hidden_layers")]
    pub n_layer: usize,
    /// Number of equally sized query/key/value heads.
    #[serde(alias = "num_attention_heads")]
    pub n_head: usize,
    /// Number of learned absolute-position embeddings.
    #[serde(alias = "max_position_embeddings")]
    pub n_positions: usize,
    /// MLP intermediate width; absence selects four times the residual width.
    pub n_inner: Option<usize>,
    /// Source graph's nonlinearity; never inferred from its weights.
    pub activation_function: Gpt2Activation,
    /// Epsilon added to each affine LayerNorm variance.
    pub layer_norm_epsilon: f64,
    /// Whether attention scores are divided by the square root of head width.
    pub scale_attn_weights: bool,
    /// Whether block `i` additionally divides scores by `i + 1`.
    pub scale_attn_by_inverse_layer_idx: bool,
    /// Alternate upcast/reordered attention graph; currently rejected explicitly.
    pub reorder_and_upcast_attn: bool,
    /// Cross-attention topology; currently rejected explicitly.
    pub add_cross_attention: bool,
    /// Output storage must be tied to the token embedding in this implementation.
    pub tie_word_embeddings: bool,
    /// Experimental visibility policy. `false` makes stateless attention bidirectional.
    pub causal: bool,
}

impl Default for Gpt2Config {
    /// Describes the causal, tied-readout GPT-2 small evaluation graph with tanh GELU.
    fn default() -> Self {
        Self {
            model_type: "gpt2".into(),
            vocab_size: 50_257,
            n_embd: 768,
            n_layer: 12,
            n_head: 12,
            n_positions: 1024,
            n_inner: None,
            activation_function: Gpt2Activation::GeluNew,
            layer_norm_epsilon: 1e-5,
            scale_attn_weights: true,
            scale_attn_by_inverse_layer_idx: false,
            reorder_and_upcast_attn: false,
            add_cross_attention: false,
            tie_word_embeddings: true,
            causal: true,
        }
    }
}

impl Gpt2Config {
    /// Returns the explicit or conventional MLP width, rejecting overflow.
    pub fn intermediate_size(&self) -> Result<usize> {
        self.n_inner
            .or_else(|| self.n_embd.checked_mul(4))
            .filter(|width| *width != 0)
            .ok_or_else(|| Error::configuration("gpt2.n_inner", "must be positive and fit usize"))
    }

    /// Checks the full stateless graph before any parameters are allocated.
    pub fn validate(&self) -> Result<()> {
        if self.model_type != "gpt2" {
            return Err(Error::configuration("gpt2.model_type", "expected gpt2"));
        }
        for (name, value) in [
            ("vocab_size", self.vocab_size),
            ("n_embd", self.n_embd),
            ("n_layer", self.n_layer),
            ("n_head", self.n_head),
            ("n_positions", self.n_positions),
        ] {
            if value == 0 || value > i32::MAX as usize {
                return Err(Error::configuration(
                    format!("gpt2.{name}"),
                    "must be positive and fit the portable integer index range",
                ));
            }
        }
        if !self.n_embd.is_multiple_of(self.n_head) {
            return Err(Error::configuration("gpt2.n_head", "must divide n_embd"));
        }
        if !self.layer_norm_epsilon.is_finite() || self.layer_norm_epsilon <= 0.0 {
            return Err(Error::configuration(
                "gpt2.layer_norm_epsilon",
                "must be finite and positive",
            ));
        }
        if self.reorder_and_upcast_attn || self.add_cross_attention || !self.tie_word_embeddings {
            return Err(Error::configuration(
                "gpt2",
                "this graph requires tied embeddings, no cross attention, and standard attention ordering",
            ));
        }
        for width in [self.vocab_size, self.n_positions, self.intermediate_size()?] {
            self.n_embd.checked_mul(width).ok_or_else(|| {
                Error::configuration("gpt2", "parameter dimensions overflow usize")
            })?;
        }
        self.n_embd.checked_mul(3).ok_or_else(|| {
            Error::configuration("gpt2.n_embd", "fused QKV width overflows usize")
        })?;
        Ok(())
    }
}
