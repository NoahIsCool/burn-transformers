//! Executable transformer graphs assembled from serializable architecture specifications.
//!
//! Family modules answer the question “what exactly did this checkpoint ship?”.
//! This module answers the complementary experimental question “what happens if these
//! mechanisms are recombined?”. [`ModularCausalLm`] and [`ModularSeq2SeqLm`] turn the
//! model-independent specifications in [`crate::architecture`] into ordinary Burn modules.
//!
//! # Structure
//!
//! | piece | module | role |
//! | --- | --- | --- |
//! | [`ModularLayer`] | `layer` | one block: mixer, optional cross-attention, optional feed-forward |
//! | mixers | `mixer` | grouped-query, latent (MLA), and differential attention; Gated DeltaNet; Mamba-2 |
//! | feed-forward | `feed_forward` | gated, ungated, clamped SwiGLU, routed experts |
//! | [`NormLayout`] | `norm` | identity, pre, post, or sandwich normalization per branch |
//! | stack pass | `pass` | positions, padding, overrides, and tables shared by all blocks |
//! | planning | `plan` | Verus-verified index plans (relative-position tables, history appends) |
//! | [`ModularDecodeState`] | `state` | fixed-capacity token history for reference decoding |
//!
//! Every block is built from the shared kernels through the specification's own
//! translation (for example [`crate::architecture::AttentionSpec`] to
//! [`crate::attention::GroupedQueryAttentionConfig`]), so validation and construction
//! agree. Full forwards, bidirectional forwards, staged ranges
//! ([`crate::architecture::StagedModel`]), single-block execution
//! ([`crate::experiment::HiddenTransform`]), and cached decoding all evaluate blocks through
//! one code path over one stack pass, which builds rotary tables, T5 relative-bucket
//! indices, and connectivity masks once per distinct configuration.
//!
//! # Contracts
//!
//! The builders are deliberately strict. A valid serialized specification can still ask for
//! a mechanism whose shared executable kernel has not landed yet (gated residuals). Such a
//! request fails before any parameter is allocated; it is never substituted with ordinary
//! attention or another mathematically different operation. Mamba-2 blocks run the shared
//! [`crate::mamba2`] mixer; settings its specification does not carry take the documented
//! Hugging Face `Mamba2Config` defaults (see the `mixer` module). Stateful inference uses
//! a preallocated token-history state and recomputes the stateless graph for the whole batch
//! in one pass. That reference path is exact and useful for parity experiments, while family
//! runtimes retain the optimized native-KV and recurrent caches. Semantic masks exclude
//! invalid keys for every connectivity. Sliding windows and local chunks count valid tokens,
//! matching compact cached history; explicit positional-encoding IDs can have gaps without
//! changing which recent keys remain visible.
//!
//! Serializable flags keep one unambiguous runtime meaning: an attention `output_gate` is
//! sigmoid-multiplicative unless the options select SiLU, Q/K normalization uses the layer
//! mixer's epsilon and zero-centering, the score scale is explicit
//! ([`crate::architecture::ScoreScale`], whose legacy `Auto` keeps T5-biased grouped-query
//! attention unscaled), and [`crate::architecture::FeedForwardSpec::Dense`] is the
//! three-projection gated MLP. These choices follow the shared executable primitives rather
//! than inferring behavior from a model-family name.
//!
//! # Initialization
//!
//! `init` uses Burn's default initializers so seeded experiments stay reproducible.
//! Training from scratch should use `init_with` and the spec-derived `scaled_init_policy`
//! (`N(0, std^2)` with GPT-2 depth scaling); learned T5 biases start from the reference
//! `N(0, 1/hidden_size)`.

mod causal;
mod common;
mod feed_forward;
mod init;
mod layer;
mod mixer;
mod norm;
mod pass;
mod plan;
mod seq2seq;
mod state;

pub use causal::ModularCausalLm;
pub use layer::ModularLayer;
pub use norm::{ModularNorm, NormLayout};
pub use seq2seq::{DecoderStages, EncoderStages, ModularSeq2SeqLm};
pub use state::ModularDecodeState;

#[cfg(test)]
/// Shared fixtures for the modular-model test suites.
mod test_support;
#[cfg(test)]
/// Tiny modular graphs, serialization, validation, and state-lifecycle regressions.
mod tests;
