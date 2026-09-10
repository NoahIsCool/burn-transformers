//! Executable transformer graphs assembled from serializable architecture specifications.
//!
//! Family modules answer the question “what exactly did this checkpoint ship?”.
//! This module answers the complementary experimental question “what happens if these
//! mechanisms are recombined?”. [`ModularCausalLm`] and [`ModularSeq2SeqLm`] turn the
//! model-independent specifications in [`crate::architecture`] into ordinary Burn modules.
//!
//! The builders are deliberately strict. A valid serialized specification can still ask for
//! a mixer whose shared executable kernel has not landed yet. Such a request fails before any
//! parameter is allocated; it is never substituted with ordinary attention or another
//! mathematically different operation. Stateful inference uses a preallocated token-history
//! state and recomputes the stateless graph. That reference path is exact and useful for parity
//! experiments, while family runtimes retain the optimized native-KV and recurrent caches.
//! Semantic masks exclude invalid keys for both full and sliding attention. Sliding windows
//! count valid tokens, matching compact cached history; explicit positional-encoding IDs can
//! have gaps without changing which recent keys remain visible.
//!
//! The serializable flags retain one unambiguous runtime meaning here: an attention
//! `output_gate` is sigmoid-multiplicative, Q/K normalization is per-head RMS normalization
//! using the layer mixer's epsilon, and [`crate::architecture::FeedForwardSpec::Dense`] is the
//! three-projection gated MLP documented by that specification. These choices match the shared
//! executable primitives rather than inferring behavior from a model-family name.

mod causal;
mod common;
mod seq2seq;
mod state;

pub use causal::ModularCausalLm;
pub use common::{ModularLayer, ModularNorm, NormLayout};
pub use seq2seq::ModularSeq2SeqLm;
pub use state::ModularDecodeState;

#[cfg(test)]
mod tests;
