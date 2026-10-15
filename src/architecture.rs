//! Public architecture contracts and reusable model specifications.
//!
//! Contracts separate stateless parameterized computation from mutable decode
//! sessions. [`HookedModel`] and [`StagedModel`] give experiments one interface
//! for activation hooks and staged execution across families. Shared
//! specifications describe mixer, normalization, and residual choices without
//! selecting a checkpoint family. Input/output types make tensor axes and
//! token-validity semantics explicit; family implementations validate the
//! requested capabilities before applying their numerical graph.

mod spec;
mod traits;
mod types;

pub use spec::{
    Activation, AttentionSpec, DecoderLayerSpec, DecoderSpec, DenseFeedForwardSpec,
    FeedForwardSpec, Mamba2Spec, MixerSpec, MixtureOfExpertsSpec, MlaSpec, NormPlacement,
    NormalizationSpec, PositionEncodingSpec, ResidualTopology, RopeScaling, Seq2SeqSpec,
    SlidingWindowSpec,
};
pub use traits::{
    ArchitectureConfig, CausalLanguageModel, DecodeState, HookedModel, Seq2SeqLanguageModel,
    StagedModel,
};
pub use types::{
    CausalLmInput, CausalLmOutput, DecodeInput, EncoderOutput, ForwardOptions, LogitsSelection,
    RouterDiagnostics, Seq2SeqInput, Seq2SeqOutput, TokenMask,
};
pub(crate) use types::{bool_tensor_from_values, select_bool_tensor, tensor_bool_values};
