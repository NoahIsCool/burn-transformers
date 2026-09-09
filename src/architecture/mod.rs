//! Public architecture contracts and reusable model specifications.

mod spec;
mod traits;
mod types;

pub use spec::{
    Activation, AttentionSpec, DecoderLayerSpec, DecoderSpec, DenseFeedForwardSpec,
    FeedForwardSpec, Mamba2Spec, MixerSpec, MixtureOfExpertsSpec, MlaSpec, NormPlacement,
    NormalizationSpec, PositionEncodingSpec, ResidualTopology, RopeScaling, Seq2SeqSpec,
    SlidingWindowSpec,
};
pub use traits::{ArchitectureConfig, CausalLanguageModel, DecodeState, Seq2SeqLanguageModel};
pub use types::{
    CausalLmInput, CausalLmOutput, DecodeInput, EncoderOutput, ForwardOptions, LogitsSelection,
    RouterDiagnostics, Seq2SeqInput, Seq2SeqOutput, TokenMask,
};
pub(crate) use types::{bool_tensor_from_values, select_bool_tensor, tensor_bool_values};
