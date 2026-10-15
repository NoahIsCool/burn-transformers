//! Shared fixtures for the modular-model test suites.
//!
//! Every suite builds tiny `Flex<f32, i32>` graphs from the same compact
//! specifications and initializes parameters deterministically, so failures
//! reproduce without random seeds.

use burn::{
    backend::Flex,
    module::{ModuleMapper, Param},
    tensor::{Bool, Tensor, TensorData, backend::BackendTypes},
};

use crate::{
    TokenMask,
    architecture::{
        Activation, AttentionOptions, AttentionSpec, DecoderLayerSpec, DecoderSpec,
        DenseFeedForwardSpec, FeedForwardSpec, MixerSpec, NormPlacement, NormalizationSpec,
        PositionEncodingSpec, ResidualTopology, RopeScaling, Seq2SeqSpec,
    },
};

/// CPU backend with F32 values and I32 indices for compact modular-model fixtures.
pub(super) type TestBackend = Flex<f32, i32>;

/// Standard scale-only normalization for compact fixtures.
pub(super) fn rms() -> NormalizationSpec {
    NormalizationSpec::Rms {
        epsilon: 1e-6,
        zero_centered: false,
    }
}

/// Full two-head attention without an explicit position transform.
pub(super) fn attention(position: PositionEncodingSpec) -> MixerSpec {
    MixerSpec::Attention(AttentionSpec {
        query_heads: 2,
        key_value_heads: 1,
        head_dim: 4,
        query_key_norm: true,
        output_gate: false,
        position,
    })
}

/// Bias-free gated feed-forward branch used by most fixtures.
pub(super) fn dense_ffn() -> FeedForwardSpec {
    FeedForwardSpec::Dense(DenseFeedForwardSpec {
        intermediate_size: 12,
        activation: Activation::Silu,
        bias: false,
    })
}

/// One pre-normalized additive residual layer.
pub(super) fn layer(mixer: MixerSpec) -> DecoderLayerSpec {
    DecoderLayerSpec {
        mixer,
        feed_forward: dense_ffn(),
        mixer_norm: rms(),
        feed_forward_norm: rms(),
        norm_placement: NormPlacement::Pre,
        residual: ResidualTopology::Add,
    }
}

/// Small decoder whose layer schedule is supplied by the test.
pub(super) fn decoder(layers: Vec<DecoderLayerSpec>) -> DecoderSpec {
    DecoderSpec {
        vocabulary_size: 32,
        hidden_size: 8,
        layers,
        final_norm: rms(),
        tie_word_embeddings: false,
        max_position_embeddings: 32,
    }
}

/// Small causal decoder with RoPE attention followed by DeltaNet.
pub(super) fn causal_spec() -> DecoderSpec {
    decoder(vec![
        layer(attention(PositionEncodingSpec::Rope {
            theta: 10_000.0,
            rotary_dim: 4,
            scaling: RopeScaling::None,
        })),
        layer(MixerSpec::DeltaNet {
            key_heads: 1,
            value_heads: 2,
            key_head_dim: 4,
            value_head_dim: 4,
            convolution_kernel: 2,
        }),
    ])
}

/// One-layer encoder-decoder fixture with plain absolute-free attention.
pub(super) fn seq2seq_spec() -> Seq2SeqSpec {
    Seq2SeqSpec {
        encoder: decoder(vec![layer(attention(PositionEncodingSpec::None))]),
        decoder: decoder(vec![layer(attention(PositionEncodingSpec::None))]),
        cross_attention_heads: 2,
        cross_attention_head_dim: 4,
        cross_attention_options: AttentionOptions::default(),
    }
}

/// Constructs a token-validity mask from a nested boolean array.
pub(super) fn mask<const B: usize, const S: usize>(
    values: [[bool; S]; B],
    device: &<TestBackend as BackendTypes>::Device,
) -> TokenMask<TestBackend> {
    TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
        values.into(),
        device,
    ))
}

/// Replaces every lazy float parameter with reproducible, non-degenerate fixture values.
///
/// The mapping consumes the graph and initializes every parameter before any model clone can
/// occur. Matrix entries vary by their ordinal and parameter traversal order, while norm scales
/// remain one. This avoids random global seeds and independently initialized lazy clones.
#[derive(Default)]
pub(super) struct DeterministicParameters {
    /// Stable traversal ordinal used to distinguish equally shaped projection matrices.
    ordinal: usize,
}

impl ModuleMapper<TestBackend> for DeterministicParameters {
    /// Materializes deterministic weights without changing the original parameter identities.
    fn map_float<const D: usize>(
        &mut self,
        param: Param<Tensor<TestBackend, D>>,
    ) -> Param<Tensor<TestBackend, D>> {
        self.ordinal += 1;
        let ordinal = self.ordinal;
        param.map(|tensor| {
            let shape = tensor.dims();
            let values = (0..shape.iter().product())
                .map(|index| {
                    if D == 1 {
                        1.0
                    } else {
                        (((index + 1) * (ordinal + 3)) as f32 * 0.17).sin() * 0.2
                    }
                })
                .collect::<Vec<_>>();
            Tensor::from_data(TensorData::new(values, shape), &tensor.device())
        })
    }
}
