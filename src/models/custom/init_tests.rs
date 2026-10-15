//! Spec-derived initialization policies: residual-projection names and initial logit scale.

use burn::{
    backend::Flex,
    tensor::{Int, Tensor, TensorData, backend::Backend},
};

use super::*;
use crate::{
    CausalLmInput, ForwardOptions, LogitsSelection,
    architecture::{AttentionSpec, PositionEncodingSpec},
    models::custom::{
        ModularCausalLm, ModularSeq2SeqLm,
        test_support::{attention, decoder, layer, seq2seq_spec},
    },
};

/// Residual-projection names present in a policy, sorted.
fn names(policy: &InitPolicy) -> Vec<&str> {
    let mut names = policy
        .residual_projections
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>();
    names.sort_unstable();
    names
}

/// Names follow the blocks actually present, and every policy applies to its model.
#[test]
fn residual_projection_names_follow_the_blocks_present() {
    let device = Default::default();
    let mut attention_only = layer(attention(PositionEncodingSpec::None));
    attention_only.feed_forward = FeedForwardSpec::None;
    let delta = layer(MixerSpec::DeltaNet {
        key_heads: 1,
        value_heads: 2,
        key_head_dim: 4,
        value_head_dim: 4,
        convolution_kernel: 2,
    });
    let cases = [
        (vec![attention_only.clone()], vec!["o_proj"]),
        (vec![delta.clone()], vec!["down_proj", "out_proj"]),
        (
            vec![attention_only, delta],
            vec!["down_proj", "o_proj", "out_proj"],
        ),
    ];
    for (layers, expected) in cases {
        let spec = decoder(layers);
        let policy = ModularCausalLm::<Flex>::scaled_init_policy(&spec, 0.02);
        assert_eq!(names(&policy), expected);
        assert_eq!(
            policy.residual_scaling,
            crate::initialization::ResidualScaling::DepthScaled {
                layers: spec.layers.len()
            }
        );
        ModularCausalLm::<Flex>::init_with(spec, &policy, &device).unwrap();
    }
    let policy = ModularSeq2SeqLm::<Flex>::scaled_init_policy(&seq2seq_spec(), 0.02);
    assert_eq!(names(&policy), vec!["down_proj", "o_proj"]);
    ModularSeq2SeqLm::<Flex>::init_with(seq2seq_spec(), &policy, &device).unwrap();
}

/// CUS-8: a tied readout starts with `O(1)` logits under the scaled policy, `~sqrt(d)` without.
#[test]
fn scaled_initialization_keeps_tied_logits_well_conditioned() {
    /// CPU backend with F32 values and I32 token IDs.
    type B = Flex<f32, i32>;
    let device = Default::default();
    let width = 256;
    let mut spec = decoder(vec![layer(MixerSpec::Attention(AttentionSpec {
        query_heads: 4,
        key_value_heads: 4,
        head_dim: 64,
        query_key_norm: false,
        output_gate: false,
        position: PositionEncodingSpec::None,
    }))]);
    spec.hidden_size = width;
    spec.vocabulary_size = 512;
    spec.tie_word_embeddings = true;
    let ids = (0..32)
        .map(|token| (token * 37 + 11) % 512)
        .collect::<Vec<i32>>();
    let tokens = Tensor::<B, 2, Int>::from_data(TensorData::new(ids.clone(), [1, 32]), &device);
    // The input token's own logit contains its embedding twice and is excluded,
    // as in `crate::initialization`'s analysis.
    let logit_std = |model: ModularCausalLm<B>| {
        let logits = model
            .forward(
                CausalLmInput {
                    token_ids: tokens.clone(),
                    token_mask: None,
                    position_ids: None,
                },
                &ForwardOptions {
                    logits: LogitsSelection::All,
                    ..Default::default()
                },
            )
            .unwrap()
            .logits
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let others = logits
            .chunks(512)
            .zip(&ids)
            .flat_map(|(row, &own)| {
                row.iter()
                    .enumerate()
                    .filter(move |(column, _)| *column != own as usize)
                    .map(|(_, value)| f64::from(*value))
            })
            .collect::<Vec<_>>();
        let mean = others.iter().sum::<f64>() / others.len() as f64;
        (others
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / others.len() as f64)
            .sqrt()
    };
    B::seed(&device, 7);
    let default_std = logit_std(ModularCausalLm::init(spec.clone(), &device).unwrap());
    B::seed(&device, 7);
    let policy = ModularCausalLm::<B>::scaled_init_policy(&spec, 0.02);
    let scaled_std = logit_std(ModularCausalLm::init_with(spec, &policy, &device).unwrap());
    let root_width = (width as f64).sqrt();
    assert!(
        default_std > 0.5 * root_width,
        "Burn default std {default_std}"
    );
    assert!(
        (scaled_std / (0.02 * root_width) - 1.0).abs() < 0.3,
        "scaled std {scaled_std}"
    );
}
