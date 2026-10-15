//! Mamba-2 in modular specifications: translation, shared-kernel parity, padding, caching,
//! causality, hooks, initialization, and gradients.
//!
//! The modular Mamba-2 block must be exactly the shared [`crate::mamba2`] mixer
//! wired into a residual block: the same weights give the same output, padded
//! rows equal their unpadded scans, cached decoding equals the stateless
//! forward (in causal stacks and in seq2seq decoders with cross-attention),
//! and every non-causal execution request is rejected.

use burn::{
    module::Module,
    tensor::{DType, Int, Tensor, TensorData, Tolerance, backend::BackendTypes},
};

use super::*;
use crate::{
    CausalLanguageModel, CausalLmInput, DecodeInput, DecodeState, EncoderOutput, ForwardOptions,
    LogitsSelection, Seq2SeqInput, Seq2SeqLanguageModel,
    architecture::{
        DecoderSpec, FeedForwardSpec, Mamba2Spec, PositionEncodingSpec, RopeScaling, Seq2SeqSpec,
    },
    cache::CacheConfig,
    experiment::ExecutionContext,
    intervention::{Hooks, Layer, Stack},
    models::custom::{
        ModularCausalLm, ModularDecodeState, ModularLayer, ModularSeq2SeqLm,
        test_support::{
            DeterministicParameters, TestBackend, attention, decoder, layer, mask, rms,
            seq2seq_spec,
        },
    },
};

/// Device of the deterministic CPU fixtures.
type Device = <TestBackend as BackendTypes>::Device;

/// A two-group Mamba-2 geometry: four heads of width two, state four, kernel three.
fn mamba_spec() -> Mamba2Spec {
    Mamba2Spec {
        heads: 4,
        head_dim: 2,
        state_dim: 4,
        convolution_kernel: 3,
        group_size: 2,
    }
}

/// A causal stack of RoPE attention followed by a Mamba-2 block.
fn hybrid_spec() -> DecoderSpec {
    decoder(vec![
        layer(attention(PositionEncodingSpec::Rope {
            theta: 10_000.0,
            rotary_dim: 4,
            scaling: RopeScaling::None,
        })),
        layer(MixerSpec::Mamba2(mamba_spec())),
    ])
}

/// Builds `spec` with deterministic, non-degenerate parameters.
fn hybrid_model(spec: DecoderSpec, device: &Device) -> ModularCausalLm<TestBackend> {
    ModularCausalLm::<TestBackend>::init(spec, device)
        .unwrap()
        .map(&mut DeterministicParameters::default())
}

/// Deterministic `[batch, tokens, 8]` residual inputs.
fn residuals(batch: usize, tokens: usize, device: &Device) -> Tensor<TestBackend, 3> {
    let values = (0..batch * tokens * 8)
        .map(|index| ((index * 5 + 1) as f32 * 0.37).sin())
        .collect::<Vec<_>>();
    Tensor::from_data(TensorData::new(values, [batch, tokens, 8]), device)
}

/// Token input with an optional mask and implicit positions.
fn tokens<const B: usize, const S: usize>(
    ids: [[i32; S]; B],
    valid: Option<[[bool; S]; B]>,
    device: &Device,
) -> CausalLmInput<TestBackend> {
    CausalLmInput {
        token_ids: Tensor::<TestBackend, 2, Int>::from_ints(ids, device),
        token_mask: valid.map(|valid| mask(valid, device)),
        position_ids: None,
    }
}

/// Logits for every position.
fn all_logits() -> ForwardOptions {
    ForwardOptions {
        logits: LogitsSelection::All,
        ..ForwardOptions::default()
    }
}

/// F32 tolerance for re-associated chunked scans (padding adds exact identity steps).
fn scan_tolerance() -> Tolerance<f32> {
    Tolerance::rel_abs(1e-5, 1e-5)
}

/// The spec's geometry and the documented Hugging Face defaults reach the shared kernel.
#[test]
fn mamba2_spec_translates_to_the_shared_kernel_with_reference_defaults() {
    let config = mamba2_config(&mamba_spec(), 8, 1e-5);
    assert_eq!(
        config,
        Mamba2Config {
            hidden_size: 8,
            num_heads: 4,
            head_dim: 2,
            state_size: 4,
            groups: 2,
            convolution_kernel: 3,
            chunk_size: 256,
            time_step_clamp: TimeStepClamp::chunked_only(TimeStepLimits::at_least(0.0)),
            normalization_epsilon: 1e-5,
            projection_bias: false,
            convolution_bias: true,
        }
    );
    config.validate().unwrap();
    // Group sizes that are zero or do not divide the heads (both rejected by
    // spec validation) translate to zero groups, which the kernel rejects
    // instead of dividing by zero or silently regrouping the heads.
    for group_size in [0, 3] {
        let degenerate = Mamba2Spec {
            group_size,
            ..mamba_spec()
        };
        assert_eq!(mamba2_config(&degenerate, 8, 1e-5).groups, 0);
        assert!(mamba2_config(&degenerate, 8, 1e-5).validate().is_err());
    }
}

/// The modular mixer is the shared Mamba-2 mixer: identical weights give identical outputs.
#[test]
fn modular_mamba2_mixer_equals_the_shared_kernel() {
    let device = Default::default();
    let mixer =
        ModularMixer::<TestBackend>::init(&MixerSpec::Mamba2(mamba_spec()), 8, rms(), &device)
            .unwrap()
            .map(&mut DeterministicParameters::default());
    let ModularMixer::Mamba2(mamba) = &mixer else {
        panic!("a Mamba-2 spec builds the Mamba-2 mixer");
    };
    assert_eq!(
        mamba.config(),
        mamba2_config(&mamba_spec(), 8, rms().epsilon())
    );
    assert_eq!(mixer.connectivity(), None);
    assert_eq!(mixer.attention_heads(), None);
    assert_eq!(mixer.recurrent_name(), Some("Mamba2"));
    assert!(mixer.activation_points(Layer::decoder(0)).is_empty());

    let hidden = residuals(2, 5, &device);
    let context = ExecutionContext::contiguous(2, 5, &device).unwrap();
    let pass = StackPass::from_context(&context, Stack::Decoder, DType::F32);
    let modular = mixer
        .forward(hidden.clone(), &pass, Layer::decoder(0), &mut Hooks::new())
        .unwrap();
    let shared = mamba
        .forward(hidden.clone(), None, Mamba2Mode::ChunkedPrefill)
        .unwrap();
    modular
        .clone()
        .into_data()
        .assert_approx_eq::<f32>(&shared.into_data(), Tolerance::absolute(0.0));
    // The chunked scan agrees with the transparent token-by-token recurrence.
    let recurrent = mamba.forward(hidden, None, Mamba2Mode::Recurrent).unwrap();
    modular
        .into_data()
        .assert_approx_eq::<f32>(&recurrent.into_data(), scan_tolerance());
}

/// Left- and right-padded rows of a Mamba-2 stack equal their unpadded forwards.
#[test]
fn mamba2_padding_is_semantic_in_both_directions() {
    let device = Default::default();
    let model = hybrid_model(hybrid_spec(), &device);
    let unpadded = model
        .forward(tokens([[5, 6, 7]], None, &device), &all_logits())
        .unwrap()
        .logits;
    let batched = model
        .forward(
            tokens(
                [[0, 0, 5, 6, 7], [5, 6, 7, 0, 0], [1, 2, 3, 4, 5]],
                Some([
                    [false, false, true, true, true],
                    [true, true, true, false, false],
                    [true; 5],
                ]),
                &device,
            ),
            &all_logits(),
        )
        .unwrap()
        .logits;
    let expected = unpadded.into_data();
    batched
        .clone()
        .slice([0..1, 2..5])
        .into_data()
        .assert_approx_eq::<f32>(&expected, scan_tolerance());
    batched
        .clone()
        .slice([1..2, 0..3])
        .into_data()
        .assert_approx_eq::<f32>(&expected, scan_tolerance());
    // The fully valid row is unaffected by its padded neighbours.
    let alone = model
        .forward(tokens([[1, 2, 3, 4, 5]], None, &device), &all_logits())
        .unwrap()
        .logits;
    batched
        .narrow(0, 2, 1)
        .into_data()
        .assert_approx_eq::<f32>(&alone.into_data(), scan_tolerance());
}

/// Ragged cached prefill, decode, and rollback agree with the stateless forward.
#[test]
fn mamba2_cached_decoding_matches_the_stateless_forward() {
    let device = Default::default();
    let model = hybrid_model(hybrid_spec(), &device);
    let options = ForwardOptions::default();
    let mut state = model
        .new_decode_state(CacheConfig::new(2, 8).unwrap(), 2, &device)
        .unwrap();
    // Row 0 prefills [3, 4, 5]; row 1 is left-padded and prefills [6, 7].
    model
        .prefill(
            tokens(
                [[3, 4, 5], [0, 6, 7]],
                Some([[true; 3], [false, true, true]]),
                &device,
            ),
            &mut state,
            &options,
        )
        .unwrap();
    assert_eq!(state.lengths(), [3, 2]);
    let step = |state: &mut ModularDecodeState<TestBackend>, ids: [[i32; 1]; 2]| {
        model
            .decode(
                DecodeInput {
                    token_ids: Tensor::from_ints(ids, &device),
                    token_mask: None,
                    position_ids: None,
                },
                state,
                &options,
            )
            .unwrap()
            .logits
    };
    let last = |ids: &[i32]| {
        let input = CausalLmInput {
            token_ids: Tensor::<TestBackend, 2, Int>::from_data(
                TensorData::new(ids.to_vec(), [1, ids.len()]),
                &device,
            ),
            token_mask: None,
            position_ids: None,
        };
        model.forward(input, &options).unwrap().logits
    };
    let assert_rows = |cached: Tensor<TestBackend, 3>, rows: [&[i32]; 2]| {
        for (row, history) in rows.into_iter().enumerate() {
            cached
                .clone()
                .narrow(0, row, 1)
                .into_data()
                .assert_approx_eq::<f32>(&last(history).into_data(), scan_tolerance());
        }
    };
    let first = step(&mut state, [[8], [9]]);
    assert_rows(first, [&[3, 4, 5, 8], &[6, 7, 9]]);
    let second = step(&mut state, [[10], [11]]);
    assert_rows(second, [&[3, 4, 5, 8, 10], &[6, 7, 9, 11]]);

    // Rolling back recomputes the recurrence from the retained prefix.
    state.truncate(&[4, 2]).unwrap();
    let rewound = step(&mut state, [[12], [13]]);
    assert_rows(rewound, [&[3, 4, 5, 8, 12], &[6, 7, 13]]);
    // Reordering moves each row's whole recurrent history.
    state.reorder(&[1, 0]).unwrap();
    let swapped = step(&mut state, [[14], [15]]);
    assert_rows(swapped, [&[6, 7, 13, 14], &[3, 4, 5, 8, 12, 15]]);
}

/// A seq2seq model whose decoder runs RoPE attention, then Mamba-2, each followed by cross-attention.
///
/// The encoder is the fixture's single bidirectional attention block, so the
/// decoder's Mamba-2 state is the only recurrent state in the graph.
fn hybrid_seq2seq(device: &Device) -> ModularSeq2SeqLm<TestBackend> {
    ModularSeq2SeqLm::<TestBackend>::init(
        Seq2SeqSpec {
            decoder: hybrid_spec(),
            ..seq2seq_spec()
        },
        device,
    )
    .unwrap()
    .map(&mut DeterministicParameters::default())
}

/// One unpadded row of token IDs, of any length.
fn row(ids: &[i32], device: &Device) -> CausalLmInput<TestBackend> {
    CausalLmInput {
        token_ids: Tensor::from_data(TensorData::new(ids.to_vec(), [1, ids.len()]), device),
        token_mask: None,
        position_ids: None,
    }
}

/// Three ragged source rows: full, right-padded to `[7, 8]`, and full.
fn ragged_sources(device: &Device) -> CausalLmInput<TestBackend> {
    tokens(
        [[4, 5, 6], [7, 8, 0], [4, 8, 6]],
        Some([[true; 3], [true, true, false], [true; 3]]),
        device,
    )
}

/// The unpadded source of each row of [`ragged_sources`].
const SOURCES: [&[i32]; 3] = [&[4, 5, 6], &[7, 8], &[4, 8, 6]];

/// Decoder padding around a Mamba-2 block is semantic under cross-attention too.
///
/// Each row of a ragged batch (left-padded target, right-padded target and
/// source, fully valid) must equal the unpadded forward of that row alone.
#[test]
fn mamba2_seq2seq_decoder_padding_matches_unpadded_rows() {
    let device = Default::default();
    let model = hybrid_seq2seq(&device);
    let batched = model
        .forward(
            Seq2SeqInput {
                encoder: ragged_sources(&device),
                decoder: tokens(
                    [[0, 0, 1, 2, 3], [1, 2, 3, 0, 0], [1, 2, 3, 4, 5]],
                    Some([
                        [false, false, true, true, true],
                        [true, true, true, false, false],
                        [true; 5],
                    ]),
                    &device,
                ),
            },
            &all_logits(),
        )
        .unwrap()
        .decoder
        .logits;
    let alone = |source: &[i32], target: &[i32]| {
        model
            .forward(
                Seq2SeqInput {
                    encoder: row(source, &device),
                    decoder: row(target, &device),
                },
                &all_logits(),
            )
            .unwrap()
            .decoder
            .logits
            .into_data()
    };
    for (batch_row, valid, target) in [
        (0, 2..5, &[1, 2, 3][..]),
        (1, 0..3, &[1, 2, 3][..]),
        (2, 0..5, &[1, 2, 3, 4, 5][..]),
    ] {
        batched
            .clone()
            .slice([batch_row..batch_row + 1, valid])
            .into_data()
            .assert_approx_eq::<f32>(&alone(SOURCES[batch_row], target), scan_tolerance());
    }
}

/// Ragged cached seq2seq decoding through Mamba-2 and cross-attention matches the stateless forward.
///
/// Prefill a left-padded target batch, decode two steps, roll one row back,
/// and reorder the rows (re-encoding the sources in the new order, since the
/// caller owns the encoder output). Every step's logits must equal the last
/// position of a fresh stateless forward over that row's source and history.
#[test]
fn mamba2_seq2seq_cached_decoding_matches_the_stateless_forward() {
    let device = Default::default();
    let model = hybrid_seq2seq(&device);
    let options = ForwardOptions::default();
    let sources = |order: [usize; 2]| {
        let ids = order.map(|source| {
            let mut ids = [0; 3];
            ids[..SOURCES[source].len()].copy_from_slice(SOURCES[source]);
            ids
        });
        let valid = order.map(|source| [0, 1, 2].map(|column| column < SOURCES[source].len()));
        model
            .encode(tokens(ids, Some(valid), &device), &options)
            .unwrap()
    };
    let mut encoder = sources([0, 1]);
    let mut state = model
        .new_decode_state(CacheConfig::new(2, 8).unwrap(), 2, &device)
        .unwrap();
    // Row 0 prefills [0, 1, 2]; row 1 is left-padded and prefills [0, 3].
    model
        .prefill_decoder(
            &encoder,
            tokens(
                [[0, 1, 2], [0, 0, 3]],
                Some([[true; 3], [false, true, true]]),
                &device,
            ),
            &mut state,
            &options,
        )
        .unwrap();
    assert_eq!(state.lengths(), [3, 2]);
    let step = |encoder: &EncoderOutput<TestBackend>,
                state: &mut ModularDecodeState<TestBackend>,
                ids: [[i32; 1]; 2]| {
        model
            .decode(
                encoder,
                DecodeInput {
                    token_ids: Tensor::from_ints(ids, &device),
                    token_mask: None,
                    position_ids: None,
                },
                state,
                &options,
            )
            .unwrap()
            .decoder
            .logits
    };
    let assert_rows = |cached: Tensor<TestBackend, 3>, rows: [(usize, &[i32]); 2]| {
        for (batch_row, (source, history)) in rows.into_iter().enumerate() {
            let expected = model
                .forward(
                    Seq2SeqInput {
                        encoder: row(SOURCES[source], &device),
                        decoder: row(history, &device),
                    },
                    &options,
                )
                .unwrap()
                .decoder
                .logits;
            cached
                .clone()
                .narrow(0, batch_row, 1)
                .into_data()
                .assert_approx_eq::<f32>(&expected.into_data(), scan_tolerance());
        }
    };
    let first = step(&encoder, &mut state, [[4], [5]]);
    assert_rows(first, [(0, &[0, 1, 2, 4]), (1, &[0, 3, 5])]);
    let second = step(&encoder, &mut state, [[6], [7]]);
    assert_rows(second, [(0, &[0, 1, 2, 4, 6]), (1, &[0, 3, 5, 7])]);

    // Rolling back recomputes the recurrence from the retained prefix.
    state.truncate(&[4, 2]).unwrap();
    let rewound = step(&encoder, &mut state, [[8], [9]]);
    assert_rows(rewound, [(0, &[0, 1, 2, 4, 8]), (1, &[0, 3, 9])]);
    // Reordering moves each row's recurrent history; the caller reorders the sources.
    state.reorder(&[1, 0]).unwrap();
    encoder = sources([1, 0]);
    let swapped = step(&encoder, &mut state, [[10], [11]]);
    assert_rows(swapped, [(1, &[0, 3, 9, 10]), (0, &[0, 1, 2, 4, 8, 11])]);
}

/// Mamba-2 is a causal scan: bidirectional, encoder, and non-causal policies are rejected.
#[test]
fn mamba2_rejects_every_noncausal_execution() {
    let device = Default::default();
    let model = hybrid_model(hybrid_spec(), &device);
    let error = model
        .forward_bidirectional(tokens([[1, 2]], None, &device), &ForwardOptions::default())
        .unwrap_err();
    assert!(
        matches!(error, crate::Error::BackendCapability(_)),
        "{error:?}"
    );
    assert!(error.to_string().contains("Mamba2"));

    let encoder = Seq2SeqSpec {
        encoder: decoder(vec![layer(MixerSpec::Mamba2(mamba_spec()))]),
        ..seq2seq_spec()
    };
    let error = ModularSeq2SeqLm::<TestBackend>::init(encoder, &device).unwrap_err();
    assert!(
        error.to_string().contains("causal recurrent scan"),
        "{error}"
    );

    // A seq2seq decoder may use Mamba-2: its self-mixing is causal.
    let decoder_side = Seq2SeqSpec {
        decoder: decoder(vec![layer(MixerSpec::Mamba2(mamba_spec()))]),
        ..seq2seq_spec()
    };
    ModularSeq2SeqLm::<TestBackend>::init(decoder_side, &device).unwrap();

    // A standalone block evaluated as an encoder layer defaults to bidirectional mixing.
    let block = ModularLayer::<TestBackend>::init(
        &layer(MixerSpec::Mamba2(mamba_spec())),
        8,
        None,
        &device,
    )
    .unwrap();
    let context = ExecutionContext::contiguous(1, 3, &device).unwrap();
    let error = block
        .run(residuals(1, 3, &device), &context, Layer::encoder(0))
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("Mamba2 requires ordinary causal execution")
    );
    block
        .run(residuals(1, 3, &device), &context, Layer::decoder(0))
        .expect("decoder-stack execution is causal");
}

/// The mixer boundaries of a Mamba-2 block are hookable, and editing them changes the logits.
#[test]
fn mamba2_blocks_expose_their_mixer_boundaries() {
    let device = Default::default();
    let model = hybrid_model(hybrid_spec(), &device);
    let layer = Layer::decoder(1);
    let points = model.activation_points();
    assert!(points.contains(&layer.mixer_input().info()));
    assert!(points.contains(&layer.mixer_output().info()));
    let input = tokens([[1, 2, 3]], None, &device);
    let clean = model.forward(input.clone(), &all_logits()).unwrap().logits;
    let mut hooks = Hooks::new();
    let mixed = hooks.capture(layer.mixer_output());
    hooks.edit(layer.mixer_output().token(2), |output| output.zeros_like());
    let edited = model
        .forward_with_hooks(input, &all_logits(), &mut hooks)
        .unwrap()
        .logits;
    assert_eq!(hooks.get(&mixed).unwrap().dims(), [1, 3, 8]);
    // Only the edited token's logits change: the mixer output feeds no other position.
    edited
        .clone()
        .slice([0..1, 0..2])
        .into_data()
        .assert_approx_eq::<f32>(
            &clean.clone().slice([0..1, 0..2]).into_data(),
            Tolerance::absolute(0.0),
        );
    let change = (edited.slice([0..1, 2..3]) - clean.slice([0..1, 2..3]))
        .abs()
        .max()
        .into_scalar();
    assert!(change > 1e-6);
}

/// The spec-derived policy depth-scales the Mamba-2 output projection.
#[test]
fn scaled_init_policy_covers_mamba2_output_projections() {
    let device = Default::default();
    let mut block = layer(MixerSpec::Mamba2(mamba_spec()));
    block.feed_forward = FeedForwardSpec::None;
    let spec = decoder(vec![block.clone(), block]);
    let policy = ModularCausalLm::<TestBackend>::scaled_init_policy(&spec, 0.02);
    assert_eq!(policy.residual_projections, ["out_proj"]);
    let model = ModularCausalLm::<TestBackend>::init_with(spec, &policy, &device).unwrap();
    let logits = model
        .forward(tokens([[1, 2, 3]], None, &device), &all_logits())
        .unwrap()
        .logits;
    assert!(logits.is_finite().all().into_scalar());
}

/// Every Mamba-2 parameter, and the embeddings beneath it, receive gradients.
#[cfg(feature = "autodiff")]
#[test]
fn mamba2_modular_graph_is_autodiff_safe() {
    use burn::backend::Autodiff;

    /// Reverse-mode CPU backend.
    type AdBackend = Autodiff<TestBackend>;
    let device = Default::default();
    let mixer =
        ModularMixer::<AdBackend>::init(&MixerSpec::Mamba2(mamba_spec()), 8, rms(), &device)
            .unwrap();
    let ModularMixer::Mamba2(mamba) = &mixer else {
        panic!("a Mamba-2 spec builds the Mamba-2 mixer");
    };
    let values = (0..2 * 4 * 8)
        .map(|index| ((index * 3 + 2) as f32 * 0.29).cos())
        .collect::<Vec<_>>();
    let hidden = Tensor::<AdBackend, 3>::from_data(TensorData::new(values, [2, 4, 8]), &device);
    let context = ExecutionContext::contiguous(2, 4, &device).unwrap();
    let pass = StackPass::from_context(&context, Stack::Decoder, DType::F32);
    let gradients = mixer
        .forward(hidden, &pass, Layer::decoder(0), &mut Hooks::new())
        .unwrap()
        .square()
        .sum()
        .backward();
    for (name, present) in [
        (
            "in_proj.weight",
            mamba.in_proj.weight.grad(&gradients).is_some(),
        ),
        (
            "conv1d.weight",
            mamba.conv1d.weight.grad(&gradients).is_some(),
        ),
        (
            "conv1d.bias",
            mamba
                .conv1d
                .bias
                .as_ref()
                .is_some_and(|bias| bias.grad(&gradients).is_some()),
        ),
        ("dt_bias", mamba.dt_bias.grad(&gradients).is_some()),
        ("A_log", mamba.a_log.grad(&gradients).is_some()),
        ("D", mamba.d.grad(&gradients).is_some()),
        ("norm.weight", mamba.norm.weight.grad(&gradients).is_some()),
        (
            "out_proj.weight",
            mamba.out_proj.weight.grad(&gradients).is_some(),
        ),
    ] {
        assert!(present, "{name} received no gradient");
    }

    let model = ModularCausalLm::<AdBackend>::init(hybrid_spec(), &device).unwrap();
    let gradients = model
        .forward(
            CausalLmInput {
                token_ids: Tensor::<AdBackend, 2, Int>::from_ints([[1, 2, 3]], &device),
                token_mask: None,
                position_ids: None,
            },
            &ForwardOptions::default(),
        )
        .unwrap()
        .logits
        .sum()
        .backward();
    assert!(model.embed_tokens.weight.grad(&gradients).is_some());
}
