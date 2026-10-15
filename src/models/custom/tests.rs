//! Tiny executable, validation, serialization, state-lifecycle, and autodiff tests.

use burn::{
    module::{Module, Param},
    tensor::{Int, Tensor, TensorData, Tolerance, backend::BackendTypes},
};

use crate::{
    CausalLanguageModel, CausalLmInput, DecodeInput, DecodeState, ForwardOptions, LogitsSelection,
    Seq2SeqInput, Seq2SeqLanguageModel, TokenMask,
    architecture::{
        Activation, AttentionSpec, DecoderSpec, FeedForwardSpec, Mamba2Spec, MixerSpec,
        MixtureOfExpertsSpec, MlaSpec, NormPlacement, NormalizationSpec, PositionEncodingSpec,
        ResidualTopology, RopeScaling, ScoreScale, SlidingWindowSpec,
    },
    attention::AttentionConnectivity,
    cache::CacheConfig,
    intervention::Stack,
    moe::RoutingPolicy,
    position::T5RelativePositionBiasConfig,
};

use super::{ModularCausalLm, ModularSeq2SeqLm, pass::StackPass, test_support::*};

#[path = "state_contracts_tests.rs"]
/// Sparse decode diagnostics and transactional failure contract tests.
mod state_contracts;

#[path = "family_parity_tests.rs"]
/// Modular graphs configured like family blocks reproduce the family outputs.
mod family_parity;

#[path = "perf_tests.rs"]
/// Ignored release-mode timings of the pass tables and batched history writes.
mod perf;

/// Builds a fully initialized single-layer full/sliding attention graph for parity tests.
fn deterministic_attention_model(
    window: Option<usize>,
    position: PositionEncodingSpec,
    device: &<TestBackend as BackendTypes>::Device,
) -> ModularCausalLm<TestBackend> {
    let attention = AttentionSpec {
        query_heads: 2,
        key_value_heads: 1,
        head_dim: 4,
        query_key_norm: true,
        output_gate: false,
        position,
    };
    let mixer = match window {
        Some(window) => MixerSpec::SlidingAttention(SlidingWindowSpec { attention, window }),
        None => MixerSpec::Attention(attention),
    };
    ModularCausalLm::init(decoder(vec![layer(mixer)]), device)
        .unwrap()
        .map(&mut DeterministicParameters::default())
}

/// Constructs one unbatched token row with optional semantic mask and exact RoPE positions.
fn parity_input(
    ids: &[i32],
    validity: Option<&[bool]>,
    positions: Option<&[i32]>,
    device: &<TestBackend as BackendTypes>::Device,
) -> CausalLmInput<TestBackend> {
    let shape = [1, ids.len()];
    CausalLmInput {
        token_ids: Tensor::from_data(TensorData::new(ids.to_vec(), shape), device),
        token_mask: validity.map(|values| {
            assert_eq!(values.len(), ids.len());
            TokenMask::new(Tensor::from_bool(
                TensorData::new(values.to_vec(), shape),
                device,
            ))
        }),
        position_ids: positions.map(|values| {
            assert_eq!(values.len(), ids.len());
            Tensor::from_data(TensorData::new(values.to_vec(), shape), device)
        }),
    }
}

/// Requires shape-preserving, finite F32 parity at the component acceptance tolerance.
fn assert_finite_attention_parity(actual: &TensorData, expected: &TensorData) {
    assert_eq!(actual.shape, expected.shape);
    assert!(
        actual
            .as_slice::<f32>()
            .unwrap()
            .iter()
            .all(|value| value.is_finite())
    );
    assert!(
        expected
            .as_slice::<f32>()
            .unwrap()
            .iter()
            .all(|value| value.is_finite())
    );
    actual.assert_approx_eq::<f32>(expected, Tolerance::absolute(1e-4));
}

/// Compares one token's vocabulary vector while keeping batch/sequence axes explicit.
fn assert_token_logits_match(
    actual: &Tensor<TestBackend, 3>,
    actual_row: usize,
    actual_column: usize,
    expected: &Tensor<TestBackend, 3>,
    expected_column: usize,
) {
    let actual = actual
        .clone()
        .slice([
            actual_row..actual_row + 1,
            actual_column..actual_column + 1,
            0..32,
        ])
        .into_data();
    let expected = expected
        .clone()
        .slice([0..1, expected_column..expected_column + 1, 0..32])
        .into_data();
    assert_finite_attention_parity(&actual, &expected);
}

/// Masked columns must not consume a trailing window when cached history compacts them.
#[test]
fn sliding_reference_parity_with_interior_padding() {
    let device = Default::default();
    for sliding in [false, true] {
        let model = deterministic_attention_model(
            sliding.then_some(2),
            PositionEncodingSpec::None,
            &device,
        );
        let options = ForwardOptions::default();
        let stateless = model
            .forward(
                CausalLmInput {
                    token_ids: Tensor::from_ints([[1, 0, 2]], &device),
                    token_mask: Some(mask([[true, false, true]], &device)),
                    position_ids: None,
                },
                &options,
            )
            .unwrap()
            .logits
            .into_data();
        let compact = model
            .forward(
                CausalLmInput {
                    token_ids: Tensor::from_ints([[1, 2]], &device),
                    token_mask: None,
                    position_ids: None,
                },
                &options,
            )
            .unwrap()
            .logits
            .into_data();
        let mut prefill_state = model
            .new_decode_state(CacheConfig::new(1, 8).unwrap(), 1, &device)
            .unwrap();
        let prefill = model
            .prefill(
                CausalLmInput {
                    token_ids: Tensor::from_ints([[1, 0, 2]], &device),
                    token_mask: Some(mask([[true, false, true]], &device)),
                    position_ids: None,
                },
                &mut prefill_state,
                &options,
            )
            .unwrap()
            .logits
            .into_data();
        let mut decode_state = model
            .new_decode_state(CacheConfig::new(1, 8).unwrap(), 1, &device)
            .unwrap();
        model
            .prefill(
                CausalLmInput {
                    token_ids: Tensor::from_ints([[1, 0]], &device),
                    token_mask: Some(mask([[true, false]], &device)),
                    position_ids: None,
                },
                &mut decode_state,
                &options,
            )
            .unwrap();
        assert_eq!(decode_state.lengths(), &[1]);
        let decoded = model
            .decode(
                DecodeInput {
                    token_ids: Tensor::from_ints([[2]], &device),
                    token_mask: None,
                    position_ids: None,
                },
                &mut decode_state,
                &options,
            )
            .unwrap()
            .logits
            .into_data();
        assert_eq!(prefill_state.lengths(), &[2]);
        assert_eq!(decode_state.lengths(), &[2]);
        assert_finite_attention_parity(&compact, &prefill);
        assert_finite_attention_parity(&compact, &decoded);
        assert_finite_attention_parity(&stateless, &prefill);
    }
}

/// Left, right, interior, and explicitly all-valid padding preserve every valid token's logits.
#[test]
fn full_and_sliding_attention_ignore_padding_columns() {
    let device = Default::default();
    let layouts = [
        (vec![1, 2, 3], vec![true, true, true]),
        (vec![0, 0, 1, 2, 3], vec![false, false, true, true, true]),
        (vec![1, 2, 3, 0, 0], vec![true, true, true, false, false]),
        (
            vec![0, 1, 0, 2, 0, 3, 0],
            vec![false, true, false, true, false, true, false],
        ),
    ];
    for window in [None, Some(1), Some(2), Some(8), Some(usize::MAX)] {
        let model = deterministic_attention_model(
            window,
            PositionEncodingSpec::Rope {
                theta: 10_000.0,
                rotary_dim: 4,
                scaling: RopeScaling::None,
            },
            &device,
        );
        let options = ForwardOptions {
            logits: LogitsSelection::All,
            ..Default::default()
        };
        let compact = model
            .forward(parity_input(&[1, 2, 3], None, None, &device), &options)
            .unwrap()
            .logits;
        for (ids, validity) in &layouts {
            let padded = model
                .forward(parity_input(ids, Some(validity), None, &device), &options)
                .unwrap()
                .logits;
            for (compact_column, padded_column) in validity
                .iter()
                .enumerate()
                .filter_map(|(column, &valid)| valid.then_some(column))
                .enumerate()
            {
                assert_token_logits_match(&padded, 0, padded_column, &compact, compact_column);
            }
        }
    }
}

/// A window counts recent tokens, not gaps or offsets in explicit positional-encoding IDs.
#[test]
fn sliding_window_cardinality_is_independent_of_explicit_positions() {
    let device = Default::default();
    let ids = [1, 2, 3, 4];
    let positions = [2, 7, 15, 24];
    for window in [1, 2, 4, 8] {
        let model = deterministic_attention_model(
            Some(window),
            PositionEncodingSpec::Rope {
                theta: 10_000.0,
                rotary_dim: 4,
                scaling: RopeScaling::None,
            },
            &device,
        );
        let options = ForwardOptions::default();
        let start = ids.len().saturating_sub(window);
        // A one-layer graph's last query depends only on its visible raw token embeddings.
        // Evaluating exactly that suffix is an independent control for the window boundary.
        let suffix = model
            .forward(
                parity_input(&ids[start..], None, Some(&positions[start..]), &device),
                &options,
            )
            .unwrap()
            .logits
            .into_data();
        let full = model
            .forward(
                parity_input(&ids, None, Some(&positions), &device),
                &options,
            )
            .unwrap()
            .logits
            .into_data();
        assert_finite_attention_parity(&full, &suffix);
        let padded_input = || {
            parity_input(
                &[0, 1, 0, 2, 3, 0, 4, 0],
                Some(&[false, true, false, true, true, false, true, false]),
                Some(&[0, 2, 0, 7, 15, 0, 24, 0]),
                &device,
            )
        };
        let padded = model
            .forward(padded_input(), &options)
            .unwrap()
            .logits
            .into_data();
        assert_finite_attention_parity(&padded, &suffix);
        let mut state = model
            .new_decode_state(CacheConfig::new(1, 8).unwrap(), 1, &device)
            .unwrap();
        let cached = model
            .prefill(padded_input(), &mut state, &options)
            .unwrap()
            .logits
            .into_data();
        assert_finite_attention_parity(&cached, &suffix);
        assert_eq!(state.lengths(), &[4]);
    }
}

/// Ragged chunking, reorder, duplicate beams, and truncation preserve compact history semantics.
#[test]
fn masked_attention_continuations_survive_chunking_reorder_and_truncate() {
    let device = Default::default();
    for window in [None, Some(2)] {
        let model = deterministic_attention_model(
            window,
            PositionEncodingSpec::Rope {
                theta: 10_000.0,
                rotary_dim: 4,
                scaling: RopeScaling::None,
            },
            &device,
        );
        let all = ForwardOptions {
            logits: LogitsSelection::All,
            ..Default::default()
        };
        let mut state = model
            .new_decode_state(CacheConfig::new(2, 8).unwrap(), 2, &device)
            .unwrap();
        model
            .prefill(
                CausalLmInput {
                    token_ids: Tensor::from_ints([[1, 0, 2], [0, 3, 0]], &device),
                    token_mask: Some(mask([[true, false, true], [false, true, false]], &device)),
                    position_ids: Some(Tensor::from_ints([[2, 0, 5], [0, 7, 0]], &device)),
                },
                &mut state,
                &all,
            )
            .unwrap();
        assert_eq!(state.lengths(), &[2, 1]);
        let chunk = model
            .decode(
                DecodeInput {
                    token_ids: Tensor::from_ints([[0, 4, 5], [6, 0, 7]], &device),
                    token_mask: Some(mask([[false, true, true], [true, false, true]], &device)),
                    position_ids: Some(Tensor::from_ints([[0, 10, 13], [11, 0, 15]], &device)),
                },
                &mut state,
                &all,
            )
            .unwrap()
            .logits;
        assert_eq!(state.lengths(), &[4, 3]);
        let first = model
            .forward(
                parity_input(&[1, 2, 4, 5], None, Some(&[2, 5, 10, 13]), &device),
                &all,
            )
            .unwrap()
            .logits;
        let second = model
            .forward(
                parity_input(&[3, 6, 7], None, Some(&[7, 11, 15]), &device),
                &all,
            )
            .unwrap()
            .logits;
        let batched = model
            .forward(
                CausalLmInput {
                    token_ids: Tensor::from_ints([[1, 0, 2, 0, 4, 5], [0, 3, 0, 6, 0, 7]], &device),
                    token_mask: Some(mask(
                        [
                            [true, false, true, false, true, true],
                            [false, true, false, true, false, true],
                        ],
                        &device,
                    )),
                    position_ids: Some(Tensor::from_ints(
                        [[2, 0, 5, 0, 10, 13], [0, 7, 0, 11, 0, 15]],
                        &device,
                    )),
                },
                &all,
            )
            .unwrap()
            .logits;
        assert_token_logits_match(&batched, 0, 4, &first, 2);
        assert_token_logits_match(&batched, 0, 5, &first, 3);
        assert_token_logits_match(&batched, 1, 3, &second, 1);
        assert_token_logits_match(&batched, 1, 5, &second, 2);
        assert_token_logits_match(&chunk, 0, 1, &first, 2);
        assert_token_logits_match(&chunk, 0, 2, &first, 3);
        assert_token_logits_match(&chunk, 1, 0, &second, 1);
        assert_token_logits_match(&chunk, 1, 2, &second, 2);
        for (row, column) in [(0, 0), (1, 1)] {
            assert!(
                chunk
                    .clone()
                    .slice([row..row + 1, column..column + 1, 0..32])
                    .into_data()
                    .as_slice::<f32>()
                    .unwrap()
                    .iter()
                    .all(|&value| value == 0.0)
            );
        }

        state.reorder(&[1, 0]).unwrap();
        assert_eq!(state.lengths(), &[3, 4]);
        state.truncate(&[1, 2]).unwrap();
        assert_eq!(state.lengths(), &[1, 2]);
        let last = ForwardOptions::default();
        let after_truncate = model
            .decode(
                DecodeInput {
                    token_ids: Tensor::from_ints([[8, 0], [0, 9]], &device),
                    token_mask: Some(mask([[true, false], [false, true]], &device)),
                    position_ids: Some(Tensor::from_ints([[20, 0], [0, 21]], &device)),
                },
                &mut state,
                &last,
            )
            .unwrap()
            .logits;
        assert_eq!(state.lengths(), &[2, 3]);
        for (row, ids, positions) in [
            (0, vec![3, 8], vec![7, 20]),
            (1, vec![1, 2, 9], vec![2, 5, 21]),
        ] {
            let fresh = model
                .forward(parity_input(&ids, None, Some(&positions), &device), &last)
                .unwrap()
                .logits;
            assert_token_logits_match(&after_truncate, row, 0, &fresh, 0);
        }

        state.reorder(&[0, 0]).unwrap();
        assert_eq!(state.lengths(), &[2, 2]);
        let after_duplicate = model
            .decode(
                DecodeInput {
                    token_ids: Tensor::from_ints([[10], [11]], &device),
                    token_mask: None,
                    position_ids: Some(Tensor::from_ints([[25], [26]], &device)),
                },
                &mut state,
                &last,
            )
            .unwrap()
            .logits;
        assert_eq!(state.lengths(), &[3, 3]);
        for (row, token, position) in [(0, 10, 25), (1, 11, 26)] {
            let fresh = model
                .forward(
                    parity_input(&[3, 8, token], None, Some(&[7, 20, position]), &device),
                    &last,
                )
                .unwrap()
                .logits;
            assert_token_logits_match(&after_duplicate, row, 0, &fresh, 0);
        }
    }
}

/// Full attention preserves encoder and decoder valid-token outputs under independent padding.
#[test]
fn seq2seq_full_attention_padding_preserves_encoded_and_decoded_outputs() {
    let device = Default::default();
    let model = ModularSeq2SeqLm::<TestBackend>::init(seq2seq_spec(), &device)
        .unwrap()
        .map(&mut DeterministicParameters::default());
    let options = ForwardOptions::default();
    let source_ids = [4, 5, 6];
    let target_ids = [1, 2, 3];
    let compact_encoder = model
        .encode(parity_input(&source_ids, None, None, &device), &options)
        .unwrap();
    let compact_logits = model
        .forward(
            Seq2SeqInput {
                encoder: parity_input(&source_ids, None, None, &device),
                decoder: parity_input(&target_ids, None, None, &device),
            },
            &options,
        )
        .unwrap()
        .decoder
        .logits
        .into_data();
    let source_layouts = [
        (vec![0, 4, 5, 6], vec![false, true, true, true]),
        (vec![4, 5, 6, 0], vec![true, true, true, false]),
        (vec![4, 0, 5, 0, 6], vec![true, false, true, false, true]),
    ];
    let target_layouts = [
        (vec![0, 1, 2, 3], vec![false, true, true, true]),
        (vec![1, 2, 3, 0], vec![true, true, true, false]),
        (vec![1, 0, 2, 0, 3], vec![true, false, true, false, true]),
    ];
    for (source_ids, source_validity) in &source_layouts {
        let padded_encoder = model
            .encode(
                parity_input(source_ids, Some(source_validity), None, &device),
                &options,
            )
            .unwrap();
        for (compact_column, padded_column) in source_validity
            .iter()
            .enumerate()
            .filter_map(|(column, &valid)| valid.then_some(column))
            .enumerate()
        {
            let padded = padded_encoder
                .hidden_state
                .clone()
                .slice([0..1, padded_column..padded_column + 1, 0..8])
                .into_data();
            let compact = compact_encoder
                .hidden_state
                .clone()
                .slice([0..1, compact_column..compact_column + 1, 0..8])
                .into_data();
            assert_finite_attention_parity(&padded, &compact);
        }
        for (target_ids, target_validity) in &target_layouts {
            let decoder = || parity_input(target_ids, Some(target_validity), None, &device);
            let padded = model
                .forward(
                    Seq2SeqInput {
                        encoder: parity_input(source_ids, Some(source_validity), None, &device),
                        decoder: decoder(),
                    },
                    &options,
                )
                .unwrap()
                .decoder
                .logits
                .into_data();
            assert_finite_attention_parity(&padded, &compact_logits);
            let mut state = model
                .new_decode_state(CacheConfig::new(1, 8).unwrap(), 1, &device)
                .unwrap();
            let cached = model
                .prefill_decoder(&padded_encoder, decoder(), &mut state, &options)
                .unwrap()
                .decoder
                .logits
                .into_data();
            assert_finite_attention_parity(&cached, &compact_logits);
            assert_eq!(state.lengths(), &[3]);
            let decoded = model
                .decode(
                    &padded_encoder,
                    DecodeInput {
                        token_ids: Tensor::from_ints([[0, 7]], &device),
                        token_mask: Some(mask([[false, true]], &device)),
                        position_ids: None,
                    },
                    &mut state,
                    &options,
                )
                .unwrap()
                .decoder
                .logits;
            assert_eq!(state.lengths(), &[4]);
            let fresh = model
                .forward(
                    Seq2SeqInput {
                        encoder: parity_input(&[4, 5, 6], None, None, &device),
                        decoder: parity_input(&[1, 2, 3, 7], None, None, &device),
                    },
                    &options,
                )
                .unwrap()
                .decoder
                .logits;
            assert_finite_attention_parity(&decoded.into_data(), &fresh.into_data());
        }
    }
}

/// Heterogeneous attention and recurrent layers execute as one decoder graph.
#[test]
fn causal_heterogeneous_forward_returns_requested_diagnostics() {
    let device = Default::default();
    let model = ModularCausalLm::<TestBackend>::init(causal_spec(), &device).unwrap();
    let output = model
        .forward(
            CausalLmInput {
                token_ids: Tensor::from_ints([[1, 2, 3], [4, 5, 0]], &device),
                token_mask: Some(mask([[true, true, true], [true, true, false]], &device)),
                position_ids: None,
            },
            &ForwardOptions {
                logits: LogitsSelection::All,
                hidden_states: true,
                ..Default::default()
            },
        )
        .expect("heterogeneous forward");
    assert_eq!(output.logits.dims(), [2, 3, 32]);
    assert_eq!(output.hidden_states.len(), 4);
    assert_eq!(output.final_hidden_state.unwrap().dims(), [2, 3, 8]);
    assert!(output.attentions.is_empty());
}

/// Sparse custom layers expose raw router scores and a differentiable balancing loss.
#[test]
fn causal_moe_exposes_opt_in_router_diagnostics() {
    let device = Default::default();
    let mut moe_layer = layer(attention(PositionEncodingSpec::None));
    moe_layer.feed_forward = FeedForwardSpec::MixtureOfExperts(MixtureOfExpertsSpec {
        routed_experts: 4,
        experts_per_token: 2,
        shared_experts: 1,
        expert_intermediate_size: 6,
        activation: Activation::ReluSquared,
        renormalize_selected: true,
        routed_scale: 0.75,
        routing: RoutingPolicy::default(),
        shared_expert_gate: false,
        expert_bias: false,
    });
    let model = ModularCausalLm::<TestBackend>::init(decoder(vec![moe_layer]), &device).unwrap();
    let output = model
        .forward(
            CausalLmInput {
                token_ids: Tensor::from_ints([[1, 2]], &device),
                token_mask: None,
                position_ids: None,
            },
            &ForwardOptions {
                router_logits: true,
                auxiliary_losses: true,
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(output.routers.len(), 1);
    assert_eq!(output.routers[0].logits.as_ref().unwrap().dims(), [1, 2, 4]);
    assert_eq!(
        output.routers[0]
            .load_balancing_loss
            .as_ref()
            .unwrap()
            .dims(),
        [1]
    );
}

/// Sliding attention, a sigmoid output gate, LayerNorm, and sandwich placement compose.
#[test]
fn sliding_sandwich_layernorm_graph_is_executable() {
    let device = Default::default();
    let attention = AttentionSpec {
        query_heads: 2,
        key_value_heads: 2,
        head_dim: 4,
        query_key_norm: false,
        output_gate: true,
        position: PositionEncodingSpec::None,
    };
    let mut block = layer(MixerSpec::SlidingAttention(SlidingWindowSpec {
        attention: attention.clone(),
        window: 2,
    }));
    block.norm_placement = NormPlacement::Sandwich;
    block.mixer_norm = NormalizationSpec::Layer {
        epsilon: 1e-5,
        bias: true,
    };
    block.feed_forward_norm = NormalizationSpec::Layer {
        epsilon: 1e-5,
        bias: false,
    };
    let model = ModularCausalLm::<TestBackend>::init(decoder(vec![block]), &device).unwrap();
    let data = model
        .forward(
            CausalLmInput {
                token_ids: Tensor::from_ints([[1, 2, 3, 4]], &device),
                token_mask: None,
                position_ids: None,
            },
            &ForwardOptions::default(),
        )
        .unwrap()
        .logits
        .into_data();
    assert!(
        data.as_slice::<f32>()
            .unwrap()
            .iter()
            .all(|x| x.is_finite())
    );
}

/// T5 relative buckets select unscaled learned-bias attention in a custom decoder.
#[test]
fn t5_relative_bias_is_executable_outside_t5_family_wiring() {
    let device = Default::default();
    let spec = decoder(vec![layer(attention(
        PositionEncodingSpec::T5RelativeBias {
            buckets: 8,
            max_distance: 32,
            bidirectional: false,
        },
    ))]);
    let model = ModularCausalLm::<TestBackend>::init(spec, &device).unwrap();
    assert_eq!(
        model
            .forward(
                CausalLmInput {
                    token_ids: Tensor::from_ints([[1, 2, 3]], &device),
                    token_mask: None,
                    position_ids: None,
                },
                &ForwardOptions::default(),
            )
            .unwrap()
            .logits
            .dims(),
        [1, 1, 32]
    );
}

/// Explicit, gapped T5 positions select buckets from actual displacement rather than indices.
#[test]
fn t5_relative_bias_honors_explicit_position_gaps() {
    let device = Default::default();
    let mut bias = T5RelativePositionBiasConfig::new(1, 8, 32, false)
        .init::<TestBackend>(&device)
        .unwrap();
    bias.weight = Param::from_tensor(
        Tensor::<TestBackend, 1, Int>::arange(0..8, &device)
            .float()
            .reshape([8, 1]),
    );

    let pass = StackPass::new(
        Tensor::from_ints([[5, 7, 10]], &device),
        None,
        11,
        Stack::Decoder,
        burn::tensor::DType::F32,
    );
    let actual = pass.relative_bias(&bias).unwrap();
    actual.into_data().assert_approx_eq::<f32>(
        &Tensor::<TestBackend, 4>::from_floats(
            [[[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 3.0, 0.0]]]],
            &device,
        )
        .into_data(),
        Tolerance::absolute(1e-6),
    );
}

/// YaRN rotary scaling and conventional post-normalization remain independently selectable.
#[test]
fn yarn_and_post_norm_graph_is_executable() {
    let device = Default::default();
    let mut block = layer(attention(PositionEncodingSpec::Rope {
        theta: 10_000.0,
        rotary_dim: 4,
        scaling: RopeScaling::Yarn {
            factor: 2.0,
            original_max_positions: 16,
            attention_factor: Some(1.1),
            beta_slow: 1.0,
            beta_fast: 32.0,
            truncate: true,
        },
    }));
    block.norm_placement = NormPlacement::Post;
    let model = ModularCausalLm::<TestBackend>::init(decoder(vec![block]), &device).unwrap();
    let output = model
        .forward(
            CausalLmInput {
                token_ids: Tensor::from_ints([[1, 2]], &device),
                token_mask: None,
                position_ids: Some(Tensor::from_ints([[16, 17]], &device)),
            },
            &ForwardOptions::default(),
        )
        .unwrap();
    assert_eq!(output.logits.dims(), [1, 1, 32]);
}

/// Tied output projection owns no duplicate linear parameter matrix.
#[test]
fn tied_causal_head_reuses_embedding_weight() {
    let device = Default::default();
    let mut spec = decoder(vec![layer(attention(PositionEncodingSpec::None))]);
    spec.tie_word_embeddings = true;
    let model = ModularCausalLm::<TestBackend>::init(spec, &device).unwrap();
    assert!(model.lm_head.is_none());
    assert_eq!(
        model
            .forward(
                CausalLmInput {
                    token_ids: Tensor::from_ints([[1, 2]], &device),
                    token_mask: None,
                    position_ids: None,
                },
                &ForwardOptions::default(),
            )
            .unwrap()
            .logits
            .dims(),
        [1, 1, 32]
    );
}

/// Serialized heterogeneous topology is lossless across JSON round trips.
#[test]
fn modular_spec_round_trips_through_json() {
    let spec = causal_spec();
    let json = serde_json::to_string_pretty(&spec).unwrap();
    let decoded: DecoderSpec = serde_json::from_str(&json).unwrap();
    assert_eq!(decoded, spec);
}

/// Invalid MLA rotary dimensions, unsupported Mamba2, and undefined residual gates fail explicitly.
#[test]
fn unavailable_or_ambiguous_components_are_rejected() {
    let device = Default::default();
    let mla = MixerSpec::Mla(MlaSpec {
        query_heads: 2,
        content_head_dim: 2,
        rotary_head_dim: 2,
        value_head_dim: 4,
        key_value_lora_rank: 4,
        query_lora_rank: None,
        position: PositionEncodingSpec::Rope {
            theta: 10_000.0,
            rotary_dim: 4,
            scaling: RopeScaling::None,
        },
        score_scale: ScoreScale::Auto,
        connectivity: AttentionConnectivity::Global,
    });
    let error = ModularCausalLm::<TestBackend>::init(decoder(vec![layer(mla)]), &device)
        .expect_err("MLA rotary dimension exceeds its rotary channel width");
    assert!(error.to_string().contains("rotary"));

    let mamba = MixerSpec::Mamba2(Mamba2Spec {
        heads: 2,
        head_dim: 4,
        state_dim: 8,
        convolution_kernel: 4,
        group_size: 1,
    });
    let error = ModularCausalLm::<TestBackend>::init(decoder(vec![layer(mamba)]), &device)
        .expect_err("Mamba2 kernel is not executable yet");
    assert!(error.to_string().contains("requests Mamba2"));

    let mut gated = layer(attention(PositionEncodingSpec::None));
    gated.residual = ResidualTopology::Gated;
    let error = ModularCausalLm::<TestBackend>::init(decoder(vec![gated]), &device)
        .expect_err("gate semantics are underspecified");
    assert!(error.to_string().contains("does not define its activation"));
}

/// Stateless and prefill-plus-decode paths agree exactly at the appended token.
#[test]
fn causal_reference_decode_matches_stateless_forward() {
    let device = Default::default();
    let model = ModularCausalLm::<TestBackend>::init(causal_spec(), &device).unwrap();
    let full = model
        .forward(
            CausalLmInput {
                token_ids: Tensor::from_ints([[1, 2, 3, 4]], &device),
                token_mask: None,
                position_ids: None,
            },
            &ForwardOptions::default(),
        )
        .unwrap()
        .logits;
    let mut state = model
        .new_decode_state(CacheConfig::new(1, 8).unwrap(), 1, &device)
        .unwrap();
    assert_eq!(state.allocated_integer_elements(), 16);
    model
        .prefill(
            CausalLmInput {
                token_ids: Tensor::from_ints([[1, 2, 3]], &device),
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .unwrap();
    let cached = model
        .decode(
            DecodeInput {
                token_ids: Tensor::from_ints([[4]], &device),
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .unwrap()
        .logits;
    cached
        .into_data()
        .assert_approx_eq::<f32>(&full.into_data(), Tolerance::absolute(2e-4));
    assert_eq!(state.lengths(), &[4]);
}

/// Explicit absolute positions and selective logits retain their meaning in cached execution.
#[test]
fn cached_explicit_positions_and_selection_match_stateless() {
    let device = Default::default();
    let model = ModularCausalLm::<TestBackend>::init(
        decoder(vec![layer(attention(PositionEncodingSpec::Rope {
            theta: 10_000.0,
            rotary_dim: 4,
            scaling: RopeScaling::None,
        }))]),
        &device,
    )
    .unwrap();
    let full = model
        .forward(
            CausalLmInput {
                token_ids: Tensor::from_ints([[1, 2, 3, 4]], &device),
                token_mask: None,
                position_ids: Some(Tensor::from_ints([[7, 9, 12, 13]], &device)),
            },
            &ForwardOptions {
                logits: LogitsSelection::Positions(vec![2, 3]),
                ..Default::default()
            },
        )
        .unwrap()
        .logits;
    let mut state = model
        .new_decode_state(CacheConfig::new(1, 8).unwrap(), 1, &device)
        .unwrap();
    model
        .prefill(
            CausalLmInput {
                token_ids: Tensor::from_ints([[1, 2]], &device),
                token_mask: None,
                position_ids: Some(Tensor::from_ints([[7, 9]], &device)),
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .unwrap();
    let cached = model
        .decode(
            DecodeInput {
                token_ids: Tensor::from_ints([[3, 4]], &device),
                token_mask: None,
                position_ids: Some(Tensor::from_ints([[12, 13]], &device)),
            },
            &mut state,
            &ForwardOptions {
                logits: LogitsSelection::All,
                ..Default::default()
            },
        )
        .unwrap()
        .logits;
    cached
        .into_data()
        .assert_approx_eq::<f32>(&full.into_data(), Tolerance::absolute(2e-4));
}

/// Ragged cached steps compact valid tokens, zero inactive outputs, and preserve row lengths.
#[test]
fn causal_decode_compacts_ragged_steps() {
    let device = Default::default();
    let model = ModularCausalLm::<TestBackend>::init(
        decoder(vec![layer(attention(PositionEncodingSpec::None))]),
        &device,
    )
    .unwrap();
    let mut state = model
        .new_decode_state(CacheConfig::new(2, 8).unwrap(), 2, &device)
        .unwrap();
    model
        .prefill(
            CausalLmInput {
                token_ids: Tensor::from_ints([[1, 2], [3, 4]], &device),
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .unwrap();
    let output = model
        .decode(
            DecodeInput {
                token_ids: Tensor::from_ints([[5, 6], [0, 0]], &device),
                token_mask: Some(mask([[true, true], [false, false]], &device)),
                position_ids: None,
            },
            &mut state,
            &ForwardOptions {
                logits: LogitsSelection::All,
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(state.lengths(), &[4, 2]);
    assert_eq!(output.logits.dims(), [2, 2, 32]);
    let inactive = output.logits.slice([1..2, 0..2, 0..32]).into_data();
    assert!(
        inactive
            .as_slice::<f32>()
            .unwrap()
            .iter()
            .all(|&x| x == 0.0)
    );
}

/// Reset, duplicate reorder, and truncation update external history lifecycle consistently.
#[test]
fn causal_state_lifecycle_is_checked() {
    let device = Default::default();
    let model = ModularCausalLm::<TestBackend>::init(
        decoder(vec![layer(attention(PositionEncodingSpec::None))]),
        &device,
    )
    .unwrap();
    let mut state = model
        .new_decode_state(CacheConfig::new(2, 6).unwrap(), 2, &device)
        .unwrap();
    model
        .prefill(
            CausalLmInput {
                token_ids: Tensor::from_ints([[1, 2], [3, 4]], &device),
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .unwrap();
    state.reorder(&[1, 1]).unwrap();
    assert_eq!(state.lengths(), &[2, 2]);
    state.truncate(&[1, 0]).unwrap();
    assert_eq!(state.lengths(), &[1, 0]);
    state.reset().unwrap();
    assert_eq!(state.lengths(), &[0, 0]);
    let error = state
        .truncate(&[1, 0])
        .expect_err("truncation cannot extend state");
    assert!(error.to_string().contains("cannot truncate"));
}

/// A failed over-capacity append leaves committed lengths unchanged.
#[test]
fn decode_capacity_failure_is_atomic() {
    let device = Default::default();
    let model = ModularCausalLm::<TestBackend>::init(
        decoder(vec![layer(attention(PositionEncodingSpec::None))]),
        &device,
    )
    .unwrap();
    let mut state = model
        .new_decode_state(CacheConfig::new(1, 2).unwrap(), 1, &device)
        .unwrap();
    model
        .prefill(
            CausalLmInput {
                token_ids: Tensor::from_ints([[1, 2]], &device),
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .unwrap();
    let error = model
        .decode(
            DecodeInput {
                token_ids: Tensor::from_ints([[3]], &device),
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .expect_err("capacity must be checked before mutation");
    assert!(error.to_string().contains("exceeds capacity"));
    assert_eq!(state.lengths(), &[2]);
}

/// State construction rejects disagreement between cache sequence count and requested batch.
#[test]
fn decode_state_rejects_batch_configuration_mismatch() {
    let device = Default::default();
    let model = ModularCausalLm::<TestBackend>::init(
        decoder(vec![layer(attention(PositionEncodingSpec::None))]),
        &device,
    )
    .unwrap();
    let error = model
        .new_decode_state(CacheConfig::new(2, 8).unwrap(), 1, &device)
        .expect_err("batch and cache geometry must agree");
    assert!(error.to_string().contains("expected 2 sequences"));
}

/// State topology identity prevents passing a valid state to a different custom graph.
#[test]
fn decode_rejects_state_from_another_topology() {
    let device = Default::default();
    let first = ModularCausalLm::<TestBackend>::init(
        decoder(vec![layer(attention(PositionEncodingSpec::None))]),
        &device,
    )
    .unwrap();
    let second = ModularCausalLm::<TestBackend>::init(causal_spec(), &device).unwrap();
    let mut state = first
        .new_decode_state(CacheConfig::new(1, 8).unwrap(), 1, &device)
        .unwrap();
    let error = second
        .prefill(
            CausalLmInput {
                token_ids: Tensor::from_ints([[1]], &device),
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .expect_err("topologies must not share state");
    assert!(error.to_string().contains("different model topology"));
}

/// Public input validation reports token, mask, position, and diagnostic errors without panics.
#[test]
fn forward_validation_is_contextual() {
    let device = Default::default();
    let model = ModularCausalLm::<TestBackend>::init(
        decoder(vec![layer(attention(PositionEncodingSpec::None))]),
        &device,
    )
    .unwrap();
    let error = model
        .forward(
            CausalLmInput {
                token_ids: Tensor::from_ints([[32]], &device),
                token_mask: None,
                position_ids: None,
            },
            &ForwardOptions::default(),
        )
        .expect_err("out-of-range token");
    assert!(error.to_string().contains("values in 0..32"));

    let error = model
        .forward(
            CausalLmInput {
                token_ids: Tensor::from_ints([[1, 2]], &device),
                token_mask: Some(mask([[false, false]], &device)),
                position_ids: None,
            },
            &ForwardOptions::default(),
        )
        .expect_err("fully masked rows are invalid");
    assert!(error.to_string().contains("fully masked row"));

    let error = model
        .forward(
            CausalLmInput {
                token_ids: Tensor::from_ints([[1]], &device),
                token_mask: None,
                position_ids: Some(Tensor::from_ints([[32]], &device)),
            },
            &ForwardOptions::default(),
        )
        .expect_err("out-of-range position");
    assert!(error.to_string().contains("position_ids"));

    let error = model
        .forward(
            CausalLmInput {
                token_ids: Tensor::from_ints([[1]], &device),
                token_mask: None,
                position_ids: None,
            },
            &ForwardOptions {
                attentions: true,
                ..Default::default()
            },
        )
        .expect_err("attention capture is unavailable");
    assert!(error.to_string().contains("probability capture"));
}

/// Bidirectional encoding, causal decoding, and cross attention produce the public shapes.
#[test]
fn seq2seq_forward_executes_both_stacks() {
    let device = Default::default();
    let model = ModularSeq2SeqLm::<TestBackend>::init(seq2seq_spec(), &device).unwrap();
    let output = model
        .forward(
            Seq2SeqInput {
                encoder: CausalLmInput {
                    token_ids: Tensor::from_ints([[4, 5, 6]], &device),
                    token_mask: None,
                    position_ids: None,
                },
                decoder: CausalLmInput {
                    token_ids: Tensor::from_ints([[0, 1]], &device),
                    token_mask: None,
                    position_ids: None,
                },
            },
            &ForwardOptions {
                logits: LogitsSelection::All,
                hidden_states: true,
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(output.encoder.hidden_state.dims(), [1, 3, 8]);
    assert_eq!(output.encoder.hidden_states.len(), 3);
    assert_eq!(output.decoder.logits.dims(), [1, 2, 32]);
    assert_eq!(output.decoder.hidden_states.len(), 3);
    assert!(output.cross_attentions.is_empty());
}

/// Seq2seq prefill and reference decode agree with a complete target forward.
#[test]
fn seq2seq_cached_decoder_matches_stateless_forward() {
    let device = Default::default();
    let model = ModularSeq2SeqLm::<TestBackend>::init(seq2seq_spec(), &device).unwrap();
    let source = CausalLmInput {
        token_ids: Tensor::from_ints([[4, 5, 6]], &device),
        token_mask: None,
        position_ids: None,
    };
    let encoder = model
        .encode(source.clone(), &ForwardOptions::default())
        .unwrap();
    let full = model
        .forward(
            Seq2SeqInput {
                encoder: source,
                decoder: CausalLmInput {
                    token_ids: Tensor::from_ints([[0, 1, 2]], &device),
                    token_mask: None,
                    position_ids: None,
                },
            },
            &ForwardOptions::default(),
        )
        .unwrap()
        .decoder
        .logits;
    let mut state = model
        .new_decode_state(CacheConfig::new(1, 8).unwrap(), 1, &device)
        .unwrap();
    model
        .prefill_decoder(
            &encoder,
            CausalLmInput {
                token_ids: Tensor::from_ints([[0, 1]], &device),
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .unwrap();
    let cached = model
        .decode(
            &encoder,
            DecodeInput {
                token_ids: Tensor::from_ints([[2]], &device),
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .unwrap()
        .decoder
        .logits;
    cached
        .into_data()
        .assert_approx_eq::<f32>(&full.into_data(), Tolerance::absolute(2e-4));
}

/// Causal recurrent and trailing-window mixers cannot be placed in a bidirectional encoder.
#[test]
fn seq2seq_rejects_directionally_invalid_encoder_mixers() {
    let device = Default::default();
    let mut spec = seq2seq_spec();
    spec.encoder.layers[0].mixer = MixerSpec::DeltaNet {
        key_heads: 1,
        value_heads: 2,
        key_head_dim: 4,
        value_head_dim: 4,
        convolution_kernel: 2,
    };
    let error = ModularSeq2SeqLm::<TestBackend>::init(spec, &device)
        .expect_err("causal recurrence is invalid in encoder");
    assert!(error.to_string().contains("bidirectional encoder"));

    let mut spec = seq2seq_spec();
    let attention = match attention(PositionEncodingSpec::None) {
        MixerSpec::Attention(attention) => attention,
        _ => unreachable!(),
    };
    spec.encoder.layers[0].mixer = MixerSpec::SlidingAttention(SlidingWindowSpec {
        attention,
        window: 2,
    });
    let error = ModularSeq2SeqLm::<TestBackend>::init(spec, &device)
        .expect_err("trailing window is invalid in encoder");
    assert!(error.to_string().contains("symmetric bidirectional"));

    // The explicit forms of a past-only window are rejected the same way (CUS-6).
    let sliding = crate::architecture::AttentionOptions {
        connectivity: AttentionConnectivity::SlidingWindow { window: 2 },
        ..Default::default()
    };
    let mla = MlaSpec {
        query_heads: 2,
        content_head_dim: 4,
        rotary_head_dim: 2,
        value_head_dim: 4,
        key_value_lora_rank: 4,
        query_lora_rank: None,
        position: PositionEncodingSpec::None,
        score_scale: ScoreScale::Auto,
        connectivity: AttentionConnectivity::SlidingWindow { window: 2 },
    };
    for mixer in [
        MixerSpec::AttentionWithOptions {
            attention: attention_spec(),
            options: sliding,
        },
        MixerSpec::Mla(mla.clone()),
    ] {
        let mut spec = seq2seq_spec();
        spec.encoder.layers[0].mixer = mixer;
        let error = ModularSeq2SeqLm::<TestBackend>::init(spec, &device)
            .expect_err("trailing window is invalid in encoder");
        assert!(error.to_string().contains("past-only"), "{error}");
    }
    // Symmetric local chunks are valid encoder connectivity for both attention kinds.
    for mixer in [
        MixerSpec::AttentionWithOptions {
            attention: attention_spec(),
            options: crate::architecture::AttentionOptions {
                connectivity: AttentionConnectivity::ChunkedLocal { chunk: 2 },
                ..Default::default()
            },
        },
        MixerSpec::Mla(MlaSpec {
            connectivity: AttentionConnectivity::ChunkedLocal { chunk: 2 },
            ..mla.clone()
        }),
    ] {
        let mut spec = seq2seq_spec();
        spec.encoder.layers[0].mixer = mixer;
        ModularSeq2SeqLm::<TestBackend>::init(spec, &device).unwrap();
    }
}

/// The grouped-query topology of the shared [`attention`] fixture.
fn attention_spec() -> AttentionSpec {
    match attention(PositionEncodingSpec::None) {
        MixerSpec::Attention(attention) => attention,
        _ => unreachable!("the fixture is compact attention"),
    }
}

/// A last-token prediction depends only on the keys its connectivity allows.
///
/// For a one-layer causal graph, the last query's output is a function of the
/// visible raw embeddings alone. Evaluating exactly those tokens (with the same
/// explicit positions) is therefore an independent control for sliding MLA
/// windows and chunked-local grouped-query attention; padding consumes neither.
#[test]
fn local_connectivity_limits_the_last_token_to_its_visible_keys() {
    let device = Default::default();
    let ids = [1, 2, 3, 4, 5];
    let positions = [3, 4, 5, 6, 7];
    let rope = PositionEncodingSpec::Rope {
        theta: 10_000.0,
        rotary_dim: 4,
        scaling: RopeScaling::None,
    };
    let sliding_mla = MixerSpec::Mla(MlaSpec {
        query_heads: 2,
        content_head_dim: 4,
        rotary_head_dim: 4,
        value_head_dim: 4,
        key_value_lora_rank: 4,
        query_lora_rank: None,
        position: rope.clone(),
        score_scale: ScoreScale::Auto,
        connectivity: AttentionConnectivity::SlidingWindow { window: 2 },
    });
    let chunked = MixerSpec::AttentionWithOptions {
        attention: AttentionSpec {
            position: rope,
            ..attention_spec()
        },
        options: crate::architecture::AttentionOptions {
            connectivity: AttentionConnectivity::ChunkedLocal { chunk: 3 },
            ..Default::default()
        },
    };
    // Sliding window 2 sees the last two tokens; chunks of 3 put tokens 3..5 together.
    for (mixer, visible) in [(sliding_mla, 3..5), (chunked, 3..5)] {
        let model = ModularCausalLm::<TestBackend>::init(decoder(vec![layer(mixer)]), &device)
            .unwrap()
            .map(&mut DeterministicParameters::default());
        let options = ForwardOptions::default();
        let control = model
            .forward(
                parity_input(
                    &ids[visible.clone()],
                    None,
                    Some(&positions[visible]),
                    &device,
                ),
                &options,
            )
            .unwrap()
            .logits
            .into_data();
        let full = model
            .forward(
                parity_input(&ids, None, Some(&positions), &device),
                &options,
            )
            .unwrap()
            .logits
            .into_data();
        assert_finite_attention_parity(&full, &control);
        let padded = model
            .forward(
                parity_input(
                    &[0, 1, 2, 0, 3, 4, 0, 5],
                    Some(&[false, true, true, false, true, true, false, true]),
                    Some(&[0, 3, 4, 0, 5, 6, 0, 7]),
                    &device,
                ),
                &options,
            )
            .unwrap()
            .logits
            .into_data();
        assert_finite_attention_parity(&padded, &control);
    }
}

/// Encoder MoE math is allowed, but unavailable encoder router output is rejected explicitly.
#[test]
fn seq2seq_encoder_router_diagnostics_are_not_silently_dropped() {
    let device = Default::default();
    let mut spec = seq2seq_spec();
    spec.encoder.layers[0].feed_forward = FeedForwardSpec::MixtureOfExperts(MixtureOfExpertsSpec {
        routed_experts: 2,
        experts_per_token: 1,
        shared_experts: 0,
        expert_intermediate_size: 4,
        activation: Activation::Silu,
        renormalize_selected: true,
        routed_scale: 1.0,
        routing: RoutingPolicy::default(),
        shared_expert_gate: false,
        expert_bias: false,
    });
    let model = ModularSeq2SeqLm::<TestBackend>::init(spec, &device).unwrap();
    let error = model
        .encode(
            CausalLmInput {
                token_ids: Tensor::from_ints([[1, 2]], &device),
                token_mask: None,
                position_ids: None,
            },
            &ForwardOptions {
                router_logits: true,
                ..Default::default()
            },
        )
        .expect_err("EncoderOutput cannot carry router diagnostics");
    assert!(error.to_string().contains("no router-diagnostic field"));
}

/// Causal stateless execution retains a gradient path to embeddings and the LM head.
#[cfg(feature = "autodiff")]
#[test]
fn causal_stateless_graph_is_autodiff_safe() {
    use burn::backend::Autodiff;

    /// Reverse-mode CPU backend used to check the stateless graph's embedding gradients.
    type AdBackend = Autodiff<TestBackend>;
    let device = Default::default();
    let model = ModularCausalLm::<AdBackend>::init(
        decoder(vec![layer(attention(PositionEncodingSpec::None))]),
        &device,
    )
    .unwrap();
    let loss = model
        .forward(
            CausalLmInput {
                token_ids: Tensor::<AdBackend, 2, Int>::from_ints([[1, 2]], &device),
                token_mask: None,
                position_ids: None,
            },
            &ForwardOptions::default(),
        )
        .unwrap()
        .logits
        .sum();
    let gradients = loss.backward();
    assert!(model.embed_tokens.weight.grad(&gradients).is_some());
}

/// Encoder-decoder stateless execution retains gradients through both embedding tables.
#[cfg(feature = "autodiff")]
#[test]
fn seq2seq_stateless_graph_is_autodiff_safe() {
    use burn::backend::Autodiff;

    /// Reverse-mode CPU backend used to check the stateless graph's embedding gradients.
    type AdBackend = Autodiff<TestBackend>;
    let device = Default::default();
    let model = ModularSeq2SeqLm::<AdBackend>::init(seq2seq_spec(), &device).unwrap();
    let loss = model
        .forward(
            Seq2SeqInput {
                encoder: CausalLmInput {
                    token_ids: Tensor::<AdBackend, 2, Int>::from_ints([[4, 5]], &device),
                    token_mask: None,
                    position_ids: None,
                },
                decoder: CausalLmInput {
                    token_ids: Tensor::<AdBackend, 2, Int>::from_ints([[0, 1]], &device),
                    token_mask: None,
                    position_ids: None,
                },
            },
            &ForwardOptions::default(),
        )
        .unwrap()
        .decoder
        .logits
        .sum();
    let gradients = loss.backward();
    assert!(model.encoder_embed_tokens.weight.grad(&gradients).is_some());
    assert!(model.decoder_embed_tokens.weight.grad(&gradients).is_some());
}

/// Records every float parameter of a module that received no gradient.
#[cfg(feature = "autodiff")]
struct GradientAudit<'a, B: burn::tensor::backend::AutodiffBackend> {
    /// Gradients of one backward pass.
    gradients: &'a B::Gradients,
    /// Module path of the parameter being visited.
    path: Vec<String>,
    /// Paths of parameters without a gradient.
    missing: Vec<String>,
    /// Number of float parameters visited.
    visited: usize,
}

#[cfg(feature = "autodiff")]
impl<B: burn::tensor::backend::AutodiffBackend> burn::module::ModuleVisitor<B>
    for GradientAudit<'_, B>
{
    /// Descends into a named submodule.
    fn enter_module(&mut self, name: &str, _container_type: &str) {
        self.path.push(name.to_owned());
    }

    /// Leaves the current submodule.
    fn exit_module(&mut self, _name: &str, _container_type: &str) {
        self.path.pop();
    }

    /// Records whether this parameter's tensor has a gradient.
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        self.visited += 1;
        if param.val().grad(self.gradients).is_none() {
            self.missing.push(self.path.join("."));
        }
    }
}

/// Every parameter of a graph using each new mechanism is a trainable autodiff leaf.
///
/// Covers option-rich attention (Llama 3 RoPE, sinks, per-projection biases,
/// full-width Q/K norm, SiLU gate, chunked-local connectivity), learned T5
/// bias through the shared pass table, sliding MLA, a gated shared expert,
/// clamped and ungated MLPs, an attention-only block, and differential
/// attention, over padded input.
/// Routed experts that no token selects legitimately receive no gradient.
#[cfg(feature = "autodiff")]
#[test]
fn every_mechanism_keeps_parameters_trainable() {
    use burn::backend::Autodiff;

    use crate::{
        architecture::{
            AttentionOptions, ClampedSwiGluSpec, DenseFeedForwardSpec, OutputGateActivation,
        },
        attention::{ProjectionBias, QueryKeyNormScope},
    };

    /// Reverse-mode CPU backend.
    type AdBackend = Autodiff<TestBackend>;
    let device = Default::default();
    let options = AttentionOptions {
        connectivity: AttentionConnectivity::ChunkedLocal { chunk: 2 },
        attention_sinks: true,
        projection_bias: ProjectionBias::query_key_value_only(),
        query_key_norm_scope: QueryKeyNormScope::FullWidth,
        output_gate_activation: OutputGateActivation::Silu,
        ..AttentionOptions::default()
    };
    let mut rich = layer(MixerSpec::AttentionWithOptions {
        attention: AttentionSpec {
            query_heads: 2,
            key_value_heads: 1,
            head_dim: 4,
            query_key_norm: true,
            output_gate: true,
            position: PositionEncodingSpec::Rope {
                theta: 10_000.0,
                rotary_dim: 4,
                scaling: RopeScaling::Llama3 {
                    factor: 8.0,
                    low_freq_factor: 1.0,
                    high_freq_factor: 4.0,
                    original_max_positions: 4,
                },
            },
        },
        options,
    });
    rich.feed_forward = FeedForwardSpec::ClampedSwiGlu(ClampedSwiGluSpec {
        intermediate_size: 6,
        bias: true,
        limit: 7.0,
        alpha: 1.702,
        up_offset: 1.0,
    });
    let mut t5 = layer(attention(PositionEncodingSpec::T5RelativeBias {
        buckets: 8,
        max_distance: 16,
        bidirectional: false,
    }));
    t5.feed_forward = FeedForwardSpec::Ungated(DenseFeedForwardSpec {
        intermediate_size: 6,
        activation: Activation::GeluTanh,
        bias: true,
    });
    let mut mla = layer(MixerSpec::Mla(MlaSpec {
        query_heads: 2,
        content_head_dim: 4,
        rotary_head_dim: 2,
        value_head_dim: 4,
        key_value_lora_rank: 4,
        query_lora_rank: Some(4),
        position: PositionEncodingSpec::Rope {
            theta: 10_000.0,
            rotary_dim: 2,
            scaling: RopeScaling::None,
        },
        score_scale: ScoreScale::Unscaled,
        connectivity: AttentionConnectivity::SlidingWindow { window: 2 },
    }));
    mla.feed_forward = FeedForwardSpec::MixtureOfExperts(MixtureOfExpertsSpec {
        routed_experts: 2,
        experts_per_token: 2,
        shared_experts: 1,
        expert_intermediate_size: 4,
        activation: Activation::Silu,
        renormalize_selected: true,
        routed_scale: 1.0,
        routing: RoutingPolicy::default(),
        shared_expert_gate: true,
        expert_bias: false,
    });
    let mut attention_only = layer(attention(PositionEncodingSpec::None));
    attention_only.feed_forward = FeedForwardSpec::None;
    let model =
        ModularCausalLm::<AdBackend>::init(decoder(vec![rich, t5, mla, attention_only]), &device)
            .unwrap();
    let loss = model
        .forward(
            CausalLmInput {
                token_ids: Tensor::<AdBackend, 2, Int>::from_ints(
                    [[1, 2, 3, 4, 5], [0, 6, 7, 8, 0]],
                    &device,
                ),
                token_mask: Some(TokenMask::new(Tensor::from_bool(
                    [[true; 5], [false, true, true, true, false]].into(),
                    &device,
                ))),
                position_ids: None,
            },
            &ForwardOptions {
                logits: LogitsSelection::All,
                ..Default::default()
            },
        )
        .unwrap()
        .logits
        .powi_scalar(2)
        .mean();
    let gradients = loss.backward();
    let mut audit = GradientAudit::<AdBackend> {
        gradients: &gradients,
        path: Vec::new(),
        missing: Vec::new(),
        visited: 0,
    };
    model.visit(&mut audit);
    assert!(audit.visited > 40, "visited {}", audit.visited);
    assert!(
        audit.missing.is_empty(),
        "no gradient for {:?}",
        audit.missing
    );
}
