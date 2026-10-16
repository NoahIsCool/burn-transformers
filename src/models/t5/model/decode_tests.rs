//! Tests cached T5 decoding, state preservation, stateless parity, and gradient connectivity.

use burn::{
    backend::Flex,
    module::{Initializer, Module, ModuleMapper, Param},
    tensor::{Bool, DType, Int, Tensor, TensorData, Tolerance, backend::BackendTypes},
};

use super::super::{
    PARAMETERS, TargetColumns, ValidatedTarget, defer_encoder_output, defer_token_input,
};
use super::*;
use crate::{
    DecodeState, LogitsSelection,
    models::shared::DeferredChecks,
    models::t5::{T5FeedForwardProjection, T5LayerCrossAttention},
};

/// Deterministic CPU backend used by cache parity tests.
type TestBackend = Flex<f32, i32>;

/// Device selected by the deterministic test backend.
type TestDevice = <TestBackend as BackendTypes>::Device;

/// Complete tiny encoder/decoder binding used by input-contract regressions.
type ContractFixture = (
    T5ForConditionalGeneration<TestBackend>,
    EncoderOutput<TestBackend>,
    T5DecodeState<TestBackend>,
);

/// Stateless decoding against an arbitrary encoder result: the cached-parity oracle.
///
/// Production `forward` validates source and target together and produces the
/// encoder result itself. These tests instead pair one target with supplied,
/// possibly invalid, encoder results, so the oracle validates both with one
/// readback (target layout, encoder devices/extents/dtype, then target tokens,
/// target mask, and encoder mask values) and runs the production decoder path.
impl<B: Backend> T5ForConditionalGeneration<B> {
    /// Runs the decoder without persistent KV or cross-attention state.
    fn decode_stateless(
        &self,
        encoder: &EncoderOutput<B>,
        decoder: CausalLmInput<B>,
        options: &ForwardOptions,
    ) -> Result<CausalLmOutput<B>> {
        validate_options(options)?;
        let model_device = self.shared.weight.lazy_device();
        let mut checks = DeferredChecks::new();
        let target = defer_token_input(
            &decoder,
            self.config.vocab_size,
            &model_device,
            DECODER_INPUT,
            "decoder",
            &mut checks,
        )?;
        ensure_same_device(
            "t5.encoder_output.hidden_state",
            &encoder.hidden_state.device(),
            &model_device,
            PARAMETERS,
        )?;
        if let Some(mask) = &encoder.token_mask {
            ensure_same_device(
                "t5.encoder_output.token_mask",
                &mask.as_tensor().device(),
                &model_device,
                PARAMETERS,
            )?;
        }
        defer_encoder_output(
            &mut checks,
            encoder,
            decoder.token_ids.dims()[0],
            self.config.d_model,
            self.shared.weight.val().dtype(),
        )?;
        let target = ValidatedTarget::read(checks, &decoder, target, None, TargetColumns::Read)?;
        self.decode_validated(encoder, decoder, options, target.columns)
    }
}

/// Builds a small topology with configurable FFN and vocabulary-head layouts.
fn tiny_config(projection: T5FeedForwardProjection, tied: bool, layers: usize) -> T5Config {
    T5Config {
        vocab_size: 24,
        d_model: 8,
        d_kv: 4,
        d_ff: 16,
        num_layers: layers,
        num_decoder_layers: Some(layers),
        num_heads: 2,
        relative_attention_num_buckets: 8,
        relative_attention_max_distance: 16,
        dropout_rate: 0.0,
        feed_forward_proj: projection,
        tie_word_embeddings: tied,
        ..T5Config::default()
    }
}

/// Public encoder tensors with a different dtype must fail before backend matrix multiplication.
#[test]
fn release_contract_encoder_dtype_rejection_preserves_unbound_state() {
    let device = Default::default();
    let model = T5ForConditionalGeneration::<TestBackend>::init(
        tiny_config(T5FeedForwardProjection::Relu, true, 1),
        &device,
    )
    .unwrap();
    let encoder = encoded_source(&model, &device);
    let mut state = model
        .new_decode_state(CacheConfig::new(1, 4).unwrap(), 1, &device)
        .unwrap();
    let before = contract_snapshot(&state);
    for dtype in [DType::F16, DType::F64] {
        let mut invalid = encoder.clone();
        invalid.hidden_state = invalid.hidden_state.cast(dtype);
        for masked in [false, true] {
            let mut input = target(&[0], &device);
            if masked {
                input.token_mask =
                    Some(TokenMask::new(Tensor::from_bool([[true]].into(), &device)));
            }
            let error = model
                .decode_stateless(&invalid, input, &ForwardOptions::default())
                .unwrap_err();
            assert!(matches!(error, Error::BackendCapability(_)));
            assert!(
                error
                    .to_string()
                    .contains("t5.encoder_output.hidden_state dtype"),
                "{error}"
            );
        }
        let error = model
            .prefill_decoder(
                &invalid,
                target(&[0], &device),
                &mut state,
                &ForwardOptions::default(),
            )
            .expect_err("different encoder dtype must return a contextual error");
        assert!(matches!(error, Error::BackendCapability(_)));
        assert!(
            error.to_string().contains("decoder parameters F32"),
            "{error}"
        );
        assert!(
            error.to_string().contains(&format!("found {dtype:?}")),
            "{error}"
        );
        assert_eq!(contract_snapshot(&state), before);
    }
    let actual = model
        .prefill_decoder(
            &encoder,
            target(&[0], &device),
            &mut state,
            &ForwardOptions::default(),
        )
        .unwrap();
    let expected = model
        .decode_stateless(&encoder, target(&[0], &device), &ForwardOptions::default())
        .unwrap();
    assert_release_close(actual.decoder.logits, expected.logits);
    assert_eq!(state.lengths(), &[1]);
}

/// A bound source rejects dtype changes atomically even though same-dtype caller values are ignored.
#[test]
fn release_contract_encoder_dtype_rejection_preserves_populated_state() {
    let device = Default::default();
    let (model, encoder, mut state) = contract_fixture(&device, true);
    let before = contract_snapshot(&state);
    for dtype in [DType::F16, DType::F64] {
        let mut invalid = encoder.clone();
        invalid.hidden_state = invalid.hidden_state.cast(dtype);
        for prefill in [true, false] {
            let result = if prefill {
                model.prefill_decoder(
                    &invalid,
                    target(&[4], &device),
                    &mut state,
                    &ForwardOptions::default(),
                )
            } else {
                model.decode(
                    &invalid,
                    DecodeInput {
                        token_ids: target(&[4], &device).token_ids,
                        token_mask: None,
                        position_ids: None,
                    },
                    &mut state,
                    &ForwardOptions::default(),
                )
            };
            let error = result.expect_err("bound encoder dtype cannot change");
            assert!(matches!(error, Error::BackendCapability(_)));
            assert!(error.to_string().contains("bound encoder F32"), "{error}");
            assert!(
                error.to_string().contains(&format!("found {dtype:?}")),
                "{error}"
            );
            assert_eq!(contract_snapshot(&state), before);
        }
    }
    let actual = model
        .decode(
            &encoder,
            DecodeInput {
                token_ids: target(&[4], &device).token_ids,
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .unwrap();
    let expected = model
        .decode_stateless(
            &encoder,
            target(&[0, 3, 4], &device),
            &ForwardOptions::default(),
        )
        .unwrap();
    assert_release_close(actual.decoder.logits, expected.logits);
    assert_eq!(state.lengths(), &[3]);
}

/// Selects a real parameter dtype while retaining T5 normalization scales in F32.
struct RuntimeDtypeParameters(
    /// Dtype applied to non-normalization parameters.
    DType,
);

impl ModuleMapper<TestBackend> for RuntimeDtypeParameters {
    /// Replaces lazy weights with deterministic small matrices and unit F32 normalization scales.
    fn map_float<const D: usize>(
        &mut self,
        param: Param<Tensor<TestBackend, D>>,
    ) -> Param<Tensor<TestBackend, D>> {
        param.map(|tensor| {
            let shape = tensor.dims();
            let values = (0..shape.iter().product())
                .map(|index| {
                    if D == 1 {
                        1.0
                    } else {
                        (index % 19) as f32 * 0.01 - 0.09
                    }
                })
                .collect::<Vec<_>>();
            Tensor::<TestBackend, D>::from_data(TensorData::new(values, shape), &tensor.device())
                .cast(if D == 1 { DType::F32 } else { self.0 })
        })
    }
}

/// Actual parameter/cache dtype takes precedence over the Flex backend's default F32 element type.
#[test]
fn release_contract_matching_runtime_dtypes_accept_mixed_precision_norms() {
    let device = Default::default();
    for dtype in [DType::F32, DType::F16, DType::F64] {
        let model = T5ForConditionalGeneration::<TestBackend>::init(
            tiny_config(T5FeedForwardProjection::Relu, true, 1),
            &device,
        )
        .unwrap()
        .map(&mut RuntimeDtypeParameters(dtype));
        assert_eq!(model.shared.weight.val().dtype(), dtype);
        assert_eq!(
            model.decoder.final_layer_norm.weight.val().dtype(),
            DType::F32
        );
        let encoder = encoded_source(&model, &device);
        assert_eq!(encoder.hidden_state.dtype(), dtype);
        let mut state = model
            .new_decode_state(CacheConfig::new(1, 3).unwrap(), 1, &device)
            .unwrap();
        let prefill = model
            .prefill_decoder(
                &encoder,
                target(&[0], &device),
                &mut state,
                &ForwardOptions::default(),
            )
            .unwrap();
        let decoded = model
            .decode(
                &encoder,
                DecodeInput {
                    token_ids: target(&[3], &device).token_ids,
                    token_mask: None,
                    position_ids: None,
                },
                &mut state,
                &ForwardOptions::default(),
            )
            .unwrap();
        let stateless = model
            .decode_stateless(
                &encoder,
                target(&[0, 3], &device),
                &ForwardOptions::default(),
            )
            .unwrap();
        for logits in [
            prefill.decoder.logits,
            decoded.decoder.logits,
            stateless.logits,
        ] {
            assert_eq!(logits.dims(), [1, 1, 24]);
            assert_eq!(logits.dtype(), dtype);
            assert!(
                logits
                    .cast(DType::F32)
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap()
                    .into_iter()
                    .all(f32::is_finite)
            );
        }
        assert_eq!(state.lengths(), &[2]);
        assert_eq!(
            state
                .tensor_state()
                .main()
                .native_kv(0)
                .unwrap()
                .keys(0)
                .unwrap()
                .dtype(),
            dtype
        );
    }
}

/// Compares equally shaped finite F32 outputs independently of tensor assertion shortcuts.
fn assert_release_close<const D: usize>(
    left: Tensor<TestBackend, D>,
    right: Tensor<TestBackend, D>,
) {
    assert_eq!(left.dims(), right.dims());
    let left = left.into_data().to_vec::<f32>().unwrap();
    let right = right.into_data().to_vec::<f32>().unwrap();
    for (index, (left, right)) in left.into_iter().zip(right).enumerate() {
        assert!(left.is_finite() && right.is_finite(), "element {index}");
        assert!(
            (left - right).abs() <= 1e-4,
            "element {index}: {left} != {right}"
        );
    }
}

/// Checks the public encoder binding, including diagnostics and logical mask values.
fn assert_release_encoder(
    actual: &EncoderOutput<TestBackend>,
    expected: &EncoderOutput<TestBackend>,
) {
    assert_release_close(actual.hidden_state.clone(), expected.hidden_state.clone());
    assert_eq!(actual.hidden_states.len(), expected.hidden_states.len());
    for (actual, expected) in actual.hidden_states.iter().zip(&expected.hidden_states) {
        assert_release_close(actual.clone(), expected.clone());
    }
    assert_eq!(actual.attentions.len(), expected.attentions.len());
    for (actual, expected) in actual.attentions.iter().zip(&expected.attentions) {
        assert_release_close(actual.clone(), expected.clone());
    }
    assert_eq!(
        actual
            .token_mask
            .as_ref()
            .map(|mask| mask.as_tensor().clone().int().into_data()),
        expected
            .token_mask
            .as_ref()
            .map(|mask| mask.as_tensor().clone().int().into_data()),
    );
}

/// Beam changes must move gapped source masks and diagnostics together with immutable cross-KV.
#[test]
fn release_contract_masked_encoder_reorder_and_duplicate_preserve_continuation() {
    let device = Default::default();
    let model = T5ForConditionalGeneration::<TestBackend>::init(
        tiny_config(T5FeedForwardProjection::GatedGelu, false, 2),
        &device,
    )
    .unwrap();
    let options = ForwardOptions {
        hidden_states: true,
        ..ForwardOptions::default()
    };
    let encoder = model
        .encode(
            CausalLmInput {
                token_ids: Tensor::from_ints([[8, 0, 9, 0], [0, 10, 0, 11]], &device),
                token_mask: Some(TokenMask::new(Tensor::from_bool(
                    [[true, false, true, false], [false, true, false, true]].into(),
                    &device,
                ))),
                position_ids: None,
            },
            &options,
        )
        .unwrap();
    let mut state = model
        .new_decode_state(CacheConfig::new(2, 6).unwrap(), 2, &device)
        .unwrap();
    model
        .prefill_decoder(
            &encoder,
            CausalLmInput {
                token_ids: Tensor::from_ints([[0, 3], [0, 4]], &device),
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &options,
        )
        .unwrap();
    state.reorder(&[1, 0]).unwrap();
    let selector = Tensor::<TestBackend, 1, Int>::from_ints([1, 0], &device);
    let reordered = EncoderOutput {
        hidden_state: encoder.hidden_state.clone().select(0, selector.clone()),
        token_mask: Some(TokenMask::new(Tensor::from_bool(
            [[false, true, false, true], [true, false, true, false]].into(),
            &device,
        ))),
        hidden_states: encoder
            .hidden_states
            .iter()
            .map(|hidden| hidden.clone().select(0, selector.clone()))
            .collect(),
        attentions: Vec::new(),
    };
    // The public contract consumes the state's retained encoder after a beam
    // reorder, not same-shaped stale caller values or the caller's old mask.
    let actual = model
        .decode(
            &encoder,
            DecodeInput {
                token_ids: Tensor::from_ints([[5], [6]], &device),
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &options,
        )
        .unwrap();
    let expected = model
        .decode_stateless(
            &reordered,
            CausalLmInput {
                token_ids: Tensor::from_ints([[0, 4, 5], [0, 3, 6]], &device),
                token_mask: None,
                position_ids: None,
            },
            &options,
        )
        .unwrap();
    assert_release_close(actual.decoder.logits, expected.logits);
    assert_release_encoder(&actual.encoder, &reordered);
    assert_eq!(state.tensor_state().lengths(), &[3, 3]);

    state.reorder(&[0, 0]).unwrap();
    let selector = Tensor::<TestBackend, 1, Int>::from_ints([0, 0], &device);
    let duplicated = EncoderOutput {
        hidden_state: reordered.hidden_state.select(0, selector.clone()),
        token_mask: Some(TokenMask::new(Tensor::from_bool(
            [[false, true, false, true], [false, true, false, true]].into(),
            &device,
        ))),
        hidden_states: reordered
            .hidden_states
            .into_iter()
            .map(|hidden| hidden.select(0, selector.clone()))
            .collect(),
        attentions: Vec::new(),
    };
    let actual = model
        .decode(
            &encoder,
            DecodeInput {
                token_ids: Tensor::from_ints([[7], [8]], &device),
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &options,
        )
        .unwrap();
    let expected = model
        .decode_stateless(
            &duplicated,
            CausalLmInput {
                token_ids: Tensor::from_ints([[0, 4, 5, 7], [0, 4, 5, 8]], &device),
                token_mask: None,
                position_ids: None,
            },
            &options,
        )
        .unwrap();
    assert_release_close(actual.decoder.logits, expected.logits);
    assert_release_encoder(&actual.encoder, &duplicated);
    assert_eq!(state.lengths(), &[4, 4]);

    state.reset().unwrap();
    assert!(!state.has_encoder_cache());
    let rebound = model
        .prefill_decoder(
            &encoder,
            CausalLmInput {
                token_ids: Tensor::from_ints([[0], [0]], &device),
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &options,
        )
        .unwrap();
    assert_release_encoder(&rebound.encoder, &encoder);
    let expected = model
        .decode_stateless(
            &encoder,
            CausalLmInput {
                token_ids: Tensor::from_ints([[0], [0]], &device),
                token_mask: None,
                position_ids: None,
            },
            &options,
        )
        .unwrap();
    assert_release_close(rebound.decoder.logits, expected.logits);
    assert_eq!(state.lengths(), &[1, 1]);
}

/// Invalid output selections leave both cache banks intact and permit an ordered duplicate retry.
#[test]
fn release_contract_selective_logits_rejections_preserve_bound_history() {
    let device = Default::default();
    let (model, encoder, mut state) = contract_fixture(&device, true);
    let before = contract_snapshot(&state);
    for positions in [vec![], vec![2], vec![0, usize::MAX]] {
        let options = ForwardOptions {
            logits: LogitsSelection::Positions(positions),
            ..ForwardOptions::default()
        };
        for prefill in [true, false] {
            let input = target(&[4, 5], &device);
            let result = if prefill {
                model.prefill_decoder(&encoder, input, &mut state, &options)
            } else {
                model.decode(
                    &encoder,
                    DecodeInput {
                        token_ids: input.token_ids,
                        token_mask: None,
                        position_ids: None,
                    },
                    &mut state,
                    &options,
                )
            };
            let error = result.unwrap_err();
            assert!(
                matches!(error, Error::Shape { ref tensor, .. } if tensor == "forward_options.logits.positions")
            );
            assert_eq!(contract_snapshot(&state), before);
        }
    }
    let actual = model
        .decode(
            &encoder,
            DecodeInput {
                token_ids: target(&[4, 5], &device).token_ids,
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &ForwardOptions {
                logits: LogitsSelection::Positions(vec![1, 0, 1]),
                ..ForwardOptions::default()
            },
        )
        .unwrap();
    let expected = model
        .decode_stateless(
            &encoder,
            target(&[0, 3, 4, 5], &device),
            &ForwardOptions {
                logits: LogitsSelection::All,
                ..ForwardOptions::default()
            },
        )
        .unwrap();
    assert_release_close(
        actual.decoder.logits,
        expected.logits.select(
            1,
            Tensor::<TestBackend, 1, Int>::from_ints([3, 2, 3], &device),
        ),
    );
    assert_eq!(state.lengths(), &[4]);
}

/// An impossible portable index capacity is rejected before allocating a multi-billion-token cache.
#[test]
fn release_contract_capacity_rejects_unrepresentable_indices_before_allocation() {
    let device = Default::default();
    let model = T5ForConditionalGeneration::<TestBackend>::init(
        tiny_config(T5FeedForwardProjection::Relu, true, 1),
        &device,
    )
    .unwrap();
    let error = model
        .new_decode_state(
            CacheConfig::new(1, i32::MAX as usize + 1).unwrap(),
            1,
            &device,
        )
        .unwrap_err();
    assert!(
        matches!(error, Error::Configuration { ref field, .. } if field == "t5.decode_state.capacity")
    );
    assert!(
        error
            .to_string()
            .contains("portable signed position-index range")
    );
}

/// Encodes a fixed source prompt for one decoder row.
fn encoded_source(
    model: &T5ForConditionalGeneration<TestBackend>,
    device: &TestDevice,
) -> EncoderOutput<TestBackend> {
    model
        .encode(
            CausalLmInput {
                token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[8, 9, 10]], device),
                token_mask: None,
                position_ids: None,
            },
            &ForwardOptions::default(),
        )
        .expect("source encoding")
}

/// Creates an unmasked target segment from one row of IDs.
fn target(ids: &[i32], device: &TestDevice) -> CausalLmInput<TestBackend> {
    CausalLmInput {
        token_ids: Tensor::from_data(TensorData::new(ids.to_vec(), [1, ids.len()]), device),
        token_mask: None,
        position_ids: None,
    }
}

/// Compares two floating tensors at the crate's F32 golden tolerance.
fn assert_close<const D: usize>(left: Tensor<TestBackend, D>, right: Tensor<TestBackend, D>) {
    left.into_data()
        .assert_approx_eq::<f32>(&right.into_data(), Tolerance::absolute(1e-4));
}

/// Materializes every row's committed self-KV slots `[0, L_b)` in every layer.
///
/// Batched decoding writes each layer's tokens ahead of attention, so a
/// rejected call may leave bytes in slots beyond the lengths; those slots are
/// unreachable and excluded here. Atomicity is therefore asserted on the
/// logical cache contents and lengths, which is what later reads observe.
fn self_cache_snapshot(state: &T5DecodeState<TestBackend>) -> Vec<Vec<f32>> {
    let cache = state
        .tensors
        .main()
        .native_kv(SELF_KV_GROUP)
        .expect("self cache");
    let mut snapshot = Vec::new();
    for layer in 0..state.self_spec.layers() {
        for row in 0..state.tensors.batch_size() {
            let length = state.lengths()[row];
            let (keys, values) = match cache
                .read_sequence(layer, row, length)
                .expect("committed cache read")
                .materialize()
            {
                Some((keys, values)) => (
                    keys.into_data().to_vec::<f32>().expect("F32 keys"),
                    values.into_data().to_vec::<f32>().expect("F32 values"),
                ),
                None => (Vec::new(), Vec::new()),
            };
            snapshot.push(keys);
            snapshot.push(values);
        }
    }
    snapshot
}

/// Materializes all physical self-KV slots, including those beyond the lengths.
fn physical_cache_snapshot(state: &T5DecodeState<TestBackend>) -> Vec<f32> {
    let cache = state
        .tensors
        .main()
        .native_kv(SELF_KV_GROUP)
        .expect("self cache");
    let mut values = Vec::new();
    for layer in 0..state.self_spec.layers() {
        for tensor in [cache.keys(layer).unwrap(), cache.values(layer).unwrap()] {
            values.extend(
                tensor
                    .clone()
                    .into_data()
                    .to_vec::<f32>()
                    .expect("F32 cache"),
            );
        }
    }
    values
}

/// Includes source tensors and diagnostics because a rejected operation must retain its binding.
#[derive(Debug, PartialEq)]
struct ContractSnapshot {
    /// Materialized self-attention keys and values for every layer and sequence.
    self_kv: Vec<Vec<f32>>,
    /// Bound encoder tensors, projected cross-attention keys and values, and diagnostics.
    cross: Option<Vec<TensorData>>,
    /// Committed decoder prefix length for each sequence.
    lengths: Vec<usize>,
}

/// Captures the entire self cache and the immutable projected encoder bank.
fn contract_snapshot(state: &T5DecodeState<TestBackend>) -> ContractSnapshot {
    ContractSnapshot {
        self_kv: self_cache_snapshot(state),
        cross: state.cross.as_ref().map(|cross| {
            let mut data = vec![cross.encoder.hidden_state.clone().into_data()];
            if let Some(mask) = &cross.encoder.token_mask {
                data.push(mask.as_tensor().clone().into_data());
            }
            data.extend(
                cross
                    .keys
                    .iter()
                    .chain(&cross.values)
                    .map(|tensor| tensor.clone().into_data()),
            );
            data.extend(
                cross
                    .encoder
                    .hidden_states
                    .iter()
                    .map(|tensor| tensor.clone().into_data()),
            );
            data.extend(
                cross
                    .encoder
                    .attentions
                    .iter()
                    .map(|tensor| tensor.clone().into_data()),
            );
            data
        }),
        lengths: state.lengths().to_vec(),
    }
}

/// Prefills nonzero history so rejected appends cannot silently erase a valid prefix.
fn contract_fixture(device: &TestDevice, masked: bool) -> ContractFixture {
    let model = T5ForConditionalGeneration::init(
        tiny_config(T5FeedForwardProjection::Relu, true, 2),
        device,
    )
    .unwrap();
    let mut encoder = encoded_source(&model, device);
    if masked {
        encoder.token_mask = Some(TokenMask::new(Tensor::from_bool(
            [[true, false, true]].into(),
            device,
        )));
    }
    let mut state = model
        .new_decode_state(CacheConfig::new(1, 6).unwrap(), 1, device)
        .unwrap();
    model
        .prefill_decoder(
            &encoder,
            target(&[0, 3], device),
            &mut state,
            &ForwardOptions::default(),
        )
        .unwrap();
    (model, encoder, state)
}

/// Bad target rectangles fail before modifying an already bound encoder or self-KV prefix.
#[test]
fn cache_contract_invalid_rectangles_are_atomic() {
    let device = Default::default();
    let (model, encoder, mut state) = contract_fixture(&device, false);
    let before = contract_snapshot(&state);
    for (case, context) in [
        (0, "t5.decoder.input_ids"),
        (1, "batch"),
        (2, "t5.decoder.token_mask"),
        (3, "t5.decoder.position_ids"),
    ] {
        let mut input = DecodeInput {
            token_ids: target(&[4], &device).token_ids,
            token_mask: None,
            position_ids: None,
        };
        match case {
            0 => input.token_ids = Tensor::zeros([1, 0], &device),
            1 => input.token_ids = Tensor::zeros([2, 1], &device),
            2 => {
                input.token_mask = Some(TokenMask::new(Tensor::from_bool(
                    [[true, true]].into(),
                    &device,
                )))
            }
            3 => input.position_ids = Some(Tensor::zeros([1, 2], &device)),
            _ => unreachable!(),
        }
        let error = model
            .decode(&encoder, input, &mut state, &ForwardOptions::default())
            .unwrap_err();
        assert!(error.to_string().contains(context), "case {case}: {error}");
        assert_eq!(contract_snapshot(&state), before, "case {case}");
    }
}

/// Encoder batch, source length, and width are immutable after cross-attention binding.
#[test]
fn cache_contract_encoder_geometry_binding_is_atomic() {
    let device = Default::default();
    let (model, encoder, mut state) = contract_fixture(&device, false);
    let before = contract_snapshot(&state);
    for shape in [[2, 3, 8], [1, 2, 8], [1, 3, 7]] {
        let mut changed = encoder.clone();
        changed.hidden_state = Tensor::zeros(shape, &device);
        let error = model
            .decode(
                &changed,
                DecodeInput {
                    token_ids: target(&[4], &device).token_ids,
                    token_mask: None,
                    position_ids: None,
                },
                &mut state,
                &ForwardOptions::default(),
            )
            .unwrap_err();
        assert!(
            error.to_string().contains("t5.encoder_output.hidden_state"),
            "{error}"
        );
        assert!(
            error.to_string().contains("bound to decoder state"),
            "{error}"
        );
        assert_eq!(contract_snapshot(&state), before);
    }
}

/// Neither adding/removing a source mask nor changing its rectangle can alter an existing binding.
#[test]
fn cache_contract_encoder_mask_binding_is_atomic() {
    let device = Default::default();
    for masked in [false, true] {
        let (model, encoder, mut state) = contract_fixture(&device, masked);
        let before = contract_snapshot(&state);
        for wrong_shape in [false, true] {
            let mut changed = encoder.clone();
            changed.token_mask = if masked && !wrong_shape {
                None
            } else if wrong_shape {
                Some(TokenMask::new(Tensor::from_bool(
                    [[true, true]].into(),
                    &device,
                )))
            } else {
                Some(TokenMask::new(Tensor::from_bool(
                    [[true, true, true]].into(),
                    &device,
                )))
            };
            let error = model
                .decode(
                    &changed,
                    DecodeInput {
                        token_ids: target(&[4], &device).token_ids,
                        token_mask: None,
                        position_ids: None,
                    },
                    &mut state,
                    &ForwardOptions::default(),
                )
                .unwrap_err();
            assert!(
                error.to_string().contains("t5.encoder_output.token_mask"),
                "{error}"
            );
            assert_eq!(contract_snapshot(&state), before);
        }
    }
}

/// Same-shaped encoder arguments do not replace the documented retained source of truth.
#[test]
fn cache_contract_same_shape_encoder_values_do_not_rebind() {
    let device = Default::default();
    let (model, encoder, mut state) = contract_fixture(&device, true);
    let expected = model
        .decode_stateless(
            &encoder,
            target(&[0, 3, 4], &device),
            &ForwardOptions::default(),
        )
        .unwrap();
    let bound = contract_snapshot(&state).cross;
    let mut changed = encoder.clone();
    changed.hidden_state = Tensor::zeros([1, 3, 8], &device);
    changed.token_mask = Some(TokenMask::new(Tensor::from_bool(
        [[false, true, false]].into(),
        &device,
    )));
    let actual = model
        .decode(
            &changed,
            DecodeInput {
                token_ids: target(&[4], &device).token_ids,
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .unwrap();
    assert_close(actual.decoder.logits, expected.logits);
    assert_eq!(contract_snapshot(&state).cross, bound);
    assert_eq!(state.lengths(), &[3]);
}

/// A corrupted cross-layer count fails public decode before the valid self cache is changed.
#[test]
fn cache_contract_cross_geometry_rejection_is_atomic() {
    let device = Default::default();
    let (model, encoder, mut state) = contract_fixture(&device, false);
    state.cross.as_mut().unwrap().keys.pop();
    let before = contract_snapshot(&state);
    let error = model
        .decode(
            &encoder,
            DecodeInput {
                token_ids: target(&[4], &device).token_ids,
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("cross cache has 1 layers for a 2-layer decoder")
    );
    assert_eq!(contract_snapshot(&state), before);
}

/// Cached prefill exactly reproduces stateless logits and diagnostics.
#[test]
fn prefill_matches_stateless_for_dense_gated_tied_and_untied_models() {
    let device = Default::default();
    for (projection, tied) in [
        (T5FeedForwardProjection::Relu, true),
        (T5FeedForwardProjection::GatedGelu, false),
        (T5FeedForwardProjection::GatedSilu, false),
    ] {
        let model = T5ForConditionalGeneration::<TestBackend>::init(
            tiny_config(projection, tied, 2),
            &device,
        )
        .expect("tiny model");
        let encoder = encoded_source(&model, &device);
        let decoder = target(&[0, 4, 5], &device);
        let options = ForwardOptions {
            logits: LogitsSelection::All,
            hidden_states: true,
            ..ForwardOptions::default()
        };
        let stateless = model
            .decode_stateless(&encoder, decoder.clone(), &options)
            .expect("stateless decoder");
        let mut state = model
            .new_decode_state(CacheConfig::new(1, 8).unwrap(), 1, &device)
            .expect("decode state");
        let cached = model
            .prefill_decoder(&encoder, decoder, &mut state, &options)
            .expect("cached prefill")
            .decoder;

        assert_eq!(state.lengths(), [3]);
        assert!(state.has_encoder_cache());
        assert_eq!(state.encoder_sequence_length(), Some(3));
        assert_close(cached.logits, stateless.logits);
        assert_close(
            cached.final_hidden_state.expect("cached final"),
            stateless.final_hidden_state.expect("stateless final"),
        );
        assert_eq!(cached.hidden_states.len(), stateless.hidden_states.len());
        for (cached, stateless) in cached
            .hidden_states
            .into_iter()
            .zip(stateless.hidden_states)
        {
            assert_close(cached, stateless);
        }
    }
}

/// Multi-token and recurrent decode agree with the matching stateless suffix.
#[test]
fn chunked_and_recurrent_decode_match_full_forward() {
    let device = Default::default();
    let model = T5ForConditionalGeneration::<TestBackend>::init(
        tiny_config(T5FeedForwardProjection::GatedGelu, false, 2),
        &device,
    )
    .unwrap();
    let encoder = encoded_source(&model, &device);
    let options = ForwardOptions {
        logits: LogitsSelection::All,
        hidden_states: true,
        ..ForwardOptions::default()
    };
    let mut state = model
        .new_decode_state(CacheConfig::new(1, 8).unwrap(), 1, &device)
        .unwrap();
    model
        .prefill_decoder(&encoder, target(&[0, 3], &device), &mut state, &options)
        .unwrap();
    let chunk = model
        .decode(
            &encoder,
            DecodeInput {
                token_ids: target(&[4, 5], &device).token_ids,
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &options,
        )
        .expect("two-token decode")
        .decoder;
    let full = model
        .decode_stateless(&encoder, target(&[0, 3, 4, 5], &device), &options)
        .unwrap();
    assert_close(chunk.logits, full.logits.narrow(1, 2, 2));
    assert_close(
        chunk.final_hidden_state.unwrap(),
        full.final_hidden_state.unwrap().narrow(1, 2, 2),
    );
    assert_eq!(state.lengths(), [4]);

    let step = model
        .decode(
            &encoder,
            DecodeInput {
                token_ids: target(&[6], &device).token_ids,
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &options,
        )
        .expect("single-token decode")
        .decoder;
    let full = model
        .decode_stateless(&encoder, target(&[0, 3, 4, 5, 6], &device), &options)
        .unwrap();
    assert_close(step.logits, full.logits.narrow(1, 4, 1));
    assert_eq!(state.lengths(), [5]);
}

/// Ragged and gapped semantic masks have identical stateless/prefill meaning.
#[test]
fn ragged_gapped_prefill_matches_stateless_valid_positions() {
    let device = Default::default();
    let model = T5ForConditionalGeneration::<TestBackend>::init(
        tiny_config(T5FeedForwardProjection::GatedGelu, false, 2),
        &device,
    )
    .unwrap();
    let encoder = model
        .encode(
            CausalLmInput {
                token_ids: Tensor::from_ints([[8, 9, 10], [11, 12, 13]], &device),
                token_mask: Some(TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
                    [[true, false, true], [true, true, false]].into(),
                    &device,
                ))),
                position_ids: None,
            },
            &ForwardOptions::default(),
        )
        .unwrap();
    let decoder = CausalLmInput {
        token_ids: Tensor::from_ints([[0, 1, 4], [0, 5, 1]], &device),
        token_mask: Some(TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
            [[true, false, true], [true, true, false]].into(),
            &device,
        ))),
        position_ids: None,
    };
    let options = ForwardOptions {
        logits: LogitsSelection::All,
        hidden_states: true,
        ..ForwardOptions::default()
    };
    let stateless = model
        .decode_stateless(&encoder, decoder.clone(), &options)
        .unwrap();
    let mut state = model
        .new_decode_state(CacheConfig::new(2, 6).unwrap(), 2, &device)
        .unwrap();
    let cached = model
        .prefill_decoder(&encoder, decoder, &mut state, &options)
        .unwrap()
        .decoder;
    assert_eq!(state.lengths(), [2, 2]);
    assert_close(cached.logits, stateless.logits);
    assert_close(
        cached.final_hidden_state.unwrap(),
        stateless.final_hidden_state.unwrap(),
    );
    assert!(
        cached
            .hidden_states
            .iter()
            .all(|hidden| hidden.dims() == [2, 3, 8])
    );
}

/// Finished rows ignore masked sentinel IDs and preserve every physical slot.
#[test]
fn mixed_and_all_inactive_decode_rows_are_exact_no_ops() {
    let device = Default::default();
    let model = T5ForConditionalGeneration::<TestBackend>::init(
        tiny_config(T5FeedForwardProjection::Relu, true, 2),
        &device,
    )
    .unwrap();
    let encoder = model
        .encode(
            CausalLmInput {
                token_ids: Tensor::from_ints([[8, 9], [10, 11]], &device),
                token_mask: None,
                position_ids: None,
            },
            &ForwardOptions::default(),
        )
        .unwrap();
    let mut state = model
        .new_decode_state(CacheConfig::new(2, 6).unwrap(), 2, &device)
        .unwrap();
    model
        .prefill_decoder(
            &encoder,
            CausalLmInput {
                token_ids: Tensor::from_ints([[0, 3], [0, 99]], &device),
                token_mask: Some(TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
                    [[true, true], [true, false]].into(),
                    &device,
                ))),
                position_ids: None,
            },
            &mut state,
            &ForwardOptions {
                logits: LogitsSelection::All,
                ..ForwardOptions::default()
            },
        )
        .unwrap();
    assert_eq!(state.lengths(), [2, 1]);
    let before = self_cache_snapshot(&state);
    let output = model
        .decode(
            &encoder,
            DecodeInput {
                token_ids: Tensor::from_ints([[4, 5], [-1, 99]], &device),
                token_mask: Some(TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
                    [[true, true], [false, false]].into(),
                    &device,
                ))),
                position_ids: None,
            },
            &mut state,
            &ForwardOptions {
                logits: LogitsSelection::All,
                hidden_states: true,
                ..ForwardOptions::default()
            },
        )
        .expect("mixed decode");
    assert_eq!(state.lengths(), [4, 1]);
    let after = self_cache_snapshot(&state);
    let per_row_layer_tensors = 2;
    for layer in 0..2 {
        let row_one_start = (layer * 2 + 1) * per_row_layer_tensors;
        assert_eq!(after[row_one_start], before[row_one_start]);
        assert_eq!(after[row_one_start + 1], before[row_one_start + 1]);
    }
    assert!(
        output
            .decoder
            .logits
            .slice([1..2, 0..2, 0..24])
            .into_data()
            .to_vec::<f32>()
            .unwrap()
            .iter()
            .all(|value| *value == 0.0)
    );
    assert!(
        output
            .decoder
            .final_hidden_state
            .unwrap()
            .slice([1..2, 0..2, 0..8])
            .into_data()
            .to_vec::<f32>()
            .unwrap()
            .iter()
            .all(|value| *value == 0.0)
    );
    assert!(output.decoder.hidden_states.iter().all(|hidden| {
        hidden
            .clone()
            .slice([1..2, 0..2, 0..8])
            .into_data()
            .to_vec::<f32>()
            .unwrap()
            .iter()
            .all(|value| *value == 0.0)
    }));

    let before = self_cache_snapshot(&state);
    let lengths = state.lengths().to_vec();
    let output = model
        .decode(
            &encoder,
            DecodeInput {
                token_ids: Tensor::from_ints([[99], [-7]], &device),
                token_mask: Some(TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
                    [[false], [false]].into(),
                    &device,
                ))),
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .expect("all-inactive decode");
    assert_eq!(state.lengths(), lengths);
    assert_eq!(self_cache_snapshot(&state), before);
    assert!(
        output
            .decoder
            .logits
            .into_data()
            .to_vec::<f32>()
            .unwrap()
            .iter()
            .all(|value| *value == 0.0)
    );
}

/// A later-layer error leaves lengths, cross state, and all physical KV unchanged.
#[test]
fn later_layer_failure_is_result_atomic() {
    let device = Default::default();
    let mut model = T5ForConditionalGeneration::<TestBackend>::init(
        tiny_config(T5FeedForwardProjection::Relu, true, 2),
        &device,
    )
    .unwrap();
    let encoder = encoded_source(&model, &device);
    let mut state = model
        .new_decode_state(CacheConfig::new(1, 6).unwrap(), 1, &device)
        .unwrap();
    model
        .prefill_decoder(
            &encoder,
            target(&[0, 3], &device),
            &mut state,
            &ForwardOptions::default(),
        )
        .unwrap();
    let before = self_cache_snapshot(&state);
    let lengths = state.lengths().to_vec();
    let cross_before = state.cross.as_ref().unwrap().keys[0].clone().into_data();
    model.decoder.block[1].cross_attention = None;
    let error = model
        .decode(
            &encoder,
            DecodeInput {
                token_ids: target(&[4], &device).token_ids,
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .expect_err("corrupt later layer");
    assert!(error.to_string().contains("cross_attention"));
    assert_eq!(state.lengths(), lengths);
    assert_eq!(self_cache_snapshot(&state), before);
    assert_eq!(
        state.cross.as_ref().unwrap().keys[0].clone().into_data(),
        cross_before
    );
}

/// Capacity failure occurs before model execution and preserves the complete state.
#[test]
fn capacity_boundary_failure_preserves_state() {
    let device = Default::default();
    let model = T5ForConditionalGeneration::<TestBackend>::init(
        tiny_config(T5FeedForwardProjection::GatedGelu, false, 1),
        &device,
    )
    .unwrap();
    let encoder = encoded_source(&model, &device);
    let mut state = model
        .new_decode_state(CacheConfig::new(1, 2).unwrap(), 1, &device)
        .unwrap();
    model
        .prefill_decoder(
            &encoder,
            target(&[0, 3], &device),
            &mut state,
            &ForwardOptions::default(),
        )
        .unwrap();
    let before = self_cache_snapshot(&state);
    let error = model
        .decode(
            &encoder,
            DecodeInput {
                token_ids: target(&[4], &device).token_ids,
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .expect_err("full cache");
    assert!(error.to_string().contains("capacity"));
    assert_eq!(state.lengths(), [2]);
    assert_eq!(self_cache_snapshot(&state), before);
}

/// Decoder steps consume cached encoder K/V even if projection weights change.
#[test]
fn cross_attention_key_values_are_projected_only_during_prefill() {
    let device = Default::default();
    let mut model = T5ForConditionalGeneration::<TestBackend>::init(
        tiny_config(T5FeedForwardProjection::Relu, true, 2),
        &device,
    )
    .unwrap();
    let encoder = encoded_source(&model, &device);
    let mut state = model
        .new_decode_state(CacheConfig::new(1, 5).unwrap(), 1, &device)
        .unwrap();
    model
        .prefill_decoder(
            &encoder,
            target(&[0], &device),
            &mut state,
            &ForwardOptions::default(),
        )
        .unwrap();
    for block in &mut model.decoder.block {
        let cross = block.cross_attention.as_mut().unwrap();
        cross.enc_dec_attention.attention.k_proj.weight = Initializer::Constant { value: f64::NAN }
            .init(
                cross.enc_dec_attention.attention.k_proj.weight.shape(),
                &device,
            );
        cross.enc_dec_attention.attention.v_proj.weight = Initializer::Constant { value: f64::NAN }
            .init(
                cross.enc_dec_attention.attention.v_proj.weight.shape(),
                &device,
            );
    }
    let output = model
        .decode(
            &encoder,
            DecodeInput {
                token_ids: target(&[4], &device).token_ids,
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .expect("decode must not execute changed K/V projections");
    assert!(
        output
            .decoder
            .logits
            .into_data()
            .to_vec::<f32>()
            .unwrap()
            .iter()
            .all(|value| value.is_finite())
    );
}

/// Gapped encoder masks exclude cached source columns exactly as stateless attention does.
#[test]
fn cached_cross_attention_preserves_gapped_encoder_masks() {
    let device = Default::default();
    let model = T5ForConditionalGeneration::<TestBackend>::init(
        tiny_config(T5FeedForwardProjection::GatedSilu, false, 1),
        &device,
    )
    .unwrap();
    let mask = TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
        [[true, false, true]].into(),
        &device,
    ));
    let encoder = EncoderOutput {
        hidden_state: Tensor::from_data(
            TensorData::new(
                vec![
                    0.1_f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 10_000.0, -10_000.0, 9_000.0,
                    -9_000.0, 8_000.0, -8_000.0, 7_000.0, -7_000.0, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3,
                    0.2, 0.1,
                ],
                [1, 3, 8],
            ),
            &device,
        ),
        token_mask: Some(mask),
        hidden_states: Vec::new(),
        attentions: Vec::new(),
    };
    let options = ForwardOptions {
        logits: LogitsSelection::All,
        ..ForwardOptions::default()
    };
    let stateless = model
        .decode_stateless(&encoder, target(&[0, 4], &device), &options)
        .unwrap();
    let mut state = model
        .new_decode_state(CacheConfig::new(1, 4).unwrap(), 1, &device)
        .unwrap();
    let cached = model
        .prefill_decoder(&encoder, target(&[0, 4], &device), &mut state, &options)
        .unwrap();
    assert_close(cached.decoder.logits, stateless.logits);
}

/// Reorder, truncate, and reset coordinate both self and cross state.
#[test]
fn lifecycle_operations_coordinate_self_and_cross_caches() {
    let device = Default::default();
    let model = T5ForConditionalGeneration::<TestBackend>::init(
        tiny_config(T5FeedForwardProjection::Relu, true, 1),
        &device,
    )
    .unwrap();
    let encoder = model
        .encode(
            CausalLmInput {
                token_ids: Tensor::from_ints([[8, 9], [10, 11]], &device),
                token_mask: None,
                position_ids: None,
            },
            &ForwardOptions::default(),
        )
        .unwrap();
    let mut state = model
        .new_decode_state(CacheConfig::new(2, 5).unwrap(), 2, &device)
        .unwrap();
    model
        .prefill_decoder(
            &encoder,
            CausalLmInput {
                token_ids: Tensor::from_ints([[0, 3], [0, 99]], &device),
                token_mask: Some(TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
                    [[true, true], [true, false]].into(),
                    &device,
                ))),
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .unwrap();
    let cross_before = state.cross.as_ref().unwrap().keys[0].clone();
    state.reorder(&[1, 0]).expect("beam reorder");
    assert_eq!(state.lengths(), [1, 2]);
    assert_close(
        state.cross.as_ref().unwrap().keys[0]
            .clone()
            .slice([0..1, 0..2, 0..2, 0..4]),
        cross_before.slice([1..2, 0..2, 0..2, 0..4]),
    );
    let first_reordered_cross = state.cross.as_ref().unwrap().keys[0].clone();
    state.reorder(&[0, 0]).expect("beam duplication");
    assert_eq!(state.lengths(), [1, 1]);
    assert_close(
        state.cross.as_ref().unwrap().keys[0]
            .clone()
            .slice([1..2, 0..2, 0..2, 0..4]),
        first_reordered_cross.slice([0..1, 0..2, 0..2, 0..4]),
    );
    state.truncate(&[0, 1]).expect("full-cache rewind");
    assert_eq!(state.lengths(), [0, 1]);
    assert!(state.has_encoder_cache());
    state.reset().expect("state reset");
    assert_eq!(state.lengths(), [0, 0]);
    assert!(!state.has_encoder_cache());
    assert!(
        physical_cache_snapshot(&state)
            .into_iter()
            .all(|value| value == 0.0)
    );
}

/// A full-KV rewind hides and overwrites the stale suffix on the next decode.
#[test]
fn truncate_then_decode_matches_the_shortened_prefix() {
    let device = Default::default();
    let model = T5ForConditionalGeneration::<TestBackend>::init(
        tiny_config(T5FeedForwardProjection::GatedGelu, false, 2),
        &device,
    )
    .unwrap();
    let encoder = encoded_source(&model, &device);
    let mut state = model
        .new_decode_state(CacheConfig::new(1, 6).unwrap(), 1, &device)
        .unwrap();
    model
        .prefill_decoder(
            &encoder,
            target(&[0, 3, 4], &device),
            &mut state,
            &ForwardOptions::default(),
        )
        .unwrap();
    state.truncate(&[1]).expect("full cache supports rewind");
    assert!(state.has_encoder_cache());
    let cached = model
        .decode(
            &encoder,
            DecodeInput {
                token_ids: target(&[5], &device).token_ids,
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .unwrap();
    let stateless = model
        .decode_stateless(
            &encoder,
            target(&[0, 5], &device),
            &ForwardOptions::default(),
        )
        .unwrap();
    assert_close(cached.decoder.logits, stateless.logits);
    assert_eq!(state.lengths(), [2]);
    let snapshot = self_cache_snapshot(&state);
    assert!(state.truncate(&[3]).is_err());
    assert_eq!(state.lengths(), [2]);
    assert_eq!(self_cache_snapshot(&state), snapshot);
}

/// Invalid lifecycle/input operations are contextual and do not initialize cross state.
#[test]
fn invalid_cached_inputs_fail_before_state_mutation() {
    let device = Default::default();
    let model = T5ForConditionalGeneration::<TestBackend>::init(
        tiny_config(T5FeedForwardProjection::Relu, true, 1),
        &device,
    )
    .unwrap();
    let encoder = encoded_source(&model, &device);
    let mut state = model
        .new_decode_state(CacheConfig::new(1, 4).unwrap(), 1, &device)
        .unwrap();
    let error = model
        .decode(
            &encoder,
            DecodeInput {
                token_ids: target(&[1], &device).token_ids,
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .expect_err("decode before prefill");
    assert!(error.to_string().contains("successful decoder prefill"));
    assert!(!state.has_encoder_cache());

    let error = model
        .prefill_decoder(
            &encoder,
            CausalLmInput {
                token_ids: target(&[0], &device).token_ids,
                token_mask: None,
                position_ids: Some(Tensor::zeros([1, 1], &device)),
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .expect_err("explicit positions are unsupported");
    assert!(matches!(error, Error::BackendCapability(_)));
    assert_eq!(state.lengths(), [0]);
    assert!(!state.has_encoder_cache());

    let error = model
        .prefill_decoder(
            &encoder,
            CausalLmInput {
                token_ids: Tensor::from_ints([[99]], &device),
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .expect_err("active invalid ID");
    assert!(error.to_string().contains("0..24"));
    assert_eq!(state.lengths(), [0]);
    assert!(!state.has_encoder_cache());

    let error = model
        .prefill_decoder(
            &encoder,
            CausalLmInput {
                token_ids: Tensor::from_ints([[99]], &device),
                token_mask: Some(TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
                    [[false]].into(),
                    &device,
                ))),
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .expect_err("prefill cannot contain an inactive row");
    assert!(error.to_string().contains("fully masked prompt row"));
    assert_eq!(state.lengths(), [0]);
    assert!(!state.has_encoder_cache());

    assert!(state.reorder(&[]).is_err());
    assert!(state.reorder(&[1]).is_err());
    assert_eq!(state.lengths(), [0]);
}

/// Initial cross projection is also deferred when the output head fails.
#[test]
fn initial_prefill_output_failure_does_not_bind_encoder() {
    let device = Default::default();
    let mut model = T5ForConditionalGeneration::<TestBackend>::init(
        tiny_config(T5FeedForwardProjection::Relu, true, 1),
        &device,
    )
    .unwrap();
    let encoder = encoded_source(&model, &device);
    let mut state = model
        .new_decode_state(CacheConfig::new(1, 4).unwrap(), 1, &device)
        .unwrap();
    model.lm_head = Some(super::super::t5_linear(8, 24, 1.0, &device));
    model
        .prefill_decoder(
            &encoder,
            target(&[0], &device),
            &mut state,
            &ForwardOptions::default(),
        )
        .expect_err("tied topology with an independent head");
    assert_eq!(state.lengths(), [0]);
    assert!(!state.has_encoder_cache());
    assert!(
        self_cache_snapshot(&state)
            .into_iter()
            .flatten()
            .all(|value| value == 0.0)
    );
}

/// Corrupt cross-attention topology is rejected instead of indexing cache vectors.
#[test]
fn incompatible_model_and_state_layout_is_rejected() {
    let device = Default::default();
    let one = T5ForConditionalGeneration::<TestBackend>::init(
        tiny_config(T5FeedForwardProjection::Relu, true, 1),
        &device,
    )
    .unwrap();
    let two = T5ForConditionalGeneration::<TestBackend>::init(
        tiny_config(T5FeedForwardProjection::Relu, true, 2),
        &device,
    )
    .unwrap();
    let encoder = encoded_source(&two, &device);
    let mut state = one
        .new_decode_state(CacheConfig::new(1, 4).unwrap(), 1, &device)
        .unwrap();
    let error = two
        .prefill_decoder(
            &encoder,
            target(&[0], &device),
            &mut state,
            &ForwardOptions::default(),
        )
        .expect_err("state belongs to another topology");
    assert!(error.to_string().contains("incompatible model layout"));
}

/// Test-only construction ensures a decoder block without cross attention errors cleanly.
#[test]
fn cross_projection_rejects_missing_decoder_component() {
    let device = Default::default();
    let mut model = T5ForConditionalGeneration::<TestBackend>::init(
        tiny_config(T5FeedForwardProjection::Relu, true, 1),
        &device,
    )
    .unwrap();
    let encoder = encoded_source(&model, &device);
    model.decoder.block[0].cross_attention = None::<T5LayerCrossAttention<TestBackend>>;
    let error = T5CrossAttentionCache::project(&model, &encoder)
        .expect_err("missing cross layer must fail");
    assert!(error.to_string().contains("cross_attention"));
}

/// Full-precision stateless execution retains gradients for training v2.
#[cfg(feature = "autodiff")]
#[test]
fn stateless_encoder_decoder_graph_has_parameter_gradients() {
    use burn::backend::Autodiff;

    /// Autodiff wrapper used to verify encoder and decoder parameter gradients.
    type AdBackend = Autodiff<TestBackend>;

    let device = Default::default();
    let model = T5ForConditionalGeneration::<AdBackend>::init(
        tiny_config(T5FeedForwardProjection::GatedGelu, false, 1),
        &device,
    )
    .unwrap();
    let output = model
        .forward(
            Seq2SeqInput {
                encoder: CausalLmInput {
                    token_ids: Tensor::<AdBackend, 2, Int>::from_ints([[8, 9]], &device),
                    token_mask: None,
                    position_ids: None,
                },
                decoder: CausalLmInput {
                    token_ids: Tensor::<AdBackend, 2, Int>::from_ints([[0, 1, 4]], &device),
                    token_mask: Some(TokenMask::new(Tensor::<AdBackend, 2, Bool>::from_bool(
                        [[true, false, true]].into(),
                        &device,
                    ))),
                    position_ids: None,
                },
            },
            &ForwardOptions {
                logits: LogitsSelection::All,
                ..ForwardOptions::default()
            },
        )
        .expect("autodiff forward");
    let gradients =
        burn::optim::GradientsParams::from_grads(output.decoder.logits.sum().backward(), &model);
    assert!(
        gradients
            .get::<TestBackend, 2>(model.lm_head.as_ref().unwrap().weight.id)
            .is_some()
    );
    assert!(
        gradients
            .get::<TestBackend, 1>(model.decoder.final_layer_norm.weight.id)
            .is_some()
    );
    assert!(
        gradients
            .get::<TestBackend, 2>(
                model.decoder.block[0]
                    .feed_forward
                    .gated
                    .as_ref()
                    .unwrap()
                    .down_proj
                    .weight
                    .id
            )
            .is_some()
    );
    assert!(
        gradients
            .get::<TestBackend, 2>(
                model.encoder.block[0]
                    .self_attention
                    .self_attention
                    .attention
                    .q_proj
                    .weight
                    .id
            )
            .is_some()
    );
    assert!(
        gradients
            .get::<TestBackend, 2>(
                model.decoder.block[0]
                    .cross_attention
                    .as_ref()
                    .unwrap()
                    .enc_dec_attention
                    .attention
                    .q_proj
                    .weight
                    .id
            )
            .is_some()
    );
    assert!(
        gradients
            .get::<TestBackend, 2>(
                model.decoder.block[0]
                    .cross_attention
                    .as_ref()
                    .unwrap()
                    .enc_dec_attention
                    .attention
                    .k_proj
                    .weight
                    .id
            )
            .is_some()
    );
    assert!(
        gradients
            .get::<TestBackend, 2>(model.shared.weight.id)
            .is_some()
    );
}

/// Builds a `[rows, columns]` validity mask (`true == valid`) on `device`.
fn validity<const R: usize, const C: usize>(
    values: [[bool; C]; R],
    device: &TestDevice,
) -> TokenMask<TestBackend> {
    TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
        values.into(),
        device,
    ))
}

/// Builds a target or source input from `[rows, columns]` IDs and an optional mask.
fn padded_input<const R: usize, const C: usize>(
    ids: [[i32; C]; R],
    mask: Option<[[bool; C]; R]>,
    device: &TestDevice,
) -> CausalLmInput<TestBackend> {
    CausalLmInput {
        token_ids: Tensor::<TestBackend, 2, Int>::from_ints(ids, device),
        token_mask: mask.map(|mask| validity(mask, device)),
        position_ids: None,
    }
}

/// Reads a float tensor into exact row-major host values.
fn exact<const D: usize>(tensor: Tensor<TestBackend, D>) -> Vec<f32> {
    tensor.into_data().to_vec::<f32>().expect("f32 tensor data")
}

/// The fused single-readback `forward` reproduces separately validated stages bit for bit.
///
/// Source and target use left, interior, and right padding, so the masked
/// compaction path consumes the valid columns read with the validation.
#[test]
fn fused_forward_matches_separately_validated_encode_and_decode() {
    let device = Default::default();
    let model = T5ForConditionalGeneration::<TestBackend>::init(
        tiny_config(T5FeedForwardProjection::GatedGelu, false, 2),
        &device,
    )
    .unwrap();
    let source = padded_input(
        [[8, 9, 10, 0], [8, 0, 9, 10]],
        Some([[true, true, true, false], [true, false, true, true]]),
        &device,
    );
    let target = padded_input(
        [[0, 3, 0, 4], [0, 0, 5, 6]],
        Some([[true, true, false, true], [false, false, true, true]]),
        &device,
    );
    let options = ForwardOptions {
        logits: LogitsSelection::All,
        hidden_states: true,
        ..ForwardOptions::default()
    };
    let fused = model
        .forward(
            Seq2SeqInput {
                encoder: source.clone(),
                decoder: target.clone(),
            },
            &options,
        )
        .unwrap();
    let encoder = model.encode(source, &options).unwrap();
    let staged = model.decode_stateless(&encoder, target, &options).unwrap();
    assert_eq!(
        exact(fused.encoder.hidden_state),
        exact(encoder.hidden_state)
    );
    assert_eq!(exact(fused.decoder.logits), exact(staged.logits));
    assert_eq!(
        fused.decoder.hidden_states.len(),
        staged.hidden_states.len()
    );
    for (fused, staged) in fused
        .decoder
        .hidden_states
        .into_iter()
        .zip(staged.hidden_states)
    {
        assert_eq!(exact(fused), exact(staged));
    }
}

/// Complete passes validate both inputs before encoding: layout errors first,
/// then source values, then target values, with and without hooks.
#[test]
fn complete_pass_validation_reports_layout_then_source_then_target_errors() {
    let device = Default::default();
    let model = T5ForConditionalGeneration::<TestBackend>::init(
        tiny_config(T5FeedForwardProjection::Relu, true, 1),
        &device,
    )
    .unwrap();
    let run = |input: Seq2SeqInput<TestBackend>, hooked: bool| {
        if hooked {
            let mut hooks = crate::intervention::Hooks::new();
            hooks.edit(
                crate::intervention::Layer::decoder(0).cross_attention_output(),
                |activation| activation,
            );
            model.forward_with_hooks(input, &ForwardOptions::default(), &mut hooks)
        } else {
            model.forward(input, &ForwardOptions::default())
        }
    };
    let shape = |error: Error| match error {
        Error::Shape { tensor, actual, .. } => (tensor, actual),
        other => panic!("expected a shape error, found {other:?}"),
    };
    for hooked in [false, true] {
        // Source token values precede target token values.
        let error = run(
            Seq2SeqInput {
                encoder: padded_input([[99, 1]], None, &device),
                decoder: padded_input([[0, 99]], None, &device),
            },
            hooked,
        )
        .unwrap_err();
        assert_eq!(
            shape(error),
            (
                "t5.encoder.input_ids".into(),
                "observed range 1..=99".into()
            )
        );

        // A target layout error precedes any value error.
        let mut decoder = padded_input([[0, 1]], None, &device);
        decoder.token_mask = Some(validity([[true, true, true]], &device));
        let error = run(
            Seq2SeqInput {
                encoder: padded_input([[99, 1]], None, &device),
                decoder,
            },
            hooked,
        )
        .unwrap_err();
        assert_eq!(shape(error).0, "t5.decoder.token_mask");

        // A fully masked target row is rejected after valid source values.
        let error = run(
            Seq2SeqInput {
                encoder: padded_input([[4, 1], [2, 3]], None, &device),
                decoder: padded_input(
                    [[0, 1], [0, 2]],
                    Some([[true, true], [false, false]]),
                    &device,
                ),
            },
            hooked,
        )
        .unwrap_err();
        assert_eq!(
            shape(error),
            ("t5.decoder.token_mask".into(), "a fully masked row".into())
        );

        // A fully masked source row is named after the source input.
        let error = run(
            Seq2SeqInput {
                encoder: padded_input([[4, 1]], Some([[false, false]]), &device),
                decoder: padded_input([[0, 1]], None, &device),
            },
            hooked,
        )
        .unwrap_err();
        assert_eq!(
            shape(error),
            ("t5.encoder.token_mask".into(), "a fully masked row".into())
        );

        // Mismatched source and target batches are still rejected after encoding.
        let error = run(
            Seq2SeqInput {
                encoder: padded_input([[4, 1], [2, 3]], None, &device),
                decoder: padded_input([[0, 1]], None, &device),
            },
            hooked,
        )
        .unwrap_err();
        assert_eq!(shape(error).0, "t5.encoder_output.hidden_state");
    }
}

/// The test oracle reports a supplied fully masked source row after target values.
#[test]
fn supplied_source_mask_is_checked_after_target_values() {
    let device = Default::default();
    let model = T5ForConditionalGeneration::<TestBackend>::init(
        tiny_config(T5FeedForwardProjection::Relu, true, 1),
        &device,
    )
    .unwrap();
    let mut encoder = encoded_source(&model, &device);
    encoder.token_mask = Some(validity([[false, false, false]], &device));
    let options = ForwardOptions::default();
    let error = model
        .decode_stateless(&encoder, target(&[0, 99], &device), &options)
        .unwrap_err();
    assert!(
        error.to_string().contains("t5.decoder.input_ids"),
        "{error}"
    );
    let error = model
        .decode_stateless(&encoder, target(&[0, 1], &device), &options)
        .unwrap_err();
    assert!(
        matches!(&error, Error::Shape { tensor, actual, .. }
            if tensor == "t5.encoder_output.token_mask" && actual == "a fully masked row"),
        "{error}"
    );
}

/// Cached cross projections follow the crate's linear contract instead of panicking.
///
/// A cross-attention key projection stored in another dtype than the encoder
/// states is reported as a contextual capability error, exactly as the
/// stateless cross-attention projection reports it.
#[test]
fn cross_projection_rejects_mismatched_weight_dtypes_contextually() {
    let device = Default::default();
    let mut model = T5ForConditionalGeneration::<TestBackend>::init(
        tiny_config(T5FeedForwardProjection::Relu, true, 1),
        &device,
    )
    .unwrap();
    let encoder = encoded_source(&model, &device);
    let attention = &mut model.decoder.block[0]
        .cross_attention
        .as_mut()
        .unwrap()
        .enc_dec_attention
        .attention;
    attention.k_proj.weight = attention
        .k_proj
        .weight
        .clone()
        .map(|weight| weight.cast(DType::F64));
    let error = T5CrossAttentionCache::project(&model, &encoder).unwrap_err();
    assert!(matches!(error, Error::BackendCapability(_)), "{error}");
    assert!(error.to_string().contains("linear.weight dtype"), "{error}");
    let mut state = model
        .new_decode_state(CacheConfig::new(1, 4).unwrap(), 1, &device)
        .unwrap();
    assert!(
        model
            .prefill_decoder(
                &encoder,
                target(&[0], &device),
                &mut state,
                &ForwardOptions::default(),
            )
            .is_err()
    );
    assert!(!state.has_encoder_cache());
}

/// Returns whether a Flex tensor occupies contiguous row-major storage.
fn is_contiguous<const D: usize>(tensor: Tensor<TestBackend, D>) -> bool {
    let primitive: burn::backend::flex::FlexTensor = tensor.into_primitive().tensor();
    primitive.is_contiguous()
}

/// Projects one layer's cross keys or values as the strided head view decode used to cache.
fn strided_cross_heads(
    model: &T5ForConditionalGeneration<TestBackend>,
    encoder: &EncoderOutput<TestBackend>,
    layer: usize,
    keys: bool,
) -> Tensor<TestBackend, 4> {
    let attention = &model.decoder.block[layer]
        .cross_attention
        .as_ref()
        .unwrap()
        .enc_dec_attention
        .attention;
    let projection = if keys {
        &attention.k_proj
    } else {
        &attention.v_proj
    };
    let [batch, source, _] = encoder.hidden_state.dims();
    operations::linear(encoder.hidden_state.clone(), projection)
        .unwrap()
        .reshape([batch, source, attention.key_value_heads, attention.head_dim])
        .swap_dims(1, 2)
}

/// Prefill stores cross keys and values contiguously on Flex with exactly the projected values.
///
/// Flex's attention copied the former strided head view on every decode
/// step; the cache now holds attention-ready storage from prefill on, and a
/// beam reorder keeps it contiguous.
#[test]
fn cached_cross_projections_are_attention_ready_and_exact() {
    let device = Default::default();
    let model = T5ForConditionalGeneration::<TestBackend>::init(
        tiny_config(T5FeedForwardProjection::GatedGelu, false, 2),
        &device,
    )
    .unwrap();
    let encoder = model
        .encode(
            CausalLmInput {
                token_ids: Tensor::from_ints([[8, 9, 10], [11, 12, 13]], &device),
                token_mask: None,
                position_ids: None,
            },
            &ForwardOptions::default(),
        )
        .unwrap();
    let mut state = model
        .new_decode_state(CacheConfig::new(2, 4).unwrap(), 2, &device)
        .unwrap();
    model
        .prefill_decoder(
            &encoder,
            CausalLmInput {
                token_ids: Tensor::from_ints([[0], [0]], &device),
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .unwrap();
    let cross = state.cross.as_ref().unwrap();
    for layer in 0..2 {
        for (cached, keys) in [(&cross.keys[layer], true), (&cross.values[layer], false)] {
            let strided = strided_cross_heads(&model, &encoder, layer, keys);
            assert!(!is_contiguous(strided.clone()));
            assert!(is_contiguous(cached.clone()), "layer {layer} keys={keys}");
            assert_eq!(cached.clone().into_data(), strided.into_data());
        }
    }
    let before = state.cross.as_ref().unwrap().values[1].clone();
    state.reorder(&[1, 1]).unwrap();
    let after = state.cross.as_ref().unwrap().values[1].clone();
    assert!(is_contiguous(after.clone()));
    assert_eq!(
        after.narrow(0, 0, 1).into_data(),
        before.narrow(0, 1, 1).into_data()
    );
}

/// Decoding from attention-ready cross storage is bit-identical to decoding from the strided view.
///
/// The layout change is a pure permutation of storage, and Flex's attention
/// made the same contiguous copy itself before, so every logit must match
/// exactly, not merely within tolerance.
#[test]
fn attention_ready_cross_storage_leaves_decode_logits_bit_identical() {
    let device = Default::default();
    let model = T5ForConditionalGeneration::<TestBackend>::init(
        tiny_config(T5FeedForwardProjection::Relu, true, 2),
        &device,
    )
    .unwrap();
    let encoder = encoded_source(&model, &device);
    let mut contiguous = model
        .new_decode_state(CacheConfig::new(1, 6).unwrap(), 1, &device)
        .unwrap();
    model
        .prefill_decoder(
            &encoder,
            target(&[0, 3], &device),
            &mut contiguous,
            &ForwardOptions::default(),
        )
        .unwrap();
    let mut strided = model
        .new_decode_state(CacheConfig::new(1, 6).unwrap(), 1, &device)
        .unwrap();
    model
        .prefill_decoder(
            &encoder,
            target(&[0, 3], &device),
            &mut strided,
            &ForwardOptions::default(),
        )
        .unwrap();
    let cross = strided.cross.as_mut().unwrap();
    for layer in 0..2 {
        cross.keys[layer] = strided_cross_heads(&model, &encoder, layer, true);
        cross.values[layer] = strided_cross_heads(&model, &encoder, layer, false);
    }
    for token in [5, 7, 2] {
        let step = |state: &mut T5DecodeState<TestBackend>| {
            model
                .decode(
                    &encoder,
                    DecodeInput {
                        token_ids: target(&[token], &device).token_ids,
                        token_mask: None,
                        position_ids: None,
                    },
                    state,
                    &ForwardOptions::default(),
                )
                .unwrap()
                .decoder
                .logits
                .into_data()
        };
        assert_eq!(step(&mut contiguous), step(&mut strided));
    }
}
