//! Encoder-decoder hook propagation, staged-state validation, supplied encoder
//! outputs, and the architecture-trait entry points of [`ModularSeq2SeqLm`].

use burn::{
    module::Module,
    tensor::{Int, Tensor, Tolerance, backend::BackendTypes},
};

use super::*;
use crate::{
    DecodeState, LogitsSelection,
    experiment::AttentionPolicy,
    models::{
        custom::test_support::{DeterministicParameters, TestBackend, mask, seq2seq_spec},
        shared::test_support::{
            assert_edit_failures_propagate, assert_staged_edit_failures_propagate,
        },
    },
};

/// Device of every fixture.
type Device = <TestBackend as BackendTypes>::Device;

/// The one-layer encoder-decoder fixture with eager attention and deterministic weights.
fn model(device: &Device) -> ModularSeq2SeqLm<TestBackend> {
    ModularSeq2SeqLm::init(seq2seq_spec(), device)
        .unwrap()
        .with_attention_precision(AttentionPrecisionMode::EagerFp32Softmax)
        .map(&mut DeterministicParameters::default())
}

/// A two-row source of three tokens; the second row is right-padded.
fn source(device: &Device) -> CausalLmInput<TestBackend> {
    CausalLmInput {
        token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[1, 2, 3], [4, 5, 0]], device),
        token_mask: Some(mask([[true, true, true], [true, true, false]], device)),
        position_ids: None,
    }
}

/// A two-row target of two tokens.
fn target(device: &Device) -> CausalLmInput<TestBackend> {
    CausalLmInput {
        token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[6, 7], [8, 9]], device),
        token_mask: None,
        position_ids: None,
    }
}

/// Source and target together.
fn input(device: &Device) -> Seq2SeqInput<TestBackend> {
    Seq2SeqInput {
        encoder: source(device),
        decoder: target(device),
    }
}

/// Unwraps a configuration error's field.
fn configuration_field(error: Error) -> String {
    match error {
        Error::Configuration { field, .. } => field,
        other => panic!("expected a configuration error, found {other:?}"),
    }
}

/// A failing edit at any encoder, cross-attention, or decoder point aborts the pass.
#[test]
fn every_point_of_both_stacks_propagates_edit_failures() {
    let device = Default::default();
    let model = model(&device);
    assert_edit_failures_propagate(&model.activation_points(), |hooks| {
        model.forward_with_hooks(input(&device), &ForwardOptions::default(), hooks)
    });
    assert_edit_failures_propagate(&model.encoder_activation_points(), |hooks| {
        model.encode_with_hooks(source(&device), &ForwardOptions::default(), hooks)
    });
}

/// Encoder and decoder staged views propagate edit failures at each of their points.
#[test]
fn staged_stacks_propagate_edit_failures() {
    let device = Default::default();
    let model = model(&device);
    let encoder_points = model.encoder_activation_points();
    assert_staged_edit_failures_propagate(
        &model.encoder_stages(),
        &source(&device),
        &encoder_points,
    );
    let context = model
        .encoder_stages()
        .run_stack(source(&device))
        .unwrap()
        .into_encoder_context()
        .unwrap();
    let decoder_points = model
        .activation_points()
        .into_iter()
        .filter(|point| !encoder_points.contains(point))
        .collect::<Vec<_>>();
    assert_staged_edit_failures_propagate(
        &model.decoder_stages(context),
        &target(&device),
        &decoder_points,
    );
    assert_eq!(model.encoder_layers().len(), 1);
    assert_eq!(model.decoder_layers().len(), 1);
}

/// The trait entry points delegate to the inherent model and bound the decoder capacity.
#[test]
fn architecture_trait_entry_points_match_the_inherent_model() {
    let device = Default::default();
    let model = model(&device);
    let options = ForwardOptions::default();
    let inherent = model.forward(input(&device), &options).unwrap();
    let generic = Seq2SeqLanguageModel::forward(&model, input(&device), &options).unwrap();
    assert_eq!(
        inherent.decoder.logits.into_data(),
        generic.decoder.logits.into_data()
    );
    let encoded = Seq2SeqLanguageModel::encode(&model, source(&device), &options).unwrap();
    assert_eq!(
        encoded.hidden_state.into_data(),
        inherent.encoder.hidden_state.into_data()
    );
    ArchitectureConfig::validate(model.spec()).unwrap();
    let error = model
        .new_decode_state(CacheConfig::new(2, 33).unwrap(), 2, &device)
        .unwrap_err();
    assert!(
        matches!(&error, Error::Configuration { field, reason }
            if field == "custom.seq2seq.decode_state.capacity"
                && reason == "33 exceeds decoder max_position_embeddings 32"),
        "{error:?}"
    );
}

/// Invalid source tokens and logits positions fail with the offending tensor's name.
#[test]
fn invalid_source_tokens_and_logits_positions_are_rejected() {
    let device = Default::default();
    let model = model(&device);
    let mut bad = input(&device);
    bad.encoder.token_ids =
        Tensor::<TestBackend, 2, Int>::from_ints([[1, 2, 32], [4, 5, 0]], &device);
    let error = model.forward(bad, &ForwardOptions::default()).unwrap_err();
    assert!(
        matches!(&error, Error::Shape { tensor, .. } if tensor == "custom.seq2seq.encoder.token_ids"),
        "{error:?}"
    );
    let error = model
        .forward(
            input(&device),
            &ForwardOptions {
                logits: LogitsSelection::Positions(vec![2]),
                ..ForwardOptions::default()
            },
        )
        .unwrap_err();
    assert!(
        matches!(&error, Error::Shape { tensor, .. } if tensor == "forward_options.logits.positions"),
        "{error:?}"
    );
}

/// A supplied encoder output must match the decoder batch, the source width, and its mask grid.
#[test]
fn supplied_encoder_outputs_are_checked_before_decoding() {
    let device = Default::default();
    let model = model(&device);
    let encoded = model
        .encode(source(&device), &ForwardOptions::default())
        .unwrap();
    let mut state = model
        .new_decode_state(CacheConfig::new(2, 4).unwrap(), 2, &device)
        .unwrap();
    let narrow = EncoderOutput {
        hidden_state: encoded.hidden_state.clone().narrow(0, 0, 1),
        token_mask: None,
        ..encoded.clone()
    };
    match model
        .prefill_decoder(
            &narrow,
            target(&device),
            &mut state,
            &ForwardOptions::default(),
        )
        .unwrap_err()
    {
        Error::Shape {
            tensor,
            expected,
            actual,
        } => assert_eq!(
            (tensor.as_str(), expected.as_str(), actual.as_str()),
            (
                "custom.seq2seq.encoder_output",
                "[2, source>0, 8]",
                "[1, 3, 8]"
            )
        ),
        other => panic!("expected an encoder-output shape error, found {other:?}"),
    }
    let misshapen_mask = EncoderOutput {
        token_mask: Some(mask([[true, true], [true, true]], &device)),
        ..encoded
    };
    let error = model
        .prefill_decoder(
            &misshapen_mask,
            target(&device),
            &mut state,
            &ForwardOptions::default(),
        )
        .unwrap_err();
    assert!(
        matches!(&error, Error::Shape { tensor, .. } if tensor == ENCODER_MASK),
        "{error:?}"
    );
    assert_eq!(state.lengths(), [0, 0], "failed prefills commit nothing");
}

/// Staged states are checked for width, override addresses, and source positions.
#[test]
fn staged_states_reject_foreign_widths_overrides_and_source_positions() {
    let device = Default::default();
    let model = model(&device);
    let encoder = model.encoder_stages();
    let state = encoder.embed(source(&device)).unwrap();

    let mut narrow = state.clone();
    narrow.hidden = Tensor::zeros([2, 3, 7], &device);
    match model.run_encoder_layers(narrow, 0..1).unwrap_err() {
        Error::Shape {
            tensor,
            expected,
            actual,
        } => assert_eq!(
            (tensor.as_str(), expected.as_str(), actual.as_str()),
            (
                "execution.hidden",
                "nonempty [batch, token, 8]",
                "[2, 3, 7]"
            )
        ),
        other => panic!("expected a shape error, found {other:?}"),
    }

    for layer in [Layer::encoder(1), Layer::decoder(1)] {
        let mut overridden = state.clone();
        overridden.context = overridden
            .context
            .with_layer_attention(layer, AttentionPolicy::bidirectional());
        assert_eq!(
            configuration_field(model.run_encoder_layers(overridden, 0..1).unwrap_err()),
            "custom.seq2seq.execution.layer_attention"
        );
    }
    // An override addressing an existing block of the other stack is accepted.
    let mut foreign = state.clone();
    foreign.context = foreign
        .context
        .with_layer_attention(Layer::decoder(0), AttentionPolicy::causal());
    model.run_encoder_layers(foreign, 0..1).unwrap();

    let mut context = encoder
        .run_stack(source(&device))
        .unwrap()
        .into_encoder_context()
        .unwrap();
    context.positions = Tensor::<TestBackend, 2, Int>::from_ints([[0, 1, 2], [0, -1, 2]], &device);
    let decoder = model.decoder_stages(context);
    let embedded = decoder.embed(target(&device)).unwrap();
    let error = decoder.run_layers(embedded, 0..1).unwrap_err();
    assert!(
        matches!(&error, Error::Configuration { field, reason }
            if field == "execution.encoder.positions" && reason == "source positions must be nonnegative"),
        "{error:?}"
    );
}

/// Staged embedding of either stack validates token IDs against that stack's vocabulary.
///
/// The decoder vocabulary is shrunk to 16 while the encoder keeps 32, so ID
/// 20 is valid only for the source stack: the encoder embeds it, the decoder
/// rejects it with its own `0..16` limit, and an ID past both vocabularies is
/// reported against the encoder's `0..32` limit.
#[test]
fn staged_embedding_validates_token_ids() {
    let device = Default::default();
    let mut spec = seq2seq_spec();
    spec.decoder.vocabulary_size = 16;
    let model = ModularSeq2SeqLm::<TestBackend>::init(spec, &device)
        .unwrap()
        .map(&mut DeterministicParameters::default());
    let token_error = |error: &Error, limit: &str, observed: &str| {
        matches!(error, Error::Shape { tensor, expected, actual }
            if tensor == "custom.seq2seq.embed.token_ids"
                && *expected == format!("token values in 0..{limit}")
                && *actual == format!("observed range {observed}"))
    };

    let mut beyond_both = source(&device);
    beyond_both.token_ids =
        Tensor::<TestBackend, 2, Int>::from_ints([[1, 2, 40], [4, 5, 0]], &device);
    let error = model.embed_encoder(beyond_both).unwrap_err();
    assert!(token_error(&error, "32", "0..=40"), "{error:?}");

    let mut shared = source(&device);
    shared.token_ids = Tensor::<TestBackend, 2, Int>::from_ints([[1, 2, 20], [4, 5, 0]], &device);
    assert_eq!(
        model.embed_encoder(shared).unwrap().hidden.dims(),
        [2, 3, 8]
    );

    let context = model
        .encoder_stages()
        .run_stack(source(&device))
        .unwrap()
        .into_encoder_context()
        .unwrap();
    let mut beyond_decoder = target(&device);
    beyond_decoder.token_ids = Tensor::<TestBackend, 2, Int>::from_ints([[6, 20], [8, 9]], &device);
    let error = model
        .embed_decoder(beyond_decoder, context.clone())
        .unwrap_err();
    assert!(token_error(&error, "16", "6..=20"), "{error:?}");
    let staged = model.decoder_stages(context);
    assert_eq!(
        staged.embed(target(&device)).unwrap().hidden.dims(),
        [2, 2, 8]
    );
}

/// The encoder-decoder fixture with per-stack embedding scales and a decoder logit cap.
fn readout_model(
    scales: (Option<f64>, Option<f64>),
    final_logit_softcap: Option<f64>,
    device: &Device,
) -> ModularSeq2SeqLm<TestBackend> {
    let mut spec = seq2seq_spec();
    (spec.encoder.embedding_scale, spec.decoder.embedding_scale) = scales;
    spec.decoder.final_logit_softcap = final_logit_softcap;
    ModularSeq2SeqLm::init(spec, device)
        .unwrap()
        .map(&mut DeterministicParameters::default())
}

/// Options requesting every logit and every retained hidden state.
fn retained() -> ForwardOptions {
    ForwardOptions {
        logits: LogitsSelection::All,
        hidden_states: true,
        ..ForwardOptions::default()
    }
}

/// Each stack scales its own embeddings, and the decoder cap applies to every readout path.
///
/// The source and target embedding states (stateless and staged) are their
/// stacks' scales times the unscaled ones; capped logits equal
/// `c · tanh(z / c)` of the uncapped twin's, in the stateless forward and in
/// a cached prefill.
#[test]
fn stack_embedding_scales_and_the_decoder_cap_apply_independently() {
    let device = Default::default();
    let plain = readout_model((None, None), None, &device);
    let scaled = readout_model((Some(2.0), Some(0.5)), None, &device);
    let reference = plain.forward(input(&device), &retained()).unwrap();
    let output = scaled.forward(input(&device), &retained()).unwrap();
    output.encoder.hidden_states[0]
        .clone()
        .into_data()
        .assert_approx_eq::<f32>(
            &(reference.encoder.hidden_states[0].clone() * 2.0).into_data(),
            Tolerance::absolute(0.0),
        );
    output.decoder.hidden_states[0]
        .clone()
        .into_data()
        .assert_approx_eq::<f32>(
            &(reference.decoder.hidden_states[0].clone() * 0.5).into_data(),
            Tolerance::absolute(0.0),
        );
    let staged = scaled.embed_encoder(source(&device)).unwrap();
    staged.hidden.into_data().assert_approx_eq::<f32>(
        &(reference.encoder.hidden_states[0].clone() * 2.0).into_data(),
        Tolerance::absolute(0.0),
    );

    let cap = 0.05;
    let capped = readout_model((None, None), Some(cap), &device);
    let expected = reference
        .decoder
        .logits
        .clone()
        .div_scalar(cap)
        .tanh()
        .mul_scalar(cap);
    capped
        .forward(input(&device), &retained())
        .unwrap()
        .decoder
        .logits
        .into_data()
        .assert_approx_eq::<f32>(&expected.clone().into_data(), Tolerance::absolute(1e-7));
    let encoded = capped
        .encode(source(&device), &ForwardOptions::default())
        .unwrap();
    let mut state = capped
        .new_decode_state(CacheConfig::new(2, 4).unwrap(), 2, &device)
        .unwrap();
    let prefill = capped
        .prefill_decoder(
            &encoded,
            target(&device),
            &mut state,
            &ForwardOptions::default(),
        )
        .unwrap();
    prefill.decoder.logits.into_data().assert_approx_eq::<f32>(
        &expected.slice([0..2, 1..2]).into_data(),
        Tolerance::absolute(1e-6),
    );
}

/// An encoder stack has no vocabulary readout, so a cap there is rejected before allocation.
#[test]
fn an_encoder_logit_cap_is_rejected() {
    let device: Device = Default::default();
    let mut spec = seq2seq_spec();
    spec.encoder.final_logit_softcap = Some(30.0);
    let error = ModularSeq2SeqLm::<TestBackend>::init(spec, &device).unwrap_err();
    assert_eq!(configuration_field(error), "encoder.final_logit_softcap");
}
