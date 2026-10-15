//! Batched stateful decoding against the previous per-row algorithm and stateless passes.
//!
//! [`reference_call`] is the per-row implementation that preceded batched
//! decoding, kept verbatim in spirit as a test oracle: every row is compacted
//! to its valid tokens, run alone through every layer with its cached history
//! concatenated explicitly, an explicit `k > L + q` causal mask, a per-row
//! cross-attention slice, and a per-row contiguous bias; the appends are then
//! published row by row. The batched path must agree with it on logits,
//! hidden states, lengths, and cache contents for ragged, gapped, inactive,
//! truncated, and reordered rows.

use std::time::Instant;

use burn::{
    backend::Flex,
    tensor::{Bool, Int, Tensor, TensorData, Tolerance},
};
use proptest::prelude::*;

use super::*;
use crate::{
    LogitsSelection,
    attention::{AttentionProjection, padding_attention_mask},
    models::shared::{rows::concat_rows, select_hidden_with_last_columns},
    models::t5::{T5FeedForwardProjection, T5ForConditionalGeneration},
};

/// Deterministic CPU backend for parity tests.
type TestBackend = Flex<f32, i32>;

/// Builds a small topology; `layers` decoder and encoder blocks.
fn config(projection: T5FeedForwardProjection, tied: bool, layers: usize) -> T5Config {
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

/// Bottom-right causal exclusion `key > baseline + query` for one row, `[1, heads, q, k]`.
fn causal_mask(baseline: usize, queries: usize, heads: usize) -> Tensor<TestBackend, 4, Bool> {
    let device = Default::default();
    let keys = baseline + queries;
    let query = Tensor::<TestBackend, 1, Int>::arange(baseline as i64..keys as i64, &device)
        .reshape([1, 1, queries, 1])
        .expand([1, heads, queries, keys]);
    let key = Tensor::<TestBackend, 1, Int>::arange(0..keys as i64, &device)
        .reshape([1, 1, 1, keys])
        .expand([1, heads, queries, keys]);
    key.greater(query)
}

/// Runs one compact row through every decoder layer as the per-row algorithm did.
#[allow(clippy::too_many_arguments)]
fn reference_row(
    model: &T5ForConditionalGeneration<TestBackend>,
    tokens: Tensor<TestBackend, 2, Int>,
    row: usize,
    baseline: usize,
    state: &TensorDecodeState<TestBackend>,
    cross: &T5CrossAttentionCache<TestBackend>,
    step: &mut TensorStepWrites<TestBackend>,
    retain: bool,
) -> Result<(Tensor<TestBackend, 3>, Vec<Tensor<TestBackend, 3>>)> {
    let device = tokens.device();
    let queries = tokens.dims()[1];
    let bias =
        model
            .decoder
            .relative_table()?
            .forward(queries, baseline + queries, baseline, &device)?;
    let mut hidden = model.shared.forward(tokens);
    let mut states = Vec::new();
    if retain {
        states.push(hidden.clone());
    }
    for (layer, block) in model.decoder.block.iter().enumerate() {
        let residual = hidden.clone();
        let attention = &block.self_attention.self_attention.attention;
        let mut projection =
            attention.project_self(block.self_attention.layer_norm.forward(hidden))?;
        let (new_keys, new_values) = (projection.key.clone(), projection.value.clone());
        let history = state
            .main()
            .native_kv(SELF_KV_GROUP)?
            .read_sequence(layer, row, baseline)?;
        if let Some((keys, values)) = history.materialize() {
            projection.key = Tensor::cat(vec![keys, projection.key], 2);
            projection.value = Tensor::cat(vec![values, projection.value], 2);
        }
        let mask = causal_mask(baseline, queries, attention.query_heads);
        hidden = residual + attention.attend(projection, Some(mask), Some(bias.clone()), false)?;
        step.push_kv_row(SELF_KV_GROUP, layer, row, new_keys, new_values);

        let residual = hidden.clone();
        let cross_layer = block.cross_attention.as_ref().unwrap();
        let cross_attention = &cross_layer.enc_dec_attention.attention;
        let normalized = cross_layer.layer_norm.forward(hidden);
        let [_, _, width] = normalized.dims();
        let query = crate::operations::linear(normalized, &cross_attention.q_proj)?
            .reshape([
                1,
                queries,
                cross_attention.query_heads,
                cross_attention.head_dim,
            ])
            .swap_dims(1, 2);
        let source = cross.source_sequence();
        let row_slice = |tensor: &Tensor<TestBackend, 4>| tensor.clone().narrow(0, row, 1);
        let source_mask = cross
            .encoder
            .token_mask
            .as_ref()
            .map(|mask| TokenMask::new(mask.as_tensor().clone().narrow(0, row, 1)));
        let cross_mask = padding_attention_mask(
            None,
            source_mask.as_ref(),
            1,
            cross_attention.query_heads,
            queries,
            source,
        )?;
        hidden = residual
            + cross_attention.attend(
                AttentionProjection {
                    query,
                    key: row_slice(&cross.keys[layer]),
                    value: row_slice(&cross.values[layer]),
                    gate: None,
                },
                cross_mask,
                None,
                false,
            )?;
        debug_assert_eq!(width, model.config.d_model);

        let residual = hidden.clone();
        let normalized = block.feed_forward.layer_norm.forward(hidden);
        let projected = match (&block.feed_forward.dense, &block.feed_forward.gated) {
            (Some(dense), None) => dense.forward(normalized)?,
            (None, Some(gated)) => gated.forward(normalized)?,
            _ => unreachable!("tiny fixtures select one topology"),
        };
        hidden = residual + projected;
        if retain {
            states.push(hidden.clone());
        }
    }
    hidden = model.decoder.final_layer_norm.forward(hidden);
    if retain {
        states.push(hidden.clone());
    }
    Ok((hidden, states))
}

/// The per-row stateful algorithm that preceded batched decoding (test oracle).
fn reference_call(
    model: &T5ForConditionalGeneration<TestBackend>,
    encoder: &EncoderOutput<TestBackend>,
    decoder: CausalLmInput<TestBackend>,
    state: &mut T5DecodeState<TestBackend>,
    options: &ForwardOptions,
    phase: StatefulPhase,
) -> Result<Seq2SeqOutput<TestBackend>> {
    let prepared = prepare_stateful_input(model, encoder, &decoder, state, phase)?;
    let pending_cross = state
        .cross
        .is_none()
        .then(|| T5CrossAttentionCache::project(model, encoder))
        .transpose()?;
    let cross = pending_cross.as_ref().or(state.cross.as_ref()).unwrap();
    let device = decoder.token_ids.device();
    let mut step = state.tensors.step(&prepared.token_counts)?;
    let mut rows = Vec::new();
    let mut diagnostics: Vec<Vec<Tensor<TestBackend, 3>>> =
        vec![Vec::new(); model.decoder.block.len() + 2];
    for row in 0..prepared.batch {
        let Some(plan) = prepared.compact_row::<TestBackend>(row, &device)? else {
            let empty = Tensor::zeros([1, prepared.sequence, model.config.d_model], &device);
            rows.push(empty.clone());
            for destination in &mut diagnostics {
                destination.push(empty.clone());
            }
            continue;
        };
        let baseline = state.tensors.lengths()[row];
        let (hidden, states) = reference_row(
            model,
            plan.compact(&decoder.token_ids),
            row,
            baseline,
            &state.tensors,
            cross,
            &mut step,
            options.hidden_states,
        )?;
        rows.push(plan.scatter(hidden)?);
        for (destination, compact) in diagnostics.iter_mut().zip(states) {
            destination.push(plan.scatter(compact)?);
        }
    }
    let hidden = concat_rows(rows, "reference.hidden")?;
    let hidden_states = if options.hidden_states {
        diagnostics
            .into_iter()
            .map(|rows| concat_rows(rows, "reference.hidden_states"))
            .collect::<Result<Vec<_>>>()?
    } else {
        Vec::new()
    };
    let selected =
        select_hidden_with_last_columns(hidden.clone(), prepared.last_columns(), &options.logits)?;
    let logits = model.project_logits(selected)?;
    let bound_encoder = cross.encoder.clone();
    state.tensors.publish_step(step)?;
    if let Some(cross) = pending_cross {
        state.cross = Some(cross);
    }
    Ok(Seq2SeqOutput {
        decoder: CausalLmOutput {
            logits,
            final_hidden_state: options.hidden_states.then_some(hidden),
            hidden_states,
            attentions: Vec::new(),
            routers: Vec::new(),
            load_balancing_loss: None,
        },
        encoder: bound_encoder,
        cross_attentions: Vec::new(),
    })
}

/// Builds a `[rows, columns]` target from IDs and an optional validity mask.
fn input(ids: &[&[i64]], mask: Option<&[&[bool]]>) -> CausalLmInput<TestBackend> {
    let device = Default::default();
    let (rows, columns) = (ids.len(), ids[0].len());
    CausalLmInput {
        token_ids: Tensor::from_data(TensorData::new(ids.concat(), [rows, columns]), &device),
        token_mask: mask.map(|mask| {
            TokenMask::new(Tensor::from_data(
                TensorData::new(mask.concat(), [rows, columns]),
                &device,
            ))
        }),
        position_ids: None,
    }
}

/// Every committed slot `[0, L_b)` of every layer and row, as host values.
fn committed(state: &T5DecodeState<TestBackend>) -> Vec<Vec<f32>> {
    let cache = state.tensors.main().native_kv(SELF_KV_GROUP).unwrap();
    let mut values = Vec::new();
    for layer in 0..state.self_spec.layers() {
        for row in 0..state.tensors.batch_size() {
            if let Some((keys, slots)) = cache
                .read_sequence(layer, row, state.lengths()[row])
                .unwrap()
                .materialize()
            {
                values.push(keys.into_data().to_vec::<f32>().unwrap());
                values.push(slots.into_data().to_vec::<f32>().unwrap());
            }
        }
    }
    values
}

/// Asserts two output sets agree within the F32 parity tolerance.
fn assert_outputs_match(
    actual: &Seq2SeqOutput<TestBackend>,
    expected: &Seq2SeqOutput<TestBackend>,
    label: &str,
) {
    let tolerance = Tolerance::absolute(2e-5);
    actual
        .decoder
        .logits
        .clone()
        .into_data()
        .assert_approx_eq::<f32>(&expected.decoder.logits.clone().into_data(), tolerance);
    assert_eq!(
        actual.decoder.hidden_states.len(),
        expected.decoder.hidden_states.len(),
        "{label}: diagnostic count"
    );
    for (left, right) in actual
        .decoder
        .hidden_states
        .iter()
        .zip(&expected.decoder.hidden_states)
    {
        left.clone()
            .into_data()
            .assert_approx_eq::<f32>(&right.clone().into_data(), tolerance);
    }
}

/// Asserts both states hold the same lengths and committed cache contents.
fn assert_states_match(
    actual: &T5DecodeState<TestBackend>,
    expected: &T5DecodeState<TestBackend>,
    label: &str,
) {
    assert_eq!(actual.lengths(), expected.lengths(), "{label}: lengths");
    for (left, right) in committed(actual).iter().zip(committed(expected).iter()) {
        assert_eq!(left.len(), right.len(), "{label}: committed extent");
        for (a, b) in left.iter().zip(right) {
            assert!((a - b).abs() <= 2e-5, "{label}: cache {a} vs {b}");
        }
    }
}

/// One stateful call applied identically to the batched and the reference state.
enum Call {
    /// Initial prompt (binds the encoder).
    Prefill(CausalLmInput<TestBackend>),
    /// Continuation segment.
    Decode(CausalLmInput<TestBackend>),
    /// Rollback to the given lengths.
    Truncate(Vec<usize>),
    /// Beam-style row reorder.
    Reorder(Vec<usize>),
}

/// Runs a call script on two states and compares every output and state.
fn assert_script_matches_reference(
    model: &T5ForConditionalGeneration<TestBackend>,
    encoder: &EncoderOutput<TestBackend>,
    batch: usize,
    calls: Vec<Call>,
) {
    let device = Default::default();
    let options = ForwardOptions {
        logits: LogitsSelection::All,
        hidden_states: true,
        ..ForwardOptions::default()
    };
    let mut batched = model
        .new_decode_state(CacheConfig::new(batch, 12).unwrap(), batch, &device)
        .unwrap();
    let mut reference = model
        .new_decode_state(CacheConfig::new(batch, 12).unwrap(), batch, &device)
        .unwrap();
    for (index, call) in calls.into_iter().enumerate() {
        let label = format!("call {index}");
        match call {
            Call::Prefill(target) => {
                let actual = model
                    .prefill_decoder(encoder, target.clone(), &mut batched, &options)
                    .unwrap();
                let expected = reference_call(
                    model,
                    encoder,
                    target,
                    &mut reference,
                    &options,
                    StatefulPhase::Prefill,
                )
                .unwrap();
                assert_outputs_match(&actual, &expected, &label);
            }
            Call::Decode(target) => {
                let actual = model
                    .decode(
                        encoder,
                        DecodeInput {
                            token_ids: target.token_ids.clone(),
                            token_mask: target.token_mask.clone(),
                            position_ids: None,
                        },
                        &mut batched,
                        &options,
                    )
                    .unwrap();
                let expected = reference_call(
                    model,
                    encoder,
                    target,
                    &mut reference,
                    &options,
                    StatefulPhase::Decode,
                )
                .unwrap();
                assert_outputs_match(&actual, &expected, &label);
            }
            Call::Truncate(lengths) => {
                batched.truncate(&lengths).unwrap();
                reference.truncate(&lengths).unwrap();
            }
            Call::Reorder(indices) => {
                batched.reorder(&indices).unwrap();
                reference.reorder(&indices).unwrap();
            }
        }
        assert_states_match(&batched, &reference, &label);
    }
}

/// Encodes three source rows, the second with gapped padding.
fn three_sources(model: &T5ForConditionalGeneration<TestBackend>) -> EncoderOutput<TestBackend> {
    model
        .encode(
            input(
                &[&[8, 9, 10, 11], &[12, 13, 0, 14], &[15, 16, 17, 18]],
                Some(&[
                    &[true, true, true, true],
                    &[true, true, false, true],
                    &[true, true, true, true],
                ]),
            ),
            &ForwardOptions::default(),
        )
        .unwrap()
}

/// Ragged, gapped, inactive, truncated, and reordered rows match the per-row algorithm.
#[test]
fn batched_calls_match_the_per_row_reference() {
    let device = Default::default();
    for (projection, tied) in [
        (T5FeedForwardProjection::Relu, true),
        (T5FeedForwardProjection::GatedGelu, false),
    ] {
        let model =
            T5ForConditionalGeneration::<TestBackend>::init(config(projection, tied, 2), &device)
                .unwrap();
        let encoder = three_sources(&model);
        assert_script_matches_reference(
            &model,
            &encoder,
            3,
            vec![
                // Counts 3, 2 (left padding), 2 (interior padding).
                Call::Prefill(input(
                    &[&[0, 3, 4], &[99, 0, 5], &[0, 77, 6]],
                    Some(&[
                        &[true, true, true],
                        &[false, true, true],
                        &[true, false, true],
                    ]),
                )),
                // Ragged baselines (3, 2, 2) with an inactive middle row.
                Call::Decode(input(
                    &[&[7, 0], &[-1, -1], &[8, 9]],
                    Some(&[&[true, false], &[false, false], &[true, true]]),
                )),
                // Unmasked single-token step at lengths (4, 2, 4).
                Call::Decode(input(&[&[10], &[11], &[12]], None)),
                Call::Truncate(vec![2, 3, 1]),
                Call::Decode(input(
                    &[&[13, 14], &[15, 16], &[17, 18]],
                    Some(&[&[false, true], &[true, true], &[true, false]]),
                )),
                Call::Reorder(vec![2, 2, 0]),
                Call::Decode(input(&[&[19], &[20], &[21]], None)),
            ],
        );
    }
}

/// Beam-style steps (one token per active row, finished rows inactive) match the reference.
#[test]
fn beam_style_steps_match_the_per_row_reference() {
    let device = Default::default();
    let model = T5ForConditionalGeneration::<TestBackend>::init(
        config(T5FeedForwardProjection::GatedGelu, true, 3),
        &device,
    )
    .unwrap();
    let encoder = three_sources(&model);
    let mut calls = vec![Call::Prefill(input(&[&[0], &[0], &[0]], None))];
    for step in 0..5_i64 {
        let finished = step >= 3;
        calls.push(Call::Reorder(if step % 2 == 0 {
            vec![0, 0, 2]
        } else {
            vec![1, 0, 2]
        }));
        calls.push(Call::Decode(input(
            &[&[3 + step], &[4 + step], &[5 + step]],
            Some(&[&[true], &[true], &[!finished]]),
        )));
    }
    assert_script_matches_reference(&model, &encoder, 3, calls);
}

/// A later-layer failure publishes nothing; the next call equals a clean reference run.
#[test]
fn failed_batched_call_changes_no_logical_state() {
    let device = Default::default();
    let model = T5ForConditionalGeneration::<TestBackend>::init(
        config(T5FeedForwardProjection::Relu, true, 2),
        &device,
    )
    .unwrap();
    let encoder = three_sources(&model);
    let options = ForwardOptions {
        logits: LogitsSelection::All,
        ..ForwardOptions::default()
    };
    let mut state = model
        .new_decode_state(CacheConfig::new(3, 8).unwrap(), 3, &device)
        .unwrap();
    let mut reference = model
        .new_decode_state(CacheConfig::new(3, 8).unwrap(), 3, &device)
        .unwrap();
    let prompt = input(&[&[0, 3], &[0, 4], &[0, 5]], None);
    model
        .prefill_decoder(&encoder, prompt.clone(), &mut state, &options)
        .unwrap();
    model
        .prefill_decoder(&encoder, prompt, &mut reference, &options)
        .unwrap();
    let before = committed(&state);
    let mut broken = model.clone();
    broken.decoder.block[1].cross_attention = None;
    let step = input(&[&[6], &[7], &[8]], None);
    let decode = |model: &T5ForConditionalGeneration<TestBackend>,
                  state: &mut T5DecodeState<TestBackend>| {
        model.decode(
            &encoder,
            DecodeInput {
                token_ids: step.token_ids.clone(),
                token_mask: None,
                position_ids: None,
            },
            state,
            &options,
        )
    };
    let error = decode(&broken, &mut state).unwrap_err();
    assert!(error.to_string().contains("cross_attention"), "{error}");
    assert_eq!(state.lengths(), [2, 2, 2]);
    assert_eq!(committed(&state), before);
    // The abandoned write-ahead slots are unreachable and overwritten next time.
    let actual = decode(&model, &mut state).unwrap();
    let expected = decode(&model, &mut reference).unwrap();
    assert_eq!(
        actual.decoder.logits.into_data(),
        expected.decoder.logits.into_data()
    );
    assert_states_match(&state, &reference, "after failure");
}

/// Group-preserving reorders keep the cross binding in place and match a full reorder.
#[test]
fn group_preserving_reorder_skips_the_cross_gather() {
    let device = Default::default();
    let model = T5ForConditionalGeneration::<TestBackend>::init(
        config(T5FeedForwardProjection::Relu, true, 1),
        &device,
    )
    .unwrap();
    // Two sources with two beams each; beams of one source share encoder rows.
    let source = model
        .encode(
            input(&[&[8, 9], &[10, 11]], None),
            &ForwardOptions::default(),
        )
        .unwrap();
    let expand = Tensor::<TestBackend, 1, Int>::from_ints([0, 0, 1, 1], &device);
    let encoder = EncoderOutput {
        hidden_state: source.hidden_state.select(0, expand),
        token_mask: None,
        hidden_states: Vec::new(),
        attentions: Vec::new(),
    };
    let prompt = input(&[&[0, 3], &[0, 4], &[0, 5], &[0, 6]], None);
    let mut grouped = model
        .new_decode_state(CacheConfig::new(4, 6).unwrap(), 4, &device)
        .unwrap();
    let mut full = model
        .new_decode_state(CacheConfig::new(4, 6).unwrap(), 4, &device)
        .unwrap();
    for state in [&mut grouped, &mut full] {
        model
            .prefill_decoder(&encoder, prompt.clone(), state, &ForwardOptions::default())
            .unwrap();
    }
    let cross_before = grouped.cross.as_ref().unwrap().keys[0].clone().into_data();
    grouped.reorder_within_groups(&[1, 1, 3, 2], 2).unwrap();
    full.reorder(&[1, 1, 3, 2]).unwrap();
    assert_eq!(
        grouped.cross.as_ref().unwrap().keys[0].clone().into_data(),
        cross_before,
        "the cross binding is not gathered"
    );
    assert_eq!(committed(&grouped), committed(&full));
    let step = input(&[&[7], &[8], &[9], &[10]], None);
    let decode = |state: &mut T5DecodeState<TestBackend>| {
        model
            .decode(
                &encoder,
                DecodeInput {
                    token_ids: step.token_ids.clone(),
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
    assert_eq!(decode(&mut grouped), decode(&mut full));
    // Leaving a group, or a group size that does not divide the batch, changes nothing.
    let lengths = grouped.lengths().to_vec();
    for (indices, group) in [
        (vec![2, 0, 3, 2], 2),
        (vec![0, 1, 2, 3], 3),
        (vec![0, 1, 2, 3], 0),
    ] {
        assert!(grouped.reorder_within_groups(&indices, group).is_err());
    }
    assert_eq!(grouped.lengths(), lengths);
}

/// Slot geometry: masks see exactly the causal history and uniform lengths share one bias.
#[test]
fn step_geometry_positions_and_shared_lengths() {
    let geometry = StepGeometry::new(&[4, 2, 4], &[2, 0, 1], 2).unwrap();
    assert_eq!(geometry.mask_positions, [4, 5, 1, 1, 4, 4]);
    assert_eq!(geometry.bias_positions, [4, 5, 2, 3, 4, 5]);
    // The inactive row's shorter history does not break the shared active length.
    assert_eq!(geometry.shared_length, Some(4));
    assert_eq!(geometry.max_position, 5);
    let ragged = StepGeometry::new(&[3, 1], &[1, 1], 1).unwrap();
    assert_eq!(ragged.shared_length, None);
    let empty_history = StepGeometry::new(&[0], &[0], 1).unwrap();
    assert_eq!(empty_history.mask_positions, [-1]);
}

proptest! {
    /// The exclusion plane equals `key > visible position` for every slot.
    #[test]
    fn step_exclusions_follow_the_causal_predicate(
        rows in proptest::collection::vec((0_usize..6, 0_usize..4), 1..4),
    ) {
        let lengths: Vec<usize> = rows.iter().map(|row| row.0).collect();
        let counts: Vec<usize> = rows.iter().map(|row| row.1).collect();
        let block = counts.iter().copied().max().unwrap_or(0).max(1);
        let key_slots = lengths
            .iter()
            .zip(&counts)
            .map(|(length, count)| length + count)
            .max()
            .unwrap()
            .max(1);
        let geometry = StepGeometry::new(&lengths, &counts, block).unwrap();
        let plane = geometry
            .exclusions::<TestBackend>(key_slots, &Default::default())
            .unwrap()
            .into_data()
            .to_vec::<bool>()
            .unwrap();
        for (row, (&length, &count)) in lengths.iter().zip(&counts).enumerate() {
            for slot in 0..block {
                for key in 0..key_slots {
                    let visible = if count == 0 {
                        length as i64 - 1
                    } else {
                        (length + slot.min(count - 1)) as i64
                    };
                    let excluded = plane[(row * block + slot) * key_slots + key];
                    prop_assert_eq!(excluded, key as i64 > visible);
                    // A valid query never sees a slot beyond its row's appended history.
                    if slot < count && key >= length + count {
                        prop_assert!(excluded);
                    }
                }
            }
        }
    }
}

/// Measures batched against per-row decoding on Flex (release builds only).
///
/// Run with `cargo test --release --lib batched_decode_benchmark -- --ignored --nocapture`.
#[test]
#[ignore = "timing benchmark; run explicitly in release mode"]
fn batched_decode_benchmark() {
    let device = Default::default();
    let model = T5ForConditionalGeneration::<TestBackend>::init(
        T5Config {
            vocab_size: 512,
            d_model: 256,
            d_kv: 64,
            d_ff: 1024,
            num_layers: 4,
            num_decoder_layers: Some(4),
            num_heads: 4,
            relative_attention_num_buckets: 32,
            relative_attention_max_distance: 128,
            feed_forward_proj: T5FeedForwardProjection::GatedGelu,
            tie_word_embeddings: false,
            ..T5Config::default()
        },
        &device,
    )
    .unwrap();
    for batch in [1_usize, 4, 8, 16] {
        let source_ids: Vec<i64> = (0..batch * 32).map(|index| (index % 500) as i64).collect();
        let encoder = model
            .encode(
                CausalLmInput {
                    token_ids: Tensor::from_data(TensorData::new(source_ids, [batch, 32]), &device),
                    token_mask: None,
                    position_ids: None,
                },
                &ForwardOptions::default(),
            )
            .unwrap();
        let options = ForwardOptions {
            logits: LogitsSelection::Last,
            ..ForwardOptions::default()
        };
        let steps = 32;
        // Best of three alternating runs per path; the first pair also warms caches.
        let mut timings = [f64::INFINITY; 2];
        for (round, batched) in [true, false, true, false, true, false]
            .into_iter()
            .enumerate()
        {
            let mut state = model
                .new_decode_state(CacheConfig::new(batch, steps + 1).unwrap(), batch, &device)
                .unwrap();
            let prompt = CausalLmInput {
                token_ids: Tensor::zeros([batch, 1], &device),
                token_mask: None,
                position_ids: None,
            };
            let started = Instant::now();
            let mut call = |input: CausalLmInput<TestBackend>, phase| {
                let output = if batched {
                    match phase {
                        StatefulPhase::Prefill => {
                            model.prefill_decoder(&encoder, input, &mut state, &options)
                        }
                        StatefulPhase::Decode => model.decode(
                            &encoder,
                            DecodeInput {
                                token_ids: input.token_ids,
                                token_mask: input.token_mask,
                                position_ids: None,
                            },
                            &mut state,
                            &options,
                        ),
                    }
                } else {
                    reference_call(&model, &encoder, input, &mut state, &options, phase)
                };
                // Read the logits back, as generation does, so the work completes.
                output.unwrap().decoder.logits.into_data();
            };
            call(prompt, StatefulPhase::Prefill);
            for step in 0..steps {
                call(
                    CausalLmInput {
                        token_ids: Tensor::from_data(
                            TensorData::new(vec![(step % 500) as i64; batch], [batch, 1]),
                            &device,
                        ),
                        token_mask: None,
                        position_ids: None,
                    },
                    StatefulPhase::Decode,
                );
            }
            let slot = round % 2;
            timings[slot] = timings[slot].min(started.elapsed().as_secs_f64());
        }
        eprintln!(
            "T5 decode batch={batch} steps={steps}: batched={:.4}s per-row={:.4}s speedup={:.2}x",
            timings[0],
            timings[1],
            timings[1] / timings[0]
        );
    }
}
