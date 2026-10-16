//! Equivalence of batched cached execution with the former per-row algorithm.
//!
//! [`per_row_forward`] is the pre-batching implementation kept as an oracle:
//! every row is compacted and evaluated on its own `[1, n_b]` slice, full
//! attention concatenates that row's history with its new keys and applies an
//! explicit `k > W_b + q` mask, DeltaNet runs on the row's own state, and the
//! results are scattered back. The tests drive the oracle and the batched
//! path through identical prefill/decode schedules with ragged, left-, right-,
//! and interior-padded rows and finished rows, and compare outputs, hidden
//! states, and every row's readable cache contents.

use burn::tensor::{
    Bool, Int, Tensor, TensorData, Tolerance,
    backend::{Backend, BackendTypes},
};

use super::*;
use crate::{
    DecodeInput, DecodeState, LogitsSelection,
    architecture::TokenMask,
    models::qwen3_5::{Qwen3_5LayerType, Qwen3_5TextConfig},
    models::shared::rows::concat_rows,
    position::{apply_rotary, rotary_angles},
};

/// CPU Flex backend with F32 activations and portable I32 indices.
type TestBackend = burn::backend::Flex<f32, i32>;

/// Runs one stateful call with the former per-row algorithm (no transaction).
fn per_row_forward<B: Backend>(
    model: &Qwen3_5ForCausalLm<B>,
    input: CausalLmInput<B>,
    state: &mut Qwen3_5DecodeState<B>,
    options: &ForwardOptions,
    phase: StatefulPhase,
) -> Result<CausalLmOutput<B>> {
    assert!(
        !state.tensors.has_active_transaction(),
        "oracle is non-transactional"
    );
    state.validate_for_model(model)?;
    let prepared = prepare_stateful_input(model, &input, state, phase)?;
    validate_logits_selection(&options.logits, prepared.sequence)?;
    let device = input.token_ids.device();
    let mut writes = state.tensors.step(&prepared.token_counts)?;
    let mut final_rows = Vec::new();
    let mut diagnostic_rows: Vec<Vec<Tensor<B, 3>>> = if options.hidden_states {
        (0..model.model.layers.len() + 2)
            .map(|_| Vec::new())
            .collect()
    } else {
        Vec::new()
    };
    for row in 0..prepared.batch {
        let Some(plan) = prepared.compact_row::<B>(row, &device)? else {
            let empty = Tensor::zeros(
                [1, prepared.sequence, model.config().hidden_size],
                (&device, model.model.embed_tokens.weight.val().dtype()),
            );
            final_rows.push(empty.clone());
            for destination in &mut diagnostic_rows {
                destination.push(empty.clone());
            }
            continue;
        };
        let baseline = state.tensors.lengths()[row];
        let positions = match &input.position_ids {
            Some(positions) => plan.compact(positions),
            None => implicit_position_ids(baseline, plan.count(), &device, "positions")?,
        };
        let rotary = rotary_angles(
            model.model.rope,
            positions,
            prepared.evaluated_contexts[row],
        )?;
        let mut hidden = model
            .model
            .embed_tokens
            .forward(plan.compact(&input.token_ids));
        let mut diagnostics = vec![hidden.clone()];
        for (layer, &slot) in model.model.layers.iter().zip(&state.layout.schedule) {
            let residual = hidden.clone();
            let normalized = layer.input_layernorm.forward(hidden);
            let mixed = match slot {
                Qwen3_5LayerCache::FullAttention(cache_layer) => {
                    let attention = layer.self_attn.as_ref().unwrap();
                    let mut projection = attention.project_self(normalized)?;
                    (projection.query, projection.key) =
                        apply_rotary(projection.query, projection.key, &rotary)?;
                    writes.push_kv_row(
                        NATIVE_KV_GROUP,
                        cache_layer,
                        row,
                        projection.key.clone(),
                        projection.value.clone(),
                    );
                    let history = state
                        .tensors
                        .main()
                        .native_kv(NATIVE_KV_GROUP)?
                        .read_sequence(cache_layer, row, baseline)?;
                    if let Some((keys, values)) = history.materialize() {
                        projection.key = Tensor::cat(vec![keys, projection.key], 2);
                        projection.value = Tensor::cat(vec![values, projection.value], 2);
                    }
                    let queries = projection.query.dims()[2];
                    let mask =
                        cached_causal_mask::<B>(baseline, queries, attention.query_heads, &device);
                    attention.attend(projection, Some(mask), None, false)?
                }
                Qwen3_5LayerCache::DeltaNet(cache_layer) => {
                    let cached = state
                        .tensors
                        .main()
                        .recurrent(RECURRENT_GROUP)?
                        .read_sequence(cache_layer, row)?;
                    let mut recurrent = GatedDeltaNetState {
                        convolution: cached.convolution().clone(),
                        recurrent: cached.matrix().clone(),
                    };
                    let mixed = layer.linear_attn.as_ref().unwrap().forward_with_state(
                        normalized,
                        None,
                        &mut recurrent,
                        phase.delta_mode(),
                    )?;
                    writes.push_recurrent_row(
                        RECURRENT_GROUP,
                        cache_layer,
                        row,
                        recurrent.recurrent,
                        recurrent.convolution,
                    );
                    mixed
                }
            };
            hidden = finish_residual(layer, residual, mixed)?;
            diagnostics.push(hidden.clone());
        }
        hidden = model.model.norm.forward(hidden);
        diagnostics.push(hidden.clone());
        final_rows.push(plan.scatter(hidden)?);
        for (destination, diagnostic) in diagnostic_rows.iter_mut().zip(diagnostics) {
            destination.push(plan.scatter(diagnostic)?);
        }
    }
    let hidden = concat_rows(final_rows, "qwen.hidden")?;
    let hidden_states = diagnostic_rows
        .into_iter()
        .map(|rows| concat_rows(rows, "qwen.hidden_states"))
        .collect::<Result<Vec<_>>>()?;
    let selected =
        select_hidden_with_last_columns(hidden.clone(), prepared.last_columns(), &options.logits)?;
    let logits = model.project_hidden(selected)?;
    state.tensors.publish_step(writes)?;
    Ok(CausalLmOutput {
        logits,
        final_hidden_state: options.hidden_states.then_some(hidden),
        hidden_states,
        attentions: Vec::new(),
        routers: Vec::new(),
        load_balancing_loss: None,
    })
}

/// The former explicit cached mask `excluded(q, k) = k > W + q` as `[1, heads, Q, W + Q]`.
fn cached_causal_mask<B: Backend>(
    baseline: usize,
    queries: usize,
    heads: usize,
    device: &B::Device,
) -> Tensor<B, 4, Bool> {
    let keys = baseline + queries;
    let query = Tensor::<B, 1, Int>::arange(baseline as i64..keys as i64, device)
        .reshape([1, 1, queries, 1])
        .expand([1, heads, queries, keys]);
    let key = Tensor::<B, 1, Int>::arange(0..keys as i64, device)
        .reshape([1, 1, 1, keys])
        .expand([1, heads, queries, keys]);
    key.greater(query)
}

/// Builds a hybrid model with two full-attention layers and every cache family.
fn hybrid_model(device: &<TestBackend as BackendTypes>::Device) -> Qwen3_5ForCausalLm<TestBackend> {
    TestBackend::seed(device, 401);
    Qwen3_5ForCausalLm::init(
        Qwen3_5TextConfig {
            vocab_size: 32,
            hidden_size: 8,
            intermediate_size: 16,
            num_hidden_layers: 4,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 4,
            max_position_embeddings: 96,
            layer_types: vec![
                Qwen3_5LayerType::LinearAttention,
                Qwen3_5LayerType::FullAttention,
                Qwen3_5LayerType::LinearAttention,
                Qwen3_5LayerType::FullAttention,
            ],
            linear_key_head_dim: 4,
            linear_value_head_dim: 4,
            linear_num_key_heads: 1,
            linear_num_value_heads: 2,
            linear_conv_kernel_dim: 3,
            partial_rotary_factor: Some(0.5),
            rope_theta: Some(10_000.0),
            mtp_num_hidden_layers: 0,
            ..Default::default()
        },
        device,
    )
    .unwrap()
}

/// Requests every logit position and the complete hidden-state trace.
fn full_options() -> ForwardOptions {
    ForwardOptions {
        logits: LogitsSelection::All,
        hidden_states: true,
        ..Default::default()
    }
}

/// Builds a `[B, S]` input with deterministic token IDs and an optional mask.
fn masked_input(
    mask: &[Vec<bool>],
    seed: i64,
    device: &<TestBackend as BackendTypes>::Device,
) -> CausalLmInput<TestBackend> {
    let (batch, sequence) = (mask.len(), mask[0].len());
    let tokens = (0..batch * sequence)
        .map(|index| (index as i64 * 7 + seed) % 31)
        .collect::<Vec<_>>();
    let flags = mask.iter().flatten().copied().collect::<Vec<_>>();
    let all_valid = flags.iter().all(|&valid| valid);
    CausalLmInput {
        token_ids: Tensor::from_data(TensorData::new(tokens, [batch, sequence]), device),
        token_mask: (!all_valid).then(|| {
            TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_data(
                TensorData::new(flags, [batch, sequence]),
                device,
            ))
        }),
        position_ids: None,
    }
}

/// Every row's readable cache contents: KV `[0, L_b)` and recurrent states.
fn readable_cache(state: &Qwen3_5DecodeState<TestBackend>) -> Vec<Vec<f32>> {
    let bank = state.tensor_state().main();
    let mut contents = Vec::new();
    for row in 0..state.batch_size() {
        let length = state.lengths()[row];
        let kv = bank.native_kv(NATIVE_KV_GROUP).unwrap();
        for layer in 0..kv.spec().layers() {
            for plane in [kv.keys(layer).unwrap(), kv.values(layer).unwrap()] {
                let [_, heads, _, width] = plane.dims();
                if length > 0 {
                    contents.push(
                        plane
                            .clone()
                            .slice([row..row + 1, 0..heads, 0..length, 0..width])
                            .into_data()
                            .to_vec::<f32>()
                            .unwrap(),
                    );
                }
            }
        }
        let recurrent = bank.recurrent(RECURRENT_GROUP).unwrap();
        for layer in 0..recurrent.spec().layers() {
            contents.push(
                recurrent
                    .matrix(layer)
                    .unwrap()
                    .clone()
                    .narrow(0, row, 1)
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap(),
            );
            contents.push(
                recurrent
                    .convolution(layer)
                    .unwrap()
                    .clone()
                    .narrow(0, row, 1)
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap(),
            );
        }
    }
    contents
}

/// Asserts two outputs agree within the recurrent/chunked F32 tolerance.
fn assert_outputs_close(
    batched: &CausalLmOutput<TestBackend>,
    oracle: &CausalLmOutput<TestBackend>,
) {
    let tolerance = Tolerance::rel_abs(2e-4, 2e-5);
    batched
        .logits
        .clone()
        .into_data()
        .assert_approx_eq::<f32>(&oracle.logits.clone().into_data(), tolerance);
    assert_eq!(batched.hidden_states.len(), oracle.hidden_states.len());
    for (batched, oracle) in batched.hidden_states.iter().zip(&oracle.hidden_states) {
        batched
            .clone()
            .into_data()
            .assert_approx_eq::<f32>(&oracle.clone().into_data(), tolerance);
    }
}

/// Asserts both decode states hold the same lengths and readable cache contents.
fn assert_caches_close(
    batched: &Qwen3_5DecodeState<TestBackend>,
    oracle: &Qwen3_5DecodeState<TestBackend>,
) {
    assert_eq!(batched.lengths(), oracle.lengths());
    for (batched, oracle) in readable_cache(batched)
        .into_iter()
        .zip(readable_cache(oracle))
    {
        TensorData::new(batched.clone(), [batched.len()]).assert_approx_eq::<f32>(
            &TensorData::new(oracle, [batched.len()]),
            Tolerance::rel_abs(2e-4, 2e-5),
        );
    }
}

/// Ragged prefill and mixed decode steps match the per-row algorithm call for call.
#[test]
fn batched_calls_match_the_per_row_algorithm() {
    let device = Default::default();
    let model = hybrid_model(&device);
    let config = CacheConfig::new(3, 48).unwrap();
    let mut batched_state = model.new_decode_state(config, 3, &device).unwrap();
    let mut oracle_state = model.new_decode_state(config, 3, &device).unwrap();
    // Left, interior, and right padding with three different prompt lengths.
    let schedule: Vec<(StatefulPhase, Vec<Vec<bool>>)> = vec![
        (
            StatefulPhase::Prefill,
            vec![
                vec![false, false, true, true, true, true, true],
                vec![true, false, true, true, false, true, false],
                vec![true, true, true, true, true, true, true],
            ],
        ),
        // One token per row.
        (
            StatefulPhase::Decode,
            vec![vec![true], vec![true], vec![true]],
        ),
        // A finished row beside two active rows.
        (
            StatefulPhase::Decode,
            vec![vec![true], vec![false], vec![true]],
        ),
        // Multi-token, ragged decode.
        (
            StatefulPhase::Decode,
            vec![
                vec![true, true, false],
                vec![false, true, true],
                vec![true, false, false],
            ],
        ),
        // A long chunked prefill segment for every row.
        (
            StatefulPhase::Decode,
            vec![vec![true; 14], vec![true; 14], vec![true; 14]],
        ),
    ];
    for (step, (phase, mask)) in schedule.into_iter().enumerate() {
        let input = masked_input(&mask, step as i64 * 5 + 3, &device);
        let batched = match phase {
            StatefulPhase::Prefill => {
                model.prefill(input.clone(), &mut batched_state, &full_options())
            }
            StatefulPhase::Decode => model.decode(
                DecodeInput {
                    token_ids: input.token_ids.clone(),
                    token_mask: input.token_mask.clone(),
                    position_ids: None,
                },
                &mut batched_state,
                &full_options(),
            ),
        }
        .unwrap();
        let oracle =
            per_row_forward(&model, input, &mut oracle_state, &full_options(), phase).unwrap();
        assert_outputs_close(&batched, &oracle);
        assert_caches_close(&batched_state, &oracle_state);
    }
}

/// Explicit positions and every logits selection match the per-row algorithm.
#[test]
fn explicit_positions_and_selections_match_the_per_row_algorithm() {
    let device = Default::default();
    let model = hybrid_model(&device);
    let config = CacheConfig::new(2, 48).unwrap();
    for selection in [
        LogitsSelection::Last,
        LogitsSelection::All,
        LogitsSelection::Positions(vec![3, 0, 2]),
    ] {
        let mut batched_state = model.new_decode_state(config, 2, &device).unwrap();
        let mut oracle_state = model.new_decode_state(config, 2, &device).unwrap();
        let mut input = masked_input(
            &[vec![false, true, true, true], vec![true, true, false, true]],
            11,
            &device,
        );
        input.position_ids = Some(Tensor::from_data(
            TensorData::new(vec![0_i64, 4, 5, 9, 2, 3, 0, 7], [2, 4]),
            &device,
        ));
        let options = ForwardOptions {
            logits: selection,
            hidden_states: true,
            ..Default::default()
        };
        let batched = model
            .prefill(input.clone(), &mut batched_state, &options)
            .unwrap();
        let oracle = per_row_forward(
            &model,
            input,
            &mut oracle_state,
            &options,
            StatefulPhase::Prefill,
        )
        .unwrap();
        assert_outputs_close(&batched, &oracle);
        assert_caches_close(&batched_state, &oracle_state);
    }
}

/// Measures batched against per-row cached execution on a small hybrid model.
///
/// Eight rows prefill 48-token prompts, once left-padded to ragged lengths
/// 48, 44, ..., 20 and once unpadded, and then decode 16 steps of one token
/// each; both paths alternate and report the median of five runs. Run with
/// `cargo test --release --lib qwen3_5::model::decode::per_row_tests::batched_timing -- --ignored --nocapture`.
#[test]
#[ignore = "timing measurement; run explicitly in release mode"]
fn batched_timing() {
    let device = Default::default();
    TestBackend::seed(&device, 5);
    let model = Qwen3_5ForCausalLm::<TestBackend>::init(
        Qwen3_5TextConfig {
            vocab_size: 512,
            hidden_size: 256,
            intermediate_size: 512,
            num_hidden_layers: 4,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 64,
            max_position_embeddings: 128,
            layer_types: vec![
                Qwen3_5LayerType::LinearAttention,
                Qwen3_5LayerType::LinearAttention,
                Qwen3_5LayerType::LinearAttention,
                Qwen3_5LayerType::FullAttention,
            ],
            linear_key_head_dim: 64,
            linear_value_head_dim: 64,
            linear_num_key_heads: 2,
            linear_num_value_heads: 4,
            linear_conv_kernel_dim: 4,
            ..Default::default()
        },
        &device,
    )
    .unwrap();
    let batch = 8;
    let prompt = 48;
    for ragged in [true, false] {
        let mask = (0..batch)
            .map(|row| {
                (0..prompt)
                    .map(|column| !ragged || column >= row * 4)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let prompt_input = masked_input(&mask, 3, &device);
        let decode_input = |step: i64| masked_input(&vec![vec![true]; batch], step, &device);
        /// Batched or per-row cached execution under comparison: applies one
        /// stateful prefill or decode call and returns its `[batch, sequence, vocabulary]` logits.
        type Runner<'a> = dyn Fn(
                CausalLmInput<TestBackend>,
                &mut Qwen3_5DecodeState<TestBackend>,
                StatefulPhase,
            ) -> Tensor<TestBackend, 3>
            + 'a;
        let batched: &Runner<'_> = &|input, state, phase| {
            match phase {
                StatefulPhase::Prefill => model.prefill(input, state, &ForwardOptions::default()),
                StatefulPhase::Decode => model.decode(
                    DecodeInput {
                        token_ids: input.token_ids,
                        token_mask: input.token_mask,
                        position_ids: None,
                    },
                    state,
                    &ForwardOptions::default(),
                ),
            }
            .unwrap()
            .logits
        };
        let per_row: &Runner<'_> = &|input, state, phase| {
            per_row_forward(&model, input, state, &ForwardOptions::default(), phase)
                .unwrap()
                .logits
        };
        let mut timings = [(Vec::new(), Vec::new()), (Vec::new(), Vec::new())];
        for _ in 0..5 {
            for (runner, (prefill, decode)) in
                [batched, per_row].into_iter().zip(timings.iter_mut())
            {
                let mut state = model
                    .new_decode_state(CacheConfig::new(batch, 96).unwrap(), batch, &device)
                    .unwrap();
                let start = std::time::Instant::now();
                let _ =
                    runner(prompt_input.clone(), &mut state, StatefulPhase::Prefill).into_data();
                prefill.push(start.elapsed());
                let start = std::time::Instant::now();
                for step in 0..16 {
                    let _ =
                        runner(decode_input(step), &mut state, StatefulPhase::Decode).into_data();
                }
                decode.push(start.elapsed() / 16);
            }
        }
        let median = |samples: &mut Vec<std::time::Duration>| {
            samples.sort();
            samples[samples.len() / 2]
        };
        let [(batched_prefill, batched_decode), (row_prefill, row_decode)] = &mut timings;
        let (batched_prefill, batched_decode) = (median(batched_prefill), median(batched_decode));
        let (row_prefill, row_decode) = (median(row_prefill), median(row_decode));
        println!(
            "batch={batch} prompt={prompt} ragged={ragged}: prefill batched {batched_prefill:?} \
         per-row {row_prefill:?} ({:.2}x); decode step batched {batched_decode:?} \
         per-row {row_decode:?} ({:.2}x)",
            row_prefill.as_secs_f64() / batched_prefill.as_secs_f64(),
            row_decode.as_secs_f64() / batched_decode.as_secs_f64()
        );
    }
}
