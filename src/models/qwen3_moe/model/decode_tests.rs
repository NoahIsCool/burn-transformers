//! Equivalence of batched Qwen3-MoE cached execution with the former per-row algorithm.
//!
//! [`per_row_forward`] is the pre-batching implementation kept as an oracle:
//! each row is compacted and evaluated alone, attention concatenates that
//! row's history with its new keys under the explicit mask
//! `k > L_b + q || k + window <= L_b + q`, and router logits are scattered
//! back per row. The tests drive both paths through identical schedules with
//! ragged, left-, interior-, and right-padded rows, finished rows, and
//! sliding-window layers, and compare logits, hidden states, router
//! diagnostics, the auxiliary loss, and every row's readable cache.

use burn::tensor::{
    Bool, Int, Tensor, TensorData, Tolerance,
    backend::{Backend, BackendTypes},
};

use super::*;
use crate::{
    DecodeState, LogitsSelection,
    architecture::{RouterDiagnostics, TokenMask},
    checkpoint::TensorSink,
    models::{
        config::{RopeParameters, RopeType},
        qwen3_moe::{Qwen3MoeCheckpointOptions, Qwen3MoeConfig},
        shared::{implicit_position_ids, rows::concat_rows},
    },
    position::{apply_rotary, rotary_angles},
};

/// CPU Flex backend with F32 activations and portable I32 indices.
type TestBackend = burn::backend::Flex<f32, i32>;

/// Runs one stateful call with the former per-row algorithm.
fn per_row_forward<B: Backend>(
    model: &Qwen3MoeForCausalLm<B>,
    input: CausalLmInput<B>,
    state: &mut Qwen3MoeDecodeState<B>,
    options: &ForwardOptions,
    phase: StatefulPhase,
) -> Result<CausalLmOutput<B>> {
    let prepared = prepare_stateful_input(model, &input, state, phase)?;
    validate_logits_selection(&options.logits, prepared.sequence)?;
    let device = input.token_ids.device();
    let retain_routers = model.retains_router_logits(options);
    let sparse_layers = model
        .model
        .layers
        .iter()
        .filter(|layer| layer.sparse_mlp.is_some())
        .count();
    let mut writes = state.tensors.step(&prepared.token_counts)?;
    let mut final_rows = Vec::new();
    let mut diagnostic_rows: Vec<Vec<Tensor<B, 3>>> = if options.hidden_states {
        (0..model.model.layers.len() + 2)
            .map(|_| Vec::new())
            .collect()
    } else {
        Vec::new()
    };
    let mut router_rows: Vec<Vec<Tensor<B, 3>>> = if retain_routers {
        (0..sparse_layers).map(|_| Vec::new()).collect()
    } else {
        Vec::new()
    };
    let dtype = model.model.embed_tokens.weight.val().dtype();
    for row in 0..prepared.batch {
        let Some(plan) = prepared.compact_row::<B>(row, &device)? else {
            let empty = |width: usize| {
                Tensor::<B, 3>::zeros([1, prepared.sequence, width], (&device, dtype))
            };
            final_rows.push(empty(model.config().hidden_size));
            for destination in &mut diagnostic_rows {
                destination.push(empty(model.config().hidden_size));
            }
            for destination in &mut router_rows {
                destination.push(empty(model.config().num_experts));
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
        let mut routers = Vec::new();
        for (index, layer) in model.model.layers.iter().enumerate() {
            let residual = hidden.clone();
            let normalized = layer.input_layernorm.forward(hidden);
            let mut projection = layer.self_attn.project_self(normalized)?;
            (projection.query, projection.key) =
                apply_rotary(projection.query, projection.key, &rotary)?;
            writes.push_kv_row(
                NATIVE_KV_GROUP,
                index,
                row,
                projection.key.clone(),
                projection.value.clone(),
            );
            let history = state
                .tensors
                .main()
                .native_kv(NATIVE_KV_GROUP)?
                .read_sequence(index, row, baseline)?;
            if let Some((keys, values)) = history.materialize() {
                projection.key = Tensor::cat(vec![keys, projection.key], 2);
                projection.value = Tensor::cat(vec![values, projection.value], 2);
            }
            let queries = projection.query.dims()[2];
            let mask = cached_attention_mask::<B>(
                baseline,
                queries,
                layer.self_attn.query_heads,
                model.config().layer_sliding_window(index),
                &device,
            );
            let hidden_after_attention = residual
                + layer
                    .self_attn
                    .attend(projection, Some(mask), None, false)?;
            let normalized = layer
                .post_attention_layernorm
                .forward(hidden_after_attention.clone());
            let (mlp, logits) = layer.feed_forward(normalized, Layer::decoder(index), None)?;
            hidden = hidden_after_attention + mlp;
            diagnostics.push(hidden.clone());
            routers.extend(logits);
        }
        hidden = model.model.norm.forward(hidden);
        diagnostics.push(hidden.clone());
        final_rows.push(plan.scatter(hidden)?);
        for (destination, diagnostic) in diagnostic_rows.iter_mut().zip(diagnostics) {
            destination.push(plan.scatter(diagnostic)?);
        }
        for (destination, logits) in router_rows.iter_mut().zip(routers) {
            destination.push(plan.scatter(logits)?);
        }
    }
    let hidden = concat_rows(final_rows, "hidden")?;
    let hidden_states = diagnostic_rows
        .into_iter()
        .map(|rows| concat_rows(rows, "hidden_states"))
        .collect::<Result<Vec<_>>>()?;
    let router_logits = router_rows
        .into_iter()
        .map(|rows| concat_rows(rows, "router_logits"))
        .collect::<Result<Vec<_>>>()?;
    let selected =
        select_hidden_with_last_columns(hidden.clone(), prepared.last_columns(), &options.logits)?;
    let output = model.finish_selected(
        &input,
        options,
        selected,
        options.hidden_states.then_some(hidden.clone()),
        Qwen3MoeBackboneOutput {
            hidden,
            hidden_states,
            router_logits,
        },
    )?;
    state.tensors.publish_step(writes)?;
    Ok(output)
}

/// The former explicit cached mask `k > W + q || (window && k + window <= W + q)`.
fn cached_attention_mask<B: Backend>(
    baseline: usize,
    queries: usize,
    heads: usize,
    window: Option<usize>,
    device: &B::Device,
) -> Tensor<B, 4, Bool> {
    let keys = baseline + queries;
    let query = Tensor::<B, 1, Int>::arange(baseline as i64..keys as i64, device)
        .reshape([1, 1, queries, 1])
        .expand([1, heads, queries, keys]);
    let key = Tensor::<B, 1, Int>::arange(0..keys as i64, device)
        .reshape([1, 1, 1, keys])
        .expand([1, heads, queries, keys]);
    let causal = key.clone().greater(query.clone());
    match window {
        Some(window) => causal.bool_or((key + window as i64).lower_equal(query)),
        None => causal,
    }
}

/// Builds a three-layer model (dense, sparse, sparse) whose every block uses a sliding window.
fn windowed_model(
    device: &<TestBackend as BackendTypes>::Device,
) -> Qwen3MoeForCausalLm<TestBackend> {
    let config = Qwen3MoeConfig {
        vocab_size: 32,
        hidden_size: 8,
        intermediate_size: 12,
        num_hidden_layers: 3,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: 4,
        max_position_embeddings: 64,
        rope_theta: Some(10_000.0),
        use_sliding_window: true,
        sliding_window: Some(3),
        decoder_sparse_step: 1,
        mlp_only_layers: vec![0],
        moe_intermediate_size: 6,
        num_experts: 4,
        num_experts_per_tok: 2,
        norm_topk_prob: true,
        output_router_logits: true,
        ..Default::default()
    };
    let contract = config
        .checkpoint_contract(Qwen3MoeCheckpointOptions::default())
        .unwrap();
    let mut model = Qwen3MoeForCausalLm::init(config, device).unwrap();
    for (tensor_index, expected) in contract.expected_tensors().iter().enumerate() {
        let shape = expected
            .shape
            .iter()
            .map(|&dimension| usize::try_from(dimension).unwrap())
            .collect::<Vec<_>>();
        let elements = shape.iter().product();
        let values = (0..elements)
            .map(|element| {
                let code = (element * 5 + tensor_index * 7) % 29;
                (code as f32 - 14.0) * 0.03
            })
            .collect::<Vec<_>>();
        TensorSink::write_tensor(&mut model, &expected.name, TensorData::new(values, shape))
            .unwrap();
    }
    model
}

/// Builds a `[B, S]` input with deterministic token IDs, sentinel padding, and an optional mask.
fn masked_input(
    mask: &[Vec<bool>],
    seed: i64,
    device: &<TestBackend as BackendTypes>::Device,
) -> CausalLmInput<TestBackend> {
    let (batch, sequence) = (mask.len(), mask[0].len());
    let flags = mask.iter().flatten().copied().collect::<Vec<_>>();
    // Padded columns hold an out-of-vocabulary sentinel that must stay inert.
    let tokens = flags
        .iter()
        .enumerate()
        .map(|(index, &valid)| {
            if valid {
                (index as i64 * 7 + seed) % 31
            } else {
                99
            }
        })
        .collect::<Vec<_>>();
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

/// Every row's readable KV slots `[0, L_b)` across all layers.
fn readable_cache(state: &Qwen3MoeDecodeState<TestBackend>) -> Vec<Vec<f32>> {
    let cache = state
        .tensor_state()
        .main()
        .native_kv(NATIVE_KV_GROUP)
        .unwrap();
    let mut contents = Vec::new();
    for (row, &length) in state.lengths().iter().enumerate() {
        for layer in 0..cache.spec().layers() {
            for plane in [cache.keys(layer).unwrap(), cache.values(layer).unwrap()] {
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
    }
    contents
}

/// Asserts two outputs agree on logits, traces, router diagnostics, and loss.
fn assert_outputs_close(
    batched: &CausalLmOutput<TestBackend>,
    oracle: &CausalLmOutput<TestBackend>,
) {
    let tolerance = Tolerance::rel_abs(2e-4, 2e-5);
    let close = |left: &Tensor<TestBackend, 3>, right: &Tensor<TestBackend, 3>| {
        left.clone()
            .into_data()
            .assert_approx_eq::<f32>(&right.clone().into_data(), tolerance);
    };
    close(&batched.logits, &oracle.logits);
    assert_eq!(batched.hidden_states.len(), oracle.hidden_states.len());
    for (left, right) in batched.hidden_states.iter().zip(&oracle.hidden_states) {
        close(left, right);
    }
    assert_eq!(batched.routers.len(), oracle.routers.len());
    for (left, right) in batched.routers.iter().zip(&oracle.routers) {
        let logits = |router: &RouterDiagnostics<TestBackend>| router.logits.clone().unwrap();
        close(&logits(left), &logits(right));
    }
    match (&batched.load_balancing_loss, &oracle.load_balancing_loss) {
        (Some(left), Some(right)) => left
            .clone()
            .into_data()
            .assert_approx_eq::<f32>(&right.clone().into_data(), tolerance),
        (None, None) => {}
        _ => panic!("auxiliary loss presence differs"),
    }
}

/// Ragged prefill and mixed decode match the per-row algorithm, sliding windows included.
#[test]
fn batched_calls_match_the_per_row_algorithm() {
    let device = Default::default();
    let model = windowed_model(&device);
    let config = CacheConfig::new(3, 40).unwrap();
    let mut batched_state = model.new_decode_state(config, 3, &device).unwrap();
    let mut oracle_state = model.new_decode_state(config, 3, &device).unwrap();
    let options = ForwardOptions {
        logits: LogitsSelection::All,
        hidden_states: true,
        router_logits: true,
        auxiliary_losses: true,
        ..Default::default()
    };
    let schedule: Vec<(StatefulPhase, Vec<Vec<bool>>)> = vec![
        (
            StatefulPhase::Prefill,
            vec![
                vec![false, false, true, true, true, true],
                vec![true, false, true, true, false, true],
                vec![true, true, true, true, true, true],
            ],
        ),
        (
            StatefulPhase::Decode,
            vec![vec![true], vec![true], vec![true]],
        ),
        (
            StatefulPhase::Decode,
            vec![vec![true], vec![false], vec![true]],
        ),
        (
            StatefulPhase::Decode,
            vec![
                vec![true, true, false],
                vec![false, true, true],
                vec![true, false, false],
            ],
        ),
        (
            StatefulPhase::Decode,
            vec![vec![true; 5], vec![true; 5], vec![true; 5]],
        ),
    ];
    for (step, (phase, mask)) in schedule.into_iter().enumerate() {
        let input = masked_input(&mask, step as i64 * 3 + 1, &device);
        let batched = match phase {
            StatefulPhase::Prefill => model.prefill(input.clone(), &mut batched_state, &options),
            StatefulPhase::Decode => model.decode(
                DecodeInput {
                    token_ids: input.token_ids.clone(),
                    token_mask: input.token_mask.clone(),
                    position_ids: None,
                },
                &mut batched_state,
                &options,
            ),
        }
        .unwrap();
        let oracle = per_row_forward(&model, input, &mut oracle_state, &options, phase).unwrap();
        assert_outputs_close(&batched, &oracle);
        assert_eq!(batched_state.lengths(), oracle_state.lengths());
        for (left, right) in readable_cache(&batched_state)
            .into_iter()
            .zip(readable_cache(&oracle_state))
        {
            TensorData::new(left.clone(), [left.len()]).assert_approx_eq::<f32>(
                &TensorData::new(right, [left.len()]),
                Tolerance::rel_abs(2e-4, 2e-5),
            );
        }
    }
}

/// Explicit positions and each logits selection match the per-row algorithm.
#[test]
fn explicit_positions_and_selections_match_the_per_row_algorithm() {
    let device = Default::default();
    let model = windowed_model(&device);
    let config = CacheConfig::new(2, 40).unwrap();
    for selection in [
        LogitsSelection::Last,
        LogitsSelection::All,
        LogitsSelection::Positions(vec![2, 0]),
    ] {
        let mut batched_state = model.new_decode_state(config, 2, &device).unwrap();
        let mut oracle_state = model.new_decode_state(config, 2, &device).unwrap();
        let mut input = masked_input(
            &[vec![true, false, true, true], vec![false, true, true, true]],
            5,
            &device,
        );
        input.position_ids = Some(Tensor::from_data(
            TensorData::new(vec![3_i64, -1, 4, 9, -1, 0, 6, 7], [2, 4]),
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
    }
}

/// A context-dependent rotary policy rejects the first active row past its limit.
///
/// LongRoPE switches frequency tables once the evaluated context passes the
/// original training context `L = 2`. Row 0 of the batch stays within it
/// (two tokens), row 1 does not (three tokens), so the prefill fails with
/// row 1's context before any cache mutation.
#[test]
fn context_dependent_rope_rejects_the_first_row_past_the_limit() {
    let device = Default::default();
    let config = Qwen3MoeConfig {
        vocab_size: 32,
        hidden_size: 8,
        intermediate_size: 12,
        num_hidden_layers: 2,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: 4,
        max_position_embeddings: 64,
        rope_parameters: Some(RopeParameters {
            rope_type: RopeType::LongRope,
            factor: Some(2.0),
            original_max_position_embeddings: Some(2),
            short_factor: Some(vec![1.0, 1.0]),
            long_factor: Some(vec![1.5, 2.0]),
            ..Default::default()
        }),
        decoder_sparse_step: 1,
        moe_intermediate_size: 6,
        num_experts: 4,
        num_experts_per_tok: 2,
        ..Default::default()
    };
    let model = Qwen3MoeForCausalLm::<TestBackend>::init(config, &device).unwrap();
    let mut state = model
        .new_decode_state(CacheConfig::new(2, 4).unwrap(), 2, &device)
        .unwrap();
    let error = model
        .prefill(
            masked_input(
                &[vec![true, true, false], vec![true, true, true]],
                3,
                &device,
            ),
            &mut state,
            &ForwardOptions::default(),
        )
        .expect_err("row 1 passes the original LongRoPE window");
    assert!(matches!(error, Error::BackendCapability(_)));
    assert!(
        error.to_string().contains(
            "Qwen3 MoE cached decoding with a context-dependent RoPE policy cannot extend the evaluated context to 3 beyond 2"
        ),
        "{error}"
    );
    assert_eq!(state.lengths(), &[0, 0]);

    // Within the limit every row is accepted.
    model
        .prefill(
            masked_input(&[vec![true, true], vec![false, true]], 3, &device),
            &mut state,
            &ForwardOptions::default(),
        )
        .expect("contexts within the original window are cacheable");
    assert_eq!(state.lengths(), &[2, 1]);
}

/// Measures batched against per-row cached execution on a small MoE model.
///
/// Eight rows prefill 32-token prompts (unpadded and left-padded to ragged
/// lengths) and decode 16 one-token steps; both paths alternate and report
/// the median of five runs. Run with
/// `cargo test --release --lib qwen3_moe::model::decode::tests::batched_timing -- --ignored --nocapture`.
#[test]
#[ignore = "timing measurement; run explicitly in release mode"]
fn batched_timing() {
    let device = Default::default();
    TestBackend::seed(&device, 3);
    let model = Qwen3MoeForCausalLm::<TestBackend>::init(
        Qwen3MoeConfig {
            vocab_size: 512,
            hidden_size: 128,
            intermediate_size: 256,
            num_hidden_layers: 4,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 32,
            max_position_embeddings: 128,
            rope_theta: Some(10_000.0),
            decoder_sparse_step: 1,
            moe_intermediate_size: 64,
            num_experts: 16,
            num_experts_per_tok: 4,
            norm_topk_prob: true,
            ..Default::default()
        },
        &device,
    )
    .unwrap();
    let batch = 8;
    let prompt = 32;
    for ragged in [false, true] {
        let mask = (0..batch)
            .map(|row| {
                (0..prompt)
                    .map(|column| !ragged || column >= row * 3)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let prompt_input = masked_input(&mask, 3, &device);
        let options = ForwardOptions::default();
        let mut timings = [(Vec::new(), Vec::new()), (Vec::new(), Vec::new())];
        for _ in 0..5 {
            for (batched, (prefill, decode)) in [true, false].into_iter().zip(timings.iter_mut()) {
                let mut state = model
                    .new_decode_state(CacheConfig::new(batch, 64).unwrap(), batch, &device)
                    .unwrap();
                let run = |input: CausalLmInput<TestBackend>,
                           state: &mut Qwen3MoeDecodeState<TestBackend>,
                           phase: StatefulPhase| {
                    if batched {
                        match phase {
                            StatefulPhase::Prefill => model.prefill(input, state, &options),
                            StatefulPhase::Decode => model.decode(
                                DecodeInput {
                                    token_ids: input.token_ids,
                                    token_mask: input.token_mask,
                                    position_ids: None,
                                },
                                state,
                                &options,
                            ),
                        }
                    } else {
                        per_row_forward(&model, input, state, &options, phase)
                    }
                    .unwrap()
                    .logits
                    .into_data()
                };
                let start = std::time::Instant::now();
                run(prompt_input.clone(), &mut state, StatefulPhase::Prefill);
                prefill.push(start.elapsed());
                let start = std::time::Instant::now();
                for step in 0..16 {
                    run(
                        masked_input(&vec![vec![true]; batch], step, &device),
                        &mut state,
                        StatefulPhase::Decode,
                    );
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
