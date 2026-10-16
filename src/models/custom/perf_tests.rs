//! Flex timings of the pass tables against the per-layer code they replace.
//!
//! These measurements are ignored by default; run them in release mode:
//!
//! ```text
//! cargo test --locked --release --lib -- --ignored --nocapture models::custom::tests::perf
//! ```
//!
//! The batched history-write timing lives with the decode state
//! (`models::custom::state::tests`), which owns the history tensors.
//!
//! `tied_vocabulary_decode_throughput` is an end-to-end series rather than an
//! old/new comparison: it times cached decode of randomly initialized
//! tied-vocabulary decoders (tiny and small modular stacks and a small
//! Qwen3.5 hybrid with the released vocabulary) and prints one JSON line per
//! configuration. Build it from two source revisions and compare the lines;
//! `TRANSFORMERS_BENCH_SOURCE_LABEL` names the revision in the output. Each
//! model is drawn from a fixed backend seed, so run the test alone
//! (`--exact --test-threads=1`) and the final-logit checksums of two revisions
//! must agree up to F32 reassociation (the FNV-1a hash of their bits shows
//! whether they are bit-identical).
//!
//! Each comparison times the former algorithm (kept here verbatim as an
//! oracle) and the new one on identical inputs, checks that both produce the
//! same values wherever attention reads them, and prints the median of several
//! repetitions. Relative-bias tables must match exactly; sliding masks must
//! match on every valid query row, because padded query rows are unconstrained
//! by design (their outputs are discarded). Timings include
//! every device synchronization the algorithm performs. One element of each
//! result is read back, so lazy backends cannot skip work, without charging
//! either side for copying a full plane to the host (attention consumes the
//! planes on the device).

use std::time::Instant;

use burn::tensor::{Bool, Int, Tensor, TensorData, backend::BackendTypes, module::embedding};

use super::*;
use crate::{
    attention::padding_attention_mask, models::custom::pass::StackPass,
    position::T5RelativePositionBiasConfig,
};

/// Device of every measurement.
type Device = <TestBackend as BackendTypes>::Device;

/// Forces a 4-D tensor's computation by reading back one element.
fn force<K: burn::tensor::TensorKind<TestBackend> + burn::tensor::BasicOps<TestBackend>>(
    tensor: Tensor<TestBackend, 4, K>,
) {
    tensor.slice([0..1, 0..1, 0..1, 0..1]).into_data();
}

/// Median wall-clock milliseconds of `repetitions` runs of `run`, after one warm-up.
fn median_ms(repetitions: usize, mut run: impl FnMut()) -> f64 {
    run();
    let mut samples = (0..repetitions)
        .map(|_| {
            let start = Instant::now();
            run();
            start.elapsed().as_secs_f64() * 1e3
        })
        .collect::<Vec<_>>();
    samples.sort_by(f64::total_cmp);
    samples[samples.len() / 2]
}

/// Former per-layer T5 bias: read positions back, loop over every pair on the host, upload, gather.
fn per_layer_relative_bias(
    bias: &crate::position::T5RelativePositionBias<TestBackend>,
    positions: Tensor<TestBackend, 2, Int>,
) -> Tensor<TestBackend, 4> {
    let [batch, sequence] = positions.dims();
    let device = positions.device();
    let positions = positions
        .into_data()
        .convert::<i64>()
        .into_vec::<i64>()
        .unwrap();
    let buckets = bias.buckets();
    let mut indices = Vec::with_capacity(batch * sequence * sequence);
    for row in positions.chunks_exact(sequence) {
        for &query in row {
            for &key in row {
                indices.push(buckets.bucket_for_positions(query as usize, key as usize) as i64);
            }
        }
    }
    let indices = Tensor::<TestBackend, 2, Int>::from_data(
        TensorData::new(indices, [batch * sequence, sequence]),
        &device,
    );
    let weight = bias.weight.val();
    let heads = weight.dims()[1];
    embedding(weight, indices)
        .reshape([batch, sequence, sequence, heads])
        .permute([0, 3, 1, 2])
}

/// Former per-layer trailing-window mask over `[batch, heads, S, S]` integer planes.
fn per_layer_sliding_mask(
    token_mask: &TokenMask<TestBackend>,
    heads: usize,
    window: usize,
) -> Tensor<TestBackend, 4, Bool> {
    let [batch, sequence] = token_mask.as_tensor().dims();
    let padding =
        padding_attention_mask(None, Some(token_mask), batch, heads, sequence, sequence).unwrap();
    let ordinals = token_mask.as_tensor().clone().int().cumsum(1);
    let query = ordinals
        .clone()
        .reshape([batch, 1, sequence, 1])
        .expand([batch, heads, sequence, sequence]);
    let key = ordinals
        .reshape([batch, 1, 1, sequence])
        .expand([batch, heads, sequence, sequence]);
    let too_old = (query - key).greater_equal_elem(window as i64);
    padding.unwrap().bool_or(too_old)
}

/// Gapped per-row positions `[batch, sequence]` and a mask with left padding in odd rows.
fn positions_and_mask(
    batch: usize,
    sequence: usize,
    device: &Device,
) -> (Tensor<TestBackend, 2, Int>, TokenMask<TestBackend>) {
    let positions = (0..batch * sequence)
        .map(|cell| ((cell % sequence) * 2 + cell / sequence) as i32)
        .collect::<Vec<_>>();
    let validity = (0..batch * sequence)
        .map(|cell| (cell / sequence).is_multiple_of(2) || cell % sequence >= sequence / 8)
        .collect::<Vec<_>>();
    (
        Tensor::from_data(TensorData::new(positions, [batch, sequence]), device),
        TokenMask::new(Tensor::from_bool(
            TensorData::new(validity, [batch, sequence]),
            device,
        )),
    )
}

/// T5 bias for `layers` blocks: one host loop and readback per layer versus one table per pass.
#[test]
#[ignore = "timing measurement; run in release with --ignored --nocapture"]
fn relative_bias_tables_versus_per_layer_host_loops() {
    let device = Default::default();
    let (batch, sequence, heads, layers) = (4, 256, 8, 8);
    let (positions, _) = positions_and_mask(batch, sequence, &device);
    let context = 2 * sequence + batch;
    let biases = (0..layers)
        .map(|_| {
            T5RelativePositionBiasConfig::new(heads, 32, 128, true)
                .init::<TestBackend>(&device)
                .unwrap()
        })
        .collect::<Vec<_>>();
    let old = median_ms(5, || {
        for bias in &biases {
            force(per_layer_relative_bias(bias, positions.clone()));
        }
    });
    let new = median_ms(5, || {
        let pass = StackPass::new(
            positions.clone(),
            None,
            context,
            crate::intervention::Stack::Decoder,
            burn::tensor::DType::F32,
        );
        for bias in &biases {
            force(pass.relative_bias(bias).unwrap());
        }
    });
    // Default positions `0..S` in every row: the table drops the batch axis.
    let contiguous = Tensor::<TestBackend, 1, Int>::arange(0..sequence as i64, &device)
        .reshape([1, sequence])
        .expand([batch, sequence]);
    let old_contiguous = median_ms(5, || {
        for bias in &biases {
            force(per_layer_relative_bias(bias, contiguous.clone()));
        }
    });
    let new_contiguous = median_ms(5, || {
        let pass = StackPass::new(
            contiguous.clone(),
            None,
            sequence,
            crate::intervention::Stack::Decoder,
            burn::tensor::DType::F32,
        )
        .with_row_invariant_positions(true);
        for bias in &biases {
            force(pass.relative_bias(bias).unwrap());
        }
    });
    let pass = StackPass::new(
        positions.clone(),
        None,
        context,
        crate::intervention::Stack::Decoder,
        burn::tensor::DType::F32,
    );
    pass.relative_bias(&biases[0])
        .unwrap()
        .into_data()
        .assert_eq(
            &per_layer_relative_bias(&biases[0], positions).into_data(),
            true,
        );
    // The shared default-position table broadcasts over the batch axis.
    let shared = StackPass::new(
        contiguous.clone(),
        None,
        sequence,
        crate::intervention::Stack::Decoder,
        burn::tensor::DType::F32,
    )
    .with_row_invariant_positions(true)
    .relative_bias(&biases[0])
    .unwrap();
    assert_eq!(shared.dims()[0], 1);
    shared
        .expand([batch, heads, sequence, sequence])
        .into_data()
        .assert_eq(
            &per_layer_relative_bias(&biases[0], contiguous).into_data(),
            true,
        );
    println!(
        "T5 bias, batch {batch}, tokens {sequence}, heads {heads}, {layers} layers, per-row gapped positions: per-layer host {old:.2} ms, pass table {new:.2} ms ({:.1}x)",
        old / new
    );
    println!(
        "T5 bias, batch {batch}, tokens {sequence}, heads {heads}, {layers} layers, default positions: per-layer host {old_contiguous:.2} ms, shared pass table {new_contiguous:.2} ms ({:.1}x)",
        old_contiguous / new_contiguous
    );
}

/// Sliding masks for `layers` blocks: per-layer `[b, h, S, S]` integer planes versus one pass plane.
#[test]
#[ignore = "timing measurement; run in release with --ignored --nocapture"]
fn connectivity_planes_versus_per_layer_integer_masks() {
    let device = Default::default();
    let (batch, sequence, heads, layers, window) = (2, 512, 8, 8, 128);
    let (_, token_mask) = positions_and_mask(batch, sequence, &device);
    let positions = crate::attention::valid_token_ordinals(&token_mask);
    let pass_plane = |pass: &StackPass<TestBackend>| {
        pass.self_attention(
            crate::intervention::Layer::decoder(0),
            AttentionConnectivity::SlidingWindow { window },
            heads,
        )
        .unwrap()
        .mask
        .unwrap()
    };
    let new_pass = || {
        StackPass::new(
            positions.clone(),
            Some(&token_mask),
            sequence,
            crate::intervention::Stack::Decoder,
            burn::tensor::DType::F32,
        )
    };
    // Equal on every valid query row; padded query rows are unconstrained by design.
    let valid_queries = token_mask
        .as_tensor()
        .clone()
        .reshape([batch, 1, sequence, 1])
        .expand([batch, heads, sequence, sequence]);
    pass_plane(&new_pass())
        .bool_and(valid_queries.clone())
        .into_data()
        .assert_eq(
            &per_layer_sliding_mask(&token_mask, heads, window)
                .bool_and(valid_queries)
                .into_data(),
            true,
        );
    let old = median_ms(5, || {
        for _ in 0..layers {
            force(per_layer_sliding_mask(&token_mask, heads, window));
        }
    });
    let new = median_ms(5, || {
        let pass = new_pass();
        for _ in 0..layers {
            force(pass_plane(&pass));
        }
    });
    println!(
        "sliding masks, batch {batch}, tokens {sequence}, heads {heads}, {layers} layers: per-layer {old:.2} ms, pass plane {new:.2} ms ({:.1}x)",
        old / new
    );
}

/// End-to-end stateless forward of a mixed modular stack (new path only, for scale).
#[test]
#[ignore = "timing measurement; run in release with --ignored --nocapture"]
fn mixed_stack_forward_time() {
    let device: Device = Default::default();
    let (batch, sequence) = (4, 256);
    let mut spec = decoder(
        (0..4)
            .map(|index| {
                layer(match index % 2 {
                    0 => attention(PositionEncodingSpec::T5RelativeBias {
                        buckets: 32,
                        max_distance: 128,
                        bidirectional: false,
                    }),
                    _ => MixerSpec::SlidingAttention(crate::architecture::SlidingWindowSpec {
                        attention: AttentionSpec {
                            query_heads: 8,
                            key_value_heads: 2,
                            head_dim: 16,
                            query_key_norm: true,
                            output_gate: false,
                            position: PositionEncodingSpec::Rope {
                                theta: 10_000.0,
                                rotary_dim: 16,
                                scaling: RopeScaling::None,
                            },
                        },
                        window: 64,
                    }),
                })
            })
            .collect(),
    );
    spec.hidden_size = 128;
    spec.max_position_embeddings = 1024;
    for block in &mut spec.layers {
        if let MixerSpec::Attention(attention) = &mut block.mixer {
            attention.query_heads = 8;
            attention.key_value_heads = 8;
            attention.head_dim = 16;
        }
        block.feed_forward = FeedForwardSpec::Dense(crate::architecture::DenseFeedForwardSpec {
            intermediate_size: 256,
            activation: Activation::Silu,
            bias: false,
        });
    }
    let model = ModularCausalLm::<TestBackend>::init(spec, &device).unwrap();
    let (positions, token_mask) = positions_and_mask(batch, sequence, &device);
    let input = CausalLmInput {
        token_ids: Tensor::zeros([batch, sequence], &device),
        token_mask: Some(token_mask),
        position_ids: Some(positions),
    };
    let elapsed = median_ms(5, || {
        model
            .forward(input.clone(), &ForwardOptions::default())
            .unwrap()
            .logits
            .into_data();
    });
    println!(
        "mixed T5-bias/sliding stack, 4 layers, batch {batch}, tokens {sequence}, width 128: {elapsed:.2} ms per forward"
    );
}

/// Minimum, median and maximum of measured samples, in their own unit.
fn spread(mut samples: Vec<f64>) -> (f64, f64, f64) {
    samples.sort_by(f64::total_cmp);
    (
        samples[0],
        samples[samples.len() / 2],
        samples[samples.len() - 1],
    )
}

/// Times batch-one cached decode of `model` and prints one JSON line.
///
/// Every repetition allocates a fresh state, prefills `prompt` tokens, then
/// decodes `steps` single tokens. The decode interval ends after reading back
/// the final logits, so no lazily queued work escapes it; prefill is reported
/// separately. Two warm-up repetitions (which also materialize lazily
/// initialized parameters) precede `repetitions` measured ones.
fn time_cached_decode<M: CausalLanguageModel<TestBackend>>(
    name: &str,
    model: &M,
    vocabulary: usize,
    repetitions: usize,
) {
    let device: Device = Default::default();
    let (prompt, steps) = (32, 32);
    let token = |index: usize| ((index * 7919 + 13) % vocabulary) as i32;
    let prompt_ids = (0..prompt).map(token).collect::<Vec<_>>();
    let options = ForwardOptions {
        logits: LogitsSelection::Last,
        ..ForwardOptions::default()
    };
    let run = || {
        let mut state = model
            .new_decode_state(CacheConfig::new(1, prompt + steps).unwrap(), 1, &device)
            .unwrap();
        let started = Instant::now();
        model
            .prefill(
                CausalLmInput {
                    token_ids: Tensor::from_data(
                        TensorData::new(prompt_ids.clone(), [1, prompt]),
                        &device,
                    ),
                    token_mask: None,
                    position_ids: None,
                },
                &mut state,
                &options,
            )
            .unwrap()
            .logits
            .into_data();
        let prefill = started.elapsed().as_secs_f64();
        let started = Instant::now();
        let mut logits = None;
        for step in 0..steps {
            logits = Some(
                model
                    .decode(
                        DecodeInput {
                            token_ids: Tensor::from_data(
                                TensorData::new(vec![token(prompt + step)], [1, 1]),
                                &device,
                            ),
                            token_mask: None,
                            position_ids: None,
                        },
                        &mut state,
                        &options,
                    )
                    .unwrap()
                    .logits,
            );
        }
        let values = logits.unwrap().into_data().to_vec::<f32>().unwrap();
        let decode = started.elapsed().as_secs_f64();
        let checksum = values.iter().map(|&value| f64::from(value)).sum::<f64>();
        // FNV-1a over the little-endian bits of every final logit.
        let bits = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| {
                (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3)
            });
        (prefill * 1e3, steps as f64 / decode, checksum, bits)
    };
    run();
    run();
    let samples = (0..repetitions).map(|_| run()).collect::<Vec<_>>();
    let (prefill_min, prefill_median, prefill_max) =
        spread(samples.iter().map(|sample| sample.0).collect());
    let (decode_min, decode_median, decode_max) =
        spread(samples.iter().map(|sample| sample.1).collect());
    let label = std::env::var("TRANSFORMERS_BENCH_SOURCE_LABEL").unwrap_or_default();
    println!(
        "TIED_DECODE_JSON={{\"source\":\"{label}\",\"config\":\"{name}\",\"vocabulary\":{vocabulary},\"prompt_tokens\":32,\"decode_steps\":32,\"repetitions\":{repetitions},\"prefill_ms\":[{prefill_min:.3},{prefill_median:.3},{prefill_max:.3}],\"decode_tokens_per_second\":[{decode_min:.2},{decode_median:.2},{decode_max:.2}],\"final_logit_sum\":{:.6},\"final_logits_fnv1a\":\"{:016x}\"}}",
        samples[0].2, samples[0].3
    );
}

/// Tied modular decoder: RoPE grouped-query attention and gated SiLU MLPs.
fn tied_modular_decoder(
    hidden: usize,
    vocabulary: usize,
    layers: usize,
    heads: (usize, usize, usize),
    intermediate: usize,
) -> DecoderSpec {
    let (query_heads, key_value_heads, head_dim) = heads;
    let block = crate::architecture::DecoderLayerSpec {
        mixer: MixerSpec::Attention(AttentionSpec {
            query_heads,
            key_value_heads,
            head_dim,
            query_key_norm: false,
            output_gate: false,
            position: PositionEncodingSpec::Rope {
                theta: 10_000.0,
                rotary_dim: head_dim,
                scaling: RopeScaling::None,
            },
        }),
        feed_forward: FeedForwardSpec::Dense(crate::architecture::DenseFeedForwardSpec {
            intermediate_size: intermediate,
            activation: Activation::Silu,
            bias: false,
        }),
        ..layer(attention(PositionEncodingSpec::None))
    };
    DecoderSpec {
        vocabulary_size: vocabulary,
        hidden_size: hidden,
        layers: vec![block; layers],
        final_norm: rms(),
        tie_word_embeddings: true,
        max_position_embeddings: 128,
    }
}

/// Flex cached-decode throughput of tied-vocabulary decoders (see the module documentation).
///
/// `tiny` (width 64, vocabulary 1,024) is dominated by per-call overhead;
/// `small` (width 576, vocabulary 49,152, SmolLM-135M's table) and
/// `qwen3.5-small` (width 1,024, vocabulary 248,320, four hybrid layers) are
/// dominated by the tied readout and the projections. Each model is drawn
/// from its own fixed backend seed, so checksums are comparable across
/// revisions.
#[test]
#[ignore = "timing measurement; run in release with --ignored --nocapture"]
fn tied_vocabulary_decode_throughput() {
    use burn::tensor::backend::Backend;

    let device: Device = Default::default();
    TestBackend::seed(&device, 2026);
    let tiny = ModularCausalLm::<TestBackend>::init(
        tied_modular_decoder(64, 1_024, 2, (4, 2, 16), 128),
        &device,
    )
    .unwrap();
    time_cached_decode("modular-tiny", &tiny, 1_024, 9);
    TestBackend::seed(&device, 2027);
    let small = ModularCausalLm::<TestBackend>::init(
        tied_modular_decoder(576, 49_152, 4, (9, 3, 64), 1_536),
        &device,
    )
    .unwrap();
    time_cached_decode("modular-small", &small, 49_152, 9);
    TestBackend::seed(&device, 2028);
    let qwen = crate::models::qwen3_5::Qwen3_5ForCausalLm::<TestBackend>::init(
        crate::models::qwen3_5::Qwen3_5TextConfig {
            vocab_size: 248_320,
            hidden_size: 1_024,
            intermediate_size: 3_584,
            num_hidden_layers: 4,
            num_attention_heads: 8,
            num_key_value_heads: 2,
            head_dim: 256,
            max_position_embeddings: 128,
            tie_word_embeddings: true,
            linear_num_key_heads: 16,
            linear_num_value_heads: 16,
            ..Default::default()
        },
        &device,
    )
    .unwrap();
    time_cached_decode("qwen3.5-small", &qwen, 248_320, 5);
}
