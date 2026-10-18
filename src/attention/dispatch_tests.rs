//! Grouped attention dispatch: scalar-oracle parity, folding, causal alignment, and sinks.

use burn::{
    backend::Flex,
    tensor::{Bool, DType, Tensor, TensorData, module::attention},
};

use super::*;
use crate::attention::mask::bottom_right_causal_mask;

/// Deterministic F32 CPU backend for every dispatch test.
type TestBackend = Flex<f32, i32>;

/// One attention problem in plain host vectors, shared by the oracle and Burn.
#[derive(Clone, Debug)]
struct Problem {
    /// Batch rows.
    batch: usize,
    /// Query heads.
    query_heads: usize,
    /// Native key/value heads.
    key_value_heads: usize,
    /// Query tokens.
    queries: usize,
    /// Key tokens.
    keys: usize,
    /// Query/key channels.
    head_dim: usize,
    /// Value channels.
    value_dim: usize,
    /// Optional exclusions `[batch, query_head, query, key]`.
    mask: Option<Vec<bool>>,
    /// Optional bias `[1, query_head, query, key]`, broadcast over batch.
    bias: Option<Vec<f32>>,
    /// Optional per-query-head sink logits.
    sinks: Option<Vec<f32>>,
    /// Whether bottom-right causal masking applies.
    causal: bool,
    /// Optional explicit score scale.
    scale: Option<f64>,
    /// Optional logit softcap.
    softcap: Option<f64>,
}

/// Deterministic pseudo-random values in roughly `[-1.1, 1.1]`.
fn values(count: usize, seed: usize) -> Vec<f32> {
    (0..count)
        .map(|index| ((index * 37 + seed * 11) % 23) as f32 * 0.1 - 1.1)
        .collect()
}

impl Problem {
    /// A dense problem without masks, bias, sinks, or causality.
    fn new(
        batch: usize,
        query_heads: usize,
        key_value_heads: usize,
        queries: usize,
        keys: usize,
    ) -> Self {
        Self {
            batch,
            query_heads,
            key_value_heads,
            queries,
            keys,
            head_dim: 3,
            value_dim: 2,
            mask: None,
            bias: None,
            sinks: None,
            causal: false,
            scale: None,
            softcap: None,
        }
    }

    /// Query values `[batch, query_head, query, head_dim]`.
    fn query_values(&self) -> Vec<f32> {
        values(
            self.batch * self.query_heads * self.queries * self.head_dim,
            1,
        )
    }

    /// Key values `[batch, kv_head, key, head_dim]`.
    fn key_values(&self) -> Vec<f32> {
        values(
            self.batch * self.key_value_heads * self.keys * self.head_dim,
            2,
        )
    }

    /// Value values `[batch, kv_head, key, value_dim]`.
    fn value_values(&self) -> Vec<f32> {
        values(
            self.batch * self.key_value_heads * self.keys * self.value_dim,
            3,
        )
    }

    /// Adds a padding-style mask that leaves `query 0` of batch 0 with no visible key.
    fn with_padding(mut self) -> Self {
        let mut mask = vec![false; self.batch * self.query_heads * self.queries * self.keys];
        for batch in 0..self.batch {
            for head in 0..self.query_heads {
                for query in 0..self.queries {
                    for key in 0..self.keys {
                        let index = ((batch * self.query_heads + head) * self.queries + query)
                            * self.keys
                            + key;
                        // Batch 0's first query is fully padded; key 0 of batch 1 is padding.
                        mask[index] = (batch == 0 && query == 0) || (batch == 1 && key == 0);
                    }
                }
            }
        }
        self.mask = Some(mask);
        self
    }

    /// Adds a batch-broadcast additive bias.
    fn with_bias(mut self) -> Self {
        self.bias = Some(values(self.query_heads * self.queries * self.keys, 5));
        self
    }

    /// Builds Burn tensors for the problem with every float operand cast to `dtype`.
    #[allow(clippy::type_complexity)]
    fn tensors_in(
        &self,
        dtype: DType,
    ) -> (
        Tensor<TestBackend, 4>,
        Tensor<TestBackend, 4>,
        Tensor<TestBackend, 4>,
        Option<Tensor<TestBackend, 4, Bool>>,
        Option<Tensor<TestBackend, 4>>,
        Option<Tensor<TestBackend, 1>>,
    ) {
        let device = Default::default();
        let float = |data: Vec<f32>, shape: [usize; 4]| {
            Tensor::<TestBackend, 4>::from_data(TensorData::new(data, shape), &device).cast(dtype)
        };
        (
            float(
                self.query_values(),
                [self.batch, self.query_heads, self.queries, self.head_dim],
            ),
            float(
                self.key_values(),
                [self.batch, self.key_value_heads, self.keys, self.head_dim],
            ),
            float(
                self.value_values(),
                [self.batch, self.key_value_heads, self.keys, self.value_dim],
            ),
            self.mask.clone().map(|mask| {
                Tensor::from_data(
                    TensorData::new(
                        mask,
                        [self.batch, self.query_heads, self.queries, self.keys],
                    ),
                    &device,
                )
            }),
            self.bias
                .clone()
                .map(|bias| float(bias, [1, self.query_heads, self.queries, self.keys])),
            self.sinks.clone().map(|sinks| {
                Tensor::<TestBackend, 1>::from_data(
                    TensorData::new(sinks, [self.query_heads]),
                    &device,
                )
                .cast(dtype)
            }),
        )
    }

    /// Burn attention options for the problem.
    fn options(&self) -> AttentionModuleOptions {
        AttentionModuleOptions {
            scale: self.scale,
            softcap: self.softcap,
            is_causal: self.causal,
        }
    }

    /// Evaluates the problem through the crate's grouped dispatch.
    fn evaluate(&self, precision: AttentionPrecisionMode) -> Result<Vec<f32>> {
        self.evaluate_in(precision, DType::F32)
    }

    /// Evaluates the problem in `dtype` and returns the output widened to F32.
    fn evaluate_in(&self, precision: AttentionPrecisionMode, dtype: DType) -> Result<Vec<f32>> {
        let (query, key, value, mask, bias, sinks) = self.tensors_in(dtype);
        let output = evaluate(
            Operands {
                query,
                key,
                value,
                mask,
                bias,
                sinks,
            },
            self.options(),
            precision,
        )?;
        assert_eq!(
            output.dtype(),
            dtype,
            "attention must keep the operand dtype"
        );
        Ok(output.cast(DType::F32).into_data().to_vec::<f32>().unwrap())
    }

    /// Straightforward scalar reference in F64, with zero output for rows without support.
    fn oracle(&self) -> Vec<f32> {
        let query = self.query_values();
        let key = self.key_values();
        let value = self.value_values();
        let groups = self.query_heads / self.key_value_heads;
        let scale = self
            .scale
            .unwrap_or_else(|| (self.head_dim as f64).sqrt().recip());
        let offset = self.keys as i64 - self.queries as i64;
        let mut output = Vec::new();
        for batch in 0..self.batch {
            for head in 0..self.query_heads {
                let kv = head / groups;
                for row in 0..self.queries {
                    let mut logits = Vec::new();
                    for column in 0..self.keys {
                        let score_index = ((batch * self.query_heads + head) * self.queries + row)
                            * self.keys
                            + column;
                        let excluded = self.mask.as_ref().is_some_and(|mask| mask[score_index])
                            || (self.causal && column as i64 > row as i64 + offset);
                        if excluded {
                            continue;
                        }
                        let mut dot = 0.0_f64;
                        for channel in 0..self.head_dim {
                            let q = query[((batch * self.query_heads + head) * self.queries + row)
                                * self.head_dim
                                + channel];
                            let k = key[((batch * self.key_value_heads + kv) * self.keys + column)
                                * self.head_dim
                                + channel];
                            dot += f64::from(q) * f64::from(k);
                        }
                        let mut logit = dot * scale;
                        if let Some(softcap) = self.softcap {
                            logit = (logit / softcap).tanh() * softcap;
                        }
                        if let Some(bias) = &self.bias {
                            logit +=
                                f64::from(bias[(head * self.queries + row) * self.keys + column]);
                        }
                        logits.push((column, logit));
                    }
                    let sink = self.sinks.as_ref().map(|sinks| f64::from(sinks[head]));
                    let maximum = logits
                        .iter()
                        .map(|(_, logit)| *logit)
                        .chain(sink)
                        .fold(f64::NEG_INFINITY, f64::max);
                    let denominator = logits
                        .iter()
                        .map(|(_, logit)| (logit - maximum).exp())
                        .chain(sink.map(|sink| (sink - maximum).exp()))
                        .sum::<f64>();
                    for channel in 0..self.value_dim {
                        let mut mixed = 0.0;
                        for (column, logit) in &logits {
                            let v = value[((batch * self.key_value_heads + kv) * self.keys
                                + column)
                                * self.value_dim
                                + channel];
                            mixed += (logit - maximum).exp() / denominator * f64::from(v);
                        }
                        output.push(mixed as f32);
                    }
                }
            }
        }
        output
    }

    /// Evaluates the previous implementation: repeated K/V and an explicit causal mask.
    fn repeated_reference(&self, precision: AttentionPrecisionMode) -> Vec<f32> {
        self.repeated_reference_in(precision, DType::F32)
    }

    /// Evaluates the previous repeated-K/V implementation in `dtype`, widened to F32.
    fn repeated_reference_in(&self, precision: AttentionPrecisionMode, dtype: DType) -> Vec<f32> {
        let (query, key, value, mask, bias, _) = self.tensors_in(dtype);
        let key = repeat_key_value(key, self.query_heads).unwrap();
        let value = repeat_key_value(value, self.query_heads).unwrap();
        let causal = (self.causal && self.queries > 1).then(|| {
            bottom_right_causal_mask(
                self.batch,
                self.query_heads,
                self.queries,
                self.keys,
                &Default::default(),
            )
            .unwrap()
        });
        let mask = crate::attention::mask::union_exclusions(mask, causal);
        let options = AttentionModuleOptions {
            is_causal: false,
            ..self.options()
        };
        let output = match precision {
            AttentionPrecisionMode::BackendDefault => {
                attention(query, key, value, mask, bias, options)
            }
            AttentionPrecisionMode::EagerFp32Softmax => {
                let scale = self.scale.unwrap_or_else(|| default_scale(self.head_dim));
                let scores = scaled_scores(query, key, scale, self.softcap);
                grouped_mix(
                    eager_probabilities(scores, mask.as_ref(), bias, None),
                    value,
                )
            }
        };
        output.cast(DType::F32).into_data().to_vec::<f32>().unwrap()
    }
}

/// Asserts elementwise closeness within the documented F32 tolerance.
fn assert_close(actual: &[f32], expected: &[f32], context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}: length");
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= 2e-5 + 2e-5 * expected.abs(),
            "{context}: element {index}: {actual} != {expected}"
        );
    }
}

/// Both operation graphs agree with a scalar F64 oracle across groups, alignments, masks, and bias.
#[test]
fn grouped_dispatch_matches_scalar_oracle_for_every_alignment() {
    for (query_heads, key_value_heads) in [(2, 2), (4, 2), (4, 1), (6, 2)] {
        for (queries, keys) in [(1, 5), (3, 5), (5, 5), (4, 3)] {
            for causal in [false, true] {
                for (padded, biased) in [(false, false), (true, false), (true, true)] {
                    let mut problem = Problem::new(2, query_heads, key_value_heads, queries, keys);
                    problem.causal = causal;
                    if padded {
                        problem = problem.with_padding();
                    }
                    if biased {
                        problem = problem.with_bias();
                    }
                    let expected = problem.oracle();
                    for precision in [
                        AttentionPrecisionMode::BackendDefault,
                        AttentionPrecisionMode::EagerFp32Softmax,
                    ] {
                        let context = format!(
                            "{precision:?} heads={query_heads}/{key_value_heads} q={queries} k={keys} causal={causal} padded={padded} bias={biased}"
                        );
                        assert_close(&problem.evaluate(precision).unwrap(), &expected, &context);
                    }
                }
            }
        }
    }
}

/// Folding is numerically equivalent to the previous repeat-K/V implementation.
#[test]
fn folded_groups_match_previous_repeated_key_value_path() {
    for (query_heads, key_value_heads) in [(4, 2), (8, 2), (4, 4)] {
        for (queries, keys, causal) in [(1, 7, false), (1, 7, true), (3, 7, true), (6, 6, true)] {
            let mut problem = Problem::new(2, query_heads, key_value_heads, queries, keys)
                .with_padding()
                .with_bias();
            problem.causal = causal;
            problem.scale = Some(0.37);
            problem.softcap = Some(1.5);
            for precision in [
                AttentionPrecisionMode::BackendDefault,
                AttentionPrecisionMode::EagerFp32Softmax,
            ] {
                assert_close(
                    &problem.evaluate(precision).unwrap(),
                    &problem.repeated_reference(precision),
                    &format!("{precision:?} {query_heads}/{key_value_heads} q={queries} k={keys}"),
                );
            }
        }
    }
}

/// Asserts elementwise closeness within an absolute reduced-precision tolerance.
fn assert_within(actual: &[f32], expected: &[f32], tolerance: f32, context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}: length");
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "{context}: element {index}: {actual} != {expected} (tolerance {tolerance})"
        );
    }
}

/// Reduced-precision tolerances calibrated on Flex: `(dtype, between graphs, against F64)`.
///
/// Fixture outputs are convex combinations of values in `[-1.1, 1.1]` with
/// scaled scores up to about 5 at head width 128. The eager graph rounds those
/// scores to the operand dtype before its F32 softmax (relative `2^-11` in F16,
/// `2^-8` in BF16), which alone moves outputs by a few `1e-3`. Every product
/// is an F32 sum rounded once (the eager F16 products on Flex through
/// [`f32_accumulated_matmul`]). Observed maxima on Flex: the folded and
/// repeated graphs are equal in both dtypes (up to the sign of zero on fully
/// padded rows); against F64, F16 `1.6e-3` eager and `7.9e-4` fused, BF16
/// `9.8e-3` eager and `5.1e-3` fused. While Flex's F16 GEMM still formed the
/// eager products with F16 partial sums, eager F16 differed between graphs
/// by up to `2.0e-3` and from F64 by `7.6e-3`. The bounds below, unchanged by
/// that fix, leave at least 1.6x headroom and match common F16/BF16 attention
/// parity tolerances.
const REDUCED_PRECISION: [(DType, f32, f32); 2] =
    [(DType::F16, 4e-3, 1.25e-2), (DType::BF16, 1.6e-2, 1.6e-2)];

/// Folding stays equivalent to repeated K/V in F16 and BF16 at production head widths.
///
/// Both graphs see identical rounded operands; folding changes only matmul
/// shapes, and every product is an F32 sum rounded once, so the graphs can
/// differ only where F32 sums in another order round differently. On these
/// fixtures they are equal on Flex (up to the sign of zero on fully padded
/// rows); the tolerance keeps the test independent of summation order. Both
/// stay within the same distance of the F64 oracle.
#[test]
fn folded_groups_match_repeated_key_value_in_reduced_precision() {
    for head_dim in [64, 128] {
        for (queries, keys, causal) in [(1, 9, true), (3, 9, true), (4, 4, true)] {
            let mut problem = Problem::new(2, 8, 2, queries, keys).with_padding();
            problem.head_dim = head_dim;
            problem.value_dim = head_dim;
            problem.causal = causal;
            let oracle = problem.oracle();
            for (dtype, between, against_oracle) in REDUCED_PRECISION {
                for precision in [
                    AttentionPrecisionMode::BackendDefault,
                    AttentionPrecisionMode::EagerFp32Softmax,
                ] {
                    let context =
                        format!("{precision:?} {dtype:?} d={head_dim} q={queries} k={keys}");
                    let folded = problem.evaluate_in(precision, dtype).unwrap();
                    let repeated = problem.repeated_reference_in(precision, dtype);
                    assert_within(&folded, &repeated, between, &context);
                    assert_within(&folded, &oracle, against_oracle, &context);
                    assert_within(&repeated, &oracle, against_oracle, &context);
                }
            }
        }
    }
}

/// Native sinks in F16 and BF16 track eager sinks and the F64 oracle at head width 128.
///
/// With `scale = 1 / sqrt(128)` (not a power of two) the native sink key holds
/// `round(s_h / scale)`, so its logit carries one extra rounding in the operand
/// dtype. It stays within the eager graph's own reduced-precision error (on
/// Flex the native graph is the more accurate one: `7.2e-4` versus `2.7e-3`
/// against F64 in F16).
#[test]
fn native_sinks_track_eager_and_oracle_in_reduced_precision() {
    let mut problem = Problem::new(2, 8, 2, 3, 9).with_padding();
    problem.head_dim = 128;
    problem.causal = true;
    problem.sinks = Some(values(8, 7).iter().map(|value| value * 4.0).collect());
    let oracle = problem.oracle();
    for (dtype, between, against_oracle) in REDUCED_PRECISION {
        let native = problem
            .evaluate_in(AttentionPrecisionMode::BackendDefault, dtype)
            .unwrap();
        let eager = problem
            .evaluate_in(AttentionPrecisionMode::EagerFp32Softmax, dtype)
            .unwrap();
        assert_within(
            &native,
            &oracle,
            against_oracle,
            &format!("{dtype:?} native"),
        );
        assert_within(&eager, &oracle, against_oracle, &format!("{dtype:?} eager"));
        assert_within(
            &native,
            &eager,
            between,
            &format!("{dtype:?} native vs eager"),
        );
    }
}

/// Native and eager graphs agree for a cached causal suffix (`Q < K`) with GQA groups.
///
/// This is the bottom-right contract that CubeCL flash kernels would violate
/// if `is_causal` were forwarded for non-square planes.
#[test]
fn native_causal_suffix_matches_eager_bottom_right_alignment() {
    let mut problem = Problem::new(1, 4, 2, 4, 9);
    problem.causal = true;
    let native = problem
        .evaluate(AttentionPrecisionMode::BackendDefault)
        .unwrap();
    let eager = problem
        .evaluate(AttentionPrecisionMode::EagerFp32Softmax)
        .unwrap();
    assert_close(&native, &eager, "causal suffix");
    // Row 0 sits at key position 5. Under top-left alignment it would see only
    // key 0, whose context is exactly value row 0 of KV head 0.
    let first_value = &problem.value_values()[..problem.value_dim];
    assert!(
        native[..problem.value_dim]
            .iter()
            .zip(first_value)
            .any(|(actual, value)| (actual - value).abs() > 1e-3)
    );
}

/// One causal query row sees every key, so causality must not change the result.
#[test]
fn single_query_causality_is_a_no_op() {
    for precision in [
        AttentionPrecisionMode::BackendDefault,
        AttentionPrecisionMode::EagerFp32Softmax,
    ] {
        let mut problem = Problem::new(2, 4, 2, 1, 6).with_padding();
        let open = problem.evaluate(precision).unwrap();
        problem.causal = true;
        assert_eq!(problem.evaluate(precision).unwrap(), open);
    }
}

/// More queries than keys leave the leading rows without support; they produce zero context.
#[test]
fn rows_before_the_first_key_produce_zero_context() {
    let mut problem = Problem::new(1, 2, 1, 4, 2);
    problem.causal = true;
    for precision in [
        AttentionPrecisionMode::BackendDefault,
        AttentionPrecisionMode::EagerFp32Softmax,
    ] {
        let output = problem.evaluate(precision).unwrap();
        // Rows 0 and 1 of both heads see no key.
        for head in 0..2 {
            let row_width = problem.value_dim;
            let head_offset = head * problem.queries * row_width;
            assert!(
                output[head_offset..head_offset + 2 * row_width]
                    .iter()
                    .all(|value| *value == 0.0),
                "{precision:?}"
            );
        }
    }
}

/// Learned sinks follow the extra-softmax-column definition in both graphs.
#[test]
fn attention_sinks_match_extra_column_oracle_in_both_graphs() {
    for (query_heads, key_value_heads) in [(2, 2), (4, 2), (6, 3)] {
        for (queries, keys, causal) in [(1, 4, false), (3, 4, true), (4, 4, true), (2, 5, false)] {
            for (padded, biased) in [(false, false), (true, true)] {
                let mut problem = Problem::new(2, query_heads, key_value_heads, queries, keys);
                problem.causal = causal;
                problem.sinks = Some(values(query_heads, 7).iter().map(|v| v * 2.0).collect());
                problem.scale = Some(0.61);
                if padded {
                    problem = problem.with_padding();
                }
                if biased {
                    problem = problem.with_bias();
                }
                let expected = problem.oracle();
                for precision in [
                    AttentionPrecisionMode::BackendDefault,
                    AttentionPrecisionMode::EagerFp32Softmax,
                ] {
                    assert_close(
                        &problem.evaluate(precision).unwrap(),
                        &expected,
                        &format!(
                            "{precision:?} sinks {query_heads}/{key_value_heads} q={queries} k={keys} causal={causal} padded={padded}"
                        ),
                    );
                }
            }
        }
    }
}

/// Output of Hugging Face's gpt-oss `eager_attention_forward` for the sink fixture.
///
/// Pinned with transformers 5.2.0 and torch 2.10.0 (F32, CPU): queries, keys,
/// and values are this module's `values(n, 1 | 2 | 3)` fixtures in
/// `[2, 4, 3, 3]`, `[2, 2, 5, 3]`, and `[2, 2, 5, 2]`; sinks are
/// `[0.4, -1.3, 0.9, 2.2]`; `scaling = 0.61`; and the additive mask is `-inf`
/// on the `Problem::with_padding` pattern plus bottom-right causal links. The
/// reference output `[batch, query, head, value]` is listed here transposed to
/// `[batch, head, query, value]`.
const GPT_OSS_SINK_REFERENCE: [f64; 48] = [
    0.0,
    0.0,
    0.068_462_461,
    -0.209_003_896,
    -0.087_072_886_5,
    0.163_870_886,
    0.0,
    0.0,
    0.111_773_424,
    -0.489_854_485,
    0.023_477_122_2,
    0.057_496_853_2,
    0.0,
    0.0,
    0.153_382_54,
    -0.162_919_626,
    0.087_014_392,
    -0.306_541_92,
    0.0,
    0.0,
    0.146_412_939,
    0.084_311_373_5,
    0.063_471_913_3,
    -0.060_960_695_1,
    -0.042_540_907_9,
    0.074_061_967_4,
    0.077_267_386,
    0.149_545_819,
    0.153_732_3,
    -0.011_979_484,
    -0.558_887_064,
    0.268_511_772,
    -0.397_918_999,
    0.612_480_164,
    -0.215_120_971,
    0.037_361_770_9,
    0.117_741_667,
    0.179_854_989,
    0.009_546_427_99,
    0.313_024_312,
    -0.046_354_945_7,
    0.283_165_902,
    0.010_084_477_3,
    0.053_996_451_2,
    -0.000_787_153_898,
    0.106_649_496,
    0.014_614_789_7,
    0.010_276_235_6,
];

/// Both graphs reproduce the pinned gpt-oss sink attention, including a fully excluded row.
#[test]
fn attention_sinks_match_pinned_gpt_oss_reference() {
    let mut problem = Problem::new(2, 4, 2, 3, 5).with_padding();
    problem.causal = true;
    problem.scale = Some(0.61);
    problem.sinks = Some(vec![0.4, -1.3, 0.9, 2.2]);
    for precision in [
        AttentionPrecisionMode::BackendDefault,
        AttentionPrecisionMode::EagerFp32Softmax,
    ] {
        assert_close(
            &problem.evaluate(precision).unwrap(),
            &GPT_OSS_SINK_REFERENCE.map(|value| value as f32),
            &format!("{precision:?} gpt-oss reference"),
        );
    }
}

/// A row whose keys are all excluded attends only to the zero-value sink.
#[test]
fn sink_absorbs_fully_excluded_rows_and_default_scale_is_supported() {
    let mut problem = Problem::new(2, 2, 1, 2, 3).with_padding();
    problem.sinks = Some(vec![0.5, -1.0]);
    for precision in [
        AttentionPrecisionMode::BackendDefault,
        AttentionPrecisionMode::EagerFp32Softmax,
    ] {
        let output = problem.evaluate(precision).unwrap();
        assert_close(&output, &problem.oracle(), &format!("{precision:?}"));
        // Batch 0 query 0 is fully padded for both heads.
        for head in 0..2 {
            let offset = head * problem.queries * problem.value_dim;
            assert!(
                output[offset..offset + problem.value_dim]
                    .iter()
                    .all(|value| *value == 0.0)
            );
        }
    }
}

/// Eager sink probabilities omit the sink column and sum to `1 - p_sink`.
#[test]
fn eager_sink_probabilities_drop_the_sink_column() {
    let device = Default::default();
    let scores = Tensor::<TestBackend, 4>::from_floats([[[[0.0, 0.0]], [[1.0, -1.0]]]], &device);
    let sinks = Tensor::<TestBackend, 1>::from_floats([0.0, 3.0_f32.ln()], &device);
    let probabilities = eager_probabilities(scores, None, None, Some(sinks))
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    let denominator = 1.0_f32.exp() + (-1.0_f32).exp() + 3.0;
    let expected = [
        1.0 / 3.0,
        1.0 / 3.0,
        1.0_f32.exp() / denominator,
        (-1.0_f32).exp() / denominator,
    ];
    assert_close(&probabilities, &expected, "sink probabilities");
}

/// Native sinks reject softcapping instead of silently capping the sink logit.
#[test]
fn native_sinks_reject_softcap() {
    let mut problem = Problem::new(1, 2, 1, 2, 2);
    problem.sinks = Some(vec![0.0, 0.0]);
    problem.softcap = Some(2.0);
    let error = problem
        .evaluate(AttentionPrecisionMode::BackendDefault)
        .unwrap_err();
    assert!(error.to_string().contains("softcap"));
    // The eager graph appends the sink after softcapping and supports it.
    assert_close(
        &problem
            .evaluate(AttentionPrecisionMode::EagerFp32Softmax)
            .unwrap(),
        &problem.oracle(),
        "eager softcap with sinks",
    );
}

/// Every operand mismatch reports a contextual error before backend dispatch.
#[test]
fn malformed_operands_return_contextual_errors() {
    let device = Default::default();
    let tensor = |shape: [usize; 4]| Tensor::<TestBackend, 4>::zeros(shape, &device);
    let run = |query: [usize; 4],
               key: [usize; 4],
               value: [usize; 4],
               mask: Option<[usize; 4]>,
               bias: Option<[usize; 4]>,
               sinks: Option<usize>| {
        evaluate(
            Operands {
                query: tensor(query),
                key: tensor(key),
                value: tensor(value),
                mask: mask.map(|shape| Tensor::<TestBackend, 4, Bool>::zeros(shape, &device)),
                bias: bias.map(tensor),
                sinks: sinks.map(|heads| Tensor::<TestBackend, 1>::zeros([heads], &device)),
            },
            AttentionModuleOptions::default(),
            AttentionPrecisionMode::EagerFp32Softmax,
        )
        .unwrap_err()
        .to_string()
    };
    let q = [1, 4, 2, 3];
    let kv = [1, 2, 5, 3];
    for (error, expected) in [
        (
            run([0, 4, 2, 3], kv, kv, None, None, None),
            "attention.operands",
        ),
        (
            run(q, [1, 2, 0, 3], [1, 2, 0, 3], None, None, None),
            "attention.operands",
        ),
        (
            run(q, [1, 3, 5, 3], [1, 3, 5, 3], None, None, None),
            "attention.head_grouping",
        ),
        (
            run(q, [1, 2, 5, 4], kv, None, None, None),
            "attention.operands",
        ),
        (
            run(q, kv, [1, 2, 4, 3], None, None, None),
            "attention.operands",
        ),
        (
            run(q, kv, kv, Some([1, 4, 2, 4]), None, None),
            "attention.mask",
        ),
        (
            run(q, kv, kv, None, Some([2, 4, 2, 5]), None),
            "attention.bias",
        ),
        (
            run(q, kv, kv, None, Some([1, 2, 2, 5]), None),
            "attention.bias",
        ),
        (run(q, kv, kv, None, None, Some(2)), "attention.sinks"),
    ] {
        assert!(error.contains(expected), "{error} lacks {expected}");
    }
    let error = repeat_key_value(tensor([1, 2, 1, 3]), 3).unwrap_err();
    assert!(error.to_string().contains("head_grouping"));
}

/// Fully excluded rows have exact-zero probabilities; other rows are normalized.
#[test]
fn fp32_softmax_zeroes_unsupported_rows_and_preserves_dtype() {
    let device = Default::default();
    let scores = Tensor::<TestBackend, 4>::from_floats(
        [[[
            [f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY],
            [0.5, f32::NEG_INFINITY, -0.25],
        ]]],
        &device,
    );
    let probabilities = fp32_softmax(scores.clone())
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    assert_eq!(&probabilities[..4], &[0.0, 0.0, 0.0, probabilities[3]]);
    assert_eq!(probabilities[4], 0.0);
    assert!((probabilities[3] + probabilities[5] - 1.0).abs() < 1e-6);
    let half = fp32_softmax(scores.cast(DType::F16));
    assert_eq!(half.dtype(), DType::F16);
}

/// The group selector marks folded row `j * q + t` with channel `j`.
#[test]
fn group_selector_is_one_hot_by_group_member() {
    let selector = group_selector::<TestBackend>(3, 2, 6, DType::F32, &Default::default())
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    assert_eq!(
        selector,
        [
            1.0, 0.0, 0.0, 1.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
        ]
    );
}

proptest::proptest! {
    #![proptest_config(proptest::test_runner::Config::with_cases(32))]

    /// Folding maps `[b, kv*g + j, t, x]` to `[b, kv, j*q + t, x]` and unfolding inverts it.
    #[test]
    fn folding_relocates_rows_and_round_trips(
        batch in 1usize..=2,
        key_value_heads in 1usize..=3,
        groups in 1usize..=3,
        rows in 1usize..=4,
        width in 1usize..=3,
    ) {
        let query_heads = key_value_heads * groups;
        let shape = [batch, query_heads, rows, width];
        let data = (0..shape.iter().product::<usize>()).map(|index| index as f32).collect::<Vec<_>>();
        let tensor = Tensor::<TestBackend, 4>::from_data(TensorData::new(data.clone(), shape), &Default::default());
        let folded = fold_groups(tensor, key_value_heads);
        proptest::prop_assert_eq!(folded.dims(), [batch, key_value_heads, groups * rows, width]);
        let folded_data = folded.clone().into_data().to_vec::<f32>().unwrap();
        for b in 0..batch {
            for kv in 0..key_value_heads {
                for j in 0..groups {
                    for t in 0..rows {
                        for x in 0..width {
                            let source = ((b * query_heads + kv * groups + j) * rows + t) * width + x;
                            let target = ((b * key_value_heads + kv) * groups * rows + j * rows + t) * width + x;
                            proptest::prop_assert_eq!(folded_data[target], data[source]);
                        }
                    }
                }
            }
        }
        let restored = unfold_groups(folded, query_heads).into_data().to_vec::<f32>().unwrap();
        proptest::prop_assert_eq!(restored, data);
    }

    /// Random grouped problems match the scalar oracle in both graphs.
    #[test]
    fn random_grouped_problems_match_the_oracle(
        key_value_heads in 1usize..=2,
        groups in 1usize..=3,
        queries in 1usize..=4,
        extra_keys in 0usize..=3,
        causal in proptest::bool::ANY,
        padded in proptest::bool::ANY,
        sinks in proptest::bool::ANY,
    ) {
        let query_heads = key_value_heads * groups;
        let mut problem = Problem::new(2, query_heads, key_value_heads, queries, queries + extra_keys);
        problem.causal = causal;
        if padded {
            problem = problem.with_padding();
        }
        if sinks {
            problem.sinks = Some(values(query_heads, 9));
        }
        let expected = problem.oracle();
        for precision in [AttentionPrecisionMode::BackendDefault, AttentionPrecisionMode::EagerFp32Softmax] {
            let actual = problem.evaluate(precision).unwrap();
            for (actual, expected) in actual.iter().zip(&expected) {
                proptest::prop_assert!((actual - expected).abs() <= 2e-5 + 2e-5 * expected.abs());
            }
        }
    }
}

/// Gradients reach queries, native keys, and native values through both folded graphs.
#[cfg(feature = "autodiff")]
#[test]
fn folded_attention_is_differentiable() {
    use burn::backend::Autodiff;
    /// Differentiable CPU backend for the folded-attention gradient smoke test.
    type Ad = Autodiff<TestBackend>;
    let device = Default::default();
    for precision in [
        AttentionPrecisionMode::BackendDefault,
        AttentionPrecisionMode::EagerFp32Softmax,
    ] {
        let query =
            Tensor::<Ad, 4>::from_data(TensorData::new(values(16, 1), [1, 4, 2, 2]), &device)
                .require_grad();
        let key = Tensor::<Ad, 4>::from_data(TensorData::new(values(12, 2), [1, 2, 3, 2]), &device)
            .require_grad();
        let value =
            Tensor::<Ad, 4>::from_data(TensorData::new(values(12, 3), [1, 2, 3, 2]), &device)
                .require_grad();
        let output = grouped_query_attention(
            query.clone(),
            key.clone(),
            value.clone(),
            None,
            None,
            AttentionModuleOptions {
                is_causal: true,
                ..Default::default()
            },
            precision,
        )
        .unwrap();
        let gradients = (output.clone() * output).sum().backward();
        for gradient in [
            query.grad(&gradients),
            key.grad(&gradients),
            value.grad(&gradients),
        ] {
            let gradient = gradient
                .expect("gradient exists")
                .into_data()
                .to_vec::<f32>()
                .unwrap();
            assert!(gradient.iter().all(|value| value.is_finite()));
            assert!(gradient.iter().any(|value| *value != 0.0));
        }
    }
}

/// Measures decode-shaped GQA attention: folded native KV versus explicit repetition.
///
/// Run with `cargo test --release --lib grouped_decode_attention_benchmark -- --ignored --nocapture`.
#[test]
#[ignore = "timing benchmark; run explicitly in release mode"]
fn grouped_decode_attention_benchmark() {
    use std::time::Instant;
    let device = Default::default();
    let (batch, query_heads, key_value_heads, keys, head_dim) = (1, 32, 4, 4096, 128);
    let query = Tensor::<TestBackend, 4>::random(
        [batch, query_heads, 1, head_dim],
        burn::tensor::Distribution::Default,
        &device,
    );
    let key = Tensor::<TestBackend, 4>::random(
        [batch, key_value_heads, keys, head_dim],
        burn::tensor::Distribution::Default,
        &device,
    );
    let value = key.clone() * 0.5;
    let mask = Tensor::<TestBackend, 4, Bool>::zeros([batch, query_heads, 1, keys], &device);
    let iterations = 20;
    for precision in [
        AttentionPrecisionMode::BackendDefault,
        AttentionPrecisionMode::EagerFp32Softmax,
    ] {
        let time = |folded: bool| {
            let started = Instant::now();
            for _ in 0..iterations {
                let output = if folded {
                    grouped_query_attention(
                        query.clone(),
                        key.clone(),
                        value.clone(),
                        Some(mask.clone()),
                        None,
                        AttentionModuleOptions::default(),
                        precision,
                    )
                    .unwrap()
                } else {
                    grouped_query_attention(
                        query.clone(),
                        repeat_key_value(key.clone(), query_heads).unwrap(),
                        repeat_key_value(value.clone(), query_heads).unwrap(),
                        Some(mask.clone()),
                        None,
                        AttentionModuleOptions::default(),
                        precision,
                    )
                    .unwrap()
                };
                let _ = output.into_data();
            }
            started.elapsed().as_secs_f64() * 1e3 / f64::from(iterations)
        };
        // Warm both paths once before timing.
        let _ = (time(true), time(false));
        let (folded, repeated) = (time(true), time(false));
        println!(
            "{precision:?} decode q=1 k={keys} heads={query_heads}/{key_value_heads} d={head_dim}: folded {folded:.3} ms, repeated {repeated:.3} ms, speedup {:.2}x",
            repeated / folded
        );
    }
}

/// Autotuned Metal kernels keep bottom-right causal alignment for a cached suffix.
///
/// CubeCL flash kernels align `is_causal` top-left, so this guards the
/// explicit-mask dispatch for `Q < K`. Long keys make the autotuner consider
/// flash candidates. Run serially on a Metal host with
/// `cargo test --release --features attention-kernel-tracing --lib
/// attention::dispatch::tests::metal_causal_suffix_matches_eager_reference
/// -- --ignored --exact --nocapture --test-threads=1`.
#[cfg(all(feature = "attention-kernel-tracing", target_os = "macos"))]
#[test]
#[ignore = "requires a Metal-capable macOS host and serialized attention autotuning"]
fn metal_causal_suffix_matches_eager_reference() {
    use burn::backend::Metal;
    use burn::tensor::Distribution;
    /// Autotuned Metal backend with F32 activations.
    type Gpu = Metal<f32, i32>;
    let device = Default::default();
    let (queries, keys, query_heads, key_value_heads, head_dim) = (64, 4097, 4, 2, 64);
    let random =
        |shape: [usize; 4]| Tensor::<Gpu, 4>::random(shape, Distribution::Default, &device);
    let query = random([1, query_heads, queries, head_dim]);
    let key = random([1, key_value_heads, keys, head_dim]);
    let value = random([1, key_value_heads, keys, head_dim]);
    let options = AttentionModuleOptions {
        is_causal: true,
        ..Default::default()
    };
    let run = |precision| {
        let output = grouped_query_attention(
            query.clone(),
            key.clone(),
            value.clone(),
            None,
            None,
            options,
            precision,
        )
        .unwrap();
        <Gpu as burn::tensor::backend::Backend>::sync(&device)
            .expect("synchronize Metal attention");
        output.into_data().to_vec::<f32>().unwrap()
    };
    // Warm the autotuner before the compared call.
    let _ = run(AttentionPrecisionMode::BackendDefault);
    let native = run(AttentionPrecisionMode::BackendDefault);
    let eager = run(AttentionPrecisionMode::EagerFp32Softmax);
    let maximum = native
        .iter()
        .zip(&eager)
        .map(|(native, eager)| (native - eager).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        maximum < 1e-3,
        "native and eager causal suffixes differ by {maximum}"
    );
}

/// Rounds F32 values to a 16-bit float `dtype` with Flex's own cast, and widens them back.
fn cast_through(dtype: DType, values: &[f32]) -> Vec<f32> {
    Tensor::<TestBackend, 1>::from_data(
        TensorData::new(values.to_vec(), [values.len()]),
        &Default::default(),
    )
    .cast(dtype)
    .cast(DType::F32)
    .into_data()
    .to_vec::<f32>()
    .unwrap()
}

/// Half-precision operands `[1, 1, rows, 128]` with values exactly representable in F16 and BF16.
fn half_operand(rows: usize, seed: usize, dtype: DType) -> Tensor<TestBackend, 4> {
    let data = (0..rows * 128)
        .map(|index| ((index * 53 + seed * 29) % 61) as f32 / 32.0 - 0.9375)
        .collect::<Vec<_>>();
    Tensor::<TestBackend, 4>::from_data(
        TensorData::new(data, [1, 1, rows, 128]),
        &Default::default(),
    )
    .cast(dtype)
}

/// Eager half-precision scores are the rounded `Q K^T` times the unrounded scale, rounded once, as in Transformers.
///
/// Transformers' eager attention computes `torch.matmul(query, key.T) *
/// scaling` with `scaling` a Python float, which PyTorch's CPU kernel
/// multiplies in F32 and rounds once: `x * s` equals `(x.float() * s).to(dtype)`
/// on every finite BF16 and F16 `x` (the review round's torch probe; MPS
/// agrees except that it flushes BF16 subnormals). With 128-wide heads
/// `scaling = 1 / sqrt(128)` is exact in neither format, so rounding it to
/// the dtype first, as Burn's `tensor * scalar` does, moves some scores by
/// one unit in the last place; this test detects that. The matmul's own
/// rounding is read back from the graph's product
/// ([`f32_accumulated_matmul`]), so only the scale's rounding point is under
/// test.
#[test]
fn half_precision_eager_scores_multiply_by_the_unrounded_scale() {
    let scale = default_scale(128);
    for dtype in [DType::BF16, DType::F16] {
        let (query, key) = (half_operand(3, 1, dtype), half_operand(64, 2, dtype));
        let product = f32_accumulated_matmul(query.clone(), key.clone().swap_dims(2, 3))
            .cast(DType::F32)
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let unrounded = product
            .iter()
            .map(|value| value * scale as f32)
            .collect::<Vec<_>>();
        let expected = cast_through(dtype, &unrounded);
        let scores = scaled_scores(query.clone(), key.clone(), scale, None);
        assert_eq!(scores.dtype(), dtype);
        let actual = scores.cast(DType::F32).into_data().to_vec::<f32>().unwrap();
        assert_eq!(
            actual, expected,
            "{dtype:?}: the unrounded scale, rounded once"
        );
        // The scale rounded to the dtype first gives other scores on these operands.
        let rounded_scale = cast_through(dtype, &[scale as f32])[0];
        let rounded = cast_through(
            dtype,
            &product
                .iter()
                .map(|value| value * rounded_scale)
                .collect::<Vec<_>>(),
        );
        assert_ne!(
            rounded, expected,
            "{dtype:?}: the test detects a rounded scale"
        );
        // F32 keeps Burn's own graph bit for bit.
        let wide = scaled_scores(query.cast(DType::F32), key.cast(DType::F32), scale, None);
        let wide_product = half_operand(3, 1, DType::F32)
            .matmul(half_operand(64, 2, DType::F32).swap_dims(2, 3))
            * scale;
        assert_eq!(wide.into_data(), wide_product.into_data());
    }
}

/// The fused half-precision path equals F32 attention of the same operands rounded once, like PyTorch's SDPA.
///
/// PyTorch's fused `scaled_dot_product_attention` on half-precision operands
/// accumulates scores, softmax, and the value product in F32 and rounds the
/// output once. On Flex, Burn's fused `attention` does the same: its F16 and
/// BF16 output equals its F32 output on the widened operands, cast once.
#[test]
fn half_precision_fused_attention_is_the_f32_attention_rounded_once() {
    let problem = Problem {
        head_dim: 16,
        value_dim: 8,
        causal: true,
        ..Problem::new(1, 4, 2, 5, 7)
    };
    for dtype in [DType::BF16, DType::F16] {
        let (query, key, value, mask, bias, _) = problem.tensors_in(dtype);
        let operands = |query: Tensor<TestBackend, 4>,
                        key: Tensor<TestBackend, 4>,
                        value: Tensor<TestBackend, 4>| Operands {
            query,
            key,
            value,
            mask: mask.clone(),
            bias: bias.clone(),
            sinks: None,
        };
        let half = evaluate(
            operands(query.clone(), key.clone(), value.clone()),
            problem.options(),
            AttentionPrecisionMode::BackendDefault,
        )
        .unwrap();
        assert_eq!(half.dtype(), dtype);
        let wide = evaluate(
            operands(
                query.cast(DType::F32),
                key.cast(DType::F32),
                value.cast(DType::F32),
            ),
            problem.options(),
            AttentionPrecisionMode::BackendDefault,
        )
        .unwrap();
        assert_eq!(
            half.cast(DType::F32).into_data(),
            wide.cast(dtype).cast(DType::F32).into_data(),
            "{dtype:?}"
        );
    }
}

/// One grouped eager problem whose exact sums are exact in F32 but not in F16 or BF16.
///
/// Queries and keys are multiples of `1/64` in `[-0.75, 0.75]` with eight
/// channels, so every `Q K^T` partial sum is a multiple of `2^-12` below
/// `4.5` in magnitude: exact in F32, but not in F16 (whose spacing is
/// `2^-11` from `0.5`) or BF16. Values are multiples of `1/8` in `[-1, 1]`
/// and the bias lies in `[-0.45, 0.45]`, so with the score scale
/// `1 / sqrt(8)` every probability-times-value sum is exact in F32 too
/// ([`EagerGrid::emulate`] asserts both). Four query heads share two key/value heads; three queries
/// over five keys make the causal plane non-square (bottom-right aligned),
/// and the explicit mask excludes every key of batch 0, head 1, query 0 (a
/// fully masked row) and one more link.
struct EagerGrid {
    /// Queries `[2, 4, 3, 8]`.
    query: Vec<f32>,
    /// Keys `[2, 2, 5, 8]`.
    key: Vec<f32>,
    /// Values `[2, 2, 5, 4]`.
    value: Vec<f32>,
    /// Explicit exclusions `[2, 4, 3, 5]` (`true == excluded`), before causality.
    mask: Vec<bool>,
    /// Additive bias `[1, 4, 3, 5]`, broadcast over the batch.
    bias: Vec<f32>,
}

/// One rounding point of [`EagerGrid::emulate`] moved elsewhere, to show the grid detects it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EagerVariant {
    /// The five rounding points of Transformers' eager attention.
    Reference,
    /// The score product `Q K^T` keeps dtype partial sums (`acc = round(acc + q k)`).
    ScorePartialSums,
    /// The value product `P V` keeps dtype partial sums.
    ValuePartialSums,
    /// The bias is added to the widened scores, without rounding the sum.
    WideBias,
    /// Softmax runs in the dtype instead of F32.
    DtypeSoftmax,
}

impl EagerGrid {
    /// Batch rows.
    const BATCH: usize = 2;
    /// Query heads.
    const QUERY_HEADS: usize = 4;
    /// Native key/value heads.
    const KEY_VALUE_HEADS: usize = 2;
    /// Query tokens.
    const QUERIES: usize = 3;
    /// Key tokens.
    const KEYS: usize = 5;
    /// Query/key channels.
    const HEAD_DIM: usize = 8;
    /// Value channels.
    const VALUE_DIM: usize = 4;

    /// Builds the deterministic grid operands.
    fn new() -> Self {
        // `steps` multiples of `half_range / steps` on either side of zero.
        let grid = |count: usize, seed: usize, steps: usize, half_range: f32| {
            (0..count)
                .map(|index| {
                    let step = (index * 29 + seed * 13) % (2 * steps + 1);
                    (step as f32 - steps as f32) * half_range / steps as f32
                })
                .collect::<Vec<_>>()
        };
        let (batch, heads, queries, keys) =
            (Self::BATCH, Self::QUERY_HEADS, Self::QUERIES, Self::KEYS);
        let mut mask = vec![false; batch * heads * queries * keys];
        // Batch 0, head 1, query 0: every key excluded.
        mask[keys * queries..keys * queries + keys].fill(true);
        // Batch 1, head 2, query 2, key 4: one more excluded link.
        mask[((heads + 2) * queries + 2) * keys + 4] = true;
        Self {
            query: grid(batch * heads * queries * Self::HEAD_DIM, 1, 48, 0.75),
            key: grid(
                batch * Self::KEY_VALUE_HEADS * keys * Self::HEAD_DIM,
                2,
                48,
                0.75,
            ),
            value: grid(
                batch * Self::KEY_VALUE_HEADS * keys * Self::VALUE_DIM,
                3,
                8,
                1.0,
            ),
            mask,
            bias: grid(heads * queries * keys, 4, 16, 0.45),
        }
    }

    /// The operands cast to `dtype` on Flex, with the explicit mask.
    fn operands(&self, dtype: DType) -> Operands<TestBackend> {
        self.operands_on(dtype, &Default::default())
    }

    /// The operands cast to `dtype` on `device`, with the explicit mask.
    fn operands_on<B: Backend>(&self, dtype: DType, device: &B::Device) -> Operands<B> {
        let tensor = |values: &[f32], shape: [usize; 4]| {
            Tensor::<B, 4>::from_data(TensorData::new(values.to_vec(), shape), device).cast(dtype)
        };
        Operands {
            query: tensor(
                &self.query,
                [
                    Self::BATCH,
                    Self::QUERY_HEADS,
                    Self::QUERIES,
                    Self::HEAD_DIM,
                ],
            ),
            key: tensor(
                &self.key,
                [
                    Self::BATCH,
                    Self::KEY_VALUE_HEADS,
                    Self::KEYS,
                    Self::HEAD_DIM,
                ],
            ),
            value: tensor(
                &self.value,
                [
                    Self::BATCH,
                    Self::KEY_VALUE_HEADS,
                    Self::KEYS,
                    Self::VALUE_DIM,
                ],
            ),
            mask: Some(Tensor::<B, 4, Bool>::from_data(
                TensorData::new(
                    self.mask.clone(),
                    [Self::BATCH, Self::QUERY_HEADS, Self::QUERIES, Self::KEYS],
                ),
                device,
            )),
            bias: Some(tensor(
                &self.bias,
                [1, Self::QUERY_HEADS, Self::QUERIES, Self::KEYS],
            )),
            sinks: None,
        }
    }

    /// Whether link `(batch, head, query, key)` is excluded, by the mask or by bottom-right causality.
    fn excluded(&self, batch: usize, head: usize, query: usize, key: usize) -> bool {
        let index = ((batch * Self::QUERY_HEADS + head) * Self::QUERIES + query) * Self::KEYS + key;
        self.mask[index] || key > query + Self::KEYS - Self::QUERIES
    }

    /// `[batch, head, row, column]` sums `Σ_i left(row, i) right(i, column)` over grouped heads.
    ///
    /// `left` is `[batch, query_head, rows, inner]` and `right` is
    /// `[batch, kv_head, inner, columns]` read through `right_index(kv, i,
    /// column)`. A reference sum is formed exactly in F64 and asserted exact
    /// in F32, so its summation order cannot matter; with `partial` every
    /// partial sum is rounded by it instead.
    fn grouped_sums(
        left: &[f32],
        right: impl Fn(usize, usize, usize, usize) -> f32,
        [rows, inner, columns]: [usize; 3],
        partial: Option<&dyn Fn(f32) -> f32>,
    ) -> Vec<f32> {
        let groups = Self::QUERY_HEADS / Self::KEY_VALUE_HEADS;
        let mut sums = Vec::with_capacity(Self::BATCH * Self::QUERY_HEADS * rows * columns);
        for batch in 0..Self::BATCH {
            for head in 0..Self::QUERY_HEADS {
                for row in 0..rows {
                    for column in 0..columns {
                        let term = |index: usize| {
                            f64::from(
                                left[((batch * Self::QUERY_HEADS + head) * rows + row) * inner
                                    + index],
                            ) * f64::from(right(batch, head / groups, index, column))
                        };
                        sums.push(match partial {
                            Some(round) => (0..inner)
                                .fold(0.0_f32, |sum, index| round(sum + term(index) as f32)),
                            None => {
                                let sum = (0..inner).map(term).sum::<f64>();
                                assert_eq!(f64::from(sum as f32), sum, "{sum} is not exact in F32");
                                sum as f32
                            }
                        });
                    }
                }
            }
        }
        sums
    }

    /// Transformers' eager graph in `dtype` (or a [`EagerVariant`] of it), emulated on the host.
    ///
    /// Returns the probabilities `[batch, head, query, key]` and the output
    /// `[batch, head, query, value]`, widened to F32. `round` is Flex's own
    /// F32-to-`dtype` cast ([`cast_through`]), so the reference emulation
    /// fixes the rounding points and nothing else:
    ///
    /// - `p_qk = round(Σ q k)`: the matrix product, an exact sum rounded once
    ///   (`torch.matmul` accumulates half-precision operands in F32);
    /// - `s = round(p_qk * f32(scale))`: a Python-float scale in F32 opmath;
    /// - `t = round(s + bias)`, `-inf` where excluded: Transformers adds its
    ///   additive mask in the dtype;
    /// - `p = round(softmax_f32(t))`, zero on a fully excluded row (the
    ///   crate's definition): `softmax(..., dtype=torch.float32).to(dtype)`;
    /// - `out = round(Σ p v)`: the value product, an exact sum rounded once.
    ///
    /// The F32 softmax itself is Burn's (`activation::softmax` of an F32
    /// tensor of the emulated `t`): two F32 softmax implementations may differ
    /// in the last F32 bit, which is no rounding point.
    fn emulate(&self, dtype: DType, variant: EagerVariant) -> (Vec<f32>, Vec<f32>) {
        let (batch, heads, queries, keys) =
            (Self::BATCH, Self::QUERY_HEADS, Self::QUERIES, Self::KEYS);
        let round = |values: &[f32]| cast_through(dtype, values);
        let round_one = |value: f32| match dtype {
            DType::BF16 => burn::tensor::bf16::from_f32(value).to_f32(),
            _ => burn::tensor::f16::from_f32(value).to_f32(),
        };
        let partial =
            |moved: EagerVariant| (variant == moved).then_some(&round_one as &dyn Fn(f32) -> f32);
        let (query, key, value, bias) = (
            round(&self.query),
            round(&self.key),
            round(&self.value),
            round(&self.bias),
        );
        let products = Self::grouped_sums(
            &query,
            |batch, kv, channel, column| {
                key[((batch * Self::KEY_VALUE_HEADS + kv) * keys + column) * Self::HEAD_DIM
                    + channel]
            },
            [queries, Self::HEAD_DIM, keys],
            partial(EagerVariant::ScorePartialSums),
        );
        let scale = default_scale(Self::HEAD_DIM) as f32;
        let scores = round(
            &round(&products)
                .iter()
                .map(|product| product * scale)
                .collect::<Vec<_>>(),
        );
        let biased = scores
            .iter()
            .enumerate()
            .map(|(index, score)| score + bias[index % (heads * queries * keys)])
            .collect::<Vec<_>>();
        let mut biased = match variant {
            EagerVariant::WideBias => biased,
            _ => round(&biased),
        };
        let mut fully_excluded = vec![false; batch * heads * queries];
        for b in 0..batch {
            for h in 0..heads {
                for q in 0..queries {
                    let row = (b * heads + h) * queries + q;
                    for k in 0..keys {
                        if self.excluded(b, h, q, k) {
                            biased[row * keys + k] = f32::NEG_INFINITY;
                        }
                    }
                    fully_excluded[row] = (0..keys).all(|k| self.excluded(b, h, q, k));
                    if fully_excluded[row] {
                        // The graph zeroes unsupported rows before and after normalization.
                        biased[row * keys..(row + 1) * keys].fill(0.0);
                    }
                }
            }
        }
        let shape = [batch, heads, queries, keys];
        let wide = Tensor::<TestBackend, 4>::from_data(
            TensorData::new(biased, shape),
            &Default::default(),
        );
        let softmax = match variant {
            EagerVariant::DtypeSoftmax => {
                burn::tensor::activation::softmax(wide.cast(dtype), 3).cast(DType::F32)
            }
            _ => burn::tensor::activation::softmax(wide, 3),
        };
        let mut probabilities = round(&softmax.into_data().to_vec::<f32>().unwrap());
        for (row, &excluded) in fully_excluded.iter().enumerate() {
            if excluded {
                probabilities[row * keys..(row + 1) * keys].fill(0.0);
            }
        }
        let output = Self::grouped_sums(
            &probabilities,
            |batch, kv, key, channel| {
                value[((batch * Self::KEY_VALUE_HEADS + kv) * keys + key) * Self::VALUE_DIM
                    + channel]
            },
            [queries, keys, Self::VALUE_DIM],
            partial(EagerVariant::ValuePartialSums),
        );
        (probabilities, round(&output))
    }
}

/// Transformers' `eager_attention_forward` on [`EagerGrid`], as BF16 bits `[batch, head, query, value]`.
///
/// Pinned with transformers 5.2.0 and torch 2.10.0 on the CPU by
/// `acceptance/measurements/half-precision-review-20261018/probes/probe_torch_half.py`
/// (Llama's eager function, excluded links at `finfo(bfloat16).min` as in
/// Transformers' masks, `scaling = 8 ** -0.5`). Elements 12 to 15 are the
/// fully masked row, which Transformers averages over every key and the crate
/// defines as zero.
const TRANSFORMERS_EAGER_BF16: [u16; 96] = [
    0x3cc4, 0xbddc, 0x3e17, 0xbef4, 0xbd8c, 0x3ddb, 0xbc48, 0xbe31, 0x3e54, 0xbe04, 0xbed9, 0x3cfe,
    0x3e1a, 0xbd4d, 0xbe80, 0xbccd, 0x3ea3, 0x3dfe, 0xbe39, 0xbe04, 0x3e88, 0xbcb9, 0xbe44, 0xbe5d,
    0xbee0, 0x3e5d, 0x3ee0, 0xbe40, 0x3c60, 0x3e9a, 0x3db2, 0xbf0a, 0xbd8a, 0x3d04, 0x3e44, 0xbe52,
    0xbf11, 0x3ef1, 0x3e9e, 0xbea2, 0xbeae, 0x3e91, 0x3e4e, 0xbed9, 0x3ca9, 0x3e30, 0x3da0, 0xbe8c,
    0xbe8f, 0xbda9, 0x3f18, 0xbcf0, 0xbed7, 0xbd26, 0x3ee9, 0xbe2e, 0x3cde, 0xbdc1, 0x3e17, 0x3e8e,
    0xbe64, 0xbdd6, 0x3f27, 0x3ce4, 0xbf11, 0x3e41, 0x3e9d, 0xbea3, 0xbdb0, 0xbd20, 0x3e67, 0x3e28,
    0x3de2, 0xbf03, 0xbe26, 0x3eb8, 0xbe82, 0xbe03, 0x3e7b, 0xbb88, 0xbe2c, 0xbef0, 0x3e23, 0x3da8,
    0xbd52, 0xbf2d, 0x3e0a, 0x3e4c, 0xbe0b, 0xbec5, 0x3e64, 0x3dea, 0x3d54, 0xbe42, 0xbdfc, 0x3d98,
];

/// Transformers' `eager_attention_forward` on [`EagerGrid`] in F16, pinned like [`TRANSFORMERS_EAGER_BF16`].
const TRANSFORMERS_EAGER_F16: [u16; 96] = [
    0x260a, 0xaee2, 0x30bc, 0xb7a2, 0xac59, 0x2ebf, 0xa206, 0xb187, 0x32a4, 0xb029, 0xb6ca, 0x27f4,
    0x30cc, 0xaa66, 0xb400, 0xa666, 0x3520, 0x2ff9, 0xb1cc, 0xb01d, 0x3437, 0xa5cc, 0xb21f, 0xb2e1,
    0xb6ff, 0x32dc, 0x3701, 0xb1fe, 0x22e6, 0x34d1, 0x2d90, 0xb84e, 0xac51, 0x283d, 0x321d, 0xb294,
    0xb889, 0x378a, 0x34ef, 0xb512, 0xb571, 0x3488, 0x3272, 0xb6c8, 0x2534, 0x3185, 0x2cfd, 0xb460,
    0xb479, 0xad54, 0x38c4, 0xa78c, 0xb6b8, 0xa922, 0x3748, 0xb171, 0x26f8, 0xae09, 0x30b3, 0x346f,
    0xb319, 0xaebe, 0x393a, 0x2738, 0xb88b, 0x320a, 0x34e9, 0xb516, 0xad78, 0xa8fa, 0x3334, 0x3144,
    0x2f1a, 0xb81c, 0xb136, 0x35c6, 0xb413, 0xb014, 0x33d9, 0x9ca4, 0xb156, 0xb77e, 0x310e, 0x2d55,
    0xaa84, 0xb968, 0x304e, 0x325f, 0xb059, 0xb629, 0x3320, 0x2f4f, 0x2a8e, 0xb212, 0xafd0, 0x2cc0,
];

/// Counts the elements whose bits differ.
fn differing_bits(actual: &[f32], expected: &[f32]) -> usize {
    assert_eq!(actual.len(), expected.len());
    actual
        .iter()
        .zip(expected)
        .filter(|(actual, expected)| actual.to_bits() != expected.to_bits())
        .count()
}

/// Eager half-precision attention rounds where Transformers' eager attention does, on grouped heads with causal and fully masked rows.
///
/// [`EagerGrid::emulate`] lists the five rounding points. The probabilities
/// ([`eager_probabilities`] of [`scaled_scores`], the hook path's pattern)
/// and the output of [`eager`] must equal the emulation bit for bit in BF16
/// and F16, and the output must equal Transformers' own
/// `eager_attention_forward` ([`TRANSFORMERS_EAGER_BF16`],
/// [`TRANSFORMERS_EAGER_F16`]) on every row with a visible key. Each
/// [`EagerVariant`] must differ from the emulation, so the grid detects dtype
/// partial sums in either product, a bias added after widening, and a softmax
/// in the dtype. On Flex the F16 products are formed in F32
/// ([`f32_accumulated_matmul`]); Flex's own F16 GEMM, which keeps F16 partial
/// sums, fails this test in either product.
#[test]
fn half_precision_eager_attention_rounds_like_transformers_eager() {
    let problem = EagerGrid::new();
    let options = AttentionModuleOptions {
        scale: None,
        softcap: None,
        is_causal: true,
    };
    let device = Default::default();
    for dtype in [DType::BF16, DType::F16] {
        let (probabilities, output) = problem.emulate(dtype, EagerVariant::Reference);
        let operands = problem.operands(dtype);
        let geometry = operands.geometry().unwrap();
        let exclusion = union_exclusions(
            operands.mask.clone(),
            causal_exclusion(true, geometry, &device).unwrap(),
        );
        let pattern = eager_probabilities(
            scaled_scores(
                operands.query.clone(),
                operands.key.clone(),
                default_scale(EagerGrid::HEAD_DIM),
                None,
            ),
            exclusion.as_ref(),
            operands.bias.clone(),
            None,
        );
        assert_eq!(pattern.dtype(), dtype);
        let pattern = pattern
            .cast(DType::F32)
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        assert_eq!(
            differing_bits(&pattern, &probabilities),
            0,
            "{dtype:?} probabilities\nactual   {pattern:?}\nexpected {probabilities:?}"
        );
        let actual = eager(operands, options, geometry).unwrap();
        assert_eq!(actual.dtype(), dtype);
        let actual = actual.cast(DType::F32).into_data().to_vec::<f32>().unwrap();
        assert_eq!(
            differing_bits(&actual, &output),
            0,
            "{dtype:?} output\nactual   {actual:?}\nexpected {output:?}"
        );
        // The fully excluded row is zero; every other output is Transformers' own.
        let row = EagerGrid::QUERIES * EagerGrid::VALUE_DIM;
        assert!(
            actual[row..row + EagerGrid::VALUE_DIM]
                .iter()
                .all(|&value| value == 0.0)
        );
        let pinned = match dtype {
            DType::BF16 => {
                TRANSFORMERS_EAGER_BF16.map(|bits| burn::tensor::bf16::from_bits(bits).to_f32())
            }
            _ => TRANSFORMERS_EAGER_F16.map(|bits| burn::tensor::f16::from_bits(bits).to_f32()),
        };
        let supported =
            |values: &[f32]| [&values[..row], &values[row + EagerGrid::VALUE_DIM..]].concat();
        assert_eq!(
            differing_bits(&supported(&actual), &supported(&pinned)),
            0,
            "{dtype:?}: the output differs from Transformers' eager_attention_forward"
        );
        for variant in [
            EagerVariant::ScorePartialSums,
            EagerVariant::ValuePartialSums,
            EagerVariant::WideBias,
            EagerVariant::DtypeSoftmax,
        ] {
            let (other_probabilities, other_output) = problem.emulate(dtype, variant);
            let moved = differing_bits(&other_probabilities, &probabilities)
                + differing_bits(&other_output, &output);
            eprintln!("{dtype:?} {variant:?}: {moved} probabilities and outputs move");
            assert!(moved > 0, "{dtype:?}: the grid does not detect {variant:?}");
        }
    }
}

/// Metal's eager F16 attention (Muse-Glimmer's half-precision default) equals Transformers' eager attention on [`EagerGrid`].
///
/// [`f32_accumulated_matmul`] keeps Metal's own F16 products, which are F32
/// sums rounded once (`operations_tests.rs`,
/// `metal_f16_matmul_is_an_f32_sum_rounded_once`), so on Metal the eager graph
/// must already round where Transformers' `eager_attention_forward` does: its
/// output must equal [`TRANSFORMERS_EAGER_F16`] on every row with a visible
/// key, and the fully masked row must be zero. Burn 0.21 cannot store BF16 on
/// Metal. Run serially on a Metal host with `cargo test --release --features
/// metal --lib
/// attention::dispatch::tests::metal_half_precision_eager_attention_matches_transformers_eager
/// -- --ignored --exact --nocapture --test-threads=1`.
#[cfg(all(
    feature = "metal",
    target_os = "macos",
    not(any(feature = "rocm", feature = "attention-kernel-tracing"))
))]
#[test]
#[ignore = "requires a Metal-capable macOS host"]
fn metal_half_precision_eager_attention_matches_transformers_eager() {
    use burn::backend::Metal;

    let device = Default::default();
    let problem = EagerGrid::new();
    let operands = problem.operands_on::<Metal>(DType::F16, &device);
    let geometry = operands.geometry().unwrap();
    let options = AttentionModuleOptions {
        scale: None,
        softcap: None,
        is_causal: true,
    };
    let actual = eager(operands, options, geometry)
        .unwrap()
        .cast(DType::F32)
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    let pinned = TRANSFORMERS_EAGER_F16.map(|bits| burn::tensor::f16::from_bits(bits).to_f32());
    let row = EagerGrid::QUERIES * EagerGrid::VALUE_DIM;
    let supported =
        |values: &[f32]| [&values[..row], &values[row + EagerGrid::VALUE_DIM..]].concat();
    let differing = differing_bits(&supported(&actual), &supported(&pinned));
    eprintln!(
        "Metal F16 eager attention: {differing} of {} supported outputs differ from Transformers' eager_attention_forward; fully masked row {:?}",
        actual.len() - EagerGrid::VALUE_DIM,
        &actual[row..row + EagerGrid::VALUE_DIM]
    );
    assert_eq!(differing, 0);
    assert!(
        actual[row..row + EagerGrid::VALUE_DIM]
            .iter()
            .all(|&value| value == 0.0)
    );
}
