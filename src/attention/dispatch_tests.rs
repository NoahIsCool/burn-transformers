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

    /// Builds Burn tensors for the problem.
    #[allow(clippy::type_complexity)]
    fn tensors(
        &self,
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
            Tensor::<TestBackend, 4>::from_data(TensorData::new(data, shape), &device)
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
                Tensor::from_data(TensorData::new(sinks, [self.query_heads]), &device)
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
        let (query, key, value, mask, bias, sinks) = self.tensors();
        Ok(evaluate(
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
        )?
        .into_data()
        .to_vec::<f32>()
        .unwrap())
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
        let (query, key, value, mask, bias, _) = self.tensors();
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
        output.into_data().to_vec::<f32>().unwrap()
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
    let selector = group_selector::<TestBackend>(3, 2, DType::F32, &Default::default())
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
