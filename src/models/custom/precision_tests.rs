//! End-to-end F64 decoder execution: dtype preservation, F32 parity, and F64-only invariants.
//!
//! The fixture is a two-block decoder-only model with RMS pre-normalization,
//! grouped-query attention with Q/K RMS normalization and RoPE, a SwiGLU
//! feed-forward branch, final RMS normalization, and an untied vocabulary head.
//! Its F32 parameters are widened exactly to F64, so both models compute the
//! same function and differ only in arithmetic precision.

use burn::{
    module::{Module, ModuleMapper, Param},
    tensor::{DType, Tensor, TensorData},
};

use super::{ModularCausalLm, TestBackend, attention, decoder, layer};
use crate::{
    CausalLmInput, ForwardOptions, LogitsSelection,
    architecture::{DecoderSpec, PositionEncodingSpec, RopeScaling},
    attention::AttentionPrecisionMode,
    intervention::{Capture, Hooks, Layer, Stack},
    precision::test_support::{
        CastParameters, ROTARY_SHIFT, assert_matches_f64_reference, assert_rotary_shift_invariant,
        assert_rotary_shift_sensitive, f64_values, host_masked_softmax, host_rms_norm,
        scaled_max_difference,
    },
};

/// Residual width of the fixture decoder.
const HIDDEN: usize = 8;

/// Per-head query/key width, including the Q/K normalization width.
const HEAD_DIM: usize = 4;

/// Number of tokens per row in the fixture inputs.
const TOKENS: usize = 6;

/// RMS normalization epsilon used by every norm in the fixture specification.
const EPSILON: f64 = 1e-6;

/// Effective scale of channel `c` in every RMS normalization of the fixture.
///
/// Scales depend only on the channel, so width-8 residual norms and width-4 Q/K
/// norms share a host-side closed form while still differing from one.
fn norm_scale(channel: usize) -> f32 {
    1.0 + 0.25 * ((channel + 1) as f32 * 0.7).sin()
}

/// Deterministic non-dyadic F32 matrices and non-unit channel-dependent norm scales.
#[derive(Default)]
struct NonUnitParameters(
    /// Number of floating parameters already replaced.
    usize,
);

impl ModuleMapper<TestBackend> for NonUnitParameters {
    /// Replaces one parameter, keyed by traversal order for matrices and by channel for vectors.
    fn map_float<const D: usize>(
        &mut self,
        param: Param<Tensor<TestBackend, D>>,
    ) -> Param<Tensor<TestBackend, D>> {
        self.0 += 1;
        let ordinal = self.0;
        param.map(|tensor| {
            let shape = tensor.dims();
            let values = (0..shape.iter().product::<usize>())
                .map(|index| {
                    if D == 1 {
                        norm_scale(index)
                    } else {
                        (((index + 1) * (ordinal + 3)) as f32 * 0.17).sin() * 0.3
                    }
                })
                .collect::<Vec<_>>();
            Tensor::from_data(TensorData::new(values, shape), &tensor.device())
        })
    }
}

/// Two RoPE attention blocks with a context large enough for shifted positions.
fn precision_spec() -> DecoderSpec {
    let rope = PositionEncodingSpec::Rope {
        theta: 10_000.0,
        rotary_dim: HEAD_DIM,
        scaling: RopeScaling::None,
    };
    let mut spec = decoder(vec![layer(attention(rope.clone())), layer(attention(rope))]);
    spec.max_position_embeddings = ROTARY_SHIFT as usize + TOKENS;
    spec
}

/// Returns the F32 fixture and its exactly widened F64 copy for one attention mode.
fn models(
    mode: AttentionPrecisionMode,
) -> (ModularCausalLm<TestBackend>, ModularCausalLm<TestBackend>) {
    let single = ModularCausalLm::init(precision_spec(), &Default::default())
        .expect("valid fixture specification")
        .map(&mut NonUnitParameters::default())
        .with_attention_precision(mode);
    let double = single.clone().map(&mut CastParameters(DType::F64));
    (single, double)
}

/// Two token rows with explicit positions starting at `offset`.
fn tokens(offset: i32) -> CausalLmInput<TestBackend> {
    let device = Default::default();
    let positions = core::array::from_fn::<i32, TOKENS, _>(|index| offset + index as i32);
    CausalLmInput {
        token_ids: Tensor::from_ints([[3, 17, 5, 29, 8, 1], [30, 2, 11, 7, 19, 4]], &device),
        token_mask: None,
        position_ids: Some(Tensor::from_ints([positions, positions], &device)),
    }
}

/// Requests logits for every token so whole sequences are compared.
fn all_logits() -> ForwardOptions {
    ForwardOptions {
        logits: LogitsSelection::All,
        ..Default::default()
    }
}

/// Host channel scales for a normalization of `width` channels.
fn host_scales(width: usize) -> Vec<f64> {
    (0..width)
        .map(|channel| f64::from(norm_scale(channel)))
        .collect()
}

/// Reads one capture and requires F64 storage.
fn captured<const D: usize>(
    hooks: &Hooks<'_, TestBackend>,
    capture: &Capture<D>,
    name: &str,
) -> Vec<f64> {
    let data = hooks.get(capture).expect("captured activation").into_data();
    assert_eq!(data.dtype, DType::F64, "{name} must stay F64");
    f64_values(&data)
}

/// F64 parameters keep every decoder activation in F64, matching host F64 operations.
///
/// Embeddings, both residual streams, all RMS normalizations, rotated Q/K,
/// SwiGLU activations, eager scores/patterns, final normalization, and logits
/// are F64. RMS normalizations, SiLU, and eager softmax match host F64
/// references at `1e-12`, and logits match the F32 model within F32 tolerance.
#[test]
fn f64_parameters_keep_every_decoder_activation_in_f64() {
    for mode in [
        AttentionPrecisionMode::BackendDefault,
        AttentionPrecisionMode::EagerFp32Softmax,
    ] {
        let eager = mode == AttentionPrecisionMode::EagerFp32Softmax;
        let (single, double) = models(mode);
        let mut hooks = Hooks::new();
        let embedding = hooks.capture(Stack::Decoder.embedding());
        let final_norm = hooks.capture(Stack::Decoder.final_norm());
        let blocks = (0..2)
            .map(|index| {
                let block = Layer::decoder(index);
                let attention = block.self_attention();
                (
                    [
                        hooks.capture(block.residual_pre()),
                        hooks.capture(block.attention_input()),
                        hooks.capture(block.attention_output()),
                        hooks.capture(block.mlp_input()),
                        hooks.capture(block.mlp_pre_activation()),
                        hooks.capture(block.mlp_activation()),
                        hooks.capture(block.mlp_output()),
                        hooks.capture(block.residual_post()),
                    ],
                    [
                        hooks.capture(attention.query()),
                        hooks.capture(attention.key()),
                        hooks.capture(attention.value()),
                        hooks.capture(attention.head_context()),
                    ],
                    eager.then(|| {
                        [
                            hooks.capture(attention.scores()),
                            hooks.capture(attention.pattern()),
                        ]
                    }),
                )
            })
            .collect::<Vec<_>>();
        let output = double
            .forward_with_hooks(tokens(0), &all_logits(), &mut hooks)
            .expect("F64 forward");
        let context = format!("{mode:?}");

        let mut residual = captured(&hooks, &embedding, "embedding");
        for (index, (residuals, projections, planes)) in blocks.iter().enumerate() {
            let [pre, normalized, mixed, mlp_input, gate, activated, _, post] = residuals
                .each_ref()
                .map(|capture| captured(&hooks, capture, "residual"));
            for capture in projections {
                captured(&hooks, capture, "attention projection");
            }
            let block_context = format!("{context} block {index}");
            assert_eq!(
                pre, residual,
                "{block_context}: block input is the prior residual"
            );
            let scales = host_scales(HIDDEN);
            assert_matches_f64_reference(
                &TensorData::new(normalized, [pre.len()]),
                &host_rms_norm(&pre, &scales, EPSILON),
                1e-6,
                &format!("{block_context} mixer RMSNorm"),
            );
            let middle = pre
                .iter()
                .zip(&mixed)
                .map(|(x, y)| x + y)
                .collect::<Vec<_>>();
            assert_matches_f64_reference(
                &TensorData::new(mlp_input, [middle.len()]),
                &host_rms_norm(&middle, &scales, EPSILON),
                1e-6,
                &format!("{block_context} feed-forward RMSNorm"),
            );
            let silu = gate
                .iter()
                .map(|value| value / (1.0 + (-value).exp()))
                .collect::<Vec<_>>();
            assert_matches_f64_reference(
                &TensorData::new(activated, [silu.len()]),
                &silu,
                1e-6,
                &format!("{block_context} SiLU gate"),
            );
            if let Some([scores, pattern]) = planes {
                let scores = captured(&hooks, scores, "scores");
                let pattern = captured(&hooks, pattern, "pattern");
                // Plane rows are `[batch, head, query]`; a key is excluded after its query.
                let excluded = (0..scores.len())
                    .map(|index| index % TOKENS > (index / TOKENS) % TOKENS)
                    .collect::<Vec<_>>();
                assert_matches_f64_reference(
                    &TensorData::new(pattern, [scores.len()]),
                    &host_masked_softmax(&scores, Some(&excluded), TOKENS),
                    1e-6,
                    &format!("{block_context} eager causal softmax"),
                );
            }
            residual = post;
        }
        assert_matches_f64_reference(
            &TensorData::new(
                captured(&hooks, &final_norm, "final norm"),
                [residual.len()],
            ),
            &host_rms_norm(&residual, &host_scales(HIDDEN), EPSILON),
            1e-6,
            &format!("{context} final RMSNorm"),
        );

        let logits = output.logits.into_data();
        assert_eq!(logits.dtype, DType::F64, "{context} logits");
        assert_eq!(logits.shape.dims::<3>(), [2, TOKENS, 32]);
        let reference = single
            .forward(tokens(0), &all_logits())
            .expect("F32 forward")
            .logits
            .into_data();
        assert_eq!(reference.dtype, DType::F32);
        // Identical parameter values: only F32 arithmetic separates the two runs.
        let difference = scaled_max_difference(&reference, &logits);
        assert!(
            difference <= 1e-5,
            "{context}: F64 and F32 logits differ by {difference:e}"
        );
    }
}

/// F64 decoders are invariant to a common rotary position shift to F64 accuracy.
///
/// RoPE scores depend only on position differences, and this decoder has no
/// absolute positional input, so shifting every position by `2^22` preserves
/// the exact function. F64 tables keep the shifted logits within `1e-9`; the
/// same shift moves F32 logits by more than `1e-6`, so F32 rotary phases in an
/// F64 model would fail this check. Both attention dispatch modes are covered.
#[test]
fn f64_decoder_logits_are_invariant_to_rotary_position_shifts() {
    for mode in [
        AttentionPrecisionMode::BackendDefault,
        AttentionPrecisionMode::EagerFp32Softmax,
    ] {
        let (single, double) = models(mode);
        let logits = |model: &ModularCausalLm<TestBackend>, offset: i32| {
            model
                .forward(tokens(offset), &all_logits())
                .expect("shifted forward")
                .logits
                .into_data()
        };
        assert_rotary_shift_invariant(
            &logits(&double, 0),
            &logits(&double, ROTARY_SHIFT),
            &format!("{mode:?}"),
        );
        assert_rotary_shift_sensitive(
            &logits(&single, 0),
            &logits(&single, ROTARY_SHIFT),
            &format!("{mode:?}"),
        );
    }
}
