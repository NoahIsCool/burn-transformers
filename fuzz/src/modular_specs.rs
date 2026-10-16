//! Modular architecture specifications: validation, serialization, construction, execution.
//!
//! The specification module promises that "a spec that validates can be
//! constructed": [`DecoderSpec::validate`] and [`Seq2SeqSpec::validate`] build
//! and validate the same shared component configurations that the modular
//! models build from. The harness generates typed specifications covering
//! every mixer (grouped-query attention in all three forms, MLA, differential
//! attention, Gated DeltaNet, Mamba2), every feed-forward operator (gated,
//! ungated, clamped SwiGLU, routed experts with every routing policy, none),
//! every position encoding and RoPE scaling rule, every connectivity, score
//! scale, soft cap, sink, bias, norm-scope, gate, normalization, placement,
//! and residual option, with small dimensions that are sometimes zero,
//! non-divisible, or mismatched. It checks:
//!
//! 1. **Totality.** Validation never panics and fails only with
//!    [`Error::Configuration`].
//! 2. **Round trip.** A valid spec serializes and parses back to the same JSON,
//!    integers and floats exactly (see `crate::floats_round_trip`), and still
//!    validates.
//! 3. **Construction.** [`ModularCausalLm::init`] (or
//!    [`ModularSeq2SeqLm::init`]) succeeds exactly for valid specs without the
//!    documented capability gaps (gated residuals, Mamba2, positions beyond
//!    `i32`, and recurrent or sliding-window encoder layers), which fail with
//!    [`Error::BackendCapability`] or a configuration error.
//! 4. **Execution.** A constructed model with moderate parameters evaluates a
//!    padded two-row batch to finite logits of the expected shape.
//! 5. **Cache parity.** For causal stacks without routed experts (whose top-k
//!    may flip on rounding-level differences), a prefill of three tokens plus one
//!    decode step reproduces the stateless forward's last logits.

use burn::{
    backend::Flex,
    tensor::{Int, Tensor, TensorData},
};
use transformers::{
    CausalLanguageModel, CausalLmInput, DecodeInput, Error, ForwardOptions, LogitsSelection,
    Seq2SeqInput, Seq2SeqLanguageModel, TokenMask,
    architecture::{
        Activation, AttentionOptions, AttentionSpec, ClampedSwiGluSpec, DecoderLayerSpec,
        DecoderSpec, DenseFeedForwardSpec, DifferentialAttentionSpec, FeedForwardSpec, Mamba2Spec,
        MixerSpec, MixtureOfExpertsSpec, MlaSpec, NormPlacement, NormalizationSpec,
        OutputGateActivation, PositionEncodingSpec, ResidualTopology, RopeScaling, ScoreScale,
        Seq2SeqSpec, SlidingWindowSpec,
    },
    attention::{AttentionConnectivity, ProjectionBias, QueryKeyNormScope},
    cache::CacheConfig,
    models::custom::{ModularCausalLm, ModularSeq2SeqLm},
    moe::{
        ExpertSelection, GroupLimitedSelection, GroupScore, PrunedExpertScore,
        RenormalizationDenominator, RouterProjectionPrecision, RouterScoring, RoutingPolicy,
        ScoringOrder,
    },
};

use crate::Bytes;

/// Host backend of every model in this harness.
type Backend = Flex;

/// Relative tolerance between cached and stateless logits.
const PARITY_TOLERANCE: f64 = 2e-3;
/// Tokens per row of the execution inputs.
const SEQUENCE: usize = 4;

/// Byte decoder that draws moderate values, or boundary values in extreme mode.
struct Draw<'a, 'b> {
    /// Underlying total decoder.
    bytes: &'b mut Bytes<'a>,
    /// Whether floats may come from [`crate::INTERESTING_FLOATS`].
    extreme: bool,
}

impl Draw<'_, '_> {
    /// Whether the next draw takes its table's typical first entry (7 times in 8).
    ///
    /// A specification makes a few dozen draws, and every table lists rarer or
    /// invalid entries after the first, so a higher rate would leave almost no
    /// valid specifications and a lower one almost no invalid fields.
    fn typical(&mut self) -> bool {
        self.bytes.byte() < 224
    }

    /// A small dimension; entries after the first are rarer or invalid.
    fn size(&mut self, table: &[usize]) -> usize {
        if self.typical() {
            table[0]
        } else {
            self.bytes.pick(table)
        }
    }

    /// A float from `table`, or a boundary float in extreme mode.
    fn float(&mut self, table: &[f64]) -> f64 {
        if self.extreme && self.bytes.flag() {
            self.bytes.interesting_float()
        } else if self.typical() {
            table[0]
        } else {
            self.bytes.pick(table)
        }
    }

    /// A Boolean.
    fn flag(&mut self) -> bool {
        self.bytes.flag()
    }

    /// One of `count` variants.
    fn variant(&mut self, count: usize) -> usize {
        self.bytes.below(count)
    }
}

/// Decodes a normalization: RMS or LayerNorm with a mostly valid epsilon.
fn normalization(draw: &mut Draw<'_, '_>) -> NormalizationSpec {
    let epsilon = draw.float(&[1e-6, 1e-5, 0.0, -1.0]);
    if draw.flag() {
        NormalizationSpec::Rms {
            epsilon,
            zero_centered: draw.flag(),
        }
    } else {
        NormalizationSpec::Layer {
            epsilon,
            bias: draw.flag(),
        }
    }
}

/// Decodes a RoPE scaling rule for `rotary` channels.
fn rope_scaling(draw: &mut Draw<'_, '_>, rotary: usize) -> RopeScaling {
    match draw.variant(7) {
        0 => RopeScaling::None,
        1 => RopeScaling::Linear {
            factor: draw.float(&[2.0, 1.0, 0.5]),
        },
        2 => RopeScaling::DynamicNtk {
            factor: draw.float(&[2.0, 1.0, 0.5]),
            original_max_positions: draw.size(&[4, 1, 0]),
        },
        3 => RopeScaling::Yarn {
            factor: draw.float(&[4.0, 1.0, 0.5]),
            original_max_positions: draw.size(&[4, 1, 0]),
            attention_factor: draw.flag().then(|| draw.float(&[1.1, 0.0])),
            beta_slow: draw.float(&[1.0, 2.0, 32.0]),
            beta_fast: draw.float(&[32.0, 4.0, 1.0]),
            truncate: draw.flag(),
        },
        4 => RopeScaling::Llama3 {
            factor: draw.float(&[8.0, 1.0]),
            low_freq_factor: draw.float(&[1.0, 2.0]),
            high_freq_factor: draw.float(&[4.0, 1.0]),
            original_max_positions: draw.size(&[4, 1, 0]),
        },
        5 => {
            let count = draw.size(&[rotary / 2, 1, 0]);
            let short_factors = (0..count).map(|_| draw.float(&[1.0, 2.0, 0.5])).collect();
            let long_count = draw.size(&[count, count + 1]);
            let long_factors = (0..long_count).map(|_| draw.float(&[2.0, 4.0])).collect();
            RopeScaling::LongRope {
                short_factors,
                long_factors,
                original_max_positions: draw.size(&[4, 1, 0]),
                factor: draw.float(&[2.0, 1.0]),
                attention_factor: draw.flag().then(|| draw.float(&[1.2, -1.0])),
            }
        }
        _ => RopeScaling::Proportional {
            rotated_fraction: draw.float(&[0.5, 0.0, 1.0, 1.5]),
            factor: draw.float(&[1.0, 2.0]),
        },
    }
}

/// Decodes a position encoding for heads of `head_dim` channels.
fn position(draw: &mut Draw<'_, '_>, head_dim: usize) -> PositionEncodingSpec {
    match draw.variant(4) {
        0 => PositionEncodingSpec::None,
        1 | 2 => {
            let rotary_dim = draw.size(&[head_dim, 2, 0, head_dim + 2, 3]);
            PositionEncodingSpec::Rope {
                theta: draw.float(&[10_000.0, 1_000_000.0, 2.0, 1.0, 0.5]),
                rotary_dim,
                scaling: rope_scaling(draw, rotary_dim),
            }
        }
        _ => PositionEncodingSpec::T5RelativeBias {
            buckets: draw.size(&[8, 32, 4, 3, 2]),
            max_distance: draw.size(&[16, 128, 4, 1]),
            bidirectional: draw.flag(),
        },
    }
}

/// Decodes a connectivity rule; zero widths are invalid.
fn connectivity(draw: &mut Draw<'_, '_>) -> AttentionConnectivity {
    match draw.variant(4) {
        0 | 1 => AttentionConnectivity::Global,
        2 => AttentionConnectivity::SlidingWindow {
            window: draw.size(&[2, 1, 3, 0]),
        },
        _ => AttentionConnectivity::ChunkedLocal {
            chunk: draw.size(&[2, 1, 3, 0]),
        },
    }
}

/// Decodes a score scale.
fn score_scale(draw: &mut Draw<'_, '_>) -> ScoreScale {
    match draw.variant(4) {
        0 => ScoreScale::Auto,
        1 => ScoreScale::InverseSqrtHeadDim,
        2 => ScoreScale::Unscaled,
        _ => ScoreScale::Fixed {
            value: draw.float(&[0.5, 1.0, 0.0, -1.0]),
        },
    }
}

/// Decodes grouped-query attention geometry and position.
fn attention(draw: &mut Draw<'_, '_>) -> AttentionSpec {
    let head_dim = draw.size(&[4, 2, 8, 0, 3]);
    AttentionSpec {
        query_heads: draw.size(&[2, 4, 1, 0, 3]),
        key_value_heads: draw.size(&[1, 2, 4, 0]),
        head_dim,
        query_key_norm: draw.flag(),
        output_gate: draw.flag(),
        position: position(draw, head_dim),
    }
}

/// Decodes every grouped-query attention option for `attention`.
///
/// Options that contradict the compact flags (a full-width Q/K-norm scope
/// without Q/K norm, a SiLU gate without a gate, sinks with a soft cap) are
/// drawn only one time in eight, so most specifications stay valid.
fn attention_options(draw: &mut Draw<'_, '_>, attention: &AttentionSpec) -> AttentionOptions {
    let attention_sinks = draw.flag();
    let softcap_allowed = !attention_sinks || !draw.typical();
    let full_width_allowed = attention.query_key_norm || !draw.typical();
    let silu_allowed = attention.output_gate || !draw.typical();
    AttentionOptions {
        connectivity: connectivity(draw),
        score_scale: score_scale(draw),
        logit_softcap: (softcap_allowed && draw.flag()).then(|| draw.float(&[30.0, 1.0, 0.0])),
        attention_sinks,
        projection_bias: ProjectionBias {
            query: draw.flag(),
            key: draw.flag(),
            value: draw.flag(),
            output: draw.flag(),
        },
        query_key_norm_scope: if full_width_allowed && draw.flag() {
            QueryKeyNormScope::FullWidth
        } else {
            QueryKeyNormScope::PerHead
        },
        output_gate_activation: if silu_allowed && draw.flag() {
            OutputGateActivation::Silu
        } else {
            OutputGateActivation::Sigmoid
        },
    }
}

/// Decodes one sequence mixer of any kind.
fn mixer(draw: &mut Draw<'_, '_>) -> MixerSpec {
    match draw.variant(9) {
        0 | 1 => MixerSpec::Attention(attention(draw)),
        2 => MixerSpec::SlidingAttention(SlidingWindowSpec {
            attention: attention(draw),
            window: draw.size(&[2, 1, 0]),
        }),
        3 | 4 => {
            let attention = attention(draw);
            let options = attention_options(draw, &attention);
            MixerSpec::AttentionWithOptions { attention, options }
        }
        5 => {
            let rotary_head_dim = draw.size(&[2, 4, 0, 3]);
            MixerSpec::Mla(MlaSpec {
                query_heads: draw.size(&[2, 1, 0]),
                content_head_dim: draw.size(&[2, 4, 0]),
                rotary_head_dim,
                value_head_dim: draw.size(&[2, 4, 0]),
                key_value_lora_rank: draw.size(&[4, 2, 0]),
                query_lora_rank: draw.flag().then(|| draw.size(&[4, 0])),
                position: position(draw, rotary_head_dim),
                score_scale: score_scale(draw),
                connectivity: connectivity(draw),
            })
        }
        6 => {
            let head_dim = draw.size(&[4, 2, 0]);
            let position = match position(draw, head_dim) {
                // The kernel has no additive-bias input; keep T5 rare.
                PositionEncodingSpec::T5RelativeBias { .. } if draw.typical() => {
                    PositionEncodingSpec::None
                }
                other => other,
            };
            MixerSpec::DifferentialAttention(DifferentialAttentionSpec {
                query_heads: draw.size(&[2, 4, 1, 3]),
                key_value_heads: draw.size(&[1, 2, 0]),
                head_dim,
                lambda_init: draw.float(&[0.2, 0.8, 0.0, 1.0]),
                position,
                connectivity: connectivity(draw),
                bias: draw.flag(),
            })
        }
        7 => MixerSpec::DeltaNet {
            key_heads: draw.size(&[1, 2, 0]),
            value_heads: draw.size(&[2, 1, 4, 3]),
            key_head_dim: draw.size(&[2, 4, 0]),
            value_head_dim: draw.size(&[2, 4]),
            convolution_kernel: draw.size(&[2, 1, 4, 0]),
        },
        _ => MixerSpec::Mamba2(Mamba2Spec {
            heads: draw.size(&[2, 1, 0]),
            head_dim: 2,
            state_dim: 2,
            convolution_kernel: 2,
            group_size: draw.size(&[1, 2, 3]),
        }),
    }
}

/// Decodes a routing policy, including group-limited selection.
fn routing(draw: &mut Draw<'_, '_>) -> RoutingPolicy {
    if draw.variant(3) != 0 {
        return RoutingPolicy::default();
    }
    RoutingPolicy {
        scoring: if draw.flag() {
            RouterScoring::Sigmoid
        } else {
            RouterScoring::Softmax
        },
        order: if draw.flag() {
            ScoringOrder::TopKThenScore
        } else {
            ScoringOrder::ScoreThenTopK
        },
        selection: if draw.flag() {
            ExpertSelection::GroupLimited(GroupLimitedSelection {
                groups: draw.size(&[2, 1, 3]),
                top_groups: draw.size(&[1, 2, 0]),
                group_score: if draw.flag() {
                    GroupScore::TopTwoSum
                } else {
                    GroupScore::Max
                },
                pruned_score: if draw.flag() {
                    PrunedExpertScore::Zero
                } else {
                    PrunedExpertScore::NegativeInfinity
                },
            })
        } else {
            ExpertSelection::Greedy
        },
        correction_bias: draw.flag(),
        router_bias: draw.flag(),
        projection: if draw.flag() {
            RouterProjectionPrecision::Fp32
        } else {
            RouterProjectionPrecision::ModelDtype
        },
        denominator: match draw.variant(3) {
            0 => RenormalizationDenominator::ClampedBelow(f32::MIN_POSITIVE),
            1 => RenormalizationDenominator::PlusEpsilon(1e-20),
            _ => RenormalizationDenominator::ClampedBelow(-1.0),
        },
    }
}

/// Decodes a feed-forward operator of any kind.
fn feed_forward(draw: &mut Draw<'_, '_>) -> FeedForwardSpec {
    let activation = |draw: &mut Draw<'_, '_>| {
        draw.bytes.pick(&[
            Activation::Silu,
            Activation::Relu,
            Activation::Gelu,
            Activation::GeluTanh,
            Activation::ReluSquared,
        ])
    };
    match draw.variant(6) {
        0 => FeedForwardSpec::Dense(DenseFeedForwardSpec {
            intermediate_size: draw.size(&[8, 4, 0]),
            activation: activation(draw),
            bias: draw.flag(),
        }),
        1 => FeedForwardSpec::Ungated(DenseFeedForwardSpec {
            intermediate_size: draw.size(&[8, 4, 0]),
            activation: activation(draw),
            bias: draw.flag(),
        }),
        2 => FeedForwardSpec::ClampedSwiGlu(ClampedSwiGluSpec {
            intermediate_size: draw.size(&[8, 0]),
            bias: draw.flag(),
            limit: draw.float(&[7.0, 0.0, -1.0]),
            alpha: draw.float(&[1.702, 1.0, 0.0]),
            up_offset: draw.float(&[1.0, 0.0]),
        }),
        3 | 4 => {
            let shared_experts = draw.size(&[0, 1]);
            // A gate needs a shared expert; keep the invalid pairing rare.
            let gate_allowed = shared_experts > 0 || !draw.typical();
            FeedForwardSpec::MixtureOfExperts(MixtureOfExpertsSpec {
                routed_experts: draw.size(&[4, 2, 0]),
                experts_per_token: draw.size(&[2, 1, 3, 0]),
                shared_experts,
                expert_intermediate_size: draw.size(&[4, 0]),
                activation: activation(draw),
                renormalize_selected: draw.flag(),
                routed_scale: draw.float(&[1.0, 2.5, 0.0]),
                routing: routing(draw),
                shared_expert_gate: gate_allowed && draw.flag(),
                expert_bias: draw.flag(),
            })
        }
        _ => FeedForwardSpec::None,
    }
}

/// Decodes one block.
fn layer(draw: &mut Draw<'_, '_>) -> DecoderLayerSpec {
    DecoderLayerSpec {
        mixer: mixer(draw),
        feed_forward: feed_forward(draw),
        mixer_norm: normalization(draw),
        feed_forward_norm: normalization(draw),
        norm_placement: draw.bytes.pick(&[
            NormPlacement::Pre,
            NormPlacement::Post,
            NormPlacement::Sandwich,
        ]),
        residual: if draw.variant(8) == 0 {
            ResidualTopology::Gated
        } else {
            ResidualTopology::Add
        },
    }
}

/// Decodes one stack of up to two layers.
fn stack(draw: &mut Draw<'_, '_>, hidden_size: usize) -> DecoderSpec {
    DecoderSpec {
        vocabulary_size: draw.size(&[16, 7, 1, 0]),
        hidden_size,
        layers: (0..draw.size(&[1, 2, 0])).map(|_| layer(draw)).collect(),
        final_norm: normalization(draw),
        tie_word_embeddings: draw.flag(),
        max_position_embeddings: draw.size(&[16, 8, 1 << 31, 0]),
    }
}

/// Whether the modular models can execute a layer.
///
/// Gated residuals are not executable. Mamba-2 mixers execute through the
/// shared `transformers::mamba2` kernel, except with a width-one causal
/// convolution, which construction rejects with a configuration error.
fn layer_is_executable(layer: &DecoderLayerSpec) -> bool {
    layer.residual != ResidualTopology::Gated
        && !matches!(&layer.mixer, MixerSpec::Mamba2(mamba) if mamba.convolution_kernel < 2)
}

/// Whether a layer uses routed experts, whose top-k may flip on rounding noise.
fn has_routed_experts(spec: &DecoderSpec) -> bool {
    spec.layers
        .iter()
        .any(|layer| matches!(layer.feed_forward, FeedForwardSpec::MixtureOfExperts(_)))
}

/// Builds a two-row causal input; row 1 is left-padded by one token.
fn causal_input(vocabulary: usize, tokens: &[i32], columns: usize) -> CausalLmInput<Backend> {
    let device = Default::default();
    let modulus = i32::try_from(vocabulary).expect("small vocabulary");
    let ids: Vec<i32> = tokens
        .iter()
        .map(|token| token.rem_euclid(modulus))
        .collect();
    let mask: Vec<i32> = (0..2 * columns)
        .map(|index| i32::from(index != columns))
        .collect();
    CausalLmInput {
        token_ids: Tensor::<Backend, 2, Int>::from_data(
            TensorData::new(ids, [2, columns]),
            &device,
        ),
        token_mask: Some(TokenMask::new(
            Tensor::<Backend, 2, Int>::from_data(TensorData::new(mask, [2, columns]), &device)
                .bool(),
        )),
        position_ids: None,
    }
}

/// Reads logits to host values and checks their shape.
fn logits(output: Tensor<Backend, 3>, shape: [usize; 3]) -> Vec<f32> {
    assert_eq!(output.dims(), shape);
    output.into_data().to_vec::<f32>().expect("F32 logits")
}

/// Options requesting logits at every position.
fn all_logits() -> ForwardOptions {
    ForwardOptions {
        logits: LogitsSelection::All,
        ..ForwardOptions::default()
    }
}

/// Runs the stateless forward and, without routed experts, the cached parity check.
fn execute_causal(model: &ModularCausalLm<Backend>, spec: &DecoderSpec, tokens: &[i32]) {
    let vocabulary = spec.vocabulary_size;
    let input = causal_input(vocabulary, &tokens[..2 * SEQUENCE], SEQUENCE);
    let output = CausalLanguageModel::forward(model, input, &all_logits()).expect("valid input");
    let full = logits(output.logits, [2, SEQUENCE, vocabulary]);
    assert!(
        full.iter().all(|value| value.is_finite()),
        "non-finite logits for {spec:?}"
    );
    if has_routed_experts(spec) {
        return;
    }
    let device = Default::default();
    let mut state = model
        .new_decode_state(CacheConfig::new(2, SEQUENCE).unwrap(), 2, &device)
        .expect("capacity within max_position_embeddings");
    let prompt: Vec<i32> = (0..2)
        .flat_map(|row| tokens[row * SEQUENCE..row * SEQUENCE + SEQUENCE - 1].to_vec())
        .collect();
    model
        .prefill(
            causal_input(vocabulary, &prompt, SEQUENCE - 1),
            &mut state,
            &all_logits(),
        )
        .expect("valid prefill");
    let modulus = i32::try_from(vocabulary).unwrap();
    let last: Vec<i32> = (0..2)
        .map(|row| tokens[row * SEQUENCE + SEQUENCE - 1].rem_euclid(modulus))
        .collect();
    let step = model
        .decode(
            DecodeInput {
                token_ids: Tensor::<Backend, 2, Int>::from_data(
                    TensorData::new(last, [2, 1]),
                    &device,
                ),
                position_ids: None,
                token_mask: None,
            },
            &mut state,
            &all_logits(),
        )
        .expect("valid decode");
    let cached = logits(step.logits, [2, 1, vocabulary]);
    for row in 0..2 {
        for token in 0..vocabulary {
            let expected = f64::from(full[(row * SEQUENCE + SEQUENCE - 1) * vocabulary + token]);
            let actual = f64::from(cached[row * vocabulary + token]);
            assert!(
                (actual - expected).abs() <= PARITY_TOLERANCE * expected.abs().max(1.0),
                "row {row} token {token}: cached {actual}, stateless {expected} for {spec:?}"
            );
        }
    }
}

/// Checks that `result` failed with an error a validated spec may still produce.
fn assert_capability_error<T>(result: transformers::Result<T>) {
    match result {
        Err(Error::BackendCapability(_) | Error::Configuration { .. }) => {}
        Err(other) => panic!("unexpected construction error {other:?}"),
        Ok(_) => panic!("construction succeeded despite a documented capability gap"),
    }
}

/// How far a specification got through the checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Outcome {
    /// Validation rejected it.
    Invalid,
    /// It validated but has a documented capability gap.
    Unsupported,
    /// It validated and constructed; execution was skipped.
    Constructed,
    /// It validated, constructed, and executed.
    Executed,
}

/// Checks validation, round trip, construction, and execution of a causal stack.
fn check_causal(spec: &DecoderSpec, moderate: bool, tokens: &[i32]) -> Outcome {
    let validation = spec.validate();
    if let Err(error) = &validation {
        assert!(matches!(error, Error::Configuration { .. }), "{error:?}");
        return Outcome::Invalid;
    }
    let encoded = serde_json::to_value(spec).expect("specs serialize");
    let decoded: DecoderSpec = serde_json::from_value(encoded.clone()).expect("specs parse");
    assert!(crate::json_equal(
        &encoded,
        &serde_json::to_value(&decoded).unwrap()
    ));
    assert!(decoded.validate().is_ok());
    let executable = spec.layers.iter().all(layer_is_executable)
        && spec.max_position_embeddings <= i32::MAX as usize;
    let model = ModularCausalLm::<Backend>::init(spec.clone(), &Default::default());
    if !executable {
        assert_capability_error(model);
        return Outcome::Unsupported;
    }
    let model = model.expect("validated executable specs construct");
    if moderate && spec.max_position_embeddings >= SEQUENCE {
        execute_causal(&model, spec, tokens);
        Outcome::Executed
    } else {
        Outcome::Constructed
    }
}

/// Checks validation, round trip, construction, and execution of an encoder-decoder.
fn check_seq2seq(spec: &Seq2SeqSpec, moderate: bool, tokens: &[i32]) -> Outcome {
    if let Err(error) = spec.validate() {
        assert!(matches!(error, Error::Configuration { .. }), "{error:?}");
        return Outcome::Invalid;
    }
    let encoded = serde_json::to_value(spec).expect("specs serialize");
    let decoded: Seq2SeqSpec = serde_json::from_value(encoded.clone()).expect("specs parse");
    assert!(crate::json_equal(
        &encoded,
        &serde_json::to_value(&decoded).unwrap()
    ));
    let encoder_executable = spec.encoder.layers.iter().all(|layer| {
        layer_is_executable(layer)
            && !layer.mixer.is_recurrent()
            && !matches!(
                layer.mixer.connectivity(),
                Some(AttentionConnectivity::SlidingWindow { .. })
            )
    });
    let executable = encoder_executable
        && spec.decoder.layers.iter().all(layer_is_executable)
        && spec.encoder.max_position_embeddings <= i32::MAX as usize
        && spec.decoder.max_position_embeddings <= i32::MAX as usize;
    let model = ModularSeq2SeqLm::<Backend>::init(spec.clone(), &Default::default());
    if !executable {
        assert_capability_error(model);
        return Outcome::Unsupported;
    }
    let model = model.expect("validated executable specs construct");
    let fits = spec.encoder.max_position_embeddings >= SEQUENCE
        && spec.decoder.max_position_embeddings >= SEQUENCE;
    if moderate && fits {
        let input = Seq2SeqInput {
            encoder: causal_input(
                spec.encoder.vocabulary_size,
                &tokens[..2 * SEQUENCE],
                SEQUENCE,
            ),
            decoder: causal_input(
                spec.decoder.vocabulary_size,
                &tokens[2 * SEQUENCE..4 * SEQUENCE],
                SEQUENCE,
            ),
        };
        let output =
            Seq2SeqLanguageModel::forward(&model, input, &all_logits()).expect("valid input");
        let values = logits(
            output.decoder.logits,
            [2, SEQUENCE, spec.decoder.vocabulary_size],
        );
        assert!(
            values.iter().all(|value| value.is_finite()),
            "non-finite logits for {spec:?}"
        );
        Outcome::Executed
    } else {
        Outcome::Constructed
    }
}

/// Decodes and checks one specification.
///
/// Byte layout: a mode byte (bit 0 selects an encoder-decoder, values of at
/// least 224 select extreme floats that only validation and construction
/// see), the hidden width, then each stack (vocabulary, layer count, layers,
/// final norm, tying, maximum positions) and, for encoder-decoders, the
/// cross-attention heads, width, and options, then 16 token bytes. Every
/// dimension and float draw takes its typical value seven times in eight.
pub fn exercise(input: &[u8]) {
    exercise_outcome(input);
}

/// [`exercise`], returning how far the decoded specification got.
fn exercise_outcome(input: &[u8]) -> Outcome {
    let mut bytes = Bytes::new(input);
    let mode = bytes.byte();
    let extreme = mode >= 224;
    let mut draw = Draw {
        bytes: &mut bytes,
        extreme,
    };
    let hidden_size = draw.size(&[8, 4, 16, 0]);
    if mode & 1 == 0 {
        let spec = stack(&mut draw, hidden_size);
        let tokens: Vec<i32> = (0..4 * SEQUENCE)
            .map(|_| i32::from(draw.bytes.byte()))
            .collect();
        check_causal(&spec, !extreme, &tokens)
    } else {
        let encoder = stack(&mut draw, hidden_size);
        let decoder_hidden = draw.size(&[hidden_size, 4]);
        let decoder = stack(&mut draw, decoder_hidden);
        let spec = Seq2SeqSpec {
            encoder,
            decoder,
            cross_attention_heads: draw.size(&[2, 1, 0]),
            cross_attention_head_dim: draw.size(&[4, 2, 0]),
            cross_attention_options: if draw.flag() {
                let cross = AttentionSpec {
                    query_heads: 1,
                    key_value_heads: 1,
                    head_dim: 1,
                    query_key_norm: false,
                    output_gate: false,
                    position: PositionEncodingSpec::None,
                };
                attention_options(&mut draw, &cross)
            } else {
                AttentionOptions::default()
            },
        };
        let tokens: Vec<i32> = (0..4 * SEQUENCE)
            .map(|_| i32::from(draw.bytes.byte()))
            .collect();
        check_seq2seq(&spec, !extreme, &tokens)
    }
}

#[cfg(test)]
#[path = "modular_specs_tests.rs"]
mod tests;
