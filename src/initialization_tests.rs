//! Initialization tests: realized statistics per role, structural contracts, and initial loss.

use burn::{
    backend::Flex,
    module::{Module, Param},
    nn::{Embedding, EmbeddingConfig, Linear, LinearConfig, LinearLayout},
    record::{BinBytesRecorder, FullPrecisionSettings, Recorder},
    tensor::{Int, Tensor, backend::Backend},
};

use super::*;

/// Deterministic F32 CPU backend.
type B = Flex<f32, i32>;

/// One block with an ordinary projection, a residual projection, and a scale vector.
#[derive(Module, Debug)]
struct Block<B: Backend> {
    /// Expanding projection with bias, `width -> 4 width`.
    up: Linear<B>,
    /// Residual output projection with bias, `4 width -> width`.
    o_proj: Linear<B>,
    /// Normalization-like scale, initialized to ones.
    scale: Param<Tensor<B, 1>>,
}

/// Embedding, blocks, and a column-layout head, mirroring a small language model.
#[derive(Module, Debug)]
struct Model<B: Backend> {
    /// Token embedding `[vocabulary, width]`.
    embed: Embedding<B>,
    /// Residual blocks.
    blocks: Vec<Block<B>>,
    /// Vocabulary projection stored in checkpoint-native column layout.
    head: Linear<B>,
}

/// Width of the fixture's residual stream.
const WIDTH: usize = 64;
/// Fixture vocabulary size.
const VOCABULARY: usize = 512;

/// Identity mapper that materializes lazy parameters, keeping identity, mapper, and trainability.
struct Materialize;

impl<B: Backend> burn::module::ModuleMapper<B> for Materialize {
    /// Initializes the parameter and returns it unchanged.
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let (id, value, mapper) = param.consume();
        Param::from_mapped_value(id, value, mapper)
    }
}

/// Builds the fixture with Burn's default initializers, materialized.
///
/// Burn parameters are lazy, and clones of an unmaterialized parameter each
/// draw their own value; materializing first makes clones compare equal.
fn model<B: Backend>(device: &B::Device) -> Model<B> {
    let block = || Block {
        up: LinearConfig::new(WIDTH, 4 * WIDTH).init(device),
        o_proj: LinearConfig::new(4 * WIDTH, WIDTH).init(device),
        scale: Param::from_tensor(Tensor::ones([WIDTH], device)),
    };
    Model {
        embed: EmbeddingConfig::new(VOCABULARY, WIDTH).init(device),
        blocks: vec![block(), block()],
        // The crate's column-layout constructor keeps the stored weight an
        // autodiff leaf (Burn 0.21's own column layout does not).
        head: crate::operations::init_linear(
            &LinearConfig::new(WIDTH, VOCABULARY)
                .with_bias(false)
                .with_layout(LinearLayout::Col),
            device,
        ),
    }
    .map(&mut Materialize)
}

/// Host copy of a tensor's values in F64.
fn values<const D: usize>(tensor: Tensor<B, D>) -> Vec<f64> {
    tensor
        .into_data()
        .to_vec::<f32>()
        .unwrap()
        .into_iter()
        .map(f64::from)
        .collect()
}

/// Sample mean and (population) standard deviation.
fn moments(values: &[f64]) -> (f64, f64) {
    let count = values.len() as f64;
    let mean = values.iter().sum::<f64>() / count;
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / count;
    (mean, variance.sqrt())
}

/// Asserts a tensor looks like `N(0, std^2)`: standard deviation within 5%, mean near zero.
fn assert_normal<const D: usize>(tensor: Tensor<B, D>, std: f64, name: &str) {
    let samples = values(tensor);
    let (mean, actual) = moments(&samples);
    assert!(
        (actual / std - 1.0).abs() < 0.05,
        "{name}: std {actual} != {std}"
    );
    assert!(
        mean.abs() < 4.0 * std / (samples.len() as f64).sqrt(),
        "{name}: mean {mean}"
    );
}

/// Each role receives its distribution: N(0, 0.02), depth-scaled residual outputs, zero biases.
#[test]
fn roles_receive_their_distributions_and_residual_outputs_are_depth_scaled() {
    let device = Default::default();
    B::seed(&device, 0);
    let policy = InitPolicy::normal(0.02).with_depth_scaling(2, ["o_proj"]);
    let (model, report) = policy.apply(model::<B>(&device)).unwrap();
    assert_normal(model.embed.weight.val(), 0.02, "embedding");
    assert_normal(model.head.weight.val(), 0.02, "head");
    for (index, block) in model.blocks.iter().enumerate() {
        assert_normal(block.up.weight.val(), 0.02, "up");
        // 1 / sqrt(2 * 2 layers) = 1/2.
        assert_normal(block.o_proj.weight.val(), 0.01, "o_proj");
        for bias in [&block.up.bias, &block.o_proj.bias] {
            assert!(
                values(bias.as_ref().unwrap().val())
                    .iter()
                    .all(|v| *v == 0.0)
            );
        }
        assert_eq!(values(block.scale.val()), vec![1.0; WIDTH], "block {index}");
    }
    let role = |path: &str| {
        report
            .parameters
            .iter()
            .find(|parameter| parameter.path == path)
            .unwrap_or_else(|| panic!("{path} missing from {report:?}"))
            .clone()
    };
    assert_eq!(role("embed.weight").role, ParameterRole::Embedding);
    assert_eq!(role("blocks.0.up.weight").role, ParameterRole::Projection);
    assert_eq!(
        role("blocks.1.o_proj.weight").role,
        ParameterRole::ResidualProjection
    );
    assert_eq!(
        role("blocks.1.o_proj.weight").init,
        ParameterInit::Normal { std: 0.01 }
    );
    assert_eq!(role("blocks.0.o_proj.bias").role, ParameterRole::Bias);
    assert_eq!(role("blocks.0.scale").role, ParameterRole::Other);
    assert_eq!(role("blocks.0.scale").init, ParameterInit::Keep);
    assert_eq!(report.parameters.len(), 12);
    assert!(
        report
            .parameters
            .windows(2)
            .all(|pair| pair[0].path < pair[1].path)
    );
}

/// Keeping every role leaves values and identities exactly as constructed.
#[test]
fn keep_policy_changes_nothing() {
    let device = Default::default();
    let original = model::<B>(&device);
    let (kept, report) = InitPolicy::keep().apply(original.clone()).unwrap();
    assert_eq!(kept.embed.weight.id, original.embed.weight.id);
    assert_eq!(
        values(kept.embed.weight.val()),
        values(original.embed.weight.val())
    );
    assert_eq!(
        values(kept.head.weight.val()),
        values(original.head.weight.val())
    );
    assert!(
        report
            .parameters
            .iter()
            .all(|parameter| parameter.init == ParameterInit::Keep)
    );
    assert_eq!(InitPolicy::default(), InitPolicy::keep());
}

/// Overrides beat role rules, match whole path components, and the last match wins.
#[test]
fn overrides_take_precedence_in_registration_order() {
    let device = Default::default();
    let original = model::<B>(&device);
    let policy = InitPolicy::normal(0.02)
        .with_override("blocks", ParameterInit::Zeros)
        .with_override("blocks.0.up", ParameterInit::Keep)
        .with_override("head.weight", ParameterInit::Zeros);
    let (model, _) = policy.apply(original.clone()).unwrap();
    assert_eq!(
        values(model.blocks[0].up.weight.val()),
        values(original.blocks[0].up.weight.val())
    );
    for tensor in [
        model.blocks[0].o_proj.weight.val(),
        model.blocks[1].up.weight.val(),
        model.head.weight.val(),
    ] {
        assert!(values(tensor).iter().all(|value| *value == 0.0));
    }
    assert!(
        values(model.blocks[1].scale.val())
            .iter()
            .all(|value| *value == 0.0)
    );
    // `block` is not a component prefix of `blocks`.
    let partial = InitPolicy::keep().with_override("block", ParameterInit::Zeros);
    let (model, _) = partial.apply(original.clone()).unwrap();
    assert_eq!(
        values(model.blocks[0].scale.val()),
        values(original.blocks[0].scale.val())
    );
    // Residual scaling still applies to an overridden residual projection.
    let scaled = InitPolicy::keep()
        .with_depth_scaling(8, ["o_proj"])
        .with_override("blocks", ParameterInit::Normal { std: 1.0 });
    assert_eq!(
        scaled.resolve("blocks.0.o_proj.weight", ParameterRole::ResidualProjection),
        ParameterInit::Normal { std: 0.25 }
    );
}

/// Identity, dtype, device, trainability, and the column-layout record mapper survive.
#[test]
fn replacements_preserve_identity_trainability_and_record_layout() {
    let device = Default::default();
    let original = model::<B>(&device).no_grad();
    let (initialized, _) = InitPolicy::normal(0.5).apply(original.clone()).unwrap();
    assert_eq!(initialized.head.weight.id, original.head.weight.id);
    assert_eq!(
        initialized.blocks[1].up.weight.id,
        original.blocks[1].up.weight.id
    );
    assert_eq!(
        initialized.head.weight.val().dims(),
        original.head.weight.val().dims()
    );
    assert_eq!(
        initialized.head.weight.val().dtype(),
        original.head.weight.val().dtype()
    );
    assert!(!initialized.head.weight.val().is_require_grad());

    let recorder = BinBytesRecorder::<FullPrecisionSettings>::default();
    let bytes = recorder
        .record(initialized.clone().into_record(), ())
        .expect("record serializes");
    let reloaded = model::<B>(&device).load_record(
        recorder
            .load(bytes, &device)
            .expect("record deserializes into a fresh column-layout model"),
    );
    assert_eq!(
        values(reloaded.head.weight.val()),
        values(initialized.head.weight.val())
    );
}

/// Two fields with one identity receive one tensor, decided by the first field visited.
#[test]
fn tied_fields_share_one_replacement() {
    /// Two fields holding the same parameter identity.
    #[derive(Module, Debug)]
    struct Tied<B: Backend> {
        /// First registered owner.
        first: Param<Tensor<B, 2>>,
        /// Alias of `first`.
        second: Param<Tensor<B, 2>>,
    }
    let device = Default::default();
    let shared = Param::from_tensor(Tensor::<B, 2>::ones([8, 8], &device));
    let tied = Tied {
        first: shared.clone(),
        second: shared,
    };
    let policy = InitPolicy {
        other: ParameterInit::Normal { std: 1.0 },
        ..InitPolicy::keep()
    }
    .with_override("second", ParameterInit::Keep);
    let (tied, report) = policy.apply(tied).unwrap();
    assert_eq!(tied.first.id, tied.second.id);
    assert_eq!(values(tied.first.val()), values(tied.second.val()));
    assert!(values(tied.first.val()).iter().any(|value| *value != 1.0));
    assert!(
        report
            .parameters
            .iter()
            .all(|parameter| parameter.init == ParameterInit::Normal { std: 1.0 })
    );
}

/// Packed quantized weights are always kept; their dense neighbors are still initialized.
#[test]
fn packed_parameters_are_kept() {
    use crate::checkpoint::quantization::{
        QuantizationGroupSize, SymmetricQuantization, WeightQuantizationConfig,
        quantize_tensor_data,
    };
    let device = Default::default();
    let packed = quantize_tensor_data(
        &burn::tensor::TensorData::new(
            (0..64).map(|i| i as f32 / 8.0 - 4.0).collect::<Vec<_>>(),
            [2, 32],
        ),
        WeightQuantizationConfig {
            scheme: SymmetricQuantization::Q4,
            group_size: QuantizationGroupSize::new(32).unwrap(),
        },
    )
    .unwrap()
    .into_burn_qfloat_data()
    .unwrap();
    let mut linear = LinearConfig::new(2, 32).init::<B>(&device);
    linear.weight = Param::from_tensor(Tensor::from_data(packed, &device));
    let original = linear.weight.val().into_data();
    let (linear, report) = InitPolicy::normal(0.02).apply(linear).unwrap();
    assert_eq!(linear.weight.val().into_data(), original);
    let weight = report
        .parameters
        .iter()
        .find(|parameter| parameter.path == "weight")
        .unwrap();
    assert_eq!(weight.init, ParameterInit::Keep);
    assert!(
        values(linear.bias.unwrap().val())
            .iter()
            .all(|value| *value == 0.0)
    );
}

/// Invalid deviations, depths, and names are rejected before any parameter changes.
#[test]
fn invalid_policies_are_rejected_with_field_paths() {
    for std in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let normal = ParameterInit::Normal { std };
        for (policy, field) in [
            (
                InitPolicy {
                    embedding: normal,
                    ..InitPolicy::keep()
                },
                "init.embedding",
            ),
            (InitPolicy::normal(std), "init.embedding"),
            (
                InitPolicy {
                    bias: normal,
                    ..InitPolicy::keep()
                },
                "init.bias",
            ),
            (
                InitPolicy {
                    other: normal,
                    ..InitPolicy::keep()
                },
                "init.other",
            ),
            (
                InitPolicy::keep().with_override("head", normal),
                "init.overrides[0].init",
            ),
        ] {
            let error = policy.validate().unwrap_err();
            assert!(
                matches!(&error, crate::Error::Configuration { field: actual, .. } if actual == field),
                "{std}: {error}"
            );
        }
    }
    let cases = [
        (
            InitPolicy::keep().with_depth_scaling(0, ["o_proj"]),
            "init.residual_scaling.layers",
        ),
        (
            InitPolicy::keep().with_depth_scaling(2, [""]),
            "init.residual_projections",
        ),
        (
            InitPolicy::keep().with_override("", ParameterInit::Zeros),
            "init.overrides[0].path",
        ),
    ];
    for (policy, field) in cases {
        let error = policy.apply(model::<B>(&Default::default())).unwrap_err();
        assert!(
            matches!(&error, crate::Error::Configuration { field: actual, .. } if actual == field),
            "{error}"
        );
    }
}

/// Policies and reports round-trip through JSON with readable tags.
#[test]
fn policies_and_reports_round_trip_through_json() {
    let policy = InitPolicy::normal(0.02)
        .with_depth_scaling(12, ["o_proj", "down_proj"])
        .with_override("lm_head.weight", ParameterInit::Zeros);
    let json = serde_json::to_value(&policy).unwrap();
    assert_eq!(json["embedding"]["kind"], "normal");
    assert_eq!(json["residual_scaling"]["kind"], "depth_scaled");
    assert_eq!(serde_json::from_value::<InitPolicy>(json).unwrap(), policy);
    let (_, report) = policy.apply(model::<B>(&Default::default())).unwrap();
    let json = serde_json::to_string(&report).unwrap();
    assert_eq!(serde_json::from_str::<InitReport>(&json).unwrap(), report);
}

/// Tied embeddings: Burn's `N(0, 1)` gives logit std ~ sqrt(d); `N(0, 0.02)` starts near ln V.
///
/// The final RMS normalization makes `||h||^2 ~ d`, so tied logits of tokens
/// other than the input have standard deviation `sigma * sqrt(d)`. This test
/// measures both that logit scale and the initial cross entropy for a
/// one-layer modular model.
#[test]
fn tied_embedding_logit_scale_follows_sigma_sqrt_d() {
    use crate::{
        CausalLmInput, ForwardOptions, LogitsSelection,
        architecture::{
            Activation, AttentionSpec, DecoderLayerSpec, DecoderSpec, DenseFeedForwardSpec,
            FeedForwardSpec, MixerSpec, NormPlacement, NormalizationSpec, PositionEncodingSpec,
            ResidualTopology,
        },
        models::custom::ModularCausalLm,
        training::cross_entropy,
    };
    let device = Default::default();
    let (width, vocabulary, batch, tokens) = (256, 512, 4, 32);
    let norm = NormalizationSpec::Rms {
        epsilon: 1e-6,
        zero_centered: false,
    };
    let spec = DecoderSpec {
        vocabulary_size: vocabulary,
        hidden_size: width,
        layers: vec![DecoderLayerSpec {
            mixer: MixerSpec::Attention(AttentionSpec {
                query_heads: 4,
                key_value_heads: 4,
                head_dim: 64,
                query_key_norm: false,
                output_gate: false,
                position: PositionEncodingSpec::None,
            }),
            feed_forward: FeedForwardSpec::Dense(DenseFeedForwardSpec {
                intermediate_size: 512,
                activation: Activation::Silu,
                bias: false,
            }),
            mixer_norm: norm,
            feed_forward_norm: norm,
            norm_placement: NormPlacement::Pre,
            residual: ResidualTopology::Add,
        }],
        final_norm: norm,
        tie_word_embeddings: true,
        max_position_embeddings: 64,
    };
    let ids = (0..batch * tokens)
        .map(|index| ((index * 37 + 11) % vocabulary) as i32)
        .collect::<Vec<_>>();
    let targets = ids
        .iter()
        .map(|id| (id * 5 + 3) % vocabulary as i32)
        .collect::<Vec<_>>();
    let measure = |policy: &InitPolicy| {
        B::seed(&device, 1);
        let model = ModularCausalLm::<B>::init(spec.clone(), &device).unwrap();
        let (model, _) = policy.apply(model).unwrap();
        let logits = model
            .forward(
                CausalLmInput {
                    token_ids: Tensor::<B, 2, Int>::from_data(
                        burn::tensor::TensorData::new(ids.clone(), [batch, tokens]),
                        &device,
                    ),
                    token_mask: None,
                    position_ids: None,
                },
                &ForwardOptions {
                    logits: LogitsSelection::All,
                    ..Default::default()
                },
            )
            .unwrap()
            .logits
            .reshape([batch * tokens, vocabulary]);
        // The input token's own logit is excluded: `h` still contains its
        // embedding, so that one entry per row scales like sigma^2 d instead.
        let other_logits = values(logits.clone())
            .chunks(vocabulary)
            .zip(&ids)
            .flat_map(|(row, &own)| {
                row.iter()
                    .enumerate()
                    .filter(move |(column, _)| *column != own as usize)
                    .map(|(_, value)| *value)
            })
            .collect::<Vec<_>>();
        let (_, std) = moments(&other_logits);
        let loss = cross_entropy(
            logits,
            Tensor::from_data(
                burn::tensor::TensorData::new(targets.clone(), [batch * tokens]),
                &device,
            ),
        )
        .unwrap()
        .into_scalar();
        (std, f64::from(loss))
    };
    let uniform_loss = (vocabulary as f64).ln();
    let (burn_std, burn_loss) = measure(&InitPolicy::keep());
    assert!(
        (burn_std / (width as f64).sqrt() - 1.0).abs() < 0.2,
        "Burn default logit std {burn_std}"
    );
    assert!(
        burn_loss > 3.0 * uniform_loss,
        "Burn default loss {burn_loss}"
    );
    let (small_std, small_loss) = measure(&InitPolicy::normal(0.02));
    assert!(
        (small_std / (0.02 * (width as f64).sqrt()) - 1.0).abs() < 0.2,
        "N(0, 0.02) logit std {small_std}"
    );
    assert!(
        (small_loss - uniform_loss).abs() < 0.5,
        "N(0, 0.02) loss {small_loss} vs ln V {uniform_loss}"
    );
}

/// Initialized parameters remain trainable leaves on an autodiff backend.
#[cfg(feature = "autodiff")]
#[test]
fn initialized_parameters_receive_gradients() {
    use burn::backend::Autodiff;
    /// Differentiable CPU backend.
    type Ad = Autodiff<B>;
    let device = Default::default();
    let (model, _) = InitPolicy::normal(0.02)
        .with_depth_scaling(2, ["o_proj"])
        .apply(model::<Ad>(&device))
        .unwrap();
    let tokens = Tensor::<Ad, 2, Int>::from_ints([[1, 2, 3]], &device);
    let mut hidden = model.embed.forward(tokens);
    for block in &model.blocks {
        let expanded = block.up.forward(hidden.clone());
        hidden = hidden + block.o_proj.forward(expanded) * block.scale.val().reshape([1, 1, WIDTH]);
    }
    let loss = model.head.forward(hidden).powi_scalar(2).mean();
    let gradients = loss.backward();
    assert!(model.embed.weight.val().grad(&gradients).is_some());
    assert!(model.head.weight.val().grad(&gradients).is_some());
    for block in &model.blocks {
        assert!(block.up.weight.val().grad(&gradients).is_some());
        assert!(block.o_proj.weight.val().grad(&gradients).is_some());
        assert!(block.scale.val().grad(&gradients).is_some());
    }
}

proptest::proptest! {
    /// Resolution equals the last matching override (else the role rule), times residual scaling.
    #[test]
    fn resolution_matches_override_precedence_and_scaling(
        overrides in proptest::collection::vec((0usize..4, 0usize..3), 0..5),
        path_choice in 0usize..4,
        role_choice in 0usize..5,
        layers in 1usize..9,
    ) {
        let paths = ["a", "a.b", "a.b.weight", "ab.weight"];
        let inits = [ParameterInit::Keep, ParameterInit::Zeros, ParameterInit::Normal { std: 0.5 }];
        let roles = [
            ParameterRole::Embedding,
            ParameterRole::Projection,
            ParameterRole::ResidualProjection,
            ParameterRole::Bias,
            ParameterRole::Other,
        ];
        let mut policy = InitPolicy {
            other: ParameterInit::Normal { std: 2.0 },
            ..InitPolicy::normal(1.0)
        }
        .with_depth_scaling(layers, ["b"]);
        for &(path, init) in &overrides {
            policy = policy.with_override(paths[path], inits[init]);
        }
        let path = paths[path_choice];
        let role = roles[role_choice];
        let matching = overrides.iter().rev().find(|&&(candidate, _)| {
            let prefix = paths[candidate];
            path == prefix || path.starts_with(&format!("{prefix}."))
        });
        let base = match (matching, role) {
            (Some(&(_, init)), _) => inits[init],
            (None, ParameterRole::Embedding | ParameterRole::Projection | ParameterRole::ResidualProjection) => ParameterInit::Normal { std: 1.0 },
            (None, ParameterRole::Bias) => ParameterInit::Zeros,
            (None, ParameterRole::Other) => ParameterInit::Normal { std: 2.0 },
        };
        let expected = match (base, role) {
            (ParameterInit::Normal { std }, ParameterRole::ResidualProjection) => {
                ParameterInit::Normal { std: std / (2.0 * layers as f64).sqrt() }
            }
            (base, _) => base,
        };
        proptest::prop_assert_eq!(policy.resolve(path, role), expected);
    }
}
