//! Differential attention against pinned DiffLlama references, oracles, and invalid inputs.

use burn::{
    backend::Flex,
    tensor::{Tensor, TensorData},
};

use super::*;

/// Deterministic F32 CPU backend.
type TestBackend = Flex<f32, i32>;

/// Reference fixture values, evaluated in F64 exactly as the generator script does.
fn values(count: usize, seed: usize, scale: f64) -> Vec<f32> {
    (0..count)
        .map(|index| {
            let base = ((index * 37 + seed * 11) % 23) as f64 * 0.1 - 1.1;
            (base * scale / 0.1) as f32
        })
        .collect()
}

/// `DiffLlamaAttention(layer_idx=1)` outputs pinned with transformers 5.2.0 / torch 2.10.0 (F32).
///
/// Config: hidden 8, 4 heads, head_dim 2, `rms_norm_eps = 1e-5`, no bias. The
/// q/k/v/o weights are `values(out * in, seed 1..=4, 0.05)` in `[out, in]`
/// order, lambdas are `q1 = [0.3, -0.2]`, `k1 = [0.5, 0.4]`, `q2 = [-0.1, 0.6]`,
/// `k2 = [0.2, -0.3]`, hidden states are `values(24, 9, 0.1)` as `[1, 3, 8]`,
/// RoPE is the identity, and the additive mask is causal.
const DIFF_LLAMA_REFERENCES: [(usize, [f64; 24]); 3] = [
    (
        2,
        [
            0.226_941_466,
            -0.294_352_323,
            0.367_269_009,
            -0.422_728_658,
            -0.944_022_417,
            0.563_861_966,
            0.280_386_746,
            -0.240_907_028,
            0.190_628_827,
            -0.351_260_453,
            0.395_694_137,
            -0.385_857_761,
            -0.927_747_011,
            0.556_552_231,
            0.303_859_115,
            -0.238_030_151,
            -0.324_741_095,
            -0.810_522_02,
            0.851_326_466,
            -0.197_055_638,
            -0.682_836_533,
            0.403_409_541,
            0.271_890_283,
            -0.213_890_642,
        ],
    ),
    (
        1,
        [
            0.475_599_08,
            0.076_368_078_6,
            0.442_329_764,
            -0.209_067_002,
            -0.608_298_004,
            -0.242_336_228,
            0.375_791_311,
            -0.023_439_675_6,
            0.477_986_336,
            0.068_473_070_9,
            0.443_264_425,
            -0.201_738_223,
            -0.611_251_473,
            -0.235_268_503,
            0.376_203_775,
            -0.033_309_474_6,
            0.363_595_188,
            0.384_741_247,
            0.335_952_371,
            -0.682_816_386,
            -0.661_670_327,
            -0.651_649_237,
            0.398_286_581,
            0.419_432_7,
        ],
    ),
    (
        4,
        [
            -0.076_086_476_4,
            -0.154_322_743,
            -0.924_852_61,
            0.278_203_249,
            0.199_967_027,
            0.727_919_161,
            0.037_821_382_3,
            -0.040_414_855_1,
            -0.004_637_897_01,
            -0.071_753_323_1,
            -0.946_728_528,
            0.245_573_267,
            0.178_457_767,
            0.687_556_386,
            0.039_458_453_7,
            -0.027_657_009_7,
            0.160_981_774,
            0.109_093_577,
            -0.868_222_296,
            0.169_390_649,
            0.117_502_391,
            0.576_503_932,
            -0.055_221_438_4,
            -0.107_109_644,
        ],
    ),
];

/// Builds the reference layer with DiffLlama's deterministic weights and lambdas.
fn reference_layer(
    key_value_heads: usize,
    precision: AttentionPrecisionMode,
) -> DifferentialAttention<TestBackend> {
    let device = Default::default();
    let mut config = DifferentialAttentionConfig::new(8, 4, key_value_heads, 2, 1);
    config.norm_epsilon = 1e-5;
    config.precision_mode = precision;
    let mut layer = config.init::<TestBackend>(&device).unwrap();
    for (seed, linear) in [
        &mut layer.q_proj,
        &mut layer.k_proj,
        &mut layer.v_proj,
        &mut layer.o_proj,
    ]
    .into_iter()
    .enumerate()
    {
        // Runtime weights are [in, out]; the reference values are [out, in].
        let [input, output] = linear.weight.val().dims();
        let reference = Tensor::<TestBackend, 2>::from_data(
            TensorData::new(values(output * input, seed + 1, 0.05), [output, input]),
            &device,
        );
        linear.weight = Param::from_tensor(reference.transpose());
    }
    let vector = |values: [f32; 2]| Param::from_tensor(Tensor::from_floats(values, &device));
    layer.lambda_q1 = vector([0.3, -0.2]);
    layer.lambda_k1 = vector([0.5, 0.4]);
    layer.lambda_q2 = vector([-0.1, 0.6]);
    layer.lambda_k2 = vector([0.2, -0.3]);
    layer
}

/// Both graphs reproduce DiffLlama for grouped, multi-query, and multi-head layouts.
#[test]
fn differential_attention_matches_pinned_diffllama_reference() {
    let device = Default::default();
    let hidden = Tensor::<TestBackend, 3>::from_data(
        TensorData::new(values(24, 9, 0.1), [1, 3, 8]),
        &device,
    );
    for (key_value_heads, reference) in DIFF_LLAMA_REFERENCES {
        for precision in [
            AttentionPrecisionMode::BackendDefault,
            AttentionPrecisionMode::EagerFp32Softmax,
        ] {
            let layer = reference_layer(key_value_heads, precision);
            let output = layer
                .attend(layer.project_self(hidden.clone()).unwrap(), None, true)
                .unwrap()
                .into_data()
                .to_vec::<f32>()
                .unwrap();
            for (actual, expected) in output.into_iter().zip(reference) {
                let actual = f64::from(actual);
                assert!(
                    (actual - expected).abs() <= 2e-5 + 2e-5 * expected.abs(),
                    "kv={key_value_heads} {precision:?}: {actual} vs {expected}"
                );
            }
        }
    }
}

/// An odd KV head count above one pairs per query head and matches the repeated layout.
#[test]
fn odd_key_value_head_counts_pair_per_query_head() {
    let device = Default::default();
    let mut config = DifferentialAttentionConfig::new(6, 6, 3, 2, 2);
    config.precision_mode = AttentionPrecisionMode::EagerFp32Softmax;
    let grouped = config.init::<TestBackend>(&device).unwrap();
    let hidden = Tensor::<TestBackend, 3>::from_data(
        TensorData::new(values(18, 4, 0.1), [1, 3, 6]),
        &device,
    );
    let projection = grouped.project_self(hidden).unwrap();
    // The same projection with K/V explicitly repeated to six heads.
    let mut repeated_config = config.clone();
    repeated_config.key_value_heads = 6;
    let mut repeated = repeated_config.init::<TestBackend>(&device).unwrap();
    // Lazily initialized parameters must be materialized explicitly to be shared.
    let output_weight = Tensor::<TestBackend, 2>::from_data(
        TensorData::new(values(12 * 6, 5, 0.1), [12, 6]),
        &device,
    );
    let mut grouped = grouped;
    grouped.o_proj.weight = Param::from_tensor(output_weight.clone());
    repeated.o_proj.weight = Param::from_tensor(output_weight);
    let repeated_projection = AttentionProjection {
        query: projection.query.clone(),
        key: repeat_key_value(projection.key.clone(), 6).unwrap(),
        value: repeat_key_value(projection.value.clone(), 6).unwrap(),
        gate: None,
    };
    grouped
        .attend(projection, None, true)
        .unwrap()
        .into_data()
        .assert_approx_eq::<f32>(
            &repeated
                .attend(repeated_projection, None, true)
                .unwrap()
                .into_data(),
            burn::tensor::Tolerance::absolute(1e-6),
        );
}

/// The lambda reparameterization and depth schedule follow their closed forms.
#[test]
fn lambda_follows_reparameterization_and_depth_schedule() {
    assert!((depth_lambda_init(0) - 0.2).abs() < 1e-12);
    assert!((depth_lambda_init(1) - (0.8 - 0.6 * (-0.3_f64).exp())).abs() < 1e-12);
    assert!(depth_lambda_init(40) < 0.8 && depth_lambda_init(40) > 0.79);
    let layer = reference_layer(2, AttentionPrecisionMode::BackendDefault);
    let lambda = layer
        .lambda(DType::F32)
        .into_data()
        .to_vec::<f32>()
        .unwrap()[0];
    let expected = (0.3_f64 * 0.5 - 0.2 * 0.4).exp() - (-0.1_f64 * 0.2 - 0.6 * 0.3).exp()
        + depth_lambda_init(1);
    assert!((f64::from(lambda) - expected).abs() < 1e-6);
}

/// Invalid configurations and projections fail with context.
#[test]
fn invalid_differential_layers_fail_with_context() {
    let base = DifferentialAttentionConfig::new(8, 4, 2, 2, 0);
    let cases = [
        (
            DifferentialAttentionConfig::new(8, 3, 1, 2, 0),
            "query_heads",
        ),
        (
            DifferentialAttentionConfig::new(8, 4, 3, 2, 0),
            "differential_attention",
        ),
        (
            DifferentialAttentionConfig::new(0, 4, 2, 2, 0),
            "differential_attention",
        ),
        (
            DifferentialAttentionConfig {
                lambda_init: f64::NAN,
                ..base.clone()
            },
            "lambda_init",
        ),
        (
            DifferentialAttentionConfig {
                norm_epsilon: 0.0,
                ..base.clone()
            },
            "norm_epsilon",
        ),
    ];
    for (config, field) in cases {
        let error = config.validate().unwrap_err().to_string();
        assert!(error.contains(field), "{error}");
    }
    let device = Default::default();
    let layer = base.init::<TestBackend>(&device).unwrap();
    let error = layer
        .project_self(Tensor::zeros([1, 2, 7], &device))
        .unwrap_err();
    assert!(error.to_string().contains("differential_attention.input"));
    let mut projection = layer
        .project_self(Tensor::zeros([1, 2, 8], &device))
        .unwrap();
    projection.key = projection.key.narrow(1, 0, 1);
    let error = layer.attend(projection, None, true).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("differential_attention.projection")
    );
}

/// All four lambda vectors and the projections receive gradients.
#[cfg(feature = "autodiff")]
#[test]
fn differential_attention_is_trainable() {
    use burn::backend::Autodiff;
    /// Differentiable CPU backend for the differential-attention gradient smoke test.
    type Ad = Autodiff<TestBackend>;
    let device = Default::default();
    let layer = DifferentialAttentionConfig::new(8, 4, 2, 2, 1)
        .init::<Ad>(&device)
        .unwrap();
    let hidden =
        Tensor::<Ad, 3>::from_data(TensorData::new(values(24, 9, 0.1), [1, 3, 8]), &device);
    let output = layer
        .attend(layer.project_self(hidden).unwrap(), None, true)
        .unwrap();
    let gradients = (output.clone() * output).sum().backward();
    for parameter in [
        &layer.lambda_q1,
        &layer.lambda_k1,
        &layer.lambda_q2,
        &layer.lambda_k2,
    ] {
        assert!(parameter.grad(&gradients).is_some());
    }
    assert!(layer.q_proj.weight.grad(&gradients).is_some());
}
