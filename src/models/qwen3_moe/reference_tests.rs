//! Golden parity with the pinned Transformers 5.2.0 Qwen3-MoE reference.
//!
//! The reference is `modeling_qwen3_moe.py` at Transformers commit
//! `7d9754a05193eb79b1d86aa744b622b8068008cd` with eager attention and eager
//! experts. Weights are not stored: both sides derive every tensor from its
//! released checkpoint name, so the fixture is a formula plus 112 logits.
//!
//! # Weight formula
//!
//! For checkpoint tensor `name` with `seed = sum(UTF-8 bytes of name)`,
//! element `k` (row-major, Hugging Face layout) is
//! `r_k = sin(0.37 (k + 1) + 0.11 seed)` in F64, then `1 + 0.1 r_k` for
//! RMSNorm weights (names ending in `norm.weight`) and `0.5 r_k` otherwise,
//! rounded once to F32. The reference fuses each sparse layer's experts as
//! `gate_up_proj[e] = cat(gate_proj_e, up_proj_e)` and `down_proj[e]`, built
//! from the per-expert names `model.layers.{i}.mlp.experts.{e}.*_proj.weight`
//! that the released checkpoint (and this crate's loader) uses.
//!
//! # Regenerating
//!
//! ```text
//! sys.path.insert(0, "<transformers@7d9754a>/src")
//! config = Qwen3MoeConfig(vocab_size=16, hidden_size=8, intermediate_size=12,
//!     num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
//!     max_position_embeddings=32, rms_norm_eps=1e-6, tie_word_embeddings=False,
//!     rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
//!     use_sliding_window=True, sliding_window=3, max_window_layers=1,
//!     decoder_sparse_step=1, moe_intermediate_size=6, num_experts=4,
//!     num_experts_per_tok=2, norm_topk_prob=True, mlp_only_layers=[0])
//! config._attn_implementation = config._experts_implementation = "eager"
//! model = Qwen3MoeForCausalLM(config).eval()   # then install the formula
//! model(input_ids=torch.tensor([[3, 1, 4, 1, 5, 9, 2]])).logits[0]
//! ```

use burn::{
    backend::Flex,
    tensor::{Int, Tensor, TensorData, Tolerance},
};

use super::*;
use crate::{
    CausalLanguageModel, DecodeInput, ForwardOptions, LogitsSelection,
    cache::CacheConfig,
    checkpoint::{TensorOperation, TensorSink, TensorTransform},
    models::{config::RopeParameters, qwen3_moe::Qwen3MoeCheckpointOptions},
};

/// CPU Flex backend with F32 activations and portable I32 indices.
type TestBackend = Flex<f32, i32>;

/// Prompt evaluated by the reference.
const TOKENS: [i32; 7] = [3, 1, 4, 1, 5, 9, 2];

/// Reference logits `[position, vocabulary]` for [`TOKENS`].
const REFERENCE_LOGITS: [[f32; 16]; 7] = [
    [
        0.8342373,
        -0.72065586,
        0.58337545,
        -0.42691052,
        0.25640643,
        -0.0774703,
        -0.10401338,
        0.28207648,
        -0.45086354,
        0.6048237,
        -0.7388941,
        0.8486656,
        -0.9305284,
        0.98179054,
        -1.000766,
        0.986831,
    ],
    [
        0.5118939,
        -0.39176553,
        0.2587536,
        -0.1172325,
        -0.028143764,
        0.17259449,
        -0.31136945,
        0.43990487,
        -0.55397385,
        0.6498253,
        -0.72430694,
        0.77496946,
        -0.8001467,
        0.79901105,
        -0.77159953,
        0.71881354,
    ],
    [
        0.03553009,
        -0.02956438,
        0.022626638,
        -0.014944315,
        0.0067710876,
        0.001625061,
        -0.009967804,
        0.017982602,
        -0.025406241,
        0.031994224,
        -0.037530124,
        0.04183188,
        -0.044757888,
        0.046212077,
        -0.046146482,
        0.044563413,
    ],
    [
        0.53721434,
        -0.41117054,
        0.271605,
        -0.12310791,
        -0.02943772,
        0.18101537,
        -0.3266402,
        0.46152335,
        -0.5812291,
        0.68182087,
        -0.7599908,
        0.81316817,
        -0.839604,
        0.83842933,
        -0.8096826,
        0.75430906,
    ],
    [
        1.0476491,
        -0.95721674,
        0.835306,
        -0.6859259,
        0.51398885,
        -0.32514906,
        0.12561679,
        0.07804656,
        -0.27914333,
        0.47106028,
        -0.6474863,
        0.8026196,
        -0.93135834,
        1.029469,
        -1.0937252,
        1.1220138,
    ],
    [
        0.21707922,
        -0.2954175,
        0.3640409,
        -0.4206925,
        0.46350968,
        -0.4910841,
        0.50250894,
        -0.49740875,
        0.47595096,
        -0.4388413,
        0.38730025,
        -0.3230228,
        0.24812251,
        -0.1650626,
        0.076574564,
        0.014431596,
    ],
    [
        -0.2515519,
        0.17890507,
        -0.100375,
        0.018543929,
        0.063896954,
        -0.1442365,
        0.21983278,
        -0.28819978,
        0.34708923,
        -0.39456457,
        0.42906448,
        -0.4494545,
        0.45506394,
        -0.44570845,
        0.42169574,
        -0.38381526,
    ],
];

/// The reference configuration: a dense block 0, a sparse block 1, and `max_window_layers = 1`.
///
/// Transformers 5.2 ignores `max_window_layers` and windows every block, so
/// this configuration separates the pinned semantics from the 4.51 rule that
/// kept block 0 on full attention.
fn reference_config() -> Qwen3MoeConfig {
    Qwen3MoeConfig {
        vocab_size: 16,
        hidden_size: 8,
        intermediate_size: 12,
        num_hidden_layers: 2,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: 4,
        max_position_embeddings: 32,
        rms_norm_eps: 1e-6,
        tie_word_embeddings: false,
        rope_parameters: Some(RopeParameters {
            rope_theta: Some(10_000.0),
            ..RopeParameters::default()
        }),
        rope_theta: None,
        use_sliding_window: true,
        sliding_window: Some(3),
        max_window_layers: 1,
        decoder_sparse_step: 1,
        moe_intermediate_size: 6,
        num_experts: 4,
        num_experts_per_tok: 2,
        norm_topk_prob: true,
        mlp_only_layers: vec![0],
        ..Qwen3MoeConfig::default()
    }
}

/// Evaluates the module-level weight formula for checkpoint tensor `name`.
fn formula(name: &str, elements: usize) -> Vec<f32> {
    let seed = name.bytes().map(f64::from).sum::<f64>();
    let norm = name.ends_with("norm.weight");
    (0..elements)
        .map(|element| {
            let raw = (0.37 * (element as f64 + 1.0) + 0.11 * seed).sin();
            (if norm { 1.0 + 0.1 * raw } else { 0.5 * raw }) as f32
        })
        .collect()
}

/// Installs the formula through the checkpoint contract's name mapping.
fn reference_model(
    device: &<TestBackend as burn::tensor::backend::BackendTypes>::Device,
) -> Qwen3MoeForCausalLm<TestBackend> {
    let config = reference_config();
    let contract = config
        .checkpoint_contract(Qwen3MoeCheckpointOptions::default())
        .unwrap();
    let mut model = Qwen3MoeForCausalLm::init(config, device).unwrap();
    let mut installed = 0;
    for operation in contract.mappings().operations() {
        let TensorOperation::Map {
            source,
            target,
            transform,
        } = operation
        else {
            panic!("untied Qwen3-MoE maps every tensor one to one: {operation:?}");
        };
        assert_eq!(*transform, TensorTransform::Identity, "{source}");
        let shape = contract
            .expected_tensors()
            .iter()
            .find(|expected| &expected.name == target)
            .unwrap()
            .shape
            .iter()
            .map(|&dimension| usize::try_from(dimension).unwrap())
            .collect::<Vec<_>>();
        let values = formula(source, shape.iter().product());
        TensorSink::write_tensor(&mut model, target, TensorData::new(values, shape)).unwrap();
        installed += 1;
    }
    assert_eq!(installed, contract.expected_tensors().len());
    model
}

/// Reference logits for positions `range` as `[1, len, vocabulary]`.
fn reference(
    range: std::ops::Range<usize>,
    device: &<TestBackend as burn::tensor::backend::BackendTypes>::Device,
) -> Tensor<TestBackend, 3> {
    let rows = REFERENCE_LOGITS[range.clone()]
        .iter()
        .flatten()
        .copied()
        .collect::<Vec<_>>();
    Tensor::from_data(TensorData::new(rows, [1, range.len(), 16]), device)
}

/// Tolerance for F32 logits of magnitude about one after two tiny blocks.
///
/// Both sides compute in F32; remaining differences are reduction order in
/// widths of at most 12 and the libm `exp`/`sin`/`cos` of each runtime.
fn tolerance() -> Tolerance<f32> {
    Tolerance::absolute(2e-5).set_relative(2e-5)
}

/// Stateless logits match the 5.2 reference, which windows every block.
#[test]
fn windowed_forward_matches_the_pinned_reference() {
    let device = Default::default();
    let model = reference_model(&device);
    let logits = model
        .forward(
            crate::CausalLmInput {
                token_ids: Tensor::<TestBackend, 1, Int>::from_ints(TOKENS, &device).unsqueeze(),
                token_mask: None,
                position_ids: None,
            },
            &ForwardOptions {
                logits: LogitsSelection::All,
                ..Default::default()
            },
        )
        .unwrap()
        .logits;
    logits
        .into_data()
        .assert_approx_eq::<f32>(&reference(0..7, &device).into_data(), tolerance());
}

/// Cached prefill and one-token decode steps reproduce the same reference positions.
#[test]
fn windowed_cached_decode_matches_the_pinned_reference() {
    let device = Default::default();
    let model = reference_model(&device);
    let mut state = model
        .new_decode_state(CacheConfig::new(1, 8).unwrap(), 1, &device)
        .unwrap();
    let all = ForwardOptions {
        logits: LogitsSelection::All,
        ..Default::default()
    };
    let prefill = model
        .prefill(
            crate::CausalLmInput {
                token_ids: Tensor::<TestBackend, 1, Int>::from_ints(&TOKENS[..4], &device)
                    .unsqueeze(),
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &all,
        )
        .unwrap()
        .logits;
    prefill
        .into_data()
        .assert_approx_eq::<f32>(&reference(0..4, &device).into_data(), tolerance());
    for (position, &token) in TOKENS.iter().enumerate().skip(4) {
        let logits = model
            .decode(
                DecodeInput {
                    token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[token]], &device),
                    token_mask: None,
                    position_ids: None,
                },
                &mut state,
                &all,
            )
            .unwrap()
            .logits;
        logits.into_data().assert_approx_eq::<f32>(
            &reference(position..position + 1, &device).into_data(),
            tolerance(),
        );
    }
}
