//! Tiny seeded gpt-oss models and inputs shared by the family's unit tests.
//!
//! The tiny configuration has three layers (sliding, full, sliding) of width
//! 16 with a window of 3, far below the test capacities, so cached tests
//! cover ring storage and wraparound; four experts with two selected per
//! token; and YaRN RoPE. Models are materialized eagerly with weights that
//! are a pure function of the seed (see `Randomize`), so tests can run in
//! parallel and clone models safely.

use burn::{
    backend::Flex,
    module::{Module, ModuleMapper, Param},
    tensor::{Bool, Int, Tensor, TensorData, backend::Backend},
};

use crate::{
    CausalLmInput, TokenMask,
    models::gpt_oss::{GptOssConfig, GptOssForCausalLm},
};

/// Deterministic F32 CPU backend.
pub(crate) type B = Flex<f32, i32>;

/// Vocabulary of every tiny model.
pub(crate) const VOCABULARY: usize = 32;

/// Sliding window of the tiny model.
pub(crate) const WINDOW: usize = 3;

/// Returns the tiny configuration: three layers alternating sliding and full attention.
pub(crate) fn tiny_config() -> GptOssConfig {
    let rope_parameters = crate::models::config::RopeParameters {
        factor: Some(4.0),
        original_max_position_embeddings: Some(16),
        ..GptOssConfig::default_rope_parameters()
    };
    GptOssConfig {
        vocab_size: VOCABULARY,
        hidden_size: 16,
        intermediate_size: 12,
        num_hidden_layers: 3,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dim: 4,
        num_local_experts: 4,
        num_experts_per_tok: 2,
        sliding_window: Some(WINDOW),
        layer_types: GptOssConfig::default_layer_types(3),
        max_position_embeddings: 64,
        rope_parameters,
        rms_norm_eps: 1e-6,
        ..GptOssConfig::default()
    }
}

/// Replaces every parameter by a deterministic draw from a local generator.
///
/// Flex's seeded RNG is process-global, so tests running in parallel would
/// interleave its draws; a local SplitMix64 stream keeps every tiny model a
/// pure function of its seed. Matrices draw `U(-0.35, 0.35)`; rank-one
/// parameters (norm scales, biases, and sinks) draw `1 + U(-0.3, 0.3)`, so
/// norms, biases, and sinks are never trivial.
struct Randomize {
    /// SplitMix64 state.
    state: u64,
}

impl Randomize {
    /// Returns the next uniform draw in `[-1, 1)`.
    fn next_unit(&mut self) -> f32 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        // The top 24 bits give an exact F32 fraction in [0, 1).
        ((z >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

impl<B: Backend> ModuleMapper<B> for Randomize {
    /// Replaces one floating parameter with deterministic values of the same shape.
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let (id, tensor, mapper) = param.consume();
        let shape = tensor.shape();
        let (center, spread) = if D == 1 { (1.0, 0.3) } else { (0.0, 0.35) };
        let values: Vec<f32> = (0..shape.num_elements())
            .map(|_| center + spread * self.next_unit())
            .collect();
        let tensor = Tensor::from_data(TensorData::new(values, shape), &tensor.device());
        Param::from_mapped_value(id, tensor, mapper)
    }
}

/// Builds a seeded, fully materialized model of `config`.
pub(crate) fn model_from(config: GptOssConfig, seed: u64) -> GptOssForCausalLm<B> {
    GptOssForCausalLm::init(config, &Default::default())
        .expect("valid tiny configuration")
        .map(&mut Randomize { state: seed })
}

/// Builds the seeded tiny model.
pub(crate) fn tiny_model(seed: u64) -> GptOssForCausalLm<B> {
    model_from(tiny_config(), seed)
}

/// Uploads a row-major `[rows, columns]` integer matrix.
pub(crate) fn ints<const R: usize, const C: usize>(values: [[i64; C]; R]) -> Tensor<B, 2, Int> {
    Tensor::from_data(
        TensorData::new(values.concat(), [R, C]),
        &Default::default(),
    )
}

/// Uploads a row-major `[rows, columns]` validity matrix as a token mask.
pub(crate) fn mask<const R: usize, const C: usize>(values: [[bool; C]; R]) -> TokenMask<B> {
    let indicators = values.map(|row| row.map(i64::from));
    let valid: Tensor<B, 2, Bool> = ints(indicators).equal_elem(1);
    TokenMask::new(valid)
}

/// Builds a stateless or prefill input.
pub(crate) fn input(
    tokens: Tensor<B, 2, Int>,
    token_mask: Option<TokenMask<B>>,
    positions: Option<Tensor<B, 2, Int>>,
) -> CausalLmInput<B> {
    CausalLmInput {
        token_ids: tokens,
        token_mask,
        position_ids: positions,
    }
}

/// Reads an F32 tensor to the host.
pub(crate) fn host<const D: usize>(tensor: Tensor<B, D>) -> Vec<f32> {
    tensor.into_data().to_vec::<f32>().expect("F32 tensor")
}

/// Returns the largest absolute elementwise difference.
pub(crate) fn max_gap(left: &[f32], right: &[f32]) -> f32 {
    assert_eq!(left.len(), right.len(), "compared tensors differ in size");
    left.iter()
        .zip(right)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max)
}

/// Asserts two tensors agree within `tolerance` and returns the gap.
pub(crate) fn assert_close<const D: usize>(
    actual: Tensor<B, D>,
    expected: Tensor<B, D>,
    tolerance: f32,
    context: &str,
) -> f32 {
    assert_eq!(actual.dims(), expected.dims(), "{context}: shape");
    let gap = max_gap(&host(actual), &host(expected));
    assert!(
        gap <= tolerance,
        "{context}: max |Δ| = {gap:e} > {tolerance:e}"
    );
    gap
}
