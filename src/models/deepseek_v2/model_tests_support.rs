//! Deterministic tiny topology shared by family-local tests.

use burn::{tensor::TensorData, tensor::backend::Backend};

use super::*;
use crate::{
    checkpoint::{TensorOperation, TensorSink},
    models::deepseek_v2::{
        DeepseekScoringFunction, DeepseekTopKMethod, DeepseekV2CheckpointOptions,
    },
};

/// Returns a tiny graph with one dense and one sparse block.
pub(crate) fn tiny_config() -> DeepseekV2Config {
    DeepseekV2Config {
        vocab_size: 32,
        hidden_size: 8,
        intermediate_size: 12,
        moe_intermediate_size: 6,
        num_hidden_layers: 2,
        num_attention_heads: 2,
        num_key_value_heads: 2,
        n_shared_experts: 2,
        n_routed_experts: 4,
        ep_size: 1,
        routed_scaling_factor: 1.0,
        kv_lora_rank: 5,
        q_lora_rank: None,
        qk_nope_head_dim: 3,
        qk_rope_head_dim: 2,
        v_head_dim: 4,
        topk_method: DeepseekTopKMethod::Greedy,
        n_group: 1,
        topk_group: 1,
        num_experts_per_tok: 2,
        moe_layer_freq: 1,
        first_k_dense_replace: 1,
        norm_topk_prob: false,
        scoring_func: DeepseekScoringFunction::Softmax,
        aux_loss_alpha: 0.001,
        seq_aux: true,
        hidden_act: "silu".into(),
        max_position_embeddings: 32,
        initializer_range: 0.02,
        rms_norm_eps: 1e-6,
        use_cache: true,
        pretraining_tp: 1,
        tie_word_embeddings: false,
        rope_theta: 10_000.0,
        rope_scaling: None,
        attention_bias: false,
        attention_dropout: 0.0,
        use_mla: true,
        pad_token_id: None,
        bos_token_id: None,
        eos_token_id: None,
        model_type: "deepseek_v2".into(),
    }
}

/// Installs deterministic parameters through the strict family sink.
pub(crate) fn initialized_model<B: Backend>(
    config: DeepseekV2Config,
    device: &B::Device,
) -> DeepseekV2ForCausalLm<B> {
    let contract = config
        .checkpoint_contract(DeepseekV2CheckpointOptions::default())
        .expect("tiny contract");
    let mut model = DeepseekV2ForCausalLm::init(config, device).expect("tiny model");
    for (tensor_index, expected) in contract.expected_tensors().iter().enumerate() {
        let shape = expected
            .shape
            .iter()
            .map(|&dimension| usize::try_from(dimension).expect("tiny dimension"))
            .collect::<Vec<_>>();
        let elements = shape.iter().product();
        let values = (0..elements)
            .map(|element| {
                let code = (element + tensor_index * 7) % 23;
                (code as f32 - 11.0) * 0.002
            })
            .collect::<Vec<_>>();
        TensorSink::write_tensor(&mut model, &expected.name, TensorData::new(values, shape))
            .unwrap_or_else(|error| panic!("{}: {error}", expected.name));
    }
    model
}

/// Returns the 32-bit FNV-1a hash of a checkpoint tensor name.
fn fnv1a(name: &str) -> u32 {
    name.bytes().fold(0x811C_9DC5_u32, |hash, byte| {
        (hash ^ u32::from(byte)).wrapping_mul(0x0100_0193)
    })
}

/// Deterministic value of element `index` of checkpoint tensor `name`.
///
/// This is the weight function of the Hugging Face golden generator (module
/// documentation of `model/reference_tests.rs`): with `code = (7 index +
/// fnv1a(name) mod 23) mod 23`, norms hold `1 + 0.02 (code - 11)`, correction
/// biases `0.03 (code - 11)`, router gates `0.15 (code - 11)`, and every other
/// matrix `0.06 (code - 11)`. Values are formed in F64 and rounded once to
/// F32, exactly as `torch.tensor(..., dtype=float32)` rounds the Python floats.
pub(crate) fn reference_value(name: &str, index: usize) -> f32 {
    let seed = (fnv1a(name) % 23) as usize;
    let code = (index * 7 + seed) % 23;
    let centered = code as f64 - 11.0;
    let value = if name.ends_with("layernorm.weight") || name.ends_with("norm.weight") {
        1.0 + centered * 0.02
    } else if name.ends_with("e_score_correction_bias") {
        centered * 0.03
    } else if name.ends_with("gate.weight") {
        centered * 0.15
    } else {
        centered * 0.06
    };
    value as f32
}

/// Installs [`reference_value`] weights for every tensor of the strict contract.
///
/// Values are keyed by the *checkpoint* name of each mapped tensor (for
/// example `model.layers.1.mlp.experts.0.gate_proj.weight`), exactly as the
/// generator keys the Hugging Face state dict.
pub(crate) fn reference_model<B: Backend>(
    config: DeepseekV2Config,
    device: &B::Device,
) -> DeepseekV2ForCausalLm<B> {
    let contract = config
        .checkpoint_contract(DeepseekV2CheckpointOptions::default())
        .expect("reference contract");
    let mut model = DeepseekV2ForCausalLm::init(config, device).expect("reference model");
    for operation in contract.mappings().operations() {
        let TensorOperation::Map { source, target, .. } = operation else {
            continue;
        };
        let expected = contract
            .expected_tensors()
            .iter()
            .find(|expected| &expected.name == target)
            .expect("every mapped target is expected");
        let shape = expected
            .shape
            .iter()
            .map(|&dimension| usize::try_from(dimension).expect("tiny dimension"))
            .collect::<Vec<_>>();
        let values = (0..shape.iter().product())
            .map(|index| reference_value(source, index))
            .collect::<Vec<_>>();
        TensorSink::write_tensor(&mut model, target, TensorData::new(values, shape))
            .unwrap_or_else(|error| panic!("{target}: {error}"));
    }
    model
}

/// The tiny DeepSeek-V3-style topology of the Hugging Face golden.
///
/// Sigmoid `noaux_tc` routing with correction bias, a low-rank query, and
/// YaRN whose `max_position_embeddings / original_max_position_embeddings`
/// equals `factor`, so the DeepSeek and Hugging Face factor conventions agree.
pub(crate) fn reference_v3_config() -> DeepseekV2Config {
    DeepseekV2Config {
        n_shared_experts: 1,
        n_routed_experts: 8,
        routed_scaling_factor: 2.5,
        q_lora_rank: Some(6),
        qk_rope_head_dim: 4,
        topk_method: DeepseekTopKMethod::NoAuxTc,
        n_group: 4,
        topk_group: 2,
        num_experts_per_tok: 3,
        norm_topk_prob: true,
        scoring_func: DeepseekScoringFunction::Sigmoid,
        rope_scaling: Some(crate::models::config::RopeParameters {
            rope_type: crate::models::config::RopeType::Yarn,
            factor: Some(4.0),
            original_max_position_embeddings: Some(8),
            beta_fast: Some(32.0),
            beta_slow: Some(1.0),
            mscale: Some(0.707),
            mscale_all_dim: Some(0.707),
            ..Default::default()
        }),
        ..tiny_config()
    }
}

/// The tiny DeepSeek-V2 topology of the Hugging Face golden.
///
/// Softmax `group_limited_greedy` routing without renormalization, a routed
/// scale of 1.5, and default RoPE.
pub(crate) fn reference_v2_config() -> DeepseekV2Config {
    DeepseekV2Config {
        n_shared_experts: 2,
        n_routed_experts: 8,
        routed_scaling_factor: 1.5,
        kv_lora_rank: 6,
        qk_nope_head_dim: 4,
        topk_method: DeepseekTopKMethod::GroupLimitedGreedy,
        n_group: 4,
        topk_group: 2,
        num_experts_per_tok: 3,
        ..tiny_config()
    }
}
