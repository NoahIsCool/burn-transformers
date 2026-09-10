//! Deterministic tiny topology shared by family-local tests.

use burn::{tensor::TensorData, tensor::backend::Backend};

use super::*;
use crate::{checkpoint::TensorSink, models::deepseek_v2::DeepseekV2CheckpointOptions};

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
