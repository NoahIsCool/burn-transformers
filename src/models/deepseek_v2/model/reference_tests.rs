//! Tiny-model parity with Hugging Face Transformers 5.2 DeepSeek-V2 and V3.
//!
//! The goldens in `goldens.rs` cover what restating closed forms cannot:
//! YaRN frequencies, amplitude, and softmax correction beyond the original
//! context; low-rank queries; sigmoid `noaux_tc` and softmax
//! `group_limited_greedy` routing at model level; padding; and the official
//! auxiliary losses. Weights come from `tests_support::reference_value`.
//!
//! Tolerances: logits agree within `2e-5` absolute (`|logit| < 3`, F32 GEMM
//! and softmax reassociation between Burn and PyTorch), router logits within
//! `1e-5`, and routing weights within `2e-6`, matching the shared router's
//! HF tolerance. The generator (run as `python3 -I generator.py`) is:
//!
//! ```text
//! """Tiny-random DeepSeek goldens from Hugging Face Transformers 5.2 (torch, F32 CPU).
//!
//! Weights are a deterministic function of each checkpoint tensor name, so the
//! Rust tests install identical values through the strict checkpoint sink.
//! """
//! import json, math, sys
//! import torch
//! import torch.nn.functional as F
//! from transformers import DeepseekV2Config, DeepseekV3Config
//! from transformers.models.deepseek_v2.modeling_deepseek_v2 import DeepseekV2ForCausalLM
//! from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3ForCausalLM
//!
//! torch.set_grad_enabled(False)
//!
//!
//! def fnv1a(name: str) -> int:
//!     h = 0x811C9DC5
//!     for b in name.encode():
//!         h ^= b
//!         h = (h * 0x01000193) & 0xFFFFFFFF
//!     return h
//!
//!
//! def values(name, shape):
//!     n = math.prod(shape)
//!     seed = fnv1a(name) % 23
//!     out = []
//!     for i in range(n):
//!         code = (i * 7 + seed) % 23
//!         if name.endswith("layernorm.weight") or name.endswith("norm.weight"):
//!             out.append(1.0 + (code - 11) * 0.02)
//!         elif name.endswith("e_score_correction_bias"):
//!             out.append((code - 11) * 0.03)
//!         elif name.endswith("gate.weight"):
//!             out.append((code - 11) * 0.15)
//!         else:
//!             out.append((code - 11) * 0.06)
//!     return torch.tensor(out, dtype=torch.float32).reshape(shape)
//!
//!
//! def install(model, n_layers, first_dense, experts):
//!     sd = model.state_dict()
//!     new = {}
//!     for key, tensor in sd.items():
//!         if key.endswith("experts.gate_up_proj"):
//!             prefix = key[: -len("gate_up_proj")]
//!             mats = []
//!             for e in range(experts):
//!                 gate = values(f"{prefix}{e}.gate_proj.weight", list(tensor.shape[1:2]) and [tensor.shape[1] // 2, tensor.shape[2]])
//!                 up = values(f"{prefix}{e}.up_proj.weight", [tensor.shape[1] // 2, tensor.shape[2]])
//!                 mats.append(torch.cat([gate, up], 0))
//!             new[key] = torch.stack(mats)
//!         elif key.endswith("experts.down_proj"):
//!             prefix = key[: -len("down_proj")]
//!             new[key] = torch.stack([values(f"{prefix}{e}.down_proj.weight", list(tensor.shape[1:])) for e in range(experts)])
//!         else:
//!             new[key] = values(key, list(tensor.shape))
//!     model.load_state_dict(new, strict=True)
//!
//!
//! def capture_gates(model, layers):
//!     """Captures each sparse block's F32 router logits `[tokens, experts]` from its input."""
//!     logits = {}
//!     for i in layers:
//!         mlp = model.model.layers[i].mlp
//!         def hook(_m, args, i=i, mlp=mlp):
//!             hidden = args[0].reshape(-1, args[0].shape[-1])
//!             logits[i] = F.linear(hidden.float(), mlp.gate.weight.float())
//!         mlp.register_forward_pre_hook(hook)
//!     return logits
//!
//!
//! def official_aux(scores, topk_idx, bsz, seq_len, top_k, experts, seq_aux):
//!     # Verbatim transcription of MoEGate's aux loss in modeling_deepseek.py@604d566 (alpha omitted).
//!     topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
//!     if seq_aux:
//!         scores_for_seq_aux = scores.view(bsz, seq_len, -1)
//!         ce = torch.zeros(bsz, experts)
//!         ce.scatter_add_(1, topk_idx_for_aux_loss, torch.ones(bsz, seq_len * top_k)).div_(seq_len * top_k / experts)
//!         return (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean().item()
//!     mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=experts)
//!     ce = mask_ce.float().mean(0)
//!     Pi = scores.mean(0)
//!     fi = ce * experts
//!     return (Pi * fi).sum().item()
//!
//!
//! def fmt(xs):
//!     return "[" + ", ".join(repr(float(x)) for x in xs) + "]"
//!
//!
//! result = {}
//!
//! # DeepSeek-V3: sigmoid/noaux_tc + correction bias, q_lora, YaRN beyond the original context.
//! v3 = DeepseekV3Config(
//!     vocab_size=32, hidden_size=8, intermediate_size=12, moe_intermediate_size=6,
//!     num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
//!     n_shared_experts=1, n_routed_experts=8, routed_scaling_factor=2.5,
//!     kv_lora_rank=5, q_lora_rank=6, qk_rope_head_dim=4, v_head_dim=4, qk_nope_head_dim=3,
//!     n_group=4, topk_group=2, num_experts_per_tok=3, first_k_dense_replace=1,
//!     norm_topk_prob=True, max_position_embeddings=32, rms_norm_eps=1e-6,
//!     rope_parameters={"rope_type": "yarn", "rope_theta": 10000.0, "factor": 4.0,
//!                      "original_max_position_embeddings": 8, "beta_fast": 32.0, "beta_slow": 1.0,
//!                      "mscale": 0.707, "mscale_all_dim": 0.707},
//!     attention_bias=False, tie_word_embeddings=False, rope_interleave=True,
//! )
//! v3._attn_implementation = "eager"
//! v3._experts_implementation = "eager"
//! m3 = DeepseekV3ForCausalLM(v3).eval()
//! install(m3, 2, 1, 8)
//! gates3 = capture_gates(m3, [1])
//! ids3 = torch.tensor([[3, 17, 5, 9, 30, 12], [7, 1, 22, 4, 4, 19]])
//! pos3 = torch.tensor([[20, 21, 22, 23, 24, 25], [0, 1, 2, 3, 4, 5]])
//! out3 = m3(input_ids=ids3, position_ids=pos3).logits
//! result["v3_logits"] = out3.flatten().tolist()
//! result["v3_router_logits"] = gates3[1].flatten().tolist()
//! moe3 = m3.model.layers[1].mlp
//! idx3, w3 = moe3.route_tokens_to_experts(gates3[1])
//! order = torch.argsort(idx3, dim=-1)
//! result["v3_ids_sorted"] = torch.gather(idx3, 1, order).flatten().tolist()
//! result["v3_weights_sorted"] = torch.gather(w3, 1, order).flatten().tolist()
//!
//! # DeepSeek-V2: softmax group_limited_greedy, no renormalization, routed scale, default RoPE.
//! v2 = DeepseekV2Config(
//!     vocab_size=32, hidden_size=8, intermediate_size=12, moe_intermediate_size=6,
//!     num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
//!     n_shared_experts=2, n_routed_experts=8, routed_scaling_factor=1.5,
//!     kv_lora_rank=6, q_lora_rank=None, qk_rope_head_dim=2, v_head_dim=4, qk_nope_head_dim=4,
//!     n_group=4, topk_group=2, num_experts_per_tok=3, first_k_dense_replace=1,
//!     topk_method="group_limited_greedy", norm_topk_prob=False, max_position_embeddings=32,
//!     rms_norm_eps=1e-6, rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
//!     attention_bias=False, tie_word_embeddings=False,
//! )
//! v2._attn_implementation = "eager"
//! v2._experts_implementation = "eager"
//! m2 = DeepseekV2ForCausalLM(v2).eval()
//! m2.model.layers[1].mlp.num_experts = 8  # 5.2 reads an attribute it never sets
//! install(m2, 2, 1, 8)
//! gates2 = capture_gates(m2, [1])
//! ids2 = torch.tensor([[3, 17, 5, 9, 30, 12], [0, 0, 22, 4, 4, 19]])
//! mask2 = torch.tensor([[1, 1, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1]])
//! pos2 = torch.tensor([[0, 1, 2, 3, 4, 5], [0, 0, 0, 1, 2, 3]])
//! out2 = m2(input_ids=ids2, attention_mask=mask2, position_ids=pos2).logits
//! result["v2_logits"] = out2.flatten().tolist()
//! moe2 = m2.model.layers[1].mlp
//! idx2, w2 = moe2.route_tokens_to_experts(gates2[1].view(2, 6, 8))
//! result["v2_router_logits"] = gates2[1].flatten().tolist()
//! order2 = torch.argsort(idx2, dim=-1)
//! result["v2_ids_sorted"] = torch.gather(idx2, 1, order2).flatten().tolist()
//! result["v2_weights_sorted"] = torch.gather(w2, 1, order2).flatten().tolist()
//!
//! # Official aux losses on row 0 alone (no padding) and on the valid tokens of both rows.
//! scores = gates2[1].view(2, 6, 8).softmax(dim=-1, dtype=torch.float32)
//! idx = idx2.view(2, 6, 3)
//! for seq_aux in (True, False):
//!     result[f"aux_row0_seq{int(seq_aux)}"] = official_aux(scores[0:1].reshape(6, 8), idx[0:1].reshape(6, 3), 1, 6, 3, 8, seq_aux)
//! # Padded batch: rows keep only their valid tokens (the reference has no padding).
//! valid1 = mask2[1].bool()
//! s1, i1 = scores[1][valid1], idx[1][valid1]
//! result["aux_padded_seq1"] = (official_aux(scores[0].reshape(6, 8), idx[0].reshape(6, 3), 1, 6, 3, 8, True)
//!                               + official_aux(s1, i1, 1, int(valid1.sum()), 3, 8, True)) / 2
//! result["aux_padded_seq0"] = official_aux(torch.cat([scores[0], s1]), torch.cat([idx[0], i1]), 1, 6 + int(valid1.sum()), 3, 8, False)
//!
//! json.dump(result, sys.stdout)
//! ```

use burn::{
    backend::Flex,
    tensor::{Bool, Int, Tensor, TensorData, Tolerance},
};

use super::{goldens::*, *};
use crate::{LogitsSelection, architecture::TokenMask};

/// Deterministic F32 CPU backend with I32 indices.
type TestBackend = Flex<f32, i32>;

/// Builds a `[2, 6]` integer tensor from row-major values.
fn grid(values: [[i64; 6]; 2]) -> Tensor<TestBackend, 2, Int> {
    Tensor::from_ints(values, &Default::default())
}

/// Requests all logits and router logits.
fn diagnostics() -> ForwardOptions {
    ForwardOptions {
        logits: LogitsSelection::All,
        router_logits: true,
        ..ForwardOptions::default()
    }
}

/// Sorts each token's `K` routed `(id, weight)` pairs by expert ID.
fn sorted_routes(ids: &[i64], weights: &[f32], k: usize) -> (Vec<i64>, Vec<f32>) {
    let mut sorted_ids = Vec::with_capacity(ids.len());
    let mut sorted_weights = Vec::with_capacity(weights.len());
    for (token_ids, token_weights) in ids.chunks_exact(k).zip(weights.chunks_exact(k)) {
        let mut pairs = token_ids
            .iter()
            .copied()
            .zip(token_weights.iter().copied())
            .collect::<Vec<_>>();
        pairs.sort_by_key(|&(id, _)| id);
        sorted_ids.extend(pairs.iter().map(|&(id, _)| id));
        sorted_weights.extend(pairs.iter().map(|&(_, weight)| weight));
    }
    (sorted_ids, sorted_weights)
}

/// Routes the model's own layer-1 router logits `[2, 6, 8]` with the layer's policy.
///
/// Router logits are compared with the golden separately, so this checks the
/// discrete selections and weights that the model derives from them.
fn layer_one_routes(
    model: &DeepseekV2ForCausalLm<TestBackend>,
    router_logits: Tensor<TestBackend, 3>,
) -> (Vec<i64>, Vec<f32>) {
    let sparse = &model.model.layers[1]
        .sparse_mlp
        .as_ref()
        .expect("layer one is sparse")
        .routed;
    let decision = sparse
        .router_config()
        .route(
            router_logits.reshape([12, 8]),
            sparse.correction_bias.as_ref().map(|bias| bias.val()),
        )
        .expect("routing succeeds");
    let ids = decision
        .experts
        .into_data()
        .convert::<i64>()
        .to_vec::<i64>()
        .unwrap();
    let weights = decision.weights.into_data().to_vec::<f32>().unwrap();
    sorted_routes(&ids, &weights, 3)
}

/// YaRN beyond the original context, q-LoRA, and sigmoid `noaux_tc` routing match HF V3.
#[test]
fn deepseek_v3_style_model_matches_transformers() {
    let device = Default::default();
    let model = tests_support::reference_model::<TestBackend>(
        tests_support::reference_v3_config(),
        &device,
    );
    let output = model
        .forward(
            CausalLmInput {
                token_ids: grid([[3, 17, 5, 9, 30, 12], [7, 1, 22, 4, 4, 19]]),
                token_mask: None,
                position_ids: Some(grid([[20, 21, 22, 23, 24, 25], [0, 1, 2, 3, 4, 5]])),
            },
            &diagnostics(),
        )
        .unwrap();
    output.logits.into_data().assert_approx_eq::<f32>(
        &TensorData::new(V3_LOGITS.to_vec(), [2, 6, 32]),
        Tolerance::absolute(2e-5),
    );
    output.routers[0]
        .logits
        .clone()
        .unwrap()
        .into_data()
        .assert_approx_eq::<f32>(
            &TensorData::new(V3_ROUTER_LOGITS.to_vec(), [2, 6, 8]),
            Tolerance::absolute(1e-5),
        );
    let (ids, weights) = layer_one_routes(&model, output.routers[0].logits.clone().unwrap());
    assert_eq!(ids, V3_IDS_SORTED);
    TensorData::new(weights, [36]).assert_approx_eq::<f32>(
        &TensorData::new(V3_WEIGHTS_SORTED.to_vec(), [36]),
        Tolerance::absolute(2e-6),
    );
}

/// Softmax `group_limited_greedy` routing and left padding match HF V2 on valid tokens.
#[test]
fn deepseek_v2_group_limited_model_matches_transformers_on_valid_tokens() {
    let device = Default::default();
    let model = tests_support::reference_model::<TestBackend>(
        tests_support::reference_v2_config(),
        &device,
    );
    let mask = Tensor::<TestBackend, 2, Bool>::from_bool(
        [
            [true, true, true, true, true, true],
            [false, false, true, true, true, true],
        ]
        .into(),
        &device,
    );
    let output = model
        .forward(
            CausalLmInput {
                token_ids: grid([[3, 17, 5, 9, 30, 12], [0, 0, 22, 4, 4, 19]]),
                token_mask: Some(TokenMask::new(mask)),
                position_ids: Some(grid([[0, 1, 2, 3, 4, 5], [0, 0, 0, 1, 2, 3]])),
            },
            &diagnostics(),
        )
        .unwrap();
    let logits = output.logits.into_data().to_vec::<f32>().unwrap();
    let router = output.routers[0]
        .logits
        .clone()
        .unwrap()
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    // Padded columns are excluded: Transformers does not define them.
    for (row, column) in (0..6)
        .map(|column| (0, column))
        .chain((2..6).map(|c| (1, c)))
    {
        let token = row * 6 + column;
        let at = |values: &[f32], width: usize| values[token * width..(token + 1) * width].to_vec();
        TensorData::new(at(&logits, 32), [32]).assert_approx_eq::<f32>(
            &TensorData::new(at(&V2_LOGITS, 32), [32]),
            Tolerance::absolute(2e-5),
        );
        TensorData::new(at(&router, 8), [8]).assert_approx_eq::<f32>(
            &TensorData::new(at(&V2_ROUTER_LOGITS, 8), [8]),
            Tolerance::absolute(1e-5),
        );
    }
    let (ids, weights) = layer_one_routes(&model, output.routers[0].logits.clone().unwrap());
    for token in (0..6).chain(8..12) {
        let route = token * 3..(token + 1) * 3;
        assert_eq!(
            ids[route.clone()],
            V2_IDS_SORTED[route.clone()],
            "token {token}"
        );
        TensorData::new(weights[route.clone()].to_vec(), [3]).assert_approx_eq::<f32>(
            &TensorData::new(V2_WEIGHTS_SORTED[route].to_vec(), [3]),
            Tolerance::absolute(2e-6),
        );
    }
}
