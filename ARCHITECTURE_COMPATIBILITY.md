# Transformer architecture compatibility

This document tracks configuration compatibility only. A checked configuration
means the crate can deserialize the official text topology (including wrapper and
legacy aliases) and reject contradictory tensor shapes before model allocation. It
does **not** claim that Burn layers, checkpoint records, tokenizers, multimodal
frontends, or end-to-end inference are implemented.

## Compatibility matrix

| Family | Reference checkpoint/config | Serialized identity | Wrapper and aliases | Topology represented and validated |
|---|---|---|---|---|
| Qwen3.5 / Qwen3.6 / Qwen3.8 dense hybrid | [Qwen3.8-27B `config.json`](https://huggingface.co/Qwen/Qwen3.8-27B/blob/72a217a/config.json), [Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B/blob/feea018b31f89dc0950e61da42577a7a4ab09169/config.json) | Inner text config remains `qwen3_5_text` for Qwen3.8 | Direct or nested `text_config`; `rope_scaling` alias; `Qwen38TextConfig` Rust alias | Explicit or interval-derived GDN versus full-attention schedule, gated-Q projection, GQA, linear head/state widths, convolution, multimodal RoPE sectioning, and MTP depth/embedding policy |
| Qwen3 MoE | [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/d47d535/config.json) | `qwen3_moe` | Direct or any supported text wrapper; `num_local_experts` and `rope_scaling` aliases | GQA/head dimensions, dense-versus-MoE cadence, dense overrides, expert width/count/top-k, router normalization, RoPE, and optional sliding window |
| DeepSeek-V2 / V2-Lite | [DeepSeek-V2](https://huggingface.co/deepseek-ai/DeepSeek-V2/blob/main/config.json), [DeepSeek-V2-Lite](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/blob/main/config.json) | `deepseek_v2` | Direct or `text_config`, `language_config`, or `llm_config`; old/new RoPE key aliases; nullable Q LoRA rank | MLA Q/K content and rotary widths, independent V width, Q/KV latent ranks, dense prefix and MoE cadence, grouped top-k capacity, shared/routed experts, and YaRN constraints |
| Muse Glimmer text decoder | [Muse-Glimmer-30B](https://huggingface.co/meta-models/Muse-Glimmer-30B/blob/46ac57a/config.json) | Inner `muse_glimmer_text` | Nested `text_config` or direct text config; `hidden_act` and `rope_scaling` aliases | GQA, explicit or backward-derived sliding/full schedule, per-layer RoPE/NoPE, sliding window, QK scaling, post norms, and final-logit scaling/softcap |
| T5 / T5 v1.1 / FLAN-T5 | [FLAN-T5-XL](https://huggingface.co/google/flan-t5-xl/blob/8772db7a7a11f7b08e6be7d7088f7a7fd4813bc5/config.json), [upstream T5 schema](https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/configuration_t5.py) | `t5` | Canonical names plus `hidden_size`, `head_dim`, `intermediate_size`, and standard head/depth aliases | Independent residual and attention-inner widths, asymmetric encoder/decoder depth, relative-position buckets/range, dense versus gated FFN layout, output-scaling flag, and decoder start/token IDs |
| UL2 | [Google UL2](https://huggingface.co/google/ul2/blob/71c34c5c6e3e54c5bf2cee360c3edfaa42acc1b3/config.json) | `t5` | `Ul2Config` Rust alias; accepts historic derived `dense_act_fn` and `is_gated_act` | T5 encoder-decoder topology plus gated-SiLU FFN consistency |
| Nemotron-H / Nemotron Nano | [Nemotron-H-8B](https://huggingface.co/nvidia/Nemotron-H-8B-Base-8K/blob/main/config.json), [Nemotron-3-Nano-30B-A3B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/blob/2fae43de5f4df1b5e58d6c44803dce1d19e9175f/config.json) | `nemotron_h` | Compact `hybrid_override_pattern` or modern `layers_block_type`/`layer_types`; historic `mamba_*` aliases; `attention_head_dim`; bare legacy `Infinity` | Exact heterogeneous Mamba/attention/MLP/MoE sequence, attention GQA/RoPE, Mamba state/head/group/convolution/time-step topology, expert routing, redundant norm consistency, and optional MTP block sequence |

## Loader behavior

`ModelConfig::from_json_str` first parses strict JSON after one deliberately narrow
compatibility repair: an unquoted positive `Infinity` token is converted to `null`.
NVIDIA's original Nemotron-H configs used that Python JSON extension for an
unbounded time-step maximum, while newer artifacts use `null`. Strings containing
the word `Infinity` are unchanged.

For a multimodal repository, only `text_config` is consumed. The compatibility
aliases `language_config` and `llm_config` support older/custom wrappers. Vision
metadata is ignored because this crate milestone covers transformer text backbones
only. Unknown non-topology metadata is also ignored so harmless Hub metadata does
not break loading; unknown enum values and contradictory shape fields fail closed.

## Known gaps and non-claims

- The schemas do not yet prove checkpoint tensor names or tensor dtypes; the
  checkpoint loader must compare every loaded tensor against dimensions derived
  here.
- Qwen3.8 uses the Qwen3.5 architecture discriminator in the official artifact.
  No invented `qwen3_8_text` serialized value is accepted.
- Muse Glimmer vision topology is intentionally not represented. Only its nested
  text decoder is extracted.
- Nemotron-H's historical configs are not RFC-compliant JSON because they contain
  `Infinity`; only the published positive-infinity case is repaired. `NaN` and
  negative infinity remain errors.
- Quantization, tensor/expert parallel partitioning, DFlash/MTP execution, and
  tokenizer/generation policy are outside these configuration contracts.
