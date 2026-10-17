//! Scratch diagnostic: dumps one sequence's residual stream and logits from a local Llama-family or Gemma checkpoint.
//!
//! Usage: `burn-dump {flex|metal} {f32|bf16|f16} MODEL_DIR TOKENS OUT_DIR [fused|eager] [PREFILL]`,
//! where `TOKENS` is a comma-separated list of token IDs. Writes raw little-endian
//! F32 files: `embedding.bin`, `layer<i>.bin` (residual after block `i`),
//! `final_norm.bin` (each `[tokens, hidden]`), `logits.bin` (`[tokens, V]`) from
//! one hooked stateless pass, and `logits_plain.bin` from an ordinary stateless
//! pass. `eager` selects the eager F32-softmax attention graph instead of the
//! backend's fused kernel. With `PREFILL > 0`, the first `PREFILL` tokens are
//! prefilled into a decode state and every later token is decoded one at a
//! time; `decode_logits.bin` holds the logits of each decoded position
//! (`[tokens - PREFILL, V]`, the stateless rows `PREFILL..tokens`).

use std::{io::Write, path::PathBuf};

use burn::{
    backend::{Flex, Metal},
    tensor::{DType, Int, Tensor, TensorData, backend::Backend},
};
use transformers::{
    AutoCausalLm, AutoLoadOptions, CausalLmInput, DecodeInput, ForwardOptions, LogitsSelection,
    architecture::StagedModel,
    attention::AttentionPrecisionMode,
    cache::CacheConfig,
    checkpoint::{DTypePolicy, LoadOptions, TensorDType},
    intervention::{Hooks, Layer, Stack},
};

/// Reads a tensor to the host as F32.
fn host<B: Backend, const D: usize>(tensor: Tensor<B, D>) -> Vec<f32> {
    tensor.cast(DType::F32).into_data().to_vec::<f32>().unwrap()
}

/// Writes values as raw little-endian F32.
fn write(path: PathBuf, values: &[f32]) {
    let mut file = std::fs::File::create(path).unwrap();
    for value in values {
        file.write_all(&value.to_le_bytes()).unwrap();
    }
}

/// Uploads one row of token IDs.
fn row<B: Backend>(tokens: &[i64], device: &B::Device) -> Tensor<B, 2, Int> {
    Tensor::<B, 2, Int>::from_data(TensorData::new(tokens.to_vec(), [1, tokens.len()]), device)
}

/// Wraps one row of token IDs as an unpadded stateless input.
fn input<B: Backend>(tokens: &[i64], device: &B::Device) -> CausalLmInput<B> {
    CausalLmInput {
        token_ids: row::<B>(tokens, device),
        token_mask: None,
        position_ids: None,
    }
}

/// Loads the checkpoint in `dtype` and dumps every captured point, the logits, and optionally the decode path.
fn run<B: Backend>(
    device: &B::Device,
    model_dir: PathBuf,
    dtype: TensorDType,
    tokens: Vec<i64>,
    out: PathBuf,
    eager: bool,
    prefill: usize,
) {
    let options = AutoLoadOptions {
        checkpoint: LoadOptions {
            dtype_policy: DTypePolicy::Cast(dtype),
            ..LoadOptions::default()
        },
        load_tokenizer: false,
        ..AutoLoadOptions::default()
    };
    let mut model = AutoCausalLm::<B>::load_local(&model_dir, &options, device)
        .expect("strict load")
        .into_model();
    if eager {
        model = match model {
            AutoCausalLm::Llama(llama) => {
                AutoCausalLm::Llama(llama.with_attention_precision(AttentionPrecisionMode::EagerFp32Softmax))
            }
            AutoCausalLm::Gemma(gemma) => {
                AutoCausalLm::Gemma(gemma.with_attention_precision(AttentionPrecisionMode::EagerFp32Softmax))
            }
            _ => panic!("eager attention is wired for the Llama family and Gemma only"),
        };
    }
    let layers = StagedModel::num_layers(&model);
    let mut points = vec![Stack::Decoder.embedding()];
    points.extend((0..layers).map(|index| Layer::decoder(index).residual_post()));
    points.push(Stack::Decoder.final_norm());
    let mut hooks = Hooks::new();
    let handles: Vec<_> = points.iter().map(|point| hooks.capture(point)).collect();
    let every = ForwardOptions {
        logits: LogitsSelection::All,
        ..ForwardOptions::default()
    };
    let output = model
        .forward_with_hooks(input::<B>(&tokens, device), &every, &mut hooks)
        .unwrap();
    std::fs::create_dir_all(&out).unwrap();
    write(out.join("logits.bin"), &host(output.logits));
    for (index, handle) in handles.iter().enumerate() {
        let name = match index {
            0 => "embedding".to_string(),
            i if i == layers + 1 => "final_norm".to_string(),
            i => format!("layer{}", i - 1),
        };
        write(out.join(format!("{name}.bin")), &host(hooks.get(handle).unwrap()));
    }
    let plain = model.forward(input::<B>(&tokens, device), &every).unwrap();
    write(out.join("logits_plain.bin"), &host(plain.logits));
    if prefill > 0 {
        let mut state = model
            .new_decode_state(CacheConfig::new(1, tokens.len()).unwrap(), 1, device)
            .unwrap();
        let _ = model
            .prefill(input::<B>(&tokens[..prefill], device), &mut state, &ForwardOptions::default())
            .unwrap();
        let mut decoded = Vec::new();
        for position in prefill..tokens.len() {
            let next = DecodeInput {
                token_ids: row::<B>(&tokens[position..=position], device),
                position_ids: None,
                token_mask: None,
            };
            decoded.extend(host(
                model
                    .decode(next, &mut state, &ForwardOptions::default())
                    .unwrap()
                    .logits,
            ));
        }
        write(out.join("decode_logits.bin"), &decoded);
    }
    eprintln!(
        "wrote {} points for {} tokens (eager={eager}, prefill={prefill}) to {}",
        points.len(),
        tokens.len(),
        out.display()
    );
}

/// Parses the arguments and dispatches to the backend.
fn main() {
    let arguments: Vec<String> = std::env::args().collect();
    let dtype = match arguments[2].as_str() {
        "f32" => TensorDType::F32,
        "bf16" => TensorDType::BF16,
        "f16" => TensorDType::F16,
        other => panic!("unknown dtype {other}"),
    };
    let model_dir = PathBuf::from(&arguments[3]);
    let tokens: Vec<i64> = arguments[4].split(',').map(|t| t.parse().unwrap()).collect();
    let out = PathBuf::from(&arguments[5]);
    let eager = match arguments.get(6).map(String::as_str) {
        None | Some("fused") => false,
        Some("eager") => true,
        Some(other) => panic!("unknown attention graph {other}"),
    };
    let prefill: usize = arguments.get(7).map(|p| p.parse().unwrap()).unwrap_or(0);
    match arguments[1].as_str() {
        "flex" => run::<Flex<f32, i32>>(&Default::default(), model_dir, dtype, tokens, out, eager, prefill),
        "metal" => run::<Metal>(&Default::default(), model_dir, dtype, tokens, out, eager, prefill),
        other => panic!("unknown backend {other}"),
    }
}
