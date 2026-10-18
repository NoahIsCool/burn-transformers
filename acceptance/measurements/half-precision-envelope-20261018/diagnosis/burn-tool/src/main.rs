//! Scratch diagnostic (half2 S9, not committed): per-block activations of a Llama-family checkpoint on Metal.
//!
//! Usage: `block-dump {f32|f16} MODEL_DIR TOKENS_FILE OUT_DIR`, where
//! `TOKENS_FILE` holds one prompt per line as comma-separated token IDs. The
//! checkpoint loads once on Metal in the given dtype (strict automatic
//! loading, `DTypePolicy::Cast`). For prompt `i` the tool writes raw
//! little-endian F32 files under `OUT_DIR/prompt<i>/`: `logits_plain.bin`
//! (an ordinary stateless pass, `[tokens, V]`), and from one hooked pass
//! `logits.bin` and, for every block `l`, `l<l>.pre.bin`, `l<l>.attn.bin`,
//! `l<l>.mlp.bin`, `l<l>.post.bin` (`[tokens, hidden]`: the residual entering
//! the block, the attention and MLP outputs before their residual additions,
//! and the residual after the block). The default graph exposes no attention
//! scores, so the model is then switched to the eager graph
//! (`AttentionPrecisionMode::EagerFp32Softmax`, scores multiplied by the scale
//! as Transformers' eager attention does), and a second loop over the prompts
//! writes `logits_eager.bin`, the same four points per block as
//! `l<l>.<point>.eager.bin`, and every block's scaled scores as
//! `l<l>.scores.bin` (`[heads, tokens, tokens]`). Half-precision values
//! convert to F32 exactly.

use std::{io::Write, path::PathBuf};

use burn::{
    backend::Metal,
    tensor::{DType, Int, Tensor, TensorData, backend::Backend},
};
use transformers::{
    AutoCausalLm, AutoLoadOptions, CausalLmInput, ForwardOptions, LogitsSelection,
    architecture::StagedModel,
    attention::AttentionPrecisionMode,
    checkpoint::{DTypePolicy, LoadOptions, TensorDType},
    intervention::{Hooks, Layer},
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

/// Wraps one row of token IDs as an unpadded stateless input.
fn input<B: Backend>(tokens: &[i64], device: &B::Device) -> CausalLmInput<B> {
    CausalLmInput {
        token_ids: Tensor::<B, 2, Int>::from_data(
            TensorData::new(tokens.to_vec(), [1, tokens.len()]),
            device,
        ),
        token_mask: None,
        position_ids: None,
    }
}

/// Loads the checkpoint once and dumps every prompt's per-block activations.
fn run<B: Backend>(device: &B::Device, model_dir: PathBuf, dtype: TensorDType, prompts: Vec<Vec<i64>>, out: PathBuf) {
    let options = AutoLoadOptions {
        checkpoint: LoadOptions {
            dtype_policy: DTypePolicy::Cast(dtype),
            ..LoadOptions::default()
        },
        load_tokenizer: false,
        ..AutoLoadOptions::default()
    };
    let model = AutoCausalLm::<B>::load_local(&model_dir, &options, device)
        .expect("strict load")
        .into_model();
    let layers = StagedModel::num_layers(&model);
    let every = ForwardOptions {
        logits: LogitsSelection::All,
        ..ForwardOptions::default()
    };
    for (index, tokens) in prompts.iter().enumerate() {
        let target = out.join(format!("prompt{index}"));
        std::fs::create_dir_all(&target).unwrap();
        let plain = model.forward(input::<B>(tokens, device), &every).unwrap();
        write(target.join("logits_plain.bin"), &host(plain.logits));
        residual_pass(&model, layers, tokens, device, &every, &target, "");
        eprintln!("prompt {index}: {} tokens, default graph, written to {}", tokens.len(), target.display());
    }
    let model = match model {
        AutoCausalLm::Llama(llama) => {
            AutoCausalLm::Llama(llama.with_attention_precision(AttentionPrecisionMode::EagerFp32Softmax))
        }
        _ => panic!("the eager graph is wired for the Llama family only"),
    };
    for (index, tokens) in prompts.iter().enumerate() {
        let target = out.join(format!("prompt{index}"));
        residual_pass(&model, layers, tokens, device, &every, &target, ".eager");
        let mut hooks = Hooks::new();
        let scores: Vec<_> = (0..layers)
            .map(|layer| hooks.capture(Layer::decoder(layer).self_attention().scores()))
            .collect();
        let output = model
            .forward_with_hooks(input::<B>(tokens, device), &every, &mut hooks)
            .unwrap();
        write(target.join("logits_eager.bin"), &host(output.logits));
        for (layer, capture) in scores.iter().enumerate() {
            write(target.join(format!("l{layer}.scores.bin")), &host(hooks.get(capture).unwrap()));
        }
        eprintln!("prompt {index}: {} tokens, eager graph, written to {}", tokens.len(), target.display());
    }
}

/// One hooked stateless pass writing each block's residual, attention, and MLP points with `suffix`.
fn residual_pass<B: Backend>(
    model: &AutoCausalLm<B>,
    layers: usize,
    tokens: &[i64],
    device: &B::Device,
    every: &ForwardOptions,
    target: &std::path::Path,
    suffix: &str,
) {
    let mut hooks = Hooks::new();
    let captures: Vec<_> = (0..layers)
        .map(|layer| {
            let at = Layer::decoder(layer);
            [
                ("pre", hooks.capture(at.residual_pre())),
                ("attn", hooks.capture(at.attention_output())),
                ("mlp", hooks.capture(at.mlp_output())),
                ("post", hooks.capture(at.residual_post())),
            ]
        })
        .collect();
    let output = model
        .forward_with_hooks(input::<B>(tokens, device), every, &mut hooks)
        .unwrap();
    write(target.join(format!("logits{}.bin", if suffix.is_empty() { "" } else { "_hooked_eager" })), &host(output.logits));
    for (layer, points) in captures.iter().enumerate() {
        for (name, capture) in points {
            write(target.join(format!("l{layer}.{name}{suffix}.bin")), &host(hooks.get(capture).unwrap()));
        }
    }
}

/// Parses the arguments and runs on Metal.
fn main() {
    let arguments: Vec<String> = std::env::args().collect();
    let dtype = match arguments[1].as_str() {
        "f32" => TensorDType::F32,
        "f16" => TensorDType::F16,
        other => panic!("unknown dtype {other}"),
    };
    let model_dir = PathBuf::from(&arguments[2]);
    let prompts: Vec<Vec<i64>> = std::fs::read_to_string(&arguments[3])
        .unwrap()
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| line.split(',').map(|t| t.trim().parse().unwrap()).collect())
        .collect();
    run::<Metal>(&Default::default(), model_dir, dtype, prompts, PathBuf::from(&arguments[4]));
}
