//! Small input, loading, and verification helpers; training stays in the example.

use burn::{
    module::{Module, ModuleVisitor, Param, ParamId},
    optim::GradientsParams,
    tensor::{Int, Tensor, backend::Backend},
};
use sha2::{Digest, Sha256};
use std::path::PathBuf;
use transformers::{
    CausalLmInput,
    checkpoint::{DTypePolicy, LoadOptions, TensorDType},
    models::qwen3_5::{
        Qwen3_5BackboneRecipe, Qwen3_5LayerType, Qwen3_5RecipeLoadOptions, Qwen3_5TextConfig,
        Qwen3_5TextModel,
    },
};

pub struct Options {
    pub model_dir: Option<PathBuf>,
    pub steps: usize,
    pub output: PathBuf,
}

impl Options {
    pub fn parse() -> Result<Self, Box<dyn std::error::Error>> {
        let mut model_dir = None;
        let mut steps = 60;
        let mut output = None;
        let mut args = std::env::args().skip(1);
        while let Some(argument) = args.next() {
            if argument == "--help" {
                println!(
                    "Usage: train_qwen_adapter [--model-dir LOCAL_CHECKPOINT] [--steps 60] [--output-dir NEW_DIRECTORY]"
                );
                std::process::exit(0);
            }
            let value = args.next().ok_or("expected a value after each option")?;
            match argument.as_str() {
                "--model-dir" => model_dir = Some(PathBuf::from(value)),
                "--steps" => steps = value.parse()?,
                "--output-dir" => output = Some(PathBuf::from(value)),
                _ => {
                    return Err(format!(
                        "unknown option {argument}; use --model-dir, --steps, or --output-dir"
                    )
                    .into());
                }
            }
        }
        if steps < 4 {
            return Err("--steps must be at least 4 for the midpoint resume check".into());
        }
        let output = if let Some(path) = output {
            std::fs::create_dir(&path)?;
            path
        } else {
            tempfile::Builder::new()
                .prefix("qwen-adapter-training-")
                .tempdir()?
                .keep()
        };
        Ok(Self {
            model_dir,
            steps,
            output,
        })
    }
}

pub fn backbone<B: Backend>(
    options: &Options,
    device: &B::Device,
) -> Result<(Qwen3_5TextModel<B>, String), Box<dyn std::error::Error>> {
    let recipe = if let Some(root) = &options.model_dir {
        Qwen3_5BackboneRecipe::from_local_config(root)?.prefix(2)?
    } else {
        Qwen3_5BackboneRecipe::from_config(tiny_config())?
    };
    std::fs::write(
        options.output.join("backbone_recipe.json"),
        serde_json::to_vec_pretty(&recipe)?,
    )?;
    if let Some(root) = &options.model_dir {
        let loaded = recipe.load_pretrained::<B>(
            root,
            &Qwen3_5RecipeLoadOptions {
                checkpoint: LoadOptions {
                    dtype_policy: DTypePolicy::Cast(TensorDType::F32),
                    ..Default::default()
                },
                ..Default::default()
            },
            device,
        )?;
        println!(
            "Loaded two frozen blocks from {} source bytes.",
            loaded.checkpoint.stats.source_payload_bytes
        );
        Ok((loaded.model, root.display().to_string()))
    } else {
        Ok((
            recipe.init_random(device)?,
            "random Qwen architecture, seed 42".into(),
        ))
    }
}

/// Eight toy sequences; the label distinguishes final tokens 1/2 from 3/4.
pub fn batch<B: Backend>(device: &B::Device) -> (CausalLmInput<B>, Tensor<B, 1, Int>) {
    (
        CausalLmInput {
            token_ids: Tensor::<B, 2, Int>::from_ints(
                [
                    [1, 2, 1],
                    [2, 1, 2],
                    [3, 4, 1],
                    [4, 3, 2],
                    [1, 2, 3],
                    [2, 1, 4],
                    [3, 4, 3],
                    [4, 3, 4],
                ],
                device,
            ),
            token_mask: None,
            position_ids: None,
        },
        Tensor::<B, 1, Int>::from_ints([0, 0, 0, 0, 1, 1, 1, 1], device),
    )
}

pub fn gradient_norm<B: Backend>(gradients: &GradientsParams, id: ParamId) -> f32 {
    let values = gradients
        .get::<B, 2>(id)
        .expect("adapter must receive gradients")
        .to_data()
        .to_vec::<f32>()
        .expect("F32 gradient");
    assert!(values.iter().all(|value| value.is_finite()));
    let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
    assert!(norm > 0.0, "the gradient route must remain connected");
    norm
}

/// Hash every parameter's exact tensor bytes without duplicating a large backbone record.
pub fn fingerprint<B: Backend>(model: &impl Module<B>, frozen: bool) -> String {
    struct Visitor {
        hash: Sha256,
        frozen: bool,
    }
    impl<B: Backend> ModuleVisitor<B> for Visitor {
        fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
            let value = param.val();
            if self.frozen {
                assert!(
                    !value.is_require_grad(),
                    "backbone parameter must be frozen"
                );
            }
            let data = value.to_data();
            self.hash
                .update(format!("{:?}:{:?}", data.shape, data.dtype).as_bytes());
            self.hash.update(&*data.bytes);
        }
    }
    let mut visitor = Visitor {
        hash: Sha256::new(),
        frozen,
    };
    model.visit(&mut visitor);
    format!("{:x}", visitor.hash.finalize())
}

fn tiny_config() -> Qwen3_5TextConfig {
    Qwen3_5TextConfig {
        vocab_size: 16,
        hidden_size: 8,
        intermediate_size: 12,
        num_hidden_layers: 2,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: 4,
        max_position_embeddings: 32,
        layer_types: vec![
            Qwen3_5LayerType::LinearAttention,
            Qwen3_5LayerType::FullAttention,
        ],
        linear_key_head_dim: 4,
        linear_value_head_dim: 4,
        linear_num_key_heads: 1,
        linear_num_value_heads: 2,
        linear_conv_kernel_dim: 3,
        partial_rotary_factor: Some(1.0),
        rope_theta: Some(10_000.0),
        mtp_num_hidden_layers: 0,
        ..Default::default()
    }
}
