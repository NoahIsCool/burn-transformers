//! A Qwen prefix, an intermediate bottleneck, and a custom classifier head.
//!
//! CPU demo with random weights:
//! `cargo run --release --locked --example qwen_prefix -- --random`
//!
//! Reuse the first four blocks of a local pretrained checkpoint:
//! `cargo run --release --locked --example qwen_prefix -- MODEL_DIR 4`
//!
//! The adapter/classifier are newly initialized. This demonstrates composition
//! and activation access, not trained sentiment accuracy. See `train_qwen_adapter`
//! for the native Burn training and optimizer-resume workflow.
//!
//! Token IDs have axes `[batch, token]`; staged hidden states retain
//! `[batch, token, hidden]` with the Qwen layer's causal semantics. Final RMS
//! normalization precedes an explicit width-changing bottleneck and token mean;
//! the readout retains a singleton token axis, `[batch, 1, 2]`. Inputs here have
//! no padding. Pretrained tensors are explicitly cast to F32 on Flex, and loading
//! a prefix skips later blocks and the original vocabulary head. All components
//! belong to one Burn module; this stateless demo uses neither dropout nor an
//! external decode cache.

#[cfg(not(feature = "flex"))]
/// Report the missing Flex feature required by the prefix demonstration.
fn main() {
    eprintln!("qwen_prefix requires the flex feature");
    std::process::exit(1);
}

#[cfg(feature = "flex")]
/// Run local prefix initialization/loading and the custom readout demonstration.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    experiment::run()
}

#[cfg(feature = "flex")]
mod experiment {
    //! Selective Qwen prefix loading and stateless F32 hidden-state readout.

    use std::env;

    use burn::{
        backend::Flex,
        module::Module,
        nn::{Linear, LinearConfig},
        tensor::{Int, Tensor, backend::Backend},
    };
    use transformers::{
        CausalLmInput,
        checkpoint::{DTypePolicy, LoadOptions, TensorDType},
        intervention::{Hooks, Layer},
        models::qwen3_5::{
            Qwen3_5BackboneRecipe, Qwen3_5LayerType, Qwen3_5RecipeLoadOptions, Qwen3_5TextConfig,
            Qwen3_5TextModel,
        },
    };

    /// CPU backend used to initialize or load the selected Qwen blocks in F32.
    type B = Flex<f32, i32>;

    // Register every trainable component in one ordinary Burn module. The
    // backbone itself contains only the selected blocks, embedding, and norm.
    /// Registers a selected Qwen prefix and an untrained classification head.
    #[derive(Module, Debug)]
    struct PrefixClassifier<B: Backend> {
        /// Embedding, selected decoder blocks, and final normalization of the Qwen prefix.
        backbone: Qwen3_5TextModel<B>,
        /// Projection reducing each normalized residual to the head's bottleneck width.
        bottleneck: Linear<B>,
        /// Binary task readout from the bottleneck features.
        classifier: Linear<B>,
    }

    /// Construct a selected Qwen prefix, capture its final MLP token, and attach an untrained bottleneck/head.
    /// Local checkpoint tensors are cast explicitly to F32; no later blocks or vocabulary head are needed.
    pub fn run() -> Result<(), Box<dyn std::error::Error>> {
        let device = Default::default();
        B::seed(&device, 42);
        let arguments = env::args().skip(1).collect::<Vec<_>>();
        let random = arguments.first().is_none_or(|value| value == "--random");
        let count = arguments
            .get(1)
            .map(|value| value.parse::<usize>())
            .transpose()?
            .unwrap_or(2);
        let recipe = if random {
            Qwen3_5BackboneRecipe::from_config(tiny_config())?.prefix(count)?
        } else {
            Qwen3_5BackboneRecipe::from_local_config(&arguments[0])?.prefix(count)?
        };

        // Make `recipe` mutable to edit these layer fields before initialization. For
        // pretrained edits, choose Reject or Reinitialize explicitly when a
        // changed shape can no longer reuse its source tensor.
        // recipe.layers[0].intermediate_size = 24;

        let backbone = if random {
            println!("Initializing {count} random Qwen blocks; no checkpoint is read.");
            recipe.init_random::<B>(&device)?
        } else {
            let loaded = recipe.load_pretrained::<B>(
                &arguments[0],
                &Qwen3_5RecipeLoadOptions {
                    checkpoint: LoadOptions {
                        dtype_policy: DTypePolicy::Cast(TensorDType::F32),
                        ..Default::default()
                    },
                    ..Default::default()
                },
                &device,
            )?;
            println!(
                "Loaded {} reused and {} converted parameters; omitted {} source parameters; initialized {} new parameters.",
                loaded.report.reused.len(),
                loaded.report.transformed.len(),
                loaded.report.omitted.len(),
                loaded.report.newly_initialized.len()
            );
            println!(
                "Read {} checkpoint payload bytes; later blocks and the vocabulary head were not loaded.",
                loaded.checkpoint.stats.source_payload_bytes
            );
            loaded.model
        };
        let width = backbone.config.hidden_size;
        let bottleneck_width = (width / 2).max(1);
        let experiment = PrefixClassifier {
            backbone,
            bottleneck: LinearConfig::new(width, bottleneck_width).init(&device),
            classifier: LinearConfig::new(bottleneck_width, 2).init(&device),
        };

        // Supply token IDs from your task's tokenizer. Fixed valid IDs keep this
        // structural example independent of text processing and datasets.
        let input = CausalLmInput {
            token_ids: Tensor::<B, 2, Int>::from_ints([[1, 2, 3]], &device),
            token_mask: None,
            position_ids: None,
        };
        let state = experiment.backbone.embed(&input)?;
        let mut hooks = Hooks::new();
        let neurons = hooks.capture(Layer::decoder(count - 1).mlp_activation().token(2));
        let state = experiment
            .backbone
            .run_layers_with_hooks(state, 0..count, &mut hooks)?;
        println!(
            "Captured last-block neurons: {:?}",
            hooks.get(&neurons)?.dims()
        );

        // Normalization and width changes are explicit. The bottleneck changes
        // the latent width; the small classifier consumes that new representation.
        let state = experiment.backbone.normalize(state)?;
        let adapted = experiment.bottleneck.forward(state.hidden);
        let logits = experiment.classifier.forward(adapted.mean_dim(1));
        assert_eq!(logits.dims(), [1, 1, 2]);
        println!(
            "Untrained two-class logits: {:?}",
            logits.into_data().to_vec::<f32>()?
        );
        Ok(())
    }

    /// Define the small recurrent-plus-full-attention Qwen recipe used for the no-download demonstration.
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
}
