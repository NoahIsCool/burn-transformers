//! Train the modular-addition transformer from scratch with Burn's AdamW.
//! Run `--features autodiff --example grokking_train -- --steps 10` for a CPU smoke.
//! See mechanistic/grokking_training/README.md for the full grokking experiment.

#[cfg(all(feature = "std", feature = "flex", feature = "autodiff"))]
#[path = "mechanistic/grokking_model.rs"]
mod model;
#[cfg(all(feature = "std", feature = "flex", feature = "autodiff"))]
#[path = "mechanistic/grokking_training/study.rs"]
mod study;

#[cfg(all(feature = "std", feature = "flex", feature = "autodiff"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = study::Settings::from_args()?;
    match config.backend.as_str() {
        "flex" => study::train::<burn::backend::Autodiff<burn::backend::Flex>>(config),
        #[cfg(feature = "metal")]
        "metal" => study::train::<burn::backend::Autodiff<burn::backend::Metal>>(config),
        name => Err(format!("unsupported backend {name}; Metal needs --features metal").into()),
    }
}

/// This is the entire training step. Burn owns autodiff, optimizer state, and
/// parameter traversal; the transformer is an ordinary registered Burn module.
#[cfg(all(feature = "std", feature = "flex", feature = "autodiff"))]
fn training_step<B, O>(
    model: model::GrokkingTransformer<B>,
    optimizer: &mut O,
    inputs: burn::tensor::Tensor<B, 2, burn::tensor::Int>,
    targets: burn::tensor::Tensor<B, 1, burn::tensor::Int>,
    learning_rate: f64,
) -> transformers::Result<model::GrokkingTransformer<B>>
where
    B: burn::tensor::backend::AutodiffBackend,
    O: burn::optim::Optimizer<model::GrokkingTransformer<B>, B>,
{
    use burn::{nn::loss::CrossEntropyLossConfig, optim::GradientsParams};
    let loss_function = CrossEntropyLossConfig::new().init(&inputs.device());
    let logits = model.forward(inputs)?;
    let loss = loss_function.forward(logits, targets);
    let gradients = GradientsParams::from_grads(loss.backward(), &model);
    Ok(optimizer.step(learning_rate, model, gradients))
}

#[cfg(not(all(feature = "std", feature = "flex", feature = "autodiff")))]
fn main() {
    eprintln!("grokking_train requires --features std,flex,autodiff");
    std::process::exit(1);
}
