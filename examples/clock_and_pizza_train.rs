//! Train the Clock/Pizza architecture from random weights, then inspect its learned activations.
//! Run: cargo run --release --features autodiff --example clock_and_pizza_train -- --smoke
//!
//! The unmasked attention rule is `alpha * softmax(Q K^T / sqrt(32)) + (1-alpha) J`,
//! where every entry of J is one. Inputs are `[batch, 2]`; final-token logits are
//! `[batch, 59]`. Training uses F32 cross entropy and full-batch AdamW with ten
//! warmup updates; the first update has zero learning rate. Held-out accuracy and
//! activation geometry are separate outcomes, as described in the study manifest.
#[allow(dead_code)]
#[path = "mechanistic/clock_and_pizza/model.rs"]
mod model;
#[path = "mechanistic/clock_and_pizza_training/study.rs"]
mod study;

/// Parse the run settings and choose a supported native Burn autodiff backend.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use burn::{
        backend::{Autodiff, Flex},
        module::AutodiffModule,
        optim::{AdamWConfig, GradientsParams, Optimizer},
        record::{BinFileRecorder, FullPrecisionSettings, Recorder},
        tensor::backend::AutodiffBackend,
    };
    use transformers::intervention::Hooks;

    /// Optimize the registered transformer with the fixed AdamW schedule.
    /// The same hooks remain in the training graph; evaluation uses the inner
    /// backend, and the final artifacts include detached activation diagnostics.
    fn train<B: AutodiffBackend>(
        options: study::Options,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let device = B::Device::default();
        B::seed(&device, options.seed);
        let mut model = model::Transformer::<B>::init_random(options.alpha, &device)?;
        let mut optimizer = AdamWConfig::new()
            .with_beta_2(0.98)
            .with_epsilon(1e-8)
            .with_weight_decay(2.0)
            .init();
        let data = study::Dataset::new(options.seed);
        let (inputs, labels) = data.training_batch::<B>(&device, options.smoke);
        let mut run = study::Run::begin(&options, &data, &model)?;
        run.measure(0, &model.valid(), &data)?;

        for step in 0..options.steps {
            // The experiment's structural attention rule and activation hooks are
            // part of an ordinary differentiable Burn module.
            let logits = model.forward_ids_with_hooks(inputs.clone(), &mut Hooks::new())?;
            let gradients =
                transformers::training::cross_entropy(logits, labels.clone())?.backward();
            let gradients = GradientsParams::from_grads(gradients, &model);
            let learning_rate = 0.001 * (step as f64 / 10.0).min(1.0);
            model = optimizer.step(learning_rate, model, gradients);
            if (step + 1) % options.evaluate_every == 0 || step + 1 == options.steps {
                run.measure(step + 1, &model.valid(), &data)?;
            }
        }
        BinFileRecorder::<FullPrecisionSettings>::new()
            .record(optimizer.to_record(), options.output.join("optimizer"))?;
        let symmetry = study::gradient_symmetry(&model)?;
        std::fs::write(
            options.output.join("gradient_symmetry.json"),
            serde_json::to_vec_pretty(&symmetry)?,
        )?;
        run.finish(&model.valid(), &data)?;
        Ok(())
    }

    let options = study::Options::parse()?;
    match options.backend.as_str() {
        "flex" => train::<Autodiff<Flex>>(options),
        #[cfg(all(feature = "metal", target_os = "macos"))]
        "metal" => train::<Autodiff<burn::backend::Metal>>(options),
        _ => Err("backend requires its Cargo feature: flex or metal (macOS)".into()),
    }
}
