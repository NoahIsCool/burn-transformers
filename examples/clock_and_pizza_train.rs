//! Train the Clock/Pizza architecture from random weights, then inspect its learned activations.
//! Run: cargo run --release --features autodiff --example clock_and_pizza_train -- --smoke
#[cfg(feature = "autodiff")]
#[allow(dead_code)]
#[path = "mechanistic/clock_and_pizza/model.rs"]
mod model;
#[cfg(feature = "autodiff")]
#[path = "mechanistic/clock_and_pizza_training/study.rs"]
mod study;

#[cfg(feature = "autodiff")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use burn::{
        backend::{Autodiff, Flex},
        module::AutodiffModule,
        optim::{AdamWConfig, GradientsParams, Optimizer},
        record::{BinFileRecorder, FullPrecisionSettings, Recorder},
        tensor::backend::AutodiffBackend,
    };
    use transformers::intervention::Hooks;

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

#[cfg(not(feature = "autodiff"))]
fn main() {
    eprintln!(
        "Enable autodiff: cargo run --release --features autodiff --example clock_and_pizza_train -- --smoke"
    );
}
