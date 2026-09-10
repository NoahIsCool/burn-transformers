//! Train an inserted adapter and replacement head through a frozen Qwen block.
//!
//! Random CPU smoke: `cargo run --release --locked --offline --features autodiff --example train_qwen_adapter`
//! Local weights: append `-- --model-dir /path/to/Qwen3.5-0.8B --steps 40`.
//! This learns eight synthetic token-label pairs; it is not sentiment evaluation.
//!
//! The two-block F32 backbone keeps causal token order and accepts unpadded
//! `[batch, 3]` IDs. Its parameters are frozen on the autodiff backend, preserving
//! derivatives through the later block to an inserted residual bottleneck:
//! `h + 0.1 up(tanh(down(h)))`. A head reads the last token into `[batch, 2]` logits.
//! The objective is categorical cross entropy plus `0.01 * mean(z^2)`, where `z`
//! is a selected, gradient-connected MLP activation after that later frozen block.
//! No dropout or decode cache is used. Native Burn records store trainable parts
//! and Adam moments separately from the frozen backbone; restored parameter IDs
//! reconnect optimizer state for an exact next-step comparison.

#[cfg(not(all(feature = "flex", feature = "autodiff")))]
/// Report missing Flex/autodiff features before attempting adapter training.
fn main() {
    eprintln!("train_qwen_adapter requires the flex and autodiff features");
    std::process::exit(1);
}

#[cfg(all(feature = "flex", feature = "autodiff"))]
#[path = "training/qwen_adapter_support.rs"]
mod support;

#[cfg(all(feature = "flex", feature = "autodiff"))]
/// Run the frozen-backbone adapter experiment and native Adam resume checks.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    experiment::run()
}

#[cfg(all(feature = "flex", feature = "autodiff"))]
mod experiment {
    //! Registered adapter/head training through a causally executed frozen Qwen prefix.

    use burn::{
        backend::{Autodiff, Flex},
        module::{AutodiffModule, Module},
        nn::{Linear, LinearConfig},
        optim::{AdamConfig, GradientsParams, Optimizer},
        record::{BinBytesRecorder, FullPrecisionSettings, Recorder},
        tensor::{Tensor, backend::Backend},
    };
    use transformers::{
        CausalLmInput, Result,
        intervention::{Hooks, Layer},
        models::qwen3_5::Qwen3_5TextModel,
        training::cross_entropy,
    };

    use super::support::{self, Options};

    type Cpu = Flex<f32, i32>;
    type Ad = Autodiff<Cpu>;
    const LEARNING_RATE: f64 = 0.01;
    const AUXILIARY_WEIGHT: f32 = 0.01;

    #[derive(Module, Debug)]
    struct Adapter<B: Backend> {
        down: Linear<B>,
        up: Linear<B>,
    }

    impl<B: Backend> Adapter<B> {
        /// Map `[batch, token, hidden]` through `up(tanh(down(h)))`, restoring the original hidden width.
        fn forward(&self, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
            self.up.forward(self.down.forward(hidden).tanh())
        }
    }

    // Save these small records separately from the original frozen checkpoint.
    #[derive(Module, Debug)]
    struct TrainableParts<B: Backend> {
        adapter: Adapter<B>,
        head: Linear<B>,
    }

    impl<B: Backend> TrainableParts<B> {
        /// Register a bounded bottleneck adapter and binary head for the supplied backbone width.
        fn init(width: usize, device: &B::Device) -> Self {
            let bottleneck = (width / 2).clamp(1, 16);
            Self {
                adapter: Adapter {
                    down: LinearConfig::new(width, bottleneck).init(device),
                    up: LinearConfig::new(bottleneck, width).init(device),
                },
                head: LinearConfig::new(width, 2).init(device),
            }
        }
    }

    #[derive(Module, Debug)]
    struct Experiment<B: Backend> {
        backbone: Qwen3_5TextModel<B>,
        trainable: TrainableParts<B>,
    }

    impl<B: Backend> Experiment<B> {
        /// Insert a residual adapter after block zero and return last-token logits plus selected downstream MLP features.
        /// The selected capture remains gradient-connected, allowing its auxiliary loss to reach the adapter through a frozen block.
        fn forward(&self, input: &CausalLmInput<B>) -> Result<(Tensor<B, 2>, Tensor<B, 3>)> {
            let mut hooks = Hooks::new();
            hooks.edit(Layer::decoder(0).residual_post(), |hidden| {
                hidden.clone() + self.trainable.adapter.forward(hidden) * 0.1
            });
            // The auxiliary loss uses selected channels AFTER the later frozen
            // block's attention and MLP input projection. Ordinary capture()
            // detaches; capture_with_grad() explicitly keeps this loss connected.
            let channels = self.backbone.config.intermediate_size.min(8);
            let selected = hooks.capture_with_grad(
                Layer::decoder(1)
                    .mlp_activation()
                    .token(2)
                    .channels(0..channels),
            );
            let state = self.backbone.embed(input)?;
            let state = self
                .backbone
                .run_layers_with_hooks(state, 0..2, &mut hooks)?;
            let state = self.backbone.normalize(state)?;
            let logits = self
                .trainable
                .head
                .forward(state.hidden.narrow(1, 2, 1))
                .squeeze_dim(1);
            Ok((logits, hooks.get(&selected)?))
        }
    }

    /// Train only the adapter/head and verify both task and auxiliary gradient routes through the frozen backbone.
    /// Save and restore trainable parameters with Adam state, compare the next update exactly, and report the synthetic task metrics.
    pub fn run() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let options = Options::parse()?;
        let device = Default::default();
        Ad::seed(&device, 42);
        let (backbone, source) = support::backbone::<Ad>(&options, &device)?;

        // Freeze parameters, keeping the autodiff backend. Later frozen blocks
        // must still differentiate with respect to the adapter's output.
        let backbone = backbone.no_grad();
        let width = backbone.config.hidden_size;
        let frozen_before = support::fingerprint(&backbone, true);
        let mut model = Experiment {
            backbone: backbone.clone(),
            trainable: TrainableParts::init(width, &device),
        };
        let adapter_before = support::fingerprint(&model.trainable.adapter, false);
        let head_before = support::fingerprint(&model.trainable.head, false);
        let (input, labels) = support::batch::<Ad>(&device);
        let mut optimizer = AdamConfig::new().init::<Ad, Experiment<Ad>>();

        // Check each route separately: the task loss reaches the adapter through
        // block 1, and the selected auxiliary tensor provides its own gradient.
        let (logits, _) = model.forward(&input)?;
        let task = cross_entropy(logits, labels.clone())?;
        let initial_loss = task.clone().into_scalar();
        let task_gradients = GradientsParams::from_grads(task.backward(), &model);
        let task_adapter_gradient =
            support::gradient_norm::<Cpu>(&task_gradients, model.trainable.adapter.down.weight.id);
        assert_eq!(
            task_gradients.len(),
            6,
            "only adapter and head parameters receive gradients"
        );
        let (_, selected) = model.forward(&input)?;
        let auxiliary = selected.square().mean();
        let initial_auxiliary_loss = auxiliary.clone().into_scalar();
        let auxiliary_gradients = GradientsParams::from_grads(auxiliary.backward(), &model);
        let auxiliary_adapter_gradient = support::gradient_norm::<Cpu>(
            &auxiliary_gradients,
            model.trainable.adapter.down.weight.id,
        );

        let step = |model: &Experiment<Ad>| -> Result<GradientsParams> {
            let (logits, selected) = model.forward(&input)?;
            let loss = cross_entropy(logits, labels.clone())?
                + selected.square().mean() * AUXILIARY_WEIGHT;
            Ok(GradientsParams::from_grads(loss.backward(), model))
        };
        for _ in 0..options.steps / 2 {
            let gradients = step(&model)?;
            model = optimizer.step(LEARNING_RATE, model, gradients);
        }

        // Save ordinary Burn records, including Adam's moments and step counts.
        // Parameter IDs restored by the model record reconnect optimizer state.
        let recorder = BinBytesRecorder::<FullPrecisionSettings>::default();
        let parts_bytes = recorder.record(model.trainable.clone().into_record(), ())?;
        let optimizer_bytes = recorder.record(optimizer.to_record(), ())?;
        let parts_path = options.output.join("trainable-midpoint.bin");
        let optimizer_path = options.output.join("adam-midpoint.bin");
        std::fs::write(&parts_path, parts_bytes)?;
        std::fs::write(&optimizer_path, optimizer_bytes)?;
        let mut resumed = Experiment {
            backbone,
            trainable: TrainableParts::init(width, &device)
                .load_record(recorder.load(std::fs::read(parts_path)?, &device)?),
        };
        let mut resumed_optimizer = AdamConfig::new()
            .init::<Ad, Experiment<Ad>>()
            .load_record(recorder.load(std::fs::read(optimizer_path)?, &device)?);
        assert_eq!(
            support::fingerprint(&model.trainable, false),
            support::fingerprint(&resumed.trainable, false)
        );

        // An uninterrupted step and a restored step must produce identical weights.
        let gradients = step(&model)?;
        model = optimizer.step(LEARNING_RATE, model, gradients);
        let gradients = step(&resumed)?;
        resumed = resumed_optimizer.step(LEARNING_RATE, resumed, gradients);
        assert_eq!(
            support::fingerprint(&model.trainable, false),
            support::fingerprint(&resumed.trainable, false)
        );
        drop(model);
        for _ in options.steps / 2 + 1..options.steps {
            let gradients = step(&resumed)?;
            resumed = resumed_optimizer.step(LEARNING_RATE, resumed, gradients);
        }

        // valid() removes autodiff only for evaluation, after training is done.
        let inference = resumed.valid();
        let (evaluation_input, evaluation_labels) = support::batch::<Cpu>(&device);
        let (logits, selected) = inference.forward(&evaluation_input)?;
        let final_loss = cross_entropy(logits.clone(), evaluation_labels.clone())?.into_scalar();
        let accuracy = logits
            .argmax(1)
            .squeeze_dim::<1>(1)
            .equal(evaluation_labels)
            .float()
            .mean()
            .into_scalar();
        let final_auxiliary_loss = selected.clone().square().mean().into_scalar();
        let frozen_after = support::fingerprint(&resumed.backbone, true);
        assert_eq!(
            frozen_before, frozen_after,
            "every frozen parameter byte must remain identical"
        );
        assert_ne!(
            adapter_before,
            support::fingerprint(&resumed.trainable.adapter, false)
        );
        assert_ne!(
            head_before,
            support::fingerprint(&resumed.trainable.head, false)
        );
        assert!(
            final_loss < initial_loss,
            "the synthetic task loss must decrease"
        );
        std::fs::write(
            options.output.join("trainable-final.bin"),
            recorder.record(resumed.trainable.clone().into_record(), ())?,
        )?;
        std::fs::write(
            options.output.join("adam-final.bin"),
            recorder.record(resumed_optimizer.to_record(), ())?,
        )?;
        let report = serde_json::json!({
            "scope": "Synthetic training and serialization smoke; no sentiment or pretrained-model quality claim",
            "source": source, "steps": options.steps, "learning_rate": LEARNING_RATE,
            "auxiliary_weight": AUXILIARY_WEIGHT, "initial_task_loss": initial_loss,
            "final_task_loss": final_loss, "training_set_accuracy": accuracy,
            "initial_auxiliary_loss": initial_auxiliary_loss, "final_auxiliary_loss": final_auxiliary_loss,
            "initial_total_loss": initial_loss + AUXILIARY_WEIGHT * initial_auxiliary_loss,
            "final_total_loss": final_loss + AUXILIARY_WEIGHT * final_auxiliary_loss,
            "task_gradient_through_frozen_block": task_adapter_gradient,
            "selected_auxiliary_gradient_through_frozen_block": auxiliary_adapter_gradient,
            "capture_shape": selected.dims(), "capture_dtype": format!("{:?}", selected.dtype()),
            "frozen_parameters_sha256_before": frozen_before, "frozen_parameters_sha256_after": frozen_after,
            "adapter_changed": true, "head_changed": true, "resume_next_step_exact": true,
            "saved_step": options.steps / 2, "seed": 42,
            "data_order": "same fixed batch of eight examples on every optimizer step", "held_out_evaluation": false,
        });
        std::fs::write(
            options.output.join("report.json"),
            serde_json::to_vec_pretty(&report)?,
        )?;
        println!(
            "Task loss: {initial_loss:.6} -> {final_loss:.6}; synthetic training accuracy: {:.1}%",
            accuracy * 100.0
        );
        println!(
            "Frozen backbone unchanged; adapter/head updated; restored Adam step matched exactly."
        );
        println!("Records and report: {}", options.output.display());
        Ok(())
    }
}
