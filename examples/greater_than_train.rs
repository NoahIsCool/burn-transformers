//! Learn a small replacement for a GPT-2 MLP using clean-source information.
//!
//! This is a new distillation extension, not the Greater-Than paper's training.
//! See mechanistic/greater_than_training/protocol.json before running the study.
//!
//! The student computes up(tanh(down(h_clean))) with widths 768 → 32 → 768,
//! replacing block 10's MLP output at token twelve in a corrupted causal forward.
//! It minimizes full-vocabulary `KL(oracle-patched || student-patched)` using
//! native Burn Adam. Frozen later components transmit activation gradients.
//! Clean input features explicitly supply year information removed by corruption;
//! this is conditioned distillation, not recovery from corrupted information alone.
//! All training uses F32; host F64 metrics compare the fixed development/held-out
//! split only after the predetermined endpoint, without validation-driven selection.

#[cfg(not(all(feature = "flex", feature = "autodiff")))]
/// Explain the required Cargo features and exit before accessing local artifacts.
fn main() {
    eprintln!("greater_than_train requires flex and autodiff");
    std::process::exit(1);
}

#[cfg(all(feature = "flex", feature = "autodiff"))]
#[path = "mechanistic/greater_than_training/support.rs"]
mod support;

#[cfg(all(feature = "flex", feature = "autodiff"))]
/// Run the fixed conditioned replacement experiment and propagate artifact errors.
fn main() -> support::Result<()> {
    experiment::run()
}

#[cfg(all(feature = "flex", feature = "autodiff"))]
mod experiment {
    //! Registered student ownership and native Adam execution for the fixed protocol.
    //!
    //! Freeze the backbone parameters while retaining gradients through its suffix.
    //! Clean-source caches are inputs, and teacher logits are detached KL targets.

    use super::support::{self, Cpu, Result};
    use burn::{
        backend::Autodiff,
        grad_clipping::GradientClippingConfig,
        module::{AutodiffModule, Module},
        nn::{Linear, LinearConfig},
        optim::{AdamConfig, GradientsParams, Optimizer},
        record::{BinBytesRecorder, FullPrecisionSettings, Recorder},
        tensor::{Tensor, backend::Backend},
    };
    use std::{fs, time::Instant};
    use transformers::{
        CausalLmInput, ForwardOptions,
        intervention::{Hooks, Layer},
        models::gpt2::Gpt2ForCausalLm,
        training::kl_divergence,
    };

    type Ad = Autodiff<Cpu>;
    const LEARNING_RATE: f64 = 0.003;
    const BATCH_SIZE: usize = 8;
    const SEED: u64 = 20260910;

    #[derive(Module, Debug)]
    struct Student<B: Backend> {
        down: Linear<B>,
        up: Linear<B>,
    }

    impl<B: Backend> Student<B> {
        /// Initialize the two registered projections of the 768 → 32 → 768 student.
        ///
        /// The caller seeds Burn before construction; both biases and weights learn.
        fn init(device: &B::Device) -> Self {
            Self {
                down: LinearConfig::new(768, 32).init(device),
                up: LinearConfig::new(32, 768).init(device),
            }
        }

        /// Map clean features `[batch, 1, 768]` through `up(tanh(down(h)))`.
        ///
        /// The 32-channel bottleneck limits capacity without changing the selected
        /// MLP output shape expected by the downstream transformer.
        fn forward(&self, clean_mlp_input: Tensor<B, 3>) -> Tensor<B, 3> {
            self.up.forward(self.down.forward(clean_mlp_input).tanh())
        }
    }

    #[derive(Module, Debug)]
    struct Experiment<B: Backend> {
        backbone: Gpt2ForCausalLm<B>,
        student: Student<B>,
    }

    impl<B: Backend> Experiment<B> {
        /// Replace MLP 10 at token 12 and return `[batch, 50257]` final logits.
        ///
        /// Clean features provide the missing year information to the student; the
        /// ordinary destination run remains corrupted. Frozen block 11 and readout
        /// remain differentiable with respect to the replacement activation.
        fn forward(
            &self,
            corrupted: CausalLmInput<B>,
            clean_features: Tensor<B, 3>,
        ) -> transformers::Result<Tensor<B, 2>> {
            let mut hooks = Hooks::new();
            // Cached CLEAN features supply the missing year information. The
            // student replaces this one output inside an ordinary CORRUPT run.
            // GPT-2 block 11 and the tied readout still differentiate its output.
            hooks.edit(Layer::decoder(10).mlp_output().token(12), |_| {
                self.student.forward(clean_features.clone())
            });
            Ok(self
                .backbone
                .forward_with_hooks(corrupted, &ForwardOptions::default(), &mut hooks)?
                .logits
                .squeeze_dim(1))
        }
    }

    /// Train for the protocol endpoint using native Adam and evaluate both frozen splits.
    ///
    /// Only four student parameter tensors receive gradients. Save and restore
    /// student parameters and optimizer state halfway through, require the next
    /// update to match exactly, and check frozen backbone bytes after training.
    pub fn run() -> Result<()> {
        let started = Instant::now();
        let options = support::Options::parse()?;
        let study = support::Study::load(&options)?;
        let backbone = study.load_model()?;
        let device = Default::default();
        let development = study.prepare(&backbone, "development", options.smoke)?;
        let frozen = backbone.train::<Ad>().no_grad();
        let frozen_before = support::fingerprint(&frozen, true);
        Ad::seed(&device, SEED);
        let mut model = Experiment {
            backbone: frozen.clone(),
            student: Student::init(&device),
        };
        let initial_student = model.student.clone();
        let initial_student_hash = support::fingerprint(&initial_student, false);
        let adam = AdamConfig::new().with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)));
        let mut optimizer = adam.init::<Ad, Experiment<Ad>>();
        let order = support::training_order(development.len());
        let steps = if options.smoke { 2 } else { 256 };
        let indices_for = |step: usize| {
            (0..BATCH_SIZE)
                .map(|j| order[(step * BATCH_SIZE + j) % order.len()])
                .collect::<Vec<_>>()
        };
        let training_loss = |model: &Experiment<Ad>, indices: &[usize]| -> Result<Tensor<Ad, 1>> {
            let input = study.input::<Ad>(&development.source_indices(indices), true);
            let features = Tensor::<Ad, 3>::from_inner(development.features(indices));
            let teacher = Tensor::<Ad, 2>::from_inner(development.teacher(indices));
            Ok(kl_divergence(
                model.forward(input, features)?,
                teacher.detach(),
            )?)
        };

        let probe = training_loss(&model, &indices_for(0))?;
        let initial_batch_kl = probe.clone().into_scalar();
        let gradients = GradientsParams::from_grads(probe.backward(), &model);
        assert_eq!(
            gradients.len(),
            4,
            "only the two student projections receive parameter gradients"
        );
        let gradient_norm = support::gradient_norm(&gradients, model.student.down.weight.id);
        let train_started = Instant::now();
        let mut loss_history = Vec::new();
        for step in 0..steps / 2 {
            let loss = training_loss(&model, &indices_for(step))?;
            loss_history.push(loss.clone().into_scalar());
            let gradients = GradientsParams::from_grads(loss.backward(), &model);
            model = optimizer.step(LEARNING_RATE, model, gradients);
            if (step + 1) % 32 == 0 {
                eprintln!("Learned replacement: {}/{} steps", step + 1, steps);
            }
        }

        // Restore ordinary Burn records and Adam's per-parameter state. This
        // includes a next-step comparison against uninterrupted optimization.
        let recorder = BinBytesRecorder::<FullPrecisionSettings>::default();
        let student_path = options.output.join("student-midpoint.bin");
        let adam_path = options.output.join("adam-midpoint.bin");
        fs::write(
            &student_path,
            recorder.record(model.student.clone().into_record(), ())?,
        )?;
        fs::write(&adam_path, recorder.record(optimizer.to_record(), ())?)?;
        let mut resumed = Experiment {
            backbone: frozen,
            student: Student::init(&device)
                .load_record(recorder.load(fs::read(student_path)?, &device)?),
        };
        let mut resumed_optimizer = adam
            .init::<Ad, Experiment<Ad>>()
            .load_record(recorder.load(fs::read(adam_path)?, &device)?);
        let indices = indices_for(steps / 2);
        let loss = training_loss(&model, &indices)?;
        loss_history.push(loss.clone().into_scalar());
        let gradients = GradientsParams::from_grads(loss.backward(), &model);
        model = optimizer.step(LEARNING_RATE, model, gradients);
        let loss = training_loss(&resumed, &indices)?;
        let gradients = GradientsParams::from_grads(loss.backward(), &resumed);
        resumed = resumed_optimizer.step(LEARNING_RATE, resumed, gradients);
        assert_eq!(
            support::fingerprint(&model.student, false),
            support::fingerprint(&resumed.student, false)
        );
        drop(model);
        for step in steps / 2 + 1..steps {
            let loss = training_loss(&resumed, &indices_for(step))?;
            loss_history.push(loss.clone().into_scalar());
            let gradients = GradientsParams::from_grads(loss.backward(), &resumed);
            resumed = resumed_optimizer.step(LEARNING_RATE, resumed, gradients);
            if (step + 1) % 32 == 0 {
                eprintln!("Learned replacement: {}/{} steps", step + 1, steps);
            }
        }
        let training_seconds = train_started.elapsed().as_secs_f64();
        let frozen_after = support::fingerprint(&resumed.backbone, true);
        assert_eq!(frozen_before, frozen_after);
        let final_student_hash = support::fingerprint(&resumed.student, false);
        assert_ne!(initial_student_hash, final_student_hash);
        fs::write(
            options.output.join("student-final.bin"),
            recorder.record(resumed.student.clone().into_record(), ())?,
        )?;
        fs::write(
            options.output.join("adam-final.bin"),
            recorder.record(resumed_optimizer.to_record(), ())?,
        )?;

        // Held-out clean prompts enter evaluation only after the fixed endpoint.
        // No early stopping, checkpoint selection, or validation-driven tuning.
        let trained = resumed.valid();
        let initial = Experiment {
            backbone: trained.backbone.clone(),
            student: initial_student.valid(),
        };
        let mut rows = Vec::new();
        for split in ["development", "held_out"] {
            if options.smoke && split == "held_out" {
                continue;
            }
            let cache = if split == "development" {
                development.clone()
            } else {
                study.prepare(&trained.backbone, split, false)?
            };
            for start in (0..cache.len()).step_by(BATCH_SIZE) {
                let indices = (start..(start + BATCH_SIZE).min(cache.len())).collect::<Vec<_>>();
                let input = study.input::<Cpu>(&cache.source_indices(&indices), true);
                let features = cache.features(&indices);
                let initial_logits = initial.forward(input.clone(), features.clone())?;
                let trained_logits = trained.forward(input, features)?;
                rows.extend(study.metrics(&cache, &indices, initial_logits, trained_logits)?);
            }
        }
        study.finish(&options, rows, serde_json::json!({
            "steps": steps, "initial_batch_kl": initial_batch_kl,
            "loss_history": loss_history, "training_seconds": training_seconds,
            "total_seconds": started.elapsed().as_secs_f64(), "student_gradient_through_frozen_block": gradient_norm,
            "frozen_sha256_before": frozen_before, "frozen_sha256_after": frozen_after,
            "initial_student_sha256": initial_student_hash, "final_student_sha256": final_student_hash,
            "resume_next_step_exact": true, "student_parameters": trained.student.num_params(),
        }))
    }
}
