//! Train new affine lenses on frozen Pythia activations using ordinary Burn SGD.
//!
//! Prepare the pinned backbone with `examples/mechanistic/tuned_lens/prepare.py`.
//! Run: cargo run --release --locked --features autodiff,tokenizers --example tuned_lens_train
//! The companion protocol explains the deliberately small, disjoint data split.
//!
//! Each correction learns h + Wh + b from zero parameters, minimizing
//! `KL(teacher || lens)` over the full 50,304-token vocabulary. Detached residuals
//! have axes `[tokens=32, hidden=768]`; the frozen final decoder still propagates
//! gradients into the correction. Native Nesterov SGD runs for a fixed endpoint
//! on twelve documents, with four disjoint documents reserved for evaluation.
//! F32 graph arithmetic and global gradient clipping are paired with F64 host
//! evaluation. This small protocol is an adaptation of the paper's training scale.

#[path = "mechanistic/tuned_lens_training/study.rs"]
mod study;

/// Fit each registered affine correction with native Burn SGD, then evaluate held-out data.
///
/// The frozen backbone supplies detached residuals and teacher logits. A midpoint
/// record round trip must reproduce the next weight and momentum update.
fn main() -> study::Result<()> {
    use burn::{module::Module, optim::Optimizer};
    use std::time::Instant;
    use study::{Lens, checkpoint, evaluate, fingerprint, gradients, optimizer};

    let started = Instant::now();
    let settings = study::Settings::read()?;
    let (backbone, records, provenance) = study::load(&settings)?;
    let backbone_before = fingerprint(&backbone);

    // The backbone runs once per document. Hooks select raw residual states;
    // detached captures are appropriate because this backbone will stay frozen.
    let cache = study::capture(&backbone, &records, &settings.stages)?;
    let training = &cache[..settings.train_records];
    let validation = &cache[settings.train_records..];
    let mut results = Vec::new();

    for (slot, &stage) in settings.stages.iter().enumerate() {
        // This owns registered Burn modules. Only the affine correction learns;
        // gradients still pass through the frozen final LayerNorm and readout.
        let mut lens = Lens::new(&backbone);
        let identity_error = study::identity_error(&lens, &backbone, &training[0], slot)?;
        let decoder_before = lens.frozen_fingerprint();
        let mut optim = optimizer();
        let initial = evaluate(&lens, validation, slot)?;
        let mut curve = Vec::new();
        let mut resume_error = 0.0;
        let mut first_gradient_norm = 0.0;
        let layer_started = Instant::now();

        for step in 0..settings.steps {
            let batch = &training[step % training.len()];
            let logits = lens.forward(batch.hidden[slot].clone());
            let loss = transformers::training::kl_divergence(logits, batch.teacher.clone())?;
            let loss_value = study::scalar(loss.clone())?;
            let (grads, norm) = gradients(loss.backward(), &lens)?;
            if step == 0 {
                first_gradient_norm = norm;
                study::export_gradient_oracle(
                    &settings, stage, &lens, batch, slot, &grads, loss_value,
                )?;
            }
            lens = optim.step(settings.learning_rate(step), lens, grads);
            if step == 0 {
                study::export_first_update(&settings, stage, &lens)?;
            }
            curve.push(serde_json::json!({"step": step + 1, "training_kl": loss_value, "gradient_norm_before_clipping": norm}));
            if (step + 1) % 20 == 0 {
                println!(
                    "stage {stage}, step {}: training KL {loss_value:.6}",
                    step + 1
                );
            }

            if step + 1 == settings.steps / 2 {
                // Save model and momentum separately with Burn recorders. Resume
                // from disk, then compare the next update to the untouched run.
                let (restored, resumed_optim) = checkpoint(&settings, stage, &lens, &optim)?;
                resume_error = study::compare_next_step(
                    lens.clone(),
                    optim.clone(),
                    restored.clone(),
                    resumed_optim.clone(),
                    &training[(step + 1) % training.len()],
                    slot,
                    settings.learning_rate(step + 1),
                )?;
                lens = restored;
                optim = resumed_optim;
            }
        }

        let final_metrics = evaluate(&lens, validation, slot)?;
        let frozen_unchanged = decoder_before == lens.frozen_fingerprint();
        study::save_final(&settings, stage, &lens, &optim)?;
        let item = serde_json::json!({
            "stage": stage, "initial": initial, "final": final_metrics,
            "first_gradient_norm": first_gradient_norm,
            "initial_identity_max_abs": identity_error,
            "resume_next_step_max_abs": resume_error,
            "frozen_decoder_unchanged": frozen_unchanged,
            "translator_parameters": lens.translator.num_params(),
            "seconds": layer_started.elapsed().as_secs_f64(), "curve": curve,
        });
        println!(
            "stage {stage}: held-out KL {:.6} -> {:.6}; next-token CE {:.6} -> {:.6}",
            initial.kl, final_metrics.kl, initial.ce, final_metrics.ce
        );
        results.push(item);
    }

    study::report(
        &settings,
        provenance,
        results,
        backbone_before == fingerprint(&backbone),
        started.elapsed().as_secs_f64(),
    )
}
