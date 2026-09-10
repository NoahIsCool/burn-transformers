//! Train MLA encoder/decoder modules and a borrowed latent adapter using native AdamW.
//!
//! Source IDs are `[batch, 2]`; decoder queries `[10, 11]` ask for warm and cool
//! counts. Both decoder slots may see each other because their inputs are known
//! queries, never target answers. All source/query tokens are valid. Three-class
//! logits `[batch, query, 3]` flatten to `[batch * 2, 3]` with matching labels for
//! mean categorical cross entropy. A registered adapter edits compressed cross
//! KV states before expansion, preserving gradients into encoder and MLA modules.
//! F32 native Burn AdamW uses weight decay 0.1; dropout and caches are absent.
//! The final model record checks parameter reload, not optimizer-state resumption.

#[path = "support.rs"]
mod support;

use super::experiment::EncoderDecoder;
use burn::{
    backend::Autodiff,
    module::{AutodiffModule, Module},
    optim::{AdamWConfig, GradientsParams, Optimizer},
    record::{BinFileRecorder, FullPrecisionSettings},
    tensor::{Int, Tensor, TensorData, backend::Backend},
};
use std::{path::Path, time::Instant};
use support::{Cpu, Report, Result};
use transformers::training::cross_entropy;

type Ad = Autodiff<Cpu>;
const STEPS: usize = 600;
const LEARNING_RATE: f64 = 0.003;
const SEED: u64 = 101;

/// Train the bidirectional MLA counting task with its fixed AdamW budget in ignored local storage.
pub fn run() -> Result<()> {
    train(Path::new("target/training/mla_encoder_decoder"), STEPS)
}

/// Optimize both query answers through encoder, decoder, cross-MLA, and the borrowed latent adapter.
/// Check representative gradient/update paths and compare held-out logits after native model-record reload.
fn train(root: &Path, steps: usize) -> Result<()> {
    let started = Instant::now();
    std::fs::create_dir_all(root)?;
    let device = Default::default();
    Cpu::seed(&device, SEED);
    let mut model = EncoderDecoder::<Ad>::init(&device)?;
    let (train, held_out) = dataset();
    let (source, queries, labels) = batch::<Cpu>(&held_out);
    let initial = support::evaluate(
        model
            .valid()
            .training_logits(source, queries)?
            .reshape([held_out.len() * 2, 3]),
        labels,
    )?;
    let before = support::snapshot(model.training_weights())?;
    let mut adam = AdamWConfig::new().with_weight_decay(0.1).init();

    // Inputs are known query tokens, not shifted target labels. Both decoder
    // positions use bidirectional attention; the two labels depend on the source.
    for step in 0..steps {
        let (source, queries, labels) = batch::<Ad>(&train);
        // training_logits uses a temporary edit that borrows the registered
        // latent_adapter. Gradients cross that callback into encoder and MLA.
        let logits = model
            .training_logits(source, queries)?
            .reshape([train.len() * 2, 3]);
        let loss = cross_entropy(logits, labels)?;
        let gradients = GradientsParams::from_grads(loss.backward(), &model);
        if step == 0 {
            support::check_gradients(model.training_weights(), &gradients)?;
        }
        model = adam.step(LEARNING_RATE, model, gradients);
    }

    let valid = model.valid();
    let (source, queries, labels) = batch::<Cpu>(&train);
    let final_train_scores = support::evaluate(
        valid
            .training_logits(source, queries)?
            .reshape([train.len() * 2, 3]),
        labels,
    )?;
    let (source, queries, labels) = batch::<Cpu>(&held_out);
    let logits = valid.training_logits(source, queries)?;
    let final_scores = support::evaluate(logits.clone().reshape([held_out.len() * 2, 3]), labels)?;
    let parameter_changes =
        support::changes(&before, support::snapshot(valid.training_weights())?)?;
    let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
    // Save the trained module: Burn 0.21 valid() drops parameter layout mappers.
    model.save_file(root.join("model"), &recorder)?;
    let reloaded =
        EncoderDecoder::<Cpu>::init(&device)?.load_file(root.join("model"), &recorder, &device)?;
    let (source, queries, _) = batch::<Cpu>(&held_out);
    let reload_logit_error = support::max_error(logits, reloaded.training_logits(source, queries)?);
    Report {
        task: "Bidirectional MLA category counting",
        seed: SEED,
        learning_rate: LEARNING_RATE,
        optimizer: "AdamW",
        weight_decay: 0.1,
        steps,
        train_examples: train.len(),
        held_out_examples: held_out.len(),
        initial,
        final_scores,
        final_train_scores,
        parameter_changes,
        reload_logit_error,
        seconds: started.elapsed().as_secs_f64(),
    }
    .save(root)
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Example {
    source: [i32; 2],
    labels: [i32; 2],
}

/// Enumerate all ordered source pairs with warm/cool counts, withholding pairs by a fixed arithmetic rule.
fn dataset() -> (Vec<Example>, Vec<Example>) {
    // Source symbols 1..4 are cool colors, 5..8 are warm colors. Query 10 asks
    // for the warm count, query 11 for the cool count; each answer is 0, 1, or 2.
    // Hold out source pairs. No target answer is included in the decoder input.
    let (mut train, mut test) = (Vec::new(), Vec::new());
    for a in 1..=8 {
        for b in 1..=8 {
            let warm = i32::from(a >= 5) + i32::from(b >= 5);
            let row = Example {
                source: [a, b],
                labels: [warm, 2 - warm],
            };
            if (a + b) % 5 == 0 {
                test.push(row);
            } else {
                train.push(row);
            }
        }
    }
    (train, test)
}

/// Pack source pairs, repeated known query IDs, and example-major flattened count labels.
/// The label order matches reshaping `[batch, 2, 3]` logits into `[batch * 2, 3]`.
fn batch<B: Backend>(
    rows: &[Example],
) -> (Tensor<B, 2, Int>, Tensor<B, 2, Int>, Tensor<B, 1, Int>) {
    let device = Default::default();
    (
        Tensor::from_data(
            TensorData::new(
                rows.iter().flat_map(|r| r.source).collect::<Vec<_>>(),
                [rows.len(), 2],
            ),
            &device,
        ),
        Tensor::from_data(
            TensorData::new([10, 11].repeat(rows.len()), [rows.len(), 2]),
            &device,
        ),
        Tensor::from_data(
            TensorData::new(
                rows.iter().flat_map(|r| r.labels).collect::<Vec<_>>(),
                [rows.len() * 2],
            ),
            &device,
        ),
    )
}

#[cfg(test)]
#[path = "mla_encoder_decoder_tests.rs"]
mod tests;
