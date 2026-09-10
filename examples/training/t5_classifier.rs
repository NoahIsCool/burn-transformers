//! Native Burn optimization of the heterogeneous T5 graph from the parent example.
//! The synthetic task labels positive/negative adjectives, not real sentiment.
//!
//! Each example contains four token slots and one binary adjective label. Boolean
//! masks mark nonzero IDs as valid; bidirectional attention and masked mean pooling
//! ignore padding. The fixed split withholds noun/adjective combinations while
//! retaining every vocabulary item in training. Full-batch F32 categorical cross
//! entropy trains the encoder, inserted adapter, and readout with native Burn Adam.
//! The parent recipe sets dropout to zero. The final parameter record is saved
//! from the training owner to preserve layout metadata; this helper checks model
//! reload equivalence and does not save Adam state for a resumed training run.

#[path = "support.rs"]
mod support;

use super::experiment::SentimentClassifier;
use burn::{
    backend::Autodiff,
    module::{AutodiffModule, Module},
    optim::{AdamConfig, GradientsParams, Optimizer},
    record::{BinFileRecorder, FullPrecisionSettings},
    tensor::{Bool, Int, Tensor, TensorData, backend::Backend},
};
use std::{path::Path, time::Instant};
use support::{Cpu, Report, Result};
use transformers::{CausalLmInput, TokenMask, training::cross_entropy};

type Ad = Autodiff<Cpu>;
const STEPS: usize = 100;
const LEARNING_RATE: f64 = 0.01;
const SEED: u64 = 73;

/// Train the synthetic adjective classifier with the fixed seed and update budget in local ignored storage.
pub fn run() -> Result<()> {
    train(Path::new("target/training/t5_classifier"), STEPS)
}

/// Optimize the staged T5 graph using full-batch Adam, then verify held-out learning and parameter-record reload.
/// The first update checks representative gradient routes; the final record retains every registered model parameter.
fn train(root: &Path, steps: usize) -> Result<()> {
    let started = Instant::now();
    std::fs::create_dir_all(root)?;
    let device = Default::default();
    Cpu::seed(&device, SEED);
    let mut model = SentimentClassifier::<Ad>::init(&device)?;
    let (train, held_out) = dataset();
    let (input, labels) = batch::<Cpu>(&held_out);
    let initial = support::evaluate(model.valid().logits(input)?, labels)?;
    let before = support::snapshot(model.training_weights())?;
    let mut adam = AdamConfig::new().init();

    // An ordinary Burn loop: differentiate through embed → stages → adapter → head.
    for step in 0..steps {
        let (input, labels) = batch::<Ad>(&train);
        let loss = cross_entropy(model.logits(input)?, labels)?;
        let gradients = GradientsParams::from_grads(loss.backward(), &model);
        if step == 0 {
            support::check_gradients(model.training_weights(), &gradients)?;
        }
        model = adam.step(LEARNING_RATE, model, gradients);
    }

    // valid() uses the same parameters on the non-autodiff backend for evaluation.
    let valid = model.valid();
    let (input, labels) = batch::<Cpu>(&train);
    let final_train_scores = support::evaluate(valid.logits(input)?, labels)?;
    let (input, labels) = batch::<Cpu>(&held_out);
    let logits = valid.logits(input)?;
    let final_scores = support::evaluate(logits.clone(), labels)?;
    let parameter_changes =
        support::changes(&before, support::snapshot(valid.training_weights())?)?;

    // Native Burn records contain all registered parameters, including the adapter.
    let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
    // Save the trained module: Burn 0.21 valid() drops parameter layout mappers.
    model.save_file(root.join("model"), &recorder)?;
    let reloaded = SentimentClassifier::<Cpu>::init(&device)?.load_file(
        root.join("model"),
        &recorder,
        &device,
    )?;
    let reload_logit_error =
        support::max_error(logits, reloaded.logits(batch::<Cpu>(&held_out).0)?);
    Report {
        task: "Synthetic T5 adjective sentiment",
        seed: SEED,
        learning_rate: LEARNING_RATE,
        optimizer: "Adam",
        weight_decay: 0.0,
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
    tokens: [i32; 4],
    label: i32,
}

/// Enumerate padded/unpadded adjective sentences and deterministically withhold noun/adjective combinations.
fn dataset() -> (Vec<Example>, Vec<Example>) {
    // 0=pad, 1=this, 2=film, 3=book, 4=story, 5=game, 6=is,
    // 7=good, 8=great, 9=bad, 10=awful, 11=really.
    // Hold out noun/adjective combinations; every word occurs during training.
    let (mut train, mut test) = (Vec::new(), Vec::new());
    for noun in 2..=5 {
        for adjective in 7..=10 {
            for tokens in [[1, noun, adjective, 0], [noun, 6, 11, adjective]] {
                let row = Example {
                    tokens,
                    label: i32::from(adjective <= 8),
                };
                if (noun + adjective) % 4 == 0 {
                    test.push(row);
                } else {
                    train.push(row);
                }
            }
        }
    }
    (train, test)
}

/// Pack four-token rows and binary labels, marking nonzero token IDs as valid in the Boolean padding mask.
fn batch<B: Backend>(rows: &[Example]) -> (CausalLmInput<B>, Tensor<B, 1, Int>) {
    let device = Default::default();
    let ids: Vec<_> = rows.iter().flat_map(|r| r.tokens).collect();
    let mask = ids.iter().map(|&id| id != 0).collect::<Vec<_>>();
    (
        CausalLmInput {
            token_ids: Tensor::from_data(TensorData::new(ids, [rows.len(), 4]), &device),
            token_mask: Some(TokenMask::new(Tensor::<B, 2, Bool>::from_data(
                TensorData::new(mask, [rows.len(), 4]),
                &device,
            ))),
            position_ids: None,
        },
        Tensor::from_data(
            TensorData::new(
                rows.iter().map(|r| r.label).collect::<Vec<_>>(),
                [rows.len()],
            ),
            &device,
        ),
    )
}

#[cfg(test)]
#[path = "t5_classifier_tests.rs"]
mod tests;
