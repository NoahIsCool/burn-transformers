//! Train the open mixer and both registered width adapters with native Burn Adam.

#[path = "support.rs"]
mod support;

use super::experiment::Classifier;
use burn::{
    backend::Autodiff,
    module::{AutodiffModule, Module},
    optim::{AdamConfig, GradientsParams, Optimizer},
    record::{BinFileRecorder, FullPrecisionSettings},
    tensor::{Int, Tensor, TensorData, backend::Backend},
};
use std::{path::Path, time::Instant};
use support::{Cpu, Report, Result};
use transformers::{
    intervention::{Hooks, Layer},
    training::cross_entropy,
};

type Ad = Autodiff<Cpu>;
const STEPS: usize = 120;
const LEARNING_RATE: f64 = 0.01;
const SEED: u64 = 89;

pub fn run() -> Result<()> {
    train(Path::new("target/training/custom_composition"), STEPS)
}

fn train(root: &Path, steps: usize) -> Result<()> {
    let started = Instant::now();
    std::fs::create_dir_all(root)?;
    let device = Default::default();
    Cpu::seed(&device, SEED);
    let mut model = Classifier::<Ad>::init(&device)?;
    let (train, held_out) = dataset();
    let (tokens, labels) = batch::<Cpu>(&held_out);
    let initial = support::evaluate(model.valid().forward(tokens, &mut Hooks::new())?, labels)?;
    let before = support::snapshot(model.training_weights())?;
    let mut adam = AdamConfig::new().init();

    for step in 0..steps {
        let (tokens, labels) = batch::<Ad>(&train);
        let mut hooks = Hooks::new();
        // The auxiliary loss differentiates through an explicit capture. Default
        // detached captures would not contribute this regularization gradient.
        let features = hooks.capture_with_grad(Layer::encoder(1).mlp_activation());
        let logits = model.forward(tokens, &mut hooks)?;
        let loss = cross_entropy(logits, labels)? + hooks.get(&features)?.square().mean() * 1e-4;
        let gradients = GradientsParams::from_grads(loss.backward(), &model);
        if step == 0 {
            support::check_gradients(model.training_weights(), &gradients)?;
        }
        model = adam.step(LEARNING_RATE, model, gradients);
    }

    let valid = model.valid();
    let (tokens, labels) = batch::<Cpu>(&train);
    let final_train_scores = support::evaluate(valid.forward(tokens, &mut Hooks::new())?, labels)?;
    let (tokens, labels) = batch::<Cpu>(&held_out);
    let logits = valid.forward(tokens, &mut Hooks::new())?;
    let final_scores = support::evaluate(logits.clone(), labels)?;
    let parameter_changes =
        support::changes(&before, support::snapshot(valid.training_weights())?)?;
    let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
    // Save the trained module: Burn 0.21 valid() drops parameter layout mappers.
    model.save_file(root.join("model"), &recorder)?;
    let reloaded =
        Classifier::<Cpu>::init(&device)?.load_file(root.join("model"), &recorder, &device)?;
    let reload_logit_error = support::max_error(
        logits,
        reloaded.forward(batch::<Cpu>(&held_out).0, &mut Hooks::new())?,
    );
    Report {
        task: "Custom mixer color majority",
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
    tokens: [i32; 3],
    label: i32,
}

fn dataset() -> (Vec<Example>, Vec<Example>) {
    // Tokens 1=blue, 2=green, 3=red, 4=orange; classify a warm-color majority.
    // Disjoint sequences may share multisets: this bag-of-tokens task intentionally
    // tests a permutation-invariant mixer, not general language understanding.
    let (mut train, mut test) = (Vec::new(), Vec::new());
    for a in 1..=4 {
        for b in 1..=4 {
            for c in 1..=4 {
                let tokens = [a, b, c];
                let label = i32::from(tokens.iter().filter(|&&x| x >= 3).count() >= 2);
                let row = Example { tokens, label };
                if (a + 2 * b + 3 * c) % 5 == 0 {
                    test.push(row);
                } else {
                    train.push(row);
                }
            }
        }
    }
    (train, test)
}

fn batch<B: Backend>(rows: &[Example]) -> (Tensor<B, 2, Int>, Tensor<B, 1, Int>) {
    let device = Default::default();
    (
        Tensor::from_data(
            TensorData::new(
                rows.iter().flat_map(|r| r.tokens).collect::<Vec<_>>(),
                [rows.len(), 3],
            ),
            &device,
        ),
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
mod tests {
    use super::*;
    #[test]
    fn custom_mixer_and_width_adapters_learn_and_survive_native_records() {
        let (train_rows, held) = dataset();
        assert_eq!(train_rows.len() + held.len(), 64);
        assert!(held.iter().all(|r| !train_rows.contains(r)));
        train(tempfile::tempdir().unwrap().path(), STEPS).unwrap();
    }
}
