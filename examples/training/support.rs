//! Small measurement helpers; optimization stays in each example's native Burn loop.

use std::{collections::BTreeMap, path::Path};

use burn::{
    backend::Flex,
    module::Param,
    optim::GradientsParams,
    tensor::{Int, Tensor, backend::Backend},
};
use serde::Serialize;

pub type Cpu = Flex<f32, i32>;
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
pub type Weights<'a, B> = Vec<(&'static str, &'a Param<Tensor<B, 2>>)>;
pub type Snapshot = BTreeMap<String, Vec<f32>>;

#[derive(Clone, Copy, Debug, Serialize)]
pub struct Scores {
    pub loss: f64,
    pub accuracy: f64,
}

pub fn evaluate(logits: Tensor<Cpu, 2>, labels: Tensor<Cpu, 1, Int>) -> Result<Scores> {
    let loss =
        transformers::training::cross_entropy(logits.clone(), labels.clone())?.into_scalar() as f64;
    let predictions = logits
        .argmax(1)
        .squeeze_dim::<1>(1)
        .into_data()
        .to_vec::<i32>()?;
    let expected = labels.into_data().to_vec::<i32>()?;
    let correct = predictions
        .iter()
        .zip(&expected)
        .filter(|(a, b)| a == b)
        .count();
    Ok(Scores {
        loss,
        accuracy: correct as f64 / expected.len() as f64,
    })
}

// Borrow the original parameters: cloning an uninitialized Burn Param can sample
// a different random value when that clone is first accessed.
pub fn snapshot<B: Backend>(weights: Weights<'_, B>) -> Result<Snapshot> {
    weights
        .into_iter()
        .map(|(name, parameter)| Ok((name.into(), parameter.val().into_data().to_vec::<f32>()?)))
        .collect()
}

pub fn check_gradients<B: Backend>(
    weights: Weights<'_, B>,
    gradients: &GradientsParams,
) -> Result<()> {
    for (name, parameter) in weights {
        let values = gradients
            .get::<Cpu, 2>(parameter.id)
            .ok_or_else(|| format!("no gradient for {name}"))?
            .into_data()
            .to_vec::<f32>()?;
        if values.iter().any(|v| !v.is_finite()) || values.iter().all(|v| *v == 0.0) {
            return Err(format!("{name} must receive a finite, nonzero gradient").into());
        }
    }
    Ok(())
}

pub fn changes(before: &Snapshot, after: Snapshot) -> Result<BTreeMap<String, f64>> {
    after
        .into_iter()
        .map(|(name, values)| {
            let original = before.get(&name).ok_or("missing initial parameter")?;
            if values.len() != original.len() || values.iter().any(|x| !x.is_finite()) {
                return Err(format!("invalid updated parameter: {name}").into());
            }
            let change = values
                .iter()
                .zip(original)
                .map(|(a, b)| (*a as f64 - *b as f64).abs())
                .fold(0.0, f64::max);
            if change <= 1e-7 {
                return Err(format!("optimizer did not update {name}").into());
            }
            Ok((name, change))
        })
        .collect()
}

pub fn max_error<const D: usize>(a: Tensor<Cpu, D>, b: Tensor<Cpu, D>) -> f64 {
    (a - b).abs().max().into_scalar() as f64
}

#[derive(Serialize)]
pub struct Report {
    pub task: &'static str,
    pub seed: u64,
    pub learning_rate: f64,
    pub optimizer: &'static str,
    pub weight_decay: f64,
    pub steps: usize,
    pub train_examples: usize,
    pub held_out_examples: usize,
    pub initial: Scores,
    pub final_scores: Scores,
    pub final_train_scores: Scores,
    pub parameter_changes: BTreeMap<String, f64>,
    pub reload_logit_error: f64,
    pub seconds: f64,
}

impl Report {
    pub fn save(self, root: &Path) -> Result<()> {
        let passed = self.final_scores.loss < self.initial.loss * 0.5
            && self.final_scores.accuracy >= 0.875
            && self.reload_logit_error <= 1e-6;
        std::fs::write(root.join("result.json"), serde_json::to_vec_pretty(&self)?)?;
        println!(
            "{}: held-out loss {:.4} → {:.4}; accuracy {:.1}% → {:.1}%; {} {} steps in {:.2}s",
            self.task,
            self.initial.loss,
            self.final_scores.loss,
            self.initial.accuracy * 100.0,
            self.final_scores.accuracy * 100.0,
            self.steps,
            self.optimizer,
            self.seconds
        );
        println!(
            "Training loss {:.4}, accuracy {:.1}%; learning rate {}",
            self.final_train_scores.loss,
            self.final_train_scores.accuracy * 100.0,
            self.learning_rate
        );
        println!(
            "{} parameter groups changed; reload error {:.2e}; records: {}",
            self.parameter_changes.len(),
            self.reload_logit_error,
            root.display()
        );
        if !passed {
            return Err(
                "learning or reload checks failed; measurements retained in result.json".into(),
            );
        }
        Ok(())
    }
}
