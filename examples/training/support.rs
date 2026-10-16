//! Small measurement helpers; optimization stays in each example's native Burn loop.
//!
//! Evaluation takes F32 `[examples, classes]` logits and I32 `[examples]` labels;
//! cross entropy is averaged over examples and reported in nats. Gradient and
//! update checks sample named rank-two parameter groups, not every model tensor.
//! Snapshots retain host F32 values for comparison, while records are written by
//! the caller's owning Burn module. These helpers introduce no masking, dropout,
//! optimizer state, or gradient detachment policy into the training graph.

use std::{collections::BTreeMap, path::Path};

use burn::{
    backend::Flex,
    module::Param,
    optim::GradientsParams,
    tensor::{Int, Tensor, backend::Backend},
};
use serde::Serialize;

/// Deterministic F32/I32 Flex backend used by the local training fixtures.
pub type Cpu = Flex<f32, i32>;
/// Fallible measurement or artifact operation with a contextual displayable error.
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
/// Named borrowed rank-two parameter groups selected for gradient and update checks.
pub type Weights<'a, B> = Vec<(&'static str, &'a Param<Tensor<B, 2>>)>;
/// Host F32 parameter values indexed by their experiment-readable group names.
pub type Snapshot = BTreeMap<String, Vec<f32>>;

/// Mean classification loss and accuracy for one explicit evaluation split.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct Scores {
    /// Mean categorical cross entropy in nats, promoted to F64 for reporting.
    pub loss: f64,
    /// Fraction of examples whose argmax class matches the integer label.
    pub accuracy: f64,
}

/// Measure mean categorical cross entropy and argmax accuracy for aligned F32 logits and I32 labels.
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
/// Read the original named parameter tensors into host F32 snapshots without cloning lazy parameter owners.
pub fn snapshot<B: Backend>(weights: Weights<'_, B>) -> Result<Snapshot> {
    weights
        .into_iter()
        .map(|(name, parameter)| Ok((name.into(), parameter.val().into_data().to_vec::<f32>()?)))
        .collect()
}

/// Require finite, nonzero rank-two Flex gradients for every sampled parameter group.
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

/// Return each named matrix's maximum absolute update, rejecting missing, nonfinite, or effectively unchanged values.
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

/// Return the maximum absolute F32 tensor difference as an F64 reporting scalar.
pub fn max_error<const D: usize>(a: Tensor<Cpu, D>, b: Tensor<Cpu, D>) -> f64 {
    (a - b).abs().max().into_scalar() as f64
}

/// Return the largest absolute F32 element as an F64 reporting scalar.
pub fn max_abs<const D: usize>(tensor: Tensor<Cpu, D>) -> f64 {
    tensor.abs().max().into_scalar() as f64
}

/// Largest reload deviation allowed per unit of logit magnitude.
///
/// A reloaded parameter can be a transposed view where the trained one was
/// contiguous. Values are identical, but a CPU GEMM may then sum in another
/// order: x86 AVX kernels differ by a few F32 units in the last place, which
/// is about `1e-7` of the logit. Twenty times that, relative to the largest
/// logit, still catches any parameter that failed to round-trip.
const RELOAD_RELATIVE_TOLERANCE: f64 = 2e-6;

/// Synthetic-task evidence saved before learning/reload checks are enforced.
#[derive(Serialize)]
pub struct Report {
    /// Human-readable synthetic task description.
    pub task: &'static str,
    /// Backend seed used before initialization.
    pub seed: u64,
    /// Constant learning rate supplied to the native Burn optimizer.
    pub learning_rate: f64,
    /// Native Burn optimizer name used by the example.
    pub optimizer: &'static str,
    /// Configured weight-decay coefficient; zero when no decay is applied.
    pub weight_decay: f64,
    /// Number of completed full-batch optimizer updates.
    pub steps: usize,
    /// Number of training sequences, before flattening any query slots.
    pub train_examples: usize,
    /// Number of held-out sequences.
    pub held_out_examples: usize,
    /// Held-out metrics before the first optimizer update.
    pub initial: Scores,
    /// Held-out metrics after training.
    pub final_scores: Scores,
    /// Training-split metrics after training.
    pub final_train_scores: Scores,
    /// Maximum absolute changes for the sampled named parameter groups.
    pub parameter_changes: BTreeMap<String, f64>,
    /// Maximum absolute held-out logit difference after reloading the model record.
    pub reload_logit_error: f64,
    /// Largest absolute held-out logit before reloading, which scales the reload tolerance.
    pub reload_logit_scale: f64,
    /// Elapsed wall time for initialization, training, evaluation, and record checks.
    pub seconds: f64,
}

impl Report {
    /// Persist metrics before enforcing held-out learning and model-reload tolerances.
    /// Failed checks leave `result.json` available for inspection.
    pub fn save(self, root: &Path) -> Result<()> {
        let passed = self.final_scores.loss < self.initial.loss * 0.5
            && self.final_scores.accuracy >= 0.875
            && self.reload_logit_error
                <= RELOAD_RELATIVE_TOLERANCE * self.reload_logit_scale.max(1.0);
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
