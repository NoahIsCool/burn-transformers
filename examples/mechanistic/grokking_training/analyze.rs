//! Interpret this run's learned frequencies through the same public hooks as v0.2.
use crate::model::{GrokkingTransformer, NEURONS, P, mlp_activation};
use burn::module::Module;
use burn::tensor::{Tensor, TensorData, backend::Backend};
use serde_json::{Value, json};
use transformers::intervention::Hooks;

pub fn learned_circuit<B: Backend>(
    model: &GrokkingTransformer<B>,
    frequencies: &[usize],
) -> Result<Value, Box<dyn std::error::Error>> {
    let directions = model.fourier_directions(frequencies);
    let mut columns = directions.clone();
    for column in 0..columns.len() {
        for previous in 0..column {
            let dot: f64 = columns[column]
                .iter()
                .zip(&columns[previous])
                .map(|(a, b)| a * b)
                .sum();
            let (before, current) = columns.split_at_mut(column);
            for (value, &basis) in current[0].iter_mut().zip(&before[previous]) {
                *value -= dot * basis;
            }
        }
        let norm = columns[column].iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm < 1e-10 {
            return Err(
                "learned Fourier directions are rank deficient; no ablation claimed".into(),
            );
        }
        for value in &mut columns[column] {
            *value /= norm;
        }
    }
    let device = model
        .devices()
        .into_iter()
        .next()
        .ok_or("model has no device")?;
    let width = frequencies.len() * 2;
    let matrix = |columns: &[Vec<f64>]| {
        let values: Vec<f32> = (0..NEURONS)
            .flat_map(|row| columns.iter().map(move |col| col[row] as f32))
            .collect();
        Tensor::<B, 2>::from_data(TensorData::new(values, [NEURONS, width]), &device)
    };
    let basis = matrix(&columns);
    let readout = matrix(&directions);
    let project = |x: Tensor<B, 2>| x.matmul(basis.clone()).matmul(basis.clone().transpose());
    let mut conditions = [Metrics::default(); 4];
    let mut fits = vec![Fit::default(); width];
    let mut identity_error = 0.0_f32;
    let mut restoration_error = 0.0_f32;
    for start in (0..P * P).step_by(256) {
        let count = (P * P - start).min(256);
        let (residual, activation) = model.encode(start, count)?;
        let baseline =
            model.continue_with_hooks(residual.clone(), activation.clone(), &mut Hooks::new())?;
        let mut hooks = Hooks::new();
        let captured = hooks.capture(mlp_activation());
        let observed =
            model.continue_with_hooks(residual.clone(), activation.clone(), &mut hooks)?;
        let neurons = hooks.get(&captured)?;
        let measurements = neurons
            .clone()
            .matmul(readout.clone())
            .into_data()
            .to_vec::<f32>()?;
        for (offset, row) in measurements.chunks_exact(width).enumerate() {
            let pair = start + offset;
            let sum = pair / P + pair % P;
            for (i, &frequency) in frequencies.iter().enumerate() {
                let phase = std::f64::consts::TAU * (frequency * sum) as f64 / P as f64;
                fits[i * 2].add(phase.cos(), row[i * 2] as f64);
                fits[i * 2 + 1].add(phase.sin(), row[i * 2 + 1] as f64);
            }
        }
        let mut keep = Hooks::new();
        keep.edit(mlp_activation(), &project);
        let retained =
            model.continue_with_hooks(residual.clone(), activation.clone(), &mut keep)?;
        let mut remove = Hooks::new();
        remove.edit(mlp_activation(), |x| x.clone() - project(x));
        let removed =
            model.continue_with_hooks(residual.clone(), activation.clone(), &mut remove)?;
        let saved = project(neurons);
        let mut restore = Hooks::new();
        restore.edit(mlp_activation(), |x| x.clone() - project(x));
        restore.edit(mlp_activation(), |x| x + saved.clone());
        let restored = model.continue_with_hooks(residual, activation, &mut restore)?;
        let baseline = baseline.into_data().to_vec::<f32>()?;
        let observed = observed.into_data().to_vec::<f32>()?;
        let retained = retained.into_data().to_vec::<f32>()?;
        let removed = removed.into_data().to_vec::<f32>()?;
        let restored = restored.into_data().to_vec::<f32>()?;
        for ((&a, b), c) in baseline.iter().zip(observed).zip(&restored) {
            identity_error = identity_error.max((a - b).abs());
            restoration_error = restoration_error.max((a - c).abs());
        }
        for (metrics, values) in conditions
            .iter_mut()
            .zip([baseline, retained, removed, restored])
        {
            metrics.add(start, &values);
        }
    }
    if identity_error > 1e-6 || restoration_error > 1e-3 {
        return Err(format!(
            "hook control failed: capture {identity_error}, restoration {restoration_error}"
        )
        .into());
    }
    let conditions: Vec<Value> = ["baseline", "keep_learned_10d", "remove_learned_10d", "remove_then_restore"].iter()
        .zip(conditions).map(|(name, m)| json!({"name":name,"accuracy":m.correct as f64/(P*P) as f64,"cross_entropy":m.loss/(P*P) as f64})).collect();
    Ok(json!({
        "frequency_selection": "top five final neuron-to-logit Fourier pair powers; no checkpoint frequencies reused",
        "frequencies": frequencies,
        "evaluation_pairs": P*P,
        "analysis_classes": P,
        "conditions": conditions,
        "legacy_zero_intercept_fve": fits.iter().map(Fit::fve).collect::<Vec<_>>(),
        "affine_sinusoid_variance_explained": fits.iter().map(Fit::affine_fve).collect::<Vec<_>>(),
        "fit_note": "The legacy archived-model diagnostic fits y=A*x without an intercept but divides by centered variance; a nonzero activation mean can make it negative. The affine diagnostic fits y=mean(y)+A*x and reports explained centered variance. Both are disclosed; neither was a training success criterion.",
        "identity_max_logit_error":identity_error,
        "restoration_max_logit_error":restoration_error,
        "claim": "Mechanistic analysis of this freshly trained run; not reproduction of the archived seed-zero Table 1 entries"
    }))
}

#[derive(Clone, Copy, Default)]
struct Metrics {
    correct: usize,
    loss: f64,
}
impl Metrics {
    fn add(&mut self, start: usize, logits: &[f32]) {
        // Training includes '=', while the author's Fourier analysis drops it.
        for (offset, row) in logits.chunks_exact(P + 1).enumerate() {
            let row = &row[..P];
            let pair = start + offset;
            let target = (pair / P + pair % P) % P;
            let maximum = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
            self.loss += row
                .iter()
                .map(|&v| (v as f64 - maximum).exp())
                .sum::<f64>()
                .ln()
                + maximum
                - row[target] as f64;
            self.correct += usize::from(
                row.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .unwrap()
                    .0
                    == target,
            );
        }
    }
}
#[derive(Clone, Copy, Default)]
struct Fit {
    x2: f64,
    xy: f64,
    y: f64,
    y2: f64,
}
impl Fit {
    fn add(&mut self, x: f64, y: f64) {
        self.x2 += x * x;
        self.xy += x * y;
        self.y += y;
        self.y2 += y * y;
    }
    fn fve(&self) -> f64 {
        1.0 - (self.y2 - self.xy * self.xy / self.x2) / (self.y2 - self.y * self.y / (P * P) as f64)
    }
    fn affine_fve(&self) -> f64 {
        (self.xy * self.xy / self.x2) / (self.y2 - self.y * self.y / (P * P) as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn affine_sinusoid_fit_retains_a_nonzero_activation_mean() {
        let mut fit = Fit::default();
        for a in 0..P {
            for b in 0..P {
                let x = (std::f64::consts::TAU * 7.0 * (a + b) as f64 / P as f64).cos();
                fit.add(x, 3.0 + 2.0 * x);
            }
        }
        assert!(
            fit.fve() < 0.0,
            "legacy metric penalizes the unmodeled mean"
        );
        assert!((fit.affine_fve() - 1.0).abs() < 1e-10);
    }
}
