//! Reproduce Table 1 of Nanda et al. (ICLR 2023), then intervene on its circuit.
//!
//! First: `python3 examples/mechanistic/grokking_prepare.py`
//! Then: `cargo run --release --locked --example grokking_fourier`
//! See `examples/mechanistic/grokking.md` for provenance and precise claim limits.
//!
//! The archived checkpoint defines five Fourier frequencies and ten directions
//! in the 512-channel MLP activation. F64 orthonormalization produces U; F32
//! hooks apply `x U U^T` to retain the circuit or subtract it to ablate it. The
//! original residual and biases remain intact. This complete-grid checkpoint
//! analysis complements from-scratch optimization in `grokking_train`.

#[path = "mechanistic/grokking_model.rs"]
mod model;

/// Reproduce the archived Fourier fits and test keep/remove/restore interventions.
/// Evaluate every modular pair and compare independent reference logits before
/// claiming that the ten-dimensional neuron subspace preserves the solution.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use burn::tensor::{Tensor, TensorData};
    use model::{B, GrokkingTransformer, HIDDEN, NEURONS, P};
    use transformers::intervention::Hooks;

    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "target/mechanistic/grokking.safetensors".into());
    let model = GrokkingTransformer::load(&path)?;
    let device = Default::default();

    // 1. The paper predicts ten directions: sine/cosine for five frequencies.
    //    W_out W_U maps each MLP neuron to output logits. Fourier-transforming
    //    its output axis identifies the corresponding directions in neuron space.
    let directions = model.fourier_directions(&[14, 35, 41, 42, 52]);
    let readout_directions: Vec<f32> = (0..NEURONS)
        .flat_map(|row| directions.iter().map(move |column| column[row] as f32))
        .collect();
    let readout_directions =
        Tensor::<B, 2>::from_data(TensorData::new(readout_directions, [NEURONS, 10]), &device);
    let basis = orthonormalize(directions);
    let basis = Tensor::<B, 2>::from_data(TensorData::new(basis, [NEURONS, 10]), &device);
    let project = |activation: Tensor<B, 2>| {
        activation
            .matmul(basis.clone())
            .matmul(basis.clone().transpose())
    };

    // 2. Evaluate every possible (a,b), in small CPU batches. Capture the neurons
    //    with the same public Hooks used by built-in models, then project, remove,
    //    or restore the circuit before the model's ordinary readout.
    let mut baseline = Metrics::default();
    let mut retained = Metrics::default();
    let mut removed = Metrics::default();
    let mut restored = Metrics::default();
    let mut no_op_max_error = 0.0_f32;
    let mut reference_max_error = 0.0_f32;
    let mut restoration_max_error = 0.0_f32;
    let mut fourier_fit = [FourierFit::default(); 10];
    for start in (0..P * P).step_by(256) {
        let count = (P * P - start).min(256);
        let (residual, activation) = model.encode(start, count)?;
        assert_eq!(residual.dims(), [count, HIDDEN]);
        let point = model::mlp_activation();
        let original =
            model.continue_with_hooks(residual.clone(), activation.clone(), &mut Hooks::new())?;

        let mut observe = Hooks::new();
        let neurons = observe.capture(&point);
        let no_op =
            model.continue_with_hooks(residual.clone(), activation.clone(), &mut observe)?;
        let captured = observe.get(&neurons)?;
        let readout = values(captured.clone().matmul(readout_directions.clone()));
        for (offset, row) in readout.chunks_exact(10).enumerate() {
            let pair = start + offset;
            let sum = pair / P + pair % P;
            for (column, &frequency) in [14, 35, 41, 42, 52].iter().enumerate() {
                let phase = std::f64::consts::TAU * (frequency * sum) as f64 / P as f64;
                fourier_fit[2 * column].add(phase.cos(), row[2 * column] as f64);
                fourier_fit[2 * column + 1].add(phase.sin(), row[2 * column + 1] as f64);
            }
        }
        let mut keep = Hooks::new();
        keep.edit(&point, &project);
        let retained_logits =
            model.continue_with_hooks(residual.clone(), activation.clone(), &mut keep)?;

        let mut remove = Hooks::new();
        remove.edit(&point, |x| x.clone() - project(x));
        let removed_logits =
            model.continue_with_hooks(residual.clone(), activation.clone(), &mut remove)?;

        // Actions run in registration order. Add the saved circuit contribution
        // back after removing it, while preserving every other part of the graph.
        let kept = project(captured);
        let mut restore = Hooks::new();
        restore.edit(&point, |x| x.clone() - project(x));
        restore.edit(&point, |x| x + kept.clone());
        let restored_logits = model.continue_with_hooks(residual, activation, &mut restore)?;
        let original = values(original);
        for (left, right) in original.iter().zip(values(no_op)) {
            no_op_max_error = no_op_max_error.max((left - right).abs());
        }
        // Independent NumPy implementation of the authors' equations, pinned
        // weights, first 8 pairs. This guards a mutually-consistent wrong port.
        if start == 0 {
            let reference: Vec<f32> =
                serde_json::from_str(include_str!("mechanistic/grokking_reference_logits.json"))?;
            assert_eq!(reference.len(), 8 * P, "eight full reference-logit rows");
            for (actual, expected) in original.iter().zip(reference) {
                reference_max_error = reference_max_error.max((actual - expected).abs());
            }
        }
        baseline.add(start, &original);
        retained.add(start, &values(retained_logits));
        removed.add(start, &values(removed_logits));
        let restored_values = values(restored_logits);
        for (actual, expected) in restored_values.iter().zip(&original) {
            restoration_max_error = restoration_max_error.max((actual - expected).abs());
        }
        restored.add(start, &restored_values);
    }
    println!(
        "Published seed-0 checkpoint, epoch 40,000; all {} input pairs",
        P * P
    );
    println!("Table 1: fraction of variance explained by cos/sin(2πk(a+b)/113)");
    let published = [
        0.932, 0.935, 0.968, 0.965, 0.970, 0.970, 0.964, 0.964, 0.974, 0.982,
    ];
    for (index, fit) in fourier_fit.iter().enumerate() {
        let fve = fit.fve();
        println!(
            "k={:2} {:3}: {:.3}% (paper {:.1}%)",
            [14, 35, 41, 42, 52][index / 2],
            if index % 2 == 0 { "cos" } else { "sin" },
            fve * 100.0,
            published[index] * 100.0
        );
        assert!(
            (fve - published[index]).abs() < 0.002,
            "Table 1 should reproduce within 0.2 percentage points"
        );
    }
    println!("Additional experiment: intervene on the full model, retaining residual and biases");
    println!(
        "{:22} {:>12} {:>14}",
        "MLP intervention", "accuracy", "cross entropy"
    );
    for (label, metrics) in [
        ("unchanged", &baseline),
        ("keep 10D subspace", &retained),
        ("remove 10D subspace", &removed),
        ("remove then restore", &restored),
    ] {
        println!(
            "{label:22} {:11.4}% {:14.8e}",
            metrics.accuracy() * 100.0,
            metrics.loss / (P * P) as f64
        );
    }
    println!(
        "no-op max error: {no_op_max_error:.2e}; reference max error: {reference_max_error:.2e}; restoration max error: {restoration_max_error:.2e}"
    );
    assert!(
        no_op_max_error < 1e-5,
        "capturing activations must preserve outputs"
    );
    assert!(
        reference_max_error < 1e-3,
        "Burn must agree with the independent forward equations"
    );
    assert!(baseline.accuracy() > 0.999 && retained.accuracy() > 0.999);
    assert!(
        removed.accuracy() < 0.2,
        "removing the predicted subspace should sharply reduce accuracy"
    );
    assert!(
        restored.accuracy() > 0.999,
        "restoring activations should recover the solution"
    );
    assert!(
        restoration_max_error < 1e-3,
        "restoration should recover full logits"
    );
    assert!(
        retained.loss < baseline.loss,
        "this checkpoint should retain low loss under restriction"
    );
    Ok(())
}

/// Modified Gram-Schmidt produces an orthonormal basis for the same span.
/// Input columns have 512 F64 entries; reject near-dependent directions before
/// casting the row-major `[512, directions]` basis to F32 for projection.
fn orthonormalize(mut columns: Vec<Vec<f64>>) -> Vec<f32> {
    for column in 0..columns.len() {
        for previous in 0..column {
            let dot: f64 = columns[column]
                .iter()
                .zip(&columns[previous])
                .map(|(a, b)| a * b)
                .sum();
            for row in 0..columns[column].len() {
                columns[column][row] -= dot * columns[previous][row];
            }
        }
        let norm = columns[column].iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(norm > 1e-8, "Fourier directions must be independent");
        for value in &mut columns[column] {
            *value /= norm;
        }
    }
    (0..model::NEURONS)
        .flat_map(|row| columns.iter().map(move |column| column[row] as f32))
        .collect()
}

/// Transfer a rank-two F32 activation to the host and reject nonfinite values.
/// Assert its flattened length matches both tensor axes before statistical use.
fn values(tensor: burn::tensor::Tensor<model::B, 2>) -> Vec<f32> {
    let shape = tensor.dims();
    let values = tensor.into_data().to_vec::<f32>().expect("F32 activations");
    assert_eq!(values.len(), shape[0] * shape[1]);
    assert!(
        values.iter().all(|value| value.is_finite()),
        "activations must be finite"
    );
    values
}

/// Host-side accumulated correctness and negative log likelihood over the complete modular grid.
#[derive(Default)]
struct Metrics {
    /// Number of grid pairs whose largest numeric-class logit matches the modular sum.
    correct: usize,
    /// Sum of per-pair negative log likelihoods in nats over 113 numeric classes.
    loss: f64,
}

impl Metrics {
    /// Accumulate complete 113-class rows beginning at grid index `start`.
    /// Use F64 `log1p(sum exp(other-target))` so near-zero checkpoint losses
    /// remain measurable even when logits themselves were computed in F32.
    fn add(&mut self, start: usize, logits: &[f32]) {
        assert_eq!(logits.len() % model::P, 0, "complete logit rows");
        for (offset, row) in logits.chunks_exact(model::P).enumerate() {
            let pair = start + offset;
            let target = (pair / model::P + pair % model::P) % model::P;
            let prediction = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0;
            self.correct += usize::from(prediction == target);
            // f64 log1p keeps near-zero losses measurable with F32 logits.
            let target_logit = row[target] as f64;
            let others: f64 = row
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != target)
                .map(|(_, value)| (*value as f64 - target_logit).exp())
                .sum();
            self.loss += others.ln_1p();
        }
    }
    /// Normalize the accumulated correct predictions by the complete 113² grid.
    fn accuracy(&self) -> f64 {
        self.correct as f64 / (model::P * model::P) as f64
    }
}

/// Sufficient statistics for a least-squares sinusoid fit over the complete grid.
#[derive(Clone, Copy, Default)]
struct FourierFit {
    /// Sum of squared theoretical sinusoid values, `Σ x²`.
    x2: f64,
    /// Sum of sinusoid/activation products, `Σ xy`.
    xy: f64,
    /// Sum of activation projections, `Σ y`, used to center total variation.
    y: f64,
    /// Sum of squared activation projections, `Σ y²`.
    y2: f64,
}

impl FourierFit {
    /// Accumulate F64 moments for the theoretical sinusoid x and activation projection y.
    fn add(&mut self, x: f64, y: f64) {
        self.x2 += x * x;
        self.xy += x * y;
        self.y += y;
        self.y2 += y * y;
    }
    /// Return `1-SSE/SST` for the paper’s zero-intercept least-squares sinusoid fit.
    /// Each nonzero sinusoid has zero mean on the complete grid; SST centers y.
    fn fve(&self) -> f64 {
        // Every selected sinusoid has exactly zero mean over all 113² pairs.
        let residual_sum_squares = self.y2 - self.xy * self.xy / self.x2;
        let variance_sum = self.y2 - self.y * self.y / (model::P * model::P) as f64;
        1.0 - residual_sum_squares / variance_sum
    }
}
