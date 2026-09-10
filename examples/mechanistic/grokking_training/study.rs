//! A plain Burn training loop: model -> loss -> backward -> AdamW -> model.
//!
//! The checked-in split fixes 3,830 training and 8,939 held-out modular-addition
//! pairs. Full-batch F32 optimization follows the pinned AdamW protocol; F64
//! host-side loss and Fourier measurements avoid adding a diagnostic gradient.
//! Model and optimizer records preserve native Burn parameter ownership. Source,
//! protocol, and split hashes bind resumable training to its original experiment;
//! read-only analysis records both historical and current source identities.
use crate::model::{GrokkingTransformer, HIDDEN, NEURONS, P};
use burn::{
    module::{AutodiffModule, Module},
    optim::{AdamWConfig, Optimizer},
    record::{BinFileRecorder, FullPrecisionSettings, Recorder},
    tensor::{
        Int, Tensor, TensorData,
        backend::{AutodiffBackend, Backend},
    },
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{error::Error, fs, path::PathBuf, time::Instant};

#[path = "analyze.rs"]
mod analyze;

type AnyResult<T> = Result<T, Box<dyn Error>>;

#[derive(Clone, Debug, Serialize, Deserialize)]
/// Serializable command-line settings recorded with each training checkpoint.
pub struct Settings {
    /// Execution backend name; feature availability is checked by the entry point.
    pub backend: String,
    /// Total target updates, including any updates already saved in a resumed run.
    pub steps: usize,
    /// Backend initialization seed; the checked-in data split is independent.
    pub seed: u64,
    /// Directory for source provenance, learning curves, and native Burn records.
    pub output: PathBuf,
    /// Load the latest model and optimizer instead of overwriting an existing run.
    pub resume: bool,
    /// Time disposable updates with explicit device synchronization.
    pub benchmark: bool,
    #[serde(default)]
    /// Load a checkpoint for circuit analysis without optimizer updates.
    pub analyze_only: bool,
}
impl Settings {
    /// Parse explicit run options and reject zero steps or resumable benchmarks.
    /// `--analyze` implies loading a checkpoint but permits a different analysis backend.
    pub fn from_args() -> AnyResult<Self> {
        let mut config = Self {
            backend: "flex".into(),
            steps: 40000,
            seed: 0,
            output: "target/mechanistic/grokking-training".into(),
            resume: false,
            benchmark: false,
            analyze_only: false,
        };
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--backend" => config.backend = args.next().ok_or("--backend needs a value")?,
                "--steps" => config.steps = args.next().ok_or("--steps needs a value")?.parse()?,
                "--seed" => config.seed = args.next().ok_or("--seed needs a value")?.parse()?,
                "--output" => config.output = args.next().ok_or("--output needs a path")?.into(),
                "--resume" => config.resume = true,
                "--analyze" => {
                    config.resume = true;
                    config.analyze_only = true;
                }
                "--benchmark" => config.benchmark = true,
                _ => return Err(format!("unknown argument {arg}").into()),
            }
        }
        if config.steps == 0 {
            return Err("--steps must be positive".into());
        }
        if config.resume && config.benchmark {
            return Err("benchmark runs are disposable timing probes and cannot resume".into());
        }
        Ok(config)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Observation {
    step: usize,
    elapsed_seconds: f64,
    train_loss: f64,
    test_loss: f64,
    train_accuracy: f64,
    test_accuracy: f64,
    embedding_fourier_power: Vec<f64>,
    readout_fourier_power: Vec<f64>,
}
#[derive(Serialize, Deserialize)]
struct Checkpoint {
    step: usize,
    directory: PathBuf,
    settings: Settings,
    observations: Vec<Observation>,
    train_split_sha256: String,
}

/// Train, resume, benchmark, or analyze the registered one-block transformer.
/// Normal runs use full-batch AdamW and a ten-update linear warmup from zero;
/// periodic native records include optimizer moments and the exact split.
/// At 40,000 updates, test the preregistered delayed-generalization criterion
/// and analyze frequencies selected from this run’s final readout spectrum.
pub fn train<B: AutodiffBackend>(config: Settings) -> AnyResult<()> {
    let device = B::Device::default();
    B::seed(&device, config.seed);
    fs::create_dir_all(&config.output)?;
    if !config.resume && config.output.join("latest.json").exists() {
        return Err(
            "output already contains a run; use --resume or a new --output directory".into(),
        );
    }
    #[derive(Deserialize)]
    struct Split {
        train_indices: Vec<usize>,
    }
    let train = serde_json::from_str::<Split>(include_str!("train_split.json"))?.train_indices;
    let test: Vec<usize> = (0..P * P).filter(|pair| !train.contains(pair)).collect();
    let split_hash = format!("{:x}", Sha256::digest(include_bytes!("train_split.json")));
    let (inputs, targets) = batch::<B>(&train, &device);
    let mut model = GrokkingTransformer::<B>::init_random(&device)?;
    let mut optimizer = AdamWConfig::new()
        .with_beta_1(0.9)
        .with_beta_2(0.98)
        .with_epsilon(1e-8)
        .with_weight_decay(1.0)
        .init();
    let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
    if !config.resume {
        recorder.record(
            model.clone().into_record(),
            config.output.join("initial-model"),
        )?;
        write_json(
            config.output.join("run.json"),
            &serde_json::json!({
                "settings": config,
                "protocol_sha256": format!("{:x}", Sha256::digest(include_bytes!("manifest.json"))),
                "initial_model_sha256": file_hash(config.output.join("initial-model.bin"))?,
                "source_sha256": source_fingerprints()
            }),
        )?;
    }
    let mut start_step = 0;
    let mut observations = Vec::new();
    if config.resume {
        let checkpoint: Checkpoint =
            serde_json::from_slice(&fs::read(config.output.join("latest.json"))?)?;
        if checkpoint.train_split_sha256 != split_hash
            || !config.analyze_only
                && (checkpoint.settings.seed != config.seed
                    || checkpoint.settings.backend != config.backend)
        {
            return Err("resume requires the same seed, backend and dataset split".into());
        }
        let provenance: serde_json::Value =
            serde_json::from_slice(&fs::read(config.output.join("run.json"))?)?;
        if !config.analyze_only {
            validate_resume(&config, &checkpoint, &provenance)?;
        }
        model = model.load_record(recorder.load(checkpoint.directory.join("model"), &device)?);
        if config.analyze_only {
            let valid = model.valid();
            let (_, readout) = valid.fourier_weights();
            let frequencies = strongest_frequencies(&fourier_power(&readout, NEURONS, true));
            let mut report = analyze::learned_circuit(&valid, &frequencies)?;
            report["checkpoint_step"] = serde_json::json!(checkpoint.step);
            report["trained_backend"] = serde_json::json!(checkpoint.settings.backend);
            report["analysis_backend"] = serde_json::json!(config.backend);
            report["trained_source_sha256"] = provenance["source_sha256"].clone();
            report["analysis_source_sha256"] = source_fingerprints();
            write_json(config.output.join("learned-circuit-replay.json"), &report)?;
            println!(
                "Read-only analysis of checkpoint {} saved to learned-circuit-replay.json",
                checkpoint.step
            );
            return Ok(());
        }
        optimizer =
            optimizer.load_record(recorder.load(checkpoint.directory.join("optimizer"), &device)?);
        start_step = checkpoint.step;
        observations = checkpoint.observations;
    }
    let previous_elapsed = observations.last().map_or(0.0, |row| row.elapsed_seconds);
    let started = Instant::now();
    if start_step == 0 && !config.benchmark {
        observations.push(observe(&model.valid(), &train, &test, 0, 0.0)?);
        write_json(config.output.join("progress.json"), &observations)?;
    }
    // The author's LambdaLR starts at zero and increments AFTER each update.
    // These are full-batch updates; there is no shuffling or accumulation loop.
    for step in start_step..config.steps {
        let learning_rate = 1e-3 * (step as f64 / 10.0).min(1.0);
        model = crate::training_step(
            model,
            &mut optimizer,
            inputs.clone(),
            targets.clone(),
            learning_rate,
        )?;
        let completed = step + 1;
        if config.benchmark {
            // Force completion, so the timing measures actual work on lazy GPUs.
            let _ = model.forward(inputs.clone())?.mean().into_scalar();
            println!(
                "step {completed}: elapsed {:.3}s",
                started.elapsed().as_secs_f64()
            );
        } else if completed == config.steps
            || completed <= 1000 && completed % 100 == 0
            || completed % 500 == 0
        {
            let elapsed = previous_elapsed + started.elapsed().as_secs_f64();
            let row = observe(&model.valid(), &train, &test, completed, elapsed)?;
            println!(
                "step {completed:5}: train {:.2}% loss {:.6}, test {:.2}% loss {:.6}, {:.1}s",
                row.train_accuracy * 100.0,
                row.train_loss,
                row.test_accuracy * 100.0,
                row.test_loss,
                elapsed
            );
            observations.push(row);
            write_json(config.output.join("progress.json"), &observations)?;
        }
        if completed == config.steps || !config.benchmark && completed % 1000 == 0 {
            let directory = config.output.join(format!("checkpoint-{completed:05}"));
            fs::create_dir_all(&directory)?;
            recorder.record(model.clone().into_record(), directory.join("model"))?;
            recorder.record(optimizer.to_record(), directory.join("optimizer"))?;
            let checkpoint = Checkpoint {
                step: completed,
                directory,
                settings: config.clone(),
                observations: observations.clone(),
                train_split_sha256: split_hash.clone(),
            };
            write_json(config.output.join("latest.json"), &checkpoint)?;
        }
    }
    let elapsed = started.elapsed().as_secs_f64();
    println!(
        "{} updates in {elapsed:.3}s ({:.4}s/update); records: {}",
        config.steps - start_step,
        elapsed / (config.steps - start_step).max(1) as f64,
        config.output.display()
    );
    if config.steps >= 40000 && !config.benchmark {
        let early_memorization = observations
            .iter()
            .find(|row| row.train_accuracy > 0.99 && row.test_accuracy < 0.5);
        let generalization = early_memorization.and_then(|early| {
            observations
                .iter()
                .find(|row| row.step > early.step && row.test_accuracy > 0.99)
        });
        let reproduced =
            matches!((early_memorization, generalization), (Some(a), Some(b)) if b.step > a.step);
        write_json(
            config.output.join("outcome.json"),
            &serde_json::json!({
                "delayed_generalization_observed": reproduced,
                "memorization_step": early_memorization.map(|row| row.step),
                "generalization_step": generalization.map(|row| row.step),
                "claim": "One from-scratch Burn run, not the original seeded training curve"
            }),
        )?;
        let powers = &observations
            .last()
            .ok_or("no final observation")?
            .readout_fourier_power;
        let frequencies = strongest_frequencies(powers);
        let analysis = analyze::learned_circuit(&model.valid(), &frequencies)?;
        write_json(config.output.join("learned-circuit.json"), &analysis)?;
        println!(
            "Learned Fourier frequencies: {frequencies:?}; analysis saved to learned-circuit.json"
        );
        if !reproduced {
            return Err(
                "completed run did not meet the preregistered grokking criterion; results retained"
                    .into(),
            );
        }
    }
    Ok(())
}

/// Reject backward step targets, disposable benchmarks, and changed protocol/source.
/// Seed, backend, and split compatibility are checked by the caller before this
/// guard; read-only analysis deliberately bypasses optimizer-resume restrictions.
fn validate_resume(
    config: &Settings,
    checkpoint: &Checkpoint,
    provenance: &serde_json::Value,
) -> AnyResult<()> {
    if config.steps < checkpoint.step {
        return Err(format!(
            "--steps {} is below saved step {}; use --analyze for a read-only replay",
            config.steps, checkpoint.step
        )
        .into());
    }
    if config.benchmark || checkpoint.settings.benchmark {
        return Err("benchmark runs are disposable timing probes and cannot resume".into());
    }
    let protocol = format!("{:x}", Sha256::digest(include_bytes!("manifest.json")));
    if provenance["protocol_sha256"].as_str() != Some(protocol.as_str()) {
        return Err("frozen protocol changed since this run; start a new output directory".into());
    }
    if provenance["source_sha256"] != source_fingerprints() {
        return Err("training sources changed since this run; restore its sources to resume, or use --analyze to inspect it without training".into());
    }
    Ok(())
}

/// Hash compile-time source bytes that define the training and analysis graph.
/// Documentation edits also change these exact identities; historical evidence
/// is preserved, and resuming requires the original source snapshot.
fn source_fingerprints() -> serde_json::Value {
    let hash = |bytes: &[u8]| format!("{:x}", Sha256::digest(bytes));
    serde_json::json!({
        "model": hash(include_bytes!("../grokking_model.rs")),
        "training_loop": hash(include_bytes!("study.rs")),
        "entrypoint": hash(include_bytes!("../../grokking_train.rs")),
        "analysis": hash(include_bytes!("analyze.rs")),
        "attention": hash(include_bytes!("../../../src/attention.rs")),
        "operations": hash(include_bytes!("../../../src/operations.rs")),
        "intervention": hash(include_bytes!("../../../src/intervention.rs")),
        "cargo_lock": hash(include_bytes!("../../../Cargo.lock"))
    })
}

/// Select five nonzero Fourier pairs by descending power, breaking ties by k.
/// Input powers correspond to frequencies 1 through floor(113/2).
fn strongest_frequencies(powers: &[f64]) -> Vec<usize> {
    let mut frequencies: Vec<usize> = (1..=P / 2).collect();
    frequencies.sort_by(|&a, &b| powers[b - 1].total_cmp(&powers[a - 1]).then(a.cmp(&b)));
    frequencies.truncate(5);
    frequencies
}

/// Convert grid indices `a*113+b` to `[batch,3]` integer `[a,b,=]` inputs
/// and `[batch]` targets `(a+b) mod 113` on the requested device.
fn batch<B: Backend>(
    pairs: &[usize],
    device: &B::Device,
) -> (Tensor<B, 2, Int>, Tensor<B, 1, Int>) {
    let ids: Vec<i32> = pairs
        .iter()
        .flat_map(|&pair| [(pair / P) as i32, (pair % P) as i32, P as i32])
        .collect();
    let targets: Vec<i32> = pairs
        .iter()
        .map(|&pair| ((pair / P + pair % P) % P) as i32)
        .collect();
    (
        Tensor::from_data(TensorData::new(ids, [pairs.len(), 3]), device),
        Tensor::from_data(TensorData::new(targets, [pairs.len()]), device),
    )
}

/// Return mean cross entropy and accuracy for the supplied nonempty split.
/// Use 512-row inference chunks and F64 log-sum-exp of finite F32 logits;
/// all 114 training classes, including equals, contribute to normalization.
fn evaluate<B: Backend>(model: &GrokkingTransformer<B>, pairs: &[usize]) -> AnyResult<(f64, f64)> {
    let mut loss = 0.0;
    let mut correct = 0;
    let device = model
        .devices()
        .into_iter()
        .next()
        .ok_or("model has no device")?;
    for chunk in pairs.chunks(512) {
        let (inputs, _) = batch::<B>(chunk, &device);
        let logits = model.forward(inputs)?.into_data().to_vec::<f32>()?;
        for (&pair, row) in chunk.iter().zip(logits.chunks_exact(P + 1)) {
            if !row.iter().all(|v| v.is_finite()) {
                return Err("nonfinite logits".into());
            }
            let target = (pair / P + pair % P) % P;
            let maximum = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
            loss += row
                .iter()
                .map(|&v| (v as f64 - maximum).exp())
                .sum::<f64>()
                .ln()
                + maximum
                - row[target] as f64;
            correct += usize::from(
                row.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .unwrap()
                    .0
                    == target,
            );
        }
    }
    Ok((
        loss / pairs.len() as f64,
        correct as f64 / pairs.len() as f64,
    ))
}

/// Snapshot train/test metrics and normalized embedding/readout Fourier energy.
/// Diagnostics consume inference tensors and do not contribute training gradients.
fn observe<B: Backend>(
    model: &GrokkingTransformer<B>,
    train: &[usize],
    test: &[usize],
    step: usize,
    elapsed_seconds: f64,
) -> AnyResult<Observation> {
    let (train_loss, train_accuracy) = evaluate(model, train)?;
    let (test_loss, test_accuracy) = evaluate(model, test)?;
    let (embedding, readout) = model.fourier_weights();
    Ok(Observation {
        step,
        elapsed_seconds,
        train_loss,
        test_loss,
        train_accuracy,
        test_accuracy,
        embedding_fourier_power: fourier_power(&embedding, HIDDEN, false),
        readout_fourier_power: fourier_power(&readout, NEURONS, true),
    })
}

/// Nonzero sine/cosine energy by frequency, normalized to sum to one.
/// Read F32 weights as `[113, channels]`, or `[channels, 113]` when transposed.
/// Sum squared sine and cosine coefficients over channels in F64, excluding
/// the DC component; an all-zero spectrum stays zero without division by zero.
fn fourier_power(weights: &[f32], channels: usize, transposed: bool) -> Vec<f64> {
    let mut powers: Vec<f64> = (1..=P / 2)
        .map(|frequency| {
            (0..channels)
                .map(|channel| {
                    let mut cosine = 0.0;
                    let mut sine = 0.0;
                    for token in 0..P {
                        let phase = std::f64::consts::TAU * (frequency * token) as f64 / P as f64;
                        let value = weights[if transposed {
                            channel * P + token
                        } else {
                            token * channels + channel
                        }] as f64;
                        cosine += value * phase.cos();
                        sine += value * phase.sin();
                    }
                    cosine * cosine + sine * sine
                })
                .sum()
        })
        .collect();
    let total: f64 = powers.iter().sum();
    for value in &mut powers {
        *value /= total.max(f64::MIN_POSITIVE);
    }
    powers
}

/// Write readable JSON through a sibling temporary file, then rename it into place.
/// Readers therefore see a complete checkpoint pointer or progress snapshot.
fn write_json(path: PathBuf, value: &impl Serialize) -> AnyResult<()> {
    let temp = path.with_extension("json.tmp");
    fs::write(&temp, serde_json::to_vec_pretty(value)?)?;
    fs::rename(temp, path)?;
    Ok(())
}

#[cfg(test)]
#[path = "study_tests.rs"]
mod tests;

/// Identify an emitted checkpoint by SHA-256 of its exact stored bytes.
fn file_hash(path: PathBuf) -> AnyResult<String> {
    Ok(format!("{:x}", Sha256::digest(fs::read(path)?)))
}
