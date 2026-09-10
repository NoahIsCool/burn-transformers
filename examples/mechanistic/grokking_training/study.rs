//! A plain Burn training loop: model -> loss -> backward -> AdamW -> model.
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
pub struct Settings {
    pub backend: String,
    pub steps: usize,
    pub seed: u64,
    pub output: PathBuf,
    pub resume: bool,
    pub benchmark: bool,
    #[serde(default)]
    pub analyze_only: bool,
}
impl Settings {
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

fn strongest_frequencies(powers: &[f64]) -> Vec<usize> {
    let mut frequencies: Vec<usize> = (1..=P / 2).collect();
    frequencies.sort_by(|&a, &b| powers[b - 1].total_cmp(&powers[a - 1]).then(a.cmp(&b)));
    frequencies.truncate(5);
    frequencies
}

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

fn write_json(path: PathBuf, value: &impl Serialize) -> AnyResult<()> {
    let temp = path.with_extension("json.tmp");
    fs::write(&temp, serde_json::to_vec_pretty(value)?)?;
    fs::rename(temp, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::{
        backend::{Autodiff, Flex},
        nn::loss::CrossEntropyLossConfig,
        optim::GradientsParams,
        tensor::ElementConversion,
    };
    use transformers::intervention::Hooks;
    type TestBackend = Autodiff<Flex>;

    #[test]
    fn native_optimizer_discovers_every_parameter_and_updates_the_transformer() {
        let device = Default::default();
        TestBackend::seed(&device, 42);
        let model = GrokkingTransformer::<TestBackend>::init_random(&device).unwrap();
        assert_eq!(model.num_params(), 226816);
        let (input, targets) = batch::<TestBackend>(&[0, 2, 113, 791, 4001, 12211], &device);
        let mut hooks = Hooks::new();
        let captured = hooks.capture_with_grad(crate::model::mlp_activation());
        let logits = model.forward_with_hooks(input.clone(), &mut hooks).unwrap();
        let baseline = logits.clone().into_data().to_vec::<f32>().unwrap();
        let ce = CrossEntropyLossConfig::new()
            .init(&device)
            .forward(logits, targets);
        let loss = ce
            + hooks
                .get(&captured)
                .unwrap()
                .square()
                .mean()
                .mul_scalar(0.001);
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        assert_eq!(
            grads.len(),
            11,
            "embedding, positions, four attention matrices, four MLP tensors, readout"
        );
        let mut optimizer = AdamWConfig::new()
            .with_epsilon(1e-8)
            .with_weight_decay(1.0)
            .init();
        let model = optimizer.step(1e-3, model, grads);
        let after = model
            .forward(input)
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        assert!(after.iter().all(|value| value.is_finite()));
        assert!(
            baseline
                .iter()
                .zip(after)
                .any(|(before, after)| (before - after).abs() > 1e-5)
        );
    }

    #[test]
    fn connected_activation_loss_reaches_the_prefix_without_the_readout() {
        let device = Default::default();
        TestBackend::seed(&device, 9);
        let model = GrokkingTransformer::<TestBackend>::init_random(&device).unwrap();
        let (input, _) = batch::<TestBackend>(&[1, 7, 113, 902], &device);
        let mut hooks = Hooks::new();
        let captured = hooks.capture_with_grad(crate::model::mlp_activation());
        let _ = model.forward_with_hooks(input, &mut hooks).unwrap();
        let auxiliary = hooks.get(&captured).unwrap().square().mean();
        let gradients = GradientsParams::from_grads(auxiliary.backward(), &model);
        assert_eq!(
            gradients.len(),
            8,
            "an intermediate loss reaches embedding, positions, attention and MLP input; the downstream MLP output/readout are outside its graph"
        );
    }

    #[test]
    fn burn_model_and_optimizer_records_resume_the_next_update_exactly() {
        let device = Default::default();
        TestBackend::seed(&device, 7);
        let model = GrokkingTransformer::<TestBackend>::init_random(&device).unwrap();
        let (input, targets) = batch::<TestBackend>(&[1, 40, 700, 5555], &device);
        let loss_function = CrossEntropyLossConfig::new().init(&device);
        let mut optimizer = AdamWConfig::new()
            .with_beta_2(0.98)
            .with_epsilon(1e-8)
            .with_weight_decay(1.0)
            .init();
        let loss = loss_function.forward(model.forward(input.clone()).unwrap(), targets.clone());
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        let model = optimizer.step(1e-3, model, grads);
        let dir = tempfile::tempdir().unwrap();
        let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
        recorder
            .record(model.clone().into_record(), dir.path().join("model"))
            .unwrap();
        recorder
            .record(optimizer.to_record(), dir.path().join("optimizer"))
            .unwrap();
        let restored = GrokkingTransformer::<TestBackend>::init_random(&device)
            .unwrap()
            .load_record(recorder.load(dir.path().join("model"), &device).unwrap());
        let mut restored_optimizer = optimizer.clone().load_record(
            recorder
                .load(dir.path().join("optimizer"), &device)
                .unwrap(),
        );
        let loss_a = loss_function.forward(model.forward(input.clone()).unwrap(), targets.clone());
        let loss_b = loss_function.forward(restored.forward(input.clone()).unwrap(), targets);
        assert_eq!(
            loss_a.clone().into_scalar().elem::<f32>(),
            loss_b.clone().into_scalar().elem::<f32>()
        );
        let grads_a = GradientsParams::from_grads(loss_a.backward(), &model);
        let grads_b = GradientsParams::from_grads(loss_b.backward(), &restored);
        let a = optimizer.step(1e-3, model, grads_a);
        let b = restored_optimizer.step(1e-3, restored, grads_b);
        assert_eq!(
            a.forward(input.clone())
                .unwrap()
                .into_data()
                .to_vec::<f32>()
                .unwrap(),
            b.forward(input)
                .unwrap()
                .into_data()
                .to_vec::<f32>()
                .unwrap()
        );
    }

    #[test]
    fn resume_rejects_earlier_targets_benchmarks_and_changed_training_sources() {
        let config = Settings {
            backend: "flex".into(),
            steps: 20,
            seed: 0,
            output: "unused".into(),
            resume: true,
            benchmark: false,
            analyze_only: false,
        };
        let mut checkpoint = Checkpoint {
            step: 10,
            directory: "unused".into(),
            settings: config.clone(),
            observations: vec![],
            train_split_sha256: String::new(),
        };
        let mut provenance = serde_json::json!({
            "protocol_sha256": format!("{:x}", Sha256::digest(include_bytes!("manifest.json"))),
            "source_sha256": source_fingerprints()
        });
        assert!(validate_resume(&config, &checkpoint, &provenance).is_ok());
        let mut earlier = config.clone();
        earlier.steps = 9;
        assert!(
            validate_resume(&earlier, &checkpoint, &provenance)
                .unwrap_err()
                .to_string()
                .contains("below saved")
        );
        checkpoint.settings.benchmark = true;
        assert!(
            validate_resume(&config, &checkpoint, &provenance)
                .unwrap_err()
                .to_string()
                .contains("timing probes")
        );
        checkpoint.settings.benchmark = false;
        provenance["source_sha256"]["model"] = serde_json::json!("changed");
        assert!(
            validate_resume(&config, &checkpoint, &provenance)
                .unwrap_err()
                .to_string()
                .contains("training sources")
        );
        provenance["source_sha256"] = source_fingerprints();
        provenance["protocol_sha256"] = serde_json::json!("changed");
        assert!(
            validate_resume(&config, &checkpoint, &provenance)
                .unwrap_err()
                .to_string()
                .contains("protocol changed")
        );
    }

    #[test]
    fn fourier_power_identifies_a_known_frequency_and_ignores_dc() {
        let weights: Vec<f32> = (0..P)
            .map(|token| {
                (3.0 + (std::f64::consts::TAU * 7.0 * token as f64 / P as f64).cos()) as f32
            })
            .collect();
        let powers = fourier_power(&weights, 1, false);
        assert!(powers[6] > 1.0 - 1e-12);
        assert!((powers.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }
}

fn file_hash(path: PathBuf) -> AnyResult<String> {
    Ok(format!("{:x}", Sha256::digest(fs::read(path)?)))
}
