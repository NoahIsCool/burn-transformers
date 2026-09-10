//! Reproducible data, measurements, and artifacts; the training loop is in the example.
use crate::model::{P, Transformer};
use burn::{
    module::Module,
    record::{BinFileRecorder, FullPrecisionSettings},
    tensor::{Int, Tensor, TensorData, backend::Backend},
};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{error::Error, path::PathBuf, time::Instant};
use transformers::intervention::{Hooks, Layer};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[derive(Clone, Serialize)]
pub struct Options {
    pub backend: String,
    pub alpha: f64,
    pub seed: u64,
    pub steps: usize,
    pub evaluate_every: usize,
    pub smoke: bool,
    pub output: PathBuf,
}
impl Options {
    pub fn parse() -> Result<Self> {
        let mut result = Self {
            backend: "flex".into(),
            alpha: 0.,
            seed: 20260910,
            steps: 2000,
            evaluate_every: 100,
            smoke: false,
            output: "target/mechanistic/clock_and_pizza_training/alpha0".into(),
        };
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--smoke" => {
                    result.smoke = true;
                    result.steps = 5;
                    result.evaluate_every = 5;
                }
                "--paper" => result.steps = 20_000,
                "--backend" => result.backend = args.next().ok_or("missing backend")?,
                "--alpha" => result.alpha = args.next().ok_or("missing alpha")?.parse()?,
                "--seed" => result.seed = args.next().ok_or("missing seed")?.parse()?,
                "--steps" => result.steps = args.next().ok_or("missing steps")?.parse()?,
                "--evaluate-every" => {
                    result.evaluate_every = args.next().ok_or("missing interval")?.parse()?
                }
                "--output" => result.output = args.next().ok_or("missing output")?.into(),
                _ => return Err(format!("unknown option {arg}").into()),
            }
        }
        if result.steps == 0 || result.evaluate_every == 0 || !(0.0..=1.0).contains(&result.alpha) {
            return Err(
                "steps and evaluation interval must be positive; alpha must be in [0,1]".into(),
            );
        }
        Ok(result)
    }
}

pub struct Dataset {
    pub train: Vec<usize>,
    pub test: Vec<usize>,
}
impl Dataset {
    pub fn new(seed: u64) -> Self {
        let mut pairs: Vec<usize> = (0..P * P).collect();
        let mut rng = seed;
        for index in (1..pairs.len()).rev() {
            rng = rng.wrapping_add(0x9e3779b97f4a7c15);
            let mut value = rng;
            value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
            value ^= value >> 31;
            pairs.swap(index, (value % (index as u64 + 1)) as usize);
        }
        let test = pairs.split_off(P * P * 4 / 5);
        Self { train: pairs, test }
    }
    pub fn training_batch<B: Backend>(
        &self,
        device: &B::Device,
        smoke: bool,
    ) -> (Tensor<B, 2, Int>, Tensor<B, 1, Int>) {
        let pairs = if smoke {
            &self.train[..32]
        } else {
            &self.train[..]
        };
        let ids: Vec<i32> = pairs
            .iter()
            .flat_map(|p| [(p / P) as i32, (p % P) as i32])
            .collect();
        let labels: Vec<i32> = pairs.iter().map(|p| ((p / P + p % P) % P) as i32).collect();
        (
            Tensor::from_data(TensorData::new(ids, [pairs.len(), 2]), device),
            Tensor::from_data(TensorData::new(labels, [pairs.len()]), device),
        )
    }
}

#[derive(Serialize)]
struct Measurement {
    step: usize,
    seconds: f64,
    train_accuracy: f64,
    test_accuracy: f64,
    train_cross_entropy: f64,
    test_cross_entropy: f64,
    distance_irrelevance: f64,
}

pub struct Run {
    options: Options,
    started: Instant,
    measurements: Vec<Measurement>,
}
impl Run {
    pub fn begin<B: Backend>(
        options: &Options,
        data: &Dataset,
        model: &Transformer<B>,
    ) -> Result<Self> {
        std::fs::create_dir_all(&options.output)?;
        if options.output.join("protocol.json").exists() {
            return Err("output already has a protocol.json; choose a new --output to preserve prior evidence".into());
        }
        let sources: &[(&str, &[u8])] = &[
            (
                "examples/clock_and_pizza_train.rs",
                include_bytes!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/examples/clock_and_pizza_train.rs"
                )),
            ),
            (
                "examples/mechanistic/clock_and_pizza/model.rs",
                include_bytes!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/examples/mechanistic/clock_and_pizza/model.rs"
                )),
            ),
            (
                "examples/mechanistic/clock_and_pizza_training/study.rs",
                include_bytes!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/examples/mechanistic/clock_and_pizza_training/study.rs"
                )),
            ),
            (
                "examples/mechanistic/clock_and_pizza_training/manifest.json",
                include_bytes!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/examples/mechanistic/clock_and_pizza_training/manifest.json"
                )),
            ),
            (
                "src/training.rs",
                include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/training.rs")),
            ),
            (
                "src/attention.rs",
                include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/attention.rs")),
            ),
            (
                "src/intervention.rs",
                include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/intervention.rs")),
            ),
            (
                "src/operations.rs",
                include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/operations.rs")),
            ),
            (
                "Cargo.lock",
                include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.lock")),
            ),
        ];
        let mut source_sha256 = serde_json::Map::new();
        for (path, bytes) in sources {
            source_sha256.insert(
                (*path).into(),
                serde_json::Value::String(format!("{:x}", Sha256::digest(bytes))),
            );
        }
        let initial_embedding: Vec<u8> = model
            .embedding
            .val()
            .into_data()
            .to_vec::<f32>()?
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let protocol = serde_json::json!({ "options": options, "modulus": P, "train_pairs":data.train,"test_pairs":data.test,
            "compiled_source_sha256":source_sha256, "executable_sha256":format!("{:x}",Sha256::digest(std::fs::read(std::env::current_exe()?)?)), "initial_embedding_sha256":format!("{:x}",Sha256::digest(initial_embedding)), "parameter_count":model.num_params(), "architecture":"one block, width128, heads4x32, biased ReLU512, no normalization, unmasked alpha*softmax+(1-alpha)*J",
            "optimizer":{"name":"Burn AdamW", "learning_rate":0.001,"beta1":0.9,"beta2":0.98,"epsilon":1e-8,"weight_decay":2.0,"warmup":"update step starts at 0; lr=0.001*min(step/10,1)"},
            "training_loss":"Differentiable pairwise-reduced F32 cross entropy (paper casts logits to F64); evaluation CE host F64",
            "claim":"fresh seeded training adaptation; no equivalence to unpublished original seed/split; no filtering or retries based on learned mechanism" });
        std::fs::write(
            options.output.join("protocol.json"),
            serde_json::to_vec_pretty(&protocol)?,
        )?;
        Ok(Self {
            options: options.clone(),
            started: Instant::now(),
            measurements: Vec::new(),
        })
    }
    pub fn measure<B: Backend>(
        &mut self,
        step: usize,
        model: &Transformer<B>,
        data: &Dataset,
    ) -> Result<()> {
        let logits = all_logits(model)?;
        let (train_accuracy, train_cross_entropy) = metrics(&logits, &data.train);
        let (test_accuracy, test_cross_entropy) = metrics(&logits, &data.test);
        let q = distance_irrelevance(&logits);
        println!(
            "step {step:5}: train={train_accuracy:.4} test={test_accuracy:.4} CE={test_cross_entropy:.6} q={q:.4} elapsed={:.1}s",
            self.started.elapsed().as_secs_f64()
        );
        self.measurements.push(Measurement {
            step,
            seconds: self.started.elapsed().as_secs_f64(),
            train_accuracy,
            test_accuracy,
            train_cross_entropy,
            test_cross_entropy,
            distance_irrelevance: q,
        });
        std::fs::write(
            self.options.output.join("learning_curve.json"),
            serde_json::to_vec_pretty(&self.measurements)?,
        )?;
        Ok(())
    }
    pub fn finish<B: Backend>(&self, model: &Transformer<B>, _data: &Dataset) -> Result<()> {
        let mut hooks = Hooks::new();
        let pattern = hooks.capture(Layer::decoder(0).self_attention().pattern());
        let neurons = hooks.capture(Layer::decoder(0).mlp_activation().token(1));
        let (mut logits, mut patterns, mut activations) = (Vec::new(), Vec::new(), Vec::new());
        for start in (0..P * P).step_by(128) {
            logits.extend(
                model
                    .forward_with_hooks(start, 128.min(P * P - start), &mut hooks)?
                    .into_data()
                    .to_vec::<f32>()?,
            );
            patterns.extend(hooks.get(&pattern)?.into_data().to_vec::<f32>()?);
            activations.extend(hooks.get(&neurons)?.into_data().to_vec::<f32>()?);
        }
        let reference = all_logits(model)?;
        let capture_max_error = reference
            .iter()
            .zip(&logits)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        if capture_max_error != 0.0 {
            return Err(
                format!("Activation captures changed logits by {capture_max_error}").into(),
            );
        }
        let embedding = model.embedding.val().into_data().to_vec::<f32>()?;
        let mut hashes = serde_json::Map::new();
        for (name, values) in [
            ("logits", logits),
            ("embedding", embedding),
            ("attention", patterns),
            ("neurons", activations),
        ] {
            let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            hashes.insert(name.into(),serde_json::json!({"values":values.len(),"sha256":format!("{:x}",Sha256::digest(&bytes))}));
            std::fs::write(self.options.output.join(format!("{name}.f32")), bytes)?;
        }
        model.clone().save_file(
            self.options.output.join("model"),
            &BinFileRecorder::<FullPrecisionSettings>::new(),
        )?;
        let reloaded = model.clone().load_file(
            self.options.output.join("model"),
            &BinFileRecorder::<FullPrecisionSettings>::new(),
            &model.embedding.device(),
        )?;
        let reference = all_logits(model)?;
        let replay = all_logits(&reloaded)?;
        let reload_max_error = reference
            .iter()
            .zip(&replay)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        if reload_max_error != 0.0 {
            return Err(
                format!("Burn record round trip changed logits by {reload_max_error}").into(),
            );
        }
        std::fs::write(
            self.options.output.join("result.json"),
            serde_json::to_vec_pretty(
                &serde_json::json!({"options":self.options,"measurements":self.measurements,"seconds":self.started.elapsed().as_secs_f64(),"artifacts":hashes, "record_round_trip_max_logit_error":reload_max_error, "capture_max_logit_error":capture_max_error}),
            )?,
        )?;
        Ok(())
    }
}

fn all_logits<B: Backend>(model: &Transformer<B>) -> Result<Vec<f32>> {
    let mut values = Vec::with_capacity(P * P * P);
    for start in (0..P * P).step_by(128) {
        values.extend(
            model
                .forward_with_hooks(start, 128.min(P * P - start), &mut Hooks::new())?
                .into_data()
                .to_vec::<f32>()?,
        );
    }
    Ok(values)
}
fn metrics(logits: &[f32], pairs: &[usize]) -> (f64, f64) {
    let mut correct = 0;
    let mut loss = 0.;
    for &pair in pairs {
        let row = &logits[pair * P..(pair + 1) * P];
        let label = (pair / P + pair % P) % P;
        let predicted = row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0;
        correct += usize::from(predicted == label);
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        loss += row
            .iter()
            .map(|&v| (v as f64 - max).exp())
            .sum::<f64>()
            .ln()
            + max
            - row[label] as f64;
    }
    (
        correct as f64 / pairs.len() as f64,
        loss / pairs.len() as f64,
    )
}
fn std(values: &[f64]) -> f64 {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64).sqrt()
}
fn distance_irrelevance(logits: &[f32]) -> f64 {
    let correct: Vec<_> = (0..P * P)
        .map(|pair| logits[pair * P + (pair / P + pair % P) % P] as f64)
        .collect();
    (0..P)
        .map(|d| {
            std(&(0..P)
                .map(|a| correct[a * P + (a + d) % P])
                .collect::<Vec<_>>())
        })
        .sum::<f64>()
        / P as f64
        / std(&correct)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::{
        backend::{Autodiff, Flex},
        module::AutodiffModule,
        optim::{AdamWConfig, GradientsParams, Optimizer},
    };
    #[test]
    fn split_is_deterministic_disjoint_and_complete() {
        let data = Dataset::new(20260910);
        assert_eq!(data.train.len(), 2784);
        assert_eq!(data.test.len(), 697);
        assert_eq!(data.train, Dataset::new(20260910).train);
        let mut all = data.train.clone();
        all.extend(data.test);
        all.sort();
        assert_eq!(all, (0..P * P).collect::<Vec<_>>());
    }
    #[test]
    fn optimizer_discovers_recipe_parameters_and_hooks_preserve_learning() {
        type Ad = Autodiff<Flex>;
        let device = Default::default();
        Ad::seed(&device, 42);
        for alpha in [0., 1.] {
            let model = Transformer::<Ad>::init_random(alpha, &device).unwrap();
            assert_eq!(model.num_params(), 212608);
            let diagnostic = gradient_symmetry(&model).unwrap();
            assert_eq!(
                diagnostic["individual_cosines"].as_array().unwrap().len(),
                100
            );
            let (inputs, labels) = Dataset::new(42).training_batch::<Ad>(&device, true);
            let mut hooks = Hooks::new();
            let captured = hooks.capture_with_grad(Layer::decoder(0).mlp_activation());
            hooks.edit(Layer::decoder(0).mlp_output(), |hidden| hidden);
            let logits = model
                .forward_ids_with_hooks(inputs.clone(), &mut hooks)
                .unwrap();
            let auxiliary = hooks.get(&captured).unwrap().square().mean();
            let auxiliary_gradients = auxiliary.backward();
            let embedding_gradient = model
                .embedding
                .val()
                .grad(&auxiliary_gradients)
                .unwrap()
                .into_data()
                .to_vec::<f32>()
                .unwrap();
            assert!(embedding_gradient.iter().all(|v| v.is_finite()));
            assert!(embedding_gradient.iter().any(|v| v.abs() > 1e-8));
            // A new graph is needed after backward has consumed the first one.
            drop(logits);
            let logits = model
                .forward_ids_with_hooks(inputs.clone(), &mut hooks)
                .unwrap();
            let loss = transformers::training::cross_entropy(logits, labels.clone()).unwrap();
            let before = loss.clone().into_scalar();
            let gradients = loss.backward();
            let embedding_before = model.embedding.val().into_data().to_vec::<f32>().unwrap();
            let gradients = GradientsParams::from_grads(gradients, &model);
            let mut optimizer = AdamWConfig::new()
                .with_beta_2(0.98)
                .with_epsilon(1e-8)
                .with_weight_decay(2.)
                .init();
            let trained = optimizer.step(0.001, model, gradients);
            assert_ne!(
                embedding_before,
                trained.embedding.val().into_data().to_vec::<f32>().unwrap()
            );
            let (inputs, labels) = Dataset::new(42).training_batch::<Flex>(&device, true);
            let logits = trained
                .valid()
                .forward_ids_with_hooks(inputs, &mut Hooks::new())
                .unwrap();
            let after = transformers::training::cross_entropy(logits, labels)
                .unwrap()
                .into_scalar();
            assert!(
                after < before,
                "alpha={alpha}: optimizer should lower tiny-batch CE {before}->{after}"
            );
        }
    }
}

/// Paper Definition 4.1 on 100 fixed triples. The diagnostic deliberately makes
/// token activations new leaves; it does not change the training graph or weights.
pub fn gradient_symmetry<B: burn::tensor::backend::AutodiffBackend>(
    model: &Transformer<B>,
) -> Result<serde_json::Value> {
    let mut state = 20260911u64;
    let mut draw = || {
        state = state.wrapping_add(0x9e3779b97f4a7c15);
        let mut value = state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
        ((value ^ (value >> 31)) % P as u64) as i32
    };
    let triples: Vec<[i32; 3]> = (0..100).map(|_| [draw(), draw(), draw()]).collect();
    let inputs = Tensor::<B, 2, Int>::from_data(
        TensorData::new(
            triples.iter().flat_map(|t| [t[0], t[1]]).collect(),
            [100, 2],
        ),
        &model.embedding.device(),
    );
    let classes = Tensor::<B, 2, Int>::from_data(
        TensorData::new(triples.iter().map(|t| t[2]).collect(), [100, 1]),
        &model.embedding.device(),
    );
    let mut gradient_input = None;
    let mut hooks = Hooks::new();
    hooks.edit(crate::model::token_embedding(), |tokens| {
        let leaf = tokens.detach().require_grad();
        // Burn retains leaf gradients. Keep that exact tensor, not a reshaped
        // capture view, for this diagnostic's derivative readout.
        gradient_input = Some(leaf.clone());
        leaf
    });
    let selected = model
        .forward_ids_with_hooks(inputs, &mut hooks)?
        .gather(1, classes)
        .sum();
    drop(hooks);
    let gradients = selected.backward();
    let gradient = gradient_input
        .ok_or("input diagnostic point was not visited")?
        .grad(&gradients)
        .ok_or("input diagnostic leaf has no gradient")?
        .into_data()
        .to_vec::<f32>()?;
    let similarities: Vec<f64> = gradient
        .chunks_exact(2 * crate::model::HIDDEN)
        .map(|pair| {
            let (a, b) = pair.split_at(crate::model::HIDDEN);
            let dot = a
                .iter()
                .zip(b)
                .map(|(&x, &y)| x as f64 * y as f64)
                .sum::<f64>();
            let norm_a = a.iter().map(|&v| (v as f64).powi(2)).sum::<f64>().sqrt();
            let norm_b = b.iter().map(|&v| (v as f64).powi(2)).sum::<f64>().sqrt();
            dot / (norm_a * norm_b)
        })
        .collect();
    if similarities.iter().any(|v| !v.is_finite()) {
        return Err("gradient symmetricity has a zero/nonfinite gradient norm".into());
    }
    Ok(
        serde_json::json!({"mean_cosine_similarity":similarities.iter().sum::<f64>()/100.,"individual_cosines":similarities,"triples":triples,"sampling_seed":20260911,"method":"d selected raw logit / d token activation; diagnostic-only detached leaf before position addition"}),
    )
}
