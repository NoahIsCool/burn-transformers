//! Artifact checks, fixed data, and numerical reporting for the learned replacement.
//!
//! # Conditioned latent replacement
//!
//! Cache clean MLP 10 input `[batch, 1, 768]` and oracle-patched teacher logits
//! `[batch, 50257]`. The teacher inserts clean MLP output at token twelve into a
//! corrupted causal forward; a trainable bottleneck predicts a substitute from
//! the clean input features. This makes the missing information source explicit.
//! No padding or decode cache is used, and detached caches keep the fixed backbone
//! out of feature extraction gradients while its later blocks remain differentiable.
//!
//! # Evaluation and protocol boundaries
//!
//! Train in F32 and compute host F64 `KL(oracle || candidate)` over the entire
//! vocabulary. Numeric preference and cutoff sharpness use the clean year and
//! retain full-vocabulary probability mass. SHA-256 fixes artifact identities and
//! batch order; create-only output directories preserve earlier runs. The 97/97
//! clean-prompt split is fixed, although corrupted controls can repeat. Report
//! this as a new small distillation experiment, not paper training reproduction.

use burn::{
    backend::Flex,
    module::{Module, ModuleVisitor, Param, ParamId},
    optim::GradientsParams,
    tensor::{Int, Tensor, TensorData, backend::Backend},
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fs, io::Read, path::PathBuf};
use transformers::{
    CausalLmInput, ForwardOptions,
    checkpoint::LoadOptions,
    intervention::{Hooks, Layer},
    models::gpt2::{Gpt2Activation, Gpt2Config, Gpt2ForCausalLm},
};

pub type Cpu = Flex<f32, i32>;
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

pub struct Options {
    pub prepared: PathBuf,
    pub output: PathBuf,
    pub smoke: bool,
}

impl Options {
    /// Read local paths and the smoke flag, creating a new output directory.
    ///
    /// Reject unknown or incomplete options and existing result directories.
    /// Smoke mode uses development examples only and cannot establish held-out results.
    pub fn parse() -> Result<Self> {
        let mut prepared = PathBuf::from("target/mechanistic/greater_than");
        let mut output = None;
        let mut smoke = false;
        let mut args = std::env::args().skip(1);
        while let Some(argument) = args.next() {
            match argument.as_str() {
                "--smoke" => smoke = true,
                "--prepared-dir" => {
                    prepared = PathBuf::from(args.next().ok_or("missing prepared directory")?)
                }
                "--output-dir" => {
                    output = Some(PathBuf::from(
                        args.next().ok_or("missing output directory")?,
                    ))
                }
                "--help" => {
                    println!(
                        "greater_than_train [--smoke] [--prepared-dir DIRECTORY] [--output-dir NEW_DIRECTORY]"
                    );
                    std::process::exit(0);
                }
                _ => return Err(format!("unknown argument {argument}").into()),
            }
        }
        let output = output.unwrap_or_else(|| {
            PathBuf::from(if smoke {
                "target/mechanistic/greater_than_training/smoke"
            } else {
                "target/mechanistic/greater_than_training/full"
            })
        });
        if let Some(parent) = output.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::create_dir(&output)?; // Results are create-only; keep earlier runs intact.
        Ok(Self {
            prepared,
            output,
            smoke,
        })
    }
}

#[derive(Deserialize)]
struct Fixture {
    entries: Vec<Prompt>,
}

#[derive(Deserialize)]
struct Prompt {
    clean_ids: Vec<i32>,
    corrupt_ids: Vec<i32>,
    split: String,
    year: usize,
    readout_position: usize,
    start_suffix_position: usize,
}

pub struct Study {
    prepared: PathBuf,
    fixture: Fixture,
    protocol: Value,
    numeric_ids: Vec<usize>,
    source_hashes: BTreeMap<String, String>,
    executable_sha256: String,
}

#[derive(Clone)]
pub struct Cached {
    indices: Vec<usize>,
    features: Tensor<Cpu, 3>,
    teacher: Tensor<Cpu, 2>,
    clean: Tensor<Cpu, 2>,
    corrupt: Tensor<Cpu, 2>,
}

impl Cached {
    /// Return the number of prompt pairs retained in this split cache.
    pub fn len(&self) -> usize {
        self.indices.len()
    }
    /// Map cache-local batch indices back to the immutable fixture row indices.
    pub fn source_indices(&self, indices: &[usize]) -> Vec<usize> {
        indices.iter().map(|&i| self.indices[i]).collect()
    }
    /// Select cached clean MLP inputs as `[batch, 1, 768]` F32 features.
    pub fn features(&self, indices: &[usize]) -> Tensor<Cpu, 3> {
        self.features.clone().select(0, selection(indices))
    }
    /// Select oracle-patched full-vocabulary logits as `[batch, 50257]` F32 targets.
    pub fn teacher(&self, indices: &[usize]) -> Tensor<Cpu, 2> {
        self.teacher.clone().select(0, selection(indices))
    }
}

/// Convert in-range cache row indices to a rank-one Flex integer selection tensor.
fn selection(indices: &[usize]) -> Tensor<Cpu, 1, Int> {
    Tensor::from_data(
        TensorData::new(indices.iter().map(|&i| i as i32).collect(), [indices.len()]),
        &Default::default(),
    )
}

impl Study {
    /// Verify the fixed fixture and checkpoint hashes and preserve this run's protocol.
    ///
    /// Require 97 examples per split and the declared token alignment before
    /// loading weights; record source and executable identities separately.
    pub fn load(options: &Options) -> Result<Self> {
        let protocol: Value = serde_json::from_slice(include_bytes!("protocol.json"))?;
        let bytes = fs::read(options.prepared.join("fixture.json"))?;
        if hash(&bytes)
            != protocol["fixture"]["sha256"]
                .as_str()
                .ok_or("missing fixture hash")?
        {
            return Err("prepared Greater-Than fixture differs from frozen protocol".into());
        }
        for name in ["model.safetensors", "config.json", "tokenizer.json"] {
            if file_hash(&options.prepared.join("model").join(name))?
                != protocol["backbone"]["artifact_sha256"][name]
                    .as_str()
                    .ok_or("missing checkpoint hash")?
            {
                return Err(format!("checkpoint hash mismatch: {name}").into());
            }
        }
        let fixture: Fixture = serde_json::from_slice(&bytes)?;
        for split in ["development", "held_out"] {
            if fixture
                .entries
                .iter()
                .filter(|entry| entry.split == split)
                .count()
                != 97
            {
                return Err("expected 97 prompts per frozen split".into());
            }
        }
        for entry in &fixture.entries {
            if entry.clean_ids.len() != 13
                || entry.corrupt_ids.len() != 13
                || entry.readout_position != 12
                || entry.start_suffix_position != 8
            {
                return Err("fixture token alignment differs from protocol".into());
            }
        }
        let numeric_ids =
            serde_json::from_value(protocol["evaluation"]["numeric_token_ids"].clone())?;
        let source_hashes = source_inventory()?;
        fs::write(
            options.output.join("protocol.json"),
            include_bytes!("protocol.json"),
        )?;
        Ok(Self {
            prepared: options.prepared.clone(),
            fixture,
            protocol,
            numeric_ids,
            source_hashes,
            executable_sha256: file_hash(&std::env::current_exe()?)?,
        })
    }

    /// Strictly load the pinned 768-wide, twelve-block GPT-2 small checkpoint.
    ///
    /// Use exact GELU from the author's graph rather than the config's GELU
    /// approximation; this explicit graph translation leaves checkpoint values intact.
    pub fn load_model(&self) -> Result<Gpt2ForCausalLm<Cpu>> {
        let mut config: Gpt2Config =
            serde_json::from_slice(&fs::read(self.prepared.join("model/config.json"))?)?;
        config.activation_function = Gpt2Activation::Gelu;
        if config.n_embd != 768 || config.n_layer != 12 {
            return Err("expected the pinned GPT-2 small graph".into());
        }
        let mut model = Gpt2ForCausalLm::init(config, &Default::default())?;
        model.load_checkpoint(self.prepared.join("model"), &LoadOptions::default())?;
        Ok(model)
    }

    /// Gather selected clean or corrupted fixture rows into `[batch, 13]` token IDs.
    ///
    /// The caller supplies valid row indices. These unpadded sequences use
    /// causal attention and ordinary positions on the requested backend.
    pub fn input<B: Backend>(&self, indices: &[usize], corrupt: bool) -> CausalLmInput<B> {
        let ids = indices
            .iter()
            .flat_map(|&index| {
                let row = &self.fixture.entries[index];
                if corrupt {
                    row.corrupt_ids.clone()
                } else {
                    row.clean_ids.clone()
                }
            })
            .collect::<Vec<_>>();
        CausalLmInput {
            token_ids: Tensor::<B, 2, Int>::from_data(
                TensorData::new(ids, [indices.len(), 13]),
                &Default::default(),
            ),
            token_mask: None,
            position_ids: None,
        }
    }

    /// Cache one split's clean features and clean, corrupted, and oracle-patched logits.
    ///
    /// The oracle inserts clean MLP 10 output at token twelve into a corrupted
    /// forward. The student instead receives clean MLP input at that point; cached
    /// features and targets are detached on the non-autodiff CPU backend.
    pub fn prepare(
        &self,
        model: &Gpt2ForCausalLm<Cpu>,
        split: &str,
        smoke: bool,
    ) -> Result<Cached> {
        let indices = self
            .fixture
            .entries
            .iter()
            .enumerate()
            .filter(|(_, row)| row.split == split)
            .map(|(index, _)| index)
            .take(if smoke { 8 } else { 97 })
            .collect::<Vec<_>>();
        let mut features = Vec::new();
        let mut clean = Vec::new();
        let mut corrupt = Vec::new();
        let mut teacher = Vec::new();
        for batch in indices.chunks(8) {
            let mut observe = Hooks::new();
            let clean_input = observe.capture(Layer::decoder(10).mlp_input().token(12));
            let clean_output = observe.capture(Layer::decoder(10).mlp_output().token(12));
            clean.push(
                model
                    .forward_with_hooks(
                        self.input(batch, false),
                        &ForwardOptions::default(),
                        &mut observe,
                    )?
                    .logits
                    .squeeze_dim(1),
            );
            features.push(observe.get(&clean_input)?);
            let replacement = observe.get(&clean_output)?;
            corrupt.push(
                model
                    .forward(self.input(batch, true), &ForwardOptions::default())?
                    .logits
                    .squeeze_dim(1),
            );
            let mut patch = Hooks::new();
            patch.edit(Layer::decoder(10).mlp_output().token(12), |_| {
                replacement.clone()
            });
            teacher.push(
                model
                    .forward_with_hooks(
                        self.input(batch, true),
                        &ForwardOptions::default(),
                        &mut patch,
                    )?
                    .logits
                    .squeeze_dim(1),
            );
        }
        eprintln!(
            "Cached {} {split} clean-source features and oracle targets",
            indices.len()
        );
        Ok(Cached {
            indices,
            features: Tensor::cat(features, 0),
            teacher: Tensor::cat(teacher, 0),
            clean: Tensor::cat(clean, 0),
            corrupt: Tensor::cat(corrupt, 0),
        })
    }

    /// Compare five conditions using F64 probabilities over the full 50,257-token vocabulary.
    ///
    /// Teacher KL is `KL(oracle || condition)`. Numeric metrics retain the
    /// original clean year cutoff, and recovery is absent for a negligible
    /// clean/corrupt gap. Input logits must match the selected cache batch.
    pub fn metrics(
        &self,
        cache: &Cached,
        indices: &[usize],
        initial: Tensor<Cpu, 2>,
        trained: Tensor<Cpu, 2>,
    ) -> Result<Vec<Row>> {
        let teacher = cache.teacher(indices).to_data().to_vec::<f32>()?;
        let clean = cache
            .clean
            .clone()
            .select(0, selection(indices))
            .to_data()
            .to_vec::<f32>()?;
        let corrupt = cache
            .corrupt
            .clone()
            .select(0, selection(indices))
            .to_data()
            .to_vec::<f32>()?;
        let initial = initial.to_data().to_vec::<f32>()?;
        let trained = trained.to_data().to_vec::<f32>()?;
        let mut rows = Vec::new();
        for (batch_index, &local) in indices.iter().enumerate() {
            let index = cache.indices[local];
            let prompt = &self.fixture.entries[index];
            let slice = batch_index * 50_257..(batch_index + 1) * 50_257;
            let teacher_log_probs = log_probabilities(&teacher[slice.clone()])?;
            let clean_pd = numeric_metrics(
                &log_probabilities(&clean[slice.clone()])?,
                &self.numeric_ids,
                prompt.year % 100,
            )
            .0;
            let corrupt_pd = numeric_metrics(
                &log_probabilities(&corrupt[slice.clone()])?,
                &self.numeric_ids,
                prompt.year % 100,
            )
            .0;
            let gap = clean_pd - corrupt_pd;
            for (condition, values) in [
                ("clean", &clean),
                ("corrupt", &corrupt),
                ("oracle", &teacher),
                ("initial_student", &initial),
                ("trained_student", &trained),
            ] {
                let log_probs = log_probabilities(&values[slice.clone()])?;
                let teacher_kl = teacher_log_probs
                    .iter()
                    .zip(&log_probs)
                    .map(|(teacher, student)| teacher.exp() * (teacher - student))
                    .sum::<f64>();
                let (probability_difference, cutoff_sharpness, numeric_mass, numeric_probabilities) =
                    numeric_metrics(&log_probs, &self.numeric_ids, prompt.year % 100);
                rows.push(Row {
                    index,
                    split: prompt.split.clone(),
                    suffix: prompt.year % 100,
                    condition: condition.into(),
                    teacher_kl,
                    probability_difference,
                    cutoff_sharpness,
                    numeric_mass,
                    clean_gap_recovery: if gap.abs() > 1e-6 {
                        Some((probability_difference - corrupt_pd) / gap)
                    } else {
                        None
                    },
                    numeric_probabilities,
                });
            }
        }
        Ok(rows)
    }

    /// Write raw rows, descriptive split summaries, artifact hashes, and training evidence.
    ///
    /// Record source drift explicitly and keep engineering checks separate from
    /// scientific effect sizes; the fixed training endpoint is not selected by results.
    pub fn finish(&self, options: &Options, rows: Vec<Row>, training: Value) -> Result<()> {
        let mut groups: BTreeMap<String, Vec<&Row>> = BTreeMap::new();
        for row in &rows {
            groups
                .entry(format!("{}/{}", row.split, row.condition))
                .or_default()
                .push(row);
        }
        let summary = groups.iter().map(|(name, rows)| (name.clone(), json!({
            "teacher_kl": statistics(rows.iter().map(|row| row.teacher_kl).collect()),
            "probability_difference": statistics(rows.iter().map(|row| row.probability_difference).collect()),
            "cutoff_sharpness": statistics(rows.iter().map(|row| row.cutoff_sharpness).collect()),
            "numeric_mass": statistics(rows.iter().map(|row| row.numeric_mass).collect()),
            "clean_gap_recovery": statistics(rows.iter().filter_map(|row| row.clean_gap_recovery).collect()),
        }))).collect::<BTreeMap<_, _>>();
        let mut artifacts = BTreeMap::new();
        for entry in fs::read_dir(&options.output)? {
            let path = entry?.path();
            if path.is_file() {
                artifacts.insert(
                    path.file_name().unwrap().to_string_lossy().to_string(),
                    json!({"sha256": file_hash(&path)?, "bytes": fs::metadata(&path)?.len()}),
                );
            }
        }
        let result = json!({
            "schema_version": 1, "protocol": self.protocol, "protocol_sha256": hash(include_bytes!("protocol.json")),
            "engineering_checks_passed": true, "smoke_only": options.smoke,
            "training": training, "summary": summary, "rows": rows,
            "source_sha256": self.source_hashes, "source_unchanged_during_execution": self.source_hashes == source_inventory()?,
            "executable_sha256": self.executable_sha256, "command": std::env::args().collect::<Vec<_>>(),
            "generated_artifacts": artifacts,
        });
        fs::write(
            options.output.join("result.json"),
            serde_json::to_vec_pretty(&result)?,
        )?;
        println!("{}", serde_json::to_string_pretty(&result["summary"])?);
        println!("Results: {}", options.output.join("result.json").display());
        Ok(())
    }
}

#[derive(Serialize)]
pub struct Row {
    index: usize,
    split: String,
    suffix: usize,
    condition: String,
    teacher_kl: f64,
    probability_difference: f64,
    cutoff_sharpness: f64,
    numeric_mass: f64,
    clean_gap_recovery: Option<f64>,
    numeric_probabilities: Vec<f64>,
}

/// Compute stable F64 log-softmax for exactly 50,257 finite F32 logits.
///
/// Subtract the row maximum before exponentiation; numeric token subsets
/// must use this full-vocabulary denominator rather than renormalizing themselves.
fn log_probabilities(values: &[f32]) -> Result<Vec<f64>> {
    if values.len() != 50_257 || values.iter().any(|value| !value.is_finite()) {
        return Err("expected 50257 finite F32 logits".into());
    }
    let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let log_sum = values
        .iter()
        .map(|&value| (value as f64 - maximum).exp())
        .sum::<f64>()
        .ln();
    Ok(values
        .iter()
        .map(|&value| value as f64 - maximum - log_sum)
        .collect())
}

/// Return signed greater-than preference, local cutoff sharpness, numeric mass, and curve.
///
/// The protocol supplies 100 numeric token IDs and an original clean suffix
/// in 02..98. Difference is upper-tail mass minus mass at or below the cutoff;
/// sharpness is `p[y+1] - p[y-1]` without numeric-subset renormalization.
fn numeric_metrics(log_probs: &[f64], ids: &[usize], suffix: usize) -> (f64, f64, f64, Vec<f64>) {
    let probabilities = ids
        .iter()
        .map(|&id| log_probs[id].exp())
        .collect::<Vec<_>>();
    let difference = probabilities[suffix + 1..].iter().sum::<f64>()
        - probabilities[..=suffix].iter().sum::<f64>();
    (
        difference,
        probabilities[suffix + 1] - probabilities[suffix - 1],
        probabilities.iter().sum(),
        probabilities,
    )
}

/// Compute count, mean, and sample standard deviation without treating prompts as iid.
///
/// Empty samples have null statistics; a singleton reports zero spread.
fn statistics(values: Vec<f64>) -> Value {
    if values.is_empty() {
        return json!({"count": 0, "mean": null, "sample_sd": null});
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let sd = if values.len() > 1 {
        (values
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / (values.len() - 1) as f64)
            .sqrt()
    } else {
        0.0
    };
    json!({"count": values.len(), "mean": mean, "sample_sd": sd})
}

/// Sort row indices by a fixed SHA-256 key to avoid backend-dependent shuffling.
pub fn training_order(count: usize) -> Vec<usize> {
    let mut indices = (0..count).collect::<Vec<_>>();
    indices.sort_by_key(|index| {
        hash(format!("greater-than-learned-replacement-v1/{index}").as_bytes())
    });
    indices
}

/// Require finite, nonzero F32 gradients on the selected student matrix.
///
/// The reported Euclidean norm proves a parameter receives signal through
/// the frozen suffix; it is measured before the optimizer's per-tensor clipping.
pub fn gradient_norm(gradients: &GradientsParams, id: ParamId) -> f32 {
    let values = gradients
        .get::<Cpu, 2>(id)
        .expect("student must receive gradients")
        .to_data()
        .to_vec::<f32>()
        .expect("F32 gradient");
    assert!(values.iter().all(|value| value.is_finite()));
    let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
    assert!(norm > 0.0);
    norm
}

/// Hash registered float layouts and bytes, optionally asserting all parameters are frozen.
pub fn fingerprint<B: Backend>(model: &impl Module<B>, frozen: bool) -> String {
    struct Visitor {
        hash: Sha256,
        frozen: bool,
    }
    impl<B: Backend> ModuleVisitor<B> for Visitor {
        /// Check the requested freeze invariant and hash one tensor's shape, dtype, and raw values.
        fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
            let value = param.val();
            if self.frozen {
                assert!(!value.is_require_grad());
            }
            let data = value.to_data();
            self.hash
                .update(format!("{:?}:{:?}", data.shape, data.dtype).as_bytes());
            self.hash.update(&*data.bytes);
        }
    }
    let mut visitor = Visitor {
        hash: Sha256::new(),
        frozen,
    };
    model.visit(&mut visitor);
    format!("{:x}", visitor.hash.finalize())
}

/// Hash the declared graph, protocol, fixture, and lockfile paths for this run.
///
/// This explicit dependency inventory complements the executable hash; it
/// is not a claim that every repository file participates in the experiment.
fn source_inventory() -> Result<BTreeMap<String, String>> {
    [
        "examples/greater_than_train.rs",
        "examples/mechanistic/greater_than_training/support.rs",
        "examples/mechanistic/greater_than_training/protocol.json",
        "examples/mechanistic/greater_than/fixture.json",
        "src/models/gpt2.rs",
        "src/models/gpt2/model.rs",
        "src/models/gpt2/checkpoint.rs",
        "src/intervention.rs",
        "src/training.rs",
        "Cargo.lock",
    ]
    .into_iter()
    .map(|name| {
        Ok((
            name.into(),
            file_hash(&PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(name))?,
        ))
    })
    .collect()
}

/// Return SHA-256 for bytes, including deterministic training-order keys.
fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// Stream SHA-256 with a 1 MiB buffer to bound memory for the large backbone file.
fn file_hash(path: &std::path::Path) -> Result<String> {
    let mut stream = fs::File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let count = stream.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(format!("{:x}", digest.finalize()))
}
