//! Artifact verification, fixed protocol, and numerical/checkpoint controls.
//!
//! # Learning a residual decoder
//!
//! A lens maps h to `readout(LayerNorm(h + Wh + b))`. Zero W and b initialize the
//! raw lens. Residuals `[tokens=32, hidden=768]` and teacher logits
//! `[tokens=32, vocabulary=50304]` are cached without gradients; the frozen final
//! normalization/readout preserve input gradients while only W and b learn.
//! Native SGD minimizes `KL(teacher || lens)`, with one global gradient norm
//! across weight and bias before Nesterov momentum and weight decay.
//!
//! # Bounded protocol and numerical evidence
//!
//! The causal backbone sees no padding or decode cache. Twelve disjoint documents
//! train for 100 steps and four evaluate, a smaller adaptation of paper training.
//! Graph arithmetic stays F32; host F64 log-sum-exp computes held-out metrics.
//! Independent first-step exports, parameter fingerprints, and native momentum
//! record round trips test numerical and training compatibility without fitting
//! tolerances or selecting checkpoints from held-out outcomes.

use burn::{
    backend::{Autodiff, Flex},
    module::{Initializer, Module, ModuleVisitor, Param},
    nn::{LayerNorm, Linear, LinearConfig},
    optim::{
        GradientsParams, Optimizer, Sgd, SgdConfig, adaptor::OptimizerAdaptor,
        decay::WeightDecayConfig, momentum::MomentumConfig,
    },
    record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder},
    tensor::{Tensor, TensorData, backend::Backend},
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::Read,
    path::{Path, PathBuf},
};
use transformers::{
    CausalLmInput, ForwardOptions, LogitsSelection,
    checkpoint::LoadOptions,
    intervention::{Hooks, Layer, Stack},
    models::gpt_neox::{GptNeoXConfig, GptNeoXForCausalLm},
};

/// CPU inference backend for frozen-data evaluation.
pub type Cpu = Flex<f32, i32>;
/// Autodifferentiable CPU backend for fitting affine residual corrections.
pub type B = Autodiff<Cpu>;
/// Experiment result carrying contextual artifact, gradient, and checkpoint errors.
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
/// Native SGD adaptor owning momentum state for a trainable lens module.
pub type SgdOptimizer = OptimizerAdaptor<Sgd<Cpu>, Lens<B>, B>;

/// Pinned training settings and separate input/output artifact directories.
pub struct Settings {
    /// Directory containing the frozen model and reference preparation artifacts.
    pub base: PathBuf,
    /// Directory receiving fresh training fixtures, native records, and results.
    pub output: PathBuf,
    /// Residual stages to fit: zero denotes embeddings and i denotes post-block i - 1.
    pub stages: Vec<usize>,
    /// Number of leading cached documents used for fitting; later documents are held out.
    pub train_records: usize,
    /// Number of optimizer updates in the fixed training schedule.
    pub steps: usize,
    /// Embedded protocol containing pinned source and base-artifact identities.
    protocol: Value,
}

impl Settings {
    /// Read the embedded protocol and optional artifact/output directories.
    ///
    /// Use stages 0, 6, and 11, twelve training documents, and 100 updates;
    /// reject the same base and output path to protect the reference directory.
    pub fn read() -> Result<Self> {
        let protocol: Value = serde_json::from_str(include_str!("manifest.json"))?;
        let base = std::env::args()
            .nth(1)
            .map(PathBuf::from)
            .unwrap_or_else(|| "target/mechanistic/tuned_lens".into());
        let output = std::env::args()
            .nth(2)
            .map(PathBuf::from)
            .unwrap_or_else(|| "target/mechanistic/tuned_lens_training".into());
        if base == output {
            return Err(
                "training output must differ from the frozen v0.2 artifact directory".into(),
            );
        }
        if !base.join("preparation.json").is_file() {
            return Err(format!(
                "{}: no prepared artifacts; run python3 examples/mechanistic/tuned_lens/prepare.py, \
                 or pass the prepared directory as the first argument",
                base.display()
            )
            .into());
        }
        fs::create_dir_all(&output)?;
        Ok(Self {
            base,
            output,
            stages: vec![0, 6, 11],
            train_records: 12,
            steps: 100,
            protocol,
        })
    }

    /// Return `0.02 * (1 - step / steps)` for zero-based steps before the endpoint.
    pub fn learning_rate(&self, step: usize) -> f64 {
        0.02 * (1.0 - step as f64 / self.steps as f64)
    }
}

/// Trainable residual affine correction followed by the frozen final decoder.
#[derive(Module, Debug)]
pub struct Lens<B: Backend> {
    /// Width-preserving correction `Wh + b`, initialized to zero and added to the input.
    pub translator: Linear<B>,
    /// Frozen final LayerNorm that retains input-gradient propagation.
    final_norm: LayerNorm<B>,
    /// Frozen vocabulary projection that retains input-gradient propagation.
    unembedding: Linear<B>,
}

impl<T: Backend> Lens<T> {
    /// Initialize `h + Wh + b` as identity and register a frozen final decoder.
    ///
    /// Only the correction has trainable parameters; the LayerNorm and
    /// unembedding remain in the graph so their input gradients reach it.
    pub fn new(backbone: &GptNeoXForCausalLm<T>) -> Self {
        let width = backbone.embed_out.weight.val().dims()[0];
        Self {
            translator: LinearConfig::new(width, width)
                .with_initializer(Initializer::Zeros)
                .init(&Default::default()),
            final_norm: backbone.final_layer_norm.clone().no_grad(),
            unembedding: backbone.embed_out.clone().no_grad(),
        }
    }
}

impl<T: Backend> Lens<T> {
    /// Decode `[tokens, hidden]` residuals into `[tokens, vocabulary]` logits.
    ///
    /// Apply the residual affine correction before the frozen final LayerNorm
    /// and unembedding, matching the raw lens when the correction is zero.
    pub fn forward(&self, hidden: Tensor<T, 2>) -> Tensor<T, 2> {
        let translated = hidden.clone() + self.translator.forward(hidden);
        self.unembedding
            .forward(self.final_norm.forward(translated))
    }

    /// Hash the final LayerNorm and readout separately to detect accidental updates.
    pub fn frozen_fingerprint(&self) -> (String, String) {
        (
            fingerprint(&self.final_norm),
            fingerprint(&self.unembedding),
        )
    }
}

/// Construct native SGD with Nesterov momentum 0.9 and weight decay 0.001.
///
/// Global gradient clipping is applied separately before this optimizer step.
pub fn optimizer() -> SgdOptimizer {
    SgdConfig::new()
        .with_momentum(Some(
            MomentumConfig::new()
                .with_momentum(0.9)
                .with_dampening(0.0)
                .with_nesterov(true),
        ))
        .with_weight_decay(Some(WeightDecayConfig::new(0.001)))
        .init()
}

/// One prepared document with exact source identity and token IDs.
#[derive(Deserialize, Serialize)]
pub struct Record {
    /// Stable source-document index defining the frozen fitting/validation split.
    pub record_index: usize,
    /// BOS-prefixed document token IDs truncated to the protocol's 32 positions.
    pub input_ids: Vec<i32>,
}

/// Verify protocol hashes and load a frozen backbone with sixteen fixed documents.
///
/// Records 4..15 train and records 16..19 validate; each receives BOS and is
/// truncated to 32 tokens. Save their exact IDs and the compiled-source hashes.
pub fn load(settings: &Settings) -> Result<(GptNeoXForCausalLm<B>, Vec<Record>, Value)> {
    let mut artifacts = serde_json::Map::new();
    for (name, expected) in settings.protocol["base_artifacts"].as_object().unwrap() {
        let path = settings.base.join(name);
        let actual = json!({"bytes": fs::metadata(&path)?.len(), "sha256": file_hash(&path)?});
        if &actual != expected {
            return Err(
                format!("{name}: artifact differs from the pinned training protocol").into(),
            );
        }
        artifacts.insert(name.clone(), actual);
    }
    let config: GptNeoXConfig =
        serde_json::from_slice(&fs::read(settings.base.join("model/config.json"))?)?;
    let mut backbone = GptNeoXForCausalLm::<B>::init(config, &Default::default())?;
    backbone.load_checkpoint(settings.base.join("model"), &LoadOptions::default())?;
    let backbone = backbone.no_grad();
    let tokenizer = tokenizers::Tokenizer::from_file(settings.base.join("model/tokenizer.json"))
        .map_err(|e| std::io::Error::other(e.to_string()))?;
    let documents = fs::read_to_string(settings.base.join("pile_test_first25.jsonl"))?;
    let mut records = Vec::new();
    for (record_index, line) in documents.lines().enumerate().skip(4).take(16) {
        let document: Value = serde_json::from_str(line)?;
        let encoding = tokenizer
            .encode(document["text"].as_str().ok_or("missing text")?, false)
            .map_err(|e| std::io::Error::other(e.to_string()))?;
        let input_ids = std::iter::once(0)
            .chain(encoding.get_ids().iter().map(|&id| id as i32))
            .take(32)
            .collect::<Vec<_>>();
        if input_ids.len() != 32 {
            return Err(
                format!("record {record_index} has fewer than the pinned 32 tokens").into(),
            );
        }
        records.push(Record {
            record_index,
            input_ids,
        });
    }
    if records.len() != 16 {
        return Err("expected 12 train and 4 validation documents".into());
    }
    write_json(&settings.output.join("fixture.json"), &records)?;
    let provenance = json!({"artifacts": artifacts,
        "manifest_sha256": hash(include_bytes!("manifest.json")),
        "fixture_sha256": file_hash(&settings.output.join("fixture.json"))?,
        "training_source_sha256": hash(include_bytes!("../../tuned_lens_train.rs")),
        "study_source_sha256": hash(include_bytes!("study.rs")),
    });
    Ok((backbone, records, provenance))
}

/// Detached residuals, teacher logits, and token IDs for one fixed document.
pub struct CachedRecord {
    /// Captured `[tokens, hidden]` residual tensors in the requested stage order.
    pub hidden: Vec<Tensor<B, 2>>,
    /// Final-model teacher logits with axes `[tokens, vocabulary]`.
    pub teacher: Tensor<B, 2>,
    /// Input token IDs; each next token supplies the previous position's CE target.
    pub tokens: Vec<i32>,
}

/// Cache detached `[32, 768]` residuals and `[32, 50304]` teacher logits per document.
///
/// Stage zero selects embeddings; stage i selects the residual after block
/// i - 1. Inputs have no padding, and the backbone retains causal attention.
/// Caching reduces compute because neither backbone nor data changes during fitting.
pub fn capture(
    backbone: &GptNeoXForCausalLm<B>,
    records: &[Record],
    stages: &[usize],
) -> Result<Vec<CachedRecord>> {
    let options = ForwardOptions {
        logits: LogitsSelection::All,
        ..Default::default()
    };
    let mut cache = Vec::new();
    for record in records {
        let mut hooks = Hooks::new();
        let handles = stages
            .iter()
            .map(|&stage| {
                if stage == 0 {
                    hooks.capture(Stack::Decoder.embedding())
                } else {
                    hooks.capture(Layer::decoder(stage - 1).residual_post())
                }
            })
            .collect::<Vec<_>>();
        let input = CausalLmInput {
            token_ids: Tensor::from_data(
                TensorData::new(record.input_ids.clone(), [1, 32]),
                &Default::default(),
            ),
            token_mask: None,
            position_ids: None,
        };
        let output = backbone.forward_with_hooks(input, &options, &mut hooks)?;
        let hidden = handles
            .iter()
            .map(|handle| hooks.get(handle).map(|tensor| tensor.reshape([32, 768])))
            .collect::<transformers::Result<Vec<_>>>()?;
        cache.push(CachedRecord {
            hidden,
            teacher: output.logits.reshape([32, 50304]).detach(),
            tokens: record.input_ids.clone(),
        });
    }
    Ok(cache)
}

/// Global clipping across both affine tensors, before Burn's weight decay and
/// Nesterov update. Burn's built-in Norm option instead clips each tensor alone.
pub fn gradients(
    grads: <B as burn::tensor::backend::AutodiffBackend>::Gradients,
    lens: &Lens<B>,
) -> Result<(GradientsParams, f64)> {
    let mut grads = GradientsParams::from_grads(grads, lens);
    if grads.len() != 2 {
        return Err(format!(
            "expected exactly two translator gradients, found {}",
            grads.len()
        )
        .into());
    }
    let bias_id = lens.translator.bias.as_ref().unwrap().id;
    let weight = grads
        .remove::<Cpu, 2>(lens.translator.weight.id)
        .ok_or("missing translator weight gradient")?;
    let bias = grads
        .remove::<Cpu, 1>(bias_id)
        .ok_or("missing translator bias gradient")?;
    let mut squared = 0.0_f64;
    for value in weight
        .to_data()
        .iter::<f32>()
        .chain(bias.to_data().iter::<f32>())
    {
        squared += f64::from(value).powi(2);
    }
    let norm = squared.sqrt();
    if !norm.is_finite() || norm == 0.0 {
        return Err("translator gradients must be finite and nonzero".into());
    }
    let scale = (1.0 / (norm + 1e-6)).min(1.0);
    grads.register(lens.translator.weight.id, weight * scale);
    grads.register(bias_id, bias * scale);
    Ok((grads, norm))
}

/// Mean evaluation losses and their distinct token populations.
#[derive(Clone, Copy, Serialize)]
pub struct Metrics {
    /// Mean forward KL from final-model teacher to lens predictions, in nats per position.
    pub kl: f64,
    /// Mean next-token negative log likelihood, in nats per valid target.
    pub ce: f64,
    /// Number of prediction positions included in forward KL.
    pub kl_tokens: usize,
    /// Number of shifted targets included in cross entropy.
    pub ce_tokens: usize,
}

/// Measure mean `KL(teacher || lens)` and next-token CE using host F64 arithmetic.
///
/// The F32 decoder runs on the inference backend. KL counts all 32 positions
/// per record; CE counts 31 shifted targets, with no target after the final token.
/// The caller supplies a nonempty split and an existing residual slot.
pub fn evaluate(lens: &Lens<B>, data: &[CachedRecord], slot: usize) -> Result<Metrics> {
    use burn::module::AutodiffModule;
    let lens = lens.valid();
    let mut kl = 0.0;
    let mut ce = 0.0;
    for batch in data {
        let logits = lens.forward(batch.hidden[slot].clone().inner());
        let values = logits.into_data().to_vec::<f32>()?;
        let teacher = batch.teacher.to_data().to_vec::<f32>()?;
        for token in 0..32 {
            let row = &values[token * 50304..(token + 1) * 50304];
            let target = &teacher[token * 50304..(token + 1) * 50304];
            let normalizer = logsumexp(row);
            let teacher_normalizer = logsumexp(target);
            for (&s, &t) in row.iter().zip(target) {
                let logp = f64::from(t) - teacher_normalizer;
                kl += logp.exp() * (logp - (f64::from(s) - normalizer));
            }
            if token < 31 {
                ce += normalizer - f64::from(row[batch.tokens[token + 1] as usize]);
            }
        }
    }
    Ok(Metrics {
        kl: kl / (data.len() * 32) as f64,
        ce: ce / (data.len() * 31) as f64,
        kl_tokens: data.len() * 32,
        ce_tokens: data.len() * 31,
    })
}

/// Compute `max(x) + ln(sum(exp(x - max(x))))` in F64 from a finite F32 row.
fn logsumexp(row: &[f32]) -> f64 {
    let maximum = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    maximum
        + row
            .iter()
            .map(|&value| (f64::from(value) - maximum).exp())
            .sum::<f64>()
            .ln()
}

/// Round-trip affine parameters and SGD momentum through native Burn recorders.
///
/// Restore the translator into a fresh registered module so parameter IDs and
/// optimizer state align; retain the same frozen decoder without serializing it.
pub fn checkpoint(
    settings: &Settings,
    stage: usize,
    lens: &Lens<B>,
    optim: &SgdOptimizer,
) -> Result<(Lens<B>, SgdOptimizer)> {
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    let model_path = settings
        .output
        .join(format!("stage_{stage}_half_translator"));
    let optimizer_path = settings
        .output
        .join(format!("stage_{stage}_half_optimizer"));
    recorder.record(lens.translator.clone().into_record(), model_path.clone())?;
    recorder.record(optim.to_record(), optimizer_path.clone())?;
    let mut restored = lens.clone();
    let [input, output] = lens.translator.weight.val().dims();
    restored.translator = LinearConfig::new(input, output)
        .with_initializer(Initializer::Zeros)
        .init::<B>(&Default::default())
        .load_record(recorder.load(model_path, &Default::default())?);
    let resumed = optimizer().load_record(recorder.load(optimizer_path, &Default::default())?);
    Ok((restored, resumed))
}

/// Apply the same KL batch and SGD step to uninterrupted and restored states.
///
/// Return the largest absolute difference across translator weight and bias;
/// this validates optimizer momentum continuation as well as parameter loading.
pub fn compare_next_step(
    mut original: Lens<B>,
    mut original_optim: SgdOptimizer,
    mut restored: Lens<B>,
    mut restored_optim: SgdOptimizer,
    batch: &CachedRecord,
    slot: usize,
    lr: f64,
) -> Result<f32> {
    for (lens, optim) in [
        (&mut original, &mut original_optim),
        (&mut restored, &mut restored_optim),
    ] {
        let loss = transformers::training::kl_divergence(
            lens.forward(batch.hidden[slot].clone()),
            batch.teacher.clone(),
        )?;
        let (grads, _) = gradients(loss.backward(), lens)?;
        *lens = optim.step(lr, lens.clone(), grads);
    }
    let weight = max_difference(
        original.translator.weight.val(),
        restored.translator.weight.val(),
    )?;
    let bias = max_difference(
        original.translator.bias.unwrap().val(),
        restored.translator.bias.unwrap().val(),
    )?;
    Ok(weight.max(bias))
}

/// Save final affine and momentum records without copying the frozen backbone.
pub fn save_final(
    settings: &Settings,
    stage: usize,
    lens: &Lens<B>,
    optim: &SgdOptimizer,
) -> Result<()> {
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    recorder.record(
        lens.translator.clone().into_record(),
        settings.output.join(format!("stage_{stage}_translator")),
    )?;
    recorder.record(
        optim.to_record(),
        settings.output.join(format!("stage_{stage}_optimizer")),
    )?;
    Ok(())
}

/// Read the first element of the expected one-element F32 loss tensor as F64.
pub fn scalar(tensor: Tensor<B, 1>) -> Result<f64> {
    Ok(f64::from(tensor.into_data().to_vec::<f32>()?[0]))
}

/// Return the maximum absolute difference of same-shaped F32 tensors.
fn max_difference<const D: usize>(left: Tensor<B, D>, right: Tensor<B, D>) -> Result<f32> {
    Ok((left - right).abs().max().into_data().to_vec::<f32>()?[0])
}

/// Compare a newly zero-initialized lens with the backbone raw residual readout.
///
/// Both paths receive the same 32-token residual; the maximum logit error
/// checks correction placement, final normalization, and unembedding order.
pub fn identity_error(
    lens: &Lens<B>,
    backbone: &GptNeoXForCausalLm<B>,
    batch: &CachedRecord,
    slot: usize,
) -> Result<f32> {
    max_difference(
        lens.forward(batch.hidden[slot].clone()),
        backbone
            .project_hidden(batch.hidden[slot].clone().reshape([1, 32, 768]))?
            .reshape([32, 50304]),
    )
}

/// Hash each registered float tensor in traversal order, including its shape and dtype.
pub fn fingerprint<T: Backend, M: Module<T>>(model: &M) -> String {
    /// Accumulates a digest over registered parameter layouts and exact values.
    struct HashParameters(
        /// Incremental SHA-256 state in native parameter visitation order.
        Sha256,
    );
    impl<T: Backend> ModuleVisitor<T> for HashParameters {
        /// Append one parameter layout and raw values without including unstable parameter IDs.
        fn visit_float<const D: usize>(&mut self, parameter: &Param<Tensor<T, D>>) {
            let data = parameter.val().into_data();
            self.0
                .update(format!("{:?}:{:?}", data.shape, data.dtype).as_bytes());
            self.0.update(data.as_bytes());
        }
    }
    let mut visitor = HashParameters(Sha256::new());
    model.visit(&mut visitor);
    format!("{:x}", visitor.0.finalize())
}

/// Return SHA-256 for embedded protocol or source bytes.
fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// Stream SHA-256 with a 1 MiB buffer so checkpoint size does not dictate host memory.
fn file_hash(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

/// Serialize readable JSON with a trailing newline to the experiment output path.
fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    fs::write(path, format!("{}\n", serde_json::to_string_pretty(value)?))?;
    Ok(())
}

/// Export the initial batch, frozen decoder, and globally clipped gradients as F32.
///
/// The independent Python oracle can reproduce one loss and update without
/// rerunning Pythia; metrics record `[tokens, hidden, vocabulary] = [32, 768, 50304]`.
pub fn export_gradient_oracle(
    settings: &Settings,
    stage: usize,
    lens: &Lens<B>,
    batch: &CachedRecord,
    slot: usize,
    grads: &GradientsParams,
    loss: f64,
) -> Result<()> {
    // A small independent Python reference can verify this exact initial batch
    // and gradient without running or training the whole transformer again.
    let prefix = settings.output.join(format!("stage_{stage}_first_step"));
    fs::create_dir_all(&prefix)?;
    for (name, data) in [
        ("hidden", batch.hidden[slot].to_data()),
        ("teacher", batch.teacher.to_data()),
        ("unembedding", lens.unembedding.weight.val().into_data()),
        ("norm_weight", lens.final_norm.gamma.val().into_data()),
        (
            "norm_bias",
            lens.final_norm.beta.as_ref().unwrap().val().into_data(),
        ),
        (
            "gradient_weight",
            grads
                .get::<Cpu, 2>(lens.translator.weight.id)
                .unwrap()
                .into_data(),
        ),
        (
            "gradient_bias",
            grads
                .get::<Cpu, 1>(lens.translator.bias.as_ref().unwrap().id)
                .unwrap()
                .into_data(),
        ),
    ] {
        fs::write(prefix.join(format!("{name}.f32")), data.as_bytes())?;
    }
    write_json(
        &prefix.join("metrics.json"),
        &json!({"loss":loss,"shape":[32,768,50304],"gradients":"after global norm clip1; before weight decay and optimizer"}),
    )
}

/// Write the first updated affine tensors for the independent optimizer comparison.
pub fn export_first_update(settings: &Settings, stage: usize, lens: &Lens<B>) -> Result<()> {
    let prefix = settings.output.join(format!("stage_{stage}_first_step"));
    fs::write(
        prefix.join("updated_weight.f32"),
        lens.translator.weight.val().into_data().as_bytes(),
    )?;
    fs::write(
        prefix.join("updated_bias.f32"),
        lens.translator
            .bias
            .as_ref()
            .unwrap()
            .val()
            .into_data()
            .as_bytes(),
    )?;
    Ok(())
}

/// Retain per-stage metrics and require frozen parameters, resume parity, and lower held-out KL.
///
/// A failed check returns an error after saving its measurements; the endpoint
/// and tolerances stay fixed rather than selecting a favorable checkpoint.
pub fn report(
    settings: &Settings,
    provenance: Value,
    layers: Vec<Value>,
    backbone_unchanged: bool,
    seconds: f64,
) -> Result<()> {
    let passed = backbone_unchanged
        && layers.iter().all(|item| {
            item["frozen_decoder_unchanged"] == true
                && item["initial_identity_max_abs"].as_f64().unwrap() <= 1e-6
                && item["resume_next_step_max_abs"].as_f64().unwrap() <= 1e-6
                && item["final"]["kl"].as_f64().unwrap() < item["initial"]["kl"].as_f64().unwrap()
        });
    let report = json!({"status":if passed {"passed"} else {"failed"}, "provenance":provenance,
        "protocol":settings.protocol, "backbone_unchanged":backbone_unchanged,
        "layers":layers, "wall_seconds":seconds});
    write_json(&settings.output.join("result.json"), &report)?;
    if !passed {
        return Err("training result failed a frozen-backbone, resume, or held-out KL check; result retained".into());
    }
    Ok(())
}

#[cfg(test)]
#[path = "study_tests.rs"]
mod tests;
