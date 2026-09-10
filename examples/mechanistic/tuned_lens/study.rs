//! Local artifact loading and small reporting helpers for `tuned_lens.rs`.
//!
//! # Residual decoding and precision
//!
//! Released affine corrections act on `[batch, token, hidden]` residual states
//! before the frozen final normalization and readout. Compare their full-vocabulary
//! `KL(final || candidate)` and shifted next-token cross entropy. Each document is
//! unpadded, causal, and independent; there is no decode cache to reset or reorder.
//! The graph stays F32. Pairwise vocabulary reductions limit Flex tail-mass error,
//! while host F64 aggregation keeps per-document and dataset totals distinct.
//!
//! # Artifact and evidence boundaries
//!
//! Validate checkpoint, translator, tokenizer, fixture, and independent oracle
//! identities before measurement. Parameter import explicitly transposes PyTorch
//! `[output, input]` matrices. The short fixed fixture evaluates released weights;
//! it does not train translators or reproduce the paper's full data population.

use std::{fs, io::Read, path::Path};

use burn::{
    backend::Flex,
    module::{Module, Param},
    nn::{Linear, LinearConfig},
    tensor::{Int, Tensor, TensorData, backend::Backend},
};
use safetensors::{Dtype, SafeTensors};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use transformers::{
    CausalLmInput,
    checkpoint::LoadOptions,
    models::gpt_neox::{GptNeoXConfig, GptNeoXModel},
};

pub type B = Flex<f32, i32>;
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// Owning both modules registers backbone and auxiliary parameters with Burn.
#[derive(Module, Debug)]
pub struct Experiment<B: Backend> {
    pub backbone: GptNeoXModel<B>,
    pub translators: Vec<Linear<B>>,
}

impl Experiment<B> {
    /// Verify pinned files, strictly load the backbone, and import released translators.
    ///
    /// PyTorch affine weights `[output, input]` are transposed into Burn
    /// `[input, output]` matrices. The four unpadded records must retain the
    /// protocol token bounds, vocabulary, and BOS token.
    pub fn load(root: &Path) -> Result<(Self, Fixture, serde_json::Value)> {
        let preparation: serde_json::Value =
            serde_json::from_slice(&fs::read(root.join("preparation.json"))?)?;
        if preparation["fingerprint"]["manifest_sha256"].as_str()
            != Some(&source_hash(include_bytes!("manifest.json")))
            || preparation["fingerprint"]["prepare_sha256"].as_str()
                != Some(&source_hash(include_bytes!("prepare.py")))
        {
            return Err("preparation is stale; rerun prepare.py with the current protocol".into());
        }
        for (name, path) in [
            ("model", "model/model.safetensors"),
            ("config", "model/config.json"),
            ("lens", "lens.safetensors"),
            ("fixture", "fixture.json"),
            ("tokenizer", "model/tokenizer.json"),
            ("tokenizer_config", "model/tokenizer_config.json"),
            ("special_tokens_map", "model/special_tokens_map.json"),
            ("tensor_equivalence", "tensor_equivalence.json"),
            ("pile_records", "pile_test_first25.jsonl"),
        ] {
            let expected = &preparation["outputs"][name];
            if expected["path"].as_str() != Some(path)
                || expected["bytes"].as_u64() != Some(fs::metadata(root.join(path))?.len())
                || expected["sha256"].as_str() != Some(&file_hash(&root.join(path))?)
            {
                return Err(
                    format!("{path}: prepared artifact does not match preparation.json").into(),
                );
            }
        }
        let config: GptNeoXConfig =
            serde_json::from_slice(&fs::read(root.join("model/config.json"))?)?;
        let device = Default::default();
        let mut backbone = GptNeoXModel::init(config.clone(), &device)?;
        backbone.load_checkpoint(root.join("model"), &LoadOptions::default())?;
        let lens = Tensors::load(&root.join("lens.safetensors"))?;
        let mut translators = Vec::new();
        for index in 0..config.num_hidden_layers {
            let mut translator =
                LinearConfig::new(config.hidden_size, config.hidden_size).init(&device);
            // PyTorch stores [output,input]; ordinary Burn Linear uses [input,output].
            translator.weight = Param::from_tensor(
                lens.tensor(
                    &format!("translators.{index}.weight"),
                    [config.hidden_size, config.hidden_size],
                )?
                .transpose(),
            );
            translator.bias = Some(Param::from_tensor(
                lens.tensor(&format!("translators.{index}.bias"), [config.hidden_size])?,
            ));
            translators.push(translator);
        }
        let fixture: Fixture = serde_json::from_slice(&fs::read(root.join("fixture.json"))?)?;
        if fixture.records.len() != 4
            || fixture.records.iter().enumerate().any(|(index, record)| {
                record.record_index != index
                    || record.input_ids.len() < 8
                    || record.input_ids.len() > 128
                    || record.input_ids[0] != 0
                    || record
                        .input_ids
                        .iter()
                        .any(|&id| id < 0 || id >= config.vocab_size as i64)
            })
        {
            return Err(
                "fixture must contain the four pinned records with valid bounded token sequences"
                    .into(),
            );
        }
        Ok((
            Self {
                backbone,
                translators,
            },
            fixture,
            preparation,
        ))
    }
}

/// Bind the independent oracle to the exact prepared fixture and current scripts.
pub fn load_reference(
    root: &Path,
    preparation: &serde_json::Value,
) -> Result<(serde_json::Value, Tensors)> {
    let reference: serde_json::Value =
        serde_json::from_slice(&fs::read(root.join("reference.json"))?)?;
    let provenance = &reference["provenance"];
    if provenance["preparation_sha256"].as_str()
        != Some(&file_hash(&root.join("preparation.json"))?)
        || provenance["manifest_sha256"].as_str()
            != Some(&source_hash(include_bytes!("manifest.json")))
        || provenance["reference_script_sha256"].as_str()
            != Some(&source_hash(include_bytes!("reference.py")))
        || provenance["inputs"] != preparation["outputs"]
    {
        return Err("reference provenance is stale; rerun reference.py".into());
    }
    let output = &reference["outputs"]["reference"];
    let path = root.join("reference.safetensors");
    if output["path"].as_str() != Some("reference.safetensors")
        || output["bytes"].as_u64() != Some(fs::metadata(&path)?.len())
        || output["sha256"].as_str() != Some(&file_hash(&path)?)
    {
        return Err("reference.safetensors does not match its reference report".into());
    }
    Ok((reference, Tensors::load(&path)?))
}

#[derive(Deserialize)]
pub struct Fixture {
    pub records: Vec<Record>,
}

#[derive(Deserialize)]
pub struct Record {
    pub record_index: usize,
    pub input_ids: Vec<i64>,
}

impl Record {
    /// Build unpadded `[1, tokens]` IDs with causal model attention and default positions.
    pub fn input(&self) -> CausalLmInput<B> {
        CausalLmInput {
            token_ids: Tensor::from_data(
                TensorData::new(self.input_ids.clone(), [1, self.input_ids.len()]),
                &Default::default(),
            ),
            token_mask: None,
            position_ids: None,
        }
    }
}

pub struct Tensors {
    bytes: Vec<u8>,
}

impl Tensors {
    /// Read a local safetensors container without interpreting tensor contents yet.
    pub fn load(path: &Path) -> Result<Self> {
        Ok(Self {
            bytes: fs::read(path)?,
        })
    }

    /// Read one finite F32 tensor with exactly the requested name, rank, and axes.
    ///
    /// Safetensors stores little-endian row-major values; reject shape or dtype
    /// mismatches before constructing the Flex tensor.
    pub fn tensor<const D: usize>(&self, name: &str, shape: [usize; D]) -> Result<Tensor<B, D>> {
        let tensors = SafeTensors::deserialize(&self.bytes)?;
        let view = tensors.tensor(name)?;
        if view.dtype() != Dtype::F32 || view.shape() != shape {
            return Err(format!(
                "{name}: expected F32 {shape:?}, found {:?} {:?}",
                view.dtype(),
                view.shape()
            )
            .into());
        }
        let values: Vec<_> = view
            .data()
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect();
        if values.iter().any(|value| !value.is_finite()) {
            return Err(format!("{name}: nonfinite reference or lens parameter").into());
        }
        Ok(Tensor::from_data(
            TensorData::new(values, shape),
            &Default::default(),
        ))
    }
}

/// Stream a local artifact through SHA-256 using a bounded 1 MiB host buffer.
pub fn file_hash(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

/// Hash embedded source bytes so the report identifies the compiled example.
fn source_hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// Normalize the full vocabulary without a long sequential F32 reduction.
pub fn log_probabilities(logits: Tensor<B, 3>) -> Result<Tensor<B, 3>> {
    let shifted = logits.clone() - logits.max_dim(2);
    let log_normalizer = sum_vocabulary(shifted.clone().exp())?.log();
    Ok(shifted - log_normalizer)
}

// Flex's scalar path sums a row sequentially. At 50,304 terms it loses enough
// tail mass to bias KL. Pairwise F32 reductions retain the original dtype and
// mathematical definition; the backbone and independent oracle stay unchanged.
/// Reduce `[batch, token, 50304]` to `[batch, token, 1]` with pairwise F32 sums.
///
/// Pad odd intermediate widths with additive zeros. Balanced addition limits
/// tail-mass loss compared with one long sequential reduction without changing
/// the distribution, dtype, or pinned vocabulary.
fn sum_vocabulary(mut values: Tensor<B, 3>) -> Result<Tensor<B, 3>> {
    let [batch, tokens, vocabulary] = values.dims();
    if vocabulary != 50_304 {
        return Err(
            "this study's vocabulary reduction requires the pinned 50,304-token model".into(),
        );
    }
    let mut width = vocabulary;
    while width > 1 {
        if !width.is_multiple_of(2) {
            let zero = Tensor::zeros([batch, tokens, 1], &values.device());
            values = Tensor::cat(vec![values, zero], 2);
            width += 1;
        }
        width /= 2;
        values = values
            .reshape([batch, tokens, width, 2])
            .sum_dim(3)
            .reshape([batch, tokens, width]);
    }
    Ok(values)
}

/// Per-record loss sums. Distribution arithmetic is F32; host aggregation is F64.
pub fn losses(
    candidate_log_probs: Tensor<B, 3>,
    final_log_probs: Tensor<B, 3>,
    ids: &[i64],
) -> Result<(f64, f64)> {
    let tokens = ids.len();
    let kl = sum_vocabulary(
        final_log_probs.clone().exp() * (final_log_probs - candidate_log_probs.clone()),
    )?
    .reshape([tokens]);
    // Position t predicts token t+1. The final prediction has no CE target.
    let labels = Tensor::<B, 3, Int>::from_data(
        TensorData::new(ids[1..].to_vec(), [1, tokens - 1, 1]),
        &Default::default(),
    );
    let ce = -candidate_log_probs
        .narrow(1, 0, tokens - 1)
        .gather(2, labels)
        .reshape([tokens - 1]);
    let sum = |x: Tensor<B, 1>| -> Result<f64> {
        let values = x.into_data().to_vec::<f32>()?;
        if values.iter().any(|value| !value.is_finite()) {
            return Err("nonfinite loss".into());
        }
        Ok(values.into_iter().map(f64::from).sum())
    };
    Ok((sum(kl)?, sum(ce)?))
}

/// Return the finite maximum absolute F32 logit difference, rejecting unequal shapes.
pub fn max_error(left: Tensor<B, 3>, right: Tensor<B, 3>) -> Result<f32> {
    if left.dims() != right.dims() {
        return Err("parity tensor shape mismatch".into());
    }
    let error = (left - right).abs().max().into_scalar();
    if !error.is_finite() {
        return Err("nonfinite parity error".into());
    }
    Ok(error)
}

#[derive(Clone, Default, Serialize)]
pub struct LayerMetrics {
    pub index: usize,
    pub point: String,
    pub raw_kl: f64,
    pub tuned_kl: f64,
    pub raw_ce: f64,
    pub tuned_ce: f64,
}

/// Unnormalized loss totals and controls collected by the visible experiment.
pub struct Measurements {
    pub layers: Vec<LayerMetrics>,
    pub input_tokens: usize,
    pub ce_tokens: usize,
    pub final_ce: f64,
    pub parity_error: f32,
    pub observation_error: f32,
    pub replacement_error: f32,
    pub replacement_translator: usize,
    pub wall_seconds: f64,
}

/// Normalize token totals, check frozen parity tolerances, and retain the result JSON.
///
/// KL uses every prediction position; next-token CE excludes each document
/// final position. A failed reference or identity check returns an error only
/// after writing the measured values and their artifact/source provenance.
pub fn write_report(
    root: &Path,
    mut measured: Measurements,
    reference: &serde_json::Value,
    preparation: &serde_json::Value,
) -> Result<()> {
    if reference["metrics"]["input_tokens"].as_u64() != Some(measured.input_tokens as u64)
        || reference["metrics"]["ce_tokens"].as_u64() != Some(measured.ce_tokens as u64)
    {
        return Err("reference and experiment used different loss populations".into());
    }
    measured.final_ce /= measured.ce_tokens as f64;
    let mut metric_error = 0.0_f64;
    println!(
        "Pythia-160M-v0 / released Tuned Lens: {} KL positions, {} CE targets",
        measured.input_tokens, measured.ce_tokens
    );
    println!("Stage       raw KL  tuned KL   raw CE  tuned CE  (nats/token)");
    for layer in &mut measured.layers {
        if reference["metrics"]["layers"][layer.index]["point"].as_str() != Some(&layer.point) {
            return Err("reference and experiment used different activation boundaries".into());
        }
        layer.raw_kl /= measured.input_tokens as f64;
        layer.tuned_kl /= measured.input_tokens as f64;
        layer.raw_ce /= measured.ce_tokens as f64;
        layer.tuned_ce /= measured.ce_tokens as f64;
        for (name, observed) in [
            ("raw_kl", layer.raw_kl),
            ("tuned_kl", layer.tuned_kl),
            ("raw_ce", layer.raw_ce),
            ("tuned_ce", layer.tuned_ce),
        ] {
            let expected = reference["metrics"]["layers"][layer.index][name]
                .as_f64()
                .ok_or("missing reference metric")?;
            metric_error = metric_error.max((observed - expected).abs());
            if (observed - expected).abs() > 0.0002 {
                eprintln!(
                    "Stage {} {name}: observed={observed:.9}, reference={expected:.9}",
                    layer.index
                );
            }
        }
        println!(
            "{:>5}       {:7.3}   {:7.3}  {:7.3}   {:7.3}",
            layer.index, layer.raw_kl, layer.tuned_kl, layer.raw_ce, layer.tuned_ce
        );
    }
    metric_error = metric_error.max(
        (measured.final_ce
            - reference["metrics"]["final_ce"]
                .as_f64()
                .ok_or("missing final CE")?)
        .abs(),
    );
    // These limits were fixed in the protocol before the first Rust measurement.
    let parity_passed = measured.parity_error <= 0.005 && metric_error <= 0.0002;
    let controls_passed = measured.observation_error <= 1e-6 && measured.replacement_error <= 1e-6;
    println!(
        "Final-model CE: {:.6}; maximum reference logit error: {:.6}",
        measured.final_ce, measured.parity_error
    );
    println!(
        "Maximum metric error: {metric_error:.8}; observe/identity errors: {:.8}/{:.8}",
        measured.observation_error, measured.replacement_error
    );
    let result = serde_json::json!({
        "schema_version": 1, "scope": "short-context bounded adaptation; released lens, no fitting",
        "backend": "Burn Flex", "dtype": "F32", "wall_seconds": measured.wall_seconds,
        "metrics": { "input_tokens": measured.input_tokens, "ce_tokens": measured.ce_tokens, "final_ce": measured.final_ce, "layers": measured.layers },
        "parity": { "passed": parity_passed, "maximum_logit_error": measured.parity_error, "maximum_metric_error": metric_error, "logit_tolerance": 0.005, "metric_tolerance": 0.0002 },
        "controls": { "passed": controls_passed, "observation_logit_error": measured.observation_error, "zero_translator_logit_error": measured.replacement_error, "replacement_translator": measured.replacement_translator },
        "provenance": {
            "preparation_sha256": file_hash(&root.join("preparation.json"))?,
            "reference_sha256": file_hash(&root.join("reference.json"))?,
            "prepared_outputs": preparation["outputs"],
            "compiled_source_sha256": {
                "examples/tuned_lens.rs": source_hash(include_bytes!("../../tuned_lens.rs")),
                "examples/mechanistic/tuned_lens/study.rs": source_hash(include_bytes!("study.rs")),
                "src/models/gpt_neox/model.rs": source_hash(include_bytes!("../../../src/models/gpt_neox/model.rs")),
                "src/models/gpt_neox/load.rs": source_hash(include_bytes!("../../../src/models/gpt_neox/load.rs"))
            }
        },
        "visualization": "predictions.html"
    });
    fs::write(
        root.join("rust_result.json"),
        serde_json::to_string_pretty(&result)? + "\n",
    )?;
    println!(
        "Wrote {}/predictions.html and rust_result.json",
        root.display()
    );
    if !parity_passed || !controls_passed {
        return Err(
            "reference parity or control failed; see rust_result.json for the measured values"
                .into(),
        );
    }
    Ok(())
}

/// A readable token table; color encodes the confidence of each top prediction.
pub fn prediction_cells(
    log_probs: Tensor<B, 3>,
    start: usize,
    count: usize,
    tokenizer: &tokenizers::Tokenizer,
) -> Result<Vec<(String, f32)>> {
    let window = log_probs.narrow(1, start, count);
    let ids = window
        .clone()
        .argmax(2)
        .reshape([count])
        .into_data()
        .to_vec::<i32>()?;
    let confidence = window
        .exp()
        .max_dim(2)
        .reshape([count])
        .into_data()
        .to_vec::<f32>()?;
    ids.into_iter()
        .zip(confidence)
        .map(|(id, probability)| {
            let text = tokenizer
                .decode(&[id as u32], false)
                .map_err(|error| std::io::Error::other(error.to_string()))?;
            Ok((text, probability))
        })
        .collect()
}

/// Write an escaped HTML table of next-token predictions and their probabilities.
///
/// Color reflects full-vocabulary confidence; the display is a predictive
/// decoder comparison and does not establish a causal circuit.
pub fn write_prediction_table(
    root: &Path,
    labels: &[String],
    rows: &[(String, Vec<(String, f32)>)],
) -> Result<()> {
    let escape = |text: &str| {
        text.replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
            .replace('"', "&quot;")
    };
    let mut html = String::from(
        "<!doctype html><html lang=\"en\"><meta charset=\"utf-8\"><title>Tuned Lens: latent predictions</title><style>body{font:16px system-ui;max-width:1200px;margin:32px auto;color:#142235}h1{font-size:28px}table{border-collapse:collapse;width:100%;font:13px ui-monospace,monospace}td,th{border:1px solid #dce2ea;padding:9px 6px;text-align:left;white-space:pre-wrap}th{background:#f1f4f8}small{display:block;color:#46586c;margin-top:4px}.wrap{overflow:auto}</style><h1>What does each layer predict next?</h1><p>Released Pythia-160M-v0 and its pretrained Tuned Lens. Columns show the preceding input token; cells show the next-token prediction. Darker cells mean higher confidence. These are predictive decoders, not causal claims.</p><div class=\"wrap\"><table><thead><tr><th>Readout</th>",
    );
    for label in labels {
        html.push_str(&format!("<th>{}</th>", escape(label)));
    }
    html.push_str("</tr></thead><tbody>");
    for (name, cells) in rows {
        html.push_str(&format!("<tr><th>{}</th>", escape(name)));
        for (token, probability) in cells {
            let alpha = 0.06 + 0.45 * probability.clamp(0.0, 1.0);
            html.push_str(&format!(
                "<td style=\"background:rgba(30,132,120,{alpha:.3})\">{}<small>{:.1}%</small></td>",
                escape(token),
                probability * 100.0
            ));
        }
        html.push_str("</tr>");
    }
    html.push_str("</tbody></table></div><p>Fixture: last eight positions of the first of four fixed Pile test records. Each record uses at most 128 input tokens. The study is a short-context adaptation of the paper's comparison method.</p></html>");
    fs::write(root.join("predictions.html"), html)?;
    Ok(())
}

#[cfg(test)]
#[path = "study_tests.rs"]
mod tests;
