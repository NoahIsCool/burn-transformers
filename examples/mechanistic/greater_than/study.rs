//! Pinned local artifacts and reporting; model/intervention execution stays in main.
//!
//! # Probability and intervention contracts
//!
//! The fixed corpus contains 194 pairs of 13-token prompts. Only token eight
//! changes to the corrupted suffix; all conditions read token twelve and retain
//! the original clean suffix y as their cutoff. F32 causal forwards have no
//! padding or decode cache. Analyze full-vocabulary logits in F64 before selecting
//! the 100 numeric tokens: preference is `sum(p[y+1..]) - sum(p[..=y])`, sharpness
//! is `p[y+1] - p[y-1]`, and numeric mass is `sum(p)` without renormalization.
//!
//! # Measurement boundaries
//!
//! Node recovery divides the change from corrupt by the signed clean/corrupt gap,
//! unless that gap is negligible. Direct MLP-to-readout changes are separate from
//! node interventions. Clean prompts are disjoint across splits, but corrupted
//! prompts repeat; sample standard deviations are descriptive, not iid uncertainty.
//! Pinned F32 oracles, exact GELU wiring, identity controls, and source inventories
//! validate this bounded adaptation without claiming the paper's complete circuit.

use burn::{
    backend::Flex,
    tensor::{Int, Tensor, TensorData},
};
use safetensors::{Dtype, SafeTensors};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::Read,
    ops::Range,
    path::Path,
};
use transformers::{
    CausalLmInput,
    checkpoint::LoadOptions,
    models::gpt2::{Gpt2Activation, Gpt2Config, Gpt2ForCausalLm},
};

/// CPU backend used for F32 GPT-2 execution and activation comparisons.
pub type B = Flex<f32, i32>;
/// Experiment result preserving contextual artifact, tensor, and validation errors.
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// Immutable inventory of aligned clean/corrupt prompt pairs.
#[derive(Deserialize)]
struct Fixture {
    /// Ordered prompt pairs, including the fixed development and evaluation splits.
    entries: Vec<Prompt>,
}
/// One aligned prompt pair with the original year cutoff and intervention positions.
#[derive(Deserialize)]
struct Prompt {
    /// Original year-comparison prompt text.
    clean: String,
    /// Prompt text after replacing the start-year suffix with 01.
    corrupt: String,
    /// Token IDs for the original clean prompt.
    clean_ids: Vec<i32>,
    /// Token IDs for the aligned corrupted prompt.
    corrupt_ids: Vec<i32>,
    /// Frozen split label used when aggregating outcomes.
    split: String,
    /// Original clean start year; its final two digits define the comparison cutoff.
    year: usize,
    /// Zero-based final token position used for the prediction and MLP interventions.
    readout_position: usize,
    /// Zero-based token position containing the start-year suffix.
    start_suffix_position: usize,
}
/// Independent reference metrics for one prompt and intervention condition.
#[derive(Clone, Deserialize)]
struct ReferenceRow {
    /// Index of the prompt in the immutable fixture.
    index: usize,
    /// Name of the independently evaluated intervention or control.
    condition: String,
    /// Reference probability mass above the clean cutoff minus mass at or below it.
    pd: f64,
    /// Reference neighboring probability difference `p[y+1] - p[y-1]`.
    sharpness: f64,
    /// Reference total full-vocabulary probability assigned to the 100 numeric tokens.
    numeric_mass: f64,
    /// Reference probabilities ordered by numeric suffix, without subset renormalization.
    probabilities: Vec<f64>,
}
/// Predeclared thresholds for numerical agreement and resolvable recovery gaps.
#[derive(Clone, Copy, Deserialize)]
struct Tolerances {
    /// Maximum absolute error permitted for independent full-vocabulary logits.
    oracle_logits_absolute: f64,
    /// Absolute term in the captured-MLP element tolerance.
    captured_mlp_absolute: f64,
    /// Relative term multiplying each reference MLP element's absolute value.
    captured_mlp_relative: f64,
    /// Maximum absolute discrepancy in the greater-than probability difference.
    probability_difference_absolute: f64,
    /// Maximum absolute discrepancy in neighboring cutoff sharpness.
    cutoff_sharpness_absolute: f64,
    /// Maximum absolute discrepancy in an individual numeric-token probability.
    numeric_probabilities_absolute: f64,
    /// Maximum absolute logit change permitted for identity and restoration controls.
    controls_logits_absolute: f64,
    /// Absolute clean/corrupt gap below which recovery is left undefined.
    negligible_gap: f64,
}
/// Measured metrics for one prompt and intervention condition.
#[derive(Serialize)]
struct Row {
    /// Index of the prompt in the immutable fixture.
    index: usize,
    /// Frozen prompt split retained for separate aggregation.
    split: String,
    /// Original clean year modulo 100, used as the numeric cutoff.
    suffix: usize,
    /// Name of the executed intervention or control.
    condition: String,
    /// Probability mass above the cutoff minus mass at or below it.
    pd: f64,
    /// Neighboring cutoff contrast `p[y+1] - p[y-1]`.
    sharpness: f64,
    /// Total probability of numeric suffix tokens under the full-vocabulary softmax.
    numeric_mass: f64,
    /// Signed difference between the clean and corrupt probability-difference metrics.
    clean_corrupt_gap: f64,
    /// Condition probability difference minus the corresponding clean value.
    change_from_clean: f64,
    /// Fraction of the signed clean/corrupt gap recovered; absent when undefined or inapplicable.
    recovery: Option<f64>,
    /// Numeric probability curve retained only for selected conditions.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    probabilities: Vec<f64>,
}

/// Validation measures numerical agreement and identities, not a fitted effect size.
#[derive(Default, Serialize)]
struct Validation {
    /// Whether every required numerical agreement threshold passed.
    passed: bool,
    /// Largest absolute discrepancy against independent oracle logits.
    maximum_oracle_logit_error: f64,
    /// Largest absolute discrepancy in captured MLP values.
    maximum_mlp_error: f64,
    /// Number of MLP elements exceeding the combined absolute and relative tolerance.
    mlp_tolerance_failures: usize,
    /// Largest absolute discrepancy in the greater-than probability difference.
    maximum_pd_error: f64,
    /// Largest absolute discrepancy in neighboring cutoff sharpness.
    maximum_sharpness_error: f64,
    /// Largest absolute discrepancy in any numeric-token probability.
    maximum_numeric_probability_error: f64,
    /// Largest absolute discrepancy in total numeric-token probability mass.
    maximum_numeric_mass_error: f64,
    /// Largest absolute logit change across identity and restoration controls.
    maximum_control_logit_error: f64,
    /// Largest absolute discrepancy between explicit residual projection and clean logits.
    maximum_readout_identity_error: f64,
    /// Largest absolute discrepancy between final-layer direct-path and reverse-node logits.
    maximum_final_layer_path_node_error: f64,
    /// Maximum absolute clean-logit discrepancy indexed by control condition.
    control_errors: BTreeMap<String, f64>,
    /// Number of full-vocabulary oracle logit elements compared.
    oracle_elements_compared: usize,
    /// Number of captured MLP elements compared against the oracle.
    mlp_elements_compared: usize,
    /// Number of prompt/condition metric rows compared against independent references.
    metric_rows_compared: usize,
}

/// Validates prepared artifacts and accumulates greater-than intervention evidence.
pub struct Study {
    /// Immutable aligned prompt pairs and split assignments.
    fixture: Fixture,
    /// Embedded experiment specification with pinned artifact identities and thresholds.
    manifest: Value,
    /// Prepared artifact metadata and fingerprints loaded from disk.
    preparation: Value,
    /// Independent reference report including provenance and expected metrics.
    reference: Value,
    /// Reference metrics indexed by condition name and immutable prompt index.
    expected: BTreeMap<(String, usize), ReferenceRow>,
    /// Prepared independent tensor values used for logit and MLP parity.
    oracle: Oracle,
    /// Numerical thresholds loaded from the pinned experiment manifest.
    tolerances: Tolerances,
    /// Vocabulary token IDs ordered by the numeric suffixes 00 through 99.
    indices: Vec<usize>,
    /// Fixture row indices for which independent full tensor outputs are retained.
    oracle_indices: Vec<usize>,
    /// Measured prompt/condition metrics accumulated for the final report.
    rows: Vec<Row>,
    /// Condition/prompt keys already recorded, used to reject duplicates and omissions.
    seen: BTreeSet<(String, usize)>,
    /// Clean full-vocabulary logits retained for controls in the current batch.
    clean_logits: BTreeMap<usize, Vec<f32>>,
    /// Final-layer reverse-node logits retained for the matching direct-path identity.
    reverse_last: BTreeMap<usize, Vec<f32>>,
    /// Accumulated numerical errors and comparison counts.
    validation: Validation,
    /// Declared source-path SHA-256 digests captured before execution.
    source_inventory: BTreeMap<String, String>,
    /// SHA-256 digest of the executable producing the measurements.
    executable_sha256: String,
}

impl Study {
    /// Verify pinned checkpoint, fixture, and oracle artifacts before constructing GPT-2.
    ///
    /// Select the author graph's exact GELU explicitly, preserving all pretrained
    /// weights. Require the complete independent condition inventory and capture
    /// source/binary identities before any intervention measurement.
    pub fn load(root: &Path) -> Result<(Gpt2ForCausalLm<B>, Self)> {
        let source_inventory = source_inventory()?;
        let executable_sha256 = file_hash(&std::env::current_exe()?)?;
        let manifest: Value = serde_json::from_slice(include_bytes!("manifest.json"))?;
        let preparation: Value = serde_json::from_slice(&fs::read(root.join("preparation.json"))?)?;
        let fingerprint = &preparation["fingerprint"];
        check_string(
            &fingerprint["manifest_sha256"],
            &hash(include_bytes!("manifest.json")),
            "prepared manifest",
        )?;
        check_string(
            &fingerprint["prepare_sha256"],
            &hash(include_bytes!("prepare.py")),
            "preparation script",
        )?;
        check_string(
            &fingerprint["fixture_sha256"],
            &hash(include_bytes!("fixture.json")),
            "prepared fixture",
        )?;
        for (relative, expected) in preparation["outputs"]
            .as_object()
            .ok_or("missing prepared outputs")?
        {
            verify_artifact(root, relative, expected)?;
        }
        for name in ["model.safetensors", "config.json", "tokenizer.json"] {
            check_string(
                &manifest["artifact_sha256"][name],
                &file_hash(&root.join("model").join(name))?,
                name,
            )?;
        }
        let fixture_bytes = fs::read(root.join("fixture.json"))?;
        check_string(
            &manifest["fixture_sha256"],
            &hash(&fixture_bytes),
            "fixture",
        )?;
        let fixture: Fixture = serde_json::from_slice(&fixture_bytes)?;
        validate_fixture(&fixture)?;
        let mut config: Gpt2Config =
            serde_json::from_slice(&fs::read(root.join("model/config.json"))?)?;
        // HF config says gelu_new; the independently verified author graph says gelu.
        // This override is explicit and does not change a single pretrained weight.
        config.activation_function = Gpt2Activation::Gelu;
        let mut model = Gpt2ForCausalLm::init(config, &Default::default())?;
        model.load_checkpoint(root.join("model"), &LoadOptions::default())?;
        let reference: Value = serde_json::from_slice(&fs::read(root.join("reference.json"))?)?;
        for (field, expected) in [
            ("manifest_sha256", hash(include_bytes!("manifest.json"))),
            ("fixture_sha256", hash(include_bytes!("fixture.json"))),
            (
                "preparation_sha256",
                file_hash(&root.join("preparation.json"))?,
            ),
            ("prepare_sha256", hash(include_bytes!("prepare.py"))),
            ("reference_sha256", hash(include_bytes!("reference.py"))),
        ] {
            check_string(&reference["provenance"][field], &expected, field)?;
        }
        verify_artifact(
            root,
            "reference.safetensors",
            &reference["artifacts"]["reference.safetensors"],
        )?;
        let expected_rows: Vec<ReferenceRow> = serde_json::from_value(reference["rows"].clone())?;
        let mut expected = BTreeMap::new();
        for row in expected_rows {
            if row.index >= fixture.entries.len()
                || row.probabilities.len() != 100
                || expected
                    .insert((row.condition.clone(), row.index), row)
                    .is_some()
            {
                return Err("invalid or duplicated independent reference row".into());
            }
        }
        let conditions = manifest["conditions"]
            .as_array()
            .ok_or("missing conditions")?;
        if expected.len() != conditions.len() * fixture.entries.len() {
            return Err("reference has an incomplete condition inventory".into());
        }
        for condition in conditions {
            for index in 0..fixture.entries.len() {
                if !expected
                    .contains_key(&(condition.as_str().ok_or("condition name")?.into(), index))
                {
                    return Err("reference is missing a frozen condition".into());
                }
            }
        }
        let tolerances = serde_json::from_value(manifest["numerical_tolerances"].clone())?;
        let indices = serde_json::from_value(manifest["numeric_token_ids"].clone())?;
        let oracle_indices = serde_json::from_value(manifest["oracle_indices"].clone())?;
        Ok((
            model,
            Self {
                fixture,
                manifest,
                preparation,
                reference,
                expected,
                oracle: Oracle {
                    bytes: fs::read(root.join("reference.safetensors"))?,
                },
                tolerances,
                indices,
                oracle_indices,
                rows: Vec::new(),
                seen: BTreeSet::new(),
                clean_logits: BTreeMap::new(),
                reverse_last: BTreeMap::new(),
                validation: Validation::default(),
                source_inventory,
                executable_sha256,
            },
        ))
    }

    /// Return the number of aligned clean/corrupt prompt pairs in the fixed fixture.
    pub fn len(&self) -> usize {
        self.fixture.entries.len()
    }

    /// Build `[batch, 13]` clean or corrupted IDs for the same prompt range.
    ///
    /// All positions are real tokens; causal attention and default positions apply
    /// without padding, and the corrupted year suffix occupies token eight.
    pub fn input(&self, range: Range<usize>, corrupt: bool) -> CausalLmInput<B> {
        let batch = range.len();
        let ids: Vec<i32> = range
            .flat_map(|index| {
                let prompt = &self.fixture.entries[index];
                if corrupt {
                    prompt.corrupt_ids.clone()
                } else {
                    prompt.clean_ids.clone()
                }
            })
            .collect();
        CausalLmInput {
            token_ids: Tensor::<B, 2, Int>::from_data(
                TensorData::new(ids, [batch, 13]),
                &Default::default(),
            ),
            token_mask: None,
            position_ids: None,
        }
    }

    /// Validate `[batch, 1, 50257]` logits and record one frozen condition per prompt.
    ///
    /// Normalize over the full vocabulary in F64, compare reference logits and
    /// metrics, and retain same-run identity controls. Selected conditions also
    /// retain the full curve of 100 numeric suffix probabilities.
    pub fn record(&mut self, condition: &str, start: usize, logits: Tensor<B, 3>) -> Result<()> {
        let [batch, sequence, vocab] = logits.dims();
        if sequence != 1 || vocab != 50_257 || start + batch > self.len() {
            return Err("expected [batch, 1, 50257] final-token logits".into());
        }
        let data = tensor_values(logits)?;
        for (offset, logits) in data.chunks_exact(vocab).enumerate() {
            let index = start + offset;
            let key = (condition.to_string(), index);
            if !self.seen.insert(key.clone()) {
                return Err("duplicated measurement".into());
            }
            let expected = self
                .expected
                .get(&key)
                .ok_or("condition is absent from frozen reference")?;
            let probabilities = numeric_probabilities(logits, &self.indices)?;
            let suffix = self.fixture.entries[index].year % 100;
            let (pd, sharpness, numeric_mass) = metrics(&probabilities, suffix)?;
            let v = &mut self.validation;
            v.maximum_pd_error = v.maximum_pd_error.max((pd - expected.pd).abs());
            v.maximum_sharpness_error = v
                .maximum_sharpness_error
                .max((sharpness - expected.sharpness).abs());
            v.maximum_numeric_mass_error = v
                .maximum_numeric_mass_error
                .max((numeric_mass - expected.numeric_mass).abs());
            for (a, b) in probabilities.iter().zip(&expected.probabilities) {
                if !b.is_finite() {
                    return Err("reference probabilities must be finite".into());
                }
                v.maximum_numeric_probability_error =
                    v.maximum_numeric_probability_error.max((a - b).abs());
            }
            v.metric_rows_compared += 1;
            if let Some(oracle_index) = self
                .oracle_indices
                .iter()
                .position(|&selected| selected == index)
            {
                let expected = self.oracle.rows(
                    &format!("logits.{condition}"),
                    &[self.oracle_indices.len(), vocab],
                    oracle_index,
                    1,
                )?;
                v.maximum_oracle_logit_error = v
                    .maximum_oracle_logit_error
                    .max(max_error(logits, &expected)?);
                v.oracle_elements_compared += vocab;
            }
            if condition == "clean" {
                self.clean_logits.insert(index, logits.to_vec());
            }
            if condition == "node_reverse_11" {
                self.reverse_last.insert(index, logits.to_vec());
            }
            if condition == "direct_11" {
                v.maximum_final_layer_path_node_error =
                    v.maximum_final_layer_path_node_error.max(max_error(
                        logits,
                        self.reverse_last
                            .get(&index)
                            .ok_or("missing final node control")?,
                    )?);
            }
            if condition == "empty_clean"
                || condition == "observe_clean"
                || condition.starts_with("identity_")
                || condition.starts_with("restore_")
            {
                let error = max_error(
                    logits,
                    self.clean_logits
                        .get(&index)
                        .ok_or("missing clean control")?,
                )?;
                v.maximum_control_logit_error = v.maximum_control_logit_error.max(error);
                v.control_errors
                    .entry(condition.into())
                    .and_modify(|x| *x = x.max(error))
                    .or_insert(error);
            }
            let retain_curve = matches!(
                condition,
                "clean" | "corrupt" | "node_patch_10" | "node_patch_all_late"
            );
            self.rows.push(Row {
                index,
                split: self.fixture.entries[index].split.clone(),
                suffix,
                condition: condition.into(),
                pd,
                sharpness,
                numeric_mass,
                clean_corrupt_gap: 0.0,
                change_from_clean: 0.0,
                recovery: None,
                probabilities: if retain_curve {
                    probabilities
                } else {
                    Vec::new()
                },
            });
        }
        Ok(())
    }

    /// Compare selected `[batch, 1, 768]` MLP outputs with the pinned F32 oracle.
    ///
    /// Each element must satisfy `abs(error) <= atol + rtol * abs(reference)`;
    /// record failures and comparison counts for the final validation report.
    pub fn check_capture(
        &mut self,
        source: &str,
        layer: usize,
        start: usize,
        hidden: Tensor<B, 3>,
    ) -> Result<()> {
        let batch = hidden.dims()[0];
        if hidden.dims() != [batch, 1, 768] {
            return Err("MLP capture must select exactly END's 768 channels".into());
        }
        let expected = self.oracle.rows(
            &format!("mlp.{source}.{layer}"),
            &[self.len(), 1, 768],
            start,
            batch,
        )?;
        for (actual, expected) in tensor_values(hidden)?.iter().zip(&expected) {
            let error = (*actual as f64 - *expected as f64).abs();
            self.validation.maximum_mlp_error = self.validation.maximum_mlp_error.max(error);
            if error
                > self.tolerances.captured_mlp_absolute
                    + self.tolerances.captured_mlp_relative * (*expected as f64).abs()
            {
                self.validation.mlp_tolerance_failures += 1;
            }
            self.validation.mlp_elements_compared += 1;
        }
        Ok(())
    }

    /// Compare projection of the clean final residual with ordinary clean logits.
    pub fn check_readout(&mut self, start: usize, logits: Tensor<B, 3>) -> Result<()> {
        for (offset, actual) in tensor_values(logits)?.chunks_exact(50_257).enumerate() {
            self.validation.maximum_readout_identity_error = self
                .validation
                .maximum_readout_identity_error
                .max(max_error(
                    actual,
                    self.clean_logits
                        .get(&(start + offset))
                        .ok_or("missing clean readout")?,
                )?);
        }
        Ok(())
    }

    /// Release cached clean and final-node control logits after all batch conditions run.
    pub fn finish_batch(&mut self, range: Range<usize>) {
        for index in range {
            self.clean_logits.remove(&index);
            self.reverse_last.remove(&index);
        }
    }

    /// Finalize signed effects and numerical controls, then persist all measurements.
    ///
    /// Require unchanged crate sources and a complete condition inventory. Node
    /// recovery is measured relative to the clean/corrupt gap; direct-path changes
    /// remain separate. Failed numerical gates retain their raw result before erroring.
    pub fn finish(mut self, root: &Path, wall_seconds: f64) -> Result<()> {
        if source_inventory()? != self.source_inventory {
            return Err(
                "crate sources changed during execution; rerun against a stable source tree".into(),
            );
        }
        if self.seen.len() != self.expected.len() {
            return Err("not every frozen condition was measured".into());
        }
        let clean: BTreeMap<_, _> = self
            .rows
            .iter()
            .filter(|r| r.condition == "clean")
            .map(|r| (r.index, r.pd))
            .collect();
        let corrupt: BTreeMap<_, _> = self
            .rows
            .iter()
            .filter(|r| r.condition == "corrupt")
            .map(|r| (r.index, r.pd))
            .collect();
        for row in &mut self.rows {
            row.clean_corrupt_gap = clean[&row.index] - corrupt[&row.index];
            row.change_from_clean = row.pd - clean[&row.index];
            // A direct-path measurement reports signed change from clean. It is
            // distinct from restoring the corrupt destination's node behavior.
            row.recovery = if row.condition.starts_with("direct_") {
                None
            } else {
                recovery(
                    row.pd,
                    clean[&row.index],
                    corrupt[&row.index],
                    self.tolerances.negligible_gap,
                )
            };
        }
        self.rows
            .sort_by(|a, b| a.index.cmp(&b.index).then(a.condition.cmp(&b.condition)));
        let t = self.tolerances;
        let v = &mut self.validation;
        v.passed = v.maximum_oracle_logit_error <= t.oracle_logits_absolute
            && v.mlp_tolerance_failures == 0
            && v.maximum_pd_error <= t.probability_difference_absolute
            && v.maximum_sharpness_error <= t.cutoff_sharpness_absolute
            && v.maximum_numeric_probability_error <= t.numeric_probabilities_absolute
            && v.maximum_control_logit_error <= t.controls_logits_absolute
            && v.maximum_readout_identity_error <= t.oracle_logits_absolute
            && v.maximum_final_layer_path_node_error <= t.oracle_logits_absolute;
        let mut summaries = BTreeMap::new();
        let mut gaps = BTreeMap::new();
        for split in ["development", "held_out"] {
            let mut conditions = BTreeMap::new();
            for condition in self.manifest["conditions"].as_array().ok_or("conditions")? {
                let condition = condition.as_str().ok_or("condition")?;
                let rows: Vec<_> = self
                    .rows
                    .iter()
                    .filter(|r| r.split == split && r.condition == condition)
                    .collect();
                conditions.insert(condition.to_string(), summary(&rows));
            }
            let baseline: Vec<_> = self
                .rows
                .iter()
                .filter(|r| r.split == split && r.condition == "clean")
                .collect();
            gaps.insert(split, json!({"statistics": statistics(&baseline.iter().map(|r| r.clean_corrupt_gap).collect::<Vec<_>>()),
                "near_zero_count": baseline.iter().filter(|r| r.clean_corrupt_gap.abs() <= t.negligible_gap).count()}));
            summaries.insert(split, conditions);
        }
        let source_hashes = BTreeMap::from([
            (
                "examples/greater_than.rs",
                hash(include_bytes!("../../greater_than.rs")),
            ),
            (
                "examples/mechanistic/greater_than/study.rs",
                hash(include_bytes!("study.rs")),
            ),
            (
                "examples/mechanistic/greater_than/manifest.json",
                hash(include_bytes!("manifest.json")),
            ),
            (
                "examples/mechanistic/greater_than/fixture.json",
                hash(include_bytes!("fixture.json")),
            ),
            (
                "examples/mechanistic/greater_than/prepare.py",
                hash(include_bytes!("prepare.py")),
            ),
            (
                "examples/mechanistic/greater_than/reference.py",
                hash(include_bytes!("reference.py")),
            ),
            (
                "src/models/gpt2.rs",
                hash(include_bytes!("../../../src/models/gpt2.rs")),
            ),
            (
                "src/models/gpt2/model.rs",
                hash(include_bytes!("../../../src/models/gpt2/model.rs")),
            ),
            (
                "src/models/gpt2/checkpoint.rs",
                hash(include_bytes!("../../../src/models/gpt2/checkpoint.rs")),
            ),
        ]);
        let result = json!({
            "protocol": "greater-than-v1", "scope": "Bounded activation-patching adaptation with separate direct MLP-to-readout measurements",
            "backend": "Burn 0.21 Flex F32 CPU; F64 host probability analysis", "activation": "exact GELU, matching the author's declared graph",
            "wall_seconds": wall_seconds, "validation": self.validation,
            "expected": {"numerical_tolerances": self.manifest["numerical_tolerances"], "scientific_expectations": self.manifest["scientific_expectations"]},
            "provenance": {"source_sha256": source_hashes,
                "manifest_sha256": hash(include_bytes!("manifest.json")), "fixture_sha256": hash(include_bytes!("fixture.json")),
                "preparation_sha256": file_hash(&root.join("preparation.json"))?,
                "reference_sha256": file_hash(&root.join("reference.json"))?, "prepared_artifacts": self.preparation["outputs"],
                "reference_provenance": self.reference["provenance"], "reference_artifacts": self.reference["artifacts"],
                "repository_base_commit": git_revision(), "working_tree_changes_present": git_changes_present(),
                "crate_source_sha256": self.source_inventory, "executable_sha256": self.executable_sha256},
            "algebraic_invariants": "readout(clean final residual)=clean logits; direct_11=node_reverse_11 up to F32 subtraction/readdition; checked under frozen oracle tolerance",
            "sample_counts": self.manifest["split_counts"], "summary": summaries, "baseline_gap": gaps, "rows": self.rows,
            "limitations": ["Clean prompts are disjoint across splits; corrupted controls repeat and ten are shared across splits.",
                "Per-prompt SD is descriptive variation, not an iid confidence interval.",
                "Node patching does not reproduce the full published path-patched circuit.",
                "Zero ablation removes the complete biased MLP output; this is an added control.",
                "Full original rust-circuit execution, training, quantized GPT-2 import, and the paper's 10000-sample aggregate are not validated."]
        });
        fs::write(
            root.join("result.json"),
            serde_json::to_vec_pretty(&result)?,
        )?;
        println!(
            "Held-out clean prompts: probability difference, sharpness, signed change from clean"
        );
        for name in [
            "clean",
            "corrupt",
            "node_patch_2",
            "node_patch_8",
            "node_patch_9",
            "node_patch_10",
            "node_patch_11",
            "node_patch_all_late",
            "direct_10",
        ] {
            let summary = &result["summary"]["held_out"][name];
            println!(
                "{name:22} PD={:.6} sharpness={:.6} change={:.6}",
                summary["mean_pd"].as_f64().unwrap_or_default(),
                summary["mean_sharpness"].as_f64().unwrap_or_default(),
                summary["mean_change_from_clean"]
                    .as_f64()
                    .unwrap_or_default()
            );
        }
        println!(
            "wrote {}; runtime {:.2}s; numerical validation {}",
            root.join("result.json").display(),
            wall_seconds,
            if self.validation.passed {
                "passed"
            } else {
                "FAILED"
            }
        );
        if !self.validation.passed {
            return Err(
                "Greater-Than numerical validation failed; raw measurements were retained".into(),
            );
        }
        Ok(())
    }
}

/// Check all 194 aligned pairs, bounded IDs, year cutoffs, and clean split uniqueness.
///
/// Only token eight may differ, every sequence has 13 tokens, and readout
/// occurs at token twelve. Repeated corrupted prompts are allowed by the protocol.
fn validate_fixture(fixture: &Fixture) -> Result<()> {
    if fixture.entries.len() != 194 {
        return Err("fixture must contain 194 fixed pairs".into());
    }
    let mut clean_prompts = BTreeSet::new();
    for (index, p) in fixture.entries.iter().enumerate() {
        let expected_split = if index < 97 {
            "development"
        } else {
            "held_out"
        };
        let differences: Vec<_> = p
            .clean_ids
            .iter()
            .zip(&p.corrupt_ids)
            .enumerate()
            .filter(|(_, (a, b))| a != b)
            .map(|(i, _)| i)
            .collect();
        if p.split != expected_split
            || p.year % 100 != 2 + index % 97
            || !(1100..1800).contains(&p.year)
            || p.clean_ids.len() != 13
            || p.corrupt_ids.len() != 13
            || differences != [8]
            || p.clean_ids[0] != 50_256
            || p.corrupt_ids[8] != 486
            || p.clean_ids
                .iter()
                .chain(&p.corrupt_ids)
                .any(|&id| !(0..50_257).contains(&id))
            || p.readout_position != 12
            || p.start_suffix_position != 8
            || p.clean == p.corrupt
            || !clean_prompts.insert(&p.clean)
        {
            return Err(format!("prompt {index} violates the fixed aligned fixture").into());
        }
    }
    Ok(())
}

/// Stable analysis in F64, using every original vocabulary logit in the denominator.
fn numeric_probabilities(logits: &[f32], indices: &[usize]) -> Result<Vec<f64>> {
    if logits.is_empty()
        || logits.iter().any(|x| !x.is_finite())
        || indices.iter().any(|&i| i >= logits.len())
    {
        return Err("probability analysis needs finite logits and valid numeric indices".into());
    }
    let maximum = logits
        .iter()
        .map(|&x| x as f64)
        .fold(f64::NEG_INFINITY, f64::max);
    let denominator: f64 = logits.iter().map(|&x| (x as f64 - maximum).exp()).sum();
    Ok(indices
        .iter()
        .map(|&i| (logits[i] as f64 - maximum).exp() / denominator)
        .collect())
}

/// Compute probability difference, cutoff sharpness, and total numeric mass.
///
/// For original clean suffix y, return `sum(p[y+1..]) - sum(p[..=y])`,
/// `p[y+1] - p[y-1]`, and `sum(p)`. Keep probabilities unrenormalized over
/// the 100 numeric tokens and require the valid cutoff range 02..98.
fn metrics(p: &[f64], suffix: usize) -> Result<(f64, f64, f64)> {
    if p.len() != 100 || !(2..=98).contains(&suffix) {
        return Err(
            "metrics require original clean suffix 02..98 and 100 numeric probabilities".into(),
        );
    }
    Ok((
        p[suffix + 1..].iter().sum::<f64>() - p[..=suffix].iter().sum::<f64>(),
        p[suffix + 1] - p[suffix - 1],
        p.iter().sum(),
    ))
}
/// Return `(patched - corrupt) / (clean - corrupt)` when the gap is resolvable.
///
/// Use the absolute gap only for the threshold check; preserve its sign and
/// allow recovery outside [0, 1]. A negligible gap produces an absent value.
fn recovery(patched: f64, clean: f64, corrupt: f64, threshold: f64) -> Option<f64> {
    let gap = clean - corrupt;
    (gap.abs() > threshold).then(|| (patched - corrupt) / gap)
}
/// Summarize count, arithmetic mean, and sample standard deviation in F64.
///
/// Empty populations yield null statistics; a singleton has zero descriptive
/// spread. These are prompt variation summaries, not confidence intervals.
fn statistics(values: &[f64]) -> Value {
    if values.is_empty() {
        return json!({"mean": null, "sd": null, "count": 0});
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let sd = if values.len() < 2 {
        0.0
    } else {
        (values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64).sqrt()
    };
    json!({"mean": mean, "sd": sd, "count": values.len()})
}
/// Aggregate one split/condition while keeping undefined and inapplicable recovery distinct.
fn summary(rows: &[&Row]) -> Value {
    let pd = statistics(&rows.iter().map(|r| r.pd).collect::<Vec<_>>());
    let sharpness = statistics(&rows.iter().map(|r| r.sharpness).collect::<Vec<_>>());
    let recoveries = statistics(&rows.iter().filter_map(|r| r.recovery).collect::<Vec<_>>());
    let change = statistics(&rows.iter().map(|r| r.change_from_clean).collect::<Vec<_>>());
    let applicable = rows
        .first()
        .is_some_and(|row| !row.condition.starts_with("direct_"));
    json!({"count": rows.len(), "mean_pd": pd["mean"], "sd_pd": pd["sd"],
        "mean_sharpness": sharpness["mean"], "sd_sharpness": sharpness["sd"],
        "mean_numeric_mass": statistics(&rows.iter().map(|r| r.numeric_mass).collect::<Vec<_>>())["mean"],
        "mean_recovery": recoveries["mean"], "sd_recovery": recoveries["sd"],
        "recovery_applicable": applicable,
        "undefined_recovery": if applicable { rows.iter().filter(|r| r.recovery.is_none()).count() } else { 0 },
        "mean_change_from_clean": change["mean"], "sd_change_from_clean": change["sd"]})
}

/// Owns the prepared safetensors bytes used as an independent numerical reference.
struct Oracle {
    /// Immutable serialized oracle tensors decoded on demand.
    bytes: Vec<u8>,
}
impl Oracle {
    /// Read a bounded leading-axis range from a finite F32 oracle tensor.
    ///
    /// Require its complete expected shape before decoding contiguous
    /// little-endian rows; trailing axes determine each row's element stride.
    fn rows(&self, name: &str, shape: &[usize], start: usize, count: usize) -> Result<Vec<f32>> {
        let tensors = SafeTensors::deserialize(&self.bytes)?;
        let view = tensors.tensor(name)?;
        if view.dtype() != Dtype::F32 || view.shape() != shape || start + count > shape[0] {
            return Err(format!("{name}: independent oracle shape/dtype mismatch").into());
        }
        let stride: usize = shape[1..].iter().product();
        let bytes = &view.data()[start * stride * 4..(start + count) * stride * 4];
        let values: Vec<_> = bytes
            .chunks_exact(4)
            .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
            .collect();
        if values.iter().any(|x| !x.is_finite()) {
            return Err("nonfinite oracle values".into());
        }
        Ok(values)
    }
}
/// Read a Flex tensor as row-major F32 host values with a contextual conversion error.
fn tensor_values<const D: usize>(tensor: Tensor<B, D>) -> Result<Vec<f32>> {
    Ok(tensor
        .into_data()
        .to_vec::<f32>()
        .map_err(|error| format!("tensor data read: {error:?}"))?)
}
/// Compute F64 maximum absolute error after checking equal lengths and finite values.
fn max_error(a: &[f32], b: &[f32]) -> Result<f64> {
    if a.len() != b.len() || a.iter().chain(b).any(|x| !x.is_finite()) {
        return Err("numerical comparison needs equal finite vectors".into());
    }
    Ok(a.iter()
        .zip(b)
        .map(|(&a, &b)| (a as f64 - b as f64).abs())
        .fold(0.0, f64::max))
}
/// Hash embedded protocol, fixture, or source bytes with SHA-256.
fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
/// Stream a local file into SHA-256 with a bounded 1 MiB buffer.
fn file_hash(path: &Path) -> Result<String> {
    let mut source = fs::File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let read = source.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(format!("{:x}", digest.finalize()))
}
/// Require exact string equality for a named pinned provenance field.
fn check_string(actual: &Value, expected: &str, label: &str) -> Result<()> {
    if actual.as_str() != Some(expected) {
        return Err(format!("{label}: stale or mismatched pinned provenance").into());
    }
    Ok(())
}
/// Reject non-normal path components, then check the local artifact size and hash.
///
/// The lexical check excludes absolute paths and parent traversal; the
/// prepared directory itself is trusted, including its existing symlinks.
fn verify_artifact(root: &Path, relative: &str, expected: &Value) -> Result<()> {
    if Path::new(relative)
        .components()
        .any(|p| !matches!(p, std::path::Component::Normal(_)))
    {
        return Err("artifact paths must stay within the prepared directory".into());
    }
    let path = root.join(relative);
    if expected["bytes"].as_u64() != Some(fs::metadata(&path)?.len()) {
        return Err(format!("{relative}: artifact length mismatch").into());
    }
    check_string(&expected["sha256"], &file_hash(&path)?, relative)
}
/// Read the local HEAD identity, returning absence when Git cannot report it.
fn git_revision() -> Option<String> {
    std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().into())
}

/// Record whether Git reports tracked or ordinary untracked changes when available.
fn git_changes_present() -> Option<bool> {
    std::process::Command::new("git")
        .args(["status", "--porcelain", "--untracked-files=normal"])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| !output.stdout.is_empty())
}

/// Bind the run to every crate source, including the shared transformer mechanics.
fn source_inventory() -> Result<BTreeMap<String, String>> {
    /// Recursively inventory Rust files under the crate source directory using relative paths.
    fn visit(root: &Path, directory: &Path, files: &mut BTreeMap<String, String>) -> Result<()> {
        for entry in fs::read_dir(directory)? {
            let entry = entry?;
            let path = entry.path();
            if entry.file_type()?.is_dir() {
                visit(root, &path, files)?;
            } else if path.extension().is_some_and(|extension| extension == "rs") {
                files.insert(
                    path.strip_prefix(root)?.to_string_lossy().into(),
                    file_hash(&path)?,
                );
            }
        }
        Ok(())
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut files = BTreeMap::new();
    visit(root, &root.join("src"), &mut files)?;
    for name in ["Cargo.toml", "Cargo.lock"] {
        files.insert(name.into(), file_hash(&root.join(name))?);
    }
    Ok(files)
}

#[cfg(test)]
#[path = "study_tests.rs"]
mod tests;
