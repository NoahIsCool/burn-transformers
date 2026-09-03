//! Validated bundles returned by architecture-specific local model loaders.
//!
//! Loading has several outputs that must remain associated with the executable
//! module: tokenizer state, generation defaults, immutable checkpoint
//! provenance, and the exact tensor accounting report. [`LoadedModel`] keeps
//! those values together so an application cannot accidentally report one
//! revision while executing another model. The bundle performs no network I/O;
//! [`ModelSourceMetadata`] canonicalizes an already-present local directory.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::{Error, GenerationConfig, Result, checkpoint::LoadReport};

/// Tokenizer type carried by [`LoadedModel`] when tokenizer support is enabled.
#[cfg(feature = "tokenizers")]
pub type ModelTokenizer = crate::LocalTokenizer;

/// Uninhabited tokenizer marker used when the optional tokenizer feature is off.
///
/// Consequently, `LoadedModel::tokenizer()` can only return `None` in a build
/// without `tokenizers`, while the bundle's public shape remains unchanged.
#[cfg(not(feature = "tokenizers"))]
#[derive(Debug)]
pub enum ModelTokenizer {}

/// Immutable identity and local origin of a loaded model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelSourceMetadata {
    /// Canonical local directory containing the model artifacts.
    root: PathBuf,
    /// Architecture discriminator read from the checkpoint configuration.
    model_type: String,
    /// Optional upstream repository name, such as `Qwen/Qwen3.5-0.8B`.
    repository: Option<String>,
    /// Optional immutable upstream commit or content revision.
    revision: Option<String>,
}

impl ModelSourceMetadata {
    /// Validates and canonicalizes one existing local artifact directory.
    pub fn local(root: impl AsRef<Path>, model_type: impl Into<String>) -> Result<Self> {
        let supplied_root = root.as_ref();
        let root = supplied_root
            .canonicalize()
            .map_err(|source| Error::io(supplied_root, source))?;
        if !root.is_dir() {
            return Err(Error::Artifact {
                path: root,
                reason: "model source root must be a directory".into(),
            });
        }
        let model_type = validate_label("model_type", model_type.into())?;
        Ok(Self {
            root,
            model_type,
            repository: None,
            revision: None,
        })
    }

    /// Attaches a non-empty upstream repository identifier.
    pub fn with_repository(mut self, repository: impl Into<String>) -> Result<Self> {
        self.repository = Some(validate_label("repository", repository.into())?);
        Ok(self)
    }

    /// Attaches a non-empty immutable revision identifier.
    pub fn with_revision(mut self, revision: impl Into<String>) -> Result<Self> {
        self.revision = Some(validate_label("revision", revision.into())?);
        Ok(self)
    }

    /// Returns the canonical local artifact root.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Returns the checkpoint architecture discriminator.
    pub fn model_type(&self) -> &str {
        &self.model_type
    }

    /// Returns the upstream repository identifier when recorded.
    pub fn repository(&self) -> Option<&str> {
        self.repository.as_deref()
    }

    /// Returns the immutable upstream revision when recorded.
    pub fn revision(&self) -> Option<&str> {
        self.revision.as_deref()
    }
}

/// An executable model and all host-side state established by a strict load.
#[derive(Debug)]
pub struct LoadedModel<M> {
    model: M,
    tokenizer: Option<ModelTokenizer>,
    generation_defaults: GenerationConfig,
    source: ModelSourceMetadata,
    load_report: LoadReport,
}

impl<M> LoadedModel<M> {
    /// Builds a bundle after validating generation and checkpoint invariants.
    ///
    /// Explicitly allowlisted omissions remain valid, but unexplained missing or
    /// unexpected tensors are rejected. The report is finalized here so every
    /// exposed category has deterministic ordering and no duplicates.
    pub fn new(
        model: M,
        tokenizer: Option<ModelTokenizer>,
        generation_defaults: GenerationConfig,
        source: ModelSourceMetadata,
        mut load_report: LoadReport,
        vocabulary_size: usize,
    ) -> Result<Self> {
        generation_defaults.validate(vocabulary_size)?;
        load_report.finalize();
        if !load_report.is_strictly_valid() {
            return Err(Error::Checkpoint(format!(
                "cannot construct LoadedModel: {} missing and {} unexpected tensors remain",
                load_report.missing.len(),
                load_report.unexpected.len()
            )));
        }
        Ok(Self {
            model,
            tokenizer,
            generation_defaults,
            source,
            load_report,
        })
    }

    /// Returns the executable Burn module.
    pub fn model(&self) -> &M {
        &self.model
    }

    /// Returns mutable access to the executable module without touching metadata.
    pub fn model_mut(&mut self) -> &mut M {
        &mut self.model
    }

    /// Returns the local tokenizer when this load attached one.
    pub fn tokenizer(&self) -> Option<&ModelTokenizer> {
        self.tokenizer.as_ref()
    }

    /// Returns validated generation defaults normalized for this vocabulary.
    pub fn generation_defaults(&self) -> &GenerationConfig {
        &self.generation_defaults
    }

    /// Returns immutable local and upstream source metadata.
    pub fn source(&self) -> &ModelSourceMetadata {
        &self.source
    }

    /// Returns deterministic tensor-level load accounting.
    pub fn load_report(&self) -> &LoadReport {
        &self.load_report
    }

    /// Consumes the bundle and returns the executable module.
    pub fn into_model(self) -> M {
        self.model
    }

    /// Consumes the bundle without discarding any component.
    pub fn into_parts(
        self,
    ) -> (
        M,
        Option<ModelTokenizer>,
        GenerationConfig,
        ModelSourceMetadata,
        LoadReport,
    ) {
        (
            self.model,
            self.tokenizer,
            self.generation_defaults,
            self.source,
            self.load_report,
        )
    }
}

/// Rejects empty, surrounding-whitespace, and control-character provenance.
fn validate_label(field: &str, value: String) -> Result<String> {
    if value.is_empty() || value.trim() != value {
        return Err(Error::configuration(
            format!("model_source.{field}"),
            "must be non-empty and contain no surrounding whitespace",
        ));
    }
    if value.chars().any(char::is_control) {
        return Err(Error::configuration(
            format!("model_source.{field}"),
            "must not contain control characters",
        ));
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use crate::checkpoint::LoadReport;

    use super::{LoadedModel, ModelSourceMetadata};

    /// Source metadata canonicalizes roots and preserves pinned provenance.
    #[test]
    fn local_source_is_canonical_and_pinned() {
        let directory = tempfile::tempdir().expect("temporary source root");
        let source = ModelSourceMetadata::local(directory.path(), "qwen3_5_text")
            .expect("valid local source")
            .with_repository("Qwen/Qwen3.5-0.8B")
            .expect("valid repository")
            .with_revision("2fc06364715b967f1860aea9cf38778875588b17")
            .expect("valid revision");
        assert!(source.root().is_absolute());
        assert_eq!(source.model_type(), "qwen3_5_text");
        assert_eq!(source.repository(), Some("Qwen/Qwen3.5-0.8B"));
        assert_eq!(
            source.revision(),
            Some("2fc06364715b967f1860aea9cf38778875588b17")
        );
    }

    /// Invalid roots and ambiguous textual identifiers produce contextual errors.
    #[test]
    fn local_source_rejects_invalid_roots_and_labels() {
        let directory = tempfile::tempdir().expect("temporary source root");
        let file = directory.path().join("config.json");
        std::fs::write(&file, b"{}").expect("fixture file");
        assert!(ModelSourceMetadata::local(&file, "t5").is_err());
        assert!(ModelSourceMetadata::local(directory.path().join("missing"), "t5").is_err());
        assert!(ModelSourceMetadata::local(directory.path(), "").is_err());

        let source = ModelSourceMetadata::local(directory.path(), "t5").expect("valid source");
        assert!(
            source
                .clone()
                .with_repository(" google/flan-t5-small")
                .is_err()
        );
        assert!(source.with_revision("bad\nrevision").is_err());
    }

    /// Construction finalizes a strict report and round-trips every component.
    #[test]
    fn loaded_model_validates_and_preserves_components() {
        let directory = tempfile::tempdir().expect("temporary source root");
        let source = ModelSourceMetadata::local(directory.path(), "t5").expect("valid source");
        let mut report = LoadReport::new();
        report.applied = vec!["z.weight".into(), "a.weight".into(), "a.weight".into()];
        let loaded = LoadedModel::new(
            17_u8,
            None,
            Default::default(),
            source.clone(),
            report,
            32_128,
        )
        .expect("strict load should bundle");
        assert_eq!(*loaded.model(), 17);
        assert!(loaded.tokenizer().is_none());
        assert_eq!(loaded.source(), &source);
        assert_eq!(loaded.load_report().applied, ["a.weight", "z.weight"]);
        assert_eq!(loaded.generation_defaults().max_new_tokens, 256);

        let (model, tokenizer, _, unpacked_source, unpacked_report) = loaded.into_parts();
        assert_eq!(model, 17);
        assert!(tokenizer.is_none());
        assert_eq!(unpacked_source, source);
        assert_eq!(unpacked_report.applied, ["a.weight", "z.weight"]);
    }

    /// Invalid generation controls and non-strict reports cannot escape loaders.
    #[test]
    fn loaded_model_rejects_invalid_runtime_metadata() {
        let directory = tempfile::tempdir().expect("temporary source root");
        let source = ModelSourceMetadata::local(directory.path(), "t5").expect("valid source");
        let invalid_generation = crate::GenerationConfig {
            max_new_tokens: 0,
            ..Default::default()
        };
        assert!(
            LoadedModel::new(
                (),
                None,
                invalid_generation,
                source.clone(),
                LoadReport::new(),
                32
            )
            .is_err()
        );

        let mut report = LoadReport::new();
        report.missing = vec!["decoder.weight".into(), "decoder.weight".into()];
        report.unexpected.push("foreign.weight".into());
        let error = LoadedModel::new((), None, Default::default(), source, report, 32)
            .expect_err("non-strict report must fail");
        assert!(error.to_string().contains("1 missing and 1 unexpected"));
    }
}
