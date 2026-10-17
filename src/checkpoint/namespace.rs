//! Detection of the language-model namespace of checkpoints published in two layouts.
//!
//! Several families ship the same language weights in two parameter layouts:
//! inside a multimodal checkpoint, where every language tensor lives below
//! `model.language_model.`, and as a standalone text checkpoint, whose token
//! embedding is `model.embed_tokens.weight`. Loaders choose the matching
//! architecture contract from the tensor names alone, so the classification
//! must be total and exclusive: a name set with only wrapped names is
//! [`LanguageNamespace::Wrapped`], one with only the text-only embedding is
//! [`LanguageNamespace::TextOnly`], and a name set with neither or both is an
//! error. No filename or configuration heuristics are involved.
//!
//! Families whose official releases use more namespaces (Gemma's original
//! multimodal release, for example) keep their own detector.

use super::{CheckpointError, CheckpointManifest, Result};

/// Prefix of every language tensor in a multimodal (wrapped) checkpoint.
const WRAPPED_PREFIX: &str = "model.language_model.";

/// Token embedding that identifies a standalone text checkpoint.
const TEXT_ONLY_EMBEDDING: &str = "model.embed_tokens.weight";

/// Parameter namespace holding a checkpoint's language model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LanguageNamespace {
    /// Released multimodal layout: `model.language_model.*`.
    Wrapped,
    /// Standalone text layout: `model.embed_tokens.weight` and siblings.
    TextOnly,
}

impl LanguageNamespace {
    /// Classifies a validated tensor-name set into exactly one language namespace.
    ///
    /// `family` names the model family in diagnostics, such as `Qwen`.
    ///
    /// # Errors
    ///
    /// [`CheckpointError::LanguageNamespaceNotFound`] when no name has the
    /// wrapped prefix and none is the text-only embedding, and
    /// [`CheckpointError::AmbiguousLanguageNamespace`] when both kinds occur.
    pub(crate) fn classify<'a>(
        names: impl IntoIterator<Item = &'a str>,
        family: &'static str,
    ) -> Result<Self> {
        let mut wrapped = false;
        let mut text_only = false;
        for name in names {
            wrapped |= name.starts_with(WRAPPED_PREFIX);
            text_only |= name == TEXT_ONLY_EMBEDDING;
        }
        match (wrapped, text_only) {
            (true, false) => Ok(Self::Wrapped),
            (false, true) => Ok(Self::TextOnly),
            (false, false) => Err(CheckpointError::LanguageNamespaceNotFound { family }),
            (true, true) => Err(CheckpointError::AmbiguousLanguageNamespace { family }),
        }
    }

    /// Classifies the complete tensor inventory of a discovered checkpoint.
    ///
    /// # Errors
    ///
    /// As [`Self::classify`].
    pub(crate) fn of_manifest(manifest: &CheckpointManifest, family: &'static str) -> Result<Self> {
        Self::classify(manifest.tensors.keys().map(String::as_str), family)
    }
}

#[cfg(test)]
#[path = "namespace_tests.rs"]
mod tests;
