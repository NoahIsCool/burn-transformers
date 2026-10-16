//! Context-rich errors returned by public transformer operations.
//!
//! User-provided configurations and artifacts are never trusted. Fallible
//! public operations therefore return [`Error`] instead of relying on panics
//! from lower-level tensor shape assumptions.

use std::path::PathBuf;

/// The result type used by this crate.
pub type Result<T> = core::result::Result<T, Error>;

/// A recoverable transformer runtime error.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// An architecture configuration violates a mathematical invariant.
    #[error("invalid configuration at {field}: {reason}")]
    Configuration {
        /// Dot-separated configuration field name.
        field: String,
        /// Human-readable invariant that was violated.
        reason: String,
    },

    /// A tensor has an incompatible rank or extent.
    #[error("shape mismatch for {tensor}: expected {expected}, found {actual}")]
    Shape {
        /// Logical tensor or parameter name.
        tensor: String,
        /// Expected tensor shape description.
        expected: String,
        /// Observed tensor shape description.
        actual: String,
    },

    /// An operation would violate cache state invariants.
    #[error("cache error: {0}")]
    Cache(
        /// Description of the violated cache invariant.
        String,
    ),

    /// A typed tensor-cache validation or lifecycle error.
    #[error("tensor cache error: {0}")]
    TensorCache(
        /// Typed cache failure retained for programmatic inspection.
        #[from]
        crate::cache::TensorCacheError,
    ),

    /// Invalid decode-cache dimensions passed to [`crate::cache::CacheConfig`].
    ///
    /// Converted automatically, so `CacheConfig::new(batch, capacity)?` works
    /// in any function returning [`Result`].
    #[error("cache configuration error: {0}")]
    CacheConfig(
        /// Typed dimension or window failure retained for programmatic inspection.
        #[from]
        crate::cache::CacheConfigError,
    ),

    /// Invalid rotary-embedding dimensions or context-scaling parameters.
    ///
    /// Converted automatically, so `RopeConfig::new(..)?` and the scaling
    /// constructors compose with `?` in any function returning [`Result`].
    #[error("rope configuration error: {0}")]
    Rope(
        /// Typed rotary failure retained for programmatic inspection.
        #[from]
        crate::position::RopeConfigError,
    ),

    /// A local model directory or one of its files is invalid.
    #[error("artifact error at {path}: {reason}")]
    Artifact {
        /// Local artifact path.
        path: PathBuf,
        /// Description of the invalid artifact.
        reason: String,
    },

    /// Checkpoint tensors fail strict validation or application.
    #[error("checkpoint error: {0}")]
    Checkpoint(
        /// Checkpoint validation or application failure with tensor context.
        String,
    ),

    /// Tokenizer files or operations are invalid.
    #[error("tokenizer error: {0}")]
    Tokenizer(
        /// Invalid tokenizer artifact or failed text operation.
        String,
    ),

    /// The selected backend cannot execute a required operation faithfully.
    #[error("backend capability error: {0}")]
    BackendCapability(
        /// Required operation or numerical contract unsupported by this backend.
        String,
    ),

    /// Generation parameters or state are invalid.
    #[error("generation error: {0}")]
    Generation(
        /// Invalid generation configuration, state, or model boundary result.
        String,
    ),

    /// An activation hook, edit callback, or model constraint failed while a
    /// model was executing inside a hook scope.
    ///
    /// Registration mistakes found before execution (unknown points, impossible
    /// axes) remain [`Error::Configuration`]. The message includes the cause's
    /// text; `cause` keeps the original error for matching. It is deliberately
    /// not exposed through [`std::error::Error::source`], so chain printers do
    /// not repeat it.
    #[error("intervention at {point} failed: {reason}")]
    Intervention {
        /// Semantic activation point, such as `decoder.layers.3.mlp.activation`.
        point: String,
        /// What failed at that point, including the cause's message when present.
        reason: String,
        /// Original error returned by a user callback or model constraint.
        cause: Option<Box<Error>>,
    },

    /// No compiled architecture matches a checkpoint model type.
    #[error("unsupported architecture: {0}")]
    UnsupportedArchitecture(
        /// Checkpoint model type for which no architecture is registered.
        String,
    ),

    /// A local filesystem operation failed.
    #[error("I/O error at {path}: {source}")]
    Io {
        /// Path involved in the operation.
        path: PathBuf,
        /// Underlying filesystem error.
        #[source]
        source: std::io::Error,
    },

    /// A JSON artifact could not be decoded.
    #[error("invalid JSON at {path}: {source}")]
    Json {
        /// JSON file being decoded.
        path: PathBuf,
        /// Underlying JSON syntax or data error.
        #[source]
        source: serde_json::Error,
    },
}

impl Error {
    /// Constructs a configuration error with a stable field path.
    pub fn configuration(field: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::Configuration {
            field: field.into(),
            reason: reason.into(),
        }
    }

    /// Constructs a shape error without allocating tensor data.
    pub fn shape(
        tensor: impl Into<String>,
        expected: impl Into<String>,
        actual: impl Into<String>,
    ) -> Self {
        Self::Shape {
            tensor: tensor.into(),
            expected: expected.into(),
            actual: actual.into(),
        }
    }

    /// Constructs a runtime intervention failure at a semantic activation point.
    pub fn intervention(point: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::Intervention {
            point: point.into(),
            reason: reason.into(),
            cause: None,
        }
    }

    /// Constructs an intervention failure that wraps the error a callback returned.
    ///
    /// The cause's message is appended to `reason` so the failure reads fully
    /// on one line; the cause itself stays available for matching.
    pub fn intervention_caused_by(
        point: impl Into<String>,
        reason: impl Into<String>,
        cause: Error,
    ) -> Self {
        Self::Intervention {
            point: point.into(),
            reason: format!("{}: {cause}", reason.into()),
            cause: Some(Box::new(cause)),
        }
    }

    /// Wraps a filesystem error with the path that gave it meaning.
    pub fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Self::Io {
            path: path.into(),
            source,
        }
    }

    /// Wraps a JSON error with its source artifact path.
    pub fn json(path: impl Into<PathBuf>, source: serde_json::Error) -> Self {
        Self::Json {
            path: path.into(),
            source,
        }
    }
}

#[cfg(test)]
#[path = "error_tests.rs"]
mod tests;
