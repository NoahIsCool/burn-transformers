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
    Cache(String),

    /// A typed tensor-cache validation or lifecycle error.
    #[error("tensor cache error: {0}")]
    TensorCache(#[from] crate::cache::TensorCacheError),

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
    Checkpoint(String),

    /// Tokenizer files or operations are invalid.
    #[error("tokenizer error: {0}")]
    Tokenizer(String),

    /// The selected backend cannot execute a required operation faithfully.
    #[error("backend capability error: {0}")]
    BackendCapability(String),

    /// Generation parameters or state are invalid.
    #[error("generation error: {0}")]
    Generation(String),

    /// No compiled architecture matches a checkpoint model type.
    #[error("unsupported architecture: {0}")]
    UnsupportedArchitecture(String),

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
