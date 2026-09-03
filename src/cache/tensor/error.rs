//! Typed validation and lifecycle errors for tensor-backed caches.

use burn::tensor::DType;

use crate::cache::{DecodeStateError, RingPlanError};

/// Result returned by tensor-cache construction and mutation operations.
pub type TensorCacheResult<T> = core::result::Result<T, TensorCacheError>;

/// Recoverable failure while configuring or mutating tensor-backed state.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum TensorCacheError {
    /// A required cache dimension was zero.
    #[error("cache dimension `{field}` must be non-zero")]
    ZeroDimension {
        /// Name of the invalid dimension.
        field: &'static str,
    },

    /// Cache storage was configured with a non-floating-point data type.
    #[error("cache dtype must be floating point, found {dtype:?}")]
    NonFloatDType {
        /// Invalid runtime data type.
        dtype: DType,
    },

    /// The selected backend cannot generally store and update the data type.
    #[error("backend does not support general cache operations for {dtype:?}")]
    UnsupportedDType {
        /// Unsupported runtime data type.
        dtype: DType,
    },

    /// The requested batch disagreed with the metadata configuration.
    #[error("cache config tracks {configured} sequences but batch size is {requested}")]
    BatchSizeMismatch {
        /// Sequence count encoded in [`crate::cache::CacheConfig`].
        configured: usize,
        /// Batch size requested by the model.
        requested: usize,
    },

    /// A ring storage mode was selected without a sliding-window width.
    #[error("{storage} storage requires CacheConfig::sliding_window")]
    SlidingWindowRequired {
        /// Human-readable storage mode.
        storage: &'static str,
    },

    /// Doubling a mirrored ring width overflowed `usize`.
    #[error("mirrored ring width 2 * {window} overflows usize")]
    MirroredCapacityOverflow {
        /// Undoubled sliding-window width.
        window: usize,
    },

    /// A layer index was outside a tensor bank.
    #[error("layer index {layer} is out of bounds for {layers} layers")]
    LayerOutOfBounds {
        /// Invalid layer index.
        layer: usize,
        /// Available layer count.
        layers: usize,
    },

    /// A cache-bank index was outside the requested collection.
    #[error("{kind} cache index {index} is out of bounds for {count} caches")]
    CacheOutOfBounds {
        /// Kind of state cache being indexed.
        kind: &'static str,
        /// Invalid cache index.
        index: usize,
        /// Available cache count.
        count: usize,
    },

    /// A sequence index was outside the configured batch.
    #[error("sequence index {sequence} is out of bounds for batch size {batch_size}")]
    SequenceOutOfBounds {
        /// Invalid sequence index.
        sequence: usize,
        /// Configured batch size.
        batch_size: usize,
    },

    /// A batch index cannot be represented by Burn's portable i32 selector.
    #[error("batch index {index} exceeds the i32 indexing range")]
    BatchIndexOverflow {
        /// Invalid source row index.
        index: usize,
    },

    /// An input tensor did not have the documented shape.
    #[error("shape mismatch for {tensor}: expected {expected:?}, found {actual:?}")]
    ShapeMismatch {
        /// Semantic name of the tensor.
        tensor: &'static str,
        /// Required extents.
        expected: Vec<usize>,
        /// Observed extents.
        actual: Vec<usize>,
    },

    /// An input tensor's runtime dtype differed from the cache allocation.
    #[error("dtype mismatch for {tensor}: expected {expected:?}, found {actual:?}")]
    DTypeMismatch {
        /// Semantic name of the tensor.
        tensor: &'static str,
        /// Cache allocation dtype.
        expected: DType,
        /// Input tensor dtype.
        actual: DType,
    },

    /// An input tensor resides on a different backend device.
    #[error("device mismatch for {tensor}: tensor must reside on the cache device")]
    DeviceMismatch {
        /// Semantic name of the tensor.
        tensor: &'static str,
    },

    /// A direct write exceeded logical context capacity.
    #[error("write [{start}, {end}) exceeds logical cache capacity {capacity}")]
    WriteExceedsCapacity {
        /// First absolute position requested.
        start: usize,
        /// Exclusive absolute end requested.
        end: usize,
        /// Configured logical capacity.
        capacity: usize,
    },

    /// An absolute write-end calculation overflowed `usize`.
    #[error("write start {start} plus token count {tokens} overflows usize")]
    WritePositionOverflow {
        /// First absolute position requested.
        start: usize,
        /// Number of new tokens.
        tokens: usize,
    },

    /// A read requested a logical length beyond capacity.
    #[error("read length {length} exceeds logical cache capacity {capacity}")]
    ReadExceedsCapacity {
        /// Requested exclusive logical end.
        length: usize,
        /// Configured logical capacity.
        capacity: usize,
    },

    /// Speculative staging was not preallocated for this cache.
    #[error("speculative staging is disabled; configure a non-zero staging capacity")]
    StagingDisabled,

    /// A second speculative transaction was started before finishing the first.
    #[error("a tensor-cache transaction is already active")]
    TransactionAlreadyActive,

    /// A speculative operation was requested without a transaction.
    #[error("no tensor-cache transaction is active")]
    TransactionNotActive,

    /// The supplied transaction mark is stale or belongs to another state.
    #[error("tensor-cache mark is stale or belongs to another state")]
    StaleTransactionMark,

    /// A lifecycle mutation cannot run during a speculative transaction.
    #[error("cannot {operation} while a tensor-cache transaction is active")]
    TransactionActive {
        /// Name of the rejected lifecycle operation.
        operation: &'static str,
    },

    /// Appending a proposal would exceed its staging allocation.
    #[error("staging {requested} tokens exceeds staging capacity {capacity}")]
    StagingCapacityExceeded {
        /// Requested staged suffix length.
        requested: usize,
        /// Preallocated proposal capacity.
        capacity: usize,
    },

    /// A layer did not stage exactly the logical speculative suffix.
    #[error(
        "incomplete {kind} staging at layer {layer}, sequence {sequence}: expected {expected} tokens, found {actual}"
    )]
    IncompleteStaging {
        /// Kind of tensor cache.
        kind: &'static str,
        /// Layer within the cache.
        layer: usize,
        /// Sequence within the batch.
        sequence: usize,
        /// Logical proposal length.
        expected: usize,
        /// Physically staged length.
        actual: usize,
    },

    /// A commit accepted more tokens than were proposed for one sequence.
    #[error("sequence {sequence} cannot accept {accepted} tokens from proposal length {proposed}")]
    AcceptedPrefixExceedsProposal {
        /// Affected sequence index.
        sequence: usize,
        /// Requested accepted-prefix length.
        accepted: usize,
        /// Logical proposal length.
        proposed: usize,
    },

    /// A wrapped ring cannot reconstruct history exposed by a rewind.
    #[error(
        "cannot truncate wrapped {storage} sequence {sequence} from {current} to {requested}; reset to zero or retain a history journal"
    )]
    UnsafeWrappedTruncate {
        /// Ring storage mode.
        storage: &'static str,
        /// Affected sequence index.
        sequence: usize,
        /// Current logical length.
        current: usize,
        /// Requested earlier logical length.
        requested: usize,
    },

    /// A recurrent terminal state cannot be inverted to an intermediate step.
    #[error(
        "cannot truncate recurrent sequence {sequence} from {current} to intermediate length {requested}; store state checkpoints"
    )]
    UnsafeRecurrentTruncate {
        /// Affected sequence index.
        sequence: usize,
        /// Current logical length.
        current: usize,
        /// Requested earlier logical length.
        requested: usize,
    },

    /// An attached bank has incompatible config or device affinity.
    #[error("cannot attach {bank} bank: {reason}")]
    IncompatibleBank {
        /// Logical bank name.
        bank: &'static str,
        /// Specific incompatibility.
        reason: &'static str,
    },

    /// MTP storage was enabled more than once.
    #[error("MTP cache bank is already enabled")]
    MtpAlreadyEnabled,

    /// Backend-independent logical metadata rejected a transition.
    #[error(transparent)]
    Metadata(#[from] DecodeStateError),

    /// Ring-buffer index planning rejected a transition.
    #[error(transparent)]
    Ring(#[from] RingPlanError),
}

impl From<TensorCacheError> for crate::Error {
    fn from(value: TensorCacheError) -> Self {
        Self::Cache(value.to_string())
    }
}
