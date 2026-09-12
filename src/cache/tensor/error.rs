//! Typed validation and lifecycle errors for tensor-backed caches.
//!
//! Errors retain the layer, tensor, axis, or lifecycle boundary that failed.
//! Shape and byte-count failures are checked before backend allocation; stale
//! marks and invalid accepted prefixes are checked before cache publication. The
//! error hierarchy distinguishes these recoverable input failures from backend
//! resource limits rather than treating successful metadata validation as a
//! promise that device allocation or arithmetic will succeed.

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

    /// Multiplying tensor dimensions overflowed the host address space.
    #[error("shape element count for {tensor} overflows usize")]
    TensorShapeOverflow {
        /// Semantic name of the tensor being validated.
        tensor: &'static str,
    },

    /// A tensor exceeds the portable signed-index range used by Burn backends.
    #[error("{tensor} requires {elements} elements, exceeding portable limit {limit}")]
    TensorElementLimitExceeded {
        /// Semantic name of the tensor being validated.
        tensor: &'static str,
        /// Requested scalar element count.
        elements: usize,
        /// Largest element count accepted before backend allocation.
        limit: usize,
    },

    /// Computing the byte size of a cache allocation overflowed `usize`.
    #[error(
        "allocation byte count for {tensor} overflows usize ({elements} elements of {element_size} bytes)"
    )]
    AllocationSizeOverflow {
        /// Semantic name of the allocation group.
        tensor: &'static str,
        /// Number of scalar elements in the allocation group.
        elements: usize,
        /// Bytes used by one scalar element.
        element_size: usize,
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

    /// A runtime tensor index cannot be represented by Burn's portable i32 type.
    #[error("{axis} index {index} exceeds the i32 indexing range")]
    TensorIndexOverflow {
        /// Logical axis being indexed.
        axis: &'static str,
        /// Invalid source index.
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

    /// A whole-batch staging write encountered different row offsets.
    #[error(
        "cannot stage a uniform recurrent batch at layer {layer}: sequence {sequence} offset {actual} differs from {expected}"
    )]
    NonUniformStagingOffset {
        /// Recurrent layer within the cache group.
        layer: usize,
        /// First row whose offset differs.
        sequence: usize,
        /// Uniform offset required by the batch assignment.
        expected: usize,
        /// Existing offset for the differing row.
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

    /// An MTP-specific operation was requested without an auxiliary bank.
    #[error("MTP cache bank is not enabled")]
    MtpNotEnabled,

    /// MTP staged tokens require an independent accepted-prefix vector.
    #[error("MTP staged tokens require an explicit independent accepted prefix")]
    MtpAcceptedPrefixRequired,

    /// Backend-independent logical metadata rejected a transition.
    #[error(transparent)]
    Metadata(
        /// Underlying logical-state validation failure.
        #[from]
        DecodeStateError,
    ),

    /// Ring-buffer index planning rejected a transition.
    #[error(transparent)]
    Ring(
        /// Underlying ring indexing or accepted-prefix validation failure.
        #[from]
        RingPlanError,
    ),
}
