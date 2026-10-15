//! Decode-cache metadata and ring-buffer planning.
//!
//! Autoregressive inference separates a *logical* token position from the
//! physical slot holding its key/value state. Logical sequence lengths grow
//! monotonically up to the configured context capacity. Sliding-window layers
//! may reuse a bounded set of physical slots with `slot = position mod window`.
//! Keeping those two coordinate systems separate prevents wrapped storage from
//! accidentally resetting RoPE positions or causal-mask offsets.
//!
//! Speculative decoding adds a second distinction: some appended states are
//! tentative. [`TransactionalCache`] records the logical lengths before a
//! proposal, then either keeps an accepted prefix or restores the mark. Logical
//! rollback alone is sufficient for append-only storage, but not for a wrapped
//! ring: speculative writes could overwrite still-visible baseline slots.
//! [`plan_staged_ring_transaction`] therefore models the safe policy of keeping
//! proposals in separate linear storage and copying only the accepted prefix
//! into the committed ring.

mod config;
mod lengths;
mod retention;
mod ring;
mod state;
pub mod tensor;
mod transaction;

pub use config::{CacheConfig, CacheConfigError};
pub use ring::{
    MirroredRingReadPlan, MirroredRingWriteSegment, RingPlanError, RingReadPlan, RingReadSegment,
    RingWritePlan, RingWriteSegment, StagedRingTransactionPlan, plan_mirrored_ring_read,
    plan_mirrored_ring_write, plan_ring_read, plan_ring_write, plan_staged_ring_transaction,
    sliding_window_range,
};
pub use state::{DecodeState, DecodeStateError, LengthTransition, SequenceLengths};
pub use tensor::{
    KvCacheRead, KvCacheReadSegment, KvCacheStorage, NativeKvCache, NativeKvCacheSpec,
    RecurrentBatchStateRead, RecurrentStateCache, RecurrentStateRead, RecurrentStateSpec,
    TensorCacheBank, TensorCacheError, TensorCacheMark, TensorCacheResult, TensorDecodeState,
};
pub use transaction::{CacheMark, TransactionError, TransactionalCache};
