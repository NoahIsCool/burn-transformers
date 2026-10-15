//! Decode-cache metadata, verified planning, and tensor-backed storage.
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
//!
//! # Module map
//!
//! * [`SequenceLengths`] holds committed and working lengths; its arithmetic
//!   is the Verus-verified ledger in `lengths.rs`, shared with
//!   [`TensorDecodeState`], so both report committed lengths from `lengths()`.
//! * `retention.rs` (rewind safety, speculative working windows, ring slot
//!   positions), `ring.rs` (ring write/read plans), and `rows.rs` (batched row
//!   runs) are pure integer planners verified in place by `verus/cache.rs`.
//! * [`tensor`] owns the Burn tensors: key/value and latent groups
//!   ([`NativeKvCache`]), recurrent state ([`RecurrentStateCache`]), banks,
//!   and the decode state with its step write sets ([`TensorStepWrites`]).

mod config;
mod lengths;
mod retention;
mod ring;
mod rows;
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
    KvBatchAppend, KvBatchRead, KvBatchReadPlan, KvBatchSegment, KvCacheRead, KvCacheReadSegment,
    KvCacheStorage, NativeKvCache, NativeKvCacheSpec, RecurrentBatchStateRead, RecurrentStateCache,
    RecurrentStateRead, RecurrentStateSpec, StepBank, TensorCacheBank, TensorCacheError,
    TensorCacheMark, TensorCacheResult, TensorDecodeState, TensorStepWrites, TokenAlignment,
};
pub use transaction::{CacheMark, TransactionError, TransactionalCache};
