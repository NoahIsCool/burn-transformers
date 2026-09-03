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
//! proposal, then either keeps an accepted prefix or restores the mark. Cache
//! tensors may leave rejected bytes in memory because lengths are the source of
//! truth; subsequent writes overwrite those unreachable slots.

mod config;
mod ring;
mod state;
mod transaction;

pub use config::{CacheConfig, CacheConfigError};
pub use ring::{
    MirroredRingReadPlan, MirroredRingWriteSegment, RingPlanError, RingReadPlan, RingReadSegment,
    RingWritePlan, RingWriteSegment, plan_mirrored_ring_read, plan_mirrored_ring_write,
    plan_ring_read, plan_ring_write, sliding_window_range,
};
pub use state::{DecodeState, DecodeStateError, LengthTransition};
pub use transaction::{CacheMark, TransactionError, TransactionalCache};
