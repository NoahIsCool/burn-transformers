//! Preallocated Burn tensor storage for autoregressive inference.
//!
//! Cache tensors deliberately live outside [`burn::module::Module`]. They are
//! mutable session state, not checkpoint parameters, and serializing them with
//! model weights would couple inference history to the differentiable module
//! graph. The types in this module retain fixed tensor shapes for the lifetime
//! of a session while exposing logical length changes through
//! [`TensorDecodeState`].
//!
//! Burn tensor updates are ownership-based: [`burn::tensor::Tensor::inplace`]
//! permits a backend to reuse an uniquely owned allocation, but Burn 0.21 does
//! not promise that `slice_assign` or batch reordering is allocation-free, and
//! Burn Flex 0.21's `slice_assign` in fact copies its whole destination. Cache
//! writes therefore go through the crate's `operations::assign_region` and
//! `operations::zero_region`, which on inference Flex write only the new
//! slots into a uniquely owned plane and keep its buffer, so a token write
//! costs `O(written slots)` rather than `O(capacity)`; other backends use
//! `slice_assign`. "Preallocated" guarantees stable cache *capacity and
//! shape*; physical buffer reuse is guaranteed on inference Flex while no read
//! view of the plane is alive, and remains a backend optimization elsewhere.
//! Batch reordering (`select`) still allocates.
//!
//! # Model step workflow
//!
//! A cached forward pass appends `n_b` tokens to each row `b`:
//!
//! 1. [`TensorDecodeState::step`] starts a [`TensorStepWrites`] for the step.
//! 2. Each attention layer reads its history for the whole batch, either
//!    through a [`KvBatchReadPlan`] built once per step and group
//!    ([`TensorDecodeState::kv_read_plan`]) plus the step's own tokens
//!    ([`TensorDecodeState::step_token_segment`]), or, for full storage, by
//!    writing the new tokens ahead ([`TensorDecodeState::write_kv_ahead`]) and
//!    reading one zero-copy view ([`TensorDecodeState::read_kv_through`]).
//!    [`KvBatchSegment::exclusion_mask`] turns key positions into the causal,
//!    sliding-window, and padding mask, and [`KvBatchSegment::masked_values`]
//!    zeroes the unspecified values of invalid slots.
//! 3. Each layer records its new keys/values and recurrent states in the step.
//! 4. [`TensorDecodeState::publish_step`] validates the whole write set and
//!    commits it, or stages it inside a speculative transaction.
//!
//! A step belongs to the state and bank version it was created at, and a
//! read plan to the group (and transaction) that built it; using either after
//! the state moved on, or with another state or group, is an error rather
//! than a silent misread.
//!
//! The per-row APIs (`read_sequence`, `prepare_direct_writes`, …) remain for
//! families that still run one row at a time.

mod allocation;
mod error;
mod kv;
mod plan;
mod recurrent;
mod state;

pub(crate) use allocation::{validate_repeated_allocation, validate_tensor_shape};
pub use error::{TensorCacheError, TensorCacheResult};
pub use kv::{
    KvBatchAppend, KvBatchRead, KvBatchReadPlan, KvBatchSegment, KvCacheRead, KvCacheReadSegment,
    KvCacheStorage, NativeKvCache, NativeKvCacheSpec,
};
pub use recurrent::{
    RecurrentBatchStateRead, RecurrentStateCache, RecurrentStateRead, RecurrentStateSpec,
};
pub use state::{
    StepBank, TensorCacheBank, TensorCacheMark, TensorDecodeState, TensorStepWrites, TokenAlignment,
};

#[cfg(test)]
#[path = "tensor_tests.rs"]
mod tests;
