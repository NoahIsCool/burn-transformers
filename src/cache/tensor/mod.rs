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
//! not promise that `slice_assign` or batch reordering is allocation-free.
//! "Preallocated" here therefore guarantees stable cache *capacity and shape*,
//! while physical buffer reuse remains a backend optimization.

mod allocation;
mod error;
mod kv;
mod recurrent;
mod state;

pub use error::{TensorCacheError, TensorCacheResult};
pub use kv::{KvCacheRead, KvCacheReadSegment, KvCacheStorage, NativeKvCache, NativeKvCacheSpec};
pub use recurrent::{
    RecurrentBatchStateRead, RecurrentStateCache, RecurrentStateRead, RecurrentStateSpec,
};
pub use state::{TensorCacheBank, TensorCacheMark, TensorDecodeState};

#[cfg(test)]
mod tests;
