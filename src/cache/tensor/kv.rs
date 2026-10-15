//! Native key/value and latent attention cache tensors.
//!
//! Every committed layer stores one or two *planes* shaped
//! `[batch, heads, time, dim]`: keys and values for ordinary attention, or a
//! single latent per token (`heads = 1`, `dim = width`) for multi-head latent
//! attention and similar compressed caches. Keeping time next to the feature
//! dimension makes a chronological slice directly consumable by attention
//! kernels, and every storage, staging, batching, and lifecycle operation
//! below applies identically to each plane.
//!
//! Full storage allocates `time = capacity`; a standard sliding ring
//! allocates `time = window`; a mirrored ring allocates `time = 2 * window`
//! and writes each slot twice so the retained suffix is one contiguous tensor
//! view. The window is the group's own
//! ([`NativeKvCacheSpec::with_sliding_window`]) or else the bank's.
//!
//! Speculative proposals are staged in separate linear buffers, or, for full
//! storage with in-place staging, directly in committed slots beyond the
//! committed length; see [`NativeKvCacheSpec::with_in_place_staging`]. Batched
//! whole-layer reads and writes live in the `batch` submodule.

use core::num::NonZeroUsize;
use core::ops::Range;

use burn::tensor::{DType, Int, Tensor, TensorData, backend::Backend};

use crate::cache::retention::{WorkingWindow, first_unsafe_ring_rewind, ring_working_window};
use crate::cache::{
    CacheConfig, DecodeStateError, plan_mirrored_ring_write, plan_ring_read, plan_ring_write,
    plan_staged_ring_transaction,
};

use super::{
    TensorCacheError, TensorCacheResult, validate_repeated_allocation, validate_tensor_shape,
};

mod batch;

pub(crate) use batch::PreparedNativeKvBatchWrite;
pub use batch::{KvBatchAppend, KvBatchRead, KvBatchReadPlan, KvBatchSegment};

/// Physical time-axis strategy used by a native key/value cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KvCacheStorage {
    /// Stores every logical position at its absolute time index.
    Full,
    /// Stores `position mod window`, producing at most two chronological views.
    SlidingRing,
    /// Duplicates each ring slot at `slot + window` for one contiguous view.
    MirroredSlidingRing,
}

impl KvCacheStorage {
    /// Returns a stable human-readable storage name for diagnostics.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::SlidingRing => "sliding ring",
            Self::MirroredSlidingRing => "mirrored sliding ring",
        }
    }
}

/// Validated tensor geometry for a homogeneous group of attention layers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NativeKvCacheSpec {
    /// Number of layers with identical key/value geometry.
    layers: NonZeroUsize,
    /// Number of stored key/value heads per layer.
    key_value_heads: NonZeroUsize,
    /// Scalar channels in each key or value head.
    head_dimension: NonZeroUsize,
    /// Mapping from logical positions to committed time slots.
    storage: KvCacheStorage,
    /// Floating-point dtype shared by committed and proposed tensors.
    dtype: DType,
    /// Maximum proposal tokens per row; zero disables staging.
    staging_capacity: usize,
    /// Group-specific ring width; `None` falls back to the bank's config window.
    sliding_window: Option<NonZeroUsize>,
    /// Whether proposals are written straight into committed full storage.
    in_place_staging: bool,
    /// Whether the group stores one latent plane instead of a key/value pair.
    latent: bool,
}

impl NativeKvCacheSpec {
    /// Creates a native KV specification without speculative staging.
    ///
    /// # Errors
    ///
    /// Returns an error when any dimension is zero or `dtype` is not a normal
    /// floating-point tensor type.
    pub fn new(
        layers: usize,
        key_value_heads: usize,
        head_dimension: usize,
        storage: KvCacheStorage,
        dtype: DType,
    ) -> TensorCacheResult<Self> {
        let layers = non_zero("layers", layers)?;
        let key_value_heads = non_zero("key_value_heads", key_value_heads)?;
        let head_dimension = non_zero("head_dimension", head_dimension)?;
        validate_float_dtype(dtype)?;
        Ok(Self {
            layers,
            key_value_heads,
            head_dimension,
            storage,
            dtype,
            staging_capacity: 0,
            sliding_window: None,
            in_place_staging: false,
            latent: false,
        })
    }

    /// Creates a single-plane per-token latent specification.
    ///
    /// Some attention variants cache one compressed vector per token instead
    /// of a key/value pair: multi-head latent attention (MLA) stores the
    /// normalized KV latent next to the shared rotary key, and cross-layer or
    /// quantized caches have similar layouts. A latent group stores
    /// `[batch, 1, time, width]` per layer and otherwise behaves exactly like a
    /// KV group: full or ring storage, speculative staging (separate or in
    /// place), batched I/O, step writes, reorder, truncate, and reset. Use the
    /// `*_latent*` methods of [`NativeKvCache`] and [`TensorStepWrites`]
    /// (`[rows, tokens, width]` inputs) instead of the key/value pair methods.
    ///
    /// [`TensorStepWrites`]: super::TensorStepWrites
    ///
    /// # Errors
    ///
    /// Returns an error when `layers` or `width` is zero or `dtype` is not a
    /// normal floating-point type.
    pub fn latent(
        layers: usize,
        width: usize,
        storage: KvCacheStorage,
        dtype: DType,
    ) -> TensorCacheResult<Self> {
        let mut spec = Self::new(layers, 1, width, storage, dtype)?;
        spec.latent = true;
        Ok(spec)
    }

    /// Returns whether this is a single-plane latent group.
    #[must_use]
    pub const fn is_latent(self) -> bool {
        self.latent
    }

    /// Returns the number of per-token tensors stored per layer: 2 for keys
    /// and values, 1 for a latent group.
    #[must_use]
    pub const fn plane_count(self) -> usize {
        if self.latent { 1 } else { 2 }
    }

    /// Sets the maximum number of proposal tokens staged per sequence.
    ///
    /// A value of zero disables speculative staging. Staging tensors retain the
    /// same `[batch, kv_heads, staging_capacity, head_dim]` layout as committed
    /// tensors and are allocated once with the cache.
    #[must_use]
    pub const fn with_staging_capacity(mut self, staging_capacity: usize) -> Self {
        self.staging_capacity = staging_capacity;
        self
    }

    /// Returns the number of homogeneous attention layers.
    #[must_use]
    pub const fn layers(self) -> usize {
        self.layers.get()
    }

    /// Returns the number of key/value heads per layer.
    #[must_use]
    pub const fn key_value_heads(self) -> usize {
        self.key_value_heads.get()
    }

    /// Returns the scalar width of each key/value head.
    #[must_use]
    pub const fn head_dimension(self) -> usize {
        self.head_dimension.get()
    }

    /// Returns the committed physical storage strategy.
    #[must_use]
    pub const fn storage(self) -> KvCacheStorage {
        self.storage
    }

    /// Returns the runtime floating-point dtype of every tensor.
    #[must_use]
    pub const fn dtype(self) -> DType {
        self.dtype
    }

    /// Returns the preallocated speculative suffix capacity per sequence.
    #[must_use]
    pub const fn staging_capacity(self) -> usize {
        self.staging_capacity
    }

    /// Stages proposals directly in committed full storage instead of separate buffers.
    ///
    /// Full storage never evicts, and every slot at or beyond a row's
    /// committed length is unreachable by committed reads. Proposal token `j`
    /// of row `b` can therefore be written straight to slot `C_b + j`:
    /// accepting a prefix only advances the committed length, rejecting it
    /// leaves bytes no read can reach (the next write overwrites them), and a
    /// working read is one contiguous view `[0, C_b + P_b)` with no staging
    /// concatenation. No staging tensors are allocated; `staging_capacity`
    /// still bounds the proposal length and must be non-zero for the group to
    /// join a transaction.
    ///
    /// The trade-off is *logical* rather than byte-exact isolation: after a
    /// rollback, slots beyond the committed length hold rejected proposals.
    /// Ring storage cannot use this mode because a proposal would overwrite a
    /// still-visible committed slot; [`NativeKvCache::new`] rejects it.
    #[must_use]
    pub const fn with_in_place_staging(mut self) -> Self {
        self.in_place_staging = true;
        self
    }

    /// Returns whether proposals are staged in place in committed storage.
    #[must_use]
    pub const fn in_place_staging(self) -> bool {
        self.in_place_staging
    }

    /// Returns whether this group allocates separate proposal tensors.
    const fn allocates_staging(self) -> bool {
        self.staging_capacity > 0 && !self.in_place_staging
    }

    /// Gives this group its own sliding-window width.
    ///
    /// Ring storage normally uses [`CacheConfig::sliding_window`], which every
    /// group of a bank shares. A group-specific width lets one decode state mix,
    /// for example, 512-token and 4096-token local layers, or follow a
    /// per-layer window schedule by grouping layers of equal width. The width
    /// must not exceed the logical capacity and requires ring storage;
    /// [`NativeKvCache::new`] validates both.
    #[must_use]
    pub const fn with_sliding_window(mut self, window: NonZeroUsize) -> Self {
        self.sliding_window = Some(window);
        self
    }

    /// Returns the group-specific sliding window, if one was set.
    #[must_use]
    pub const fn sliding_window(self) -> Option<NonZeroUsize> {
        self.sliding_window
    }

    /// Returns the ring width this group uses under `config`.
    ///
    /// The group's own window takes precedence over the config's shared
    /// window; full storage has no ring width.
    #[must_use]
    pub fn effective_window(self, config: CacheConfig) -> Option<NonZeroUsize> {
        match self.storage {
            KvCacheStorage::Full => None,
            KvCacheStorage::SlidingRing | KvCacheStorage::MirroredSlidingRing => {
                self.sliding_window.or(config.sliding_window())
            }
        }
    }

    /// Returns the exact time-axis extent allocated for committed tensors.
    ///
    /// Full storage allocates the logical capacity, a standard ring its
    /// window `W`, and a mirrored ring `2W`. This is the authoritative answer
    /// that [`CacheConfig::physical_capacity`] can only approximate, because the
    /// config does not know a group's storage mode.
    ///
    /// # Errors
    ///
    /// Returns an error for ring storage without any window, a window larger
    /// than the logical capacity, a window on full storage, in-place staging
    /// on ring storage, or `2W` overflow.
    pub fn committed_time_slots(self, config: CacheConfig) -> TensorCacheResult<usize> {
        if self.in_place_staging && self.storage != KvCacheStorage::Full {
            return Err(TensorCacheError::InPlaceStagingRequiresFullStorage {
                storage: self.storage.as_str(),
            });
        }
        if let Some(window) = self.sliding_window {
            if self.storage == KvCacheStorage::Full {
                return Err(TensorCacheError::InvalidSlidingWindow {
                    storage: self.storage.as_str(),
                    window: window.get(),
                    capacity: config.capacity(),
                    reason: "full storage retains every position; use a ring storage mode",
                });
            }
            if window.get() > config.capacity() {
                return Err(TensorCacheError::InvalidSlidingWindow {
                    storage: self.storage.as_str(),
                    window: window.get(),
                    capacity: config.capacity(),
                    reason: "the window exceeds the logical capacity",
                });
            }
        }
        let window = || {
            self.effective_window(config)
                .ok_or(TensorCacheError::SlidingWindowRequired {
                    storage: self.storage.as_str(),
                })
        };
        match self.storage {
            KvCacheStorage::Full => Ok(config.capacity()),
            KvCacheStorage::SlidingRing => Ok(window()?.get()),
            KvCacheStorage::MirroredSlidingRing => {
                let window = window()?;
                window
                    .get()
                    .checked_mul(2)
                    .ok_or(TensorCacheError::MirroredCapacityOverflow {
                        window: window.get(),
                    })
            }
        }
    }
}

/// One chronological key/value segment returned by a cache read.
#[derive(Debug, Clone)]
pub struct KvCacheReadSegment<B: Backend> {
    /// Absolute positions represented by this chronological tensor slice.
    absolute_positions: Range<usize>,
    /// Key slice shaped `[1, kv_heads, segment_time, head_dim]`.
    keys: Tensor<B, 4>,
    /// Value slice with the same geometry as the keys.
    values: Tensor<B, 4>,
}

impl<B: Backend> KvCacheReadSegment<B> {
    /// Returns the absolute token positions represented by this segment.
    #[must_use]
    pub fn absolute_positions(&self) -> Range<usize> {
        self.absolute_positions.clone()
    }

    /// Borrows keys shaped `[1, kv_heads, segment_time, head_dim]`.
    #[must_use]
    pub const fn keys(&self) -> &Tensor<B, 4> {
        &self.keys
    }

    /// Borrows values shaped `[1, kv_heads, segment_time, head_dim]`.
    #[must_use]
    pub const fn values(&self) -> &Tensor<B, 4> {
        &self.values
    }
}

/// Chronological native-KV read split across committed and staged storage.
#[derive(Debug, Clone)]
pub struct KvCacheRead<B: Backend> {
    /// Absolute interval covered by all returned segments.
    retained_positions: Range<usize>,
    /// Committed and staged tensor slices in chronological order.
    segments: Vec<KvCacheReadSegment<B>>,
}

impl<B: Backend> KvCacheRead<B> {
    /// Returns the absolute key interval visible to the attention layer.
    #[must_use]
    pub fn retained_positions(&self) -> Range<usize> {
        self.retained_positions.clone()
    }

    /// Returns chronological tensor segments.
    ///
    /// A committed standard ring contributes at most two segments and an
    /// active proposal contributes one more linear staging segment.
    #[must_use]
    pub fn segments(&self) -> &[KvCacheReadSegment<B>] {
        &self.segments
    }

    /// Materializes split ring segments as one chronological key/value pair.
    ///
    /// Full and mirrored caches already have one segment and only clone their
    /// tensor handles. A standard wrapped ring invokes [`Tensor::cat`] and may
    /// allocate an attention input. Empty cache reads return `None` because
    /// Burn backends do not uniformly support zero-length slicing.
    #[must_use]
    pub fn materialize(&self) -> Option<(Tensor<B, 4>, Tensor<B, 4>)> {
        self.materialize_pair()
    }

    /// Materializes a latent group's read as one `[1, time, width]` tensor.
    ///
    /// A latent group stores a single plane, which both `keys()` and
    /// `values()` of each segment return; this joins the segments and drops
    /// the singleton head axis (a free reshape of a contiguous row).
    #[must_use]
    pub fn materialize_latent(&self) -> Option<Tensor<B, 3>> {
        let (latent, _) = self.materialize_pair()?;
        let [rows, _, time, width] = latent.dims();
        Some(latent.reshape([rows, time, width]))
    }

    /// Shared segment concatenation behind both materialize helpers.
    fn materialize_pair(&self) -> Option<(Tensor<B, 4>, Tensor<B, 4>)> {
        match self.segments.as_slice() {
            [] => None,
            [segment] => Some((segment.keys.clone(), segment.values.clone())),
            segments => Some((
                Tensor::cat(
                    segments
                        .iter()
                        .map(|segment| segment.keys.clone())
                        .collect(),
                    2,
                ),
                Tensor::cat(
                    segments
                        .iter()
                        .map(|segment| segment.values.clone())
                        .collect(),
                    2,
                ),
            )),
        }
    }
}

/// Committed planes and optional staging planes for one layer.
///
/// A plane is one per-token tensor `[batch, heads, time, dim]`: keys and
/// values for a KV group (planes 0 and 1), or the single latent of a latent
/// group. Every operation applies identically to each plane.
#[derive(Debug)]
struct KvLayer<B: Backend> {
    /// Committed planes `[batch, heads, physical_time, dim]`.
    planes: Vec<Tensor<B, 4>>,
    /// Linear proposal planes `[batch, heads, staging_capacity, dim]`, when allocated.
    staged: Option<Vec<Tensor<B, 4>>>,
}

impl<B: Backend> KvLayer<B> {
    /// Returns the plane read as keys (plane 0).
    fn keys(&self) -> &Tensor<B, 4> {
        &self.planes[0]
    }

    /// Returns the plane read as values: plane 1, or the latent plane itself.
    fn values(&self) -> &Tensor<B, 4> {
        self.planes.get(1).unwrap_or(&self.planes[0])
    }
}

/// One fully planned direct key/value append.
///
/// The fields are private so only [`NativeKvCache::prepare_sequence_write`] can
/// construct this capability. Applying it performs tensor assignments only;
/// every recoverable shape, device, capacity, and ring-planning error has
/// already been returned to the caller.
#[derive(Debug)]
pub(crate) struct PreparedNativeKvWrite<B: Backend> {
    /// Validated destination layer index.
    layer: usize,
    /// Validated destination batch-row index.
    sequence: usize,
    /// Validated input planes `[1, heads, tokens, dim]`, one per stored plane.
    planes: Vec<Tensor<B, 4>>,
    /// Prevalidated source intervals and committed destinations.
    segments: Vec<PreparedDirectSegment>,
}

/// Source slice and one or two committed destinations for a direct append.
#[derive(Debug)]
struct PreparedDirectSegment {
    /// Contiguous interval within the validated input tensors.
    input: Range<usize>,
    /// One ring interval, or both copies for mirrored storage.
    destinations: Vec<Range<usize>>,
}

/// Fully validated physical writes for one sequence's accepted prefix.
#[derive(Debug)]
struct KvSequenceCommitPlan {
    /// Accepted staging intervals and their committed destinations.
    segments: Vec<KvCommitSegment>,
}

/// One staging interval and its one or two committed destinations.
#[derive(Debug)]
struct KvCommitSegment {
    /// Contiguous interval within the linear proposal allocation.
    input: Range<usize>,
    /// One committed interval, or both copies for mirrored storage.
    destinations: Vec<Range<usize>>,
}

/// Opaque, prevalidated commit capability consumed by the cache coordinator.
///
/// Construction checks every logical count, layer staging length, and ring
/// index before any committed tensor is mutated. Its private fields prevent a
/// caller from fabricating a partially valid commit.
#[derive(Debug)]
pub(super) struct KvCommitPlan<B: Backend> {
    /// Validated staging handles in committed-layer order; cloning keeps tensor data on device.
    layers: Vec<KvStagingRead<B>>,
    /// Accepted source intervals and physical destinations for each batch row.
    sequences: Vec<KvSequenceCommitPlan>,
}

/// Staging planes captured before any committed tensor is changed.
#[derive(Debug)]
struct KvStagingRead<B: Backend> {
    /// Proposal planes `[batch, heads, staging_capacity, dim]`.
    planes: Vec<Tensor<B, 4>>,
}

/// Preallocated native key/value (or latent) cache for one homogeneous layer group.
///
/// This type contains runtime state and intentionally does not implement
/// [`burn::module::Module`]. Direct committed writes are available for normal
/// prefill/decode. During a model-level speculative transaction, writes go to
/// the separate staging tensors and only an accepted prefix is copied into the
/// committed allocation (or, with in-place staging, proposals are written to
/// their final slots and committing only advances lengths).
///
/// Key/value groups use the `*_sequence`/`*_batch` pair methods; latent
/// groups ([`NativeKvCacheSpec::latent`]) use the `*_latent_*` methods with
/// `[rows, tokens, width]` tensors. Supplying the wrong kind returns
/// [`TensorCacheError::PlaneKindMismatch`].
#[derive(Debug)]
pub struct NativeKvCache<B: Backend> {
    /// Validated logical batch, capacity, and retention geometry.
    config: CacheConfig,
    /// Immutable layer, head, storage, dtype, and staging geometry.
    spec: NativeKvCacheSpec,
    /// Device shared by all tensors in this cache group.
    device: B::Device,
    /// Committed and optional proposed storage for every layer.
    layers: Vec<KvLayer<B>>,
    /// Proposed token counts indexed by layer and batch row.
    staged_lengths: Vec<Vec<usize>>,
    /// Whether a transaction currently owns the proposal allocations.
    staging_active: bool,
    /// Physical time-axis extent, including both mirrored halves.
    committed_time_slots: usize,
    /// Resolved ring width (group window, else config window); `None` for full storage.
    window: Option<NonZeroUsize>,
    /// Committed lengths captured when staging began, when the coordinator supplied them.
    ///
    /// In-place staging writes proposal `j` of row `b` to slot
    /// `baseline[b] + j`, and working reads check that callers pass this same
    /// committed length.
    staging_baseline: Option<Vec<usize>>,
}

impl<B: Backend> NativeKvCache<B> {
    /// Allocates zeroed committed and optional staging tensors.
    ///
    /// # Errors
    ///
    /// Returns an error for a batch/config mismatch, a ring without a sliding
    /// window, mirrored-width overflow, or a dtype unsupported by the backend.
    pub fn new(
        config: CacheConfig,
        batch_size: usize,
        spec: NativeKvCacheSpec,
        device: &B::Device,
    ) -> TensorCacheResult<Self> {
        validate_batch(config, batch_size)?;
        if !B::supports_dtype(device, spec.dtype()) {
            return Err(TensorCacheError::UnsupportedDType {
                dtype: spec.dtype(),
            });
        }
        let committed_time_slots = spec.committed_time_slots(config)?;
        let committed_shape = [
            batch_size,
            spec.key_value_heads(),
            committed_time_slots,
            spec.head_dimension(),
        ];
        let committed_elements =
            validate_tensor_shape("committed KV tensor", &committed_shape, spec.dtype())?;
        let plane_layers = spec.layers().checked_mul(spec.plane_count()).ok_or(
            TensorCacheError::TensorShapeOverflow {
                tensor: "committed KV planes",
            },
        )?;
        validate_repeated_allocation(
            "committed KV planes",
            committed_elements,
            plane_layers,
            spec.dtype(),
        )?;
        if spec.allocates_staging() {
            let staging_shape = [
                batch_size,
                spec.key_value_heads(),
                spec.staging_capacity(),
                spec.head_dimension(),
            ];
            let staging_elements =
                validate_tensor_shape("staged KV tensor", &staging_shape, spec.dtype())?;
            validate_repeated_allocation(
                "staged KV planes",
                staging_elements,
                plane_layers,
                spec.dtype(),
            )?;
        }
        let layers = B::memory_persistent_allocations(
            device,
            (spec, batch_size, committed_time_slots, device.clone()),
            |(spec, batch_size, committed_time_slots, allocation_device)| {
                let committed_shape = [
                    batch_size,
                    spec.key_value_heads(),
                    committed_time_slots,
                    spec.head_dimension(),
                ];
                let staging_shape = [
                    batch_size,
                    spec.key_value_heads(),
                    spec.staging_capacity(),
                    spec.head_dimension(),
                ];
                let zeros = |shape: [usize; 4]| -> Vec<Tensor<B, 4>> {
                    (0..spec.plane_count())
                        .map(|_| Tensor::zeros(shape, (&allocation_device, spec.dtype())))
                        .collect()
                };
                (0..spec.layers())
                    .map(|_| KvLayer {
                        planes: zeros(committed_shape),
                        staged: spec.allocates_staging().then(|| zeros(staging_shape)),
                    })
                    .collect()
            },
        );

        Ok(Self {
            config,
            spec,
            device: device.clone(),
            layers,
            staged_lengths: vec![vec![0; batch_size]; spec.layers()],
            staging_active: false,
            committed_time_slots,
            window: spec.effective_window(config),
            staging_baseline: None,
        })
    }

    /// Returns the immutable tensor geometry.
    #[must_use]
    pub const fn spec(&self) -> NativeKvCacheSpec {
        self.spec
    }

    /// Returns the logical capacity and sliding-window policy.
    #[must_use]
    pub const fn config(&self) -> CacheConfig {
        self.config
    }

    /// Returns the number of committed time-axis slots per tensor.
    #[must_use]
    pub const fn committed_time_slots(&self) -> usize {
        self.committed_time_slots
    }

    /// Returns the device to which all cache tensors are bound.
    #[must_use]
    pub const fn device(&self) -> &B::Device {
        &self.device
    }

    /// Borrows a layer's complete committed key allocation.
    ///
    /// For a latent group this is the latent plane `[batch, 1, time, width]`.
    ///
    /// # Errors
    ///
    /// Returns an error when `layer` is out of bounds.
    pub fn keys(&self, layer: usize) -> TensorCacheResult<&Tensor<B, 4>> {
        Ok(self.layer(layer)?.keys())
    }

    /// Borrows a layer's complete committed value allocation.
    ///
    /// For a latent group this is the same latent plane as [`keys`](Self::keys).
    ///
    /// # Errors
    ///
    /// Returns an error when `layer` is out of bounds.
    pub fn values(&self, layer: usize) -> TensorCacheResult<&Tensor<B, 4>> {
        Ok(self.layer(layer)?.values())
    }

    /// Writes one sequence's committed keys and values at absolute `start`.
    ///
    /// Inputs use `[1, kv_heads, tokens, head_dim]`. Ring modes apply
    /// `physical_slot = absolute_position mod window`; oversized chunks retain
    /// only their newest window. Direct writes are forbidden while speculative
    /// staging is active.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid indices, shape/dtype/device mismatch,
    /// position overflow, logical-capacity overflow, or an active transaction.
    pub fn write_sequence(
        &mut self,
        layer: usize,
        sequence: usize,
        start: usize,
        keys: Tensor<B, 4>,
        values: Tensor<B, 4>,
    ) -> TensorCacheResult<()> {
        let write = self.prepare_sequence_write(layer, sequence, start, keys, values)?;
        self.apply_prepared_writes(vec![write]);
        Ok(())
    }

    /// Writes one sequence's committed latents `[1, tokens, width]` at absolute `start`.
    ///
    /// The latent counterpart of [`write_sequence`](Self::write_sequence).
    ///
    /// # Errors
    ///
    /// Returns an error for a key/value group or the errors of
    /// [`write_sequence`](Self::write_sequence).
    pub fn write_latent_sequence(
        &mut self,
        layer: usize,
        sequence: usize,
        start: usize,
        latent: Tensor<B, 3>,
    ) -> TensorCacheResult<()> {
        let planes = self.latent_planes(latent)?;
        let write = self.prepare_planes_write(layer, sequence, start, planes)?;
        self.apply_prepared_writes(vec![write]);
        Ok(())
    }

    /// Validates and physically plans one direct committed append.
    ///
    /// Separating preparation from application lets a model validate every
    /// layer and batch-row update before publishing any of them. This is the
    /// tensor-cache analogue of a transaction prepare phase: ring arithmetic,
    /// shapes, dtypes, devices, and capacity are all recoverable errors, while
    /// applying the returned opaque value contains no fallible branch.
    ///
    /// # Errors
    ///
    /// Returns the same validation errors as [`write_sequence`](Self::write_sequence).
    pub(crate) fn prepare_sequence_write(
        &self,
        layer: usize,
        sequence: usize,
        start: usize,
        keys: Tensor<B, 4>,
        values: Tensor<B, 4>,
    ) -> TensorCacheResult<PreparedNativeKvWrite<B>> {
        let planes = self.pair_planes(keys, values)?;
        self.prepare_planes_write(layer, sequence, start, planes)
    }

    /// Plane-generic core of [`prepare_sequence_write`](Self::prepare_sequence_write).
    pub(crate) fn prepare_planes_write(
        &self,
        layer: usize,
        sequence: usize,
        start: usize,
        planes: Vec<Tensor<B, 4>>,
    ) -> TensorCacheResult<PreparedNativeKvWrite<B>> {
        if self.staging_active {
            return Err(TensorCacheError::TransactionActive {
                operation: "write committed KV",
            });
        }
        let token_count = self.validate_planes(layer, sequence, &planes)?;
        validate_write_end(start, token_count, self.config.capacity())?;
        let segments = self.prepare_direct_segments(start, token_count)?;
        Ok(PreparedNativeKvWrite {
            layer,
            sequence,
            planes,
            segments,
        })
    }

    /// Applies one or more writes returned by [`prepare_sequence_write`](Self::prepare_sequence_write).
    ///
    /// This method is intentionally infallible: callers may prepare every
    /// update in a public model invocation, perform the logical-length advance,
    /// and then publish all tensor writes without exposing an intermediate
    /// recoverable-error state.
    pub(crate) fn apply_prepared_writes(&mut self, writes: Vec<PreparedNativeKvWrite<B>>) {
        for write in writes {
            let cache_layer = &mut self.layers[write.layer];
            for segment in write.segments {
                for destination in segment.destinations {
                    for (plane, input) in cache_layer.planes.iter_mut().zip(&write.planes) {
                        assign_segment(
                            plane,
                            input,
                            write.sequence,
                            segment.input.clone(),
                            destination.clone(),
                            self.spec,
                        );
                    }
                }
            }
        }
    }

    /// Reads one sequence's retained chronological key/value interval.
    ///
    /// Full storage returns `[0, sequence_length)`. Ring storage returns
    /// `[max(0, sequence_length - window), sequence_length)`. A standard ring
    /// may return two segments; mirrored storage always returns at most one.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid index or a length beyond logical
    /// capacity.
    pub fn read_sequence(
        &self,
        layer: usize,
        sequence: usize,
        sequence_length: usize,
    ) -> TensorCacheResult<KvCacheRead<B>> {
        let cache_layer = self.layer(layer)?;
        validate_sequence(sequence, self.config.sequence_count())?;
        if sequence_length > self.config.capacity() {
            return Err(TensorCacheError::ReadExceedsCapacity {
                length: sequence_length,
                capacity: self.config.capacity(),
            });
        }
        let retained = match self.window {
            Some(window) => sequence_length.min(window.get()),
            None => sequence_length,
        };
        Ok(KvCacheRead {
            retained_positions: sequence_length - retained..sequence_length,
            segments: self.committed_segments(cache_layer, sequence, sequence_length, retained)?,
        })
    }

    /// Reads committed history followed by the physically staged proposal.
    ///
    /// `committed_length` is the stable transaction baseline for this row. For
    /// a window `W` and staged suffix `P`, the visible interval is
    /// `[max(0, committed_length + P - W), committed_length + P)`. The method
    /// trims older committed ring entries before appending the linear staging
    /// segment, so iterative drafting observes every earlier proposal token.
    /// The trimming arithmetic is the Verus-verified `ring_working_window`.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid indices, capacity overflow, or inconsistent
    /// staging allocation. Outside a transaction this is identical to
    /// [`read_sequence`](Self::read_sequence).
    pub fn read_working_sequence(
        &self,
        layer: usize,
        sequence: usize,
        committed_length: usize,
    ) -> TensorCacheResult<KvCacheRead<B>> {
        let cache_layer = self.layer(layer)?;
        validate_sequence(sequence, self.config.sequence_count())?;
        if self.staging_active {
            self.validate_baseline(sequence, committed_length)?;
        }
        let staged = self.staged_lengths[layer][sequence];
        if !self.staging_active || staged == 0 {
            return self.read_sequence(layer, sequence, committed_length);
        }
        let working_length = validate_write_end(committed_length, staged, self.config.capacity())?;
        if self.spec.in_place_staging() {
            // Proposals already sit in committed slots `[C, C + P)`.
            return self.read_sequence(layer, sequence, working_length);
        }
        let plan = match self.window {
            Some(window) => ring_working_window(committed_length, staged, window.get()),
            None => WorkingWindow {
                retained_start: 0,
                committed_keep: committed_length,
                staged_skip: 0,
            },
        };
        let mut segments =
            self.committed_segments(cache_layer, sequence, committed_length, plan.committed_keep)?;
        segments.push(read_staged_segment(
            cache_layer,
            sequence,
            committed_length + plan.staged_skip..working_length,
            plan.staged_skip..staged,
            self.spec,
        )?);
        Ok(KvCacheRead {
            retained_positions: plan.retained_start..working_length,
            segments,
        })
    }

    /// Slices the newest `keep` committed positions before `length` in chronological order.
    ///
    /// The caller guarantees `keep <= length` and, for ring storage,
    /// `keep <= window`, so every requested position is still stored. Full and
    /// mirrored storage return one contiguous segment; a standard ring returns
    /// at most two. `keep == 0` returns no segment because Burn backends do not
    /// uniformly support zero-length slices.
    fn committed_segments(
        &self,
        layer: &KvLayer<B>,
        sequence: usize,
        length: usize,
        keep: usize,
    ) -> TensorCacheResult<Vec<KvCacheReadSegment<B>>> {
        let start = length - keep;
        let Some(keep) = NonZeroUsize::new(keep) else {
            return Ok(Vec::new());
        };
        Ok(match self.spec.storage() {
            KvCacheStorage::Full => vec![read_segment(
                layer,
                sequence,
                start..length,
                start..length,
                self.spec,
            )],
            KvCacheStorage::SlidingRing => plan_ring_read(length, keep, self.window()?)?
                .segments()
                .iter()
                .map(|segment| {
                    read_segment(
                        layer,
                        sequence,
                        segment.absolute(),
                        segment.physical(),
                        self.spec,
                    )
                })
                .collect(),
            KvCacheStorage::MirroredSlidingRing => {
                // Slot `s` is duplicated at `s + W`, so any run of at most `W`
                // positions starting at slot `start mod W` is contiguous.
                let physical_start = start % self.window()?.get();
                vec![read_segment(
                    layer,
                    sequence,
                    start..length,
                    physical_start..physical_start + keep.get(),
                    self.spec,
                )]
            }
        })
    }

    /// Appends one sequence's proposal values to preallocated staging storage.
    ///
    /// Staging uses linear proposal offsets independent of ring slots. This is
    /// the central rollback invariant: rejected values never overwrite a
    /// committed slot, even when the ring was already full. With
    /// [in-place staging](NativeKvCacheSpec::with_in_place_staging), full
    /// storage instead receives proposal `j` at slot `baseline + j`, beyond
    /// every committed read.
    ///
    /// # Errors
    ///
    /// Returns an error outside an active transaction, or for invalid tensor
    /// metadata and staging- or logical-capacity overflow.
    pub fn stage_sequence(
        &mut self,
        layer: usize,
        sequence: usize,
        keys: Tensor<B, 4>,
        values: Tensor<B, 4>,
    ) -> TensorCacheResult<()> {
        let planes = self.pair_planes(keys, values)?;
        let stage = self.prepare_row_stage(layer, sequence, planes)?;
        self.apply_batch_writes(vec![stage]);
        Ok(())
    }

    /// Stages one sequence's proposal latents `[1, tokens, width]`.
    ///
    /// The latent counterpart of [`stage_sequence`](Self::stage_sequence).
    ///
    /// # Errors
    ///
    /// Returns an error for a key/value group or the errors of
    /// [`stage_sequence`](Self::stage_sequence).
    pub fn stage_latent_sequence(
        &mut self,
        layer: usize,
        sequence: usize,
        latent: Tensor<B, 3>,
    ) -> TensorCacheResult<()> {
        let planes = self.latent_planes(latent)?;
        let stage = self.prepare_row_stage(layer, sequence, planes)?;
        self.apply_batch_writes(vec![stage]);
        Ok(())
    }

    /// Returns the number of staged proposal tokens for a layer and sequence.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid layer or sequence.
    pub fn staged_length(&self, layer: usize, sequence: usize) -> TensorCacheResult<usize> {
        self.layer(layer)?;
        validate_sequence(sequence, self.config.sequence_count())?;
        Ok(self.staged_lengths[layer][sequence])
    }

    /// Returns whether the cache is accepting proposal writes.
    #[must_use]
    pub const fn has_active_staging(&self) -> bool {
        self.staging_active
    }

    /// Validates whether an absolute rewind preserves all ring-visible values.
    ///
    /// The decode state has already rejected extensions, so `requested[i] <=
    /// current[i]`. Rewinding a wrapped ring by any positive amount exposes
    /// positions that have already been overwritten; rewinding to zero is safe
    /// because the affected row is explicitly cleared by
    /// [`apply_truncate`](Self::apply_truncate). The predicate is the
    /// Verus-verified `first_unsafe_ring_rewind`, proved sound and complete
    /// against the ring eviction model.
    pub(crate) fn validate_truncate(
        &self,
        current: &[usize],
        requested: &[usize],
    ) -> TensorCacheResult<()> {
        let Some(window) = self.window else {
            return Ok(());
        };
        match first_unsafe_ring_rewind(current, requested, window.get()) {
            None => Ok(()),
            Some(sequence) => Err(TensorCacheError::UnsafeWrappedTruncate {
                storage: self.spec.storage().as_str(),
                sequence,
                current: current[sequence],
                requested: requested[sequence],
            }),
        }
    }

    /// Clears rows reset to zero after truncate validation succeeds.
    pub(crate) fn apply_truncate(&mut self, current: &[usize], requested: &[usize]) {
        for (sequence, (&from, &to)) in current.iter().zip(requested).enumerate() {
            if from != 0 && to == 0 {
                self.zero_sequence(sequence);
            }
        }
    }

    /// Enables empty proposal staging for a model-level transaction.
    ///
    /// `baseline` holds each row's committed length, which in-place staging
    /// uses as the first proposal slot and working reads verify.
    pub(crate) fn begin_staging_from(&mut self, baseline: &[usize]) -> TensorCacheResult<()> {
        if baseline.len() != self.config.sequence_count() {
            return Err(DecodeStateError::SequenceCountMismatch {
                expected: self.config.sequence_count(),
                actual: baseline.len(),
            }
            .into());
        }
        self.begin_staging_with(Some(baseline.to_vec()))
    }

    /// Enables separate-buffer staging without a known baseline (unit tests only).
    ///
    /// Working reads then trust the caller's committed length, and in-place
    /// staging is refused because it needs the baseline as its write offset.
    #[cfg(test)]
    pub(crate) fn begin_staging(&mut self) -> TensorCacheResult<()> {
        self.begin_staging_with(None)
    }

    /// Shared transition into the staging state.
    fn begin_staging_with(&mut self, baseline: Option<Vec<usize>>) -> TensorCacheResult<()> {
        if self.staging_active {
            return Err(TensorCacheError::TransactionAlreadyActive);
        }
        if self.spec.staging_capacity() == 0 {
            return Err(TensorCacheError::StagingDisabled);
        }
        for lengths in &mut self.staged_lengths {
            lengths.fill(0);
        }
        self.staging_baseline = baseline;
        self.staging_active = true;
        Ok(())
    }

    /// Checks a caller's committed length against the transaction baseline.
    fn validate_baseline(&self, sequence: usize, committed_length: usize) -> TensorCacheResult<()> {
        match &self.staging_baseline {
            Some(baseline) if baseline[sequence] != committed_length => {
                Err(TensorCacheError::TransactionCursorMismatch {
                    sequence,
                    cursor: "committed length",
                    expected: baseline[sequence],
                    actual: committed_length,
                })
            }
            _ => Ok(()),
        }
    }

    /// Builds an opaque commit plan after validating every layer and row.
    pub(super) fn prepare_staging_commit(
        &self,
        baseline: &[usize],
        proposed: &[usize],
        accepted: &[usize],
    ) -> TensorCacheResult<KvCommitPlan<B>> {
        if !self.staging_active {
            return Err(TensorCacheError::TransactionNotActive);
        }
        validate_commit_vector_lengths(self.config.sequence_count(), baseline, proposed, accepted)?;
        for layer in 0..self.spec.layers() {
            for sequence in 0..self.config.sequence_count() {
                let actual = self.staged_lengths[layer][sequence];
                if actual != proposed[sequence] {
                    return Err(TensorCacheError::IncompleteStaging {
                        kind: "native KV",
                        layer,
                        sequence,
                        expected: proposed[sequence],
                        actual,
                    });
                }
                if accepted[sequence] > proposed[sequence] {
                    return Err(TensorCacheError::AcceptedPrefixExceedsProposal {
                        sequence,
                        accepted: accepted[sequence],
                        proposed: proposed[sequence],
                    });
                }
            }
        }

        if self.spec.in_place_staging() {
            // Accepted proposals already occupy their committed slots.
            return Ok(KvCommitPlan {
                layers: Vec::new(),
                sequences: Vec::new(),
            });
        }
        let mut sequences = Vec::with_capacity(self.config.sequence_count());
        for sequence in 0..self.config.sequence_count() {
            let count = accepted[sequence];
            let staging = plan_staged_ring_transaction(baseline[sequence], proposed[sequence])?;
            let segments = match self.spec.storage() {
                KvCacheStorage::Full => {
                    let end =
                        validate_write_end(baseline[sequence], count, self.config.capacity())?;
                    (count > 0)
                        .then(|| KvCommitSegment {
                            input: 0..count,
                            destinations: core::iter::once(baseline[sequence]..end).collect(),
                        })
                        .into_iter()
                        .collect()
                }
                KvCacheStorage::SlidingRing => staging
                    .commit_prefix(count, self.window()?)?
                    .segments()
                    .iter()
                    .map(|segment| KvCommitSegment {
                        input: segment.input(),
                        destinations: core::iter::once(segment.physical()).collect(),
                    })
                    .collect(),
                KvCacheStorage::MirroredSlidingRing => {
                    staging.commit_prefix(count, self.window()?)?;
                    plan_mirrored_ring_write(baseline[sequence], count, self.window()?)?
                        .iter()
                        .map(|segment| KvCommitSegment {
                            input: segment.input(),
                            destinations: vec![segment.primary(), segment.mirror()],
                        })
                        .collect()
                }
            };
            sequences.push(KvSequenceCommitPlan { segments });
        }
        let layers = self
            .layers
            .iter()
            .map(|layer| {
                Ok(KvStagingRead {
                    planes: self.staged_planes(layer)?.to_vec(),
                })
            })
            .collect::<TensorCacheResult<_>>()?;
        Ok(KvCommitPlan { layers, sequences })
    }

    /// Applies a previously validated plan without any fallible index planning.
    pub(super) fn apply_staging_commit(&mut self, plan: KvCommitPlan<B>) {
        let spec = self.spec;
        for (layer, staging) in self.layers.iter_mut().zip(plan.layers) {
            for (sequence, sequence_plan) in plan.sequences.iter().enumerate() {
                for segment in &sequence_plan.segments {
                    for (plane, staged) in layer.planes.iter_mut().zip(&staging.planes) {
                        let accepted = staged.clone().slice([
                            sequence..sequence + 1,
                            0..spec.key_value_heads(),
                            segment.input.clone(),
                            0..spec.head_dimension(),
                        ]);
                        for destination in &segment.destinations {
                            assign_segment(
                                plane,
                                &accepted,
                                sequence,
                                0..segment.input.len(),
                                destination.clone(),
                                spec,
                            );
                        }
                    }
                }
            }
        }
        self.finish_staging();
    }

    /// Discards all staged lengths without touching committed ring slots.
    pub(crate) fn rollback_staging(&mut self) {
        self.finish_staging();
    }

    /// Clears all committed and staging tensors for session reuse.
    pub(crate) fn reset(&mut self) {
        for layer in &mut self.layers {
            let staged = layer.staged.iter_mut().flatten();
            for plane in layer.planes.iter_mut().chain(staged) {
                zero_tensor(plane);
            }
        }
        for lengths in &mut self.staged_lengths {
            lengths.fill(0);
        }
        self.staging_baseline = None;
        self.staging_active = false;
    }

    /// Reorders or duplicates the committed tensors' batch rows.
    ///
    /// `selection` holds source rows already validated by the decode state.
    /// Reordering is only legal outside a transaction, where every staged
    /// length is zero and the staging tensors are unreachable scratch space;
    /// they are therefore left untouched instead of being gathered.
    pub(crate) fn reorder_rows(&mut self, selection: &Tensor<B, 1, Int>) {
        debug_assert!(!self.staging_active);
        debug_assert!(
            self.staged_lengths
                .iter()
                .flatten()
                .all(|&length| length == 0)
        );
        for layer in &mut self.layers {
            for plane in &mut layer.planes {
                select_batch(plane, selection);
            }
        }
    }

    /// Returns a checked immutable layer reference.
    fn layer(&self, layer: usize) -> TensorCacheResult<&KvLayer<B>> {
        self.layers
            .get(layer)
            .ok_or(TensorCacheError::LayerOutOfBounds {
                layer,
                layers: self.layers.len(),
            })
    }

    /// Returns the resolved ring width required by either ring representation.
    fn window(&self) -> TensorCacheResult<NonZeroUsize> {
        self.window.ok_or(TensorCacheError::SlidingWindowRequired {
            storage: self.spec.storage().as_str(),
        })
    }

    /// Converts a key/value input pair into planes, rejecting latent groups.
    pub(crate) fn pair_planes(
        &self,
        keys: Tensor<B, 4>,
        values: Tensor<B, 4>,
    ) -> TensorCacheResult<Vec<Tensor<B, 4>>> {
        if self.spec.is_latent() {
            return Err(TensorCacheError::PlaneKindMismatch {
                expected: "one latent tensor; use the latent methods",
            });
        }
        Ok(vec![keys, values])
    }

    /// Converts a `[rows, tokens, width]` latent input into its single plane.
    pub(crate) fn latent_planes(
        &self,
        latent: Tensor<B, 3>,
    ) -> TensorCacheResult<Vec<Tensor<B, 4>>> {
        if !self.spec.is_latent() {
            return Err(TensorCacheError::PlaneKindMismatch {
                expected: "a key/value pair; use the key/value methods",
            });
        }
        // `[rows, tokens, width]` -> `[rows, 1, tokens, width]` adds the
        // singleton head axis without moving data.
        Ok(vec![latent.unsqueeze_dim(1)])
    }

    /// Validates one row's input planes `[1, heads, tokens, dim]` and returns their token extent.
    fn validate_planes(
        &self,
        layer: usize,
        sequence: usize,
        planes: &[Tensor<B, 4>],
    ) -> TensorCacheResult<usize> {
        self.layer(layer)?;
        validate_sequence(sequence, self.config.sequence_count())?;
        validate_plane_count(self.spec, planes.len())?;
        let token_count = planes[0].dims()[2];
        let expected = [
            1,
            self.spec.key_value_heads(),
            token_count,
            self.spec.head_dimension(),
        ];
        for (plane, name) in planes.iter().zip(PLANE_NAMES) {
            validate_tensor(plane, name, expected, self.spec.dtype(), &self.device)?;
        }
        Ok(token_count)
    }

    /// Borrows a layer's staging planes, which must exist with every plane.
    fn staged_planes<'a>(&self, layer: &'a KvLayer<B>) -> TensorCacheResult<&'a [Tensor<B, 4>]> {
        match &layer.staged {
            Some(planes) if planes.len() == self.spec.plane_count() => Ok(planes),
            _ => Err(TensorCacheError::StagingDisabled),
        }
    }

    /// Computes every physical destination for a validated direct append.
    fn prepare_direct_segments(
        &self,
        start: usize,
        token_count: usize,
    ) -> TensorCacheResult<Vec<PreparedDirectSegment>> {
        if token_count == 0 {
            return Ok(Vec::new());
        }
        match self.spec.storage() {
            KvCacheStorage::Full => Ok(vec![PreparedDirectSegment {
                input: 0..token_count,
                destinations: core::iter::once(start..start + token_count).collect(),
            }]),
            KvCacheStorage::SlidingRing => {
                let window = self.window()?;
                Ok(plan_ring_write(start, token_count, window)?
                    .segments()
                    .iter()
                    .map(|segment| PreparedDirectSegment {
                        input: segment.input(),
                        destinations: core::iter::once(segment.physical()).collect(),
                    })
                    .collect())
            }
            KvCacheStorage::MirroredSlidingRing => {
                let window = self.window()?;
                Ok(plan_mirrored_ring_write(start, token_count, window)?
                    .into_iter()
                    .map(|segment| PreparedDirectSegment {
                        input: segment.input(),
                        destinations: vec![segment.primary(), segment.mirror()],
                    })
                    .collect())
            }
        }
    }

    /// Clears one batch row in all committed and staging tensors.
    fn zero_sequence(&mut self, sequence: usize) {
        let spec = self.spec;
        let committed_slots = self.committed_time_slots;
        let staging_slots = spec.staging_capacity();
        for layer in &mut self.layers {
            for plane in &mut layer.planes {
                zero_4d_row(
                    plane,
                    sequence,
                    spec.key_value_heads(),
                    committed_slots,
                    spec.head_dimension(),
                );
            }
            for plane in layer.staged.iter_mut().flatten() {
                zero_4d_row(
                    plane,
                    sequence,
                    spec.key_value_heads(),
                    staging_slots,
                    spec.head_dimension(),
                );
            }
        }
    }

    /// Ends staging and makes all staged bytes logically unreachable.
    fn finish_staging(&mut self) {
        for lengths in &mut self.staged_lengths {
            lengths.fill(0);
        }
        self.staging_baseline = None;
        self.staging_active = false;
    }
}

/// Creates a non-zero dimension or a field-specific error.
fn non_zero(field: &'static str, value: usize) -> TensorCacheResult<NonZeroUsize> {
    NonZeroUsize::new(value).ok_or(TensorCacheError::ZeroDimension { field })
}

/// Rejects integer, boolean, and quantized runtime cache tensors.
fn validate_float_dtype(dtype: DType) -> TensorCacheResult<()> {
    if dtype.is_float() {
        Ok(())
    } else {
        Err(TensorCacheError::NonFloatDType { dtype })
    }
}

/// Ensures the caller's batch matches the metadata sequence count.
pub(crate) fn validate_batch(config: CacheConfig, batch_size: usize) -> TensorCacheResult<()> {
    if config.sequence_count() == batch_size {
        Ok(())
    } else {
        Err(TensorCacheError::BatchSizeMismatch {
            configured: config.sequence_count(),
            requested: batch_size,
        })
    }
}

/// Validates one sequence index before any backend indexing call.
fn validate_sequence(sequence: usize, batch_size: usize) -> TensorCacheResult<()> {
    if sequence < batch_size {
        Ok(())
    } else {
        Err(TensorCacheError::SequenceOutOfBounds {
            sequence,
            batch_size,
        })
    }
}

/// Checks absolute write arithmetic and logical capacity.
fn validate_write_end(start: usize, tokens: usize, capacity: usize) -> TensorCacheResult<usize> {
    let end = start
        .checked_add(tokens)
        .ok_or(TensorCacheError::WritePositionOverflow { start, tokens })?;
    if end > capacity {
        Err(TensorCacheError::WriteExceedsCapacity {
            start,
            end,
            capacity,
        })
    } else {
        Ok(end)
    }
}

/// Checks shape, dtype, and device before invoking a backend assignment.
fn validate_tensor<B: Backend>(
    tensor: &Tensor<B, 4>,
    name: &'static str,
    expected: [usize; 4],
    dtype: DType,
    device: &B::Device,
) -> TensorCacheResult<()> {
    let actual = tensor.dims();
    if actual != expected {
        return Err(TensorCacheError::ShapeMismatch {
            tensor: name,
            expected: expected.to_vec(),
            actual: actual.to_vec(),
        });
    }
    if tensor.dtype() != dtype {
        return Err(TensorCacheError::DTypeMismatch {
            tensor: name,
            expected: dtype,
            actual: tensor.dtype(),
        });
    }
    if tensor.device() != *device {
        return Err(TensorCacheError::DeviceMismatch { tensor: name });
    }
    Ok(())
}

/// Assigns one non-empty source interval to one physical time interval.
fn assign_segment<B: Backend>(
    destination: &mut Tensor<B, 4>,
    source: &Tensor<B, 4>,
    sequence: usize,
    input: Range<usize>,
    physical: Range<usize>,
    spec: NativeKvCacheSpec,
) {
    let values = source.clone().slice([
        0..1,
        0..spec.key_value_heads(),
        input,
        0..spec.head_dimension(),
    ]);
    destination.inplace(|tensor| {
        tensor.slice_assign(
            [
                sequence..sequence + 1,
                0..spec.key_value_heads(),
                physical,
                0..spec.head_dimension(),
            ],
            values,
        )
    });
}

/// Constructs one read segment from committed physical slots.
fn read_segment<B: Backend>(
    layer: &KvLayer<B>,
    sequence: usize,
    absolute_positions: Range<usize>,
    physical: Range<usize>,
    spec: NativeKvCacheSpec,
) -> KvCacheReadSegment<B> {
    let slices = [
        sequence..sequence + 1,
        0..spec.key_value_heads(),
        physical,
        0..spec.head_dimension(),
    ];
    KvCacheReadSegment {
        absolute_positions,
        keys: layer.keys().clone().slice(slices.clone()),
        values: layer.values().clone().slice(slices),
    }
}

/// Constructs one read segment from linear proposal staging storage.
fn read_staged_segment<B: Backend>(
    layer: &KvLayer<B>,
    sequence: usize,
    absolute_positions: Range<usize>,
    staged: Range<usize>,
    spec: NativeKvCacheSpec,
) -> TensorCacheResult<KvCacheReadSegment<B>> {
    let slices = [
        sequence..sequence + 1,
        0..spec.key_value_heads(),
        staged,
        0..spec.head_dimension(),
    ];
    let planes = match &layer.staged {
        Some(planes) if planes.len() == spec.plane_count() => planes,
        _ => return Err(TensorCacheError::StagingDisabled),
    };
    let keys = planes[0].clone().slice(slices.clone());
    let values = planes
        .get(1)
        .map_or_else(|| keys.clone(), |plane| plane.clone().slice(slices));
    Ok(KvCacheReadSegment {
        absolute_positions,
        keys,
        values,
    })
}

/// Diagnostic names of the planes, in storage order.
const PLANE_NAMES: [&str; 2] = ["keys", "values"];

/// Requires one input tensor per stored plane.
fn validate_plane_count(spec: NativeKvCacheSpec, actual: usize) -> TensorCacheResult<()> {
    if actual == spec.plane_count() {
        Ok(())
    } else {
        Err(TensorCacheError::PlaneKindMismatch {
            expected: if spec.is_latent() {
                "one latent tensor; use the latent methods"
            } else {
                "a key/value pair; use the key/value methods"
            },
        })
    }
}

/// Validates all per-sequence vectors before direct indexing during planning.
fn validate_commit_vector_lengths(
    expected: usize,
    baseline: &[usize],
    proposed: &[usize],
    accepted: &[usize],
) -> TensorCacheResult<()> {
    for actual in [baseline.len(), proposed.len(), accepted.len()] {
        if actual != expected {
            return Err(DecodeStateError::SequenceCountMismatch { expected, actual }.into());
        }
    }
    Ok(())
}

/// Validates fixed-size beam reordering indices.
pub(crate) fn validate_reorder_indices(
    indices: &[usize],
    batch_size: usize,
) -> TensorCacheResult<()> {
    if indices.len() != batch_size {
        return Err(TensorCacheError::BatchSizeMismatch {
            configured: batch_size,
            requested: indices.len(),
        });
    }
    for &sequence in indices {
        validate_sequence(sequence, batch_size)?;
    }
    Ok(())
}

/// Builds the device-local integer tensor used by Burn batch selection.
pub(crate) fn selection_tensor<B: Backend>(
    indices: &[usize],
    device: &B::Device,
) -> TensorCacheResult<Tensor<B, 1, Int>> {
    let values: Vec<i32> = indices
        .iter()
        .map(|&index| {
            i32::try_from(index).map_err(|_| TensorCacheError::BatchIndexOverflow { index })
        })
        .collect::<TensorCacheResult<_>>()?;
    Ok(Tensor::from_ints(
        TensorData::new(values, [indices.len()]),
        device,
    ))
}

/// Applies the same batch gather to one rank-four state tensor.
fn select_batch<B: Backend>(tensor: &mut Tensor<B, 4>, indices: &Tensor<B, 1, Int>) {
    tensor.inplace(|value| value.select(0, indices.clone()));
}

/// Replaces a tensor with same-shape zeros, preserving dtype and device.
fn zero_tensor<B: Backend, const D: usize>(tensor: &mut Tensor<B, D>) {
    tensor.inplace(|value| value.zeros_like());
}

/// Clears one batch row of a rank-four tensor without changing its shape.
fn zero_4d_row<B: Backend>(
    tensor: &mut Tensor<B, 4>,
    sequence: usize,
    heads: usize,
    time: usize,
    dimension: usize,
) {
    let zeros = Tensor::zeros(
        [1, heads, time, dimension],
        (&tensor.device(), tensor.dtype()),
    );
    tensor.inplace(|value| {
        value.slice_assign(
            [sequence..sequence + 1, 0..heads, 0..time, 0..dimension],
            zeros,
        )
    });
}

#[cfg(test)]
#[path = "kv_tests.rs"]
mod tests;
