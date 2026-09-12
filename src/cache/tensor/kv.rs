//! Native key/value attention cache tensors.
//!
//! Every committed layer uses `[batch, kv_heads, time, head_dim]`. Keeping time
//! next to the feature dimension makes a chronological key/value slice directly
//! consumable by attention kernels. Full storage allocates `time = capacity`;
//! a standard sliding ring allocates `time = window`; a mirrored ring allocates
//! `time = 2 * window` and writes each slot twice so the retained suffix is one
//! contiguous tensor view.

use core::num::NonZeroUsize;
use core::ops::Range;

use burn::tensor::{DType, Int, Tensor, TensorData, backend::Backend};

use crate::cache::{
    CacheConfig, DecodeStateError, plan_mirrored_ring_read, plan_mirrored_ring_write,
    plan_ring_read, plan_ring_write, plan_staged_ring_transaction,
};

use super::{
    TensorCacheError, TensorCacheResult, validate_repeated_allocation, validate_tensor_shape,
};

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
        })
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

/// Committed tensors and paired optional staging allocations for one attention layer.
#[derive(Debug)]
struct KvLayer<B: Backend> {
    /// Committed keys shaped `[batch, kv_heads, physical_time, head_dim]`.
    keys: Tensor<B, 4>,
    /// Committed values with the same geometry as the keys.
    values: Tensor<B, 4>,
    /// Linear proposal keys, allocated together with staged values.
    staged_keys: Option<Tensor<B, 4>>,
    /// Linear proposal values, allocated together with staged keys.
    staged_values: Option<Tensor<B, 4>>,
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
    /// Validated input keys shaped `[1, kv_heads, tokens, head_dim]`.
    keys: Tensor<B, 4>,
    /// Validated input values with matching shape, dtype, and device.
    values: Tensor<B, 4>,
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

/// Paired staging allocations captured before any committed tensor is changed.
#[derive(Debug)]
struct KvStagingRead<B: Backend> {
    /// Proposed keys shaped `[batch, kv_heads, staging_capacity, head_dimension]`.
    keys: Tensor<B, 4>,
    /// Proposed values with the same geometry and dtype as the keys.
    values: Tensor<B, 4>,
}

/// Preallocated native key/value cache for one homogeneous layer group.
///
/// This type contains runtime state and intentionally does not implement
/// [`burn::module::Module`]. Direct committed writes are available for normal
/// prefill/decode. During a model-level speculative transaction, writes go to
/// the separate staging tensors and only an accepted prefix is copied into the
/// committed allocation.
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
        let committed_time_slots = committed_time_slots(config, spec.storage())?;
        let committed_shape = [
            batch_size,
            spec.key_value_heads(),
            committed_time_slots,
            spec.head_dimension(),
        ];
        let committed_elements =
            validate_tensor_shape("committed KV tensor", &committed_shape, spec.dtype())?;
        validate_repeated_allocation(
            "committed KV keys",
            committed_elements,
            spec.layers(),
            spec.dtype(),
        )?;
        validate_repeated_allocation(
            "committed KV values",
            committed_elements,
            spec.layers(),
            spec.dtype(),
        )?;
        if spec.staging_capacity() > 0 {
            let staging_shape = [
                batch_size,
                spec.key_value_heads(),
                spec.staging_capacity(),
                spec.head_dimension(),
            ];
            let staging_elements =
                validate_tensor_shape("staged KV tensor", &staging_shape, spec.dtype())?;
            validate_repeated_allocation(
                "staged KV keys",
                staging_elements,
                spec.layers(),
                spec.dtype(),
            )?;
            validate_repeated_allocation(
                "staged KV values",
                staging_elements,
                spec.layers(),
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
                (0..spec.layers())
                    .map(|_| KvLayer {
                        keys: Tensor::zeros(committed_shape, (&allocation_device, spec.dtype())),
                        values: Tensor::zeros(committed_shape, (&allocation_device, spec.dtype())),
                        staged_keys: (spec.staging_capacity() > 0).then(|| {
                            Tensor::zeros(staging_shape, (&allocation_device, spec.dtype()))
                        }),
                        staged_values: (spec.staging_capacity() > 0).then(|| {
                            Tensor::zeros(staging_shape, (&allocation_device, spec.dtype()))
                        }),
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
    /// # Errors
    ///
    /// Returns an error when `layer` is out of bounds.
    pub fn keys(&self, layer: usize) -> TensorCacheResult<&Tensor<B, 4>> {
        Ok(&self.layer(layer)?.keys)
    }

    /// Borrows a layer's complete committed value allocation.
    ///
    /// # Errors
    ///
    /// Returns an error when `layer` is out of bounds.
    pub fn values(&self, layer: usize) -> TensorCacheResult<&Tensor<B, 4>> {
        Ok(&self.layer(layer)?.values)
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
        if self.staging_active {
            return Err(TensorCacheError::TransactionActive {
                operation: "write committed KV",
            });
        }
        let token_count = self.validate_pair(layer, sequence, &keys, &values)?;
        validate_write_end(start, token_count, self.config.capacity())?;
        let segments = self.prepare_direct_segments(start, token_count)?;
        Ok(PreparedNativeKvWrite {
            layer,
            sequence,
            keys,
            values,
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
                    assign_segment(
                        &mut cache_layer.keys,
                        &write.keys,
                        write.sequence,
                        segment.input.clone(),
                        destination.clone(),
                        self.spec,
                    );
                    assign_segment(
                        &mut cache_layer.values,
                        &write.values,
                        write.sequence,
                        segment.input.clone(),
                        destination,
                        self.spec,
                    );
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
        let layer = self.layer(layer)?;
        validate_sequence(sequence, self.config.sequence_count())?;
        if sequence_length > self.config.capacity() {
            return Err(TensorCacheError::ReadExceedsCapacity {
                length: sequence_length,
                capacity: self.config.capacity(),
            });
        }
        if sequence_length == 0 {
            return Ok(KvCacheRead {
                retained_positions: 0..0,
                segments: Vec::new(),
            });
        }

        match self.spec.storage() {
            KvCacheStorage::Full => Ok(KvCacheRead {
                retained_positions: 0..sequence_length,
                segments: vec![read_segment(
                    layer,
                    sequence,
                    0..sequence_length,
                    0..sequence_length,
                    self.spec,
                )],
            }),
            KvCacheStorage::SlidingRing => {
                let window = self.window()?;
                let plan = plan_ring_read(sequence_length, window, window)?;
                let segments = plan
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
                    .collect();
                Ok(KvCacheRead {
                    retained_positions: plan.retained_positions(),
                    segments,
                })
            }
            KvCacheStorage::MirroredSlidingRing => {
                let window = self.window()?;
                let plan = plan_mirrored_ring_read(sequence_length, window)?;
                Ok(KvCacheRead {
                    retained_positions: plan.retained_positions(),
                    segments: vec![read_segment(
                        layer,
                        sequence,
                        plan.retained_positions(),
                        plan.physical(),
                        self.spec,
                    )],
                })
            }
        }
    }

    /// Reads committed history followed by the physically staged proposal.
    ///
    /// `committed_length` is the stable transaction baseline for this row. For
    /// a window `W` and staged suffix `P`, the visible interval is
    /// `[max(0, committed_length + P - W), committed_length + P)`. The method
    /// trims older committed ring entries before appending the linear staging
    /// segment, so iterative drafting observes every earlier proposal token.
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
        let staged = self.staged_lengths[layer][sequence];
        if !self.staging_active || staged == 0 {
            return self.read_sequence(layer, sequence, committed_length);
        }
        let working_length = validate_write_end(committed_length, staged, self.config.capacity())?;
        let (committed_keep, staged_start, retained_start) = match self.spec.storage() {
            KvCacheStorage::Full => (committed_length, 0, 0),
            KvCacheStorage::SlidingRing | KvCacheStorage::MirroredSlidingRing => {
                let window = self.window()?.get();
                let staged_keep = staged.min(window);
                let committed_keep = committed_length.min(window - staged_keep);
                (
                    committed_keep,
                    staged - staged_keep,
                    working_length - committed_keep - staged_keep,
                )
            }
        };

        let mut segments = Vec::with_capacity(3);
        if let Some(keep) = NonZeroUsize::new(committed_keep) {
            let absolute_start = committed_length - committed_keep;
            match self.spec.storage() {
                KvCacheStorage::Full => segments.push(read_segment(
                    cache_layer,
                    sequence,
                    absolute_start..committed_length,
                    absolute_start..committed_length,
                    self.spec,
                )),
                KvCacheStorage::SlidingRing => {
                    let window = self.window()?;
                    let plan = plan_ring_read(committed_length, keep, window)?;
                    segments.extend(plan.segments().iter().map(|segment| {
                        read_segment(
                            cache_layer,
                            sequence,
                            segment.absolute(),
                            segment.physical(),
                            self.spec,
                        )
                    }));
                }
                KvCacheStorage::MirroredSlidingRing => {
                    let window = self.window()?.get();
                    let physical_start = absolute_start % window;
                    segments.push(read_segment(
                        cache_layer,
                        sequence,
                        absolute_start..committed_length,
                        physical_start..physical_start + committed_keep,
                        self.spec,
                    ));
                }
            }
        }
        let staged_kept = staged - staged_start;
        if staged_kept > 0 {
            segments.push(read_staged_segment(
                cache_layer,
                sequence,
                committed_length + staged_start..working_length,
                staged_start..staged,
                self.spec,
            )?);
        }
        Ok(KvCacheRead {
            retained_positions: retained_start..working_length,
            segments,
        })
    }

    /// Appends one sequence's proposal values to preallocated staging storage.
    ///
    /// Staging uses linear proposal offsets independent of ring slots. This is
    /// the central rollback invariant: rejected values never overwrite a
    /// committed slot, even when the ring was already full.
    ///
    /// # Errors
    ///
    /// Returns an error outside an active transaction, or for invalid tensor
    /// metadata and staging-capacity overflow.
    pub fn stage_sequence(
        &mut self,
        layer: usize,
        sequence: usize,
        keys: Tensor<B, 4>,
        values: Tensor<B, 4>,
    ) -> TensorCacheResult<()> {
        if !self.staging_active {
            return Err(TensorCacheError::TransactionNotActive);
        }
        let token_count = self.validate_pair(layer, sequence, &keys, &values)?;
        let start = self.staged_lengths[layer][sequence];
        let end =
            start
                .checked_add(token_count)
                .ok_or(TensorCacheError::StagingCapacityExceeded {
                    requested: usize::MAX,
                    capacity: self.spec.staging_capacity(),
                })?;
        if end > self.spec.staging_capacity() {
            return Err(TensorCacheError::StagingCapacityExceeded {
                requested: end,
                capacity: self.spec.staging_capacity(),
            });
        }
        if token_count == 0 {
            return Ok(());
        }

        let spec = self.spec;
        let cache_layer = self.layer_mut(layer)?;
        let staged_keys = cache_layer
            .staged_keys
            .as_mut()
            .ok_or(TensorCacheError::StagingDisabled)?;
        let staged_values = cache_layer
            .staged_values
            .as_mut()
            .ok_or(TensorCacheError::StagingDisabled)?;
        assign_segment(
            staged_keys,
            &keys,
            sequence,
            0..token_count,
            start..end,
            spec,
        );
        assign_segment(
            staged_values,
            &values,
            sequence,
            0..token_count,
            start..end,
            spec,
        );
        self.staged_lengths[layer][sequence] = end;
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
    /// Rewinding a wrapped ring by any positive amount exposes positions that
    /// have already been overwritten. Rewinding to zero is safe because the
    /// affected row is explicitly cleared by [`apply_truncate`](Self::apply_truncate).
    pub(crate) fn validate_truncate(
        &self,
        current: &[usize],
        requested: &[usize],
    ) -> TensorCacheResult<()> {
        if self.spec.storage() == KvCacheStorage::Full {
            return Ok(());
        }
        let window = self.window()?.get();
        for (sequence, (&from, &to)) in current.iter().zip(requested).enumerate() {
            if to < from && to != 0 && from > window {
                return Err(TensorCacheError::UnsafeWrappedTruncate {
                    storage: self.spec.storage().as_str(),
                    sequence,
                    current: from,
                    requested: to,
                });
            }
        }
        Ok(())
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
    pub(crate) fn begin_staging(&mut self) -> TensorCacheResult<()> {
        if self.staging_active {
            return Err(TensorCacheError::TransactionAlreadyActive);
        }
        if self.spec.staging_capacity() == 0 {
            return Err(TensorCacheError::StagingDisabled);
        }
        for lengths in &mut self.staged_lengths {
            lengths.fill(0);
        }
        self.staging_active = true;
        Ok(())
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
                    keys: layer
                        .staged_keys
                        .as_ref()
                        .ok_or(TensorCacheError::StagingDisabled)?
                        .clone(),
                    values: layer
                        .staged_values
                        .as_ref()
                        .ok_or(TensorCacheError::StagingDisabled)?
                        .clone(),
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
                    let staged_keys = staging.keys.clone().slice([
                        sequence..sequence + 1,
                        0..spec.key_value_heads(),
                        segment.input.clone(),
                        0..spec.head_dimension(),
                    ]);
                    let staged_values = staging.values.clone().slice([
                        sequence..sequence + 1,
                        0..spec.key_value_heads(),
                        segment.input.clone(),
                        0..spec.head_dimension(),
                    ]);
                    for destination in &segment.destinations {
                        assign_segment(
                            &mut layer.keys,
                            &staged_keys,
                            sequence,
                            0..segment.input.len(),
                            destination.clone(),
                            spec,
                        );
                        assign_segment(
                            &mut layer.values,
                            &staged_values,
                            sequence,
                            0..segment.input.len(),
                            destination.clone(),
                            spec,
                        );
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
            zero_tensor(&mut layer.keys);
            zero_tensor(&mut layer.values);
            if let Some(tensor) = &mut layer.staged_keys {
                zero_tensor(tensor);
            }
            if let Some(tensor) = &mut layer.staged_values {
                zero_tensor(tensor);
            }
        }
        for lengths in &mut self.staged_lengths {
            lengths.fill(0);
        }
        self.staging_active = false;
    }

    /// Reorders or duplicates every tensor's batch dimension.
    pub(crate) fn reorder(&mut self, indices: &[usize]) -> TensorCacheResult<()> {
        if self.staging_active {
            return Err(TensorCacheError::TransactionActive {
                operation: "reorder KV",
            });
        }
        validate_reorder_indices(indices, self.config.sequence_count())?;
        let selection = selection_tensor::<B>(indices, &self.device)?;
        for layer in &mut self.layers {
            select_batch(&mut layer.keys, &selection);
            select_batch(&mut layer.values, &selection);
            if let Some(tensor) = &mut layer.staged_keys {
                select_batch(tensor, &selection);
            }
            if let Some(tensor) = &mut layer.staged_values {
                select_batch(tensor, &selection);
            }
        }
        for lengths in &mut self.staged_lengths {
            let old = lengths.clone();
            *lengths = indices.iter().map(|&index| old[index]).collect();
        }
        Ok(())
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

    /// Returns a checked mutable layer reference.
    fn layer_mut(&mut self, layer: usize) -> TensorCacheResult<&mut KvLayer<B>> {
        let layers = self.layers.len();
        self.layers
            .get_mut(layer)
            .ok_or(TensorCacheError::LayerOutOfBounds { layer, layers })
    }

    /// Returns the configured window required by either ring representation.
    fn window(&self) -> TensorCacheResult<NonZeroUsize> {
        self.config
            .sliding_window()
            .ok_or(TensorCacheError::SlidingWindowRequired {
                storage: self.spec.storage().as_str(),
            })
    }

    /// Validates an input key/value pair and returns its token extent.
    fn validate_pair(
        &self,
        layer: usize,
        sequence: usize,
        keys: &Tensor<B, 4>,
        values: &Tensor<B, 4>,
    ) -> TensorCacheResult<usize> {
        self.layer(layer)?;
        validate_sequence(sequence, self.config.sequence_count())?;
        let token_count = keys.dims()[2];
        let expected = [
            1,
            self.spec.key_value_heads(),
            token_count,
            self.spec.head_dimension(),
        ];
        validate_tensor(keys, "keys", expected, self.spec.dtype(), &self.device)?;
        validate_tensor(values, "values", expected, self.spec.dtype(), &self.device)?;
        Ok(token_count)
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
                let window = self.config.sliding_window().ok_or(
                    TensorCacheError::SlidingWindowRequired {
                        storage: self.spec.storage().as_str(),
                    },
                )?;
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
                let window = self.config.sliding_window().ok_or(
                    TensorCacheError::SlidingWindowRequired {
                        storage: self.spec.storage().as_str(),
                    },
                )?;
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
            zero_4d_row(
                &mut layer.keys,
                sequence,
                spec.key_value_heads(),
                committed_slots,
                spec.head_dimension(),
            );
            zero_4d_row(
                &mut layer.values,
                sequence,
                spec.key_value_heads(),
                committed_slots,
                spec.head_dimension(),
            );
            if let Some(tensor) = &mut layer.staged_keys {
                zero_4d_row(
                    tensor,
                    sequence,
                    spec.key_value_heads(),
                    staging_slots,
                    spec.head_dimension(),
                );
            }
            if let Some(tensor) = &mut layer.staged_values {
                zero_4d_row(
                    tensor,
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

/// Computes the committed tensor's physical time-axis extent.
fn committed_time_slots(config: CacheConfig, storage: KvCacheStorage) -> TensorCacheResult<usize> {
    match storage {
        KvCacheStorage::Full => Ok(config.capacity()),
        KvCacheStorage::SlidingRing => config.sliding_window().map(NonZeroUsize::get).ok_or(
            TensorCacheError::SlidingWindowRequired {
                storage: storage.as_str(),
            },
        ),
        KvCacheStorage::MirroredSlidingRing => {
            let window =
                config
                    .sliding_window()
                    .ok_or(TensorCacheError::SlidingWindowRequired {
                        storage: storage.as_str(),
                    })?;
            window
                .get()
                .checked_mul(2)
                .ok_or(TensorCacheError::MirroredCapacityOverflow {
                    window: window.get(),
                })
        }
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
        keys: layer.keys.clone().slice(slices.clone()),
        values: layer.values.clone().slice(slices),
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
    let keys = layer
        .staged_keys
        .as_ref()
        .ok_or(TensorCacheError::StagingDisabled)?
        .clone()
        .slice(slices.clone());
    let values = layer
        .staged_values
        .as_ref()
        .ok_or(TensorCacheError::StagingDisabled)?
        .clone()
        .slice(slices);
    Ok(KvCacheReadSegment {
        absolute_positions,
        keys,
        values,
    })
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
