//! Recurrent matrix and convolution state for linear-time sequence mixers.
//!
//! Delta-style recurrence updates a matrix state such as
//! `S_t = decay_t * S_(t-1) + update_t`; selective state-space layers likewise
//! carry a terminal matrix/vector plus a causal-convolution history. Unlike a
//! KV cache, the terminal state is not invertible. Speculative decoding must
//! therefore stage the state *after every proposed token* and select the
//! accepted terminal snapshot, rather than trying to subtract rejected updates.

use core::num::NonZeroUsize;

use burn::tensor::{DType, Int, Tensor, backend::Backend};

use crate::cache::CacheConfig;

use super::kv::{selection_tensor, validate_batch, validate_reorder_indices};
use super::{TensorCacheError, TensorCacheResult};

/// Validated geometry for DeltaNet/Mamba2-style recurrent layer state.
///
/// Each committed layer owns a matrix tensor shaped
/// `[batch, state_heads, state_rows, state_columns]` and a causal convolution
/// tensor shaped `[batch, convolution_channels, convolution_width]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RecurrentStateSpec {
    layers: NonZeroUsize,
    state_heads: NonZeroUsize,
    state_rows: NonZeroUsize,
    state_columns: NonZeroUsize,
    convolution_channels: NonZeroUsize,
    convolution_width: NonZeroUsize,
    dtype: DType,
    staging_capacity: usize,
}

impl RecurrentStateSpec {
    /// Creates recurrent state geometry without speculative snapshots.
    ///
    /// # Errors
    ///
    /// Returns an error when any dimension is zero or `dtype` is not a normal
    /// floating-point type.
    pub fn new(
        layers: usize,
        state_heads: usize,
        state_rows: usize,
        state_columns: usize,
        convolution_channels: usize,
        convolution_width: usize,
        dtype: DType,
    ) -> TensorCacheResult<Self> {
        if !dtype.is_float() {
            return Err(TensorCacheError::NonFloatDType { dtype });
        }
        Ok(Self {
            layers: non_zero("layers", layers)?,
            state_heads: non_zero("state_heads", state_heads)?,
            state_rows: non_zero("state_rows", state_rows)?,
            state_columns: non_zero("state_columns", state_columns)?,
            convolution_channels: non_zero("convolution_channels", convolution_channels)?,
            convolution_width: non_zero("convolution_width", convolution_width)?,
            dtype,
            staging_capacity: 0,
        })
    }

    /// Sets the number of post-token state snapshots retained per proposal.
    #[must_use]
    pub const fn with_staging_capacity(mut self, staging_capacity: usize) -> Self {
        self.staging_capacity = staging_capacity;
        self
    }

    /// Returns the number of recurrent layers in this homogeneous group.
    #[must_use]
    pub const fn layers(self) -> usize {
        self.layers.get()
    }

    /// Returns the number of independent recurrent matrices per layer.
    #[must_use]
    pub const fn state_heads(self) -> usize {
        self.state_heads.get()
    }

    /// Returns each recurrent matrix's row count.
    #[must_use]
    pub const fn state_rows(self) -> usize {
        self.state_rows.get()
    }

    /// Returns each recurrent matrix's column count.
    #[must_use]
    pub const fn state_columns(self) -> usize {
        self.state_columns.get()
    }

    /// Returns the channel count of the causal-convolution history.
    #[must_use]
    pub const fn convolution_channels(self) -> usize {
        self.convolution_channels.get()
    }

    /// Returns the retained convolution history width.
    #[must_use]
    pub const fn convolution_width(self) -> usize {
        self.convolution_width.get()
    }

    /// Returns the runtime floating-point dtype of all state tensors.
    #[must_use]
    pub const fn dtype(self) -> DType {
        self.dtype
    }

    /// Returns the maximum speculative snapshot count per sequence.
    #[must_use]
    pub const fn staging_capacity(self) -> usize {
        self.staging_capacity
    }

    /// Returns the committed matrix tensor shape for a batch.
    #[must_use]
    pub const fn matrix_shape(self, batch_size: usize) -> [usize; 4] {
        [
            batch_size,
            self.state_heads(),
            self.state_rows(),
            self.state_columns(),
        ]
    }

    /// Returns the committed causal-convolution tensor shape for a batch.
    #[must_use]
    pub const fn convolution_shape(self, batch_size: usize) -> [usize; 3] {
        [
            batch_size,
            self.convolution_channels(),
            self.convolution_width(),
        ]
    }
}

#[derive(Debug)]
struct RecurrentLayer<B: Backend> {
    matrix: Tensor<B, 4>,
    convolution: Tensor<B, 3>,
    staged_matrix: Option<Tensor<B, 5>>,
    staged_convolution: Option<Tensor<B, 4>>,
}

/// Preallocated terminal recurrent state and speculative state snapshots.
///
/// The speculative matrix layout is
/// `[batch, proposal, state_heads, state_rows, state_columns]`; convolution
/// snapshots use `[batch, proposal, channels, width]`. Snapshot `i` represents
/// state *after* proposal token `i`, so accepting `a > 0` selects `a - 1`.
#[derive(Debug)]
pub struct RecurrentStateCache<B: Backend> {
    config: CacheConfig,
    spec: RecurrentStateSpec,
    device: B::Device,
    layers: Vec<RecurrentLayer<B>>,
    staged_lengths: Vec<Vec<usize>>,
    staging_active: bool,
}

impl<B: Backend> RecurrentStateCache<B> {
    /// Allocates zeroed terminal state and optional proposal snapshots.
    ///
    /// # Errors
    ///
    /// Returns an error for a batch/config mismatch or a dtype unsupported by
    /// the selected backend device.
    pub fn new(
        config: CacheConfig,
        batch_size: usize,
        spec: RecurrentStateSpec,
        device: &B::Device,
    ) -> TensorCacheResult<Self> {
        validate_batch(config, batch_size)?;
        if !B::supports_dtype(device, spec.dtype()) {
            return Err(TensorCacheError::UnsupportedDType {
                dtype: spec.dtype(),
            });
        }
        let layers = B::memory_persistent_allocations(
            device,
            (spec, batch_size, device.clone()),
            |(spec, batch_size, allocation_device)| {
                let matrix_shape = spec.matrix_shape(batch_size);
                let convolution_shape = spec.convolution_shape(batch_size);
                let staged_matrix_shape = [
                    batch_size,
                    spec.staging_capacity(),
                    spec.state_heads(),
                    spec.state_rows(),
                    spec.state_columns(),
                ];
                let staged_convolution_shape = [
                    batch_size,
                    spec.staging_capacity(),
                    spec.convolution_channels(),
                    spec.convolution_width(),
                ];
                (0..spec.layers())
                    .map(|_| RecurrentLayer {
                        matrix: Tensor::zeros(
                            matrix_shape,
                            (&allocation_device, spec.dtype()),
                        ),
                        convolution: Tensor::zeros(
                            convolution_shape,
                            (&allocation_device, spec.dtype()),
                        ),
                        staged_matrix: (spec.staging_capacity() > 0).then(|| {
                            Tensor::zeros(staged_matrix_shape, (&allocation_device, spec.dtype()))
                        }),
                        staged_convolution: (spec.staging_capacity() > 0).then(|| {
                            Tensor::zeros(
                                staged_convolution_shape,
                                (&allocation_device, spec.dtype()),
                            )
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
        })
    }

    /// Returns the immutable recurrent-state geometry.
    #[must_use]
    pub const fn spec(&self) -> RecurrentStateSpec {
        self.spec
    }

    /// Returns the device to which all recurrent tensors are bound.
    #[must_use]
    pub const fn device(&self) -> &B::Device {
        &self.device
    }

    /// Borrows a layer's committed matrix state.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid layer index.
    pub fn matrix(&self, layer: usize) -> TensorCacheResult<&Tensor<B, 4>> {
        Ok(&self.layer(layer)?.matrix)
    }

    /// Borrows a layer's committed causal-convolution history.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid layer index.
    pub fn convolution(&self, layer: usize) -> TensorCacheResult<&Tensor<B, 3>> {
        Ok(&self.layer(layer)?.convolution)
    }

    /// Replaces one sequence's committed recurrent terminal state.
    ///
    /// Inputs are shaped `[1, state_heads, state_rows, state_columns]` and
    /// `[1, convolution_channels, convolution_width]` respectively.
    ///
    /// # Errors
    ///
    /// Returns an error for active staging, invalid indices, or tensor
    /// shape/dtype/device mismatch.
    pub fn set_sequence_state(
        &mut self,
        layer: usize,
        sequence: usize,
        matrix: Tensor<B, 4>,
        convolution: Tensor<B, 3>,
    ) -> TensorCacheResult<()> {
        if self.staging_active {
            return Err(TensorCacheError::TransactionActive {
                operation: "write committed recurrent state",
            });
        }
        self.validate_state_pair(layer, sequence, &matrix, &convolution)?;
        let spec = self.spec;
        let state = self.layer_mut(layer)?;
        assign_matrix_row(&mut state.matrix, sequence, matrix, spec);
        assign_convolution_row(&mut state.convolution, sequence, convolution, spec);
        Ok(())
    }

    /// Appends post-token recurrent snapshots for one speculative sequence.
    ///
    /// Matrix snapshots use `[1, tokens, state_heads, rows, columns]` and
    /// convolution snapshots use `[1, tokens, channels, width]`.
    ///
    /// # Errors
    ///
    /// Returns an error outside a transaction, for invalid tensor metadata, or
    /// when the accumulated proposal exceeds preallocated staging capacity.
    pub fn stage_sequence_snapshots(
        &mut self,
        layer: usize,
        sequence: usize,
        matrices: Tensor<B, 5>,
        convolutions: Tensor<B, 4>,
    ) -> TensorCacheResult<()> {
        if !self.staging_active {
            return Err(TensorCacheError::TransactionNotActive);
        }
        let token_count = self.validate_snapshot_pair(layer, sequence, &matrices, &convolutions)?;
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
        let state = self.layer_mut(layer)?;
        let staged_matrix = state
            .staged_matrix
            .as_mut()
            .ok_or(TensorCacheError::StagingDisabled)?;
        staged_matrix.inplace(|tensor| {
            tensor.slice_assign(
                [
                    sequence..sequence + 1,
                    start..end,
                    0..spec.state_heads(),
                    0..spec.state_rows(),
                    0..spec.state_columns(),
                ],
                matrices,
            )
        });
        let staged_convolution = state
            .staged_convolution
            .as_mut()
            .ok_or(TensorCacheError::StagingDisabled)?;
        staged_convolution.inplace(|tensor| {
            tensor.slice_assign(
                [
                    sequence..sequence + 1,
                    start..end,
                    0..spec.convolution_channels(),
                    0..spec.convolution_width(),
                ],
                convolutions,
            )
        });
        self.staged_lengths[layer][sequence] = end;
        Ok(())
    }

    /// Returns whether proposal snapshots are currently active.
    #[must_use]
    pub const fn has_active_staging(&self) -> bool {
        self.staging_active
    }

    /// Rejects every non-zero intermediate recurrent rewind.
    pub(crate) fn validate_truncate(
        &self,
        current: &[usize],
        requested: &[usize],
    ) -> TensorCacheResult<()> {
        for (sequence, (&from, &to)) in current.iter().zip(requested).enumerate() {
            if to < from && to != 0 {
                return Err(TensorCacheError::UnsafeRecurrentTruncate {
                    sequence,
                    current: from,
                    requested: to,
                });
            }
        }
        Ok(())
    }

    /// Restores the exact initial state for rows truncated to zero.
    pub(crate) fn apply_truncate(&mut self, current: &[usize], requested: &[usize]) {
        for (sequence, (&from, &to)) in current.iter().zip(requested).enumerate() {
            if from != 0 && to == 0 {
                self.zero_sequence(sequence);
            }
        }
    }

    /// Enables empty proposal snapshots for a model-level transaction.
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

    /// Validates complete per-token snapshots before any commit mutation.
    pub(crate) fn validate_staging_commit(
        &self,
        proposed: &[usize],
        accepted: &[usize],
    ) -> TensorCacheResult<()> {
        if !self.staging_active {
            return Err(TensorCacheError::TransactionNotActive);
        }
        for layer in 0..self.spec.layers() {
            for sequence in 0..self.config.sequence_count() {
                let actual = self.staged_lengths[layer][sequence];
                if actual != proposed[sequence] {
                    return Err(TensorCacheError::IncompleteStaging {
                        kind: "recurrent",
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
        Ok(())
    }

    /// Selects the accepted terminal snapshot for every recurrent state row.
    pub(crate) fn commit_staging(&mut self, accepted: &[usize]) -> TensorCacheResult<()> {
        let spec = self.spec;
        for layer_index in 0..spec.layers() {
            for (sequence, &count) in accepted.iter().enumerate() {
                if count == 0 {
                    continue;
                }
                let snapshot = count - 1;
                let state = self.layer(layer_index)?;
                let matrix = state
                    .staged_matrix
                    .as_ref()
                    .ok_or(TensorCacheError::StagingDisabled)?
                    .clone()
                    .slice([
                        sequence..sequence + 1,
                        snapshot..snapshot + 1,
                        0..spec.state_heads(),
                        0..spec.state_rows(),
                        0..spec.state_columns(),
                    ])
                    .reshape(spec.matrix_shape(1));
                let convolution = state
                    .staged_convolution
                    .as_ref()
                    .ok_or(TensorCacheError::StagingDisabled)?
                    .clone()
                    .slice([
                        sequence..sequence + 1,
                        snapshot..snapshot + 1,
                        0..spec.convolution_channels(),
                        0..spec.convolution_width(),
                    ])
                    .reshape(spec.convolution_shape(1));
                let state = self.layer_mut(layer_index)?;
                assign_matrix_row(&mut state.matrix, sequence, matrix, spec);
                assign_convolution_row(&mut state.convolution, sequence, convolution, spec);
            }
        }
        self.finish_staging();
        Ok(())
    }

    /// Discards proposal snapshots without altering committed terminal state.
    pub(crate) fn rollback_staging(&mut self) {
        self.finish_staging();
    }

    /// Clears all committed and speculative recurrent tensors.
    pub(crate) fn reset(&mut self) {
        for state in &mut self.layers {
            zero_tensor(&mut state.matrix);
            zero_tensor(&mut state.convolution);
            if let Some(tensor) = &mut state.staged_matrix {
                zero_tensor(tensor);
            }
            if let Some(tensor) = &mut state.staged_convolution {
                zero_tensor(tensor);
            }
        }
        for lengths in &mut self.staged_lengths {
            lengths.fill(0);
        }
        self.staging_active = false;
    }

    /// Reorders or duplicates every recurrent batch row.
    pub(crate) fn reorder(&mut self, indices: &[usize]) -> TensorCacheResult<()> {
        if self.staging_active {
            return Err(TensorCacheError::TransactionActive {
                operation: "reorder recurrent state",
            });
        }
        validate_reorder_indices(indices, self.config.sequence_count())?;
        let selection = selection_tensor::<B>(indices, &self.device)?;
        for state in &mut self.layers {
            select_batch_4(&mut state.matrix, &selection);
            select_batch_3(&mut state.convolution, &selection);
            if let Some(tensor) = &mut state.staged_matrix {
                select_batch_5(tensor, &selection);
            }
            if let Some(tensor) = &mut state.staged_convolution {
                select_batch_4(tensor, &selection);
            }
        }
        for lengths in &mut self.staged_lengths {
            let old = lengths.clone();
            *lengths = indices.iter().map(|&index| old[index]).collect();
        }
        Ok(())
    }

    /// Returns a checked immutable layer reference.
    fn layer(&self, layer: usize) -> TensorCacheResult<&RecurrentLayer<B>> {
        self.layers
            .get(layer)
            .ok_or(TensorCacheError::LayerOutOfBounds {
                layer,
                layers: self.layers.len(),
            })
    }

    /// Returns a checked mutable layer reference.
    fn layer_mut(&mut self, layer: usize) -> TensorCacheResult<&mut RecurrentLayer<B>> {
        let layers = self.layers.len();
        self.layers
            .get_mut(layer)
            .ok_or(TensorCacheError::LayerOutOfBounds { layer, layers })
    }

    /// Validates one committed matrix/convolution pair.
    fn validate_state_pair(
        &self,
        layer: usize,
        sequence: usize,
        matrix: &Tensor<B, 4>,
        convolution: &Tensor<B, 3>,
    ) -> TensorCacheResult<()> {
        self.layer(layer)?;
        validate_sequence(sequence, self.config.sequence_count())?;
        validate_tensor_4(
            matrix,
            "recurrent matrix",
            self.spec.matrix_shape(1),
            self.spec.dtype(),
            &self.device,
        )?;
        validate_tensor_3(
            convolution,
            "convolution state",
            self.spec.convolution_shape(1),
            self.spec.dtype(),
            &self.device,
        )
    }

    /// Validates proposal snapshots and returns their token count.
    fn validate_snapshot_pair(
        &self,
        layer: usize,
        sequence: usize,
        matrices: &Tensor<B, 5>,
        convolutions: &Tensor<B, 4>,
    ) -> TensorCacheResult<usize> {
        self.layer(layer)?;
        validate_sequence(sequence, self.config.sequence_count())?;
        let token_count = matrices.dims()[1];
        validate_tensor_5(
            matrices,
            "recurrent matrix snapshots",
            [
                1,
                token_count,
                self.spec.state_heads(),
                self.spec.state_rows(),
                self.spec.state_columns(),
            ],
            self.spec.dtype(),
            &self.device,
        )?;
        validate_tensor_4(
            convolutions,
            "convolution snapshots",
            [
                1,
                token_count,
                self.spec.convolution_channels(),
                self.spec.convolution_width(),
            ],
            self.spec.dtype(),
            &self.device,
        )?;
        Ok(token_count)
    }

    /// Clears one sequence's committed and staged rows.
    fn zero_sequence(&mut self, sequence: usize) {
        let spec = self.spec;
        for state in &mut self.layers {
            let matrix = Tensor::zeros(spec.matrix_shape(1), (&self.device, spec.dtype()));
            assign_matrix_row(&mut state.matrix, sequence, matrix, spec);
            let convolution =
                Tensor::zeros(spec.convolution_shape(1), (&self.device, spec.dtype()));
            assign_convolution_row(&mut state.convolution, sequence, convolution, spec);
        }
    }

    /// Ends staging and makes every proposal snapshot unreachable.
    fn finish_staging(&mut self) {
        for lengths in &mut self.staged_lengths {
            lengths.fill(0);
        }
        self.staging_active = false;
    }
}

/// Creates a non-zero recurrent dimension or a field-specific error.
fn non_zero(field: &'static str, value: usize) -> TensorCacheResult<NonZeroUsize> {
    NonZeroUsize::new(value).ok_or(TensorCacheError::ZeroDimension { field })
}

/// Validates one sequence index before tensor slicing.
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

/// Validates rank-three shape, dtype, and device.
fn validate_tensor_3<B: Backend>(
    tensor: &Tensor<B, 3>,
    name: &'static str,
    expected: [usize; 3],
    dtype: DType,
    device: &B::Device,
) -> TensorCacheResult<()> {
    validate_tensor_metadata::<B, 3>(
        tensor.dims(),
        tensor.dtype(),
        tensor.device(),
        name,
        expected,
        dtype,
        device,
    )
}

/// Validates rank-four shape, dtype, and device.
fn validate_tensor_4<B: Backend>(
    tensor: &Tensor<B, 4>,
    name: &'static str,
    expected: [usize; 4],
    dtype: DType,
    device: &B::Device,
) -> TensorCacheResult<()> {
    validate_tensor_metadata::<B, 4>(
        tensor.dims(),
        tensor.dtype(),
        tensor.device(),
        name,
        expected,
        dtype,
        device,
    )
}

/// Validates rank-five shape, dtype, and device.
fn validate_tensor_5<B: Backend>(
    tensor: &Tensor<B, 5>,
    name: &'static str,
    expected: [usize; 5],
    dtype: DType,
    device: &B::Device,
) -> TensorCacheResult<()> {
    validate_tensor_metadata::<B, 5>(
        tensor.dims(),
        tensor.dtype(),
        tensor.device(),
        name,
        expected,
        dtype,
        device,
    )
}

/// Shares tensor metadata checks across the recurrent tensor ranks.
fn validate_tensor_metadata<B: Backend, const D: usize>(
    actual_shape: [usize; D],
    actual_dtype: DType,
    actual_device: B::Device,
    name: &'static str,
    expected_shape: [usize; D],
    expected_dtype: DType,
    expected_device: &B::Device,
) -> TensorCacheResult<()> {
    if actual_shape != expected_shape {
        return Err(TensorCacheError::ShapeMismatch {
            tensor: name,
            expected: expected_shape.to_vec(),
            actual: actual_shape.to_vec(),
        });
    }
    if actual_dtype != expected_dtype {
        return Err(TensorCacheError::DTypeMismatch {
            tensor: name,
            expected: expected_dtype,
            actual: actual_dtype,
        });
    }
    if actual_device != *expected_device {
        return Err(TensorCacheError::DeviceMismatch { tensor: name });
    }
    Ok(())
}

/// Assigns one committed matrix row.
fn assign_matrix_row<B: Backend>(
    destination: &mut Tensor<B, 4>,
    sequence: usize,
    matrix: Tensor<B, 4>,
    spec: RecurrentStateSpec,
) {
    destination.inplace(|tensor| {
        tensor.slice_assign(
            [
                sequence..sequence + 1,
                0..spec.state_heads(),
                0..spec.state_rows(),
                0..spec.state_columns(),
            ],
            matrix,
        )
    });
}

/// Assigns one committed convolution-history row.
fn assign_convolution_row<B: Backend>(
    destination: &mut Tensor<B, 3>,
    sequence: usize,
    convolution: Tensor<B, 3>,
    spec: RecurrentStateSpec,
) {
    destination.inplace(|tensor| {
        tensor.slice_assign(
            [
                sequence..sequence + 1,
                0..spec.convolution_channels(),
                0..spec.convolution_width(),
            ],
            convolution,
        )
    });
}

/// Applies a batch gather to a rank-three state tensor.
fn select_batch_3<B: Backend>(tensor: &mut Tensor<B, 3>, indices: &Tensor<B, 1, Int>) {
    tensor.inplace(|value| value.select(0, indices.clone()));
}

/// Applies a batch gather to a rank-four state tensor.
fn select_batch_4<B: Backend>(tensor: &mut Tensor<B, 4>, indices: &Tensor<B, 1, Int>) {
    tensor.inplace(|value| value.select(0, indices.clone()));
}

/// Applies a batch gather to a rank-five state tensor.
fn select_batch_5<B: Backend>(tensor: &mut Tensor<B, 5>, indices: &Tensor<B, 1, Int>) {
    tensor.inplace(|value| value.select(0, indices.clone()));
}

/// Replaces a tensor with same-shape zeros, preserving dtype and device.
fn zero_tensor<B: Backend, const D: usize>(tensor: &mut Tensor<B, D>) {
    tensor.inplace(|value| value.zeros_like());
}
