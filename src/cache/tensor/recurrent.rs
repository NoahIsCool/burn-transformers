//! Recurrent matrix and convolution state for linear-time sequence mixers.
//!
//! Delta-style recurrence updates a matrix state such as
//! `S_t = decay_t * S_(t-1) + update_t`; selective state-space layers likewise
//! carry a terminal matrix/vector plus a causal-convolution history. Unlike a
//! KV cache, the terminal state is not invertible. Speculative decoding must
//! therefore stage the state *after every proposed token* and select the
//! accepted terminal snapshot, rather than trying to subtract rejected updates.

use core::num::NonZeroUsize;

use burn::tensor::{DType, Int, Tensor, TensorData, backend::Backend};

use crate::cache::{CacheConfig, DecodeStateError};

use super::allocation::{validate_repeated_allocation, validate_tensor_shape};
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
    matrix_dtype: DType,
    convolution_dtype: DType,
    staging_capacity: usize,
}

impl RecurrentStateSpec {
    /// Creates recurrent state geometry without speculative snapshots.
    ///
    /// # Errors
    ///
    /// Returns an error when any dimension is zero or `dtype` is not a normal
    /// floating-point type. Both state families use `dtype`; use
    /// [`new_with_dtypes`](Self::new_with_dtypes) when a numerically stable
    /// recurrent matrix and lower-precision convolution history differ.
    pub fn new(
        layers: usize,
        state_heads: usize,
        state_rows: usize,
        state_columns: usize,
        convolution_channels: usize,
        convolution_width: usize,
        dtype: DType,
    ) -> TensorCacheResult<Self> {
        Self::new_with_dtypes(
            layers,
            state_heads,
            state_rows,
            state_columns,
            convolution_channels,
            convolution_width,
            dtype,
            dtype,
        )
    }

    /// Creates recurrent state geometry with independent storage dtypes.
    ///
    /// DeltaNet commonly keeps the recurrent matrix in FP32 while retaining
    /// convolution history in the projection dtype. These tensors interact
    /// through explicit model math rather than concatenation, so they need not
    /// share a representation.
    ///
    /// # Errors
    ///
    /// Returns an error when any dimension is zero or either dtype is not a
    /// normal floating-point tensor type.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_dtypes(
        layers: usize,
        state_heads: usize,
        state_rows: usize,
        state_columns: usize,
        convolution_channels: usize,
        convolution_width: usize,
        matrix_dtype: DType,
        convolution_dtype: DType,
    ) -> TensorCacheResult<Self> {
        validate_float_dtype(matrix_dtype)?;
        validate_float_dtype(convolution_dtype)?;
        Ok(Self {
            layers: non_zero("layers", layers)?,
            state_heads: non_zero("state_heads", state_heads)?,
            state_rows: non_zero("state_rows", state_rows)?,
            state_columns: non_zero("state_columns", state_columns)?,
            convolution_channels: non_zero("convolution_channels", convolution_channels)?,
            convolution_width: non_zero("convolution_width", convolution_width)?,
            matrix_dtype,
            convolution_dtype,
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

    /// Returns the recurrent matrix accumulation dtype.
    #[must_use]
    pub const fn matrix_dtype(self) -> DType {
        self.matrix_dtype
    }

    /// Returns the causal-convolution history dtype.
    #[must_use]
    pub const fn convolution_dtype(self) -> DType {
        self.convolution_dtype
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

/// One validated direct replacement of a sequence's terminal recurrent state.
///
/// Construction checks both matrix and convolution tensors before the opaque
/// capability is returned. Applying a collection of capabilities therefore has
/// no recoverable branch between its first and last tensor publication.
#[derive(Debug)]
pub(crate) struct PreparedRecurrentStateWrite<B: Backend> {
    layer: usize,
    sequence: usize,
    matrix: Tensor<B, 4>,
    convolution: Tensor<B, 3>,
}

/// One sequence's recurrent state selected for the next decode step.
#[derive(Debug, Clone)]
pub struct RecurrentStateRead<B: Backend> {
    matrix: Tensor<B, 4>,
    convolution: Tensor<B, 3>,
    staged_snapshot: Option<usize>,
}

/// Whole-batch recurrent state visible to one model decode invocation.
#[derive(Debug, Clone)]
pub struct RecurrentBatchStateRead<B: Backend> {
    matrix: Tensor<B, 4>,
    convolution: Tensor<B, 3>,
    staged_rows: Vec<bool>,
}

impl<B: Backend> RecurrentBatchStateRead<B> {
    /// Borrows matrices shaped `[batch, state_heads, rows, columns]`.
    #[must_use]
    pub const fn matrix(&self) -> &Tensor<B, 4> {
        &self.matrix
    }

    /// Borrows histories shaped `[batch, convolution_channels, width]`.
    #[must_use]
    pub const fn convolution(&self) -> &Tensor<B, 3> {
        &self.convolution
    }

    /// Identifies rows selected from proposal snapshots instead of baseline.
    #[must_use]
    pub fn staged_rows(&self) -> &[bool] {
        &self.staged_rows
    }
}

impl<B: Backend> RecurrentStateRead<B> {
    /// Borrows the matrix shaped `[1, state_heads, rows, columns]`.
    #[must_use]
    pub const fn matrix(&self) -> &Tensor<B, 4> {
        &self.matrix
    }

    /// Borrows convolution history shaped `[1, channels, width]`.
    #[must_use]
    pub const fn convolution(&self) -> &Tensor<B, 3> {
        &self.convolution
    }

    /// Returns the proposal snapshot index, or `None` for committed state.
    #[must_use]
    pub const fn staged_snapshot(&self) -> Option<usize> {
        self.staged_snapshot
    }
}

/// Opaque, fully validated recurrent-state commit capability.
#[derive(Debug)]
pub(super) struct RecurrentCommitPlan {
    accepted: Vec<usize>,
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
        if !B::supports_dtype(device, spec.matrix_dtype()) {
            return Err(TensorCacheError::UnsupportedDType {
                dtype: spec.matrix_dtype(),
            });
        }
        if !B::supports_dtype(device, spec.convolution_dtype()) {
            return Err(TensorCacheError::UnsupportedDType {
                dtype: spec.convolution_dtype(),
            });
        }
        let matrix_shape = spec.matrix_shape(batch_size);
        let matrix_elements = validate_tensor_shape(
            "committed recurrent matrix",
            &matrix_shape,
            spec.matrix_dtype(),
        )?;
        validate_repeated_allocation(
            "committed recurrent matrices",
            matrix_elements,
            spec.layers(),
            spec.matrix_dtype(),
        )?;
        let convolution_shape = spec.convolution_shape(batch_size);
        let convolution_elements = validate_tensor_shape(
            "committed convolution state",
            &convolution_shape,
            spec.convolution_dtype(),
        )?;
        validate_repeated_allocation(
            "committed convolution states",
            convolution_elements,
            spec.layers(),
            spec.convolution_dtype(),
        )?;
        if spec.staging_capacity() > 0 {
            let staged_matrix_shape = [
                batch_size,
                spec.staging_capacity(),
                spec.state_heads(),
                spec.state_rows(),
                spec.state_columns(),
            ];
            let staged_matrix_elements = validate_tensor_shape(
                "staged recurrent matrix",
                &staged_matrix_shape,
                spec.matrix_dtype(),
            )?;
            validate_repeated_allocation(
                "staged recurrent matrices",
                staged_matrix_elements,
                spec.layers(),
                spec.matrix_dtype(),
            )?;
            let staged_convolution_shape = [
                batch_size,
                spec.staging_capacity(),
                spec.convolution_channels(),
                spec.convolution_width(),
            ];
            let staged_convolution_elements = validate_tensor_shape(
                "staged convolution state",
                &staged_convolution_shape,
                spec.convolution_dtype(),
            )?;
            validate_repeated_allocation(
                "staged convolution states",
                staged_convolution_elements,
                spec.layers(),
                spec.convolution_dtype(),
            )?;
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
                            (&allocation_device, spec.matrix_dtype()),
                        ),
                        convolution: Tensor::zeros(
                            convolution_shape,
                            (&allocation_device, spec.convolution_dtype()),
                        ),
                        staged_matrix: (spec.staging_capacity() > 0).then(|| {
                            Tensor::zeros(
                                staged_matrix_shape,
                                (&allocation_device, spec.matrix_dtype()),
                            )
                        }),
                        staged_convolution: (spec.staging_capacity() > 0).then(|| {
                            Tensor::zeros(
                                staged_convolution_shape,
                                (&allocation_device, spec.convolution_dtype()),
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

    /// Returns the shared logical capacity and sequence count.
    #[must_use]
    pub const fn config(&self) -> CacheConfig {
        self.config
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

    /// Borrows a layer's whole-batch matrix and convolution state.
    ///
    /// This is the preferred decode interface: one model kernel can update all
    /// batch rows without a host-side loop over sequences.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid layer index.
    pub fn batch_state(&self, layer: usize) -> TensorCacheResult<(&Tensor<B, 4>, &Tensor<B, 3>)> {
        let layer = self.layer(layer)?;
        Ok((&layer.matrix, &layer.convolution))
    }

    /// Replaces a layer's complete batch state in one ownership transfer.
    ///
    /// Inputs use `[batch, state_heads, rows, columns]` and
    /// `[batch, convolution_channels, width]`. One full-axis assignment avoids
    /// per-sequence host loops and is suitable for prefill or fused decode.
    ///
    /// # Errors
    ///
    /// Returns an error for active staging, an invalid layer, or incompatible
    /// tensor shape, dtype, or device.
    pub fn set_batch_state(
        &mut self,
        layer: usize,
        matrix: Tensor<B, 4>,
        convolution: Tensor<B, 3>,
    ) -> TensorCacheResult<()> {
        if self.staging_active {
            return Err(TensorCacheError::TransactionActive {
                operation: "write committed recurrent state",
            });
        }
        self.validate_batch_state_pair(layer, &matrix, &convolution)?;
        let spec = self.spec;
        let batch_size = self.config.sequence_count();
        let state = self.layer_mut(layer)?;
        state.matrix.inplace(|tensor| {
            tensor.slice_assign(
                [
                    0..batch_size,
                    0..spec.state_heads(),
                    0..spec.state_rows(),
                    0..spec.state_columns(),
                ],
                matrix,
            )
        });
        state.convolution.inplace(|tensor| {
            tensor.slice_assign(
                [
                    0..batch_size,
                    0..spec.convolution_channels(),
                    0..spec.convolution_width(),
                ],
                convolution,
            )
        });
        Ok(())
    }

    /// Reads one sequence's committed recurrent terminal state.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid layer or sequence index.
    pub fn read_sequence(
        &self,
        layer: usize,
        sequence: usize,
    ) -> TensorCacheResult<RecurrentStateRead<B>> {
        let state = self.layer(layer)?;
        validate_sequence(sequence, self.config.sequence_count())?;
        Ok(RecurrentStateRead {
            matrix: state.matrix.clone().slice([
                sequence..sequence + 1,
                0..self.spec.state_heads(),
                0..self.spec.state_rows(),
                0..self.spec.state_columns(),
            ]),
            convolution: state.convolution.clone().slice([
                sequence..sequence + 1,
                0..self.spec.convolution_channels(),
                0..self.spec.convolution_width(),
            ]),
            staged_snapshot: None,
        })
    }

    /// Reads the latest state visible to an iterative speculative step.
    ///
    /// Snapshot `i` is the state after proposal token `i`; consequently the
    /// working terminal state is snapshot `staged_length - 1`. Before the first
    /// proposal token, or outside a transaction, committed state is returned.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid indices or inconsistent staging storage.
    pub fn read_working_sequence(
        &self,
        layer: usize,
        sequence: usize,
    ) -> TensorCacheResult<RecurrentStateRead<B>> {
        let state = self.layer(layer)?;
        validate_sequence(sequence, self.config.sequence_count())?;
        let staged = self.staged_lengths[layer][sequence];
        if !self.staging_active || staged == 0 {
            return self.read_sequence(layer, sequence);
        }
        let snapshot = staged - 1;
        let matrix = state
            .staged_matrix
            .as_ref()
            .ok_or(TensorCacheError::StagingDisabled)?
            .clone()
            .slice([
                sequence..sequence + 1,
                snapshot..snapshot + 1,
                0..self.spec.state_heads(),
                0..self.spec.state_rows(),
                0..self.spec.state_columns(),
            ])
            .reshape(self.spec.matrix_shape(1));
        let convolution = state
            .staged_convolution
            .as_ref()
            .ok_or(TensorCacheError::StagingDisabled)?
            .clone()
            .slice([
                sequence..sequence + 1,
                snapshot..snapshot + 1,
                0..self.spec.convolution_channels(),
                0..self.spec.convolution_width(),
            ])
            .reshape(self.spec.convolution_shape(1));
        Ok(RecurrentStateRead {
            matrix,
            convolution,
            staged_snapshot: Some(snapshot),
        })
    }

    /// Reads the latest working recurrent state for the entire batch.
    ///
    /// A synthetic snapshot axis concatenates committed state at index zero
    /// with proposal snapshots at indices `1..`. A single device-side
    /// [`Tensor::gather_nd`] then selects `(batch_row, staged_length)` for every
    /// row, supporting ragged proposals without host-side tensor slicing.
    /// Outside staging this only clones the committed tensor handles.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid layer, inconsistent staging allocation,
    /// or an index outside Burn's portable signed selector range.
    pub fn read_working_batch(
        &self,
        layer: usize,
    ) -> TensorCacheResult<RecurrentBatchStateRead<B>> {
        let state = self.layer(layer)?;
        let staged_lengths = &self.staged_lengths[layer];
        if !self.staging_active || staged_lengths.iter().all(|&length| length == 0) {
            return Ok(RecurrentBatchStateRead {
                matrix: state.matrix.clone(),
                convolution: state.convolution.clone(),
                staged_rows: vec![false; self.config.sequence_count()],
            });
        }
        let staged_matrix = state
            .staged_matrix
            .as_ref()
            .ok_or(TensorCacheError::StagingDisabled)?;
        let staged_convolution = state
            .staged_convolution
            .as_ref()
            .ok_or(TensorCacheError::StagingDisabled)?;
        let mut coordinates = Vec::with_capacity(self.config.sequence_count() * 2);
        for (sequence, &staged_length) in staged_lengths.iter().enumerate() {
            coordinates.push(i32::try_from(sequence).map_err(|_| {
                TensorCacheError::TensorIndexOverflow {
                    axis: "batch",
                    index: sequence,
                }
            })?);
            coordinates.push(i32::try_from(staged_length).map_err(|_| {
                TensorCacheError::TensorIndexOverflow {
                    axis: "proposal",
                    index: staged_length,
                }
            })?);
        }
        let indices = Tensor::<B, 2, Int>::from_ints(
            TensorData::new(coordinates, [self.config.sequence_count(), 2]),
            &self.device,
        );
        let matrix_source = Tensor::cat(
            vec![
                state.matrix.clone().unsqueeze_dim::<5>(1),
                staged_matrix.clone(),
            ],
            1,
        );
        let convolution_source = Tensor::cat(
            vec![
                state.convolution.clone().unsqueeze_dim::<4>(1),
                staged_convolution.clone(),
            ],
            1,
        );
        Ok(RecurrentBatchStateRead {
            matrix: matrix_source.gather_nd::<2, 4>(indices.clone()),
            convolution: convolution_source.gather_nd::<2, 3>(indices),
            staged_rows: staged_lengths.iter().map(|&length| length > 0).collect(),
        })
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
        let write = self.prepare_sequence_state(layer, sequence, matrix, convolution)?;
        self.apply_prepared_state_writes(vec![write]);
        Ok(())
    }

    /// Validates one direct terminal-state replacement without mutating storage.
    ///
    /// Models can prepare every layer and row first, then apply the opaque
    /// writes only after their complete forward pass succeeds.
    ///
    /// # Errors
    ///
    /// Returns the same validation errors as
    /// [`set_sequence_state`](Self::set_sequence_state).
    pub(crate) fn prepare_sequence_state(
        &self,
        layer: usize,
        sequence: usize,
        matrix: Tensor<B, 4>,
        convolution: Tensor<B, 3>,
    ) -> TensorCacheResult<PreparedRecurrentStateWrite<B>> {
        if self.staging_active {
            return Err(TensorCacheError::TransactionActive {
                operation: "write committed recurrent state",
            });
        }
        self.validate_state_pair(layer, sequence, &matrix, &convolution)?;
        Ok(PreparedRecurrentStateWrite {
            layer,
            sequence,
            matrix,
            convolution,
        })
    }

    /// Applies writes returned by [`prepare_sequence_state`](Self::prepare_sequence_state).
    ///
    /// All indexing and tensor compatibility checks happened during
    /// preparation, leaving only infallible Burn tensor assignments here.
    pub(crate) fn apply_prepared_state_writes(
        &mut self,
        writes: Vec<PreparedRecurrentStateWrite<B>>,
    ) {
        for write in writes {
            let state = &mut self.layers[write.layer];
            assign_matrix_row(&mut state.matrix, write.sequence, write.matrix, self.spec);
            assign_convolution_row(
                &mut state.convolution,
                write.sequence,
                write.convolution,
                self.spec,
            );
        }
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

    /// Appends a uniform token block of recurrent snapshots for every row.
    ///
    /// Inputs use `[batch, tokens, state_heads, rows, columns]` and
    /// `[batch, tokens, channels, width]`. All rows must currently share one
    /// staging offset; ragged callers can use
    /// [`stage_sequence_snapshots`](Self::stage_sequence_snapshots) as the
    /// explicit fallback.
    ///
    /// # Errors
    ///
    /// Returns an error outside a transaction, for incompatible tensor
    /// metadata, a ragged starting offset, or staging-capacity overflow.
    pub fn stage_batch_snapshots(
        &mut self,
        layer: usize,
        matrices: Tensor<B, 5>,
        convolutions: Tensor<B, 4>,
    ) -> TensorCacheResult<()> {
        if !self.staging_active {
            return Err(TensorCacheError::TransactionNotActive);
        }
        let token_count = self.validate_batch_snapshot_pair(layer, &matrices, &convolutions)?;
        let offsets = &self.staged_lengths[layer];
        let start = offsets[0];
        if let Some((sequence, &actual)) = offsets
            .iter()
            .enumerate()
            .find(|(_, offset)| **offset != start)
        {
            return Err(TensorCacheError::NonUniformStagingOffset {
                layer,
                sequence,
                expected: start,
                actual,
            });
        }
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
        let batch_size = self.config.sequence_count();
        let state = self.layer_mut(layer)?;
        state
            .staged_matrix
            .as_mut()
            .ok_or(TensorCacheError::StagingDisabled)?
            .inplace(|tensor| {
                tensor.slice_assign(
                    [
                        0..batch_size,
                        start..end,
                        0..spec.state_heads(),
                        0..spec.state_rows(),
                        0..spec.state_columns(),
                    ],
                    matrices,
                )
            });
        state
            .staged_convolution
            .as_mut()
            .ok_or(TensorCacheError::StagingDisabled)?
            .inplace(|tensor| {
                tensor.slice_assign(
                    [
                        0..batch_size,
                        start..end,
                        0..spec.convolution_channels(),
                        0..spec.convolution_width(),
                    ],
                    convolutions,
                )
            });
        self.staged_lengths[layer].fill(end);
        Ok(())
    }

    /// Returns whether proposal snapshots are currently active.
    #[must_use]
    pub const fn has_active_staging(&self) -> bool {
        self.staging_active
    }

    /// Returns the number of recurrent snapshots staged for one sequence.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid layer or sequence.
    pub fn staged_length(&self, layer: usize, sequence: usize) -> TensorCacheResult<usize> {
        self.layer(layer)?;
        validate_sequence(sequence, self.config.sequence_count())?;
        Ok(self.staged_lengths[layer][sequence])
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

    /// Builds an opaque commit plan after validating every staged snapshot.
    pub(super) fn prepare_staging_commit(
        &self,
        proposed: &[usize],
        accepted: &[usize],
    ) -> TensorCacheResult<RecurrentCommitPlan> {
        if !self.staging_active {
            return Err(TensorCacheError::TransactionNotActive);
        }
        validate_commit_vector_lengths(self.config.sequence_count(), proposed, accepted)?;
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
        Ok(RecurrentCommitPlan {
            accepted: accepted.to_vec(),
        })
    }

    /// Applies a prevalidated plan without further fallible state transitions.
    pub(super) fn apply_staging_commit(&mut self, plan: RecurrentCommitPlan) {
        let spec = self.spec;
        for layer_index in 0..spec.layers() {
            for (sequence, &count) in plan.accepted.iter().enumerate() {
                if count == 0 {
                    continue;
                }
                let snapshot = count - 1;
                let state = &self.layers[layer_index];
                let matrix = state
                    .staged_matrix
                    .as_ref()
                    .expect("commit plan requires allocated recurrent staging")
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
                    .expect("commit plan requires allocated recurrent staging")
                    .clone()
                    .slice([
                        sequence..sequence + 1,
                        snapshot..snapshot + 1,
                        0..spec.convolution_channels(),
                        0..spec.convolution_width(),
                    ])
                    .reshape(spec.convolution_shape(1));
                let state = &mut self.layers[layer_index];
                assign_matrix_row(&mut state.matrix, sequence, matrix, spec);
                assign_convolution_row(&mut state.convolution, sequence, convolution, spec);
            }
        }
        self.finish_staging();
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
            self.spec.matrix_dtype(),
            &self.device,
        )?;
        validate_tensor_3(
            convolution,
            "convolution state",
            self.spec.convolution_shape(1),
            self.spec.convolution_dtype(),
            &self.device,
        )
    }

    /// Validates one whole-batch matrix/convolution pair.
    fn validate_batch_state_pair(
        &self,
        layer: usize,
        matrix: &Tensor<B, 4>,
        convolution: &Tensor<B, 3>,
    ) -> TensorCacheResult<()> {
        self.layer(layer)?;
        validate_tensor_4(
            matrix,
            "recurrent matrix batch",
            self.spec.matrix_shape(self.config.sequence_count()),
            self.spec.matrix_dtype(),
            &self.device,
        )?;
        validate_tensor_3(
            convolution,
            "convolution state batch",
            self.spec.convolution_shape(self.config.sequence_count()),
            self.spec.convolution_dtype(),
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
            self.spec.matrix_dtype(),
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
            self.spec.convolution_dtype(),
            &self.device,
        )?;
        Ok(token_count)
    }

    /// Validates whole-batch proposal snapshots and returns their token count.
    fn validate_batch_snapshot_pair(
        &self,
        layer: usize,
        matrices: &Tensor<B, 5>,
        convolutions: &Tensor<B, 4>,
    ) -> TensorCacheResult<usize> {
        self.layer(layer)?;
        let token_count = matrices.dims()[1];
        validate_tensor_5(
            matrices,
            "recurrent matrix snapshot batch",
            [
                self.config.sequence_count(),
                token_count,
                self.spec.state_heads(),
                self.spec.state_rows(),
                self.spec.state_columns(),
            ],
            self.spec.matrix_dtype(),
            &self.device,
        )?;
        validate_tensor_4(
            convolutions,
            "convolution snapshot batch",
            [
                self.config.sequence_count(),
                token_count,
                self.spec.convolution_channels(),
                self.spec.convolution_width(),
            ],
            self.spec.convolution_dtype(),
            &self.device,
        )?;
        Ok(token_count)
    }

    /// Clears one sequence's committed and staged rows.
    fn zero_sequence(&mut self, sequence: usize) {
        let spec = self.spec;
        for state in &mut self.layers {
            let matrix = Tensor::zeros(spec.matrix_shape(1), (&self.device, spec.matrix_dtype()));
            assign_matrix_row(&mut state.matrix, sequence, matrix, spec);
            let convolution = Tensor::zeros(
                spec.convolution_shape(1),
                (&self.device, spec.convolution_dtype()),
            );
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

/// Rejects integer, boolean, and quantized recurrent state tensors.
fn validate_float_dtype(dtype: DType) -> TensorCacheResult<()> {
    if dtype.is_float() {
        Ok(())
    } else {
        Err(TensorCacheError::NonFloatDType { dtype })
    }
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

/// Validates proposal and acceptance vector sizes before direct indexing.
fn validate_commit_vector_lengths(
    expected: usize,
    proposed: &[usize],
    accepted: &[usize],
) -> TensorCacheResult<()> {
    for actual in [proposed.len(), accepted.len()] {
        if actual != expected {
            return Err(DecodeStateError::SequenceCountMismatch { expected, actual }.into());
        }
    }
    Ok(())
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
