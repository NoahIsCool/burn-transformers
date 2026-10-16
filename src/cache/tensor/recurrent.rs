//! Recurrent matrix and convolution state for linear-time sequence mixers.
//!
//! Delta-style recurrence updates a matrix state such as
//! `S_t = decay_t * S_(t-1) + update_t`; selective state-space layers likewise
//! carry a terminal matrix/vector plus a causal-convolution history. Unlike a
//! KV cache, the terminal state is not invertible. Speculative decoding must
//! therefore stage the state *after every proposed token* and select the
//! accepted terminal snapshot, rather than trying to subtract rejected updates.

use core::num::NonZeroUsize;
use core::ops::Range;

use burn::tensor::{Bool, DType, Int, Tensor, TensorData, backend::Backend};

use crate::cache::retention::first_inexact_recurrent_rewind;
use crate::cache::{CacheConfig, DecodeStateError};

use super::kv::validate_batch;
use super::plan::{check_staged_commit, checked_product, rows_cleared_by_truncate, staging_end};
use super::{
    TensorCacheError, TensorCacheResult, validate_repeated_allocation, validate_tensor_shape,
};

mod plan;

pub(crate) use self::plan::{first_nonuniform_offset, last_snapshot};
use self::plan::{flat_snapshot_indices, uniform_snapshot};

/// Validated geometry for DeltaNet/Mamba2-style recurrent layer state.
///
/// Each committed layer owns a matrix tensor shaped
/// `[batch, state_heads, state_rows, state_columns]` and a causal convolution
/// tensor shaped `[batch, convolution_channels, convolution_width]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RecurrentStateSpec {
    /// Number of layers with identical recurrent geometry.
    layers: NonZeroUsize,
    /// Independent matrix-state heads per layer.
    state_heads: NonZeroUsize,
    /// Rows in each recurrent state matrix.
    state_rows: NonZeroUsize,
    /// Columns in each recurrent state matrix.
    state_columns: NonZeroUsize,
    /// Channels in the causal convolution history.
    convolution_channels: NonZeroUsize,
    /// History positions retained per convolution channel.
    convolution_width: NonZeroUsize,
    /// Floating-point precision of accumulated matrix state.
    matrix_dtype: DType,
    /// Floating-point precision of convolution history.
    convolution_dtype: DType,
    /// Maximum proposed terminal snapshots per row; zero disables staging.
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

/// Committed recurrent state and paired optional proposal snapshots for one layer.
#[derive(Debug)]
struct RecurrentLayer<B: Backend> {
    /// Committed state shaped `[batch, heads, rows, columns]`.
    matrix: Tensor<B, 4>,
    /// Committed convolution history shaped `[batch, channels, width]`.
    convolution: Tensor<B, 3>,
    /// Proposed matrices shaped `[batch, proposal, heads, rows, columns]`.
    staged_matrix: Option<Tensor<B, 5>>,
    /// Proposed histories shaped `[batch, proposal, channels, width]`.
    staged_convolution: Option<Tensor<B, 4>>,
}

/// One validated direct replacement of a sequence's terminal recurrent state.
///
/// Construction checks both matrix and convolution tensors before the opaque
/// capability is returned. Applying a collection of capabilities therefore has
/// no recoverable branch between its first and last tensor publication.
#[derive(Debug)]
pub(crate) struct PreparedRecurrentStateWrite<B: Backend> {
    /// Validated destination layer index.
    layer: usize,
    /// Validated destination batch-row index.
    sequence: usize,
    /// Validated single-row replacement matrix state.
    matrix: Tensor<B, 4>,
    /// Validated single-row replacement convolution history.
    convolution: Tensor<B, 3>,
}

/// A validated whole-batch terminal-state replacement.
#[derive(Debug)]
pub(crate) struct PreparedRecurrentBatchWrite<B: Backend> {
    /// Destination layer.
    layer: usize,
    /// New matrices `[batch, heads, rows, columns]`.
    matrix: Tensor<B, 4>,
    /// New histories `[batch, channels, width]`.
    convolution: Tensor<B, 3>,
    /// Rows that keep committed state (`true`), or `None` when all rows change.
    keep: Option<Tensor<B, 1, Bool>>,
}

/// A validated staging of post-token snapshots for a contiguous row range.
#[derive(Debug)]
pub(crate) struct PreparedRecurrentStage<B: Backend> {
    /// Destination layer.
    layer: usize,
    /// Destination rows; the input's batch axis covers exactly these rows.
    rows: Range<usize>,
    /// Snapshot slots `[start, start + tokens)` receiving the input.
    snapshots: Range<usize>,
    /// Matrix snapshots `[rows, tokens, heads, rows, columns]`.
    matrices: Tensor<B, 5>,
    /// Convolution snapshots `[rows, tokens, channels, width]`.
    convolutions: Tensor<B, 4>,
}

/// One sequence's recurrent state selected for the next decode step.
#[derive(Debug, Clone)]
pub struct RecurrentStateRead<B: Backend> {
    /// Selected matrix shaped `[1, heads, rows, columns]`.
    matrix: Tensor<B, 4>,
    /// Selected convolution history shaped `[1, channels, width]`.
    convolution: Tensor<B, 3>,
    /// Zero-based proposal snapshot index, or none for committed state.
    staged_snapshot: Option<usize>,
}

/// Whole-batch recurrent state visible to one model decode invocation.
#[derive(Debug, Clone)]
pub struct RecurrentBatchStateRead<B: Backend> {
    /// Selected matrices shaped `[batch, heads, rows, columns]`.
    matrix: Tensor<B, 4>,
    /// Selected histories shaped `[batch, channels, width]`.
    convolution: Tensor<B, 3>,
    /// Whether each batch row comes from a proposed snapshot.
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
pub(super) struct RecurrentCommitPlan<B: Backend> {
    /// Validated staging handles in committed-layer order; no tensor data is copied.
    layers: Vec<RecurrentStagingRead<B>>,
    /// Number of proposed snapshots retained for each batch row.
    accepted: Vec<usize>,
}

/// Paired recurrent staging allocations retained by a validated commit plan.
#[derive(Debug)]
struct RecurrentStagingRead<B: Backend> {
    /// Matrix snapshots shaped `[batch, proposal, heads, rows, columns]`.
    matrix: Tensor<B, 5>,
    /// Convolution snapshots shaped `[batch, proposal, channels, width]`.
    convolution: Tensor<B, 4>,
}

/// Preallocated terminal recurrent state and speculative state snapshots.
///
/// The speculative matrix layout is
/// `[batch, proposal, state_heads, state_rows, state_columns]`; convolution
/// snapshots use `[batch, proposal, channels, width]`. Snapshot `i` represents
/// state *after* proposal token `i`, so accepting `a > 0` selects `a - 1`.
#[derive(Debug)]
pub struct RecurrentStateCache<B: Backend> {
    /// Validated logical batch and context capacity.
    config: CacheConfig,
    /// Immutable state geometry, independent dtypes, and staging capacity.
    spec: RecurrentStateSpec,
    /// Device shared by every committed and proposed state tensor.
    device: B::Device,
    /// Terminal states and optional proposal snapshots for each layer.
    layers: Vec<RecurrentLayer<B>>,
    /// Proposed snapshot counts indexed by layer and batch row.
    staged_lengths: Vec<Vec<usize>>,
    /// Whether a transaction currently owns the snapshot allocations.
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
    /// Batched models read every row at once here (or through
    /// [`read_working_batch`](Self::read_working_batch) during speculation)
    /// and publish all rows with one
    /// [`TensorStepWrites::push_recurrent_batch`](super::TensorStepWrites::push_recurrent_batch),
    /// avoiding a host-side loop over sequences.
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
        let rows = self.config.sequence_count();
        let write = self.prepare_batch_state(layer, matrix, convolution, &vec![true; rows])?;
        self.apply_batch_state_writes(vec![write]);
        Ok(())
    }

    /// Validates a whole-batch terminal-state replacement that keeps inactive rows.
    ///
    /// Rows with `active[b] == false` keep their committed state: applying
    /// selects them from the old tensor with one `mask_where`, so a ragged
    /// step needs no per-row assignment. When every row is active the input
    /// replaces the state with a single full-range assignment.
    pub(crate) fn prepare_batch_state(
        &self,
        layer: usize,
        matrix: Tensor<B, 4>,
        convolution: Tensor<B, 3>,
        active: &[bool],
    ) -> TensorCacheResult<PreparedRecurrentBatchWrite<B>> {
        if self.staging_active {
            return Err(TensorCacheError::TransactionActive {
                operation: "write committed recurrent state",
            });
        }
        self.validate_batch_state_pair(layer, &matrix, &convolution)?;
        let rows = self.config.sequence_count();
        if active.len() != rows {
            return Err(DecodeStateError::SequenceCountMismatch {
                expected: rows,
                actual: active.len(),
            }
            .into());
        }
        let keep = active.contains(&false).then(|| {
            Tensor::<B, 1, Bool>::from_data(
                TensorData::new(active.iter().map(|&row| !row).collect::<Vec<_>>(), [rows]),
                &self.device,
            )
        });
        Ok(PreparedRecurrentBatchWrite {
            layer,
            matrix,
            convolution,
            keep,
        })
    }

    /// Applies prepared whole-batch state writes; every fallible check already ran.
    pub(crate) fn apply_batch_state_writes(&mut self, writes: Vec<PreparedRecurrentBatchWrite<B>>) {
        let spec = self.spec;
        let rows = self.config.sequence_count();
        for write in writes {
            let state = &mut self.layers[write.layer];
            let (mut matrix, mut convolution) = (write.matrix, write.convolution);
            if let Some(keep) = write.keep {
                let matrix_mask = keep
                    .clone()
                    .reshape([rows, 1, 1, 1])
                    .expand(spec.matrix_shape(rows));
                let convolution_mask = keep
                    .reshape([rows, 1, 1])
                    .expand(spec.convolution_shape(rows));
                matrix = matrix.mask_where(matrix_mask, state.matrix.clone());
                convolution = convolution.mask_where(convolution_mask, state.convolution.clone());
            }
            state.matrix.inplace(|tensor| {
                tensor.slice_assign(
                    [
                        0..rows,
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
                        0..rows,
                        0..spec.convolution_channels(),
                        0..spec.convolution_width(),
                    ],
                    convolution,
                )
            });
        }
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
        // The verified `last_snapshot` selects snapshot `staged - 1`, the
        // state after the last proposal token.
        let snapshot = match last_snapshot(staged) {
            Some(snapshot) if self.staging_active => snapshot,
            _ => return self.read_sequence(layer, sequence),
        };
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
    /// Row `b` with `s_b` staged snapshots reads snapshot `s_b - 1`, or its
    /// committed state when `s_b == 0`. Three cases avoid touching more state
    /// than they return:
    ///
    /// * no row staged (or no transaction): the committed handles are cloned;
    /// * every row staged the same `s > 0`: one slice of snapshot `s - 1` over
    ///   the whole batch;
    /// * ragged: the contiguous `[batch, proposal, ...]` staging tensor is
    ///   viewed as `[batch * proposal, ...]` and one `select` gathers flat row
    ///   `b * proposal + max(s_b, 1) - 1`; rows with `s_b == 0` then take their
    ///   committed state through one `mask_where`.
    ///
    /// Each case copies at most twice the returned state and never
    /// concatenates committed state onto the staging buffer.
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
        let staged_rows: Vec<bool> = staged_lengths.iter().map(|&length| length > 0).collect();
        if !self.staging_active || !staged_rows.contains(&true) {
            return Ok(RecurrentBatchStateRead {
                matrix: state.matrix.clone(),
                convolution: state.convolution.clone(),
                staged_rows,
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
        let batch_size = self.config.sequence_count();
        let spec = self.spec;
        if let Some(snapshot) = uniform_snapshot(staged_lengths) {
            return Ok(RecurrentBatchStateRead {
                matrix: staged_matrix
                    .clone()
                    .slice([
                        0..batch_size,
                        snapshot..snapshot + 1,
                        0..spec.state_heads(),
                        0..spec.state_rows(),
                        0..spec.state_columns(),
                    ])
                    .reshape(spec.matrix_shape(batch_size)),
                convolution: staged_convolution
                    .clone()
                    .slice([
                        0..batch_size,
                        snapshot..snapshot + 1,
                        0..spec.convolution_channels(),
                        0..spec.convolution_width(),
                    ])
                    .reshape(spec.convolution_shape(batch_size)),
                staged_rows,
            });
        }

        let flat_rows =
            flat_snapshot_indices(staged_lengths, spec.staging_capacity()).map_err(|index| {
                TensorCacheError::TensorIndexOverflow {
                    axis: "batch * proposal",
                    index,
                }
            })?;
        let selector =
            Tensor::<B, 1, Int>::from_ints(TensorData::new(flat_rows, [batch_size]), &self.device);
        // The staged snapshots are reshaped to `[batch * proposal, ...]`.
        // Allocation already validated that product as a factor of the staged
        // shape; the verified product reports an unrepresentable reshape
        // extent as a shape overflow instead of wrapping.
        let flat = checked_product(batch_size, spec.staging_capacity()).ok_or(
            TensorCacheError::TensorShapeOverflow {
                tensor: "staged recurrent snapshots",
            },
        )?;
        let mut matrix = staged_matrix
            .clone()
            .reshape([
                flat,
                spec.state_heads(),
                spec.state_rows(),
                spec.state_columns(),
            ])
            .select(0, selector.clone());
        let mut convolution = staged_convolution
            .clone()
            .reshape([flat, spec.convolution_channels(), spec.convolution_width()])
            .select(0, selector);
        if staged_rows.contains(&false) {
            // `true` marks rows without a proposal, which keep committed state.
            let committed_rows = Tensor::<B, 1, Bool>::from_data(
                TensorData::new(
                    staged_rows
                        .iter()
                        .map(|&staged| !staged)
                        .collect::<Vec<_>>(),
                    [batch_size],
                ),
                &self.device,
            );
            let matrix_mask = committed_rows
                .clone()
                .reshape([batch_size, 1, 1, 1])
                .expand(spec.matrix_shape(batch_size));
            let convolution_mask = committed_rows
                .reshape([batch_size, 1, 1])
                .expand(spec.convolution_shape(batch_size));
            matrix = matrix.mask_where(matrix_mask, state.matrix.clone());
            convolution = convolution.mask_where(convolution_mask, state.convolution.clone());
        }
        Ok(RecurrentBatchStateRead {
            matrix,
            convolution,
            staged_rows,
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
        let stage = self.prepare_row_snapshots(layer, sequence, matrices, convolutions)?;
        self.apply_snapshot_stages(vec![stage]);
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
        let stage = self.prepare_batch_snapshots(layer, matrices, convolutions)?;
        self.apply_snapshot_stages(vec![stage]);
        Ok(())
    }

    /// Validates staging one row's `[1, tokens, ...]` snapshots without mutating storage.
    pub(crate) fn prepare_row_snapshots(
        &self,
        layer: usize,
        sequence: usize,
        matrices: Tensor<B, 5>,
        convolutions: Tensor<B, 4>,
    ) -> TensorCacheResult<PreparedRecurrentStage<B>> {
        if !self.staging_active {
            return Err(TensorCacheError::TransactionNotActive);
        }
        let tokens = self.validate_snapshot_pair(layer, sequence, &matrices, &convolutions)?;
        let start = self.staged_lengths[layer][sequence];
        self.prepare_stage(
            layer,
            sequence..sequence + 1,
            start,
            tokens,
            matrices,
            convolutions,
        )
    }

    /// Validates staging a uniform `[batch, tokens, ...]` snapshot block without mutating storage.
    pub(crate) fn prepare_batch_snapshots(
        &self,
        layer: usize,
        matrices: Tensor<B, 5>,
        convolutions: Tensor<B, 4>,
    ) -> TensorCacheResult<PreparedRecurrentStage<B>> {
        if !self.staging_active {
            return Err(TensorCacheError::TransactionNotActive);
        }
        let tokens = self.validate_batch_snapshot_pair(layer, &matrices, &convolutions)?;
        let offsets = &self.staged_lengths[layer];
        let start = offsets[0];
        if let Some(sequence) = first_nonuniform_offset(offsets) {
            return Err(TensorCacheError::NonUniformStagingOffset {
                layer,
                sequence,
                expected: start,
                actual: offsets[sequence],
            });
        }
        let rows = 0..self.config.sequence_count();
        self.prepare_stage(layer, rows, start, tokens, matrices, convolutions)
    }

    /// Checks staging capacity and allocation for snapshots `[start, start + tokens)` of `rows`.
    fn prepare_stage(
        &self,
        layer: usize,
        rows: Range<usize>,
        start: usize,
        tokens: usize,
        matrices: Tensor<B, 5>,
        convolutions: Tensor<B, 4>,
    ) -> TensorCacheResult<PreparedRecurrentStage<B>> {
        let end =
            staging_end(start, tokens, self.spec.staging_capacity()).map_err(|requested| {
                TensorCacheError::StagingCapacityExceeded {
                    requested,
                    capacity: self.spec.staging_capacity(),
                }
            })?;
        let state = self.layer(layer)?;
        if tokens > 0 && (state.staged_matrix.is_none() || state.staged_convolution.is_none()) {
            return Err(TensorCacheError::StagingDisabled);
        }
        Ok(PreparedRecurrentStage {
            layer,
            rows,
            snapshots: start..end,
            matrices,
            convolutions,
        })
    }

    /// Applies prepared snapshot stages; every fallible check already ran.
    pub(crate) fn apply_snapshot_stages(&mut self, stages: Vec<PreparedRecurrentStage<B>>) {
        let spec = self.spec;
        for stage in stages {
            if stage.snapshots.is_empty() {
                continue;
            }
            let state = &mut self.layers[stage.layer];
            if let (Some(staged_matrix), Some(staged_convolution)) =
                (&mut state.staged_matrix, &mut state.staged_convolution)
            {
                let rows = stage.rows.clone();
                let snapshots = stage.snapshots.clone();
                staged_matrix.inplace(|tensor| {
                    tensor.slice_assign(
                        [
                            rows.clone(),
                            snapshots.clone(),
                            0..spec.state_heads(),
                            0..spec.state_rows(),
                            0..spec.state_columns(),
                        ],
                        stage.matrices,
                    )
                });
                staged_convolution.inplace(|tensor| {
                    tensor.slice_assign(
                        [
                            rows,
                            snapshots,
                            0..spec.convolution_channels(),
                            0..spec.convolution_width(),
                        ],
                        stage.convolutions,
                    )
                });
            }
            for sequence in stage.rows {
                self.staged_lengths[stage.layer][sequence] = stage.snapshots.end;
            }
        }
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

    /// Borrows every row's staged snapshot count for one layer.
    pub(crate) fn staged_lengths_of(&self, layer: usize) -> TensorCacheResult<&[usize]> {
        self.layer(layer)?;
        Ok(&self.staged_lengths[layer])
    }

    /// Rejects every non-zero intermediate recurrent rewind.
    ///
    /// The decode state has already rejected extensions. A terminal state can
    /// only stay unchanged or restart from zero (the Verus-verified
    /// `first_inexact_recurrent_rewind`).
    pub(crate) fn validate_truncate(
        &self,
        current: &[usize],
        requested: &[usize],
    ) -> TensorCacheResult<()> {
        match first_inexact_recurrent_rewind(current, requested) {
            None => Ok(()),
            Some(sequence) => Err(TensorCacheError::UnsafeRecurrentTruncate {
                sequence,
                current: current[sequence],
                requested: requested[sequence],
            }),
        }
    }

    /// Restores the exact initial state for rows truncated to zero.
    ///
    /// The verified `rows_cleared_by_truncate` selects exactly the rows that
    /// held tokens and are truncated to zero.
    pub(crate) fn apply_truncate(&mut self, current: &[usize], requested: &[usize]) {
        for (sequence, cleared) in rows_cleared_by_truncate(current, requested)
            .into_iter()
            .enumerate()
        {
            if cleared {
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
    ) -> TensorCacheResult<RecurrentCommitPlan<B>> {
        if !self.staging_active {
            return Err(TensorCacheError::TransactionNotActive);
        }
        validate_commit_vector_lengths(self.config.sequence_count(), proposed, accepted)?;
        // Every layer holds one staged count per row, so the verified scan's
        // precondition follows from the vector-length check above.
        check_staged_commit(&self.staged_lengths, proposed, accepted)
            .map_err(|violation| TensorCacheError::staged_commit("recurrent", violation))?;
        let layers = self
            .layers
            .iter()
            .map(|layer| {
                Ok(RecurrentStagingRead {
                    matrix: layer
                        .staged_matrix
                        .as_ref()
                        .ok_or(TensorCacheError::StagingDisabled)?
                        .clone(),
                    convolution: layer
                        .staged_convolution
                        .as_ref()
                        .ok_or(TensorCacheError::StagingDisabled)?
                        .clone(),
                })
            })
            .collect::<TensorCacheResult<_>>()?;
        Ok(RecurrentCommitPlan {
            layers,
            accepted: accepted.to_vec(),
        })
    }

    /// Applies a prevalidated plan without further fallible state transitions.
    pub(super) fn apply_staging_commit(&mut self, plan: RecurrentCommitPlan<B>) {
        let spec = self.spec;
        for (state, staging) in self.layers.iter_mut().zip(plan.layers) {
            for (sequence, &count) in plan.accepted.iter().enumerate() {
                // Accepting `count > 0` tokens commits snapshot `count - 1`.
                let Some(snapshot) = last_snapshot(count) else {
                    continue;
                };
                let matrix = staging
                    .matrix
                    .clone()
                    .slice([
                        sequence..sequence + 1,
                        snapshot..snapshot + 1,
                        0..spec.state_heads(),
                        0..spec.state_rows(),
                        0..spec.state_columns(),
                    ])
                    .reshape(spec.matrix_shape(1));
                let convolution = staging
                    .convolution
                    .clone()
                    .slice([
                        sequence..sequence + 1,
                        snapshot..snapshot + 1,
                        0..spec.convolution_channels(),
                        0..spec.convolution_width(),
                    ])
                    .reshape(spec.convolution_shape(1));
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

    /// Reorders or duplicates the committed recurrent batch rows.
    ///
    /// `selection` holds source rows already validated by the decode state.
    /// Outside a transaction the snapshot tensors are unreachable scratch
    /// space (every staged length is zero), so only committed state moves;
    /// gathering `[batch, proposal, ...]` snapshots would cost `1 + S` times
    /// the committed state for no observable effect.
    pub(crate) fn reorder_rows(&mut self, selection: &Tensor<B, 1, Int>) {
        debug_assert!(!self.staging_active);
        debug_assert!(
            self.staged_lengths
                .iter()
                .flatten()
                .all(|&length| length == 0)
        );
        for state in &mut self.layers {
            select_batch_4(&mut state.matrix, selection);
            select_batch_3(&mut state.convolution, selection);
        }
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

/// Replaces a tensor with same-shape zeros, preserving dtype and device.
fn zero_tensor<B: Backend, const D: usize>(tensor: &mut Tensor<B, D>) {
    tensor.inplace(|value| value.zeros_like());
}

#[cfg(test)]
#[path = "recurrent_tests.rs"]
mod tests;
