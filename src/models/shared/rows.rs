//! Stateful row preparation: one readback, compaction plans, and exact scatter.
//!
//! Prefill and decode evaluate a rectangular `[B, S]` input against an
//! external cache with per-row committed lengths `L_b`. Padding must neither
//! consume positions nor enter the cache, so each row `b` is *compacted* to its
//! valid columns `c_{b,0} < … < c_{b,n_b-1}`, evaluated on `[1, n_b, …]`, and
//! *scattered* back into `[1, S, …]` with exact zeros at padding. A row with
//! `n_b = 0` (permitted during decode) is a no-op that returns zeros.
//!
//! [`prepare_stateful_rows`] validates the whole call before any model work:
//! shapes and devices, then one device-to-host transfer of the mask, token IDs,
//! and explicit positions. On the host it derives the valid columns, checks the
//! empty-row policy and every append `L_b + n_b <= capacity`, and validates
//! token and position values over valid columns only, so padding sentinels are
//! inert. The per-row rotary extent is `1 + max p_b` for explicit positions and
//! `L_b + n_b` for the implicit positions `L_b, …, L_b + n_b - 1`.
//!
//! [`CompactRow`] realizes the verified [`plan::plan_scatter`] layouts with a
//! constant number of kernels per tensor (identity, one concatenation, or one
//! gather), instead of one slice per valid token.

use burn::tensor::{BasicOps, Int, Tensor, TensorData, TensorKind, backend::Backend};

use crate::{
    CausalLmInput, Error, Result,
    cache::{DecodeStateError, TensorCacheError},
    position::RopeScaling,
};

use super::{
    checks::{ensure_grid, ensure_nonempty, ensure_same_device, range_error, read_int_tensors},
    input::{EmptyRows, InputNames},
    plan::{self, AppendError, ScatterLayout},
};

/// Contract for one stateful (prefill or decode) entry point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct StatefulContract<'a> {
    /// Tensor names reported in errors.
    pub(crate) names: InputNames<'a>,
    /// Owner named in device errors, e.g. `Qwen decode state`.
    pub(crate) owner: &'a str,
    /// Exclusive bound `V` on valid token IDs.
    pub(crate) vocabulary: usize,
    /// Exclusive bound on valid explicit positions; `None` leaves them unchecked.
    pub(crate) position_limit: Option<usize>,
    /// Prefill rejects fully masked rows; decode permits them as no-ops.
    pub(crate) empty_rows: EmptyRows,
}

/// Host plan for one validated rectangular stateful call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct StatefulRows {
    /// Number of rows `B`, equal to the decode state's batch size.
    pub(crate) batch: usize,
    /// Number of input columns `S`, including padding.
    pub(crate) sequence: usize,
    /// Valid source columns of each row in increasing order.
    pub(crate) valid_indices: Vec<Vec<usize>>,
    /// Number of tokens appended to each row; zero marks an inactive row.
    pub(crate) token_counts: Vec<usize>,
    /// Per-row rotary extent: `1 + max p_b` (explicit) or `L_b + n_b` (implicit).
    pub(crate) evaluated_contexts: Vec<usize>,
    /// Each row's last valid column, or `S - 1` for an inactive row.
    last_columns: Vec<usize>,
}

impl StatefulRows {
    /// Materializes row `row`'s compaction and scatter plan on `device`.
    ///
    /// Returns `None` for an inactive row, which must produce zeros without
    /// evaluating the model or touching its cache.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Shape`] for an out-of-range row and
    /// [`Error::Configuration`] for indices beyond Burn's signed range.
    pub(crate) fn compact_row<B: Backend>(
        &self,
        row: usize,
        device: &B::Device,
    ) -> Result<Option<CompactRow<B>>> {
        let columns = self.valid_indices.get(row).ok_or_else(|| {
            Error::shape(
                "stateful.row",
                format!("a row below {}", self.batch),
                row.to_string(),
            )
        })?;
        if columns.is_empty() {
            return Ok(None);
        }
        CompactRow::new(row, columns, self.sequence, device).map(Some)
    }

    /// Iterates `(row, evaluated_context)` over rows that append at least one token.
    pub(crate) fn active_contexts(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.evaluated_contexts
            .iter()
            .zip(&self.token_counts)
            .enumerate()
            .filter(|(_, (_, count))| **count > 0)
            .map(|(row, (&context, _))| (row, context))
    }

    /// Returns each row's last valid column for [`LogitsSelection::Last`].
    ///
    /// An inactive row reports `S - 1`; its hidden states are zero, so any
    /// column selects the same values while keeping the common case a slice.
    ///
    /// [`LogitsSelection::Last`]: crate::LogitsSelection::Last
    pub(crate) fn last_columns(&self) -> &[usize] {
        &self.last_columns
    }
}

/// Validates a stateful call and plans every row before any model work.
///
/// `lengths` are the per-row cursors the call appends to (committed or
/// working lengths, depending on the family's transaction model) and
/// `capacity` is the cache's logical capacity. Exactly one blocking
/// device-to-host transfer is performed.
///
/// # Errors
///
/// Returns [`Error::Shape`] for empty or mismatched extents, rejected empty
/// rows, and out-of-range valid tokens or positions;
/// [`Error::TensorCache`] for batch mismatches, cursor overflow, or capacity
/// overflow; and [`Error::BackendCapability`] for device mismatches or a
/// failed readback.
pub(crate) fn prepare_stateful_rows<B: Backend>(
    input: &CausalLmInput<B>,
    device: &B::Device,
    lengths: &[usize],
    capacity: usize,
    contract: &StatefulContract<'_>,
) -> Result<StatefulRows> {
    let names = contract.names;
    let [batch, sequence] = input.token_ids.dims();
    ensure_nonempty(batch, sequence, names.token_ids)?;
    if batch != lengths.len() {
        return Err(TensorCacheError::BatchSizeMismatch {
            configured: lengths.len(),
            requested: batch,
        }
        .into());
    }
    ensure_same_device(
        names.token_ids,
        &input.token_ids.device(),
        device,
        contract.owner,
    )?;
    if let Some(mask) = &input.token_mask {
        ensure_same_device(
            names.token_mask,
            &mask.as_tensor().device(),
            device,
            contract.owner,
        )?;
        ensure_grid(mask.as_tensor().dims(), batch, sequence, names.token_mask)?;
    }
    if let Some(positions) = &input.position_ids {
        ensure_same_device(
            names.position_ids,
            &positions.device(),
            device,
            contract.owner,
        )?;
        ensure_grid(positions.dims(), batch, sequence, names.position_ids)?;
    }

    let host = HostInput::read(input, batch, sequence, names.token_ids)?;
    let valid_indices = host.valid_indices(batch, sequence);
    let last_columns = host.last_columns(batch, sequence);
    if contract.empty_rows == EmptyRows::Reject && valid_indices.iter().any(Vec::is_empty) {
        return Err(Error::shape(
            names.token_mask,
            "at least one valid token in every prefill row",
            "a fully masked prompt row",
        ));
    }
    let token_counts = valid_indices.iter().map(Vec::len).collect::<Vec<_>>();
    let ends = plan::plan_appends(lengths, &token_counts, capacity)
        .map_err(|error| append_error(error, capacity))?;

    let mut evaluated_contexts = Vec::with_capacity(batch);
    for (row, columns) in valid_indices.iter().enumerate() {
        let window = row * sequence..(row + 1) * sequence;
        if let Some((minimum, maximum)) = plan::value_range(&host.tokens[window.clone()], columns)
            && !plan::range_within(minimum, maximum, contract.vocabulary)
        {
            return Err(range_error(
                names.token_ids,
                "token",
                contract.vocabulary,
                minimum,
                maximum,
            ));
        }
        let explicit = host
            .positions
            .as_ref()
            .and_then(|positions| plan::value_range(&positions[window], columns));
        evaluated_contexts.push(match explicit {
            Some((minimum, maximum)) => {
                if let Some(limit) = contract.position_limit
                    && !plan::range_within(minimum, maximum, limit)
                {
                    return Err(range_error(
                        names.position_ids,
                        "position",
                        limit,
                        minimum,
                        maximum,
                    ));
                }
                plan::context_from_maximum(maximum).ok_or_else(|| {
                    Error::configuration(names.position_ids, "positions must be non-negative")
                })?
            }
            // Implicit positions L_b..L_b + n_b, or an inactive row's unchanged cursor.
            None => ends[row],
        });
    }
    Ok(StatefulRows {
        batch,
        sequence,
        valid_indices,
        token_counts,
        evaluated_contexts,
        last_columns,
    })
}

/// Host copies of the mask, token IDs, and explicit positions of one call.
#[derive(Debug)]
struct HostInput {
    /// Row-major validity flags, or `None` when every token is valid.
    mask: Option<Vec<bool>>,
    /// Row-major token IDs.
    tokens: Vec<i64>,
    /// Row-major explicit positions, when supplied.
    positions: Option<Vec<i64>>,
}

impl HostInput {
    /// Reads every per-token tensor of `input` in one transfer.
    fn read<B: Backend>(
        input: &CausalLmInput<B>,
        batch: usize,
        sequence: usize,
        label: &str,
    ) -> Result<Self> {
        let elements = batch * sequence;
        let flat = |tensor: Tensor<B, 2, Int>| tensor.reshape([elements]);
        let mut tensors = Vec::with_capacity(3);
        if let Some(mask) = &input.token_mask {
            tensors.push(flat(mask.as_tensor().clone().int()));
        }
        tensors.push(flat(input.token_ids.clone()));
        if let Some(positions) = &input.position_ids {
            tensors.push(flat(positions.clone()));
        }
        let expected = elements * tensors.len();
        let mut values = read_int_tensors(tensors, label)?;
        if values.len() != expected {
            return Err(Error::BackendCapability(format!(
                "{label} readback returned {} values, expected {expected}",
                values.len()
            )));
        }
        // Split in reverse registration order: positions, tokens, then mask.
        let positions = input
            .position_ids
            .is_some()
            .then(|| values.split_off(values.len() - elements));
        let tokens = values.split_off(values.len() - elements);
        let mask = input
            .token_mask
            .is_some()
            .then(|| values.into_iter().map(|value| value != 0).collect());
        Ok(Self {
            mask,
            tokens,
            positions,
        })
    }

    /// Returns every row's last valid column, using `S - 1` for inactive rows.
    fn last_columns(&self, batch: usize, sequence: usize) -> Vec<usize> {
        match &self.mask {
            Some(mask) => mask
                .chunks_exact(sequence)
                .map(|row| plan::last_valid_column(row).unwrap_or(sequence - 1))
                .collect(),
            None => vec![sequence - 1; batch],
        }
    }

    /// Lists every row's valid columns, treating an absent mask as all valid.
    fn valid_indices(&self, batch: usize, sequence: usize) -> Vec<Vec<usize>> {
        match &self.mask {
            Some(mask) => mask
                .chunks_exact(sequence)
                .map(plan::valid_columns)
                .collect(),
            None => vec![(0..sequence).collect::<Vec<_>>(); batch],
        }
    }
}

/// Converts a planning failure into the shared cache-metadata error.
fn append_error(error: AppendError, capacity: usize) -> Error {
    let metadata = match error {
        AppendError::Overflow {
            row,
            length,
            additional,
        } => DecodeStateError::LengthOverflow {
            sequence: row,
            length,
            additional,
        },
        AppendError::Capacity { row, requested } => DecodeStateError::CapacityExceeded {
            sequence: row,
            requested,
            capacity,
        },
    };
    TensorCacheError::Metadata(metadata).into()
}

/// Device-side layout of one compact row.
#[derive(Debug, Clone)]
enum RowLayout<B: Backend> {
    /// Every column is valid.
    Identity,
    /// One valid run between `leading` and `trailing` padding columns.
    Contiguous {
        /// Padding columns before the run.
        leading: usize,
        /// Padding columns after the run.
        trailing: usize,
    },
    /// Interior padding handled by index tensors.
    Gather {
        /// Valid source columns `c_k`, shaped `[n]`.
        columns: Tensor<B, 1, Int>,
        /// Compact source per output column (`n` addresses a zero row), shaped `[S]`.
        sources: Tensor<B, 1, Int>,
    },
}

/// Compaction and scatter plan for one active row, materialized once per call.
///
/// Index tensors are uploaded once and reused for the token IDs, positions,
/// final states, every requested hidden state, and router diagnostics.
#[derive(Debug, Clone)]
pub(crate) struct CompactRow<B: Backend> {
    /// Batch row this plan addresses.
    row: usize,
    /// Rectangular width `S`.
    sequence: usize,
    /// Number of valid columns `n > 0`.
    count: usize,
    /// Kernel strategy for compaction and scatter.
    layout: RowLayout<B>,
}

impl<B: Backend> CompactRow<B> {
    /// Plans row `row` from its valid columns `0 <= c_0 < … < c_{n-1} < sequence`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Shape`] when the columns are empty, unsorted, or out of
    /// range, and [`Error::Configuration`] for indices beyond Burn's signed range.
    pub(crate) fn new(
        row: usize,
        columns: &[usize],
        sequence: usize,
        device: &B::Device,
    ) -> Result<Self> {
        if columns.is_empty() || !plan::is_sorted_unique_below(columns, sequence) {
            return Err(Error::shape(
                "stateful.valid_columns",
                format!("non-empty sorted unique columns below {sequence}"),
                format!("{columns:?}"),
            ));
        }
        let layout = match plan::plan_scatter(columns, sequence) {
            ScatterLayout::Identity => RowLayout::Identity,
            ScatterLayout::Contiguous { leading, trailing } => {
                RowLayout::Contiguous { leading, trailing }
            }
            ScatterLayout::Gather { sources } => RowLayout::Gather {
                columns: index_tensor(columns, device, "stateful.valid_columns")?,
                sources: index_tensor(&sources, device, "stateful.scatter_sources")?,
            },
        };
        Ok(Self {
            row,
            sequence,
            count: columns.len(),
            layout,
        })
    }

    /// Returns the number of valid tokens `n` in this row.
    pub(crate) const fn count(&self) -> usize {
        self.count
    }

    /// Selects this row's valid columns from a rectangular `[B, S]` tensor.
    ///
    /// Returns `[1, n]`: the row itself when dense, a slice for one run, and a
    /// single index-select otherwise.
    pub(crate) fn compact<K>(&self, rectangular: &Tensor<B, 2, K>) -> Tensor<B, 2, K>
    where
        K: TensorKind<B> + BasicOps<B>,
    {
        let row = rectangular
            .clone()
            .slice([self.row..self.row + 1, 0..self.sequence]);
        match &self.layout {
            RowLayout::Identity => row,
            RowLayout::Contiguous { leading, .. } => row.narrow(1, *leading, self.count),
            RowLayout::Gather { columns, .. } => row.select(1, columns.clone()),
        }
    }

    /// Restores compact `[1, n, d]` values to `[1, S, d]` with zeros at padding.
    ///
    /// Values are copied exactly, the dtype is preserved, and the operation is
    /// differentiable (concatenation and index-select both have exact
    /// adjoints).
    ///
    /// # Errors
    ///
    /// Returns [`Error::Shape`] when `compact` is not `[1, n, d]`.
    pub(crate) fn scatter<K>(&self, compact: Tensor<B, 3, K>) -> Result<Tensor<B, 3, K>>
    where
        K: TensorKind<B> + BasicOps<B>,
    {
        let [rows, count, width] = compact.dims();
        if rows != 1 || count != self.count {
            return Err(Error::shape(
                "stateful.compact_row",
                format!("[1, {}, width]", self.count),
                format!("{:?}", compact.dims()),
            ));
        }
        let zeros = |columns: usize| {
            Tensor::<B, 3, K>::zeros([1, columns, width], (&compact.device(), compact.dtype()))
        };
        Ok(match &self.layout {
            RowLayout::Identity => compact,
            RowLayout::Contiguous { leading, trailing } => {
                let mut segments = Vec::with_capacity(3);
                if *leading > 0 {
                    segments.push(zeros(*leading));
                }
                let trailing_zeros = (*trailing > 0).then(|| zeros(*trailing));
                segments.push(compact);
                segments.extend(trailing_zeros);
                Tensor::cat(segments, 1)
            }
            // Entry n of [compact; 0] is the zero row addressed by padded columns.
            RowLayout::Gather { sources, .. } => {
                let padded = Tensor::cat(vec![compact.clone(), zeros(1)], 1);
                padded.select(1, sources.clone())
            }
        })
    }
}

/// Uploads host indices (columns, rows, or batch permutations) as a `[len]` index tensor.
///
/// # Errors
///
/// Returns [`Error::Configuration`] naming `name` when an index exceeds `i64::MAX`.
pub(crate) fn index_tensor<B: Backend>(
    indices: &[usize],
    device: &B::Device,
    name: &str,
) -> Result<Tensor<B, 1, Int>> {
    let values = plan::signed_indices(indices)
        .ok_or_else(|| Error::configuration(name, "index exceeds the signed tensor-index range"))?;
    Ok(Tensor::from_data(
        TensorData::new(values, [indices.len()]),
        device,
    ))
}

/// Creates compact implicit positions `[[L, L + 1, …, L + n - 1]]` on `device`.
///
/// # Errors
///
/// Returns [`Error::Configuration`] naming `name` when a position exceeds `i64`.
pub(crate) fn implicit_position_ids<B: Backend>(
    baseline: usize,
    count: usize,
    device: &B::Device,
    name: &str,
) -> Result<Tensor<B, 2, Int>> {
    let values = plan::implicit_positions(baseline, count)
        .ok_or_else(|| Error::configuration(name, "implicit position exceeds i64"))?;
    Ok(Tensor::from_data(
        TensorData::new(values, [1, count]),
        device,
    ))
}

/// Rejects a dynamic-NTK rotary context that cached keys cannot follow.
///
/// Dynamic NTK rescales the rotary base `θ` once the evaluated context exceeds
/// the original window `L₀`. Keys cached under the previous base would then
/// disagree with queries rotated under the new one, so cached decoding supports
/// only `context <= L₀`; static scalings are always accepted. `family` names
/// the model in the error.
///
/// # Errors
///
/// Returns [`Error::BackendCapability`] for a dynamic-NTK context beyond `L₀`.
pub(crate) fn ensure_static_rotary_context(
    scaling: RopeScaling,
    context: usize,
    family: &str,
) -> Result<()> {
    if let RopeScaling::DynamicNtk(dynamic) = scaling
        && context > dynamic.original_max_position_embeddings()
    {
        return Err(Error::BackendCapability(format!(
            "{family} dynamic-NTK decoding beyond the original context requires re-rotating cached keys"
        )));
    }
    Ok(())
}

/// Concatenates per-row `[1, S, d]` results into `[B, S, d]`.
///
/// Rows must share one dtype; a mismatch (for example an F16 placeholder next
/// to F32 router logits) is reported instead of reaching a backend `cat`,
/// which may panic or silently reinterpret data.
///
/// # Errors
///
/// Returns [`Error::Shape`] for an empty row list and
/// [`Error::BackendCapability`] for mixed dtypes.
pub(crate) fn concat_rows<B, K>(rows: Vec<Tensor<B, 3, K>>, name: &str) -> Result<Tensor<B, 3, K>>
where
    B: Backend,
    K: TensorKind<B> + BasicOps<B>,
{
    let Some(first) = rows.first() else {
        return Err(Error::shape(name, "at least one batch row", "no rows"));
    };
    let dtype = first.dtype();
    if let Some(mismatch) = rows.iter().find(|row| row.dtype() != dtype) {
        return Err(Error::BackendCapability(format!(
            "{name} rows must share one dtype: found {dtype:?} and {:?}",
            mismatch.dtype()
        )));
    }
    Ok(Tensor::cat(rows, 0))
}

#[cfg(test)]
#[path = "rows_tests.rs"]
mod tests;
