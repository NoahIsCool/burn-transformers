//! Host layout of one batched cache step: the block, its slots, and the way back.
//!
//! A cached call appends `n_b` valid tokens to row `b`, whose committed
//! length is `L_b`. The whole batch runs as one `[batch, Q]` *block* in which
//! row `b`'s tokens occupy the contiguous slots `[c_b, c_b + n_b)`:
//!
//! * when every row's valid columns already form one run (no padding, left
//!   or right padding), the block is the input's narrowest common column
//!   window `[lo, hi)`, cut with one slice, and `c_b` is the row's first
//!   valid column minus `lo`;
//! * otherwise (interior padding) every row is compacted with one gather into
//!   a right-padded block of width `Q = max_b n_b`, and `c_b = 0`.
//!
//! Slot `t` of row `b` holds cache position `L_b + (t - c_b)` inside the run;
//! outside it, a *fill* position (the row's last new position, else its last
//! cached one, else 0), so a padding slot still sees a key and stays finite.
//! These are the causal and sliding-window query positions, and the implicit
//! rotary positions. Inactive rows (`n_b = 0`, decode only) are all padding.
//!
//! Results return to the caller's `[batch, S]` columns by one gather from the
//! block extended with a zero slot: column `j = v_k` (the row's `k`-th valid
//! column) reads slot `c_b + k`, every padded column reads the zero slot, so
//! padding gets exact zeros. The integer plan is the Verus-verified
//! `plan_step` in `plan.rs`, which also proves that every row fits the
//! block, so the per-row plans it composes are called within their contracts.

use burn::tensor::{Bool, Int, Tensor, TensorData, backend::Backend};

use crate::{Error, Result, architecture::bool_tensor_from_values, models::shared::StatefulRows};

use super::super::plan::{StepPlanError, plan_step, signed_indices};

/// How the block is cut from the rectangular input.
#[derive(Debug, Clone, PartialEq, Eq)]
enum BlockSource {
    /// Every row is one run: the block is input columns `[offset, offset + block)`.
    Slice {
        /// First input column of the block.
        offset: usize,
    },
    /// Interior padding: slot `t` of row `b` gathers input column `sources[b * block + t]`.
    Gather {
        /// Row-major `[batch, block]` source columns.
        sources: Vec<usize>,
    },
}

/// The host plan of one batched step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct StepLayout {
    /// Rows `B`.
    batch: usize,
    /// Input columns `S`.
    sequence: usize,
    /// Block width `Q >= max_b n_b`.
    block: usize,
    /// Where the block comes from.
    source: BlockSource,
    /// First block slot `c_b` of each row's new tokens.
    columns: Vec<usize>,
    /// New tokens `n_b` of each row.
    counts: Vec<usize>,
    /// Row-major `[batch, block]` cache position of every slot.
    positions: Vec<i64>,
    /// Row-major `[batch, block]` flags: slot holds one of the row's new tokens.
    valid: Vec<bool>,
    /// Row-major `[batch, sequence]` block slot restoring each input column (`block` = zero slot).
    restore: Vec<usize>,
    /// Block slot of each row's last new token (slot zero for an inactive row).
    last: Vec<usize>,
}

impl StepLayout {
    /// Plans the step of validated `rows` appending at committed lengths `starts`.
    ///
    /// The integer layout is the Verus-verified `plan_step`; this wrapper
    /// only maps its errors and concatenates its per-row plans row-major.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Shape`] when no row is active (callers handle that
    /// case without a step) and [`Error::Configuration`] when a cache
    /// position overflows `usize` or `i64`.
    pub(super) fn plan(rows: &StatefulRows, starts: &[usize]) -> Result<Self> {
        let (batch, sequence) = (rows.batch, rows.sequence);
        let plan =
            plan_step(&rows.valid_indices, starts, sequence).map_err(|error| match error {
                StepPlanError::NoActiveRow => Error::shape(
                    "decoder.step",
                    "at least one row with a new token",
                    "only inactive rows",
                ),
                StepPlanError::PositionOverflow => Error::configuration(
                    "decoder.step.positions",
                    "a cache position overflows usize",
                ),
                StepPlanError::PositionBeyondI64 => {
                    Error::configuration("decoder.step.positions", "a cache position exceeds i64")
                }
            })?;
        let block = plan.block;
        let mut positions = Vec::with_capacity(batch * block);
        let mut valid = Vec::with_capacity(batch * block);
        let mut restore = Vec::with_capacity(batch * sequence);
        let mut sources = Vec::new();
        let mut last = Vec::with_capacity(batch);
        for row in plan.rows {
            positions.extend(row.positions);
            valid.extend(row.valid);
            restore.extend(row.restore);
            sources.extend(row.sources);
            last.push(row.last);
        }
        let source = match plan.offset {
            Some(offset) => BlockSource::Slice { offset },
            None => BlockSource::Gather { sources },
        };
        Ok(Self {
            batch,
            sequence,
            block,
            source,
            columns: plan.columns,
            counts: rows.token_counts.clone(),
            positions,
            valid,
            restore,
            last,
        })
    }

    /// Returns whether every block slot holds a new token (no padding slot).
    pub(super) fn is_dense(&self) -> bool {
        self.valid.iter().all(|&valid| valid)
    }

    /// Returns each row's first block slot `c_b`.
    pub(super) fn columns(&self) -> &[usize] {
        &self.columns
    }

    /// Returns each row's new-token count `n_b`.
    pub(super) fn counts(&self) -> &[usize] {
        &self.counts
    }

    /// Cuts the `[batch, Q]` block out of a rectangular `[batch, S]` tensor.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Configuration`] for indices beyond `i64`.
    pub(super) fn cut<B: Backend>(
        &self,
        rectangular: &Tensor<B, 2, Int>,
    ) -> Result<Tensor<B, 2, Int>> {
        match &self.source {
            BlockSource::Slice { offset } => Ok(rectangular.clone().narrow(1, *offset, self.block)),
            BlockSource::Gather { sources } => {
                let device = rectangular.device();
                let index = int_tensor(sources, [self.batch, self.block], &device)?;
                Ok(rectangular.clone().gather(1, index))
            }
        }
    }

    /// Uploads the `[batch, Q]` cache position of every slot.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Configuration`] only for an inconsistent layout.
    pub(super) fn slot_positions<B: Backend>(
        &self,
        device: &B::Device,
    ) -> Result<Tensor<B, 2, Int>> {
        Ok(Tensor::from_data(
            TensorData::new(self.positions.clone(), [self.batch, self.block]),
            device,
        ))
    }

    /// Uploads `[batch, Q]` slot flags: `true` where a slot holds a new token.
    pub(super) fn slot_validity<B: Backend>(&self, device: &B::Device) -> Tensor<B, 2, Bool> {
        bool_tensor_from_values(self.valid.clone(), [self.batch, self.block], device)
    }

    /// Cuts the `[batch, Q]` token block, with token 0 in every padding slot.
    ///
    /// Stateful calls validate token IDs only at valid columns, so padding
    /// columns may hold arbitrary sentinels (`-1`, out-of-vocabulary IDs);
    /// they must never reach the embedding lookup. Padding outputs are
    /// discarded, so any in-vocabulary ID serves; 0 always is one.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::cut`].
    pub(super) fn tokens<B: Backend>(
        &self,
        token_ids: &Tensor<B, 2, Int>,
    ) -> Result<Tensor<B, 2, Int>> {
        let tokens = self.cut(token_ids)?;
        if self.is_dense() {
            return Ok(tokens);
        }
        let padding = self.slot_validity::<B>(&tokens.device()).bool_not();
        Ok(tokens.mask_fill(padding, 0))
    }

    /// Returns rotary positions: explicit positions on new tokens, slot positions elsewhere.
    ///
    /// Explicit positions at padded columns may be arbitrary sentinels, so
    /// they are replaced by the slot's fill position before rotation.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::cut`].
    pub(super) fn rotary_positions<B: Backend>(
        &self,
        explicit: Option<&Tensor<B, 2, Int>>,
        slot_positions: &Tensor<B, 2, Int>,
    ) -> Result<Tensor<B, 2, Int>> {
        let Some(explicit) = explicit else {
            return Ok(slot_positions.clone());
        };
        let explicit = self.cut(explicit)?;
        if self.is_dense() {
            return Ok(explicit);
        }
        let padding = self.slot_validity::<B>(&explicit.device()).bool_not();
        Ok(explicit.mask_where(padding, slot_positions.clone()))
    }

    /// Restores `[batch, Q, d]` block states to `[batch, S, d]` with zeros at padding.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Shape`] when `block_states` is not `[batch, Q, d]`.
    pub(super) fn restore<B: Backend>(&self, block_states: Tensor<B, 3>) -> Result<Tensor<B, 3>> {
        let [batch, block, width] = block_states.dims();
        if batch != self.batch || block != self.block {
            return Err(Error::shape(
                "decoder.step.block_states",
                format!("[{}, {}, width]", self.batch, self.block),
                format!("{:?}", block_states.dims()),
            ));
        }
        let device = block_states.device();
        let zero_slot = Tensor::zeros([batch, 1, width], (&device, block_states.dtype()));
        let extended = Tensor::cat(vec![block_states, zero_slot], 1);
        let index = int_tensor(&self.restore, [batch, self.sequence], &device)?
            .unsqueeze_dim::<3>(2)
            .expand([batch, self.sequence, width]);
        Ok(extended.gather(1, index))
    }

    /// Selects each row's last new token `[batch, 1, d]`; inactive rows are zero.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Configuration`] for indices beyond `i64`.
    pub(super) fn last_states<B: Backend>(
        &self,
        block_states: Tensor<B, 3>,
    ) -> Result<Tensor<B, 3>> {
        let [batch, _, width] = block_states.dims();
        let device = block_states.device();
        let index = int_tensor(&self.last, [batch, 1], &device)?
            .unsqueeze_dim::<3>(2)
            .expand([batch, 1, width]);
        let selected = block_states.gather(1, index);
        if self.counts.iter().all(|&count| count > 0) {
            return Ok(selected);
        }
        let inactive: Vec<bool> = self.counts.iter().map(|&count| count == 0).collect();
        let inactive = bool_tensor_from_values::<B, 3>(inactive, [batch, 1, 1], &device)
            .expand([batch, 1, width]);
        Ok(selected.mask_fill(inactive, 0.0))
    }
}

/// Uploads row-major host indices as a `shape` Int tensor.
fn int_tensor<B: Backend, const D: usize>(
    values: &[usize],
    shape: [usize; D],
    device: &B::Device,
) -> Result<Tensor<B, D, Int>> {
    let values = signed_indices(values)
        .ok_or_else(|| Error::configuration("decoder.step.index", "an index exceeds i64"))?;
    Ok(Tensor::from_data(TensorData::new(values, shape), device))
}

#[cfg(test)]
#[path = "step_tests.rs"]
mod tests;
