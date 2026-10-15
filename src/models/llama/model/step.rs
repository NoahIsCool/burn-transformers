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
//! padding gets exact zeros. The integer plans come from the Verus-verified
//! functions in `plan.rs`.

use burn::tensor::{Bool, Int, Tensor, TensorData, backend::Backend};

use crate::{Error, Result, architecture::bool_tensor_from_values, models::shared::StatefulRows};

use super::super::plan::{compaction_sources, is_one_run, restore_slots, slot_positions};

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
}

impl StepLayout {
    /// Plans the step of validated `rows` appending at committed lengths `starts`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Shape`] when no row is active (callers handle that
    /// case without a step) and [`Error::Configuration`] when a cache
    /// position overflows `usize` or `i64`.
    pub(super) fn plan(rows: &StatefulRows, starts: &[usize]) -> Result<Self> {
        let (batch, sequence) = (rows.batch, rows.sequence);
        let valid_columns = &rows.valid_indices;
        let active = valid_columns.iter().filter(|columns| !columns.is_empty());
        let (lo, hi) = active.fold((usize::MAX, 0), |(lo, hi), columns| {
            (lo.min(columns[0]), hi.max(columns[columns.len() - 1] + 1))
        });
        if hi == 0 {
            return Err(Error::shape(
                "decoder.step",
                "at least one row with a new token",
                "only inactive rows",
            ));
        }
        let contiguous = valid_columns.iter().all(|columns| is_one_run(columns));
        let (source, block) = if contiguous {
            (BlockSource::Slice { offset: lo }, hi - lo)
        } else {
            let block = rows.token_counts.iter().copied().max().unwrap_or(0);
            let sources = valid_columns
                .iter()
                .flat_map(|columns| compaction_sources(columns, block))
                .collect();
            (BlockSource::Gather { sources }, block)
        };
        let columns: Vec<usize> = valid_columns
            .iter()
            .map(|columns| match (&source, columns.first()) {
                (BlockSource::Slice { offset }, Some(&first)) => first - offset,
                _ => 0,
            })
            .collect();
        let mut layout = Self {
            batch,
            sequence,
            block,
            source,
            counts: rows.token_counts.clone(),
            columns,
            positions: Vec::with_capacity(batch * block),
            valid: Vec::with_capacity(batch * block),
            restore: Vec::with_capacity(batch * sequence),
        };
        for row in 0..batch {
            layout.plan_row(row, starts[row], &valid_columns[row])?;
        }
        Ok(layout)
    }

    /// Appends row `row`'s slot positions, slot flags, and restore indices.
    fn plan_row(&mut self, row: usize, start: usize, valid_columns: &[usize]) -> Result<()> {
        let (column, count) = (self.columns[row], self.counts[row]);
        let positions = slot_positions(start, column, count, self.block).ok_or_else(|| {
            Error::configuration("decoder.step.positions", "a cache position overflows usize")
        })?;
        for position in positions {
            self.positions.push(i64::try_from(position).map_err(|_| {
                Error::configuration("decoder.step.positions", "a cache position exceeds i64")
            })?);
        }
        self.valid
            .extend((0..self.block).map(|slot| (column..column + count).contains(&slot)));
        self.restore.extend(restore_slots(
            valid_columns,
            self.sequence,
            self.block,
            column,
        ));
        Ok(())
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
        let last: Vec<usize> = self
            .columns
            .iter()
            .zip(&self.counts)
            .map(|(&column, &count)| (column + count).saturating_sub(1))
            .collect();
        let index = int_tensor(&last, [batch, 1], &device)?
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
    let values = values
        .iter()
        .map(|&value| i64::try_from(value))
        .collect::<core::result::Result<Vec<_>, _>>()
        .map_err(|_| Error::configuration("decoder.step.index", "an index exceeds i64"))?;
    Ok(Tensor::from_data(TensorData::new(values, shape), device))
}

#[cfg(test)]
#[path = "step_tests.rs"]
mod tests;
