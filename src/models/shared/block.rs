//! Device-side packing of ragged stateful rows into one `[B, T]` block.
//!
//! The host plan comes from the Verus-verified [`super::block_plan`] module
//! (see its documentation for the equations); this module uploads it once per
//! call and applies it to tensors: [`BlockLayout::compact`] gathers each
//! row's valid tokens to the left of the block, [`BlockLayout::compact_inert`]
//! additionally overwrites padding entries with a known-valid value (token
//! IDs and explicit positions are validated only at valid columns, so the
//! source column a padding entry gathers may hold any sentinel, and a finished
//! row's padding gathers such an unvalidated column), [`BlockLayout::scatter`]
//! restores `[B, S, X]` results with exact zeros at padding (one gather from
//! the block with an appended zero row), and [`BlockLayout::visibility`]
//! describes the block's queries against one batched cache read. Fully valid
//! calls (every row has all `S` columns) upload no maps and run no gather or
//! scatter.
//!
//! The layout is family-independent. DeepSeek-V2 (latent attention) and Muse
//! Glimmer (grouped-query attention over full and ring caches) use it, since
//! both need [`BlockLayout::visibility`]'s position-ordered key limits for
//! bounded attention. Families whose cached attention needs no key limits do
//! not use `BlockLayout`: Qwen3.5's `CompactBatch` (which gpt-oss and
//! Qwen3-MoE also use) applies this module's host plan
//! [`super::block_plan::row_maps`] through `qwen3_5::plan::compact_row_plan`,
//! the Llama family plans its own steps with `llama::plan::plan_step`, and
//! Nemotron-H uses its flat `CompactPlan`, which also feeds the Mamba-2 scan
//! only the valid tokens.

use burn::tensor::{BasicOps, Bool, Int, Tensor, TensorData, TensorKind, backend::Backend};

use crate::{
    Error, Result, cache::KvBatchSegment, mla::PositionalVisibility, models::shared::StatefulRows,
};

use super::{
    block_plan::{
        RowMaps, batch_extents, clamp_limits, identity_layout, implicit_positions, key_limits,
        query_slots, query_validity, row_maps,
    },
    plan::signed_indices,
};

/// Device-side packing of one call's rows into a `[B, T]` block.
///
/// `None` from [`BlockLayout::new`] means every row is inactive.
pub(crate) struct BlockLayout<B: Backend> {
    /// Rectangular width `S`.
    sequence: usize,
    /// Packed width `T = max_b n_b`.
    block: usize,
    /// `[B, T]` source columns, or `None` when packing is the identity.
    gather: Option<Tensor<B, 2, Int>>,
    /// `[B, S]` block entries (`T` = zero row), or `None` for the identity.
    scatter: Option<Tensor<B, 2, Int>>,
    /// Implicit positions `[B, T]`: `L_b + k` for valid entries, zero for padding.
    implicit_positions: Tensor<B, 2, Int>,
    /// Query cache slots `[B, T]` (`L_b + k`, or `L_b` for padding).
    query_slots: Tensor<B, 2, Int>,
    /// Query validity `[B, T]` (`k < n_b`), or `None` when every entry is valid.
    query_valid: Option<Tensor<B, 2, Bool>>,
    /// Key-prefix limits of the block's query rows over position-ordered keys.
    key_limits: Vec<usize>,
}

impl<B: Backend> BlockLayout<B> {
    /// Plans the block for validated rows appended at `baselines` (per-row cache lengths).
    ///
    /// # Errors
    ///
    /// Returns [`Error::Configuration`] when a cursor or index overflows.
    pub(crate) fn new(
        prepared: &StatefulRows,
        baselines: &[usize],
        device: &B::Device,
    ) -> Result<Option<Self>> {
        let counts = &prepared.token_counts;
        let (block, extent) = batch_extents(baselines, counts).ok_or_else(|| {
            Error::configuration("stateful.batch", "cache cursors overflow usize")
        })?;
        if block == 0 {
            return Ok(None);
        }
        let (batch, sequence) = (prepared.batch, prepared.sequence);
        // Every host plan below is verified in `verus/models.rs`.
        let identity = identity_layout(counts, block, sequence);
        let upload = |values: Vec<usize>, shape: [usize; 2]| -> Result<Tensor<B, 2, Int>> {
            let signed = signed_indices(&values)
                .ok_or_else(|| Error::configuration("stateful.batch", "index exceeds i64"))?;
            Ok(Tensor::from_data(TensorData::new(signed, shape), device))
        };
        let (gather, scatter) = if identity {
            (None, None)
        } else {
            let maps = prepared
                .valid_indices
                .iter()
                .map(|columns| row_maps(columns, sequence, block))
                .collect::<Vec<RowMaps>>();
            (
                Some(upload(
                    maps.iter().flat_map(|row| row.gather.clone()).collect(),
                    [batch, block],
                )?),
                Some(upload(
                    maps.iter().flat_map(|row| row.scatter.clone()).collect(),
                    [batch, sequence],
                )?),
            )
        };
        let slots = query_slots(baselines, counts, block, extent).concat();
        let implicit = implicit_positions(baselines, counts, block).concat();
        let query_valid = query_validity(counts, block)
            .map(|rows| Tensor::from_data(TensorData::new(rows.concat(), [batch, block]), device));
        Ok(Some(Self {
            sequence,
            block,
            gather,
            scatter,
            implicit_positions: upload(implicit, [batch, block])?,
            query_slots: upload(slots, [batch, block])?,
            query_valid,
            key_limits: key_limits(baselines, counts, block, extent),
        }))
    }

    /// Returns the packed width `T`.
    pub(crate) const fn block(&self) -> usize {
        self.block
    }

    /// Returns implicit positions `[B, T]` (`L_b + k`, zero at padding).
    pub(crate) fn implicit_positions(&self) -> Tensor<B, 2, Int> {
        self.implicit_positions.clone()
    }

    /// Packs a rectangular `[B, S]` tensor into the `[B, T]` block.
    pub(crate) fn compact<K>(&self, rectangular: Tensor<B, 2, K>) -> Tensor<B, 2, K>
    where
        K: TensorKind<B> + BasicOps<B>,
    {
        match &self.gather {
            Some(gather) => rectangular.gather(1, gather.clone()),
            None => rectangular,
        }
    }

    /// Packs a rectangular `[B, S]` integer tensor and sets every padding entry to `fill`.
    ///
    /// Callers validate token IDs and explicit positions only at a row's
    /// valid columns, so masked columns may carry arbitrary sentinels (an
    /// out-of-vocabulary pad ID, `-1`, `i32::MAX`). [`Self::compact`] gathers
    /// padding entries from column `c_0` of a row, or from column `0` of a
    /// finished row whose every column is masked, so without this step a
    /// sentinel would reach an embedding gather (an out-of-bounds read) or a
    /// rotary table. `fill` must itself be valid for the consumer (token `0`,
    /// position `0`); padding entries are excluded from attention and from
    /// every scattered output, so the value never influences a result.
    /// Calls without padding entries (`query_valid` is `None`) skip the fill.
    pub(crate) fn compact_inert(
        &self,
        rectangular: Tensor<B, 2, Int>,
        fill: i64,
    ) -> Tensor<B, 2, Int> {
        let packed = self.compact(rectangular);
        match &self.query_valid {
            Some(valid) => packed.mask_fill(valid.clone().bool_not(), fill),
            None => packed,
        }
    }

    /// Restores block results `[B, T, X]` to `[B, S, X]` with exact zeros at padding.
    pub(crate) fn scatter<K>(&self, block: Tensor<B, 3, K>) -> Tensor<B, 3, K>
    where
        K: TensorKind<B> + BasicOps<B>,
    {
        let Some(scatter) = &self.scatter else {
            return block;
        };
        let [batch, _, width] = block.dims();
        let zero = Tensor::<B, 3, K>::zeros([batch, 1, width], (&block.device(), block.dtype()));
        let padded = Tensor::cat(vec![block, zero], 1);
        let index = scatter.clone().reshape([batch, self.sequence, 1]).expand([
            batch,
            self.sequence,
            width,
        ]);
        padded.gather(1, index)
    }

    /// Describes this block's queries against one batched cache read.
    ///
    /// Query `k` of row `b` sits at slot `L_b + k`; padded queries are
    /// invalid. When the read's keys are in position order (full storage,
    /// `positional`), the verified key limits bound each query chunk's key
    /// prefix; otherwise (ring slot order) every chunk reads all keys.
    pub(crate) fn visibility(
        &self,
        segment: &KvBatchSegment<B>,
        positional: bool,
    ) -> PositionalVisibility<B> {
        let keys = segment.key_positions().dims()[1];
        let limits = if positional {
            clamp_limits(&self.key_limits, keys)
        } else {
            vec![keys; self.block]
        };
        PositionalVisibility::cached(
            self.query_slots.clone(),
            self.query_valid.clone(),
            segment.key_positions().clone(),
            segment.key_valid().cloned(),
            limits,
        )
    }
}

#[cfg(test)]
#[path = "block_tests.rs"]
/// Packing, inert padding, scattering, and visibility geometry of ragged blocks.
mod tests;
