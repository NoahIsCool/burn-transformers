//! Tensor realization of right-padded compact batches.
//!
//! [`CompactBatch`] uploads the index maps of a [`CompactPlan`] and applies
//! them with one `select` each over row-major flattened tensors (see the
//! planning module for the equations). Every map that may address padding is
//! applied to its source with one zero row appended, so padding receives
//! exact zeros without a scatter. Index tensors are uploaded on first use and
//! reused for every tensor of the call (final states, per-layer diagnostics,
//! router outputs).
//!
//! Dense batches, where every compact slot is valid, skip the packing maps,
//! and identity batches (no padding at all, the usual unmasked prefill and
//! decode) skip every map: the model then runs on the caller's tensors as is.

use std::cell::OnceCell;

use burn::tensor::{BasicOps, Bool, Int, Tensor, TensorData, TensorKind, backend::Backend};

use crate::{Error, Result, models::shared::index_tensor};

use super::plan::{CompactPlan, plan_compaction, query_positions};

/// One stateful call's compact layout and lazily uploaded index maps.
#[derive(Debug)]
pub(super) struct CompactBatch<B: Backend> {
    /// Host plan of every map.
    plan: CompactPlan,
    /// Device receiving every uploaded index.
    device: B::Device,
    /// `[B, N]`, `true` at padded compact slots; `None` for a dense batch.
    padding: Option<Tensor<B, 2, Bool>>,
    /// Uploaded `scatter` map, `[B * S]`.
    scatter: OnceCell<Tensor<B, 1, Int>>,
    /// Uploaded `packed` map, `[T]`.
    packed: OnceCell<Tensor<B, 1, Int>>,
    /// Uploaded `unpack` map, `[B * N]`.
    unpack: OnceCell<Tensor<B, 1, Int>>,
}

impl<B: Backend> CompactBatch<B> {
    /// Plans and prepares the compact layout of rows with valid columns `columns`.
    ///
    /// # Errors
    ///
    /// Returns a shape error for unsorted or out-of-range columns and a
    /// configuration error when an index exceeds Burn's signed range.
    pub(super) fn new(columns: &[Vec<usize>], sequence: usize, device: &B::Device) -> Result<Self> {
        let plan = plan_compaction(columns, sequence).ok_or_else(|| {
            Error::shape(
                "nemotron_h.compact_batch",
                format!("sorted unique valid columns below {sequence} with representable extents"),
                format!("{columns:?}"),
            )
        })?;
        let padding = (!plan.is_dense()).then(|| {
            let flags = plan
                .unpack
                .iter()
                .map(|&source| source == plan.tokens())
                .collect::<Vec<_>>();
            Tensor::from_data(TensorData::new(flags, [plan.batch(), plan.width]), device)
        });
        Ok(Self {
            plan,
            device: device.clone(),
            padding,
            scatter: OnceCell::new(),
            packed: OnceCell::new(),
            unpack: OnceCell::new(),
        })
    }

    /// Returns the host plan.
    pub(super) const fn plan(&self) -> &CompactPlan {
        &self.plan
    }

    /// Returns the compact width `N`.
    pub(super) const fn width(&self) -> usize {
        self.plan.width
    }

    /// Returns the valid tokens of every row.
    pub(super) fn counts(&self) -> &[usize] {
        &self.plan.counts
    }

    /// Compacts rectangular token IDs `[B, S]` into `[B, N]`; padded slots hold token zero.
    ///
    /// Padding columns of the input may hold arbitrary sentinels, so padded
    /// compact slots are overwritten with the always-valid ID zero.
    ///
    /// # Errors
    ///
    /// Returns a shape error when `tokens` is not `[B, S]` and a configuration
    /// error for an index beyond Burn's signed range.
    pub(super) fn compact_tokens(&self, tokens: Tensor<B, 2, Int>) -> Result<Tensor<B, 2, Int>> {
        let [batch, sequence] = tokens.dims();
        if [batch, sequence] != [self.plan.batch(), self.plan.sequence] {
            return Err(Error::shape(
                "nemotron_h.compact_batch.tokens",
                format!("[{}, {}]", self.plan.batch(), self.plan.sequence),
                format!("{:?}", tokens.dims()),
            ));
        }
        if self.plan.is_identity() {
            return Ok(tokens);
        }
        let gather = index_tensor(
            &self.plan.gather,
            &self.device,
            "nemotron_h.compact_batch.gather",
        )?;
        let compact = tokens
            .reshape([batch * sequence])
            .select(0, gather)
            .reshape([batch, self.plan.width]);
        Ok(match &self.padding {
            Some(padding) => compact.mask_fill(padding.clone(), 0),
            None => compact,
        })
    }

    /// Restores compact `[B, N, D]` values to `[B, S, D]` with zeros at padding.
    ///
    /// # Errors
    ///
    /// Returns a shape error for a mis-sized input.
    pub(super) fn to_rectangular<K>(&self, compact: Tensor<B, 3, K>) -> Result<Tensor<B, 3, K>>
    where
        K: TensorKind<B> + BasicOps<B>,
    {
        let width = self.check_compact(&compact, "nemotron_h.compact_batch.to_rectangular")?;
        if self.plan.is_identity() {
            return Ok(compact);
        }
        let rows = self.plan.batch() * self.plan.width;
        let index = self.index(&self.scatter, &self.plan.scatter, "scatter")?;
        Ok(with_zero_row(compact.reshape([rows, width]))
            .select(0, index)
            .reshape([self.plan.batch(), self.plan.sequence, width]))
    }

    /// Packs the valid tokens of compact `[B, N, D]` values into `[T, D]`.
    ///
    /// # Errors
    ///
    /// Returns a shape error for a mis-sized input.
    pub(super) fn pack<K>(&self, compact: Tensor<B, 3, K>) -> Result<Tensor<B, 2, K>>
    where
        K: TensorKind<B> + BasicOps<B>,
    {
        let width = self.check_compact(&compact, "nemotron_h.compact_batch.pack")?;
        let flat = compact.reshape([self.plan.batch() * self.plan.width, width]);
        if self.plan.is_dense() {
            return Ok(flat);
        }
        let index = self.index(&self.packed, &self.plan.packed, "packed")?;
        Ok(flat.select(0, index))
    }

    /// Restores packed `[T, D]` values to compact `[B, N, D]` with zeros at padding.
    ///
    /// # Errors
    ///
    /// Returns a shape error when `packed` does not hold `T` tokens.
    pub(super) fn unpack<K>(&self, packed: Tensor<B, 2, K>) -> Result<Tensor<B, 3, K>>
    where
        K: TensorKind<B> + BasicOps<B>,
    {
        let [tokens, width] = packed.dims();
        if tokens != self.plan.tokens() {
            return Err(Error::shape(
                "nemotron_h.compact_batch.unpack",
                format!("[{}, width]", self.plan.tokens()),
                format!("{:?}", packed.dims()),
            ));
        }
        let shape = [self.plan.batch(), self.plan.width, width];
        if self.plan.is_dense() {
            return Ok(packed.reshape(shape));
        }
        let index = self.index(&self.unpack, &self.plan.unpack, "unpack")?;
        Ok(with_zero_row(packed).select(0, index).reshape(shape))
    }

    /// Selects each row's last valid compact token as `[B, 1, D]`; inactive rows are zero.
    ///
    /// # Errors
    ///
    /// Returns a shape error for a mis-sized input.
    pub(super) fn last_tokens<K>(&self, compact: Tensor<B, 3, K>) -> Result<Tensor<B, 3, K>>
    where
        K: TensorKind<B> + BasicOps<B>,
    {
        let width = self.check_compact(&compact, "nemotron_h.compact_batch.last_tokens")?;
        let rows = self.plan.batch();
        if self.plan.is_dense() {
            return Ok(compact.narrow(1, self.plan.width - 1, 1));
        }
        let index = index_tensor(
            &self.plan.last,
            &self.device,
            "nemotron_h.compact_batch.last",
        )?;
        Ok(
            with_zero_row(compact.reshape([rows * self.plan.width, width]))
                .select(0, index)
                .reshape([rows, 1, width]),
        )
    }

    /// Absolute positions `[B, N]` of every compact slot after `baselines`.
    ///
    /// # Errors
    ///
    /// Returns a configuration error when a position exceeds `i64`.
    pub(super) fn query_positions(&self, baselines: &[usize]) -> Result<Tensor<B, 2, Int>> {
        let positions = query_positions(baselines, self.plan.width).ok_or_else(|| {
            Error::configuration(
                "nemotron_h.position",
                "cached position exceeds the signed index range",
            )
        })?;
        Ok(Tensor::from_data(
            TensorData::new(positions, [baselines.len(), self.plan.width]),
            &self.device,
        ))
    }

    /// Requires `[B, N, D]` and returns `D`.
    fn check_compact<K>(&self, compact: &Tensor<B, 3, K>, name: &str) -> Result<usize>
    where
        K: TensorKind<B> + BasicOps<B>,
    {
        let [batch, width, channels] = compact.dims();
        if [batch, width] != [self.plan.batch(), self.plan.width] || width == 0 {
            return Err(Error::shape(
                name,
                format!("[{}, {}, channels]", self.plan.batch(), self.plan.width),
                format!("{:?}", compact.dims()),
            ));
        }
        Ok(channels)
    }

    /// Returns an uploaded map, uploading it on first use.
    fn index(
        &self,
        cell: &OnceCell<Tensor<B, 1, Int>>,
        indices: &[usize],
        name: &str,
    ) -> Result<Tensor<B, 1, Int>> {
        if let Some(index) = cell.get() {
            return Ok(index.clone());
        }
        let index = index_tensor(indices, &self.device, name)?;
        Ok(cell.get_or_init(|| index).clone())
    }
}

/// Appends one zero row to `[rows, D]`, the target of every padding index.
fn with_zero_row<B, K>(flat: Tensor<B, 2, K>) -> Tensor<B, 2, K>
where
    B: Backend,
    K: TensorKind<B> + BasicOps<B>,
{
    let [_, width] = flat.dims();
    let zero = Tensor::<B, 2, K>::zeros([1, width], (&flat.device(), flat.dtype()));
    Tensor::cat(vec![flat, zero], 0)
}

#[cfg(test)]
#[path = "batch_tests.rs"]
mod tests;
