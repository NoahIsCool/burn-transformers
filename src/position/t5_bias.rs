//! Learned T5 relative-position bias tensors.
//!
//! T5 does not add an absolute position vector to token embeddings. Instead, a
//! query at position `i` and key at position `j` select a learned scalar per
//! attention head from a bucketed displacement `j - i`. The scalar table has
//! shape `[buckets, heads]`; lookup produces `[1, heads, query, key]`, which is
//! broadcast across the batch and added to the raw, unscaled `QK^T` scores.
//!
//! Because the bucket depends only on the displacement, a `[query, key]` bias
//! has just `query + key - 1` distinct buckets. [`T5RelativeBucketTable`]
//! evaluates those on the host in `O(query + key)` with the overflow-safe
//! reference algorithm of [`T5RelativePositionBuckets`] and uploads one short
//! vector. The device then builds the `[query, key]` table index
//! `j + (query - 1 - i)` as the broadcast sum of the key range and the
//! reversed query range, gathers bucket IDs, and looks up the learned table.
//! No per-element host work or quadratic upload remains, and the learned
//! lookup stays an autodiff-safe embedding.
//!
//! The index formula is specified by [`super::relative_table_index`], whose
//! host implementation Verus proves overflow-free, in bounds, and
//! displacement-preserving. The integer checks of
//! [`T5RelativePositionBias::bucket_table`] and
//! [`T5RelativePositionBiasConfig::validate`] are the verified plans
//! `plan_relative_table` and `relative_bias_shape_fault`: an accepted geometry
//! has the exact anchor and table length, and every `(query, key)` pair reads
//! the entry whose displacement from the anchor equals its true key-minus-query
//! displacement, including a query offset. The proofs cover those scalar
//! functions, not the device kernel; a unit test compares every device entry
//! with the scalar index.

use burn::{
    module::{Initializer, Module, Param},
    tensor::{Int, Tensor, TensorData, backend::Backend, module::embedding},
};

use crate::{Error, Result};

use super::T5RelativePositionBuckets;
use super::index_plan::{
    RelativeBiasShapeFault, RelativeTableFault, plan_relative_table, relative_bias_shape_fault,
};

/// Configuration for a learned T5 relative-attention bias table.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct T5RelativePositionBiasConfig {
    /// Number of independently biased attention heads.
    pub num_heads: usize,
    /// Total number of learned logarithmic displacement buckets.
    pub num_buckets: usize,
    /// Absolute distance at which the final logarithmic bucket saturates.
    pub max_distance: usize,
    /// Whether past and future positions use separate halves of the table.
    pub bidirectional: bool,
    /// Standard deviation used only before checkpoint weights are loaded.
    pub initializer_std: f64,
}

impl T5RelativePositionBiasConfig {
    /// Creates a relative-bias configuration with unit initialization scale.
    ///
    /// The unit standard deviation suits loading checkpoint weights, which
    /// replace the initialization. Randomly initialized T5 experiments should
    /// call [`Self::with_reference_initialization`], which reproduces the
    /// reference `initializer_factor * d_model^-0.5` (about 0.044 for
    /// `d_model = 512`, roughly 20 times smaller than one).
    pub fn new(
        num_heads: usize,
        num_buckets: usize,
        max_distance: usize,
        bidirectional: bool,
    ) -> Self {
        Self {
            num_heads,
            num_buckets,
            max_distance,
            bidirectional,
            initializer_std: 1.0,
        }
    }

    /// Selects the standard deviation for randomly initialized experiments.
    pub fn with_initializer_std(mut self, initializer_std: f64) -> Self {
        self.initializer_std = initializer_std;
        self
    }

    /// Uses the reference T5 initialization `initializer_factor * d_model^-0.5`.
    ///
    /// Hugging Face `T5PreTrainedModel._init_weights` draws the relative bias
    /// table from `N(0, (initializer_factor / sqrt(d_model))^2)`; checkpoints
    /// use `initializer_factor = 1`. Invalid inputs (zero width, non-finite or
    /// negative factor) produce a non-finite or negative standard deviation that
    /// [`Self::validate`] rejects.
    pub fn with_reference_initialization(self, d_model: usize, initializer_factor: f64) -> Self {
        let std = if d_model == 0 {
            f64::NAN
        } else {
            initializer_factor / (d_model as f64).sqrt()
        };
        self.with_initializer_std(std)
    }

    /// Validates table dimensions, bucket geometry, and initialization values.
    ///
    /// The integer shape checks (non-zero heads, `i64` bucket IDs, and a
    /// representable `buckets * heads` table) are the Verus-verified
    /// `relative_bias_shape_fault`.
    pub fn validate(&self) -> Result<()> {
        match relative_bias_shape_fault(self.num_heads, self.num_buckets) {
            Some(RelativeBiasShapeFault::ZeroHeads) => {
                return Err(Error::configuration(
                    "t5_relative_bias.num_heads",
                    "must be non-zero",
                ));
            }
            Some(RelativeBiasShapeFault::BucketIndexRange) => {
                return Err(Error::configuration(
                    "t5_relative_bias.num_buckets",
                    "must fit in the signed integer index representation",
                ));
            }
            Some(RelativeBiasShapeFault::TableOverflow) => {
                return Err(Error::configuration(
                    "t5_relative_bias.shape",
                    "bucket count times head count overflows usize",
                ));
            }
            None => {}
        }
        T5RelativePositionBuckets::new(self.num_buckets, self.max_distance, self.bidirectional)
            .map_err(|error| Error::configuration("t5_relative_bias.buckets", error.to_string()))?;
        if !self.initializer_std.is_finite() || self.initializer_std < 0.0 {
            return Err(Error::configuration(
                "t5_relative_bias.initializer_std",
                "must be finite and non-negative",
            ));
        }
        Ok(())
    }

    /// Initializes the trainable `[buckets, heads]` scalar table.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Result<T5RelativePositionBias<B>> {
        self.validate()?;
        let buckets =
            T5RelativePositionBuckets::new(self.num_buckets, self.max_distance, self.bidirectional)
                .map_err(|error| {
                    Error::configuration("t5_relative_bias.buckets", error.to_string())
                })?;
        Ok(T5RelativePositionBias {
            weight: Initializer::Normal {
                mean: 0.0,
                std: self.initializer_std,
            }
            .init([self.num_buckets, self.num_heads], device),
            buckets,
        })
    }
}

/// Learned relative-position bias shared by all layers in one T5 stack.
#[derive(Module, Debug)]
pub struct T5RelativePositionBias<B: Backend> {
    /// Bias lookup table stored as `[bucket, head]`, matching T5 checkpoints.
    pub weight: Param<Tensor<B, 2>>,
    /// Validated displacement-to-bucket mapping, which owns no parameters.
    #[module(skip)]
    buckets: T5RelativePositionBuckets,
}

impl<B: Backend> T5RelativePositionBias<B> {
    /// Builds additive bias shaped `[1, heads, query_length, key_length]`.
    ///
    /// `query_offset` is the absolute position of the first query. Stateless
    /// encoder/decoder passes use zero; cached decoding can request only a suffix
    /// while keys still begin at position zero. This evaluates
    /// [`Self::bucket_table`] and [`Self::forward_with_table`]; callers that
    /// reuse one geometry across layers can build the table once.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::bucket_table`].
    pub fn forward(
        &self,
        query_length: usize,
        key_length: usize,
        query_offset: usize,
        device: &B::Device,
    ) -> Result<Tensor<B, 4>> {
        let table = self.bucket_table(query_length, key_length, query_offset)?;
        self.forward_with_table(&table, device)
    }

    /// Computes the `query_length + key_length - 1` displacement buckets on the host.
    ///
    /// Entry `t` holds the bucket of the anchor query at absolute position
    /// `query_offset + query_length - 1` and key position `t`. Every
    /// `(query, key)` pair reads the entry with the same displacement, so this
    /// table determines the complete bias. The work is linear in the lengths.
    ///
    /// # Errors
    ///
    /// Returns a shape error for an empty axis; a configuration error when the
    /// `query * key` output, the offset, or an `i64` table index cannot be
    /// represented; and a backend-capability error when host memory for the
    /// table cannot be reserved.
    pub fn bucket_table(
        &self,
        query_length: usize,
        key_length: usize,
        query_offset: usize,
    ) -> Result<T5RelativeBucketTable> {
        // The verified plan rejects, in order: an empty axis; a `query * key`
        // output that cannot exist; a query block whose last position
        // `query_offset + query_length - 1` is unrepresentable; and a table
        // whose indices exceed the device's `i64` range. The table length
        // `q + k - 1 <= q * k` needs no separate check.
        let plan = plan_relative_table(query_length, key_length, query_offset).map_err(
            |fault| match fault {
                RelativeTableFault::EmptyAxis => Error::shape(
                    "t5_relative_bias",
                    "non-zero query and key lengths",
                    format!("query={query_length}, key={key_length}"),
                ),
                RelativeTableFault::OutputOverflow => Error::configuration(
                    "t5_relative_bias.shape",
                    "query length times key length overflows usize",
                ),
                RelativeTableFault::OffsetOverflow => Error::configuration(
                    "t5_relative_bias.query_offset",
                    "offset plus query length overflows usize",
                ),
                RelativeTableFault::IndexRange => Error::configuration(
                    "t5_relative_bias.shape",
                    "query length plus key length exceeds the i64 index range",
                ),
            },
        )?;
        let table_length = plan.length;
        let anchor = plan.anchor;
        let mut buckets = Vec::new();
        buckets.try_reserve_exact(table_length).map_err(|_| {
            Error::BackendCapability(format!(
                "could not reserve {table_length} host buckets for T5 relative bias"
            ))
        })?;
        for key_position in 0..table_length {
            let bucket = self.buckets.bucket_for_positions(anchor, key_position);
            // Configuration validation bounds the bucket count by i64::MAX.
            buckets.push(i64::try_from(bucket).map_err(|_| {
                Error::configuration(
                    "t5_relative_bias.num_buckets",
                    "bucket index does not fit in i64",
                )
            })?);
        }
        Ok(T5RelativeBucketTable {
            buckets,
            mapping: self.buckets,
            query_length,
            key_length,
        })
    }

    /// Looks up the learned bias for a precomputed displacement table.
    ///
    /// Gathers bucket IDs from the uploaded `query + key - 1` table with the
    /// device index `j + (query_length - 1 - i)` (the formula of
    /// [`super::relative_table_index`]) and passes them through the learned
    /// `[buckets, heads]` embedding, giving `[1, heads, query_length,
    /// key_length]`. The bucket IDs are identical to a pairwise evaluation, so
    /// the result equals it exactly.
    ///
    /// # Errors
    ///
    /// Returns a configuration error when `table` was built by a bias with a
    /// different bucket mapping (bucket count, saturation distance, or
    /// direction mode): its IDs would index this bias's embedding out of
    /// range or select the wrong learned scalars.
    pub fn forward_with_table(
        &self,
        table: &T5RelativeBucketTable,
        device: &B::Device,
    ) -> Result<Tensor<B, 4>> {
        if table.mapping != self.buckets {
            return Err(Error::configuration(
                "t5_relative_bias.table",
                format!(
                    "bucket table was built for {:?}, but this bias uses {:?}",
                    table.mapping, self.buckets
                ),
            ));
        }
        let query_length = table.query_length;
        let key_length = table.key_length;
        let buckets = Tensor::<B, 1, Int>::from_data(
            TensorData::new(table.buckets.clone(), [table.buckets.len()]),
            device,
        );
        let bucket_ids = buckets
            .select(
                0,
                table_entries::<B>(query_length, key_length, device)
                    .reshape([query_length * key_length]),
            )
            .reshape([query_length, key_length]);
        Ok(embedding(self.weight.val(), bucket_ids)
            .permute([2, 0, 1])
            .unsqueeze_dim(0))
    }

    /// Returns the validated host bucket mapping used by this table.
    #[must_use]
    pub const fn buckets(&self) -> T5RelativePositionBuckets {
        self.buckets
    }
}

/// Builds the `[query, key]` displacement-table index on the device.
///
/// Entry `(i, j)` is [`super::relative_table_index`]`(i, j, query_length) =
/// j + (query_length - 1 - i)`: the key range broadcast against the reversed
/// query range `query_length - 1, ..., 0`. Callers pass lengths accepted by
/// [`T5RelativePositionBias::bucket_table`], which guarantees every entry is
/// below `query_length + key_length - 1` and fits in `i64`.
fn table_entries<B: Backend>(
    query_length: usize,
    key_length: usize,
    device: &B::Device,
) -> Tensor<B, 2, Int> {
    let keys = Tensor::<B, 1, Int>::arange(0..key_length as i64, device)
        .reshape([1, key_length])
        .expand([query_length, key_length]);
    let reversed_queries = Tensor::<B, 1, Int>::arange(0..query_length as i64, device)
        .flip([0])
        .reshape([query_length, 1])
        .expand([query_length, key_length]);
    keys + reversed_queries
}

/// Host displacement buckets for one `[query, key]` relative-bias geometry.
///
/// Built by [`T5RelativePositionBias::bucket_table`]. Entry `t` is the bucket
/// for displacement `t - (query_length - 1)` relative to the anchor query; the
/// table is immutable and can be reused by every layer that shares the
/// geometry, offset, and bucket mapping. It records that mapping, so
/// [`T5RelativePositionBias::forward_with_table`] rejects a table built by a
/// differently configured bias.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct T5RelativeBucketTable {
    /// `query_length + key_length - 1` bucket IDs in increasing displacement order.
    buckets: Vec<i64>,
    /// Bucket mapping of the bias that built the table.
    mapping: T5RelativePositionBuckets,
    /// Number of queries the table serves.
    query_length: usize,
    /// Number of keys the table serves.
    key_length: usize,
}

impl T5RelativeBucketTable {
    /// Returns the bucket IDs ordered by increasing key-minus-query displacement.
    #[must_use]
    pub fn bucket_ids(&self) -> &[i64] {
        &self.buckets
    }

    /// Returns the bucket mapping of the bias that built this table.
    #[must_use]
    pub const fn mapping(&self) -> T5RelativePositionBuckets {
        self.mapping
    }

    /// Returns the number of queries.
    #[must_use]
    pub const fn query_length(&self) -> usize {
        self.query_length
    }

    /// Returns the number of keys.
    #[must_use]
    pub const fn key_length(&self) -> usize {
        self.key_length
    }
}

#[cfg(test)]
#[path = "t5_bias_tests.rs"]
mod tests;
