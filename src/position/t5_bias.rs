//! Learned T5 relative-position bias tensors.
//!
//! T5 does not add an absolute position vector to token embeddings. Instead, a
//! query at position `i` and key at position `j` select a learned scalar per
//! attention head from a bucketed displacement `j - i`. The scalar table has
//! shape `[buckets, heads]`; lookup produces `[1, heads, query, key]`, which is
//! broadcast across the batch and added to the raw, unscaled `QK^T` scores.
//!
//! Bucket computation intentionally remains on the host. Its `query * key`
//! integer matrix is tiny relative to attention scores, contains no trainable
//! values, and reuses the overflow-safe reference algorithm in
//! [`T5RelativePositionBuckets`]. The learned lookup and all downstream
//! attention mathematics remain backend tensors and therefore autodiff-safe.

use burn::{
    module::{Initializer, Module, Param},
    tensor::{Int, Tensor, TensorData, backend::Backend, module::embedding},
};

use crate::{Error, Result};

use super::T5RelativePositionBuckets;

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
    /// Creates a relative-bias configuration with T5's common initialization scale.
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

    /// Validates table dimensions, bucket geometry, and initialization values.
    pub fn validate(&self) -> Result<()> {
        if self.num_heads == 0 {
            return Err(Error::configuration(
                "t5_relative_bias.num_heads",
                "must be non-zero",
            ));
        }
        if self.num_buckets > i64::MAX as usize {
            return Err(Error::configuration(
                "t5_relative_bias.num_buckets",
                "must fit in the signed integer index representation",
            ));
        }
        if self.num_buckets.checked_mul(self.num_heads).is_none() {
            return Err(Error::configuration(
                "t5_relative_bias.shape",
                "bucket count times head count overflows usize",
            ));
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
    /// while keys still begin at position zero.
    pub fn forward(
        &self,
        query_length: usize,
        key_length: usize,
        query_offset: usize,
        device: &B::Device,
    ) -> Result<Tensor<B, 4>> {
        if query_length == 0 || key_length == 0 {
            return Err(Error::shape(
                "t5_relative_bias",
                "non-zero query and key lengths",
                format!("query={query_length}, key={key_length}"),
            ));
        }
        let _query_end = query_offset.checked_add(query_length).ok_or_else(|| {
            Error::configuration(
                "t5_relative_bias.query_offset",
                "offset plus query length overflows usize",
            )
        })?;
        let element_count = query_length.checked_mul(key_length).ok_or_else(|| {
            Error::configuration(
                "t5_relative_bias.shape",
                "query length times key length overflows usize",
            )
        })?;
        let mut indices = Vec::new();
        indices.try_reserve_exact(element_count).map_err(|_| {
            Error::BackendCapability(format!(
                "could not reserve {element_count} host indices for T5 relative bias"
            ))
        })?;
        for query in 0..query_length {
            let query_position = query_offset + query;
            for key_position in 0..key_length {
                let bucket = self
                    .buckets
                    .bucket_for_positions(query_position, key_position);
                indices.push(i64::try_from(bucket).map_err(|_| {
                    Error::configuration(
                        "t5_relative_bias.num_buckets",
                        "bucket index does not fit in i64",
                    )
                })?);
            }
        }

        let indices = Tensor::<B, 2, Int>::from_data(
            TensorData::new(indices, [query_length, key_length]),
            device,
        );
        Ok(embedding(self.weight.val(), indices)
            .permute([2, 0, 1])
            .unsqueeze_dim(0))
    }

    /// Returns the validated host bucket mapping used by this table.
    #[must_use]
    pub const fn buckets(&self) -> T5RelativePositionBuckets {
        self.buckets
    }
}

#[cfg(test)]
mod tests {
    use burn::{
        backend::Flex,
        module::Param,
        tensor::{Int, Tensor, TensorData, Tolerance},
    };

    use super::*;

    /// Learned lookup values follow the exact `[head, query, key]` permutation.
    #[test]
    fn tensor_bias_matches_bucket_lookup_and_head_layout() {
        let device = Default::default();
        let mut bias = T5RelativePositionBiasConfig::new(2, 4, 16, true)
            .init::<Flex>(&device)
            .expect("valid bias");
        bias.weight = Param::from_tensor(Tensor::from_floats(
            [[0.0, 1.0], [10.0, 11.0], [20.0, 21.0], [30.0, 31.0]],
            &device,
        ));

        let actual = bias
            .forward(2, 3, 0, &device)
            .expect("valid tensor geometry");
        assert_eq!(actual.dims(), [1, 2, 2, 3]);
        let buckets = bias.buckets();
        let mut expected = vec![0.0_f32; 12];
        for head in 0..2 {
            for query in 0..2 {
                for key in 0..3 {
                    let bucket = buckets.bucket_for_positions(query, key);
                    expected[head * 6 + query * 3 + key] = bucket as f32 * 10.0 + head as f32;
                }
            }
        }
        actual.into_data().assert_approx_eq::<f32>(
            &TensorData::new(expected, [1, 2, 2, 3]),
            Tolerance::absolute(1e-6),
        );
    }

    /// Cached-query offsets select displacements against absolute key positions.
    #[test]
    fn query_offset_matches_absolute_bucket_positions() {
        let device = Default::default();
        let mut bias = T5RelativePositionBiasConfig::new(1, 8, 32, false)
            .init::<Flex>(&device)
            .expect("valid bias");
        bias.weight = Param::from_tensor(
            Tensor::<Flex, 1, Int>::arange(0..8, &device)
                .float()
                .reshape([8, 1]),
        );

        let actual = bias
            .forward(1, 4, 3, &device)
            .expect("one cached query")
            .into_data();
        actual.assert_approx_eq::<f32>(
            &TensorData::from([[[[3.0, 2.0, 1.0, 0.0]]]]),
            Tolerance::absolute(1e-6),
        );
    }

    /// Every invalid public shape or scalar is rejected before tensor construction.
    #[test]
    fn validation_rejects_zero_invalid_and_overflowing_inputs() {
        let device = Default::default();
        for config in [
            T5RelativePositionBiasConfig::new(0, 4, 16, true),
            T5RelativePositionBiasConfig::new(1, 3, 16, true),
            T5RelativePositionBiasConfig::new(1, 8, 4, false),
            T5RelativePositionBiasConfig::new(usize::MAX, 2, 8, false),
            T5RelativePositionBiasConfig::new(1, 8, 32, false).with_initializer_std(f64::NAN),
        ] {
            assert!(config.validate().is_err());
        }
        let bias = T5RelativePositionBiasConfig::new(1, 8, 32, false)
            .init::<Flex>(&device)
            .expect("valid bias");
        assert!(bias.forward(0, 1, 0, &device).is_err());
        assert!(bias.forward(1, 0, 0, &device).is_err());
        assert!(bias.forward(2, 1, usize::MAX, &device).is_err());
    }
}
