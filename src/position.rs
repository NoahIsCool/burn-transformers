//! Position-index and relative-position algorithms for transformer attention.
//!
//! Absolute position IDs and relative buckets solve different problems.
//! [`derive_batched_position_ids`] assigns contiguous absolute indices only to
//! valid tokens, so left padding does not rotate otherwise identical prompts by
//! different phases. [`RopeConfig`] validates the paired rotary subspace used
//! by RoPE, whose frequencies have the form
//! `theta^(-2 i / rotary_dimension)`. [`T5RelativePositionBuckets`] instead
//! compresses signed query-key displacement into exact buckets: short distances
//! remain distinct while long distances are spaced logarithmically.

mod ids;
mod rope;
mod t5;
mod t5_bias;
mod tensor;

pub use ids::{PositionIdError, derive_batched_position_ids, derive_position_ids};
pub use rope::{
    DynamicNtkScaling, LinearRopeScaling, RopeConfig, RopeConfigError, RopeScaling, YarnConfig,
    rotary_dimension_from_partial_factor,
};
pub use t5::{T5BucketError, T5RelativePositionBuckets};
pub use t5_bias::{T5RelativePositionBias, T5RelativePositionBiasConfig};
pub use tensor::{RotaryAngles, apply_rotary, inverse_frequencies, mrope_angles, rotary_angles};
