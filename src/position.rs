//! Position-index and relative-position algorithms for transformer attention.
//!
//! Absolute position IDs and relative buckets solve different problems.
//! [`derive_batched_position_ids`] assigns contiguous absolute indices only to
//! valid tokens, so left padding does not rotate otherwise identical prompts by
//! different phases. [`RopeConfig`] validates the paired rotary subspace used
//! by RoPE, whose frequencies have the form
//! `theta^(-2 i / rotary_dimension)`, together with a [`RopeScaling`]
//! context-extension policy (linear, dynamic NTK, YaRN, Llama 3, LongRoPE, or
//! proportional). [`RotaryFrequencies`] evaluates that policy on the host once,
//! and [`rotary_angles_from_frequencies`] / [`apply_rotary_to`] build and apply
//! F32 cosine/sine tables on the device. [`T5RelativePositionBuckets`] instead
//! compresses signed query-key displacement into exact buckets: short distances
//! remain distinct while long distances are spaced logarithmically, and
//! [`T5RelativePositionBias`] looks them up through a linear-size displacement
//! table.
//!
//! Pure integer index plans ([`mrope_frequency_axis`], [`relative_table_length`],
//! the T5 bucket assembly, and [`derive_batched_position_ids`]) are verified in
//! place with Verus through `verus/position.rs`. [`relative_table_index`] is
//! verified the same way as the specification of the device-side T5 table
//! index, which a unit test checks entry by entry against it.

mod frequencies;
mod ids;
mod index_plan;
mod rope;
mod rope_scaling;
mod t5;
mod t5_bias;
mod tensor;

pub use frequencies::{RotaryFrequencies, inverse_frequencies};
pub use ids::{PositionIdError, derive_batched_position_ids, derive_position_ids};
pub use index_plan::{mrope_frequency_axis, relative_table_index, relative_table_length};
pub use rope::{
    DynamicNtkScaling, LinearRopeScaling, RopeConfig, RopeConfigError, RopeScaling, YarnConfig,
    rotary_dimension_from_partial_factor,
};
pub use rope_scaling::{
    Llama3RopeScaling, LongRopeScaling, MAX_LONG_ROPE_FREQUENCIES, ProportionalRopeScaling,
};
pub use t5::{T5BucketError, T5RelativePositionBuckets};
pub use t5_bias::{T5RelativeBucketTable, T5RelativePositionBias, T5RelativePositionBiasConfig};
pub use tensor::{
    RotaryAngles, apply_rotary, apply_rotary_to, interleaved_to_split_half, mrope_angles,
    rotary_angles, rotary_angles_from_frequencies, split_half_to_interleaved,
};
