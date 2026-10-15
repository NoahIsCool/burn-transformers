//! Nemotron-H names for the shared Mamba-2 selective state-space mixer.
//!
//! Nemotron-H's `linear_attention` blocks are plain Mamba-2 mixers with
//! grouped `B/C`, a depthwise causal convolution, and a gated grouped RMS
//! normalization. The mathematics, precision policy, and cache invariants live
//! once in [`crate::mamba2`] so other hybrid families and custom compositions
//! can reuse them; these aliases keep the family's established names.

pub use crate::mamba2::{Mamba2Mode, TimeStepClamp, TimeStepLimits};

/// Numerical and tensor geometry of one Nemotron-H Mamba-2 block.
pub type NemotronMamba2Config = crate::mamba2::Mamba2Config;

/// Checkpoint-compatible Nemotron-H Mamba-2 mixer.
pub type NemotronMamba2<B> = crate::mamba2::Mamba2<B>;

/// External matrix and convolution state of one Nemotron-H Mamba-2 block.
pub type NemotronMamba2State<B> = crate::mamba2::Mamba2State<B>;

/// Recurrent decode or chunked prefill execution of a Nemotron-H Mamba-2 block.
pub type NemotronMamba2Mode = crate::mamba2::Mamba2Mode;

/// Gated grouped RMS normalization of a Nemotron-H Mamba-2 block.
pub type NemotronGroupedGatedRmsNorm<B> = crate::mamba2::GatedGroupRmsNorm<B>;
