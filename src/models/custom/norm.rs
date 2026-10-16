//! Normalization operators and their placement around one residual branch.
//!
//! A residual branch with sublayer `F` and normalization `N` is evaluated in
//! one of four topologies:
//!
//! | layout | output | parameters |
//! | --- | --- | --- |
//! | identity | `x + F(x)` | none |
//! | pre | `x + F(N(x))` | `before` |
//! | post | `N(x + F(x))` | `after` |
//! | sandwich | `x + N_after(F(N_before(x)))` | `before`, `after` |
//!
//! The layout is encoded entirely by which of the two optional norms exist, so
//! no flag can disagree with the allocated parameters and every combination
//! has exactly one meaning. Normalization acts on the channel axis of
//! `[batch, token, hidden]` states and preserves shape, dtype, and device.

use burn::{
    module::Module,
    nn::{LayerNorm, LayerNormConfig},
    tensor::{Tensor, backend::Backend},
};

use crate::{
    Error, Result,
    architecture::{NormPlacement, NormalizationSpec},
    normalization::{RmsNorm, RmsNormConfig},
};

/// Runtime normalization supporting both scale-only RMS and mean-subtracting LayerNorm.
#[derive(Module, Debug)]
pub enum ModularNorm<B: Backend> {
    /// Crate-native RMS normalization, including zero-centered checkpoint weights.
    Rms(
        /// RMS normalization parameters and epsilon for the residual width.
        RmsNorm<B>,
    ),
    /// Burn LayerNorm with an optional learned additive bias.
    Layer(
        /// Mean-centering layer normalization parameters and epsilon.
        LayerNorm<B>,
    ),
}

impl<B: Backend> ModularNorm<B> {
    /// Initializes the exact normalization convention requested by a specification.
    pub fn init(spec: NormalizationSpec, hidden_size: usize, device: &B::Device) -> Result<Self> {
        spec.validate("custom.normalization")?;
        if hidden_size == 0 {
            return Err(Error::configuration(
                "custom.normalization.hidden_size",
                "must be non-zero",
            ));
        }
        match spec {
            NormalizationSpec::Rms {
                epsilon,
                zero_centered,
            } => RmsNormConfig::new(hidden_size)
                .with_epsilon(epsilon)
                .with_zero_centered(zero_centered)
                .init(device)
                .map(Self::Rms),
            NormalizationSpec::Layer { epsilon, bias } => Ok(Self::Layer(
                LayerNormConfig::new(hidden_size)
                    .with_epsilon(epsilon)
                    .with_bias(bias)
                    .init(device),
            )),
        }
    }

    /// Normalizes the final channel of a `[batch, sequence, hidden]` tensor.
    pub fn forward(&self, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
        match self {
            Self::Rms(norm) => norm.forward(hidden),
            Self::Layer(norm) => norm.forward(hidden),
        }
    }
}

/// Before/after normalization parameters for one residual sublayer.
///
/// See the module documentation for the four layouts; which norms are present
/// fully determines the computation.
#[derive(Module, Debug)]
pub struct NormLayout<B: Backend> {
    /// Normalization applied to the sublayer input (pre and sandwich layouts).
    before: Option<ModularNorm<B>>,
    /// Normalization applied after the sublayer (post and sandwich layouts).
    after: Option<ModularNorm<B>>,
}

impl<B: Backend> NormLayout<B> {
    /// Allocates exactly the normalization parameters implied by one placement.
    pub fn init(
        spec: NormalizationSpec,
        placement: NormPlacement,
        hidden_size: usize,
        device: &B::Device,
    ) -> Result<Self> {
        let before = matches!(placement, NormPlacement::Pre | NormPlacement::Sandwich)
            .then(|| ModularNorm::init(spec, hidden_size, device))
            .transpose()?;
        let after = matches!(placement, NormPlacement::Post | NormPlacement::Sandwich)
            .then(|| ModularNorm::init(spec, hidden_size, device))
            .transpose()?;
        Ok(Self { before, after })
    }

    /// Creates an explicit unnormalized residual branch, `x + F(x)`.
    pub fn identity() -> Self {
        Self {
            before: None,
            after: None,
        }
    }

    /// Applies the optional normalization immediately before a sublayer.
    ///
    /// This never fails; the `Result` keeps the signature stable for
    /// composition code that propagates errors uniformly.
    pub fn prepare(&self, hidden: Tensor<B, 3>) -> Result<Tensor<B, 3>> {
        Ok(match &self.before {
            Some(norm) => norm.forward(hidden),
            None => hidden,
        })
    }

    /// Combines a sublayer value with its residual according to the layout.
    ///
    /// `Pre` and identity compute `x + F`; `Post` computes `N(x + F)`;
    /// sandwich normalization computes `x + N_after(F)`, the form used by
    /// sandwich-normalized decoders such as Muse Glimmer. This never fails.
    pub fn finish(&self, residual: Tensor<B, 3>, mixed: Tensor<B, 3>) -> Result<Tensor<B, 3>> {
        Ok(match (&self.before, &self.after) {
            (_, None) => residual + mixed,
            (None, Some(post)) => post.forward(residual + mixed),
            (Some(_), Some(sandwich)) => residual + sandwich.forward(mixed),
        })
    }
}

#[cfg(test)]
#[path = "norm_tests.rs"]
/// Normalization construction contracts.
mod tests;
