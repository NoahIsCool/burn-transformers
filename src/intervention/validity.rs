//! Per-execution activation validity, independent of hook registrations.
//!
//! A point's Boolean indicator is broadcast over its declared tensor axes.
//! Applying it computes `where(valid, activation, 0)` while preserving shape,
//! dtype, and device. A hidden-state mask `[batch, token, 1]` can therefore suppress
//! all channels of a padded token before observation and after each edit. This
//! execution invariant is independent of whether any user callback is registered.

use burn::tensor::{Bool, Tensor, backend::Backend};

use super::{Point, PointInfo};
use crate::{Error, Result};

/// A broadcastable validity mask for one exact activation point.
///
/// `true` denotes a real activation. Invalid elements are forced to zero before
/// captures or callbacks, and after every edit. This describes an execution
/// invariant such as padding; it does not register an action or request a tensor
/// that an optimized kernel would otherwise leave unmaterialized.
#[derive(Clone, Debug)]
pub struct PointValidity<B: Backend> {
    /// Exact point name and semantic axes to which the indicator applies.
    pub(super) point: PointInfo,
    /// Flattened Boolean indicator with `true` denoting a valid activation.
    valid: Tensor<B, 1, Bool>,
    /// Original broadcast dimensions restored when the activation is emitted.
    shape: Vec<usize>,
}

impl<B: Backend> PointValidity<B> {
    /// Associate a typed point with a positive, broadcastable boolean mask.
    ///
    /// Rank is checked here; broadcast extents and device are checked against the
    /// actual tensor when the model emits the point. For `[batch, token, channel]`
    /// activations, a `[batch, token, 1]` mask applies to every channel.
    pub fn new<const D: usize>(point: Point<D>, valid: Tensor<B, D, Bool>) -> Result<Self> {
        if D == 0 || valid.dims().contains(&0) {
            return Err(Error::configuration(
                &point.info().name,
                "activation validity must have positive rank and extents",
            ));
        }
        Ok(Self {
            point: point.info(),
            shape: valid.dims().to_vec(),
            valid: valid.flatten(0, D - 1),
        })
    }

    /// Zeros invalid elements after checking rank, broadcast extents, and device.
    ///
    /// The mask uses `true == valid`; inversion is confined to Burn's `mask_fill`
    /// boundary. Shape, floating dtype, and gradients of valid elements are preserved.
    pub(super) fn apply<const D: usize>(&self, tensor: Tensor<B, D>) -> Result<Tensor<B, D>> {
        let shape: [usize; D] = self.shape.clone().try_into().map_err(|_| {
            Error::configuration(
                &self.point.name,
                "activation validity rank differs from the emitted tensor",
            )
        })?;
        let actual = tensor.dims();
        if !crate::experiment::bounds::broadcasts_to(&shape, &actual) {
            return Err(Error::shape(
                &self.point.name,
                format!("validity broadcastable to {actual:?}"),
                format!("{shape:?}"),
            ));
        }
        if self.valid.device() != tensor.device() {
            return Err(Error::configuration(
                &self.point.name,
                "activation validity and emitted tensor must share a device",
            ));
        }
        Ok(tensor.mask_fill(
            self.valid.clone().reshape(shape).expand(actual).bool_not(),
            0.0,
        ))
    }
}
