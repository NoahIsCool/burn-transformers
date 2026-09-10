//! Compatibility imports for the shared multi-head latent-attention implementation.
//!
//! New experiments can import the same parameter modules from [`crate::mla`].

pub(crate) use crate::mla::MlaMixOutput;
pub use crate::mla::*;
