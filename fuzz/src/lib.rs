#![deny(missing_docs)]
#![forbid(unsafe_code)]

//! Bounded fuzz oracles shared by libFuzzer entry points and ordinary unit tests.
//!
//! Inputs never initialize a model or read downloaded weights. Configuration parsing keeps
//! adversarial numeric dimensions while bounding layer counts that control validation loops.
//! Tensor tests use at most four sequences and sixteen tokens on the host-only Flex backend.
//! Artifact discovery sees only tiny, disposable local files and strict metadata limits.

pub mod artifacts;
pub mod cache;
pub mod configuration;
pub mod experiment_interfaces;
pub mod generation;
pub mod masks;

/// Maximum bytes consumed from any one fuzzer input.
pub const MAX_INPUT_BYTES: usize = 4096;

/// A total byte decoder: exhausted input yields zero, and loops are separately bounded.
pub(crate) struct Bytes<'a> {
    /// Remaining unconsumed bytes.
    remaining: &'a [u8],
}

impl<'a> Bytes<'a> {
    /// Restricts the decoder to the shared input-size budget.
    pub(crate) fn new(input: &'a [u8]) -> Self {
        Self {
            remaining: &input[..input.len().min(MAX_INPUT_BYTES)],
        }
    }

    /// Returns one byte, or zero when the input has ended.
    pub(crate) fn byte(&mut self) -> u8 {
        match self.remaining.split_first() {
            Some((&value, rest)) => {
                self.remaining = rest;
                value
            }
            None => 0,
        }
    }

    /// Samples small values plus machine-integer overflow boundaries without allocation.
    pub(crate) fn size(&mut self) -> usize {
        match self.byte() {
            255 => usize::MAX,
            254 => usize::MAX - 1,
            value => usize::from(value),
        }
    }

    /// Decodes every IEEE-754 bit pattern, including NaN, infinities and signed zero.
    pub(crate) fn float(&mut self) -> f64 {
        let bits = core::array::from_fn(|_| self.byte());
        f64::from_le_bytes(bits)
    }
}
