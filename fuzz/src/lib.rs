#![deny(missing_docs)]
#![forbid(unsafe_code)]

//! Bounded fuzz oracles shared by libFuzzer entry points and ordinary unit tests.
//!
//! Inputs never read downloaded weights; the only models built are modular specifications of
//! at most two layers and sixteen channels. Configuration parsing keeps adversarial numeric
//! dimensions while bounding layer counts that control validation loops. Tensor tests use at
//! most four sequences and sixteen tokens on the host-only Flex backend. Artifact and
//! tokenizer discovery see only tiny, disposable local files and strict metadata limits.
//!
//! Every harness decodes its input with [`Bytes`], a total decoder: exhausted input reads as
//! zero, so every byte string is a valid test case and libFuzzer never wastes executions on
//! rejected framing. Each module documents its byte layout next to its `exercise` function.

pub mod artifacts;
pub mod attention_connectivity;
pub mod cache;
pub mod chat_templates;
pub mod configuration;
pub mod experiment_interfaces;
pub mod generation;
pub mod logits_processing;
pub mod masks;
pub mod modular_specs;
pub mod rope_parameters;
pub mod safetensors_headers;

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

    /// Returns whether the low bit of the next byte is set.
    pub(crate) fn flag(&mut self) -> bool {
        self.byte() & 1 == 1
    }

    /// Returns a value in `0..bound` (zero when `bound` is zero).
    ///
    /// The modulo bias is irrelevant for fuzzing: every value stays reachable.
    pub(crate) fn below(&mut self, bound: usize) -> usize {
        if bound == 0 {
            0
        } else {
            usize::from(self.byte()) % bound
        }
    }

    /// Selects one entry of a non-empty table.
    pub(crate) fn pick<T: Clone>(&mut self, table: &[T]) -> T {
        table[self.below(table.len())].clone()
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

    /// Mostly boundary values that validators treat specially, sometimes raw bits.
    ///
    /// Raw IEEE-754 patterns rarely land on `0`, `1`, or the edges of `f32`, so a
    /// fuzzer reading only raw bits would almost never reach the branches a
    /// validator spends most of its code on. One selector byte chooses between
    /// [`INTERESTING_FLOATS`] (selector below 200) and an arbitrary bit pattern.
    pub(crate) fn interesting_float(&mut self) -> f64 {
        let selector = self.byte();
        if selector < 200 {
            INTERESTING_FLOATS[usize::from(selector) % INTERESTING_FLOATS.len()]
        } else {
            self.float()
        }
    }

    /// Returns up to `limit` raw bytes (fewer when the input ends).
    pub(crate) fn take(&mut self, limit: usize) -> &'a [u8] {
        let length = limit.min(self.remaining.len());
        let (taken, rest) = self.remaining.split_at(length);
        self.remaining = rest;
        taken
    }
}

/// Boundary floats for [`Bytes::interesting_float`].
///
/// The table covers the signed zeros, the identities `1` and `0.5`/`2` of
/// multiplicative parameters, non-finite values, the extremes of `f64`, and the
/// values just past the `f32` range (`f32::MAX * 2`, `1e-46`) at which a
/// finite `f64` parameter becomes infinite or zero once a kernel rounds it to
/// `f32`.
pub(crate) const INTERESTING_FLOATS: [f64; 24] = [
    0.0,
    -0.0,
    1.0,
    -1.0,
    0.5,
    2.0,
    0.25,
    1.5,
    32.0,
    100.0,
    1e-300,
    1e300,
    f64::MIN_POSITIVE,
    f64::MAX,
    f64::NAN,
    f64::INFINITY,
    f64::NEG_INFINITY,
    1.0 + f64::EPSILON,
    1.0 - f64::EPSILON / 2.0,
    f32::MAX as f64,
    f32::MAX as f64 * 2.0,
    f32::MIN_POSITIVE as f64,
    1e-46,
    10_000.0,
];

/// Whether `right` is an acceptable JSON round trip of the float `left`.
///
/// `serde_json` serializes the shortest exact decimal but, without its
/// `float_roundtrip` feature (which neither crate enables), parses long
/// mantissas with large exponents approximately: a campaign found
/// `3.7997969716361836e-262` read back four ulps away. Values must agree to a
/// relative `1e-14`, or absolutely within `f64::MIN_POSITIVE` for subnormals.
pub(crate) fn floats_round_trip(left: f64, right: f64) -> bool {
    let difference = (left - right).abs();
    left == right
        || difference <= 1e-14 * left.abs().max(right.abs())
        || difference <= f64::MIN_POSITIVE
}

/// Whether two JSON values are equal, with floats compared by [`floats_round_trip`].
///
/// Integers (`u64`/`i64` numbers) must match exactly, and an integer never
/// equals a float: only when both numbers are floats does the relative
/// tolerance apply, so a round trip that turned a token ID or context length
/// into a float, or moved a large integer, is still reported.
pub(crate) fn json_equal(left: &serde_json::Value, right: &serde_json::Value) -> bool {
    use serde_json::Value;
    match (left, right) {
        (Value::Number(left), Value::Number(right)) => {
            left == right
                || (left.is_f64()
                    && right.is_f64()
                    && matches!(
                        (left.as_f64(), right.as_f64()),
                        (Some(left), Some(right)) if floats_round_trip(left, right)
                    ))
        }
        (Value::Array(left), Value::Array(right)) => {
            left.len() == right.len() && left.iter().zip(right).all(|(a, b)| json_equal(a, b))
        }
        (Value::Object(left), Value::Object(right)) => {
            left.len() == right.len()
                && left.iter().all(|(key, value)| {
                    right.get(key).is_some_and(|other| json_equal(value, other))
                })
        }
        _ => left == right,
    }
}

/// Deterministic pseudo-random inputs for unit tests that stand in for a fuzzer.
///
/// A 32-bit linear congruential generator (Numerical Recipes constants) emits
/// `count` inputs of `length` bytes each, taking the high byte of every state.
/// The stream depends only on `seed`, so failures reproduce exactly.
#[cfg(test)]
pub(crate) fn lcg_inputs(seed: u32, count: usize, length: usize) -> Vec<Vec<u8>> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            (0..length)
                .map(|_| {
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    state.to_le_bytes()[3]
                })
                .collect()
        })
        .collect()
}

#[cfg(test)]
#[path = "lib_tests.rs"]
mod tests;
