//! Fuzz causal, sliding-window, and chunked-local attention masks and masked attention.
//!
//! Masks must match a scalar position oracle for padded batches and shifted key
//! starts, window key ranges must be exact, and grouped-query attention under the
//! mask must match an `f64` softmax in both precision modes.

#![no_main]

libfuzzer_sys::fuzz_target!(|data: &[u8]| {
    transformers_fuzz::attention_connectivity::exercise(data);
});
