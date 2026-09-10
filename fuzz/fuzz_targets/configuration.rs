//! Fuzz family configuration parsers and normalized architecture validators.
//!
//! Delegates to bounded JSON round trips without allocating model parameters;
//! extreme dimensions remain available to expose arithmetic validation errors.

#![no_main]

libfuzzer_sys::fuzz_target!(|data: &[u8]| {
    transformers_fuzz::configuration::exercise(data);
});
