//! Fuzz generated SafeTensors headers (including microscaling dtypes) and shard indexes.
//!
//! Discovery must agree with an independent byte-accounting oracle on every
//! accept/reject decision and inventoried field, and the official SafeTensors
//! parser must accept every file discovery accepts.

#![no_main]

libfuzzer_sys::fuzz_target!(|data: &[u8]| {
    transformers_fuzz::safetensors_headers::exercise(data);
});
