//! Fuzz Hugging Face `rope_parameters` through the shared RoPE translator.
//!
//! Arbitrary flat and per-layer-type mappings and family contexts must either
//! translate into a configuration with finite, non-negative frequencies or fail
//! with a contextual configuration error; `Ignore` must accept whatever
//! `Reject` accepts, identically.

#![no_main]

libfuzzer_sys::fuzz_target!(|data: &[u8]| {
    transformers_fuzz::rope_parameters::exercise(data);
});
