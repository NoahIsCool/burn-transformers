//! Fuzz Hugging Face `rope_parameters` through the shared RoPE translator.
//!
//! Arbitrary flat and per-layer-type mappings and family contexts must either
//! translate into a configuration with non-negative frequencies (finite inside
//! the envelope `theta >= 1`, LongRoPE factors `>= 2^-64`; beyond it `f32`
//! overflow is open finding ROPE-F32-1/2) or fail with a contextual
//! configuration error under either unused-field policy; `Ignore` must accept
//! whatever `Reject` accepts, identically.

#![no_main]

libfuzzer_sys::fuzz_target!(|data: &[u8]| {
    transformers_fuzz::rope_parameters::exercise(data);
});
