//! Fuzz modular architecture specifications from validation to cached execution.
//!
//! Valid specifications must round-trip through JSON, construct unless they
//! hit a documented capability gap, run to finite logits, and (without routed
//! experts) reproduce the stateless forward through prefill and decode.

#![no_main]

libfuzzer_sys::fuzz_target!(|data: &[u8]| {
    transformers_fuzz::modular_specs::exercise(data);
});
