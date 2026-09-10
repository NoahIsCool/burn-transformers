//! Fuzz generation-parameter validation and token-stopping transitions.
//!
//! Arbitrary floating-point bits exercise numerical guards; bounded token streams
//! check EOS, longest-stop, and length precedence against an independent oracle.

#![no_main]

libfuzzer_sys::fuzz_target!(|data: &[u8]| {
    transformers_fuzz::generation::exercise(data);
});
