//! Fuzz checkpoint-index discovery and SafeTensors artifact validation.
//!
//! The shared harness writes only fixed names inside a disposable directory,
//! bounds headers and shard counts, and checks that accepted paths cannot escape.

#![no_main]

libfuzzer_sys::fuzz_target!(|data: &[u8]| {
    transformers_fuzz::artifacts::exercise(data);
});
