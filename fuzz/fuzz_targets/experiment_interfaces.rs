#![no_main]

//! libFuzzer entry point for bounded experiment-interface oracles.

libfuzzer_sys::fuzz_target!(|data: &[u8]| {
    transformers_fuzz::experiment_interfaces::exercise(data);
});
