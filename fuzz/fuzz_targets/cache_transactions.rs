//! Fuzz cache mutations and transactional ring-buffer behavior.
//!
//! Independent logical-length and `slot = position mod capacity` oracles check
//! append, truncate, reorder, reset, commit, rollback, and rejected mutations.

#![no_main]

libfuzzer_sys::fuzz_target!(|data: &[u8]| {
    transformers_fuzz::cache::exercise(data);
});
