//! Fuzz cache mutations and transactional ring-buffer behavior.
//!
//! Independent logical-length and `slot = position mod capacity` oracles check
//! append, truncate, reorder, reset, commit, rollback, and rejected mutations.
//! The same input then drives the tensor-backed decode state (KV, ring,
//! mirrored, latent, and recurrent groups through step write sets) against a
//! per-row token-history oracle.

#![no_main]

libfuzzer_sys::fuzz_target!(|data: &[u8]| {
    transformers_fuzz::cache::exercise(data);
    transformers_fuzz::cache::exercise_tensor_state(data);
});
