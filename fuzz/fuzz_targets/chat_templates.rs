//! Fuzz untrusted chat templates and messages through `LocalTokenizer::render_chat`.
//!
//! Rendering must never panic, must fail contextually on documented argument
//! errors and sandbox limits, must be deterministic with a pinned clock, and
//! must stay within time and output bounds.

#![no_main]

libfuzzer_sys::fuzz_target!(|data: &[u8]| {
    transformers_fuzz::chat_templates::exercise(data);
});
