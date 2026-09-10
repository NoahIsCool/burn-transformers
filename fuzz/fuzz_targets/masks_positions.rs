//! Fuzz padding-mask polarity, broadcast geometry, and absolute positions.
//!
//! The Flex harness checks `[batch,head,query,key]` exclusion masks against a
//! scalar oracle and tests padding offsets, malformed shapes, and integer overflow.

#![no_main]

libfuzzer_sys::fuzz_target!(|data: &[u8]| {
    transformers_fuzz::masks::exercise(data);
});
