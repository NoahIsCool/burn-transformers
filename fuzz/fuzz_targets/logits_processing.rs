//! Fuzz logits processing with ragged, out-of-vocabulary token history.
//!
//! A scalar `f64` oracle checks bias, EOS suppression before `min_new_tokens`,
//! no-repeat n-grams, the three penalties, and the sampling warpers; rebuilt,
//! incrementally recorded, stateless, and legacy masked histories must agree
//! bit for bit.

#![no_main]

libfuzzer_sys::fuzz_target!(|data: &[u8]| {
    transformers_fuzz::logits_processing::exercise(data);
});
