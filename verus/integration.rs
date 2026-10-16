//! Verus harness for the experiment-integration package's pure translation
//! logic.
//!
//! Verifies, in place, the specifications attached to:
//!
//! - `src/models/auto/defaults.rs`: `sampling_top_k` is Hugging Face's
//!   sampling `top_k` rule for `generation_config.json`. An explicit zero
//!   disables top-k and nothing else does; otherwise the requested (or default
//!   50) cutoff is clamped to the vocabulary, and for any nonempty vocabulary
//!   the result lies in `1..=vocabulary`, so a resolved checkpoint default
//!   always passes `GenerationConfig::validate`. `clamped_min_new_tokens` is
//!   `min(declared, max_new_tokens)` and lowers the declared minimum exactly
//!   when it exceeds the bound (when the loader reports it as unapplied).
//!   `canonical_token_ids` accepts an EOS or padding list exactly when every
//!   ID is in the vocabulary and fits `u32`, reporting the first offending ID
//!   in order, and otherwise returns the strictly increasing list of exactly
//!   the distinct IDs via the recursive `sorted_unique` over `merge_unique`.
//!   Beyond that result, the termination measure `⌈log₂(end - start)⌉` proves
//!   the split balanced (recursion depth at most `⌈log₂ n⌉`, by
//!   `lemma_halves_shrink` and `lemma_ceil_log2_monotonic`), and
//!   `merge_unique` is proved to run at most `|left| + |right|` iterations;
//!   the resulting `O(n log n)` total is the standard per-level sum, not a
//!   Verus theorem.
//! - `src/models/custom/mamba2_geometry.rs`: `mamba2_groups`, the Mamba-2 head
//!   grouping of a modular specification. A group count exists exactly when
//!   the group size is positive and divides the heads, and it then multiplies
//!   back to the head count, so translation never silently regroups heads.
//!
//! See `verus/README.md` for the method, toolchain, and command. Parsing,
//! tensor construction, floating-point defaults (`top_p`, `min_p`,
//! temperature, penalties), and the remaining generation-default semantics
//! are covered by the Rust test suite.

#![cfg_attr(verus_keep_ghost, feature(proc_macro_hygiene))]
#![allow(unused)]

#[path = "../src/models/auto/defaults.rs"]
pub mod defaults;

#[path = "../src/models/custom/mamba2_geometry.rs"]
pub mod mamba2_geometry;
