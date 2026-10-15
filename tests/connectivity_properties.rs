//! Metamorphic properties of attention connectivity masks.
//!
//! The fuzz target `attention_connectivity` compares each mask entry with a
//! scalar oracle. These properties relate masks of *different* requests:
//!
//! - **Window monotonicity.** Widening a sliding window never hides a key.
//! - **Padding invariance.** Inserting padded key columns before the query
//!   suffix changes nothing among real tokens, because windows and chunks count
//!   valid-token ordinals ("padding consumes no window").
//! - **Chunk periodicity.** Shifting the key start by a multiple of the chunk
//!   width leaves a chunked-local mask unchanged.
//!
//! Cases default to 96 and honor `PROPTEST_CASES`; the RNG seed is fixed.

#![cfg(feature = "flex")]

use burn::{
    backend::Flex,
    tensor::{Int, Tensor, TensorData},
};
use proptest::prelude::*;
use transformers::{
    TokenMask,
    attention::{AttentionConnectivity, MaskGeometry},
};

/// Deterministic CPU backend with F32 values and I32 indices.
type Cpu = Flex<f32, i32>;

/// Number of cases: `PROPTEST_CASES` when set, else `default`.
fn cases(default: u32) -> u32 {
    std::env::var("PROPTEST_CASES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

/// Builds the `[1, keys]` key mask of `valid`.
fn key_mask(valid: &[bool]) -> TokenMask<Cpu> {
    let values: Vec<i32> = valid.iter().map(|&valid| i32::from(valid)).collect();
    TokenMask::new(
        Tensor::<Cpu, 2, Int>::from_data(
            TensorData::new(values, [1, valid.len()]),
            &Default::default(),
        )
        .bool(),
    )
}

/// Exclusions of one `[1, 1, queries, keys]` request as host values (`true == excluded`).
fn exclusions(
    connectivity: AttentionConnectivity,
    causal: bool,
    queries: usize,
    keys: usize,
    key_start: usize,
    valid: Option<&[bool]>,
) -> Vec<bool> {
    let mask = valid.map(key_mask);
    let geometry = MaskGeometry::new(1, 1, queries, keys).with_key_start(key_start);
    connectivity
        .exclusion_mask::<Cpu>(causal, geometry, None, mask.as_ref(), &Default::default())
        .expect("valid request")
        .map_or_else(
            || vec![false; queries * keys],
            |mask| mask.into_data().to_vec::<bool>().expect("Boolean mask"),
        )
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: cases(96),
        rng_seed: proptest::test_runner::RngSeed::Fixed(20_261_015),
        ..ProptestConfig::default()
    })]

    /// A wider sliding window excludes a subset of what a narrower one excludes.
    #[test]
    fn widening_a_window_never_hides_a_key(
        keys in 1usize..=12,
        queries_seed in any::<prop::sample::Index>(),
        window in 1usize..=12,
        causal in any::<bool>(),
    ) {
        let queries = queries_seed.index(keys) + 1;
        let narrow = exclusions(AttentionConnectivity::SlidingWindow { window }, causal, queries, keys, 0, None);
        let wide = exclusions(
            AttentionConnectivity::SlidingWindow { window: window + 1 },
            causal,
            queries,
            keys,
            0,
            None,
        );
        for (&narrow, &wide) in narrow.iter().zip(&wide) {
            prop_assert!(!wide || narrow);
        }
    }

    /// Padded key columns before the queries do not change exclusions among real tokens.
    #[test]
    fn padding_consumes_no_window_or_chunk(
        real in 1usize..=8,
        queries_seed in any::<prop::sample::Index>(),
        padding in proptest::collection::vec(0usize..=2, 8),
        rule in 0usize..3,
        width in 1usize..=5,
        causal in any::<bool>(),
    ) {
        let queries = queries_seed.index(real) + 1;
        let connectivity = match rule {
            0 => AttentionConnectivity::Global,
            1 => AttentionConnectivity::SlidingWindow { window: width },
            _ => AttentionConnectivity::ChunkedLocal { chunk: width },
        };
        // Insert `padding[j]` padded columns before each real key that precedes the queries.
        let mut valid = Vec::new();
        let mut columns = Vec::new();
        for (key, &pads) in padding.iter().enumerate().take(real) {
            if key < real - queries {
                valid.extend(std::iter::repeat_n(false, pads));
            }
            columns.push(valid.len());
            valid.push(true);
        }
        let padded_keys = valid.len();
        let compact = exclusions(connectivity, causal, queries, real, 0, Some(&vec![true; real]));
        let padded = exclusions(connectivity, causal, queries, padded_keys, 0, Some(&valid));
        for query in 0..queries {
            for key in 0..real {
                prop_assert_eq!(
                    padded[query * padded_keys + columns[key]],
                    compact[query * real + key],
                    "query {} key {} of {:?}", query, key, valid
                );
            }
        }
    }

    /// Shifting the key start by whole chunks leaves a chunked-local mask unchanged.
    #[test]
    fn chunk_masks_are_periodic_in_the_key_start(
        keys in 1usize..=10,
        queries_seed in any::<prop::sample::Index>(),
        chunk in 1usize..=6,
        start in 0usize..12,
        periods in 1usize..1000,
        causal in any::<bool>(),
    ) {
        let queries = queries_seed.index(keys) + 1;
        let connectivity = AttentionConnectivity::ChunkedLocal { chunk };
        let base = exclusions(connectivity, causal, queries, keys, start, None);
        let shifted = exclusions(connectivity, causal, queries, keys, start + periods * chunk, None);
        prop_assert_eq!(base, shifted);
    }
}
