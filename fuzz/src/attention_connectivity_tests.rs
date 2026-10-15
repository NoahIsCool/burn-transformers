//! Deterministic regression seeds for the attention-connectivity fuzz harness.

use super::*;

/// A problem without token masks.
fn plain(
    connectivity: AttentionConnectivity,
    causal: bool,
    queries: usize,
    keys: usize,
) -> Problem {
    Problem {
        connectivity,
        causal,
        geometry: MaskGeometry::new(2, 2, queries, keys),
        query_valid: None,
        key_valid: None,
    }
}

/// Builds, checks against the oracle, and runs attention for one problem.
fn check(problem: &Problem) {
    let built = build(problem);
    assert_eq!(built.is_some(), problem.accepts(), "{problem:?}");
    check_window_ranges(problem);
    if let Some(mask) = built {
        let values = mask_values(problem, mask);
        check_mask(problem, &values);
        check_attention(&mut Bytes::new(&[1, 2, 3, 4, 5, 6, 7]), problem, &values);
    }
}

/// Every rule with and without causality over cached prefixes.
#[test]
fn rules_match_the_scalar_oracle() {
    for connectivity in [
        AttentionConnectivity::Global,
        AttentionConnectivity::SlidingWindow { window: 1 },
        AttentionConnectivity::SlidingWindow { window: 3 },
        AttentionConnectivity::SlidingWindow { window: usize::MAX },
        AttentionConnectivity::ChunkedLocal { chunk: 2 },
        AttentionConnectivity::ChunkedLocal { chunk: 5 },
    ] {
        for causal in [false, true] {
            for (queries, keys) in [(1, 1), (3, 7), (7, 7), (1, 8)] {
                check(&plain(connectivity, causal, queries, keys));
            }
        }
    }
}

/// Padding consumes no window capacity, and fully padded rows exclude everything.
#[test]
fn padded_rows_use_valid_token_ordinals() {
    let mut padded = plain(
        AttentionConnectivity::SlidingWindow { window: 2 },
        true,
        3,
        5,
    );
    padded.key_valid = Some((
        2,
        5,
        vec![
            true, false, true, false, true, // ordinals 0, -, 1, -, 2
            false, false, false, false, false, // a fully padded row
        ],
    ));
    padded.query_valid = Some((2, 3, vec![true, false, true, false, false, false]));
    check(&padded);
    // Row 0, query 2 sits at key column 4 (ordinal 2); window 2 keeps ordinals 1 and 2.
    assert!(expected_excluded(&padded, 0, 2, 0));
    assert!(!expected_excluded(&padded, 0, 2, 2));
}

/// Chunk boundaries are absolute: a key start shifts them, even near `usize::MAX`.
#[test]
fn chunk_boundaries_follow_the_key_start() {
    for key_start in [0, 1, 3, usize::MAX - 8] {
        let mut shifted = plain(
            AttentionConnectivity::ChunkedLocal { chunk: 4 },
            false,
            6,
            6,
        );
        shifted.geometry = shifted.geometry.with_key_start(key_start);
        check(&shifted);
    }
    let mut overflowing = plain(AttentionConnectivity::Global, false, 2, 2);
    overflowing.geometry = overflowing.geometry.with_key_start(usize::MAX);
    assert!(!overflowing.accepts());
    check(&overflowing);
}

/// Documented rejections: zero rules, empty axes, misaligned lengths, and mis-shaped masks.
#[test]
fn invalid_requests_are_rejected() {
    check(&plain(
        AttentionConnectivity::SlidingWindow { window: 0 },
        true,
        2,
        2,
    ));
    check(&plain(
        AttentionConnectivity::ChunkedLocal { chunk: 0 },
        false,
        2,
        2,
    ));
    check(&plain(AttentionConnectivity::Global, true, 0, 2));
    check(&plain(
        AttentionConnectivity::SlidingWindow { window: 2 },
        true,
        4,
        2,
    ));
    // Unpadded global causal attention may have more queries than keys.
    let mut leading = plain(AttentionConnectivity::Global, true, 4, 2);
    assert!(leading.accepts());
    check(&leading);
    leading.key_valid = Some((2, 2, vec![true; 4]));
    assert!(!leading.accepts());
    check(&leading);
    let mut misshaped = plain(AttentionConnectivity::Global, false, 2, 2);
    misshaped.query_valid = Some((3, 2, vec![true; 6]));
    check(&misshaped);
}

/// Committed seeds keep their meaning.
#[test]
fn committed_seeds_pass() {
    for seed in [
        &include_bytes!("../seeds/attention_connectivity/sliding-padded")[..],
        &include_bytes!("../seeds/attention_connectivity/chunked-key-start")[..],
        &include_bytes!("../seeds/attention_connectivity/causal-more-queries")[..],
    ] {
        assert!(exercise(seed), "{seed:?}");
    }
    exercise(&[]);
    exercise(&[255; 128]);
}

/// Pseudo-random inputs cover many rule, geometry, and mask mixes.
#[test]
fn pseudo_random_inputs_agree_with_the_oracles() {
    let accepted = crate::lcg_inputs(0x636f_6e6e, 512, 160)
        .iter()
        .filter(|input| exercise(input))
        .count();
    // Both outcomes are common, so neither branch of the oracle is vacuous.
    assert!(
        (128..=448).contains(&accepted),
        "{accepted} of 512 accepted"
    );
}
