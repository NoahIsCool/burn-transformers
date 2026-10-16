//! Executable checks of the Verus-verified MLA widths and key prefixes against brute force.

use super::*;

/// Extents are checked in field order, then rotary parity, then the query rank.
#[test]
fn axis_faults_follow_field_order() {
    assert_eq!(mla_axis_fault(8, 2, 3, 2, 4, 5, None), None);
    assert_eq!(mla_axis_fault(8, 2, 3, 2, 4, 5, Some(3)), None);
    let zero = [
        (MlaAxis::HiddenSize, [0, 0, 0, 0, 0, 0]),
        (MlaAxis::NumHeads, [1, 0, 0, 0, 0, 0]),
        (MlaAxis::QkNopeHeadDim, [1, 1, 0, 0, 0, 0]),
        (MlaAxis::QkRopeHeadDim, [1, 1, 1, 0, 0, 0]),
        (MlaAxis::ValueHeadDim, [1, 1, 1, 2, 0, 0]),
        (MlaAxis::KvLoraRank, [1, 1, 1, 2, 1, 0]),
    ];
    for (axis, [hidden, heads, nope, rope, value, rank]) in zero {
        assert_eq!(
            mla_axis_fault(hidden, heads, nope, rope, value, rank, Some(0)),
            Some(MlaAxisFault::Zero(axis))
        );
    }
    assert_eq!(
        mla_axis_fault(1, 1, 1, 3, 1, 1, Some(0)),
        Some(MlaAxisFault::OddRotary)
    );
    assert_eq!(
        mla_axis_fault(1, 1, 1, 2, 1, 1, Some(0)),
        Some(MlaAxisFault::ZeroQueryRank)
    );
}

/// Successful widths are exact for direct and low-rank queries.
#[test]
fn widths_are_exact() {
    let expected = MlaWidths {
        qk_head_dim: 5,
        compressed_width: 7,
        expanded_width: 14,
        query_width: 10,
        value_width: 8,
    };
    assert_eq!(mla_widths(8, 2, 3, 2, 4, 5, None), Ok(expected));
    assert_eq!(mla_widths(8, 2, 3, 2, 4, 5, Some(6)), Ok(expected));
}

/// The shared width helpers equal checked arithmetic and name their own faults.
#[test]
fn shared_width_helpers_are_exact() {
    for (nope, rope) in [
        (3, 2),
        (0, 0),
        (usize::MAX, 0),
        (usize::MAX, 1),
        (1, usize::MAX),
    ] {
        assert_eq!(
            qk_head_width(nope, rope),
            nope.checked_add(rope).ok_or(MlaWidthFault::QkHeadDim)
        );
        assert_eq!(
            compressed_width(nope, rope),
            nope.checked_add(rope).ok_or(MlaWidthFault::CompressedWidth)
        );
    }
    assert_eq!(expanded_width(2, 3, 4), Ok(14));
    assert_eq!(
        expanded_width(1, usize::MAX, 1),
        Err(MlaWidthFault::ExpandedPerHead)
    );
    // The per-head sum is checked before the head product.
    assert_eq!(
        expanded_width(2, usize::MAX, 1),
        Err(MlaWidthFault::ExpandedPerHead)
    );
    assert_eq!(
        expanded_width(usize::MAX, 1, 1),
        Err(MlaWidthFault::ExpandedHeads)
    );
    assert_eq!(expanded_width(usize::MAX, 0, 1), Ok(usize::MAX));
}

/// Width faults name the first overflowing sum, product, or matrix.
#[test]
fn width_faults_name_the_first_overflow() {
    let max = usize::MAX;
    assert_eq!(
        mla_widths(1, 1, max, 2, 1, 1, None),
        Err(MlaWidthFault::QkHeadDim)
    );
    assert_eq!(
        mla_widths(1, 1, 1, 2, 1, max, None),
        Err(MlaWidthFault::CompressedWidth)
    );
    assert_eq!(
        mla_widths(1, 1, 1, 2, max, 1, None),
        Err(MlaWidthFault::ExpandedPerHead)
    );
    assert_eq!(
        mla_widths(1, max, 1, 2, 1, 1, None),
        Err(MlaWidthFault::ExpandedHeads)
    );
    assert_eq!(
        mla_widths(1, max / 2, 1, 4, 1, 1, None),
        Err(MlaWidthFault::QueryWidth)
    );
    let matrix = |matrix, input, output| MlaWidthFault::Matrix {
        matrix,
        input,
        output,
    };
    assert_eq!(
        mla_widths(max, 1, 1, 2, 1, 1, None),
        Err(matrix(MlaMatrix::Query, max, 3))
    );
    assert_eq!(
        mla_widths(max, 1, 1, 2, 1, 1, Some(2)),
        Err(matrix(MlaMatrix::QueryDown, max, 2))
    );
    assert_eq!(
        mla_widths(1, 1, 1, 2, 1, 1, Some(max / 2 + 1)),
        Err(matrix(MlaMatrix::QueryUp, max / 2 + 1, 3))
    );
    assert_eq!(
        mla_widths(max / 3, 1, 1, 2, 1, 4, None),
        Err(matrix(MlaMatrix::JointKeyValue, max / 3, 6))
    );
    assert_eq!(
        mla_widths(1, 2, 1, 2, 1, max / 4 + 1, None),
        Err(matrix(MlaMatrix::KeyValueExpansion, max / 4 + 1, 4))
    );
    assert_eq!(
        mla_widths(2, 1, 1, 2, max / 2 + 1, 1, None),
        Err(matrix(MlaMatrix::Output, max / 2 + 1, 2))
    );
}

/// Suffix bounds are each row's exact causal prefix and satisfy the contract.
#[test]
fn suffix_limits_are_exact_causal_prefixes() {
    assert_eq!(suffix_key_limits(3, 3), [1, 2, 3]);
    assert_eq!(suffix_key_limits(2, 5), [4, 5]);
    assert!(suffix_key_limits(0, 4).is_empty());
    for keys in 0..=7usize {
        for queries in 0..=keys {
            let limits = suffix_key_limits(queries, keys);
            assert!(key_limits_valid(&limits, queries, keys));
            for (row, &limit) in limits.iter().enumerate() {
                let visible = (0..keys).filter(|&key| key + queries <= keys + row).count();
                assert_eq!(limit, visible, "Q={queries} K={keys} row={row}");
            }
        }
    }
}

/// Brute-force statement of the prefix contract.
fn contract(limits: &[usize], queries: usize, keys: usize) -> bool {
    limits.len() == queries
        && limits.iter().all(|&limit| (1..=keys).contains(&limit))
        && limits.windows(2).all(|pair| pair[0] <= pair[1])
}

/// The validator accepts exactly contract-satisfying bounds.
#[test]
fn limit_validation_matches_the_contract() {
    for (limits, queries, keys, valid) in [
        (vec![2, 3], 2, 3, true),
        (vec![3], 2, 3, false),
        (vec![3, 2], 2, 3, false),
        (vec![0, 3], 2, 3, false),
        (vec![2, 4], 2, 3, false),
        (vec![], 0, 0, true),
        (vec![3, 3, 3], 3, 3, true),
    ] {
        assert_eq!(
            key_limits_valid(&limits, queries, keys),
            valid,
            "{limits:?}"
        );
        assert_eq!(contract(&limits, queries, keys), valid);
    }
}

/// A chunk's key prefix is the bound of its last row and covers every earlier row.
#[test]
fn chunk_key_end_covers_every_row_of_the_chunk() {
    let limits = [1, 3, 3, 6];
    assert_eq!(chunk_key_end(&limits, 1), 1);
    assert_eq!(chunk_key_end(&limits, 3), 3);
    assert_eq!(chunk_key_end(&limits, 4), 6);
}

/// Window widths saturate exactly at `i64::MAX`.
#[test]
fn window_widths_saturate_at_the_signed_limit() {
    assert_eq!(saturating_window(0), 0);
    assert_eq!(saturating_window(4096), 4096);
    assert_eq!(saturating_window(i64::MAX as usize), i64::MAX);
    assert_eq!(saturating_window(i64::MAX as usize + 1), i64::MAX);
    assert_eq!(saturating_window(usize::MAX), i64::MAX);
}

/// Visibility metadata faults: zero windows first, then any extent or bound mismatch.
#[test]
fn visibility_faults_check_window_then_extents() {
    let ok = |window, query, key, query_valid, key_valid, limits: &[usize]| {
        visibility_fault(window, query, key, query_valid, key_valid, limits, 2, 2, 3)
    };
    assert_eq!(ok(None, [2, 2], [1, 3], None, None, &[2, 3]), None);
    assert_eq!(
        ok(Some(4), [1, 2], [2, 3], Some([2, 2]), Some([2, 3]), &[3, 3]),
        None
    );
    // A zero window wins even over broken extents.
    assert_eq!(
        ok(Some(0), [5, 5], [5, 5], None, None, &[]),
        Some(VisibilityFault::ZeroWindow)
    );
    for (query, key, query_valid, key_valid, limits) in [
        ([3, 2], [1, 3], None, None, vec![2, 3]),
        ([2, 2], [3, 3], None, None, vec![2, 3]),
        ([2, 1], [1, 3], None, None, vec![2, 3]),
        ([2, 2], [1, 4], None, None, vec![2, 3]),
        ([2, 2], [1, 3], Some([1, 2]), None, vec![2, 3]),
        ([2, 2], [1, 3], None, Some([2, 2]), vec![2, 3]),
        ([2, 2], [1, 3], None, Some([1, 3]), vec![2, 3]),
        ([2, 2], [1, 3], None, None, vec![3, 2]),
        ([2, 2], [1, 3], None, None, vec![2]),
    ] {
        assert_eq!(
            ok(Some(1), query, key, query_valid, key_valid, &limits),
            Some(VisibilityFault::Shape),
            "{query:?} {key:?} {query_valid:?} {key_valid:?} {limits:?}"
        );
    }
}

/// Projected components must match the configured, non-empty layout exactly.
#[test]
fn projection_extents_match_the_configured_layout() {
    let layout = MlaHeads {
        heads: 2,
        content: 3,
        rotary: 4,
        rank: 5,
    };
    let content = [1, 2, 6, 3];
    let rotary = [1, 2, 6, 4];
    let latent = [1, 7, 5];
    let key = [1, 1, 7, 4];
    assert!(projection_matches(layout, content, rotary, latent, key));
    for (content, rotary, latent, key) in [
        ([0, 2, 6, 3], [0, 2, 6, 4], [0, 7, 5], [0, 1, 7, 4]),
        ([1, 2, 0, 3], [1, 2, 0, 4], latent, key),
        (content, rotary, [1, 0, 5], [1, 1, 0, 4]),
        ([1, 3, 6, 3], [1, 3, 6, 4], latent, key),
        ([1, 2, 6, 2], rotary, latent, key),
        (content, rotary, [2, 7, 5], key),
        (content, rotary, [1, 7, 4], key),
        (content, [1, 2, 5, 4], latent, key),
        (content, [1, 2, 6, 2], latent, key),
        (content, rotary, latent, [1, 2, 7, 4]),
        (content, rotary, latent, [1, 1, 6, 4]),
        (content, [2, 2, 6, 4], latent, key),
        (content, rotary, latent, [2, 1, 7, 4]),
        (content, rotary, latent, [1, 1, 7, 3]),
    ] {
        assert!(!projection_matches(layout, content, rotary, latent, key));
    }
}

/// Bias extents broadcast when every axis is one or equal to the score plane.
#[test]
fn bias_extents_broadcast_axis_by_axis() {
    let shape = [2, 3, 4, 5];
    assert!(broadcasts_over(shape, shape));
    assert!(broadcasts_over([1, 1, 1, 1], shape));
    assert!(broadcasts_over([1, 3, 1, 5], shape));
    for axis in 0..4 {
        let mut extent = shape;
        extent[axis] += 1;
        assert!(!broadcasts_over(extent, shape));
        extent[axis] = 0;
        assert!(!broadcasts_over(extent, shape));
    }
}

proptest::proptest! {
    #![proptest_config(proptest::test_runner::Config::with_cases(256))]

    /// The validator agrees with the brute-force contract on random bounds.
    #[test]
    fn random_limits_validate_like_the_contract(
        limits in proptest::collection::vec(0usize..=6, 0..=6),
        queries in 0usize..=6,
        keys in 0usize..=6,
    ) {
        proptest::prop_assert_eq!(
            key_limits_valid(&limits, queries, keys),
            contract(&limits, queries, keys)
        );
    }

    /// Successful widths agree with 128-bit arithmetic, and failures overflow somewhere.
    #[test]
    fn random_widths_agree_with_wide_arithmetic(
        hidden in proptest::prop_oneof![1usize..=64, (usize::MAX / 8)..=usize::MAX],
        heads in proptest::prop_oneof![1usize..=8, (usize::MAX / 8)..=usize::MAX],
        nope in 1usize..=64,
        rope in 1usize..=64,
        value in proptest::prop_oneof![1usize..=64, (usize::MAX / 8)..=usize::MAX],
        rank in 1usize..=64,
        query_rank in proptest::option::of(proptest::prop_oneof![1usize..=64, (usize::MAX / 8)..=usize::MAX]),
    ) {
        // Products in 128 bits; `None` means beyond even 128 bits.
        let product = |a: u128, b: u128| a.checked_mul(b);
        let fits = |value: Option<u128>| value.is_some_and(|value| value <= usize::MAX as u128);
        let query = product(heads as u128, (nope + rope) as u128);
        let per_head = nope as u128 + value as u128;
        let expanded = product(heads as u128, per_head);
        let value_width = product(heads as u128, value as u128);
        let all_fit = fits(Some(per_head))
            && fits(query)
            && fits(expanded)
            && fits(product(hidden as u128, (rank + rope) as u128))
            && match query_rank {
                None => fits(query.and_then(|query| product(hidden as u128, query))),
                Some(q) => {
                    fits(product(hidden as u128, q as u128))
                        && fits(query.and_then(|query| product(q as u128, query)))
                }
            }
            && fits(expanded.and_then(|expanded| product(rank as u128, expanded)))
            && fits(value_width.and_then(|width| product(width, hidden as u128)));
        match mla_widths(hidden, heads, nope, rope, value, rank, query_rank) {
            Ok(widths) => {
                proptest::prop_assert!(all_fit);
                proptest::prop_assert_eq!(Some(widths.query_width as u128), query);
                proptest::prop_assert_eq!(Some(widths.expanded_width as u128), expanded);
                proptest::prop_assert_eq!(Some(widths.value_width as u128), value_width);
            }
            Err(_) => proptest::prop_assert!(!all_fit),
        }
    }
}
