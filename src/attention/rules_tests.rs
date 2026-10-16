//! Executable checks of the Verus-verified connectivity rules against brute-force definitions.

use super::*;

/// What a reduced local rule excludes, mirroring the `local_excluded` specification.
fn local_excludes(rule: LocalRule, query: usize, key: usize) -> bool {
    match rule {
        LocalRule::Unrestricted => false,
        LocalRule::Window(window) => key + window <= query,
        LocalRule::Chunk { width, phase } => (query + phase) / width != (key + phase) / width,
    }
}

/// The window oracle equals `window == 0 || key + window <= query` in wide arithmetic.
#[test]
fn window_oracle_matches_wide_definition() {
    for query in 0..12usize {
        for key in 0..12usize {
            for window in 0..8usize {
                assert_eq!(
                    window_excludes(query, key, window),
                    window == 0 || key + window <= query,
                    "q={query} k={key} W={window}"
                );
            }
        }
    }
    // The gap form cannot overflow where `key + window` would.
    assert!(!window_excludes(usize::MAX, usize::MAX, usize::MAX));
    assert!(window_excludes(usize::MAX, 0, usize::MAX));
    assert!(!window_excludes(usize::MAX - 1, 0, usize::MAX));
}

/// The chunk oracle compares floor-divided positions and rejects zero chunks.
#[test]
fn chunk_oracle_matches_floor_division() {
    for query in 0..12usize {
        for key in 0..12usize {
            assert!(chunk_excludes(query, key, 0));
            for chunk in 1..6usize {
                assert_eq!(
                    chunk_excludes(query, key, chunk),
                    query / chunk != key / chunk
                );
            }
        }
    }
}

/// Window reductions exclude exactly the window's links on every local pair.
#[test]
fn window_reductions_preserve_exclusions() {
    for span in 1..=9usize {
        for window in 1..=12usize {
            let rule = reduce_window(window, span);
            assert_eq!(rule == LocalRule::Unrestricted, window >= span);
            if window < span {
                assert_eq!(rule, LocalRule::Window(window));
            }
            // Only a rule that excludes nothing collapses.
            if rule != LocalRule::Unrestricted {
                assert!(local_excludes(rule, span - 1, 0));
            }
            for query in 0..span {
                for key in 0..span {
                    assert_eq!(
                        local_excludes(rule, query, key),
                        key + window <= query,
                        "W={window} span={span} q={query} k={key}"
                    );
                }
            }
        }
    }
    assert_eq!(reduce_window(usize::MAX, 3), LocalRule::Unrestricted);
}

/// Chunk reductions exclude exactly the absolute chunk boundaries crossed.
#[test]
fn chunk_reductions_preserve_absolute_boundaries() {
    for chunk in 1..=8usize {
        for key_start in 0..=17usize {
            for span in 1..=10usize {
                let rule = reduce_chunk(chunk, key_start, span);
                if let LocalRule::Chunk { width, phase } = rule {
                    assert!(1 <= width && width <= span && phase < width);
                }
                assert!(!matches!(rule, LocalRule::Window(_)));
                // Collapse happens exactly when the span crosses no chunk boundary.
                assert_eq!(
                    rule == LocalRule::Unrestricted,
                    key_start / chunk == (key_start + span - 1) / chunk,
                    "C={chunk} start={key_start} span={span}"
                );
                if rule != LocalRule::Unrestricted {
                    assert!(local_excludes(rule, span - 1, 0));
                }
                for query in 0..span {
                    for key in 0..span {
                        assert_eq!(
                            local_excludes(rule, query, key),
                            (key_start + query) / chunk != (key_start + key) / chunk,
                            "C={chunk} start={key_start} span={span} q={query} k={key}"
                        );
                    }
                }
            }
        }
    }
    // A chunk wider than any 32-bit integer still reduces to a small frame.
    let chunk = (1usize << 33) + 5;
    assert_eq!(
        reduce_chunk(chunk, chunk - 2, 6),
        LocalRule::Chunk { width: 6, phase: 4 }
    );
}

/// Mask geometry faults follow the documented order.
#[test]
fn mask_geometry_faults_are_ordered() {
    assert_eq!(mask_geometry_fault(1, 1, 1, 1, 0), None);
    for axes in [(0, 1, 1, 1), (1, 0, 1, 1), (1, 1, 0, 1), (1, 1, 1, 0)] {
        assert_eq!(
            mask_geometry_fault(axes.0, axes.1, axes.2, axes.3, usize::MAX),
            Some(MaskGeometryFault::EmptyAxis)
        );
    }
    let huge = i64::MAX as usize + 1;
    assert_eq!(
        mask_geometry_fault(1, 1, huge, huge, usize::MAX),
        Some(MaskGeometryFault::QueryTooLong)
    );
    assert_eq!(
        mask_geometry_fault(1, 1, 1, huge, usize::MAX),
        Some(MaskGeometryFault::KeyTooLong)
    );
    assert_eq!(mask_geometry_fault(1, 1, 1, i64::MAX as usize, 0), None);
    assert_eq!(
        mask_geometry_fault(1, 1, 1, 2, usize::MAX - 1),
        Some(MaskGeometryFault::KeyStartOverflow)
    );
    assert_eq!(mask_geometry_fault(1, 1, 1, 2, usize::MAX - 2), None);
}

/// Short key timelines are rejected except for unpadded or non-causal global attention.
#[test]
fn short_keys_are_rejected_for_local_or_padded_causal_masks() {
    for global in [false, true] {
        for causal in [false, true] {
            for key_masked in [false, true] {
                let excused = global && !(causal && key_masked);
                assert_eq!(
                    rejects_short_keys(global, causal, key_masked, 3, 2),
                    !excused
                );
                assert!(!rejects_short_keys(global, causal, key_masked, 2, 2));
                assert!(!rejects_short_keys(global, causal, key_masked, 2, 3));
            }
        }
    }
}

/// Positions are requested only when causality or a restricting rule needs them.
#[test]
fn position_sources_follow_the_rule_and_key_mask() {
    assert_eq!(
        position_source(true, false, false, 3, 2),
        PositionSource::Unneeded
    );
    assert_eq!(
        position_source(true, false, true, 3, 2),
        PositionSource::Unneeded
    );
    assert_eq!(
        position_source(true, true, false, 3, 2),
        PositionSource::Columns
    );
    assert_eq!(
        position_source(false, false, false, 2, 5),
        PositionSource::Columns
    );
    assert_eq!(
        position_source(true, true, true, 2, 5),
        PositionSource::Ordinals { query_offset: 3 }
    );
    assert_eq!(
        position_source(false, false, true, 4, 4),
        PositionSource::Ordinals { query_offset: 0 }
    );
}

/// The aligned position peak is the largest query or key position.
#[test]
fn aligned_position_peak_is_the_largest_position() {
    for queries in 0..6i64 {
        for keys in 0..6i64 {
            let largest = (0..queries)
                .map(|row| row + keys - queries)
                .chain(0..queries)
                .chain(0..keys)
                .max()
                .unwrap_or(-1);
            let peak = aligned_position_peak(queries, keys);
            assert_eq!(peak, queries.max(keys) - 1);
            if queries > 0 || keys > 0 {
                assert_eq!(peak, largest, "Q={queries} K={keys}");
            }
        }
    }
}

/// Position peaks are `(span - 1) + extent`, rejected only beyond `i64`.
#[test]
fn position_peaks_add_the_extent_to_the_last_position() {
    assert_eq!(position_peak(5, 0), Some(4));
    assert_eq!(position_peak(5, 4), Some(8));
    assert_eq!(position_peak(1, 0), Some(0));
    assert_eq!(position_peak(0, 0), Some(-1));
    assert_eq!(position_peak(i64::MAX, 0), Some(i64::MAX - 1));
    assert_eq!(position_peak(i64::MAX, 1), Some(i64::MAX));
    assert_eq!(position_peak(i64::MAX, 2), None);
    assert_eq!(position_peak(i64::MAX / 2 + 2, i64::MAX / 2 + 1), None);
}

proptest::proptest! {
    #![proptest_config(proptest::test_runner::Config::with_cases(256))]

    /// Random huge chunks and offsets reduce to frames that keep absolute chunks.
    #[test]
    fn random_chunk_reductions_preserve_boundaries(
        chunk in proptest::prop_oneof![1usize..=32, 33usize..=1 << 40],
        key_start in proptest::prop_oneof![0usize..=64, 65usize..=1 << 40],
        span in 1usize..=24,
    ) {
        let rule = reduce_chunk(chunk, key_start, span);
        proptest::prop_assert_eq!(
            rule == LocalRule::Unrestricted,
            key_start / chunk == (key_start + span - 1) / chunk
        );
        for query in 0..span {
            for key in 0..span {
                proptest::prop_assert_eq!(
                    local_excludes(rule, query, key),
                    (key_start + query) / chunk != (key_start + key) / chunk
                );
            }
        }
    }

    /// The window oracle agrees with 128-bit arithmetic for arbitrary positions.
    #[test]
    fn random_window_oracle_matches_wide_arithmetic(
        query in proptest::num::usize::ANY,
        key in proptest::num::usize::ANY,
        window in proptest::num::usize::ANY,
    ) {
        proptest::prop_assert_eq!(
            window_excludes(query, key, window),
            window == 0 || key as u128 + window as u128 <= query as u128
        );
    }
}
