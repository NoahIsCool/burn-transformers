//! Executable checks of the Verus-verified planning core against brute-force definitions.

use super::*;

/// Shorthand for bounded limits.
fn limits(max_query_tokens: usize, max_score_elements: usize) -> Option<ChunkLimits> {
    Some(ChunkLimits {
        max_query_tokens,
        max_score_elements,
    })
}

/// Collects every chunk range of a successful plan.
fn ranges(plan: ChunkGeometry, query_sequence: usize) -> Vec<(usize, usize)> {
    (0..plan.chunk_count)
        .map(|index| {
            query_chunk_range(
                plan.query_chunk_size,
                plan.chunk_count,
                query_sequence,
                index,
            )
            .expect("index below the chunk count")
        })
        .collect()
}

/// Bounded plans take the largest chunk allowed by both limits.
#[test]
fn bounded_plans_take_the_largest_admissible_chunk() {
    // One row is 2 * 5 = 10 elements; 30 elements admit 3 rows, tokens admit 4.
    let plan = plan_query_chunks(1, 2, 7, 5, limits(4, 30)).unwrap();
    assert_eq!(
        plan,
        ChunkGeometry {
            query_chunk_size: 3,
            chunk_count: 3,
            peak_score_elements: 30,
        }
    );
    assert_eq!(ranges(plan, 7), [(0, 3), (3, 6), (6, 7)]);
    // The token limit binds when the score limit is generous.
    let plan = plan_query_chunks(1, 1, 10, 1, limits(4, 1000)).unwrap();
    assert_eq!((plan.query_chunk_size, plan.chunk_count), (4, 3));
    // Short sequences use one chunk of exactly their length.
    let plan = plan_query_chunks(1, 1, 2, 1, limits(4, 1000)).unwrap();
    assert_eq!((plan.query_chunk_size, plan.chunk_count), (2, 1));
}

/// Unbounded plans are one chunk spanning every query.
#[test]
fn unbounded_plans_are_a_single_full_chunk() {
    let plan = plan_query_chunks(2, 3, 5, 7, None).unwrap();
    assert_eq!(
        plan,
        ChunkGeometry {
            query_chunk_size: 5,
            chunk_count: 1,
            peak_score_elements: 210,
        }
    );
    assert_eq!(ranges(plan, 5), [(0, 5)]);
    assert_eq!(query_chunk_range(5, 1, 5, 1), None);
}

/// Each failure reports its own precise reason, limits before shapes.
#[test]
fn errors_name_the_exact_failure() {
    assert_eq!(
        plan_query_chunks(0, 0, 0, 0, limits(0, 0)),
        Err(ChunkPlanError::ZeroMaxQueryTokens)
    );
    assert_eq!(
        plan_query_chunks(0, 0, 0, 0, limits(1, 0)),
        Err(ChunkPlanError::ZeroMaxScoreElements)
    );
    for shape in [(0, 1, 1, 1), (1, 0, 1, 1), (1, 1, 0, 1), (1, 1, 1, 0)] {
        assert_eq!(
            plan_query_chunks(shape.0, shape.1, shape.2, shape.3, None),
            Err(ChunkPlanError::EmptyAxis)
        );
    }
    assert_eq!(
        plan_query_chunks(usize::MAX, 2, 1, 1, None),
        Err(ChunkPlanError::RowOverflow)
    );
    assert_eq!(
        plan_query_chunks(1, usize::MAX, 1, 2, None),
        Err(ChunkPlanError::RowOverflow)
    );
    assert_eq!(
        plan_query_chunks(1, 2, 8, 32, limits(8, 63)),
        Err(ChunkPlanError::RowExceedsLimit {
            row_elements: 64,
            max_score_elements: 63,
        })
    );
    assert_eq!(
        plan_query_chunks(usize::MAX / 2 + 1, 1, 2, 1, None),
        Err(ChunkPlanError::PeakOverflow)
    );
    // Bounded plans never overflow their peak, even for huge sequences.
    let plan = plan_query_chunks(1, 1, usize::MAX, 1, limits(usize::MAX, usize::MAX)).unwrap();
    assert_eq!(plan.query_chunk_size, usize::MAX);
    assert_eq!(plan.chunk_count, 1);
}

/// Sliding-window key ranges equal the brute-force union of every row's window.
#[test]
fn sliding_window_ranges_are_exact_unions_of_row_windows() {
    for query_sequence in 1..=5 {
        for key_sequence in query_sequence..=7 {
            let offset = key_sequence - query_sequence;
            for window in 1..=4 {
                for start in 0..query_sequence {
                    for end in start + 1..=query_sequence {
                        let visible = (0..key_sequence)
                            .filter(|&key| {
                                (start..end).any(|row| {
                                    let position = row + offset;
                                    key <= position && position < key + window
                                })
                            })
                            .collect::<Vec<_>>();
                        let expected = (visible[0], visible[visible.len() - 1] + 1);
                        assert_eq!(
                            sliding_window_key_range(
                                query_sequence,
                                key_sequence,
                                start,
                                end,
                                window
                            ),
                            Some(expected),
                            "Q={query_sequence} K={key_sequence} W={window} rows={start}..{end}"
                        );
                    }
                }
            }
        }
    }
}

/// Invalid key-range requests return `None` rather than a wrapped interval.
#[test]
fn sliding_window_ranges_reject_invalid_requests() {
    for (query_sequence, key_sequence, start, end, window) in [
        (3, 5, 1, 1, 2),
        (3, 5, 2, 1, 2),
        (3, 5, 0, 4, 2),
        (6, 5, 0, 1, 2),
        (3, 5, 0, 1, 0),
    ] {
        assert_eq!(
            sliding_window_key_range(query_sequence, key_sequence, start, end, window),
            None
        );
    }
}

/// The row cap is the smaller of the row limit and the query length.
#[test]
fn capped_query_rows_take_the_smaller_extent() {
    assert_eq!(capped_query_rows(4, 10), 4);
    assert_eq!(capped_query_rows(10, 4), 4);
    assert_eq!(capped_query_rows(7, 7), 7);
    assert_eq!(capped_query_rows(0, 3), 0);
    assert_eq!(capped_query_rows(usize::MAX, 3), 3);
}

/// Window key bounds are `min(cap + window - 1, K)`, saturating instead of wrapping.
#[test]
fn windowed_key_bounds_cover_chunk_windows_without_overflow() {
    assert_eq!(windowed_key_bound(4, 3, 100), 6);
    assert_eq!(windowed_key_bound(4, 3, 5), 5);
    assert_eq!(windowed_key_bound(4, 1, 100), 4);
    assert_eq!(windowed_key_bound(0, 1, 9), 0);
    assert_eq!(windowed_key_bound(usize::MAX, 2, 17), 17);
    assert_eq!(
        windowed_key_bound(usize::MAX - 1, usize::MAX, usize::MAX),
        usize::MAX
    );
}

/// Checks a chunk frame against absolute chunk indices for every local position.
fn assert_frame_preserves_chunks(chunk: usize, key_start: usize, span: usize) {
    let frame = chunk_frame(chunk, key_start, span);
    for local in 0..span {
        let absolute = (key_start + local) / chunk;
        match frame {
            Some((width, phase)) => {
                assert!(phase < width && width <= span, "frame {frame:?}");
                assert_eq!(
                    absolute,
                    key_start / chunk + (local + phase) / width,
                    "C={chunk} start={key_start} span={span} p={local}"
                );
            }
            None => assert_eq!(
                absolute,
                key_start / chunk,
                "C={chunk} start={key_start} span={span}"
            ),
        }
    }
}

/// Chunk frames keep every local position in its absolute chunk.
#[test]
fn chunk_frames_preserve_absolute_chunks() {
    for chunk in 1..=9 {
        for key_start in 0..=20 {
            for span in 1..=12 {
                assert_frame_preserves_chunks(chunk, key_start, span);
            }
        }
    }
    // Chunks starting at the first key with a phase of zero keep their width.
    assert_eq!(chunk_frame(4, 8, 10), Some((4, 0)));
    // A wide chunk with one interior boundary becomes a span-wide frame.
    assert_eq!(chunk_frame(10, 7, 5), Some((5, 2)));
    // A span inside one chunk needs no frame at all.
    assert_eq!(chunk_frame(10, 0, 10), None);
    assert_eq!(chunk_frame(usize::MAX, 3, 1 << 20), None);
}

/// Chunk widths beyond 32-bit integers still yield small, exact frames.
#[test]
fn huge_chunk_frames_stay_within_the_span() {
    let span = 1000;
    let chunk = (1usize << 32) + 2;
    // The span straddles the boundary at absolute position `chunk`.
    let key_start = chunk - 300;
    assert_eq!(
        chunk_frame(chunk, key_start, span),
        Some((span, span - 300))
    );
    assert_frame_preserves_chunks(chunk, key_start, span);
}

proptest::proptest! {
    #![proptest_config(proptest::test_runner::Config::with_cases(256))]

    /// Random frames agree with absolute chunk indices.
    #[test]
    fn random_chunk_frames_preserve_absolute_chunks(
        chunk in proptest::prop_oneof![1usize..=16, 17usize..=1 << 40],
        key_start in proptest::prop_oneof![0usize..=64, 65usize..=1 << 40],
        span in 1usize..=64,
    ) {
        assert_frame_preserves_chunks(chunk, key_start, span);
    }

    /// Every chunk of a window-bounded plan fits the key bound and re-plans as one call.
    #[test]
    fn window_bounded_chunks_replan_as_one_call(
        batch in 1usize..=3,
        heads in 1usize..=3,
        query_sequence in 1usize..=24,
        extra_keys in 0usize..=8,
        window in 1usize..=10,
        max_query_tokens in 1usize..=12,
        max_score_elements in 1usize..=600,
        bounded in proptest::bool::ANY,
    ) {
        let key_sequence = query_sequence + extra_keys;
        let limits = bounded.then_some(ChunkLimits { max_query_tokens, max_score_elements });
        let cap = limits.map_or(query_sequence, |limits| {
            capped_query_rows(limits.max_query_tokens, query_sequence)
        });
        let key_bound = windowed_key_bound(cap, window, key_sequence);
        proptest::prop_assert_eq!(key_bound, (cap + window - 1).min(key_sequence));
        let Ok(plan) = plan_query_chunks(batch, heads, query_sequence, key_bound, limits) else {
            return Ok(());
        };
        for (start, end) in ranges(plan, query_sequence) {
            let (key_start, key_end) =
                sliding_window_key_range(query_sequence, key_sequence, start, end, window)
                    .expect("planned rows have a key range");
            proptest::prop_assert!(key_end - key_start <= key_bound);
            let replan = plan_query_chunks(batch, heads, end - start, key_end - key_start, limits)
                .expect("a planned chunk re-plans");
            proptest::prop_assert_eq!(replan.chunk_count, 1);
            proptest::prop_assert_eq!(replan.query_chunk_size, end - start);
        }
    }

    /// Successful plans partition the queries, respect both limits, and report the exact peak.
    #[test]
    fn plans_partition_queries_within_limits(
        batch in 1usize..=3,
        heads in 1usize..=4,
        query_sequence in 1usize..=40,
        key_bound in 1usize..=9,
        max_query_tokens in 1usize..=16,
        max_score_elements in 1usize..=400,
    ) {
        let row = batch * heads * key_bound;
        match plan_query_chunks(batch, heads, query_sequence, key_bound, limits(max_query_tokens, max_score_elements)) {
            Ok(plan) => {
                let expected_chunk = query_sequence.min(max_query_tokens).min(max_score_elements / row);
                proptest::prop_assert_eq!(plan.query_chunk_size, expected_chunk);
                proptest::prop_assert_eq!(plan.chunk_count, query_sequence.div_ceil(expected_chunk));
                proptest::prop_assert_eq!(plan.peak_score_elements, row * expected_chunk);
                proptest::prop_assert!(plan.peak_score_elements <= max_score_elements);
                let mut next = 0;
                for (start, end) in ranges(plan, query_sequence) {
                    proptest::prop_assert_eq!(start, next);
                    proptest::prop_assert!(start < end && end - start <= expected_chunk);
                    next = end;
                }
                proptest::prop_assert_eq!(next, query_sequence);
            }
            Err(error) => {
                proptest::prop_assert_eq!(error, ChunkPlanError::RowExceedsLimit {
                    row_elements: row,
                    max_score_elements,
                });
                proptest::prop_assert!(row > max_score_elements);
            }
        }
    }
}
