//! Mamba-2 planning: widths, geometry, row bookkeeping, mode partitions, and scan blocks.

use super::*;

/// Shorthand for a row with `tokens` valid tokens in `mode`.
fn row(tokens: usize, mode: Mamba2Mode) -> Mamba2Row {
    Mamba2Row { tokens, mode }
}

/// Widths follow the fused `[z, x, B, C, dt]` layout or name the first overflow.
#[test]
fn widths_are_exact_or_name_the_first_overflow() {
    // Nemotron-H 8B-like: 128 heads of 64, 8 groups of 128 state coordinates.
    assert_eq!(
        convolution_width(128, 64, 8, 128),
        Ok(128 * 64 + 2 * 8 * 128)
    );
    assert_eq!(
        projection_width(128, 64, 8, 128),
        Ok(2 * 128 * 64 + 2 * 8 * 128 + 128)
    );
    assert_eq!(
        convolution_width(usize::MAX, 2, usize::MAX, 2),
        Err(WidthOverflow::StateChannels)
    );
    assert_eq!(
        convolution_width(usize::MAX, 2, 1, 1),
        Err(WidthOverflow::Intermediate)
    );
    assert_eq!(
        convolution_width(usize::MAX, 1, 1, 1),
        Err(WidthOverflow::Convolution)
    );
    // The projection checks the intermediate width before the B/C width.
    assert_eq!(
        projection_width(usize::MAX, 2, usize::MAX, 2),
        Err(WidthOverflow::Intermediate)
    );
    assert_eq!(
        projection_width(1, 1, usize::MAX, 2),
        Err(WidthOverflow::StateChannels)
    );
    assert_eq!(
        projection_width(usize::MAX / 2, 1, 1, 1),
        Err(WidthOverflow::Projection)
    );
}

/// Zero widths and short kernels are reported before uneven groups.
#[test]
fn geometry_reports_the_first_violation() {
    assert_eq!(check_mamba2_geometry(64, 8, 16, 16, 2, 4, 128), Ok(()));
    for (kernel, chunk, groups) in [(1, 128, 2), (4, 0, 2), (4, 128, 0), (1, 128, 3)] {
        assert_eq!(
            check_mamba2_geometry(64, 8, 16, 16, groups, kernel, chunk),
            Err(Mamba2GeometryViolation::ZeroWidth)
        );
    }
    assert_eq!(
        check_mamba2_geometry(64, 8, 16, 16, 3, 4, 128),
        Err(Mamba2GeometryViolation::Groups)
    );
}

/// Row counts, fullness, and validity flags follow right padding.
#[test]
fn right_padded_rows_are_checked_and_flagged() {
    assert!(row_counts_fit(&[3, 0, 2], 3, 3));
    assert!(!row_counts_fit(&[3, 4], 2, 3));
    assert!(!row_counts_fit(&[3], 2, 3));
    assert!(rows_full(&[3, 3], 3));
    assert!(!rows_full(&[3, 2], 3));
    assert!(rows_full(&[], 3));
    assert_eq!(
        right_padded_flags(&[2, 0, 3], 3),
        [true, true, false, false, false, false, true, true, true]
    );
    assert!(right_padded_flags(&[2, 1], 0).is_empty());
}

/// Gather indices follow their closed forms or are rejected beyond `i64`.
#[test]
fn gather_indices_follow_closed_forms() {
    assert_eq!(row_end_columns(&[2, 0], 3), Some(vec![2, 3, 4, 0, 1, 2]));
    assert_eq!(row_end_columns(&[5], 0), Some(vec![]));
    assert_eq!(row_end_columns(&[i64::MAX as usize], 2), None);
    assert_eq!(window_columns(2, 3), Some(vec![1, 2, 3, 2, 3, 4]));
    assert_eq!(window_columns(3, 0), Some(vec![]));
    // Rejecting a window column beyond `i64` needs an index table of that
    // length, so only the proof covers it; row-end columns can exceed it cheaply.
    assert_eq!(signed_rows(&[2, 0, 1]), Some(vec![2, 0, 1]));
    assert_eq!(signed_rows(&[0, i64::MAX as usize + 1]), None);
}

/// Inactive rows join the first active row's mode; groups appear in first-appearance order.
#[test]
fn mode_partitions_group_rows_and_restore_batch_order() {
    let rows = [
        row(1, Mamba2Mode::Recurrent),
        row(3, Mamba2Mode::ChunkedPrefill),
        row(0, Mamba2Mode::ChunkedPrefill),
        row(1, Mamba2Mode::Recurrent),
    ];
    assert_eq!(
        partition_modes(&rows),
        ModePartition {
            first_mode: Mamba2Mode::Recurrent,
            first: vec![0, 2, 3],
            second_mode: Mamba2Mode::ChunkedPrefill,
            second: vec![1],
            restore: vec![0, 3, 1, 2],
        }
    );
    // Every row inactive: chunked prefill holds them all.
    let idle = [row(0, Mamba2Mode::Recurrent), row(0, Mamba2Mode::Recurrent)];
    let partition = partition_modes(&idle);
    assert_eq!(partition.first_mode, Mamba2Mode::ChunkedPrefill);
    assert_eq!((partition.first, partition.second), (vec![0, 1], vec![]));
    assert_eq!(
        partition_modes(&[]),
        ModePartition {
            first_mode: Mamba2Mode::ChunkedPrefill,
            first: vec![],
            second_mode: Mamba2Mode::Recurrent,
            second: vec![],
            restore: vec![],
        }
    );
}

/// Blocks hold the chunks the plane budget admits, at least one and at most all.
#[test]
fn scan_block_chunks_clamp_the_plane_budget() {
    // One plane is 1 * 2 * 4 * 4 = 32 elements: a budget of 100 admits three.
    assert_eq!(scan_block_chunks(1, 2, 4, 100, 10), 3);
    assert_eq!(scan_block_chunks(1, 2, 4, 100, 2), 2);
    assert_eq!(scan_block_chunks(1, 2, 4, 10, 10), 1);
    assert_eq!(
        scan_block_chunks(usize::MAX, usize::MAX, 4, usize::MAX, 7),
        1
    );
    assert_eq!(scan_block_chunks(0, 2, 4, 5, 7), 5);
}

/// Scan blocks tile the tokens; only the last block is padded.
#[test]
fn scan_blocks_tile_the_tokens() {
    let block = |start, width, chunks, padding| ScanBlock {
        start,
        width,
        chunks,
        padding,
    };
    assert_eq!(
        scan_blocks(10, 4, 2),
        [block(0, 8, 2, 0), block(8, 2, 1, 2)]
    );
    assert_eq!(scan_blocks(5, 128, 3), [block(0, 5, 1, 123)]);
    assert_eq!(scan_blocks(8, 4, 2), [block(0, 8, 2, 0)]);
}

proptest::proptest! {
    #![proptest_config(proptest::test_runner::Config::with_cases(256))]

    /// Partitions hold every row once, by effective mode, with a two-sided inverse.
    #[test]
    fn mode_partitions_are_inverse_permutations(
        rows in proptest::collection::vec((0usize..3, proptest::bool::ANY), 0..12),
    ) {
        let rows = rows
            .into_iter()
            .map(|(tokens, chunked)| {
                row(tokens, if chunked { Mamba2Mode::ChunkedPrefill } else { Mamba2Mode::Recurrent })
            })
            .collect::<Vec<_>>();
        let fallback = rows
            .iter()
            .find(|row| row.tokens > 0)
            .map_or(Mamba2Mode::ChunkedPrefill, |row| row.mode);
        let effective = |index: usize| {
            if rows[index].tokens > 0 { rows[index].mode } else { fallback }
        };
        let partition = partition_modes(&rows);
        let order = [partition.first.clone(), partition.second.clone()].concat();
        proptest::prop_assert_eq!(order.len(), rows.len());
        for (position, &index) in order.iter().enumerate() {
            proptest::prop_assert_eq!(partition.restore[index], position);
        }
        for &index in &partition.first {
            proptest::prop_assert_eq!(effective(index), partition.first_mode);
        }
        for &index in &partition.second {
            proptest::prop_assert_eq!(effective(index), partition.second_mode);
        }
    }

    /// Scan blocks match the reference greedy chunk walk.
    #[test]
    fn scan_blocks_match_the_greedy_walk(
        sequence in 1usize..400,
        chunk_size in 1usize..40,
        block_chunks in 1usize..6,
    ) {
        let chunks = sequence.div_ceil(chunk_size);
        let mut expected = Vec::new();
        let mut first_chunk = 0;
        while first_chunk < chunks {
            let count = block_chunks.min(chunks - first_chunk);
            let start = first_chunk * chunk_size;
            let width = (count * chunk_size).min(sequence - start);
            expected.push(ScanBlock {
                start,
                width,
                chunks: count,
                padding: count * chunk_size - width,
            });
            first_chunk += count;
        }
        proptest::prop_assert_eq!(scan_blocks(sequence, chunk_size, block_chunks), expected);
    }

    /// Flags and gather indices match their direct definitions.
    #[test]
    fn row_indices_match_direct_definitions(
        counts in proptest::collection::vec(0usize..6, 0..5),
        sequence in 0usize..6,
        width in 0usize..4,
    ) {
        let flags = counts
            .iter()
            .flat_map(|&count| (0..sequence).map(move |column| column < count))
            .collect::<Vec<_>>();
        proptest::prop_assert_eq!(right_padded_flags(&counts, sequence), flags);
        let starts = counts
            .iter()
            .flat_map(|&count| (0..width).map(move |offset| (count + offset) as i64))
            .collect::<Vec<_>>();
        proptest::prop_assert_eq!(row_end_columns(&counts, width), Some(starts));
        let windows = (0..sequence)
            .flat_map(|token| (0..width).map(move |offset| (token + 1 + offset) as i64))
            .collect::<Vec<_>>();
        proptest::prop_assert_eq!(window_columns(sequence, width), Some(windows));
    }
}
