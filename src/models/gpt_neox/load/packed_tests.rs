//! Packed GPT-NeoX query/key/value row, byte, and staging arithmetic.

use proptest::prelude::*;

use super::*;

/// The former inline geometry check and row computation of the sink.
fn reference_rows(
    heads: usize,
    head_dim: usize,
    hidden: usize,
    expected_heads: usize,
    expected_hidden: usize,
    key_value_heads: usize,
) -> Result<PackedRows, PackedQkvError> {
    if heads == 0
        || head_dim == 0
        || hidden == 0
        || heads != expected_heads
        || hidden != expected_hidden
        || heads.checked_mul(head_dim) != Some(expected_hidden)
        || key_value_heads != heads
    {
        return Err(PackedQkvError::Geometry);
    }
    let rows = heads
        .checked_mul(head_dim)
        .and_then(|width| width.checked_mul(3))
        .ok_or(PackedQkvError::Overflow)?;
    Ok(PackedRows {
        rows,
        component_rows: rows / 3,
    })
}

/// The former inline per-head gather of the sink.
fn reference_component(
    payload: &[u8],
    heads: usize,
    head_dim: usize,
    row_bytes: usize,
    component: usize,
) -> Vec<u8> {
    let mut bytes = Vec::new();
    for head in 0..heads {
        let start = (head * 3 + component) * head_dim * row_bytes;
        let end = start + head_dim * row_bytes;
        bytes.extend_from_slice(&payload[start..end]);
    }
    bytes
}

/// A valid geometry yields `3 q d` packed rows and `q d` rows per component.
#[test]
fn packed_rows_accept_exactly_the_configured_geometry() {
    assert_eq!(
        packed_rows(2, 2, 4, 2, 4, 2),
        Ok(PackedRows {
            rows: 12,
            component_rows: 4,
        })
    );
    for (heads, head_dim, hidden, expected_heads, expected_hidden, key_value_heads) in [
        (0, 2, 4, 0, 4, 0),
        (2, 0, 4, 2, 4, 2),
        (2, 2, 0, 2, 0, 2),
        (2, 2, 4, 3, 4, 2),
        (2, 2, 4, 2, 5, 2),
        (2, 3, 4, 2, 4, 2),
        (2, 2, 4, 2, 4, 1),
        (2, usize::MAX, 4, 2, 4, 2),
    ] {
        assert_eq!(
            packed_rows(
                heads,
                head_dim,
                hidden,
                expected_heads,
                expected_hidden,
                key_value_heads
            ),
            Err(PackedQkvError::Geometry)
        );
    }
    let wide = usize::MAX / 2 + 1;
    assert_eq!(
        packed_rows(1, wide, wide, 1, wide, 1),
        Err(PackedQkvError::Overflow)
    );
}

/// Row and payload byte sizes are exact, and either product may overflow.
#[test]
fn packed_payload_is_exact_or_reports_overflow() {
    assert_eq!(
        packed_payload(12, 4, 2),
        Some(PackedPayload {
            row_bytes: 8,
            bytes: 96,
        })
    );
    assert_eq!(
        packed_payload(12, 1, 4),
        Some(PackedPayload {
            row_bytes: 4,
            bytes: 48,
        })
    );
    assert_eq!(packed_payload(1, usize::MAX, 2), None);
    assert_eq!(packed_payload(usize::MAX, 1, 2), None);
}

/// Each component gathers its head blocks in head order.
#[test]
fn packed_component_gathers_head_blocks_in_order() {
    // Two heads, one row per head, one byte per row: [q0, k0, v0, q1, k1, v1].
    let payload = [10, 20, 30, 11, 21, 31];
    assert_eq!(packed_component(&payload, 2, 1, 1, 0), vec![10, 11]);
    assert_eq!(packed_component(&payload, 2, 1, 1, 1), vec![20, 21]);
    assert_eq!(packed_component(&payload, 2, 1, 1, 2), vec![30, 31]);
}

/// Staging adds the largest component third to the base peak.
#[test]
fn packed_staging_peak_adds_the_largest_component() {
    assert_eq!(packed_staging_peak(7, &[]), Some(7));
    assert_eq!(packed_staging_peak(7, &[30, 8, 31]), Some(17));
    assert_eq!(packed_staging_peak(u64::MAX, &[2]), Some(u64::MAX));
    assert_eq!(packed_staging_peak(u64::MAX, &[3]), None);
    assert_eq!(packed_staging_peak(u64::MAX - 1, &[u64::MAX, 3]), None);
}

proptest! {
    /// The verified geometry check equals the sink's former inline checks.
    #[test]
    fn packed_rows_match_the_inline_checks(
        heads in prop_oneof![0_usize..5, Just(usize::MAX)],
        head_dim in prop_oneof![0_usize..5, Just(usize::MAX / 3 + 1)],
        hidden in prop_oneof![0_usize..17, Just(usize::MAX / 3 + 1)],
        expected_heads in 0_usize..5,
        expected_hidden in prop_oneof![0_usize..17, Just(usize::MAX / 3 + 1)],
        key_value_heads in 0_usize..5,
    ) {
        prop_assert_eq!(
            packed_rows(heads, head_dim, hidden, expected_heads, expected_hidden, key_value_heads),
            reference_rows(heads, head_dim, hidden, expected_heads, expected_hidden, key_value_heads)
        );
    }

    /// The verified gather equals the former inline gather, and the three
    /// components together read every payload byte exactly once.
    #[test]
    fn packed_components_partition_the_payload(
        heads in 1_usize..5,
        head_dim in 1_usize..4,
        row_scalars in 1_usize..5,
    ) {
        let scalars = 3 * heads * head_dim * row_scalars;
        // Distinct four-byte payload values identify each source scalar.
        let values = (0..scalars)
            .map(|index| u32::try_from(index).unwrap())
            .collect::<Vec<_>>();
        let bytes = values.iter().flat_map(|value| value.to_le_bytes()).collect::<Vec<_>>();
        let row_bytes = row_scalars * 4;
        let mut seen = Vec::new();
        for component in 0..3 {
            let gathered = packed_component(&bytes, heads, head_dim, row_bytes, component);
            prop_assert_eq!(
                &gathered,
                &reference_component(&bytes, heads, head_dim, row_bytes, component)
            );
            prop_assert_eq!(gathered.len(), heads * head_dim * row_bytes);
            seen.extend(
                gathered
                    .chunks_exact(4)
                    .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap())),
            );
        }
        seen.sort_unstable();
        prop_assert_eq!(seen, values);
    }

    /// Staging equals the former running maximum and checked addition.
    #[test]
    fn packed_staging_peak_matches_the_running_maximum(
        peak in prop_oneof![any::<u64>(), 0_u64..100],
        storage in prop::collection::vec(prop_oneof![any::<u64>(), 0_u64..100], 0..6),
    ) {
        let largest = storage.iter().fold(0_u64, |largest, &bytes| largest.max(bytes / 3));
        prop_assert_eq!(packed_staging_peak(peak, &storage), peak.checked_add(largest));
    }
}
