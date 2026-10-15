//! Executable checks of the Verus-verified operand validation and causal routing.

use super::*;

/// Queries `[2, 4, 3, 8]`, keys `[2, 2, 5, 8]`, and values `[2, 2, 5, 6]` group consistently.
const QUERY: [usize; 4] = [2, 4, 3, 8];
/// Native keys of [`QUERY`].
const KEY: [usize; 4] = [2, 2, 5, 8];
/// Native values of [`QUERY`] with a different value width.
const VALUE: [usize; 4] = [2, 2, 5, 6];

/// Consistent operands yield their geometry, with optional mask, bias, and sinks.
#[test]
fn consistent_operands_yield_their_geometry() {
    let expected = HeadGeometry {
        batch: 2,
        query_heads: 4,
        key_value_heads: 2,
        query_sequence: 3,
        key_sequence: 5,
        head_dim: 8,
    };
    assert_eq!(
        operand_geometry(QUERY, KEY, VALUE, None, None, None),
        Ok(expected)
    );
    assert_eq!(
        operand_geometry(
            QUERY,
            KEY,
            VALUE,
            Some([2, 4, 3, 5]),
            Some([1, 4, 3, 5]),
            Some(4)
        ),
        Ok(expected)
    );
    assert_eq!(
        operand_geometry(QUERY, KEY, VALUE, None, Some([2, 4, 3, 5]), None),
        Ok(expected)
    );
    assert_eq!(expected.groups(), 2);
}

/// Operand faults are reported in check order.
#[test]
fn operand_faults_follow_check_order() {
    let mask = Some([2, 4, 3, 5]);
    for (query, key, value) in [
        ([0, 4, 3, 8], KEY, VALUE),
        ([2, 0, 3, 8], KEY, VALUE),
        ([2, 4, 0, 8], KEY, VALUE),
        ([2, 4, 3, 0], KEY, VALUE),
        (QUERY, [2, 0, 5, 8], VALUE),
        (QUERY, [2, 2, 0, 8], VALUE),
        (QUERY, KEY, [2, 2, 5, 0]),
    ] {
        assert_eq!(
            operand_geometry(query, key, value, Some([9; 4]), Some([9; 4]), Some(9)),
            Err(OperandFault::EmptyAxis)
        );
    }
    assert_eq!(
        operand_geometry(QUERY, [3, 3, 5, 8], VALUE, None, None, None),
        Err(OperandFault::UngroupedHeads)
    );
    for (key, value) in [
        ([1, 2, 5, 8], VALUE),
        ([2, 2, 5, 7], VALUE),
        (KEY, [1, 2, 5, 6]),
        (KEY, [2, 1, 5, 6]),
        (KEY, [2, 2, 4, 6]),
    ] {
        assert_eq!(
            operand_geometry(QUERY, key, value, Some([9; 4]), None, None),
            Err(OperandFault::KeyValueShape)
        );
    }
    for bad_mask in [[1, 4, 3, 5], [2, 2, 3, 5], [2, 4, 1, 5], [2, 4, 3, 4]] {
        assert_eq!(
            operand_geometry(QUERY, KEY, VALUE, Some(bad_mask), Some([9; 4]), Some(9)),
            Err(OperandFault::MaskShape)
        );
    }
    for bad_bias in [[3, 4, 3, 5], [2, 1, 3, 5], [2, 4, 2, 5], [1, 4, 3, 6]] {
        assert_eq!(
            operand_geometry(QUERY, KEY, VALUE, mask, Some(bad_bias), Some(9)),
            Err(OperandFault::BiasShape)
        );
    }
    assert_eq!(
        operand_geometry(QUERY, KEY, VALUE, mask, None, Some(2)),
        Err(OperandFault::SinkShape)
    );
}

/// The configured layout of [`QUERY`], optionally gated.
const fn layout(gated: bool) -> HeadLayout {
    HeadLayout {
        query_heads: 4,
        key_value_heads: 2,
        head_dim: 8,
        gated,
    }
}

/// Projection faults are reported in check order and accept matching shapes.
#[test]
fn projection_faults_follow_check_order() {
    let same_value = [2, 2, 5, 8];
    assert_eq!(
        projection_fault(layout(false), QUERY, KEY, same_value, None, None, None),
        None
    );
    assert_eq!(
        projection_fault(
            layout(true),
            QUERY,
            KEY,
            same_value,
            Some([2, 3, 32]),
            Some([2, 4, 3, 5]),
            Some([1, 4, 3, 5])
        ),
        None
    );
    let fault = |query, key, value, gate, mask, bias| {
        projection_fault(layout(true), query, key, value, gate, mask, bias)
    };
    let junk = Some([9; 4]);
    for (query, key) in [
        ([0, 4, 3, 8], KEY),
        ([2, 4, 0, 8], KEY),
        (QUERY, [2, 2, 0, 8]),
    ] {
        assert_eq!(
            fault(query, key, [9; 4], None, junk, junk),
            Some(ProjectionFault::EmptyAxis)
        );
    }
    for query in [[2, 3, 3, 8], [2, 4, 3, 7]] {
        assert_eq!(
            fault(query, [9; 4], [9; 4], None, junk, junk),
            Some(ProjectionFault::QueryShape)
        );
    }
    for key in [[1, 2, 5, 8], [2, 1, 5, 8], [2, 2, 5, 7]] {
        assert_eq!(
            fault(QUERY, key, [9; 4], None, junk, junk),
            Some(ProjectionFault::KeyShape)
        );
    }
    for value in [[1, 2, 5, 8], [2, 1, 5, 8], [2, 2, 4, 8], VALUE] {
        assert_eq!(
            fault(QUERY, KEY, value, None, junk, junk),
            Some(ProjectionFault::ValueShape)
        );
    }
    assert_eq!(
        projection_fault(
            layout(false),
            QUERY,
            KEY,
            same_value,
            Some([2, 3, 32]),
            junk,
            junk
        ),
        Some(ProjectionFault::UnexpectedGate)
    );
    for gate in [[1, 3, 32], [2, 2, 32], [2, 3, 31]] {
        assert_eq!(
            fault(QUERY, KEY, same_value, Some(gate), junk, junk),
            Some(ProjectionFault::GateShape)
        );
    }
    assert_eq!(
        fault(QUERY, KEY, same_value, None, junk, junk),
        Some(ProjectionFault::MissingGate)
    );
    let gate = Some([2, 3, 32]);
    for mask in [[1, 4, 3, 5], [2, 2, 3, 5], [2, 4, 2, 5], [2, 4, 3, 4]] {
        assert_eq!(
            fault(QUERY, KEY, same_value, gate, Some(mask), junk),
            Some(ProjectionFault::MaskShape)
        );
    }
    for bias in [[3, 4, 3, 5], [2, 1, 3, 5], [2, 4, 2, 5], [1, 4, 3, 6]] {
        assert_eq!(
            fault(QUERY, KEY, same_value, gate, None, Some(bias)),
            Some(ProjectionFault::BiasShape)
        );
    }
}

/// Bottom-right causal exclusion `key > row + K - Q`, evaluated in wide integers.
fn bottom_right(row: usize, key: usize, queries: usize, keys: usize) -> bool {
    key as i128 > row as i128 + keys as i128 - queries as i128
}

/// A causal mask is skipped exactly for one query row, which sees every key.
#[test]
fn skipped_causal_masks_exclude_nothing() {
    for queries in 0..=4 {
        assert!(!causal_mask_needed(false, queries));
        assert_eq!(causal_mask_needed(true, queries), queries != 1);
    }
    for keys in 1..=6 {
        assert!((0..keys).all(|key| !bottom_right(0, key, 1, keys)));
    }
}

/// Native routing uses the backend causal flag only where both alignments agree.
#[test]
fn native_routes_respect_causal_alignment() {
    assert_eq!(
        native_route(true, true, 4, 4),
        NativeRoute::Sinks { causal: true }
    );
    assert_eq!(
        native_route(true, true, 1, 4),
        NativeRoute::Sinks { causal: false }
    );
    assert_eq!(native_route(true, false, 4, 4), NativeRoute::SquareCausal);
    assert_eq!(
        native_route(true, false, 1, 1),
        NativeRoute::Folded { causal: false }
    );
    assert_eq!(
        native_route(true, false, 3, 7),
        NativeRoute::Folded { causal: true }
    );
    assert_eq!(
        native_route(false, false, 4, 4),
        NativeRoute::Folded { causal: false }
    );
    for size in 2..=6 {
        for row in 0..size {
            for key in 0..size {
                assert_eq!(key > row, bottom_right(row, key, size, size));
            }
        }
    }
}
